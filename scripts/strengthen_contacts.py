#!/usr/bin/env python3
"""Prespecified CRBN contact perturbations under statically relaxed DDB1.

The Schur compliance is quasistatic and has arbitrary common spring units.
Specificity divides closure compliance by mean internal compliance. Positive
spring factors preserve the original network nullspace. No mutation efficacy,
population change, p value or FDR is inferred from these model perturbations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

# Bound BLAS work before importing numpy; independent analysis processes coexist.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, solve

try:
    import ddb1_complex_modes as legacy
    from softmode_lib import anm_hessian, contact_pairs
    from hinge_geometry import compute_geometry
except ModuleNotFoundError:
    from scripts import ddb1_complex_modes as legacy
    from scripts.softmode_lib import anm_hessian, contact_pairs
    from scripts.hinge_geometry import compute_geometry

ROOT = Path(__file__).resolve().parents[1]
legacy.CIF_CACHE = str(ROOT / "data/_cif_cache")
DEFAULT_OUTPUT = ROOT / "results/strengthening/contacts"
DEFAULT_CONFIG = ROOT / "scripts/strengthening_config.json"


def rigid_basis(xyz):
    """Six orthonormal rigid directions, rejecting collinear input."""
    centered = xyz - xyz.mean(axis=0)
    cols = [np.tile(v, (len(xyz), 1)).ravel() for v in np.eye(3)]
    cols += [np.cross(v, centered).ravel() for v in np.eye(3)]
    u, s, _ = np.linalg.svd(np.array(cols).T, full_matrices=False)
    if len(s) != 6 or s[-1] <= 1e-10 * s[0]:
        raise ValueError("CRBN coordinates do not define six rigid degrees of freedom")
    return u


def project_internal(x, rigid):
    return x - rigid @ (rigid.T @ x)


def schur_state(hessian, crbn_xyz, direction):
    """Factor the partner block and evaluate an internally gauged CRBN response.

    D must be positive definite for these coupled networks. Singular D is an
    explicit unsupported case, rather than a silent pseudoinverse fallback.
    """
    n = 3 * len(crbn_xyz)
    if not np.allclose(hessian, hessian.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Hessian is not symmetric")
    a, b, d = hessian[:n, :n], hessian[:n, n:], hessian[n:, n:]
    factor = cho_factor(d, lower=True, check_finite=True)
    dinvbt = cho_solve(factor, b.T, check_finite=False)
    effective = a - b @ dinvbt
    effective = (effective + effective.T) / 2
    eig, vec = eigh(effective, check_finite=False)
    tol = 1e-9
    if eig.min() < -tol or np.sum(eig <= tol) != 6:
        raise ValueError(f"Expected PSD effective Hessian with six zero modes: {eig[:8]}")
    rigid = rigid_basis(crbn_xyz)
    if np.linalg.norm(effective @ rigid) > 1e-7:
        raise ValueError("Effective Hessian does not annihilate CRBN rigid motions")
    q = project_internal(np.asarray(direction, dtype=float), rigid)
    retained_norm = float(np.linalg.norm(q))
    if retained_norm <= 1e-10:
        raise ValueError("Closure direction has no internal component")
    q /= retained_norm
    pinv = (vec[:, 6:] / eig[6:]) @ vec[:, 6:].T
    pinv = project_internal(project_internal(pinv, rigid).T, rigid).T
    cclose = float(q @ pinv @ q)
    mean = float(np.trace(pinv) / (n - 6))
    return {"effective": effective, "factor": factor, "dinvbt": dinvbt,
            "b": b, "pinv": pinv, "rigid": rigid, "q": q,
            "cclose": cclose, "mean": mean, "specificity": cclose / mean,
            "zero_modes": 6, "min_internal_eigenvalue": float(eig[6]),
            "target_internal_norm": retained_norm, "dof": n - 6}


def edge_columns(coords, edges):
    result = np.zeros((3 * len(coords), len(edges)))
    for k, (i, j) in enumerate(edges):
        delta = coords[j] - coords[i]
        delta /= np.linalg.norm(delta)
        result[3*i:3*i+3, k] = delta
        result[3*j:3*j+3, k] = -delta
    return result


def prepare_updates(state, columns):
    """Precompute low-rank response products, without inverting the full joint H."""
    n = len(state["q"])
    uc, ud = columns[:n], columns[n:]
    dinvud = cho_solve(state["factor"], ud, check_finite=False)
    condensed_u = uc - state["b"] @ dinvud
    # Edge forces are globally balanced. Removing numerical gauge components
    # also makes trace changes refer exclusively to internal CRBN response.
    response = state["pinv"] @ condensed_u
    response = project_internal(response, state["rigid"])
    gram = condensed_u.T @ response + ud.T @ dinvud
    gram = (gram + gram.T) / 2
    return {"gram": gram, "response_gram": response.T @ response,
            "closure_response": response.T @ state["q"]}


def perturbation_metrics(state, updates, indices, spring_factor):
    """Exact Woodbury update for a group of springs, at a fixed closure target."""
    if spring_factor <= 0:
        raise ValueError("Spring factor must be positive to preserve topology/nullspace")
    ids = np.asarray(indices, dtype=int)
    delta = spring_factor - 1.0
    gram = updates["gram"][np.ix_(ids, ids)]
    inverse_update = delta * solve(np.eye(len(ids)) + delta * gram,
                                   np.eye(len(ids)), assume_a="sym")
    closure = updates["closure_response"][ids]
    cclose = state["cclose"] - float(closure @ inverse_update @ closure)
    response_gram = updates["response_gram"][np.ix_(ids, ids)]
    mean = state["mean"] - float(np.sum(inverse_update.T * response_gram)) / state["dof"]
    if cclose <= 0 or mean <= 0:
        raise ValueError("Positive spring perturbation produced nonpositive compliance")
    specificity = cclose / mean
    return {"C_close": cclose, "mean_compliance": mean, "S_close": specificity,
            "delta_log_C_close": float(np.log(cclose / state["cclose"])),
            "delta_log_mean_compliance": float(np.log(mean / state["mean"])),
            "delta_log_S_close": float(np.log(specificity / state["specificity"]))}


def domain(residue):
    return "NTD" if residue < 187 else "HB" if residue < 318 else "TBD"


def candidate_groups(coords, residues, cutoff):
    n = len(residues)
    ii, jj, _ = contact_pairs(coords, cutoff)
    degrees = np.bincount(np.concatenate([ii, jj]), minlength=len(coords))[:n]
    edges, groups = [], {}
    for i, j in zip(ii, jj):
        cls = None
        if i < n <= j:
            cls, members = "CRBN_DDB1", [i]
        elif j < n:
            ri, rj = int(residues[i]), int(residues[j])
            if {domain(ri), domain(rj)} == {"HB", "TBD"} and abs(ri - rj) > 2:
                cls, members = "HB_TBD", [i, j]
        if cls:
            eid = len(edges)
            edges.append((int(i), int(j)))
            for member in members:
                groups.setdefault((int(residues[member]), cls), []).append(eid)
    return edges, groups, degrees


def matched_controls(rows, config):
    """Descriptive matching only, excluding edge-set duplicates in either tier."""
    result = []
    minimum = config["minimum_controls"]
    limits = config["control_calipers"]
    for row in rows:
        pool = [r for r in rows if r["contact_class"] == row["contact_class"]
                and r["domain"] == row["domain"] and r["edge_ids"] != row["edge_ids"]]
        exact = [r for r in pool if r["contact_count"] == row["contact_count"]]
        controls = exact
        tier = "exact_contact_count"
        if len(controls) < minimum:
            tier = "fixed_calipers"
            controls = [r for r in pool
                        if abs(r["contact_count"] / row["contact_count"] - 1) <= limits["contact_count_relative"] + 1e-12
                        and abs(r["joint_degree"] / row["joint_degree"] - 1) <= limits["degree_relative"] + 1e-12
                        and abs(r["axis_distance_A"] - row["axis_distance_A"]) <= limits["axis_distance_A"]]
        values = np.array([abs(r["D_g"]) for r in controls])
        value = abs(row["D_g"])
        percentile = float(100 * (np.sum(values < value) + .5*np.sum(values == value)) / len(values)) if len(values) >= minimum else None
        result.append({"residue": row["residue"], "contact_class": row["contact_class"],
                       "matching_tier": tier, "control_n": len(controls),
                       "matching_status": "adequate" if len(controls) >= minimum else "insufficient",
                       "matched_abs_effect_percentile": percentile,
                       "control_residues": ";".join(str(r["residue"]) for r in controls)})
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an unlabelled empty table: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


CONDITION_FILES = ("effects.csv", "candidates.csv", "matched_controls.csv", "edges.csv", "summary.json")


def condition_hashes(directory):
    """Require every output before accepting a completed calculation."""
    result = {}
    for name in CONDITION_FILES:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Incomplete condition output: {path}")
        item = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            with path.open() as stream:
                item["rows"] = sum(1 for _ in csv.DictReader(stream))
            if item["rows"] == 0:
                raise ValueError(f"Empty condition output: {path}")
        result[name] = item
    return result


def validated_completion(directory):
    manifest = directory / "output_hashes.json"
    if not manifest.exists():
        return False
    if json.loads(manifest.read_text()) != condition_hashes(directory):
        raise ValueError(f"Completed output checksum mismatch: {directory}")
    return True


def load_case(pdb, offline):
    confs, labels, residues, axis = legacy.load_reference()
    case = next(c for c in legacy.CASES if c[0] == pdb)
    cache = Path(legacy.CIF_CACHE) / f"{pdb}.cif.gz"
    if offline and not cache.exists():
        raise FileNotFoundError(f"Offline mmCIF missing: {cache}")
    legacy.CACHE_WRITES_ENABLED = False
    crbn = legacy.ca_coords(pdb, case[1])
    partner = legacy.ca_coords(pdb, case[2])
    crbn_raw = np.array([crbn[int(r)] for r in residues])
    rot, pc, qc = legacy.kabsch(crbn_raw, confs[labels.index(pdb)])
    nums = sorted(partner)
    xyz = (np.vstack([crbn_raw, [partner[r] for r in nums]]) - pc) @ rot + qc
    if not np.allclose(xyz[:len(residues)], confs[labels.index(pdb)], atol=1e-6):
        raise ValueError("Reference coordinate frame mismatch")
    mask = np.load(ROOT / "data/pca_diffvec.npz")["open_mask"].astype(bool)
    _, distances, _ = compute_geometry(confs, mask, residues)
    return xyz, residues, axis, distances, nums, cache


def analyse_condition(pdb, cutoff, config, output, offline, direct_check):
    start = time.monotonic()
    xyz, residues, direction, distances, partner_nums, cache = load_case(pdb, offline)
    n = len(residues)
    h = anm_hessian(xyz, cutoff)
    state = schur_state(h, xyz[:n], direction)
    edges, groups, degrees = candidate_groups(xyz, residues, cutoff)
    columns = edge_columns(xyz, edges)
    updates = prepare_updates(state, columns)
    effects, candidates = [], []
    for (residue, cls), ids in sorted(groups.items()):
        metrics = {}
        for factor in config["spring_factors"]:
            metrics[factor] = perturbation_metrics(state, updates, ids, factor)
            effects.append({"pdb": pdb, "cutoff_A": cutoff, "residue": residue,
                            "contact_class": cls, "spring_factor": factor, **metrics[factor]})
        idx = int(np.flatnonzero(residues == residue)[0])
        candidates.append({"pdb": pdb, "cutoff_A": cutoff, "residue": residue,
                           "contact_class": cls, "domain": domain(residue),
                           "contact_count": len(ids), "joint_degree": int(degrees[idx]),
                           "axis_distance_A": float(distances[idx]),
                           "edge_ids": ";".join(map(str, ids)),
                           "D_g": (metrics[1.1]["delta_log_S_close"] - metrics[.9]["delta_log_S_close"]) / .2,
                           "additional_zero_modes": 0})
    for cls in config["classes"]:
        ranked = sorted([r for r in candidates if r["contact_class"] == cls],
                        key=lambda r: (-abs(r["D_g"]), r["residue"]))
        for rank, row in enumerate(ranked, 1):
            row.update(rank=rank, class_n=len(ranked), rank_fraction=rank/len(ranked))
    direct = []
    if direct_check:
        # Independently rebuild the Schur complement for the largest-effect
        # group of each class at both endpoints. These checks do not select claims.
        for cls in config["classes"]:
            row = next(r for r in candidates if r["contact_class"] == cls and r["rank"] == 1)
            ids = groups[(row["residue"], cls)]
            u = columns[:, ids]
            for factor in [.8, 1.2]:
                altered = schur_state(h + (factor-1) * (u @ u.T), xyz[:n], state["q"])
                fast = perturbation_metrics(state, updates, ids, factor)
                relative = abs(fast["S_close"] / altered["specificity"] - 1)
                close_error = abs(fast["C_close"] / altered["cclose"] - 1)
                mean_error = abs(fast["mean_compliance"] / altered["mean"] - 1)
                if max(relative, close_error, mean_error) > 1e-7:
                    raise ValueError(f"Woodbury/direct disagreement: {relative}")
                direct.append({"contact_class": cls, "residue": row["residue"],
                               "factor": factor, "specificity_relative_error": relative,
                               "closure_compliance_relative_error": close_error,
                               "mean_compliance_relative_error": mean_error})
    key = f"{pdb}_{cutoff:g}A"
    destination = output / key
    write_csv(destination / "effects.csv", effects)
    write_csv(destination / "candidates.csv", candidates)
    write_csv(destination / "matched_controls.csv", matched_controls(candidates, config))
    edge_rows = []
    for k, (i, j) in enumerate(edges):
        memberships = [f"{r}:{c}" for (r, c), ids in groups.items() if k in ids]
        edge_rows.append({"edge_id": k, "i_node": i, "j_node": j,
                          "crbn_i_residue": int(residues[i]),
                          "j_partner": "CRBN" if j < n else "DDB1",
                          "j_residue": int(residues[j]) if j < n else partner_nums[j-n],
                          "candidate_groups": ";".join(memberships)})
    write_csv(destination / "edges.csv", edge_rows)
    summary = {"pdb": pdb, "cutoff_A": cutoff, "n_crbn": n, "n_ddb1": len(partner_nums),
               "C_close": state["cclose"], "mean_compliance": state["mean"],
               "S_close": state["specificity"], "zero_modes": state["zero_modes"],
               "minimum_internal_eigenvalue": state["min_internal_eigenvalue"],
               "target_internal_norm": state["target_internal_norm"],
               "n_unique_perturbed_edges": len(edges), "n_candidate_groups": len(groups),
               "direct_recomputation": direct, "elapsed_seconds": time.monotonic()-start,
               "source_cif_sha256": hashlib.sha256(cache.read_bytes()).hexdigest() if cache.exists() else None,
               "nullspace_rule": "All factors are strictly positive on unchanged edges; the sum-of-squares Hessian has exactly the baseline nullspace.",
               "matching_geometry": "Fixed mean-endpoint screw-axis distances; joint-network node degree at this reference/cutoff."}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (destination / "output_hashes.json").write_text(json.dumps(condition_hashes(destination), indent=2) + "\n")
    print(f"{key}: {len(groups)} groups, S={state['specificity']:.6g}, {summary['elapsed_seconds']:.1f}s", flush=True)
    return candidates, summary


def combine(output, config):
    rows = {}
    for path in output.glob("*/candidates.csv"):
        with path.open() as stream:
            for r in csv.DictReader(stream):
                key = (int(r["residue"]), r["contact_class"])
                condition = (r["pdb"], float(r["cutoff_A"]))
                rows.setdefault(key, {})[condition] = r
    discovery = (config["discovery_reference"], config["discovery_cutoff_A"])
    robust_conditions = [tuple(c) for c in config["robust_conditions"]]
    engineered = [tuple(c) for c in config["engineered_conditions"]]
    results = []
    for key, conditions in rows.items():
        if discovery not in conditions:
            continue
        base = conditions[discovery]
        sign = np.sign(float(base["D_g"]))
        def passes(condition, conditions=conditions, sign=sign):
            r = conditions.get(condition)
            return (r is not None and np.sign(float(r["D_g"])) == sign and sign != 0
                    and float(r["rank_fraction"]) <= .2 and int(r["additional_zero_modes"]) == 0)
        complete = all(c in conditions for c in robust_conditions)
        robust = complete and all(passes(c) for c in robust_conditions)
        results.append({"residue": key[0], "contact_class": key[1], "discovery_D_g": float(base["D_g"]),
                        "discovery_rank": int(base["rank"]),
                        "discovery_top5": int(base["rank"]) <= config["top_n_per_class"],
                        "all_required_conditions_observed": complete,
                        "stable_apo_model_candidate": robust,
                        "also_consistent_in_engineered_references": robust and all(passes(c) for c in engineered),
                        "condition_results": ";".join(f"{p}:{c:g}={'pass' if passes((p,c)) else 'absent' if (p,c) not in conditions else 'fail'}" for p,c in robust_conditions+engineered)})
    results.sort(key=lambda r: (r["contact_class"], r["discovery_rank"]))
    write_csv(output / "candidate_robustness.csv", results)
    write_csv(output / "discovery_top5.csv", [r for r in results if r["discovery_top5"]])
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Only frozen discovery 8CVP/15A with direct checks")
    parser.add_argument("--resume", action="store_true", help="Reuse completed conditions only if config/source hashes match")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())["contact"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [args.config, Path(__file__), ROOT / "data/crbn_ensemble.ens.npz",
              ROOT / "data/pca_diffvec.npz", ROOT / "data/crbn_residue_window.csv"]
    inputs += [ROOT / "scripts" / name for name in
               ["ddb1_complex_modes.py", "softmode_lib.py", "hinge_geometry.py", "analysis_contracts.py"]]
    inputs += [Path(legacy.CIF_CACHE) / f"{c[0]}.cif.gz" for c in legacy.CASES]
    hashes = {(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    manifest = args.output_dir / "input_hashes.json"
    if args.resume and manifest.exists() and json.loads(manifest.read_text()) != hashes:
        raise ValueError("Resume inputs differ from frozen completed results")
    manifest.write_text(json.dumps(hashes, indent=2) + "\n")
    conditions = [["8CVP", 15.0]] if args.pilot else config["robust_conditions"] + config["engineered_conditions"]
    for pdb, cutoff in conditions:
        summary = args.output_dir / f"{pdb}_{cutoff:g}A/summary.json"
        if args.resume and validated_completion(summary.parent):
            continue
        analyse_condition(pdb, cutoff, config, args.output_dir, args.offline,
                          direct_check=(pdb == "8CVP" and cutoff == 15))
    combined = combine(args.output_dir, config)
    print(f"Stable apo model candidates: {sum(r['stable_apo_model_candidate'] for r in combined)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
