#!/usr/bin/env python3
"""Reproduce CRBN directional mechanics from frozen coordinates and settings.

The output directory contains numerical source tables, input hashes and
per-condition completion manifests. Existing verified stages can be resumed.
The original strengthening workflow and its outputs are not modified.
"""
from __future__ import annotations

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

try:
    import directional_mechanics as mechanics
    import directional_contacts as contacts
    import strengthen_contacts as previous_contacts
    import ddb1_complex_modes as legacy
    from hinge_geometry import compute_geometry
except ModuleNotFoundError:
    from scripts import directional_mechanics as mechanics
    from scripts import directional_contacts as contacts
    from scripts import strengthen_contacts as previous_contacts
    from scripts import ddb1_complex_modes as legacy
    from scripts.hinge_geometry import compute_geometry

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_INPUTS = ROOT / "data/directional_reference_inputs"
STAGES = ("mechanics", "contacts", "modes", "external", "figures")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, default=json_value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_rows(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        raise ValueError(f"Cannot write an unlabelled empty table: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, default=json_value) if isinstance(v, (list, dict, tuple)) else v
                             for k, v in row.items()})
    temp.replace(path)


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def condition_id(pdb, cutoff, weighting):
    return f"{pdb}_{cutoff:g}A_{weighting}"


def mapped_geometry(pdb, crbn_xyz, residues):
    """Transport the frozen endpoint screw line using only the fixed anchor."""
    conformers, labels, window, _ = legacy.load_reference()
    if not np.array_equal(residues, window):
        raise ValueError("Residue order differs from the frozen structural window")
    open_mask = np.load(ROOT / "data/pca_diffvec.npz")["open_mask"].astype(bool)
    frozen, _, _ = compute_geometry(conformers, open_mask, residues)
    mean_open = conformers[open_mask].mean(axis=0)
    mean_closed = conformers[~open_mask].mean(axis=0)
    anchor = residues <= 317
    rotation, pc, qc = legacy.kabsch(mean_open[anchor], mean_closed[anchor])
    anchored_open = (mean_open - pc) @ rotation + qc
    transport, pc2, qc2 = legacy.kabsch(anchored_open[anchor], crbn_xyz[anchor])
    result = dict(frozen)
    result["axis_point_A"] = (np.asarray(frozen["axis_point_A"]) - pc2) @ transport + qc2
    result["axis_unit_vector"] = np.asarray(frozen["axis_unit_vector"]) @ transport
    result["mapping"] = "Frozen mean-open anchor after mean-closed anchoring, aligned to reference residues <=317"
    result["reference"] = pdb
    return result


def directional_statistics(states, geometry):
    """Evaluate the actual tangent and finite displacement without refitting q."""
    U = states["isolated"]["U"]
    q = states["isolated"]["q"]
    tangent = U.T @ geometry["observed_rotation_tangent"]
    tangent /= np.linalg.norm(tangent)
    basis = U.T @ geometry["basis"]
    coefficients = geometry["coefficients"]
    values = {}
    models = []
    for name, state in states.items():
        G = state["G"]
        mean = state["mean_compliance"]
        small = basis.T @ G @ basis
        random_c = np.einsum("bi,ij,bj->b", coefficients, small, coefficients)
        finite_c = float(q @ G @ q)
        tangent_c = float(tangent @ G @ tangent)
        values[name] = {"finite": finite_c / mean, "tangent": tangent_c / mean,
                        "null": random_c / mean}
        overlaps = np.abs(q @ state["eigenvectors"])
        best20 = int(np.argmax(overlaps[:20]))
        best60 = int(np.argmax(overlaps[:60]))
        models.append({"model": name, "C_close": finite_c, "mean_compliance": mean,
                       "S_close": finite_c / mean, "C_tangent": tangent_c,
                       "S_tangent": tangent_c / mean, "mode1_internal_overlap": float(overlaps[0]),
                       "best_mode20": best20 + 1, "best_overlap20": float(overlaps[best20]),
                       "best_mode60": best60 + 1, "best_overlap60": float(overlaps[best60]),
                       "top3_projection": float(np.linalg.norm(overlaps[:3])),
                       "minimum_internal_stiffness": float(state["eigenvalues"][0]),
                       "internal_dimension": len(q),
                       "eigenvalue_interpretation": "Effective static stiffness in common CRBN internal coordinates"})
    comparisons = []
    null_columns = {}
    for role, numerator, denominator in (("R_body", "rigid", "fixed"),
                                         ("R_internal", "flexible", "rigid"),
                                         ("R_total", "flexible", "fixed"),
                                         ("M", "flexible", "isolated")):
        null = np.log(values[numerator]["null"]) - np.log(values[denominator]["null"])
        null_columns[role] = null
        for target in ("finite", "tangent"):
            effect = float(np.log(values[numerator][target]) - np.log(values[denominator][target]))
            percentile = float(100 * (np.count_nonzero(null < effect) + 0.5*np.count_nonzero(null == effect)) / len(null))
            comparisons.append({"role": role, "target": target, "effect": effect,
                                "rotational_percentile": percentile,
                                "null_median": float(np.median(null)),
                                "null_q05": float(np.quantile(null, .05)),
                                "null_q95": float(np.quantile(null, .95)),
                                "finite_tangent_overlap": geometry["finite_tangent_overlap"],
                                "null_n": len(null),
                                "interpretation": "Descriptive local rotation comparator" if target == "tangent"
                                else "Finite target compared descriptively to a different local rotation family"})
    return models, comparisons, null_columns


def input_record(pdb, cutoff, weighting, config_path):
    files = [ROOT / "data/crbn_ensemble.ens.npz", ROOT / "data/pca_diffvec.npz",
             ROOT / "data/crbn_residue_window.csv", Path(legacy.CIF_CACHE) / f"{pdb}.cif.gz",
             REFERENCE_INPUTS / "candidate_universe.csv", REFERENCE_INPUTS / "legacy_robustness.csv"]
    return {"pdb": pdb, "cutoff_A": cutoff, "weighting": weighting,
            "config_sha256": digest(config_path),
            "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in files]}


def stage_signature(inputs, stage):
    sources = {"mechanics": ["directional_mechanics.py", "run_directional_mechanics.py"],
               "contacts": ["directional_mechanics.py", "directional_contacts.py", "run_directional_mechanics.py"],
               "modes": ["directional_mechanics.py", "directional_modes.py", "strengthen_ddb1.py"]}[stage]
    sources = sorted(set(sources + ["strengthen_contacts.py", "ddb1_complex_modes.py", "hinge_geometry.py"]))
    payload = {"inputs": inputs, "stage": stage,
               "code": {name: digest(ROOT / "scripts" / name) for name in sources}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def complete(directory, signature):
    path = directory / "completion.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record["signature"] != signature:
        return False
    for row in record["files"]:
        p = directory / row["path"]
        if not p.is_file() or digest(p) != row["sha256"]:
            return False
    return True


def finish(directory, signature, started):
    files = [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "completion.json"]
    write_json(directory / "completion.json", {"signature": signature, "seconds": time.time()-started,
                                                "files": files})


def run_case(pdb, cutoff, weighting, config, config_path, output, stages, offline):
    case = condition_id(pdb, cutoff, weighting)
    xyz, residues, direction, axis_distances, partner_residues, cache = previous_contacts.load_case(pdb, offline)
    if len(residues) != config["crbn_position_count"]:
        raise ValueError("CRBN position count differs from frozen protocol")
    inputs = input_record(pdb, cutoff, weighting, config_path)
    paths = {"mechanics": output / "analysis/mechanics" / case,
             "contacts": output / "analysis/contact_roles" / case,
             "modes": output / "analysis/mode_paths" / case}
    pending = [s for s in stages if s in paths and not complete(paths[s], stage_signature(inputs, s))]
    if not pending:
        print(f"Verified cached {case}", flush=True)
        return
    started = time.time()
    system = mechanics.build_system(xyz, len(residues), cutoff, weighting)
    if any(s in pending for s in ("mechanics", "contacts")):
        states, checks = mechanics.make_states(system, xyz[:len(residues)], direction)
        if not checks["all_order_checks_pass"]:
            raise ArithmeticError(f"Static matrix order failed for {case}: {checks}")
    context = {"pdb": pdb, "cutoff_A": cutoff, "weighting": weighting,
               "reference_type": "apo" if pdb in config["apo_references"] else "engineered"}
    if "mechanics" in pending:
        target = paths["mechanics"]
        target.mkdir(parents=True, exist_ok=True)
        frozen = mapped_geometry(pdb, xyz[:len(residues)], residues)
        geometry = mechanics.geometry_directions(xyz[:len(residues)], residues, direction, frozen,
                                                 seed=config["seed"], n_draws=config["rotation_null"]["n_draws"])
        model_rows, comparison_rows, null = directional_statistics(states, geometry)
        write_rows(target / "models.csv", [{**context, **r} for r in model_rows])
        write_rows(target / "comparisons.csv", [{**context, **r} for r in comparison_rows])
        write_json(target / "verification.json", checks)
        write_json(target / "geometry.json", {k: v for k, v in geometry.items()
                                               if k not in {"basis", "sampled_directions", "coefficients",
                                                            "finite_direction", "observed_rotation_tangent", "moving_mask"}})
        np.savez_compressed(target / "rotation_null.npz", **null, coefficients=geometry["coefficients"],
                            finite=geometry["finite_direction"], tangent=geometry["observed_rotation_tangent"],
                            basis=geometry["basis"])
        write_json(target / "inputs.json", inputs)
        finish(target, stage_signature(inputs, "mechanics"), started)
        print(f"Mechanics {case}: Rbody={checks['R_body']:.6g}, Rinternal={checks['R_internal']:.6g}, M={checks['M']:.6g}", flush=True)
    if "contacts" in pending:
        target = paths["contacts"]
        target.mkdir(parents=True, exist_ok=True)
        discovery = read_rows(REFERENCE_INPUTS / "candidate_universe.csv")
        if len(discovery) != config["contact"]["candidate_count"]:
            raise ValueError("Frozen discovery universe is incomplete")
        result = contacts.analyse_contacts(system, states, xyz, residues, axis_distances, discovery, config)
        for key in ("groups", "factor_effects", "role_factor_effects", "edges", "ridge"):
            write_rows(target / (key + ".csv"), [{**context, **row} for row in result[key]], fields=None)
        write_json(target / "summary.json", result["summary"])
        write_json(target / "inputs.json", inputs)
        finish(target, stage_signature(inputs, "contacts"), started)
        print(f"Contacts {case}: {len(result['groups'])} frozen groups", flush=True)
    if "modes" in pending:
        try:
            from directional_modes import run_modes
        except ModuleNotFoundError:
            from scripts.directional_modes import run_modes
        target = paths["modes"]
        target.mkdir(parents=True, exist_ok=True)
        run_modes(system, direction, residues, config["interface_strengths"], target, config)
        write_json(target / "inputs.json", inputs)
        finish(target, stage_signature(inputs, "modes"), started)
    print(f"Completed {case} in {time.time()-started:.1f}s", flush=True)


def verified_conditions(output, config, config_path, stage):
    """Select only expected conditions verified against current inputs and code."""
    area = {"mechanics": "mechanics", "contacts": "contact_roles", "modes": "mode_paths"}[stage]
    accepted, rejected = [], []
    for weighting in config["weightings"]:
        for cutoff in config["cutoffs_A"]:
            for pdb in config["references"]:
                case = condition_id(pdb, cutoff, weighting)
                directory = output / "analysis" / area / case
                try:
                    inputs = input_record(pdb, cutoff, weighting, config_path)
                    stored = json.loads((directory / "inputs.json").read_text())
                    valid = stored == inputs and complete(directory, stage_signature(inputs, stage))
                except (FileNotFoundError, ValueError, KeyError):
                    valid = False
                if valid:
                    accepted.append(directory)
                else:
                    rejected.append(case)
    return accepted, rejected


def consolidate(output, config, config_path=None):
    config_path = config_path or ROOT / "scripts/directional_config.json"
    tables = {}
    validation = {}
    for source, names in (("mechanics", ("models", "comparisons")),
                           ("contact_roles", ("groups", "ridge"))):
        stage = "contacts" if source == "contact_roles" else source
        accepted, rejected = verified_conditions(output, config, config_path, stage)
        validation[stage] = {"accepted": [p.name for p in accepted], "missing_or_stale": rejected}
        for name in names:
            rows = []
            for directory in accepted:
                rows.extend(read_rows(directory / f"{name}.csv"))
            destination = output / "analysis" / source / (name + "_all.csv")
            if rows:
                write_rows(destination, rows)
                tables[f"{source}/{name}"] = rows
            elif destination.exists():
                destination.unlink()
    mode_paths, rejected_modes = verified_conditions(output, config, config_path, "modes")
    validation["modes"] = {"accepted": [p.name for p in mode_paths], "missing_or_stale": rejected_modes}
    write_json(output / "verification/consolidation.json", validation)
    if not validation["contacts"]["missing_or_stale"]:
        try:
            from summarize_directional_contacts import consolidate as summarize_contacts
        except ModuleNotFoundError:
            from scripts.summarize_directional_contacts import consolidate as summarize_contacts
        summarize_contacts(output / "analysis/contact_roles", config,
                           accepted_case_paths=[output / "analysis/contact_roles" / name
                                                for name in validation["contacts"]["accepted"]])
    else:
        for name in ("candidate_summary.csv", "summary.json"):
            stale = output / "analysis/contact_roles" / name
            if stale.exists():
                stale.unlink()
    comparisons = tables.get("mechanics/comparisons", [])
    if not comparisons:
        write_json(output / "analysis/mechanics/claim_gates.json",
                   {"claim_category": "incomplete_required_conditions", "verified_conditions": 0})
        return
    indexed = {(r["pdb"], float(r["cutoff_A"]), r["weighting"], r["role"], r["target"]): r for r in comparisons}
    tol = config["claim_gate"]["positive_tolerance"]
    gates = {}
    for role in ("R_body", "R_internal"):
        primary_keys = [(p, 15., "uniform", role, "tangent") for p in config["apo_references"]]
        sensitivity_keys = [("8CVP", c, "uniform", role, "tangent") for c in (13., 18.)]
        sensitivity_keys += [(p, 15., "inverse_square", role, "tangent") for p in config["apo_references"]]
        required = primary_keys + sensitivity_keys
        available = all(k in indexed for k in required)
        primary_pass = available and all(float(indexed[k]["effect"]) > tol and float(indexed[k]["rotational_percentile"]) >= 95 for k in primary_keys)
        sign_pass = available and all(float(indexed[k]["effect"]) > tol for k in sensitivity_keys)
        finite_keys = [(p, 15., "uniform", role, "finite") for p in config["apo_references"]]
        overlap_pass = available and all(float(indexed[k]["finite_tangent_overlap"]) >= .90 for k in primary_keys)
        finite_positive = available and all(float(indexed[k]["effect"]) > tol for k in finite_keys)
        gates[role] = {"complete": available, "primary_tangent_gate": bool(primary_pass),
                       "sensitivity_sign_gate": bool(sign_pass),
                       "local_rotation_selectivity": bool(primary_pass and sign_pass),
                       "finite_tangent_transfer": bool(primary_pass and sign_pass and overlap_pass and finite_positive),
                       "finite_overlap_gate": bool(overlap_pass)}
    preservation_keys = [(p, 15., "uniform", "M", "finite") for p in config["apo_references"]]
    preservation = all(k in indexed for k in preservation_keys) and all(float(indexed[k]["effect"]) >= -tol for k in preservation_keys)
    gates["relative_specificity_preserved"] = bool(preservation)
    internal = gates["R_internal"]["finite_tangent_transfer"]
    body = gates["R_body"]["finite_tangent_transfer"]
    gates["claim_category"] = ("internal_selective_preservation" if internal and preservation else
                                "internal_selective_partial_relief" if internal else
                                "rigid_body_selective_accommodation" if body else
                                "decomposition_without_demonstrated_finite_closure_selectivity")
    if not all(gates[role]["complete"] for role in ("R_body", "R_internal")):
        gates["claim_category"] = "incomplete_required_conditions"
    gates["scope"] = "Static harmonic response, three apo references from one study; no functional or equilibrium-population inference"
    write_json(output / "analysis/mechanics/claim_gates.json", gates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/directional_config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/directional_mechanics")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--stages", choices=STAGES, nargs="+", default=list(STAGES))
    parser.add_argument("--references", nargs="+", help="Run a documented subset; claim gates remain incomplete until all required conditions exist")
    parser.add_argument("--cutoffs", type=float, nargs="+")
    parser.add_argument("--weightings", choices=("uniform", "inverse_square"), nargs="+")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    refs = args.references or config["references"]
    if any(p not in config["references"] for p in refs):
        parser.error("Reference is outside the frozen configuration")
    cutoffs = args.cutoffs or config["cutoffs_A"]
    weightings = args.weightings or config["weightings"]
    if any(c not in config["cutoffs_A"] for c in cutoffs):
        parser.error("Cutoff is outside the frozen configuration")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"config_sha256": digest(args.config), "stages": args.stages, "offline": args.offline,
              "references": refs, "cutoffs_A": cutoffs, "weightings": weightings, "status": "running"}
    report_path = output / "verification/directional_workflow.json"
    write_json(report_path, report)
    started = time.time()
    try:
        numerical = [s for s in args.stages if s in {"mechanics", "contacts", "modes"}]
        if numerical:
            for weighting in weightings:
                for cutoff in cutoffs:
                    for pdb in refs:
                        run_case(pdb, cutoff, weighting, config, args.config, output, numerical, args.offline)
            consolidate(output, config, args.config)
        if "external" in args.stages:
            try:
                from directional_external import run
            except ModuleNotFoundError:
                from scripts.directional_external import run
            run(output, offline=args.offline, config=args.config)
        if "figures" in args.stages:
            try:
                from build_directional_figures import build
            except ModuleNotFoundError:
                from scripts.build_directional_figures import build
            build(output)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report["seconds"] = time.time()-started
        write_json(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
