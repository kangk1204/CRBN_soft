#!/usr/bin/env python3
"""Observed-residue sensitivity measured in the unchanged CRBN core.

The original CRBN internal constraint is applied before inversion. Added,
experimentally observed CRBN nodes and permitted DDB1 degrees of freedom are
relaxed by positive-definite Schur solves. Both windows use the same force
probe and the same 801-dimensional mean compliance. No missing coordinates
are imputed and no pseudoinverse is used.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, block_diag
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

try:
    import directional_mechanics as dm
    import directional_contacts as dc
    import strengthen_contacts as sc
    import run_directional_mechanics as runner
    from curation_contracts import cif_loop_rows
except ModuleNotFoundError:
    from scripts import directional_mechanics as dm
    from scripts import directional_contacts as dc
    from scripts import strengthen_contacts as sc
    from scripts import run_directional_mechanics as runner
    from scripts.curation_contracts import cif_loop_rows

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("isolated", "fixed", "rigid", "flexible")
METRICS = ("C_close", "mean_compliance", "S_close")
POLICIES = ("original_edges", "expanded_incident_edges")
RESPONSE_TOL = 1e-7
MATRIX_TOL = 1e-8


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["status"]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_table(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sym(a):
    return (a + a.T) * .5


def _factor(a, name):
    if a.size == 0:
        return None
    if not np.isfinite(a).all() or not np.allclose(a, a.T, atol=1e-9, rtol=1e-10):
        raise ArithmeticError(f"{name}: nonfinite or asymmetric stiffness")
    try:
        return cho_factor(a, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        minimum = eigh(_sym(a), eigvals_only=True, subset_by_index=(0, 0))[0]
        raise ArithmeticError(f"{name}: singular or indefinite stiffness; minimum={minimum}") from error


def _solve(factor, rhs):
    return cho_solve(factor, rhs, check_finite=False)


def _core_basis(xyz, direction, U=None, q=None):
    direction = np.asarray(direction, dtype=float).reshape(-1)
    if direction.shape != (3*len(xyz),) or not np.isfinite(direction).all():
        raise ValueError("Direction must be a finite core vector")
    if U is None:
        U = dm.internal_basis(xyz)
    if q is None:
        q = U.T @ direction
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm <= dm.ZERO_TOL:
            raise ValueError("Direction has no finite core-internal component")
        q = q / norm
    if U.shape != (3 * len(xyz), 3 * len(xyz) - 6):
        raise ValueError("Invalid core internal basis")
    if not np.isfinite(U).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-8:
        raise ValueError("Invalid or nonunit projected direction")
    if not np.allclose(U.T@U, np.eye(U.shape[1]), atol=1e-8):
        raise ValueError("Core basis is not orthonormal")
    if np.max(np.abs(dm.rigid_basis(xyz).T@U)) > 1e-8:
        raise ValueError("Core basis contains rigid motion")
    return U, q


def make_state(system, n_core, direction, model, U=None, q=None):
    """Condense partner, then added nodes, keeping the original core gauge."""
    if model not in MODELS:
        raise ValueError(model)
    U, q = _core_basis(system["crbn_xyz"][:n_core], direction, U, q)
    nc = 3 * n_core
    n_crbn = system["n_crbn"]
    nx = 3 * (n_crbn - n_core)
    A = system["h_crbn_isolated"] if model == "isolated" else system["A"]
    Z = dm.rigid_basis(system["ddb1_xyz"]) if model == "rigid" else None
    if model == "flexible":
        B, D = system["B"], system["D"]
    elif model == "rigid":
        B, D = system["B"] @ Z, _sym(Z.T @ system["D"] @ Z)
    else:
        B, D = np.zeros((len(A), 0)), np.zeros((0, 0))
    fd = _factor(D, f"{model} partner")
    E = _sym(A - B @ _solve(fd, B.T)) if fd is not None else A
    Kcc = U.T @ E[:nc, :nc] @ U
    Kcx = U.T @ E[:nc, nc:]
    Kxx = E[nc:, nc:]
    fx = _factor(Kxx, f"{model} added CRBN") if nx else None
    H = _sym(Kcc - Kcx @ _solve(fx, Kcx.T)) if nx else _sym(Kcc)
    fh = _factor(H, f"{model} core")
    G = _sym(_solve(fh, np.eye(len(q))))
    values, vectors = eigh(H, check_finite=False)
    if values[0] <= dm.ZERO_TOL:
        raise ArithmeticError(f"{model} has an additional core zero mode")
    C = float(q @ G @ q)
    mean = float(np.trace(G) / len(q))
    if not np.isfinite([C, mean]).all() or min(C, mean) <= 0:
        raise ArithmeticError(f"{model} produced invalid compliance")
    residual = float(np.linalg.norm(H @ G - np.eye(len(q)), ord=np.inf))
    if residual > RESPONSE_TOL:
        raise ArithmeticError(f"{model} inverse residual {residual}")
    return {"name": model, "U": U, "q": q, "H": H, "G": G,
            "C_close": C, "mean_compliance": mean, "S_close": C / mean,
            "dof": len(q), "n_core": n_core, "n_crbn": n_crbn,
            "partner_B": B, "partner_D": D, "partner_factor": fd,
            "partner_basis": Z if model == "rigid" else ("identity" if model == "flexible" else None),
            "Kcx": Kcx, "Kxx": Kxx, "extra_factor": fx,
            "eigenvalues": values, "eigenvectors": vectors,
            "inverse_residual_inf": residual,
            "condition_number_core": float(values[-1] / values[0])}


def make_states(system, n_core, direction, U=None, q=None):
    U, q = _core_basis(system["crbn_xyz"][:n_core], direction, U, q)
    return {name: make_state(system, n_core, direction, name, U, q) for name in MODELS}


def prepare_updates(state, columns):
    """Exact low-rank covariance updates through both elimination levels."""
    nc, nr = 3 * state["n_core"], 3 * state["n_crbn"]
    crbn = columns[:nr].copy()
    partner = columns[nr:]
    if state["name"] == "isolated":
        # Interface springs are absent in this model; ignore their columns.
        crbn[:, np.any(partner != 0, axis=0)] = 0
    basis = state["partner_basis"]
    if basis is None:
        partner = np.zeros((0, columns.shape[1]))
    elif not isinstance(basis, str):
        partner = basis.T @ partner
    gram_other = np.zeros((columns.shape[1], columns.shape[1]))
    if partner.shape[0]:
        dp = _solve(state["partner_factor"], partner)
        crbn -= state["partner_B"] @ dp
        gram_other += partner.T @ dp
    uc, ux = state["U"].T @ crbn[:nc], crbn[nc:]
    if ux.shape[0]:
        dx = _solve(state["extra_factor"], ux)
        uc -= state["Kcx"] @ dx
        gram_other += ux.T @ dx
    response = state["G"] @ uc
    return {"gram": _sym(uc.T @ response + gram_other),
            "response_gram": _sym(response.T @ response),
            "closure_response": response.T @ state["q"],
            "response": response}


def constrained_matrix(system, state):
    """Independent direct matrix in [core internal, added, permitted DDB1]."""
    nr, nc = 3 * state["n_crbn"], 3 * state["n_core"]
    Q = block_diag(state["U"], np.eye(nr - nc))
    A = system["h_crbn_isolated"] if state["name"] == "isolated" else system["A"]
    ca = _sym(Q.T @ A @ Q)
    if state["partner_D"].size:
        cb = Q.T @ state["partner_B"]
        return np.block([[ca, cb], [cb.T, state["partner_D"]]])
    return ca


def verify_direct_response(system, state):
    K = constrained_matrix(system, state)
    rhs = np.zeros((len(K), 4))
    rng = np.random.default_rng(20260906)
    rhs[:state["dof"], 0] = state["q"]
    rhs[:state["dof"], 1:] = rng.normal(size=(state["dof"], 3))
    solution = _solve(_factor(K, "full constrained"), rhs)
    expected = state["G"] @ rhs[:state["dof"]]
    relative = float(np.linalg.norm(solution[:state["dof"]] - expected) / np.linalg.norm(expected))
    residual = float(np.linalg.norm(K @ solution - rhs) / np.linalg.norm(rhs))
    if max(relative, residual) > RESPONSE_TOL:
        raise ArithmeticError(f"Full/sequential response mismatch {relative}, {residual}")
    return {"model": state["name"], "core_response_relative_error": relative,
            "direct_relative_residual": residual, "rhs_count": 4, "pass": True}


def changed_system(system, columns, factor):
    """Directly add the specified spring updates to the original block matrices."""
    out = dict(system)
    nr = 3 * system["n_crbn"]
    uc, ud = columns[:nr], columns[nr:]
    change = factor - 1.0
    out["A"] = system["A"] + change * (uc @ uc.T)
    out["B"] = system["B"] + change * (uc @ ud.T)
    out["D"] = system["D"] + change * (ud @ ud.T)
    internal = ~np.any(ud != 0, axis=0)
    out["h_crbn_isolated"] = system["h_crbn_isolated"] + change * (uc[:, internal] @ uc[:, internal].T)
    return out


def load_window(pdb, offline=True):
    """Append only observed Q96SW2 positions; validate author/canonical mapping."""
    xyz, core, direction, axis_distances, partner_nums, cache = sc.load_case(pdb, offline)
    text = gzip.open(cache, "rt").read()
    chain = next(item[1] for item in sc.legacy.CASES if item[0] == pdb)
    intervals = [row for row in cif_loop_rows(text, "struct_ref_seq")
                 if row.get("pdbx_db_accession") == "Q96SW2"
                 and chain in row.get("pdbx_strand_id", "").split(",")]
    if not intervals:
        raise ValueError(f"{pdb}: missing exact Q96SW2 mapping")
    allowed = set()
    for row in intervals:
        ab, ae = int(row["pdbx_auth_seq_align_beg"]), int(row["pdbx_auth_seq_align_end"])
        db, de = int(row["db_align_beg"]), int(row["db_align_end"])
        if (ab, ae) != (db, de) or not 1 <= ab <= ae <= 442:
            raise ValueError(f"{pdb}: nonidentity author mapping requires explicit curation")
        allowed.update(range(ab, ae + 1))
    # Atom loops contain no multiline fields. First CA occurrence matches the
    # frozen loader; nonempty insertion codes and non-first models are excluded.
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("_atom_site."))
    headers = []
    while lines[start].startswith("_atom_site."):
        headers.append(lines[start].strip().split(".", 1)[1])
        start += 1
    ix = {name: i for i, name in enumerate(headers)}
    observed, names, excluded = {}, {}, []
    for line in lines[start:]:
        if line.startswith("#"):
            break
        tokens = line.split()
        if len(tokens) < len(headers) or tokens[ix["group_PDB"]] != "ATOM":
            continue
        if tokens[ix["auth_asym_id"]] != chain or tokens[ix["label_atom_id"]] != "CA":
            continue
        try:
            residue = int(tokens[ix["auth_seq_id"]])
        except ValueError:
            continue
        if "pdbx_PDB_model_num" in ix and tokens[ix["pdbx_PDB_model_num"]] != "1":
            continue
        reason = ""
        if tokens[ix["pdbx_PDB_ins_code"]] not in (".", "?"):
            reason = "insertion_code"
        elif residue not in allowed:
            reason = "outside_Q96SW2_mapping"
        if reason:
            excluded.append({"pdb": pdb, "author_residue": residue, "reason": reason})
            continue
        observed.setdefault(residue, [float(tokens[ix[key]]) for key in ("Cartn_x", "Cartn_y", "Cartn_z")])
        names.setdefault(residue, tokens[ix["label_comp_id"]])
    if not set(map(int, core)).issubset(observed):
        raise ValueError(f"{pdb}: frozen core not included in accepted mapped atoms")
    rawcore = np.asarray([observed[int(r)] for r in core])
    rotation, pc, qc = sc.legacy.kabsch(rawcore, xyz[:len(core)])
    if np.max(np.abs((rawcore - pc) @ rotation + qc - xyz[:len(core)])) > 1e-6:
        raise ValueError(f"{pdb}: expanded/core coordinate mismatch")
    added = np.array(sorted(set(observed) - set(map(int, core))), dtype=int)
    extra = (np.asarray([observed[int(r)] for r in added]) - pc) @ rotation + qc
    expanded = np.vstack([xyz[:len(core)], extra, xyz[len(core):]])
    coverage = [{"pdb": pdb, "residue": r, "domain": dc._domain(r),
                 "status": "core" if r in core else ("observed_added" if r in observed
                           else ("unobserved" if r in allowed else "outside_construct")),
                 "observed_residue_name": names.get(r, "")}
                for r in range(1, 443)]
    return {"original_xyz": xyz, "expanded_xyz": expanded, "core": core,
            "expanded_residues": np.r_[core, added], "added": added,
            "direction": direction, "axis_distances": axis_distances,
            "partner_residues": partner_nums, "coverage": coverage,
            "excluded_observed": excluded, "cache": cache, "mapping": intervals}


def edge_key(pair, residues, partner_residues):
    n = len(residues)
    nodes = [("CRBN", int(residues[i])) if i < n else ("DDB1", int(partner_residues[i-n])) for i in pair]
    return "--".join(f"{name}:{r}" for name, r in sorted(nodes))


def candidate_data(system, residues, partner_residues, universe):
    edges, found, degrees = sc.candidate_groups(system["coords"], residues, system["cutoff"])
    weights_by_pair = {tuple(pair): weight for pair, weight in zip(system["pairs"], system["weights"])}
    groups = {f"{r}:{cls}": ids for (r, cls), ids in found.items() if f"{r}:{cls}" in universe}
    used = sorted({i for ids in groups.values() for i in ids})
    old_to_new = {old: new for new, old in enumerate(used)}
    groups = {group: [old_to_new[i] for i in ids] for group, ids in groups.items()}
    edges = [edges[i] for i in used]
    weights = np.array([weights_by_pair[tuple(pair)] for pair in edges])
    columns = dc._edge_columns(system["coords"], edges, weights)
    keys = [edge_key(pair, residues, partner_residues) for pair in edges]
    return {"edges": edges, "keys": keys, "weights": weights,
            "groups": groups, "columns": columns, "degrees": degrees}


def contact_tables(system, states, data, universe, residues, old_edge_keys, policy,
                   axis_distances, factors=(.8, .9, 1.1, 1.2)):
    keys = data["keys"]
    groups = {}
    for group in universe:
        ids = data["groups"].get(group, [])
        if policy == "original_edges":
            wanted = old_edge_keys.get(group, set())
            ids = [i for i in ids if keys[i] in wanted]
            if {keys[i] for i in ids} != wanted:
                raise ValueError(f"Original spring set lost for {group}")
        groups[group] = ids
    membership = {}
    for group, ids in groups.items():
        for i in ids:
            membership.setdefault(i, []).append(group)
    updates = {name: prepare_updates(states[name], data["columns"]) for name in MODELS}
    effects, factors_out, roles, edges_out = [], [], [], []
    for group in sorted(universe):
        r, cls = group.split(":")
        ids = groups[group]
        row = {"group_id": group, "residue": int(r), "contact_class": cls,
               "domain": dc._domain(int(r)), "status": "present" if ids else "absent",
               "contact_count": len(ids), "additional_zero_modes": 0,
               "rank_eligible": bool(old_edge_keys.get(group))}
        if not ids:
            row["missing_reason"] = "No original edges in this condition" if policy == "original_edges" else "No eligible incident edges"
            effects.append(row)
            continue
        row["edge_ids"] = ";".join(keys[i] for i in ids)
        row["shared_edge_group_ids"] = ";".join(sorted({g for i in ids for g in membership[i] if g != group}))
        row["identical_edge_group_ids"] = ";".join(g for g, js in groups.items() if g != group and ids == js)
        idx = int(np.flatnonzero(residues == int(r))[0])
        row["joint_degree"] = int(data["degrees"][idx])
        row["axis_distance_A"] = float(axis_distances[int(r)])
        cache = {}
        for name in MODELS:
            der = dc._exact_derivative(states[name], updates[name], ids)
            row.update({f"{name}_{k}": v for k, v in der.items()})
            row[f"{name}_derivative_log_S_close_per_edge"] = der["derivative_log_S_close"] / len(ids)
            cache[name] = {}
            for factor in factors:
                met = dc._perturbation_metrics(states[name], updates[name], ids, factor)
                cache[name][factor] = met
                factors_out.append({"group_id": group, "model": name, "spring_factor": factor, **met})
            row[f"{name}_D_g"] = (cache[name][1.1]["delta_log_S_close"] - cache[name][.9]["delta_log_S_close"]) / .2
            row[f"{name}_D_g_per_edge"] = row[f"{name}_D_g"] / len(ids)
        for factor in factors:
            rr = {"group_id": group, "spring_factor": factor}
            for role, num, den in (("R_body", "rigid", "fixed"), ("R_internal", "flexible", "rigid"), ("M", "flexible", "isolated")):
                for metric in METRICS:
                    rr[f"delta_{role}_delta_log_{metric}"] = cache[num][factor][f"delta_log_{metric}"] - cache[den][factor][f"delta_log_{metric}"]
            roles.append(rr)
        for role, num, den in (("R_body", "rigid", "fixed"), ("R_internal", "flexible", "rigid"), ("M", "flexible", "isolated")):
            row[f"delta_{role}_D_g"] = row[f"{num}_D_g"] - row[f"{den}_D_g"]
            row[f"delta_{role}_derivative_log_S_close"] = row[f"{num}_derivative_log_S_close"] - row[f"{den}_derivative_log_S_close"]
        effects.append(row)
    for name in MODELS:
        dc._rank_groups([row for row in effects if row["rank_eligible"]], name)
    for i in sorted(membership):
        pair = data["edges"][i]
        row = {"edge_id": keys[i], "i_node": pair[0], "j_node": pair[1],
               "distance_A": float(np.linalg.norm(system["coords"][pair[0]] - system["coords"][pair[1]])),
               "weight": float(data["weights"][i]), "group_ids": ";".join(sorted(membership[i])),
               "shared": len(membership[i]) > 1}
        for name in MODELS:
            row.update({f"{name}_{k}": v for k, v in dc._exact_derivative(states[name], updates[name], [i]).items()})
        edges_out.append(row)
    return {"groups": effects, "factors": factors_out, "roles": roles,
            "edges": edges_out, "updates": updates, "groups_indices": groups}


def verify_orders(original, expanded):
    checks = {}
    for window, states in (("original", original), ("expanded", expanded)):
        for left, right in (("isolated", "flexible"), ("flexible", "rigid"), ("rigid", "fixed")):
            checks[f"{window}_{left}_stiffness_le_{right}"] = dm._psd_diagnostic(states[left]["H"], states[right]["H"], MATRIX_TOL)
            a, b = states[right]["C_close"], states[left]["C_close"]
            checks[f"{window}_{right}_C_le_{left}"] = {"difference": b-a, "pass": a <= b + RESPONSE_TOL * max(abs(a), abs(b), 1)}
    for model in MODELS:
        checks[f"{model}_original_stiffness_le_expanded"] = dm._psd_diagnostic(original[model]["H"], expanded[model]["H"], MATRIX_TOL)
        a, b = expanded[model]["C_close"], original[model]["C_close"]
        checks[f"{model}_expanded_C_le_original"] = {"difference": b-a, "pass": a <= b + RESPONSE_TOL * max(abs(a), abs(b), 1)}
    if not all(bool(item["pass"]) for item in checks.values()):
        raise ArithmeticError(f"Matrix/compliance ordering failed: {checks}")
    return checks


def graph_diagnostics(system):
    pairs = system["pairs"]
    n = len(system["coords"])
    graph = csr_matrix((np.ones(2*len(pairs)), (np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])), shape=(n,n))
    nc = system["n_crbn"]
    nj = int(connected_components(graph, directed=False, return_labels=False))
    ni = int(connected_components(graph[:nc,:nc], directed=False, return_labels=False))
    if nj != 1 or ni != 1:
        raise ArithmeticError(f"Disconnected network: joint={nj}, isolated={ni}")
    return {"joint_components": nj, "isolated_components": ni,
            "n_edges": len(pairs), "n_interface_edges": int(np.sum(system["edge_types"] == "interface")),
            "nonnegative_weights": bool(np.all(system["weights"] > 0))}


def compare_legacy(pdb, cutoff, weighting, states):
    source = ROOT / "119_crbn_directional_mechanics_20260906" / "analysis/mechanics" / runner.condition_id(pdb, cutoff, weighting) / "models.csv"
    if not source.exists():
        source = ROOT / "results/directional_mechanics" / "analysis/mechanics" / runner.condition_id(pdb, cutoff, weighting) / "models.csv"
    if not source.exists():
        source = ROOT / "results/directional_mechanics" / "mechanics" / runner.condition_id(pdb, cutoff, weighting) / "models.csv"
    if not source.exists():
        return {"status": "historical_table_not_bundled", "public_rebuild_note": "No-extra regression is independently unit tested"}
    rows = read_table(source)
    errors = []
    for row in rows:
        for metric in METRICS:
            expected = float(row[metric])
            errors.append(abs(states[row["model"]][metric] / expected - 1))
    error = max(errors)
    if error > RESPONSE_TOL:
        raise ArithmeticError(f"Legacy mechanics regression {error}")
    return {"status": "verified", "maximum_relative_error": error,
            "source": str(source.relative_to(ROOT)), "sha256": runner.digest(source)}


def compare_legacy_contacts(pdb, cutoff, weighting, groups):
    key = runner.condition_id(pdb, cutoff, weighting)
    roots = (ROOT/"119_crbn_directional_mechanics_20260906/analysis",
             ROOT/"results/directional_mechanics/analysis", ROOT/"results/directional_mechanics")
    source = next((base/"contact_roles"/key/"groups.csv" for base in roots if (base/"contact_roles"/key/"groups.csv").exists()), None)
    if source is None:
        return {"status": "historical_table_not_bundled"}
    old = {r["group_id"]: r for r in read_table(source)}
    errors = []
    for row in groups:
        prior = old[row["group_id"]]
        if (row["status"] == "present") != (prior["status"] == "present"):
            raise ArithmeticError(f"Original contact presence changed: {row['group_id']}")
        if row["status"] != "present":
            continue
        if int(row["contact_count"]) != int(prior["contact_count"]):
            raise ArithmeticError(f"Original contact count changed: {row['group_id']}")
        for name in ("fixed","rigid","flexible"):
            errors.append(abs(float(row[f"{name}_D_g"])-float(prior[f"{name}_D_g"])))
    maximum = max(errors)
    if maximum > 1e-9:
        raise ArithmeticError(f"Original contact finite-difference regression {maximum}")
    return {"status": "verified", "maximum_D_g_absolute_error": maximum,
            "group_count": len(groups), "source": str(source.relative_to(ROOT)),
            "sha256": runner.digest(source)}


def verify_contact_update(system, states, data, tables, direction, extensive=False):
    """Direct block rebuilding plus derivative checks; no perturbed topology."""
    check_rows = []
    for cls in ("CRBN_DDB1", "HB_TBD"):
        present = [r for r in tables["groups"] if r["contact_class"] == cls and r["status"] == "present"]
        if not present:
            continue
        preferred = "221:CRBN_DDB1" if cls == "CRBN_DDB1" else "262:HB_TBD"
        selected = next((r for r in present if r["group_id"] == preferred), present[0])
        group = selected["group_id"]
        ids = tables["groups_indices"][group]
        columns = data["columns"][:, ids]
        for name in (MODELS if extensive else ("flexible",)):
            state = states[name]
            update = tables["updates"][name]
            exact = dc._exact_derivative(state, update, ids)
            low = dc._perturbation_metrics(state, update, ids, 1-1e-4)
            high = dc._perturbation_metrics(state, update, ids, 1+1e-4)
            derivative_error = max(abs((high[f"delta_log_{m}"]-low[f"delta_log_{m}"])/2e-4-exact[f"derivative_log_{m}"]) for m in METRICS)
            if derivative_error > 1e-7:
                raise ArithmeticError(f"Derivative check failed {group} {name}: {derivative_error}")
            for factor in ((.8,1.2) if extensive else (1.2,)):
                altered = changed_system(system, columns, factor)
                direct = make_state(altered, state["n_core"], direction, name, state["U"], state["q"])
                fast = dc._perturbation_metrics(state, update, ids, factor)
                errors = {f"{m}_relative_error": abs(fast[m]/direct[m]-1) for m in METRICS}
                if max(errors.values()) > RESPONSE_TOL:
                    raise ArithmeticError(f"Direct perturbation mismatch: {group}, {name}, {errors}")
                check_rows.append({"group_id": group, "model": name, "spring_factor": factor,
                                   **errors, "exact_vs_central_derivative_abs_error": derivative_error, "pass": True})
    return check_rows


def run_condition(pdb, cutoff, weighting, config, config_path, output, universe):
    started = time.monotonic()
    context = {"pdb": pdb, "cutoff_A": float(cutoff), "weighting": weighting,
               "reference_type": "apo" if pdb in config["apo_references"] else "engineered"}
    identifier = runner.condition_id(pdb, cutoff, weighting)
    target = output / "conditions" / identifier
    source_files = [Path(__file__), ROOT/"scripts/directional_mechanics.py", ROOT/"scripts/directional_contacts.py",
                    ROOT/"scripts/run_directional_mechanics.py", ROOT/"scripts/strengthen_contacts.py",
                    ROOT/"scripts/ddb1_complex_modes.py", ROOT/"scripts/curation_contracts.py",
                    ROOT/"scripts/hinge_geometry.py", ROOT/"data/crbn_ensemble.ens.npz",
                    ROOT/"data/pca_diffvec.npz", ROOT/"data/crbn_residue_window.csv",
                    ROOT/"data/directional_reference_inputs/candidate_universe.csv",
                    ROOT/"data/directional_reference_inputs/legacy_robustness.csv",
                    Path(sc.legacy.CIF_CACHE)/f"{pdb}.cif.gz"]
    inputs = {"condition": context, "config_sha256": runner.digest(config_path),
              "files": [{"path": str(p.relative_to(ROOT)), "sha256": runner.digest(p)} for p in source_files]}
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if runner.complete(target, signature):
        print(f"Verified cached window {identifier}", flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    case = load_window(pdb, True)
    n_core = len(case["core"])
    if n_core != config["window_extension"]["core_position_count"]:
        raise ValueError("Frozen core count changed")
    original_system = dm.build_system(case["original_xyz"], n_core, cutoff, weighting)
    original = make_states(original_system, n_core, case["direction"])
    expanded_system = dm.build_system(case["expanded_xyz"], len(case["expanded_residues"]), cutoff, weighting)
    expanded = make_states(expanded_system, n_core, case["direction"], original["isolated"]["U"], original["isolated"]["q"])
    verifications = {"original_graph": graph_diagnostics(original_system),
                     "expanded_graph": graph_diagnostics(expanded_system),
                     "orders": verify_orders(original, expanded),
                     "legacy": compare_legacy(pdb, cutoff, weighting, original)}
    # Original edges and their weights must survive the addition exactly.
    def all_edges(system, residues):
        return {edge_key(pair, residues, case["partner_residues"]): float(w)
                for pair, w in zip(system["pairs"], system["weights"])}
    old_edges = all_edges(original_system, case["core"])
    new_edges = all_edges(expanded_system, case["expanded_residues"])
    differences = [abs(w-new_edges.get(edge, float("inf"))) for edge,w in old_edges.items()]
    if max(differences) > 1e-10:
        raise ArithmeticError("Existing network edges or weights changed")
    verifications["original_edge_weight_max_abs_difference"] = max(differences)
    verifications["full_sequential_static_response"] = [verify_direct_response(expanded_system, state) for state in expanded.values()]
    frozen = runner.mapped_geometry(pdb, case["original_xyz"][:n_core], case["core"])
    geometry = dm.geometry_directions(case["original_xyz"][:n_core], case["core"], case["direction"], frozen,
                                      seed=config["seed"], n_draws=config["rotation_null"]["n_draws"])
    model_rows, comparison_rows = [], []
    for window, states in (("original", original), ("expanded", expanded)):
        models, comparisons, _ = runner.directional_statistics(states, geometry)
        extra = len(case["added"]) if window == "expanded" else 0
        for row in models:
            model_rows.append({**context, "window": window, "n_core": n_core,
                               "n_added": extra, "n_crbn": n_core+extra, **row})
        comparison_rows.extend({**context, "window": window, **row} for row in comparisons)
    write_table(target/"models.csv", model_rows)
    write_table(target/"comparisons.csv", comparison_rows)
    write_table(target/"coverage.csv", [{"reference_type": context["reference_type"], **row} for row in case["coverage"]])
    runner.write_json(target/"mapping.json", {"intervals": case["mapping"], "excluded_observed": case["excluded_observed"],
                                             "core_order": case["core"], "added_order": case["added"]})
    print(f"Window mechanics {identifier}: added={len(case['added'])}; "
          f"S_iso {original['isolated']['S_close']:.6g}->{expanded['isolated']['S_close']:.6g}; "
          f"S_flex {original['flexible']['S_close']:.6g}->{expanded['flexible']['S_close']:.6g}", flush=True)
    original_data = candidate_data(original_system, case["core"], case["partner_residues"], universe)
    old_group_keys = {g: {original_data["keys"][i] for i in ids} for g,ids in original_data["groups"].items()}
    axis_distances = {int(r): float(d) for r,d in zip(case["core"], case["axis_distances"])}
    all_groups, all_factors, all_roles, all_edge_effects, direct_checks = [], [], [], [], []
    for window, system, states, residues in (("original", original_system, original, case["core"]),
                                             ("expanded", expanded_system, expanded, case["expanded_residues"])):
        data = original_data if window == "original" else candidate_data(system, residues, case["partner_residues"], universe)
        for policy in (("original_edges",) if window == "original" else POLICIES):
            tables = contact_tables(system, states, data, universe, residues, old_group_keys, policy, axis_distances,
                                    tuple(config["contact"]["spring_factors"]))
            if window == "original":
                verifications["legacy_contacts"] = compare_legacy_contacts(pdb, cutoff, weighting, tables["groups"])
            tag = {**context, "window": window, "policy": policy}
            all_groups.extend({**tag, **r} for r in tables["groups"])
            all_factors.extend({**tag, **r} for r in tables["factors"])
            all_roles.extend({**tag, **r} for r in tables["roles"])
            all_edge_effects.extend({**tag, **r} for r in tables["edges"])
            if window == "expanded":
                extensive = pdb == "8CVP" and cutoff == 15 and weighting == "uniform"
                direct_checks.extend({"window": window, "policy": policy, **r} for r in
                                     verify_contact_update(system, states, data, tables, case["direction"], extensive))
            del tables
    write_table(target/"group_effects.csv.gz", all_groups)
    write_table(target/"factor_effects.csv.gz", all_factors)
    write_table(target/"role_factor_effects.csv.gz", all_roles)
    write_table(target/"edge_effects.csv.gz", all_edge_effects)
    verifications["direct_contact_checks"] = direct_checks
    verifications["metrics"] = [{"window": window, "model": name,
                                  "condition_number_core": s["condition_number_core"],
                                  "inverse_residual_inf": s["inverse_residual_inf"],
                                  "minimum_core_eigenvalue": float(s["eigenvalues"][0])}
                                 for window, states in (("original",original),("expanded",expanded)) for name,s in states.items()]
    verifications["nullspace_rule"] = "All positive spring factors preserve the baseline nullspace on unchanged topology; baseline core and nuisance blocks are positive definite."
    verifications["all_verification_pass"] = True
    verifications["elapsed_seconds"] = time.monotonic()-started
    runner.write_json(target/"verification.json", verifications)
    runner.write_json(target/"inputs.json", inputs)
    runner.finish(target, signature, time.time()-verifications["elapsed_seconds"])
    print(f"Completed window {identifier} in {verifications['elapsed_seconds']:.1f}s", flush=True)


def _truth(value):
    return str(value).lower() in ("true", "1", "yes")


def robustness_table(group_rows, config):
    legacy = {f"{r['residue']}:{r['contact_class']}": r for r in
              read_table(ROOT/"data/directional_reference_inputs/legacy_robustness.csv")}
    by_key = {(r["pdb"], float(r["cutoff_A"]), r["weighting"], r["window"], r["policy"], r["group_id"]): r for r in group_rows}
    required = [("8CVP", c, "uniform") for c in (13.,15.,18.)] + [(p,15.,"uniform") for p in config["apo_references"] if p != "8CVP"]
    engineered = [(p,15.,"uniform") for p in config["engineered_references"]]
    weighted = [(p,float(c),"inverse_square") for p in config["references"] for c in config["cutoffs_A"]]
    rows = []
    for policy in POLICIES:
        for group, old in sorted(legacy.items()):
            discovery = by_key.get(("8CVP",15.,"uniform","expanded",policy,group))
            sign = np.sign(float(discovery["flexible_D_g"])) if discovery and discovery["status"] == "present" else 0
            results = {}
            for p,c,w in required+engineered+weighted:
                row = by_key.get((p,c,w,"expanded",policy,group))
                if not row or row["status"] != "present" or row.get("flexible_rank_fraction", "") == "":
                    result = "absent"
                elif float(row["flexible_rank_fraction"]) <= .20 and np.sign(float(row["flexible_D_g"])) == sign and sign != 0:
                    result = "pass"
                else:
                    result = "fail"
                results[(p,c,w)] = result
            def passed(conditions, condition_results=results):
                return all(condition_results[x] == "pass" for x in conditions)
            original_sign = np.sign(float(old["discovery_D_g"]))
            rows.append({"group_id": group, "residue": int(old["residue"]), "contact_class": old["contact_class"],
                         "policy": policy, "legacy_stable_apo": _truth(old["stable_apo_model_candidate"]),
                         "legacy_engineered": _truth(old["also_consistent_in_engineered_references"]),
                         "expanded_apo_stable": passed(required),
                         "expanded_engineered_consistent": passed(required+engineered),
                         "expanded_all_weighted_stable": passed(required+engineered+weighted),
                         "expanded_discovery_sign": int(sign),
                         "same_sign_as_original_discovery": bool(sign != 0 and sign == original_sign),
                         "expanded_discovery_D_g": discovery.get("flexible_D_g", "") if discovery else "",
                         "condition_results": ";".join(f"{p}:{c:g}:{w}={v}" for (p,c,w),v in results.items())})
    return rows


def consolidate(output, config, expected_conditions):
    paths = [output/"conditions"/runner.condition_id(p,c,w) for p,c,w in expected_conditions]
    models, comparisons, groups = [], [], []
    coverage = {}
    records = []
    for directory in paths:
        verification = json.loads((directory/"verification.json").read_text())
        if not verification["all_verification_pass"]:
            raise ArithmeticError(f"Unverified condition {directory}")
        models.extend(read_table(directory/"models.csv"))
        comparisons.extend(read_table(directory/"comparisons.csv"))
        groups.extend(read_table(directory/"group_effects.csv.gz"))
        for row in read_table(directory/"coverage.csv"):
            coverage[(row["pdb"], int(row["residue"]))] = row
        records.append({"condition": directory.name, "elapsed_seconds": verification["elapsed_seconds"],
                        "verification_sha256": runner.digest(directory/"verification.json")})
    write_table(output/"models.csv", models)
    write_table(output/"comparisons.csv", comparisons)
    write_table(output/"coverage.csv", coverage.values())
    write_table(output/"group_effects.csv", groups)
    robust = robustness_table(groups, config)
    write_table(output/"robustness.csv", robust)
    table_index = []
    for directory in paths:
        for filename in ("factor_effects.csv.gz", "role_factor_effects.csv.gz", "edge_effects.csv.gz"):
            path = directory/filename
            with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                row_count = sum(1 for _ in reader)
            table_index.append({"condition": directory.name, "table": filename.removesuffix(".csv.gz") if hasattr(str,"removesuffix") else filename[:-7],
                                "path": str(path.relative_to(output)), "row_count": row_count,
                                "compressed_bytes": path.stat().st_size, "sha256": runner.digest(path),
                                "columns": ";".join(fields)})
    write_table(output/"table_index.csv", table_index)
    # Older pilot aggregates are redundant with the indexed condition tables.
    for filename in ("factor_effects.csv.gz", "role_factor_effects.csv.gz", "edge_effects.csv.gz"):
        stale = output/filename
        if stale.exists():
            stale.unlink()
    primary = [r for r in comparisons if r["window"] == "expanded" and r["weighting"] == "uniform"
               and float(r["cutoff_A"]) == 15 and r["target"] == "finite" and r["pdb"] in config["apo_references"]]
    primary_map = {(r["pdb"],r["role"]):float(r["effect"]) for r in primary}
    complete = len(paths) == len(config["references"])*len(config["cutoffs_A"])*len(config["weightings"])
    gate = complete and all(primary_map[(p,"R_body")] > primary_map[(p,"R_internal")] > 1e-7
                            and primary_map[(p,"M")] < -1e-7 for p in config["apo_references"])
    summary = {"status": "complete" if complete else "partial", "condition_count": len(paths),
               "all_verification_pass": True, "expected_condition_count": 30,
               "core_position_count": 269, "core_internal_dimension": 801,
               "model_count": len(models), "contact_group_row_count": len(groups),
               "sum_condition_seconds": sum(r["elapsed_seconds"] for r in records),
               "body_dominant_partial_relief_primary_apo_gate": bool(gate),
               "interpretation": "Static responses measured only in the same original core; no dynamical frequency claim for Schur stiffnesses.",
               "rank_pool": "Original discovery142 IDs and original per-condition present-group eligibility; absent effects are never zero.",
               "robustness_rule": "Expanded discovery sign retained and flexible rank fraction <=0.20 in the required apo conditions; old flags remain separate.",
               "conditions": records,
               "legacy_eight_retained": {policy: [r["group_id"] for r in robust if r["policy"]==policy and r["legacy_stable_apo"] and r["expanded_apo_stable"]] for policy in POLICIES},
               "expanded_apo_stable_count": {policy: sum(r["expanded_apo_stable"] for r in robust if r["policy"]==policy) for policy in POLICIES}}
    runner.write_json(output/"summary.json", summary)
    return summary


def run(config_path, output_dir, offline=True, *, references=None, cutoffs=None, weightings=None):
    """Reproduce paired mechanics and both contact policies, with verified resume."""
    config_path, output = Path(config_path).resolve(), Path(output_dir).resolve()
    config = json.loads(config_path.read_text())
    if not offline:
        # Online acquisition is owned by the external-data runner. Reuse these
        # frozen coordinate inputs; never replace them inside this analysis.
        print("Window analysis uses the frozen local coordinate inputs", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    discovery = read_table(ROOT/"data/directional_reference_inputs/candidate_universe.csv")
    universe = {f"{r['residue']}:{r['contact_class']}" for r in discovery}
    if len(universe) != config["contact"]["candidate_count"]:
        raise ValueError("Frozen contact universe count differs")
    selected = [(p,float(c),w) for p in (references or config["references"])
                for c in (cutoffs or config["cutoffs_A"]) for w in (weightings or config["weightings"])]
    for pdb,cutoff,weighting in selected:
        run_condition(pdb,cutoff,weighting,config,config_path,output,universe)
    return consolidate(output,config,selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"scripts/review_extensions_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--references", nargs="+")
    parser.add_argument("--cutoffs", nargs="+", type=float)
    parser.add_argument("--weightings", nargs="+")
    args = parser.parse_args()
    print(json.dumps(run(args.config,args.output_dir,args.offline,references=args.references,
                         cutoffs=args.cutoffs,weightings=args.weightings), indent=2))


if __name__ == "__main__":
    main()
