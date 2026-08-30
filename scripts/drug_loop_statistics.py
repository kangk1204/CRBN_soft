#!/usr/bin/env python3
"""How strongly can the ligand-site mobility claim be worded?

The pre-specified UniProt ligand annotations (378/380/386) are not an exhaustive
definition of the thalidomide-binding pocket.  This analysis therefore keeps that trio
as the primary annotation-based comparison and reports two structural sensitivity
definitions separately: the annotation trio plus W400/F402, and the seven residues in
the 5FQD LVY heavy-atom contact shell that are present in the common analysis window.
The four additional 5FQD contacts in the sensor loop (350--353) are absent from that
window and are recorded as missing rather than silently discarded.

Three objections must be met before any residue-set comparison can stand:

(a) SMALL SAMPLE. For 3 vs 4 residues, the minimum attainable exact two-sided
    Mann-Whitney p is 2/C(7,3) = 0.0571 (one-sided minimum 0.0286). Reporting the test
    without its discreteness overstates its resolution, so both tails and both floors
    are computed here.

(b) THE RIGHT NULL IS DOMAIN-MATCHED. Both residue sets lie in the TBD. The question is
    not "are these residues mobile relative to the whole protein" (they are, trivially,
    because the TBD is the mobile lobe) but "are they mobile relative to other TBD
    residues". A permutation null over random same-size TBD sets answers exactly that.

(c) MOBILITY MEASURE CONFOUNDS. Two are tested. (i) Contact number: a residue with few
    neighbours fluctuates more in any elastic network, so a mobility claim may merely
    restate surface exposure; the percentiles are recomputed after regressing mobility
    on contact number. (ii) Crystallographic B-factors are the EXPERIMENTAL measure of
    local mobility; if the model profile simply tracks them, the claim adds nothing.
    Both are computed on three profiles: the ANM 10-mode fluctuation, the ensemble PCA
    fluctuation, and the experimental B-factor profile from the X-ray subset.

Percentiles use one explicit weak empirical convention throughout:
``100 * mean(profile <= selected_value)``.  The result determines the wording.  The
annotation-trio result is definition-specific; structural-shell sensitivity results
must accompany it.  The B-factor comparison is a null result and is reported as such.

Inputs   data/crbn_ensemble.ens.npz, data/crbn_residue_window.csv,
         data/crbn_curation_log.csv, data/crbn_residue_fluctuations.csv,
         RCSB mmCIF (B-factors; cached in data/_cif_cache)
Outputs  data/drug_loop_statistics.json
Usage    python scripts/drug_loop_statistics.py [--verify] [--no-network]
"""
import argparse
import csv
import gzip
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmode_lib as L
import reproduce_tensor as R
from analysis_contracts import assert_tree_close, atomic_write_json

UNIPROT_LIGAND_ANNOTATIONS = (378, 380, 386)
ANNOTATED_PLUS_W400_F402 = (378, 380, 386, 400, 402)
CONTACT_SHELL_COMMON_WINDOW = (377, 378, 379, 380, 386, 400, 402)
CONTACT_SHELL_SENSOR_LOOP_MISSING = (350, 351, 352, 353)
CONTACT_SHELL_FULL_5FQD = (
    *CONTACT_SHELL_SENSOR_LOOP_MISSING,
    *CONTACT_SHELL_COMMON_WINDOW,
)
POCKET_DEFINITIONS = {
    "uniprot_ligand_annotations": UNIPROT_LIGAND_ANNOTATIONS,
    "annotated_plus_W400_F402": ANNOTATED_PLUS_W400_F402,
    "5fqd_4.5A_contact_shell_common_window": CONTACT_SHELL_COMMON_WINDOW,
}
# Backwards-compatible name for the pre-specified annotation trio.  It must not be
# interpreted as an exhaustive pocket definition.
DRUG = list(UNIPROT_LIGAND_ANNOTATIONS)
ZN = [323, 326, 391, 394]                 # structural zinc site
PERCENTILE_CONVENTION = "100 * mean(profile <= selected value)"
TBD = (318, 426)
CUTOFF = 15.0
NMODES_FLUCT = 10
NDRAW = 20000
SEED = 20260720
REF = "8CVP"
MIN_BFACTOR_COVERAGE = 0.9


def validate_core_inputs(ensemble, resnums, curation_rows, fluctuation_rows):
    """Validate the shared conformer, residue, and reference-profile ordering."""
    required = {"_confs", "_labels"}
    missing = required - set(ensemble.files)
    if missing:
        raise ValueError(f"ensemble artifact is missing required arrays: {sorted(missing)}")

    conformers = np.asarray(ensemble["_confs"], dtype=float)
    raw_labels = np.asarray(ensemble["_labels"])
    if (
        conformers.ndim != 3
        or conformers.shape[0] < 2
        or conformers.shape[1] < 1
        or conformers.shape[2] != 3
        or not np.isfinite(conformers).all()
    ):
        raise ValueError(
            "ensemble coordinates must be a finite n x residues x 3 array, "
            f"found {conformers.shape}"
        )
    if raw_labels.shape != (conformers.shape[0],):
        raise ValueError(
            f"ensemble labels must have shape ({conformers.shape[0]},), "
            f"found {raw_labels.shape}"
        )
    labels = [str(value).strip().upper() for value in raw_labels]
    if any(len(label) != 4 or not label.isalnum() for label in labels):
        raise ValueError("ensemble labels must be four-character alphanumeric PDB identifiers")
    if len(labels) != len(set(labels)):
        raise ValueError("ensemble labels must be unique")
    if labels.count(REF) != 1:
        raise ValueError(f"ensemble must contain exactly one {REF} reference conformer")

    residues = np.asarray(resnums)
    if residues.ndim != 1 or residues.shape[0] != conformers.shape[1]:
        raise ValueError(
            "residue labels must be one-dimensional and match the coordinate order, "
            f"found {residues.shape} for {conformers.shape[1]} residues"
        )
    if residues.dtype.kind not in "iu":
        raise ValueError(f"residue labels must be integers, found dtype={residues.dtype}")
    residues = residues.astype(int, copy=False)
    if len(np.unique(residues)) != len(residues):
        raise ValueError("residue labels must be unique")
    if np.any(np.diff(residues) <= 0):
        raise ValueError("residue labels must be strictly increasing in coordinate order")
    required_residues = set(ZN)
    for definition in POCKET_DEFINITIONS.values():
        required_residues.update(definition)
    missing_residues = sorted(required_residues - set(residues.tolist()))
    if missing_residues:
        raise ValueError(f"functional residues are missing from the analysis window: {missing_residues}")

    methods = {}
    allowed_methods = {"X-ray", "cryo-EM"}
    for row in curation_rows:
        pdb = str(row.get("pdb", "")).strip().upper()
        method = str(row.get("method", "")).strip()
        if not pdb or not method or pdb in methods:
            raise ValueError(f"invalid or duplicate curation row for {pdb!r}")
        if method not in allowed_methods:
            raise ValueError(f"invalid structure method for {pdb}: {method!r}")
        methods[pdb] = method
    if set(methods) != set(labels):
        raise ValueError(
            "curation labels do not exactly match the ensemble; "
            f"missing={sorted(set(labels) - set(methods))}, "
            f"extra={sorted(set(methods) - set(labels))}"
        )

    if len(fluctuation_rows) != len(residues):
        raise ValueError(
            f"reference fluctuation table has {len(fluctuation_rows)} rows; "
            f"expected {len(residues)}"
        )
    try:
        reference_residues = np.array(
            [int(row["resnum"]) for row in fluctuation_rows],
            dtype=int,
        )
        reference_anm = np.array(
            [float(row["anm_sqfluct"]) for row in fluctuation_rows],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reference fluctuation table has invalid required fields") from exc
    if not np.array_equal(reference_residues, residues):
        raise ValueError("reference fluctuation residue order does not match the analysis window")
    if not np.isfinite(reference_anm).all():
        raise ValueError("reference ANM fluctuation profile contains non-finite values")
    return conformers, labels, residues, methods, reference_anm


def percentile_of(values, idx, universe=None):
    """Mean weak empirical percentile for the residues at ``idx``.

    For each selected value ``x``, the percentile is
    ``100 * mean(universe_values <= x)``.  Including ties and the selected residue
    matches ``context_stats.py`` and the Figure 4 source-data convention.
    """
    u = np.arange(len(values)) if universe is None else np.asarray(universe)
    v = values[u]
    pct = []
    for k in idx:
        pct.append(100.0 * (v <= values[k]).mean())
    return float(np.mean(pct))


def mannwhitney_one_sided(a, b):
    """Exact Mann-Whitney by full label enumeration.

    The return keeps the historical one-sided ``P(a > b)`` value and also reports the
    symmetric two-sided tail used by ``context_stats.py`` and Figure 4.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, m = len(a), len(b)
    U = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    pooled = np.concatenate([a, b])
    ge = extreme = tot = 0
    permutation_u = []
    centre = n * m / 2.0
    observed_deviation = abs(U - centre)
    for c in combinations(range(n + m), n):
        aa = pooled[list(c)]
        bb = pooled[[i for i in range(n + m) if i not in c]]
        u = sum((x > y) + 0.5 * (x == y) for x in aa for y in bb)
        tot += 1
        permutation_u.append(float(u))
        if u >= U:
            ge += 1
        if abs(u - centre) >= observed_deviation - 1e-12:
            extreme += 1
    two_sided_floor = min(
        sum(abs(candidate - centre) >= abs(value - centre) - 1e-12
            for candidate in permutation_u) / tot
        for value in permutation_u
    )
    return {
        "U": float(U),
        "p_one_sided": ge / tot,
        "p_two_sided": min(1.0, extreme / tot),
        "n_permutations": tot,
        # Historical alias is the one-sided floor.
        "p_minimum_attainable": 1.0 / tot,
        "p_minimum_attainable_one_sided": 1.0 / tot,
        "p_minimum_attainable_two_sided": two_sided_floor,
        "n_a": n,
        "n_b": m,
    }


def residualise(y, x):
    """y after removing its linear dependence on x (OLS)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def matched_set_null(profile, tbd_idx, obs_idx, ndraw=NDRAW, seed=SEED):
    """Empirical p against random same-size residue sets drawn from the TBD."""
    rng = np.random.default_rng(seed)
    ranks = np.array([100.0 * (profile <= v).mean() for v in profile])
    obs = float(ranks[obs_idx].mean())
    draws = np.array([ranks[rng.choice(tbd_idx, len(obs_idx), replace=False)].mean()
                      for _ in range(ndraw)])
    n_ex = int((draws >= obs).sum())
    return {"observed_percentile": obs, "null_mean": float(draws.mean()),
            "null_sd": float(draws.std(ddof=1)), "n_draws": ndraw,
            "n_exceedances": n_ex, "p_empirical": (n_ex + 1) / (ndraw + 1),
            "z": float((obs - draws.mean()) / draws.std(ddof=1))}


def trio_null(profile, tbd_idx, obs_idx, ndraw=NDRAW, seed=SEED):
    """Compatibility wrapper for the three-residue annotation-set null."""
    if len(obs_idx) != 3:
        raise ValueError("trio_null requires exactly three observed residues")
    return matched_set_null(profile, tbd_idx, obs_idx, ndraw=ndraw, seed=seed)


def fetch_bfactor_cif(pdb, cache_only=False):
    """Read a mmCIF from the local analysis cache, optionally without network fallback."""
    path = R.CACHE / f"{pdb.upper()}.cif.gz"
    if cache_only:
        if not path.exists():
            raise FileNotFoundError(f"{pdb}: cached mmCIF not found at {path}")
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return R.fetch_cif(pdb)


def choose_crbn_chain_for_bfactor(ca, pdb, resnums):
    """Use the same metadata-validated CRBN-chain rule as the coordinate tensor."""
    chains = R.crbn_chains(pdb)
    if not chains:
        raise RuntimeError(f"{pdb}: no CRBN entity chain in data/_rcsb_meta.json")
    ch, n = R.select_chain(ca, pdb)
    min_n = int(np.ceil(MIN_BFACTOR_COVERAGE * len(resnums)))
    if n < min_n:
        raise RuntimeError(f"{pdb}: CRBN chain {ch} resolves {n}/{len(resnums)} "
                           f"window residues, below expected {min_n}/{len(resnums)}")
    return ch, n, chains


def bfactor_profile(labels, resnums, methods, cache_only=False, verbose=True):
    """Ensemble-average normalised Ca B-factor profile from the X-RAY entries only.

    Cryo-EM entries are excluded: the B field there is an ADP/blur factor from a
    different refinement target and is not comparable to a crystallographic B-factor.
    Each structure is z-scored over the window before averaging so that per-structure
    scaling and resolution do not dominate.
    """
    prof, used, chain_rows, alt, occ = [], [], [], 0, 0
    for pdb in labels:
        if methods.get(pdb) != "X-ray":
            continue
        ca, stats = L.parse_atoms(fetch_bfactor_cif(pdb, cache_only=cache_only))
        ch, n, chains = choose_crbn_chain_for_bfactor(ca, pdb, resnums)
        b = np.array([ca[ch][r]["b"] if r in ca[ch] else np.nan for r in resnums])
        finite_n = int(np.isfinite(b).sum())
        min_n = int(np.ceil(MIN_BFACTOR_COVERAGE * len(resnums)))
        if finite_n < min_n:
            raise RuntimeError(f"{pdb}: CRBN chain {ch} has finite B-factors for "
                               f"{finite_n}/{len(resnums)} window residues, below "
                               f"expected {min_n}/{len(resnums)}")
        alt += sum(1 for r in resnums if r in ca[ch] and ca[ch][r]["alt"] not in (".", "?"))
        occ += sum(1 for r in resnums if r in ca[ch] and ca[ch][r]["occ"] < 0.999)
        z = (b - np.nanmean(b)) / np.nanstd(b)
        prof.append(z)
        used.append(pdb)
        chain_rows.append({"pdb": pdb, "chain": ch, "coverage": int(n),
                           "crbn_entity_chains": list(chains)})
    if not prof:
        return None, used, {"n_altloc_ca": alt, "n_partial_occupancy_ca": occ,
                            "chain_selection": chain_rows}
    P = np.array(prof)
    return np.nanmean(P, 0), used, {"n_altloc_ca": alt, "n_partial_occupancy_ca": occ,
                                    "chain_selection": chain_rows,
                                    "chain_selection_rule":
                                        "metadata-validated reproduce_tensor.select_chain: "
                                        "use committed primary-chain override where present, "
                                        "otherwise lowest exact-Q96SW2 auth-chain id"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="recompute and compare without writing")
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="require every crystallographic source record to be cached",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verify = bool(args.verify)
    L.CACHE_WRITES_ENABLED = not verify
    R.CACHE_WRITES_ENABLED = not verify
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    with open("data/crbn_residue_window.csv", encoding="utf-8", newline="") as handle:
        window_rows = list(csv.DictReader(handle))
    try:
        resnums = np.array([int(row["author_resnum"]) for row in window_rows], dtype=int)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("analysis-window table has invalid author_resnum values") from exc
    with open("data/crbn_curation_log.csv", encoding="utf-8", newline="") as handle:
        curation_rows = list(csv.DictReader(handle))
    with open("data/crbn_residue_fluctuations.csv", encoding="utf-8", newline="") as handle:
        fluctuation_rows = list(csv.DictReader(handle))
    confs, labels, resnums, methods, anm_comm = validate_core_inputs(
        ens,
        resnums,
        curation_rows,
        fluctuation_rows,
    )
    idx = {int(r): k for k, r in enumerate(resnums)}
    definition_indices = {
        name: [idx[r] for r in residues]
        for name, residues in POCKET_DEFINITIONS.items()
    }
    zn_i = [idx[r] for r in ZN]
    tbd_i = np.array([k for k, r in enumerate(resnums) if TBD[0] <= r <= TBD[1]])

    ref = confs[labels.index(REF)]
    w, V = L.modes(L.anm_hessian(ref, CUTOFF), 20)
    if (
        w.ndim != 1
        or w.shape[0] < NMODES_FLUCT
        or V.shape != (ref.size, w.shape[0])
        or not np.isfinite(w).all()
        or not np.isfinite(V).all()
        or np.any(w <= 0)
    ):
        raise ValueError(f"invalid ANM eigensystem: eigenvalues={w.shape}, modes={V.shape}")
    anm_f = L.sqfluct(w, V, NMODES_FLUCT)
    # ensemble PCA fluctuation: eigenvalue-weighted over the top 10 PCs
    P, var, scores = L.ensemble_pca(confs)
    if (
        P.ndim != 2
        or P.shape[0] != ref.size
        or P.shape[1] < 10
        or scores.ndim != 2
        or scores.shape[0] != len(confs)
        or scores.shape[1] < 10
        or not np.isfinite(P).all()
        or not np.isfinite(var).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError(
            f"invalid PCA result: modes={P.shape}, variance={var.shape}, scores={scores.shape}"
        )
    ev = (scores ** 2).sum(0) / (len(confs) - 1)
    pca_f = np.zeros(len(resnums))
    for m in range(10):
        pca_f += ev[m] * (P[:, m].reshape(-1, 3) ** 2).sum(1)
    # contact number at the ANM cutoff
    ci, cj, _ = L.contact_pairs(ref, CUTOFF)
    contact = np.bincount(np.concatenate([ci, cj]), minlength=len(ref)).astype(float)

    for name, profile in {
        "ANM fluctuation": anm_f,
        "PCA fluctuation": pca_f,
        "contact count": contact,
    }.items():
        values = np.asarray(profile)
        if values.shape != resnums.shape or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite and match the residue order")

    # committed profiles must match the recomputed ones
    rho_check = L.spearman(anm_f, anm_comm)[0]

    out = {"meta": {
        "primary_definition": "uniprot_ligand_annotations",
        "primary_definition_scope": (
            "pre-specified UniProt (S)-thalidomide ligand annotations; not an exhaustive "
            "structural pocket definition"
        ),
        # Kept for readers of the previous schema; its scope is made explicit above.
        "drug_residues": DRUG,
        "pocket_definitions_common_window": {
            name: list(residues) for name, residues in POCKET_DEFINITIONS.items()
        },
        "5fqd_4.5A_contact_shell_full": list(CONTACT_SHELL_FULL_5FQD),
        "5fqd_contact_residues_missing_from_common_window":
            list(CONTACT_SHELL_SENSOR_LOOP_MISSING),
        "missing_contact_reason": (
            "residues 350-353 lie in the sensor-loop segment omitted from the common "
            "269-residue window because the genuine-apo open structures do not resolve it"
        ),
        "percentile_convention": PERCENTILE_CONVENTION,
        "zn_residues": ZN,
        "tbd_window": list(TBD),
        "n_tbd_residues": int(len(tbd_i)),
        "cutoff": CUTOFF,
        "n_modes_fluctuation": NMODES_FLUCT,
        "n_draws": NDRAW,
        "seed": SEED,
        "reference": REF,
        "spearman_recomputed_vs_committed_anm_profile": rho_check,
    }}

    # ------------------------------------------------ B-factor profile ---
    no_network = bool(args.no_network)
    bprof, bpdbs, bstats = bfactor_profile(
        labels,
        resnums,
        methods,
        cache_only=no_network,
    )
    if bprof is None:
        raise RuntimeError("no complete X-ray B-factor profiles were available")

    profiles = {"anm_fluctuation": anm_f, "pca_fluctuation": pca_f}
    if np.asarray(bprof).shape != resnums.shape or not np.isfinite(bprof).all():
        raise ValueError("experimental B-factor profile must be finite and match residue order")
    profiles["experimental_bfactor"] = bprof
    out["bfactor_provenance"] = {
        "n_xray_entries_used": len(bpdbs), "entries": bpdbs,
        "normalisation": "per-structure z-score over the 269-residue window, then averaged",
        "cryoem_excluded": True,
        "cryoem_exclusion_reason": "the B field in cryo-EM depositions is an ADP/blur "
                                   "factor from a different refinement target and is not "
                                   "comparable to a crystallographic B-factor",
        **bstats}

    # ------------------------------- definition-specific percentile comparison ---
    print("weak empirical percentile of each residue definition in each mobility profile:")
    comparison_by_definition = {}
    for definition_name, selected_i in definition_indices.items():
        selected_residues = list(POCKET_DEFINITIONS[definition_name])
        definition_comparison = {}
        print(f"  [{definition_name}] residues={selected_residues}")
        for profile_name, prof in profiles.items():
            d_win = percentile_of(prof, selected_i)
            z_win = percentile_of(prof, zn_i)
            d_tbd = percentile_of(prof, selected_i, tbd_i)
            z_tbd = percentile_of(prof, zn_i, tbd_i)
            matched_null = matched_set_null(prof, tbd_i, selected_i)
            result = {
                "definition": definition_name,
                "residues": selected_residues,
                "group_percentile_window": d_win,
                "zn_percentile_window": z_win,
                "group_percentile_within_tbd": d_tbd,
                "zn_percentile_within_tbd": z_tbd,
                "difference_window": d_win - z_win,
                "group_values": [float(prof[k]) for k in selected_i],
                "zn_values": [float(prof[k]) for k in zn_i],
                "mannwhitney": mannwhitney_one_sided(prof[selected_i], prof[zn_i]),
                "matched_set_null_tbd": matched_null,
            }
            if definition_name == "uniprot_ligand_annotations":
                # Legacy field names remain aliases for the explicitly scoped trio.
                result.update({
                    "drug_percentile_window": d_win,
                    "drug_percentile_within_tbd": d_tbd,
                    "drug_values": result["group_values"],
                    "trio_null_tbd_matched": matched_null,
                })
            definition_comparison[profile_name] = result
            print(
                f"    {profile_name:22s} group {d_win:5.1f} vs Zn {z_win:5.1f} "
                f"(window) | within-TBD {d_tbd:5.1f} vs {z_tbd:5.1f} | "
                f"set-null p = {matched_null['p_empirical']:.4f} | "
                f"MW p2 = {result['mannwhitney']['p_two_sided']:.3f}"
            )
        comparison_by_definition[definition_name] = definition_comparison
    comparison = comparison_by_definition["uniprot_ligand_annotations"]
    out["definition_comparison"] = comparison_by_definition
    # Backwards-compatible alias: this is the UniProt annotation trio only.
    out["three_way_comparison"] = comparison

    # --------------------------------------------- contact-number confound ---
    conf_by_definition = {}
    for definition_name, selected_i in definition_indices.items():
        definition_conf = {}
        for profile_name, prof in profiles.items():
            rho, p, n = L.spearman(contact, prof)
            res = residualise(prof, contact)
            contact_null = matched_set_null(res, tbd_i, selected_i)
            result = {
                "definition": definition_name,
                "residues": list(POCKET_DEFINITIONS[definition_name]),
                "spearman_contact_vs_profile": {"rho": rho, "p": p, "n": n},
                "group_percentile_after_contact_residualisation":
                    percentile_of(res, selected_i),
                "zn_percentile_after_contact_residualisation": percentile_of(res, zn_i),
                "matched_set_null_after_contact_residualisation": contact_null,
            }
            if bprof is not None and profile_name != "experimental_bfactor":
                resb = residualise(
                    prof,
                    np.nan_to_num(bprof, nan=float(np.nanmean(bprof))),
                )
                bfactor_null = matched_set_null(resb, tbd_i, selected_i)
                result.update({
                    "group_percentile_after_bfactor_residualisation":
                        percentile_of(resb, selected_i),
                    "zn_percentile_after_bfactor_residualisation":
                        percentile_of(resb, zn_i),
                    "matched_set_null_after_bfactor_residualisation": bfactor_null,
                })
            if definition_name == "uniprot_ligand_annotations":
                result.update({
                    "drug_percentile_after_contact_residualisation":
                        result["group_percentile_after_contact_residualisation"],
                    "trio_null_after_contact_residualisation": contact_null,
                })
                if "group_percentile_after_bfactor_residualisation" in result:
                    result.update({
                        "drug_percentile_after_bfactor_residualisation":
                            result["group_percentile_after_bfactor_residualisation"],
                        "trio_null_after_bfactor_residualisation":
                            result["matched_set_null_after_bfactor_residualisation"],
                    })
            definition_conf[profile_name] = result
        conf_by_definition[definition_name] = definition_conf
    conf = conf_by_definition["uniprot_ligand_annotations"]
    out["confound_control_by_definition"] = conf_by_definition
    # Backwards-compatible alias: UniProt annotation trio only.
    out["confound_control"] = conf
    print("contact-number confound:")
    for name, c in conf.items():
        s = c["spearman_contact_vs_profile"]
        print(f"  {name:22s} rho(contact, mobility) = {s['rho']:+.3f} "
              f"(p = {s['p']:.1e}); after residualisation annotations "
              f"{c['drug_percentile_after_contact_residualisation']:.1f} vs Zn "
              f"{c['zn_percentile_after_contact_residualisation']:.1f}")

    # ------------------------------- cross-correlation among the profiles ---
    cross = {}
    names = list(profiles)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            rho, p, n = L.spearman(profiles[names[a]], profiles[names[b]])
            cross[f"{names[a]}__vs__{names[b]}"] = {"rho": rho, "p": p, "n": n}
    out["profile_cross_correlation"] = cross
    print("profile cross-correlation:")
    for k, v in cross.items():
        print(f"  {k:52s} rho = {v['rho']:+.3f}")

    # --------------------------------------------- multiple comparisons ---
    tests = [("trio null, ANM", comparison["anm_fluctuation"]
              ["trio_null_tbd_matched"]["p_empirical"]),
             ("trio null, PCA", comparison["pca_fluctuation"]
              ["trio_null_tbd_matched"]["p_empirical"])]
    if bprof is not None:
        tests.append(("trio null, B-factor", comparison["experimental_bfactor"]
                      ["trio_null_tbd_matched"]["p_empirical"]))
    tests += [("Mann-Whitney, ANM", comparison["anm_fluctuation"]
               ["mannwhitney"]["p_one_sided"]),
              ("Mann-Whitney, PCA", comparison["pca_fluctuation"]
               ["mannwhitney"]["p_one_sided"])]
    ps = sorted(p for _, p in tests)
    m = len(ps)
    bh = [p * m / (k + 1) for k, p in enumerate(ps)]
    out["multiple_comparisons"] = {
        "n_tests_on_this_hypothesis": m,
        "tests": [{"test": t, "p": p} for t, p in tests],
        "bonferroni_threshold_for_0p05": 0.05 / m,
        "smallest_p": ps[0],
        "smallest_p_survives_bonferroni": bool(ps[0] < 0.05 / m),
        "benjamini_hochberg_adjusted_smallest": float(min(bh)),
        "note": "The three trio-null tests are three measures of the pre-specified "
                "UniProt annotation-trio hypothesis, not three hypotheses; they are "
                "reported together rather than corrected "
                "against each other, and the B-factor test is a pre-specified negative "
                "control rather than a competing test. The Bonferroni figure is given "
                "so a reader can apply the strictest possible correction.",
    }

    # --------------------------------------------------- effect sizes ---
    eff_by_definition = {}
    for definition_name, selected_i in definition_indices.items():
        definition_effects = {}
        for profile_name, prof in profiles.items():
            a, b = prof[selected_i], prof[zn_i]
            sp = np.sqrt(
                ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2)
            )
            d = float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")
            cd = float(np.mean([np.sign(x - y) for x in a for y in b]))
            rng = np.random.default_rng(SEED)
            boot = []
            ranks = np.array([100.0 * (prof <= v).mean() for v in prof])
            for _ in range(10000):
                ia = rng.choice(selected_i, len(selected_i), replace=True)
                ib = rng.choice(zn_i, len(zn_i), replace=True)
                boot.append(ranks[ia].mean() - ranks[ib].mean())
            boot = np.array(boot)
            definition_effects[profile_name] = {
                "definition": definition_name,
                "residues": list(POCKET_DEFINITIONS[definition_name]),
                "cohens_d": d,
                "cliffs_delta": cd,
                "percentile_difference":
                    float(ranks[selected_i].mean() - ranks[zn_i].mean()),
                "percentile_difference_ci95": [
                    float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)),
                ],
                "note": (
                    f"n = {len(a)} vs {len(b)}; intervals are wide by construction "
                    "and are reported to convey exactly that"
                ),
            }
        eff_by_definition[definition_name] = definition_effects
    eff = eff_by_definition["uniprot_ligand_annotations"]
    out["effect_sizes_by_definition"] = eff_by_definition
    # Backwards-compatible alias: UniProt annotation trio only.
    out["effect_sizes"] = eff

    # ------------------------------------------------- recommended wording ---
    pca_p = comparison["pca_fluctuation"]["trio_null_tbd_matched"]["p_empirical"]
    anm_p = comparison["anm_fluctuation"]["trio_null_tbd_matched"]["p_empirical"]
    b_p = (comparison["experimental_bfactor"]["trio_null_tbd_matched"]["p_empirical"]
           if bprof is not None else None)
    shell = comparison_by_definition["5fqd_4.5A_contact_shell_common_window"]
    core = comparison_by_definition["annotated_plus_W400_F402"]
    out["verdict"] = {
        "primary_statistic": "ensemble PCA fluctuation percentile of the three "
                             "pre-specified UniProt ligand annotations against a "
                             "TBD-matched random-trio null",
        "primary_p": pca_p,
        "secondary_p_anm": anm_p,
        "bfactor_null_result_p": b_p,
        "mannwhitney_p": comparison["pca_fluctuation"]["mannwhitney"]["p_one_sided"],
        "mannwhitney_p_two_sided":
            comparison["pca_fluctuation"]["mannwhitney"]["p_two_sided"],
        "supportable_claim": "The three UniProt ligand annotations occupy unusually "
                             "high PCA-fluctuation percentiles relative to TBD-matched "
                             "random trios. Their mean exceeds the zinc annotations, but "
                             "the discrete 3-vs-4 exact two-sided comparison gives "
                             "p = 0.0571. Neither result may be generalized to the entire "
                             "structural pocket.",
        "definition_sensitivity": {
            "annotated_plus_W400_F402": {
                "anm_mannwhitney_p_two_sided":
                    core["anm_fluctuation"]["mannwhitney"]["p_two_sided"],
                "pca_mannwhitney_p_two_sided":
                    core["pca_fluctuation"]["mannwhitney"]["p_two_sided"],
                "anm_percentile_difference": core["anm_fluctuation"]["difference_window"],
                "pca_percentile_difference": core["pca_fluctuation"]["difference_window"],
            },
            "5fqd_4.5A_contact_shell_common_window": {
                "anm_mannwhitney_p_two_sided":
                    shell["anm_fluctuation"]["mannwhitney"]["p_two_sided"],
                "pca_mannwhitney_p_two_sided":
                    shell["pca_fluctuation"]["mannwhitney"]["p_two_sided"],
                "anm_percentile_difference": shell["anm_fluctuation"]["difference_window"],
                "pca_percentile_difference": shell["pca_fluctuation"]["difference_window"],
            },
        },
        "not_supportable": ["that the loop is 'more disordered' (crystallographic "
                            "B-factors do not distinguish the two sites)",
                            "any significance claim resting on the 3-vs-4 two-sided "
                            "Mann-Whitney test, whose minimum attainable p is 0.0571",
                            "that residues 378/380/386 exhaustively define the structural "
                            "thalidomide pocket",
                            "that the ANM profile distinguishes the seven-residue 5FQD "
                            "contact shell from the zinc annotations"],
    }

    artifact_path = Path("data/drug_loop_statistics.json")
    if not verify:
        atomic_write_json(artifact_path, out)
    print(f"\nprimary annotation trio: PCA-fluctuation set-null p = {pca_p:.4f}; "
          f"ANM p = {anm_p:.4f}; "
          + (f"B-factor p = {b_p:.3f} (negative control)"
             if b_p is not None else "B-factor control skipped/unavailable"))

    if verify:
        with artifact_path.open(encoding="utf-8") as handle:
            committed = json.load(handle)
        assert_tree_close(out, committed, float_tolerance=1e-8)
        c = comparison
        assert abs(rho_check - 1.0) < 1e-6 or rho_check > 0.99, rho_check
        # (a) small-sample limit
        mw = c["anm_fluctuation"]["mannwhitney"]
        assert mw["n_permutations"] == 35, mw
        assert abs(mw["p_minimum_attainable"] - 1 / 35) < 1e-9
        assert abs(mw["p_one_sided"] - 0.114) < 0.01, mw["p_one_sided"]
        # (b) domain-matched nulls
        assert 0.02 < c["anm_fluctuation"]["trio_null_tbd_matched"]["p_empirical"] < 0.06, \
            c["anm_fluctuation"]["trio_null_tbd_matched"]
        assert c["pca_fluctuation"]["trio_null_tbd_matched"]["p_empirical"] < 0.01, \
            c["pca_fluctuation"]["trio_null_tbd_matched"]
        # (b2) pocket-definition sensitivity: ANM does not distinguish either
        # structural definition from zinc; PCA remains strongest for the resolved
        # seven-residue 5FQD contact shell.
        assert list(POCKET_DEFINITIONS["5fqd_4.5A_contact_shell_common_window"]) == \
            [377, 378, 379, 380, 386, 400, 402]
        shell_c = comparison_by_definition["5fqd_4.5A_contact_shell_common_window"]
        core_c = comparison_by_definition["annotated_plus_W400_F402"]
        assert shell_c["anm_fluctuation"]["mannwhitney"]["p_two_sided"] > 0.4
        assert shell_c["pca_fluctuation"]["mannwhitney"]["p_two_sided"] < 0.03
        assert shell_c["pca_fluctuation"]["mannwhitney"]["p_two_sided"] > 0.02
        assert core_c["anm_fluctuation"]["mannwhitney"]["p_two_sided"] > 0.9
        assert 0.05 < core_c["pca_fluctuation"]["mannwhitney"]["p_two_sided"] < 0.07
        # (c) contact confound is strong and negative, and the result survives it
        s = conf["anm_fluctuation"]["spearman_contact_vs_profile"]
        assert s["rho"] < -0.7, s
        assert (conf["anm_fluctuation"]["drug_percentile_after_contact_residualisation"] >
                conf["anm_fluctuation"]["zn_percentile_after_contact_residualisation"])
        if bprof is not None:
            # B-factor null result: no separation, and the model survives residualisation
            bc = c["experimental_bfactor"]
            assert abs(bc["difference_window"]) < 15, bc["difference_window"]
            assert bc["trio_null_tbd_matched"]["p_empirical"] > 0.2, \
                bc["trio_null_tbd_matched"]
            assert (conf["anm_fluctuation"]
                    ["drug_percentile_after_bfactor_residualisation"] >
                    conf["anm_fluctuation"]
                    ["zn_percentile_after_bfactor_residualisation"])
        b_msg = (f"B-factor p = {b_p:.3f} (pre-specified negative control)"
                 if b_p is not None else
                 "B-factor control skipped/unavailable")
        print(f"verify OK: PCA annotation-trio null p = {pca_p:.4f} (primary), "
              f"ANM p = {anm_p:.4f}, "
              f"{b_msg}; "
              f"Mann-Whitney p = {mw['p_one_sided']:.3f} against a floor of "
              f"{mw['p_minimum_attainable']:.4f}; contact-number rho = {s['rho']:+.2f} "
              f"and the ordering survives residualisation")


if __name__ == "__main__":
    main()
