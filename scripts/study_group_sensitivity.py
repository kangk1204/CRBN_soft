#!/usr/bin/env python3
"""Study-level (publication-DOI) sensitivity of the open->closed transition axis.

Deposited PDB entries are not independent samples: one study often contributes
several structures. The main analysis weights entries equally. This script asks
whether that weighting drives the reported ANM alignment, by

  1. grouping the 70 curated conformers by RCSB primary-citation DOI, with
     tracked publisher-verified DOI overrides or explicit series identifiers
     for entries whose RCSB records have no DOI,
  2. recomputing the open->closed axis with every STUDY weighted equally
     (each group contributes the mean of its members), and
  3. dropping each whole study in turn (leave-one-study-out).

Reported: mode-1 ANM overlap under each weighting, and the angle between the
entry-weighted and study-weighted axes.

Usage
  python scripts/study_group_sensitivity.py --fetch    # refresh DOI table from RCSB
  python scripts/study_group_sensitivity.py [--verify]
Outputs  data/study_group_sensitivity.json
"""
import argparse, csv, io, json, urllib.request
from pathlib import Path
import numpy as np

try:
    from study_groups import load_study_groups, resolve_study_groups
except ModuleNotFoundError:
    from scripts.study_groups import load_study_groups, resolve_study_groups
try:
    from analysis_contracts import (
        assert_tree_close,
        atomic_write_json,
        atomic_write_text,
        validate_ensemble_diff,
    )
except ModuleNotFoundError:
    from scripts.analysis_contracts import (
        assert_tree_close,
        atomic_write_json,
        atomic_write_text,
        validate_ensemble_diff,
    )

CUTOFF = 15.0
ROOT_D = "data/"

def anm_modes(coords, cutoff=CUTOFF, n=20):
    m = len(coords); H = np.zeros((3*m, 3*m))
    for i in range(m):
        d = coords - coords[i]; r = np.linalg.norm(d, axis=1)
        for j in range(m):
            if j == i or r[j] > cutoff or r[j] < 1e-6:
                continue
            k = np.outer(d[j], d[j]) / r[j]**2
            H[3*i:3*i+3, 3*j:3*j+3] = -k
            H[3*i:3*i+3, 3*i:3*i+3] += k
    w, v = np.linalg.eigh(H)
    idx = np.argsort(w); w, v = w[idx], v[:, idx]
    keep = w > 1e-8
    return w[keep][:n], v[:, keep][:, :n]

def fetch_dois(labels, write=True):
    G = "https://data.rcsb.org/graphql"
    q = ("query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id "
         "rcsb_primary_citation{pdbx_database_id_DOI}}}")
    out = {}
    for i in range(0, len(labels), 25):
        body = json.dumps({"query": q, "variables": {"ids": labels[i:i+25]}}).encode()
        req = urllib.request.Request(G, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as response:
            entries = json.loads(response.read())["data"]["entries"]
        for e in entries:
            c = e.get("rcsb_primary_citation") or {}
            out[e["rcsb_id"]] = (c.get("pdbx_database_id_DOI") or f"NO_DOI:{e['rcsb_id']}").lower()
    if set(out) != set(labels):
        raise RuntimeError(
            f"RCSB citation response incomplete: missing={sorted(set(labels) - set(out))}, "
            f"extra={sorted(set(out) - set(labels))}"
        )
    groups = resolve_study_groups(out, labels)
    if write:
        payload = io.StringIO(newline="")
        writer = csv.writer(payload, lineterminator="\n")
        writer.writerow(["pdb", "primary_citation_doi"])
        for key in sorted(out):
            writer.writerow([key, out[key]])
        atomic_write_text(Path(ROOT_D + "curation_study_groups.csv"), payload.getvalue())
    return groups

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="recompute and compare without writing")
    parser.add_argument("--fetch", action="store_true", help="refresh study DOI metadata from RCSB")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verify = bool(args.verify)
    ens = np.load(ROOT_D + "crbn_ensemble.ens.npz", allow_pickle=False)
    dv = np.load(ROOT_D + "pca_diffvec.npz", allow_pickle=False)
    confs, label_array, open_mask, _ = validate_ensemble_diff(ens, dv)
    labels = label_array.tolist()

    if args.fetch:
        doi = fetch_dois(labels, write=not verify)
    else:
        doi = load_study_groups(labels)
    groups = sorted({doi[l] for l in labels})

    aw, av = anm_modes(confs[labels.index("8CVP")])
    if aw.shape != (20,) or av.shape != (confs.shape[1] * 3, 20):
        raise ValueError(f"unexpected ANM eigensystem shapes: {aw.shape}, {av.shape}")
    if not np.isfinite(aw).all() or not np.isfinite(av).all() or not (aw > 0).all():
        raise ValueError("ANM eigensystem contains non-finite or non-positive values")
    if not np.allclose(av.T @ av, np.eye(av.shape[1]), atol=1e-8, rtol=0.0):
        raise ValueError("ANM eigenvectors are not orthonormal")

    def overlap(axis):
        axis = np.asarray(axis, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(axis))
        if axis.shape != (confs.shape[1] * 3,) or not np.isfinite(axis).all() or norm <= 1e-12:
            raise ValueError("state-difference axis is missing, non-finite, or degenerate")
        a = axis / norm
        ov = np.abs(av.T @ a)
        return float(ov[0]), int(np.argmax(ov)) + 1, a

    # entry-weighted (the axis used in the primary analysis)
    ax_entry = confs[~open_mask].mean(0) - confs[open_mask].mean(0)
    o_entry, r_entry, a_entry = overlap(ax_entry)

    # study-weighted: each publication contributes the mean of its own structures
    def group_mean(mask):
        per = []
        for g in groups:
            sel = np.array([doi[l] == g and mask[i] for i, l in enumerate(labels)])
            if sel.any():
                per.append(confs[sel].mean(0))
        return np.mean(per, axis=0) if per else None

    closed_group_mean, open_group_mean = group_mean(~open_mask), group_mean(open_mask)
    if closed_group_mean is None or open_group_mean is None:
        raise ValueError("study weighting requires at least one open and one closed group")
    ax_study = closed_group_mean - open_group_mean
    o_study, r_study, a_study = overlap(ax_study)
    agreement = float(abs(a_entry @ a_study))

    # leave-one-study-out
    loso = {}
    for g in groups:
        keep = np.array([doi[l] != g for l in labels])
        if not (keep & ~open_mask).any() or not (keep & open_mask).any():
            continue
        ax = confs[keep & ~open_mask].mean(0) - confs[keep & open_mask].mean(0)
        loso[g] = overlap(ax)[0]
    vals = np.array(list(loso.values()))
    if vals.size == 0 or not np.isfinite(vals).all():
        raise ValueError("leave-one-study-out sensitivity produced no finite comparisons")

    out = {"n_conformers": len(labels), "n_study_groups": len(groups),
           "largest_group_size": max(sum(doi[l] == g for l in labels) for g in groups),
           "entry_weighted_overlap": o_entry, "entry_weighted_rank": r_entry,
           "study_weighted_overlap": o_study, "study_weighted_rank": r_study,
           "axis_agreement_entry_vs_study": agreement,
           "leave_one_study_out": {"min": float(vals.min()), "max": float(vals.max()),
                                   "n": int(vals.size)}}
    if not verify:
        atomic_write_json(Path(ROOT_D + "study_group_sensitivity.json"), out, sort_keys=True)
    for k, v in out.items():
        print(f"  {k}: {v}")

    if verify:
        reference = json.loads(
            Path(ROOT_D + "study_group_sensitivity.json").read_text(encoding="utf-8")
        )
        assert_tree_close(out, reference)
        assert r_entry == 1 and r_study == 1, (r_entry, r_study)
        assert agreement > 0.99, agreement
        assert vals.min() > 0.6, vals.min()
        print("verify OK: exact input-bundle artifact matches and mode 1 remains the "
              "best-matching mode under study-equal weighting")

if __name__ == "__main__":
    main()
