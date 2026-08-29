#!/usr/bin/env python3
"""Regenerate data/crbn_md_zn_analysis.json from committed distributed analysis data.

Single source: the per-residue Ca RMSF CSV (data/crbn_md_rmsf.csv) and the
one-dimensional Zn-S bond trace (data/crbn_md_zn_timeseries.npz, key 'znS',
the mean of the four Zn-S bonds per frame). This script reproduces the summary JSON
from those committed arrays; it does not regenerate the RMSF table from trajectory
frames.

The central percentiles are group means over the common 269-residue analysis window
shared with ANM/PCA (drug_pct_mean_common269 / zn_pct_mean_common269): drug 37 / Zn 70.
The full-381-construct values (drug 29 / Zn 60) are retained for reference only. The
drug-binding loop is less mobile than the zinc site in this single closed-monomer MD on
either basis; the ensemble ANM/PCA give the opposite ordering. This historical comparison is
documented in the repository data guide.

These values are NOT plotted in main-text Fig 4b. Their input, data/crbn_md_rmsf.csv, has
no committed generating script, so it cannot sit in a main figure beside the reproducible
ANM and PCA profiles; this script only re-derives the summary JSON from that distributed
analysis product. Reinstating the MD bars requires first committing the script that
produces the RMSF from the trajectory.
"""
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]        # repo root, name-independent
DATA = ROOT / "data"
DRUG_RES = (378, 380, 386)
ZN_RES = (323, 326, 391, 394)


def prepare_inputs(rows, fluctuation_rows, zn_trace, *, expected_common=None):
    """Validate and normalize the two tables and the numeric Zn-S trace."""
    if not rows:
        raise ValueError("RMSF table is empty")
    try:
        rn = np.array([int(row["resnum"]) for row in rows], dtype=int)
        rmsf = np.array([float(row["md_rmsf"]) for row in rows], dtype=float)
        fluctuation_residues = [int(row["resnum"]) for row in fluctuation_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed residue table: {exc}") from exc

    if not np.isfinite(rmsf).all():
        raise ValueError("RMSF table contains non-finite values")
    unique, counts = np.unique(rn, return_counts=True)
    duplicates = unique[counts != 1]
    if duplicates.size:
        raise ValueError(f"each RMSF residue must occur exactly once; duplicates: {duplicates.tolist()}")
    for residue in (*DRUG_RES, *ZN_RES):
        if int(np.sum(rn == residue)) != 1:
            raise ValueError(f"required residue {residue} must occur exactly once")

    common = np.array(sorted(set(fluctuation_residues) & set(rn.tolist())), dtype=int)
    if common.size == 0:
        raise ValueError("common ANM/PCA and MD residue window is empty")
    if expected_common is not None and common.size != expected_common:
        raise ValueError(
            f"common residue window has {common.size} residues; expected {expected_common}"
        )
    cmask = np.isin(rn, common)

    zn_s = np.asarray(zn_trace)
    if not np.issubdtype(zn_s.dtype, np.number):
        raise ValueError(f"Zn-S trace is not numeric: dtype {zn_s.dtype}")
    if zn_s.ndim != 1 or zn_s.size == 0:
        raise ValueError("Zn-S trace must be a non-empty one-dimensional array")
    if not np.isfinite(zn_s).all():
        raise ValueError("Zn-S trace contains non-finite values")
    return rn, rmsf, common, cmask, zn_s


def build_summary(rn, rmsf, cmask, zn_s):
    residue_rmsf = {int(residue): float(value) for residue, value in zip(rn, rmsf)}

    def pct(residue):
        return 100.0 * float(np.mean(rmsf <= residue_rmsf[residue]))

    def pct_common(residue):
        return 100.0 * float(np.mean(rmsf[cmask] <= residue_rmsf[residue]))

    def group_mean(residues):
        return round(float(np.mean([pct(residue) for residue in residues])), 4)

    def group_mean_common(residues):
        return round(float(np.mean([pct_common(residue) for residue in residues])), 4)

    return {
        "source": "data/crbn_md_rmsf.csv (per-residue Ca RMSF) and "
                  "data/crbn_md_zn_timeseries.npz key 'znS' (mean Zn-S bond per frame)",
        "percentile_basis": "Group means over the common 269-residue analysis window shared with ANM/PCA "
                            "(drug_pct_mean_common269 / zn_pct_mean_common269); full-381-construct values "
                            "retained for reference. NOT plotted in Fig 4b: the input RMSF array has no "
                            "committed generating script, so it is reported in the repository data guide instead.",
        "znS_bond_mean_A": round(float(zn_s.mean(dtype=np.float64)), 6),
        "znS_bond_std_A": round(float(zn_s.std(dtype=np.float64)), 6),
        "znS_note": "mean of the four Zn-S bonds per frame; std is the temporal std of that "
                    "mean-bond trace over the 100 ns production run.",
        "drug_res": list(DRUG_RES),
        "zn_res": list(ZN_RES),
        "drug_pct_mean_common269": group_mean_common(DRUG_RES),
        "zn_pct_mean_common269": group_mean_common(ZN_RES),
        "drug_pct_mean": group_mean(DRUG_RES),
        "zn_pct_mean": group_mean(ZN_RES),
        "drug_pct_each": {str(residue): round(pct(residue), 4) for residue in DRUG_RES},
        "zn_pct_each": {str(residue): round(pct(residue), 4) for residue in ZN_RES},
        "drug_rmsf_each_A": {
            str(residue): round(residue_rmsf[residue], 6) for residue in DRUG_RES
        },
        "zn_rmsf_each_A": {
            str(residue): round(residue_rmsf[residue], 6) for residue in ZN_RES
        },
        "finding": "In this single 100 ns zinc-bonded MD of one closed monomer, the drug-binding loop has "
                   "lower mobility than the zinc site (common 269-residue window group-mean 37 vs 70 "
                   "percentile; 29 vs 60 on the full 381-residue construct); the ensemble ANM and "
                   "PCA give the reverse ordering (drug-binding > zinc site), which is the "
                   "analysis primary residue-level result (Fig 4b). The MD is reported as exploratory single-state "
                   "context, not as support for the ordering.",
    }


def write_json_atomic(path, value):
    """Replace a generated JSON artifact only after a complete sibling write."""
    path = Path(path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main():
    with (DATA / "crbn_md_rmsf.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (DATA / "crbn_residue_fluctuations.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        fluctuation_rows = list(csv.DictReader(handle))
    with np.load(DATA / "crbn_md_zn_timeseries.npz") as archive:
        zn_trace = archive["znS"]

    rn, rmsf, _common, cmask, zn_s = prepare_inputs(
        rows, fluctuation_rows, zn_trace, expected_common=269
    )
    out = build_summary(rn, rmsf, cmask, zn_s)
    write_json_atomic(DATA / "crbn_md_zn_analysis.json", out)
    print(f"common-269 window (matches Fig 4b):          drug {out['drug_pct_mean_common269']} / "
          f"Zn {out['zn_pct_mean_common269']}")
    print(f"full-381 construct (reference only):          drug {out['drug_pct_mean']} / "
          f"Zn {out['zn_pct_mean']}")
    print(f"znS {out['znS_bond_mean_A']} +/- {out['znS_bond_std_A']} A")


if __name__ == "__main__":
    main()
