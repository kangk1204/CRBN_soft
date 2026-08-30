#!/usr/bin/env python3
"""Regenerate data/crbn_md_zn_analysis.json from committed distributed analysis data.

Single source: the per-residue Ca RMSF CSV (data/crbn_md_rmsf.csv) and the
one-dimensional Zn-S bond trace (data/crbn_md_zn_timeseries.npz, key 'znS',
the mean of the four Zn-S bonds per frame). This script reproduces the summary JSON
from those committed arrays; it does not regenerate the RMSF table from trajectory
frames.

The three UniProt ligand annotations (378/380/386) are a pre-specified functional set,
not an exhaustive structural pocket.  Results are therefore reported separately for
that trio, the annotation trio plus W400/F402, and the 5FQD LVY heavy-atom contact
shell.  The full 5FQD shell has 11 contacts.  Four (350--353) lie in the sensor-loop
segment omitted from the common 269-residue ANM/PCA window, so the common-window shell
contains seven residues; this missingness is explicit in the output.

Every percentile uses the weak empirical definition
``100 * mean(profile <= selected_value)``.  The full-381-construct values are retained
for reference only.  This historical comparison is documented in the repository data
guide.

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
UNIPROT_LIGAND_ANNOTATIONS = (378, 380, 386)
ANNOTATED_PLUS_W400_F402 = (378, 380, 386, 400, 402)
CONTACT_SHELL_COMMON_WINDOW = (377, 378, 379, 380, 386, 400, 402)
CONTACT_SHELL_SENSOR_LOOP_MISSING = (350, 351, 352, 353)
CONTACT_SHELL_FULL_5FQD = (
    *CONTACT_SHELL_SENSOR_LOOP_MISSING,
    *CONTACT_SHELL_COMMON_WINDOW,
)
POCKET_DEFINITIONS = {
    "uniprot_ligand_annotations": {
        "full_construct": UNIPROT_LIGAND_ANNOTATIONS,
        "common_window": UNIPROT_LIGAND_ANNOTATIONS,
    },
    "annotated_plus_W400_F402": {
        "full_construct": ANNOTATED_PLUS_W400_F402,
        "common_window": ANNOTATED_PLUS_W400_F402,
    },
    "5fqd_4.5A_contact_shell": {
        "full_construct": CONTACT_SHELL_FULL_5FQD,
        "common_window": CONTACT_SHELL_COMMON_WINDOW,
    },
}
# Compatibility alias for previous callers; explicitly scoped to UniProt annotations.
DRUG_RES = UNIPROT_LIGAND_ANNOTATIONS
ZN_RES = (323, 326, 391, 394)
PERCENTILE_CONVENTION = "100 * mean(profile <= selected value)"


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
    required_residues = set(ZN_RES)
    for definition in POCKET_DEFINITIONS.values():
        required_residues.update(definition["full_construct"])
    for residue in sorted(required_residues):
        if int(np.sum(rn == residue)) != 1:
            raise ValueError(f"required residue {residue} must occur exactly once")

    if len(fluctuation_residues) != len(set(fluctuation_residues)):
        raise ValueError("reference fluctuation residues must be unique")
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
    common_residue_set = set(int(value) for value in rn[cmask])

    def pct(residue):
        return 100.0 * float(np.mean(rmsf <= residue_rmsf[residue]))

    def pct_common(residue):
        return 100.0 * float(np.mean(rmsf[cmask] <= residue_rmsf[residue]))

    def group_mean(residues):
        return round(float(np.mean([pct(residue) for residue in residues])), 4)

    def group_mean_common(residues):
        return round(float(np.mean([pct_common(residue) for residue in residues])), 4)

    def definition_summary(name, definition):
        full_residues = tuple(definition["full_construct"])
        common_residues = tuple(definition["common_window"])
        missing = tuple(residue for residue in full_residues
                        if residue not in common_residue_set)
        if any(residue not in common_residue_set for residue in common_residues):
            raise ValueError(f"{name}: declared common-window residue is absent")
        if set(full_residues) - set(residue_rmsf):
            raise ValueError(f"{name}: full-construct residue is absent from the RMSF table")
        if set(full_residues) - set(common_residues) != set(missing):
            raise ValueError(f"{name}: common-window missingness is inconsistent")
        return {
            "residues_full_construct": list(full_residues),
            "residues_common269": list(common_residues),
            "residues_missing_common269": list(missing),
            "percentile_mean_common269": group_mean_common(common_residues),
            "percentile_mean_full381": group_mean(full_residues),
            "percentile_each_common269": {
                str(residue): round(pct_common(residue), 4)
                for residue in common_residues
            },
            "percentile_each_full381": {
                str(residue): round(pct(residue), 4)
                for residue in full_residues
            },
            "rmsf_each_A": {
                str(residue): round(residue_rmsf[residue], 6)
                for residue in full_residues
            },
        }

    definitions = {
        name: definition_summary(name, definition)
        for name, definition in POCKET_DEFINITIONS.items()
    }
    primary = definitions["uniprot_ligand_annotations"]
    zn_common = group_mean_common(ZN_RES)
    zn_full = group_mean(ZN_RES)
    shell = definitions["5fqd_4.5A_contact_shell"]

    return {
        "source": "data/crbn_md_rmsf.csv (per-residue Ca RMSF) and "
                  "data/crbn_md_zn_timeseries.npz key 'znS' (mean Zn-S bond per frame)",
        "percentile_convention": PERCENTILE_CONVENTION,
        "percentile_basis": "For common269, each selected RMSF is ranked against the common "
                            "269-residue ANM/PCA window. For full381, each selected RMSF is "
                            "ranked against the 381-residue MD construct. Group values are "
                            "means of residue percentiles. NOT plotted in Fig 4b: the input "
                            "RMSF array has no committed generating script.",
        "primary_definition": "uniprot_ligand_annotations",
        "primary_definition_scope": "pre-specified UniProt (S)-thalidomide ligand "
                                    "annotations; not an exhaustive pocket definition",
        "contact_shell_provenance": {
            "structure": "5FQD chain B",
            "ligand": "LVY",
            "criterion": "protein residue with any heavy atom <= 4.5 A from any LVY heavy atom",
            "full_contact_residues": list(CONTACT_SHELL_FULL_5FQD),
            "common269_contact_residues": list(CONTACT_SHELL_COMMON_WINDOW),
            "missing_from_common269": list(CONTACT_SHELL_SENSOR_LOOP_MISSING),
            "missing_reason": "residues 350-353 are in the sensor-loop segment omitted "
                              "because the genuine-apo open structures do not resolve it",
        },
        "znS_bond_mean_A": round(float(zn_s.mean(dtype=np.float64)), 6),
        "znS_bond_std_A": round(float(zn_s.std(dtype=np.float64)), 6),
        "znS_note": "mean of the four Zn-S bonds per frame; std is the temporal std of that "
                    "mean-bond trace over the 100 ns production run.",
        "definitions": definitions,
        # Legacy aliases below refer only to the explicit UniProt annotation trio.
        "legacy_alias_scope": "uniprot_ligand_annotations",
        "drug_res": list(DRUG_RES),
        "zn_res": list(ZN_RES),
        "drug_pct_mean_common269": primary["percentile_mean_common269"],
        "zn_pct_mean_common269": zn_common,
        "drug_pct_mean": primary["percentile_mean_full381"],
        "zn_pct_mean": zn_full,
        "drug_pct_each": {str(residue): round(pct(residue), 4) for residue in DRUG_RES},
        "zn_pct_each": {str(residue): round(pct(residue), 4) for residue in ZN_RES},
        "drug_rmsf_each_A": {
            str(residue): round(residue_rmsf[residue], 6) for residue in DRUG_RES
        },
        "zn_rmsf_each_A": {
            str(residue): round(residue_rmsf[residue], 6) for residue in ZN_RES
        },
        "finding": (
            "In this single 100 ns zinc-bonded MD of one closed monomer, both the three "
            f"UniProt ligand annotations ({primary['percentile_mean_common269']:.1f}) and "
            f"the seven-residue common-window 5FQD contact shell "
            f"({shell['percentile_mean_common269']:.1f}) rank below the zinc annotations "
            f"({zn_common:.1f}) on the common-269 basis. The result is definition-specific "
            "and exploratory single-state context, not support for the ensemble ordering."
        ),
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
    print(f"common-269 zinc annotations: {out['zn_pct_mean_common269']}")
    for name, values in out["definitions"].items():
        print(f"  {name:43s} common269 {values['percentile_mean_common269']:7.4f} | "
              f"full381 {values['percentile_mean_full381']:7.4f}")
    print(f"znS {out['znS_bond_mean_A']} +/- {out['znS_bond_std_A']} A")


if __name__ == "__main__":
    main()
