#!/usr/bin/env python3
"""Validate the ZAFF Cys4 zinc-site contract for the CRBN atomistic prototype.

This script validates parameter-source completeness only.  The generated LEaP
input intentionally starts from overlapping peptide fragments, so it is marked
as geometry_not_qualified and must not be used as direct MD input.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "atomistic_parameters"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "atomistic" / "zaff_contract"
)

CRBN_SITE = {
    "pdb_id": "8CVP",
    "chain_id": "B",
    "zinc_residue": "ZN501",
    "cysteine_residues": [323, 326, 391, 394],
    "observed_sg_zn_distance_A": {"min": 2.326, "max": 2.343},
    "required_model": "four CY1 thiolate residues plus one ZN1 center",
}

EXPECTED_RESIDUE = "CY1"
EXPECTED_ZINC = "ZN1"
EXPECTED_SULFUR_TYPE = "S1"
EXPECTED_ZINC_TYPE = "ZN"
EXPECTED_CY1_CHARGE = -0.63109
EXPECTED_ZN1_CHARGE = 0.52437
EXPECTED_SITE_CHARGE = -1.99999


@dataclass(frozen=True)
class PrepAtom:
    index: int
    name: str
    atom_type: str
    tree: str
    charge: float


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_prep(path: Path) -> dict[str, list[PrepAtom]]:
    units: dict[str, list[PrepAtom]] = {}
    current: str | None = None
    reading_atoms = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) >= 3 and fields[1] == "INT" and fields[2] == "1":
            current = fields[0]
            units[current] = []
            reading_atoms = False
            continue
        if current is None:
            continue
        if line == "CORR OMIT DU   BEG":
            reading_atoms = True
            continue
        if line in {"IMPROPER", "LOOP", "DONE"}:
            reading_atoms = False
            if line == "DONE":
                current = None
            continue
        if not reading_atoms or not fields[0].isdigit():
            continue
        if len(fields) < 11:
            raise ValueError(f"Malformed PREP atom row in {path}: {raw}")
        units[current].append(
            PrepAtom(
                index=int(fields[0]),
                name=fields[1],
                atom_type=fields[2],
                tree=fields[3],
                charge=float(fields[-1]),
            )
        )
    return units


def parse_frcmod(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    sections: dict[str, dict[str, dict[str, float]]] = {"BOND": {}, "ANGL": {}}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        header = line.split()[0]
        if header in {"MASS", "BOND", "ANGL", "DIHE", "IMPROPER", "NONBON"}:
            current = "ANGL" if header == "ANGL" else header
            continue
        if current == "BOND":
            fields = line.split()
            if len(fields) >= 3 and "-" in fields[0]:
                sections["BOND"][fields[0]] = {"k_amber": float(fields[1]), "r0_A": float(fields[2])}
        elif current == "ANGL":
            fields = line.split()
            if len(fields) >= 3 and fields[0].count("-") == 2:
                sections["ANGL"][fields[0]] = {
                    "k_amber": float(fields[1]),
                    "theta0_deg": float(fields[2]),
                }
    return sections


def _atom_rows(atoms: Iterable[PrepAtom]) -> list[dict[str, object]]:
    return [
        {"index": a.index, "name": a.name, "type": a.atom_type, "tree": a.tree, "charge_e": a.charge}
        for a in atoms
        if a.atom_type != "DU"
    ]


def _failures_for_model(units: dict[str, list[PrepAtom]], frcmod: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    failures: list[str] = []
    if EXPECTED_RESIDUE not in units:
        failures.append("Expected CY1 residue block is absent from PREP")
    if EXPECTED_ZINC not in units:
        failures.append("Expected ZN1 zinc-center block is absent from PREP")
    if EXPECTED_RESIDUE not in units and "CY4" in units:
        failures.append("CY4 is present in source but is not allowed for CRBN Cys4; reject 2Cys2His template mixing")

    cy1 = units.get(EXPECTED_RESIDUE, [])
    zn1 = units.get(EXPECTED_ZINC, [])
    cy1_atoms = {a.name: a for a in cy1 if a.atom_type != "DU"}
    zn_atoms = {a.name: a for a in zn1 if a.atom_type != "DU"}
    sg = cy1_atoms.get("SG")
    zn = zn_atoms.get("ZN")
    if sg is None or sg.atom_type != EXPECTED_SULFUR_TYPE:
        failures.append("CY1 SG must have ZAFF thiolate type S1")
    if any(a.name in {"HG", "HSG"} for a in cy1_atoms.values()):
        failures.append("CY1 must not retain thiol hydrogen HG/HSG")
    if zn is None or zn.atom_type != EXPECTED_ZINC_TYPE:
        failures.append("ZN1 center must expose atom ZN with type ZN")

    cy1_charge = round(sum(a.charge for a in cy1 if a.atom_type != "DU"), 5)
    zn1_charge = round(sum(a.charge for a in zn1 if a.atom_type != "DU"), 5)
    site_charge = round(4 * cy1_charge + zn1_charge, 5)
    if abs(cy1_charge - EXPECTED_CY1_CHARGE) > 1e-5:
        failures.append(f"CY1 charge {cy1_charge:.5f} e differs from expected {EXPECTED_CY1_CHARGE:.5f} e")
    if abs(zn1_charge - EXPECTED_ZN1_CHARGE) > 1e-5:
        failures.append(f"ZN1 charge {zn1_charge:.5f} e differs from expected {EXPECTED_ZN1_CHARGE:.5f} e")
    if abs(site_charge - EXPECTED_SITE_CHARGE) > 1e-5:
        failures.append(f"Four CY1 plus ZN1 charge {site_charge:.5f} e differs from expected {EXPECTED_SITE_CHARGE:.5f} e")

    for key in ("ZN-S1",):
        if key not in frcmod["BOND"]:
            failures.append(f"Required ZAFF bond parameter {key} is absent")
    for key in ("CT-S1-ZN", "S1-ZN-S1"):
        if key not in frcmod["ANGL"]:
            failures.append(f"Required ZAFF angle parameter {key} is absent")
    return failures


def build_tleap_prototype(prep: Path, frcmod: Path) -> str:
    return "\n".join(
        [
            "# Small ZAFF parameter-completeness prototype only.",
            "# The ACE-CY1-NME fragments initially overlap and are geometry_not_qualified.",
            "# Do not infer topology or MD readiness from LEaP syntax success.",
            "source leaprc.protein.ff14SB",
            f"loadAmberPrep {prep}",
            f"loadAmberParams {frcmod}",
            "p1 = sequence { ACE CY1 NME }",
            "p2 = sequence { ACE CY1 NME }",
            "p3 = sequence { ACE CY1 NME }",
            "p4 = sequence { ACE CY1 NME }",
            "z = sequence { ZN1 }",
            "site = combine { p1 p2 p3 p4 z }",
            "bond site.2.SG site.13.ZN",
            "bond site.5.SG site.13.ZN",
            "bond site.8.SG site.13.ZN",
            "bond site.11.SG site.13.ZN",
            "check site",
            "saveamberparm site prototype.parm7 prototype.rst7",
            "quit",
            "",
        ]
    )


def validate_contract(prep: Path, frcmod: Path, table_png: Path | None = None) -> dict[str, object]:
    units = parse_prep(prep)
    parameters = parse_frcmod(frcmod)
    failures = _failures_for_model(units, parameters)
    cy1 = units.get(EXPECTED_RESIDUE, [])
    zn1 = units.get(EXPECTED_ZINC, [])
    cy1_charge = round(sum(a.charge for a in cy1 if a.atom_type != "DU"), 5)
    zn1_charge = round(sum(a.charge for a in zn1 if a.atom_type != "DU"), 5)
    site_charge = round(4 * cy1_charge + zn1_charge, 5)
    sources = {
        "prep": {"path": str(prep), "sha256": sha256_file(prep)},
        "frcmod": {"path": str(frcmod), "sha256": sha256_file(frcmod)},
    }
    if table_png is not None:
        sources["table_png"] = {"path": str(table_png), "sha256": sha256_file(table_png)}
    return {
        "status": "pass" if not failures else "fail",
        "candidate_param_model": "ZAFF CY1/S1 thiolate plus ZN1/ZN center for CRBN Cys4 site",
        "crbn_experimental_site": CRBN_SITE,
        "amber_harmonic_convention": {
            "amber_energy": "E = K * (r - r0)^2",
            "openmm_amber_reader": "OpenMM Amber reader converts automatically to k = 2 * K with units",
            "legacy_qm_curvature_factor_applied": False,
        },
        "template_mixing_policy": {
            "allowed": ["CY1", "ZN1"],
            "rejected": ["CY4 2Cys2His template", "2GIV 3Cys1His tutorial template"],
        },
        "geometry_status": "geometry_not_qualified",
        "md_direct_use": False,
        "md_direct_blocker": (
            "Prototype fragments are separate ACE-CY1-NME units placed by LEaP without alignment "
            "to observed CA/CB/SG coordinates."
        ),
        "actual_model_blocks": {
            "CY1": {"atoms": _atom_rows(cy1), "net_charge_e": cy1_charge},
            "ZN1": {"atoms": _atom_rows(zn1), "net_charge_e": zn1_charge},
        },
        "site_net_charge_e": site_charge,
        "required_parameters": {
            "bonds": {"ZN-S1": parameters["BOND"].get("ZN-S1")},
            "angles": {
                "CT-S1-ZN": parameters["ANGL"].get("CT-S1-ZN"),
                "S1-ZN-S1": parameters["ANGL"].get("S1-ZN-S1"),
            },
        },
        "source_sha256": sources,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, default=DEFAULT_SOURCE_DIR / "ZAFF.prep")
    parser.add_argument("--frcmod", type=Path, default=DEFAULT_SOURCE_DIR / "ZAFF.frcmod")
    parser.add_argument("--table-png", type=Path, default=DEFAULT_SOURCE_DIR / "ZAFF_table.png")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true", help="Accepted for provenance; the validator is local only")
    args = parser.parse_args(argv)

    result = validate_contract(args.prep, args.frcmod, args.table_png if args.table_png.exists() else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "zaff_site_contract.json", result)
    (args.output_dir / "zaff_cys4_prototype.tleap.in").write_text(
        build_tleap_prototype(args.prep, args.frcmod), encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "geometry_status": result["geometry_status"]}, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
