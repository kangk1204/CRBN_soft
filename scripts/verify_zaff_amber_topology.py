#!/usr/bin/env python3
"""Verify the ZAFF prototype topology exported by Amber LEaP.

This is a parameter/topology audit only.  It does not run MD and it treats the
overlapping prototype coordinates as geometry_not_qualified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


EXPECTED_SG_RESIDUES = (2, 5, 8, 11)
EXPECTED_ZN_RESIDUE = 13
EXPECTED_CY1_CHARGE = -0.63109
EXPECTED_ZN1_CHARGE = 0.52437
EXPECTED_SITE_CHARGE = -1.99999
AMBER_BOND_K = 32.69
AMBER_ZNS_BOND_R0_A = 2.426
AMBER_CBSZN_ANGLE_K = 64.397
AMBER_CBSZN_THETA_DEG = 101.733
AMBER_SZNZS_ANGLE_K = 35.729
AMBER_SZNZS_THETA_DEG = 109.472
KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def expected_openmm_constants() -> dict[str, dict[str, float]]:
    return {
        "ZN-S1": {
            "amber_k_kcal_mol_A2": AMBER_BOND_K,
            "amber_r0_A": AMBER_ZNS_BOND_R0_A,
            "openmm_k_kj_mol_nm2": 2 * AMBER_BOND_K * KCAL_TO_KJ / (ANGSTROM_TO_NM**2),
            "openmm_r0_nm": AMBER_ZNS_BOND_R0_A * ANGSTROM_TO_NM,
        },
        "CT-S1-ZN": {
            "amber_k_kcal_mol_rad2": AMBER_CBSZN_ANGLE_K,
            "amber_theta0_deg": AMBER_CBSZN_THETA_DEG,
            "openmm_k_kj_mol_rad2": 2 * AMBER_CBSZN_ANGLE_K * KCAL_TO_KJ,
            "openmm_theta0_rad": math.radians(AMBER_CBSZN_THETA_DEG),
        },
        "S1-ZN-S1": {
            "amber_k_kcal_mol_rad2": AMBER_SZNZS_ANGLE_K,
            "amber_theta0_deg": AMBER_SZNZS_THETA_DEG,
            "openmm_k_kj_mol_rad2": 2 * AMBER_SZNZS_ANGLE_K * KCAL_TO_KJ,
            "openmm_theta0_rad": math.radians(AMBER_SZNZS_THETA_DEG),
        },
    }


def classify_leap_warnings(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"log_present": False, "warnings_total": None, "errors_total": None, "categories": {}}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    summary = {"warnings_total": None, "errors_total": None, "notes_total": None}
    for line in lines:
        match = re.search(r"Exiting LEaP: Errors = (\d+); Warnings = (\d+); Notes = (\d+)", line)
        if match:
            summary = {
                "errors_total": int(match.group(1)),
                "warnings_total": int(match.group(2)),
                "notes_total": int(match.group(3)),
            }
    blocks: list[str] = []
    for index, line in enumerate(lines):
        if "Warning!" in line:
            block_lines = [line]
            for next_line in lines[index + 1 :]:
                if "Warning!" in next_line or next_line.startswith("Exiting LEaP"):
                    break
                block_lines.append(next_line)
            blocks.append("\n".join(block_lines))
    categories: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for block in blocks:
        lowered = block.lower()
        if "close contact of" in lowered:
            category = "close_contact_from_overlapping_prototype_geometry"
        elif "bond of" in lowered and "angstroms between" in lowered:
            category = "long_bond_from_overlapping_prototype_geometry"
        elif "unperturbed charge" in lowered:
            category = "nonzero_unit_charge_expected_for_cys4_zn_site"
        elif "could not find" in lowered or "not found" in lowered or "missing" in lowered:
            category = "parameter_warning"
        elif "parameter" in lowered and "checking parameters" not in lowered:
            category = "parameter_warning"
        else:
            category = "other_warning"
        categories[category] += 1
        examples.setdefault(category, block)
    return {
        "log_present": True,
        **summary,
        "warning_blocks_seen": len(blocks),
        "categories": dict(sorted(categories.items())),
        "examples": examples,
    }


def prep_atom_contract(prep_path: Path) -> dict[str, dict[str, dict[str, float | str]]]:
    units: dict[str, dict[str, dict[str, float | str]]] = {}
    current: str | None = None
    reading_atoms = False
    for raw in prep_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) >= 3 and fields[1] == "INT" and fields[2] == "1":
            current = fields[0]
            units[current] = {}
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
        if not reading_atoms or not fields[0].isdigit() or fields[2] == "DU":
            continue
        units[current][fields[1]] = {"type": fields[2], "charge_e": float(fields[-1])}
    for unit_name in ("CY1", "ZN1"):
        if unit_name not in units:
            raise ValueError(f"{unit_name} block is absent from PREP contract")
    return units


def _term_atom_row(atoms: list[Any], atom_types: list[str] | None, charges: list[float], index: int) -> dict[str, Any]:
    atom = atoms[index]
    return {
        "atom_index": index,
        "atom_name": atom.name,
        "atom_type": None if atom_types is None else atom_types[index],
        "residue_index_1based": atom.residue.index + 1,
        "residue_name": atom.residue.name,
        "charge_e": charges[index],
    }


def parameter_contract(
    system: Any,
    topology: Any,
    *,
    prep_path: Path | None = None,
    atom_types: list[str] | None = None,
    expected_sg_residues: tuple[int, ...] | None = None,
    prototype_crosslink_report: bool = False,
) -> dict[str, Any]:
    from openmm import HarmonicAngleForce, HarmonicBondForce, NonbondedForce, unit

    atoms = list(topology.atoms())
    if len(atoms) != system.getNumParticles():
        raise ValueError("Topology and system atom counts differ")
    if atom_types is not None and len(atom_types) != len(atoms):
        raise ValueError("Atom type count differs from topology atom count")
    prep_contract = prep_atom_contract(prep_path) if prep_path is not None else None
    constants = expected_openmm_constants()
    nonbonded = [force for force in system.getForces() if isinstance(force, NonbondedForce)]
    if len(nonbonded) != 1:
        raise ValueError("Expected exactly one NonbondedForce")
    nb_force = nonbonded[0]
    charges = [
        float(nb_force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge))
        for i in range(len(atoms))
    ]
    atom_rows = [_term_atom_row(atoms, atom_types, charges, i) for i in range(len(atoms))]

    sg_atoms = [a for a in atoms if a.name == "SG" and a.residue.name == "CY1"]
    cb_by_residue = {a.residue.index + 1: a for a in atoms if a.name == "CB" and a.residue.name == "CY1"}
    zn_atoms = [a for a in atoms if a.name == "ZN" and a.residue.name == "ZN1"]
    failures: list[str] = []
    if len(sg_atoms) != 4:
        failures.append(f"Expected exactly 4 CY1 SG atoms, observed {len(sg_atoms)}")
    if len(zn_atoms) != 1:
        failures.append(f"Expected exactly 1 ZN1 ZN atom, observed {len(zn_atoms)}")
    if atom_types is None:
        failures.append("Amber atom types are required for full ZAFF parameter validation")
    if prep_contract is None:
        failures.append("ZAFF PREP contract is required for full per-atom charge/type validation")
    if failures:
        return {
            "status": "fail",
            "scope": "ZAFF parameter/topology contract; no MD performed",
            "failures": failures,
            "counts": {"atoms": len(atoms), "residues": topology.getNumResidues()},
        }

    zn_atom = zn_atoms[0]
    sg_by_residue = {a.residue.index + 1: a for a in sg_atoms}
    site_residues = tuple(sorted(sg_by_residue))
    if expected_sg_residues is not None and site_residues != tuple(sorted(expected_sg_residues)):
        failures.append(f"Expected CY1 SG residues {list(expected_sg_residues)}, observed {list(site_residues)}")
    expected_sg_indices = [sg_by_residue[i].index for i in site_residues]
    zn_index = zn_atom.index

    bond_force = next(force for force in system.getForces() if isinstance(force, HarmonicBondForce))
    angle_force = next(force for force in system.getForces() if isinstance(force, HarmonicAngleForce))

    zns_bonds = []
    all_bond_pairs: set[tuple[int, int]] = set()
    for term_index in range(bond_force.getNumBonds()):
        a, b, length, k_value = bond_force.getBondParameters(term_index)
        a = int(a)
        b = int(b)
        all_bond_pairs.add(tuple(sorted((a, b))))
        if zn_index in (a, b) and (a in expected_sg_indices or b in expected_sg_indices):
            zns_bonds.append(
                {
                    "term_index": term_index,
                    "atoms": [a, b],
                    "atom_rows": [_term_atom_row(atoms, atom_types, charges, a), _term_atom_row(atoms, atom_types, charges, b)],
                    "length_nm": float(length.value_in_unit(unit.nanometer)),
                    "k_kj_mol_nm2": float(k_value.value_in_unit(unit.kilojoules_per_mole / unit.nanometer**2)),
                }
            )
        elif zn_index in (a, b):
            failures.append(f"Unexpected extra Zn covalent bond to atoms {a}-{b}")

    szns_angles = []
    cbszn_angles = []
    for term_index in range(angle_force.getNumAngles()):
        a, b, c, theta, k_value = angle_force.getAngleParameters(term_index)
        a = int(a)
        b = int(b)
        c = int(c)
        atom_set = {a, b, c}
        if b == zn_index and a in expected_sg_indices and c in expected_sg_indices:
            szns_angles.append(
                {
                    "term_index": term_index,
                    "atoms": [a, b, c],
                    "atom_rows": [
                        _term_atom_row(atoms, atom_types, charges, a),
                        _term_atom_row(atoms, atom_types, charges, b),
                        _term_atom_row(atoms, atom_types, charges, c),
                    ],
                    "theta_rad": float(theta.value_in_unit(unit.radian)),
                    "k_kj_mol_rad2": float(k_value.value_in_unit(unit.kilojoules_per_mole / unit.radian**2)),
                }
            )
        if zn_index in atom_set:
            for residue_id in site_residues:
                cb = cb_by_residue[residue_id].index
                sg = sg_by_residue[residue_id].index
                if [a, b, c] in ([cb, sg, zn_index], [zn_index, sg, cb]):
                    cbszn_angles.append(
                        {
                            "term_index": term_index,
                            "atoms": [a, b, c],
                            "atom_rows": [
                                _term_atom_row(atoms, atom_types, charges, a),
                                _term_atom_row(atoms, atom_types, charges, b),
                                _term_atom_row(atoms, atom_types, charges, c),
                            ],
                            "theta_rad": float(theta.value_in_unit(unit.radian)),
                            "k_kj_mol_rad2": float(
                                k_value.value_in_unit(unit.kilojoules_per_mole / unit.radian**2)
                            ),
                        }
                    )

    expected_cross_residue_bonds = set()
    if prototype_crosslink_report:
        expected_cross_residue_bonds = {
            (1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9), (10, 11), (11, 12),
            (2, 13), (5, 13), (8, 13), (11, 13),
        }
    observed_cross_residue_bonds = []
    for a, b in sorted(all_bond_pairs):
        ra = atoms[a].residue.index + 1
        rb = atoms[b].residue.index + 1
        if ra == rb:
            continue
        observed_cross_residue_bonds.append(
            {"atoms": [a, b], "atom_names": [atoms[a].name, atoms[b].name], "residues": sorted((ra, rb))}
        )
    unexpected_crosslinks = []
    if prototype_crosslink_report:
        unexpected_crosslinks = [
            row for row in observed_cross_residue_bonds if tuple(row["residues"]) not in expected_cross_residue_bonds
        ]

    exceptions = {}
    for index in range(nb_force.getNumExceptions()):
        a, b, charge_prod, sigma, epsilon = nb_force.getExceptionParameters(index)
        pair = tuple(sorted((int(a), int(b))))
        exceptions[pair] = {
            "exception_index": index,
            "charge_product_e2": float(charge_prod.value_in_unit(unit.elementary_charge**2)),
            "sigma_nm": float(sigma.value_in_unit(unit.nanometer)),
            "epsilon_kj_mol": float(epsilon.value_in_unit(unit.kilojoules_per_mole)),
        }

    def exception_rows(pairs: list[tuple[int, int]]) -> list[dict[str, Any]]:
        rows = []
        for a, b in pairs:
            pair = tuple(sorted((a, b)))
            rows.append(
                {
                    "atoms": list(pair),
                    "atom_names": [atoms[pair[0]].name, atoms[pair[1]].name],
                    "residues": [atoms[pair[0]].residue.index + 1, atoms[pair[1]].residue.index + 1],
                    "exception": exceptions.get(pair),
                }
            )
        return rows

    sg_pairs = [(expected_sg_indices[i], expected_sg_indices[j]) for i in range(4) for j in range(i + 1, 4)]
    zns_pairs = [(zn_index, sg) for sg in expected_sg_indices]
    zncb_pairs = [(zn_index, cb_by_residue[r].index) for r in site_residues]
    nonbonded_exclusions = {
        "zn_sg": exception_rows(zns_pairs),
        "sulfur_sulfur": exception_rows(sg_pairs),
        "zn_cb": exception_rows(zncb_pairs),
    }

    if len(zns_bonds) != 4:
        failures.append(f"Expected exactly 4 Zn-S bonds, observed {len(zns_bonds)}")
    if len(szns_angles) != 6:
        failures.append(f"Expected exactly 6 S-Zn-S angles, observed {len(szns_angles)}")
    if len(cbszn_angles) != 4:
        failures.append(f"Expected exactly 4 CB-S-Zn angles, observed {len(cbszn_angles)}")
    if any(atom["atom_name"] in {"HG", "HSG"} for atom in atom_rows if atom["residue_name"] == "CY1"):
        failures.append("CY1 residues contain SG proton atom HG/HSG")
    if prototype_crosslink_report and unexpected_crosslinks:
        failures.append("Unexpected inter-residue covalent crosslinks are present")

    cy1_expected = prep_contract["CY1"]
    zn1_expected = prep_contract["ZN1"]
    for atom in atom_rows:
        if atom["residue_name"] == "CY1" and atom["atom_name"] in cy1_expected:
            expected = cy1_expected[atom["atom_name"]]
            if atom["atom_type"] != expected["type"]:
                failures.append(
                    f"CY1 {atom['atom_name']} atom type {atom['atom_type']} differs from PREP {expected['type']}"
                )
            if not _close(atom["charge_e"], float(expected["charge_e"]), 1e-5):
                failures.append(
                    f"CY1 {atom['atom_name']} charge {atom['charge_e']:.5f} differs from PREP {expected['charge_e']:.5f}"
                )
        if atom["residue_name"] == "ZN1" and atom["atom_name"] in zn1_expected:
            expected = zn1_expected[atom["atom_name"]]
            if atom["atom_type"] != expected["type"]:
                failures.append(
                    f"ZN1 {atom['atom_name']} atom type {atom['atom_type']} differs from PREP {expected['type']}"
                )
            if not _close(atom["charge_e"], float(expected["charge_e"]), 1e-5):
                failures.append(
                    f"ZN1 {atom['atom_name']} charge {atom['charge_e']:.5f} differs from PREP {expected['charge_e']:.5f}"
                )

    expected_zns = constants["ZN-S1"]
    for row in zns_bonds:
        if not _close(row["length_nm"], expected_zns["openmm_r0_nm"]) or not _close(
            row["k_kj_mol_nm2"], expected_zns["openmm_k_kj_mol_nm2"]
        ):
            failures.append("Zn-S OpenMM bond constants do not match 2*Amber*K conversion")
            break
    expected_cbszn = constants["CT-S1-ZN"]
    for row in cbszn_angles:
        if not _close(row["theta_rad"], expected_cbszn["openmm_theta0_rad"]) or not _close(
            row["k_kj_mol_rad2"], expected_cbszn["openmm_k_kj_mol_rad2"]
        ):
            failures.append("CB-S-Zn OpenMM angle constants do not match 2*Amber*K conversion")
            break
    expected_sznzs = constants["S1-ZN-S1"]
    for row in szns_angles:
        if not _close(row["theta_rad"], expected_sznzs["openmm_theta0_rad"]) or not _close(
            row["k_kj_mol_rad2"], expected_sznzs["openmm_k_kj_mol_rad2"]
        ):
            failures.append("S-Zn-S OpenMM angle constants do not match 2*Amber*K conversion")
            break
    for category, rows in nonbonded_exclusions.items():
        if len(rows) != {"zn_sg": 4, "sulfur_sulfur": 6, "zn_cb": 4}[category]:
            failures.append(f"{category} exclusion set has wrong pair count")
        if any(row["exception"] is None for row in rows):
            failures.append(f"{category} exclusion set has missing NonbondedForce exception")
        if any(
            row["exception"] is not None
            and (
                abs(row["exception"]["charge_product_e2"]) > 1e-12
                or abs(row["exception"]["epsilon_kj_mol"]) > 1e-12
            )
            for row in rows
        ):
            failures.append(f"{category} exclusions are not zeroed consistently")

    residue_charges: dict[int, float] = {}
    for atom in atom_rows:
        residue = int(atom["residue_index_1based"])
        residue_charges[residue] = residue_charges.get(residue, 0.0) + float(atom["charge_e"])
    total_charge = sum(residue_charges.values())
    cy1_charge = sum(residue_charges[r] for r in site_residues)
    zn_charge = residue_charges[zn_atom.residue.index + 1]
    site_charge = cy1_charge + zn_charge
    if not _close(cy1_charge, 4 * EXPECTED_CY1_CHARGE, 1e-5):
        failures.append(f"Four CY1 charge {cy1_charge:.5f} e differs from expected {4 * EXPECTED_CY1_CHARGE:.5f} e")
    if not _close(zn_charge, EXPECTED_ZN1_CHARGE, 1e-5):
        failures.append(f"ZN1 charge {zn_charge:.5f} e differs from expected {EXPECTED_ZN1_CHARGE:.5f} e")
    if not _close(site_charge, EXPECTED_SITE_CHARGE, 1e-5):
        failures.append(f"CY1/Zn site charge {site_charge:.5f} e differs from expected {EXPECTED_SITE_CHARGE:.5f} e")

    return {
        "status": "pass" if not failures else "fail",
        "scope": "ZAFF parameter/topology contract; no MD performed",
        "counts": {
            "atoms": len(atoms),
            "residues": topology.getNumResidues(),
            "harmonic_bonds": bond_force.getNumBonds(),
            "harmonic_angles": angle_force.getNumAngles(),
            "nonbonded_exceptions": nb_force.getNumExceptions(),
        },
        "openmm_conversion_contract": {
            "amber_bond_energy": "E = K * (r - r0)^2",
            "openmm_bond_energy": "E = 0.5 * k * (r - r0)^2",
            "bond_k_conversion": "k_openmm(kJ/mol/nm^2) = 2 * K_amber(kcal/mol/A^2) * 4.184 / 0.01",
            "angle_k_conversion": "k_openmm(kJ/mol/rad^2) = 2 * K_amber(kcal/mol/rad^2) * 4.184",
            "expected_constants": constants,
        },
        "site_atoms": {
            "sg": [
                {
                    "atom_index": sg_by_residue[r].index,
                    "residue_index_1based": r,
                    "atom_type": atom_types[sg_by_residue[r].index],
                    "charge_e": charges[sg_by_residue[r].index],
                }
                for r in site_residues
            ],
            "zn": {
                "atom_index": zn_atom.index,
                "residue_index_1based": zn_atom.residue.index + 1,
                "atom_type": atom_types[zn_atom.index],
                "charge_e": charges[zn_atom.index],
            },
        },
        "charges": {
            "total_charge_e": round(total_charge, 8),
            "cy1_four_residues_charge_e": round(cy1_charge, 8),
            "zn1_charge_e": round(zn_charge, 8),
            "cy1_zn_site_charge_e": round(site_charge, 8),
            "residue_charges_e": {str(key): round(value, 8) for key, value in sorted(residue_charges.items())},
        },
        "bonded_terms": {
            "zn_s_bonds": zns_bonds,
            "s_zn_s_angles": szns_angles,
            "cb_s_zn_angles": cbszn_angles,
        },
        "nonbonded_exclusions": nonbonded_exclusions,
        "cross_residue_bonds": {
            "expected_residue_pairs": [list(pair) for pair in sorted(expected_cross_residue_bonds)],
            "observed": observed_cross_residue_bonds,
            "unexpected": unexpected_crosslinks,
        },
        "failures": failures,
    }


def verify_topology(
    prmtop: Path,
    coordinates: Path,
    leap_log: Path | None = None,
    prep_path: Path | None = None,
) -> dict[str, Any]:
    from openmm import app

    topology_file = app.AmberPrmtopFile(str(prmtop))
    coordinate_file = app.AmberInpcrdFile(str(coordinates))
    system = topology_file.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
    atoms = list(topology_file.topology.atoms())
    if len(atoms) != len(coordinate_file.positions):
        raise ValueError("Topology and coordinate atom counts differ")
    report = parameter_contract(
        system,
        topology_file.topology,
        prep_path=prep_path,
        atom_types=topology_file._prmtop.getAtomTypes(),
        expected_sg_residues=EXPECTED_SG_RESIDUES,
        prototype_crosslink_report=True,
    )
    warning_report = classify_leap_warnings(leap_log)
    failures = list(report["failures"])
    if warning_report.get("errors_total") not in (None, 0):
        failures.append("LEaP log reports errors")
    if warning_report.get("categories", {}).get("parameter_warning", 0):
        failures.append("LEaP log contains parameter-related warnings")
    report.update(
        {
            "status": "pass" if not failures else "fail",
            "failures": failures,
            "scope": "ZAFF prototype parameter/topology audit; no MD performed",
            "geometry_status": "geometry_not_qualified",
            "md_direct_use": False,
            "chemical_model_limits": [
                "The prototype proves bonded parameter transfer into Amber/OpenMM for four capped CY1 fragments plus ZN1.",
                "It does not validate CRBN production geometry because peptide fragments remain unaligned and overlapping.",
                "It does not validate force-field transferability, sampling stability, solvation, protonation outside the Cys4 site, or capped-fragment representativeness.",
            ],
            "sources": {
                "prmtop": {"path": str(prmtop), "sha256": sha256_file(prmtop)},
                "coordinates": {"path": str(coordinates), "sha256": sha256_file(coordinates)},
                "prep": None if prep_path is None else {"path": str(prep_path), "sha256": sha256_file(prep_path)},
                "leap_log": None if leap_log is None else {"path": str(leap_log), "sha256": sha256_file(leap_log)},
            },
            "leap_warnings": warning_report,
            "charges": {
                **report["charges"],
                "peptide_charges_e": {
                    "p1": round(sum(report["charges"]["residue_charges_e"][str(r)] for r in (1, 2, 3)), 8),
                    "p2": round(sum(report["charges"]["residue_charges_e"][str(r)] for r in (4, 5, 6)), 8),
                    "p3": round(sum(report["charges"]["residue_charges_e"][str(r)] for r in (7, 8, 9)), 8),
                    "p4": round(sum(report["charges"]["residue_charges_e"][str(r)] for r in (10, 11, 12)), 8),
                },
            },
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--prep", type=Path)
    parser.add_argument("--leap-log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = verify_topology(args.prmtop, args.coordinates, args.leap_log, args.prep)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
