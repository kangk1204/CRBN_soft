#!/usr/bin/env python3
"""Correct modeled L-C-alpha side chains before minimization, without running MD.

Input is a heavy-only capped PDB and the heavy-repair atom_residue_mapping.csv
(chain, residue_number, residue_name, atom, source_status, x, y, z in Angstrom).
Only ALA/LEU/MET/TYR/HIS C-alpha side chains whose every atom has a recognized
modeled source status can be reflected. Negative C-alpha ILE/THR/PRO and other
residues are not handled. A separate C-beta repair for ILE/THR reflects only
modeled CG2 through CB/CA/CG1 (ILE) or CB/CA/OG1 (THR); the other branch stays
fixed. Backbone, observed atoms, caps and metal stay fixed.

The signed volume is (N-CA) dot ((C-CA) cross (CB-CA)); a retained LEaP
ACE-MET-NME template must establish the positive L sign. Reflection is an
isometry of the entire side chain through the N/CA/C plane. This only repairs
the sign: near-planarity and clashes still require downstream preparation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path

import numpy as np

try:
    from atomistic_caps import AtomRecord, parse_pdb_atom_line, read_pdb, sha256_file
    from atomistic_input_audit import STANDARD_HEAVY_ATOMS
except ImportError:  # pragma: no cover
    from scripts.atomistic_caps import AtomRecord, parse_pdb_atom_line, read_pdb, sha256_file
    from scripts.atomistic_input_audit import STANDARD_HEAVY_ATOMS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "tests/fixtures/atomistic_caps/ace_met_nme.pdb"
DEFAULT_OUTPUT = ROOT / "results/atomistic/stereochemistry"
BACKBONE = {"N", "CA", "C", "O", "OXT"}
HANDLED_RESIDUES = {"ALA", "LEU", "MET", "TYR", "HIS"}
MODELED_STATUSES = {"repaired_internal_missing_residue_or_atom", "repaired_missing_heavy_atom"}
VOLUME_EPS = 1e-10
NEAR_FLAT_DEGREES = 10.0  # Descriptive reporting threshold, not a quality gate.
PDB_DISTANCE_ROUNDING_BOUND = np.sqrt(3.0) * 0.001 + 1e-9


def atom_key(atom: AtomRecord) -> tuple[str, int, str]:
    return atom.chain, atom.resseq, atom.name


def strict_atoms(lines: list[str]) -> list[AtomRecord]:
    atoms, seen, model_count = [], set(), 0
    for line in lines:
        if line.startswith("MODEL "):
            model_count += 1
            if model_count > 1 or line[10:14].strip() != "1":
                raise ValueError("Only one unambiguous first PDB model is supported")
        atom = parse_pdb_atom_line(line)
        if atom is None:
            continue
        if atom.altloc or atom.icode or not np.isfinite(atom.xyz).all():
            raise ValueError("Alternate locations, insertion codes and nonfinite coordinates are unsupported")
        if atom.is_hydrogen:
            raise ValueError("Repair requires a heavy-only PDB before hydrogen addition")
        key = atom_key(atom)
        if key in seen:
            raise ValueError(f"Duplicate atom identity: {key}")
        seen.add(key)
        atoms.append(atom)
    if not atoms:
        raise ValueError("Input contains no atoms")
    return atoms


def residue_groups(atoms: list[AtomRecord]) -> dict[tuple[str, int, str], dict[str, AtomRecord]]:
    groups, identities = {}, {}
    for atom in atoms:
        identity = (atom.chain, atom.resseq)
        if identities.setdefault(identity, atom.resname) != atom.resname:
            raise ValueError(f"Conflicting residue names: {identity}")
        groups.setdefault((*identity, atom.resname), {})[atom.name] = atom
    return groups


def ca_geometry(group: dict[str, AtomRecord]) -> tuple[float, float]:
    return stereo_geometry(group, "CA", "N", "C", "CB")


def stereo_geometry(group: dict[str, AtomRecord], center: str, first: str, second: str, branch: str) -> tuple[float, float]:
    if not {center, first, second, branch} <= set(group):
        raise ValueError(f"Chiral residue lacks {center}, {first}, {second} or {branch}")
    n, c, cb = (group[name].xyz - group[center].xyz for name in (first, second, branch))
    normal = np.cross(n, c)
    denominator = float(np.linalg.norm(normal) * np.linalg.norm(cb))
    if denominator < 1e-12:
        raise ValueError("Degenerate stereocenter plane or branch vector")
    volume = float(np.dot(n, np.cross(c, cb)))
    angle = float(np.degrees(np.arcsin(np.clip(abs(volume) / denominator, 0, 1))))
    return volume, angle


def beta_geometry(group: dict[str, AtomRecord], resname: str) -> tuple[float, float]:
    if resname not in {"ILE", "THR"}:
        raise ValueError("Only ILE and THR C-beta centers are handled")
    return stereo_geometry(group, "CB", "CA", "CG1" if resname == "ILE" else "OG1", "CG2")


def beta_inventory(atoms: list[AtomRecord], mapping: dict) -> dict:
    rows, observed_reference = [], {"ILE": {"positive": 0, "negative": 0}, "THR": {"positive": 0, "negative": 0}}
    for key, group in residue_groups(atoms).items():
        if key[2] not in {"ILE", "THR"}:
            continue
        volume, angle = beta_geometry(group, key[2])
        x = "CG1" if key[2] == "ILE" else "OG1"
        all_observed = all(mapping.get(atom_key(group[name]), {}).get("source_status") == "observed_input" for name in ("CA", "CB", x, "CG2"))
        if all_observed and abs(volume) > VOLUME_EPS:
            observed_reference[key[2]]["positive" if volume > 0 else "negative"] += 1
        rows.append({"residue": list(key), "signed_volume_A3": volume,
                     "cg2_out_of_plane_degrees": angle, "four_center_atoms_observed": all_observed})
    return {
        "volume_convention": "(CA-CB) dot ((X-CB) cross (CG2-CB)); X=CG1 for ILE, OG1 for THR; expected positive",
        "positive_count": sum(r["signed_volume_A3"] > VOLUME_EPS for r in rows),
        "negative_count": sum(r["signed_volume_A3"] < -VOLUME_EPS for r in rows),
        "unresolved_planar_count": sum(abs(r["signed_volume_A3"]) <= VOLUME_EPS for r in rows),
        "negative_residues": [r for r in rows if r["signed_volume_A3"] < -VOLUME_EPS],
        "minimum_absolute_volume_A3": min((abs(r["signed_volume_A3"]) for r in rows), default=None),
        "minimum_out_of_plane_angle_degrees": min((r["cg2_out_of_plane_degrees"] for r in rows), default=None),
        "near_flat_threshold_degrees_descriptive_only": NEAR_FLAT_DEGREES,
        "near_flat_residues": [r for r in rows if r["cg2_out_of_plane_degrees"] < NEAR_FLAT_DEGREES],
        "all_observed_sign_reference": observed_reference,
        "chiral_residue_count": len(rows),
    }


def chirality_inventory(atoms: list[AtomRecord]) -> dict:
    rows = []
    for key, group in residue_groups(atoms).items():
        if key[2] == "GLY" and "CB" not in group:
            continue
        if "CA" not in group and key[2] not in STANDARD_HEAVY_ATOMS:
            continue  # ACE, NME and Zn are not C-alpha stereocenters.
        volume, angle = ca_geometry(group)
        rows.append({"residue": list(key), "signed_volume_A3": volume,
                     "cb_out_of_backbone_plane_degrees": angle})
    return {
        "positive_count": sum(r["signed_volume_A3"] > VOLUME_EPS for r in rows),
        "negative_count": sum(r["signed_volume_A3"] < -VOLUME_EPS for r in rows),
        "unresolved_planar_count": sum(abs(r["signed_volume_A3"]) <= VOLUME_EPS for r in rows),
        "negative_residues": [r for r in rows if r["signed_volume_A3"] < -VOLUME_EPS],
        "near_flat_threshold_degrees_descriptive_only": NEAR_FLAT_DEGREES,
        "near_flat_residues": [r for r in rows if r["cb_out_of_backbone_plane_degrees"] < NEAR_FLAT_DEGREES],
        "minimum_out_of_plane_angle_degrees": min((r["cb_out_of_backbone_plane_degrees"] for r in rows), default=None),
        "chiral_residue_count": len(rows),
    }


def read_repair_mapping(path: Path) -> dict:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"chain", "residue_number", "residue_name", "atom", "source_status", "x", "y", "z"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Repair CSV is missing identity, provenance or coordinate columns")
        for row in reader:
            key = (row["chain"], int(row["residue_number"]), row["atom"])
            xyz = np.array([float(row[name]) for name in ("x", "y", "z")])
            if key in result or not np.isfinite(xyz).all():
                raise ValueError(f"Duplicate or nonfinite repair CSV record: {key}")
            result[key] = {"residue_name": row["residue_name"], "source_status": row["source_status"], "xyz": xyz}
    if not result:
        raise ValueError("Repair CSV contains no atoms")
    return result


def validate_mapping_coordinates(atoms: list[AtomRecord], mapping: dict) -> int:
    lookup = {atom_key(atom): atom for atom in atoms}
    observed_count = 0
    for key, source in mapping.items():
        atom = lookup.get(key)
        if atom is None or atom.resname != source["residue_name"]:
            raise ValueError(f"Repair CSV atom missing or renamed in input: {key}")
        if np.max(np.abs(atom.xyz - source["xyz"])) > 0.000501:
            raise ValueError(f"Input coordinates do not match repair CSV at PDB precision: {key}")
        observed_count += source["source_status"] == "observed_input"
    return observed_count


def reflect_points(points: np.ndarray, ca: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    normal = np.cross(n - ca, c - ca)
    length = float(np.linalg.norm(normal))
    if not np.isfinite(points).all() or length < 1e-12:
        raise ValueError("Cannot reflect nonfinite points or a degenerate plane")
    normal /= length
    return points - 2 * np.outer((points - ca) @ normal, normal)


def geometry_change(before: np.ndarray, after: np.ndarray) -> dict:
    pair_delta = 0.0
    angle_delta = 0.0
    for i, j in combinations(range(len(before)), 2):
        pair_delta = max(pair_delta, abs(float(np.linalg.norm(before[i]-before[j]) - np.linalg.norm(after[i]-after[j]))))
    # All triple angles include every intra-side-chain bond angle without
    # assuming an external bond topology. CA/N/C are also fixed anchors.
    for center in range(len(before)):
        for i, j in combinations([k for k in range(len(before)) if k != center], 2):
            angles = []
            for xyz in (before, after):
                u, v = xyz[i]-xyz[center], xyz[j]-xyz[center]
                if min(np.linalg.norm(u), np.linalg.norm(v)) < 1e-12:
                    raise ValueError("Coincident atoms prevent an angle-preservation check")
                angles.append(np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v)))
            angle_delta = max(angle_delta, abs(float(np.degrees(angles[0]-angles[1]))))
    return {"max_pair_distance_change_A": pair_delta, "max_triple_angle_change_degrees": angle_delta}


def corrected_atoms(atoms: list[AtomRecord], mapping: dict) -> tuple[list[AtomRecord], list[dict]]:
    corrections = {}
    reports = []
    for key, group in residue_groups(atoms).items():
        if not {"N", "CA", "C", "CB"} <= set(group):
            continue
        volume, _ = ca_geometry(group)
        if volume >= -VOLUME_EPS:
            continue
        if key[2] not in HANDLED_RESIDUES:
            raise ValueError(f"Unhandled negative C-alpha stereochemistry; no reflection allowed: {key}")
        required = STANDARD_HEAVY_ATOMS[key[2]]
        if not required <= set(group) or set(group) - required - {"OXT"}:
            raise ValueError(f"Incomplete or unhandled side-chain atom inventory: {key}")
        sidechain = [group[name] for name in sorted(set(group) - BACKBONE)]
        for atom in sidechain:
            source = mapping.get(atom_key(atom))
            if source is None or source["residue_name"] != atom.resname or source["source_status"] not in MODELED_STATUSES:
                raise ValueError(f"Side chain is not entirely proven modeled; observed or unknown atom: {atom_key(atom)}")
        xyz = np.array([atom.xyz for atom in sidechain])
        reflected = reflect_points(xyz, group["CA"].xyz, group["N"].xyz, group["C"].xyz)
        anchors = np.array([group[name].xyz for name in ("N", "CA", "C")])
        change = geometry_change(np.vstack((xyz, anchors)), np.vstack((reflected, anchors)))
        if change["max_pair_distance_change_A"] > 1e-10 or change["max_triple_angle_change_degrees"] > 1e-8:
            raise ValueError(f"Reflection failed its isometry check: {key}")
        for atom, point in zip(sidechain, reflected):
            corrections[atom_key(atom)] = replace(atom, x=float(point[0]), y=float(point[1]), z=float(point[2]))
        reports.append({"residue": list(key), "stereocenter": "CA", "before_signed_volume_A3": volume,
                        "fixed_anchor_names": ["N", "CA", "C"],
                        "sidechain_atom_keys": [list(atom_key(a)) for a in sidechain],
                        "floating_point_isometry": change})
    return [corrections.get(atom_key(atom), atom) for atom in atoms], reports


def corrected_beta_atoms(atoms: list[AtomRecord], mapping: dict) -> tuple[list[AtomRecord], list[dict]]:
    corrections, reports = {}, []
    for key, group in residue_groups(atoms).items():
        if key[2] not in {"ILE", "THR"}:
            continue
        volume, _ = beta_geometry(group, key[2])
        if volume >= -VOLUME_EPS:
            continue
        atom = group["CG2"]
        source = mapping.get(atom_key(atom))
        if source is None or source["residue_name"] != atom.resname or source["source_status"] not in MODELED_STATUSES:
            raise ValueError(f"C-beta repair requires modeled CG2; observed or unknown atom: {atom_key(atom)}")
        if set(group) - STANDARD_HEAVY_ATOMS[key[2]] - {"OXT"} or not STANDARD_HEAVY_ATOMS[key[2]] <= set(group):
            raise ValueError(f"Unhandled ILE/THR atom inventory: {key}")
        x = "CG1" if key[2] == "ILE" else "OG1"
        anchor_names = ["CB", "CA", x]
        anchors = np.array([group[name].xyz for name in anchor_names])
        reflected = reflect_points(np.array([atom.xyz]), *anchors)[0]
        change = geometry_change(np.vstack((atom.xyz, anchors)), np.vstack((reflected, anchors)))
        if change["max_pair_distance_change_A"] > 1e-10 or change["max_triple_angle_change_degrees"] > 1e-8:
            raise ValueError(f"C-beta branch reflection failed its isometry check: {key}")
        corrections[atom_key(atom)] = replace(atom, x=float(reflected[0]), y=float(reflected[1]), z=float(reflected[2]))
        reports.append({"residue": list(key), "stereocenter": "CB", "before_signed_volume_A3": volume,
                        "fixed_anchor_names": anchor_names, "sidechain_atom_keys": [list(atom_key(atom))],
                        "floating_point_isometry": change,
                        "unchanged_other_branch": [x, "CD1"] if key[2] == "ILE" else [x]})
    return [corrections.get(atom_key(atom), atom) for atom in atoms], reports


def validate_template(path: Path) -> dict:
    _, atoms = read_pdb(path)
    if not {"ACE", "MET", "NME"} <= {a.resname for a in atoms}:
        raise ValueError("L-sign reference must be the retained ACE-MET-NME template")
    groups = [group for key, group in residue_groups(atoms).items() if key[2] == "MET"]
    if len(groups) != 1:
        raise ValueError("L-sign template must contain exactly one MET")
    volume, angle = ca_geometry(groups[0])
    if volume <= VOLUME_EPS:
        raise ValueError("LEaP L-MET template does not establish a positive sign")
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "signed_volume_A3": volume, "cb_out_of_backbone_plane_degrees": angle}


def repair(input_pdb: Path, repair_csv: Path, output_dir: Path, template_pdb: Path = DEFAULT_TEMPLATE) -> dict:
    output_pdb = output_dir / "capped_heavy.pdb"
    if output_pdb.resolve() == input_pdb.resolve():
        raise ValueError("Output must be separate from the preserved input PDB")
    inputs = (input_pdb, repair_csv, template_pdb)
    hashes = {str(path.resolve()): sha256_file(path) for path in inputs}
    template = validate_template(template_pdb)
    lines = input_pdb.read_text(encoding="utf-8").splitlines(keepends=True)
    atoms = strict_atoms(lines)
    mapping = read_repair_mapping(repair_csv)
    observed_count = validate_mapping_coordinates(atoms, mapping)
    before = chirality_inventory(atoms)
    beta_before = beta_inventory(atoms, mapping)
    if before["unresolved_planar_count"] or beta_before["unresolved_planar_count"]:
        raise ValueError("Planar stereocenter geometry cannot be repaired by a sign reflection")
    corrected, changes = corrected_atoms(atoms, mapping)
    corrected, beta_changes = corrected_beta_atoms(corrected, mapping)
    changes += beta_changes
    lookup = {atom_key(atom): atom for atom in corrected}
    moved_keys = {tuple(key) for change in changes for key in change["sidechain_atom_keys"]}
    out_lines = []
    for line in lines:
        atom = parse_pdb_atom_line(line)
        if atom is not None and atom_key(atom) in moved_keys:
            point = lookup[atom_key(atom)]
            coordinates = f"{point.x:8.3f}{point.y:8.3f}{point.z:8.3f}"
            if len(coordinates) != 24:
                raise ValueError("Reflected coordinate overflows the fixed PDB fields")
            line = line[:30] + coordinates + line[54:]
        out_lines.append(line)
    serialized = strict_atoms(out_lines)
    after = chirality_inventory(serialized)
    beta_after = beta_inventory(serialized, mapping)
    if any(inventory[key] for inventory in (after, beta_after) for key in ("negative_count", "unresolved_planar_count")):
        raise ValueError("Serialized output still has negative or unresolved stereochemistry")
    original_lookup = {atom_key(atom): atom for atom in atoms}
    serial_lookup = {atom_key(atom): atom for atom in serialized}
    atom_changes = []
    for key in sorted(moved_keys):
        old, new = original_lookup[key], serial_lookup[key]
        atom_changes.append({"atom": list(key), "residue_name": old.resname,
                             "source_status": mapping[key]["source_status"],
                             "before_xyz_A": old.xyz.tolist(), "after_xyz_A": new.xyz.tolist()})
    for old, new in zip(lines, out_lines):
        atom = parse_pdb_atom_line(old)
        if atom is None or atom_key(atom) not in moved_keys:
            if old != new:
                raise ValueError("An unselected atom or record changed")
    groups_before, groups_after = residue_groups(atoms), residue_groups(serialized)
    for change in changes:
        key = tuple(change["residue"])
        names = [atom[2] for atom in change["sidechain_atom_keys"]] + change["fixed_anchor_names"]
        delta = geometry_change(np.array([groups_before[key][name].xyz for name in names]),
                                np.array([groups_after[key][name].xyz for name in names]))
        if delta["max_pair_distance_change_A"] > PDB_DISTANCE_ROUNDING_BOUND:
            raise ValueError("PDB serialization exceeded its coordinate-rounding distance bound")
        change["serialized_geometry_change"] = delta
        geometry = ca_geometry(groups_after[key]) if change["stereocenter"] == "CA" else beta_geometry(groups_after[key], key[2])
        change["after_signed_volume_A3"], change["after_out_of_plane_angle_degrees"] = geometry
    if hashes != {str(path.resolve()): sha256_file(path) for path in inputs}:
        raise ValueError("An input changed during repair")
    payload = "".join(out_lines)
    if output_pdb.exists() and output_pdb.read_text() != payload:
        raise FileExistsError("Refusing to overwrite a different prior stereochemistry output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text(payload, encoding="utf-8")
    report = {
        "status": "corrected_premin_only" if changes else "no_negative_ca_signs_premin_only",
        "production_ready": False, "minimization_performed": False, "MD_performed": False,
        "method": "C-alpha: reflect modeled whole side chain through N/CA/C. C-beta ILE/THR: reflect modeled CG2 only through CB/CA/X.",
        "volume_convention": "(N-CA) dot ((C-CA) cross (CB-CA)); positive L sign checked against LEaP MET",
        "limitations": "Sign repair does not fix near-planarity, clashes, improper terms, or validate chemistry; downstream minimization and stereochemical inspection remain required.",
        "script_sha256": sha256_file(Path(__file__)), "input_sha256": hashes,
        "inputs_unchanged": True, "L_sign_template": template,
        "output_pdb": str(output_pdb.resolve()), "output_pdb_sha256": sha256_file(output_pdb),
        "before": before, "after": after, "beta_before": beta_before, "beta_after": beta_after,
        "corrected_residue_count": len(changes), "corrected_alpha_count": len(changes)-len(beta_changes),
        "corrected_beta_count": len(beta_changes),
        "moved_sidechain_atom_count": len(moved_keys), "changes": changes, "changed_atoms": atom_changes,
        "preservation": {"unselected_records_byte_identical": True, "backbone_exact": True,
                         "observed_atoms_exact": True, "observed_atom_count": observed_count,
                         "Zn_exact": True, "Zn_count": sum(a.element.upper() == "ZN" for a in atoms)},
        "geometry_note": "Alpha reflection preserves whole-side-chain/anchor distances and angles. Beta reflection preserves CG2 distances/angles with CB, CA, X; the other branch is unchanged. PDB 0.001 A rounding errors are separate.",
        "post_minimization_handoff": "Check CA and ILE/THR CB signs, minimum absolute volumes and out-of-plane angles again after minimization. Near-flat positive centers remain unresolved premin geometry; sign repair is not a final chirality gate.",
        "serialized_pair_distance_rounding_bound_A": float(PDB_DISTANCE_ROUNDING_BOUND),
    }
    (output_dir / "stereochemistry.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repair-csv", type=Path, required=True)
    parser.add_argument("--l-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="All operations are local; no network access is implemented.")
    args = parser.parse_args(argv)
    report = repair(args.input, args.repair_csv, args.output_dir, args.l_template)
    print(json.dumps({key: report[key] for key in ("status", "production_ready", "corrected_residue_count", "moved_sidechain_atom_count", "output_pdb", "output_pdb_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
