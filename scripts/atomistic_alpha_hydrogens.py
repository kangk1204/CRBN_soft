#!/usr/bin/env python3
"""Reconstruct newly added C-alpha hydrogens without changing heavy atoms.

The direction opposite the sum of the three unit heavy-bond vectors is a
tetrahedral bisector approximation. It is exact for an ideal tetrahedron and
rotation/translation equivariant. It is a coordinate preparation operation,
not a new force field or a geometry/chemistry qualification. Distorted heavy
geometry is reported and remains unchanged; H-only relaxation and the complete
post-preparation qualification are separate operations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MIN_REFERENCE_VOLUME_NM3 = 1e-4
PROTEIN_RESIDUES = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL HID HIE HIP CY1 CYM CYX ASH GLH LYN".split())


@dataclass(frozen=True)
class AlphaHydrogenRepair:
    positions_nm: np.ndarray
    metadata: dict[str, Any]


def _signed_volume(n: np.ndarray, ca: np.ndarray, c: np.ndarray, fourth: np.ndarray) -> float:
    return float(np.dot(np.cross(n-ca, c-ca), fourth-ca))


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.dot(first, second) / np.linalg.norm(first) / np.linalg.norm(second)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def repair_alpha_hydrogens(structure: Any, positions_nm: np.ndarray) -> AlphaHydrogenRepair:
    """Return a copy with every non-Gly protein CA-HA reconstructed.

    ``structure`` is a parameterized ParmEd Structure/AmberParm. The actual
    CA-HA bond equilibrium distance is read from its bond type (Angstrom).
    Every target is checked before a result is returned. Neither the input
    coordinate array nor the topology is modified. A positive L C-alpha
    reference volume above 1e-4 nm^3 is required, matching the preparation
    handoff's nonplanarity threshold; this does not certify its bond angles.
    """
    xyz = np.asarray(positions_nm, dtype=float)
    if xyz.shape != (len(structure.atoms), 3) or not np.all(np.isfinite(xyz)):
        raise ValueError("positions_nm must be a finite atom_count by 3 array")
    updated = xyz.copy()
    targets: list[int] = []
    rows: list[dict[str, Any]] = []
    glycine_count = 0
    for residue in structure.residues:
        atom_names = [atom.name for atom in residue.atoms]
        is_protein = residue.name in PROTEIN_RESIDUES or {"N", "CA", "C"} <= set(atom_names)
        if not is_protein:
            continue
        label = f"{residue.name} topology residue {residue.idx}"
        if len(atom_names) != len(set(atom_names)):
            raise ValueError(f"Ambiguous duplicate atom names in {label}")
        if residue.name == "GLY":
            glycine_count += 1
            continue
        names = {atom.name: atom for atom in residue.atoms}
        required = {"N", "CA", "C", "CB", "HA"}
        if not required <= names.keys():
            raise ValueError(f"Missing non-Gly alpha-hydrogen reference atoms in {label}: {sorted(required-names.keys())}")
        for name, number in {"N": 7, "CA": 6, "C": 6, "CB": 6, "HA": 1}.items():
            if int(names[name].atomic_number) != number:
                raise ValueError(f"Unexpected topology element for {label} {name}")
        ca_atom, ha_atom = names["CA"], names["HA"]
        expected_neighbors = {names[name].idx for name in ("N", "C", "CB", "HA")}
        if {a.idx for a in ca_atom.bond_partners} != expected_neighbors:
            raise ValueError(f"Unexpected CA connectivity in {label}")
        if {a.idx for a in ha_atom.bond_partners} != {ca_atom.idx}:
            raise ValueError(f"HA must have exactly one CA bond in {label}")
        bonds = [b for b in ca_atom.bonds if {b.atom1.idx, b.atom2.idx} == {ca_atom.idx, ha_atom.idx}]
        if len(bonds) != 1 or bonds[0].type is None:
            raise ValueError(f"Missing or ambiguous parameterized CA-HA bond in {label}")
        bond_nm = float(bonds[0].type.req) * 0.1
        if not np.isfinite(bond_nm) or bond_nm <= 0:
            raise ValueError(f"Invalid CA-HA equilibrium bond distance in {label}")
        indices = {name: atom.idx for name, atom in names.items() if name in required}
        ca = xyz[ca_atom.idx]
        vectors = np.array([xyz[names[name].idx]-ca for name in ("N", "C", "CB")])
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= 1e-10):
            raise ValueError(f"Degenerate heavy-bond reference in {label}")
        volume = _signed_volume(xyz[names["N"].idx], ca, xyz[names["C"].idx], xyz[names["CB"].idx])
        if volume <= MIN_REFERENCE_VOLUME_NM3:
            raise ValueError(f"Non-L or nearly planar C-alpha reference in {label}: {volume:.9g} nm^3")
        direction = -np.sum(vectors / lengths[:, None], axis=0)
        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm) or direction_norm <= 1e-10:
            raise ValueError(f"Degenerate/ambiguous hydrogen bisector in {label}")
        direction /= direction_norm
        hydrogen = ca + bond_nm * direction
        volume_h = _signed_volume(xyz[names["N"].idx], ca, xyz[names["C"].idx], hydrogen)
        if not np.all(np.isfinite(hydrogen)) or volume_h >= 0:
            raise ValueError(f"Constructed HA is not opposite the L-CB reference in {label}")
        if not np.isclose(np.linalg.norm(hydrogen-ca), bond_nm, rtol=1e-12, atol=1e-14):
            raise ValueError(f"Constructed CA-HA bond length lost precision in {label}")
        original_h = xyz[ha_atom.idx]-ca
        if np.linalg.norm(original_h) <= 1e-10:
            original_angles = None
        else:
            original_angles = {name: _angle(vectors[i], original_h) for i, name in enumerate(("N", "C", "CB"))}
        updated[ha_atom.idx] = hydrogen
        targets.append(ha_atom.idx)
        rows.append({
            "topology_residue_index": residue.idx, "residue_name": residue.name,
            "atom_indices": indices, "CA_HA_equilibrium_nm": bond_nm,
            "pre_CA_HA_nm": float(np.linalg.norm(original_h)),
            "post_CA_HA_nm": float(np.linalg.norm(hydrogen-ca)),
            "reference_CA_signed_volume_nm3": volume,
            "pre_HA_signed_volume_nm3": _signed_volume(xyz[names["N"].idx], ca, xyz[names["C"].idx], xyz[ha_atom.idx]),
            "post_HA_signed_volume_nm3": volume_h,
            "pre_HA_angles_degrees": original_angles,
            "post_HA_angles_degrees": {name: _angle(vectors[i], direction) for i, name in enumerate(("N", "C", "CB"))},
            "unchanged_heavy_angles_degrees": {"N_CA_C": _angle(vectors[0], vectors[1]), "N_CA_CB": _angle(vectors[0], vectors[2]), "C_CA_CB": _angle(vectors[1], vectors[2])},
            "HA_displacement_nm": float(np.linalg.norm(hydrogen-xyz[ha_atom.idx])),
        })
    if not targets:
        raise ValueError("No non-Gly protein CA-HA targets found")
    untouched = np.ones(len(xyz), dtype=bool)
    untouched[targets] = False
    if not np.array_equal(updated[untouched], xyz[untouched]):
        raise AssertionError("Non-target coordinates changed")
    return AlphaHydrogenRepair(updated, {
        "method": "negative normalized sum of unit CA-to-N/C/CB vectors; tetrahedral bisector approximation",
        "selection": "all non-Gly protein CA-HA atoms, independent of earlier minimization outcomes",
        "alpha_hydrogen_count": len(targets), "alpha_hydrogen_indices": targets,
        "skipped_glycine_count": glycine_count, "minimum_reference_volume_nm3": MIN_REFERENCE_VOLUME_NM3,
        "all_non_target_coordinates_exact": True, "heavy_coordinates_exact": True,
        "topology_and_force_field_changed": False, "rows": rows,
        "geometry_qualified": False, "production_ready": False,
        "limitation": "Heavy geometry remains as supplied; distorted modeled loops and heavy contacts require separate correction/relaxation and full qualification.",
    })


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(prmtop: Path, inpcrd: Path, output_dir: Path) -> dict[str, Any]:
    """Write a new positions array, Amber restart and provenance report; no MD."""
    import parmed as pmd
    from openmm import app, unit
    try:
        from .atomistic_preparation_handoff import export_restart_with_parmed, verify_restart_roundtrip
    except ImportError:  # Executed directly from scripts/.
        from atomistic_preparation_handoff import export_restart_with_parmed, verify_restart_roundtrip

    prmtop, inpcrd, output_dir = Path(prmtop).resolve(), Path(inpcrd).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output-dir must be new; existing preparation evidence is never overwritten")
    sources = {"prmtop": prmtop, "inpcrd": inpcrd}
    initial_hashes = {key: _sha(path) for key, path in sources.items()}
    structure = pmd.load_file(str(prmtop))
    restart = app.AmberInpcrdFile(str(inpcrd))
    xyz = np.asarray(restart.positions.value_in_unit(unit.nanometer), dtype=float)
    if restart.boxVectors is None:
        raise ValueError("A periodic Amber restart with explicit box vectors is required")
    box = np.asarray(restart.boxVectors.value_in_unit(unit.nanometer), dtype=float)
    if box.shape != (3, 3) or not np.all(np.isfinite(box)) or np.linalg.det(box) <= 0:
        raise ValueError("Invalid periodic box")
    repaired = repair_alpha_hydrogens(structure, xyz)
    output_dir.mkdir(parents=True, exist_ok=False)
    positions_path = output_dir / "alpha_hydrogen_positions.npy"
    box_path = output_dir / "box_vectors_nm.npy"
    restart_path = output_dir / "alpha_hydrogen_repaired.inpcrd"
    np.save(positions_path, repaired.positions_nm)
    np.save(box_path, box)
    export_restart_with_parmed(prmtop, inpcrd, repaired.positions_nm, box, restart_path)
    roundtrip = verify_restart_roundtrip(prmtop, restart_path, repaired.positions_nm, box)
    after_hashes = {key: _sha(path) for key, path in sources.items()}
    if after_hashes != initial_hashes:
        raise RuntimeError("An input changed during alpha-hydrogen preparation; outputs are unqualified")
    report = {
        "schema_version": "1.0", "status": "alpha_hydrogen_coordinate_repair_complete",
        "scope": "coordinate preparation only; no minimization, MD or geometry qualification",
        "MD_performed": False, "minimization_performed": False,
        "atom_count": len(xyz), "repair": repaired.metadata,
        "source_files": {key: {"path": str(path), "sha256": initial_hashes[key]} for key, path in sources.items()},
        "source_sha256_after": after_hashes, "inputs_unchanged": True,
        "script_sha256": _sha(Path(__file__).resolve()),
        "outputs": {key: {"path": str(path), "sha256": _sha(path)} for key, path in {"positions": positions_path, "box": box_path, "restart": restart_path}.items()},
        "roundtrip": roundtrip, "production_ready": False,
    }
    (output_dir / "alpha_hydrogen_repair.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", required=True, type=Path)
    parser.add_argument("--inpcrd", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run(args.prmtop, args.inpcrd, args.output_dir)
    print(json.dumps({"status": report["status"], "alpha_hydrogen_count": report["repair"]["alpha_hydrogen_count"], "roundtrip": report["roundtrip"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
