#!/usr/bin/env python3
"""Bounded torsion-space reconstruction of a completely modeled protein loop.

No minimization, force field, hydrogen addition or MD is run here. All geometry
uses nm. A successful result only passes the stated loop geometry checks.

JSON --input schema (version 1):
  coordinate_unit: "nm"
  donor: {pdb_id, label_asym_id, source_sha256,
    canonical_mapping: [{canonical_residue, label_seq_id, residue_name}, ...],
    atoms: [{canonical_residue, label_seq_id, residue_name, atom_name,
             xyz_nm: [x,y,z]}, ...],
    additional_bonds: [[[residue, atom], [residue, atom]], ...]}
  target: {chain, anchor_residues: [left, right],
           loop_residues: [left+1, ..., right-1]}
All residues from left through right require their complete standard heavy
atoms. Explicit additional_bonds (even an empty list) declare connectivity
beyond standard peptide/side-chain bonds; cyclic cuts fail closed. Target
anchor N/CA/C/O coordinates come exclusively from --pdb. --source-csv is the
heavy-repair inventory, with chain/residue_number/residue_name/atom/source_status
and x/y/z (Angstrom). Every replaced atom must have a recognized modeled status.

The solver changes phi/psi and six global rigid-body degrees of freedom. Pro
phi is fixed; omega, covalent lengths/angles, side-chain conformations and ring
geometry remain fixed. Eight anchor atoms are fit with a fixed small torsion
penalty, using a fixed seeded start set. Actual junction geometry is checked
after substituting the untouched target anchors and again after PDB rounding.
Failed candidates are retained as unqualified; qualified_for_md is always false.

Optional --steric-aware adds a fixed preparatory heavy-atom separation policy.
It checks loop versus every unchanged heavy atom (observed or modeled), and
loop internal pairs, excluding graph distances one and two (bonds and 1-3).
The minimum separation is 0.20 nm; the objective uses a 0.22 nm target with
0.02 nm penalty scale. No periodic box, radii, charges or force-field energies
are used. Passing this broad gross-clash screen is not vdW/FF qualification.
The option does not alter any existing geometry gate or the fixed search budget.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from atomistic_caps import parse_pdb_atom_line, sha256_file
    from atomistic_input_audit import STANDARD_HEAVY_ATOMS
    from atomistic_modeled_stereochemistry import MODELED_STATUSES, atom_key, read_repair_mapping, strict_atoms
except ImportError:  # pragma: no cover
    from scripts.atomistic_caps import parse_pdb_atom_line, sha256_file
    from scripts.atomistic_input_audit import STANDARD_HEAVY_ATOMS
    from scripts.atomistic_modeled_stereochemistry import MODELED_STATUSES, atom_key, read_repair_mapping, strict_atoms

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts/atomistic_config.json"
BACKBONE_BONDS = (("N", "CA"), ("CA", "C"), ("C", "O"))
SIDECHAIN_BONDS = {
    "GLY": "", "ALA": "CA-CB", "SER": "CA-CB CB-OG", "CYS": "CA-CB CB-SG",
    "VAL": "CA-CB CB-CG1 CB-CG2", "THR": "CA-CB CB-OG1 CB-CG2",
    "ILE": "CA-CB CB-CG1 CB-CG2 CG1-CD1", "LEU": "CA-CB CB-CG CG-CD1 CG-CD2",
    "MET": "CA-CB CB-CG CG-SD SD-CE", "PRO": "CA-CB CB-CG CG-CD CD-N",
    "ASP": "CA-CB CB-CG CG-OD1 CG-OD2", "ASN": "CA-CB CB-CG CG-OD1 CG-ND2",
    "GLU": "CA-CB CB-CG CG-CD CD-OE1 CD-OE2",
    "GLN": "CA-CB CB-CG CG-CD CD-OE1 CD-NE2",
    "LYS": "CA-CB CB-CG CG-CD CD-CE CE-NZ",
    "ARG": "CA-CB CB-CG CG-CD CD-NE NE-CZ CZ-NH1 CZ-NH2",
    "HIS": "CA-CB CB-CG CG-ND1 ND1-CE1 CE1-NE2 NE2-CD2 CD2-CG",
    "PHE": "CA-CB CB-CG CG-CD1 CD1-CE1 CE1-CZ CZ-CE2 CE2-CD2 CD2-CG",
    "TYR": "CA-CB CB-CG CG-CD1 CD1-CE1 CE1-CZ CZ-CE2 CE2-CD2 CD2-CG CZ-OH",
    "TRP": "CA-CB CB-CG CG-CD1 CD1-NE1 NE1-CE2 CE2-CD2 CD2-CG CD2-CE3 CE3-CZ3 CZ3-CH2 CH2-CZ2 CZ2-CE2",
}
POLICY = {
    "anchor_sigma_nm": 0.005,
    "torsion_penalty_per_radian": 0.02,
    "seed": 20260907,
    "start_torsion_sd_radians": [0.0, 0.15, 0.35],
    "max_nfev_per_start": 300,
    "torsion_bound_radians": float(np.pi),
    "translation_bound_nm": 2.0,
    "anchor_rmsd_max_nm": 0.01,
    "anchor_max_error_nm": 0.02,
    "peptide_bond_range_nm": [0.11, 0.17],
    "junction_angle_range_degrees": [100.0, 140.0],
    "junction_angle_delta_max_degrees": 5.0,
    "junction_dihedral_delta_max_degrees": 5.0,
    "junction_planarity_max_degrees": 15.0,
    "l_signed_volume_min_nm3": 1e-4,
    "internal_bond_delta_max_nm": 2e-10,
    "internal_angle_delta_max_degrees": 2e-6,
    "serialized_bond_delta_max_nm": 0.000174,
    "serialized_angle_delta_max_degrees": 0.2,
}
STERIC_POLICY = {
    "mode": "steric_aware_preparation",
    "minimum_separation_nm": 0.20,
    "optimization_target_separation_nm": 0.22,
    "penalty_sigma_nm": 0.02,
    "excluded_bond_graph_distances": [1, 2],
    "include_1_4_and_more_distant_pairs": True,
    "periodic_box": False,
    "environment": "all unchanged non-loop heavy atoms, including observed and modeled atoms",
    "objective": "one nearest allowed environment contact and one nearest allowed internal contact per loop atom; hinge deficits divided by penalty_sigma_nm",
    "final_screen": "every allowed pair below minimum_separation_nm is reported before and after PDB rounding",
    "scope": "gross-clash preparation only; not van der Waals, force-field or MD qualification",
}


@dataclass(frozen=True)
class Peptide:
    keys: tuple[tuple[int, str], ...]
    residue_names: tuple[tuple[int, str], ...]
    coordinates_nm: np.ndarray
    bonds: tuple[tuple[int, int], ...]

    @property
    def index(self) -> dict:
        return {key: i for i, key in enumerate(self.keys)}


@dataclass(frozen=True)
class Torsion:
    residue: int
    kind: str
    axis: tuple[int, int]
    moving: tuple[int, ...]


@dataclass(frozen=True)
class StericContext:
    peptide_keys: tuple[tuple[int, str], ...]
    loop_indices: tuple[int, ...]
    loop_atom_keys: tuple[tuple[str, int, str, str], ...]
    loop_source_status: tuple[str, ...]
    environment_atom_keys: tuple[tuple[str, int, str, str], ...]
    environment_source_status: tuple[str, ...]
    environment_coordinates_nm: np.ndarray
    environment_excluded_indices: np.ndarray
    internal_allowed: np.ndarray
    environment_tree: cKDTree


def peptide_from_input(donor: dict) -> Peptide:
    """Validate explicit canonical identity before building standard connectivity."""
    mapping = {}
    for row in donor["canonical_mapping"]:
        residue, label, name = int(row["canonical_residue"]), int(row["label_seq_id"]), row["residue_name"]
        if residue in mapping or name not in SIDECHAIN_BONDS:
            raise ValueError("Duplicate or unsupported canonical residue identity")
        mapping[residue] = (label, name)
    residues = sorted(mapping)
    if len(residues) < 3 or residues != list(range(residues[0], residues[-1] + 1)):
        raise ValueError("Donor canonical residues must form one contiguous peptide with two anchors")
    if len({value[0] for value in mapping.values()}) != len(mapping):
        raise ValueError("Donor label_seq_id mapping must be one-to-one")
    atoms = {}
    for row in donor["atoms"]:
        residue, name = int(row["canonical_residue"]), row["atom_name"]
        key = (residue, name)
        if key in atoms or mapping.get(residue) != (int(row["label_seq_id"]), row["residue_name"]):
            raise ValueError("Duplicate atom or canonical identity mismatch")
        xyz = np.asarray(row["xyz_nm"], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Donor coordinates must be finite nm triples")
        atoms[key] = xyz
    for residue, (_, name) in mapping.items():
        if {atom for res, atom in atoms if res == residue} != STANDARD_HEAVY_ATOMS[name]:
            raise ValueError(f"Donor {residue} {name} requires exact standard heavy-atom inventory")
    keys = tuple(sorted(atoms))
    index = {key: i for i, key in enumerate(keys)}
    bonds = set()
    for residue, (_, name) in mapping.items():
        for first, second in (*BACKBONE_BONDS, *(part.split("-") for part in SIDECHAIN_BONDS[name].split())):
            bonds.add(tuple(sorted((index[residue, first], index[residue, second]))))
        if residue != residues[-1]:
            bonds.add(tuple(sorted((index[residue, "C"], index[residue + 1, "N"]))))
    for first, second in donor["additional_bonds"]:
        try:
            edge = tuple(sorted((index[tuple(first)], index[tuple(second)])))
        except KeyError as exc:
            raise ValueError("Additional bond endpoint absent from donor") from exc
        if edge[0] == edge[1] or edge in bonds:
            raise ValueError("Self or duplicate additional bond")
        bonds.add(edge)
    return Peptide(keys, tuple((r, mapping[r][1]) for r in residues), np.array([atoms[k] for k in keys]), tuple(sorted(bonds)))


def downstream_component(peptide: Peptide, axis: tuple[int, int]) -> tuple[int, ...]:
    adjacency = [set() for _ in peptide.keys]
    cut = set(axis)
    if tuple(sorted(axis)) not in peptide.bonds:
        raise ValueError("Torsion axis is not a covalent bond")
    for i, j in peptide.bonds:
        if {i, j} != cut:
            adjacency[i].add(j)
            adjacency[j].add(i)
    seen, pending = {axis[1]}, [axis[1]]
    while pending:
        for neighbor in adjacency[pending.pop()] - seen:
            seen.add(neighbor)
            pending.append(neighbor)
    if axis[0] in seen:
        raise ValueError("Torsion bond cut does not disconnect a ring or crosslink")
    return tuple(sorted(seen - cut))


def torsions_for(peptide: Peptide) -> tuple[Torsion, ...]:
    index, result = peptide.index, []
    for residue, name in peptide.residue_names:
        for kind, atom_a, atom_b in (("phi", "N", "CA"), ("psi", "CA", "C")):
            if kind == "phi" and name == "PRO":
                continue
            axis = (index[residue, atom_a], index[residue, atom_b])
            result.append(Torsion(residue, kind, axis, downstream_component(peptide, axis)))
    return tuple(result)


def rotate_about_axis(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation with explicit preserved axial and radial components."""
    points, origin, axis = (np.asarray(value, dtype=float) for value in (points, origin, axis))
    if points.ndim != 2 or points.shape[1:] != (3,) or origin.shape != (3,) or axis.shape != (3,):
        raise ValueError("Rotation requires Nx3 points, a three-vector origin and axis")
    length = float(np.linalg.norm(axis))
    if not all(np.isfinite(value).all() for value in (points, origin, axis)) or not np.isfinite(angle) or length < 1e-12:
        raise ValueError("Nonfinite or degenerate rotation axis")
    unit_axis = axis / length
    relative = points - origin
    parallel = np.outer(relative @ unit_axis, unit_axis)
    perpendicular = relative - parallel
    return origin + parallel + np.cos(angle) * perpendicular + np.sin(angle) * np.cross(unit_axis, perpendicular)


def apply_torsions(peptide: Peptide, torsions: tuple[Torsion, ...], angles: np.ndarray) -> np.ndarray:
    if np.shape(angles) != (len(torsions),) or not np.isfinite(angles).all():
        raise ValueError("One finite angle is required per torsion")
    coordinates = peptide.coordinates_nm.copy()
    for torsion, angle in zip(torsions, angles, strict=True):
        a, b = torsion.axis
        moving = list(torsion.moving)
        coordinates[moving] = rotate_about_axis(coordinates[moving], coordinates[a], coordinates[b] - coordinates[a], angle)
    return coordinates


def bond_angle(points: np.ndarray) -> float:
    a, b = points[0] - points[1], points[2] - points[1]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-15:
        raise ValueError("Degenerate bond angle")
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def dihedral(points: np.ndarray) -> float:
    axis = points[2] - points[1]
    length = np.linalg.norm(axis)
    if length < 1e-12:
        raise ValueError("Degenerate dihedral axis")
    axis = axis / length
    first, last = points[0] - points[1], points[3] - points[2]
    first, last = first - np.dot(first, axis) * axis, last - np.dot(last, axis) * axis
    if min(np.linalg.norm(first), np.linalg.norm(last)) < 1e-12:
        raise ValueError("Degenerate dihedral plane")
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, first), last), np.dot(first, last))))


def angular_delta(first: float, second: float) -> float:
    return float(abs((first - second + 180.0) % 360.0 - 180.0))


def signed_volume(points: np.ndarray) -> float:
    first, second, branch = points[[0, 2, 3]] - points[1]
    return float(np.dot(first, np.cross(second, branch)))


def geometry_inventory(peptide: Peptide, coordinates: np.ndarray) -> dict:
    index = peptide.index
    adjacency = [set() for _ in peptide.keys]
    bonds, angles, omega, planarity, alpha, beta, pro_n = {}, {}, {}, {}, {}, {}, {}
    for i, j in peptide.bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
        bonds[i, j] = float(np.linalg.norm(coordinates[i] - coordinates[j]))
    for center, neighbors in enumerate(adjacency):
        for first, last in combinations(sorted(neighbors), 2):
            angles[first, center, last] = bond_angle(coordinates[[first, center, last]])
    for residue, name in peptide.residue_names:
        if name != "GLY":
            ids = tuple(index[residue, atom] for atom in ("N", "CA", "C", "CB"))
            alpha[ids] = signed_volume(coordinates[list(ids)])
        if name in {"ILE", "THR"}:
            ids = tuple(index[residue, atom] for atom in ("CA", "CB", "CG1" if name == "ILE" else "OG1", "CG2"))
            beta[ids] = signed_volume(coordinates[list(ids)])
        if (residue + 1, "N") in index:
            ids = (index[residue, "CA"], index[residue, "C"], index[residue + 1, "N"], index[residue + 1, "CA"])
            omega[ids] = dihedral(coordinates[list(ids)])
            ids = (index[residue, "O"], *ids[1:])
            planarity[ids] = dihedral(coordinates[list(ids)])
        if name == "PRO" and (residue - 1, "C") in index:
            ids = (index[residue - 1, "C"], index[residue, "N"], index[residue, "CA"], index[residue, "CD"])
            pro_n[ids] = signed_volume(coordinates[list(ids)])
    return {"bonds": bonds, "angles": angles, "omega": omega, "planarity": planarity,
            "alpha": alpha, "beta": beta, "pro_n_volume": pro_n}


def invariance_report(peptide: Peptide, after: np.ndarray, *, residues: set[int] | None = None, serialized: bool = False) -> dict:
    before_inventory = geometry_inventory(peptide, peptide.coordinates_nm)
    after_inventory = geometry_inventory(peptide, after)
    deltas = {}
    for category, before in before_inventory.items():
        differences = []
        for ids, value in before.items():
            if residues is not None and not all(peptide.keys[i][0] in residues for i in ids):
                continue
            after_value = after_inventory[category][ids]
            differences.append(angular_delta(value, after_value) if category in {"omega", "planarity"} else abs(value - after_value))
        deltas[category] = max(differences, default=0.0)
    bond_limit = POLICY["serialized_bond_delta_max_nm"] if serialized else POLICY["internal_bond_delta_max_nm"]
    angle_limit = POLICY["serialized_angle_delta_max_degrees"] if serialized else POLICY["internal_angle_delta_max_degrees"]
    passed = deltas["bonds"] <= bond_limit and all(deltas[k] <= angle_limit for k in ("angles", "omega", "planarity"))
    # PDB rounding changes volumes; strict floating-point rotations do not.
    if not serialized:
        passed = passed and all(deltas[k] <= 2e-10 for k in ("alpha", "beta", "pro_n_volume"))
    return {"pass": bool(passed), "maximum_absolute_deltas": deltas, "serialized": serialized,
            "scope": "covalent lengths/angles, fixed omega and peptide planarity, alpha/beta/Pro-N volumes; phi/psi and nonbonded distances may change"}


def anchor_indices(peptide: Peptide) -> list[int]:
    endpoints = (peptide.residue_names[0][0], peptide.residue_names[-1][0])
    return [peptide.index[residue, atom] for residue in endpoints for atom in ("N", "CA", "C", "O")]


def build_steric_context(peptide: Peptide, target: dict, lines: list[str], mapping: dict) -> StericContext:
    """Build immutable target environment and exact bond/1-3 exclusion masks."""
    lookup, loop, _ = validate_target(peptide, target, lines, mapping)
    chain = target["chain"]
    moving = tuple(i for i, (residue, _) in enumerate(peptide.keys) if residue in loop)
    atoms = strict_atoms(lines)
    environment = [atom for atom in atoms if atom.chain != chain or atom.resseq not in loop]
    env_index = {atom_key(atom): i for i, atom in enumerate(environment)}
    adjacency = [set() for _ in peptide.keys]
    for first, second in peptide.bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)
    allowed = np.ones((len(moving), len(moving)), dtype=bool)
    moving_index = {index: i for i, index in enumerate(moving)}
    exclusions = []
    for row, index in enumerate(moving):
        # A shortest graph path of <=2 excludes bonds and 1-3 interactions,
        # including paths across both fixed target peptide junctions.
        near = {index} | adjacency[index]
        near |= {other for neighbor in adjacency[index] for other in adjacency[neighbor]}
        excluded = []
        for other in near:
            if other in moving_index:
                allowed[row, moving_index[other]] = False
            else:
                excluded.append(env_index[chain, *peptide.keys[other]])
        exclusions.append(sorted(excluded))
    width = max(map(len, exclusions), default=0)
    padded = np.full((len(moving), width), -1, dtype=int)
    for row, excluded in enumerate(exclusions):
        padded[row, :len(excluded)] = excluded
    xyz = np.array([atom.xyz / 10.0 for atom in environment])
    if not environment:
        raise ValueError("Steric preparation requires the fixed target environment")
    tree = cKDTree(xyz, copy_data=True)
    for array in (xyz, padded, allowed, tree.data):
        array.setflags(write=False)
    loop_atoms = [lookup[chain, *peptide.keys[index]] for index in moving]
    return StericContext(
        peptide.keys, moving, tuple(atom.key for atom in loop_atoms),
        tuple(mapping[atom_key(atom)]["source_status"] for atom in loop_atoms),
        tuple(atom.key for atom in environment),
        tuple(mapping.get(atom_key(atom), {}).get("source_status", "not_in_repair_inventory") for atom in environment),
        xyz, padded, allowed, tree,
    )


def steric_distances(coordinates_nm: np.ndarray, context: StericContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates_nm, dtype=float)
    if coordinates.shape != (len(context.peptide_keys), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Steric coordinates must match the finite donor nm array")
    loop = coordinates[list(context.loop_indices)]
    # At most `width` environment atoms are excluded for any moving atom;
    # querying width+1 neighbors therefore includes its nearest allowed atom.
    k = min(len(context.environment_atom_keys), context.environment_excluded_indices.shape[1] + 1)
    distances, indices = context.environment_tree.query(loop, k=k)
    distances, indices = distances.reshape(len(loop), k), indices.reshape(len(loop), k)
    excluded = (indices[:, :, None] == context.environment_excluded_indices[:, None, :]).any(axis=2)
    nearest_environment = np.where(excluded, np.inf, distances).min(axis=1)
    internal = np.linalg.norm(loop[:, None, :] - loop[None, :, :], axis=2)
    internal = np.where(context.internal_allowed, internal, np.inf)
    return nearest_environment, internal.min(axis=1), internal


def steric_residual(coordinates_nm: np.ndarray, context: StericContext) -> np.ndarray:
    environment, internal, _ = steric_distances(coordinates_nm, context)
    nearest = np.r_[environment, internal]
    return np.maximum(STERIC_POLICY["optimization_target_separation_nm"] - nearest, 0.0) / STERIC_POLICY["penalty_sigma_nm"]


def steric_report(coordinates_nm: np.ndarray, context: StericContext) -> dict:
    """Enumerate every unresolved allowed pair, not just objective neighbors."""
    nearest_environment, nearest_internal, internal = steric_distances(coordinates_nm, context)
    loop = np.asarray(coordinates_nm)[list(context.loop_indices)]
    minimum = STERIC_POLICY["minimum_separation_nm"]
    unresolved = []
    for row, neighbors in enumerate(context.environment_tree.query_ball_point(loop, minimum)):
        excluded = set(context.environment_excluded_indices[row])
        for other in sorted(set(neighbors) - excluded):
            distance = float(np.linalg.norm(loop[row] - context.environment_coordinates_nm[other]))
            if distance < minimum:
                unresolved.append({"kind": "loop_environment", "first": list(context.loop_atom_keys[row]),
                                   "second": list(context.environment_atom_keys[other]), "distance_nm": distance,
                                   "first_source_status": context.loop_source_status[row],
                                   "second_source_status": context.environment_source_status[other]})
    first, second = np.where(np.triu(internal < minimum, k=1))
    for row, other in zip(first, second, strict=True):
        unresolved.append({"kind": "loop_internal", "first": list(context.loop_atom_keys[row]),
                           "second": list(context.loop_atom_keys[other]), "distance_nm": float(internal[row, other]),
                           "first_source_status": context.loop_source_status[row],
                           "second_source_status": context.loop_source_status[other]})
    unresolved.sort(key=lambda row: (row["distance_nm"], row["kind"], row["first"], row["second"]))

    def finite_minimum(values):
        finite = values[np.isfinite(values)]
        return float(finite.min()) if finite.size else None

    excluded_environment = int(np.count_nonzero(context.environment_excluded_indices >= 0))
    internal_pairs = int(np.count_nonzero(np.triu(context.internal_allowed, k=1)))
    return {"pass": not unresolved, "minimum_separation_nm": minimum,
            "environment_atom_count": len(context.environment_atom_keys), "loop_atom_count": len(loop),
            "excluded_bonded_or_1_3_environment_pairs": excluded_environment,
            "excluded_bonded_or_1_3_internal_pairs": len(loop) * (len(loop) - 1) // 2 - internal_pairs,
            "checked_environment_pairs": len(loop) * len(context.environment_atom_keys) - excluded_environment,
            "checked_internal_pairs": internal_pairs,
            "nearest_allowed_environment_nm": finite_minimum(nearest_environment),
            "nearest_allowed_internal_nm": finite_minimum(nearest_internal),
            "unresolved_count": len(unresolved), "unresolved_pairs": unresolved,
            "scope": STERIC_POLICY["scope"]}


def close_loop(peptide: Peptide, target_anchors_nm: np.ndarray, *, max_nfev: int = 300,
               sterics: StericContext | None = None) -> dict:
    target = np.asarray(target_anchors_nm, dtype=float)
    if target.shape != (8, 3) or not np.isfinite(target).all():
        raise ValueError("Both fixed anchor N/CA/C/O coordinates are required in nm")
    if not isinstance(max_nfev, int) or not 1 <= max_nfev <= POLICY["max_nfev_per_start"]:
        raise ValueError("max_nfev must be within the fixed bounded protocol")
    if sterics is not None and sterics.peptide_keys != peptide.keys:
        raise ValueError("Steric context does not match the donor atom identities")
    torsions, anchors = torsions_for(peptide), anchor_indices(peptide)
    source = peptide.coordinates_nm[anchors]
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    initial_rotation = u @ correction @ vt

    def coordinates(parameters):
        changed = apply_torsions(peptide, torsions, parameters[6:])
        return ((changed - source_center) @ initial_rotation @ Rotation.from_rotvec(parameters[:3]).as_matrix().T
                + target_center + parameters[3:6])

    def residual(parameters):
        changed = coordinates(parameters)
        fit = (changed[anchors] - target).ravel() / POLICY["anchor_sigma_nm"]
        original = np.r_[fit, POLICY["torsion_penalty_per_radian"] * parameters[6:]]
        return original if sterics is None else np.r_[original, steric_residual(changed, sterics)]

    bounds = np.r_[np.full(3, np.pi), np.full(3, POLICY["translation_bound_nm"]), np.full(len(torsions), POLICY["torsion_bound_radians"])]
    rng, starts = np.random.default_rng(POLICY["seed"]), []
    for sd in POLICY["start_torsion_sd_radians"]:
        initial = np.zeros(len(bounds))
        if sd:
            initial[6:] = np.clip(rng.normal(0.0, sd, len(torsions)), -np.pi / 2, np.pi / 2)
        fit = least_squares(residual, initial, bounds=(-bounds, bounds), max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        starts.append({"parameters": fit.x, "cost": float(fit.cost), "nfev": int(fit.nfev),
                       "solver_success": bool(fit.success), "termination": str(fit.message), "start_sd_radians": sd})
        if sterics is not None:
            starts[-1]["steric_screen"] = steric_report(coordinates(fit.x), sterics)
    selected = min(range(len(starts)), key=lambda i: (starts[i]["cost"], i))
    best = starts[selected]
    result = coordinates(best["parameters"])
    errors = np.linalg.norm(result[anchors] - target, axis=1)
    anchor_report = {"rmsd_nm": float(np.sqrt(np.mean(errors**2))), "max_error_nm": float(errors.max()), "errors_nm": errors.tolist()}
    anchor_report["pass"] = anchor_report["rmsd_nm"] <= POLICY["anchor_rmsd_max_nm"] and anchor_report["max_error_nm"] <= POLICY["anchor_max_error_nm"]
    return {"coordinates_nm": result, "torsions": [{"residue": t.residue, "kind": t.kind, "delta_radians": float(a)} for t, a in zip(torsions, best["parameters"][6:], strict=True)],
            "selected_start": selected, "starts": [{**row, "parameters": row["parameters"].tolist()} for row in starts],
            "anchor_fit": anchor_report, "donor_invariance": invariance_report(peptide, result),
            "qualified_for_md": False}


def junction_report(peptide: Peptide, coordinates: np.ndarray) -> dict:
    index = peptide.index
    rows = []
    for left in (peptide.residue_names[0][0], peptide.residue_names[-1][0] - 1):
        ids = [index[left, "CA"], index[left, "C"], index[left + 1, "N"], index[left + 1, "CA"]]
        oxygen_ids = [index[left, "O"], *ids[1:]]
        donor, actual = peptide.coordinates_nm[ids], coordinates[ids]
        angles = [bond_angle(actual[:3]), bond_angle(actual[1:])]
        angle_deltas = [abs(a - b) for a, b in zip(angles, (bond_angle(donor[:3]), bond_angle(donor[1:])), strict=True)]
        omega, plane = dihedral(actual), dihedral(coordinates[oxygen_ids])
        omega_delta = angular_delta(omega, dihedral(donor))
        plane_delta = angular_delta(plane, dihedral(peptide.coordinates_nm[oxygen_ids]))
        planar_error = min(angular_delta(plane, 0.0), angular_delta(plane, 180.0))
        length = float(np.linalg.norm(actual[1] - actual[2]))
        low, high = POLICY["peptide_bond_range_nm"]
        angle_low, angle_high = POLICY["junction_angle_range_degrees"]
        passed = (low <= length <= high and all(angle_low <= a <= angle_high for a in angles)
                  and max(angle_deltas) <= POLICY["junction_angle_delta_max_degrees"]
                  and max(omega_delta, plane_delta) <= POLICY["junction_dihedral_delta_max_degrees"]
                  and planar_error <= POLICY["junction_planarity_max_degrees"])
        rows.append({"left_residue": left, "right_residue": left + 1, "C_N_nm": length,
                     "CA_C_N_and_C_N_CA_degrees": angles, "angle_delta_from_donor_degrees": angle_deltas,
                     "omega_degrees": omega, "omega_delta_from_donor_degrees": omega_delta,
                     "O_C_N_CA_degrees": plane, "planarity_delta_from_donor_degrees": plane_delta,
                     "absolute_planarity_error_degrees": planar_error, "pass": bool(passed)})
    return {"pass": all(row["pass"] for row in rows), "junctions": rows}


def loop_geometry_report(peptide: Peptide, coordinates: np.ndarray, loop: set[int], *, serialized: bool) -> dict:
    inventory = geometry_inventory(peptide, coordinates)
    alpha = [v for ids, v in inventory["alpha"].items() if peptide.keys[ids[0]][0] in loop]
    beta = [v for ids, v in inventory["beta"].items() if peptide.keys[ids[0]][0] in loop]
    chirality = {"alpha_min_nm3": min(alpha, default=None), "beta_min_nm3": min(beta, default=None),
                 "pass": all(v > POLICY["l_signed_volume_min_nm3"] for v in alpha + beta)}
    junctions = junction_report(peptide, coordinates)
    invariance = invariance_report(peptide, coordinates, residues=loop, serialized=serialized)
    return {"pass": bool(chirality["pass"] and junctions["pass"] and invariance["pass"]),
            "chirality": chirality, "actual_junctions": junctions, "loop_internal_invariance": invariance}


def validate_target(peptide: Peptide, target: dict, lines: list[str], mapping: dict) -> tuple[dict, set[int], int]:
    atoms = strict_atoms(lines)
    lookup = {atom_key(atom): atom for atom in atoms}
    residues = [r for r, _ in peptide.residue_names]
    if target["anchor_residues"] != [residues[0], residues[-1]] or target["loop_residues"] != residues[1:-1]:
        raise ValueError("Target anchors/loop must exactly match donor canonical mapping")
    chain, loop = target["chain"], set(target["loop_residues"])
    if not isinstance(chain, str) or len(chain) != 1:
        raise ValueError("Exactly one target PDB chain is required")
    observed = 0
    for key, row in mapping.items():
        atom = lookup.get(key)
        if atom is None or atom.resname != row["residue_name"]:
            raise ValueError("Source CSV identity missing or changed in target")
        if row["source_status"] == "observed_input":
            observed += 1
            if np.max(np.abs(atom.xyz - row["xyz"])) > 0.000501:
                raise ValueError("Observed coordinates differ from source CSV")
    for residue, name in peptide.residue_names:
        current = {a.name for a in atoms if a.chain == chain and a.resseq == residue}
        if current != STANDARD_HEAVY_ATOMS[name]:
            raise ValueError(f"Target {residue} has incomplete or unsupported heavy inventory")
        for atom_name in current:
            key = (chain, residue, atom_name)
            if lookup[key].resname != name:
                raise ValueError("Target/donor canonical residue identity mismatch")
            if residue in loop and mapping.get(key, {}).get("source_status") not in MODELED_STATUSES:
                raise ValueError(f"Refusing replacement of observed or unprovenanced loop atom: {key}")
            if residue not in loop and atom_name in {"N", "CA", "C", "O"} and mapping.get(key, {}).get("source_status") != "observed_input":
                raise ValueError("Each fixed anchor N/CA/C/O must be observed input")
    # External covalent links cannot be represented by the donor peptide graph.
    serials = {a.serial for a in atoms if a.chain == chain and a.resseq in loop}
    for line in lines:
        if line.startswith("CONECT"):
            connected = {int(line[i:i + 5]) for i in range(6, len(line.rstrip()), 5) if line[i:i + 5].strip()}
            if serials & connected:
                raise ValueError("Target loop CONECT requires explicit external connectivity handling")
        if line.startswith("SSBOND"):
            if any(line[c:c + 1] == chain and int(line[r:r + 4]) in loop for c, r in ((15, 17), (29, 31))):
                raise ValueError("Target loop disulfide is unsupported")
    return lookup, loop, observed


def graft_lines(lines: list[str], peptide: Peptide, coordinates: np.ndarray, target: dict, mapping: dict,
                *, sterics: StericContext | None = None) -> tuple[list[str], dict]:
    """Change only modeled loop coordinate columns; preserve all other bytes."""
    lookup, loop, observed = validate_target(peptide, target, lines, mapping)
    index, chain, result, changed = peptide.index, target["chain"], [], 0
    actual = coordinates.copy()
    for residue, _ in peptide.residue_names:
        if residue not in loop:
            for key, i in index.items():
                if key[0] == residue:
                    actual[i] = lookup[chain, *key].xyz / 10.0
    before_rounding = loop_geometry_report(peptide, actual, loop, serialized=False)
    before_sterics = steric_report(actual, sterics) if sterics is not None else None
    for line in lines:
        atom = parse_pdb_atom_line(line)
        if atom is not None and atom.chain == chain and atom.resseq in loop:
            xyz = coordinates[index[atom.resseq, atom.name]] * 10.0
            fields = "".join(f"{value:8.3f}" for value in xyz)
            if len(fields) != 24 or not np.isfinite(xyz).all():
                raise ValueError("Candidate coordinates exceed finite PDB field capacity")
            newline = line[:30] + fields + line[54:]
            atom = parse_pdb_atom_line(newline)
            actual[index[atom.resseq, atom.name]] = atom.xyz / 10.0
            result.append(newline)
            changed += 1
        else:
            result.append(line)
    if changed != sum(residue in loop for residue, _ in peptide.keys):
        raise ValueError("Replacement atom inventory changed unexpectedly")
    observed_unchanged = all(original == updated for original, updated in zip(lines, result, strict=True)
                             if (atom := parse_pdb_atom_line(original)) is None
                             or atom.chain != chain or atom.resseq not in loop)
    if not observed_unchanged:
        raise ValueError("Non-loop or observed PDB bytes changed")
    after_rounding = loop_geometry_report(peptide, actual, loop, serialized=True)
    report = {"observed_atoms_preserved": observed, "all_nonloop_lines_byte_identical": observed_unchanged,
              "replaced_modeled_atom_count": changed, "before_pdb_rounding": before_rounding,
              "after_pdb_rounding": after_rounding,
              "pass": bool(before_rounding["pass"] and after_rounding["pass"])}
    if sterics is not None:
        after_sterics = steric_report(actual, sterics)
        report.update({"steric_before_pdb_rounding": before_sterics, "steric_after_pdb_rounding": after_sterics,
                       "pass": bool(report["pass"] and before_sterics["pass"] and after_sterics["pass"])})
    return result, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/atomistic/loop_closure")
    parser.add_argument("--offline", action="store_true", help="Recorded; this script never uses network access")
    parser.add_argument("--steric-aware", action="store_true", help="Opt in to the fixed gross-clash preparation objective and final screen")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {"donor_json": args.input, "pdb": args.pdb, "source_csv": args.source_csv, "config": args.config, "script": Path(__file__)}
    report = {"status": "unqualified_input", "qualified_for_md": False,
              "scope": "technical loop geometry only; no force-field preparation, native-conformation validation or MD",
              "offline": args.offline, "policy": POLICY}
    if args.steric_aware:
        report["steric_policy"] = STERIC_POLICY
    try:
        report["input_sha256"] = {name: sha256_file(path) for name, path in inputs.items()}
        json.loads(args.config.read_text())  # Provenance pin; algorithm policy is fixed in source.
        request = json.loads(args.input.read_text())
        if request.get("schema_version") != 1 or request.get("coordinate_unit") != "nm":
            raise ValueError("Input requires schema_version 1 and coordinate_unit nm")
        donor, target = request["donor"], request["target"]
        if not donor.get("pdb_id") or not donor.get("label_asym_id") or len(donor.get("source_sha256", "")) != 64:
            raise ValueError("Explicit donor source identity and SHA256 are required")
        report["donor_source"] = {key: donor[key] for key in ("pdb_id", "label_asym_id", "source_sha256")}
        report["donor_source"]["verification_scope"] = "self-contained coordinate JSON is hashed; coordinates are not independently re-extracted from the claimed CIF"
        peptide = peptide_from_input(donor)
        lines = args.pdb.read_bytes().decode("ascii").splitlines(keepends=True)
        mapping = read_repair_mapping(args.source_csv)
        lookup, _, _ = validate_target(peptide, target, lines, mapping)
        anchors = np.array([lookup[target["chain"], *peptide.keys[i]].xyz / 10.0 for i in anchor_indices(peptide)])
        options = {"sterics": build_steric_context(peptide, target, lines, mapping)} if args.steric_aware else {}
        fit = close_loop(peptide, anchors, **options)
        output, graft = graft_lines(lines, peptide, fit.pop("coordinates_nm"), target, mapping, **options)
        candidate = args.output_dir / "candidate_capped_heavy.pdb"
        candidate.write_bytes("".join(output).encode("ascii"))
        passed = fit["anchor_fit"]["pass"] and fit["donor_invariance"]["pass"] and graft["pass"]
        report.update({"status": "technical_loop_geometry_pass" if passed else "unqualified_loop_geometry",
                       "fit": fit, "graft": graft, "candidate_sha256": sha256_file(candidate)})
        if any(sha256_file(path) != report["input_sha256"][name] for name, path in inputs.items()):
            report["status"] = "unqualified_input_changed"
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report["error"] = str(exc)
    (args.output_dir / "loop_closure.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "qualified_for_md": False, "output_dir": str(args.output_dir)}))
    return 0 if report["status"] == "technical_loop_geometry_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
