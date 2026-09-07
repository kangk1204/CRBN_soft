#!/usr/bin/env python3
"""Deterministically add physiological NaCl pairs to an existing Amber system.

The script preserves solute atom order and coordinates by editing the existing
Amber topology/restart with ParmEd rather than rebuilding the solute from PDB.
It retains pre-existing neutralizing ions and replaces selected TIP3P waters
with additional Na+/Cl- ions. It does not run minimization, equilibration, or MD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

AVOGADRO = 6.02214076e23
DEFAULT_MOLARITY_M = 0.150
DEFAULT_SEED = 20260907
MIN_DISTANCE_NM = 0.5
ION_RESNAMES = {"Na+", "Cl-", "NA", "CL", "Na", "Cl"}
WATER_NAMES = {"WAT", "HOH", "TIP3", "TIP3P"}
SOLUTE_EXCLUDED_RESIDUES = WATER_NAMES | ION_RESNAMES
ZN_CYS_RESIDUES = {323, 326, 391, 394}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def box_vectors_from_lengths_angles(box: Sequence[float]) -> np.ndarray:
    """Return 3x3 row-vector box matrix in nm from Amber a,b,c,alpha,beta,gamma.

    ParmEd stores lengths in Angstrom and angles in degrees. The returned matrix
    uses row vectors so Cartesian row coordinates satisfy frac @ vectors.
    """
    if len(box) != 6:
        raise ValueError("Amber box must have six values")
    a, b, c = [float(x) / 10.0 for x in box[:3]]
    alpha, beta, gamma = [math.radians(float(x)) for x in box[3:]]
    if min(a, b, c) <= 0:
        raise ValueError("box lengths must be positive")
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError("invalid triclinic box angles")
    vc = np.array([cx, cy, math.sqrt(cz2)])
    vectors = np.vstack([va, vb, vc])
    if not np.isfinite(vectors).all():
        raise ValueError("nonfinite box vectors")
    return vectors


def box_volume_nm3(box_vectors_nm: np.ndarray) -> float:
    volume = float(abs(np.linalg.det(np.asarray(box_vectors_nm, dtype=float))))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("box volume must be positive and finite")
    return volume


def salt_pair_count(volume_nm3: float, molarity_m: float = DEFAULT_MOLARITY_M) -> int:
    if not np.isfinite(volume_nm3) or volume_nm3 <= 0:
        raise ValueError("volume must be positive and finite")
    if not np.isfinite(molarity_m) or molarity_m < 0:
        raise ValueError("molarity must be finite and non-negative")
    return int(round(molarity_m * volume_nm3 * 1e-24 * AVOGADRO))


NEIGHBOR_OFFSETS = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)


def minimum_image_displacements(delta_nm: np.ndarray, box_vectors_nm: np.ndarray) -> np.ndarray:
    """Return nearest-image displacements using a 27-neighbor lattice search.

    Fractional-coordinate rounding is insufficient for reduced oblique cells such
    as truncated-octahedral boxes, where a neighboring lattice vector around the
    rounded fractional image can be closer in Cartesian distance.
    """
    delta = np.asarray(delta_nm, dtype=float)
    original_shape = delta.shape
    flat = delta.reshape(-1, 3)
    vectors = np.asarray(box_vectors_nm, dtype=float)
    inv = np.linalg.inv(vectors)
    frac = flat @ inv
    center = np.round(frac)
    best = None
    best_d2 = None
    for offset in NEIGHBOR_OFFSETS:
        cart = (frac - (center + offset)) @ vectors
        d2 = np.einsum("ij,ij->i", cart, cart)
        if best is None:
            best = cart
            best_d2 = d2
        else:
            take = d2 < best_d2
            best[take] = cart[take]
            best_d2[take] = d2[take]
    return best.reshape(original_shape)


def minimum_image_min_d2(delta_nm: np.ndarray, box_vectors_nm: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_nm, dtype=float)
    vectors = np.asarray(box_vectors_nm, dtype=float)
    inv = np.linalg.inv(vectors)
    frac = delta.reshape(-1, 3) @ inv
    center = np.round(frac)
    best_d2 = None
    for offset in NEIGHBOR_OFFSETS:
        cart = (frac - (center + offset)) @ vectors
        d2 = np.einsum("ij,ij->i", cart, cart)
        best_d2 = d2 if best_d2 is None else np.minimum(best_d2, d2)
    return best_d2.reshape(delta.shape[:-1])


def min_distance_nm_27_bruteforce(points_nm: np.ndarray, references_nm: np.ndarray, box_vectors_nm: np.ndarray, *, chunk_size: int = 128) -> np.ndarray:
    points = np.asarray(points_nm, dtype=float)
    refs = np.asarray(references_nm, dtype=float)
    if len(points) == 0:
        return np.empty((0,), dtype=float)
    if len(refs) == 0:
        return np.full((len(points),), np.inf, dtype=float)
    out = np.empty((len(points),), dtype=float)
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        delta = chunk[:, None, :] - refs[None, :, :]
        d2 = minimum_image_min_d2(delta, box_vectors_nm)
        out[start : start + len(chunk)] = np.sqrt(np.min(d2, axis=1))
    return out


def min_distance_nm_kdtree_replicated(points_nm: np.ndarray, references_nm: np.ndarray, box_vectors_nm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_nm, dtype=float)
    refs = np.asarray(references_nm, dtype=float)
    if len(points) == 0:
        return np.empty((0,), dtype=float)
    if len(refs) == 0:
        return np.full((len(points),), np.inf, dtype=float)
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - fallback is tested without SciPy requirement
        raise RuntimeError("SciPy cKDTree is unavailable") from exc
    vectors = np.asarray(box_vectors_nm, dtype=float)
    inv = np.linalg.inv(vectors)
    points_frac = points @ inv
    refs_frac = refs @ inv
    points_wrapped = (points_frac - np.floor(points_frac)) @ vectors
    refs_wrapped = (refs_frac - np.floor(refs_frac)) @ vectors
    shifts = NEIGHBOR_OFFSETS @ vectors
    replicated = (refs_wrapped[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    distances, _ = cKDTree(replicated).query(points_wrapped, k=1)
    return np.asarray(distances, dtype=float)


def min_distance_nm(points_nm: np.ndarray, references_nm: np.ndarray, box_vectors_nm: np.ndarray, *, chunk_size: int = 128) -> np.ndarray:
    points = np.asarray(points_nm, dtype=float)
    refs = np.asarray(references_nm, dtype=float)
    if len(points) == 0 or len(refs) == 0:
        return min_distance_nm_27_bruteforce(points, refs, box_vectors_nm, chunk_size=chunk_size)
    if len(points) * len(refs) >= 50_000:
        try:
            return min_distance_nm_kdtree_replicated(points, refs, box_vectors_nm)
        except RuntimeError:
            pass
    return min_distance_nm_27_bruteforce(points, refs, box_vectors_nm, chunk_size=chunk_size)


def distance_backend(points_count: int, reference_count: int) -> str:
    if points_count * reference_count < 50_000:
        return "27_neighbor_bruteforce"
    try:
        import scipy.spatial  # noqa: F401
    except Exception:
        return "27_neighbor_bruteforce_scipy_unavailable"
    return "scipy_cKDTree_27_replicated_images"


def deterministic_water_pair_selection(
    water_oxygen_indices: Sequence[int],
    water_oxygen_positions_nm: np.ndarray,
    solute_heavy_positions_nm: np.ndarray,
    ion_positions_nm: np.ndarray,
    box_vectors_nm: np.ndarray,
    pair_count: int,
    *,
    seed: int = DEFAULT_SEED,
    min_distance: float = MIN_DISTANCE_NM,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Select water oxygen atom indices to replace with Na and Cl.

    Candidates must be at least ``min_distance`` from solute heavy atoms and
    pre-existing ions. Greedy deterministic placement then requires each new ion
    site to be at least ``min_distance`` from all pre-existing and previously
    selected ion sites under triclinic minimum images.
    """
    if pair_count < 0:
        raise ValueError("pair_count must be non-negative")
    needed = 2 * pair_count
    oxygen_indices = list(map(int, water_oxygen_indices))
    water_pos = np.asarray(water_oxygen_positions_nm, dtype=float)
    if water_pos.shape != (len(oxygen_indices), 3):
        raise ValueError("water positions must be n x 3 and match oxygen index count")
    if needed == 0:
        return [], [], {"candidate_count": 0, "selected_count": 0, "minimum_selected_distance_nm": None}

    fixed_refs = np.vstack([x for x in [np.asarray(solute_heavy_positions_nm, dtype=float), np.asarray(ion_positions_nm, dtype=float)] if len(x)]) if (len(solute_heavy_positions_nm) or len(ion_positions_nm)) else np.empty((0, 3))
    d_fixed = min_distance_nm(water_pos, fixed_refs, box_vectors_nm)
    candidates = [i for i, d in enumerate(d_fixed) if d >= min_distance]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(candidates)

    selected_local: list[int] = []
    selected_positions: list[np.ndarray] = []
    ion_refs = np.asarray(ion_positions_nm, dtype=float).reshape((-1, 3)) if len(ion_positions_nm) else np.empty((0, 3))
    for local_idx in candidates:
        point = water_pos[local_idx]
        refs = ion_refs if not selected_positions else np.vstack([ion_refs, np.asarray(selected_positions)])
        if len(refs):
            dist = min_distance_nm(point.reshape(1, 3), refs, box_vectors_nm)[0]
            if dist < min_distance:
                continue
        selected_local.append(local_idx)
        selected_positions.append(point)
        if len(selected_local) == needed:
            break
    if len(selected_local) < needed:
        raise ValueError(f"insufficient waters for {pair_count} NaCl pairs: selected {len(selected_local)} of {needed}")

    selected_indices = [oxygen_indices[i] for i in selected_local]
    # Assign alternately after deterministic shuffle. Equal counts are guaranteed.
    sodium = selected_indices[:pair_count]
    chloride = selected_indices[pair_count:]
    sel_pos = water_pos[selected_local]
    selected_min = None
    if len(sel_pos) > 1:
        pair_d = []
        for i in range(len(sel_pos)):
            d = min_distance_nm(sel_pos[i + 1 :], sel_pos[i].reshape(1, 3), box_vectors_nm)
            pair_d.extend(float(x) for x in d)
        selected_min = min(pair_d) if pair_d else None
    audit = {
        "seed": int(seed),
        "min_distance_nm": min_distance,
        "requested_pair_count": int(pair_count),
        "candidate_count_after_solute_and_preexisting_ion_filter": len(candidates),
        "selected_count": len(selected_indices),
        "sodium_count": len(sodium),
        "chloride_count": len(chloride),
        "minimum_fixed_distance_nm_before_selection": float(d_fixed[candidates].min()) if candidates else None,
        "minimum_selected_pair_distance_nm": selected_min,
        "primary_distance_backend": distance_backend(len(water_pos), len(fixed_refs)),
    }
    return sodium, chloride, audit


def residue_name(residue: Any) -> str:
    return str(getattr(residue, "name", ""))


def atom_element_symbol(atom: Any) -> str | None:
    atomic_number = int(getattr(atom, "atomic_number", 0) or 0)
    if atomic_number == 1:
        return "H"
    if atomic_number == 6:
        return "C"
    if atomic_number == 7:
        return "N"
    if atomic_number == 8:
        return "O"
    if atomic_number == 11:
        return "Na"
    if atomic_number == 16:
        return "S"
    if atomic_number == 17:
        return "Cl"
    if atomic_number == 30:
        return "Zn"
    return None


def is_hydrogen(atom: Any) -> bool:
    return int(getattr(atom, "atomic_number", 0) or 0) == 1 or str(getattr(atom, "name", "")).strip().startswith("H")


def collect_real_system_inputs(parm: Any) -> dict[str, Any]:
    coords_nm = np.asarray([[a.xx, a.xy, a.xz] for a in parm.atoms], dtype=float) / 10.0
    water_oxygen_indices: list[int] = []
    water_oxygen_positions: list[np.ndarray] = []
    solute_heavy_positions: list[np.ndarray] = []
    ion_positions: list[np.ndarray] = []
    preexisting_ions: dict[str, int] = {"Na+": 0, "Cl-": 0}
    water_residue_indices: list[int] = []
    hies = []
    for residue in parm.residues:
        name = residue_name(residue)
        if name in WATER_NAMES:
            oxy = [a for a in residue.atoms if a.name == "O"]
            if len(oxy) != 1 or len(residue.atoms) != 3:
                raise ValueError(f"unsupported water residue {residue.idx}:{name}")
            atom = oxy[0]
            water_oxygen_indices.append(atom.idx)
            water_oxygen_positions.append(coords_nm[atom.idx])
            water_residue_indices.append(residue.idx)
        elif name in ION_RESNAMES:
            if len(residue.atoms) != 1:
                raise ValueError(f"unsupported ion residue {residue.idx}:{name}")
            ion_positions.append(coords_nm[residue.atoms[0].idx])
            if name == "Na+":
                preexisting_ions["Na+"] += 1
            elif name == "Cl-":
                preexisting_ions["Cl-"] += 1
        else:
            if name == "HIE":
                names = {a.name for a in residue.atoms}
                hies.append({"residue_index": residue.idx, "has_HE2": "HE2" in names, "has_HD1": "HD1" in names})
            for atom in residue.atoms:
                if not is_hydrogen(atom):
                    solute_heavy_positions.append(coords_nm[atom.idx])
    return {
        "coords_nm": coords_nm,
        "water_oxygen_indices": water_oxygen_indices,
        "water_oxygen_positions_nm": np.asarray(water_oxygen_positions, dtype=float),
        "solute_heavy_positions_nm": np.asarray(solute_heavy_positions, dtype=float),
        "ion_positions_nm": np.asarray(ion_positions, dtype=float),
        "water_residue_indices": water_residue_indices,
        "preexisting_ions": preexisting_ions,
        "hie_assignments": hies,
    }


def find_ion_template(parm: Any, residue_name_value: str) -> Any | None:
    for residue in parm.residues:
        if residue.name == residue_name_value and len(residue.atoms) == 1:
            return residue.atoms[0]
    return None


def copy_atom_nonbonded_identity(target: Any, template: Any, *, name: str, residue_name_value: str, nb_idx: int | None = None) -> None:
    target.name = name
    target.type = template.type
    target.atom_type = getattr(template, "atom_type", None)
    target.atomic_number = int(template.atomic_number)
    target.mass = float(template.mass)
    target.charge = float(template.charge)
    if nb_idx is not None:
        target.nb_idx = int(nb_idx)
    else:
        target.nb_idx = int(template.nb_idx)
    target.epsilon = float(template.epsilon)
    target.rmin = float(template.rmin)
    target.epsilon_14 = float(getattr(template, "epsilon_14", template.epsilon))
    target.rmin_14 = float(getattr(template, "rmin_14", template.rmin))
    target.residue.name = residue_name_value


def append_lj_type_from_template(parm: Any, target_indices: Sequence[int], template: Any, type_name: str) -> int:
    if not target_indices:
        raise ValueError(f"no {type_name} target atoms")
    from parmed.tools.addljtype import AddLJType

    selection = [0] * len(parm.atoms)
    for idx in target_indices:
        selection[int(idx)] = 1
    AddLJType(parm, selection, float(template.rmin), float(template.epsilon), None, None)
    new_idx = int(parm.atoms[int(target_indices[0])].nb_idx)
    parm.LJ_types[type_name] = new_idx
    return new_idx


def load_ion_template(template_prmtop: Path | None, residue_name_value: str) -> Any:
    if template_prmtop is None:
        flag = "--na-template-prmtop" if residue_name_value == "Na+" else "--cl-template-prmtop"
        raise ValueError(f"input topology lacks {residue_name_value}; {flag} is required")
    import parmed as pmd
    source = pmd.load_file(str(template_prmtop))
    template = find_ion_template(source, residue_name_value)
    if template is None:
        raise ValueError(f"template topology lacks a {residue_name_value} atom")
    return template


def ensure_ion_type(parm: Any, residue_name_value: str, selected_indices: Sequence[int], template_prmtop: Path | None) -> tuple[Any, int, str]:
    existing = find_ion_template(parm, residue_name_value)
    if existing is not None:
        return existing, int(existing.nb_idx), "input_existing_ion_type"
    template = load_ion_template(template_prmtop, residue_name_value)
    nb_idx = append_lj_type_from_template(parm, selected_indices, template, residue_name_value)
    return template, nb_idx, "template_lj_type_added_with_parmed_AddLJType"


def solute_atom_indices(parm: Any) -> set[int]:
    return {a.idx for r in parm.residues if r.name not in SOLUTE_EXCLUDED_RESIDUES for a in r.atoms}


def nonbonded_signature(parm: Any, indices: set[int], idx_to_old: dict[int, int] | None = None) -> list[dict[str, Any]]:
    idx_to_old = idx_to_old or {a.idx: a.idx for a in parm.atoms}
    rows = []
    for atom in parm.atoms:
        if atom.idx not in indices:
            continue
        rows.append({
            "old_index": int(idx_to_old[atom.idx]),
            "residue_name": atom.residue.name,
            "atom_name": atom.name,
            "type": str(atom.type),
            "atomic_number": int(atom.atomic_number),
            "mass": float(atom.mass),
            "charge": float(atom.charge),
            "nb_idx": int(atom.nb_idx),
            "rmin": float(atom.rmin),
            "epsilon": float(atom.epsilon),
            "rmin_14": float(getattr(atom, "rmin_14", atom.rmin)),
            "epsilon_14": float(getattr(atom, "epsilon_14", atom.epsilon)),
        })
    return sorted(rows, key=lambda x: x["old_index"])


def _term_type_signature(term_type: Any, names: Sequence[str]) -> tuple[Any, ...]:
    if term_type is None:
        return tuple(None for _ in names)
    return tuple(float(getattr(term_type, name)) for name in names)


def bonded_signature(parm: Any, indices: set[int], idx_to_old: dict[int, int] | None = None) -> dict[str, list[Any]]:
    idx_to_old = idx_to_old or {a.idx: a.idx for a in parm.atoms}
    def old(atom: Any) -> int:
        return int(idx_to_old[atom.idx])
    bonds = []
    for b in parm.bonds:
        if b.atom1.idx in indices and b.atom2.idx in indices:
            bonds.append((tuple(sorted((old(b.atom1), old(b.atom2)))), _term_type_signature(b.type, ("k", "req"))))
    angles = []
    for a in parm.angles:
        if {a.atom1.idx, a.atom2.idx, a.atom3.idx} <= indices:
            angles.append(((old(a.atom1), old(a.atom2), old(a.atom3)), _term_type_signature(a.type, ("k", "theteq"))))
    dihedrals = []
    for d in parm.dihedrals:
        if {d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx} <= indices:
            dihedrals.append(((old(d.atom1), old(d.atom2), old(d.atom3), old(d.atom4)), _term_type_signature(d.type, ("phi_k", "per", "phase")), bool(getattr(d, "improper", False)), bool(getattr(d, "ignore_end", False))))
    exclusions = []
    for atom in parm.atoms:
        if atom.idx not in indices:
            continue
        for partner in getattr(atom, "exclusion_partners", []):
            if partner.idx in indices:
                pair = tuple(sorted((old(atom), old(partner))))
                exclusions.append(pair)
    return {
        "bonds": sorted(bonds),
        "angles": sorted(angles),
        "dihedrals": sorted(dihedrals),
        "exclusions": sorted(set(exclusions)),
    }


def histidine_assignments(parm: Any) -> list[dict[str, Any]]:
    rows = []
    for residue in parm.residues:
        if residue.name in {"HIE", "HID", "HIP", "HIS"}:
            names = sorted(a.name for a in residue.atoms)
            rows.append({
                "residue_index": int(residue.idx),
                "residue_name": residue.name,
                "atom_names": names,
                "has_HE2": "HE2" in names,
                "has_HD1": "HD1" in names,
            })
    return rows


def validate_no_bonded_terms_for_atoms(parm: Any, atom_indices: set[int]) -> None:
    for label, terms in (("bond", parm.bonds), ("angle", parm.angles), ("dihedral", parm.dihedrals)):
        for term in terms:
            atoms = [getattr(term, name) for name in ("atom1", "atom2", "atom3", "atom4") if hasattr(term, name)]
            if any(atom.idx in atom_indices for atom in atoms):
                raise ValueError(f"new ion atom appears in leftover {label} term")


def mutate_waters_to_ions(
    parm: Any,
    *,
    sodium_oxygen_indices: Sequence[int],
    chloride_oxygen_indices: Sequence[int],
    na_template_prmtop: Path | None,
    cl_template_prmtop: Path | None,
) -> dict[str, Any]:
    original_coords_nm = np.asarray([[a.xx, a.xy, a.xz] for a in parm.atoms], dtype=float) / 10.0
    initial_atom_count = len(parm.atoms)
    solute_old = solute_atom_indices(parm)
    solute_nonbonded_before = nonbonded_signature(parm, solute_old)
    solute_bonded_before = bonded_signature(parm, solute_old)
    histidine_before = histidine_assignments(parm)

    sodium_o = [int(i) for i in sodium_oxygen_indices]
    chloride_o = [int(i) for i in chloride_oxygen_indices]
    na_template, na_nb_idx, na_source = ensure_ion_type(parm, "Na+", sodium_o, na_template_prmtop)
    cl_template, cl_nb_idx, cl_source = ensure_ion_type(parm, "Cl-", chloride_o, cl_template_prmtop)

    replacement = {int(i): "Na+" for i in sodium_o}
    replacement.update({int(i): "Cl-" for i in chloride_o})
    if len(replacement) != len(sodium_o) + len(chloride_o):
        raise ValueError("same water oxygen selected for multiple ions")
    strip_atom_indices: list[int] = []
    replacement_rows = []
    for oxygen_idx, ion_name in sorted(replacement.items()):
        atom = parm.atoms[oxygen_idx]
        residue = atom.residue
        if residue.name not in WATER_NAMES or atom.name != "O":
            raise ValueError(f"selected atom {oxygen_idx} is not a water oxygen")
        if ion_name == "Na+":
            copy_atom_nonbonded_identity(atom, na_template, name="Na+", residue_name_value="Na+", nb_idx=na_nb_idx)
        else:
            copy_atom_nonbonded_identity(atom, cl_template, name="Cl-", residue_name_value="Cl-", nb_idx=cl_nb_idx)
        for other in list(residue.atoms):
            if other.idx != oxygen_idx:
                strip_atom_indices.append(other.idx)
        replacement_rows.append({
            "old_water_oxygen_atom_index": int(oxygen_idx),
            "old_water_residue_index": int(residue.idx),
            "new_ion": ion_name,
            "position_nm": original_coords_nm[oxygen_idx].tolist(),
        })
    if len(set(strip_atom_indices)) != 2 * len(replacement):
        raise ValueError("unexpected water hydrogen strip count")
    retained_old_indices = [i for i in range(initial_atom_count) if i not in set(strip_atom_indices)]
    mask = "@" + ",".join(str(i + 1) for i in sorted(strip_atom_indices))
    parm.strip(mask)
    parm.remake_parm()
    parm.rediscover_molecules()

    if len(parm.atoms) != len(retained_old_indices):
        raise ValueError("retained atom count mismatch after stripping water hydrogens")
    old_to_new = {old: new for new, old in enumerate(retained_old_indices)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    new_coords_nm = np.asarray([[a.xx, a.xy, a.xz] for a in parm.atoms], dtype=float) / 10.0
    retained_error = np.abs(new_coords_nm - original_coords_nm[retained_old_indices]).max() if retained_old_indices else 0.0
    if retained_error > 2e-7:
        raise ValueError(f"retained atom coordinates changed: max {retained_error} nm")
    new_solute = {old_to_new[i] for i in solute_old}
    if nonbonded_signature(parm, new_solute, new_to_old) != solute_nonbonded_before:
        raise ValueError("solute atom charge/mass/type/LJ signature changed")
    if bonded_signature(parm, new_solute, new_to_old) != solute_bonded_before:
        raise ValueError("solute bonded/exclusion signature changed")
    if histidine_assignments(parm) != histidine_before:
        raise ValueError("histidine protonation atom assignment changed")
    new_ion_indices = {old_to_new[i] for i in replacement}
    validate_no_bonded_terms_for_atoms(parm, new_ion_indices)
    return {
        "replacement_rows": replacement_rows,
        "stripped_water_hydrogen_count": len(strip_atom_indices),
        "retained_atom_count": len(retained_old_indices),
        "retained_atom_coordinate_max_abs_error_nm": float(retained_error),
        "old_to_new_atom_index": [[int(old), int(new)] for old, new in old_to_new.items()],
        "solute_old_atom_indices": sorted(int(i) for i in solute_old),
        "new_ion_atom_indices": sorted(int(i) for i in new_ion_indices),
        "ion_type_sources": {"Na+": na_source, "Cl-": cl_source},
        "solute_nonbonded_signature_unchanged": True,
        "solute_bonded_exclusion_signature_unchanged": True,
        "histidine_assignments_before_after": {"before": histidine_before, "after": histidine_assignments(parm)},
    }



def _quantity_float(value: Any) -> float:
    try:
        return float(value._value)
    except AttributeError:
        return float(value)


def _force_by_name(system: Any, name: str) -> Any | None:
    for force in system.getForces():
        if force.__class__.__name__ == name:
            return force
    return None


def _rounded_tuple(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(round(_quantity_float(v), 14) for v in values)


def _openmm_custom_lj_tables(custom_force: Any) -> tuple[int, int, tuple[float, ...], tuple[float, ...]]:
    tables: dict[str, Any] = {}
    for i in range(custom_force.getNumTabulatedFunctions()):
        tables[custom_force.getTabulatedFunctionName(i)] = custom_force.getTabulatedFunction(i)
    if "acoef" not in tables or "bcoef" not in tables:
        raise ValueError("CustomNonbondedForce lacks Amber LJ coefficient tables")
    ax, ay, avals = tables["acoef"].getFunctionParameters()
    bx, by, bvals = tables["bcoef"].getFunctionParameters()
    if (ax, ay) != (bx, by):
        raise ValueError("CustomNonbondedForce LJ tables have inconsistent dimensions")
    return int(ax), int(ay), tuple(float(v) for v in avals), tuple(float(v) for v in bvals)


OPENMM_LJ_COEFF_DIGITS = 9


def _lj_ab_from_sigma_epsilon(sigma: Any, epsilon: Any) -> tuple[float, float]:
    sigma_f = _quantity_float(sigma)
    epsilon_f = _quantity_float(epsilon)
    bcoef = 4.0 * epsilon_f * sigma_f**6
    acoef = math.sqrt(max(0.0, 4.0 * epsilon_f * sigma_f**12))
    return (round(acoef, OPENMM_LJ_COEFF_DIGITS), round(bcoef, OPENMM_LJ_COEFF_DIGITS))


def _openmm_effective_lj_pair_signature(system: Any, solute_indices: set[int], idx_to_old: dict[int, int]) -> list[Any]:
    nb = _force_by_name(system, "NonbondedForce")
    if nb is None:
        return []
    custom = _force_by_name(system, "CustomNonbondedForce")
    if custom is not None:
        if custom.getNumPerParticleParameters() != 1 or custom.getPerParticleParameterName(0) != "type":
            raise ValueError("unexpected CustomNonbondedForce particle parameters")
        xsize, ysize, avals, bvals = _openmm_custom_lj_tables(custom)
        by_type: dict[int, int] = {}
        for i in solute_indices:
            params = custom.getParticleParameters(i)
            type_idx = int(round(float(params[0])))
            by_type[type_idx] = min(by_type.get(type_idx, idx_to_old[i]), idx_to_old[i])
        rows = []
        for t1, old1 in sorted(by_type.items(), key=lambda kv: kv[1]):
            for t2, old2 in sorted(by_type.items(), key=lambda kv: kv[1]):
                if old2 < old1:
                    continue
                if t1 >= xsize or t2 >= ysize:
                    raise ValueError("CustomNonbondedForce type index exceeds table dimensions")
                offset = t1 + xsize * t2
                rows.append(((old1, old2), (round(avals[offset], OPENMM_LJ_COEFF_DIGITS), round(bvals[offset], OPENMM_LJ_COEFF_DIGITS))))
        return rows

    by_lj: dict[tuple[float, float], int] = {}
    lj_by_old: dict[int, tuple[Any, Any]] = {}
    for i in solute_indices:
        _charge, sigma, epsilon = nb.getParticleParameters(i)
        key = (round(_quantity_float(sigma), 14), round(_quantity_float(epsilon), 14))
        old = idx_to_old[i]
        by_lj[key] = min(by_lj.get(key, old), old)
        lj_by_old[old] = (sigma, epsilon)
    reps = sorted((old, key) for key, old in by_lj.items())
    rows = []
    for n, (old1, key1) in enumerate(reps):
        sigma1, eps1 = key1
        for old2, key2 in reps[n:]:
            sigma2, eps2 = key2
            sigma = 0.5 * (sigma1 + sigma2)
            epsilon = math.sqrt(max(0.0, eps1 * eps2))
            rows.append(((old1, old2), _lj_ab_from_sigma_epsilon(sigma, epsilon)))
    return rows


def openmm_solute_term_signature(system: Any, solute_indices: set[int], idx_to_old: dict[int, int] | None = None) -> dict[str, Any]:
    idx_to_old = idx_to_old or {i: i for i in solute_indices}
    masses = sorted((idx_to_old[i], round(_quantity_float(system.getParticleMass(i)), 14)) for i in solute_indices)
    nb = _force_by_name(system, "NonbondedForce")
    nonbonded_particles = []
    nonbonded_exceptions = []
    if nb is not None:
        for i in solute_indices:
            charge, _sigma, _epsilon = nb.getParticleParameters(i)
            nonbonded_particles.append((idx_to_old[i], (round(_quantity_float(charge), 14),)))
        for k in range(nb.getNumExceptions()):
            a, b, chargeprod, sigma, epsilon = nb.getExceptionParameters(k)
            if int(a) in solute_indices and int(b) in solute_indices:
                nonbonded_exceptions.append((tuple(sorted((idx_to_old[int(a)], idx_to_old[int(b)]))), _rounded_tuple((chargeprod, sigma, epsilon))))
    bond_force = _force_by_name(system, "HarmonicBondForce")
    bonds = []
    if bond_force is not None:
        for k in range(bond_force.getNumBonds()):
            a, b, length, kval = bond_force.getBondParameters(k)
            if int(a) in solute_indices and int(b) in solute_indices:
                bonds.append((tuple(sorted((idx_to_old[int(a)], idx_to_old[int(b)]))), _rounded_tuple((length, kval))))
    angle_force = _force_by_name(system, "HarmonicAngleForce")
    angles = []
    if angle_force is not None:
        for k in range(angle_force.getNumAngles()):
            a, b, c, theta, kval = angle_force.getAngleParameters(k)
            if {int(a), int(b), int(c)} <= solute_indices:
                angles.append(((idx_to_old[int(a)], idx_to_old[int(b)], idx_to_old[int(c)]), _rounded_tuple((theta, kval))))
    torsion_force = _force_by_name(system, "PeriodicTorsionForce")
    torsions = []
    if torsion_force is not None:
        for k in range(torsion_force.getNumTorsions()):
            a, b, c, d, periodicity, phase, kval = torsion_force.getTorsionParameters(k)
            if {int(a), int(b), int(c), int(d)} <= solute_indices:
                torsions.append(((idx_to_old[int(a)], idx_to_old[int(b)], idx_to_old[int(c)], idx_to_old[int(d)]), int(periodicity), _rounded_tuple((phase, kval))))
    return {
        "masses": masses,
        "nonbonded_particles": sorted(nonbonded_particles),
        "nonbonded_effective_lj_pairs": _openmm_effective_lj_pair_signature(system, solute_indices, idx_to_old),
        "nonbonded_exceptions": sorted(nonbonded_exceptions),
        "bonds": sorted(bonds),
        "angles": sorted(angles),
        "torsions": sorted(torsions),
    }


def signature_sha256(signature: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_openmm_solute_terms_unchanged(input_prmtop: Path, output_prmtop: Path, solute_old_indices: Sequence[int], old_to_new_rows: Sequence[Sequence[int]]) -> dict[str, Any]:
    from openmm import app
    old_to_new = {int(old): int(new) for old, new in old_to_new_rows}
    solute_old = {int(i) for i in solute_old_indices}
    missing = sorted(i for i in solute_old if i not in old_to_new)
    if missing:
        raise ValueError(f"solute atoms removed during salt edit: {missing[:5]}")
    solute_new = {old_to_new[i] for i in solute_old}
    new_to_old = {new: old for old, new in old_to_new.items()}
    before = app.AmberPrmtopFile(str(input_prmtop)).createSystem(nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)
    after = app.AmberPrmtopFile(str(output_prmtop)).createSystem(nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)
    before_sig = openmm_solute_term_signature(before, solute_old)
    after_sig = openmm_solute_term_signature(after, solute_new, new_to_old)
    before_hash = signature_sha256(before_sig)
    after_hash = signature_sha256(after_sig)
    if before_hash != after_hash or before_sig != after_sig:
        raise ValueError("OpenMM serialized solute System terms changed after salt edit")
    return {
        "status": "pass",
        "method": "OpenMM System term extraction without Context",
        "signature_sha256_before": before_hash,
        "signature_sha256_after": after_hash,
        "solute_atom_count": len(solute_old),
        "term_counts": {key: len(value) for key, value in before_sig.items()},
    }


def apply_salt_with_parmed(
    *,
    prmtop: Path,
    inpcrd: Path,
    output_prmtop: Path,
    output_inpcrd: Path,
    output_pdb: Path | None,
    mapping_in: Path,
    mapping_out: Path,
    prep: Path,
    na_template_prmtop: Path | None,
    cl_template_prmtop: Path | None,
    pair_count_override: int | None,
    seed: int,
    molarity_m: float,
    min_distance: float,
) -> dict[str, Any]:
    import parmed as pmd
    from openmm import app, unit
    import sys

    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from verify_zaff_amber_topology import parameter_contract

    parm = pmd.load_file(str(prmtop), str(inpcrd))
    initial_residue_count = len(parm.residues)
    if parm.box is None:
        raise ValueError("input restart/topology does not define a periodic box")
    box_vectors_nm = box_vectors_from_lengths_angles(parm.box)
    volume_nm3 = box_volume_nm3(box_vectors_nm)
    pair_count = salt_pair_count(volume_nm3, molarity_m) if pair_count_override is None else int(pair_count_override)
    if pair_count <= 0:
        raise ValueError("salt pair count must be positive for this preparation")

    system_inputs = collect_real_system_inputs(parm)
    if not system_inputs["hie_assignments"]:
        raise ValueError("no HIE residues found; refusing to alter protonation-unknown system")
    bad_hie = [x for x in system_inputs["hie_assignments"] if not x["has_HE2"] or x["has_HD1"]]
    if bad_hie:
        raise ValueError(f"HIE HE2 assignment check failed: {bad_hie[:5]}")

    sodium_o, chloride_o, selection_audit = deterministic_water_pair_selection(
        system_inputs["water_oxygen_indices"],
        system_inputs["water_oxygen_positions_nm"],
        system_inputs["solute_heavy_positions_nm"],
        system_inputs["ion_positions_nm"],
        box_vectors_nm,
        pair_count,
        seed=seed,
        min_distance=min_distance,
    )

    total_charge_initial_e = float(sum(a.charge for a in parm.atoms))
    mutation_audit = mutate_waters_to_ions(
        parm,
        sodium_oxygen_indices=sodium_o,
        chloride_oxygen_indices=chloride_o,
        na_template_prmtop=na_template_prmtop,
        cl_template_prmtop=cl_template_prmtop,
    )
    total_charge_mutated_e = float(sum(a.charge for a in parm.atoms))
    if abs(total_charge_mutated_e - total_charge_initial_e) > 1e-5:
        raise ValueError(
            f"total charge changed after neutral salt insertion: "
            f"initial={total_charge_initial_e} mutated={total_charge_mutated_e}"
        )

    output_prmtop.parent.mkdir(parents=True, exist_ok=True)
    output_inpcrd.parent.mkdir(parents=True, exist_ok=True)
    parm.save(str(output_prmtop), overwrite=True)
    parm.save(str(output_inpcrd), overwrite=True)
    if output_pdb is not None:
        parm.save(str(output_pdb), overwrite=True)

    salted = pmd.load_file(str(output_prmtop), str(output_inpcrd))
    salted_coords_nm = np.asarray([[a.xx, a.xy, a.xz] for a in salted.atoms], dtype=float) / 10.0

    counts_before = system_inputs["preexisting_ions"] | {"WAT": len(system_inputs["water_oxygen_indices"])}
    counts_after = {"Na+": 0, "Cl-": 0, "WAT": 0, "HIE": 0, "CY1": 0, "ZN1": 0}
    for residue in salted.residues:
        if residue.name in counts_after:
            counts_after[residue.name] += 1
    if counts_after["Na+"] != counts_before.get("Na+", 0) + pair_count:
        raise ValueError("unexpected final Na+ count")
    if counts_after["Cl-"] != counts_before.get("Cl-", 0) + pair_count:
        raise ValueError("unexpected final Cl- count")
    if counts_after["WAT"] != counts_before["WAT"] - 2 * pair_count:
        raise ValueError("unexpected final water count")
    total_charge_after_e = float(sum(a.charge for a in salted.atoms))
    if abs(total_charge_after_e - total_charge_initial_e) > 1e-5:
        raise ValueError(
            f"total charge changed after output reload: "
            f"initial={total_charge_initial_e} mutated={total_charge_mutated_e} after={total_charge_after_e}"
        )

    mapping = load_json(mapping_in)
    retained_old = {old for old, _new in mutation_audit["old_to_new_atom_index"]}
    for key in ("core_indices", "protein_ca_indices", "zn_sg_indices"):
        if any(int(i) not in retained_old for i in mapping.get(key, [])):
            raise ValueError(f"mapping key {key} contains an atom removed during salt edit")
    if int(mapping.get("zn_atom_index", -1)) not in retained_old:
        raise ValueError("zn atom index was removed during salt edit")
    old_atom_count = mapping.get("atom_count")
    mapping["atom_count"] = len(salted.atoms)
    mapping["residue_count"] = len(salted.residues)
    mapping["topology_files"] = {"prmtop": str(output_prmtop), "inpcrd": str(output_inpcrd)}
    mapping["salt_preparation"] = {
        "status": "physiological_salt_pairs_added",
        "old_atom_count": old_atom_count,
        "new_atom_count": len(salted.atoms),
        "old_residue_count": initial_residue_count,
        "new_residue_count": len(salted.residues),
        "old_counts": counts_before,
        "new_counts": counts_after,
        "nominal_added_nacl_molarity_M": molarity_m,
        "actual_initial_box_volume_nm3": volume_nm3,
        "nominal_added_salt_pairs": pair_count,
        "total_ionic_strength_note": "Nominal 150 mM added NaCl is distinct from total ionic strength because original neutralizing ions are retained.",
        "protonation_note": "Original HIE/HE2 assignments are preserved by topology editing; no pKa validation is claimed.",
    }
    write_json(mapping_out, mapping)

    amber_top = app.AmberPrmtopFile(str(output_prmtop))
    amber_crd = app.AmberInpcrdFile(str(output_inpcrd))
    positions = np.asarray(amber_crd.positions.value_in_unit(unit.nanometer), dtype=float)
    if positions.shape != (len(list(amber_top.topology.atoms())), 3):
        raise ValueError("OpenMM topology/coordinate count mismatch after salt edit")
    system = amber_top.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)
    zaff = parameter_contract(system, amber_top.topology, prep_path=prep, atom_types=amber_top._prmtop.getAtomTypes())
    if zaff.get("status") != "pass":
        raise ValueError(f"ZAFF topology contract failed after salt edit: {zaff.get('failures')}")
    openmm_term_check = verify_openmm_solute_terms_unchanged(
        prmtop,
        output_prmtop,
        mutation_audit["solute_old_atom_indices"],
        mutation_audit["old_to_new_atom_index"],
    )

    final_mapping = load_json(mapping_out)
    input_mapping = load_json(mapping_in)
    q = np.asarray(final_mapping.get("q"), dtype=float)
    ref = np.asarray(final_mapping.get("reference_nm"), dtype=float)
    if q.shape != (807,) or ref.shape != (269, 3):
        raise ValueError("mapping q/reference shape changed")
    for frozen_key in ("core_indices", "ddb1_atom_indices", "protein_ca_indices", "zn_atom_index", "zn_sg_indices", "reference_nm", "q"):
        if final_mapping.get(frozen_key) != input_mapping.get(frozen_key):
            raise ValueError(f"frozen mapping field changed: {frozen_key}")
    core_indices = [int(i) for i in final_mapping["core_indices"]]
    if not np.allclose(positions[core_indices], salted_coords_nm[core_indices], atol=2e-7, rtol=0):
        raise ValueError("core positions are inconsistent after OpenMM reload")

    from run_atomistic_technical_pilot import derive_chemical_geometry, chemical_geometry_screen
    chemical_geometry = derive_chemical_geometry(amber_top.topology)
    chemical_screen = chemical_geometry_screen(positions, chemical_geometry)
    if not chemical_screen.get("pass"):
        raise ValueError("salted topology failed complete chemical geometry screen")

    out = {
        "schema_version": "1.0",
        "status": "pass",
        "production_ready": False,
        "scope": "deterministic physiological-salt production-input preparation; no minimization/equilibration/MD and no pKa validation",
        "offline": True,
        "seed": int(seed),
        "distance_rule": {"minimum_distance_nm": min_distance, "metric": "triclinic minimum image"},
        "salt_policy": {
            "nominal_added_nacl_molarity_M": molarity_m,
            "actual_initial_box_volume_nm3": volume_nm3,
            "added_nacl_pairs": pair_count,
            "retained_original_neutralizing_ions": counts_before,
            "final_counts": counts_after,
            "total_charge_initial_e": total_charge_initial_e,
            "total_charge_mutated_e": total_charge_mutated_e,
            "total_charge_after_e": total_charge_after_e,
            "nominal_vs_total_ionic_strength": "The added pair count represents nominal 150 mM NaCl pairs from the initial box volume. Total ionic strength also includes retained original neutralizing ions.",
        },
        "selection_audit": selection_audit,
        "replacement_audit": {
            "selected_water_oxygen_atom_indices_Na": [int(i) for i in sodium_o],
            "selected_water_oxygen_atom_indices_Cl": [int(i) for i in chloride_o],
            **mutation_audit,
        },
        "checks": {
            "all_retained_atom_coordinates_preserved": True,
            "all_solute_atom_charge_mass_type_lj_unchanged": True,
            "all_solute_bond_angle_torsion_exclusion_terms_unchanged": True,
            "new_ions_have_no_leftover_water_bonds_angles_torsions": True,
            "total_charge_preserved_by_neutral_salt_pairs": True,
            "total_charge_initial_e": total_charge_initial_e,
            "total_charge_mutated_e": total_charge_mutated_e,
            "total_charge_after_e": total_charge_after_e,
            "openmm_solute_terms_unchanged": True,
            "openmm_solute_term_check": openmm_term_check,
            "hie_residue_count": counts_after["HIE"],
            "hie_he2_assignments_preserved": True,
            "cy1_count": counts_after["CY1"],
            "zn1_count": counts_after["ZN1"],
            "zaff_parameter_contract_status": zaff.get("status"),
            "mapping_core_count": len(final_mapping["core_indices"]),
            "mapping_q_length": len(final_mapping["q"]),
            "mapping_reference_shape": [len(final_mapping["reference_nm"]), len(final_mapping["reference_nm"][0])],
            "frozen_mapping_fields_unchanged": True,
            "complete_chemical_geometry_status": "pass",
            "complete_chemical_geometry_counts": {
                "particle_count": chemical_geometry.particle_count,
                "alpha_centers": len(chemical_geometry.alpha_centers),
                "alpha_hydrogen_centers": len(chemical_geometry.alpha_hydrogen_centers),
                "beta_centers": len(chemical_geometry.beta_centers),
                "peptide_pairs": len(chemical_geometry.peptide_pairs),
            },
        },
        "complete_chemical_geometry_screen": {k: v for k, v in chemical_screen.items() if k != "failures"},
        "sources": {
            "prmtop": {"path": str(prmtop), "sha256": sha256_file(prmtop)},
            "inpcrd": {"path": str(inpcrd), "sha256": sha256_file(inpcrd)},
            "mapping": {"path": str(mapping_in), "sha256": sha256_file(mapping_in)},
            "prep": {"path": str(prep), "sha256": sha256_file(prep)},
            "na_template_prmtop": {"path": str(na_template_prmtop) if na_template_prmtop else None, "sha256": sha256_file(na_template_prmtop) if na_template_prmtop else None},
            "cl_template_prmtop": {"path": str(cl_template_prmtop) if cl_template_prmtop else None, "sha256": sha256_file(cl_template_prmtop) if cl_template_prmtop else None},
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "outputs": {
            "prmtop": {"path": str(output_prmtop), "sha256": sha256_file(output_prmtop)},
            "inpcrd": {"path": str(output_inpcrd), "sha256": sha256_file(output_inpcrd)},
            "mapping": {"path": str(mapping_out), "sha256": sha256_file(mapping_out)},
        },
    }
    if output_pdb is not None:
        out["outputs"]["pdb"] = {"path": str(output_pdb), "sha256": sha256_file(output_pdb)}
    return out


REQUIRED_INPUT_PATH_KEYS = ("prmtop", "inpcrd", "mapping", "prep")
OPTIONAL_TEMPLATE_PATH_KEYS = ("na_template_prmtop", "cl_template_prmtop")


def default_paths_from_config(config: dict[str, Any]) -> dict[str, Path | None]:
    """Return salt paths supplied by config only.

    This public CLI must not infer private 125-analysis paths. Required inputs
    must be supplied either in ``atomistic_salt`` or on the command line. Ion
    template topologies are optional and are loaded only when the input topology
    lacks the corresponding ion type.
    """
    salt = config.get("atomistic_salt", {})
    if salt is None:
        salt = {}
    if not isinstance(salt, dict):
        raise ValueError("config atomistic_salt must be an object when present")
    paths: dict[str, Path | None] = {}
    for key in REQUIRED_INPUT_PATH_KEYS + OPTIONAL_TEMPLATE_PATH_KEYS:
        value = salt.get(key)
        paths[key] = Path(value) if value not in (None, "") else None
    return paths


def require_input_paths(paths: dict[str, Path | None]) -> dict[str, Path | None]:
    missing = [key for key in REQUIRED_INPUT_PATH_KEYS if paths.get(key) is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"missing required salt input path(s): {joined}; supply via CLI or config atomistic_salt")
    for key in REQUIRED_INPUT_PATH_KEYS:
        path = paths[key]
        if path is None or not path.exists():
            raise FileNotFoundError(f"{key}: {path}")
    return paths


def reject_nonempty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--prmtop", type=Path)
    p.add_argument("--inpcrd", type=Path)
    p.add_argument("--mapping", type=Path)
    p.add_argument("--prep", type=Path)
    p.add_argument("--na-template-prmtop", type=Path, help="Topology containing a vetted Na+ ion type/parameters")
    p.add_argument("--cl-template-prmtop", type=Path, help="Topology containing a vetted Cl- ion type/parameters")
    p.add_argument("--pairs", type=int, help="Override added NaCl pair count; default derives from actual input box volume")
    p.add_argument("--molarity", type=float, default=DEFAULT_MOLARITY_M)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--min-distance-nm", type=float, default=MIN_DISTANCE_NM)
    p.add_argument("--write-pdb", action="store_true")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.offline:
        raise ValueError("--offline is required; this script performs only local file operations")
    config = load_json(args.config)
    paths = default_paths_from_config(config)
    for key in REQUIRED_INPUT_PATH_KEYS + OPTIONAL_TEMPLATE_PATH_KEYS:
        override = getattr(args, key)
        if override is not None:
            paths[key] = override
    require_input_paths(paths)
    out = args.output_dir
    reject_nonempty_output_dir(out)
    output_prmtop = out / "solvated_150mM.prmtop"
    output_inpcrd = out / "solvated_150mM.inpcrd"
    output_mapping = out / "atomistic_mapping_150mM.json"
    output_pdb = out / "solvated_150mM.pdb" if args.write_pdb else None
    report = apply_salt_with_parmed(
        prmtop=paths["prmtop"],
        inpcrd=paths["inpcrd"],
        output_prmtop=output_prmtop,
        output_inpcrd=output_inpcrd,
        output_pdb=output_pdb,
        mapping_in=paths["mapping"],
        mapping_out=output_mapping,
        prep=paths["prep"],
        na_template_prmtop=paths["na_template_prmtop"],
        cl_template_prmtop=paths["cl_template_prmtop"],
        pair_count_override=args.pairs,
        seed=args.seed,
        molarity_m=args.molarity,
        min_distance=args.min_distance_nm,
    )
    report_path = out / "salt_preparation_report.json"
    write_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "output_dir": str(out),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "added_nacl_pairs": report["salt_policy"]["added_nacl_pairs"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
