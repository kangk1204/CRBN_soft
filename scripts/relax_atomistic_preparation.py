#!/usr/bin/env python3
"""Constrained minimization for atomistic CRBN preparation only.

This is not MD and does not qualify a production model.  It performs one
OpenMM LocalEnergyMinimizer pass after the ZAFF parameter contract passes,
using temporary Cartesian restraints on explicit observed-atom indices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np


DEFAULT_RESTRAINT_K = 1000.0
DEFAULT_TOLERANCE = 10.0
DEFAULT_MAX_ITERATIONS = 2000
HEAVY_CLASH_CUTOFF_NM = 0.08
ZN_SG_BOUNDS_NM = (0.19, 0.30)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parameter_contract():
    try:
        from verify_zaff_amber_topology import parameter_contract
    except ImportError:  # pragma: no cover
        from scripts.verify_zaff_amber_topology import parameter_contract
    return parameter_contract


def choose_nonbonded_method(requested: str, has_periodic_box: bool) -> str:
    if requested == "auto":
        return "PME" if has_periodic_box else "NoCutoff"
    if requested == "PME" and not has_periodic_box:
        raise ValueError("PME requires periodic box vectors; use --nonbonded-method auto or NoCutoff for dry nonperiodic preparation")
    if requested not in {"PME", "NoCutoff"}:
        raise ValueError(f"Unsupported nonbonded method {requested!r}")
    return requested


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def input_snapshot(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()}


def changed_inputs(snapshot: dict[str, dict[str, str]]) -> list[str]:
    return [key for key, item in snapshot.items()
            if not Path(item["path"]).is_file() or sha256_file(Path(item["path"])) != item["sha256"]]


def load_restrain_indices(path: Path, atom_count: int) -> list[int]:
    payload = read_json(path)
    if isinstance(payload, list):
        raw = payload
        source_label = "top_level_list"
    elif isinstance(payload, dict):
        for key in ("restrain_indices", "observed_heavy_indices", "indices"):
            if key in payload:
                raw = payload[key]
                source_label = key
                break
        else:
            raise ValueError("Restraint JSON must contain restrain_indices, observed_heavy_indices, or indices")
    else:
        raise ValueError("Restraint JSON must be a list or object")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source_label} must be a non-empty list")
    indices = []
    for value in raw:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{source_label} contains non-integer value {value!r}")
        if value < 0 or value >= atom_count:
            raise ValueError(f"{source_label} index {value} is outside atom range 0..{atom_count - 1}")
        indices.append(value)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{source_label} contains duplicate atom indices")
    return indices


def validate_mapping_payload(mapping: dict[str, Any], atom_count: int) -> dict[str, Any]:
    required = ("core_indices", "reference_nm", "q", "ddb1_atom_indices")
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"Mapping JSON missing required fields: {missing}")
    core = _valid_indices(mapping["core_indices"], atom_count, "core_indices")
    ddb1 = _valid_indices(mapping["ddb1_atom_indices"], atom_count, "ddb1_atom_indices", allow_empty=True)
    reference = np.asarray(mapping["reference_nm"], dtype=float)
    q = np.asarray(mapping["q"], dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) != len(core):
        raise ValueError("reference_nm must be an Nx3 array matching core_indices")
    expected_q_length = 3 * len(core)
    if q.ndim != 1 or q.shape[0] != expected_q_length or not np.isfinite(q).all():
        raise ValueError(f"q must be a finite one-dimensional array with length 3*len(core_indices)={expected_q_length}")
    if not np.isfinite(reference).all():
        raise ValueError("reference_nm must be finite")
    return {
        "core_count": len(core),
        "ddb1_atom_count": len(ddb1),
        "reference_shape": list(reference.shape),
        "q_length": int(q.shape[0]),
    }


def _valid_indices(values: Iterable[Any], atom_count: int, label: str, allow_empty: bool = False) -> list[int]:
    if not isinstance(values, list) or (not values and not allow_empty):
        raise ValueError(f"{label} must be a {'possibly empty ' if allow_empty else ''}list")
    out = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{label} contains non-integer value {value!r}")
        if value < 0 or value >= atom_count:
            raise ValueError(f"{label} index {value} outside atom range 0..{atom_count - 1}")
        out.append(value)
    if len(set(out)) != len(out):
        raise ValueError(f"{label} contains duplicates")
    return out


def atom_is_heavy(atom: Any) -> bool:
    element = getattr(atom, "element", None)
    symbol = getattr(element, "symbol", None)
    if symbol is not None:
        return symbol != "H"
    return not atom.name.upper().startswith("H")


def bonded_pairs(topology: Any) -> set[tuple[int, int]]:
    pairs = set()
    for a, b in topology.bonds():
        pairs.add(tuple(sorted((int(a.index), int(b.index)))))
    return pairs


def periodic_image_deltas(delta: np.ndarray, box_vectors_nm: np.ndarray, inv_box: np.ndarray | None = None) -> np.ndarray:
    box = np.asarray(box_vectors_nm, dtype=float)
    if box.shape != (3, 3):
        raise ValueError("box_vectors_nm must be a 3x3 matrix")
    inv = np.linalg.inv(box) if inv_box is None else inv_box
    frac = np.asarray(delta, dtype=float) @ inv
    nearest = np.round(frac).astype(int)
    offsets = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=int)
    translations = (nearest + offsets) @ box
    return np.asarray(delta, dtype=float) - translations


def minimum_image_delta(delta: np.ndarray, box_vectors_nm: np.ndarray | None, inv_box: np.ndarray | None = None) -> np.ndarray:
    if box_vectors_nm is None:
        return np.asarray(delta, dtype=float)
    candidates = periodic_image_deltas(delta, box_vectors_nm, inv_box)
    return candidates[np.argmin(np.linalg.norm(candidates, axis=1))]


def _clash_row(first: Any, second: Any, distance: float) -> dict[str, Any]:
    return {
        "atoms": [first.index, second.index],
        "atom_names": [first.name, second.name],
        "residues": [
            f"{first.residue.name}{first.residue.index + 1}",
            f"{second.residue.name}{second.residue.index + 1}",
        ],
        "distance_nm": distance,
        "distance_A": 10.0 * distance,
    }


def heavy_clash_pairs(
    topology: Any,
    positions_nm: np.ndarray,
    *,
    cutoff_nm: float = HEAVY_CLASH_CUTOFF_NM,
    box_vectors_nm: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    from scipy.spatial import cKDTree

    atoms = list(topology.atoms())
    xyz = np.asarray(positions_nm, dtype=float)
    if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
        raise ValueError("positions_nm must be finite and match topology atom count")
    bonded = bonded_pairs(topology)
    heavy = [atom for atom in atoms if atom_is_heavy(atom)]
    if len(heavy) < 2:
        return []

    heavy_indices = np.array([atom.index for atom in heavy], dtype=int)
    heavy_xyz = xyz[heavy_indices]
    rows = []
    if box_vectors_nm is None:
        tree = cKDTree(heavy_xyz)
        candidate_pairs = tree.query_pairs(float(cutoff_nm), output_type="set")
        for i, j in candidate_pairs:
            first = heavy[i]
            second = heavy[j]
            pair = tuple(sorted((first.index, second.index)))
            if pair in bonded:
                continue
            distance = float(np.linalg.norm(heavy_xyz[j] - heavy_xyz[i]))
            if distance < cutoff_nm:
                rows.append(_clash_row(first, second, distance))
    else:
        box = np.asarray(box_vectors_nm, dtype=float)
        if box.shape != (3, 3) or not np.isfinite(box).all():
            raise ValueError("box_vectors_nm must be a finite 3x3 matrix")
        inv_box = np.linalg.inv(box)
        min_singular = float(np.min(np.linalg.svd(box, compute_uv=False)))
        if min_singular <= 0.0 or not np.isfinite(min_singular):
            raise ValueError("box_vectors_nm must be non-singular")
        frac = (heavy_xyz @ inv_box) % 1.0
        fractional_radius = float(cutoff_nm) / min_singular
        tree = cKDTree(frac, boxsize=1.0)
        candidate_pairs = tree.query_pairs(fractional_radius, output_type="set")
        for i, j in candidate_pairs:
            first = heavy[i]
            second = heavy[j]
            pair = tuple(sorted((first.index, second.index)))
            if pair in bonded:
                continue
            delta = xyz[second.index] - xyz[first.index]
            distance = float(np.min(np.linalg.norm(periodic_image_deltas(delta, box, inv_box), axis=1)))
            if distance < cutoff_nm:
                rows.append(_clash_row(first, second, distance))
    rows.sort(key=lambda row: row["distance_nm"])
    return rows


def locate_zn_sg_indices(topology: Any) -> tuple[int, list[int]]:
    atoms = list(topology.atoms())
    zn = [atom.index for atom in atoms if atom.name == "ZN" and atom.residue.name == "ZN1"]
    sg = [atom.index for atom in atoms if atom.name == "SG" and atom.residue.name == "CY1"]
    if len(zn) != 1 or len(sg) != 4:
        raise ValueError(f"Expected one ZN1 ZN and four CY1 SG atoms, found {len(zn)} Zn and {len(sg)} SG")
    return zn[0], sorted(sg)


def zn_sg_distances(topology: Any, positions_nm: np.ndarray, box_vectors_nm: np.ndarray | None = None) -> list[float]:
    del box_vectors_nm  # Zn-SG is a bonded geometry gate; do not minimum-image a split bond.
    zn_index, sg_indices = locate_zn_sg_indices(topology)
    xyz = np.asarray(positions_nm, dtype=float)
    return [float(np.linalg.norm(xyz[sg] - xyz[zn_index])) for sg in sg_indices]


def status_from_checks(
    *,
    finite_energies: bool,
    final_clashes: list[dict[str, Any]],
    final_zn_sg_nm: list[float],
    positions_changed: bool,
) -> str:
    metal_ok = len(final_zn_sg_nm) == 4 and all(ZN_SG_BOUNDS_NM[0] <= distance <= ZN_SG_BOUNDS_NM[1] for distance in final_zn_sg_nm)
    if finite_energies and not final_clashes and metal_ok and positions_changed:
        return "minimization_complete"
    return "failed"


def add_position_restraints(system: Any, indices: list[int], reference_nm: np.ndarray, k_kj_mol_nm2: float) -> int:
    from openmm import CustomCompoundBondForce

    force = CustomCompoundBondForce(1, "0.5*k*((x1-x0)^2+(y1-y0)^2+(z1-z0)^2)")
    force.addGlobalParameter("k", float(k_kj_mol_nm2))
    for name in ("x0", "y0", "z0"):
        force.addPerBondParameter(name)
    for index in indices:
        force.addBond([int(index)], [float(x) for x in reference_nm[index]])
    system.addForce(force)
    return system.getNumForces() - 1


def max_force_norm_kj_mol_nm(state: Any) -> float:
    from openmm import unit

    forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer), dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 3 or not np.isfinite(forces).all():
        raise ValueError("OpenMM returned non-finite forces")
    return float(np.max(np.linalg.norm(forces, axis=1)))


def state_energy_kj_mol(state: Any) -> float:
    from openmm import unit

    energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    if not np.isfinite(energy):
        raise ValueError("OpenMM returned non-finite potential energy")
    return energy


def run(
    *,
    prmtop: Path,
    inpcrd: Path,
    mapping_path: Path,
    restrain_indices_path: Path,
    prep: Path,
    output_dir: Path,
    platform_name: str = "OpenCL",
    device_index: str | None = None,
    restraint_k: float = DEFAULT_RESTRAINT_K,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    nonbonded_method: str = "auto",
) -> dict[str, Any]:
    from openmm import Context, LocalEnergyMinimizer, Platform, VerletIntegrator, app, unit

    parameter_contract = load_parameter_contract()
    sources = input_snapshot({
        "prmtop": prmtop, "inpcrd": inpcrd, "mapping": mapping_path,
        "restrain_indices": restrain_indices_path, "prep": prep,
        "script": Path(__file__).resolve(),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates = app.AmberInpcrdFile(str(inpcrd))
    has_periodic_box = coordinates.boxVectors is not None
    if has_periodic_box:
        amber = app.AmberPrmtopFile(str(prmtop), periodicBoxVectors=coordinates.boxVectors)
    else:
        amber = app.AmberPrmtopFile(str(prmtop))
    selected_nonbonded_method = choose_nonbonded_method(nonbonded_method, has_periodic_box)
    positions_nm = np.asarray(coordinates.positions.value_in_unit(unit.nanometer), dtype=float)
    atoms = list(amber.topology.atoms())
    if positions_nm.shape != (len(atoms), 3):
        raise ValueError("Amber coordinate count does not match topology atom count")
    mapping = read_json(mapping_path)
    mapping_summary = validate_mapping_payload(mapping, len(atoms))
    restrain_indices = load_restrain_indices(restrain_indices_path, len(atoms))
    system = amber.createSystem(
        nonbondedMethod=getattr(app, selected_nonbonded_method),
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=False,
    )
    zaff_contract = parameter_contract(
        system,
        amber.topology,
        prep_path=prep,
        atom_types=amber._prmtop.getAtomTypes(),
    )
    if zaff_contract["status"] != "pass":
        raise ValueError(f"ZAFF parameter contract failed before Context creation: {zaff_contract['failures']}")
    restraint_force_index = add_position_restraints(system, restrain_indices, positions_nm, restraint_k)
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    platform = Platform.getPlatformByName(platform_name)
    properties = {}
    if "Precision" in platform.getPropertyNames():
        properties["Precision"] = "double"
    if device_index is not None:
        if "DeviceIndex" not in platform.getPropertyNames():
            raise ValueError(f"Platform {platform_name} does not support DeviceIndex")
        properties["DeviceIndex"] = str(device_index)
    context = Context(system, integrator, platform, properties)
    try:
        context.setPositions(coordinates.positions)
        context.computeVirtualSites()
        initial_state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        initial_energy = state_energy_kj_mol(initial_state)
        initial_forces = max_force_norm_kj_mol_nm(initial_state)
        initial_positions = np.asarray(
            initial_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float
        )
        initial_box = None
        if coordinates.boxVectors is not None:
            initial_box = np.asarray(coordinates.boxVectors.value_in_unit(unit.nanometer), dtype=float)
        pre_clashes = heavy_clash_pairs(amber.topology, initial_positions, box_vectors_nm=initial_box)
        pre_zn_sg = zn_sg_distances(amber.topology, initial_positions, initial_box)

        LocalEnergyMinimizer.minimize(context, tolerance * unit.kilojoules_per_mole / unit.nanometer, max_iterations)
        context.computeVirtualSites()
        final_state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        final_energy = state_energy_kj_mol(final_state)
        final_forces = max_force_norm_kj_mol_nm(final_state)
        final_positions = np.asarray(final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float)
    finally:
        del context
        del integrator

    final_clashes = heavy_clash_pairs(amber.topology, final_positions, box_vectors_nm=initial_box)
    final_zn_sg = zn_sg_distances(amber.topology, final_positions, initial_box)
    all_atom_rmsd = float(np.sqrt(np.mean(np.sum((final_positions - positions_nm) ** 2, axis=1))))
    restrained_rmsd = float(
        np.sqrt(np.mean(np.sum((final_positions[restrain_indices] - positions_nm[restrain_indices]) ** 2, axis=1)))
    )
    positions_changed = bool(np.max(np.linalg.norm(final_positions - positions_nm, axis=1)) > 1e-8)
    finite_energies = bool(np.isfinite(initial_energy) and np.isfinite(final_energy) and np.isfinite(final_forces))
    status = status_from_checks(
        finite_energies=finite_energies,
        final_clashes=final_clashes,
        final_zn_sg_nm=final_zn_sg,
        positions_changed=positions_changed,
    )
    modified_inputs = changed_inputs(sources)
    if modified_inputs:
        status = "failed_input_changed_during_minimization"

    npy_path = output_dir / "minimized_positions.npy"
    np.save(npy_path, final_positions)
    if coordinates.boxVectors is not None:
        np.save(output_dir / "box_vectors_nm.npy", initial_box)
    outputs = {
        "minpositions": {"path": str(npy_path), "sha256": sha256_file(npy_path)}
    }
    if coordinates.boxVectors is not None:
        box_path = output_dir / "box_vectors_nm.npy"
        outputs["box"] = {"path": str(box_path), "sha256": sha256_file(box_path)}
    mapping_copy = output_dir / "mapping_unchanged.json"
    shutil.copyfile(mapping_path, mapping_copy)
    report = {
        "status": status,
        "scope": "pre-MD preparation minimization only; no dynamics, velocities, barostat, or production-readiness claim",
        "md_ready_production": False,
        "geometry_gate": "preparation_only_not_md_qualification",
        "restraint": {
            "kind": "temporary CustomCompoundBondForce(1) Cartesian harmonic restraint",
            "energy": "0.5*k*((x1-x0)^2+(y1-y0)^2+(z1-z0)^2)",
            "k_kj_mol_nm2": float(restraint_k),
            "restrained_atom_count": len(restrain_indices),
            "force_index": restraint_force_index,
            "reference": "initial input coordinates for explicit observed-heavy restraint indices",
        },
        "platform": {"requested": platform_name, "properties": properties},
        "system_creation": {
            "nonbonded_method_requested": nonbonded_method,
            "nonbonded_method_selected": selected_nonbonded_method,
            "has_periodic_box": has_periodic_box,
            "constraints": "HBonds",
            "rigidWater": True,
            "removeCMMotion": False,
        },
        "minimizer": {"algorithm": "OpenMM LocalEnergyMinimizer", "tolerance": tolerance, "max_iterations": max_iterations},
        "parameter_contract": {"status": zaff_contract["status"], "failures": zaff_contract["failures"]},
        "mapping_preservation": {
            "summary": mapping_summary,
            "reference_nm_and_q_unchanged": True,
            "copied_mapping_json": str(mapping_copy),
            "source_sha256": sha256_file(mapping_path),
            "copied_sha256": sha256_file(mapping_copy),
        },
        "sources": sources,
        "input_integrity": {"captured_before_Amber_parsing": True, "changed_during_run": modified_inputs},
        "outputs": outputs,
        "energies_kj_mol": {"initial": initial_energy, "final": final_energy, "finite": finite_energies},
        "forces_kj_mol_nm": {"initial_max": initial_forces, "final_max": final_forces},
        "coordinate_change": {
            "positions_changed": positions_changed,
            "all_atom_rmsd_nm": all_atom_rmsd,
            "restrained_atom_rmsd_nm": restrained_rmsd,
            "output_npy": str(npy_path),
        },
        "heavy_clashes_lt_0p8A_excluding_bonded_neighbors": {
            "cutoff_nm": HEAVY_CLASH_CUTOFF_NM,
            "pre_count": len(pre_clashes),
            "post_count": len(final_clashes),
            "pre_pairs": pre_clashes[:50],
            "post_pairs": final_clashes[:50],
        },
        "zn_sg_distances_nm": {
            "bounds": list(ZN_SG_BOUNDS_NM),
            "geometry": "raw unwrapped bonded distance; PBC minimum image is not applied to Zn-SG gate",
            "pre": pre_zn_sg,
            "post": final_zn_sg,
        },
        "acceptance": {
            "minimization_complete_requires": [
                "finite initial and final energies/forces",
                "zero final heavy-atom clashes below 0.8 A after excluding bonded neighbors",
                "four final Zn-SG distances in 1.9-3.0 A",
                "changed coordinates saved to minimized_positions.npy",
            ]
        },
    }
    write_json(output_dir / "relax_atomistic_preparation.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", type=Path, required=True)
    parser.add_argument("--inpcrd", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--restrain-indices", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument("--device-index")
    parser.add_argument("--restraint-k", type=float, default=DEFAULT_RESTRAINT_K)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--nonbonded-method",
        choices=("auto", "PME", "NoCutoff"),
        default="auto",
        help="auto uses PME for periodic/solvated inputs and NoCutoff for dry nonperiodic inputs",
    )
    parser.add_argument("--offline", action="store_true", help="Accepted for provenance; this runner performs local file/compute operations only")
    args = parser.parse_args(argv)
    if args.restraint_k <= 0:
        parser.error("--restraint-k must be positive")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    report = run(
        prmtop=args.prmtop,
        inpcrd=args.inpcrd,
        mapping_path=args.mapping,
        restrain_indices_path=args.restrain_indices,
        prep=args.prep,
        output_dir=args.output_dir,
        platform_name=args.platform,
        device_index=args.device_index,
        restraint_k=args.restraint_k,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        nonbonded_method=args.nonbonded_method,
    )
    print(json.dumps({"status": report["status"], "post_clashes": report["heavy_clashes_lt_0p8A_excluding_bonded_neighbors"]["post_count"]}, indent=2))
    return 0 if report["status"] == "minimization_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
