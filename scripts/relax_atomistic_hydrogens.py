#!/usr/bin/env python3
"""Fixed-heavy hydrogen/solvent minimization for atomistic CRBN preparation.

This stage is not MD and does not qualify production dynamics.  It keeps all
solute heavy atoms fixed by setting their OpenMM particle masses to zero, then
minimizes only hydrogens, solvent, and ions from a supplied Amber topology and
coordinate set whose heavy geometry has already been prepared separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MAX_ITERATIONS = 400
DEFAULT_TOLERANCE = 10.0
FIXED_HEAVY_TOLERANCE_NM = 1e-10
CHIRALITY_MIN_VOLUME_NM3 = 1e-4
WATER_RESIDUES = {"WAT", "HOH", "TIP3", "TIP3P", "SOL"}
ION_RESIDUES = {
    "NA", "NA+", "SOD", "K", "K+", "CL", "CL-", "CLA", "MG", "MG2", "MG2+",
    "CA", "CA2", "CA2+", "ZN", "ZN2", "ZN2+",
}
# Protein/cap/ZAFF solute in this workflow.  Solvent/ions are excluded before this check.
ALLOWED_SOLUTE_ELEMENTS = {"H", "C", "N", "O", "S", "Zn"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def input_snapshot(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()}


def changed_inputs(snapshot: dict[str, dict[str, str]]) -> list[str]:
    changed = []
    for key, item in snapshot.items():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            changed.append(key)
    return changed


def load_parameter_contract():
    try:
        from verify_zaff_amber_topology import parameter_contract
    except ImportError:  # pragma: no cover
        from scripts.verify_zaff_amber_topology import parameter_contract
    return parameter_contract


def load_restart_helpers():
    try:
        from atomistic_preparation_handoff import export_restart_with_parmed, verify_restart_roundtrip
    except ImportError:  # pragma: no cover
        from scripts.atomistic_preparation_handoff import export_restart_with_parmed, verify_restart_roundtrip
    return export_restart_with_parmed, verify_restart_roundtrip


def residue_label(atom: Any) -> str:
    residue = atom.residue
    chain = getattr(getattr(residue, "chain", None), "id", "")
    rid = getattr(residue, "id", str(getattr(residue, "index", "")))
    return f"{chain}:{residue.name}{rid}" if chain else f"{residue.name}{rid}"


def element_symbol(atom: Any) -> str | None:
    element = getattr(atom, "element", None)
    symbol = getattr(element, "symbol", None)
    if symbol:
        return str(symbol)
    return None


def is_water_or_ion_residue(residue_name: str) -> bool:
    name = residue_name.upper()
    return name in WATER_RESIDUES or name in ION_RESIDUES


def classify_mobile_and_fixed_atoms(topology: Any) -> dict[str, Any]:
    fixed_solute_heavy: list[int] = []
    mobile: list[int] = []
    hydrogens: list[int] = []
    solvent_or_ions: list[int] = []
    unknown_solute: list[dict[str, Any]] = []
    atoms = list(topology.atoms())
    for atom in atoms:
        symbol = element_symbol(atom)
        residue_name = atom.residue.name.upper()
        if is_water_or_ion_residue(residue_name):
            solvent_or_ions.append(atom.index)
            mobile.append(atom.index)
            continue
        if symbol is None or symbol not in ALLOWED_SOLUTE_ELEMENTS:
            unknown_solute.append({"index": atom.index, "name": atom.name, "residue": residue_label(atom), "element": symbol})
            continue
        if symbol == "H":
            hydrogens.append(atom.index)
            mobile.append(atom.index)
        else:
            fixed_solute_heavy.append(atom.index)
    if unknown_solute:
        raise ValueError(f"unknown solute elements in non-water/non-ion residues: {unknown_solute[:10]}")
    return {
        "fixed_solute_heavy": fixed_solute_heavy,
        "mobile": mobile,
        "hydrogens": hydrogens,
        "solvent_or_ions": solvent_or_ions,
        "atom_count": len(atoms),
    }


def set_zero_masses(system: Any, indices: list[int]) -> list[float]:
    original = []
    for index in indices:
        mass = system.getParticleMass(index)
        original.append(float(mass.value_in_unit(mass.unit)))
        system.setParticleMass(index, 0.0 * mass.unit)
    return original


def signed_volume(n: np.ndarray, ca: np.ndarray, c: np.ndarray, atom: np.ndarray) -> float:
    return float(np.dot(np.cross(n - ca, c - ca), atom - ca))


def alpha_ha_stereochemistry(topology: Any, initial_nm: np.ndarray, final_nm: np.ndarray) -> dict[str, Any]:
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    initial = np.asarray(initial_nm, dtype=float)
    final = np.asarray(final_nm, dtype=float)
    for residue in topology.residues():
        if residue.name.upper() in WATER_RESIDUES or residue.name.upper() in ION_RESIDUES:
            continue
        atoms = {atom.name: atom.index for atom in residue.atoms()}
        if residue.name == "GLY" or not {"N", "CA", "C", "CB"} <= set(atoms):
            continue
        ha_name = "HA" if "HA" in atoms else None
        if ha_name is None:
            failures.append(f"missing HA for alpha stereochemistry at {residue_label(next(residue.atoms()))}")
            continue
        n, ca, c, cb, ha = (atoms[name] for name in ("N", "CA", "C", "CB", ha_name))
        heavy_initial = signed_volume(initial[n], initial[ca], initial[c], initial[cb])
        ha_initial = signed_volume(initial[n], initial[ca], initial[c], initial[ha])
        ha_post = signed_volume(final[n], final[ca], final[c], final[ha])
        row = {
            "residue": residue_label(next(residue.atoms())),
            "heavy_initial_volume_nm3": heavy_initial,
            "ha_initial_volume_nm3": ha_initial,
            "ha_post_volume_nm3": ha_post,
            "heavy_initial_valid_L": heavy_initial > CHIRALITY_MIN_VOLUME_NM3,
            "ha_post_opposite_cb": ha_post < -CHIRALITY_MIN_VOLUME_NM3,
        }
        rows.append(row)
        if not row["heavy_initial_valid_L"]:
            failures.append(f"initial heavy N-CA-C-CB chirality not positive/nonplanar at {row['residue']}: {heavy_initial:.6g} nm^3")
        if not row["ha_post_opposite_cb"]:
            failures.append(f"post HA is not opposite CB at {row['residue']}: {ha_post:.6g} nm^3")
    if not rows:
        failures.append("No complete protein alpha/HA centers were evaluated")
    return {
        "status": "pass" if not failures else "fail",
        "checked_residue_count": len(rows),
        "minimum_abs_volume_nm3": CHIRALITY_MIN_VOLUME_NM3,
        "expected": "initial N-CA-C-CB volume > +1e-4 nm^3 and post N-CA-C-HA volume < -1e-4 nm^3",
        "failures": failures,
        "examples": rows[:30],
    }


def finite_geometry(positions_nm: np.ndarray, box_nm: np.ndarray | None) -> dict[str, Any]:
    positions = np.asarray(positions_nm, dtype=float)
    ok = positions.ndim == 2 and positions.shape[1] == 3 and np.isfinite(positions).all()
    box_ok = box_nm is not None and np.asarray(box_nm, dtype=float).shape == (3, 3) and np.isfinite(box_nm).all()
    return {"status": "pass" if ok and box_ok else "fail", "positions_finite": bool(ok), "periodic_box_finite": bool(box_ok)}


def max_delta_nm(initial: np.ndarray, final: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(np.max(np.linalg.norm(np.asarray(final)[indices] - np.asarray(initial)[indices], axis=1)))


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
    prep: Path,
    output_dir: Path,
    platform_name: str = "OpenCL",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    from openmm import Context, LocalEnergyMinimizer, Platform, VerletIntegrator, app, unit

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Hydrogen minimization output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"prmtop": prmtop, "inpcrd": inpcrd, "prep": prep, "script": Path(__file__).resolve()}
    start_snapshot = input_snapshot(paths)

    coordinates = app.AmberInpcrdFile(str(inpcrd))
    if coordinates.boxVectors is None:
        raise ValueError("periodic box vectors are required for PME hydrogen/solvent minimization")
    box_nm = np.asarray(coordinates.boxVectors.value_in_unit(unit.nanometer), dtype=float)
    amber = app.AmberPrmtopFile(str(prmtop), periodicBoxVectors=coordinates.boxVectors)
    positions_nm = np.asarray(coordinates.positions.value_in_unit(unit.nanometer), dtype=float)
    geom = finite_geometry(positions_nm, box_nm)
    if geom["status"] != "pass":
        raise ValueError(f"input geometry is not finite/periodic: {geom}")

    atoms = list(amber.topology.atoms())
    if positions_nm.shape != (len(atoms), 3):
        raise ValueError("Amber coordinate count does not match topology atom count")
    masks = classify_mobile_and_fixed_atoms(amber.topology)

    system = amber.createSystem(
        nonbondedMethod=app.PME,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    zaff_contract = load_parameter_contract()(system, amber.topology, prep_path=prep, atom_types=amber._prmtop.getAtomTypes())
    if zaff_contract["status"] != "pass":
        raise ValueError(f"ZAFF parameter contract failed before Context creation: {zaff_contract['failures']}")
    original_fixed_masses = set_zero_masses(system, masks["fixed_solute_heavy"])

    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    platform = Platform.getPlatformByName(platform_name)
    properties = {}
    if "Precision" in platform.getPropertyNames():
        properties["Precision"] = "double"
    context = Context(system, integrator, platform, properties)
    try:
        context.setPositions(coordinates.positions)
        context.setPeriodicBoxVectors(*coordinates.boxVectors)
        initial_state = context.getState(getEnergy=True, getPositions=True)
        initial_energy = state_energy_kj_mol(initial_state)
        initial_positions = np.asarray(initial_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float)
        LocalEnergyMinimizer.minimize(context, tolerance * unit.kilojoules_per_mole / unit.nanometer, max_iterations)
        final_state = context.getState(getEnergy=True, getPositions=True)
        final_energy = state_energy_kj_mol(final_state)
        final_positions = np.asarray(final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float)
    finally:
        del context
        del integrator

    fixed_delta = max_delta_nm(initial_positions, final_positions, masks["fixed_solute_heavy"])
    stereo = alpha_ha_stereochemistry(amber.topology, initial_positions, final_positions)
    finite_energies = bool(np.isfinite(initial_energy) and np.isfinite(final_energy))
    end_changed = changed_inputs(start_snapshot)
    status = "hydrogen_minimization_complete" if (
        finite_energies
        and fixed_delta <= FIXED_HEAVY_TOLERANCE_NM
        and stereo["status"] == "pass"
        and not end_changed
    ) else "failed"

    positions_path = output_dir / "hydrogen_minimized_positions.npy"
    restart_path = output_dir / "hydrogen_minimized.rst7"
    box_path = output_dir / "box_vectors_nm.npy"
    np.save(positions_path, final_positions)
    np.save(box_path, box_nm)
    export_restart, verify_restart = load_restart_helpers()
    export_restart(prmtop, inpcrd, final_positions, box_nm, restart_path)
    restart_roundtrip = verify_restart(prmtop, restart_path, final_positions, box_nm)

    outputs = {
        "positions_npy": {"path": str(positions_path), "sha256": sha256_file(positions_path)},
        "amber_restart": {"path": str(restart_path), "sha256": sha256_file(restart_path)},
        "box_vectors_npy": {"path": str(box_path), "sha256": sha256_file(box_path)},
    }
    report = {
        "status": status,
        "scope": "fixed-heavy hydrogen/solvent preparation minimization only; no velocities, dynamics, CMMotionRemover, barostat, or production-readiness claim",
        "md_qualified": False,
        "production_ready": False,
        "sources_start": start_snapshot,
        "sources_end_changed": end_changed,
        "outputs": outputs,
        "system_creation": {"nonbonded_method": "PME", "constraints": None, "rigidWater": False, "removeCMMotion": False},
        "platform": {"requested": platform_name, "properties": properties},
        "minimizer": {"algorithm": "OpenMM LocalEnergyMinimizer", "tolerance": tolerance, "max_iterations": max_iterations},
        "mask": {
            "fixed_solute_heavy_count": len(masks["fixed_solute_heavy"]),
            "mobile_count": len(masks["mobile"]),
            "hydrogen_count": len(masks["hydrogens"]),
            "solvent_or_ion_count": len(masks["solvent_or_ions"]),
            "fixed_solute_heavy_max_delta_nm": fixed_delta,
            "fixed_tolerance_nm": FIXED_HEAVY_TOLERANCE_NM,
            "original_fixed_mass_count": len(original_fixed_masses),
        },
        "finite_geometry": geom,
        "energies_kj_mol": {"initial": initial_energy, "final": final_energy, "finite": finite_energies},
        "parameter_contract": {"status": zaff_contract["status"], "failures": zaff_contract["failures"]},
        "alpha_ha_stereochemistry": stereo,
        "restart_roundtrip": restart_roundtrip,
        "heavy_clash_gate": "not evaluated as pass/fail in this H/solvent-only stage; modeled heavy clashes are handled by later staged preparation",
    }
    write_json(output_dir / "relax_atomistic_hydrogens.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", type=Path, required=True)
    parser.add_argument("--inpcrd", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--offline", action="store_true", help="Accepted for provenance; this runner performs local file/compute operations only")
    args = parser.parse_args(argv)
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    report = run(
        prmtop=args.prmtop,
        inpcrd=args.inpcrd,
        prep=args.prep,
        output_dir=args.output_dir,
        platform_name=args.platform,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )
    print(json.dumps({"status": report["status"], "fixed_solute_heavy_max_delta_nm": report["mask"]["fixed_solute_heavy_max_delta_nm"]}, indent=2))
    return 0 if report["status"] == "hydrogen_minimization_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
