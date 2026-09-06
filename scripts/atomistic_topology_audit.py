#!/usr/bin/env python3
"""Inspect an Amber CRBN system before any atomistic response simulation.

An input's presence in a publication is provenance, not a force-field
validation. This audit does not repair or simulate the supplied system.
OpenMM is an optional dependency for this separate atomistic workflow.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


def materialize(source: Path, destination: Path) -> dict:
    raw = source.read_bytes()
    body = gzip.decompress(raw) if source.suffix == ".gz" else raw
    destination.write_bytes(body)
    return {"path": str(source), "retained_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": hashlib.sha256(body).hexdigest(), "content_bytes": len(body)}


def inspect_system(system, topology, positions_nm):
    from openmm import NonbondedForce, HarmonicBondForce, HarmonicAngleForce, unit

    atoms = list(topology.atoms())
    xyz = np.asarray(positions_nm, dtype=float)
    if xyz.shape != (system.getNumParticles(), 3) or len(atoms) != len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Coordinate, topology, and system atom counts must agree and be finite")
    nb_forces = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    if len(nb_forces) != 1:
        raise ValueError("Audit requires one conventional NonbondedForce")
    nb = nb_forces[0]
    charges = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                        for i in range(len(atoms))])
    zinc = [a for a in atoms if a.element is not None and a.element.symbol == "Zn"]
    sulfurs = [a for a in atoms if a.element is not None and a.element.symbol == "S"]
    bonds = []
    angles = []
    for f in system.getForces():
        if isinstance(f, HarmonicBondForce):
            for i in range(f.getNumBonds()):
                a, b, length, k = f.getBondParameters(i)
                bonds.append((int(a), int(b), float(length.value_in_unit(unit.nanometer)),
                              float(k.value_in_unit(unit.kilojoules_per_mole / unit.nanometer**2))))
        elif isinstance(f, HarmonicAngleForce):
            for i in range(f.getNumAngles()):
                a, b, c, theta, k = f.getAngleParameters(i)
                angles.append((int(a), int(b), int(c)))
    exceptions = {}
    zinc_ids = {a.index for a in zinc}
    for i in range(nb.getNumExceptions()):
        a, b, product, sigma, epsilon = nb.getExceptionParameters(i)
        if int(a) in zinc_ids or int(b) in zinc_ids:
            exceptions[tuple(sorted((int(a), int(b))))] = {
                "charge_product_e2": float(product.value_in_unit(unit.elementary_charge**2)),
                "epsilon_kj_mol": float(epsilon.value_in_unit(unit.kilojoules_per_mole))}
    sites = []
    blockers = []
    if not zinc:
        blockers.append("CRBN structural zinc is absent")
    for z in zinc:
        nearest = sorted(sulfurs, key=lambda a: np.linalg.norm(xyz[a.index] - xyz[z.index]))[:4]
        rows = []
        for a in nearest:
            residue_atoms = list(a.residue.atoms())
            h_names = [b.name for b in residue_atoms if b.element is not None and b.element.symbol == "H"]
            rows.append({"residue_index": a.residue.index, "topology_residue_id": a.residue.id,
                         "residue_name": a.residue.name, "atom_index": a.index, "atom_name": a.name,
                         "distance_A": float(10 * np.linalg.norm(xyz[a.index] - xyz[z.index])),
                         "sulfur_charge_e": float(charges[a.index]),
                         "residue_charge_e": float(sum(charges[b.index] for b in residue_atoms)),
                         "hydrogen_names": h_names,
                         "zinc_pair_exception": exceptions.get(tuple(sorted((a.index, z.index))))})
        z_bonds = [b for b in bonds if z.index in b[:2]]
        z_angles = [a for a in angles if z.index in a]
        sg4 = len(rows) == 4 and all(r["atom_name"] == "SG" and 1.9 <= r["distance_A"] <= 3.0 for r in rows)
        protonated = any("HG" in r["hydrogen_names"] for r in rows)
        if not sg4:
            blockers.append(f"Zn {z.index}: the input does not contain four nearby cysteine SG donors")
        if protonated:
            blockers.append(f"Zn {z.index}: at least one of the nearest cysteines retains HG; chemical-state review required")
        if not z_bonds and not z_angles:
            blockers.append(f"Zn {z.index}: nonbonded-only model has no supplied Cys4-site validation")
        sites.append({"atom_index": z.index, "charge_e": float(charges[z.index]),
                      "nearest_sulfurs": rows, "sg4_geometry_screen_pass": sg4,
                      "zinc_bonds": z_bonds, "zinc_angles": z_angles})
    total = float(charges.sum())
    if abs(total - round(total)) > 1e-4:
        blockers.append("System charge is not integral within 1e-4 e")
    # Geometry and chemical-state screens cannot establish transferability of a parameter set.
    blockers.append("A source-specific metal-parameter validation record is required before production")
    return {"status": "audit_complete", "production_ready": False,
            "num_atoms": len(atoms), "num_residues": topology.getNumResidues(),
            "total_charge_e": total, "total_charge_integral_tolerance_e": 1e-4,
            "force_classes": [type(f).__name__ for f in system.getForces()],
            "zinc_sites": sites, "production_blockers": blockers,
            "interpretation": "Screens this supplied input only; it does not assess every calculation in the source study."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="All operations are local, regardless of this flag")
    args = parser.parse_args()
    from openmm import app, unit

    with tempfile.TemporaryDirectory(prefix="crbn_atomistic_audit_") as temporary:
        top_path, crd_path = Path(temporary) / "system.parm7", Path(temporary) / "system.rst7"
        sources = [materialize(args.prmtop, top_path), materialize(args.coordinates, crd_path)]
        top = app.AmberPrmtopFile(str(top_path))
        crd = app.AmberInpcrdFile(str(crd_path))
        system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=.9 * unit.nanometer,
                                  constraints=app.HBonds)
        result = inspect_system(system, top.topology, crd.positions.value_in_unit(unit.nanometer))
    result["sources"] = sources
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "topology_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "production_ready", "num_atoms", "production_blockers")}, indent=2))


if __name__ == "__main__":
    main()
