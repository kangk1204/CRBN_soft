#!/usr/bin/env python3
"""Static double/mixed validation followed by a bounded zero-force benchmark.

Qualified coordinates, reference, q and chemistry gates are shared with the
technical pilot. No minimization, implicit alignment, default-precision change
or scientific convergence claim is made. Static comparisons run first for
flexible/fixed/rigid boundaries; any failure prevents all integration benchmarks.
The nonzero probe is evaluated only as an isolated static diagnostic and reset
to zero before any dynamics. CPU/Reference are for explicit unit fixtures only.

OpenMM precision/property semantics:
https://docs.openmm.org/latest/userguide/library/04_platform_specifics.html
The installed version and actual Context properties, not the documentation
version, determine the runtime precision recorded in every result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import openmm as mm
from openmm import unit

try:
    from . import run_atomistic_technical_pilot as pilot
except ImportError:
    import run_atomistic_technical_pilot as pilot

STATIC_POLICY = {
    "energy_difference_per_original_particle_kj_mol_max": 1e-4,
    "force_rms_absolute_tolerance_kj_mol_nm": 1e-3,
    "force_rms_relative_tolerance": 1e-4,
    "analytic_energy_absolute_tolerance_kj_mol": 1e-4,
    "analytic_energy_relative_tolerance": 1e-3,
    "analytic_force_rms_absolute_tolerance_kj_mol_nm": 1e-3,
    "analytic_force_rms_relative_tolerance": 1e-4,
    "static_probe_h_kj_mol_nm": 1.0,
    "integration_probe_h_kj_mol_nm": 0.0,
    "force_rms_definition": "sqrt(mean(F_component**2)); includes all x/y/z components",
    "boundary_comparisons": "fixed versus flexible full original force; rigid versus flexible non-body force and body net force/torque transferred to anchors",
    "scope": "fixed technical screening rules; not statistical equivalence",
    "later_converged_response_precision_sensitivity_max_fraction": 0.05,
    "later_response_precision_validation_required": True,
}
BENCHMARK_MODELS = ("flexible", "fixed", "rigid")
GAUGE_GROUP, PROBE_GROUP = 30, 31


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2))) if values.size else 0.0


def force_comparison(reference, candidate):
    reference, candidate = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {"pass": False, "reason": "Nonfinite or mismatched force arrays"}
    delta = candidate - reference
    limit = STATIC_POLICY["force_rms_absolute_tolerance_kj_mol_nm"] + STATIC_POLICY["force_rms_relative_tolerance"] * rms(reference)
    return {"pass": rms(delta) <= limit, "rms_difference": rms(delta), "reference_rms": rms(reference),
            "rms_limit": limit, "maximum_absolute_component_difference": float(np.max(np.abs(delta))) if delta.size else 0.0}


def energy_comparison(reference, candidate, original_count):
    difference = abs(float(candidate) - float(reference)) / original_count
    return {"pass": bool(np.isfinite(difference) and difference <= STATIC_POLICY["energy_difference_per_original_particle_kj_mol_max"]),
            "absolute_difference_kj_mol": abs(float(candidate) - float(reference)),
            "difference_per_original_particle_kj_mol": difference, "original_particle_count": original_count}


def analytic_comparison(expected_energy, actual_energy, expected_forces, actual_forces):
    energy_limit = (STATIC_POLICY["analytic_energy_absolute_tolerance_kj_mol"]
                    + STATIC_POLICY["analytic_energy_relative_tolerance"] * abs(expected_energy))
    energy_difference = abs(actual_energy - expected_energy)
    forces = force_comparison(expected_forces, actual_forces)
    return {"pass": bool(np.isfinite(energy_difference) and energy_difference <= energy_limit and forces["pass"]),
            "expected_energy_kj_mol": float(expected_energy), "actual_energy_kj_mol": float(actual_energy),
            "absolute_energy_difference_kj_mol": float(energy_difference), "energy_limit_kj_mol": float(energy_limit),
            "forces": forces}


def _state_energy_force(context, groups=-1):
    state = context.getState(getEnergy=True, getForces=True, groups=groups)
    return (float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)),
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))


def _snapshot_comparison(reference_energy, reference_forces, energy, forces, core, original_count):
    checks = {"energy": energy_comparison(reference_energy, energy, original_count),
              "full_force": force_comparison(reference_forces, forces),
              "core_force": force_comparison(reference_forces[core], forces[core])}
    return {"pass": all(check["pass"] for check in checks.values()), **checks}


def _fixed_coordinate_diagnostics(context, energy, forces, positions, core, original_count,
                                  repeats, *, deadline=None):
    """Keep compact statistics, not a stack of full-system force arrays.

    The caller has reset the probe to h=0. Every comparison uses the unchanged
    technical energy/force thresholds. Separate-group sums are diagnostic only:
    they never replace the all-groups force that the integrator would use.
    """
    baseline_forces = np.array(forces, copy=True)
    records = []
    for repeat in range(repeats):
        if deadline is not None and time.monotonic() >= deadline:
            raise pilot._BudgetStop("Wall budget exhausted during fixed-coordinate force diagnostic")
        all_energy, all_forces = ((energy, baseline_forces) if repeat == 0
                                  else _state_energy_force(context))
        # State.getForces returns an owned per-State array in OpenMM 8.5.2;
        # explicit copying also protects these diagnostics against reused test
        # buffers or a future wrapper change during the following group calls.
        all_forces = np.array(all_forces, copy=True)
        group_energy, group_forces = 0., np.zeros_like(all_forces)
        for group in (0, GAUGE_GROUP, PROBE_GROUP):
            if deadline is not None and time.monotonic() >= deadline:
                raise pilot._BudgetStop("Wall budget exhausted before force-group diagnostic")
            value, force = _state_energy_force(context, 1 << group)
            group_energy += value
            group_forces += force
        actual_positions = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        unchanged = bool(np.array_equal(positions, actual_positions))
        repeated = _snapshot_comparison(energy, baseline_forces, all_energy, all_forces, core, original_count)
        separated = _snapshot_comparison(group_energy, group_forces, all_energy, all_forces, core, original_count)
        records.append({"repeat": repeat, "pass": bool(unchanged and repeated["pass"] and separated["pass"]),
                        "coordinates_unchanged": unchanged,
                        "all_groups_vs_first": repeated,
                        "all_groups_vs_sum_of_groups": separated})
    return {"pass": all(row["pass"] for row in records), "completed_repeats": len(records),
            "force_h_kj_mol_nm": 0., "group_ids": [0, GAUGE_GROUP, PROBE_GROUP],
            "reference": "first all-groups evaluation at the same fixed coordinates",
            "criterion": "unchanged STATIC_POLICY energy per original particle and full/core force RMS limits",
            "separate_group_sum_use": "diagnostic only; never substituted for dynamics or acceptance",
            "records": records}


def _prepared(loaded, model):
    prepared = pilot._prepare_system(loaded["system"], np.array(loaded["xyz"], copy=True),
                                      loaded["mapping"], model, loaded["settings"])
    system, xyz, gauge, body, _ = prepared
    for force in system.getForces():
        name = force.getName()
        force.setForceGroup(GAUGE_GROUP if name.startswith("atomistic_boundary:gauge:") else
                            PROBE_GROUP if name.startswith("atomistic_boundary:probe:") else 0)
        if isinstance(force, mm.NonbondedForce):
            force.setReciprocalSpaceForceGroup(0)
    return system, xyz, gauge, body


def _check_qualification(loaded, synthetic_fixture):
    if synthetic_fixture:
        return
    if loaded["qualification"].get("chemical_review", {}).get("status") != pilot.CHEMICAL_REVIEW_PASS:
        raise ValueError("The precision benchmark requires qualified chemical preparation")
    if not isinstance(loaded.get("chemical_geometry"), pilot.ChemicalGeometry):
        raise ValueError("Full topology-derived chemical geometry is mandatory")


def capture_static(loaded, model, platform_name, precision, device_index=None, *, synthetic_fixture=False,
                   disable_pme_stream=False, static_repeats=1, deadline=None):
    """No integration or coordinate adjustment; both contexts receive same xyz."""
    _check_qualification(loaded, synthetic_fixture)
    if type(static_repeats) is not int or not 1 <= static_repeats <= 100:
        raise ValueError("Static repeats must lie in 1..100")
    started = time.monotonic()
    system, xyz, gauge, body = _prepared(loaded, model)
    platform, properties = pilot.platform_options(platform_name, precision, device_index, disable_pme_stream=disable_pme_stream)
    integrator = mm.VerletIntegrator(loaded["settings"].timestep_fs / 1000)
    context = mm.Context(system, integrator, platform, properties)
    try:
        precision_metadata = pilot.precision_record(platform, context, precision, disable_pme_stream=disable_pme_stream)
        context.setPositions(xyz)
        context.computeVirtualSites()
        actual_xyz = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        energy, forces = _state_energy_force(context)
        gauge_energy, gauge_forces = _state_energy_force(context, 1 << GAUGE_GROUP)
        core = loaded["mapping"]["core_indices"]
        ref = np.asarray(loaded["mapping"]["reference_nm"])
        q = np.asarray(loaded["mapping"]["q"])
        B = np.asarray(gauge["rigid_basis"])
        # Analytic values use the exact shared input, not precision-dependent
        # readback coordinates or fitted frames.
        displacement = (np.asarray(loaded["xyz"])[core] - ref).ravel()
        g = B.T @ displacement
        k = loaded["settings"].gauge_k
        expected_gauge = np.zeros_like(xyz)
        expected_gauge[core] = (-k * B @ g).reshape(-1, 3)
        gauge_check = analytic_comparison(.5 * k * float(g @ g), gauge_energy, expected_gauge[core], gauge_forces[core])
        h = STATIC_POLICY["static_probe_h_kj_mol_nm"]
        context.setParameter(gauge["probe_parameter"], h)
        probe_energy, probe_forces = _state_energy_force(context, 1 << PROBE_GROUP)
        context.setParameter(gauge["probe_parameter"], 0.)
        expected_probe = np.zeros_like(xyz)
        expected_probe[core] = (h * q).reshape(-1, 3)
        probe_check = analytic_comparison(-h * float(q @ displacement), probe_energy, expected_probe[core], probe_forces[core])
        outside = sorted(set(range(len(xyz))) - set(core))
        for check, actual in ((gauge_check, gauge_forces), (probe_check, probe_forces)):
            check["force_scope"] = "ordered measured core; off-core forces checked separately"
            check["offcore_forces"] = force_comparison(np.zeros_like(actual[outside]), actual[outside])
            check["pass"] = bool(check["pass"] and check["offcore_forces"]["pass"])
        geometry = loaded.get("chemical_geometry")
        chemical = pilot.chemical_geometry_screen(actual_xyz, geometry) if geometry is not None else {"pass": True, "scope": "explicit synthetic fixture"}
        finite = bool(np.isfinite(actual_xyz).all() and np.isfinite(forces).all() and np.isfinite(energy))
        count = len(loaded["xyz"])
        repeated = _fixed_coordinate_diagnostics(context, energy, forces, actual_xyz, core, count,
                                                 static_repeats, deadline=deadline)
        record = {"pass": bool(finite and chemical["pass"] and gauge_check["pass"] and probe_check["pass"] and repeated["pass"]),
                  "model": model, "precision": precision_metadata, "platform": platform.getName(),
                  "platform_properties": {name: platform.getPropertyValue(context, name) for name in platform.getPropertyNames()},
                  "energy_kj_mol": energy, "gauge_analytic": gauge_check, "probe_analytic": probe_check,
                  "fixed_coordinate_diagnostics": repeated,
                  "chemical_geometry": chemical, "original_particle_count": count,
                  "engine_particle_count": len(xyz), "body": body,
                  "original_coordinate_readback_max_difference_nm": float(np.max(np.abs(actual_xyz[:count] - loaded["xyz"]))),
                  "input_coordinate_array_sha256": hashlib.sha256(np.asarray(loaded["xyz"], dtype="<f8").tobytes()).hexdigest(),
                  "elapsed_wall_seconds": time.monotonic() - started}
        arrays = {"positions_nm": actual_xyz, "forces_kj_mol_nm": forces,
                  "gauge_forces_kj_mol_nm": gauge_forces, "probe_forces_kj_mol_nm": probe_forces}
        return record, arrays
    finally:
        del context, integrator


def compare_static_snapshots(records, arrays, mapping):
    """Compare precision pairs and the unchanged-potential boundary conversions."""
    comparisons = {}
    core, body = mapping["core_indices"], mapping["ddb1_atom_indices"]
    for model in BENCHMARK_MODELS:
        double, mixed = f"double_{model}", f"mixed_{model}"
        count = records[double]["original_particle_count"]
        comparisons[f"precision_{model}"] = {
            "energy": energy_comparison(records[double]["energy_kj_mol"], records[mixed]["energy_kj_mol"], count),
            "full_force": force_comparison(arrays[double]["forces_kj_mol_nm"], arrays[mixed]["forces_kj_mol_nm"]),
            "core_force": force_comparison(arrays[double]["forces_kj_mol_nm"][core], arrays[mixed]["forces_kj_mol_nm"][core])}
    for precision in ("double", "mixed"):
        base = f"{precision}_flexible"
        count = records[base]["original_particle_count"]
        nonbody = sorted(set(range(count)) - set(body))
        center = arrays[base]["positions_nm"][body].mean(axis=0)
        baseline_forces = arrays[base]["forces_kj_mol_nm"]
        for model in ("fixed", "rigid"):
            key = f"{precision}_{model}"
            actual_forces = arrays[key]["forces_kj_mol_nm"]
            comparison = {"energy": energy_comparison(records[base]["energy_kj_mol"], records[key]["energy_kj_mol"], count),
                          "nonbody_force": force_comparison(baseline_forces[nonbody], actual_forces[nonbody])}
            if model == "fixed":
                comparison["body_force"] = force_comparison(baseline_forces[body], actual_forces[body])
            else:
                anchors = records[key]["body"]["anchor_indices"]
                comparison["body_net_force"] = force_comparison(baseline_forces[body].sum(axis=0), actual_forces[anchors].sum(axis=0))
                expected_torque = np.cross(arrays[base]["positions_nm"][body] - center, baseline_forces[body]).sum(axis=0)
                actual_torque = np.cross(arrays[key]["positions_nm"][anchors] - center, actual_forces[anchors]).sum(axis=0)
                comparison["body_net_torque_kj_mol"] = force_comparison(expected_torque, actual_torque)
            comparisons[f"boundary_{key}"] = comparison
    passed = all(record["pass"] for record in records.values()) and all(check["pass"] for row in comparisons.values() for check in row.values())
    return {"pass": bool(passed), "comparisons": comparisons}


def _integration_screen(context, system, loaded, model, initial, stage):
    state = context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=False)
    xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(xyz).all() or not np.isfinite([energy, kinetic]).all():
        raise ValueError("Nonfinite integration energy or coordinates")
    chemistry = loaded.get("chemical_geometry")
    chemical = pilot.chemical_geometry_screen(xyz, chemistry) if chemistry is not None else {"pass": True, "scope": "explicit synthetic fixture"}
    error = 0.
    for i, j, distance in (system.getConstraintParameters(k) for k in range(system.getNumConstraints())):
        expected = distance.value_in_unit(unit.nanometer)
        error = max(error, abs(np.linalg.norm(xyz[i] - xyz[j]) - expected) / expected)
    fixed = float(np.max(np.linalg.norm(xyz[loaded["mapping"]["ddb1_atom_indices"]] - initial[loaded["mapping"]["ddb1_atom_indices"]], axis=1))) if model == "fixed" else 0.
    row = {"stage": stage, "time_ps": float(state.getTime().value_in_unit(unit.picosecond)),
           "potential_kj_mol": float(energy), "chemical_geometry": chemical,
           "max_relative_constraint_error": float(error), "fixed_body_max_displacement_nm": fixed}
    mapping = loaded["mapping"]
    if "zn_atom_index" in mapping:
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        distances = np.linalg.norm(pilot._minimum_image(xyz[mapping["zn_sg_indices"]] - xyz[mapping["zn_atom_index"]], box), axis=1)
        low, high = mapping.get("zn_sg_distance_bounds_nm", [.18, .30])
        row.update(zn_sg_distances_nm=distances.tolist(), zn_sg_screen_pass=bool(np.all((distances >= low) & (distances <= high))))
    row["pass"] = bool(chemical["pass"] and (stage == "initial" or
                        (error <= max(10 * loaded["settings"].constraint_tolerance, 1e-5) and fixed <= 1e-10 and row.get("zn_sg_screen_pass", True))))
    return row


def measure_integration(loaded, model, platform_name, precision, device_index, steps, deadline, *,
                        synthetic_fixture=False, disable_pme_stream=False):
    """Fresh identical coordinates, deterministic host velocities; no minimizer."""
    _check_qualification(loaded, synthetic_fixture)
    if not isinstance(steps, int) or not 1 <= steps <= 1000:
        raise ValueError("Integration benchmark is limited to 1..1000 steps")
    setup_start = time.monotonic()
    system, xyz, gauge, _ = _prepared(loaded, model)
    settings = loaded["settings"]
    platform, properties = pilot.platform_options(platform_name, precision, device_index, disable_pme_stream=disable_pme_stream)
    integrator = mm.LangevinMiddleIntegrator(settings.temperature_K, settings.friction_per_ps, settings.timestep_fs / 1000)
    integrator.setConstraintTolerance(settings.constraint_tolerance)
    integrator.setRandomNumberSeed(settings.seed)
    context = mm.Context(system, integrator, platform, properties)
    record = {"pass": False, "model": model, "requested_steps": steps, "completed_steps": 0,
              "warmup_steps": 10, "minimization_performed": False, "screens": []}
    try:
        record["precision"] = pilot.precision_record(platform, context, precision, disable_pme_stream=disable_pme_stream)
        context.setPositions(xyz)
        context.computeVirtualSites()
        context.setParameter(gauge["probe_parameter"], 0.)
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(len(xyz))])
        velocities = np.zeros_like(xyz)
        active = masses > 0
        velocities[active] = np.random.default_rng(settings.seed + 1).normal(size=(int(active.sum()), 3)) * np.sqrt(pilot.GAS_CONSTANT * settings.temperature_K / masses[active, None])
        record["initial_velocity_array_sha256"] = hashlib.sha256(velocities.astype("<f8").tobytes()).hexdigest()
        context.setVelocities(velocities)
        context.applyVelocityConstraints(settings.constraint_tolerance)

        def screen(stage):
            row = _integration_screen(context, system, loaded, model, xyz, stage)
            record["screens"].append(row)
            if not row["pass"]:
                raise ValueError(f"Unchanged technical geometry gate failed at {stage}")
            if time.monotonic() >= deadline:
                raise pilot._BudgetStop("Cooperative wall budget reached")

        screen("initial")
        record["setup_wall_seconds"] = time.monotonic() - setup_start
        integrator.step(10)
        screen("warmup")
        measured_start = time.monotonic()
        while record["completed_steps"] < steps:
            if time.monotonic() >= deadline:
                raise pilot._BudgetStop("Cooperative wall budget reached")
            amount = min(settings.max_step_batch, steps - record["completed_steps"])
            integrator.step(amount)
            record["completed_steps"] += amount
            screen("measured")  # getState synchronizes before stopping the timer.
        elapsed = time.monotonic() - measured_start
        record.update(pass_=True, measured_wall_seconds=elapsed,
                      steps_per_second=steps / elapsed, ns_per_day=steps * settings.timestep_fs / 1e6 / elapsed * 86400)
        record["pass"] = record.pop("pass_")
    except Exception as error:
        record.update(reason=f"{type(error).__name__}: {error}")
    finally:
        try:
            final = _integration_screen(context, system, loaded, model, xyz, "final")
            record["screens"].append(final)
            record["pass"] = bool(record["pass"] and final["pass"])
        except Exception as error:
            record.update({"pass": False, "reason": f"Final screen: {type(error).__name__}: {error}"})
        context = integrator = None
    return record


def run(prmtop, inpcrd, mapping_path, config_path, qualification_path, output_dir, *,
        platform_name="OpenCL", device_index=None, steps=100, max_wall_seconds=600., static_only=False,
        disable_pme_stream=False, static_repeats=1, diagnostic_only=False,
        diagnostic_model="flexible", diagnostic_precision="double"):
    if platform_name not in pilot.GPU_PLATFORMS:
        raise ValueError("The double/mixed precision comparison requires a GPU platform")
    if not isinstance(steps, int) or not 100 <= steps <= 1000:
        raise ValueError("CLI benchmark steps must lie in 100..1000")
    if not np.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= 3600:
        raise ValueError("max_wall_seconds must lie in (0, 3600]")
    if type(static_repeats) is not int or not 1 <= static_repeats <= 100:
        raise ValueError("Static repeats must lie in 1..100")
    if diagnostic_model not in BENCHMARK_MODELS or diagnostic_precision not in ("double", "mixed"):
        raise ValueError("Invalid static diagnostic boundary or precision")
    started = time.monotonic()
    paths = {"prmtop": Path(prmtop), "inpcrd": Path(inpcrd), "mapping": Path(mapping_path),
             "config": Path(config_path), "qualification": Path(qualification_path),
             "benchmark_source": Path(__file__), "pilot_source": Path(pilot.__file__), "boundary_source": Path(pilot.boundary.__file__)}
    hashes = {name: pilot._sha256(path) for name, path in paths.items()}
    loaded = pilot.load_qualified_inputs(prmtop, inpcrd, mapping_path, config_path, qualification_path)
    if hashes != {name: pilot._sha256(path) for name, path in paths.items()}:
        raise ValueError("Precision benchmark sources changed while loading qualified inputs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "accepted_mixed": False, "default_precision": "double", "default_changed": False,
              "production_ready": False, "response_converged": False, "policy": STATIC_POLICY,
              "input_provenance": loaded["provenance"], "input_sha256": hashes,
              "settings": asdict(loaded["settings"]), "openmm_version": mm.__version__,
              "static": {}, "integration": {}, "requested_steps_per_mode": steps,
              "static_only": bool(static_only),
              "disable_pme_stream_requested": bool(disable_pme_stream),
              "static_repeats_per_context": static_repeats,
              "diagnostic_only": bool(diagnostic_only),
              "diagnostic_selection": {"model": diagnostic_model, "precision": diagnostic_precision} if diagnostic_only else None,
              "wall_budget_seconds": max_wall_seconds,
              "wall_budget_rule": "cooperative before each Context and batch; a single setup/kernel can overrun",
              "benchmark_timing_scope": "integration plus synchronized geometry reports; excludes Context setup and 10 warmup steps; no minimization"}
    arrays = {}
    try:
        selections = ([(diagnostic_precision, diagnostic_model)] if diagnostic_only else
                      [(precision, model) for precision in ("double", "mixed") for model in BENCHMARK_MODELS])
        for precision, model in selections:
            if time.monotonic() >= started + max_wall_seconds:
                raise pilot._BudgetStop("Wall budget exhausted before static Context")
            key = f"{precision}_{model}"
            record, snapshot = capture_static(loaded, model, platform_name, precision, device_index,
                                               disable_pme_stream=disable_pme_stream, static_repeats=static_repeats,
                                               deadline=started + max_wall_seconds)
            report["static"][key], arrays[key] = record, snapshot
            path = output_dir / f"static_{key}.npz"
            np.savez_compressed(path, **snapshot)
            record["artifact_sha256"] = pilot._sha256(path)
            pilot._write_json(output_dir / "precision_benchmark.json", report)
        if diagnostic_only:
            report["static_comparison"] = {"pass": None, "scope": "not evaluated in a one-Context diagnostic"}
            report["status"] = "static_diagnostic_pass" if record["pass"] else "static_diagnostic_failed"
        else:
            report["static_comparison"] = compare_static_snapshots(report["static"], arrays, loaded["mapping"])
        if diagnostic_only:
            pass  # No integration or precision/boundary acceptance can follow.
        elif not report["static_comparison"]["pass"]:
            report["status"] = "static_failed_no_integration"
        elif static_only:
            report["status"] = "static_precision_screen_pass"
        else:
            report["status"] = "static_pass"
            for precision in ("double", "mixed"):
                for model in BENCHMARK_MODELS:
                    if time.monotonic() >= started + max_wall_seconds:
                        raise pilot._BudgetStop("Wall budget exhausted before integration Context")
                    key = f"{precision}_{model}"
                    record = measure_integration(loaded, model, platform_name, precision, device_index, steps, started + max_wall_seconds,
                                                 disable_pme_stream=disable_pme_stream)
                    report["integration"][key] = record
                    pilot._write_json(output_dir / "precision_benchmark.json", report)
                    if not record["pass"]:
                        raise ValueError(f"Integration screen or budget failed for {key}")
            report.update(status="technical_precision_screen_pass", accepted_mixed=True)
            report["speedup_mixed_over_double"] = {
                model: report["integration"][f"mixed_{model}"]["steps_per_second"] / report["integration"][f"double_{model}"]["steps_per_second"]
                for model in BENCHMARK_MODELS}
    except Exception as error:
        report.update(status="incomplete_or_failed", accepted_mixed=False, reason=f"{type(error).__name__}: {error}")
    if hashes != {name: pilot._sha256(path) for name, path in paths.items()}:
        report.update(status="failed_input_changed", accepted_mixed=False)
    report["elapsed_wall_seconds"] = time.monotonic() - started
    pilot._write_json(output_dir / "precision_benchmark.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prmtop", "inpcrd", "mapping", "qualification"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=pilot.ROOT / "scripts/atomistic_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=pilot.GPU_PLATFORMS, default="OpenCL")
    parser.add_argument("--device-index")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-wall-seconds", type=float, default=600.)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--static-only", action="store_true", help="Stop after identical-coordinate checks; never start dynamics")
    parser.add_argument("--disable-pme-stream", action="store_true", help="Explicit GPU property override, verified by Context readback")
    parser.add_argument("--static-repeats", type=int, default=1, help="1..100 fixed-coordinate repetitions, including separate-group comparisons")
    parser.add_argument("--diagnostic-only", action="store_true", help="One selected Context; compact repeated-force diagnostics, no dynamics or precision acceptance")
    parser.add_argument("--diagnostic-model", choices=BENCHMARK_MODELS, default="flexible")
    parser.add_argument("--diagnostic-precision", choices=("double", "mixed"), default="double")
    args = parser.parse_args(argv)
    try:
        result = run(args.prmtop, args.inpcrd, args.mapping, args.config, args.qualification, args.output_dir,
                     platform_name=args.platform, device_index=args.device_index, steps=args.steps,
                     max_wall_seconds=args.max_wall_seconds, static_only=args.static_only,
                     disable_pme_stream=args.disable_pme_stream, static_repeats=args.static_repeats,
                     diagnostic_only=args.diagnostic_only, diagnostic_model=args.diagnostic_model,
                     diagnostic_precision=args.diagnostic_precision)
    except Exception as error:
        print(json.dumps({"status": "rejected_before_benchmark", "reason": f"{type(error).__name__}: {error}"}))
        return 2
    print(json.dumps({key: result[key] for key in ("status", "accepted_mixed", "default_changed", "elapsed_wall_seconds")}))
    return 0 if result["accepted_mixed"] or result["status"] in ("static_precision_screen_pass", "static_diagnostic_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
