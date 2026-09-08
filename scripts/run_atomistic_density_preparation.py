#!/usr/bin/env python3
"""Bounded h=0 NPT density preparation for qualified atomistic inputs.

This runner prepares density only. It does not certify equilibrium, production
readiness, or response convergence. Qualified inputs must already encode the
salt/protonation state, complete topology geometry, frozen 269-core reference
frame, and q direction used by the technical pilot.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import openmm as mm
from openmm import unit

try:
    from . import atomistic_boundary as boundary
    from . import run_atomistic_technical_pilot as pilot
except ImportError:
    import atomistic_boundary as boundary
    import run_atomistic_technical_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
GAS_CONSTANT = pilot.GAS_CONSTANT
DENSITY_DA_PER_NM3_TO_G_PER_ML = 0.00166053906660
DENSITY_MODELS = ("flexible", "isolated")
REQUIRED_OPENMM_VERSION = "8.5.2"


@dataclass(frozen=True)
class DensitySettings:
    nvt_steps: int = 0
    npt_steps: int = 20000
    timestep_fs: float = 1.
    temperature_K: float = 300.
    friction_per_ps: float = 1.
    gauge_k: float = 1000.
    seed: int = 20260907
    max_wall_seconds: float | None = 300.
    report_interval_steps: int = 100
    max_step_batch: int = 100
    constraint_tolerance: float = 1e-6
    pressure_bar: float = 1.
    barostat_frequency_steps: int = 25
    nonbonded_cutoff_nm: float = 1.
    checkpoint_interval_steps: int = 100
    disable_pme_stream: bool = False


class _BudgetStop(RuntimeError):
    pass


_TEST_CRASH_HOOK = None


def _maybe_test_crash(label):
    if _TEST_CRASH_HOOK is not None:
        _TEST_CRASH_HOOK(label)


def _read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path):
    return pilot._sha256(path)


def _artifact_record(path, *, base=None):
    path = Path(path)
    name = str(path.name) if base is None else str(path.relative_to(base))
    return {"path": name, "sha256": _file_sha256(path), "bytes": path.stat().st_size}


def _restart_contract_settings(settings):
    values = asdict(settings)
    for key in ("max_wall_seconds", "nvt_steps", "npt_steps"):
        values.pop(key, None)
    return values


def _force_type_counts(system):
    counts = {}
    for force in system.getForces():
        name = type(force).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def _require_bound_hash_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("status") != "pass":
        raise ValueError("salt_protonation_manifest.status must be pass")
    hashes = manifest.get("bound_input_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("salt_protonation_manifest requires bound_input_sha256")
    required = {"prmtop", "inpcrd", "mapping"}
    missing = required - set(hashes)
    if missing:
        raise ValueError(f"salt_protonation_manifest missing bound hash(es): {sorted(missing)}")
    for key in required:
        value = hashes.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"salt_protonation_manifest bound_input_sha256.{key} must be a SHA-256 hex digest")
    return manifest


def resolve_density_settings(config, overrides=None):
    if config.get("hydrogen_mass_repartitioning", False) is not False:
        raise ValueError("Density preparation does not permit hydrogen mass repartitioning")
    values = asdict(DensitySettings())
    values.update({"temperature_K": config.get("temperature_K", 300.),
                   "timestep_fs": config.get("timestep_fs_initial", 1.),
                   "seed": config.get("seed", 20260907)})
    block = config.get("density_preparation", {})
    if not isinstance(block, dict):
        raise ValueError("density_preparation must be an object")
    unknown = set(block) - set(values)
    if unknown:
        raise ValueError(f"Unknown density_preparation setting(s): {sorted(unknown)}")
    values.update(block)
    for key, value in (overrides or {}).items():
        if value is not None or key == "max_wall_seconds":
            values[key] = value
    integer_keys = ("nvt_steps", "npt_steps", "seed", "report_interval_steps",
                    "max_step_batch", "barostat_frequency_steps", "checkpoint_interval_steps")
    nonnegative_integer_keys = ("nvt_steps",)
    boolean_keys = ("disable_pme_stream",)
    for key in boolean_keys:
        if not isinstance(values[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key, value in values.items():
        if key in boolean_keys:
            continue
        if key == "max_wall_seconds" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} must be finite numeric")
        if key in nonnegative_integer_keys:
            if int(value) != value or value < 0 or value > 2**31-1:
                raise ValueError(f"{key} must be a nonnegative 32-bit integer")
        elif value <= 0:
            raise ValueError(f"{key} must be positive")
        if key in integer_keys and (int(value) != value or value > 2**31-1):
            raise ValueError(f"{key} must be a 32-bit integer")
    if values["timestep_fs"] != 1. or values["friction_per_ps"] != 1.:
        raise ValueError("Density preparation requires 1 fs and friction 1/ps")
    if values["temperature_K"] != 300.:
        raise ValueError("Density preparation requires 300 K")
    if values["pressure_bar"] != 1.:
        raise ValueError("Density preparation requires 1 bar")
    if int(values["barostat_frequency_steps"]) != 25:
        raise ValueError("Density preparation requires barostat frequency 25")
    if values["max_step_batch"] > 100:
        raise ValueError("max_step_batch cannot exceed 100 for cooperative wall checks")
    if values["checkpoint_interval_steps"] % values["report_interval_steps"] != 0:
        raise ValueError("checkpoint_interval_steps must be a multiple of report_interval_steps")
    for key in integer_keys:
        values[key] = int(values[key])
    return DensitySettings(**values)


def _density_production_gate(qualification, synthetic_fixture):
    if synthetic_fixture:
        return {"production_prep_label_allowed": False,
                "salt_protonation_manifest_required": False,
                "salt_protonation_manifest": None}
    manifest = qualification.get("salt_protonation_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Qualified production density preparation requires salt_protonation_manifest")
    manifest = _require_bound_hash_manifest(manifest)
    return {"production_prep_label_allowed": True,
            "salt_protonation_manifest_required": True,
            "salt_protonation_manifest": manifest}


def _reject_forces(system):
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if type(force) is mm.CMMotionRemover:
            raise ValueError("Density preparation must not contain a CMMotionRemover")
        if "Barostat" in type(force).__name__:
            raise ValueError("Input System must not already contain a barostat")
        if type(force) is mm.CustomExternalForce:
            raise ValueError("Cartesian CustomExternalForce restraints are not permitted")


def _prepare_density_system(system, xyz, mapping, settings):
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    _reject_forces(system)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                       for i in range(system.getNumParticles())])
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("Density preparation requires every particle to be massive")
    gauge = boundary.add_core_gauge_and_probe(system, mapping["core_indices"],
                                              mapping["reference_nm"], mapping["q"],
                                              settings.gauge_k, 0.)
    system = gauge.system
    barostat = mm.MonteCarloBarostat(settings.pressure_bar * unit.bar,
                                     settings.temperature_K * unit.kelvin,
                                     settings.barostat_frequency_steps)
    barostat.setRandomNumberSeed(settings.seed + 2)
    system.addForce(barostat)
    constraints = [system.getConstraintParameters(i) for i in range(system.getNumConstraints())]
    pairs = [tuple(sorted((i, j))) for i, j, _ in constraints]
    if len(set(pairs)) != len(pairs) or any(masses[i] <= 0 or masses[j] <= 0 for i, j, _ in constraints):
        raise ValueError("Density preparation requires distinct constraints on massive particles")
    dof = int(3 * system.getNumParticles() - len(constraints))
    if dof <= 0:
        raise ValueError("No positive kinetic degrees of freedom")
    return system, np.asarray(xyz, dtype=float).copy(), gauge.metadata, masses, dof, barostat


def _box_nm(state):
    return state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)


def _volume_nm3(box):
    return float(np.linalg.det(np.asarray(box, dtype=float)))


def _system_default_box_nm(system):
    box = system.getDefaultPeriodicBoxVectors()
    if box is None:
        return None
    rows = []
    for vector in box:
        if hasattr(vector, "value_in_unit"):
            vector = vector.value_in_unit(unit.nanometer)
        rows.append([float(vector[0]), float(vector[1]), float(vector[2])])
    return np.asarray(rows, dtype=float)


def _validate_initial_geometry(system, xyz):
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != system.getNumParticles():
        raise ValueError("Initial coordinates must be an n_particles x 3 array")
    if not np.isfinite(xyz).all():
        raise ValueError("Initial coordinates must be finite")
    box = _system_default_box_nm(system)
    if box is None or box.shape != (3, 3) or not np.isfinite(box).all() or _volume_nm3(box) <= 0:
        raise ValueError("Initial periodic box must be finite with positive volume")
    return xyz, box


def _failed_summary(*, output_dir, settings, model, provenance, qualification, production_gate, reason, started_at, synthetic_fixture):
    summary = {
        "schema_version": "1.0",
        "status": "failed",
        "density_preparation_completed": False,
        "equilibrium_certified": False,
        "production_ready": False,
        "response_converged": False,
        "false_until_independent_density_temperature_q_assessment": True,
        "role": "bounded_h0_npt_density_preparation_only",
        "model": model,
        "settings": asdict(settings),
        "force_h_kj_mol_nm": 0.0,
        "openmm_version": mm.__version__,
        "required_openmm_version": REQUIRED_OPENMM_VERSION,
        "openmm_version_exact_match": mm.__version__ == REQUIRED_OPENMM_VERSION,
        "platform_option_requests": {"disable_pme_stream": settings.disable_pme_stream},
        "restart_seed": settings.seed,
        "resume_count": 0,
        "completed_by_phase": {"nvt": 0, "npt": 0},
        "target_by_phase": {"nvt": settings.nvt_steps, "npt": settings.npt_steps},
        "precision": {"requested": None, "effective": None, "source": "Context not created", "default_changed": False},
        "platform": None,
        "platform_properties": {},
        "chemical_review": qualification.get("chemical_review"),
        "salt_protonation_gate": production_gate,
        "input_provenance": provenance,
        "synthetic_fixture": bool(synthetic_fixture),
        "trajectory_frames": 0,
        "diagnostic_rows_this_run": 0,
        "artifact_sha256": {},
        "input_hashes_verified": provenance.get("input_sha256", {}),
        "reason": reason,
        "elapsed_wall_seconds": time.monotonic() - started_at,
    }
    _atomic_json(Path(output_dir) / "density_preparation.json", summary)
    return summary


def _integrator(settings):
    integrator = mm.LangevinMiddleIntegrator(settings.temperature_K, settings.friction_per_ps,
                                             settings.timestep_fs / 1000)
    integrator.setRandomNumberSeed(settings.seed)
    integrator.setConstraintTolerance(settings.constraint_tolerance)
    if hasattr(integrator, "setIntegrationForceGroups"):
        integrator.setIntegrationForceGroups(-1)
    return integrator


def _enforce_real_gpu_platform_contract(platform_name, precision, settings, synthetic_fixture):
    if synthetic_fixture or platform_name not in pilot.GPU_PLATFORMS:
        return
    if precision != "double" or not settings.disable_pme_stream:
        raise ValueError(
            "Production OpenCL/CUDA density preparation requires --precision double "
            "and --disable-pme-stream; default PME stream mode is rejected for "
            "real GPU density because current validation found force inconsistency"
        )


def _platform_options(platform_name, precision, device_index, disable_pme_stream):
    signature = inspect.signature(pilot.platform_options)
    if "disable_pme_stream" in signature.parameters:
        return pilot.platform_options(platform_name, precision, device_index,
                                      disable_pme_stream=disable_pme_stream)
    if disable_pme_stream:
        raise ValueError("disable_pme_stream requires the shared pilot platform_options helper")
    return pilot.platform_options(platform_name, precision, device_index)


def _existing_frames(path):
    if not path.is_file():
        return set()
    rows = list(csv.DictReader(path.open()))
    return {(row["phase"], int(row["step"])) for row in rows}


def _write_diagnostics_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _write_handoff(output_dir, summary, positions_nm, box_nm):
    _atomic_npz(output_dir / "npt_to_nvt_handoff.npz",
                positions_nm=np.asarray(positions_nm),
                box_vectors_nm=np.asarray(box_nm))
    handoff = {
        "schema_version": "1.0",
        "role": "npt_density_to_nvt_coordinate_box_handoff_only",
        "source_manifest": summary["artifact_sha256"],
        "contains": ["positions_nm", "box_vectors_nm"],
        "excludes": ["barostat_parameters", "equilibrium_certificate", "production_certificate"],
        "equilibrium_certified": False,
        "production_ready": False,
        "requires_independent_density_temperature_q_assessment": True,
    }
    _atomic_json(output_dir / "npt_to_nvt_handoff.json", handoff)
    summary["handoff_sha256"] = {
        "npt_to_nvt_handoff.npz": _file_sha256(output_dir / "npt_to_nvt_handoff.npz"),
        "npt_to_nvt_handoff.json": _file_sha256(output_dir / "npt_to_nvt_handoff.json"),
    }


def _write_final_artifact_manifest(output_dir, summary):
    artifacts = {
        **summary["artifact_sha256"],
        "density_preparation.json": _artifact_record(output_dir / "density_preparation.json"),
    }
    for name in ("npt_to_nvt_handoff.npz", "npt_to_nvt_handoff.json"):
        path = output_dir / name
        if path.exists():
            artifacts[name] = _artifact_record(path)
    manifest = {
        "schema_version": "1.0",
        "role": "density_preparation_hash_manifest",
        "artifacts": artifacts,
        "input_sha256": summary.get("input_hashes_verified", {}),
        "system_xml_sha256": summary["system_xml_sha256"],
        "frame_sha256": summary.get("frame_sha256", []),
        "current_restart": summary.get("current_restart"),
        "production_ready": False,
        "equilibrium_certified": False,
    }
    _atomic_json(output_dir / "artifact_manifest.json", manifest)


def _run_engine(system, xyz, mapping, settings, output_dir, *, model, platform_name,
                qualification, provenance, device_index=None, started_at=None,
                chemical_geometry=None, synthetic_fixture=False, precision="double"):
    start = time.monotonic() if started_at is None else started_at
    if model not in DENSITY_MODELS:
        raise ValueError("Density preparation accepts only flexible or isolated all-massive models")
    if chemical_geometry is None and not synthetic_fixture:
        raise ValueError("Full topology-derived chemical geometry is required")
    if not synthetic_fixture and qualification.get("chemical_review", {}).get("status") != pilot.CHEMICAL_REVIEW_PASS:
        raise ValueError(f"chemical_review.status must be {pilot.CHEMICAL_REVIEW_PASS}")
    production_gate = _density_production_gate(qualification, synthetic_fixture)
    if production_gate["salt_protonation_manifest"] is not None:
        declared = production_gate["salt_protonation_manifest"]["bound_input_sha256"]
        actual = provenance.get("input_sha256", {})
        for key in ("prmtop", "inpcrd", "mapping"):
            if declared[key] != actual.get(key):
                raise ValueError(f"salt_protonation_manifest bound hash mismatch: {key}")
    if not synthetic_fixture and mm.__version__ != REQUIRED_OPENMM_VERSION:
        raise ValueError(f"Density production preparation requires OpenMM {REQUIRED_OPENMM_VERSION}")
    _enforce_real_gpu_platform_contract(platform_name, precision, settings, synthetic_fixture)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "density_preparation.json"
    restart_pointer_path = output_dir / "current_restart.json"
    diagnostics_path = output_dir / "diagnostics.csv"
    trajectory_path = output_dir / "density_preparation.npz"
    generations_dir = output_dir / "generations"
    frame_keys = set()
    previous = _read_json(summary_path) if summary_path.exists() else None
    restart_pointer = _read_json(restart_pointer_path) if restart_pointer_path.exists() else None
    if previous and previous.get("status") in {"density_completed", "failed"}:
        raise ValueError("Output directory contains a terminal density preparation")
    if previous and not restart_pointer:
        raise ValueError("Resume summary exists without current_restart.json")
    if restart_pointer and restart_pointer.get("restart_seed") != settings.seed:
        raise ValueError("Resume seed mismatch; reproducible restart must not reseed")
    if restart_pointer and restart_pointer.get("settings_contract") != _restart_contract_settings(settings):
        raise ValueError("Resume settings mismatch")
    try:
        xyz, _initial_box = _validate_initial_geometry(system, xyz)
    except Exception as error:
        if restart_pointer:
            raise
        return _failed_summary(output_dir=output_dir, settings=settings, model=model, provenance=provenance,
                               qualification=qualification, production_gate=production_gate,
                               reason=f"{type(error).__name__}: {error}", started_at=start,
                               synthetic_fixture=synthetic_fixture)

    system, xyz, gauge_meta, masses, dof, barostat = _prepare_density_system(system, xyz, mapping, settings)
    system_xml = mm.XmlSerializer.serialize(system)
    roundtrip_system = mm.XmlSerializer.deserialize(system_xml)
    system_xml_sha = _sha256_bytes(system_xml.encode())
    if restart_pointer:
        if restart_pointer.get("system_xml_sha256") != system_xml_sha:
            raise ValueError("Current restart SystemXML hash mismatch")
        existing_system_path = output_dir / "system.xml"
        if not existing_system_path.exists() or _file_sha256(existing_system_path) != system_xml_sha:
            raise ValueError("Current restart SystemXML artifact hash mismatch")
    else:
        _atomic_text(output_dir / "system.xml", system_xml)
    platform, properties = _platform_options(platform_name, precision, device_index,
                                             settings.disable_pme_stream)
    deadline = None if settings.max_wall_seconds is None else start + settings.max_wall_seconds
    integrator = _integrator(settings)
    context = None
    rows = []
    frames = []
    times = []
    last_xyz = xyz.copy()
    last_box = None
    core = mapping["core_indices"]
    ca = mapping["protein_ca_indices"]
    ref = np.asarray(mapping["reference_nm"])
    q = np.asarray(mapping["q"])
    B = np.asarray(gauge_meta["rigid_basis"])
    constraint_data = [system.getConstraintParameters(i) for i in range(system.getNumConstraints())]
    constraint_i = np.array([i for i, _, _ in constraint_data], dtype=int)
    constraint_j = np.array([j for _, j, _ in constraint_data], dtype=int)
    constraint_lengths = np.array([d.value_in_unit(unit.nanometer) for _, _, d in constraint_data])
    geometry_payload = asdict(chemical_geometry) if chemical_geometry is not None else None
    geometry_sha = hashlib.sha256(json.dumps(geometry_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if geometry_payload is not None else None
    total_mass_da = float(masses.sum())
    target_steps = {"nvt": settings.nvt_steps, "npt": settings.npt_steps}
    completed = restart_pointer.get("completed_by_phase", {"nvt": 0, "npt": 0}) if restart_pointer else {"nvt": 0, "npt": 0}
    if any(completed[phase] > target_steps[phase] for phase in ("nvt", "npt")):
        raise ValueError("Current restart phase counters exceed requested target steps")
    if restart_pointer:
        expected_counter_time = (completed["nvt"] + completed["npt"]) * settings.timestep_fs / 1000.0
        if abs(restart_pointer.get("time_ps", expected_counter_time) - expected_counter_time) > 1e-9:
            raise ValueError("Current restart phase counters do not match restart time")
    summary = {
        "schema_version": "1.0",
        "status": "started",
        "density_preparation_completed": False,
        "equilibrium_certified": False,
        "production_ready": False,
        "response_converged": False,
        "false_until_independent_density_temperature_q_assessment": True,
        "role": "bounded_h0_npt_density_preparation_only",
        "model": model,
        "settings": asdict(settings),
        "force_h_kj_mol_nm": 0.,
        "openmm_version": mm.__version__,
        "required_openmm_version": REQUIRED_OPENMM_VERSION,
        "openmm_version_exact_match": mm.__version__ == REQUIRED_OPENMM_VERSION,
        "barostat": {"class": "MonteCarloBarostat", "pressure_bar": 1.,
                      "temperature_K": 300., "frequency_steps": 25,
                      "random_seed": settings.seed + 2,
                      "force_groups_included_in_accept_reject": "all integration force groups in Context"},
        "platform_option_requests": {"disable_pme_stream": settings.disable_pme_stream},
        "restart_seed": settings.seed,
        "resume_count": int(previous.get("resume_count", 0) + 1) if previous else (1 if restart_pointer else 0),
        "completed_by_phase": dict(completed),
        "target_by_phase": target_steps,
        "precision": {"requested": precision, "effective": None, "source": "Context not yet created", "default_changed": False},
        "platform": None,
        "platform_properties": {},
        "chemical_review": qualification.get("chemical_review"),
        "salt_protonation_gate": production_gate,
        "input_provenance": provenance,
        "synthetic_fixture": bool(synthetic_fixture),
        "chemical_geometry_definition": geometry_payload,
        "chemical_geometry_sha256": geometry_sha,
        "chemical_geometry_failures": [],
        "gauge": gauge_meta,
        "system_xml_sha256": system_xml_sha,
        "system_xml_roundtrip_force_type_counts": _force_type_counts(roundtrip_system),
        "original_particle_count": len(xyz),
        "engine_particle_count": system.getNumParticles(),
        "all_particles_massive": True,
        "constraint_count": system.getNumConstraints(),
        "kinetic_dof": dof,
        "kinetic_dof_rule": "3*nparticles - nnonredundantDistanceConstraints; all particles massive; no COM removal",
        "total_mass_da": total_mass_da,
        "trajectory_frames": 0,
        "diagnostic_rows": 0,
    }
    if restart_pointer:
        if restart_pointer.get("input_hashes_verified", {}) != provenance.get("input_sha256", {}):
            raise ValueError("Current restart input hash mismatch")
        if restart_pointer.get("model") != model or float(restart_pointer.get("force_h_kj_mol_nm", 0.0)) != 0.0:
            raise ValueError("Current restart model or force mismatch")
        if restart_pointer.get("precision", {}).get("requested") != precision:
            raise ValueError("Current restart precision mismatch")
        checkpoint_record = restart_pointer["checkpoint"]
        checkpoint_path = output_dir / checkpoint_record["path"]
        if _file_sha256(checkpoint_path) != checkpoint_record["sha256"]:
            raise ValueError("Current restart checkpoint hash mismatch")
        state_record = restart_pointer["state_xml"]
        state_path = output_dir / state_record["path"]
        if _file_sha256(state_path) != state_record["sha256"]:
            raise ValueError("Current restart StateXML hash mismatch")
        trajectory_record = restart_pointer.get("trajectory_npz")
        if not trajectory_record or _file_sha256(output_dir / trajectory_record["path"]) != trajectory_record["sha256"]:
            raise ValueError("Current restart trajectory hash mismatch")
        diagnostic_record = restart_pointer.get("diagnostics_csv")
        if diagnostic_record and _file_sha256(output_dir / diagnostic_record["path"]) != diagnostic_record["sha256"]:
            raise ValueError("Current restart diagnostics hash mismatch")
        generation_record = restart_pointer.get("generation")
        if generation_record:
            manifest_record = generation_record["manifest"]
            if _file_sha256(output_dir / manifest_record["path"]) != manifest_record["sha256"]:
                raise ValueError("Current restart generation manifest hash mismatch")
        rows[:] = list(csv.DictReader((output_dir / diagnostic_record["path"]).open())) if diagnostic_record else []
        frame_keys.update((row["phase"], int(row["step"])) for row in rows)
        with np.load(output_dir / trajectory_record["path"]) as old:
            frames[:] = [frame.copy() for frame in old["core_displacement_nm"]]
            times[:] = [row.copy().tolist() for row in old["frame_time_phase_step"]]
            if old["last_positions_nm"].size:
                last_xyz = old["last_positions_nm"].copy()
            if old["last_box_vectors_nm"].shape == (3, 3):
                last_box = old["last_box_vectors_nm"].copy()
        expected_frame_sha = [
            {"phase_code": int(meta[0]), "time_ps": float(meta[1]), "step": int(meta[2]),
             "core_displacement_sha256": _sha256_bytes(np.ascontiguousarray(frame).tobytes())}
            for frame, meta in zip(frames, times)
        ]
        if expected_frame_sha != restart_pointer.get("frame_sha256", []):
            raise ValueError("Current restart frame hash mismatch")

    def stage_restart(state):
        checkpoint = context.createCheckpoint()
        state_xml = mm.XmlSerializer.serialize(state)
        checkpoint_sha = _sha256_bytes(checkpoint)
        state_sha = _sha256_bytes(state_xml.encode())
        restart_dir = output_dir / "restarts"
        checkpoint_name = f"checkpoint_{checkpoint_sha}.chk"
        state_name = f"state_{state_sha}.xml"
        checkpoint_path = restart_dir / checkpoint_name
        state_path = restart_dir / state_name
        if not checkpoint_path.exists():
            _atomic_bytes(checkpoint_path, checkpoint)
        if not state_path.exists():
            _atomic_text(state_path, state_xml)
        pointer = {
            "schema_version": "1.0",
            "completed_by_phase": dict(completed),
            "time_ps": state.getTime().value_in_unit(unit.picosecond),
            "checkpoint": {"path": f"restarts/{checkpoint_name}", "sha256": checkpoint_sha, "bytes": len(checkpoint)},
            "state_xml": {"path": f"restarts/{state_name}", "sha256": state_sha, "bytes": len(state_xml.encode())},
            "system_xml_sha256": system_xml_sha,
            "settings_contract": _restart_contract_settings(settings),
            "settings": asdict(settings),
            "input_hashes_verified": provenance.get("input_sha256", {}),
            "model": model,
            "force_h_kj_mol_nm": 0.0,
            "platform": summary.get("platform"),
            "platform_properties": dict(summary.get("platform_properties", {})),
            "precision": dict(summary.get("precision", {})),
            "restart_seed": settings.seed,
        }
        summary["current_restart"] = pointer
        return pointer

    def write_trajectory_and_pointer():
        pointer = dict(summary.get("current_restart") or {})
        if not pointer:
            raise ValueError("Cannot publish a restart generation before checkpoint staging")
        generation_key_payload = json.dumps({
            "checkpoint": pointer["checkpoint"]["sha256"],
            "state_xml": pointer["state_xml"]["sha256"],
            "completed_by_phase": pointer["completed_by_phase"],
            "frames": len(times),
            "frame_hash_seed": [float(meta[1]) for meta in times],
        }, sort_keys=True, separators=(",", ":")).encode()
        generation_name = "generation_" + _sha256_bytes(generation_key_payload)[:24]
        generation_dir = generations_dir / generation_name
        tmp_dir = generations_dir / (generation_name + ".tmp")
        if tmp_dir.exists():
            for child in tmp_dir.iterdir():
                child.unlink()
            tmp_dir.rmdir()
        tmp_dir.mkdir(parents=True)
        generation_csv = tmp_dir / "diagnostics.csv"
        generation_npz = tmp_dir / "density_preparation.npz"
        _write_diagnostics_csv(generation_csv, rows)
        _maybe_test_crash("after_generation_diagnostics")
        with generation_npz.open("wb") as handle:
            np.savez_compressed(handle,
                                core_displacement_nm=np.asarray(frames).reshape(-1, len(core), 3),
                                frame_time_phase_step=np.asarray(times),
                                last_positions_nm=np.asarray(last_xyz),
                                last_box_vectors_nm=np.asarray(last_box) if last_box is not None else np.empty((0, 3)),
                                reference_nm=ref,
                                q_ambient=q,
                                core_indices=np.asarray(core),
                                rigid_basis=B)
        _maybe_test_crash("after_generation_npz")
        frame_sha256 = [
            {"phase_code": int(meta[0]), "time_ps": float(meta[1]), "step": int(meta[2]),
             "core_displacement_sha256": _sha256_bytes(np.ascontiguousarray(frame).tobytes())}
            for frame, meta in zip(frames, times)
        ]
        if not generation_dir.exists():
            tmp_dir.replace(generation_dir)
        else:
            for child in tmp_dir.iterdir():
                child.unlink()
            tmp_dir.rmdir()
        generation_manifest_path = generation_dir / "generation_manifest.json"
        generation_manifest = {
            "schema_version": "1.0",
            "completed_by_phase": dict(completed),
            "checkpoint": pointer["checkpoint"],
            "state_xml": pointer["state_xml"],
            "system_xml_sha256": system_xml_sha,
            "frame_sha256": frame_sha256,
            "diagnostics_csv": _artifact_record(generation_dir / "diagnostics.csv", base=output_dir),
            "trajectory_npz": _artifact_record(generation_dir / "density_preparation.npz", base=output_dir),
        }
        _atomic_json(generation_manifest_path, generation_manifest)
        pointer["generation"] = {"path": str(generation_dir.relative_to(output_dir)),
                                 "manifest": _artifact_record(generation_manifest_path, base=output_dir)}
        pointer["trajectory_npz"] = _artifact_record(generation_dir / "density_preparation.npz", base=output_dir)
        pointer["diagnostics_csv"] = _artifact_record(generation_dir / "diagnostics.csv", base=output_dir)
        pointer["frame_sha256"] = frame_sha256
        pointer["trajectory_frames"] = len(times)
        pointer["diagnostic_rows"] = len(rows)
        summary["frame_sha256"] = frame_sha256
        summary["current_restart"] = pointer
        _maybe_test_crash("before_restart_pointer")
        _atomic_json(restart_pointer_path, pointer)
        _write_diagnostics_csv(diagnostics_path, rows)
        _atomic_npz(trajectory_path,
                    core_displacement_nm=np.asarray(frames).reshape(-1, len(core), 3),
                    frame_time_phase_step=np.asarray(times),
                    last_positions_nm=np.asarray(last_xyz),
                    last_box_vectors_nm=np.asarray(last_box) if last_box is not None else np.empty((0, 3)),
                    reference_nm=ref,
                    q_ambient=q,
                    core_indices=np.asarray(core),
                    rigid_basis=B)


    def sample(phase, stage, store_frame=True):
        nonlocal last_xyz, last_box
        state = context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=False)
        last_xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        last_box = _box_nm(state)
        volume = _volume_nm3(last_box)
        if not np.isfinite(last_xyz).all() or not np.isfinite(volume) or volume <= 0:
            raise FloatingPointError("Nonfinite coordinate or invalid periodic box volume")
        potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        delta = last_xyz[core] - ref
        g = B.T @ delta.ravel()
        relative_constraint_error = 0.
        if len(constraint_data):
            distances = np.linalg.norm(last_xyz[constraint_i] - last_xyz[constraint_j], axis=1)
            relative_constraint_error = float(np.max(np.abs(distances - constraint_lengths) / constraint_lengths))
        step = int(completed[phase])
        row = {
            "phase": phase,
            "stage": stage,
            "step": step,
            "time_ps": state.getTime().value_in_unit(unit.picosecond),
            "elapsed_wall_seconds": time.monotonic() - start,
            "potential_kj_mol": potential,
            "kinetic_kj_mol": kinetic,
            "kinetic_temperature_K": 2 * kinetic / (dof * GAS_CONSTANT),
            "volume_nm3": volume,
            "density_g_ml": total_mass_da / volume * DENSITY_DA_PER_NM3_TO_G_PER_ML,
            "closure_Q_nm": float(q @ delta.ravel()),
            "internal_core_rmsd_nm": float(np.linalg.norm((np.eye(len(q)) - B @ B.T) @ delta.ravel()) / np.sqrt(len(core))),
            "core_rms_displacement_nm": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
            "gauge_norm_nm": float(np.linalg.norm(g)),
            "gauge_energy_kj_mol": float(.5 * settings.gauge_k * (g @ g)),
            "protein_ca_unfitted_rmsd_nm": float(np.sqrt(np.mean(np.sum((last_xyz[ca] - xyz[ca]) ** 2, axis=1)))),
            "protein_ca_fitted_rmsd_nm": pilot._fit_rmsd(last_xyz[ca], xyz[ca]),
            "max_relative_constraint_error": relative_constraint_error,
            "constraint_gate_pass": relative_constraint_error <= max(10 * settings.constraint_tolerance, 1e-5),
        }
        row.update({f"gauge_coordinate_{i}_nm": float(value) for i, value in enumerate(g)})
        numeric_values = [value for value in row.values() if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)]
        if not np.isfinite(numeric_values).all():
            raise FloatingPointError("Nonfinite density diagnostic")
        if "zn_atom_index" in mapping:
            distances = np.linalg.norm(pilot._minimum_image(last_xyz[mapping["zn_sg_indices"]] -
                                                            last_xyz[mapping["zn_atom_index"]], last_box), axis=1)
            bounds = mapping.get("zn_sg_distance_bounds_nm", [.18, .30])
            row["zn_sg_screen_pass"] = bool(np.all((distances >= bounds[0]) & (distances <= bounds[1])))
            row.update({f"zn_sg_{i+1}_distance_nm": float(d) for i, d in enumerate(distances)})
        chemistry = None
        if chemical_geometry is not None:
            chemistry = pilot.chemical_geometry_screen(last_xyz, chemical_geometry)
            row.update({f"chemical_{key}": value for key, value in chemistry.items() if key != "failures"})
            summary["chemical_geometry_last_screen"] = {"phase": phase, "stage": stage, "step": step, **chemistry}
            if not chemistry["pass"]:
                summary["chemical_geometry_failures"].append({"phase": phase, "stage": stage, "step": step, **chemistry})
        new_row = (phase, step) not in frame_keys
        if new_row:
            frame_keys.add((phase, step))
            rows.append(row)
        if store_frame and new_row:
            frames.append(delta.copy())
            times.append([0 if phase == "nvt" else 1, state.getTime().value_in_unit(unit.picosecond), step])
        if not row["constraint_gate_pass"]:
            raise ValueError("Distance constraints failed the density geometry screen")
        if not row.get("zn_sg_screen_pass", True):
            raise ValueError("Zn-SG distances failed the declared physical screen")
        if chemistry is not None and not chemistry["pass"]:
            raise ValueError(f"Chemical geometry screen failed at {phase}:{stage}: {chemistry['failures'][:10]}")
        stage_restart(state)
        write_trajectory_and_pointer()
        return row

    try:
        if deadline is not None and time.monotonic() >= deadline:
            raise _BudgetStop("Wall budget exhausted before Context creation")
        if completed["nvt"] < settings.nvt_steps:
            barostat.setFrequency(0)
        context = mm.Context(system, integrator, platform, properties)
        summary["precision"] = pilot.precision_record(platform, context, precision,
                                                       disable_pme_stream=settings.disable_pme_stream)
        summary["platform"] = platform.getName()
        summary["platform_properties"] = {name: platform.getPropertyValue(context, name)
                                          for name in platform.getPropertyNames()}
        if restart_pointer:
            checkpoint_record = restart_pointer["checkpoint"]
            checkpoint_path = output_dir / checkpoint_record["path"]
            context.loadCheckpoint(checkpoint_path.read_bytes())
            checkpoint_state = context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=False)
            checkpoint_time = checkpoint_state.getTime().value_in_unit(unit.picosecond)
            if abs(checkpoint_time - restart_pointer.get("time_ps", checkpoint_time)) > 1e-9:
                raise ValueError("Current restart time mismatch")
            if abs(context.getParameter("core_gauge_k") - settings.gauge_k) > 1e-12:
                raise ValueError("Current restart gauge mismatch")
            if abs(context.getParameter("core_force_h")) > 1e-12:
                raise ValueError("Current restart force mismatch")
        else:
            context.setPositions(xyz)
            context.computeVirtualSites()
            context.setVelocitiesToTemperature(settings.temperature_K, settings.seed + 1)
            context.applyVelocityConstraints(settings.constraint_tolerance)
            sample("nvt" if settings.nvt_steps else "npt", "initial", False)
        for phase in ("nvt", "npt"):
            if target_steps[phase] == 0:
                continue
            frequency = 0 if phase == "nvt" else settings.barostat_frequency_steps
            if barostat.getFrequency() != frequency:
                barostat.setFrequency(frequency)
                context.reinitialize(preserveState=True)
            if completed[phase] == 0 and not restart_pointer:
                sample(phase, "phase_start", False)
            while completed[phase] < target_steps[phase]:
                if deadline is not None and time.monotonic() >= deadline:
                    raise _BudgetStop("Wall budget reached at integration batch boundary")
                remaining = target_steps[phase] - completed[phase]
                until_report = settings.report_interval_steps - completed[phase] % settings.report_interval_steps
                amount = min(settings.max_step_batch, until_report, remaining)
                integrator.step(amount)
                completed[phase] += amount
                summary["completed_by_phase"] = dict(completed)
                if completed[phase] % settings.report_interval_steps == 0 or completed[phase] == target_steps[phase]:
                    sample(phase, "density")
                    summary["status"] = "running"
                    _atomic_json(summary_path, summary)
        sample("npt", "final", True)
        summary.update(status="density_completed", density_preparation_completed=True,
                       completed_by_phase=dict(completed))
    except _BudgetStop as error:
        summary.update(status="budget_limited", density_preparation_completed=False,
                       completed_by_phase=dict(completed), reason=str(error))
    except Exception as error:
        summary.update(status="failed", density_preparation_completed=False,
                       completed_by_phase=dict(completed), reason=f"{type(error).__name__}: {error}")
    finally:
        if context is not None:
            del context
    summary["elapsed_wall_seconds"] = time.monotonic() - start
    summary["trajectory_frames"] = len(times)
    summary["diagnostic_rows_this_run"] = len(rows)
    artifact_names = ("density_preparation.npz", "diagnostics.csv", "system.xml", "current_restart.json")
    summary["artifact_sha256"] = {name: _artifact_record(output_dir / name)
                                  for name in artifact_names if (output_dir / name).exists()}
    if "current_restart" in summary:
        summary["artifact_sha256"]["checkpoint"] = summary["current_restart"]["checkpoint"]
        summary["artifact_sha256"]["state_xml"] = summary["current_restart"]["state_xml"]
    summary["input_hashes_verified"] = provenance.get("input_sha256", {})
    summary["density_preparation_completed_definition"] = (
        "NVT thermal phase, if requested, and NPT phase completed with h=0, "
        "all particles massive, no CMMotionRemover or Cartesian restraints, "
        "OpenMM MonteCarloBarostat(1 bar, 300 K, 25), finite density/temperature/Q/gauge diagnostics, "
        "constraint, full-topology stereochemistry/peptide, and optional Zn gates. "
        "This is not an equilibrium or production certificate."
    )
    _atomic_json(summary_path, summary)
    _write_final_artifact_manifest(output_dir, summary)
    if summary["density_preparation_completed"]:
        _write_handoff(output_dir, summary, last_xyz, last_box)
        _atomic_json(summary_path, summary)
        _write_final_artifact_manifest(output_dir, summary)
    return summary


def load_qualified_inputs(prmtop, inpcrd, mapping_path, config_path, qualification_path, *,
                          model="flexible", overrides=None):
    loaded = pilot.load_qualified_inputs(prmtop, inpcrd, mapping_path, config_path,
                                         qualification_path, model=model, overrides=None)
    config = _read_json(config_path)
    loaded["settings"] = resolve_density_settings(config, overrides)
    loaded["provenance"] = {**loaded["provenance"],
                            "density_runner_sha256": _file_sha256(ROOT / "scripts" / "run_atomistic_density_preparation.py")}
    return loaded


def run(prmtop, inpcrd, mapping_path, config_path, qualification_path, output_dir, *,
        model="flexible", platform_name="OpenCL", device_index=None, overrides=None, precision="double"):
    started = time.monotonic()
    loaded = load_qualified_inputs(prmtop, inpcrd, mapping_path, config_path,
                                   qualification_path, model=model, overrides=overrides)
    return _run_engine(**loaded, output_dir=output_dir, model=model, platform_name=platform_name,
                       device_index=device_index, started_at=started, precision=precision)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prmtop", "inpcrd", "mapping", "qualification"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/atomistic_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Explicit local-only intent; this runner has no network operations")
    parser.add_argument("--model", choices=DENSITY_MODELS, default="flexible")
    parser.add_argument("--platform", choices=(*pilot.GPU_PLATFORMS, "Reference", "CPU"), default="OpenCL")
    parser.add_argument("--device-index")
    parser.add_argument("--precision", choices=("double", "mixed"), default="double")
    for name, kind in (("nvt-steps", int), ("npt-steps", int), ("seed", int), ("gauge-k", float),
                       ("max-wall-seconds", float), ("report-interval-steps", int), ("max-step-batch", int),
                       ("checkpoint-interval-steps", int)):
        parser.add_argument("--" + name, type=kind)
    parser.add_argument("--disable-pme-stream", action="store_true", default=None,
                        help="Required for non-synthetic OpenCL/CUDA density preparation; default PME stream mode is rejected for production density.")
    parser.add_argument("--no-wall-limit", action="store_true",
                        help="Explicitly run without an internal wall-clock deadline; scientific step targets and restart contracts are unchanged.")
    args = parser.parse_args(argv)
    if args.no_wall_limit and args.max_wall_seconds is not None:
        parser.error("--no-wall-limit cannot be combined with --max-wall-seconds")
    overrides = {name: getattr(args, name) for name in
                 ("nvt_steps", "npt_steps", "seed", "gauge_k",
                  "report_interval_steps", "max_step_batch", "checkpoint_interval_steps",
                  "disable_pme_stream") if getattr(args, name) is not None}
    if args.max_wall_seconds is not None:
        overrides["max_wall_seconds"] = args.max_wall_seconds
    if args.no_wall_limit:
        overrides["max_wall_seconds"] = None
    try:
        result = run(args.prmtop, args.inpcrd, args.mapping, args.config, args.qualification,
                     args.output_dir, model=args.model, platform_name=args.platform,
                     device_index=args.device_index, overrides=overrides, precision=args.precision)
    except Exception as error:
        print(json.dumps({"status": "rejected_before_density_preparation",
                          "reason": f"{type(error).__name__}: {error}"}))
        return 2
    print(json.dumps({key: result[key] for key in
                      ("status", "density_preparation_completed", "equilibrium_certified",
                       "production_ready", "completed_by_phase", "elapsed_wall_seconds")}))
    return 0 if result["density_preparation_completed"] else (3 if result["status"] == "budget_limited" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
