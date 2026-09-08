"""Tiny Reference-platform tests for the bounded atomistic response segment."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

mm = pytest.importorskip("openmm")
from openmm import app

from scripts import directional_mechanics as dm
from scripts import run_atomistic_density_preparation as density
from scripts import run_atomistic_response_segment as segment
from scripts.run_atomistic_response_segment import (
    AdmissionError,
    ResumeMismatch,
    SegmentSettings,
    run_from_qualified,
    run_segment,
)

INPUT_HASHES = {"prmtop": "a" * 64, "inpcrd": "b" * 64, "mapping": "c" * 64}


XYZ = np.array(
    [
        [-0.3, -0.2, -0.1],
        [0.4, -0.1, 0.2],
        [0.05, 0.45, -0.25],
        [-0.1, 0.05, 0.55],
        [1.2, 0.0, 0.0],
    ]
)


def fixture_system():
    system = mm.System()
    force = mm.CustomExternalForce("0")
    for _ in XYZ:
        system.addParticle(12.0)
        force.addParticle(system.getNumParticles() - 1, [])
    system.addForce(force)
    return system


def density_handoff_fixture():
    topology = app.Topology()
    chain = topology.addChain("A")
    core = XYZ[:4] + 1.0
    xyz = [point.copy() for point in core]
    masses = [12.0] * 4
    for _ in range(4):
        residue = topology.addResidue("ALA", chain)
        topology.addAtom("CA", app.element.carbon, residue)
    residue = topology.addResidue("HOH", topology.addChain("W"))
    water_start = len(xyz)
    theta = np.deg2rad(104.52)
    water = np.array([[0, 0, 0], [0.09572, 0, 0], [0.09572 * np.cos(theta), 0.09572 * np.sin(theta), 0]]) + [4.7, 1.0, 1.0]
    for name, element, point, mass in (
        ("O", app.element.oxygen, water[0], 15.999),
        ("H1", app.element.hydrogen, water[1], 1.008),
        ("H2", app.element.hydrogen, water[2], 1.008),
    ):
        topology.addAtom(name, element, residue)
        xyz.append(point)
        masses.append(mass)
    xyz = np.asarray(xyz, dtype=float)
    system = mm.System()
    box = np.eye(3) * 6.0
    system.setDefaultPeriodicBoxVectors(*box)
    topology.setPeriodicBoxVectors(tuple(mm.Vec3(*row) for row in box) * density.unit.nanometer)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(1.0)
    for index, mass in enumerate(masses):
        system.addParticle(mass)
        water_index = index - water_start
        charge = (-0.834, 0.417, 0.417)[water_index] if 0 <= water_index < 3 else 0.0
        nb.addParticle(charge, 0.3, 0.0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = water_start + i, water_start + j
        system.addConstraint(a, b, float(np.linalg.norm(xyz[a] - xyz[b])))
        nb.addException(a, b, 0.0, 0.1, 0.0)
    system.addForce(nb)
    mapping = {
        "schema_version": "1.0",
        "core_indices": list(range(4)),
        "reference_nm": core.tolist(),
        "q": dm.internal_basis(core)[:, 0].tolist(),
        "ddb1_atom_indices": [],
    }
    mapping = density.pilot.validate_mapping(mapping, topology, xyz, "isolated", expected_core_count=4)
    return system, xyz, mapping


def density_complex_handoff_fixture():
    topology = app.Topology()
    chain = topology.addChain("A")
    core = XYZ[:4] + 1.0
    xyz = [point.copy() for point in core]
    masses = [12.0] * 4
    for _ in range(4):
        residue = topology.addResidue("ALA", chain)
        topology.addAtom("CA", app.element.carbon, residue)
    body = []
    residue = topology.addResidue("DDB", topology.addChain("B"))
    for name, point in (
        ("C1", [3.0, 3.1, 3.2]),
        ("C2", [3.4, 3.0, 3.3]),
        ("C3", [3.1, 3.5, 3.4]),
        ("C4", [3.2, 3.2, 3.8]),
    ):
        topology.addAtom(name, app.element.carbon, residue)
        body.append(len(xyz))
        xyz.append(np.asarray(point, dtype=float))
        masses.append(12.0)
    xyz = np.asarray(xyz, dtype=float)
    system = mm.System()
    box = np.eye(3) * 7.0
    system.setDefaultPeriodicBoxVectors(*box)
    topology.setPeriodicBoxVectors(tuple(mm.Vec3(*row) for row in box) * density.unit.nanometer)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(1.0)
    for mass in masses:
        system.addParticle(mass)
        nb.addParticle(0.0, 0.3, 0.0)
    system.addForce(nb)
    mapping = {
        "schema_version": "1.0",
        "core_indices": list(range(4)),
        "reference_nm": core.tolist(),
        "q": dm.internal_basis(core)[:, 0].tolist(),
        "ddb1_atom_indices": body,
        "protein_ca_indices": list(range(4)),
    }
    mapping = density.pilot.validate_mapping(mapping, topology, xyz, "flexible", expected_core_count=4)
    return system, xyz, mapping


def fixture_mapping():
    core = [0, 1, 2, 3]
    q = dm.internal_basis(XYZ[core])[:, 0]
    return {
        "core_indices": core,
        "reference_nm": XYZ[core].tolist(),
        "q": q.tolist(),
        "ddb1_atom_indices": [],
        "protein_ca_indices": core,
    }


def settings(**kwargs):
    values = {"steps": 4, "report_interval_steps": 2, "max_step_batch": 2, "master_seed": 12345}
    values.update(kwargs)
    return SegmentSettings(**values)


def read_current_rows(path):
    manifest = json.loads((path / "segment_manifest.json").read_text())
    with (path / manifest["current_generation"] / "response_observations.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def cert_payload(kind, phase, model, force, binding, **extra):
    payload = {
        "schema_version": "1.0",
        "status": "pass",
        "certificate_kind": kind,
        "phase": phase,
        "model": model,
        "force_kj_mol_nm": float(force),
        "binding": binding,
    }
    payload.update(extra)
    return payload


def binding_from_manifest(manifest, *, include_parent=False):
    keys = list(segment.IDENTITY_BINDING_KEYS)
    binding = {key: manifest[key] for key in keys}
    if include_parent:
        binding["parent_state_sha256"] = manifest["admission"]["parent_state"]["state_xml_sha256"]
    return binding


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_locked_force_plan(path, *, model, force, binding):
    f0 = abs(float(force))
    sigma_max = 0.25 * segment.pilot.GAS_CONSTANT * 300.0 / f0
    variance_rows = []
    q = np.asarray(fixture_mapping()["q"], dtype=float).reshape(-1)
    q_norm2 = float(q @ q)
    reference_nm = np.asarray(fixture_mapping()["reference_nm"], dtype=float)
    core_indices = np.asarray(fixture_mapping()["core_indices"], dtype=int)
    for model_name in ("flexible", "fixed", "rigid"):
        for replicate in range(3):
            replicate_id = f"{model_name}-r{replicate}"
            q_scale = sigma_max if model_name == "rigid" and replicate == 2 else 0.5 * sigma_max
            q_offset = 0.01 * (("flexible", "fixed", "rigid").index(model_name) + 1) * (replicate + 1)
            q_series = np.asarray([-q_scale, 0.0, q_scale], dtype=float) + q_offset
            core_displacement = np.asarray([(q * value / q_norm2).reshape(reference_nm.shape) for value in q_series])
            artifact = path.with_name(f"{path.stem}_{model_name}_r{replicate}_zero_calibration.npz")
            with artifact.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    core_displacement_nm=core_displacement,
                    time_ps=np.asarray([0.0, 1.0, 2.0], dtype=float),
                    reference_nm=reference_nm,
                    q_ambient=q,
                    core_indices=core_indices,
                    force_kj_mol_nm=0.0,
                )
            initialization_lineage = {
                "master_seed": 1000 + replicate,
                "velocity_seed": 2000 + replicate,
                "thermostat_seed": 3000 + replicate,
                "parent_state_sha256": binding.get("parent_state_sha256", "p" * 64),
                "initial_positions_sha256": f"{4000 + replicate:064x}",
                "initial_velocities_sha256": f"{5000 + replicate:064x}",
                "stream": model_name,
            }
            calibration_manifest = write_json(
                path.with_name(f"{path.stem}_{model_name}_r{replicate}_manifest.json"),
                {
                    "schema_version": "1.0",
                    "status": "segment_complete",
                    "phase": "zero_calibration",
                    "model": model_name,
                    "replicate_id": replicate_id,
                    "force_kj_mol_nm": 0.0,
                    "seed_lineage": initialization_lineage,
                    "admission": {"parent_state": {"state_xml_sha256": initialization_lineage["parent_state_sha256"]}},
                    "current_generation": f"generation_{model_name}_{replicate}",
                    "generation_sha256": {"segment_observables.npz": segment._sha256(artifact)},
                    **{key: binding[key] for key in segment.IDENTITY_BINDING_KEYS},
                },
            )
            stationarity = write_json(
                path.with_name(f"{path.stem}_{model_name}_r{replicate}_stationarity.json"),
                {
                    "schema_version": "1.0",
                    "status": "pass",
                    "certificate_kind": "zero_calibration_stationarity_acceptance",
                    "model": model_name,
                    "replicate_id": replicate_id,
                    "artifact_sha256": segment._sha256(artifact),
                    "artifact_selection": {"observable": "q_ambient_dot_core_displacement_nm", "variance_ddof": 1},
                    "binding": {
                        key: binding[key]
                        for key in ("data_input_hashes", "core_indices_sha256", "reference_nm_sha256", "q_ambient_sha256")
                    },
                },
            )
            variance_rows.append(
                {
                    "model": model_name,
                    "replicate_id": replicate_id,
                    "sample_variance_Q_nm2": float(np.var(q_series, ddof=1)),
                    "initialization_lineage": initialization_lineage,
                    "binding": dict(binding),
                    "zero_calibration_artifact": {"path": artifact.name, "sha256": segment._sha256(artifact)},
                    "zero_calibration_manifest": {"path": calibration_manifest.name, "sha256": segment._sha256(calibration_manifest)},
                    "stationarity_certificate": {
                        "path": stationarity.name,
                        "sha256": segment._sha256(stationarity),
                        "certificate_kind": "zero_calibration_stationarity_acceptance",
                    },
                }
            )
    _sigma, calibration_data_sha256 = segment._validate_variance_rows(path.parent, variance_rows, binding)
    source_payload = cert_payload("zero_force_calibration", "zero_calibration", model, 0.0, binding)
    source_payload["zero_calibration_data_sha256"] = calibration_data_sha256
    source = write_json(
        path.with_name(path.stem + "_source_calibration.json"),
        source_payload,
    )
    return write_json(
        path,
        {
            "schema_version": "1.0",
            "status": "locked",
            "model": model,
            "force_grid_kj_mol_nm": [-2 * f0, -f0, 0.0, f0, 2 * f0],
            "force_multipliers": [-2, -1, 0, 1, 2],
            "f0_kj_mol_nm": f0,
            "sigma_max_nm": sigma_max,
            "locked_f0_rule": "0.25RT_over_sigma_max",
            "source_calibration_certificate": {
                "path": source.name,
                "sha256": segment._sha256(source),
                "certificate_kind": "zero_force_calibration",
            },
            "zero_calibration_data_sha256": calibration_data_sha256,
            "accepted_zero_calibration_variance_rows": variance_rows,
            "binding": binding,
        },
    )


def test_zero_segment_restart_resumes_generation_without_duplicate_frames(tmp_path):
    first = run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(),
        output_dir=tmp_path,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        synthetic_fixture=True,
    )
    assert first["segment_complete"]
    assert first["synthetic_fixture"] is True
    assert first["response_converged"] is False
    rows = read_current_rows(tmp_path)
    assert [float(row["time_ps"]) for row in rows] == pytest.approx([0.0, 0.002, 0.004])
    assert first["seed_lineage"]["velocity_seed"] != first["seed_lineage"]["thermostat_seed"]

    resumed = run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(),
        output_dir=tmp_path,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        synthetic_fixture=True,
        resume=True,
    )
    assert resumed["last_committed_step"] == 4
    assert read_current_rows(tmp_path) == rows
    generation = tmp_path / resumed["current_generation"]
    npz = np.load(generation / "segment_observables.npz")
    assert npz["core_displacement_nm"].shape == (3, 4, 3)
    assert npz["force_kj_mol_nm"] == pytest.approx(0.0)
    assert (generation / "segment.chk").exists()
    assert (generation / "segment_state.xml").exists()


def test_resume_rejects_changed_synthetic_fixture_manifest_role(tmp_path):
    run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(), settings=settings(),
        output_dir=tmp_path, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True,
    )
    manifest_path = tmp_path / "segment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["synthetic_fixture"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2))
    with pytest.raises(ResumeMismatch, match="synthetic_fixture"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(), settings=settings(),
            output_dir=tmp_path, phase="zero_equilibration", model="isolated", replicate_id="r1",
            force_h=0.0, platform_name="Reference", synthetic_fixture=True, resume=True,
        )


def test_non_synthetic_segment_manifest_records_false_role(tmp_path, monkeypatch):
    monkeypatch.setattr(segment.pilot, "validate_mapping", lambda mapping, topology, xyz, model: mapping)
    monkeypatch.setattr(segment.pilot, "chemical_geometry_screen", lambda xyz, geometry: {"pass": True, "failures": []})
    result = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=False, topology=app.Topology(),
        qualification={"chemical_review": {"status": segment.pilot.CHEMICAL_REVIEW_PASS}},
        chemical_geometry=object(),
    )
    assert result["synthetic_fixture"] is False


def test_force_energy_matches_boundary_analytic_expression(tmp_path):
    run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(steps=2, report_interval_steps=1, max_step_batch=1, gauge_k=7.0),
        output_dir=tmp_path,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        synthetic_fixture=True,
    )
    q = np.asarray(fixture_mapping()["q"])
    system = mm.XmlSerializer.deserialize((tmp_path / "segment_system.xml").read_text())
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    try:
        displaced = XYZ.copy()
        displacement = np.zeros((4, 3))
        displacement.reshape(-1)[:] = 0.01 * q
        displaced[:4] += displacement
        context.setPositions(displaced)
        context.setParameter("core_force_h", 0.3)
        state = context.getState(getEnergy=True, getForces=True)
        closure = float(q @ displacement.ravel())
        assert state.getPotentialEnergy()._value == pytest.approx(-0.3 * closure, abs=1e-12)
        np.testing.assert_allclose(state.getForces(asNumpy=True)._value[:4].ravel(), 0.3 * q, atol=1e-12)
    finally:
        del context


def test_zero_calibration_requires_bound_equilibrium_certificate(tmp_path):
    with pytest.raises(AdmissionError, match="zero_calibration requires"):
        run_segment(
            system=fixture_system(),
            xyz=XYZ,
            mapping=fixture_mapping(),
            settings=settings(),
            output_dir=tmp_path,
            phase="zero_calibration",
            model="isolated",
            replicate_id="r1",
            force_h=0.0,
            platform_name="Reference",
            synthetic_fixture=True,
        )


def test_parent_state_loaded_into_nonzero_force_segment_with_bound_certificates(tmp_path):
    zero = tmp_path / "zero"
    parent = run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(),
        output_dir=zero,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        synthetic_fixture=True,
    )
    parent_state_hash = segment._sha256(zero / parent["current_generation"] / "segment_state.xml")
    binding = binding_from_manifest(parent)
    binding["parent_state_sha256"] = parent_state_hash
    eq_cert = write_json(tmp_path / "zero_force_eq_cert.json", cert_payload("zero_force_equilibrium", "force_equilibration", "isolated", 0.0, binding))
    locked = write_locked_force_plan(tmp_path / "locked_force.json", model="isolated", force=1.0, binding=binding)
    out = tmp_path / "force_eq"
    result = run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(steps=6),
        output_dir=out,
        phase="force_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=1.0,
        locked_force_plan=locked,
        parent_state=zero,
        equilibration_certificate=eq_cert,
        platform_name="Reference",
        synthetic_fixture=True,
    )
    assert result["phase_start_step"] == 4
    assert result["last_committed_step"] == 10
    assert result["phase_completed_steps"] == 6
    assert result["admission"]["parent_state"]["state_xml_sha256"] == parent_state_hash
    assert result["admission"]["locked_force_plan"]["f0_kj_mol_nm"] == pytest.approx(1.0)
    assert float(read_current_rows(out)[0]["time_ps"]) == pytest.approx(0.004)


def test_force_sampling_requires_matching_force_equilibration_certificate(tmp_path):
    zero_dir = tmp_path / "zero"
    zero = run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
        output_dir=zero_dir,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        synthetic_fixture=True,
    )
    zero_binding = binding_from_manifest(zero)
    zero_binding["parent_state_sha256"] = segment._sha256(zero_dir / zero["current_generation"] / "segment_state.xml")
    eq_cert = write_json(tmp_path / "zero_force_eq_cert.json", cert_payload("zero_force_equilibrium", "force_equilibration", "isolated", 0.0, zero_binding))
    force_plan = write_locked_force_plan(tmp_path / "locked_force.json", model="isolated", force=0.5, binding=zero_binding)
    force_eq_dir = tmp_path / "force_eq"
    force_eq = run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(steps=4, report_interval_steps=1, max_step_batch=1),
        output_dir=force_eq_dir,
        phase="force_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.5,
        locked_force_plan=force_plan,
        parent_state=zero_dir,
        equilibration_certificate=eq_cert,
        platform_name="Reference",
        synthetic_fixture=True,
    )
    sampling_binding = binding_from_manifest(force_eq)
    sampling_binding["parent_state_sha256"] = segment._sha256(force_eq_dir / force_eq["current_generation"] / "segment_state.xml")
    sampling_plan = write_locked_force_plan(tmp_path / "locked_sampling_force.json", model="isolated", force=0.5, binding=sampling_binding)
    with pytest.raises(AdmissionError, match="force_sampling requires"):
        run_segment(
            system=fixture_system(),
            xyz=XYZ,
            mapping=fixture_mapping(),
            settings=settings(steps=6, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path / "out",
            phase="force_sampling",
            model="isolated",
            replicate_id="r1",
            force_h=0.5,
            locked_force_plan=sampling_plan,
            parent_state=force_eq_dir,
            platform_name="Reference",
            synthetic_fixture=True,
        )


def test_locked_force_plan_requires_hash_bound_source_calibration_and_exact_f0_grid(tmp_path):
    zero_dir = tmp_path / "zero"
    zero = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
        output_dir=zero_dir, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True,
    )
    binding = binding_from_manifest(zero)
    binding["parent_state_sha256"] = segment._sha256(zero_dir / zero["current_generation"] / "segment_state.xml")
    eq_cert = write_json(tmp_path / "zero_force_eq_cert.json", cert_payload("zero_force_equilibrium", "force_equilibration", "isolated", 0.0, binding))
    bad_plan = write_json(
        tmp_path / "bad_locked_force.json",
        {
            "schema_version": "1.0",
            "status": "locked",
            "model": "isolated",
            "force_grid_kj_mol_nm": [0.5],
            "f0_kj_mol_nm": 0.5,
            "sigma_max_nm": 0.25 * segment.pilot.GAS_CONSTANT * 300.0 / 0.5,
            "force_multipliers": [1],
            "locked_f0_rule": "0.25RT_over_sigma_max",
            "source_certificate_sha256": "b" * 64,
            "binding": binding,
        },
    )
    with pytest.raises(AdmissionError, match="include zero|exactly"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
            settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path / "force_eq", phase="force_equilibration", model="isolated",
            replicate_id="r1", force_h=0.5, locked_force_plan=bad_plan, parent_state=zero_dir,
            equilibration_certificate=eq_cert, platform_name="Reference", synthetic_fixture=True,
        )


def test_locked_force_plan_derives_variance_from_engine_npz_and_distinct_seed_lineages(tmp_path):
    zero_dir = tmp_path / "zero"
    zero = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
        output_dir=zero_dir, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True,
    )
    binding = binding_from_manifest(zero)
    binding["parent_state_sha256"] = segment._sha256(zero_dir / zero["current_generation"] / "segment_state.xml")
    eq_cert = write_json(tmp_path / "zero_force_eq_cert.json", cert_payload("zero_force_equilibrium", "force_equilibration", "isolated", 0.0, binding))
    good_plan = write_locked_force_plan(tmp_path / "locked_force.json", model="isolated", force=0.5, binding=binding)

    tampered_variance = json.loads(good_plan.read_text())
    for row in tampered_variance["accepted_zero_calibration_variance_rows"]:
        row["sample_variance_Q_nm2"] *= 4.0
    tampered_variance["sigma_max_nm"] *= 2.0
    tampered_variance["f0_kj_mol_nm"] *= 0.5
    tampered_variance["force_grid_kj_mol_nm"] = [-0.5, -0.25, 0.0, 0.25, 0.5]
    bad_variance = write_json(tmp_path / "bad_variance_force.json", tampered_variance)
    with pytest.raises(AdmissionError, match="ddof=1 variance"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
            settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path / "bad_variance", phase="force_equilibration", model="isolated",
            replicate_id="r1", force_h=0.5, locked_force_plan=bad_variance, parent_state=zero_dir,
            equilibration_certificate=eq_cert, platform_name="Reference", synthetic_fixture=True,
        )

    same_effective_lineage = json.loads(good_plan.read_text())
    for model_name in ("flexible", "fixed", "rigid"):
        rows = [row for row in same_effective_lineage["accepted_zero_calibration_variance_rows"] if row["model"] == model_name]
        effective = {
            "velocity_seed": 2222,
            "thermostat_seed": 3333,
            "parent_state_sha256": binding["parent_state_sha256"],
            "initial_positions_sha256": "a" * 64,
            "initial_velocities_sha256": "b" * 64,
        }
        for offset, row in enumerate(rows):
            row["initialization_lineage"] = {"master_seed": 1000 + offset, "stream": model_name, **effective}
            manifest_record = row["zero_calibration_manifest"]
            manifest_path = tmp_path / manifest_record["path"]
            manifest = json.loads(manifest_path.read_text())
            manifest["seed_lineage"] = row["initialization_lineage"]
            manifest_path.write_text(json.dumps(manifest, indent=2))
            manifest_record["sha256"] = segment._sha256(manifest_path)
    bad_lineage = write_json(tmp_path / "bad_lineage_force.json", same_effective_lineage)
    with pytest.raises(AdmissionError, match="effective initialization lineages"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
            settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path / "bad_lineage", phase="force_equilibration", model="isolated",
            replicate_id="r1", force_h=0.5, locked_force_plan=bad_lineage, parent_state=zero_dir,
            equilibration_certificate=eq_cert, platform_name="Reference", synthetic_fixture=True,
        )

    fresh_plan = write_locked_force_plan(tmp_path / "fresh_locked_force.json", model="isolated", force=0.5, binding=binding)
    bad_aggregate = json.loads(fresh_plan.read_text())
    bad_aggregate["zero_calibration_data_sha256"] = "d" * 64
    bad_aggregate_plan = write_json(tmp_path / "bad_aggregate_force.json", bad_aggregate)
    with pytest.raises(AdmissionError, match="zero_calibration_data_sha256"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
            settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path / "bad_aggregate", phase="force_equilibration", model="isolated",
            replicate_id="r1", force_h=0.5, locked_force_plan=bad_aggregate_plan, parent_state=zero_dir,
            equilibration_certificate=eq_cert, platform_name="Reference", synthetic_fixture=True,
        )


def test_calibration_rows_bind_their_own_generation_data_hashes_not_current_segment_hashes(tmp_path):
    zero_dir = tmp_path / "zero"
    zero = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
        output_dir=zero_dir, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True,
    )
    binding = binding_from_manifest(zero)
    binding["parent_state_sha256"] = segment._sha256(zero_dir / zero["current_generation"] / "segment_state.xml")
    eq_cert = write_json(tmp_path / "zero_force_eq_cert.json", cert_payload("zero_force_equilibrium", "force_equilibration", "isolated", 0.0, binding))
    plan_path = write_locked_force_plan(tmp_path / "locked_force.json", model="isolated", force=0.5, binding=binding)
    plan = json.loads(plan_path.read_text())
    for index, row in enumerate(plan["accepted_zero_calibration_variance_rows"]):
        endpoint_hashes = {"prmtop": f"{index:064x}", "inpcrd": f"{index + 100:064x}", "mapping": f"{index + 200:064x}"}
        row["binding"]["data_input_hashes"] = endpoint_hashes
        manifest_record = row["zero_calibration_manifest"]
        manifest_path = tmp_path / manifest_record["path"]
        manifest = json.loads(manifest_path.read_text())
        manifest["data_input_hashes"] = endpoint_hashes
        manifest["input_hashes"] = endpoint_hashes
        manifest_path.write_text(json.dumps(manifest, indent=2))
        manifest_record["sha256"] = segment._sha256(manifest_path)
        cert_record = row["stationarity_certificate"]
        cert_path = tmp_path / cert_record["path"]
        cert = json.loads(cert_path.read_text())
        cert["binding"]["data_input_hashes"] = endpoint_hashes
        cert_path.write_text(json.dumps(cert, indent=2))
        cert_record["sha256"] = segment._sha256(cert_path)
    _sigma, calibration_data_sha256 = segment._validate_variance_rows(tmp_path, plan["accepted_zero_calibration_variance_rows"], binding)
    plan["zero_calibration_data_sha256"] = calibration_data_sha256
    source_record = plan["source_calibration_certificate"]
    source_path = tmp_path / source_record["path"]
    source = json.loads(source_path.read_text())
    source["zero_calibration_data_sha256"] = calibration_data_sha256
    source_path.write_text(json.dumps(source, indent=2))
    source_record["sha256"] = segment._sha256(source_path)
    row_bound_plan = write_json(tmp_path / "row_bound_force.json", plan)
    result = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path / "row_bound", phase="force_equilibration", model="isolated",
        replicate_id="r1", force_h=0.5, locked_force_plan=row_bound_plan, parent_state=zero_dir,
        equilibration_certificate=eq_cert, platform_name="Reference", synthetic_fixture=True,
    )
    assert result["status"] == "segment_complete"
    assert result["admission"]["locked_force_plan"]["f0_kj_mol_nm"] == pytest.approx(0.5)


def test_production_isolated_nonzero_force_requires_q_transport_bridge_validation(tmp_path):
    base = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(), settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path / "base", phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True, input_hashes=INPUT_HASHES,
    )
    with pytest.raises(AdmissionError, match="proper-rigid q_transport bridge"):
        segment._validate_admission(
            phase="force_equilibration",
            model="isolated",
            force_h=0.5,
            identity={key: base[key] for key in segment.IDENTITY_BINDING_KEYS},
            gauge_meta=base["gauge"],
            body_meta=base["body"],
            settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
            synthetic_fixture=False,
            locked_force_plan=None,
            parent_state=None,
            equilibration_certificate=None,
        )


def test_non_synthetic_gpu_requires_disable_pme_stream_before_context(tmp_path, monkeypatch):
    monkeypatch.setattr(segment.pilot, "validate_mapping", lambda mapping, topology, xyz, model: mapping)
    with pytest.raises(AdmissionError, match="disable-pme-stream"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
            settings=settings(steps=1, report_interval_steps=1, max_step_batch=1, master_seed=55),
            output_dir=tmp_path, phase="zero_equilibration", model="isolated", replicate_id="r1",
            force_h=0.0, platform_name="OpenCL", synthetic_fixture=False, topology=app.Topology(),
            qualification={"chemical_review": {"status": segment.pilot.CHEMICAL_REVIEW_PASS}},
            chemical_geometry=object(),
        )


def test_actual_density_runner_handoff_seeds_zero_equilibration_without_scientific_certificate(tmp_path):
    system, xyz, mapping = density_handoff_fixture()
    density_dir = tmp_path / "density"
    density_result = density._run_engine(
        system,
        xyz,
        mapping,
        density.DensitySettings(nvt_steps=1, npt_steps=1, report_interval_steps=1, checkpoint_interval_steps=1, max_wall_seconds=60.0),
        density_dir,
        model="isolated",
        platform_name="Reference",
        synthetic_fixture=True,
        qualification={"chemical_review": {"status": "pending"}},
        provenance={"input_sha256": INPUT_HASHES},
    )
    assert density_result["status"] == "density_completed", density_result.get("reason")
    assert density_result["equilibrium_certified"] is False
    assert density_result["production_ready"] is False

    parent = segment._parent_record(density_dir)
    assert parent["admission_role"] == "preparation_provenance_only_no_equilibrium_claim"

    out = tmp_path / "segment"
    result = run_segment(
        system=system,
        xyz=xyz,
        mapping=mapping,
        settings=settings(),
        output_dir=out,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r-density",
        force_h=0.0,
        parent_state=density_dir,
        platform_name="Reference",
        synthetic_fixture=True,
        input_hashes=INPUT_HASHES,
    )
    assert result["status"] == "segment_complete"
    assert result["admission"]["parent_state"]["kind"] == "density_handoff"
    assert result["admission"]["parent_state"]["accepted_as"] == "zero_equilibration_starting_coordinates_and_box_only"
    assert result["admission"]["parent_state"]["scientific_equilibrium_claim"] is False
    assert "source_certificate" not in result["admission"]
    assert result["production_ready"] is False
    assert result["response_converged"] is False
    rows = read_current_rows(out)
    assert [float(row["time_ps"]) for row in rows] == pytest.approx([0.0, 0.002, 0.004])
    with np.load(density_dir / "npt_to_nvt_handoff.npz") as handoff, np.load(out / result["current_generation"] / "segment_observables.npz") as segment_npz:
        np.testing.assert_allclose(segment_npz["core_displacement_nm"][0], handoff["positions_nm"][mapping["core_indices"]] - np.asarray(mapping["reference_nm"]), atol=1e-12)
        state = mm.XmlSerializer.deserialize((out / result["current_generation"] / "segment_state.xml").read_text())
        np.testing.assert_allclose(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(segment.unit.nanometer), handoff["box_vectors_nm"], atol=1e-12)

    cli_like = run_segment(
        system=system,
        xyz=xyz,
        mapping=mapping,
        settings=settings(),
        output_dir=tmp_path / "segment_cli_like",
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r-density-cli",
        force_h=0.0,
        parent_state=density_dir,
        platform_name="Reference",
        synthetic_fixture=True,
        input_hashes={**INPUT_HASHES, "qualification": "d" * 64, "config": "e" * 64, "source_provenance_sha256": "f" * 64},
    )
    assert cli_like["admission"]["parent_state"]["accepted_as"] == "zero_equilibration_starting_coordinates_and_box_only"
    assert cli_like["data_input_hashes"] == INPUT_HASHES
    assert set(cli_like["stage_provenance_hashes"]) == {"qualification", "config", "source_provenance_sha256"}


def test_density_handoff_cannot_seed_zero_calibration_without_equilibrium_certificate(tmp_path):
    system, xyz, mapping = density_handoff_fixture()
    density_dir = tmp_path / "density"
    density._run_engine(
        system, xyz, mapping,
        density.DensitySettings(npt_steps=1, report_interval_steps=1, checkpoint_interval_steps=1, max_wall_seconds=60.0),
        density_dir, model="isolated", platform_name="Reference", synthetic_fixture=True,
        qualification={"chemical_review": {"status": "pending"}}, provenance={"input_sha256": {}},
    )
    with pytest.raises(AdmissionError, match="only seed zero_equilibration"):
        run_segment(
            system=system, xyz=xyz, mapping=mapping, settings=settings(), output_dir=tmp_path / "segment",
            phase="zero_calibration", model="isolated", replicate_id="r1", force_h=0.0,
            parent_state=density_dir, platform_name="Reference", synthetic_fixture=True,
        )


def test_parent_segment_rejects_q_negation_model_relabel_and_wrong_input_identity(tmp_path):
    parent_dir = tmp_path / "parent"
    parent = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(), settings=settings(),
        output_dir=parent_dir, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True, input_hashes=INPUT_HASHES,
    )
    negated = dict(fixture_mapping())
    negated["q"] = (-np.asarray(negated["q"])).tolist()
    with pytest.raises(AdmissionError, match="Parent segment identity mismatch: .*q_ambient_sha256"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=negated, settings=settings(),
            output_dir=tmp_path / "q_negated", phase="zero_equilibration", model="isolated",
            replicate_id="r1", force_h=0.0, parent_state=parent_dir, platform_name="Reference",
            synthetic_fixture=True, input_hashes=INPUT_HASHES,
        )
    with pytest.raises(AdmissionError, match="data_input_hashes"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(), settings=settings(),
            output_dir=tmp_path / "wrong_input", phase="zero_equilibration", model="isolated",
            replicate_id="r1", force_h=0.0, parent_state=parent_dir, platform_name="Reference",
            synthetic_fixture=True, input_hashes={**INPUT_HASHES, "mapping": "d" * 64},
        )

    fixed_xyz = np.vstack([XYZ, [[1.0, 0.0, 0.0]]])
    fixed_system = mm.System()
    force = mm.CustomExternalForce("0")
    for _ in fixed_xyz:
        fixed_system.addParticle(12.0)
        force.addParticle(fixed_system.getNumParticles() - 1, [])
    fixed_system.addForce(force)
    fixed_mapping = {
        "core_indices": [0, 1, 2, 3],
        "reference_nm": fixed_xyz[:4].tolist(),
        "q": dm.internal_basis(fixed_xyz[:4])[:, 0].tolist(),
        "ddb1_atom_indices": [4],
        "protein_ca_indices": [0, 1, 2, 3],
    }
    fixed_parent_dir = tmp_path / "fixed_parent"
    run_segment(
        system=fixed_system, xyz=fixed_xyz, mapping=fixed_mapping,
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=fixed_parent_dir, phase="zero_equilibration", model="fixed",
        replicate_id="r1", force_h=0.0, platform_name="Reference", synthetic_fixture=True,
    )
    with pytest.raises(AdmissionError, match="model does not match"):
        run_segment(
            system=fixed_system, xyz=fixed_xyz, mapping=fixed_mapping,
            settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path / "relabel_flexible", phase="zero_equilibration", model="flexible",
            replicate_id="r1", force_h=0.0, parent_state=fixed_parent_dir,
            platform_name="Reference", synthetic_fixture=True,
        )
    assert parent["input_hashes"] == INPUT_HASHES


def test_parent_segment_steps_are_new_phase_steps_and_resume_keeps_target(tmp_path):
    parent_dir = tmp_path / "parent"
    run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(), settings=settings(steps=4),
        output_dir=parent_dir, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True, input_hashes=INPUT_HASHES,
    )
    child_dir = tmp_path / "child"
    complete = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=4, report_interval_steps=2, max_step_batch=2, max_wall_seconds=60.0),
        output_dir=child_dir, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, parent_state=parent_dir, platform_name="Reference", synthetic_fixture=True,
        input_hashes=INPUT_HASHES,
    )
    assert complete["phase_start_step"] == 4
    assert complete["target_global_step"] == 8
    assert complete["phase_completed_steps"] == 4

    step6_generation = next(path for path in child_dir.iterdir() if path.name.startswith("generation_000000000006_"))
    manifest_path = child_dir / "segment_manifest.json"
    partial = json.loads(manifest_path.read_text())
    partial.update(
        {
            "status": "budget_limited",
            "segment_complete": False,
            "current_generation": step6_generation.name,
            "last_committed_step": 6,
            "last_committed_time_ps": 0.006,
            "phase_completed_steps": 2,
            "generation_sha256": {
                name: segment._sha256(step6_generation / name)
                for name in segment.GENERATION_ARTIFACTS
            },
        }
    )
    manifest_path.write_text(json.dumps(partial, indent=2))

    resumed = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=4, report_interval_steps=2, max_step_batch=2, max_wall_seconds=60.0),
        output_dir=child_dir, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, parent_state=parent_dir, platform_name="Reference", synthetic_fixture=True, input_hashes=INPUT_HASHES,
        resume=True,
    )
    assert resumed["status"] == "segment_complete"
    assert resumed["phase_start_step"] == 4
    assert resumed["target_global_step"] == 8
    assert resumed["last_committed_step"] == 8
    assert resumed["phase_completed_steps"] == 4


def test_density_handoff_rejects_changed_input_or_q_identity(tmp_path):
    system, xyz, mapping = density_handoff_fixture()
    density_dir = tmp_path / "density"
    density._run_engine(
        system, xyz, mapping,
        density.DensitySettings(npt_steps=1, report_interval_steps=1, checkpoint_interval_steps=1, max_wall_seconds=60.0),
        density_dir, model="isolated", platform_name="Reference", synthetic_fixture=True,
        qualification={"chemical_review": {"status": "pending"}}, provenance={"input_sha256": INPUT_HASHES},
    )
    with pytest.raises(AdmissionError, match="Density handoff identity mismatch: data_input_hashes"):
        run_segment(
            system=system, xyz=xyz, mapping=mapping, settings=settings(), output_dir=tmp_path / "bad_input",
            phase="zero_equilibration", model="isolated", replicate_id="r1", force_h=0.0,
            parent_state=density_dir, platform_name="Reference", synthetic_fixture=True,
            input_hashes={**INPUT_HASHES, "mapping": "d" * 64},
        )
    negated = dict(mapping)
    negated["q"] = (-np.asarray(mapping["q"])).tolist()
    with pytest.raises(AdmissionError, match="Density handoff identity mismatch: q_ambient_sha256"):
        run_segment(
            system=system, xyz=xyz, mapping=negated, settings=settings(), output_dir=tmp_path / "bad_q",
            phase="zero_equilibration", model="isolated", replicate_id="r1", force_h=0.0,
            parent_state=density_dir, platform_name="Reference", synthetic_fixture=True,
            input_hashes=INPUT_HASHES,
        )


def test_rigid_density_handoff_builds_anchors_from_actual_parent_coordinates(tmp_path):
    system, xyz, mapping = density_complex_handoff_fixture()
    density_dir = tmp_path / "density"
    density._run_engine(
        system, xyz, mapping,
        density.DensitySettings(npt_steps=1, report_interval_steps=1, checkpoint_interval_steps=1, max_wall_seconds=60.0),
        density_dir, model="flexible", platform_name="Reference", synthetic_fixture=True,
        qualification={"chemical_review": {"status": "pending"}}, provenance={"input_sha256": INPUT_HASHES},
    )
    out = tmp_path / "rigid_segment"
    result = run_segment(
        system=system, xyz=xyz, mapping=mapping,
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=out, phase="zero_equilibration", model="rigid", replicate_id="r1",
        force_h=0.0, parent_state=density_dir, platform_name="Reference", synthetic_fixture=True,
        input_hashes=INPUT_HASHES,
    )
    with np.load(density_dir / "npt_to_nvt_handoff.npz") as handoff:
        parent_positions = handoff["positions_nm"]
    system_from_xml = mm.XmlSerializer.deserialize((out / "segment_system.xml").read_text())
    assert system_from_xml.getNumParticles() == len(xyz) + 4
    np.testing.assert_allclose(
        result["body"]["center_of_mass_nm"],
        np.mean(parent_positions[mapping["ddb1_atom_indices"]], axis=0),
        atol=1e-12,
    )
    state = mm.XmlSerializer.deserialize((out / result["current_generation"] / "segment_state.xml").read_text())
    assert len(state.getPositions()) == len(xyz) + 4
    rigid_child = run_segment(
        system=system, xyz=xyz, mapping=mapping,
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path / "rigid_child", phase="zero_equilibration", model="rigid",
        replicate_id="r1", force_h=0.0, parent_state=out, platform_name="Reference",
        synthetic_fixture=True, input_hashes=INPUT_HASHES,
    )
    assert rigid_child["status"] == "segment_complete"
    assert rigid_child["phase_start_step"] == result["last_committed_step"]

    fixed_out = tmp_path / "fixed_segment"
    fixed = run_segment(
        system=system, xyz=xyz, mapping=mapping,
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=fixed_out, phase="zero_equilibration", model="fixed", replicate_id="r1",
        force_h=0.0, parent_state=density_dir, platform_name="Reference", synthetic_fixture=True,
        input_hashes=INPUT_HASHES,
    )
    fixed_child = run_segment(
        system=system, xyz=xyz, mapping=mapping,
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path / "fixed_child", phase="zero_equilibration", model="fixed",
        replicate_id="r1", force_h=0.0, parent_state=fixed_out, platform_name="Reference",
        synthetic_fixture=True, input_hashes=INPUT_HASHES,
    )
    assert fixed_child["status"] == "segment_complete"
    assert fixed_child["phase_start_step"] == fixed["last_committed_step"]


def test_rigid_ddb1_runtime_diagnostics_include_geometry_temperature_and_gauge(tmp_path):
    xyz = np.array([
        [-0.3, -0.2, -0.1], [0.4, -0.1, 0.2], [0.05, 0.45, -0.25], [-0.1, 0.05, 0.55],
        [1.0, 0.0, 0.0], [1.4, 0.2, 0.1], [1.1, 0.7, 0.2], [1.2, 0.1, 0.8],
    ])
    system = mm.System()
    force = mm.CustomExternalForce("0")
    for _ in xyz:
        system.addParticle(12.0)
        force.addParticle(system.getNumParticles() - 1, [])
    system.addForce(force)
    core = [0, 1, 2, 3]
    mapping = {
        "core_indices": core,
        "reference_nm": xyz[core].tolist(),
        "q": dm.internal_basis(xyz[core])[:, 0].tolist(),
        "ddb1_atom_indices": [4, 5, 6, 7],
        "protein_ca_indices": core,
    }
    result = run_segment(
        system=system, xyz=xyz, mapping=mapping, settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path, phase="zero_equilibration", model="rigid", replicate_id="r1", force_h=0.0,
        platform_name="Reference", synthetic_fixture=True,
    )
    final_diag = result["runtime_diagnostics"][-1]
    assert final_diag["rigid_ddb1_geometry_pass"] is True
    assert final_diag["rigid_ddb1_max_allatom_fit_residual_nm"] <= 1e-5
    assert final_diag["rigid_ddb1_geometry_algorithm"].startswith("O(Nbody)")
    assert np.isfinite(final_diag["kinetic_temperature_K"])
    assert final_diag["kinetic_dof"] == result["kinetic_dof"]
    assert np.isfinite(final_diag["core_gauge_norm_nm"])
    assert np.isfinite(final_diag["core_internal_rmsd_nm"])


def test_density_handoff_tampered_coordinates_rejected_by_bound_hash(tmp_path):
    system, xyz, mapping = density_handoff_fixture()
    density_dir = tmp_path / "density"
    density._run_engine(
        system, xyz, mapping,
        density.DensitySettings(npt_steps=1, report_interval_steps=1, checkpoint_interval_steps=1, max_wall_seconds=60.0),
        density_dir, model="isolated", platform_name="Reference", synthetic_fixture=True,
        qualification={"chemical_review": {"status": "pending"}}, provenance={"input_sha256": {}},
    )
    with np.load(density_dir / "npt_to_nvt_handoff.npz") as old:
        positions = old["positions_nm"].copy()
        box = old["box_vectors_nm"].copy()
    positions[0, 0] += 0.01
    with (density_dir / "npt_to_nvt_handoff.npz").open("wb") as handle:
        np.savez_compressed(handle, positions_nm=positions, box_vectors_nm=box)
    with pytest.raises(AdmissionError, match="NPZ hash mismatch"):
        run_segment(
            system=system, xyz=xyz, mapping=mapping, settings=settings(), output_dir=tmp_path / "segment",
            phase="zero_equilibration", model="isolated", replicate_id="r1", force_h=0.0,
            parent_state=density_dir, platform_name="Reference", synthetic_fixture=True,
        )


def test_real_segment_rejects_without_topology_or_full_chemistry(tmp_path):
    with pytest.raises(ValueError, match="topology"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(), settings=settings(),
            output_dir=tmp_path / "no_topology", phase="zero_equilibration", model="isolated",
            replicate_id="r1", force_h=0.0, platform_name="Reference", synthetic_fixture=False,
            qualification={"chemical_review": {"status": segment.pilot.CHEMICAL_REVIEW_PASS}},
        )


def test_disable_pme_stream_plumbed_to_pilot_and_mixed_rejected(tmp_path, monkeypatch):
    calls = {}

    def fake_platform_options(platform_name, precision, device_index, *, disable_pme_stream=False):
        calls["platform"] = (platform_name, precision, device_index, disable_pme_stream)
        return mm.Platform.getPlatformByName("Reference"), {}

    def fake_precision_record(platform, context, requested, *, disable_pme_stream=False):
        calls["precision"] = (platform.getName(), requested, disable_pme_stream)
        return {"requested": requested, "effective": "unreported_platform_default", "pme_stream": {"disable_requested": disable_pme_stream}}

    monkeypatch.setattr(segment.pilot, "platform_options", fake_platform_options)
    monkeypatch.setattr(segment.pilot, "precision_record", fake_precision_record)
    result = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1, disable_pme_stream=True),
        output_dir=tmp_path / "double", phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="OpenCL", precision="double", synthetic_fixture=True,
    )
    assert calls["platform"] == ("OpenCL", "double", None, True)
    assert calls["precision"] == ("Reference", "double", True)
    assert result["precision"]["pme_stream"]["disable_requested"] is True
    with pytest.raises(ValueError, match="double precision"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
            settings=settings(steps=1, report_interval_steps=1, max_step_batch=1, disable_pme_stream=True),
            output_dir=tmp_path / "mixed", phase="zero_equilibration", model="isolated", replicate_id="r1",
            force_h=0.0, platform_name="OpenCL", precision="mixed", synthetic_fixture=True,
        )


def test_resume_rejects_manifest_and_generation_tampering(tmp_path):
    run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        synthetic_fixture=True,
    )
    manifest_path = tmp_path / "segment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["force_kj_mol_nm"] = 1.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ResumeMismatch, match="manifest mismatch"):
        run_segment(
            system=fixture_system(),
            xyz=XYZ,
            mapping=fixture_mapping(),
            settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path,
            phase="zero_equilibration",
            model="isolated",
            replicate_id="r1",
            force_h=0.0,
            platform_name="Reference",
            synthetic_fixture=True,
            resume=True,
        )

    manifest["force_kj_mol_nm"] = 0.0
    manifest_path.write_text(json.dumps(manifest))
    generation = tmp_path / manifest["current_generation"]
    (generation / "response_observations.csv").write_text("tampered\n")
    with pytest.raises(ResumeMismatch, match="Generation artifact hash"):
        run_segment(
            system=fixture_system(),
            xyz=XYZ,
            mapping=fixture_mapping(),
            settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path,
            phase="zero_equilibration",
            model="isolated",
            replicate_id="r1",
            force_h=0.0,
            platform_name="Reference",
            synthetic_fixture=True,
            resume=True,
        )


def test_resume_rejects_changed_response_runner_source_provenance(tmp_path, monkeypatch):
    run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True,
    )
    real_sha256 = segment._sha256
    response_source = segment.ROOT / "scripts" / "run_atomistic_response_segment.py"

    def fake_sha256(path):
        if Path(path).resolve() == response_source.resolve():
            return "f" * 64
        return real_sha256(path)

    monkeypatch.setattr(segment, "_sha256", fake_sha256)
    with pytest.raises(ResumeMismatch, match="source_provenance"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
            settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path, phase="zero_equilibration", model="isolated", replicate_id="r1",
            force_h=0.0, platform_name="Reference", synthetic_fixture=True, resume=True,
        )


def test_wall_budget_preserves_last_valid_generation(tmp_path, monkeypatch):
    ticks = iter([0.0, 1.0, 1.0])
    monkeypatch.setattr(segment.time, "monotonic", lambda: next(ticks, 1.0))
    result = run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(steps=4, report_interval_steps=2, max_step_batch=2, max_wall_seconds=0.5),
        output_dir=tmp_path,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        synthetic_fixture=True,
    )
    assert result["status"] == "budget_limited"
    assert result["last_committed_step"] == 0
    assert (tmp_path / result["current_generation"] / "segment.chk").exists()


def test_no_wall_limit_completes_without_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(segment.time, "monotonic", lambda: 1_000_000.0)
    result = run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(steps=4, report_interval_steps=2, max_step_batch=2, max_wall_seconds=None),
        output_dir=tmp_path,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        synthetic_fixture=True,
    )
    assert result["status"] == "segment_complete"
    assert result["settings"]["max_wall_seconds"] is None
    assert result["last_committed_step"] == 4


def test_resume_rejects_changed_wall_limit_mode(tmp_path):
    run_segment(
        system=fixture_system(),
        xyz=XYZ,
        mapping=fixture_mapping(),
        settings=settings(steps=4, report_interval_steps=2, max_step_batch=2, max_wall_seconds=None),
        output_dir=tmp_path,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        synthetic_fixture=True,
    )
    with pytest.raises(ResumeMismatch, match="settings"):
        run_segment(
            system=fixture_system(),
            xyz=XYZ,
            mapping=fixture_mapping(),
            settings=settings(steps=4, report_interval_steps=2, max_step_batch=2, max_wall_seconds=60.0),
            output_dir=tmp_path,
            phase="zero_equilibration",
            model="isolated",
            replicate_id="r1",
            force_h=0.0,
            platform_name="Reference",
            synthetic_fixture=True,
            resume=True,
        )


def test_cli_no_wall_limit_sets_null_override_and_rejects_conflict(tmp_path, monkeypatch):
    captured = {}

    def fake_run_from_qualified(**kwargs):
        captured.update(kwargs)
        return {
            "status": "segment_complete",
            "segment_complete": True,
            "response_converged": False,
            "last_committed_step": 4,
        }

    monkeypatch.setattr(segment, "run_from_qualified", fake_run_from_qualified)
    argv = [
        "--prmtop", str(tmp_path / "p.prmtop"),
        "--inpcrd", str(tmp_path / "x.inpcrd"),
        "--mapping", str(tmp_path / "mapping.json"),
        "--qualification", str(tmp_path / "qualification.json"),
        "--output-dir", str(tmp_path / "out"),
        "--phase", "zero_equilibration",
        "--model", "isolated",
        "--replicate-id", "r1",
        "--force-kj-mol-nm", "0",
        "--platform", "Reference",
        "--master-seed", "12345",
        "--no-wall-limit",
    ]
    assert segment.main(argv) == 0
    assert captured["overrides"]["max_wall_seconds"] is None

    with pytest.raises(SystemExit):
        segment.main(argv + ["--max-wall-seconds", "60"])


def test_run_from_qualified_uses_loader_interface_and_segment_settings(tmp_path, monkeypatch):
    calls = {}
    for name in ("prmtop", "inpcrd", "mapping", "qualification"):
        (tmp_path / name).write_text(name)
    config = tmp_path / "config"
    config.write_text(json.dumps({"temperature_K": 300.0, "timestep_fs_initial": 1.0, "response_segment": {"master_seed": 77, "steps": 2, "report_interval_steps": 1, "max_step_batch": 1}}))

    def fake_loader(prmtop, inpcrd, mapping_path, config_path, qualification_path, *, model):
        calls["loader"] = (prmtop, inpcrd, mapping_path, config_path, qualification_path, model)
        return {
            "system": fixture_system(),
            "xyz": XYZ,
            "mapping": fixture_mapping(),
            "settings": object(),
            "qualification": {"chemical_review": {"status": "technical_chemistry_preparation_pass"}},
            "provenance": {"loader": "fake"},
            "chemical_geometry": None,
        }

    class FakeInpcrd:
        boxVectors = None

        def __init__(self, path):
            calls["inpcrd"] = path

    class FakePrmtop:
        topology = None

        def __init__(self, path, periodicBoxVectors=None):
            calls["prmtop"] = (path, periodicBoxVectors)

    def fake_run_segment(**kwargs):
        calls["run_segment"] = kwargs
        return {"settings": kwargs["settings"], "last_committed_step": kwargs["settings"].steps}

    monkeypatch.setattr(segment.pilot, "load_qualified_inputs", fake_loader)
    monkeypatch.setattr(segment.app, "AmberInpcrdFile", FakeInpcrd)
    monkeypatch.setattr(segment.app, "AmberPrmtopFile", FakePrmtop)
    monkeypatch.setattr(segment, "run_segment", fake_run_segment)
    result = run_from_qualified(
        prmtop=tmp_path / "prmtop",
        inpcrd=tmp_path / "inpcrd",
        mapping_path=tmp_path / "mapping",
        config_path=config,
        qualification_path=tmp_path / "qualification",
        output_dir=tmp_path / "out",
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        precision="double",
        device_index=None,
        resume=False,
        overrides={},
        locked_force_plan=None,
        parent_state=None,
        equilibration_certificate=None,
        measurement_identity=tmp_path / "measurement_identity.json",
        measurement_identity_sha256="1" * 64,
    )
    assert calls["loader"][-1] == "isolated"
    assert "provenance" not in calls["run_segment"]
    assert calls["run_segment"]["measurement_identity"] == tmp_path / "measurement_identity.json"
    assert calls["run_segment"]["measurement_identity_sha256"] == "1" * 64
    assert len(calls["run_segment"]["input_hashes"]["source_provenance_sha256"]) == 64
    assert calls["run_segment"]["qualification"]["chemical_review"]["status"] == "technical_chemistry_preparation_pass"
    assert result["settings"].master_seed == 77
    assert result["last_committed_step"] == 2


def _refresh_locked_force_plan_digest(plan_path):
    plan = json.loads(plan_path.read_text())
    _sigma, calibration_data_sha256 = segment._validate_variance_rows(
        plan_path.parent, plan["accepted_zero_calibration_variance_rows"], plan["binding"]
    )
    plan["zero_calibration_data_sha256"] = calibration_data_sha256
    source_record = plan["source_calibration_certificate"]
    source_path = plan_path.parent / source_record["path"]
    source = json.loads(source_path.read_text())
    source["zero_calibration_data_sha256"] = calibration_data_sha256
    source_path.write_text(json.dumps(source, indent=2))
    source_record["sha256"] = segment._sha256(source_path)
    plan_path.write_text(json.dumps(plan, indent=2))
    return plan_path


def test_locked_force_plan_uses_actual_parent_authority_not_optional_initial_fields(tmp_path):
    zero_dir = tmp_path / "zero"
    zero = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
        output_dir=zero_dir, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True,
    )
    binding = binding_from_manifest(zero)
    binding["parent_state_sha256"] = segment._sha256(zero_dir / zero["current_generation"] / "segment_state.xml")
    eq_cert = write_json(tmp_path / "zero_force_eq_cert.json", cert_payload("zero_force_equilibrium", "force_equilibration", "isolated", 0.0, binding))
    plan_path = write_locked_force_plan(tmp_path / "locked_force.json", model="isolated", force=0.5, binding=binding)
    plan = json.loads(plan_path.read_text())
    for model_name in ("flexible", "fixed", "rigid"):
        rows = [row for row in plan["accepted_zero_calibration_variance_rows"] if row["model"] == model_name]
        for offset, row in enumerate(rows):
            row["initialization_lineage"].update({
                "thermostat_seed": 3333,
                "initial_positions_sha256": f"{offset + 1:064x}",
                "initial_velocities_sha256": f"{offset + 11:064x}",
            })
            manifest_path = tmp_path / row["zero_calibration_manifest"]["path"]
            manifest = json.loads(manifest_path.read_text())
            manifest["seed_lineage"] = row["initialization_lineage"]
            manifest_path.write_text(json.dumps(manifest, indent=2))
            row["zero_calibration_manifest"]["sha256"] = segment._sha256(manifest_path)
    bad_plan = write_json(tmp_path / "bad_optional_initial_force.json", plan)
    with pytest.raises(AdmissionError, match="effective initialization lineages"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
            settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path / "bad_optional_initial", phase="force_equilibration", model="isolated",
            replicate_id="r1", force_h=0.5, locked_force_plan=bad_plan, parent_state=zero_dir,
            equilibration_certificate=eq_cert, platform_name="Reference", synthetic_fixture=True,
        )

    parent_override = json.loads(plan_path.read_text())
    first = parent_override["accepted_zero_calibration_variance_rows"][0]
    first["initialization_lineage"]["parent_state_sha256"] = "9" * 64
    manifest_path = tmp_path / first["zero_calibration_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["seed_lineage"] = first["initialization_lineage"]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    first["zero_calibration_manifest"]["sha256"] = segment._sha256(manifest_path)
    bad_parent = write_json(tmp_path / "bad_parent_override_force.json", parent_override)
    with pytest.raises(AdmissionError, match="parent_state_sha256 does not match manifest admission parent"):
        run_segment(
            system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
            settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
            output_dir=tmp_path / "bad_parent_override", phase="force_equilibration", model="isolated",
            replicate_id="r1", force_h=0.5, locked_force_plan=bad_parent, parent_state=zero_dir,
            equilibration_certificate=eq_cert, platform_name="Reference", synthetic_fixture=True,
        )


def test_locked_force_plan_rejects_duplicate_npz_scientific_content_even_if_zip_hash_differs(tmp_path):
    zero_dir = tmp_path / "zero"
    zero = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=2, report_interval_steps=1, max_step_batch=1),
        output_dir=zero_dir, phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True,
    )
    binding = binding_from_manifest(zero)
    binding["parent_state_sha256"] = segment._sha256(zero_dir / zero["current_generation"] / "segment_state.xml")
    plan_path = write_locked_force_plan(tmp_path / "locked_force.json", model="isolated", force=0.5, binding=binding)
    plan = json.loads(plan_path.read_text())
    rows = [row for row in plan["accepted_zero_calibration_variance_rows"] if row["model"] == "rigid"]
    src_artifact = tmp_path / rows[0]["zero_calibration_artifact"]["path"]
    dup_artifact = tmp_path / rows[1]["zero_calibration_artifact"]["path"]
    with np.load(src_artifact) as payload, dup_artifact.open("wb") as handle:
        np.savez(
            handle,
            core_displacement_nm=payload["core_displacement_nm"],
            time_ps=payload["time_ps"],
            reference_nm=payload["reference_nm"],
            q_ambient=payload["q_ambient"],
            core_indices=payload["core_indices"],
            force_kj_mol_nm=payload["force_kj_mol_nm"],
        )
    assert segment._sha256(src_artifact) != segment._sha256(dup_artifact)
    rows[1]["zero_calibration_artifact"]["sha256"] = segment._sha256(dup_artifact)
    manifest_path = tmp_path / rows[1]["zero_calibration_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["generation_sha256"]["segment_observables.npz"] = segment._sha256(dup_artifact)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    rows[1]["zero_calibration_manifest"]["sha256"] = segment._sha256(manifest_path)
    cert_path = tmp_path / rows[1]["stationarity_certificate"]["path"]
    cert = json.loads(cert_path.read_text())
    cert["artifact_sha256"] = segment._sha256(dup_artifact)
    cert_path.write_text(json.dumps(cert, indent=2))
    rows[1]["stationarity_certificate"]["sha256"] = segment._sha256(cert_path)
    with pytest.raises(AdmissionError, match="scientific array content"):
        segment._validate_variance_rows(tmp_path, plan["accepted_zero_calibration_variance_rows"], binding)


MEASUREMENT_IDENTITY_PATH = Path("125_atomistic_response_validation_20260906/analysis/measurement_identity/measurement_identity.json")
MEASUREMENT_IDENTITY_SHA256 = "0f30d6d1305a4d23cfeac0797644f79d29fca0c0138aa8af41088e718bb21719"


def _topology_from_endpoint_rows(endpoint_rows):
    topology = app.Topology()
    chain = topology.addChain("B")
    rows_by_residue = {row["amber_residue_index_zero_based"]: row for row in endpoint_rows}
    next_atom_index = 0
    for residue_index in range(max(rows_by_residue) + 1):
        row = rows_by_residue.get(residue_index)
        residue_name = row["amber_residue_name"] if row else "DUM"
        residue = topology.addResidue(residue_name, chain)
        target_atom_index = row["atom_index_zero_based"] if row else next_atom_index
        while next_atom_index < target_atom_index:
            topology.addAtom("C", app.element.carbon, residue)
            next_atom_index += 1
        topology.addAtom("CA" if row else "C", app.element.carbon, residue)
        next_atom_index += 1
    return topology


def _actual_measurement_identity_payload():
    artifact = json.loads(MEASUREMENT_IDENTITY_PATH.read_text())
    mapping = json.loads(Path(artifact["endpoints"]["isolated"]["sources"]["mapping"]["path"]).read_text())
    topology = _topology_from_endpoint_rows(artifact["endpoints"]["isolated"]["endpoint_rows"])
    return artifact, mapping, topology


def test_actual_measurement_identity_validates_isolated_proper_rigid_bridge():
    if not MEASUREMENT_IDENTITY_PATH.exists():
        pytest.skip("private 125 measurement_identity artifact is not present in this checkout")
    _artifact, mapping, topology = _actual_measurement_identity_payload()
    record = segment._validate_measurement_identity(
        MEASUREMENT_IDENTITY_PATH,
        mapping=mapping,
        topology=topology,
        model="isolated",
        expected_sha256=MEASUREMENT_IDENTITY_SHA256,
        synthetic_fixture=False,
    )
    assert record["common_identity_count"] == 269
    assert record["common_identity_sha256"] == "8f2f8e1c69a3dd390a94e28d71ac6966985a1c6988547408fe0a8b91e8fae767"
    assert record["bridge"]["endpoint"] == "isolated"
    assert record["bridge"]["rotation_determinant"] == pytest.approx(1.0)


def test_measurement_identity_rejects_reflection_q_ref_and_canonical_identity_tampering(tmp_path):
    if not MEASUREMENT_IDENTITY_PATH.exists():
        pytest.skip("private 125 measurement_identity artifact is not present in this checkout")
    artifact, mapping, topology = _actual_measurement_identity_payload()
    reflected_artifact = json.loads(json.dumps(artifact))
    reflected_mapping = json.loads(json.dumps(mapping))
    reflected = np.asarray(mapping["q_transport"]["rotation_matrix"], dtype=float)
    reflected[:, 0] *= -1.0
    reflected_mapping["q_transport"]["rotation_matrix"] = reflected.tolist()
    reflected_artifact["endpoints"]["isolated"]["transport_from_joint"]["mapping_q_transport"]["rotation_matrix"] = reflected.tolist()
    reflected_path = write_json(tmp_path / "reflected_measurement_identity.json", reflected_artifact)
    with pytest.raises(AdmissionError, match=r"proper orthogonal det\+1"):
        segment._validate_measurement_identity(reflected_path, mapping=reflected_mapping, topology=topology, model="isolated", synthetic_fixture=False)

    bad_q = json.loads(json.dumps(mapping))
    bad_q["q"][0] *= -1.0
    with pytest.raises(AdmissionError, match="transported joint q"):
        segment._validate_measurement_identity(MEASUREMENT_IDENTITY_PATH, mapping=bad_q, topology=topology, model="isolated", synthetic_fixture=False)

    bad_ref = json.loads(json.dumps(mapping))
    bad_ref["reference_nm"][0][0] += 1e-4
    with pytest.raises(AdmissionError, match="transported joint reference"):
        segment._validate_measurement_identity(MEASUREMENT_IDENTITY_PATH, mapping=bad_ref, topology=topology, model="isolated", synthetic_fixture=False)

    bad_common = json.loads(json.dumps(artifact))
    bad_common["common_identity_rows"][0]["author_resnum"] += 1
    bad_common_path = write_json(tmp_path / "bad_common_measurement_identity.json", bad_common)
    with pytest.raises(AdmissionError, match="common_identity_sha256"):
        segment._validate_measurement_identity(bad_common_path, mapping=mapping, topology=topology, model="isolated", synthetic_fixture=False)

    bad_rows = json.loads(json.dumps(artifact["endpoints"]["isolated"]["endpoint_rows"]))
    bad_rows[0]["amber_residue_name"] = "GLY"
    bad_topology = _topology_from_endpoint_rows(bad_rows)
    with pytest.raises(AdmissionError, match="topology residue type mismatch"):
        segment._validate_measurement_identity(MEASUREMENT_IDENTITY_PATH, mapping=mapping, topology=bad_topology, model="isolated", synthetic_fixture=False)


def _actual_endpoint_record(endpoint_model):
    artifact = json.loads(MEASUREMENT_IDENTITY_PATH.read_text())
    endpoint = "isolated" if endpoint_model == "isolated" else "joint"
    mapping = json.loads(Path(artifact["endpoints"][endpoint]["sources"]["mapping"]["path"]).read_text())
    topology = _topology_from_endpoint_rows(artifact["endpoints"][endpoint]["endpoint_rows"])
    return segment._validate_measurement_identity(
        MEASUREMENT_IDENTITY_PATH,
        mapping=mapping,
        topology=topology,
        model=endpoint_model,
        expected_sha256=MEASUREMENT_IDENTITY_SHA256,
        synthetic_fixture=False,
    ), mapping


def _write_actual_joint_rows_force_plan(path, *, current_binding, row_binding, joint_mapping, force):
    f0 = abs(float(force))
    sigma_max = 0.25 * segment.pilot.GAS_CONSTANT * 300.0 / f0
    reference_nm = np.asarray(joint_mapping["reference_nm"], dtype=float)
    q = np.asarray(joint_mapping["q"], dtype=float).reshape(-1)
    q_norm2 = float(q @ q)
    core_indices = np.asarray(joint_mapping["core_indices"], dtype=int)
    variance_rows = []
    for model_name in ("flexible", "fixed", "rigid"):
        for replicate in range(3):
            replicate_id = f"{model_name}-joint-r{replicate}"
            q_scale = sigma_max if model_name == "rigid" and replicate == 2 else 0.5 * sigma_max
            q_offset = 0.02 * (replicate + 1) + 0.01 * (("flexible", "fixed", "rigid").index(model_name) + 1)
            q_series = np.asarray([-q_scale, 0.0, q_scale], dtype=float) + q_offset
            core_displacement = np.asarray([(q * value / q_norm2).reshape(reference_nm.shape) for value in q_series])
            artifact = path.with_name(f"{path.stem}_{model_name}_joint_r{replicate}.npz")
            with artifact.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    core_displacement_nm=core_displacement,
                    time_ps=np.asarray([0.0, 1.0, 2.0], dtype=float),
                    reference_nm=reference_nm,
                    q_ambient=q,
                    core_indices=core_indices,
                    force_kj_mol_nm=0.0,
                )
            parent_sha = f"{('abc' + model_name + str(replicate)).encode().hex():0<64}"[:64]
            initialization_lineage = {
                "master_seed": 9000 + replicate,
                "velocity_seed": 9100 + replicate,
                "thermostat_seed": 9200 + replicate,
                "parent_state_sha256": parent_sha,
            }
            manifest = write_json(
                path.with_name(f"{path.stem}_{model_name}_joint_r{replicate}_manifest.json"),
                {
                    "schema_version": "1.0",
                    "status": "segment_complete",
                    "phase": "zero_calibration",
                    "model": model_name,
                    "replicate_id": replicate_id,
                    "force_kj_mol_nm": 0.0,
                    "seed_lineage": initialization_lineage,
                    "admission": {"parent_state": {"state_xml_sha256": parent_sha}},
                    "current_generation": f"generation_{model_name}_joint_{replicate}",
                    "generation_sha256": {"segment_observables.npz": segment._sha256(artifact)},
                    **{key: row_binding[key] for key in segment.IDENTITY_BINDING_KEYS},
                },
            )
            stationarity = write_json(
                path.with_name(f"{path.stem}_{model_name}_joint_r{replicate}_stationarity.json"),
                {
                    "schema_version": "1.0",
                    "status": "pass",
                    "certificate_kind": "zero_calibration_stationarity_acceptance",
                    "model": model_name,
                    "replicate_id": replicate_id,
                    "artifact_sha256": segment._sha256(artifact),
                    "artifact_selection": {"observable": "q_ambient_dot_core_displacement_nm", "variance_ddof": 1},
                    "binding": {key: row_binding[key] for key in ("data_input_hashes", "core_indices_sha256", "reference_nm_sha256", "q_ambient_sha256")},
                },
            )
            variance_rows.append(
                {
                    "model": model_name,
                    "replicate_id": replicate_id,
                    "sample_variance_Q_nm2": float(np.var(q_series, ddof=1)),
                    "initialization_lineage": initialization_lineage,
                    "binding": dict(row_binding),
                    "zero_calibration_artifact": {"path": artifact.name, "sha256": segment._sha256(artifact)},
                    "zero_calibration_manifest": {"path": manifest.name, "sha256": segment._sha256(manifest)},
                    "stationarity_certificate": {
                        "path": stationarity.name,
                        "sha256": segment._sha256(stationarity),
                        "certificate_kind": "zero_calibration_stationarity_acceptance",
                    },
                }
            )
    _sigma, calibration_data_sha256 = segment._validate_variance_rows(path.parent, variance_rows, current_binding)
    source = write_json(
        path.with_name(path.stem + "_source_calibration.json"),
        cert_payload("zero_force_calibration", "zero_calibration", "isolated", 0.0, current_binding, zero_calibration_data_sha256=calibration_data_sha256),
    )
    return write_json(
        path,
        {
            "schema_version": "1.0",
            "status": "locked",
            "model": "isolated",
            "force_grid_kj_mol_nm": [-2 * f0, -f0, 0.0, f0, 2 * f0],
            "force_multipliers": [-2, -1, 0, 1, 2],
            "f0_kj_mol_nm": f0,
            "sigma_max_nm": sigma_max,
            "locked_f0_rule": "0.25RT_over_sigma_max",
            "source_calibration_certificate": {
                "path": source.name,
                "sha256": segment._sha256(source),
                "certificate_kind": "zero_force_calibration",
            },
            "zero_calibration_data_sha256": calibration_data_sha256,
            "accepted_zero_calibration_variance_rows": variance_rows,
            "binding": current_binding,
        },
    )


def test_actual_joint_measurement_rows_admit_isolated_force_plan_via_bridge(tmp_path):
    if not MEASUREMENT_IDENTITY_PATH.exists():
        pytest.skip("private 125 measurement_identity artifact is not present in this checkout")
    isolated_record, isolated_mapping = _actual_endpoint_record("isolated")
    joint_record, joint_mapping = _actual_endpoint_record("rigid")
    base = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path / "base", phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True, input_hashes=INPUT_HASHES,
    )
    current_binding = binding_from_manifest(base)
    current_binding.update({
        "measurement_identity": isolated_record,
        "core_indices_sha256": segment._canonical_hash(list(map(int, isolated_mapping["core_indices"]))),
        "reference_nm_sha256": isolated_record["bridge"]["isolated_reference_nm_sha256"],
        "q_ambient_sha256": isolated_record["bridge"]["isolated_q_ambient_sha256"],
    })
    row_binding = dict(current_binding)
    row_binding.update({
        "measurement_identity": joint_record,
        "core_indices_sha256": segment._canonical_hash(list(map(int, joint_mapping["core_indices"]))),
        "reference_nm_sha256": joint_record["bridge"]["joint_reference_nm_sha256"],
        "q_ambient_sha256": joint_record["bridge"]["joint_q_ambient_sha256"],
        "data_input_hashes": {"prmtop": "1" * 64, "inpcrd": "2" * 64, "mapping": "3" * 64},
    })
    plan = _write_actual_joint_rows_force_plan(tmp_path / "joint_to_iso_locked_force.json", current_binding=current_binding, row_binding=row_binding, joint_mapping=joint_mapping, force=0.5)
    evidence = segment._validate_locked_force_plan(plan, model="isolated", force_h=0.5, binding=current_binding, temperature_K=300.0)
    assert evidence["f0_kj_mol_nm"] == pytest.approx(0.5)
    assert evidence["zero_calibration_data_sha256"] == json.loads(plan.read_text())["zero_calibration_data_sha256"]


def test_measurement_identity_rejects_declared_hash_and_endpoint_fact_contradictions(tmp_path):
    if not MEASUREMENT_IDENTITY_PATH.exists():
        pytest.skip("private 125 measurement_identity artifact is not present in this checkout")
    artifact, mapping, topology = _actual_measurement_identity_payload()
    bad_joint_ref = json.loads(json.dumps(artifact))
    bad_joint_ref["joint_reference_q"]["reference_nm_sha256"] = "f" * 64
    with pytest.raises(AdmissionError, match="declared joint reference hash"):
        segment._validate_measurement_identity(write_json(tmp_path / "bad_joint_ref.json", bad_joint_ref), mapping=mapping, topology=topology, model="isolated", synthetic_fixture=False)
    bad_joint_q = json.loads(json.dumps(artifact))
    bad_joint_q["joint_reference_q"]["q_sha256"] = "f" * 64
    with pytest.raises(AdmissionError, match="declared joint q hash"):
        segment._validate_measurement_identity(write_json(tmp_path / "bad_joint_q.json", bad_joint_q), mapping=mapping, topology=topology, model="isolated", synthetic_fixture=False)
    bad_endpoint_digest = json.loads(json.dumps(artifact))
    bad_endpoint_digest["endpoints"]["isolated"]["summary"]["endpoint_identity_sha256"] = "f" * 64
    with pytest.raises(AdmissionError, match="endpoint_identity_sha256"):
        segment._validate_measurement_identity(write_json(tmp_path / "bad_endpoint_digest.json", bad_endpoint_digest), mapping=mapping, topology=topology, model="isolated", synthetic_fixture=False)
    bad_residue = json.loads(json.dumps(artifact))
    bad_residue["endpoints"]["isolated"]["endpoint_rows"][0]["amber_residue_index_zero_based"] = 999999
    with pytest.raises(AdmissionError, match="topology residue index"):
        segment._validate_measurement_identity(write_json(tmp_path / "bad_residue_index.json", bad_residue), mapping=mapping, topology=topology, model="isolated", synthetic_fixture=False)


def _synthetic_bridge_fixture(tmp_path):
    joint_ref = XYZ[:4].copy()
    joint_q = dm.internal_basis(joint_ref)[:, 0].reshape(-1)
    theta = np.pi / 2
    rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    source_center = np.asarray([0.1, 0.2, 0.3])
    target_center = np.asarray([1.0, 1.5, 2.0])
    iso_ref = (joint_ref - source_center) @ rotation + target_center
    iso_q = (joint_q.reshape(-1, 3) @ rotation).reshape(-1)
    joint_mapping = {
        "core_indices": [0, 1, 2, 3],
        "reference_nm": joint_ref.tolist(),
        "q": joint_q.tolist(),
    }
    isolated_mapping = {
        "core_indices": [0, 1, 2, 3],
        "reference_nm": iso_ref.tolist(),
        "q": iso_q.tolist(),
        "q_transport": {
            "rotation_matrix": rotation.tolist(),
            "source_center_nm": source_center.tolist(),
            "target_center_nm": target_center.tolist(),
            "no_new_q_projection": True,
        },
    }
    joint_mapping_path = write_json(tmp_path / "joint_mapping.json", joint_mapping)
    csv_path = tmp_path / "crbn_residue_window.csv"
    csv_path.write_text("index,author_resnum\n0,77\n1,78\n2,79\n3,80\n")
    common_rows = [
        {"order_index": i, "uniprot_accession": "Q96SW2", "author_resnum": 77 + i, "canonical_resname": "ALA", "atom_name": "CA"}
        for i in range(4)
    ]
    endpoints = {}
    for endpoint in ("joint", "isolated"):
        endpoint_rows = [
            {
                **row,
                "endpoint": endpoint,
                "atom_index_zero_based": i,
                "amber_residue_index_zero_based": i,
                "amber_residue_name": "ALA",
                "atom_name_in_topology": "CA",
            }
            for i, row in enumerate(common_rows)
        ]
        endpoints[endpoint] = {
            "identity_rows": common_rows,
            "endpoint_rows": endpoint_rows,
            "summary": {
                "core_count": 4,
                "first_core_atom_index": 0,
                "last_core_atom_index": 3,
                "mapping_core_indices_sha256": segment._canonical_hash([0, 1, 2, 3]),
                "endpoint_identity_sha256": segment._canonical_hash(endpoint_rows),
            },
        }
    endpoints["isolated"]["transport_from_joint"] = {
        "status": "pass",
        "formula": "ref_iso=(ref_joint-source_center)@R+target_center; q_iso=q_joint@R",
        "no_new_q_projection": True,
        "isolated_reference_nm_sha256": segment._canonical_hash(iso_ref.tolist()),
        "isolated_q_sha256": segment._canonical_hash(iso_q.tolist()),
        "mapping_q_transport": isolated_mapping["q_transport"],
    }
    artifact = {
        "schema_version": "1.0",
        "status": "pass",
        "measurement_kind": "synthetic_CRBN_Q96SW2_ordered_CA_closure_coordinate",
        "common_identity_rows": common_rows,
        "common_identity_sha256": segment._canonical_hash(common_rows),
        "core_residue_file": {"path": csv_path.name, "sha256": segment._sha256(csv_path)},
        "response_bridge": {
            "status": "ready_for_engine_binding_not_production_admission",
            "trajectory_refit_or_new_q_projection": False,
            "common_identity_excludes_endpoint_indices": True,
            "endpoint_indices_are_separate_evidence": True,
        },
        "joint_reference_q": {
            "mapping": {"path": joint_mapping_path.name, "sha256": segment._sha256(joint_mapping_path)},
            "reference_nm_sha256": segment._canonical_hash(joint_ref.tolist()),
            "q_sha256": segment._canonical_hash(joint_q.tolist()),
        },
        "endpoints": endpoints,
    }
    artifact_path = write_json(tmp_path / "measurement_identity.json", artifact)
    topology = app.Topology()
    chain = topology.addChain("B")
    for _ in range(4):
        residue = topology.addResidue("ALA", chain)
        topology.addAtom("CA", app.element.carbon, residue)
    return artifact_path, joint_mapping, isolated_mapping, topology


def test_portable_synthetic_bridge_integrates_joint_rows_with_isolated_force_plan(tmp_path):
    artifact_path, joint_mapping, isolated_mapping, topology = _synthetic_bridge_fixture(tmp_path)
    isolated_record = segment._validate_measurement_identity(artifact_path, mapping=isolated_mapping, topology=topology, model="isolated", synthetic_fixture=True)
    joint_record = segment._validate_measurement_identity(artifact_path, mapping=joint_mapping, topology=topology, model="rigid", synthetic_fixture=True)
    base = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path / "base", phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True, input_hashes=INPUT_HASHES,
    )
    current_binding = binding_from_manifest(base)
    current_binding.update({
        "measurement_identity": isolated_record,
        "reference_nm_sha256": isolated_record["bridge"]["isolated_reference_nm_sha256"],
        "q_ambient_sha256": isolated_record["bridge"]["isolated_q_ambient_sha256"],
    })
    row_binding = dict(current_binding)
    row_binding.update({
        "measurement_identity": joint_record,
        "reference_nm_sha256": joint_record["bridge"]["joint_reference_nm_sha256"],
        "q_ambient_sha256": joint_record["bridge"]["joint_q_ambient_sha256"],
    })
    plan = _write_actual_joint_rows_force_plan(tmp_path / "synthetic_joint_to_iso_force.json", current_binding=current_binding, row_binding=row_binding, joint_mapping=joint_mapping, force=0.5)
    assert segment._validate_locked_force_plan(plan, model="isolated", force_h=0.5, binding=current_binding, temperature_K=300.0)["f0_kj_mol_nm"] == pytest.approx(0.5)


def test_shared_bridge_requires_row_actual_q_ref_to_match_validated_joint_endpoint(tmp_path):
    artifact_path, joint_mapping, isolated_mapping, topology = _synthetic_bridge_fixture(tmp_path)
    isolated_record = segment._validate_measurement_identity(artifact_path, mapping=isolated_mapping, topology=topology, model="isolated", synthetic_fixture=True)
    joint_record = segment._validate_measurement_identity(artifact_path, mapping=joint_mapping, topology=topology, model="rigid", synthetic_fixture=True)
    base = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path / "base", phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True, input_hashes=INPUT_HASHES,
    )
    current_binding = binding_from_manifest(base)
    current_binding.update({
        "measurement_identity": isolated_record,
        "reference_nm_sha256": isolated_record["bridge"]["isolated_reference_nm_sha256"],
        "q_ambient_sha256": isolated_record["bridge"]["isolated_q_ambient_sha256"],
    })
    row_binding = dict(current_binding)
    row_binding.update({
        "measurement_identity": joint_record,
        "reference_nm_sha256": joint_record["bridge"]["joint_reference_nm_sha256"],
        "q_ambient_sha256": joint_record["bridge"]["joint_q_ambient_sha256"],
    })
    plan = _write_actual_joint_rows_force_plan(tmp_path / "valid_frozen_q_force.json", current_binding=current_binding, row_binding=row_binding, joint_mapping=joint_mapping, force=0.5)
    assert segment._validate_locked_force_plan(plan, model="isolated", force_h=0.5, binding=current_binding, temperature_K=300.0)

    q2_mapping = json.loads(json.dumps(joint_mapping))
    q = np.asarray(q2_mapping["q"], dtype=float)
    q2 = q.copy()
    q2[0] *= -1.0
    q2 /= np.linalg.norm(q2)
    q2_mapping["q"] = q2.tolist()
    bad_row_binding = dict(row_binding)
    bad_row_binding["q_ambient_sha256"] = segment._canonical_hash(q2.tolist())
    with pytest.raises(AdmissionError, match="Variance row measurement identity binding mismatch: q_ambient_sha256"):
        _write_actual_joint_rows_force_plan(tmp_path / "bad_q2_force.json", current_binding=current_binding, row_binding=bad_row_binding, joint_mapping=q2_mapping, force=0.5)

    bad_ref_mapping = json.loads(json.dumps(joint_mapping))
    ref2 = np.asarray(bad_ref_mapping["reference_nm"], dtype=float)
    ref2 = ref2.copy()
    ref2[0, 0] += 0.01
    bad_ref_mapping["reference_nm"] = ref2.tolist()
    bad_ref_binding = dict(row_binding)
    bad_ref_binding["reference_nm_sha256"] = segment._canonical_hash(ref2.tolist())
    with pytest.raises(AdmissionError, match="Variance row measurement identity binding mismatch: reference_nm_sha256"):
        _write_actual_joint_rows_force_plan(tmp_path / "bad_ref_force.json", current_binding=current_binding, row_binding=bad_ref_binding, joint_mapping=bad_ref_mapping, force=0.5)


def test_shared_bridge_requires_calibration_row_endpoint_to_be_joint(tmp_path):
    artifact_path, _joint_mapping, isolated_mapping, topology = _synthetic_bridge_fixture(tmp_path)
    isolated_record = segment._validate_measurement_identity(artifact_path, mapping=isolated_mapping, topology=topology, model="isolated", synthetic_fixture=True)
    base = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path / "base", phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True, input_hashes=INPUT_HASHES,
    )
    current_binding = binding_from_manifest(base)
    current_binding.update({
        "measurement_identity": isolated_record,
        "reference_nm_sha256": isolated_record["bridge"]["isolated_reference_nm_sha256"],
        "q_ambient_sha256": isolated_record["bridge"]["isolated_q_ambient_sha256"],
    })
    row_binding = dict(current_binding)
    with pytest.raises(AdmissionError, match="Variance row measurement identity endpoint must be joint"):
        _write_actual_joint_rows_force_plan(
            tmp_path / "bad_row_isolated_endpoint_force.json",
            current_binding=current_binding, row_binding=row_binding, joint_mapping=isolated_mapping, force=0.5,
        )


def test_shared_bridge_requires_current_actual_q_ref_to_match_validated_endpoint(tmp_path):
    artifact_path, joint_mapping, isolated_mapping, topology = _synthetic_bridge_fixture(tmp_path)
    isolated_record = segment._validate_measurement_identity(artifact_path, mapping=isolated_mapping, topology=topology, model="isolated", synthetic_fixture=True)
    joint_record = segment._validate_measurement_identity(artifact_path, mapping=joint_mapping, topology=topology, model="rigid", synthetic_fixture=True)
    base = run_segment(
        system=fixture_system(), xyz=XYZ, mapping=fixture_mapping(),
        settings=settings(steps=1, report_interval_steps=1, max_step_batch=1),
        output_dir=tmp_path / "base", phase="zero_equilibration", model="isolated", replicate_id="r1",
        force_h=0.0, platform_name="Reference", synthetic_fixture=True, input_hashes=INPUT_HASHES,
    )
    current_binding = binding_from_manifest(base)
    current_binding.update({
        "measurement_identity": isolated_record,
        "reference_nm_sha256": isolated_record["bridge"]["isolated_reference_nm_sha256"],
        "q_ambient_sha256": "a" * 64,
    })
    row_binding = dict(current_binding)
    row_binding.update({
        "measurement_identity": joint_record,
        "reference_nm_sha256": joint_record["bridge"]["joint_reference_nm_sha256"],
        "q_ambient_sha256": joint_record["bridge"]["joint_q_ambient_sha256"],
    })
    with pytest.raises(AdmissionError, match="Current segment measurement identity binding mismatch: q_ambient_sha256"):
        _write_actual_joint_rows_force_plan(tmp_path / "bad_current_q_force.json", current_binding=current_binding, row_binding=row_binding, joint_mapping=joint_mapping, force=0.5)
