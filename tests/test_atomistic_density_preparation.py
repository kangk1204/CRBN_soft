"""Density-preparation runner contract tests; no GPU required."""
import csv
import json
import os

import numpy as np
import pytest

mm = pytest.importorskip("openmm")
from openmm import app

from scripts import directional_mechanics as dm
from scripts import run_atomistic_density_preparation as density


TEST_PLATFORM = os.environ.get("OPENMM_TEST_PLATFORM", "Reference")
INPUT_HASHES = {"prmtop": "a" * 64, "inpcrd": "b" * 64, "mapping": "c" * 64}


def density_fixture(isolated=False, *, cartesian_restraint=False, zero_mass=False):
    topology = app.Topology()
    chain = topology.addChain("A")
    core = np.array([[-.30, -.20, -.10], [.40, -.10, .20], [.05, .45, -.25], [-.10, .05, .55]]) + 1.
    xyz = list(core)
    masses = [12.] * 4
    for _i in range(4):
        residue = topology.addResidue("ALA", chain)
        topology.addAtom("CA", app.element.carbon, residue)
    body = []
    if not isolated:
        residue = topology.addResidue("ALA", topology.addChain("B"))
        for name, element, point, mass in (
            ("CA", app.element.carbon, [3.0, 3.1, 3.2], 12.),
            ("CB", app.element.carbon, [3.1, 3.2, 3.3], 12.),
        ):
            topology.addAtom(name, element, residue)
            body.append(len(xyz))
            xyz.append(point)
            masses.append(mass)
    residue = topology.addResidue("HOH", topology.addChain("W"))
    water_start = len(xyz)
    theta = np.deg2rad(104.52)
    water = np.array([[0, 0, 0], [.09572, 0, 0],
                      [.09572 * np.cos(theta), .09572 * np.sin(theta), 0]]) + [4.7, 1., 1.]
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
    box = np.eye(3) * 6.
    system.setDefaultPeriodicBoxVectors(*box)
    topology.setPeriodicBoxVectors(tuple(mm.Vec3(*row) for row in box) * density.unit.nanometer)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(1.)
    for i, mass in enumerate(masses):
        system.addParticle(0. if zero_mass and i == len(masses) - 1 else mass)
        water_index = i - water_start
        charge = (-.834, .417, .417)[water_index] if 0 <= water_index < 3 else 0.
        nb.addParticle(charge, .3, 0.)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = water_start + i, water_start + j
        system.addConstraint(a, b, float(np.linalg.norm(xyz[a] - xyz[b])))
        nb.addException(a, b, 0., .1, 0.)
    spring = mm.HarmonicBondForce()
    for i, j in ((0, 1), (1, 2), (2, 3)):
        spring.addBond(i, j, float(np.linalg.norm(core[i] - core[j])) * 1.01, 1000.)
    system.addForce(nb)
    system.addForce(spring)
    if cartesian_restraint:
        restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter("k", 10.)
        for name in ("x0", "y0", "z0"):
            restraint.addPerParticleParameter(name)
        for index, point in enumerate(xyz):
            restraint.addParticle(index, point)
        system.addForce(restraint)
    q = dm.internal_basis(core)[:, 0]
    mapping = {
        "schema_version": "1.0",
        "core_indices": list(range(4)),
        "reference_nm": core.tolist(),
        "q": q.tolist(),
        "ddb1_atom_indices": body,
    }
    model = "isolated" if isolated else "flexible"
    mapping = density.pilot.validate_mapping(mapping, topology, xyz, model, expected_core_count=4)
    return system, xyz, mapping, model


def test_reference_density_preparation_outputs_restart_handoff_and_no_certificate(tmp_path):
    system, xyz, mapping, model = density_fixture()
    settings = density.DensitySettings(nvt_steps=1, npt_steps=2, report_interval_steps=1,
                                       checkpoint_interval_steps=1, max_wall_seconds=60.)
    result = density._run_engine(system, xyz, mapping, settings, tmp_path, model=model,
                                 platform_name=TEST_PLATFORM, synthetic_fixture=True,
                                 qualification={"chemical_review": {"status": "pending"}},
                                 provenance={"role": "synthetic_density_fixture"})
    assert result["status"] == "density_completed", result.get("reason")
    assert result["density_preparation_completed"]
    assert not result["equilibrium_certified"] and not result["production_ready"]
    assert result["completed_by_phase"] == {"nvt": 1, "npt": 2}
    assert result["barostat"] == {"class": "MonteCarloBarostat", "pressure_bar": 1.,
                                  "temperature_K": 300., "frequency_steps": 25,
                                  "random_seed": settings.seed + 2,
                                  "force_groups_included_in_accept_reject": "all integration force groups in Context"}
    serialized = (tmp_path / "system.xml").read_text()
    system_from_xml = mm.XmlSerializer.deserialize(serialized)
    force_names = [type(system_from_xml.getForce(i)).__name__ for i in range(system_from_xml.getNumForces())]
    assert force_names.count("MonteCarloBarostat") == 1
    assert force_names.count("CustomCVForce") == 1
    assert force_names.count("CustomCompoundBondForce") == 1
    restart = json.loads((tmp_path / "current_restart.json").read_text())
    assert restart["completed_by_phase"] == {"nvt": 1, "npt": 2}
    assert (tmp_path / restart["checkpoint"]["path"]).stat().st_size == restart["checkpoint"]["bytes"]
    assert (tmp_path / restart["state_xml"]["path"]).read_text().startswith("<?xml")
    assert result["artifact_sha256"]["checkpoint"] == restart["checkpoint"]
    assert result["artifact_sha256"]["state_xml"] == restart["state_xml"]
    assert len(result["frame_sha256"]) == result["trajectory_frames"]
    handoff = json.loads((tmp_path / "npt_to_nvt_handoff.json").read_text())
    assert handoff["contains"] == ["positions_nm", "box_vectors_nm"]
    assert "barostat_parameters" in handoff["excludes"]
    assert handoff["source_manifest"]["checkpoint"] == restart["checkpoint"]
    artifact_manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert artifact_manifest["artifacts"]["density_preparation.json"]["sha256"] == density._file_sha256(tmp_path / "density_preparation.json")
    assert artifact_manifest["artifacts"]["system.xml"]["sha256"] == density._file_sha256(tmp_path / "system.xml")
    assert artifact_manifest["current_restart"]["checkpoint"] == restart["checkpoint"]
    assert artifact_manifest["frame_sha256"] == result["frame_sha256"]
    rows = list(csv.DictReader((tmp_path / "diagnostics.csv").open()))
    assert len({(row["phase"], row["step"]) for row in rows}) == len(rows)
    assert {row["phase"] for row in rows} == {"nvt", "npt"}
    assert all(float(row["volume_nm3"]) > 0 for row in rows)
    assert all(np.isfinite(float(row["density_g_ml"])) for row in rows)
    assert float(rows[-1]["time_ps"]) == pytest.approx(0.003)
    with np.load(tmp_path / "density_preparation.npz") as output:
        assert output["last_positions_nm"].shape == xyz.shape
        assert output["last_box_vectors_nm"].shape == (3, 3)
        assert output["q_ambient"].shape == (12,)


@pytest.mark.parametrize("mutation, pattern", [
    ("nan_xyz", "Initial coordinates"),
    ("bad_box", "periodic box"),
])
def test_early_invalid_geometry_writes_failed_summary_without_restart_or_handoff(tmp_path, monkeypatch, mutation, pattern):
    system, xyz, mapping, model = density_fixture()
    if mutation == "nan_xyz":
        xyz = xyz.copy()
        xyz[0, 0] = np.nan
    else:
        monkeypatch.setattr(density, "_system_default_box_nm", lambda _system: np.diag([6.0, 6.0, 0.0]))
    result = density._run_engine(system, xyz, mapping,
                                 density.DensitySettings(npt_steps=1, report_interval_steps=1,
                                                         checkpoint_interval_steps=1, max_step_batch=1,
                                                         max_wall_seconds=60.),
                                 tmp_path, model=model, platform_name=TEST_PLATFORM,
                                 synthetic_fixture=True,
                                 qualification={"chemical_review": {"status": "pending"}},
                                 provenance={"input_sha256": INPUT_HASHES})
    assert result["status"] == "failed"
    assert pattern in result["reason"]
    assert not result["density_preparation_completed"]
    assert result["artifact_sha256"] == {}
    assert (tmp_path / "density_preparation.json").exists()
    assert not (tmp_path / "system.xml").exists()
    assert not (tmp_path / "current_restart.json").exists()
    assert not (tmp_path / "npt_to_nvt_handoff.json").exists()
    assert not (tmp_path / "npt_to_nvt_handoff.npz").exists()


def test_context_creation_failure_writes_failed_summary_without_restart_or_handoff(tmp_path, monkeypatch):
    system, xyz, mapping, model = density_fixture()

    def fail_context(*_args, **_kwargs):
        raise RuntimeError("simulated context creation failure")

    monkeypatch.setattr(density.mm, "Context", fail_context)
    result = density._run_engine(system, xyz, mapping,
                                 density.DensitySettings(npt_steps=1, report_interval_steps=1,
                                                         checkpoint_interval_steps=1, max_step_batch=1,
                                                         max_wall_seconds=60.),
                                 tmp_path, model=model, platform_name=TEST_PLATFORM,
                                 synthetic_fixture=True,
                                 qualification={"chemical_review": {"status": "pending"}},
                                 provenance={"input_sha256": INPUT_HASHES})
    assert result["status"] == "failed"
    assert "simulated context creation failure" in result["reason"]
    assert not result["density_preparation_completed"]
    assert "system.xml" in result["artifact_sha256"]
    assert "density_preparation.npz" not in result["artifact_sha256"]
    assert "diagnostics.csv" not in result["artifact_sha256"]
    assert "current_restart.json" not in result["artifact_sha256"]
    assert (tmp_path / "density_preparation.json").exists()
    assert (tmp_path / "artifact_manifest.json").exists()
    assert not (tmp_path / "current_restart.json").exists()
    assert not (tmp_path / "npt_to_nvt_handoff.json").exists()
    assert not (tmp_path / "npt_to_nvt_handoff.npz").exists()


@pytest.mark.parametrize("change, pattern", [
    ({"model": "fixed"}, "flexible or isolated"),
    ({"zero_mass": True}, "every particle to be massive"),
    ({"cartesian_restraint": True}, "Cartesian CustomExternalForce"),
])
def test_density_preparation_rejects_non_massive_fixed_or_cartesian_restraints(tmp_path, change, pattern):
    system, xyz, mapping, model = density_fixture(zero_mass=change.get("zero_mass", False),
                                                 cartesian_restraint=change.get("cartesian_restraint", False))
    settings = density.DensitySettings(npt_steps=1, report_interval_steps=1,
                                       checkpoint_interval_steps=1, max_wall_seconds=60.)
    with pytest.raises(ValueError, match=pattern):
        density._run_engine(system, xyz, mapping, settings, tmp_path / pattern.replace(" ", "_"),
                            model=change.get("model", model), platform_name=TEST_PLATFORM,
                            synthetic_fixture=True,
                            qualification={"chemical_review": {"status": "pending"}},
                            provenance={})


def test_resume_rejects_reseed_before_context_or_minimization(tmp_path):
    system, xyz, mapping, model = density_fixture()
    (tmp_path / "current_restart.json").write_text(json.dumps({"checkpoint": {"path": "missing"}}))
    (tmp_path / "density_preparation.json").write_text(json.dumps({
        "status": "budget_limited",
        "restart_seed": 123,
        "completed_by_phase": {"nvt": 0, "npt": 0},
    }))
    settings = density.DensitySettings(seed=456, npt_steps=1, report_interval_steps=1,
                                       checkpoint_interval_steps=1)
    with pytest.raises(ValueError, match="Resume seed mismatch"):
        density._run_engine(system, xyz, mapping, settings, tmp_path, model=model,
                            platform_name=TEST_PLATFORM, synthetic_fixture=True,
                            qualification={"chemical_review": {"status": "pending"}},
                            provenance={})


def test_interrupted_checkpoint_resume_continues_without_duplicate_frames(tmp_path, monkeypatch):
    system, xyz, mapping, model = density_fixture()
    settings = density.DensitySettings(npt_steps=3, report_interval_steps=1,
                                       checkpoint_interval_steps=1, max_step_batch=1,
                                       max_wall_seconds=.5)
    calls = {"n": 0}

    def clock():
        calls["n"] += 1
        return 1. if calls["n"] > 6 else 0.

    monkeypatch.setattr(density.time, "monotonic", clock)
    first = density._run_engine(system, xyz, mapping, settings, tmp_path, model=model,
                                platform_name=TEST_PLATFORM, synthetic_fixture=True,
                                qualification={"chemical_review": {"status": "pending"}},
                                provenance={"input_sha256": INPUT_HASHES})
    assert first["status"] == "budget_limited"
    assert first["completed_by_phase"]["npt"] == 1
    restart_before = json.loads((tmp_path / "current_restart.json").read_text())
    assert restart_before["completed_by_phase"] == {"nvt": 0, "npt": 1}
    assert restart_before["input_hashes_verified"] == INPUT_HASHES
    assert restart_before["trajectory_npz"]["sha256"] == density._file_sha256(tmp_path / "density_preparation.npz")
    assert restart_before["frame_sha256"]
    stale_summary = json.loads((tmp_path / "density_preparation.json").read_text())
    stale_summary["completed_by_phase"] = {"nvt": 0, "npt": 0}
    (tmp_path / "density_preparation.json").write_text(json.dumps(stale_summary))
    resumed = density._run_engine(system, xyz, mapping,
                                  density.DensitySettings(npt_steps=3, report_interval_steps=1,
                                                          checkpoint_interval_steps=1, max_step_batch=1,
                                                          max_wall_seconds=60.),
                                  tmp_path, model=model, platform_name=TEST_PLATFORM,
                                  synthetic_fixture=True,
                                  qualification={"chemical_review": {"status": "pending"}},
                                  provenance={"input_sha256": INPUT_HASHES})
    assert resumed["status"] == "density_completed", resumed.get("reason")
    assert resumed["resume_count"] == 1
    assert resumed["completed_by_phase"] == {"nvt": 0, "npt": 3}
    restart_after = json.loads((tmp_path / "current_restart.json").read_text())
    assert restart_after["checkpoint"]["sha256"] != restart_before["checkpoint"]["sha256"]
    rows = list(csv.DictReader((tmp_path / "diagnostics.csv").open()))
    assert sorted((row["phase"], int(row["step"])) for row in rows) == [("npt", 0), ("npt", 1), ("npt", 2), ("npt", 3)]
    assert len({(row["phase"], row["step"]) for row in rows}) == len(rows)
    with np.load(tmp_path / "density_preparation.npz") as output:
        assert output["frame_time_phase_step"][:, 2].tolist() == [1, 2, 3]
        assert len(output["core_displacement_nm"]) == len(restart_after["frame_sha256"])


def test_resume_rejects_input_or_physics_contract_mismatch_without_overwrite(tmp_path, monkeypatch):
    system, xyz, mapping, model = density_fixture()
    calls = {"n": 0}

    def clock():
        calls["n"] += 1
        return 1. if calls["n"] > 6 else 0.

    monkeypatch.setattr(density.time, "monotonic", clock)
    first = density._run_engine(
        system, xyz, mapping,
        density.DensitySettings(npt_steps=2, report_interval_steps=1, checkpoint_interval_steps=1,
                                max_step_batch=1, max_wall_seconds=.5),
        tmp_path, model=model, platform_name=TEST_PLATFORM, synthetic_fixture=True,
        qualification={"chemical_review": {"status": "pending"}},
        provenance={"input_sha256": INPUT_HASHES},
    )
    assert first["status"] == "budget_limited"
    restart_text = (tmp_path / "current_restart.json").read_text()
    system_hash = density._file_sha256(tmp_path / "system.xml")

    with pytest.raises(ValueError, match="input hash mismatch"):
        density._run_engine(
            system, xyz, mapping,
            density.DensitySettings(npt_steps=2, report_interval_steps=1, checkpoint_interval_steps=1,
                                    max_step_batch=1, max_wall_seconds=60.),
            tmp_path, model=model, platform_name=TEST_PLATFORM, synthetic_fixture=True,
            qualification={"chemical_review": {"status": "pending"}},
            provenance={"input_sha256": {**INPUT_HASHES, "mapping": "d" * 64}},
        )
    assert (tmp_path / "current_restart.json").read_text() == restart_text
    assert density._file_sha256(tmp_path / "system.xml") == system_hash

    with pytest.raises(ValueError, match="settings mismatch"):
        density._run_engine(
            system, xyz, mapping,
            density.DensitySettings(npt_steps=2, report_interval_steps=1, checkpoint_interval_steps=1,
                                    max_step_batch=1, max_wall_seconds=60., gauge_k=2000.),
            tmp_path, model=model, platform_name=TEST_PLATFORM, synthetic_fixture=True,
            qualification={"chemical_review": {"status": "pending"}},
            provenance={"input_sha256": INPUT_HASHES},
        )
    assert (tmp_path / "current_restart.json").read_text() == restart_text
    assert density._file_sha256(tmp_path / "system.xml") == system_hash


@pytest.mark.parametrize("crash_label", ["after_generation_diagnostics", "after_generation_npz", "before_restart_pointer"])
def test_crash_before_pointer_publication_resumes_from_last_valid_generation(tmp_path, monkeypatch, crash_label):
    system, xyz, mapping, model = density_fixture()
    base_settings = density.DensitySettings(npt_steps=1, report_interval_steps=1,
                                            checkpoint_interval_steps=1, max_step_batch=1,
                                            max_wall_seconds=60.)
    first = density._run_engine(system, xyz, mapping, base_settings, tmp_path, model=model,
                                platform_name=TEST_PLATFORM, synthetic_fixture=True,
                                qualification={"chemical_review": {"status": "pending"}},
                                provenance={"input_sha256": INPUT_HASHES})
    assert first["status"] == "density_completed"
    (tmp_path / "density_preparation.json").unlink()
    old_pointer_text = (tmp_path / "current_restart.json").read_text()
    old_pointer = json.loads(old_pointer_text)
    old_top_npz_hash = density._file_sha256(tmp_path / "density_preparation.npz")
    old_top_csv_hash = density._file_sha256(tmp_path / "diagnostics.csv")

    class InjectedCrash(BaseException):
        pass

    seen = {"armed": True}

    def crash_hook(label):
        if seen["armed"] and label == crash_label:
            seen["armed"] = False
            raise InjectedCrash(label)

    monkeypatch.setattr(density, "_TEST_CRASH_HOOK", crash_hook)
    with pytest.raises(InjectedCrash):
        density._run_engine(system, xyz, mapping,
                            density.DensitySettings(npt_steps=3, report_interval_steps=1,
                                                    checkpoint_interval_steps=1, max_step_batch=1,
                                                    max_wall_seconds=60.),
                            tmp_path, model=model, platform_name=TEST_PLATFORM,
                            synthetic_fixture=True,
                            qualification={"chemical_review": {"status": "pending"}},
                            provenance={"input_sha256": INPUT_HASHES})
    assert (tmp_path / "current_restart.json").read_text() == old_pointer_text
    assert density._file_sha256(tmp_path / old_pointer["trajectory_npz"]["path"]) == old_pointer["trajectory_npz"]["sha256"]
    assert density._file_sha256(tmp_path / old_pointer["diagnostics_csv"]["path"]) == old_pointer["diagnostics_csv"]["sha256"]
    assert density._file_sha256(tmp_path / "density_preparation.npz") == old_top_npz_hash
    assert density._file_sha256(tmp_path / "diagnostics.csv") == old_top_csv_hash

    monkeypatch.setattr(density, "_TEST_CRASH_HOOK", None)
    resumed = density._run_engine(system, xyz, mapping,
                                  density.DensitySettings(npt_steps=3, report_interval_steps=1,
                                                          checkpoint_interval_steps=1, max_step_batch=1,
                                                          max_wall_seconds=60.),
                                  tmp_path, model=model, platform_name=TEST_PLATFORM,
                                  synthetic_fixture=True,
                                  qualification={"chemical_review": {"status": "pending"}},
                                  provenance={"input_sha256": INPUT_HASHES})
    assert resumed["status"] == "density_completed", resumed.get("reason")
    assert resumed["completed_by_phase"] == {"nvt": 0, "npt": 3}
    rows = list(csv.DictReader((tmp_path / "diagnostics.csv").open()))
    assert sorted((row["phase"], int(row["step"])) for row in rows) == [("npt", 0), ("npt", 1), ("npt", 2), ("npt", 3)]
    assert len({(row["phase"], row["step"]) for row in rows}) == len(rows)
    pointer = json.loads((tmp_path / "current_restart.json").read_text())
    assert pointer["completed_by_phase"] == {"nvt": 0, "npt": 3}
    assert pointer["trajectory_npz"]["path"].startswith("generations/")
    with np.load(tmp_path / pointer["trajectory_npz"]["path"]) as output:
        assert output["frame_time_phase_step"][:, 2].tolist() == [1, 2, 3]


def test_resume_rejects_phase_counter_time_mismatch(tmp_path):
    system, xyz, mapping, model = density_fixture()
    first = density._run_engine(system, xyz, mapping,
                                density.DensitySettings(npt_steps=1, report_interval_steps=1,
                                                        checkpoint_interval_steps=1, max_step_batch=1,
                                                        max_wall_seconds=60.),
                                tmp_path, model=model, platform_name=TEST_PLATFORM,
                                synthetic_fixture=True,
                                qualification={"chemical_review": {"status": "pending"}},
                                provenance={"input_sha256": INPUT_HASHES})
    assert first["status"] == "density_completed"
    (tmp_path / "density_preparation.json").unlink()
    pointer_path = tmp_path / "current_restart.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["completed_by_phase"] = {"nvt": 0, "npt": 0}
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="phase counters"):
        density._run_engine(system, xyz, mapping,
                            density.DensitySettings(npt_steps=2, report_interval_steps=1,
                                                    checkpoint_interval_steps=1, max_step_batch=1,
                                                    max_wall_seconds=60.),
                            tmp_path, model=model, platform_name=TEST_PLATFORM,
                            synthetic_fixture=True,
                            qualification={"chemical_review": {"status": "pending"}},
                            provenance={"input_sha256": INPUT_HASHES})


@pytest.mark.parametrize("manifest, pattern", [
    (None, "salt_protonation_manifest"),
    ({"status": "pass"}, "bound_input_sha256"),
    ({"status": "pass", "bound_input_sha256": {"prmtop": "a" * 64, "inpcrd": "b" * 64, "mapping": "0" * 64}}, "bound hash mismatch"),
])
def test_production_label_requires_hash_bound_salt_protonation_manifest(tmp_path, manifest, pattern):
    system, xyz, mapping, model = density_fixture(isolated=True)
    settings = density.DensitySettings(npt_steps=1, report_interval_steps=1,
                                       checkpoint_interval_steps=1)
    qualification = {"chemical_review": {"status": density.pilot.CHEMICAL_REVIEW_PASS}}
    if manifest is not None:
        qualification["salt_protonation_manifest"] = manifest
    with pytest.raises(ValueError, match=pattern):
        density._run_engine(system, xyz, mapping, settings, tmp_path, model=model,
                            platform_name=TEST_PLATFORM, synthetic_fixture=False,
                            chemical_geometry=density.pilot.ChemicalGeometry(
                                particle_count=len(xyz),
                                alpha_centers=(),
                                alpha_labels=(),
                                alpha_hydrogen_centers=(),
                                beta_centers=(),
                                beta_labels=(),
                                peptide_pairs=(),
                                peptide_labels=(),
                            ),
                            qualification=qualification,
                            provenance={"input_sha256": {"prmtop": "a" * 64,
                                                          "inpcrd": "b" * 64,
                                                          "mapping": "c" * 64}})


@pytest.mark.parametrize("platform_name", ["OpenCL", "CUDA"])
@pytest.mark.parametrize("precision, disable_pme_stream", [
    ("double", False),
    ("mixed", True),
    ("mixed", False),
])
def test_real_gpu_production_requires_double_and_disabled_pme_stream_before_context(
    tmp_path, monkeypatch, platform_name, precision, disable_pme_stream
):
    system, xyz, mapping, model = density_fixture(isolated=True)
    settings = density.DensitySettings(npt_steps=1, report_interval_steps=1,
                                       checkpoint_interval_steps=1,
                                       disable_pme_stream=disable_pme_stream)
    qualification = {
        "chemical_review": {"status": density.pilot.CHEMICAL_REVIEW_PASS},
        "salt_protonation_manifest": {
            "status": "pass",
            "bound_input_sha256": dict(INPUT_HASHES),
        },
    }

    def fail_context(*_args, **_kwargs):
        raise AssertionError("Context must not be created for rejected GPU density settings")

    monkeypatch.setattr(density.mm, "Context", fail_context)
    with pytest.raises(ValueError, match="--precision double.*--disable-pme-stream"):
        density._run_engine(
            system, xyz, mapping, settings, tmp_path, model=model,
            platform_name=platform_name, synthetic_fixture=False, precision=precision,
            chemical_geometry=density.pilot.ChemicalGeometry(
                particle_count=len(xyz),
                alpha_centers=(),
                alpha_labels=(),
                alpha_hydrogen_centers=(),
                beta_centers=(),
                beta_labels=(),
                peptide_pairs=(),
                peptide_labels=(),
            ),
            qualification=qualification, provenance={"input_sha256": INPUT_HASHES},
        )
    assert not (tmp_path / "system.xml").exists()
    assert not (tmp_path / "current_restart.json").exists()
    assert not (tmp_path / "density_preparation.json").exists()


def test_cli_shared_loader_returns_density_settings_and_provenance(tmp_path, monkeypatch, capsys):
    prmtop, inpcrd, mapping_path, qualification_path = [tmp_path / name for name in
                                                        ("toy.prmtop", "toy.inpcrd", "mapping.json", "qualification.json")]
    for index, path in enumerate((prmtop, inpcrd, mapping_path, qualification_path)):
        path.write_text(f"fixture-{index}")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"density_preparation": {"npt_steps": 1, "report_interval_steps": 1,
                                                               "checkpoint_interval_steps": 1}}))
    provenance = {"input_sha256": {"prmtop": density._file_sha256(prmtop),
                                   "inpcrd": density._file_sha256(inpcrd),
                                   "mapping": density._file_sha256(mapping_path)}}

    def fake_pilot_loader(*_args, **_kwargs):
        return {"system": object(), "xyz": np.zeros((4, 3)), "mapping": {}, "settings": object(),
                "qualification": {}, "provenance": provenance, "chemical_geometry": None}

    monkeypatch.setattr(density.pilot, "load_qualified_inputs", fake_pilot_loader)
    loaded = density.load_qualified_inputs(prmtop, inpcrd, mapping_path, config_path,
                                           qualification_path, model="isolated")
    assert isinstance(loaded["settings"], density.DensitySettings)
    assert loaded["provenance"]["density_runner_sha256"] == density._file_sha256(density.ROOT / "scripts" / "run_atomistic_density_preparation.py")
    assert loaded["provenance"]["input_sha256"]["prmtop"] == density._file_sha256(prmtop)

    def fake_run(*_args, **_kwargs):
        return {"status": "density_completed", "density_preparation_completed": True,
                "equilibrium_certified": False, "production_ready": False,
                "completed_by_phase": {"nvt": 0, "npt": 1}, "elapsed_wall_seconds": 0.}

    monkeypatch.setattr(density, "run", fake_run)
    code = density.main(["--prmtop", str(prmtop), "--inpcrd", str(inpcrd), "--mapping", str(mapping_path),
                         "--qualification", str(qualification_path), "--config", str(config_path),
                         "--output-dir", str(tmp_path / "cli"), "--model", "isolated",
                         "--platform", TEST_PLATFORM, "--offline"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "density_completed"
