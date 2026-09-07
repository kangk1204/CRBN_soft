import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import atomistic_zero_force_diagnostics as diag
from scripts import directional_mechanics as dm


def sha(path: Path) -> str:
    return diag.sha256_file(path)


def reference_and_q():
    rng = np.random.default_rng(20260907)
    reference = rng.normal(size=(269, 3))
    reference[:, 0] += np.linspace(0.0, 5.0, 269)
    basis = dm.internal_basis(reference)
    q = basis[:, 0]
    return reference, q


def write_generation(root: Path, *, frames=8, force=0.0, times=None, phase="zero_equilibration", synthetic=True):
    reference, q = reference_and_q()
    if times is None:
        times = np.arange(frames, dtype=float)
    else:
        times = np.asarray(times, dtype=float)
        frames = len(times)
    core_indices = np.arange(269, dtype=int)
    basis = dm.internal_basis(reference)
    flat = []
    for i in range(frames):
        flat.append((0.01 * i) * q + (0.002 * np.sin(i)) * basis[:, 1])
    flat = np.asarray(flat, dtype=float)
    displacement = flat.reshape(frames, 269, 3)
    gen = root / "generation_000000000004_test"
    gen.mkdir(parents=True)
    (root / "segment_system.xml").write_text("<System/>")
    (gen / "segment.chk").write_bytes(b"checkpoint")
    (gen / "segment_state.xml").write_text("<State/>")
    closure = flat @ q
    with (gen / "response_observations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diag.CSV_COLUMNS)
        writer.writeheader()
        for t, c in zip(times, closure):
            writer.writerow({"model": "isolated", "replicate": "r1", "force_kj_mol_nm": force, "time_ps": float(t), "closure_nm": float(c)})
    with (gen / "segment_observables.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            core_displacement_nm=displacement,
            time_ps=times,
            reference_nm=reference,
            q_ambient=q,
            core_indices=core_indices,
            force_kj_mol_nm=float(force),
        )
    manifest = {
        "schema_version": "1.0",
        "status": "segment_complete",
        "segment_complete": True,
        "response_converged": False,
        "production_ready": False,
        "phase": phase,
        "model": "isolated",
        "replicate_id": "r1",
        "force_kj_mol_nm": float(force),
        "current_generation": gen.name,
        "generation_sha256": {name: sha(gen / name) for name in diag.GENERATION_ARTIFACTS},
        "system_xml_sha256": sha(root / "segment_system.xml"),
        "data_input_hashes": {"prmtop": "a" * 64, "inpcrd": "b" * 64, "mapping": "c" * 64},
        "stage_provenance_hashes": {},
        "source_provenance": {"response_runner_sha256": "d" * 64},
        "topology": {"provided": False},
        "mapping_sha256": "e" * 64,
        "core_indices_sha256": diag.canonical_hash(core_indices.astype(int).tolist()),
        "reference_nm_sha256": diag.canonical_hash(reference.tolist()),
        "q_ambient_sha256": diag.canonical_hash(q.tolist()),
        "gauge_sha256": "3" * 64,
        "body_sha256": "4" * 64,
        "measurement_identity": {"provided": not synthetic, "path": "identity.json"} if not synthetic else {"provided": False},
        "synthetic_fixture": synthetic,
    }
    (root / "segment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def config(path: Path):
    path.write_text(json.dumps({"temperature_K": 300.0}))
    return path


def test_diagnostic_exports_from_cpu_toy_response(tmp_path):
    segment = write_generation(tmp_path / "segment")
    out = tmp_path / "out"
    result = diag.run(segment, config(tmp_path / "config.json"), out, equilibration_ps=0.0, offline=True)
    assert result["status"] == "descriptive_zero_force_diagnostics"
    assert result["manifest_zero_force"]["force_kj_mol_nm"] == 0.0
    assert result["real_or_synthetic_role"]["role"] == "synthetic"
    assert result["diagnostics"]["n_retained_frames"] == 8
    assert result["diagnostics"]["ess"].keys() == {"Q", "Q_squared", "trace_proxy_internal_centered_norm_squared"}
    assert result["diagnostics"]["half_comparison"]["status"] == "descriptive_only"
    assert (out / "zero_force_diagnostics.json").is_file()
    assert (out / "zero_force_timeseries.npz").is_file()


def test_tampered_generation_hash_is_rejected(tmp_path):
    segment = write_generation(tmp_path / "segment")
    gen = json.loads((segment / "segment_manifest.json").read_text())["current_generation"]
    with (segment / gen / "response_observations.csv").open("a") as handle:
        handle.write("#tamper\n")
    with pytest.raises(diag.DiagnosticInputError, match="hash mismatch"):
        diag.run(segment, config(tmp_path / "config.json"), tmp_path / "out", equilibration_ps=0.0)


def test_nonzero_force_is_rejected(tmp_path):
    segment = write_generation(tmp_path / "segment", force=1.0)
    with pytest.raises(diag.DiagnosticInputError, match="finite-force"):
        diag.run(segment, config(tmp_path / "config.json"), tmp_path / "out", equilibration_ps=0.0)


def test_nonuniform_times_are_rejected(tmp_path):
    segment = write_generation(tmp_path / "segment", times=[0.0, 1.0, 2.5, 3.0, 4.0])
    with pytest.raises(diag.DiagnosticInputError, match="equally spaced"):
        diag.run(segment, config(tmp_path / "config.json"), tmp_path / "out", equilibration_ps=0.0)


def test_short_retained_data_is_unevaluable_but_preserved(tmp_path):
    segment = write_generation(tmp_path / "segment", frames=5)
    result = diag.run(segment, config(tmp_path / "config.json"), tmp_path / "out", equilibration_ps=2.0)
    assert result["status"] == "unevaluable_insufficient_data"
    assert result["diagnostics"]["n_retained_frames"] == 3
    assert result["diagnostics"]["eligible_for_response_comparison"] is False


def test_technical_pilot_directory_is_rejected(tmp_path):
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "technical_pilot.json").write_text("{}")
    with pytest.raises(diag.DiagnosticInputError, match="technical_pilot"):
        diag.run(pilot, config(tmp_path / "config.json"), tmp_path / "out", equilibration_ps=0.0)


def test_real_role_equivalent_zero_force_does_not_require_measurement_identity(tmp_path):
    segment = write_generation(tmp_path / "segment", synthetic=False)
    manifest = json.loads((segment / "segment_manifest.json").read_text())
    manifest["measurement_identity"] = {"provided": False}
    (segment / "segment_manifest.json").write_text(json.dumps(manifest))
    result = diag.run(segment, config(tmp_path / "config.json"), tmp_path / "out", equilibration_ps=0.0)
    assert result["real_or_synthetic_role"]["role"] == "real"
    assert result["real_or_synthetic_role"]["measurement_identity_required_for_zero_force_diagnostics"] is False
    assert result["endpoint_binding_verification"]["status"] == "pass"
    assert result["production_ready"] is False
    assert result["scientific_certificate"] is False
    assert result["equilibrium_certified"] is False


def test_legacy_missing_synthetic_flag_is_role_unknown_but_diagnosable(tmp_path):
    segment = write_generation(tmp_path / "segment")
    manifest = json.loads((segment / "segment_manifest.json").read_text())
    manifest.pop("synthetic_fixture")
    (segment / "segment_manifest.json").write_text(json.dumps(manifest))
    result = diag.run(segment, config(tmp_path / "config.json"), tmp_path / "out", equilibration_ps=0.0)
    assert result["real_or_synthetic_role"]["role"] == "unknown_missing_synthetic_fixture_flag"
    assert result["endpoint_binding_verification"]["status"] == "pass"


def test_actual_run_segment_reference_generation_cli_compatible(tmp_path):
    pytest.importorskip("openmm")
    import openmm as mm

    from scripts import run_atomistic_response_segment as segment

    reference, q = reference_and_q()
    system = mm.System()
    zero = mm.CustomExternalForce("0")
    for _ in range(269):
        system.addParticle(12.0)
        zero.addParticle(system.getNumParticles() - 1, [])
    system.addForce(zero)
    mapping = {
        "core_indices": list(range(269)),
        "reference_nm": reference.tolist(),
        "q": q.tolist(),
        "ddb1_atom_indices": [],
        "protein_ca_indices": list(range(269)),
    }
    segment_dir = tmp_path / "actual_segment"
    result = segment.run_segment(
        system=system,
        xyz=reference.copy(),
        mapping=mapping,
        settings=segment.SegmentSettings(
            steps=5,
            report_interval_steps=1,
            max_step_batch=1,
            max_wall_seconds=30.0,
            master_seed=20260907,
        ),
        output_dir=segment_dir,
        phase="zero_equilibration",
        model="isolated",
        replicate_id="r1",
        force_h=0.0,
        platform_name="Reference",
        synthetic_fixture=True,
        input_hashes={"prmtop": "a" * 64, "inpcrd": "b" * 64, "mapping": "c" * 64},
    )
    assert result["status"] == "segment_complete"
    expected_role = "synthetic" if result.get("synthetic_fixture") is True else "unknown_missing_synthetic_fixture_flag"
    out = tmp_path / "actual_diag"
    code = diag.main([
        "--input-dir", str(segment_dir),
        "--config", str(config(tmp_path / "config.json")),
        "--output-dir", str(out),
        "--offline",
        "--equilibration-ps", "0",
    ])
    assert code == 0
    payload = json.loads((out / "zero_force_diagnostics.json").read_text())
    assert payload["status"] == "descriptive_zero_force_diagnostics"
    assert payload["real_or_synthetic_role"]["role"] == expected_role
    assert payload["endpoint_binding_verification"]["status"] == "pass"
    assert payload["array_csv_validation"]["max_abs_csv_actual_Q_delta_nm"] < 1e-10
    assert payload["production_ready"] is False
    assert payload["scientific_certificate"] is False
    assert payload["equilibrium_certified"] is False
