"""Reference-only synthetic validation; these tests do not run a GPU."""
import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("openmm")
from scripts import benchmark_atomistic_precision as benchmark
from scripts import run_atomistic_technical_pilot as pilot
from test_atomistic_technical_pilot import tiny_fixture


def loaded_fixture():
    _, system, xyz, mapping = tiny_fixture()
    return {"system": system, "xyz": xyz, "mapping": mapping,
            "settings": pilot.Settings(steps=10, max_wall_seconds=60.),
            "chemical_geometry": None, "qualification": {"chemical_review": {"status": pilot.CHEMICAL_REVIEW_PASS}},
            "provenance": {"role": "synthetic_reference_precision_fixture"}}


@pytest.fixture
def reference_snapshots():
    loaded = loaded_fixture()
    records, arrays = {}, {}
    for model in benchmark.BENCHMARK_MODELS:
        record, snapshot = benchmark.capture_static(loaded, model, "Reference", "double", synthetic_fixture=True)
        assert record["pass"], record
        # Duplicate only to unit-test comparison arithmetic. No mixed/GPU claim.
        for label in ("double", "mixed"):
            records[f"{label}_{model}"] = copy.deepcopy(record)
            arrays[f"{label}_{model}"] = {key: value.copy() for key, value in snapshot.items()}
    return loaded, records, arrays


def test_reference_static_analytic_gauge_probe_and_boundary_invariants(reference_snapshots):
    loaded, records, arrays = reference_snapshots
    result = benchmark.compare_static_snapshots(records, arrays, loaded["mapping"])
    assert result["pass"], result
    assert len({r["input_coordinate_array_sha256"] for r in records.values()}) == 1
    for record in records.values():
        assert record["gauge_analytic"]["pass"] and record["probe_analytic"]["pass"]
        assert record["probe_analytic"]["forces"]["reference_rms"] > 0
        assert record["precision"]["effective"] == "unreported_platform_default"
    assert len(result["comparisons"]) == 7


def test_fixed_static_policy_boundary_and_force_failure_are_not_hidden(reference_snapshots):
    loaded, records, arrays = reference_snapshots
    assert benchmark.STATIC_POLICY["energy_difference_per_original_particle_kj_mol_max"] == 1e-4
    assert benchmark.STATIC_POLICY["force_rms_absolute_tolerance_kj_mol_nm"] == 1e-3
    assert benchmark.STATIC_POLICY["force_rms_relative_tolerance"] == 1e-4
    arrays["mixed_flexible"]["forces_kj_mol_nm"][loaded["mapping"]["core_indices"]] += 1.
    result = benchmark.compare_static_snapshots(records, arrays, loaded["mapping"])
    assert not result["pass"]
    assert not result["comparisons"]["precision_flexible"]["core_force"]["pass"]
    assert result["comparisons"]["precision_flexible"]["full_force"]["maximum_absolute_component_difference"] >= 1.


def test_energy_policy_uses_original_particle_count_and_analytic_limits():
    assert benchmark.energy_comparison(-1000, -999.995, 100)["pass"]
    assert not benchmark.energy_comparison(-1000, -999.98, 100)["pass"]
    expected = np.ones((4, 3))
    assert benchmark.analytic_comparison(1., 1.0005, expected, expected + .0005)["pass"]
    assert not benchmark.analytic_comparison(1., 1.01, expected, expected)["pass"]
    assert not benchmark.analytic_comparison(1., 1., expected, expected + .01)["pass"]


@pytest.mark.parametrize("model", benchmark.BENCHMARK_MODELS)
def test_reference_bounded_integration_has_no_minimization_and_keeps_zero_force(tmp_path, monkeypatch, model):
    loaded = loaded_fixture()
    monkeypatch.setattr(pilot.mm.LocalEnergyMinimizer, "minimize", lambda *_args, **_kwargs: pytest.fail("No minimization is allowed in the precision benchmark"))
    record = benchmark.measure_integration(loaded, model, "Reference", "double", None, 5,
                                           time.monotonic() + 60., synthetic_fixture=True)
    assert record["pass"], record
    assert record["completed_steps"] == 5
    assert record["warmup_steps"] == 10
    assert record["minimization_performed"] is False
    assert record["ns_per_day"] > 0
    assert {s["stage"] for s in record["screens"]} == {"initial", "warmup", "measured", "final"}


def test_late_geometry_failure_revokes_benchmark_pass(monkeypatch):
    original = benchmark._integration_screen

    def fail_final(*args):
        row = original(*args)
        if args[-1] == "final":
            row["pass"] = False
        return row

    monkeypatch.setattr(benchmark, "_integration_screen", fail_final)
    record = benchmark.measure_integration(loaded_fixture(), "fixed", "Reference", "double", None, 1,
                                           time.monotonic() + 60., synthetic_fixture=True)
    assert not record["pass"] and record["completed_steps"] == 1


@pytest.mark.parametrize("static_pass,static_only", [(False, False), (True, True)])
def test_static_failure_or_static_only_never_starts_dynamics(tmp_path, monkeypatch, reference_snapshots, static_pass, static_only):
    loaded, records, arrays = reference_snapshots
    paths = [tmp_path / name for name in ("system", "coordinates", "mapping", "config", "qualification")]
    for path in paths:
        path.write_text("synthetic fixture")
    monkeypatch.setattr(pilot, "load_qualified_inputs", lambda *_args, **_kwargs: loaded)

    def synthetic_capture(_loaded, model, _platform, precision, _device, **_kwargs):
        key = f"{precision}_{model}"
        record = copy.deepcopy(records[key])
        if not static_pass:
            record["pass"] = False
        return record, arrays[key]

    monkeypatch.setattr(benchmark, "capture_static", synthetic_capture)
    monkeypatch.setattr(benchmark, "measure_integration", lambda *_args, **_kwargs: pytest.fail("Static gate must stop before integration"))
    report = benchmark.run(*paths, tmp_path / "output", platform_name="OpenCL", static_only=static_only)
    assert report["status"] == ("static_precision_screen_pass" if static_pass else "static_failed_no_integration")
    assert not report["accepted_mixed"] and not report["default_changed"]
    assert not report["integration"]
    saved = json.loads((tmp_path / "output/precision_benchmark.json").read_text())
    assert saved["policy"] == benchmark.STATIC_POLICY
    assert saved["policy"]["later_response_precision_validation_required"] is True


def test_unqualified_or_unsupported_requests_fail_before_engine(tmp_path):
    loaded = loaded_fixture()
    with pytest.raises(ValueError, match="Full topology-derived"):
        benchmark.capture_static(loaded, "flexible", "Reference", "double")
    with pytest.raises(ValueError, match="GPU platform"):
        benchmark.run(*(tmp_path / str(i) for i in range(6)), platform_name="CPU")
    with pytest.raises(ValueError, match="100..1000"):
        benchmark.run(*(tmp_path / str(i) for i in range(6)), steps=1001)


def test_reference_fixed_coordinate_repeats_and_group_sum_are_checked():
    record, _ = benchmark.capture_static(loaded_fixture(), "flexible", "Reference", "double",
                                         synthetic_fixture=True, static_repeats=3)
    diagnostic = record["fixed_coordinate_diagnostics"]
    assert record["pass"] and diagnostic["pass"]
    assert diagnostic["completed_repeats"] == 3
    assert diagnostic["force_h_kj_mol_nm"] == 0
    assert all(row["coordinates_unchanged"] and row["all_groups_vs_sum_of_groups"]["pass"]
               for row in diagnostic["records"])


@pytest.mark.parametrize("corruption", ["late_all_groups", "stable_wrong_all_groups", "coordinates"])
def test_static_repetition_and_separate_groups_reject_corruption(monkeypatch, corruption):
    positions = np.zeros((4, 3))
    expected_force = np.ones((4, 3))
    first_force = expected_force + (1. if corruption == "stable_wrong_all_groups" else 0.)
    calls = []

    def state_force(_context, groups=-1):
        if groups == -1:
            calls.append(1)
            bad = corruption == "stable_wrong_all_groups" or (corruption == "late_all_groups" and len(calls) == 2)
            return 0., expected_force + (1. if bad else 0.)
        return 0., expected_force.copy() if groups == 1 else np.zeros_like(expected_force)

    monkeypatch.setattr(benchmark, "_state_energy_force", state_force)
    context = SimpleNamespace(getState=lambda **_: SimpleNamespace(
        getPositions=lambda **_: (positions + (1. if corruption == "coordinates" else 0.)) * benchmark.unit.nanometer))
    result = benchmark._fixed_coordinate_diagnostics(context, 0., first_force, positions, [0, 1], 4, 4)
    assert not result["pass"] and result["completed_repeats"] == 4
    if corruption == "stable_wrong_all_groups":
        assert all(row["all_groups_vs_first"]["pass"] for row in result["records"])
        assert all(not row["all_groups_vs_sum_of_groups"]["pass"] for row in result["records"])
    elif corruption == "late_all_groups":
        assert result["records"][0]["pass"] and not result["records"][2]["pass"]
        assert result["records"][3]["pass"]  # A later good value never erases the failure.


def test_one_context_diagnostic_never_claims_precision_acceptance(tmp_path, monkeypatch, reference_snapshots):
    loaded, records, arrays = reference_snapshots
    paths = [tmp_path / name for name in ("system", "coordinates", "mapping", "config", "qualification")]
    for path in paths:
        path.write_text("synthetic fixture")
    monkeypatch.setattr(pilot, "load_qualified_inputs", lambda *_args, **_kwargs: loaded)
    calls = []

    def capture(_loaded, model, _platform, precision, _device, **kwargs):
        calls.append((model, precision, kwargs))
        return copy.deepcopy(records[f"{precision}_{model}"]), arrays[f"{precision}_{model}"]

    monkeypatch.setattr(benchmark, "capture_static", capture)
    monkeypatch.setattr(benchmark, "measure_integration", lambda *_args, **_kwargs: pytest.fail("Diagnostic mode cannot integrate"))
    report = benchmark.run(*paths, tmp_path / "diagnostic", diagnostic_only=True,
                           static_repeats=20, disable_pme_stream=True)
    assert len(calls) == 1 and calls[0][:2] == ("flexible", "double")
    assert calls[0][2]["disable_pme_stream"] and calls[0][2]["static_repeats"] == 20
    assert report["status"] == "static_diagnostic_pass"
    assert report["static_comparison"]["pass"] is None
    assert not report["accepted_mixed"] and not report["integration"]
    saved = json.loads((tmp_path / "diagnostic/precision_benchmark.json").read_text())
    assert saved["disable_pme_stream_requested"] and saved["static_repeats_per_context"] == 20
    assert saved["policy"] == benchmark.STATIC_POLICY


def test_diagnostic_cli_serializes_explicit_stream_and_repeat_options(monkeypatch, tmp_path):
    captured = {}

    def run(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "static_diagnostic_pass", "accepted_mixed": False,
                "default_changed": False, "elapsed_wall_seconds": 0.}

    monkeypatch.setattr(benchmark, "run", run)
    args = [flag for name in ("prmtop", "inpcrd", "mapping", "qualification", "output-dir")
            for flag in ("--" + name, str(tmp_path / name))]
    assert benchmark.main(args + ["--diagnostic-only", "--static-repeats", "20", "--disable-pme-stream", "--offline"]) == 0
    assert captured["diagnostic_only"] and captured["disable_pme_stream"]
    assert captured["static_repeats"] == 20


def test_force_snapshot_does_not_alias_a_later_state():
    loaded = loaded_fixture()
    system, xyz, gauge, _ = benchmark._prepared(loaded, "flexible")
    integrator = pilot.mm.VerletIntegrator(.001)
    context = pilot.mm.Context(system, integrator, pilot.mm.Platform.getPlatformByName("Reference"))
    try:
        context.setPositions(xyz)
        _, force = benchmark._state_energy_force(context)
        frozen = force.copy()
        context.setParameter(gauge["probe_parameter"], 1.)
        _, later = benchmark._state_energy_force(context)
        assert not np.shares_memory(force, later)
        assert np.array_equal(force, frozen)
        assert not np.array_equal(force, later)
    finally:
        del context, integrator
