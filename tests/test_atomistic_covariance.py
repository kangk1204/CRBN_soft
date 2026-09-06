"""Analytic and random fixtures test estimators, not CRBN atomistic sampling."""

import json

import numpy as np
import pytest

from scripts import atomistic_covariance as covariance


@pytest.fixture(scope="module")
def reference_and_basis():
    reference = np.random.default_rng(91).normal(size=(269, 3))
    return reference, covariance.dm.internal_basis(reference)


def trajectory(reference_and_basis, n=64):
    reference, basis = reference_and_basis
    z = np.random.default_rng(9).normal(size=(n, 3)) * [0.2, 0.1, 0.3]
    displacement = (z @ basis[:, :3].T).reshape(n, 269, 3)
    return displacement, np.arange(n, dtype=float), reference, basis[:, 0]


def analyze(arrays, **kwargs):
    return covariance.analyze(*arrays, temperature_K=300.0, equilibration_ps=0, **kwargs)


def test_known_covariance_uses_801_denominator_and_nm_units(reference_and_basis):
    reference, basis = reference_and_basis
    n = 32
    time = np.arange(n)
    z = np.column_stack((np.cos(2 * np.pi * time / n), np.sin(2 * np.pi * time / n)))
    z *= np.sqrt(2 * (n - 1) / n) * np.array([0.2, 0.3])
    arrays = ((z @ basis[:, :2].T).reshape(n, 269, 3), time, reference, basis[:, 0])
    result = analyze(arrays)
    estimate = result["estimates"]
    rt = 0.00831446261815324 * 300
    assert estimate["variance_Q_nm2"] == pytest.approx(0.04)
    assert estimate["trace_internal_covariance_nm2"] == pytest.approx(0.13)
    assert estimate["C_nm2_mol_per_kj"] == pytest.approx(0.04 / rt)
    assert estimate["mean_compliance_nm2_mol_per_kj"] == pytest.approx(0.13 / (801 * rt))
    assert estimate["S_dimensionless"] == pytest.approx(801 * 0.04 / 0.13)
    np.testing.assert_allclose(result["timeseries"]["Q_nm"], z[:, 0], atol=1e-14)
    assert result["sample_covariance_rank_upper_bound"] == 31
    assert not result["eligible_for_response_comparison"]
    assert result["covariance_convergence"] == "unverified"


def test_common_rotation_reference_translation_and_gauge_addition_invariance(reference_and_basis):
    arrays = trajectory(reference_and_basis)
    displacement, times, reference, q = arrays
    expected = analyze(arrays)["estimates"]
    rigid = covariance.dm.rigid_basis(reference)
    gauge = np.random.default_rng(45).normal(size=(len(times), 6)) @ rigid.T
    rotation, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))
    transformed = (
        (displacement + gauge.reshape(-1, 269, 3)) @ rotation,
        times,
        reference @ rotation + [5, -7, 9],
        (q.reshape(269, 3) @ rotation).ravel(),
    )
    actual = analyze(transformed)["estimates"]
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=1e-10, abs=1e-14)


def test_gauge_only_has_no_resolved_internal_covariance(reference_and_basis):
    reference, basis = reference_and_basis
    gauge = np.random.default_rng(45).normal(size=(64, 6)) @ covariance.dm.rigid_basis(reference).T
    result = analyze((gauge.reshape(-1, 269, 3), np.arange(64), reference, basis[:, 0]))
    assert result["status"] == "unresolved_internal_variance"
    assert result["estimates"]["trace_internal_covariance_nm2"] < 1e-24
    assert result["estimates"]["S_dimensionless"] is None
    assert all(value == 0 for value in result["effective_samples"].values())


def test_units_scale_quadratically_but_S_does_not(reference_and_basis):
    arrays = trajectory(reference_and_basis)
    initial = analyze(arrays)["estimates"]
    scaled = analyze((2 * arrays[0], *arrays[1:]))["estimates"]
    for key in initial:
        assert scaled[key] == pytest.approx(initial[key] * (1 if key == "S_dimensionless" else 4))


def test_scalar_ESS_and_conditional_blocks_never_grant_covariance_convergence(reference_and_basis):
    result = analyze(trajectory(reference_and_basis, n=1024), block_size_frames=64)
    assert set(result["effective_samples"]) == {
        "Q",
        "Q_squared",
        "Q_centered_squared",
        "internal_norm_squared",
        "internal_centered_norm_squared",
    }
    uncertainty = result["uncertainty"]
    assert uncertainty["status"] == "conditional_technical_estimate"
    assert uncertainty["n_complete_blocks"] == 16
    assert all(value > 0 for value in uncertainty["conditional_technical_se"].values())
    assert uncertainty["independent_replicate_uncertainty"] == "not_available_from_one_trajectory"
    assert result["covariance_convergence"] == "unverified"
    assert not result["eligible_for_response_comparison"]


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "nan",
        "wrong_shape",
        "irregular",
        "reversed",
        "nonunit_q",
        "gauge_q",
        "degenerate_reference",
    ],
)
def test_malformed_inputs_rejected(reference_and_basis, mutation):
    arrays = list(trajectory(reference_and_basis))
    arrays = [value.copy() for value in arrays]
    if mutation == "empty":
        arrays[0], arrays[1] = arrays[0][:0], arrays[1][:0]
    elif mutation == "nan":
        arrays[0][0, 0, 0] = np.nan
    elif mutation == "wrong_shape":
        arrays[0] = arrays[0][:, :268]
    elif mutation == "irregular":
        arrays[1][1] = 1.5
    elif mutation == "reversed":
        arrays[1] = arrays[1][::-1]
    elif mutation == "nonunit_q":
        arrays[3] *= 2
    elif mutation == "gauge_q":
        arrays[3] = covariance.dm.rigid_basis(arrays[2])[:, 0]
    elif mutation == "degenerate_reference":
        arrays[2][:] = 0
    with pytest.raises(ValueError):
        analyze(arrays)


def test_equilibration_and_thermal_validation(reference_and_basis):
    arrays = trajectory(reference_and_basis)
    for temperature, discard in [(0, 0), (-1, 0), (300, -1), (300, 1000)]:
        with pytest.raises(ValueError):
            covariance.analyze(*arrays, temperature_K=temperature, equilibration_ps=discard)
    result = covariance.analyze(*arrays, temperature_K=300, equilibration_ps=10)
    assert result["n_discarded_frames"] == 10
    np.testing.assert_equal(result["timeseries"]["time_ps"], arrays[1][10:])


def test_cli_local_schema_outputs_and_nonzero_force_rejection(reference_and_basis, tmp_path):
    arrays = trajectory(reference_and_basis)
    source, config = tmp_path / "synthetic.npz", tmp_path / "config.json"
    np.savez(source, **dict(zip(covariance.REQUIRED_ARRAYS, arrays)), force_kj_mol_nm=0.0)
    config.write_text(json.dumps({"temperature_K": 300}))
    args = [
        "--input",
        str(source),
        "--config",
        str(config),
        "--output-dir",
        str(tmp_path / "verification"),
        "--offline",
    ]
    with pytest.raises(SystemExit) as exc:
        covariance.main(args)
    assert exc.value.code == 2
    assert not (tmp_path / "verification").exists()
    assert covariance.main([*args, "--equilibration-ps", "0"]) == 0
    result = json.loads((tmp_path / "verification/covariance_analysis.json").read_text())
    assert not result["eligible_for_response_comparison"]
    assert result["input_declares_zero_force"]
    with np.load(tmp_path / "verification/covariance_timeseries.npz", allow_pickle=False) as output:
        assert output["Q_nm"].shape == (64,)
    np.savez(source, **dict(zip(covariance.REQUIRED_ARRAYS, arrays)), force_kj_mol_nm=1.0)
    with pytest.raises(ValueError, match="scalar zero"):
        covariance.run(source, config, tmp_path / "reject", equilibration_ps=0)
    assert not (tmp_path / "reject").exists()
