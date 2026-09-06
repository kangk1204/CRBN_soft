"""Synthetic method fixtures only; these are not atomistic MD measurements."""

import csv
import json

import numpy as np
import pytest

from scripts import atomistic_response_analysis as response


@pytest.fixture
def config():
    return {
        "models": ["isolated"],
        "production_gate": {
            "independent_replicates": 3,
            "minimum_effective_samples_per_replicate": 50,
            "maximum_relative_half_trajectory_difference": 0.20,
            "magnitude_response_half_vs_full_force_relative_difference": 0.20,
        },
    }


def gaussian_rows(replicates=3, n=2000, susceptibility=0.4, cubic=0.0, seed=31):
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(replicates):
        for force in (-2.0, -1.0, 0.0, 1.0, 2.0):
            noise = rng.normal(0, 0.15, n)
            for time, value in enumerate(noise + susceptibility * force + cubic * force**3):
                rows.append(
                    dict(
                        model="isolated",
                        replicate=str(rep),
                        force_kj_mol_nm=force,
                        time_ps=float(time),
                        closure_nm=float(value),
                    )
                )
    return rows


def test_stationary_gaussian_recovers_known_linear_response(config):
    result = response.analyze(gaussian_rows(), config, equilibration_ps=100)
    model = result["models"][0]
    assert result["status"] == "eligible_for_response_comparison"
    assert model["mean_chi_nm2_mol_per_kj"] == pytest.approx(0.4, abs=0.01)
    chi = [row["chi_nm2_mol_per_kj"] for row in result["replicates"]]
    assert model["between_replicate_se"] == pytest.approx(np.std(chi, ddof=1) / np.sqrt(3))
    assert model["n_independent_replicate_ids"] == 3
    assert all(row["n_discarded_frames"] == 100 for row in result["series"])
    assert all(
        row["zero_force_variance_nm2"] == pytest.approx(0.15**2, rel=0.12)
        for row in result["replicates"]
    )
    assert "not_estimated" in result["mean_compliance_and_normalized_S"]


def test_frames_do_not_replace_independent_replicates(config):
    result = response.analyze(gaussian_rows(replicates=1, n=4000), config, equilibration_ps=0)
    assert result["status"] == "insufficient_sampling"
    assert result["models"][0]["between_replicate_se"] is None
    assert result["replicates"][0]["sampling_se_upper_bound"] > 0


def test_resolved_negative_conjugate_response_is_physically_ineligible(config):
    result = response.analyze(gaussian_rows(susceptibility=-0.4), config, equilibration_ps=0)
    assert result["status"] == "physical_consistency_gate_failed"
    assert not result["overall_comparison_ready"]
    assert result["models"][0]["status"] == "physical_consistency_gate_failed"
    assert all(row["physical_consistency_gate"]["status"] == "fail" for row in result["replicates"])
    assert all(row["linearity_gate"]["status"] == "pass" for row in result["replicates"])


def test_missing_configured_models_prevents_overall_comparison(config):
    config["models"] = ["isolated", "fixed", "rigid", "flexible"]
    result = response.analyze(gaussian_rows(), config, equilibration_ps=0)
    assert result["status"] == "partial_models"
    assert result["missing_models"] == ["fixed", "flexible", "rigid"]
    assert not result["overall_comparison_ready"]
    assert result["models"][0]["status"] == "eligible_for_response_comparison"


def test_fdt_diagnostic_has_correct_units_and_never_claims_consistency(config):
    config["temperature_K"] = 300.0
    result = response.analyze(gaussian_rows(), config, equilibration_ps=0)
    assert "FDT_consistency" in result["unverified_from_csv"]
    for row in result["replicates"]:
        diagnostic = row["fdt_diagnostic"]
        expected = row["zero_force_variance_nm2"] / (0.00831446261815324 * 300)
        assert diagnostic["chi_variance_over_RT_nm2_mol_per_kj"] == pytest.approx(expected)
        assert diagnostic["chi_fd_over_chi_fdt"] == pytest.approx(
            row["chi_nm2_mol_per_kj"] / expected
        )
        assert diagnostic["chi_fd_minus_chi_fdt_nm2_mol_per_kj"] == pytest.approx(
            row["chi_nm2_mol_per_kj"] - expected
        )
        assert diagnostic["status"] == "descriptive_only"
        assert diagnostic["uncertainty_and_consistency_gate"] == "not_evaluated"


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_fdt_temperature_rejected(config, temperature):
    config["temperature_K"] = temperature
    with pytest.raises(ValueError, match="temperature_K"):
        response.analyze(gaussian_rows(n=4), config, equilibration_ps=0)


@pytest.mark.parametrize("bad_time", [0.0, 1.5, -1.0])
def test_irregular_duplicate_or_reversed_time_rejected(config, bad_time):
    rows = gaussian_rows(n=80)
    rows[1]["time_ps"] = bad_time
    with pytest.raises(ValueError, match="strictly ordered and equally spaced"):
        response.analyze(rows, config, equilibration_ps=0)


def test_missing_force_in_one_replicate_rejected(config):
    rows = [
        row
        for row in gaussian_rows(n=80)
        if not (row["replicate"] == "1" and row["force_kj_mol_nm"] == 2)
    ]
    with pytest.raises(ValueError, match="missing force conditions"):
        response.analyze(rows, config, equilibration_ps=0)


def test_nonlinear_response_fails_without_discarding_replicates(config):
    result = response.analyze(gaussian_rows(cubic=0.15), config, equilibration_ps=0)
    assert result["status"] == "quality_gate_failed"
    assert len(result["replicates"]) == 3
    assert all(row["linearity_gate"]["status"] == "fail" for row in result["replicates"])
    assert (
        result["models"][0]["aggregation"] == "unweighted_all_replicates_no_quality_based_exclusion"
    )


@pytest.mark.parametrize("column", ["closure_nm", "time_ps", "force_kj_mol_nm"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_input_rejected(config, column, bad_value):
    rows = gaussian_rows(n=10)
    rows[7][column] = bad_value
    with pytest.raises(ValueError, match="must be finite"):
        response.analyze(rows, config, equilibration_ps=0)


def test_unresolved_response_is_not_an_eligible_ratio(config):
    result = response.analyze(gaussian_rows(susceptibility=0), config, equilibration_ps=0)
    assert result["status"] == "quality_gate_failed"
    assert any(row["linearity_gate"]["status"] == "unevaluable" for row in result["replicates"])


def test_low_ess_and_all_discarded_are_insufficient(config):
    for discard in (0, 1000):
        result = response.analyze(gaussian_rows(n=40), config, equilibration_ps=discard)
        assert result["status"] == "insufficient_sampling"
        json.dumps(result, allow_nan=False)


def test_ess_detects_serial_correlation():
    rng = np.random.default_rng(445)
    white = rng.normal(size=20000)
    ar = np.zeros_like(white)
    for i in range(1, len(ar)):
        ar[i] = 0.9 * ar[i - 1] + white[i]
    assert response.effective_sample_size(white) > 0.8 * len(white)
    assert 0.025 < response.effective_sample_size(ar) / len(ar) < 0.09
    assert response.effective_sample_size(np.ones(200)) == 0


def test_half_trajectory_drift_is_detected(config):
    rows = gaussian_rows()
    for row in rows:
        if row["time_ps"] >= 1000:
            row["closure_nm"] += 0.4 * row["force_kj_mol_nm"]
    result = response.analyze(rows, config, equilibration_ps=0)
    # This drift also lowers the stationary ESS estimate: sampling failure takes
    # priority, while the separately reported half-trajectory gate still fails.
    assert result["status"] == "insufficient_sampling"
    assert all(row["half_trajectory_gate"]["status"] == "fail" for row in result["replicates"])


def test_force_dependent_even_shift_reports_asymmetry(config):
    rows = gaussian_rows()
    for row in rows:
        row["closure_nm"] += 0.1 * row["force_kj_mol_nm"] ** 2
    result = response.analyze(rows, config, equilibration_ps=0)
    assert all(
        row["baseline_asymmetry_nm2_mol_per_kj"] == pytest.approx(0.2, abs=0.02)
        for row in result["replicates"]
    )


def test_cli_requires_explicit_equilibration_and_writes_local_provenance(config, tmp_path):
    source, config_path = tmp_path / "synthetic_fixture.csv", tmp_path / "config.json"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=response.COLUMNS)
        writer.writeheader()
        writer.writerows(gaussian_rows())
    config_path.write_text(json.dumps(config))
    args = [
        "--input",
        str(source),
        "--config",
        str(config_path),
        "--output-dir",
        str(tmp_path / "verification"),
        "--offline",
    ]
    with pytest.raises(SystemExit) as exc:
        response.main(args)
    assert exc.value.code == 2
    assert not (tmp_path / "verification").exists()
    assert response.main([*args, "--equilibration-ps", "0"]) == 0
    result = json.loads((tmp_path / "verification/response_analysis.json").read_text())
    assert result["status"] == "eligible_for_response_comparison"
    assert result["provenance"]["offline"]
    assert len(result["provenance"]["input_sha256"]) == 64
