#!/usr/bin/env python3
"""Analyse finite-force closure response; NumPy is the only numerical dependency.

Input closure_nm must already be the projection onto the fixed, unit-normalized
269-core closure vector. This module neither aligns coordinates nor estimates
the 801-dimensional mean compliance or its normalized S statistic. Eligibility
is a sampling/finite-force quality rule, not a significance test or proof that
the input replicate IDs represent independent initializations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts/atomistic_config.json"
DEFAULT_OUTPUT = ROOT / "results/atomistic/response_analysis"
COLUMNS = ("model", "replicate", "force_kj_mol_nm", "time_ps", "closure_nm")
MULTIPLIERS = (-2, -1, 0, 1, 2)
GAS_CONSTANT_KJ_MOL_K = 0.00831446261815324


def effective_sample_size(values: np.ndarray) -> float:
    """Conservative, N-capped FFT initial-positive-pair autocorrelation ESS.

    Biased autocovariances are paired (rho[0]+rho[1], rho[2]+rho[3], ...)
    until the first nonpositive pair. tau = -1 + 2*sum(pairs), floored at 1.
    Constant or fewer than four observations have no estimable sampling ESS.
    Stationarity is assumed; the half-trajectory gate is a separate diagnostic.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("ESS requires a finite one-dimensional series")
    if len(x) < 4:
        return 0.0
    x = x - np.mean(x)
    scale = float(np.max(np.abs(x)))
    if scale == 0:
        return 0.0
    x = x / scale
    size = 1 << (2 * len(x) - 1).bit_length()
    spectrum = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(spectrum * spectrum.conjugate(), n=size)[: len(x)]
    rho = acov / acov[0]
    pairs = rho[: 2 * (len(rho) // 2)].reshape(-1, 2).sum(axis=1)
    stop = np.flatnonzero(pairs <= 0)
    positive = pairs[: int(stop[0])] if len(stop) else pairs
    tau = max(1.0, float(-1.0 + 2.0 * np.sum(positive)))
    return float(len(x) / tau)


def _series_statistics(values: np.ndarray) -> dict:
    n = len(values)
    variance = float(np.var(values, ddof=1)) if n >= 2 else None
    ess = effective_sample_size(values)
    return {
        "n_frames": n,
        "mean_nm": float(np.mean(values)) if n else None,
        "variance_nm2": variance,
        "effective_samples": ess,
        "time_series_mean_se_nm": math.sqrt(variance / ess) if ess > 0 else None,
    }


def _slope(plus: dict, minus: dict, force: float) -> dict:
    if plus["mean_nm"] is None or minus["mean_nm"] is None:
        return {"chi_nm2_mol_per_kj": None, "sampling_se_upper_bound": None}
    # Sum, rather than quadrature, does not assume cross-force independence.
    errors = (plus["time_series_mean_se_nm"], minus["time_series_mean_se_nm"])
    return {
        "chi_nm2_mol_per_kj": (plus["mean_nm"] - minus["mean_nm"]) / (2 * force),
        "sampling_se_upper_bound": sum(errors) / (2 * force)
        if all(value is not None for value in errors)
        else None,
    }


def _resolved(slope: dict, se_multiplier: float) -> bool:
    chi, se = slope["chi_nm2_mol_per_kj"], slope["sampling_se_upper_bound"]
    return chi is not None and se is not None and abs(chi) > se_multiplier * se


def _relative_gate(
    reference: dict,
    compared: list[dict],
    difference: float | None,
    threshold: float,
    se_multiplier: float,
) -> dict:
    resolved = all(_resolved(item, se_multiplier) for item in [reference, *compared])
    relative = difference / abs(reference["chi_nm2_mol_per_kj"]) if resolved else None
    return {
        "relative_difference": relative,
        "maximum_relative_difference": threshold,
        "status": "unevaluable"
        if relative is None
        else ("pass" if relative <= threshold else "fail"),
        "reason": "response_unresolved_relative_to_sampling_error" if relative is None else None,
    }


def _settings(config: Mapping) -> dict:
    gate = config.get("production_gate", {})
    analysis = config.get("response_analysis", {})
    if sorted(config.get("force_multipliers", MULTIPLIERS)) != list(MULTIPLIERS):
        raise ValueError("This analysis requires force multipliers [-2, -1, 0, 1, 2]")
    settings = {
        "minimum_replicates": gate.get("independent_replicates", 3),
        "minimum_effective_samples_per_force_series": gate.get(
            "minimum_effective_samples_per_replicate", 50
        ),
        "linearity_relative_limit": gate.get(
            "magnitude_response_half_vs_full_force_relative_difference", 0.20
        ),
        "half_trajectory_relative_limit": gate.get(
            "maximum_relative_half_trajectory_difference", 0.20
        ),
        "relative_gate_resolution_se_multiplier": analysis.get(
            "relative_gate_resolution_se_multiplier", 2.0
        ),
    }
    for key, value in settings.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid positive finite setting: {key}")
    if (
        settings["minimum_replicates"] != int(settings["minimum_replicates"])
        or settings["minimum_replicates"] < 3
    ):
        raise ValueError("minimum independent_replicates must be an integer >= 3")
    if settings["minimum_effective_samples_per_force_series"] < 50:
        raise ValueError("minimum effective samples must be >= 50")
    if settings["relative_gate_resolution_se_multiplier"] < 2:
        raise ValueError("relative_gate_resolution_se_multiplier must be >= 2")
    return settings


def analyze(rows: Iterable[Mapping], config: Mapping, *, equilibration_ps: float) -> dict:
    """Return all replicate estimates and model quality gates without selection.

    Discard samples whose elapsed time from their series' first observation is
    less than equilibration_ps. The cutoff is mandatory, including explicit 0.
    Input row order within each series is checked, never repaired by sorting.
    F0 is read from the force column alone and must be shared by all models and
    replicates; its prior pilot selection cannot be authenticated from this CSV.
    """
    if not math.isfinite(equilibration_ps) or equilibration_ps < 0:
        raise ValueError("equilibration_ps must be finite and nonnegative")
    settings = _settings(config)
    temperature = config.get("temperature_K")
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature_K must be positive and finite")
    groups: dict[tuple, list] = {}
    allowed_models = config.get("models")
    for number, row in enumerate(rows, 2):
        if any(key not in row for key in COLUMNS):
            raise ValueError(f"Row {number}: required columns are {', '.join(COLUMNS)}")
        model, replicate = str(row["model"]).strip(), str(row["replicate"]).strip()
        if not model or not replicate or row["model"] is None or row["replicate"] is None:
            raise ValueError(f"Row {number}: model and replicate must be nonempty")
        if allowed_models is not None and model not in allowed_models:
            raise ValueError(f"Row {number}: model {model!r} is not in config.models")
        try:
            force, time, closure = (float(row[key]) for key in COLUMNS[2:])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {number}: force, time and closure must be numeric") from exc
        if not all(math.isfinite(value) for value in (force, time, closure)):
            raise ValueError(f"Row {number}: force, time and closure must be finite")
        groups.setdefault((model, replicate, force), []).append((time, closure))
    if not groups:
        raise ValueError("Input contains no observations")
    levels = sorted({key[2] for key in groups})
    positive = [value for value in levels if value > 0]
    if not positive:
        raise ValueError("Input requires zero and both signs of F0 and 2*F0")
    force0 = min(positive)
    expected = np.asarray(MULTIPLIERS) * force0
    if len(levels) != 5 or not np.allclose(levels, expected, rtol=1e-9, atol=0):
        raise ValueError(
            "All models/replicates require one shared force set: -2F0, -F0, 0, F0, 2F0"
        )
    multiplier_for = dict(zip(levels, MULTIPLIERS))
    prepared: dict[tuple, dict] = {}
    series_records = []
    for (model, replicate, force), points in sorted(groups.items()):
        array = np.asarray(points, dtype=float)
        times = array[:, 0]
        if len(times) < 2:
            raise ValueError(f"{model}/{replicate}/{force}: need >=2 times to check spacing")
        intervals = np.diff(times)
        if np.any(intervals <= 0) or not np.allclose(
            intervals, intervals[0], rtol=1e-6, atol=abs(intervals[0]) * 1e-9
        ):
            raise ValueError(
                f"{model}/{replicate}/{force}: time must be strictly ordered and equally spaced"
            )
        kept = array[times - times[0] >= equilibration_ps, 1]
        stats = _series_statistics(kept)
        record = {
            "model": model,
            "replicate": replicate,
            "force_kj_mol_nm": force,
            "force_multiplier": multiplier_for[force],
            "n_input_frames": len(times),
            "n_discarded_frames": len(times) - len(kept),
            "time_step_ps": float(intervals[0]),
            "equilibration_cutoff_absolute_ps": float(times[0] + equilibration_ps),
            **stats,
        }
        series_records.append(record)
        middle = len(kept) // 2
        prepared.setdefault((model, replicate), {})[multiplier_for[force]] = {
            "full": stats,
            "first_half": _series_statistics(kept[:middle]),
            "second_half": _series_statistics(kept[middle:]),
        }
    replicate_records = []
    resolution = settings["relative_gate_resolution_se_multiplier"]
    for (model, replicate), series in sorted(prepared.items()):
        if sorted(series) != list(MULTIPLIERS):
            raise ValueError(
                f"{model}/{replicate}: missing force conditions; need -2F0, -F0, 0, F0, 2F0"
            )
        slopes = {
            part: _slope(series[1][part], series[-1][part], force0)
            for part in ("full", "first_half", "second_half")
        }
        double = _slope(series[2]["full"], series[-2]["full"], 2 * force0)
        chi = slopes["full"]["chi_nm2_mol_per_kj"]
        large_chi = double["chi_nm2_mol_per_kj"]
        first, second = (
            slopes[part]["chi_nm2_mol_per_kj"] for part in ("first_half", "second_half")
        )
        linearity = _relative_gate(
            slopes["full"],
            [double],
            abs(large_chi - chi) if None not in (large_chi, chi) else None,
            settings["linearity_relative_limit"],
            resolution,
        )
        convergence = _relative_gate(
            slopes["full"],
            [slopes["first_half"], slopes["second_half"]],
            abs(first - second) if None not in (first, second) else None,
            settings["half_trajectory_relative_limit"],
            resolution,
        )
        means = [series[m]["full"]["mean_nm"] for m in (1, 0, -1)]
        plus_response = (means[0] - means[1]) / force0 if None not in means else None
        minus_response = (means[1] - means[2]) / force0 if None not in means else None
        asymmetry = plus_response - minus_response if None not in means else None
        sampling_pass = all(
            series[m]["full"]["effective_samples"]
            >= settings["minimum_effective_samples_per_force_series"]
            for m in MULTIPLIERS
        )
        primary_resolved = _resolved(slopes["full"], resolution)
        negative_resolved = primary_resolved and chi < 0
        zero_variance = series[0]["full"]["variance_nm2"]
        fdt_chi = (
            zero_variance / (GAS_CONSTANT_KJ_MOL_K * temperature)
            if temperature is not None and zero_variance is not None
            else None
        )
        replicate_records.append(
            {
                "model": model,
                "replicate": replicate,
                **slopes["full"],
                "double_force": double,
                "first_half": slopes["first_half"],
                "second_half": slopes["second_half"],
                "zero_force_variance_nm2": series[0]["full"]["variance_nm2"],
                "zero_force_effective_samples": series[0]["full"]["effective_samples"],
                "minimum_effective_samples_across_forces": min(
                    series[m]["full"]["effective_samples"] for m in MULTIPLIERS
                ),
                "positive_force_baseline_response_nm2_mol_per_kj": plus_response,
                "negative_force_baseline_response_nm2_mol_per_kj": minus_response,
                "baseline_asymmetry_nm2_mol_per_kj": asymmetry,
                "baseline_asymmetry_relative_to_chi": asymmetry / abs(chi)
                if _resolved(slopes["full"], resolution)
                else None,
                "sampling_gate": "pass" if sampling_pass else "fail",
                "physical_consistency_gate": {
                    "status": "fail"
                    if negative_resolved
                    else ("pass" if primary_resolved else "unevaluable"),
                    "scope": "sign_only_for_energy_minus_force_times_closure",
                    "reason": "negative_resolved_conjugate_susceptibility"
                    if negative_resolved
                    else None,
                },
                "fdt_diagnostic": {
                    "status": "descriptive_only" if fdt_chi is not None else "not_computed",
                    "temperature_K": temperature,
                    "chi_variance_over_RT_nm2_mol_per_kj": fdt_chi,
                    "chi_fd_over_chi_fdt": chi / fdt_chi
                    if chi is not None and fdt_chi is not None and fdt_chi > 0
                    else None,
                    "chi_fd_minus_chi_fdt_nm2_mol_per_kj": chi - fdt_chi
                    if chi is not None and fdt_chi is not None
                    else None,
                    "uncertainty_and_consistency_gate": "not_evaluated",
                },
                "linearity_gate": linearity,
                "half_trajectory_gate": convergence,
            }
        )
    models = []
    for model in sorted({row["model"] for row in replicate_records}):
        replicates = [row for row in replicate_records if row["model"] == model]
        values = [row["chi_nm2_mol_per_kj"] for row in replicates]
        complete = all(value is not None for value in values)
        mean = float(np.mean(values)) if complete else None
        sd = float(np.std(values, ddof=1)) if complete and len(values) > 1 else None
        se = sd / math.sqrt(len(values)) if sd is not None else None
        aggregate_resolved = mean is not None and se is not None and abs(mean) > resolution * se
        # A chance agreement between noisy replicate means cannot by itself
        # resolve a negative sign that the time-series sampling error cannot.
        physical_fail = any(
            row["physical_consistency_gate"]["status"] == "fail" for row in replicates
        )
        physical_resolved = aggregate_resolved and all(
            row["physical_consistency_gate"]["status"] == "pass" for row in replicates
        )
        sampling_pass = len(replicates) >= settings["minimum_replicates"] and all(
            row["sampling_gate"] == "pass" for row in replicates
        )
        response_pass = aggregate_resolved and all(
            row[gate]["status"] == "pass"
            for row in replicates
            for gate in ("linearity_gate", "half_trajectory_gate")
        )
        status = (
            "physical_consistency_gate_failed"
            if physical_fail
            else (
                "insufficient_sampling"
                if not sampling_pass
                else (
                    "eligible_for_response_comparison" if response_pass else "quality_gate_failed"
                )
            )
        )
        models.append(
            {
                "model": model,
                "n_independent_replicate_ids": len(replicates),
                "mean_chi_nm2_mol_per_kj": mean,
                "between_replicate_sd": sd,
                "between_replicate_se": se,
                "aggregate_response_resolved": aggregate_resolved,
                "physical_consistency_gate": "fail"
                if physical_fail
                else ("pass" if physical_resolved else "unevaluable"),
                "sampling_gate": "pass" if sampling_pass else "fail",
                "status": status,
                "aggregation": "unweighted_all_replicates_no_quality_based_exclusion",
            }
        )
    missing_models = sorted(set(allowed_models or []) - {row["model"] for row in models})
    overall_comparison_ready = not missing_models and all(
        row["status"] == "eligible_for_response_comparison" for row in models
    )
    if missing_models:
        overall_status = "partial_models"
    elif any(row["status"] == "physical_consistency_gate_failed" for row in models):
        overall_status = "physical_consistency_gate_failed"
    elif overall_comparison_ready:
        overall_status = "eligible_for_response_comparison"
    elif any(row["status"] == "insufficient_sampling" for row in models):
        overall_status = "insufficient_sampling"
    else:
        overall_status = "quality_gate_failed"
    return {
        "schema_version": "1.0",
        "status": overall_status,
        "missing_models": missing_models,
        "overall_comparison_ready": overall_comparison_ready,
        "eligibility_scope": "sampling_and_finite_force_response_only_for_present_models",
        "all_gates_are_protocol_quality_rules_not_significance_tests": True,
        "unverified_from_csv": [
            "independent_initialization_and_equilibration",
            "fixed_unit_normalized_269_core_projection",
            "pilot_only_F0_selection_and_lock",
            "topology_metal_gauge_rigid_body_and_restraint_sensitivity_gates",
            "FDT_consistency",
        ],
        "mean_compliance_and_normalized_S": "not_estimated_requires_independently_converged_801_dimensional_covariance",
        "units": {"closure": "nm", "force": "kJ mol^-1 nm^-1", "chi": "nm^2 mol kJ^-1"},
        "force0_kj_mol_nm": force0,
        "force0_source": "input_force_levels_only_not_response_values",
        "equilibration_ps": equilibration_ps,
        "equilibration_rule": "keep elapsed time >= cutoff from each series first observation",
        "ess_method": "FFT_biased_autocovariance_initial_positive_pairs_tau_at_least_one",
        "sampling_error_rule": "per_series sqrt(variance/ESS); central_slope_SE_upper_bound=(SE_plus+SE_minus)/(2F); distinct from between_replicate_SE",
        "relative_gate_rule": "abs(compared_slope_difference)/abs(full_F0_slope); full and compared slopes must exceed resolution_multiplier times sampling_SE_upper_bound",
        "fdt_diagnostic_rule": "Var_zero_force(Q)/(R*T), R=0.00831446261815324 kJ mol^-1 K^-1; descriptive only; neither uncertainty nor FDT consistency is validated",
        "settings": settings,
        "series": series_records,
        "replicates": replicate_records,
        "models": models,
    }


def run(
    input_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    equilibration_ps: float | None = None,
    offline: bool = False,
) -> dict:
    """Read local input/config and write an auditable JSON result; never network."""
    input_path, config_path, output_dir = map(Path, (input_path, config_path, output_dir))
    config_bytes, input_bytes = config_path.read_bytes(), input_path.read_bytes()
    config = json.loads(config_bytes)
    if equilibration_ps is None:
        equilibration_ps = config.get("response_analysis", {}).get("equilibration_ps")
    if equilibration_ps is None:
        raise ValueError(
            "Explicit --equilibration-ps or config.response_analysis.equilibration_ps is required (0 is allowed)"
        )
    with io.StringIO(input_bytes.decode("utf-8-sig"), newline="") as handle:
        result = analyze(csv.DictReader(handle), config, equilibration_ps=float(equilibration_ps))
    result["provenance"] = {
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "offline": True,
        "offline_flag_requested": offline,
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "response_analysis.json").write_text(serialized)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Measured closure CSV; no synthetic fallback"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--equilibration-ps", type=float, help="Explicit elapsed-time discard; 0 allowed"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require local files (the tool always operates offline)",
    )
    args = parser.parse_args(argv)
    try:
        result = run(
            args.input,
            args.config,
            args.output_dir,
            equilibration_ps=args.equilibration_ps,
            offline=args.offline,
        )
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"status": result["status"], "output": str(args.output_dir / "response_analysis.json")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
