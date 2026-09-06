#!/usr/bin/env python3
"""Technical zero-force covariance estimates in a fixed CRBN internal gauge.

NPZ schema (no pickle): core_displacement_nm=(T,269,3), time_ps=(T,),
reference_nm=(269,3), q_ambient=(807,). Displacements must be unwrapped nm
relative to that reference, with the same ordered core atoms. q is already a
unit internal vector. Optional scalar force_kj_mol_nm must equal zero. Neither
zero-force provenance nor gauge constraints during sampling can be proved here.

No frame alignment, refitted basis, or q renormalization is performed. For fixed
U=directional_mechanics.internal_basis(reference_nm), y=displacement@U and
Q=y@(U.T@q). Report C=Var(Q)/(RT), mean=tr Cov(y)/(801 RT), S=801 Var(Q)/tr Cov(y).
These are technical equilibrium covariance estimates, never evidence that an
801-dimensional covariance has converged or that models are ready to compare.

CLI writes covariance_analysis.json and covariance_timeseries.npz. The analyze
function additionally returns NumPy arrays in its 'timeseries' member. Explicit
equilibration is required, including an explicit zero; samples with elapsed
time below the cutoff are discarded. All remaining samples enter the estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from . import atomistic_response_analysis as response
    from . import directional_mechanics as dm
else:
    import atomistic_response_analysis as response
    import directional_mechanics as dm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts/atomistic_config.json"
DEFAULT_OUTPUT = ROOT / "results/atomistic/covariance"
CORE_COUNT = 269
INTERNAL_DIMENSION = 801
REQUIRED_ARRAYS = ("core_displacement_nm", "time_ps", "reference_nm", "q_ambient")


def _positive_finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")


def _estimates(internal, q_internal, thermal_energy):
    centered = internal - np.mean(internal, axis=0)
    q_centered = centered @ q_internal
    variance_q = float(q_centered @ q_centered / (len(internal) - 1))
    trace = float(np.sum(centered * centered) / (len(internal) - 1))
    return {
        "variance_Q_nm2": variance_q,
        "trace_internal_covariance_nm2": trace,
        "C_nm2_mol_per_kj": variance_q / thermal_energy,
        "mean_compliance_nm2_mol_per_kj": trace / (INTERNAL_DIMENSION * thermal_energy),
        "S_dimensionless": INTERNAL_DIMENSION * variance_q / trace if trace > 0 else None,
    }


def _block_uncertainty(internal, q_internal, thermal_energy, block_size, ess, trace_floor):
    n = len(internal)
    if block_size is None:
        block_size = max(2, math.isqrt(n))
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
        raise ValueError("block_size_frames must be an integer >= 2")
    count = n // block_size
    relevant = [ess[key] for key in ("Q", "Q_centered_squared", "internal_centered_norm_squared")]
    tau = max(n / value for value in relevant) if all(value > 0 for value in relevant) else None
    # This is an explicitly conditional diagnostic, not a independence proof.
    adequately_sized = count >= 8 and tau is not None and block_size >= 10 * tau
    estimates = [
        _estimates(internal[i * block_size : (i + 1) * block_size], q_internal, thermal_energy)
        for i in range(count)
    ]
    for item in estimates:
        if item["trace_internal_covariance_nm2"] <= trace_floor:
            item["S_dimensionless"] = None
    standard_errors = {}
    for key in ("C_nm2_mol_per_kj", "mean_compliance_nm2_mol_per_kj", "S_dimensionless"):
        values = [item[key] for item in estimates]
        standard_errors[key] = (
            float(np.std(values, ddof=1) / math.sqrt(count))
            if adequately_sized and all(value is not None for value in values)
            else None
        )
    return {
        "status": "conditional_technical_estimate"
        if adequately_sized
        else "not_estimable_with_selected_blocks",
        "block_size_frames": block_size,
        "n_complete_blocks": count,
        "n_tail_frames_omitted_from_uncertainty_only": n % block_size,
        "maximum_estimated_correlation_time_in_frames": tau,
        "adequacy_rule": "at_least_8_blocks_and_block_length_at_least_10_times_max_observed_tau",
        "assumptions": "stationarity_and_weak_between_block_dependence_are_not_validated",
        "block_centering": "each_block_uses_its_own_mean_and_ddof_1",
        "conditional_technical_se": standard_errors,
        "block_estimates": estimates,
        "independent_replicate_uncertainty": "not_available_from_one_trajectory",
        "block_size_convergence": "not_assessed",
    }


def analyze(
    core_displacement_nm,
    time_ps,
    reference_nm,
    q_ambient,
    *,
    temperature_K,
    equilibration_ps,
    block_size_frames=None,
):
    """Return descriptive statistics and retained scalar series for one trajectory."""
    _positive_finite(temperature_K, "temperature_K")
    if (
        isinstance(equilibration_ps, bool)
        or not isinstance(equilibration_ps, (int, float))
        or not math.isfinite(equilibration_ps)
        or equilibration_ps < 0
    ):
        raise ValueError("equilibration_ps must be explicitly finite and nonnegative")
    displacement, times, reference, q = (
        np.asarray(value, dtype=float)
        for value in (core_displacement_nm, time_ps, reference_nm, q_ambient)
    )
    if displacement.ndim != 3 or displacement.shape[1:] != (CORE_COUNT, 3) or len(displacement) < 4:
        raise ValueError("core_displacement_nm must have shape (T,269,3) with T >= 4")
    if (
        reference.shape != (CORE_COUNT, 3)
        or q.shape != (3 * CORE_COUNT,)
        or times.shape != (len(displacement),)
    ):
        raise ValueError("Expected reference_nm (269,3), q_ambient (807,), time_ps (T,)")
    if not all(np.isfinite(value).all() for value in (displacement, times, reference, q)):
        raise ValueError("All input arrays must be finite")
    dt = np.diff(times)
    if np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-6, atol=abs(dt[0]) * 1e-9):
        raise ValueError("time_ps must be strictly ordered and equally spaced")
    if abs(float(np.linalg.norm(q)) - 1.0) > 1e-10:
        raise ValueError("q_ambient must already have unit norm; it is never renormalized")
    basis = dm.internal_basis(reference)
    if basis.shape != (807, INTERNAL_DIMENSION):
        raise ValueError("Reference did not define the required 801-dimensional internal space")
    q_internal = basis.T @ q
    residual = float(np.linalg.norm(q - basis @ q_internal))
    if residual > 1e-10:
        raise ValueError("q_ambient must already lie in the fixed reference internal space")
    kept = times - times[0] >= equilibration_ps
    if int(np.sum(kept)) < 4:
        raise ValueError("At least four frames must remain after equilibration discard")
    flat = displacement[kept].reshape(-1, 807)
    internal = flat @ basis
    centered = internal - np.mean(internal, axis=0)
    closure = internal @ q_internal
    thermal_energy = response.GAS_CONSTANT_KJ_MOL_K * temperature_K
    estimates = _estimates(internal, q_internal, thermal_energy)
    # Bound ratio evaluation near floating-point gauge-projection residuals.
    roundoff_bound = 32 * np.finfo(float).eps * 807 * float(np.max(np.linalg.norm(flat, axis=1)))
    trace_floor = 4 * roundoff_bound**2
    trace_resolved = estimates["trace_internal_covariance_nm2"] > trace_floor
    if not trace_resolved:
        estimates["S_dimensionless"] = None
    scalar_series = {
        "Q": closure,
        "Q_squared": closure**2,
        "Q_centered_squared": (closure - np.mean(closure)) ** 2,
        "internal_norm_squared": np.sum(internal**2, axis=1),
        "internal_centered_norm_squared": np.sum(centered**2, axis=1),
    }
    ess = {key: response.effective_sample_size(value) for key, value in scalar_series.items()}
    if not trace_resolved:
        ess = {key: 0.0 for key in scalar_series}
    blocks = _block_uncertainty(
        internal, q_internal, thermal_energy, block_size_frames, ess, trace_floor
    )
    return {
        "schema_version": "1.0",
        "status": "technical_estimate_covariance_convergence_unverified"
        if trace_resolved
        else "unresolved_internal_variance",
        "covariance_convergence": "unverified",
        "eligible_for_response_comparison": False,
        "estimate_scope": "single_trajectory_zero_force_equilibrium_covariance_identity",
        "unverified_from_npz": [
            "zero_force_and_equilibrium_provenance",
            "gauge_constraints_during_sampling",
            "core_order_and_displacement_unit_provenance",
            "independent_replicates",
            "801_dimensional_covariance_convergence",
        ],
        "units": {
            "displacement_and_Q": "nm",
            "temperature": "K",
            "covariance_trace": "nm^2",
            "C_and_mean_compliance": "nm^2 mol kJ^-1",
            "S": "dimensionless",
        },
        "temperature_K": temperature_K,
        "thermal_energy_kj_mol": thermal_energy,
        "n_input_frames": len(times),
        "n_retained_frames": len(internal),
        "n_discarded_frames": int(np.sum(~kept)),
        "time_step_ps": float(dt[0]),
        "equilibration_ps": equilibration_ps,
        "equilibration_rule": "keep_elapsed_time_from_first_observation_greater_than_or_equal_to_cutoff",
        "core_position_count": CORE_COUNT,
        "internal_dimension": INTERNAL_DIMENSION,
        "sample_covariance_rank_upper_bound": min(INTERNAL_DIMENSION, len(internal) - 1),
        "q_internal_projection_residual": residual,
        "trace_resolution_floor_nm2": trace_floor,
        "fixed_basis_rule": "U=directional_mechanics.internal_basis(reference_nm); no_frame_alignment_or_basis_refit",
        "variance_rule": "center_each_coordinate_over_all_retained_frames_and_use_ddof_1",
        "formulas": {
            "C": "Var(Q)/(RT)",
            "mean_compliance": "tr(Cov_internal)/(801*RT)",
            "S": "801*Var(Q)/tr(Cov_internal)",
        },
        "estimates": estimates,
        "effective_samples": ess,
        "ess_method": "initial_positive_pairs_as_in_atomistic_response_analysis; scalar_diagnostics_do_not_prove_covariance_convergence",
        "uncertainty": blocks,
        "timeseries": {
            "time_ps": times[kept],
            "Q_nm": closure,
            "Q_squared_nm2": scalar_series["Q_squared"],
            "Q_centered_squared_nm2": scalar_series["Q_centered_squared"],
            "internal_norm_squared_nm2": scalar_series["internal_norm_squared"],
            "internal_centered_norm_squared_nm2": scalar_series["internal_centered_norm_squared"],
        },
    }


def run(
    input_path,
    config_path,
    output_dir,
    *,
    equilibration_ps=None,
    block_size_frames=None,
    offline=False,
):
    """Load only local NPZ/config bytes; preserve source hashes and scalar series."""
    input_path, config_path, output_dir = map(Path, (input_path, config_path, output_dir))
    input_bytes, config_bytes = input_path.read_bytes(), config_path.read_bytes()
    config = json.loads(config_bytes)
    options = config.get("covariance_analysis", {})
    if equilibration_ps is None:
        equilibration_ps = options.get("equilibration_ps")
    if equilibration_ps is None:
        raise ValueError(
            "Explicit --equilibration-ps or config.covariance_analysis.equilibration_ps is required"
        )
    if block_size_frames is None:
        block_size_frames = options.get("block_size_frames")
    with np.load(io.BytesIO(input_bytes), allow_pickle=False) as archive:
        missing = set(REQUIRED_ARRAYS) - set(archive.files)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {', '.join(sorted(missing))}")
        force_declared_zero = False
        if "force_kj_mol_nm" in archive.files:
            force = np.asarray(archive["force_kj_mol_nm"])
            if force.shape != () or not np.isfinite(force) or float(force) != 0:
                raise ValueError(
                    "force_kj_mol_nm must be a scalar zero for covariance response estimation"
                )
            force_declared_zero = True
        result = analyze(
            *(archive[key] for key in REQUIRED_ARRAYS),
            temperature_K=config.get("temperature_K"),
            equilibration_ps=equilibration_ps,
            block_size_frames=block_size_frames,
        )
    series = result.pop("timeseries")
    result["input_declares_zero_force"] = force_declared_zero
    result["provenance"] = {
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_sha256": {
            Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (__file__, dm.__file__, response.__file__)
        },
        "numpy_version": np.__version__,
        "offline": True,
        "offline_flag_requested": offline,
    }
    # Reject nonfinite derived statistics before creating any output artifacts.
    json.dumps(result, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "covariance_timeseries.npz", **series)
    result["timeseries_artifact"] = {
        "file": "covariance_timeseries.npz",
        "sha256": hashlib.sha256(
            (output_dir / "covariance_timeseries.npz").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "covariance_analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--offline", action="store_true", help="Local files only; this tool never uses the network"
    )
    parser.add_argument("--equilibration-ps", type=float)
    parser.add_argument("--block-size-frames", type=int)
    args = parser.parse_args(argv)
    try:
        result = run(
            args.input,
            args.config,
            args.output_dir,
            equilibration_ps=args.equilibration_ps,
            block_size_frames=args.block_size_frames,
            offline=args.offline,
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output_dir / "covariance_analysis.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
