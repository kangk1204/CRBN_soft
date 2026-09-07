#!/usr/bin/env python3
"""Descriptive diagnostics for committed zero-force atomistic response generations.

This CLI reads an existing response segment output directory containing
segment_manifest.json plus its current generation. It verifies immutable
CSV/NPZ/generation hashes and source identity, rejects finite-force and
technical-pilot-style inputs, and writes descriptive zero-force diagnostics.
It does not run OpenMM, refit frames, issue an equilibrium certificate, or
admit production response analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__:
    from . import atomistic_covariance as covariance
    from . import atomistic_response_analysis as response_analysis
    from . import directional_mechanics as dm
else:
    import atomistic_covariance as covariance
    import atomistic_response_analysis as response_analysis
    import directional_mechanics as dm

ROOT = Path(__file__).resolve().parents[1]
GENERATION_ARTIFACTS = ("segment.chk", "segment_state.xml", "response_observations.csv", "segment_observables.npz")
ZERO_PHASES = {"zero_equilibration", "zero_calibration"}
REQUIRED_NPZ = ("core_displacement_nm", "time_ps", "reference_nm", "q_ambient", "core_indices", "force_kj_mol_nm")
CSV_COLUMNS = ("model", "replicate", "force_kj_mol_nm", "time_ps", "closure_nm")


class DiagnosticInputError(ValueError):
    """Input is not an admissible zero-force response-generation artifact."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticInputError(f"JSON must contain an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def file_record(path: Path, *, base: Path | None = None) -> dict[str, Any]:
    try:
        display = str(path.relative_to(base or ROOT))
    except ValueError:
        display = str(path)
    return {"path": display, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def require_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise DiagnosticInputError(f"output directory must be new or empty: {path}")


def generation_paths(input_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = input_dir / "segment_manifest.json"
    if not manifest_path.is_file():
        if any((input_dir / name).exists() for name in ("technical_pilot.json", "technical_pilot_report.json", "pilot_report.json")):
            raise DiagnosticInputError("technical_pilot output is not accepted as zero-force equilibrium evidence")
        raise DiagnosticInputError("input-dir must be a response segment directory containing segment_manifest.json")
    manifest = read_json(manifest_path)
    generation = manifest.get("current_generation")
    if not isinstance(generation, str) or not generation:
        raise DiagnosticInputError("segment manifest lacks current_generation")
    generation_dir = input_dir / generation
    if not generation_dir.is_dir():
        raise DiagnosticInputError("current_generation directory is missing")
    return manifest, manifest_path, generation_dir


def verify_generation_hashes(manifest: Mapping[str, Any], generation_dir: Path) -> dict[str, Any]:
    declared = manifest.get("generation_sha256")
    if not isinstance(declared, Mapping):
        raise DiagnosticInputError("segment manifest lacks generation_sha256")
    missing = sorted(set(GENERATION_ARTIFACTS) - set(declared))
    if missing:
        raise DiagnosticInputError(f"generation_sha256 missing artifacts: {', '.join(missing)}")
    records = {}
    for name in GENERATION_ARTIFACTS:
        path = generation_dir / name
        if not path.is_file():
            raise DiagnosticInputError(f"generation artifact missing: {name}")
        digest = sha256_file(path)
        if digest != declared[name]:
            raise DiagnosticInputError(f"generation artifact hash mismatch: {name}")
        records[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    return {"status": "pass", "artifacts": records}


def role_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    measurement = manifest.get("measurement_identity")
    parent = manifest.get("admission", {}).get("parent_state", {})
    if manifest.get("synthetic_fixture") is True or parent.get("synthetic_fixture") is True:
        role = "synthetic"
        synthetic_fixture: bool | None = True
    elif manifest.get("synthetic_fixture") is False:
        role = "real"
        synthetic_fixture = False
    else:
        role = "unknown_missing_synthetic_fixture_flag"
        synthetic_fixture = None
    return {
        "role": role,
        "role_rule": "synthetic_fixture=True means synthetic; False means real; missing flag remains unknown for legacy manifests",
        "synthetic_fixture": synthetic_fixture,
        "measurement_identity_required_for_zero_force_diagnostics": False,
        "measurement_identity_provided": bool(isinstance(measurement, Mapping) and measurement.get("provided")),
        "measurement_identity": measurement if isinstance(measurement, Mapping) else None,
    }


def validate_manifest_zero_force(manifest: Mapping[str, Any]) -> dict[str, Any]:
    phase = manifest.get("phase")
    if phase not in ZERO_PHASES:
        raise DiagnosticInputError("only zero_equilibration or zero_calibration segment generations are accepted")
    force = float(manifest.get("force_kj_mol_nm"))
    if force != 0.0:
        raise DiagnosticInputError("finite-force segments are not accepted as zero-force equilibrium evidence")
    if manifest.get("status") not in {"segment_complete", "budget_limited", "started"}:
        raise DiagnosticInputError("unrecognized segment manifest status")
    if manifest.get("response_converged") is True or manifest.get("production_ready") is True:
        raise DiagnosticInputError("input segment must not self-certify response convergence or production readiness")
    return {
        "status": "pass",
        "phase": phase,
        "force_kj_mol_nm": force,
        "segment_status": manifest.get("status"),
        "segment_complete": bool(manifest.get("segment_complete")),
        "model": manifest.get("model"),
        "replicate_id": manifest.get("replicate_id"),
    }


def load_generation_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = sorted(set(REQUIRED_NPZ) - set(archive.files))
            if missing:
                raise DiagnosticInputError(f"segment_observables.npz missing arrays: {', '.join(missing)}")
            return {key: np.asarray(archive[key]) for key in REQUIRED_NPZ}
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise DiagnosticInputError("segment_observables.npz must be loadable with allow_pickle=False") from exc
        raise


def parse_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise DiagnosticInputError("response_observations.csv has unexpected columns")
        return list(reader)


def validate_arrays_and_csv(arrays: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, str]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    displacement = np.asarray(arrays["core_displacement_nm"], dtype=float)
    times = np.asarray(arrays["time_ps"], dtype=float)
    reference = np.asarray(arrays["reference_nm"], dtype=float)
    q = np.asarray(arrays["q_ambient"], dtype=float).reshape(-1)
    core_indices = np.asarray(arrays["core_indices"], dtype=int)
    force_values = np.asarray(arrays["force_kj_mol_nm"], dtype=float).reshape(-1)
    if displacement.ndim != 3 or displacement.shape[1:] != (269, 3):
        raise DiagnosticInputError("core_displacement_nm must have shape (T,269,3)")
    if reference.shape != (269, 3) or q.shape != (807,) or core_indices.shape != (269,) or times.shape != (len(displacement),):
        raise DiagnosticInputError("NPZ array shapes do not match the frozen 269-core measurement contract")
    if not all(np.isfinite(value).all() for value in (displacement, times, reference, q, core_indices, force_values)):
        raise DiagnosticInputError("generation arrays must be finite")
    if len(force_values) != 1 or float(force_values[0]) != 0.0:
        raise DiagnosticInputError("segment_observables.npz must declare scalar force_kj_mol_nm=0")
    if len(times) < 2:
        raise DiagnosticInputError("at least two time points are required to validate the generation time grid")
    dt = np.diff(times)
    if np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-6, atol=abs(float(dt[0])) * 1e-9):
        raise DiagnosticInputError("time_ps must be strictly ordered and equally spaced")
    if len(rows) != len(times):
        raise DiagnosticInputError("CSV/NPZ observation counts differ")
    flat = displacement.reshape(len(displacement), -1)
    actual_q = flat @ q
    csv_q = []
    for row, time_value, _actual_closure in zip(rows, times, actual_q):
        try:
            row_force = float(row["force_kj_mol_nm"])
            row_time = float(row["time_ps"])
            row_closure = float(row["closure_nm"])
        except (TypeError, ValueError) as exc:
            raise DiagnosticInputError("response_observations.csv contains nonnumeric force/time/closure") from exc
        if row.get("model") != manifest.get("model") or row.get("replicate") != str(manifest.get("replicate_id")):
            raise DiagnosticInputError("response_observations.csv model/replicate does not match manifest")
        if row_force != 0.0:
            raise DiagnosticInputError("response_observations.csv contains finite-force rows")
        if abs(row_time - float(time_value)) > 1e-12:
            raise DiagnosticInputError("response_observations.csv time does not match NPZ time_ps")
        csv_q.append(row_closure)
    csv_q_array = np.asarray(csv_q, dtype=float)
    max_q_delta = float(np.max(np.abs(csv_q_array - actual_q))) if len(actual_q) else 0.0
    if max_q_delta > 1e-10:
        raise DiagnosticInputError("response_observations.csv closure_nm does not match q dot core displacement")
    if abs(float(np.linalg.norm(q)) - 1.0) > 1e-10:
        raise DiagnosticInputError("q_ambient must be unit norm")
    basis = dm.internal_basis(reference)
    q_internal = basis.T @ q
    q_residual = float(np.linalg.norm(q - basis @ q_internal))
    if q_residual > 1e-10:
        raise DiagnosticInputError("q_ambient is not in the fixed internal subspace")
    internal = flat @ basis
    internal_rmsd = np.linalg.norm(internal, axis=1) / math.sqrt(269)
    return {
        "status": "pass",
        "n_frames": int(len(times)),
        "time_start_ps": float(times[0]),
        "time_end_ps": float(times[-1]),
        "time_step_ps": float(dt[0]),
        "actual_Q_nm": actual_q,
        "csv_Q_nm": csv_q_array,
        "max_abs_csv_actual_Q_delta_nm": max_q_delta,
        "internal_rmsd_nm": internal_rmsd,
        "q_internal_projection_residual": q_residual,
        "core_indices_sha256": canonical_hash(core_indices.astype(int).tolist()),
        "reference_nm_sha256": canonical_hash(reference.tolist()),
        "q_ambient_sha256": canonical_hash(q.tolist()),
    }


def validate_endpoint_bindings(manifest: Mapping[str, Any], validation: Mapping[str, Any], system_xml: Path) -> dict[str, Any]:
    required = (
        "data_input_hashes",
        "source_provenance",
        "mapping_sha256",
        "core_indices_sha256",
        "reference_nm_sha256",
        "q_ambient_sha256",
        "system_xml_sha256",
        "gauge_sha256",
        "body_sha256",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise DiagnosticInputError(f"segment manifest missing source identity binding(s): {', '.join(missing)}")
    data_hashes = manifest.get("data_input_hashes")
    if not isinstance(data_hashes, Mapping) or not {"prmtop", "inpcrd", "mapping"} <= set(data_hashes):
        raise DiagnosticInputError("data_input_hashes must include prmtop, inpcrd, and mapping")
    if sha256_file(system_xml) != manifest.get("system_xml_sha256"):
        raise DiagnosticInputError("segment_system.xml hash does not match manifest system_xml_sha256")
    for key in ("core_indices_sha256", "reference_nm_sha256", "q_ambient_sha256"):
        if validation.get(key) != manifest.get(key):
            raise DiagnosticInputError(f"NPZ {key} does not match manifest binding")
    return {
        "status": "pass",
        "required_manifest_bindings": list(required),
        "data_input_hash_keys": sorted(data_hashes),
        "system_xml_sha256": manifest["system_xml_sha256"],
        "core_indices_sha256": validation["core_indices_sha256"],
        "reference_nm_sha256": validation["reference_nm_sha256"],
        "q_ambient_sha256": validation["q_ambient_sha256"],
        "measurement_identity_optional_for_zero_force": True,
    }


def finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DiagnosticInputError(f"{name} must be finite numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise DiagnosticInputError(f"{name} must be >= {minimum}")
    return result


def retained_mask(times: np.ndarray, equilibration_ps: float) -> np.ndarray:
    return times - float(times[0]) >= equilibration_ps


def series_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        return {"n": 0, "mean": None, "variance_ddof1": None, "min": None, "max": None}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "variance_ddof1": float(np.var(values, ddof=1)) if len(values) >= 2 else None,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def block_statistics(times: np.ndarray, q: np.ndarray, q2: np.ndarray, trace_proxy: np.ndarray) -> dict[str, Any]:
    n = len(q)
    block_size = max(2, math.isqrt(n)) if n >= 2 else 0
    count = n // block_size if block_size else 0
    blocks = []
    for i in range(count):
        sl = slice(i * block_size, (i + 1) * block_size)
        blocks.append({
            "block_index": i,
            "start_time_ps": float(times[sl][0]),
            "end_time_ps": float(times[sl][-1]),
            "n_frames": int(block_size),
            "Q_nm": series_summary(q[sl]),
            "Q_squared_nm2": series_summary(q2[sl]),
            "trace_proxy_internal_centered_norm_squared_nm2": series_summary(trace_proxy[sl]),
        })
    return {
        "block_size_frames": int(block_size),
        "n_complete_blocks": int(count),
        "n_tail_frames_omitted_from_block_descriptives": int(n % block_size) if block_size else int(n),
        "blocks": blocks,
    }


def half_comparison(times: np.ndarray, q: np.ndarray, q2: np.ndarray, trace_proxy: np.ndarray) -> dict[str, Any]:
    n = len(q)
    if n < 4:
        return {"status": "unevaluable", "reason": "fewer_than_four_retained_frames"}
    mid = n // 2
    first = slice(0, mid)
    second = slice(n - mid, n)
    return {
        "status": "descriptive_only",
        "rule": "compare first floor(n/2) retained frames to last floor(n/2) retained frames; no stationarity certificate",
        "first_half": {
            "time_start_ps": float(times[first][0]),
            "time_end_ps": float(times[first][-1]),
            "Q_nm": series_summary(q[first]),
            "Q_squared_nm2": series_summary(q2[first]),
            "trace_proxy_internal_centered_norm_squared_nm2": series_summary(trace_proxy[first]),
        },
        "second_half": {
            "time_start_ps": float(times[second][0]),
            "time_end_ps": float(times[second][-1]),
            "Q_nm": series_summary(q[second]),
            "Q_squared_nm2": series_summary(q2[second]),
            "trace_proxy_internal_centered_norm_squared_nm2": series_summary(trace_proxy[second]),
        },
    }


def descriptive_diagnostics(arrays: Mapping[str, np.ndarray], validation: Mapping[str, Any], config: Mapping[str, Any], equilibration_ps: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    displacement = np.asarray(arrays["core_displacement_nm"], dtype=float)
    times = np.asarray(arrays["time_ps"], dtype=float)
    reference = np.asarray(arrays["reference_nm"], dtype=float)
    q = np.asarray(arrays["q_ambient"], dtype=float).reshape(-1)
    mask = retained_mask(times, equilibration_ps)
    retained = int(np.sum(mask))
    flat = displacement.reshape(len(displacement), -1)
    basis = dm.internal_basis(reference)
    internal_all = flat @ basis
    internal = internal_all[mask]
    q_retained = np.asarray(validation["actual_Q_nm"], dtype=float)[mask]
    q2 = q_retained**2
    internal_centered = internal - np.mean(internal, axis=0) if retained else internal
    trace_proxy = np.sum(internal_centered * internal_centered, axis=1) if retained else np.asarray([], dtype=float)
    retained_times = times[mask]
    series = {
        "time_ps": retained_times,
        "Q_nm": q_retained,
        "Q_squared_nm2": q2,
        "internal_rmsd_nm": np.asarray(validation["internal_rmsd_nm"], dtype=float)[mask],
        "trace_proxy_internal_centered_norm_squared_nm2": trace_proxy,
    }
    common = {
        "n_input_frames": int(len(times)),
        "n_retained_frames": retained,
        "n_discarded_frames": int(len(times) - retained),
        "equilibration_ps": float(equilibration_ps),
        "equilibration_rule": "keep elapsed time from first observation greater than or equal to --equilibration-ps",
        "actual_Q_nm": series_summary(q_retained),
        "actual_Q_squared_nm2": series_summary(q2),
        "internal_rmsd_nm": series_summary(series["internal_rmsd_nm"]),
    }
    if retained < 4:
        return {
            **common,
            "status": "unevaluable_insufficient_data",
            "reason": "fewer_than_four_frames_after_equilibration_discard",
            "eligible_for_response_comparison": False,
            "equilibrium_or_stationarity_certificate": False,
        }, series
    ess = {
        "Q": response_analysis.effective_sample_size(q_retained),
        "Q_squared": response_analysis.effective_sample_size(q2),
        "trace_proxy_internal_centered_norm_squared": response_analysis.effective_sample_size(trace_proxy),
    }
    cov = covariance.analyze(
        displacement,
        times,
        reference,
        q,
        temperature_K=config.get("temperature_K"),
        equilibration_ps=equilibration_ps,
        block_size_frames=(config.get("zero_force_diagnostics", {}) or {}).get("block_size_frames"),
    )
    cov.pop("timeseries", None)
    return {
        **common,
        "status": "descriptive_zero_force_diagnostics",
        "eligible_for_response_comparison": False,
        "equilibrium_or_stationarity_certificate": False,
        "covariance_convergence": "not_certified",
        "ess": ess,
        "block_descriptives": block_statistics(retained_times, q_retained, q2, trace_proxy),
        "half_comparison": half_comparison(retained_times, q_retained, q2, trace_proxy),
        "covariance_analysis_reused": cov,
    }, series


def source_identity_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "data_input_hashes",
        "stage_provenance_hashes",
        "source_provenance",
        "topology",
        "mapping_sha256",
        "core_indices_sha256",
        "reference_nm_sha256",
        "q_ambient_sha256",
        "system_xml_sha256",
        "gauge_sha256",
        "body_sha256",
        "measurement_identity",
    ]
    return {key: manifest.get(key) for key in keys if key in manifest}


def run(input_dir: Path, config_path: Path, output_dir: Path, *, equilibration_ps: float, offline: bool = False) -> dict[str, Any]:
    input_dir = Path(input_dir)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    require_output_dir(output_dir)
    config = read_json(config_path)
    equilibration_ps = finite_float(equilibration_ps, "equilibration_ps", minimum=0.0)
    manifest, manifest_path, gen_dir = generation_paths(input_dir)
    manifest_zero = validate_manifest_zero_force(manifest)
    role = role_record(manifest)
    generation_hashes = verify_generation_hashes(manifest, gen_dir)
    system_xml = input_dir / "segment_system.xml"
    if not system_xml.is_file():
        raise DiagnosticInputError("segment_system.xml is required for source-identity hash verification")
    arrays = load_generation_arrays(gen_dir / "segment_observables.npz")
    rows = parse_csv_rows(gen_dir / "response_observations.csv")
    array_validation = validate_arrays_and_csv(arrays, rows, manifest)
    endpoint_binding = validate_endpoint_bindings(manifest, array_validation, system_xml)
    diagnostics, series = descriptive_diagnostics(arrays, array_validation, config, equilibration_ps)
    result = {
        "schema_version": "1.0",
        "status": diagnostics["status"],
        "scope": "descriptive diagnostics for one hash-bound zero-force response generation; no production admission or equilibrium/stationarity certificate",
        "production_ready": False,
        "scientific_certificate": False,
        "equilibrium_certified": False,
        "offline": True,
        "offline_flag_requested": bool(offline),
        "input": {
            "input_dir": str(input_dir.resolve()),
            "segment_manifest": file_record(manifest_path),
            "segment_system_xml": file_record(system_xml),
            "current_generation_dir": str(gen_dir.resolve()),
        },
        "config": file_record(config_path),
        "manifest_zero_force": manifest_zero,
        "real_or_synthetic_role": role,
        "source_identity": source_identity_record(manifest),
        "endpoint_binding_verification": endpoint_binding,
        "generation_hash_verification": generation_hashes,
        "array_csv_validation": {key: value for key, value in array_validation.items() if not isinstance(value, np.ndarray)},
        "diagnostics": diagnostics,
        "negative_claims_preserved": {
            "technical_pilot_is_not_equilibrium_evidence": True,
            "finite_force_is_not_zero_force_equilibrium_evidence": True,
            "automatic_equilibrium_or_stationarity_certificate": False,
            "automatic_scientific_pass_threshold": False,
            "frame_alignment_or_refit_performed": False,
        },
        "source_sha256": {
            "atomistic_zero_force_diagnostics.py": sha256_file(Path(__file__)),
            "atomistic_covariance.py": sha256_file(Path(covariance.__file__)),
            "atomistic_response_analysis.py": sha256_file(Path(response_analysis.__file__)),
            "directional_mechanics.py": sha256_file(Path(dm.__file__)),
        },
    }
    json.dumps(result, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "zero_force_timeseries.npz", **series)
    result["timeseries_artifact"] = file_record(output_dir / "zero_force_timeseries.npz")
    write_json(output_dir / "zero_force_diagnostics.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/atomistic_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--equilibration-ps", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.input_dir, args.config, args.output_dir, equilibration_ps=args.equilibration_ps, offline=args.offline)
    except (DiagnosticInputError, ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "rejected_or_failed", "reason": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output_dir / "zero_force_diagnostics.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
