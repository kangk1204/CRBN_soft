#!/usr/bin/env python3
"""Mode continuation for the CRBN-DDB1 directional mechanics analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

try:
    import directional_mechanics as mechanics
    import strengthen_ddb1 as legacy_modes
except ModuleNotFoundError:
    from scripts import directional_mechanics as mechanics
    from scripts import strengthen_ddb1 as legacy_modes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts" / "directional_config.json"
SCHEMA_VERSION = "directional_modes.v1"
ZERO_TOL = 1e-9
LOW_OVERLAP_THRESHOLD = 0.50


def _load_config(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        with DEFAULT_CONFIG.open() as handle:
            return json.load(handle)
    if isinstance(config, (str, Path)):
        with Path(config).open() as handle:
            return json.load(handle)
    return dict(config)


def _as_float_matrix(system: dict[str, Any], key: str) -> np.ndarray:
    if key not in system:
        raise ValueError(f"system is missing required key {key!r}")
    value = np.asarray(system[key], dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"system[{key!r}] must be a square matrix")
    return (value + value.T) / 2


def _input_hash(system: dict[str, Any], direction: np.ndarray, residues: np.ndarray, cfg: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for key in (
        "coords",
        "n_crbn",
        "h_crbn_isolated",
        "h_ddb1_within",
        "A",
        "B",
        "D",
        "a_interface",
        "b_interface",
        "d_interface",
        "pairs",
        "weights",
        "edge_types",
    ):
        if key not in system:
            continue
        value = system[key]
        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
            hasher.update(key.encode())
            hasher.update(str(arr.dtype).encode())
            hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
            if arr.dtype == object:
                hasher.update(json.dumps(arr.tolist(), sort_keys=True, default=str).encode())
            else:
                hasher.update(np.ascontiguousarray(arr).tobytes())
        else:
            hasher.update(key.encode())
            hasher.update(json.dumps(value, sort_keys=True, default=str).encode())
    for key, arr in (("direction", direction), ("residues", residues)):
        data = np.asarray(arr)
        hasher.update(key.encode())
        hasher.update(str(data.dtype).encode())
        hasher.update(np.asarray(data.shape, dtype=np.int64).tobytes())
        hasher.update(np.ascontiguousarray(data).tobytes())
    relevant = {
        "schema": SCHEMA_VERSION,
        "stored_mode_count": int(cfg.get("stored_mode_count", cfg.get("sensitivity_modes", 60))),
        "primary_mode_count": int(cfg.get("primary_mode_count", 20)),
        "near_degenerate_ratio": float(cfg.get("near_degenerate_ratio", 1.20)),
    }
    hasher.update(json.dumps(relevant, sort_keys=True).encode())
    return hasher.hexdigest()


def _spring(coords: np.ndarray, i: int, j: int, weight: float) -> np.ndarray:
    delta = coords[j] - coords[i]
    distance = float(np.linalg.norm(delta))
    if distance <= ZERO_TOL:
        raise ValueError(f"zero-length spring between nodes {i} and {j}")
    unit = delta / distance
    return float(weight) * np.outer(unit, unit)


def _add_block(block: np.ndarray, i: int, j: int, spring: np.ndarray) -> None:
    si = slice(3 * i, 3 * i + 3)
    sj = slice(3 * j, 3 * j + 3)
    block[si, si] += spring
    block[sj, sj] += spring
    block[si, sj] -= spring
    block[sj, si] -= spring


def _components_from_pairs(system: dict[str, Any]) -> dict[str, np.ndarray]:
    coords = np.asarray(system["coords"], dtype=float)
    n_crbn = int(system["n_crbn"])
    crbn_dim = 3 * n_crbn
    ddb1_dim = 3 * (len(coords) - n_crbn)
    pairs = np.asarray(system["pairs"], dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("system['pairs'] must have shape m x 2")
    if "weights" in system:
        weights = np.asarray(system["weights"], dtype=float)
    else:
        weights = np.ones(len(pairs), dtype=float)
    if weights.shape != (len(pairs),):
        raise ValueError("system['weights'] must have one value per pair")

    h_crbn = np.zeros((crbn_dim, crbn_dim), dtype=float)
    h_ddb1 = np.zeros((ddb1_dim, ddb1_dim), dtype=float)
    a_interface = np.zeros((crbn_dim, crbn_dim), dtype=float)
    b_interface = np.zeros((crbn_dim, ddb1_dim), dtype=float)
    d_interface = np.zeros((ddb1_dim, ddb1_dim), dtype=float)

    for (i_raw, j_raw), weight in zip(pairs, weights):
        i = int(i_raw)
        j = int(j_raw)
        if i > j:
            i, j = j, i
        spring = _spring(coords, i, j, float(weight))
        if j < n_crbn:
            _add_block(h_crbn, i, j, spring)
        elif i >= n_crbn:
            _add_block(h_ddb1, i - n_crbn, j - n_crbn, spring)
        else:
            si = slice(3 * i, 3 * i + 3)
            sj = slice(3 * (j - n_crbn), 3 * (j - n_crbn) + 3)
            a_interface[si, si] += spring
            d_interface[sj, sj] += spring
            b_interface[si, sj] -= spring

    return {
        "h_crbn": h_crbn,
        "h_ddb1": h_ddb1,
        "a_interface": a_interface,
        "b_interface": b_interface,
        "d_interface": d_interface,
    }


def _hessian_components(system: dict[str, Any]) -> dict[str, np.ndarray]:
    if all(key in system for key in ("h_ddb1_within", "a_interface", "b_interface", "d_interface")):
        return {
            "h_crbn": _as_float_matrix(system, "h_crbn_isolated"),
            "h_ddb1": _as_float_matrix(system, "h_ddb1_within"),
            "a_interface": _as_float_matrix(system, "a_interface"),
            "b_interface": np.asarray(system["b_interface"], dtype=float),
            "d_interface": _as_float_matrix(system, "d_interface"),
        }
    if "pairs" in system and "coords" in system:
        return _components_from_pairs(system)

    h_crbn = _as_float_matrix(system, "h_crbn_isolated")
    b = np.asarray(system.get("B", np.zeros((h_crbn.shape[0], 0))), dtype=float)
    if b.size and float(np.linalg.norm(b)) > ZERO_TOL:
        raise ValueError("system with nonzero B requires pairs or explicit interface components")
    return {
        "h_crbn": h_crbn,
        "h_ddb1": _as_float_matrix(system, "D"),
        "a_interface": np.zeros_like(h_crbn),
        "b_interface": b,
        "d_interface": np.zeros_like(_as_float_matrix(system, "D")),
    }


def _joint_hessian(components: dict[str, np.ndarray], alpha: float) -> sparse.csr_matrix:
    h_crbn = components["h_crbn"] + float(alpha) * components["a_interface"]
    h_ddb1 = components["h_ddb1"] + float(alpha) * components["d_interface"]
    b = float(alpha) * components["b_interface"]
    dense = np.block([[h_crbn, b], [b.T, h_ddb1]])
    dense = (dense + dense.T) / 2
    return sparse.csr_matrix(dense)


def _dense_modes(hessian: np.ndarray, n_modes: int, rigid_modes: int) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = eigh((hessian + hessian.T) / 2, check_finite=False)
    keep = values > ZERO_TOL
    if int((~keep).sum()) < rigid_modes:
        raise ValueError("Hessian has fewer zero/near-zero modes than expected")
    return values[keep][:n_modes], vectors[:, keep][:, :n_modes]


def _slow_modes(hessian: sparse.csr_matrix, n_modes: int, rigid_modes: int) -> tuple[np.ndarray, np.ndarray]:
    dim = hessian.shape[0]
    if dim <= max(80, n_modes + rigid_modes + 24):
        return _dense_modes(hessian.toarray(), n_modes, rigid_modes)
    return legacy_modes.slow_modes_sparse(hessian, n_modes, rigid_modes=rigid_modes)


def _alpha_zero_modes(components: dict[str, np.ndarray], n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    h_crbn = sparse.csr_matrix(components["h_crbn"])
    h_ddb1 = sparse.csr_matrix(components["h_ddb1"])
    crbn_modes = min(n_modes, max(1, h_crbn.shape[0] - 6))
    ddb1_modes = min(n_modes, max(1, h_ddb1.shape[0] - 6))
    crbn_values, crbn_vectors = _slow_modes(h_crbn, crbn_modes, rigid_modes=6)
    ddb1_values, ddb1_vectors = _slow_modes(h_ddb1, ddb1_modes, rigid_modes=6)
    return legacy_modes.block_diagonal_modes(
        crbn_values,
        crbn_vectors,
        ddb1_values,
        ddb1_vectors,
        n_modes,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= ZERO_TOL:
        raise ValueError("cannot normalize a near-zero vector")
    return vector / norm


def _score_modes(
    values: np.ndarray,
    vectors: np.ndarray,
    direction: np.ndarray,
    crbn_xyz: np.ndarray,
    alpha: float,
    primary_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    crbn_dim = 3 * len(crbn_xyz)
    rigid = mechanics.rigid_basis(crbn_xyz)
    internal_direction = mechanics.project_internal(direction, rigid)
    direction_norm = float(np.linalg.norm(direction))
    internal_direction_norm = float(np.linalg.norm(internal_direction))
    internal_direction_unit = _unit(internal_direction)
    rows: list[dict[str, Any]] = []
    overlaps: list[float] = []
    internal_overlaps: list[float] = []
    amplitudes: list[float] = []
    internal_amplitudes: list[float] = []
    for idx in range(vectors.shape[1]):
        crbn_vec = np.asarray(vectors[:crbn_dim, idx], dtype=float)
        crbn_amp = float(np.linalg.norm(crbn_vec))
        crbn_internal = mechanics.project_internal(crbn_vec, rigid)
        internal_amp = float(np.linalg.norm(crbn_internal))
        overlap = 0.0 if crbn_amp <= ZERO_TOL else float(abs(crbn_vec @ direction) / (crbn_amp * direction_norm))
        internal_overlap = (
            0.0
            if internal_amp <= ZERO_TOL
            else float(abs(crbn_internal @ internal_direction_unit) / internal_amp)
        )
        overlaps.append(overlap)
        internal_overlaps.append(internal_overlap)
        amplitudes.append(crbn_amp)
        internal_amplitudes.append(internal_amp)
        rows.append(
            {
                "interface_alpha": float(alpha),
                "mode": idx + 1,
                "eigenvalue": float(values[idx]),
                "crbn_directional_overlap": overlap,
                "crbn_amplitude": crbn_amp,
                "crbn_internal_overlap": internal_overlap,
                "crbn_internal_amplitude": internal_amp,
                "crbn_raw_norm": crbn_amp,
                "direction_raw_norm": direction_norm,
                "direction_internal_norm": internal_direction_norm,
            }
        )

    primary_count = min(primary_limit, len(rows))
    internal_best = int(np.argmax(internal_overlaps[:primary_count]))
    internal_best60 = int(np.argmax(internal_overlaps))
    raw_best = int(np.argmax(overlaps[:primary_count]))
    raw_best60 = int(np.argmax(overlaps))
    return rows, {
        "interface_alpha": float(alpha),
        "n_modes_returned": int(len(rows)),
        "primary_best_limit": int(primary_count),
        "best_mode": internal_best + 1,
        "best_crbn_internal_overlap": float(internal_overlaps[internal_best]),
        "best_crbn_internal_amplitude": float(internal_amplitudes[internal_best]),
        "best_crbn_directional_overlap": float(internal_overlaps[internal_best]),
        "best_crbn_amplitude": float(internal_amplitudes[internal_best]),
        "internal_best_mode": internal_best + 1,
        "internal_best_overlap": float(internal_overlaps[internal_best]),
        "internal_best_amplitude": float(internal_amplitudes[internal_best]),
        "raw_best_mode": raw_best + 1,
        "raw_best_crbn_directional_overlap": float(overlaps[raw_best]),
        "raw_best_crbn_amplitude": float(amplitudes[raw_best]),
        "raw_sensitivity_best_mode": raw_best60 + 1,
        "raw_sensitivity_best_crbn_directional_overlap": float(overlaps[raw_best60]),
        "raw_sensitivity_best_crbn_amplitude": float(amplitudes[raw_best60]),
        "sensitivity_best_mode": internal_best60 + 1,
        "sensitivity_best_crbn_internal_overlap": float(internal_overlaps[internal_best60]),
        "sensitivity_best_crbn_internal_amplitude": float(internal_amplitudes[internal_best60]),
        "sensitivity_best_crbn_directional_overlap": float(internal_overlaps[internal_best60]),
        "sensitivity_best_crbn_amplitude": float(internal_amplitudes[internal_best60]),
        "higher_mode_21_60_changes_sensitivity_best": bool(internal_best60 >= primary_count),
    }


def _embedded_lowest_crbn_mode(components: dict[str, np.ndarray], full_dim: int) -> np.ndarray:
    values, vectors = _slow_modes(sparse.csr_matrix(components["h_crbn"]), 1, rigid_modes=6)
    _ = values
    embedded = np.zeros(full_dim, dtype=float)
    embedded[: vectors.shape[0]] = vectors[:, 0]
    return embedded


def _cluster_indices(values: np.ndarray, center_index: int, ratio: float) -> np.ndarray:
    center = float(values[center_index])
    keep = [
        idx
        for idx, value in enumerate(values)
        if max(float(value), center) / max(min(float(value), center), ZERO_TOL) <= ratio
    ]
    return np.asarray(keep, dtype=int)


def _cluster_projection(
    vectors: np.ndarray,
    cluster: np.ndarray,
    direction: np.ndarray,
    crbn_xyz: np.ndarray,
) -> float:
    crbn_dim = 3 * len(crbn_xyz)
    rigid = mechanics.rigid_basis(crbn_xyz)
    q = _unit(mechanics.project_internal(direction, rigid))
    columns = []
    for idx in cluster:
        projected = mechanics.project_internal(vectors[:crbn_dim, int(idx)], rigid)
        if np.linalg.norm(projected) > ZERO_TOL:
            columns.append(projected)
    if not columns:
        return 0.0
    u, s, _ = np.linalg.svd(np.column_stack(columns), full_matrices=False)
    retained = s > max(vectors.shape) * float(s[0]) * 1e-12
    return float(np.linalg.norm(u[:, retained].T @ q))


def _principal_angles(previous: np.ndarray | None, current: np.ndarray) -> dict[str, Any] | None:
    if previous is None:
        return None
    singular_values = np.linalg.svd(previous.T @ current, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return {
        "singular_values": [float(v) for v in singular_values],
        "principal_angles_deg": [float(np.degrees(np.arccos(v))) for v in singular_values],
    }


def _track_branches(
    alphas: list[float],
    modes_by_alpha: list[dict[str, Any]],
    start_vector: np.ndarray,
    direction: np.ndarray,
    crbn_xyz: np.ndarray,
    near_ratio: float,
    low_overlap_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    branch_ids = [list(range(modes_by_alpha[0]["vectors"].shape[1]))]
    next_branch = len(branch_ids[0])
    overlap_records: list[dict[str, Any]] = []
    first_vectors = modes_by_alpha[0]["vectors"]
    start_rank = int(np.argmax(np.abs(first_vectors.T @ start_vector)))
    primary_branch = branch_ids[0][start_rank]

    for idx in range(1, len(modes_by_alpha)):
        previous = modes_by_alpha[idx - 1]["vectors"]
        current = modes_by_alpha[idx]["vectors"]
        overlap = np.abs(previous.T @ current)
        row_ind, col_ind = linear_sum_assignment(-overlap)
        current_ids = [-1] * current.shape[1]
        assigned_cols: set[int] = set()
        pairs = []
        for row, col in zip(row_ind, col_ind):
            current_ids[int(col)] = branch_ids[idx - 1][int(row)]
            assigned_cols.add(int(col))
            pairs.append(
                {
                    "previous_mode": int(row + 1),
                    "current_mode": int(col + 1),
                    "branch_id": int(current_ids[int(col)]),
                    "full_vector_overlap": float(overlap[int(row), int(col)]),
                    "low_overlap": bool(overlap[int(row), int(col)] < low_overlap_threshold),
                }
            )
        for col in range(current.shape[1]):
            if col not in assigned_cols:
                current_ids[col] = next_branch
                next_branch += 1
        branch_ids.append(current_ids)
        overlap_records.append(
            {
                "from_alpha": float(alphas[idx - 1]),
                "to_alpha": float(alphas[idx]),
                "matrix": overlap,
                "assignments": pairs,
            }
        )

    summaries: list[dict[str, Any]] = []
    previous_cluster_basis: np.ndarray | None = None
    for idx, item in enumerate(modes_by_alpha):
        ids = branch_ids[idx]
        if primary_branch in ids:
            tracked_index = ids.index(primary_branch)
        else:
            tracked_index = -1
        values = item["values"]
        vectors = item["vectors"]
        if tracked_index >= 0:
            cluster = _cluster_indices(values, tracked_index, near_ratio)
            cluster_basis = vectors[:, cluster]
            angles = _principal_angles(previous_cluster_basis, cluster_basis)
            previous_cluster_basis = cluster_basis
            projection = _cluster_projection(vectors, cluster, direction, crbn_xyz)
            summaries.append(
                {
                    "interface_alpha": float(alphas[idx]),
                    "primary_branch_id": int(primary_branch),
                    "tracked_rank": int(tracked_index + 1),
                    "tracked_eigenvalue": float(values[tracked_index]),
                    "branch_ids_by_rank": [int(v) for v in ids],
                    "cluster_bounds": [int(cluster[0] + 1), int(cluster[-1] + 1)],
                    "cluster_modes": [int(v + 1) for v in cluster],
                    "cluster_projection": projection,
                    "prior_current_subspace": angles,
                    "boundary_flag": bool(tracked_index in {0, len(ids) - 1}),
                    "identity_interpretable": True,
                }
            )
        else:
            summaries.append(
                {
                    "interface_alpha": float(alphas[idx]),
                    "primary_branch_id": int(primary_branch),
                    "tracked_rank": None,
                    "branch_ids_by_rank": [int(v) for v in ids],
                    "boundary_flag": True,
                    "identity_interpretable": False,
                }
            )

    by_alpha = {float(a): row for a, row in zip(alphas, summaries)}
    for record in overlap_records:
        current = by_alpha[float(record["to_alpha"])]
        assignment = next(
            (
                item
                for item in record["assignments"]
                if item["branch_id"] == current["primary_branch_id"]
            ),
            None,
        )
        if assignment is not None:
            current["previous_rank"] = assignment["previous_mode"]
            current["branch_overlap_from_previous"] = assignment["full_vector_overlap"]
            current["low_overlap_flag"] = assignment["low_overlap"]
            current["identity_interpretable"] = bool(
                current["identity_interpretable"] and not assignment["low_overlap"]
            )
    summaries[0]["previous_rank"] = None
    summaries[0]["branch_overlap_from_previous"] = None
    summaries[0]["low_overlap_flag"] = False
    return summaries, overlap_records


def _safe_alpha_name(alpha: float) -> str:
    text = f"{float(alpha):.6g}".replace("-", "m").replace(".", "p")
    return f"alpha_{text}.npz"


def _write_npz(
    path: Path,
    values: np.ndarray,
    vectors: np.ndarray,
    alpha: float,
    input_sha256: str,
) -> None:
    if path.exists():
        _load_npz(path, alpha, input_sha256)
        return
    np.savez_compressed(
        path,
        eigenvalues=np.asarray(values, dtype=np.float64),
        eigenvectors=np.asarray(vectors, dtype=np.float64),
        interface_alpha=np.asarray(float(alpha), dtype=np.float64),
        input_sha256=np.asarray(input_sha256),
        schema_version=np.asarray(SCHEMA_VERSION),
    )


def _validate_mode_arrays(values: np.ndarray, vectors: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"cached mode file has non-1D eigenvalues: {path.name}")
    if vectors.ndim != 2:
        raise ValueError(f"cached mode file has non-2D eigenvectors: {path.name}")
    if vectors.shape[1] != values.shape[0]:
        raise ValueError(f"cached mode file shape mismatch: {path.name}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise ValueError(f"cached mode file has non-finite values: {path.name}")
    if not np.all(values > ZERO_TOL):
        raise ValueError(f"cached mode file contains zero/negative reported modes: {path.name}")
    return values, vectors


def _load_npz(path: Path, alpha: float, input_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as prior:
        required = {"eigenvalues", "eigenvectors", "interface_alpha", "input_sha256", "schema_version"}
        missing = required.difference(prior.files)
        if missing:
            raise ValueError(f"cached mode file is missing fields {sorted(missing)}: {path.name}")
        prior_hash = str(prior["input_sha256"])
        prior_alpha = float(prior["interface_alpha"])
        prior_schema = str(prior["schema_version"])
        if prior_hash != input_sha256 or prior_alpha != float(alpha) or prior_schema != SCHEMA_VERSION:
            raise ValueError(f"existing mode file has incompatible input hash: {path.name}")
        values, vectors = _validate_mode_arrays(prior["eigenvalues"], prior["eigenvectors"], path)
    return values, vectors


def _residual_check(
    hessian: sparse.csr_matrix,
    values: np.ndarray,
    vectors: np.ndarray,
    path: Path,
    *,
    tolerance: float = 5e-5,
) -> None:
    residual = hessian @ vectors - vectors * values.reshape(1, -1)
    scale = max(1.0, float(np.linalg.norm(hessian.toarray() if hessian.shape[0] <= 512 else hessian.data)))
    relative = float(np.linalg.norm(residual) / scale)
    if relative > tolerance:
        raise ValueError(f"cached mode file fails residual check {relative:.3g}: {path.name}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_modes(
    system: dict[str, Any],
    direction: np.ndarray,
    residues: np.ndarray,
    alphas: list[float] | tuple[float, ...] | None,
    output_dir: str | Path,
    config: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Compute alpha-dependent joint modes and branch-continuation summaries."""

    cfg = _load_config(config)
    if alphas is None:
        alphas = cfg.get("interface_strengths", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0])
    alpha_values = [float(v) for v in alphas]
    if not alpha_values:
        raise ValueError("at least one interface alpha is required")
    if alpha_values != sorted(alpha_values):
        raise ValueError("alphas must be sorted for branch continuation")

    coords = np.asarray(system["coords"], dtype=float)
    n_crbn = int(system["n_crbn"])
    crbn_xyz = coords[:n_crbn]
    direction = np.asarray(direction, dtype=float).reshape(-1)
    residues = np.asarray(residues, dtype=int).reshape(-1)
    if direction.shape != (3 * n_crbn,):
        raise ValueError("direction must be a flattened 3N_CRBN vector")
    if residues.shape != (n_crbn,):
        raise ValueError("residues must have one value per CRBN coordinate")

    stored_modes = int(cfg.get("stored_mode_count", cfg.get("sensitivity_modes", 60)))
    primary_modes = int(cfg.get("primary_mode_count", 20))
    near_ratio = float(cfg.get("near_degenerate_ratio", 1.20))
    low_overlap = float(cfg.get("mode_tracking_low_overlap", LOW_OVERLAP_THRESHOLD))
    components = _hessian_components(system)
    digest = _input_hash(system, direction, residues, cfg)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_items: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    alpha_summaries: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    for alpha in alpha_values:
        start = time.perf_counter()
        path = out_dir / _safe_alpha_name(alpha)
        expected_rigid = 12 if abs(alpha) <= ZERO_TOL else 6
        hessian = _joint_hessian(components, alpha)
        if path.exists():
            values, vectors = _load_npz(path, alpha, digest)
            _residual_check(hessian, values, vectors, path)
            cache_status = "cached"
        else:
            if abs(alpha) <= ZERO_TOL:
                values, vectors = _alpha_zero_modes(components, stored_modes)
            else:
                values, vectors = _slow_modes(hessian, stored_modes, rigid_modes=6)
            _write_npz(path, values, vectors, alpha, digest)
            cache_status = "computed"
        rows, summary = _score_modes(values, vectors, direction, crbn_xyz, alpha, primary_modes)
        elapsed = time.perf_counter() - start
        summary.update(
            {
                "expected_rigid_zero_modes": expected_rigid,
                "elapsed_seconds": float(elapsed),
                "mode_file": path.name,
                "mode_cache_status": cache_status,
            }
        )
        for row in rows:
            row.update({"mode_file": path.name})
        mode_rows.extend(rows)
        alpha_summaries.append(summary)
        mode_items.append({"alpha": alpha, "values": values, "vectors": vectors})
        outputs[f"modes_alpha_{alpha:g}"] = path.name
        print(
            f"directional_modes alpha={alpha:g} modes={vectors.shape[1]} "
            f"internal_best20={summary['best_mode']} raw_best20={summary['raw_best_mode']} "
            f"{cache_status} elapsed={elapsed:.3f}s",
            flush=True,
        )

    start_vector = _embedded_lowest_crbn_mode(components, mode_items[0]["vectors"].shape[0])
    branch_summaries, overlap_records = _track_branches(
        alpha_values,
        mode_items,
        start_vector,
        direction,
        crbn_xyz,
        near_ratio,
        low_overlap,
    )
    branch_by_alpha = {row["interface_alpha"]: row for row in branch_summaries}
    for summary in alpha_summaries:
        summary.update(branch_by_alpha[summary["interface_alpha"]])

    csv_path = out_dir / "directional_mode_scores.csv"
    _write_csv(csv_path, mode_rows)
    outputs["mode_scores_csv"] = csv_path.name
    for record in overlap_records:
        matrix_path = out_dir / (
            f"overlap_{_safe_alpha_name(record['from_alpha'])[:-4]}_to_"
            f"{_safe_alpha_name(record['to_alpha'])[:-4]}.npz"
        )
        if not matrix_path.exists():
            np.savez_compressed(
                matrix_path,
                full_vector_overlap=np.asarray(record["matrix"], dtype=np.float64),
                from_alpha=np.asarray(record["from_alpha"], dtype=np.float64),
                to_alpha=np.asarray(record["to_alpha"], dtype=np.float64),
                input_sha256=np.asarray(digest),
                schema_version=np.asarray(SCHEMA_VERSION),
            )
        outputs[f"overlap_{record['from_alpha']:g}_to_{record['to_alpha']:g}"] = matrix_path.name

    result = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": digest,
        "summary": {
            "n_crbn": int(n_crbn),
            "n_residues": int(len(residues)),
            "n_alphas": int(len(alpha_values)),
            "stored_mode_count": stored_modes,
            "primary_mode_count": primary_modes,
            "near_degenerate_ratio": near_ratio,
            "alphas": alpha_values,
        },
        "alpha_summaries": alpha_summaries,
        "branch_assignments": [
            {
                "from_alpha": record["from_alpha"],
                "to_alpha": record["to_alpha"],
                "assignments": record["assignments"],
            }
            for record in overlap_records
        ],
        "outputs": outputs,
    }
    summary_path = out_dir / "directional_modes_summary.json"
    with summary_path.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    outputs["summary_json"] = summary_path.name
    return result


__all__ = ["run_modes"]
