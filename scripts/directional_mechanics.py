#!/usr/bin/env python3
"""Core static mechanics for CRBN directional-response analyses.

The functions in this module keep the four DDB1 treatments in one CRBN
internal coordinate system.  DDB1 is either absent, fixed, restricted to six
rigid-body degrees of freedom, or fully relaxed through a static Schur
complement.  The reduced matrices are static stiffnesses, not dynamical
frequency models for the assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh


ZERO_TOL = 1e-10
ORDER_TOL = 1e-8
HB_TBD_BOUNDARY = 317


def _as_xyz(coords: np.ndarray, name: str = "coords") -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{name} must have shape n x 3")
    if len(coords) < 4 or not np.isfinite(coords).all():
        raise ValueError(f"{name} must contain finite coordinates for at least four nodes")
    return coords


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= ZERO_TOL:
        raise ValueError(f"{name} has near-zero length")
    return vector / norm


def rigid_basis(coords: np.ndarray) -> np.ndarray:
    """Six orthonormal rigid-body directions for a non-collinear node set."""

    coords = _as_xyz(coords)
    centered = coords - coords.mean(axis=0)
    cols = [np.tile(axis, (len(coords), 1)).ravel() for axis in np.eye(3)]
    cols.extend(np.cross(axis, centered).ravel() for axis in np.eye(3))
    u, s, _ = np.linalg.svd(np.column_stack(cols), full_matrices=False)
    if len(s) != 6 or s[-1] <= 1e-10 * s[0]:
        raise ValueError("coordinates do not define six independent rigid motions")
    return u[:, :6]


def internal_basis(coords: np.ndarray) -> np.ndarray:
    """Orthonormal complement to the six CRBN rigid-body directions."""

    rigid = rigid_basis(coords)
    q, _ = np.linalg.qr(rigid, mode="complete")
    return q[:, rigid.shape[1] :]


def project_internal(vector: np.ndarray, rigid: np.ndarray) -> np.ndarray:
    return vector - rigid @ (rigid.T @ vector)


def _edge_column(coords: np.ndarray, i: int, j: int) -> np.ndarray:
    delta = coords[j] - coords[i]
    direction = _unit(delta, "edge direction")
    out = np.zeros(3 * len(coords), dtype=float)
    out[3 * i : 3 * i + 3] = direction
    out[3 * j : 3 * j + 3] = -direction
    return out


def edge_columns(coords: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Return ANM spring columns U where ``H = U diag(k) U.T``."""

    coords = _as_xyz(coords)
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape m x 2")
    cols = np.zeros((3 * len(coords), len(pairs)), dtype=float)
    for col, (i, j) in enumerate(pairs):
        cols[:, col] = _edge_column(coords, int(i), int(j))
    return cols


def _contact_graph(coords: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    iu = np.triu_indices(len(coords), 1)
    keep = (dist[iu] > 1e-6) & (dist[iu] <= cutoff)
    pairs = np.column_stack([iu[0][keep], iu[1][keep]]).astype(int)
    return pairs, dist[iu][keep], dist


def _weights(distances: np.ndarray, weighting: str) -> np.ndarray:
    if weighting == "uniform":
        return np.ones_like(distances, dtype=float)
    if weighting in {"inverse_square", "r2", "distance_r2"}:
        return (15.0 / distances) ** 2
    raise ValueError("weighting must be 'uniform' or 'inverse_square'")


def _hessian_from_columns(columns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (columns * weights[None, :]) @ columns.T


def _add_pair_block(hessian: np.ndarray, i: int, j: int, spring: np.ndarray) -> None:
    si = slice(3 * i, 3 * i + 3)
    sj = slice(3 * j, 3 * j + 3)
    hessian[si, si] += spring
    hessian[sj, sj] += spring
    hessian[si, sj] -= spring
    hessian[sj, si] -= spring


def _accumulate_hessians(
    coords: np.ndarray,
    n_crbn: int,
    pairs: np.ndarray,
    distances: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    crbn_dim = 3 * n_crbn
    ddb1_dim = 3 * (len(coords) - n_crbn)
    h_crbn_isolated = np.zeros((crbn_dim, crbn_dim), dtype=float)
    A = np.zeros((crbn_dim, crbn_dim), dtype=float)
    B = np.zeros((crbn_dim, ddb1_dim), dtype=float)
    D = np.zeros((ddb1_dim, ddb1_dim), dtype=float)

    for (i_raw, j_raw), distance, weight in zip(pairs, distances, weights):
        i = int(i_raw)
        j = int(j_raw)
        delta = coords[j] - coords[i]
        spring = float(weight) * np.outer(delta, delta) / float(distance) ** 2
        if j < n_crbn:
            _add_pair_block(h_crbn_isolated, i, j, spring)
            _add_pair_block(A, i, j, spring)
        elif i >= n_crbn:
            _add_pair_block(D, i - n_crbn, j - n_crbn, spring)
        else:
            si = slice(3 * i, 3 * i + 3)
            sj = slice(3 * (j - n_crbn), 3 * (j - n_crbn) + 3)
            A[si, si] += spring
            D[sj, sj] += spring
            B[si, sj] -= spring
    return h_crbn_isolated, A, B, D


def build_system(
    coords: np.ndarray,
    n_crbn: int,
    cutoff: float,
    weighting: str = "uniform",
) -> dict[str, Any]:
    """Build CRBN/DDB1 ANM blocks from one fixed contact topology.

    ``weighting='inverse_square'`` applies ``k=(15/r)^2`` to the contacts
    selected by the same cutoff topology.  Returned ``A``, ``B`` and ``D`` are
    the full coupled blocks at interface strength 1.
    """

    coords = _as_xyz(coords)
    if not 4 <= int(n_crbn) < len(coords):
        raise ValueError("n_crbn must leave at least one partner node and four CRBN nodes")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    n_crbn = int(n_crbn)
    pairs, distances, _ = _contact_graph(coords, float(cutoff))
    weights = _weights(distances, weighting)
    h_crbn_isolated, A, B, D = _accumulate_hessians(coords, n_crbn, pairs, distances, weights)
    full_hessian = np.block([[A, B], [B.T, D]])
    crbn_dim = 3 * n_crbn

    edge_type = np.full(len(pairs), "ddb1_ddb1", dtype=object)
    edge_type[(pairs[:, 0] < n_crbn) & (pairs[:, 1] < n_crbn)] = "crbn_crbn"
    edge_type[(pairs[:, 0] < n_crbn) & (pairs[:, 1] >= n_crbn)] = "interface"

    return {
        "coords": coords,
        "crbn_xyz": coords[:n_crbn].copy(),
        "ddb1_xyz": coords[n_crbn:].copy(),
        "n_crbn": n_crbn,
        "cutoff": float(cutoff),
        "weighting": weighting,
        "pairs": pairs,
        "distances": distances,
        "weights": weights,
        "edge_types": edge_type,
        "hessian": full_hessian,
        "h_crbn_isolated": h_crbn_isolated,
        "A": A,
        "B": B,
        "D": D,
        "crbn_dim": crbn_dim,
        "ddb1_dim": full_hessian.shape[0] - crbn_dim,
    }


@dataclass(frozen=True)
class _Reduction:
    name: str
    H_eff: np.ndarray
    B_reduced: np.ndarray
    factor: tuple[np.ndarray, bool] | None
    partner_basis: np.ndarray | None | str


def _state_from_effective(
    reduction: _Reduction,
    U: np.ndarray,
    q: np.ndarray,
) -> dict[str, Any]:
    H = (U.T @ reduction.H_eff @ U)
    H = (H + H.T) / 2
    eigenvalues, eigenvectors = eigh(H, check_finite=False)
    if float(eigenvalues[0]) <= ZERO_TOL:
        raise ValueError(f"{reduction.name} reduced Hessian is not positive definite")
    G = (eigenvectors / eigenvalues) @ eigenvectors.T
    G = (G + G.T) / 2
    c_close = float(q @ G @ q)
    mean = float(np.trace(G) / len(q))
    if c_close <= 0 or mean <= 0:
        raise ValueError(f"{reduction.name} produced nonpositive compliance")
    return {
        "name": reduction.name,
        "U": U,
        "q": q,
        "H": H,
        "G": G,
        "B": U.T @ reduction.B_reduced,
        "factor": reduction.factor,
        "partner_basis": reduction.partner_basis,
        "C_close": c_close,
        "mean_compliance": mean,
        "S_close": c_close / mean,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }


def _schur_reduction(
    name: str,
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    partner_basis: np.ndarray | str,
) -> _Reduction:
    if isinstance(partner_basis, str):
        if partner_basis != "identity":
            raise ValueError("partner_basis string must be 'identity'")
        B_reduced = B
        D_reduced = D
        exposed_basis: np.ndarray | str = "identity"
    else:
        Z = np.asarray(partner_basis, dtype=float)
        B_reduced = B @ Z
        D_reduced = Z.T @ D @ Z
        exposed_basis = Z
    D_reduced = (D_reduced + D_reduced.T) / 2
    factor = cho_factor(D_reduced, lower=True, check_finite=True)
    correction = B_reduced @ cho_solve(factor, B_reduced.T, check_finite=False)
    H_eff = (A - correction + (A - correction).T) / 2
    return _Reduction(name, H_eff, B_reduced, factor, exposed_basis)


def _psd_leq(left: np.ndarray, right: np.ndarray, tol: float = ORDER_TOL) -> bool:
    diff = (right - left + right.T - left.T) / 2
    minimum = float(eigh(diff, eigvals_only=True, subset_by_index=(0, 0), check_finite=False)[0])
    scale = max(1.0, float(np.max(np.abs(np.diag(left)))), float(np.max(np.abs(np.diag(right)))))
    return bool(minimum >= -tol * scale)


def _psd_diagnostic(left: np.ndarray, right: np.ndarray, tol: float = ORDER_TOL) -> dict[str, float | bool]:
    diff = (right - left + right.T - left.T) / 2
    minimum = float(eigh(diff, eigvals_only=True, subset_by_index=(0, 0), check_finite=False)[0])
    scale = max(1.0, float(np.max(np.abs(np.diag(left)))), float(np.max(np.abs(np.diag(right)))))
    threshold = -tol * scale
    return {"min_eigenvalue": minimum, "threshold": threshold, "scale": scale, "pass": minimum >= threshold}


def make_states(
    system: dict[str, Any],
    crbn_xyz: np.ndarray,
    direction: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return isolated/fixed/rigid/flexible static states in one CRBN basis."""

    crbn_xyz = _as_xyz(crbn_xyz, "crbn_xyz")
    n_crbn = int(system["n_crbn"])
    if len(crbn_xyz) != n_crbn:
        raise ValueError("crbn_xyz length must match system['n_crbn']")
    direction = np.asarray(direction, dtype=float).reshape(-1)
    if direction.shape != (3 * n_crbn,):
        raise ValueError("direction must be a flattened 3N_CRBN vector")

    U = internal_basis(crbn_xyz)
    rigid = rigid_basis(crbn_xyz)
    q_ambient = project_internal(direction, rigid)
    q_norm = float(np.linalg.norm(q_ambient))
    if q_norm <= ZERO_TOL:
        raise ValueError("direction has no CRBN-internal component")
    q = U.T @ (q_ambient / q_norm)
    q /= np.linalg.norm(q)

    A = np.asarray(system["A"], dtype=float)
    B = np.asarray(system["B"], dtype=float)
    D = np.asarray(system["D"], dtype=float)
    isolated = np.asarray(system["h_crbn_isolated"], dtype=float)
    Z = rigid_basis(np.asarray(system["ddb1_xyz"], dtype=float))

    reductions = [
        _Reduction("isolated", isolated, np.zeros((A.shape[0], 0)), None, None),
        _Reduction("fixed", A, np.zeros((A.shape[0], 0)), None, None),
        _schur_reduction("rigid", A, B, D, Z),
        _schur_reduction("flexible", A, B, D, "identity"),
    ]
    states = {item.name: _state_from_effective(item, U, q) for item in reductions}

    order_pairs = [
        ("isolated", "flexible"),
        ("flexible", "rigid"),
        ("rigid", "fixed"),
    ]
    stiffness_order_diagnostics = {
        f"{a}_le_{b}": _psd_diagnostic(states[a]["H"], states[b]["H"]) for a, b in order_pairs
    }
    stiffness_order = {key: bool(value["pass"]) for key, value in stiffness_order_diagnostics.items()}
    inverse_order_diagnostics = {
        f"{b}_G_le_{a}_G": _psd_diagnostic(states[b]["G"], states[a]["G"])
        for a, b in order_pairs
    }
    inverse_order = {
        key: bool(value["pass"]) for key, value in inverse_order_diagnostics.items()
    }
    compliance_order = {
        "fixed_C_le_rigid_C": states["fixed"]["C_close"] <= states["rigid"]["C_close"] + ORDER_TOL,
        "rigid_C_le_flexible_C": states["rigid"]["C_close"] <= states["flexible"]["C_close"] + ORDER_TOL,
        "flexible_C_le_isolated_C": states["flexible"]["C_close"] <= states["isolated"]["C_close"] + ORDER_TOL,
    }
    checks = {
        "internal_dimension": int(U.shape[1]),
        "target_internal_norm": q_norm,
        "same_U_all_states": True,
        "same_q_all_states": True,
        "stiffness_order": stiffness_order,
        "stiffness_order_diagnostics": stiffness_order_diagnostics,
        "inverse_order": inverse_order,
        "inverse_order_diagnostics": inverse_order_diagnostics,
        "compliance_order": compliance_order,
        "all_order_checks_pass": all(stiffness_order.values())
        and all(inverse_order.values())
        and all(compliance_order.values()),
        "R_body": float(np.log(states["rigid"]["S_close"]) - np.log(states["fixed"]["S_close"])),
        "R_internal": float(
            np.log(states["flexible"]["S_close"]) - np.log(states["rigid"]["S_close"])
        ),
        "M": float(
            np.log(states["flexible"]["S_close"]) - np.log(states["isolated"]["S_close"])
        ),
    }
    return states, checks


def full_static_response(
    state: dict[str, Any],
    force_internal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Solve the reduced static problem and return CRBN plus partner response."""

    q = np.asarray(state["q"] if force_internal is None else force_internal, dtype=float)
    x = state["G"] @ q
    crbn = state["U"] @ x
    if state["factor"] is None:
        return crbn, None
    y = -cho_solve(state["factor"], state["B"].T @ x, check_finite=False)
    partner_basis = state["partner_basis"]
    if isinstance(partner_basis, str):
        partner = y
    else:
        partner = partner_basis @ y
    return crbn, partner


def _symmetric_inverse_sqrt(gram: np.ndarray) -> np.ndarray:
    values, vectors = eigh((gram + gram.T) / 2, check_finite=False)
    if float(values[0]) <= ZERO_TOL:
        raise ValueError("rotation null basis is rank deficient")
    return (vectors / np.sqrt(values)) @ vectors.T


def _frozen_axis(frozen_geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        axis = np.asarray(frozen_geometry["axis_unit_vector"], dtype=float)
        point = np.asarray(frozen_geometry["axis_point_A"], dtype=float)
    except KeyError as exc:
        raise ValueError("frozen_geometry must contain axis_unit_vector and axis_point_A") from exc
    if point.shape != (3,):
        raise ValueError("axis_point_A must be a 3-vector in the current reference frame")
    return _unit(axis, "axis_unit_vector"), point


def geometry_directions(
    crbn_xyz: np.ndarray,
    residues: np.ndarray,
    direction: np.ndarray,
    frozen_geometry: dict[str, Any],
    seed: int = 20260906,
    n_draws: int = 5000,
) -> dict[str, Any]:
    """Generate covariant TBD rotation null directions.

    ``frozen_geometry`` must already be mapped into the coordinate frame of
    ``crbn_xyz``.  The caller can obtain that mapping by Kabsch-aligning the
    frozen mean-open anchor residues to the reference anchor residues and
    applying the same transform to the stored screw-axis point and vector.
    """

    crbn_xyz = _as_xyz(crbn_xyz, "crbn_xyz")
    residues = np.asarray(residues, dtype=int)
    if residues.shape != (len(crbn_xyz),):
        raise ValueError("residues must have one entry per CRBN coordinate")
    direction = np.asarray(direction, dtype=float).reshape(-1)
    if direction.shape != (3 * len(crbn_xyz),):
        raise ValueError("direction must be a flattened 3N_CRBN vector")
    if n_draws < 0:
        raise ValueError("n_draws must be nonnegative")

    moving = residues > HB_TBD_BOUNDARY
    if int(moving.sum()) < 3:
        raise ValueError("at least three TBD residues are required")
    axis_a, axis_point = _frozen_axis(frozen_geometry)
    centroid = crbn_xyz[moving].mean(axis=0)
    pivot = axis_point + axis_a * float((centroid - axis_point) @ axis_a)

    chosen_index: int | None = None
    for idx in np.where(moving)[0][np.argsort(residues[moving])]:
        radial = crbn_xyz[idx] - pivot
        radial -= axis_a * float(radial @ axis_a)
        if np.linalg.norm(radial) > ZERO_TOL:
            chosen_index = int(idx)
            break
    if chosen_index is None:
        raise ValueError("TBD residues are collinear with the frozen screw axis")
    seed_radial = crbn_xyz[chosen_index] - pivot
    seed_radial -= axis_a * float(seed_radial @ axis_a)
    axis_b = _unit(seed_radial, "axis_b")
    axis_c = np.cross(axis_a, axis_b)
    axis_c = _unit(axis_c, "axis_c")

    raw = np.zeros((3 * len(crbn_xyz), 3), dtype=float)
    for col, axis in enumerate((axis_a, axis_b, axis_c)):
        disp = np.zeros_like(crbn_xyz)
        disp[moving] = np.cross(axis, crbn_xyz[moving] - pivot)
        raw[:, col] = disp.ravel()

    rigid = rigid_basis(crbn_xyz)
    projected = raw - rigid @ (rigid.T @ raw)
    whitening = _symmetric_inverse_sqrt(projected.T @ projected)
    basis = projected @ whitening

    finite_direction = project_internal(direction, rigid)
    finite_direction = _unit(finite_direction, "finite direction")
    axis_tangent = project_internal(raw[:, 0], rigid)
    tangent = _unit(axis_tangent, "frozen-axis rotation tangent")
    observed_coefficients = basis.T @ tangent
    finite_tangent_overlap = float(abs(finite_direction @ tangent))

    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(n_draws, 3))
    norms = np.linalg.norm(coefficients, axis=1)
    if n_draws:
        coefficients = coefficients / norms[:, None]
    sampled = coefficients @ basis.T
    return {
        "basis": basis,
        "sampled_directions": sampled,
        "coefficients": coefficients,
        "finite_direction": finite_direction,
        "observed_rotation_tangent": tangent,
        "observed_rotation_coefficients": observed_coefficients,
        "finite_tangent_overlap": finite_tangent_overlap,
        "pivot": pivot,
        "axis_point_A": axis_point,
        "intrinsic_rotation_axes": np.vstack([axis_a, axis_b, axis_c]),
        "axis_seed_residue": int(residues[chosen_index]),
        "moving_mask": moving,
        "seed": int(seed),
        "n_draws": int(n_draws),
        "definition": (
            "TBD rotation null directions use the frozen screw axis mapped into the "
            "reference frame; the pivot is the TBD centroid projected onto that axis."
        ),
    }
