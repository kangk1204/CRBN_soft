#!/usr/bin/env python3
"""Directional contact perturbations for the CRBN-DDB1 mechanics analysis.

The public entry point is :func:`analyse_contacts`.  It accepts a prepared
system and the four static response states produced by the directional-core
runner, then reports how prespecified residue-contact groups alter closure
compliance, mean compliance, and the relative closure specificity.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.linalg import cho_solve, solve

try:
    import strengthen_contacts as legacy_contacts
except ModuleNotFoundError:
    from scripts import strengthen_contacts as legacy_contacts


DEFAULT_CONFIG: dict[str, Any] = {
    "cutoff_A": 15.0,
    "classes": ("HB_TBD", "CRBN_DDB1"),
    "spring_factors": (0.8, 0.9, 1.1, 1.2),
    "central_low_factor": 0.9,
    "central_high_factor": 1.1,
    "ridge_alpha": 1.0,
    "ridge_min_training": 20,
    "ridge_min_same_domain": 10,
    "ridge_neighbor_cutoff_A": 10.0,
    "state_order": ("isolated", "fixed", "rigid", "flexible"),
    "perturbed_states": ("fixed", "rigid", "flexible"),
}

REQUIRED_STATE_FIELDS = ("U", "q", "H", "G", "B", "factor", "partner_basis")


@dataclass(frozen=True)
class ContactGroup:
    residue: int
    contact_class: str
    edge_ids: tuple[int, ...]
    group_id: str


def _merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        source = dict(config.get("contact", config))
        if "primary_cutoff_A" in config and "cutoff_A" not in source:
            source["cutoff_A"] = config["primary_cutoff_A"]
        key_map = {
            "ridge_minimum_training": "ridge_min_training",
            "ridge_minimum_same_domain": "ridge_min_same_domain",
            "ridge_exclusion_distance_A": "ridge_neighbor_cutoff_A",
        }
        for old, new in key_map.items():
            if old in source and new not in source:
                source[new] = source[old]
        if "inverse_square_reference_distance_A" in config:
            source.setdefault(
                "inverse_square_reference_distance_A",
                config["inverse_square_reference_distance_A"],
            )
        merged.update(source)
    merged["spring_factors"] = tuple(float(v) for v in merged["spring_factors"])
    merged["classes"] = tuple(str(v) for v in merged["classes"])
    merged["perturbed_states"] = tuple(str(v) for v in merged["perturbed_states"])
    return merged


def _domain(residue: int) -> str:
    if residue < 187:
        return "NTD"
    if residue < 318:
        return "HB"
    return "TBD"


def _group_id(residue: int, contact_class: str) -> str:
    return f"{int(residue)}:{contact_class}"


def _parse_discovery_keys(discovery_keys: Iterable[Any] | None) -> set[str] | None:
    if discovery_keys is None:
        return None
    parsed: set[str] = set()
    for item in discovery_keys:
        if isinstance(item, str):
            parsed.add(item if ":" in item else item.replace(",", ":"))
        elif isinstance(item, dict):
            parsed.add(_group_id(int(item["residue"]), str(item["contact_class"])))
        else:
            residue, contact_class = item
            parsed.add(_group_id(int(residue), str(contact_class)))
    return parsed


def _validate_state(name: str, state: dict[str, Any], crbn_dim: int | None = None) -> None:
    missing = [field for field in REQUIRED_STATE_FIELDS if field not in state]
    if missing:
        raise ValueError(f"{name} state is missing fields: {missing}")
    for metric in ("C_close", "mean_compliance", "S_close"):
        if metric not in state:
            raise ValueError(f"{name} state is missing baseline metric {metric}")
    u = np.asarray(state["U"], dtype=float)
    q = np.asarray(state["q"], dtype=float).reshape(-1)
    h = np.asarray(state["H"], dtype=float)
    g = np.asarray(state["G"], dtype=float)
    b = np.asarray(state["B"], dtype=float)
    if u.ndim != 2:
        raise ValueError(f"{name} state U must be a two-dimensional basis")
    if crbn_dim is not None and u.shape[0] != crbn_dim:
        raise ValueError(f"{name} state U row count does not match CRBN coordinate dimension")
    if q.shape[0] != u.shape[1]:
        raise ValueError(f"{name} state q length does not match U internal dimension")
    if h.shape != (u.shape[1], u.shape[1]) or g.shape != h.shape:
        raise ValueError(f"{name} state H/G shapes must match the internal dimension")
    if b.ndim != 2:
        raise ValueError(f"{name} state B must be a two-dimensional coupling block")
    if b.shape[1] and b.shape[0] != u.shape[1]:
        raise ValueError(f"{name} state B row count must match the internal dimension")


def _solve_factor(factor: Any, rhs: np.ndarray) -> np.ndarray:
    if rhs.size == 0:
        return np.zeros_like(rhs, dtype=float)
    if callable(factor):
        return np.asarray(factor(rhs), dtype=float)
    if isinstance(factor, tuple) and len(factor) == 2:
        return cho_solve(factor, rhs, check_finite=False)
    if hasattr(factor, "solve"):
        return np.asarray(factor.solve(rhs), dtype=float)
    return solve(np.asarray(factor, dtype=float), rhs, assume_a="sym", check_finite=False)


def _state_dof(state: dict[str, Any]) -> int:
    if "dof" in state:
        return int(state["dof"])
    return int(np.asarray(state["U"]).shape[1])


def _edge_columns(coords: np.ndarray, edges: list[tuple[int, int]], weights: np.ndarray) -> np.ndarray:
    columns = np.zeros((3 * len(coords), len(edges)), dtype=float)
    for col, ((i, j), weight) in enumerate(zip(edges, weights)):
        delta = coords[j] - coords[i]
        distance = float(np.linalg.norm(delta))
        if distance <= 0.0:
            raise ValueError(f"Zero-length edge {i}-{j}")
        unit = np.sqrt(float(weight)) * delta / distance
        columns[3 * i:3 * i + 3, col] = unit
        columns[3 * j:3 * j + 3, col] = -unit
    return columns


def _edge_weights(
    system: dict[str, Any],
    candidate_edges: list[tuple[int, int]],
    coords: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    pair_values = system.get("pairs", system.get("edge_pairs"))
    weight_values = system.get("weights", system.get("edge_weights"))
    weighting = str(system.get("weighting", system.get("weighting_scheme", "")))
    if isinstance(weight_values, str):
        weighting = weight_values
        weight_values = None
    if pair_values is None and weight_values is not None:
        raise ValueError("system weights were provided without system pairs")
    if pair_values is None or weight_values is None:
        if weighting == "inverse_square":
            reference = float(
                system.get(
                    "inverse_square_reference_distance_A",
                    config.get("inverse_square_reference_distance_A", config["cutoff_A"]),
                )
            )
            return np.asarray(
                [
                    (reference / float(np.linalg.norm(coords[j] - coords[i]))) ** 2
                    for i, j in candidate_edges
                ],
                dtype=float,
            )
        return np.ones(len(candidate_edges), dtype=float)
    pairs = [tuple(sorted(map(int, pair))) for pair in pair_values]
    weights = np.asarray(weight_values, dtype=float)
    if len(pairs) != len(weights):
        raise ValueError("system pairs and weights lengths differ")
    weight_by_pair = {pair: float(weight) for pair, weight in zip(pairs, weights)}
    out = []
    missing = []
    for edge in candidate_edges:
        key = tuple(sorted(map(int, edge)))
        if key not in weight_by_pair:
            missing.append(key)
        else:
            out.append(weight_by_pair[key])
    if missing:
        raise ValueError(f"Missing weights for candidate edges: {missing[:5]}")
    return np.asarray(out, dtype=float)


def _candidate_groups(
    system: dict[str, Any],
    coords: np.ndarray,
    residues: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[tuple[int, int]], list[ContactGroup], np.ndarray]:
    cutoff = float(system.get("cutoff", system.get("cutoff_A", config["cutoff_A"])))
    edges, groups, degrees = legacy_contacts.candidate_groups(coords, residues, cutoff)
    classes = set(config["classes"])
    contact_groups = [
        ContactGroup(
            residue=int(residue),
            contact_class=str(contact_class),
            edge_ids=tuple(int(i) for i in edge_ids),
            group_id=_group_id(int(residue), str(contact_class)),
        )
        for (residue, contact_class), edge_ids in groups.items()
        if str(contact_class) in classes
    ]
    contact_groups.sort(key=lambda g: (g.contact_class, g.residue))
    return edges, contact_groups, np.asarray(degrees, dtype=int)


def _reduced_partner_columns(state: dict[str, Any], partner_columns: np.ndarray) -> np.ndarray:
    basis = state["partner_basis"]
    if basis is None:
        return np.zeros((0, partner_columns.shape[1]), dtype=float)
    if isinstance(basis, str):
        if basis != "identity":
            raise ValueError(f"Unsupported partner_basis string: {basis}")
        return partner_columns
    return np.asarray(basis, dtype=float).T @ partner_columns


def _prepare_state_updates(state: dict[str, Any], columns: np.ndarray, crbn_dim: int) -> dict[str, np.ndarray]:
    u = np.asarray(state["U"], dtype=float)
    q = np.asarray(state["q"], dtype=float).reshape(-1)
    g = np.asarray(state["G"], dtype=float)
    b = np.asarray(state["B"], dtype=float)
    crbn_columns = columns[:crbn_dim]
    partner_columns = columns[crbn_dim:]
    uc = u.T @ crbn_columns
    ud = _reduced_partner_columns(state, partner_columns)
    if ud.shape[0] == 0:
        dinv_ud = ud
        condensed = uc
        partner_gram = np.zeros((columns.shape[1], columns.shape[1]), dtype=float)
    else:
        dinv_ud = _solve_factor(state["factor"], ud)
        condensed = uc - b @ dinv_ud
        partner_gram = ud.T @ dinv_ud
    response = g @ condensed
    gram = condensed.T @ response + partner_gram
    gram = (gram + gram.T) / 2.0
    return {
        "condensed": condensed,
        "response": response,
        "gram": gram,
        "response_gram": response.T @ response,
        "closure_response": response.T @ q,
        "partner_columns": ud,
        "partner_dinv_columns": dinv_ud,
    }


def _perturbation_metrics(
    state: dict[str, Any],
    updates: dict[str, np.ndarray],
    indices: Iterable[int],
    spring_factor: float,
) -> dict[str, float]:
    if spring_factor <= 0.0:
        raise ValueError("Spring factor must be positive")
    ids = np.asarray(tuple(indices), dtype=int)
    delta = float(spring_factor) - 1.0
    baseline_c = float(state["C_close"])
    baseline_mean = float(state["mean_compliance"])
    baseline_s = float(state["S_close"])
    if ids.size == 0 or delta == 0.0:
        return {
            "C_close": baseline_c,
            "mean_compliance": baseline_mean,
            "S_close": baseline_s,
            "delta_log_C_close": 0.0,
            "delta_log_mean_compliance": 0.0,
            "delta_log_S_close": 0.0,
        }
    gram = updates["gram"][np.ix_(ids, ids)]
    step = delta * solve(
        np.eye(len(ids)) + delta * gram,
        np.eye(len(ids)),
        assume_a="sym",
        check_finite=False,
    )
    closure = updates["closure_response"][ids]
    c_close = baseline_c - float(closure @ step @ closure)
    response_gram = updates["response_gram"][np.ix_(ids, ids)]
    mean = baseline_mean - float(np.sum(step.T * response_gram)) / _state_dof(state)
    if c_close <= 0.0 or mean <= 0.0:
        raise ValueError("Positive spring perturbation produced nonpositive compliance")
    specificity = c_close / mean
    return {
        "C_close": c_close,
        "mean_compliance": mean,
        "S_close": specificity,
        "delta_log_C_close": float(np.log(c_close / baseline_c)),
        "delta_log_mean_compliance": float(np.log(mean / baseline_mean)),
        "delta_log_S_close": float(np.log(specificity / baseline_s)),
    }


def _exact_derivative(state: dict[str, Any], updates: dict[str, np.ndarray], indices: Iterable[int]) -> dict[str, float]:
    ids = np.asarray(tuple(indices), dtype=int)
    closure = updates["closure_response"][ids]
    d_c = -float(closure @ closure)
    d_mean = -float(np.trace(updates["response_gram"][np.ix_(ids, ids)])) / _state_dof(state)
    baseline_c = float(state["C_close"])
    baseline_mean = float(state["mean_compliance"])
    d_log_c = d_c / baseline_c
    d_log_mean = d_mean / baseline_mean
    return {
        "derivative_log_C_close": float(d_log_c),
        "derivative_log_mean_compliance": float(d_log_mean),
        "derivative_log_S_close": float(d_log_c - d_log_mean),
    }


def _central_dg(metrics_by_factor: dict[float, dict[str, float]], config: dict[str, Any]) -> float:
    low = float(config["central_low_factor"])
    high = float(config["central_high_factor"])
    return float(
        (metrics_by_factor[high]["delta_log_S_close"] - metrics_by_factor[low]["delta_log_S_close"])
        / (high - low)
    )


def _rank_groups(rows: list[dict[str, Any]], state_name: str) -> None:
    for contact_class in sorted({row["contact_class"] for row in rows if row["status"] == "present"}):
        ranked = sorted(
            [row for row in rows if row["status"] == "present" and row["contact_class"] == contact_class],
            key=lambda row: (-abs(float(row[f"{state_name}_D_g"])), int(row["residue"])),
        )
        for rank, row in enumerate(ranked, 1):
            row[f"{state_name}_rank"] = rank
            row[f"{state_name}_class_n"] = len(ranked)
            row[f"{state_name}_rank_fraction"] = rank / len(ranked)


def _shared_edge_annotations(groups: list[ContactGroup]) -> dict[str, dict[str, str]]:
    by_edge: dict[int, list[str]] = {}
    by_set: dict[tuple[int, ...], list[str]] = {}
    for group in groups:
        by_set.setdefault(group.edge_ids, []).append(group.group_id)
        for edge_id in group.edge_ids:
            by_edge.setdefault(edge_id, []).append(group.group_id)
    annotations: dict[str, dict[str, str]] = {}
    for group in groups:
        shared = sorted(
            {
                other
                for edge_id in group.edge_ids
                for other in by_edge.get(edge_id, [])
                if other != group.group_id
            }
        )
        identical = sorted(other for other in by_set[group.edge_ids] if other != group.group_id)
        annotations[group.group_id] = {
            "shared_edge_group_ids": ";".join(shared),
            "identical_edge_group_ids": ";".join(identical),
        }
    return annotations


def _ridge_fit(
    present_rows: list[dict[str, Any]],
    coords: np.ndarray,
    residues: np.ndarray,
    alpha: float,
    min_training: int,
    min_same_domain: int,
    neighbor_cutoff: float,
) -> list[dict[str, Any]]:
    rows = []
    crbn_lookup = {int(residue): idx for idx, residue in enumerate(residues)}
    domains = sorted({_domain(int(row["residue"])) for row in present_rows})
    for target in present_rows:
        same_class = [row for row in present_rows if row["contact_class"] == target["contact_class"]]
        target_edges = set(target["edge_ids_tuple"])
        target_idx = crbn_lookup[int(target["residue"])]
        train = []
        for candidate in same_class:
            candidate_idx = crbn_lookup[int(candidate["residue"])]
            shares_edges = bool(target_edges.intersection(candidate["edge_ids_tuple"]))
            near = float(np.linalg.norm(coords[target_idx] - coords[candidate_idx])) <= neighbor_cutoff
            if shares_edges or near:
                continue
            train.append(candidate)
        deduplicated = {}
        for candidate in sorted(train, key=lambda row: int(row["residue"])):
            deduplicated.setdefault(candidate["edge_ids_tuple"], candidate)
        train = list(deduplicated.values())
        same_domain_count = sum(row["domain"] == target["domain"] for row in train)
        base = {
            "group_id": target["group_id"],
            "residue": target["residue"],
            "contact_class": target["contact_class"],
            "training_n": len(train),
            "same_domain_training_n": same_domain_count,
        }
        if len(train) < min_training or same_domain_count < min_same_domain:
            rows.append({**base, "status": "insufficient"})
            continue
        x_train = np.asarray([_ridge_features(row, domains) for row in train], dtype=float)
        y_train = np.asarray([row["flexible_derivative_log_S_close_per_edge"] for row in train], dtype=float)
        continuous = x_train[:, :3]
        mean = continuous.mean(axis=0)
        sd = continuous.std(axis=0)
        sd[sd == 0.0] = 1.0
        x_train[:, :3] = (continuous - mean) / sd
        x_target = np.asarray(_ridge_features(target, domains), dtype=float)
        x_target[:3] = (x_target[:3] - mean) / sd
        design = np.column_stack([np.ones(len(train)), x_train])
        penalty = np.eye(design.shape[1]) * float(alpha)
        penalty[0, 0] = 0.0
        n_train = float(len(train))
        coef = solve(
            design.T @ design / n_train + penalty,
            design.T @ y_train / n_train,
            assume_a="pos",
            check_finite=False,
        )
        prediction = float(np.r_[1.0, x_target] @ coef)
        residual = float(target["flexible_derivative_log_S_close_per_edge"] - prediction)
        mse = float(np.mean((design @ coef - y_train) ** 2))
        rows.append(
            {
                **base,
                "status": "fit",
                "observed_per_edge_derivative": target["flexible_derivative_log_S_close_per_edge"],
                "predicted_per_edge_derivative": prediction,
                "residual_per_edge_derivative": residual,
                "training_mse": mse,
                "ridge_alpha": float(alpha),
                "ridge_objective": "mean_squared_error_plus_alpha_l2",
            }
        )
    return rows


def _ridge_features(row: dict[str, Any], domains: list[str]) -> list[float]:
    return [
        float(np.log(max(float(row["contact_count"]), 1.0))),
        float(np.log(max(float(row["joint_degree"]), 1.0))),
        float(row["axis_distance_A"]),
        *[1.0 if row["domain"] == domain else 0.0 for domain in domains],
    ]


def _load_discovery_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyse_contacts(
    system: dict[str, Any],
    states: dict[str, dict[str, Any]],
    coords: np.ndarray,
    residues: np.ndarray,
    axis_distances: np.ndarray,
    discovery_keys: Iterable[Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyse prespecified contact groups across fixed, rigid and flexible states.

    ``coords`` must contain CRBN rows followed by DDB1 rows. ``residues`` and
    ``axis_distances`` describe the CRBN rows only.  ``discovery_keys`` freezes
    the residue-by-contact-class universe from the 118 discovery table; absent
    groups are reported as missing rather than scored as zero.
    """

    cfg = _merged_config(config)
    coords = np.asarray(coords, dtype=float)
    residues = np.asarray(residues, dtype=int)
    axis_distances = np.asarray(axis_distances, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be an n x 3 array")
    if len(residues) != len(axis_distances):
        raise ValueError("residues and axis_distances must have the same length")
    crbn_dim = 3 * len(residues)
    if coords.shape[0] <= len(residues):
        raise ValueError("coords must include CRBN rows followed by DDB1 rows")
    for name in cfg["perturbed_states"]:
        _validate_state(name, states[name], crbn_dim)

    universe = _parse_discovery_keys(discovery_keys)
    candidate_edges, detected_groups, degrees = _candidate_groups(system, coords, residues, cfg)
    edge_weights = _edge_weights(system, candidate_edges, coords, cfg)
    columns = _edge_columns(coords, candidate_edges, edge_weights)
    detected_by_id = {group.group_id: group for group in detected_groups}
    if universe is None:
        universe = set(detected_by_id)
    universe_groups = [detected_by_id[group_id] for group_id in sorted(universe) if group_id in detected_by_id]
    missing_ids = sorted(group_id for group_id in universe if group_id not in detected_by_id)
    annotations = _shared_edge_annotations(universe_groups)
    updates = {
        state_name: _prepare_state_updates(states[state_name], columns, crbn_dim)
        for state_name in cfg["perturbed_states"]
    }

    factor_effects: list[dict[str, Any]] = []
    role_factor_effects: list[dict[str, Any]] = []
    group_effects: list[dict[str, Any]] = []
    present_for_ridge: list[dict[str, Any]] = []
    metrics_cache: dict[tuple[str, str], dict[float, dict[str, float]]] = {}
    derivative_cache: dict[tuple[str, str], dict[str, float]] = {}

    for group in universe_groups:
        idx = int(np.flatnonzero(residues == group.residue)[0])
        group_row: dict[str, Any] = {
            "group_id": group.group_id,
            "residue": group.residue,
            "contact_class": group.contact_class,
            "domain": _domain(group.residue),
            "status": "present",
            "contact_count": len(group.edge_ids),
            "joint_degree": int(degrees[idx]),
            "axis_distance_A": float(axis_distances[idx]),
            "edge_ids": ";".join(str(edge_id) for edge_id in group.edge_ids),
            "edge_ids_tuple": group.edge_ids,
            **annotations[group.group_id],
        }
        for state_name in cfg["perturbed_states"]:
            state_metrics: dict[float, dict[str, float]] = {}
            for spring_factor in cfg["spring_factors"]:
                metrics = _perturbation_metrics(states[state_name], updates[state_name], group.edge_ids, spring_factor)
                state_metrics[float(spring_factor)] = metrics
                factor_effects.append(
                    {
                        "group_id": group.group_id,
                        "residue": group.residue,
                        "contact_class": group.contact_class,
                        "state": state_name,
                        "spring_factor": float(spring_factor),
                        **metrics,
                    }
                )
            derivative = _exact_derivative(states[state_name], updates[state_name], group.edge_ids)
            metrics_cache[(group.group_id, state_name)] = state_metrics
            derivative_cache[(group.group_id, state_name)] = derivative
            group_row[f"{state_name}_D_g"] = _central_dg(state_metrics, cfg)
            group_row[f"{state_name}_D_g_per_edge"] = group_row[f"{state_name}_D_g"] / len(group.edge_ids)
            group_row.update({f"{state_name}_{key}": value for key, value in derivative.items()})
            group_row[f"{state_name}_derivative_log_S_close_per_edge"] = (
                derivative["derivative_log_S_close"] / len(group.edge_ids)
            )
        for spring_factor in cfg["spring_factors"]:
            fixed = metrics_cache[(group.group_id, "fixed")][float(spring_factor)]
            rigid = metrics_cache[(group.group_id, "rigid")][float(spring_factor)]
            flexible = metrics_cache[(group.group_id, "flexible")][float(spring_factor)]
            role_factor_effects.append(
                {
                    "group_id": group.group_id,
                    "residue": group.residue,
                    "contact_class": group.contact_class,
                    "spring_factor": float(spring_factor),
                    "delta_R_body_delta_log_S_close": (
                        rigid["delta_log_S_close"] - fixed["delta_log_S_close"]
                    ),
                    "delta_R_internal_delta_log_S_close": (
                        flexible["delta_log_S_close"] - rigid["delta_log_S_close"]
                    ),
                    "delta_R_body_delta_log_C_close": (
                        rigid["delta_log_C_close"] - fixed["delta_log_C_close"]
                    ),
                    "delta_R_internal_delta_log_C_close": (
                        flexible["delta_log_C_close"] - rigid["delta_log_C_close"]
                    ),
                    "delta_R_body_delta_log_mean_compliance": (
                        rigid["delta_log_mean_compliance"] - fixed["delta_log_mean_compliance"]
                    ),
                    "delta_R_internal_delta_log_mean_compliance": (
                        flexible["delta_log_mean_compliance"] - rigid["delta_log_mean_compliance"]
                    ),
                }
            )
        group_row["delta_R_body_D_g"] = group_row["rigid_D_g"] - group_row["fixed_D_g"]
        group_row["delta_R_internal_D_g"] = group_row["flexible_D_g"] - group_row["rigid_D_g"]
        group_row["delta_R_body_derivative_log_S_close"] = (
            group_row["rigid_derivative_log_S_close"] - group_row["fixed_derivative_log_S_close"]
        )
        group_row["delta_R_internal_derivative_log_S_close"] = (
            group_row["flexible_derivative_log_S_close"] - group_row["rigid_derivative_log_S_close"]
        )
        group_row["delta_R_body_derivative_log_S_close_per_edge"] = (
            group_row["delta_R_body_derivative_log_S_close"] / len(group.edge_ids)
        )
        group_row["delta_R_internal_derivative_log_S_close_per_edge"] = (
            group_row["delta_R_internal_derivative_log_S_close"] / len(group.edge_ids)
        )
        group_row["flexible_derivative_log_S_close_per_edge"] = (
            group_row["flexible_derivative_log_S_close"] / len(group.edge_ids)
        )
        group_effects.append(group_row)
        present_for_ridge.append(group_row)

    for group_id in missing_ids:
        residue, contact_class = group_id.split(":", 1)
        group_effects.append(
            {
                "group_id": group_id,
                "residue": int(residue),
                "contact_class": contact_class,
                "domain": _domain(int(residue)),
                "status": "missing",
                "missing_reason": "discovery group absent under this condition",
            }
        )

    _rank_groups(group_effects, "flexible")
    ridge_rows = _ridge_fit(
        present_for_ridge,
        coords[: len(residues)],
        residues,
        alpha=float(cfg["ridge_alpha"]),
        min_training=int(cfg["ridge_min_training"]),
        min_same_domain=int(cfg["ridge_min_same_domain"]),
        neighbor_cutoff=float(cfg["ridge_neighbor_cutoff_A"]),
    )

    for row in group_effects:
        row.pop("edge_ids_tuple", None)

    state_baselines = [
        {
            "state": name,
            "C_close": float(state["C_close"]),
            "mean_compliance": float(state["mean_compliance"]),
            "S_close": float(state["S_close"]),
            "dof": _state_dof(state),
        }
        for name, state in states.items()
        if name in cfg["state_order"] or name in cfg["perturbed_states"]
    ]
    baseline_by_name = {row["state"]: row for row in state_baselines}
    summary = {
        "n_discovery_groups": len(universe),
        "n_present_groups": sum(row["status"] == "present" for row in group_effects),
        "n_missing_groups": len(missing_ids),
        "n_candidate_edges": len(candidate_edges),
        "spring_factors": list(cfg["spring_factors"]),
        "central_difference": {
            "low_factor": float(cfg["central_low_factor"]),
            "high_factor": float(cfg["central_high_factor"]),
        },
        "baseline_R_body": (
            float(np.log(baseline_by_name["rigid"]["S_close"] / baseline_by_name["fixed"]["S_close"]))
            if {"fixed", "rigid"}.issubset(baseline_by_name)
            else None
        ),
        "baseline_R_internal": (
            float(np.log(baseline_by_name["flexible"]["S_close"] / baseline_by_name["rigid"]["S_close"]))
            if {"rigid", "flexible"}.issubset(baseline_by_name)
            else None
        ),
        "baseline_M": (
            float(np.log(baseline_by_name["flexible"]["S_close"] / baseline_by_name["isolated"]["S_close"]))
            if {"isolated", "flexible"}.issubset(baseline_by_name)
            else None
        ),
        "ranking_metric": "abs(flexible_D_g), residue ascending tie break, within contact_class",
        "missing_groups_are_not_zero_effects": True,
    }

    edge_rows = []
    for edge_id, ((i, j), weight) in enumerate(zip(candidate_edges, edge_weights)):
        members = [
            group.group_id
            for group in universe_groups
            if edge_id in group.edge_ids
        ]
        row: dict[str, Any] = {
            "edge_id": edge_id,
            "i_node": int(i),
            "j_node": int(j),
            "weight": float(weight),
            "group_ids": ";".join(sorted(members)),
            "is_shared": len(members) > 1,
        }
        for state_name in cfg["perturbed_states"]:
            derivative = _exact_derivative(states[state_name], updates[state_name], [edge_id])
            row.update({f"{state_name}_{key}": value for key, value in derivative.items()})
        row["delta_R_body_derivative_log_S_close"] = (
            row["rigid_derivative_log_S_close"] - row["fixed_derivative_log_S_close"]
        )
        row["delta_R_internal_derivative_log_S_close"] = (
            row["flexible_derivative_log_S_close"] - row["rigid_derivative_log_S_close"]
        )
        edge_rows.append(row)

    return {
        "schema_version": "directional_contacts.v1",
        "summary": summary,
        "state_baselines": state_baselines,
        "groups": group_effects,
        "factor_effects": factor_effects,
        "role_factor_effects": role_factor_effects,
        "edges": edge_rows,
        "ridge": ridge_rows,
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "directional_contacts_summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in ("state_baselines", "groups", "factor_effects", "role_factor_effects", "edges", "ridge"):
        rows = result[name]
        if not rows:
            continue
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output_dir / f"directional_contacts_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Use the common coordinate-based runner; never deserialize Python objects."""
    import sys
    try:
        from run_directional_mechanics import main as run
    except ModuleNotFoundError:
        from scripts.run_directional_mechanics import main as run
    arguments = list(sys.argv[1:] if argv is None else argv)
    return run([*arguments, "--stages", "contacts"])


if __name__ == "__main__":
    raise SystemExit(main())
