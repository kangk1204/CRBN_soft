#!/usr/bin/env python3
"""Consolidate directional contact-role conditions.

The summarizer keeps the frozen discovery candidate universe, recalculates
ranks from the new flexible-model Dg values in each condition, and writes a
single candidate table for figures and manuscript text.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "directional_mechanics" / "analysis" / "contact_roles"
DEFAULT_CONFIG = ROOT / "scripts" / "directional_config.json"
DEFAULT_LEGACY = ROOT / "data" / "directional_reference_inputs" / "legacy_robustness.csv"
TOP_FRACTION = 0.20


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV without a schema: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _condition_key(pdb: str, cutoff: Any, weighting: str) -> str:
    return f"{pdb}:{float(cutoff):g}:{weighting}"


def _group_id(row: dict[str, Any]) -> str:
    if row.get("group_id"):
        return str(row["group_id"])
    return f"{int(row['residue'])}:{row['contact_class']}"


def _load_config(config: dict[str, Any] | Path | None) -> dict[str, Any]:
    if config is None:
        return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if isinstance(config, Path):
        return json.loads(config.read_text(encoding="utf-8"))
    return dict(config)


def _load_legacy(path: Path | None = None) -> list[dict[str, Any]]:
    rows = _read_csv(path or DEFAULT_LEGACY)
    for row in rows:
        row["group_id"] = _group_id(row)
        row["residue"] = int(row["residue"])
        row["legacy_discovery_D_g"] = float(row.pop("discovery_D_g"))
        row["legacy_discovery_rank"] = int(row.pop("discovery_rank"))
        row["legacy_discovery_top5"] = _as_bool(row.pop("discovery_top5"))
        row["legacy_all_required_conditions_observed"] = _as_bool(
            row.pop("all_required_conditions_observed")
        )
        row["legacy_stable_apo_model_candidate"] = _as_bool(row.pop("stable_apo_model_candidate"))
        row["legacy_also_consistent_in_engineered_references"] = _as_bool(
            row.pop("also_consistent_in_engineered_references")
        )
        row["legacy_condition_results"] = row.pop("condition_results")
    if len({_group_id(row) for row in rows}) != len(rows):
        raise ValueError("legacy robustness table contains duplicate residue/contact-class keys")
    return rows


def _condition_dir_name(pdb: str, cutoff: Any, weighting: str) -> str:
    return f"{pdb}_{float(cutoff):g}A_{weighting}"


def _expected_case_dirs(output: Path, config: dict[str, Any]) -> list[Path]:
    references = config.get("references", list(config["apo_references"]) + list(config["engineered_references"]))
    weightings = config.get("weightings", ["uniform", "inverse_square"])
    return [
        output / _condition_dir_name(pdb, cutoff, weighting)
        for weighting in weightings
        for cutoff in config["cutoffs_A"]
        for pdb in references
        if (output / _condition_dir_name(pdb, cutoff, weighting) / "groups.csv").is_file()
    ]


def _verified_case_dirs(output: Path, config: dict[str, Any], config_path: Path | None) -> tuple[list[Path], list[str]]:
    try:
        from run_directional_mechanics import verified_conditions
    except ModuleNotFoundError:
        from scripts.run_directional_mechanics import verified_conditions

    if output.name == "contact_roles" and output.parent.name == "analysis":
        root_output = output.parents[1]
    else:
        root_output = output
    accepted, rejected = verified_conditions(root_output, config, config_path or DEFAULT_CONFIG, "contacts")
    return accepted, rejected


def _case_dirs(
    output: Path,
    config: dict[str, Any],
    *,
    accepted_case_paths: Iterable[str | Path] | None = None,
    require_verified: bool = False,
    config_path: Path | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    if accepted_case_paths is not None:
        case_dirs = [Path(path) for path in accepted_case_paths]
        missing = [str(path) for path in case_dirs if not (path / "groups.csv").is_file()]
        if missing:
            raise ValueError(f"Accepted contact case path lacks groups.csv: {missing[0]}")
        return sorted(case_dirs), {
            "case_selection": "explicit_accepted_case_paths",
            "accepted_case_paths": [str(path) for path in sorted(case_dirs)],
            "rejected_conditions": [],
        }

    if require_verified:
        accepted, rejected = _verified_case_dirs(output, config, config_path)
        if rejected:
            preview = ", ".join(rejected[:5])
            raise ValueError(f"Missing or stale verified contact conditions ({len(rejected)}): {preview}")
        return sorted(accepted), {
            "case_selection": "runner_verified_conditions",
            "accepted_case_paths": [str(path) for path in sorted(accepted)],
            "rejected_conditions": rejected,
        }

    case_dirs = _expected_case_dirs(output, config)
    return sorted(case_dirs), {
        "case_selection": "config_expected_directories",
        "accepted_case_paths": [str(path) for path in sorted(case_dirs)],
        "rejected_conditions": [],
    }


def _load_cases(case_dirs: Iterable[Path]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for case_dir in case_dirs:
        groups = _read_csv(case_dir / "groups.csv")
        if not groups:
            continue
        first = groups[0]
        pdb = first["pdb"]
        cutoff = float(first["cutoff_A"])
        weighting = first["weighting"]
        key = _condition_key(pdb, cutoff, weighting)
        if key in by_condition:
            raise ValueError(f"Duplicate condition key: {key}")
        by_group: dict[str, dict[str, Any]] = {}
        for raw in groups:
            group = dict(raw)
            group["group_id"] = _group_id(group)
            group["residue"] = int(group["residue"])
            group["cutoff_A"] = float(group["cutoff_A"])
            by_group[group["group_id"]] = group
        _recalculate_flexible_ranks(by_group.values())
        by_condition[key] = by_group
        metadata[key] = {
            "pdb": pdb,
            "cutoff_A": cutoff,
            "weighting": weighting,
            "reference_type": first.get("reference_type", ""),
            "case_dir": case_dir.name,
            "group_rows": len(groups),
        }
    return by_condition, metadata


def _recalculate_flexible_ranks(rows: Iterable[dict[str, Any]]) -> None:
    present = [row for row in rows if row.get("status") == "present"]
    for contact_class in sorted({row["contact_class"] for row in present}):
        class_rows = [row for row in present if row["contact_class"] == contact_class]
        class_rows.sort(key=lambda row: (-abs(float(row["flexible_D_g"])), int(row["residue"])))
        for rank, row in enumerate(class_rows, 1):
            row["recalc_flexible_rank"] = rank
            row["recalc_flexible_class_n"] = len(class_rows)
            row["recalc_flexible_rank_fraction"] = rank / len(class_rows)


def _load_ridge(case_dirs: Iterable[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for case_dir in case_dirs:
        path = case_dir / "ridge.csv"
        if not path.is_file():
            continue
        for row in _read_csv(path):
            records[(_condition_key(row["pdb"], row["cutoff_A"], row["weighting"]), _group_id(row))] = row
    return records


def _primary_conditions(config: dict[str, Any]) -> list[str]:
    primary = float(config["primary_cutoff_A"])
    cutoffs = [float(value) for value in config["cutoffs_A"]]
    apo = list(config["apo_references"])
    discovery = config["contact"]["discovery_reference"]
    keys = [_condition_key(discovery, cutoff, "uniform") for cutoff in cutoffs]
    keys += [_condition_key(pdb, primary, "uniform") for pdb in apo if pdb != discovery]
    return list(dict.fromkeys(keys))


def _engineered_conditions(config: dict[str, Any]) -> list[str]:
    primary = float(config["primary_cutoff_A"])
    return [_condition_key(pdb, primary, "uniform") for pdb in config["engineered_references"]]


def _all_cutoff_conditions(config: dict[str, Any], references: Iterable[str], weighting: str) -> list[str]:
    return [
        _condition_key(pdb, cutoff, weighting)
        for pdb in references
        for cutoff in config["cutoffs_A"]
    ]


def _available_inverse_square_conditions(config: dict[str, Any], conditions: Iterable[str]) -> list[str]:
    wanted = []
    primary = float(config["primary_cutoff_A"])
    for pdb in config["apo_references"]:
        wanted.append(_condition_key(pdb, primary, "inverse_square"))
    for cutoff in config["cutoffs_A"]:
        wanted.append(_condition_key(config["contact"]["discovery_reference"], cutoff, "inverse_square"))
    for pdb in config["engineered_references"]:
        wanted.append(_condition_key(pdb, primary, "inverse_square"))
    available = set(conditions)
    return [key for key in dict.fromkeys(wanted) if key in available]


def _condition_status(
    row: dict[str, Any] | None,
    discovery_sign: int,
) -> tuple[str, float | None, int | None, float | None]:
    if row is None or row.get("status") != "present":
        return "absent", None, None, None
    dg = float(row["flexible_D_g"])
    rank = int(row["recalc_flexible_rank"])
    rank_fraction = float(row["recalc_flexible_rank_fraction"])
    sign = 1 if dg > 0 else -1 if dg < 0 else 0
    if discovery_sign == 0 or sign != discovery_sign or rank_fraction > TOP_FRACTION:
        return "fail", dg, rank, rank_fraction
    return "pass", dg, rank, rank_fraction


def _join_statuses(keys: Iterable[str], statuses: dict[str, str]) -> str:
    return ";".join(f"{key}={statuses.get(key, 'missing_condition')}" for key in keys)


def _connected_components(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    adjacency: dict[str, set[str]] = {row["group_id"]: set() for row in rows}
    for row in rows:
        peers = set()
        for field in ("shared_edge_group_ids", "identical_edge_group_ids"):
            peers.update(value for value in str(row.get(field, "")).split(";") if value)
        peers.intersection_update(adjacency)
        adjacency[row["group_id"]].update(peers)
        for peer in peers:
            adjacency[peer].add(row["group_id"])
    result: dict[str, tuple[int, int]] = {}
    seen: set[str] = set()
    component_id = 0
    for group_id in sorted(adjacency):
        if group_id in seen:
            continue
        component_id += 1
        queue = deque([group_id])
        component = []
        seen.add(group_id)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        for member in component:
            result[member] = (component_id, len(component))
    return result


def _copy_metrics(prefix: str, row: dict[str, Any] | None, out: dict[str, Any]) -> None:
    fields = [
        "status",
        "contact_count",
        "joint_degree",
        "axis_distance_A",
        "edge_ids",
        "shared_edge_group_ids",
        "identical_edge_group_ids",
        "fixed_D_g",
        "rigid_D_g",
        "flexible_D_g",
        "fixed_D_g_per_edge",
        "rigid_D_g_per_edge",
        "flexible_D_g_per_edge",
        "fixed_derivative_log_C_close",
        "fixed_derivative_log_mean_compliance",
        "fixed_derivative_log_S_close",
        "fixed_derivative_log_S_close_per_edge",
        "rigid_derivative_log_C_close",
        "rigid_derivative_log_mean_compliance",
        "rigid_derivative_log_S_close",
        "rigid_derivative_log_S_close_per_edge",
        "flexible_derivative_log_C_close",
        "flexible_derivative_log_mean_compliance",
        "flexible_derivative_log_S_close",
        "flexible_derivative_log_S_close_per_edge",
        "delta_R_body_D_g",
        "delta_R_internal_D_g",
        "delta_R_body_derivative_log_S_close",
        "delta_R_internal_derivative_log_S_close",
        "delta_R_body_derivative_log_S_close_per_edge",
        "delta_R_internal_derivative_log_S_close_per_edge",
        "recalc_flexible_rank",
        "recalc_flexible_class_n",
        "recalc_flexible_rank_fraction",
    ]
    for field in fields:
        out[f"{prefix}_{field}"] = "" if row is None else row.get(field, "")


def _copy_ridge(prefix: str, row: dict[str, Any] | None, out: dict[str, Any]) -> None:
    for field in (
        "status",
        "training_n",
        "same_domain_training_n",
        "observed_per_edge_derivative",
        "predicted_per_edge_derivative",
        "residual_per_edge_derivative",
        "training_mse",
        "ridge_alpha",
        "ridge_objective",
    ):
        out[f"{prefix}_ridge_{field}"] = "" if row is None else row.get(field, "")


def consolidate(
    output: str | Path = DEFAULT_OUTPUT,
    config: dict[str, Any] | Path | None = None,
    *,
    accepted_case_paths: Iterable[str | Path] | None = None,
    require_verified: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write candidate-level directional contact summaries under ``output``."""

    output = Path(output)
    cfg = _load_config(config)
    legacy_rows = _load_legacy(Path(cfg.get("legacy_robustness_csv", DEFAULT_LEGACY)))
    universe = {row["group_id"]: row for row in legacy_rows}
    case_dirs, selection = _case_dirs(
        output,
        cfg,
        accepted_case_paths=accepted_case_paths,
        require_verified=require_verified,
        config_path=Path(config_path) if config_path is not None else None,
    )
    cases, metadata = _load_cases(case_dirs)
    ridge = _load_ridge(case_dirs)
    references = cfg.get("references", list(cfg["apo_references"]) + list(cfg["engineered_references"]))
    primary_keys = _primary_conditions(cfg)
    engineered_keys = _engineered_conditions(cfg)
    engineered_all_cutoff_keys = _all_cutoff_conditions(cfg, cfg["engineered_references"], "uniform")
    inverse_all_cutoff_keys = _all_cutoff_conditions(cfg, references, "inverse_square")
    inverse_keys = _available_inverse_square_conditions(cfg, cases)
    discovery_key = _condition_key(
        cfg["contact"]["discovery_reference"],
        cfg["contact"]["discovery_cutoff_A"],
        "uniform",
    )
    if discovery_key not in cases:
        raise ValueError(f"Discovery condition missing: {discovery_key}")
    discovery_rows = cases[discovery_key]
    components = _connected_components(
        [row for gid, row in discovery_rows.items() if gid in universe and row.get("status") == "present"]
    )

    summary_rows: list[dict[str, Any]] = []
    condition_counts = {key: len(cases.get(key, {})) for key in sorted(cases)}
    for group_id, legacy in sorted(universe.items(), key=lambda item: (item[1]["contact_class"], item[1]["residue"])):
        discovery = discovery_rows.get(group_id)
        discovery_dg = _as_float(discovery.get("flexible_D_g") if discovery else None)
        discovery_sign = 1 if discovery_dg and discovery_dg > 0 else -1 if discovery_dg and discovery_dg < 0 else 0
        primary_statuses: dict[str, str] = {}
        engineered_statuses: dict[str, str] = {}
        inverse_statuses: dict[str, str] = {}
        primary_complete = True
        primary_pass = True
        for key in primary_keys:
            status, _, _, _ = _condition_status(cases.get(key, {}).get(group_id), discovery_sign)
            if key not in cases:
                status = "missing_condition"
            primary_statuses[key] = status
            primary_complete = primary_complete and status != "absent" and status != "missing_condition"
            primary_pass = primary_pass and status == "pass"
        engineered_pass = True
        engineered_complete = True
        for key in engineered_keys:
            status, _, _, _ = _condition_status(cases.get(key, {}).get(group_id), discovery_sign)
            if key not in cases:
                status = "missing_condition"
            engineered_statuses[key] = status
            engineered_complete = engineered_complete and status != "absent" and status != "missing_condition"
            engineered_pass = engineered_pass and status == "pass"
        engineered_all_cutoff_statuses: dict[str, str] = {}
        engineered_all_cutoff_pass = True
        engineered_all_cutoff_complete = True
        for key in engineered_all_cutoff_keys:
            status, _, _, _ = _condition_status(cases.get(key, {}).get(group_id), discovery_sign)
            if key not in cases:
                status = "missing_condition"
            engineered_all_cutoff_statuses[key] = status
            engineered_all_cutoff_complete = (
                engineered_all_cutoff_complete
                and status != "absent"
                and status != "missing_condition"
            )
            engineered_all_cutoff_pass = engineered_all_cutoff_pass and status == "pass"
        inverse_available = bool(inverse_keys)
        inverse_pass = inverse_available
        for key in inverse_keys:
            status, _, _, _ = _condition_status(cases.get(key, {}).get(group_id), discovery_sign)
            inverse_statuses[key] = status
            inverse_pass = inverse_pass and status == "pass"
        inverse_all_cutoff_statuses: dict[str, str] = {}
        inverse_all_cutoff_pass = True
        inverse_all_cutoff_complete = True
        for key in inverse_all_cutoff_keys:
            status, _, _, _ = _condition_status(cases.get(key, {}).get(group_id), discovery_sign)
            if key not in cases:
                status = "missing_condition"
            inverse_all_cutoff_statuses[key] = status
            inverse_all_cutoff_complete = (
                inverse_all_cutoff_complete
                and status != "absent"
                and status != "missing_condition"
            )
            inverse_all_cutoff_pass = inverse_all_cutoff_pass and status == "pass"

        component_id, component_size = components.get(group_id, (None, None))
        row: dict[str, Any] = {
            "group_id": group_id,
            "residue": legacy["residue"],
            "contact_class": legacy["contact_class"],
            "legacy_discovery_D_g": legacy["legacy_discovery_D_g"],
            "legacy_discovery_rank": legacy["legacy_discovery_rank"],
            "legacy_discovery_top5": legacy["legacy_discovery_top5"],
            "legacy_all_required_conditions_observed": legacy["legacy_all_required_conditions_observed"],
            "legacy_stable_apo_model_candidate": legacy["legacy_stable_apo_model_candidate"],
            "legacy_also_consistent_in_engineered_references": legacy[
                "legacy_also_consistent_in_engineered_references"
            ],
            "legacy_condition_results": legacy["legacy_condition_results"],
            "new_discovery_present": discovery is not None and discovery.get("status") == "present",
            "new_discovery_flexible_D_g": "" if discovery_dg is None else discovery_dg,
            "new_discovery_flexible_rank": "" if discovery is None else discovery.get("recalc_flexible_rank", ""),
            "new_discovery_flexible_rank_fraction": "" if discovery is None else discovery.get("recalc_flexible_rank_fraction", ""),
            "new_primary_apo_complete": primary_complete,
            "new_primary_apo_stable": primary_pass,
            "new_primary_apo_condition_results": _join_statuses(primary_keys, primary_statuses),
            "new_engineered_complete": engineered_complete,
            "new_engineered_consistent": engineered_pass,
            "new_engineered_condition_results": _join_statuses(engineered_keys, engineered_statuses),
            "new_engineered_all_cutoffs_complete": engineered_all_cutoff_complete,
            "new_engineered_all_cutoffs_consistent": engineered_all_cutoff_pass,
            "new_engineered_all_cutoffs_condition_results": _join_statuses(
                engineered_all_cutoff_keys,
                engineered_all_cutoff_statuses,
            ),
            "inverse_square_condition_n": len(inverse_keys),
            "inverse_square_consistent_when_available": inverse_pass,
            "inverse_square_condition_results": _join_statuses(inverse_keys, inverse_statuses),
            "inverse_square_all_cutoffs_complete": inverse_all_cutoff_complete,
            "inverse_square_all_cutoffs_consistent": inverse_all_cutoff_pass,
            "inverse_square_all_cutoffs_condition_results": _join_statuses(
                inverse_all_cutoff_keys,
                inverse_all_cutoff_statuses,
            ),
            "shared_edge_component_id": "" if component_id is None else component_id,
            "shared_edge_component_size": "" if component_size is None else component_size,
            "shared_component_representative": component_size in (None, 1) or (
                component_id is not None
                and group_id == min(gid for gid, value in components.items() if value[0] == component_id)
            ),
            "no_p_or_fdr": True,
        }
        _copy_metrics("discovery", discovery, row)
        _copy_ridge("discovery", ridge.get((discovery_key, group_id)), row)
        summary_rows.append(row)

    _write_csv(output / "candidate_summary.csv", summary_rows)
    payload = {
        "schema_version": "summarize_directional_contacts.v1",
        "output_dir": str(output),
        "candidate_universe_n": len(universe),
        "conditions_found_n": len(cases),
        "case_selection": selection["case_selection"],
        "accepted_case_paths": selection["accepted_case_paths"],
        "rejected_conditions": selection["rejected_conditions"],
        "condition_counts": condition_counts,
        "primary_apo_conditions": primary_keys,
        "engineered_conditions": engineered_keys,
        "engineered_all_cutoff_conditions": engineered_all_cutoff_keys,
        "inverse_square_conditions_available": inverse_keys,
        "inverse_square_all_cutoff_conditions": inverse_all_cutoff_keys,
        "new_primary_apo_stable_n": sum(_as_bool(row["new_primary_apo_stable"]) for row in summary_rows),
        "legacy_primary_apo_stable_n": sum(_as_bool(row["legacy_stable_apo_model_candidate"]) for row in summary_rows),
        "new_engineered_consistent_n": sum(_as_bool(row["new_engineered_consistent"]) for row in summary_rows),
        "new_primary_apo_stable_and_engineered_consistent_n": sum(
            _as_bool(row["new_primary_apo_stable"]) and _as_bool(row["new_engineered_consistent"])
            for row in summary_rows
        ),
        "new_engineered_all_cutoffs_consistent_n": sum(
            _as_bool(row["new_engineered_all_cutoffs_consistent"]) for row in summary_rows
        ),
        "legacy_engineered_consistent_n": sum(
            _as_bool(row["legacy_also_consistent_in_engineered_references"]) for row in summary_rows
        ),
        "inverse_square_all_cutoffs_complete_n": sum(
            _as_bool(row["inverse_square_all_cutoffs_complete"]) for row in summary_rows
        ),
        "inverse_square_all_cutoffs_consistent_n": sum(
            _as_bool(row["inverse_square_all_cutoffs_consistent"]) for row in summary_rows
        ),
        "new_primary_apo_stable_and_inverse_square_all_cutoffs_consistent_n": sum(
            _as_bool(row["new_primary_apo_stable"])
            and _as_bool(row["inverse_square_all_cutoffs_consistent"])
            for row in summary_rows
        ),
        "new_primary_apo_stable_and_engineered_and_inverse_square_all_cutoffs_consistent_n": sum(
            _as_bool(row["new_primary_apo_stable"])
            and _as_bool(row["new_engineered_consistent"])
            and _as_bool(row["inverse_square_all_cutoffs_consistent"])
            for row in summary_rows
        ),
        "shared_edge_component_n": len(set(value[0] for value in components.values())),
        "non_singleton_shared_edge_component_n": len(
            {value[0] for value in components.values() if value[1] > 1}
        ),
        "candidate_summary_csv": "candidate_summary.csv",
        "ranking_rule": "abs(flexible_D_g) within contact_class, residue ascending tie break",
        "missing_group_policy": "Absent groups fail robustness and are never converted to zero effects.",
        "inferential_p_or_fdr": False,
        "legacy_flags_preserved": True,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": payload, "candidate_summary": summary_rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = consolidate(args.output_dir, args.config, require_verified=True, config_path=args.config)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
