from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_directional_contacts as subject


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _case(output: Path, name: str, rows: list[dict[str, object]], ridge: list[dict[str, object]] | None = None) -> None:
    case_dir = output / name
    _write_csv(case_dir / "groups.csv", rows)
    if ridge is not None:
        _write_csv(case_dir / "ridge.csv", ridge)


def _row(pdb: str, cutoff: float, weighting: str, residue: int, cls: str, dg: float, status: str = "present", rank: int = 1) -> dict[str, object]:
    return {
        "pdb": pdb,
        "cutoff_A": cutoff,
        "weighting": weighting,
        "reference_type": "apo" if pdb.startswith("8") else "engineered",
        "group_id": f"{residue}:{cls}",
        "residue": residue,
        "contact_class": cls,
        "domain": "TBD" if residue >= 318 else "HB",
        "status": status,
        "contact_count": 2,
        "joint_degree": 10,
        "axis_distance_A": 4.0,
        "edge_ids": f"{rank};{rank + 10}",
        "shared_edge_group_ids": "222:CRBN_DDB1" if residue == 221 else "221:CRBN_DDB1" if residue == 222 else "",
        "identical_edge_group_ids": "",
        "fixed_D_g": dg / 4,
        "fixed_D_g_per_edge": dg / 8,
        "fixed_derivative_log_C_close": -0.1,
        "fixed_derivative_log_mean_compliance": -0.2,
        "fixed_derivative_log_S_close": 0.1,
        "fixed_derivative_log_S_close_per_edge": 0.05,
        "rigid_D_g": dg / 2,
        "rigid_D_g_per_edge": dg / 4,
        "rigid_derivative_log_C_close": -0.2,
        "rigid_derivative_log_mean_compliance": -0.3,
        "rigid_derivative_log_S_close": 0.1,
        "rigid_derivative_log_S_close_per_edge": 0.05,
        "flexible_D_g": dg,
        "flexible_D_g_per_edge": dg / 2,
        "flexible_derivative_log_C_close": -0.3,
        "flexible_derivative_log_mean_compliance": -0.4,
        "flexible_derivative_log_S_close": 0.1,
        "flexible_derivative_log_S_close_per_edge": 0.05,
        "delta_R_body_D_g": dg / 4,
        "delta_R_internal_D_g": dg / 2,
        "delta_R_body_derivative_log_S_close": 0.01,
        "delta_R_internal_derivative_log_S_close": 0.02,
        "delta_R_body_derivative_log_S_close_per_edge": 0.005,
        "delta_R_internal_derivative_log_S_close_per_edge": 0.01,
        "flexible_rank": rank,
        "flexible_class_n": 3,
        "flexible_rank_fraction": rank / 3,
    }


def _with_background(rows: list[dict[str, object]], pdb: str, cutoff: float, weighting: str) -> list[dict[str, object]]:
    extra = [
        _row(pdb, cutoff, weighting, residue, "CRBN_DDB1", 0.01 / (idx + 1), rank=idx + 3)
        for idx, residue in enumerate(range(230, 238))
    ]
    return rows + extra


@pytest.fixture
def synthetic_output(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy_robustness.csv"
    _write_csv(
        legacy,
        [
            {
                "residue": 221,
                "contact_class": "CRBN_DDB1",
                "discovery_D_g": 0.8,
                "discovery_rank": 1,
                "discovery_top5": "True",
                "all_required_conditions_observed": "True",
                "stable_apo_model_candidate": "True",
                "also_consistent_in_engineered_references": "False",
                "condition_results": "legacy",
            },
            {
                "residue": 222,
                "contact_class": "CRBN_DDB1",
                "discovery_D_g": 0.7,
                "discovery_rank": 2,
                "discovery_top5": "True",
                "all_required_conditions_observed": "True",
                "stable_apo_model_candidate": "False",
                "also_consistent_in_engineered_references": "False",
                "condition_results": "legacy",
            },
            {
                "residue": 289,
                "contact_class": "HB_TBD",
                "discovery_D_g": -0.6,
                "discovery_rank": 1,
                "discovery_top5": "True",
                "all_required_conditions_observed": "False",
                "stable_apo_model_candidate": "False",
                "also_consistent_in_engineered_references": "False",
                "condition_results": "legacy",
            },
        ],
    )
    monkeypatch.setattr(subject, "DEFAULT_LEGACY", legacy)
    config = {
        "primary_cutoff_A": 15.0,
        "cutoffs_A": [13.0, 15.0, 18.0],
        "apo_references": ["8CVP", "8D7X", "8D7Y"],
        "engineered_references": ["6H0F", "7U8F"],
        "contact": {"discovery_reference": "8CVP", "discovery_cutoff_A": 15.0},
    }
    output = tmp_path / "contact_roles"
    common = _with_background([
        _row("8CVP", 15.0, "uniform", 221, "CRBN_DDB1", 0.8, rank=1),
        _row("8CVP", 15.0, "uniform", 222, "CRBN_DDB1", 0.7, rank=2),
        _row("8CVP", 15.0, "uniform", 289, "HB_TBD", -0.6, rank=1),
    ], "8CVP", 15.0, "uniform")
    _case(
        output,
        "8CVP_15A_uniform",
        common,
        [
            {
                "pdb": "8CVP",
                "cutoff_A": 15.0,
                "weighting": "uniform",
                "group_id": "221:CRBN_DDB1",
                "residue": 221,
                "contact_class": "CRBN_DDB1",
                "training_n": 30,
                "same_domain_training_n": 12,
                "status": "fit",
                "observed_per_edge_derivative": 0.4,
                "predicted_per_edge_derivative": 0.3,
                "residual_per_edge_derivative": 0.1,
                "training_mse": 0.02,
                "ridge_alpha": 1.0,
                "ridge_objective": "mean_squared_error_plus_alpha_l2",
            }
        ],
    )
    for pdb, cutoff in [
        ("8CVP", 13.0),
        ("8CVP", 18.0),
        ("8D7X", 15.0),
        ("8D7Y", 15.0),
        ("6H0F", 15.0),
        ("7U8F", 15.0),
    ]:
        rows = _with_background([
            _row(pdb, cutoff, "uniform", 221, "CRBN_DDB1", 0.5, rank=1),
            _row(pdb, cutoff, "uniform", 222, "CRBN_DDB1", -0.5, rank=2),
            _row(pdb, cutoff, "uniform", 289, "HB_TBD", -0.4, status="missing", rank=1),
        ], pdb, cutoff, "uniform")
        _case(output, f"{pdb}_{cutoff:g}A_uniform", rows)
    _case(
        output,
        "8CVP_15A_inverse_square",
        _with_background(
            [_row("8CVP", 15.0, "inverse_square", 221, "CRBN_DDB1", 0.4, rank=1)],
            "8CVP",
            15.0,
            "inverse_square",
        ),
    )
    _case(
        output,
        "OLD999_15A_uniform",
        [_row("OLD999", 15.0, "uniform", 221, "CRBN_DDB1", 99.0, rank=1)],
    )
    return output, config


def test_consolidate_preserves_universe_legacy_flags_and_absence_failures(synthetic_output):
    output, config = synthetic_output
    result = subject.consolidate(output, config)
    rows = result["candidate_summary"]
    assert len(rows) == 3
    by_id = {row["group_id"]: row for row in rows}
    assert by_id["221:CRBN_DDB1"]["legacy_stable_apo_model_candidate"] is True
    assert by_id["221:CRBN_DDB1"]["new_primary_apo_stable"] is True
    assert by_id["222:CRBN_DDB1"]["new_primary_apo_stable"] is False
    assert "fail" in by_id["222:CRBN_DDB1"]["new_primary_apo_condition_results"]
    assert by_id["289:HB_TBD"]["new_primary_apo_stable"] is False
    assert "absent" in by_id["289:HB_TBD"]["new_primary_apo_condition_results"]
    assert by_id["221:CRBN_DDB1"]["new_engineered_consistent"] is True
    assert by_id["221:CRBN_DDB1"]["new_engineered_all_cutoffs_consistent"] is False
    assert "missing_condition" in by_id["221:CRBN_DDB1"]["new_engineered_all_cutoffs_condition_results"]
    assert by_id["221:CRBN_DDB1"]["inverse_square_condition_n"] == 1
    assert by_id["221:CRBN_DDB1"]["inverse_square_consistent_when_available"] is True
    assert by_id["221:CRBN_DDB1"]["inverse_square_all_cutoffs_consistent"] is False
    assert "missing_condition" in by_id["221:CRBN_DDB1"]["inverse_square_all_cutoffs_condition_results"]
    assert by_id["221:CRBN_DDB1"]["discovery_flexible_D_g"] == "0.8"
    assert by_id["221:CRBN_DDB1"]["discovery_delta_R_internal_D_g"] == "0.4"
    assert by_id["221:CRBN_DDB1"]["discovery_ridge_status"] == "fit"
    assert by_id["221:CRBN_DDB1"]["no_p_or_fdr"] is True
    assert (output / "candidate_summary.csv").exists()
    assert (output / "summary.json").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["conditions_found_n"] == 8
    assert all("OLD999" not in key for key in summary["condition_counts"])
    assert summary["new_primary_apo_stable_and_engineered_consistent_n"] == 1
    assert summary["inverse_square_all_cutoffs_consistent_n"] == 0


def test_recalculates_ranks_from_flexible_dg_not_stale_rank_field(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy_robustness.csv"
    _write_csv(
        legacy,
        [
            {"residue": 221, "contact_class": "CRBN_DDB1", "discovery_D_g": 1, "discovery_rank": 1, "discovery_top5": "True", "all_required_conditions_observed": "True", "stable_apo_model_candidate": "True", "also_consistent_in_engineered_references": "False", "condition_results": ""},
            {"residue": 222, "contact_class": "CRBN_DDB1", "discovery_D_g": 1, "discovery_rank": 2, "discovery_top5": "True", "all_required_conditions_observed": "True", "stable_apo_model_candidate": "True", "also_consistent_in_engineered_references": "False", "condition_results": ""},
        ],
    )
    monkeypatch.setattr(subject, "DEFAULT_LEGACY", legacy)
    output = tmp_path / "contact_roles"
    rows = [
        _row("8CVP", 15.0, "uniform", 221, "CRBN_DDB1", 0.1, rank=1),
        _row("8CVP", 15.0, "uniform", 222, "CRBN_DDB1", 0.9, rank=99),
    ]
    _case(output, "8CVP_15A_uniform", rows)
    config = {
        "primary_cutoff_A": 15.0,
        "cutoffs_A": [15.0],
        "apo_references": ["8CVP"],
        "engineered_references": [],
        "contact": {"discovery_reference": "8CVP", "discovery_cutoff_A": 15.0},
    }
    result = subject.consolidate(output, config)
    by_id = {row["group_id"]: row for row in result["candidate_summary"]}
    assert by_id["222:CRBN_DDB1"]["new_discovery_flexible_rank"] == 1
    assert by_id["221:CRBN_DDB1"]["new_discovery_flexible_rank"] == 2


def test_shared_edge_components_are_reported_without_independence_counting(synthetic_output):
    output, config = synthetic_output
    rows = subject.consolidate(output, config)["candidate_summary"]
    by_id = {row["group_id"]: row for row in rows}
    assert by_id["221:CRBN_DDB1"]["shared_edge_component_id"] == by_id["222:CRBN_DDB1"]["shared_edge_component_id"]
    assert by_id["221:CRBN_DDB1"]["shared_edge_component_size"] == 2
    assert sum(row["shared_component_representative"] for row in rows if row["contact_class"] == "CRBN_DDB1") == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["non_singleton_shared_edge_component_n"] == 1


def test_require_verified_fails_closed_without_runner_completion_records(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy_robustness.csv"
    _write_csv(
        legacy,
        [
            {"residue": 221, "contact_class": "CRBN_DDB1", "discovery_D_g": 1, "discovery_rank": 1, "discovery_top5": "True", "all_required_conditions_observed": "True", "stable_apo_model_candidate": "True", "also_consistent_in_engineered_references": "False", "condition_results": ""},
        ],
    )
    monkeypatch.setattr(subject, "DEFAULT_LEGACY", legacy)
    root = tmp_path / "package"
    output = root / "analysis" / "contact_roles"
    _case(output, "8CVP_15A_uniform", [_row("8CVP", 15.0, "uniform", 221, "CRBN_DDB1", 1.0)])
    config = {
        "references": ["8CVP"],
        "primary_cutoff_A": 15.0,
        "cutoffs_A": [15.0],
        "weightings": ["uniform"],
        "apo_references": ["8CVP"],
        "engineered_references": [],
        "contact": {"discovery_reference": "8CVP", "discovery_cutoff_A": 15.0},
    }
    with pytest.raises(ValueError, match="Missing or stale verified contact conditions"):
        subject.consolidate(output, config, require_verified=True)
