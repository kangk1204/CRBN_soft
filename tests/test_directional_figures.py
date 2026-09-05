from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_directional_figures as subject


REFS = [
    ("8CVP", "apo"),
    ("8D7X", "apo"),
    ("8D7Y", "apo"),
    ("6H0F", "engineered"),
    ("7U8F", "engineered"),
]


def test_unseparated_functional_assay_is_not_a_measured_abundance_endpoint():
    assert subject.availability_code("not_separated_from_functional_assays") == 0
    assert subject.availability_code("not_measured_as_separate_endpoint") == 0
    assert subject.availability_code("not separately measured") == 0
    assert subject.availability_code("qualitative_from_table_and_figures") == 1


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_minimal_root(root: Path) -> None:
    model_rows = []
    for ref_idx, (pdb, ref_type) in enumerate(REFS):
        for model_idx, model in enumerate(subject.MODEL_ORDER):
            model_rows.append(
                {
                    "pdb": pdb,
                    "cutoff_A": 15.0,
                    "weighting": "uniform",
                    "reference_type": ref_type,
                    "model": model,
                    "C_close": 10.0 + ref_idx + model_idx,
                    "mean_compliance": 0.2 + 0.01 * model_idx,
                    "S_close": 40.0 + 2.0 * ref_idx + model_idx,
                    "best_mode20": 1 + model_idx,
                    "best_overlap20": 0.4 + 0.02 * model_idx,
                }
            )
    _write_csv(root / "analysis" / "mechanics" / "models_all.csv", model_rows)

    comparison_rows = []
    for ref_idx, (pdb, ref_type) in enumerate(REFS):
        for role in ["R_body", "R_internal", "M"]:
            for target in ["finite", "tangent"]:
                comparison_rows.append(
                    {
                        "pdb": pdb,
                        "cutoff_A": 15.0,
                        "weighting": "uniform",
                        "reference_type": ref_type,
                        "role": role,
                        "target": target,
                        "effect": 0.05 * (ref_idx + 1),
                        "rotational_percentile": 40 + 7 * ref_idx,
                        "null_q95": 0.3,
                        "finite_tangent_overlap": 0.9,
                    }
                )
    _write_csv(root / "analysis" / "mechanics" / "comparisons_all.csv", comparison_rows)

    mode_summary = {
        "alpha_summaries": [
            {
                "interface_alpha": 0.0,
                "best_mode": 2,
                "raw_best_mode": 5,
                "best_crbn_directional_overlap": 0.7,
                "best_crbn_internal_overlap": 0.8,
                "internal_best_mode": 2,
                "internal_best_overlap": 0.8,
                "tracked_rank": 2,
                "cluster_projection": 0.85,
                "identity_interpretable": True,
                "low_overlap_flag": False,
            },
            {
                "interface_alpha": 1.0,
                "best_mode": 3,
                "raw_best_mode": 5,
                "best_crbn_directional_overlap": 0.65,
                "best_crbn_internal_overlap": 0.75,
                "internal_best_mode": 3,
                "internal_best_overlap": 0.75,
                "tracked_rank": 3,
                "cluster_projection": 0.82,
                "identity_interpretable": True,
                "low_overlap_flag": False,
            },
        ]
    }
    mode_dir = root / "analysis" / "mode_paths" / "8CVP_15A_uniform"
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "directional_modes_summary.json").write_text(json.dumps(mode_summary), encoding="utf-8")

    contact_rows = []
    legacy_rows = []
    for idx, residue in enumerate([185, 186, 222, 262, 289, 339, 340, 420, 421, 422]):
        contact_class = "CRBN_DDB1" if residue < 250 or residue in {339, 340} else "HB_TBD"
        contact_rows.append(
            {
                "residue": residue,
                "contact_class": contact_class,
                "domain": "NTD" if residue < 250 else "TBD",
                "contact_count": 2 + idx % 3,
                "shared_edge_group_ids": "",
                "flexible_D_g": (-1) ** idx * (0.01 + 0.002 * idx),
                "flexible_derivative_log_C_close": -0.02 + 0.002 * idx,
                "flexible_derivative_log_mean_compliance": -0.01 + 0.001 * idx,
                "delta_R_body_derivative_log_S_close": -0.008 + 0.001 * idx,
                "delta_R_internal_derivative_log_S_close": 0.004 - 0.0005 * idx,
                "flexible_rank": idx + 1,
            }
        )
        stable = residue in {186, 222, 262, 289, 339, 422}
        legacy_rows.append(
            {
                "residue": residue,
                "contact_class": contact_class,
                "discovery_rank": idx + 1,
                "discovery_top5": idx < 5,
                "stable_apo_model_candidate": stable,
                "also_consistent_in_engineered_references": residue in {222, 262, 289},
                "condition_results": "8CVP:13=pass;8CVP:15=pass;8CVP:18=pass;8D7X:15=pass;8D7Y:15=pass;6H0F:15=absent;7U8F:15=absent",
            }
        )
    _write_csv(root / "analysis" / "contact_roles" / "8CVP_15A_uniform" / "groups.csv", contact_rows)
    _write_csv(root / "data" / "directional_reference_inputs" / "legacy_robustness.csv", legacy_rows)

    site_rows = []
    for idx in range(1, 143):
        site_rows.append(
            {
                "residue": 180 + idx,
                "contact_class": "CRBN_DDB1" if idx <= 82 else "HB_TBD",
                "discovery_rank": idx,
                "stable_apo_model_candidate": idx in {1, 7, 9, 10, 20, 30, 40, 50},
                "also_consistent_in_engineered_references": idx in {9, 10, 30, 40, 50},
                "min_distance_to_A1CEG_A": 3.0 + 0.08 * idx,
                "same_residue_as_A1CEG_contact": idx <= 16,
            }
        )
    _write_csv(root / "analysis" / "external" / "candidate_9sfm_spatial_correspondence.csv", site_rows)

    blood_rows = []
    for idx in range(12):
        blood_rows.append(
            {
                "variant_id": f"V{idx + 1}",
                "primary_269_window": "inside_269_window",
                "binding_endpoint": "not_variant_resolved",
                "abundance_or_folding_endpoint": "not separately measured",
                "degradation_endpoint": "no effect" if idx % 3 else "complete loss",
                "cell_response_endpoint": "no effect" if idx % 2 else "agent-dependent",
                "candidate_overlap": "none" if idx % 4 else "same_residue",
                "stable_apo_candidate_overlap": idx == 4,
                "functional_endpoint_type": "qualitative_from_table_and_figures",
                "binding_endpoint_type": "not_variant_resolved_in_retrieved_main_pdf",
                "abundance_endpoint_type": "not_measured_as_separate_endpoint",
                "degradation_endpoint_type": "qualitative_from_table_and_figures",
                "evidence_quote": (
                    "Behaviour similar to EV"
                    if idx in {9, 10}
                    else ("Partial CRBN function observed" if idx in {4, 6, 7, 8} else "Behaviour similar to wild type CRBN")
                ),
            }
        )
    _write_csv(root / "analysis" / "external" / "blood_2025_variant_observations.csv", blood_rows)
    _write_csv(
        root / "analysis" / "external" / "saxs_guinier_refits.csv",
        [
            {
                "accession": f"SASDU{idx}",
                "condition": condition,
                "refit_rg_nm": value,
                "refit_rg_nm_stderr": 0.01,
                "low_q_qa": "pass",
            }
            for idx, (condition, value) in enumerate(
                [
                    ("apo", 2.7),
                    ("lenalidomide", 2.25),
                    ("pomalidomide", 2.32),
                    ("iberdomide", 2.37),
                    ("mezigdomide", 2.28),
                ],
                1,
            )
        ],
    )
    _write_csv(
        root / "analysis" / "external" / "oconnor_compound9_mutant_case_comparison.csv",
        [
            {
                "variant": variant,
                "primary_269_window": "inside",
                "dsf_delta_delta_tm_vs_wt_degC": delta_tm,
                "saxs_delta_rg_vs_wt_angstrom": delta_rg,
                "binding_note": "sparse case",
            }
            for variant, delta_tm, delta_rg in [
                ("WT", 0.0, 0.0),
                ("H378N", -1.4, 0.4),
                ("H378A", -2.0, 0.7),
                ("Q100A", -0.5, -0.1),
            ]
        ],
    )


def test_build_creates_directional_figures_and_relative_manifests(tmp_path: Path):
    _make_minimal_root(tmp_path)
    result = subject.build(tmp_path)

    assert Path(result["contract"]).is_file()
    assert Path(result["legends"]).is_file()
    for stem in subject.FIGURE_STEMS:
        assert (tmp_path / "manuscript" / "figures" / f"{stem}.png").is_file()
        assert (tmp_path / "manuscript" / "figures" / "vector" / f"{stem}.pdf").is_file()
        assert (tmp_path / "manuscript" / "figures" / "vector" / f"{stem}.svg").is_file()
        manifest = json.loads((tmp_path / "analysis" / "directional_figure_sources" / f"{stem}_input_manifest.json").read_text())
        for item in manifest["inputs"] + manifest["source_snapshots"] + manifest["outputs"]:
            assert not Path(item["path"]).is_absolute()
            assert item["sha256"]


def test_build_fails_closed_when_required_mechanics_missing(tmp_path: Path):
    _make_minimal_root(tmp_path)
    (tmp_path / "analysis" / "mechanics" / "models_all.csv").unlink()

    with pytest.raises(subject.MissingSource, match="models_all"):
        subject.build(tmp_path)
