from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "strengthen_controls.py"
OUTPUT = ROOT / "results" / "strengthening" / "analysis" / "controls"


def load_module():
    spec = importlib.util.spec_from_file_location("strengthen_controls", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_anm_hessian_matches_dense_library_implementation():
    module = load_module()
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    sparse_hessian = module.sparse_anm_hessian(coords, 2.0).toarray()
    dense_hessian = module.L.anm_hessian(coords, 2.0)

    np.testing.assert_allclose(sparse_hessian, dense_hessian, atol=1e-12, rtol=0.0)


def test_endpoint_scoring_is_rigid_transform_and_permutation_invariant():
    module = load_module()
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    eigenvalues = np.array([1.0, 2.0, 3.0, 4.0])
    displacement = rng.normal(size=12)
    baseline = module.score_anm_endpoint(
        None,
        displacement,
        cutoff_A=15.0,
        n_modes=4,
        eigensystem=(eigenvalues, q),
    )

    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    block_rotation = np.kron(np.eye(4), rotation)
    rotated = module.score_anm_endpoint(
        None,
        block_rotation @ displacement,
        cutoff_A=15.0,
        n_modes=4,
        eigensystem=(eigenvalues, block_rotation @ q),
    )
    assert rotated["mode1_overlap"] == pytest.approx(baseline["mode1_overlap"])
    assert rotated["best20_overlap"] == pytest.approx(baseline["best20_overlap"])
    assert rotated["top3_projection"] == pytest.approx(baseline["top3_projection"])

    residue_order = np.array([2, 0, 3, 1])
    coordinate_order = np.concatenate([np.arange(3 * i, 3 * i + 3) for i in residue_order])
    permuted = module.score_anm_endpoint(
        None,
        displacement[coordinate_order],
        cutoff_A=15.0,
        n_modes=4,
        eigensystem=(eigenvalues, q[coordinate_order]),
    )
    assert permuted["mode1_overlap"] == pytest.approx(baseline["mode1_overlap"])
    assert permuted["best20_rank"] == baseline["best20_rank"]


def test_common_window_requires_exact_269_residue_membership(tmp_path):
    module = load_module()
    short = tmp_path / "window.csv"
    short.write_text(
        "index,author_resnum\n" + "".join(f"{index},{index + 1}\n" for index in range(268)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="269 residues"):
        module.read_residue_window(short)

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "index,author_resnum\n" + "".join(f"{index},{1 if index == 5 else index}\n" for index in range(269)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        module.read_residue_window(duplicate)


def test_control_panel_percentile_excludes_existing_crbn_row(tmp_path, monkeypatch):
    module = load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    controls_csv = data_dir / "positive_controls.csv"
    controls_csv.write_text(
        "name,passes_quality,substantial_transition,motion_class,mode1_overlap_15A,"
        "best_overlap_15A,best_rank_15A,cum_overlap_top3_15A\n"
        + "".join(
            f"control_{index},True,True,hinge,{0.1 if index < 16 else 0.9},"
            f"{0.1 if index < 10 else 0.9},{index + 1},0.2\n"
            for index in range(18)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DATA", data_dir)
    primary = {
        "mode1_overlap": 0.763724842429092,
        "best20_overlap": 0.763724842429092,
        "best20_rank": 1,
        "top3_projection": 0.7920240537247211,
    }

    rows, percentiles = module.control_panel_rows(primary)

    assert len(rows) == 19
    assert sum(row["source"] == "existing_external_primary_control_panel" for row in rows) == 18
    assert rows[-1]["name"] == "CRBN 8CVP-5FQD"
    assert "not a proteome-wide percentile" in rows[-1]["panel_percentile_scope"]
    assert percentiles["mode1_overlap"] == pytest.approx(88.88888888888889)
    assert percentiles["best20_overlap"] == pytest.approx(55.55555555555556)


def test_signed_local_tangent_matches_known_axis_rigid_rotation():
    module = load_module()
    angle = np.deg2rad(82.457060)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    axis = np.array([0.0, 0.0, 1.0])
    points = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [-5.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    tangent = module.local_screw_tangent(points, np.zeros(3), axis, rotation, 0.0)
    chord = (rotation @ points.T).T - points
    cosine = abs(float(np.dot(chord.reshape(-1), tangent.reshape(-1)))) / (
        np.linalg.norm(chord) * np.linalg.norm(tangent)
    )

    assert module.signed_rotation_angle(rotation, axis) == pytest.approx(angle)
    assert module.signed_rotation_angle(rotation, -axis) == pytest.approx(-angle)
    assert cosine == pytest.approx(np.cos(angle / 2.0), abs=1e-12)


def test_generated_strengthening_outputs_have_expected_schema():
    if not OUTPUT.is_dir():
        pytest.skip("generated strengthening outputs are not part of the code-only public clone")
    expected_counts = {
        "endpoint_scores.csv": 397,
        "control_panel_comparison.csv": 19,
        "control_endpoint_rankings.csv": 36,
        "control_paired_state_effect_summary.csv": 18,
        "residue_effects.csv": 269,
        "residue_set_effects.csv": 15,
        "finite_chord_tangent.csv": 269,
    }
    for filename, expected in expected_counts.items():
        with (OUTPUT / filename).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected, filename

    summary = (OUTPUT / "strengthen_controls_summary.json").read_text(encoding="utf-8")
    assert "not a proteome-wide percentile" in summary
    assert "Exact residue-set p values are exploratory" in summary
    assert "control_endpoint_state_effects" in summary

    with (OUTPUT / "control_endpoint_rankings.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    open_rows = [row for row in rows if row["endpoint_basis_state"] == "open"]
    closed_rows = [row for row in rows if row["endpoint_basis_state"] == "closed"]
    assert len(open_rows) == len(closed_rows) == 18
    assert max(float(row["legacy_open_mode1_abs_delta"]) for row in open_rows) < 1e-8
    assert all("no imputation" in row["residue_pairing_rule"] for row in rows)
