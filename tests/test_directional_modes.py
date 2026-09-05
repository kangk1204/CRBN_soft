from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import directional_mechanics as mechanics
import directional_modes as subject


def _tetra_pair() -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    crbn = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    ddb1 = crbn + np.array([3.0, 0.7, 0.4])
    coords = np.vstack([crbn, ddb1])
    residues = np.array([187, 260, 318, 420], dtype=int)
    return coords, len(crbn), residues, mechanics.internal_basis(crbn), mechanics.internal_basis(ddb1)


def _block_system(
    crbn_values: list[float],
    ddb1_values: list[float],
    *,
    a_slopes: list[float] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    coords, n_crbn, residues, u_crbn, u_ddb1 = _tetra_pair()
    crbn_dim = 3 * n_crbn
    ddb1_dim = 3 * (len(coords) - n_crbn)
    h_crbn = u_crbn @ np.diag(crbn_values) @ u_crbn.T
    h_ddb1 = u_ddb1 @ np.diag(ddb1_values) @ u_ddb1.T
    a_interface = np.zeros((crbn_dim, crbn_dim), dtype=float)
    if a_slopes is not None:
        a_interface = u_crbn @ np.diag(a_slopes) @ u_crbn.T
    system = {
        "coords": coords,
        "n_crbn": n_crbn,
        "h_crbn_isolated": h_crbn,
        "h_ddb1_within": h_ddb1,
        "A": h_crbn + a_interface,
        "B": np.zeros((crbn_dim, ddb1_dim), dtype=float),
        "D": h_ddb1,
        "a_interface": a_interface,
        "b_interface": np.zeros((crbn_dim, ddb1_dim), dtype=float),
        "d_interface": np.zeros((ddb1_dim, ddb1_dim), dtype=float),
    }
    return system, residues, u_crbn[:, 0].copy()


def test_alpha_zero_partner_reorders_without_changing_crbn_mode(tmp_path: Path):
    system, residues, direction = _block_system(
        crbn_values=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ddb1_values=[0.5, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    result = subject.run_modes(
        system,
        direction,
        residues,
        alphas=[0.0],
        output_dir=tmp_path,
        config={"stored_mode_count": 8, "primary_mode_count": 8, "near_degenerate_ratio": 1.2},
    )

    summary = result["alpha_summaries"][0]
    assert summary["best_mode"] == 2
    assert summary["tracked_rank"] == 2
    assert summary["internal_best_overlap"] == pytest.approx(1.0)
    mode_file = tmp_path / result["outputs"]["modes_alpha_0"]
    data = np.load(mode_file, allow_pickle=False)
    assert "eigenvalues" in data
    assert "eigenvectors" in data
    assert not any(str(value).startswith("/") for value in data.files)


def test_branch_tracking_follows_full_vectors_through_rank_crossing(tmp_path: Path):
    system, residues, direction = _block_system(
        crbn_values=[1.0, 2.0, 5.0, 6.0, 7.0, 8.0],
        ddb1_values=[9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        a_slopes=[2.0, -0.8, 0.0, 0.0, 0.0, 0.0],
    )
    result = subject.run_modes(
        system,
        direction,
        residues,
        alphas=[0.0, 1.0],
        output_dir=tmp_path,
        config={"stored_mode_count": 8, "primary_mode_count": 8, "near_degenerate_ratio": 1.2},
    )

    first, second = result["alpha_summaries"]
    assert first["tracked_rank"] == 1
    assert second["tracked_rank"] == 2
    assert second["previous_rank"] == 1
    assert second["branch_overlap_from_previous"] == pytest.approx(1.0, abs=1e-10)
    assert second["identity_interpretable"] is True
    assert result["branch_assignments"][0]["assignments"]


def test_near_degenerate_cluster_reports_projection_and_subspace_angles(tmp_path: Path):
    system, residues, direction = _block_system(
        crbn_values=[1.0, 1.1, 4.0, 5.0, 6.0, 7.0],
        ddb1_values=[8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
    )
    direction = direction + mechanics.internal_basis(system["coords"][: system["n_crbn"]])[:, 1]
    result = subject.run_modes(
        system,
        direction,
        residues,
        alphas=[0.0, 0.5],
        output_dir=tmp_path,
        config={"stored_mode_count": 8, "primary_mode_count": 8, "near_degenerate_ratio": 1.2},
    )

    first, second = result["alpha_summaries"]
    assert first["cluster_modes"] == [1, 2]
    assert first["cluster_projection"] == pytest.approx(1.0)
    assert second["prior_current_subspace"] is not None
    assert second["prior_current_subspace"]["singular_values"][0] == pytest.approx(1.0)


def test_scoring_separates_crbn_amplitude_from_normalized_overlap(tmp_path: Path):
    coords, n_crbn, residues, *_ = _tetra_pair()
    system = mechanics.build_system(coords, n_crbn, cutoff=4.0)
    direction = mechanics.internal_basis(coords[:n_crbn])[:, 0]
    result = subject.run_modes(
        system,
        direction,
        residues,
        alphas=[0.0, 0.4],
        output_dir=tmp_path,
        config={"stored_mode_count": 6, "primary_mode_count": 6, "near_degenerate_ratio": 1.2},
    )
    rows = (tmp_path / result["outputs"]["mode_scores_csv"]).read_text().splitlines()
    header = rows[0].split(",")
    assert "crbn_directional_overlap" in header
    assert "crbn_amplitude" in header
    assert "crbn_internal_overlap" in header
    assert all(0.0 <= row["best_crbn_directional_overlap"] <= 1.0 for row in result["alpha_summaries"])
    assert all(row["best_crbn_amplitude"] <= 1.0 + 1e-12 for row in result["alpha_summaries"])


def test_primary_best_uses_internal_projection_when_raw_rank_differs():
    coords, n_crbn, *_ = _tetra_pair()
    crbn = coords[:n_crbn]
    rigid = mechanics.rigid_basis(crbn)
    internal = mechanics.internal_basis(crbn)
    full_dim = 3 * n_crbn
    vectors = np.zeros((full_dim, 2), dtype=float)
    vectors[:, 0] = subject._unit(rigid[:, 0] + 0.05 * internal[:, 0])
    vectors[:, 1] = internal[:, 1]
    direction = subject._unit(rigid[:, 0] + 0.2 * internal[:, 1])

    _, summary = subject._score_modes(
        np.array([1.0, 2.0]),
        vectors,
        direction,
        crbn,
        alpha=0.0,
        primary_limit=2,
    )

    assert summary["raw_best_mode"] == 1
    assert summary["best_mode"] == 2
    assert summary["internal_best_mode"] == 2
    assert summary["best_crbn_internal_overlap"] == pytest.approx(1.0)


def test_existing_outputs_reject_changed_inputs_on_resume(tmp_path: Path):
    system, residues, direction = _block_system(
        crbn_values=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ddb1_values=[8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
    )
    config = {"stored_mode_count": 6, "primary_mode_count": 6, "near_degenerate_ratio": 1.2}
    subject.run_modes(system, direction, residues, alphas=[0.0], output_dir=tmp_path, config=config)
    with pytest.raises(ValueError, match="input hash"):
        subject.run_modes(
            system,
            direction + 0.01 * mechanics.internal_basis(system["coords"][: system["n_crbn"]])[:, 1],
            residues,
            alphas=[0.0],
            output_dir=tmp_path,
            config=config,
        )


def test_resume_loads_cached_modes_and_rescores_under_current_code(tmp_path: Path):
    system, residues, direction = _block_system(
        crbn_values=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ddb1_values=[8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
    )
    config = {"stored_mode_count": 6, "primary_mode_count": 6, "near_degenerate_ratio": 1.2}
    first = subject.run_modes(system, direction, residues, alphas=[0.0], output_dir=tmp_path, config=config)
    second = subject.run_modes(system, direction, residues, alphas=[0.0], output_dir=tmp_path, config=config)

    assert first["alpha_summaries"][0]["mode_cache_status"] == "computed"
    assert second["alpha_summaries"][0]["mode_cache_status"] == "cached"
    assert second["alpha_summaries"][0]["best_mode"] == first["alpha_summaries"][0]["best_mode"]
