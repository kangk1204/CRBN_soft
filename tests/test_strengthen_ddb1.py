from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "strengthen_ddb1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("strengthen_ddb1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toy_coords():
    crbn = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    ddb1 = np.array(
        [
            [0.7, 0.7, 0.7],
            [1.7, 0.7, 0.7],
            [0.7, 1.7, 0.7],
            [0.7, 0.7, 1.7],
        ],
        dtype=float,
    )
    return crbn, ddb1


def test_interface_alpha_zero_is_exact_block_diagonal():
    module = load_module()
    crbn, ddb1 = toy_coords()
    parts = module.decompose_hessian(crbn, ddb1, cutoff=1.25)
    h0 = module.compose_joint(parts, 0.0)
    n = 3 * len(crbn)
    assert int(parts["n_interface_pairs"]) > 0
    assert np.allclose(h0[:n, n:], 0.0)
    assert np.allclose(h0[n:, :n], 0.0)
    assert np.allclose(h0[:n, :n], parts["h_crbn"])
    assert np.allclose(h0[n:, n:], parts["h_ddb1"])


def test_interface_alpha_scales_only_cross_springs():
    module = load_module()
    crbn, ddb1 = toy_coords()
    parts = module.decompose_hessian(crbn, ddb1, cutoff=1.25)
    h0 = module.compose_joint(parts, 0.0)
    h1 = module.compose_joint(parts, 1.0)
    h2 = module.compose_joint(parts, 2.0)
    assert np.allclose(h2 - h0, 2.0 * (h1 - h0))
    assert np.linalg.norm(h1 - h0) > 0.0


def test_schur_static_response_matches_full_joint_with_balanced_force():
    module = load_module()
    crbn, ddb1 = toy_coords()
    parts = module.decompose_hessian(crbn, ddb1, cutoff=2.0)
    joint = module.compose_joint(parts, 1.0)
    crbn_dim = 3 * len(crbn)
    schur, _, _, _ = module.schur_from_joint(joint, crbn_dim)
    force = module.balanced_force(np.arange(crbn_dim, dtype=float), crbn)
    full_force = np.zeros(joint.shape[0], dtype=float)
    full_force[:crbn_dim] = force
    full_response = module.static_response(joint, full_force, np.vstack([crbn, ddb1]))[:crbn_dim]
    schur_response = module.static_response(schur, force, crbn)
    rigid = module.rigid_basis(crbn)
    full_internal = module.remove_rigid(full_response, rigid)
    schur_internal = module.remove_rigid(schur_response, rigid)
    assert np.linalg.norm(full_internal - schur_internal) / np.linalg.norm(schur_internal) < 1e-8


def test_independent_full_joint_response_check_detects_wrong_schur():
    module = load_module()
    crbn, ddb1 = toy_coords()
    parts = module.decompose_hessian(crbn, ddb1, cutoff=2.0)
    joint_dense = module.compose_joint(parts, 1.0)
    joint_sparse = module.sparse.csr_matrix(joint_dense)
    crbn_dim = 3 * len(crbn)
    schur, _, _, _ = module.schur_from_joint(joint_dense, crbn_dim)
    force = module.balanced_force(np.arange(crbn_dim, dtype=float), crbn)

    error, _, _ = module.schur_full_static_response_check(
        joint_sparse, schur, crbn_dim, crbn, ddb1, force
    )
    wrong_error, _, _ = module.schur_full_static_response_check(
        joint_sparse, schur + 0.2 * np.eye(crbn_dim), crbn_dim, crbn, ddb1, force
    )

    assert error < 1e-8
    assert wrong_error > 1e-3


def test_crbn_metrics_reports_amplitude_and_near_degenerate_subspace():
    module = load_module()
    crbn, _ = toy_coords()
    hessian = np.eye(3 * len(crbn))
    hessian[:6, :6] *= 0.01
    values, vectors = module.slow_modes(hessian, 3)
    axis = vectors[:, 0] + vectors[:, 1]
    rows, summary = module.crbn_metrics(
        "isolated",
        values,
        vectors,
        3 * len(crbn),
        axis,
        module.rigid_basis(crbn),
        0.0,
        15.0,
        "TEST",
        3,
    )
    assert rows[0]["crbn_amplitude"] == pytest.approx(1.0)
    assert 0.0 <= summary["near_degenerate_internal_subspace_overlap"] <= 1.0
    assert summary["n_modes_returned"] == 3
    assert summary["near_degenerate_internal_subspace_rank"] >= 1


def test_higher_modes_do_not_change_primary_best20_fields():
    module = load_module()
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(12, 3))
    rigid = module.rigid_basis(coords)
    q, _ = np.linalg.qr(rng.normal(size=(36, 25)))
    axis = 0.5 * q[:, 0] + q[:, 20]
    values = np.arange(1, 26, dtype=float)
    _, summary = module.crbn_metrics(
        "isolated", values, q, 36, axis, rigid, 0.0, 15.0, "TEST", 25, primary_limit=20
    )
    assert summary["best_mode"] == 1
    assert summary["best60_mode"] == 21
    assert summary["higher_mode_21_60_changes_sensitivity_best"] is True


def test_sparse_slow_modes_match_dense_subspace_for_semidefinite_anm():
    module = load_module()
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    hessian = module.anm_hessian(coords, cutoff=1.75)
    dense_values, dense_vectors = module.slow_modes(hessian, 8)
    sparse_values, sparse_vectors = module.slow_modes_sparse(
        module.sparse.csr_matrix(hessian), 8, rigid_modes=6
    )

    all_values = np.linalg.eigvalsh(hessian)
    assert np.count_nonzero(all_values <= module.ZERO_TOL) == 6
    assert sparse_values == pytest.approx(dense_values, abs=1e-8)
    residuals = module.eigenpair_residuals(
        module.sparse.csr_matrix(hessian), sparse_values, sparse_vectors
    )
    assert float(residuals.max()) < 1e-9

    overlap = np.linalg.svd(dense_vectors.T @ sparse_vectors, compute_uv=False)
    rmsip = float(np.sqrt(np.mean(overlap**2)))
    assert rmsip > 0.999999


def test_sparse_slow_modes_are_repeatable_on_near_degenerate_model():
    module = load_module()
    rng = np.random.default_rng(20260905)
    coords = rng.normal(size=(14, 3))
    hessian = module.sparse.csr_matrix(module.anm_hessian(coords, cutoff=2.4))

    values_a, vectors_a = module.slow_modes_sparse(hessian, 12, rigid_modes=6)
    values_b, vectors_b = module.slow_modes_sparse(hessian, 12, rigid_modes=6)

    assert values_b == pytest.approx(values_a, abs=0.0)
    assert np.linalg.norm(np.abs(vectors_b) - np.abs(vectors_a)) == pytest.approx(0.0, abs=1e-12)


def test_near_degenerate_subspace_rank_trim_ignores_duplicate_projected_vectors():
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
    rigid = module.rigid_basis(coords)
    base = module.remove_rigid(np.arange(12, dtype=float), rigid)
    base /= np.linalg.norm(base)
    vectors = np.column_stack([base, base, np.eye(12)[:, 0]])
    values = np.array([1.0, 1.01, 5.0])
    _, summary = module.crbn_metrics(
        "isolated", values, vectors, 12, base, rigid, 0.0, 15.0, "TEST", 3, primary_limit=20
    )
    assert summary["near_degenerate_modes"] == [1, 2]
    assert summary["near_degenerate_internal_subspace_rank"] == 1
    assert summary["near_degenerate_internal_subspace_overlap"] == pytest.approx(1.0)


def test_cli_defaults_include_assigned_output_dir_and_offline_flag():
    module = load_module()
    args = module.parse_args(["--offline"])
    assert args.offline
    assert args.output_dir == module.DEFAULT_OUT
    assert args.primary_modes == 20
    assert args.sensitivity_modes == 60
