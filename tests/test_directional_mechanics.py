from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import directional_mechanics as dm


def toy_complex() -> tuple[np.ndarray, int, np.ndarray]:
    crbn = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [2.0, 2.0, 0.5],
            [2.0, 0.5, 2.0],
        ],
        dtype=float,
    )
    ddb1 = np.array(
        [
            [0.8, 0.8, 1.0],
            [3.0, 0.8, 1.0],
            [0.8, 3.0, 1.0],
            [0.8, 0.8, 3.0],
            [3.0, 3.0, 1.2],
            [3.0, 1.2, 3.0],
        ],
        dtype=float,
    )
    direction = np.linspace(-1.0, 1.0, 3 * len(crbn))
    return np.vstack([crbn, ddb1]), len(crbn), direction


def test_build_system_returns_weighted_blocks_and_edge_schema():
    coords, n_crbn, _ = toy_complex()
    uniform = dm.build_system(coords, n_crbn, cutoff=3.1)
    weighted = dm.build_system(coords, n_crbn, cutoff=3.1, weighting="inverse_square")
    weighted_alias = dm.build_system(coords, n_crbn, cutoff=3.1, weighting="r2")

    assert uniform["pairs"].shape[1] == 2
    assert uniform["weights"].shape == uniform["distances"].shape
    assert set(uniform["edge_types"]) == {"crbn_crbn", "ddb1_ddb1", "interface"}
    np.testing.assert_allclose(uniform["weights"], 1.0)
    np.testing.assert_allclose(weighted["pairs"], uniform["pairs"])
    np.testing.assert_allclose(weighted["weights"], (15.0 / weighted["distances"]) ** 2)
    np.testing.assert_allclose(weighted_alias["weights"], weighted["weights"])
    np.testing.assert_allclose(uniform["hessian"], np.block([[uniform["A"], uniform["B"]], [uniform["B"].T, uniform["D"]]]))
    assert "columns" not in uniform


def test_make_states_uses_common_internal_basis_and_orders_static_models():
    coords, n_crbn, direction = toy_complex()
    system = dm.build_system(coords, n_crbn, cutoff=3.1)
    states, checks = dm.make_states(system, coords[:n_crbn], direction)

    assert set(states) == {"isolated", "fixed", "rigid", "flexible"}
    assert checks["all_order_checks_pass"] is True
    for state in states.values():
        assert state["U"].shape == states["isolated"]["U"].shape
        np.testing.assert_allclose(state["q"], states["isolated"]["q"])
        assert state["H"].shape == (3 * n_crbn - 6, 3 * n_crbn - 6)
        assert state["G"].shape == state["H"].shape
        assert state["B"].shape[0] == 3 * n_crbn - 6
        assert state["C_close"] > 0
        assert state["mean_compliance"] > 0
        assert state["S_close"] > 0
    assert states["isolated"]["B"].shape == (3 * n_crbn - 6, 0)
    assert states["fixed"]["B"].shape == (3 * n_crbn - 6, 0)
    assert states["rigid"]["B"].shape[1] == 6
    assert states["flexible"]["B"].shape[1] == system["ddb1_dim"]
    assert states["rigid"]["partner_basis"].shape == (system["ddb1_dim"], 6)
    assert states["flexible"]["partner_basis"] == "identity"
    assert checks["stiffness_order_diagnostics"]["isolated_le_flexible"]["pass"] is True
    assert "min_eigenvalue" in checks["stiffness_order_diagnostics"]["isolated_le_flexible"]


def test_reduced_static_response_matches_flexible_solution_equations():
    coords, n_crbn, direction = toy_complex()
    system = dm.build_system(coords, n_crbn, cutoff=3.1)
    states, _ = dm.make_states(system, coords[:n_crbn], direction)
    crbn_response, partner_response = dm.full_static_response(states["flexible"])

    assert partner_response is not None
    x = states["flexible"]["U"].T @ crbn_response
    np.testing.assert_allclose(
        states["flexible"]["H"] @ x,
        states["flexible"]["q"],
        atol=1e-10,
    )
    d_residual = system["B"].T @ crbn_response + system["D"] @ partner_response
    np.testing.assert_allclose(d_residual, 0.0, atol=1e-9)


def test_contact_worker_smoke_shapes_candidate_edge_columns_against_states():
    coords, n_crbn, direction = toy_complex()
    system = dm.build_system(coords, n_crbn, cutoff=3.1)
    states, _ = dm.make_states(system, coords[:n_crbn], direction)
    candidate_pairs = system["pairs"][system["edge_types"] == "interface"][:2]
    candidate_columns = dm.edge_columns(coords, candidate_pairs)
    reduced_candidate_columns = states["fixed"]["U"].T @ candidate_columns[: 3 * n_crbn]

    assert candidate_pairs.shape == (2, 2)
    assert candidate_columns.shape == (3 * len(coords), 2)
    assert reduced_candidate_columns.shape == (3 * n_crbn - 6, 2)
    for state in states.values():
        assert state["B"].shape[0] == reduced_candidate_columns.shape[0]


def test_rotation_translation_and_permutation_invariance_for_static_states():
    coords, n_crbn, direction = toy_complex()
    base_system = dm.build_system(coords, n_crbn, cutoff=3.1)
    base_states, base_checks = dm.make_states(base_system, coords[:n_crbn], direction)

    rot = Rotation.from_rotvec([0.3, -0.7, 1.2]).as_matrix()
    transformed = coords @ rot.T + np.array([12.0, -3.0, 4.5])
    transformed_direction = (direction.reshape(n_crbn, 3) @ rot.T).ravel()
    changed_system = dm.build_system(transformed, n_crbn, cutoff=3.1)
    changed_states, changed_checks = dm.make_states(
        changed_system, transformed[:n_crbn], transformed_direction
    )
    for name in base_states:
        np.testing.assert_allclose(
            changed_states[name]["S_close"], base_states[name]["S_close"], rtol=1e-10
        )
        np.testing.assert_allclose(
            changed_states[name]["C_close"], base_states[name]["C_close"], rtol=1e-10
        )
    np.testing.assert_allclose(changed_checks["R_body"], base_checks["R_body"], atol=1e-11)

    perm_crbn = np.array([2, 0, 1, 5, 3, 4])
    perm_ddb1 = np.array([10, 7, 6, 11, 8, 9])
    perm = np.r_[perm_crbn, perm_ddb1]
    permuted = coords[perm]
    permuted_direction = direction.reshape(n_crbn, 3)[perm_crbn].ravel()
    permuted_system = dm.build_system(permuted, n_crbn, cutoff=3.1)
    permuted_states, _ = dm.make_states(permuted_system, permuted[:n_crbn], permuted_direction)
    for name in base_states:
        np.testing.assert_allclose(
            permuted_states[name]["S_close"], base_states[name]["S_close"], rtol=1e-10
        )
        np.testing.assert_allclose(
            permuted_states[name]["C_close"], base_states[name]["C_close"], rtol=1e-10
        )


def test_uniform_spring_scaling_preserves_specificity_and_scales_compliance():
    coords, n_crbn, direction = toy_complex()
    system = dm.build_system(coords, n_crbn, cutoff=3.1)
    scaled = {**system}
    for key in ("h_crbn_isolated", "A", "B", "D", "hessian"):
        scaled[key] = 4.2 * system[key]
    base_states, _ = dm.make_states(system, coords[:n_crbn], direction)
    scaled_states, _ = dm.make_states(scaled, coords[:n_crbn], direction)
    for name in base_states:
        np.testing.assert_allclose(scaled_states[name]["S_close"], base_states[name]["S_close"], rtol=1e-10)
        np.testing.assert_allclose(4.2 * scaled_states[name]["C_close"], base_states[name]["C_close"], rtol=1e-10)


def test_invalid_direction_shape_is_rejected():
    coords, n_crbn, _ = toy_complex()
    system = dm.build_system(coords, n_crbn, cutoff=3.1)
    with pytest.raises(ValueError, match="direction"):
        dm.make_states(system, coords[:n_crbn], np.ones(3 * n_crbn + 1))


def test_geometry_directions_are_covariant_with_same_rng_coefficients():
    crbn = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [4.0, 0.2, 0.5],
            [4.2, 1.5, 1.0],
            [4.1, 0.7, 2.0],
        ],
        dtype=float,
    )
    residues = np.array([180, 200, 260, 317, 318, 340, 420])
    direction = np.arange(3 * len(crbn), dtype=float)
    frozen = {"axis_unit_vector": [0.0, 0.0, 1.0], "axis_point_A": [1.0, 0.0, 0.0]}
    result = dm.geometry_directions(crbn, residues, direction, frozen, n_draws=7)
    result_axis_q = dm.geometry_directions(
        crbn, residues, result["observed_rotation_tangent"], frozen, n_draws=1
    )
    mixed_rotation = result["observed_rotation_tangent"] + 2.0 * result["basis"][:, 1]
    result_other_q = dm.geometry_directions(crbn, residues, mixed_rotation, frozen, n_draws=1)

    axis = np.array(frozen["axis_unit_vector"], dtype=float)
    point = np.array(frozen["axis_point_A"], dtype=float)
    centroid = crbn[residues > 317].mean(axis=0)
    expected_pivot = point + axis * float((centroid - point) @ axis)
    np.testing.assert_allclose(result["pivot"], expected_pivot)
    np.testing.assert_allclose(result["basis"].T @ result["basis"], np.eye(3), atol=1e-10)
    assert result["axis_seed_residue"] == 318
    assert 0.0 <= result["finite_tangent_overlap"] <= 1.0
    assert result_axis_q["finite_tangent_overlap"] == pytest.approx(1.0)
    expected_mixed_overlap = abs(
        result_other_q["finite_direction"] @ result_other_q["observed_rotation_tangent"]
    )
    assert result_other_q["finite_tangent_overlap"] == pytest.approx(expected_mixed_overlap)
    assert result_other_q["finite_tangent_overlap"] < 1.0
    np.testing.assert_allclose(
        result["observed_rotation_tangent"],
        result["basis"] @ result["observed_rotation_coefficients"],
        atol=1e-10,
    )

    rot = Rotation.from_rotvec([0.6, -0.1, 0.2]).as_matrix()
    shift = np.array([5.0, -4.0, 3.0])
    transformed = crbn @ rot.T + shift
    transformed_direction = (direction.reshape(-1, 3) @ rot.T).ravel()
    transformed_frozen = {
        "axis_unit_vector": axis @ rot.T,
        "axis_point_A": point @ rot.T + shift,
    }
    changed = dm.geometry_directions(
        transformed, residues, transformed_direction, transformed_frozen, n_draws=7
    )

    def transform_flat(values: np.ndarray) -> np.ndarray:
        return values.reshape(-1, len(crbn), 3) @ rot.T

    np.testing.assert_allclose(changed["coefficients"], result["coefficients"])
    np.testing.assert_allclose(
        changed["sampled_directions"].reshape(7, len(crbn), 3),
        transform_flat(result["sampled_directions"]),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        changed["observed_rotation_tangent"].reshape(len(crbn), 3),
        result["observed_rotation_tangent"].reshape(len(crbn), 3) @ rot.T,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        changed["finite_tangent_overlap"], result["finite_tangent_overlap"], atol=1e-11
    )
