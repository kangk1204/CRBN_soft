from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import directional_mechanics as dm
import review_window as subject


def _coordinates(n_extra: int = 2) -> tuple[np.ndarray, int, int]:
    core = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.1, 0.0],
            [0.2, 2.1, 0.1],
            [0.1, 0.2, 2.2],
            [1.7, 1.4, 1.1],
        ],
        dtype=float,
    )
    extra = np.array(
        [
            [1.0, 0.8, 2.8],
            [2.4, 1.8, 0.7],
        ],
        dtype=float,
    )[:n_extra]
    ddb1 = np.array(
        [
            [4.0, 0.2, 0.1],
            [4.2, 2.1, 0.3],
            [4.1, 0.4, 2.2],
            [5.6, 1.5, 1.2],
            [4.8, 2.4, 2.0],
        ],
        dtype=float,
    )
    coords = np.vstack([core, extra, ddb1])
    return coords, len(core) + len(extra), len(core)


def _direction(core_xyz: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(20260906)
    raw = rng.normal(size=3 * len(core_xyz))
    return dm.project_internal(raw, dm.rigid_basis(core_xyz))


def _system(n_extra: int = 2, cutoff: float = 4.2) -> tuple[dict, int, np.ndarray]:
    coords, n_crbn, n_core = _coordinates(n_extra)
    return dm.build_system(coords, n_crbn, cutoff=cutoff), n_core, _direction(coords[:n_core])


def _weighted_columns(system: dict, edge_ids: np.ndarray | list[int]) -> np.ndarray:
    columns = dm.edge_columns(system["coords"], system["pairs"][edge_ids])
    return columns * np.sqrt(system["weights"][edge_ids])[None, :]


def _low_rank_metrics(state: dict, columns: np.ndarray, factor: float) -> tuple[float, float, float]:
    updates = subject.prepare_updates(state, columns)
    change = factor - 1.0
    middle = np.eye(columns.shape[1]) + change * updates["gram"]
    changed_g = state["G"] - change * updates["response"] @ np.linalg.solve(
        middle, updates["response"].T
    )
    c_close = float(state["q"] @ changed_g @ state["q"])
    mean = float(np.trace(changed_g) / len(state["q"]))
    return c_close, mean, c_close / mean


def test_zero_added_window_reproduces_directional_mechanics_static_states() -> None:
    system, n_core, direction = _system(n_extra=0)
    expected, _ = dm.make_states(system, system["crbn_xyz"], direction)
    observed = subject.make_states(system, n_core, direction)

    for name in subject.MODELS:
        assert observed[name]["C_close"] == pytest.approx(expected[name]["C_close"], rel=1e-10)
        assert observed[name]["mean_compliance"] == pytest.approx(
            expected[name]["mean_compliance"], rel=1e-10
        )
        assert observed[name]["S_close"] == pytest.approx(expected[name]["S_close"], rel=1e-10)


@pytest.mark.parametrize("model", subject.MODELS)
def test_sequential_schur_response_matches_direct_constrained_core_block(model: str) -> None:
    system, n_core, direction = _system()
    state = subject.make_state(system, n_core, direction, model)
    direct = subject.verify_direct_response(system, state)
    full_inverse = np.linalg.inv(subject.constrained_matrix(system, state))
    core_block = full_inverse[: state["dof"], : state["dof"]]

    assert direct["pass"] is True
    np.testing.assert_allclose(state["G"], core_block, rtol=1e-10, atol=1e-10)
    assert np.trace(state["G"]) / state["dof"] == pytest.approx(state["mean_compliance"])


def test_rotation_translation_and_permutation_preserve_window_responses() -> None:
    system, n_core, direction = _system()
    base = subject.make_states(system, n_core, direction)
    coords, n_crbn, _ = _coordinates()
    rotation = Rotation.from_rotvec([0.3, -0.4, 0.7]).as_matrix()
    shift = np.array([10.0, -2.0, 5.0])
    transformed = coords @ rotation.T + shift
    transformed_direction = (direction.reshape(n_core, 3) @ rotation.T).ravel()
    changed = subject.make_states(
        dm.build_system(transformed, n_crbn, cutoff=4.2),
        n_core,
        transformed_direction,
    )

    for name in subject.MODELS:
        assert changed[name]["C_close"] == pytest.approx(base[name]["C_close"], rel=1e-10)
        assert changed[name]["S_close"] == pytest.approx(base[name]["S_close"], rel=1e-10)

    core_perm = np.array([2, 0, 4, 1, 3])
    extra_perm = np.array([1, 0])
    ddb1_perm = np.array([3, 1, 4, 0, 2])
    core = coords[:n_core][core_perm]
    extra = coords[n_core:n_crbn][extra_perm]
    ddb1 = coords[n_crbn:][ddb1_perm]
    permuted_direction = direction.reshape(n_core, 3)[core_perm].ravel()
    permuted = subject.make_states(
        dm.build_system(np.vstack([core, extra, ddb1]), n_crbn, cutoff=4.2),
        n_core,
        permuted_direction,
    )
    for name in subject.MODELS:
        assert permuted[name]["C_close"] == pytest.approx(base[name]["C_close"], rel=1e-10)
        assert permuted[name]["S_close"] == pytest.approx(base[name]["S_close"], rel=1e-10)


def test_each_window_model_obeys_absolute_compliance_order() -> None:
    system, n_core, direction = _system()
    states = subject.make_states(system, n_core, direction)

    assert states["fixed"]["C_close"] <= states["rigid"]["C_close"]
    assert states["rigid"]["C_close"] <= states["flexible"]["C_close"]
    assert states["flexible"]["C_close"] <= states["isolated"]["C_close"]


def test_uniform_scaling_preserves_specificity_and_scales_absolute_compliance() -> None:
    system, n_core, direction = _system()
    scaled = dict(system)
    scale = 3.7
    for key in ("h_crbn_isolated", "A", "B", "D", "hessian"):
        scaled[key] = scale * system[key]
    base = subject.make_states(system, n_core, direction)
    observed = subject.make_states(scaled, n_core, direction)

    for name in subject.MODELS:
        assert observed[name]["S_close"] == pytest.approx(base[name]["S_close"], rel=1e-10)
        assert scale * observed[name]["C_close"] == pytest.approx(
            base[name]["C_close"], rel=1e-10
        )


def test_nonnegative_added_springs_do_not_increase_core_direction_compliance() -> None:
    core_only, n_core_only, direction = _system(n_extra=0)
    expanded, n_core, _ = _system(n_extra=2)
    assert n_core == n_core_only
    core_states = subject.make_states(core_only, n_core, direction)
    expanded_states = subject.make_states(expanded, n_core, direction)

    for name in subject.MODELS:
        assert expanded_states[name]["C_close"] <= core_states[name]["C_close"]


def test_disconnected_added_residue_fails_instead_of_using_pseudoinverse() -> None:
    coords, n_crbn, n_core = _coordinates()
    coords[n_core] = np.array([100.0, 100.0, 100.0])
    system = dm.build_system(coords, n_crbn, cutoff=4.2)

    with pytest.raises(ArithmeticError, match="singular|zero mode"):
        subject.make_state(system, n_core, _direction(coords[:n_core]), "fixed")


@pytest.mark.parametrize("edge_type", ["crbn_crbn", "interface"])
@pytest.mark.parametrize("factor", [0.8, 0.9, 1.1, 1.2])
@pytest.mark.parametrize("model", subject.MODELS)
def test_grouped_low_rank_update_matches_direct_rebuild(
    edge_type: str, factor: float, model: str
) -> None:
    system, n_core, direction = _system()
    edge_ids = np.flatnonzero(system["edge_types"] == edge_type)[:2]
    columns = _weighted_columns(system, edge_ids)
    states = subject.make_states(system, n_core, direction)
    fast_c, fast_mean, fast_s = _low_rank_metrics(states[model], columns, factor)
    direct = subject.make_state(
        subject.changed_system(system, columns, factor),
        n_core,
        direction,
        model,
        U=states[model]["U"],
        q=states[model]["q"],
    )

    assert fast_c == pytest.approx(direct["C_close"], rel=1e-10, abs=1e-12)
    assert fast_mean == pytest.approx(direct["mean_compliance"], rel=1e-10, abs=1e-12)
    assert fast_s == pytest.approx(direct["S_close"], rel=1e-10, abs=1e-12)
