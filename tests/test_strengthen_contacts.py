"""Independent invariants for Schur compliance and contact perturbations."""
import sys
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import strengthen_contacts as subject


@pytest.fixture
def system():
    rng = np.random.default_rng(721)
    xyz = rng.normal(size=(18, 3))
    direction = rng.normal(size=30)
    hessian = subject.anm_hessian(xyz, 10.)
    state = subject.schur_state(hessian, xyz[:10], direction)
    return xyz, direction, hessian, state


def test_schur_matches_balanced_full_joint_static_response(system):
    xyz, _, h, state = system
    force = np.zeros(len(h))
    force[:30] = state["q"]
    # A global gauge gives the same receptor-internal response as a CRBN gauge.
    rigid_joint = subject.rigid_basis(xyz)
    response = solve(h + rigid_joint @ rigid_joint.T, force, assume_a="pos")
    internal = subject.project_internal(response[:30], state["rigid"])
    np.testing.assert_allclose(internal, state["pinv"] @ state["q"], atol=1e-10)
    assert np.linalg.norm(rigid_joint.T @ force) < 1e-12


@pytest.mark.parametrize("factor", [.8, .9, 1., 1.1, 1.2])
def test_woodbury_equals_independent_full_recomputation(system, factor):
    xyz, direction, h, state = system
    columns = subject.edge_columns(xyz, [(1, 6), (2, 13), (4, 17), (8, 12)])
    updates = subject.prepare_updates(state, columns)
    result = subject.perturbation_metrics(state, updates, [0, 1, 3], factor)
    u = columns[:, [0, 1, 3]]
    direct = subject.schur_state(h + (factor - 1) * u @ u.T, xyz[:10], direction)
    np.testing.assert_allclose(result["S_close"], direct["specificity"], rtol=1e-10)
    np.testing.assert_allclose(result["C_close"], direct["cclose"], rtol=1e-10)
    np.testing.assert_allclose(result["mean_compliance"], direct["mean"], rtol=1e-10)


def test_uniform_stiffness_changes_compliance_but_not_specificity(system):
    xyz, direction, h, state = system
    scaled = subject.schur_state(3.7 * h, xyz[:10], direction)
    np.testing.assert_allclose(scaled["specificity"], state["specificity"], rtol=1e-11)
    np.testing.assert_allclose(scaled["cclose"] * 3.7, state["cclose"], rtol=1e-11)


def test_rotation_translation_and_node_order_invariance(system):
    xyz, direction, _, state = system
    rot = Rotation.from_rotvec([.7, -.2, 1.3]).as_matrix()
    perm = np.r_[np.arange(9, -1, -1), np.arange(17, 9, -1)]
    transformed = (xyz @ rot.T + [5, -12, 29])[perm]
    target = (direction.reshape(-1, 3) @ rot.T)[perm[:10]].ravel()
    changed = subject.schur_state(subject.anm_hessian(transformed, 10.), transformed[:10], target)
    np.testing.assert_allclose(changed["specificity"], state["specificity"], rtol=1e-10)
    np.testing.assert_allclose(changed["cclose"], state["cclose"], rtol=1e-10)


def test_preserves_six_rigid_nulls_and_rejects_disconnected_network(system):
    xyz, direction, h, state = system
    assert state["zero_modes"] == 6
    assert np.linalg.norm(state["effective"] @ state["rigid"]) < 1e-10
    broken = h.copy()
    broken[30:, :] = 0
    broken[:, 30:] = 0
    with pytest.raises(np.linalg.LinAlgError):
        subject.schur_state(broken, xyz[:10], direction)


def test_candidates_keep_but_do_not_perturb_sequential_cross_domain_edges():
    xyz = np.random.default_rng(43).normal(size=(8, 3))
    residues = np.array([185, 187, 300, 316, 317, 318, 319])
    edges, groups, _ = subject.candidate_groups(xyz, residues, 10)
    assert (4, 5) not in edges  # 317-318 remains in H, absent from perturbation set.
    assert (3, 5) not in edges  # 316-318 sequence distance 2.
    assert (2, 5) in edges
    assert (2, 6) in edges
    assert (187, "HB_TBD") in groups
    assert (185, "HB_TBD") not in groups
    assert (185, "CRBN_DDB1") in groups
    assert subject.anm_hessian(xyz, 10)[12:15, 15:18].any()


def test_insufficient_controls_never_get_percentile():
    row = {"residue": 200, "contact_class": "HB_TBD", "domain": "HB", "edge_ids": "1;2",
           "contact_count": 2, "joint_degree": 20, "axis_distance_A": 3., "D_g": .4}
    rows = [row, {**row, "residue": 201}]
    config = {"minimum_controls": 10, "control_calipers": {"contact_count_relative": .2,
              "degree_relative": .2, "axis_distance_A": 2.}}
    matched = subject.matched_controls(rows, config)
    assert all(r["control_n"] == 0 and r["matched_abs_effect_percentile"] is None for r in matched)


def test_nonpositive_factor_rejected(system):
    xyz, _, _, state = system
    updates = subject.prepare_updates(state, subject.edge_columns(xyz, [(1, 2)]))
    with pytest.raises(ValueError, match="positive"):
        subject.perturbation_metrics(state, updates, [0], 0)


def test_resume_requires_complete_checksum_verified_outputs(tmp_path):
    assert not subject.validated_completion(tmp_path)
    for name in subject.CONDITION_FILES:
        (tmp_path / name).write_text('{}\n' if name.endswith('.json') else 'residue,effect\n200,0.2\n')
    (tmp_path / 'output_hashes.json').write_text(json.dumps(subject.condition_hashes(tmp_path)))
    assert subject.validated_completion(tmp_path)
    (tmp_path / 'effects.csv').write_text('residue,effect\n200,0.9\n')
    with pytest.raises(ValueError, match='checksum'):
        subject.validated_completion(tmp_path)
    (tmp_path / 'effects.csv').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        subject.validated_completion(tmp_path)
