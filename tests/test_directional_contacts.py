from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import directional_contacts as subject
import directional_mechanics as dm
import strengthen_contacts


def _direct_state_after_group(base_hessian, columns, edge_ids, factor, coords, n_crbn, q_vector, mode):
    u = columns[:, list(edge_ids)]
    h = base_hessian + (factor - 1.0) * (u @ u.T)
    system = dm.build_system(coords, n_crbn, cutoff=18.0, weighting="inverse_square")
    replacement = {**system, "hessian": h}
    crbn_dim = 3 * n_crbn
    replacement["A"] = h[:crbn_dim, :crbn_dim]
    replacement["B"] = h[:crbn_dim, crbn_dim:]
    replacement["D"] = h[crbn_dim:, crbn_dim:]
    replacement["h_crbn_isolated"] = system["h_crbn_isolated"]
    states, _ = dm.make_states(replacement, coords[:n_crbn], q_vector)
    return states[mode]


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(20260906)
    scale = 18.0 / 2.2
    crbn = scale * np.array(
        [
            [0.0, 0.0, 0.0],
            [1.3, 0.1, 0.0],
            [0.2, 1.4, 0.1],
            [0.1, 0.2, 1.5],
            [1.2, 1.1, 0.8],
            [2.0, 1.2, 1.1],
        ]
    )
    ddb1 = crbn[:4] + scale * np.array([1.4, 1.0, 0.9])
    coords = np.vstack([crbn, ddb1])
    residues = np.array([180, 200, 260, 289, 339, 422])
    cutoff = 18.0
    q_vector = rng.normal(size=3 * len(crbn))
    q_vector -= strengthen_contacts.rigid_basis(crbn) @ (strengthen_contacts.rigid_basis(crbn).T @ q_vector)
    system = dm.build_system(coords, len(crbn), cutoff=cutoff, weighting="inverse_square")
    states, _ = dm.make_states(system, coords[: len(crbn)], q_vector)
    return {
        "coords": coords,
        "residues": residues,
        "axis_distances": np.linspace(1.0, 6.0, len(residues)),
        "system": system,
        "states": states,
        "hessian": system["hessian"],
        "q_vector": q_vector,
        "n_crbn": len(crbn),
    }


def test_result_schema_and_missing_discovery_groups(synthetic):
    result = subject.analyse_contacts(
        synthetic["system"],
        synthetic["states"],
        synthetic["coords"],
        synthetic["residues"],
        synthetic["axis_distances"],
        discovery_keys=[{"residue": 339, "contact_class": "CRBN_DDB1"}, (999, "HB_TBD")],
        config={"spring_factors": (0.9, 1.1), "ridge_min_training": 20},
    )
    assert result["schema_version"] == "directional_contacts.v1"
    assert set(result) == {
        "schema_version",
        "summary",
        "state_baselines",
        "groups",
        "factor_effects",
        "role_factor_effects",
        "edges",
        "ridge",
    }
    assert result["summary"]["n_discovery_groups"] == 2
    assert result["summary"]["n_missing_groups"] == 1
    missing = [row for row in result["groups"] if row["status"] == "missing"]
    assert missing[0]["group_id"] == "999:HB_TBD"
    assert all(row["residue"] != 999 for row in result["factor_effects"])
    assert result["role_factor_effects"]


@pytest.mark.parametrize("state_name", ["fixed", "rigid", "flexible"])
@pytest.mark.parametrize("factor", [0.8, 0.9, 1.1, 1.2])
def test_low_rank_update_matches_direct_reassembly_for_all_partner_models(synthetic, state_name, factor):
    result = subject.analyse_contacts(
        synthetic["system"],
        synthetic["states"],
        synthetic["coords"],
        synthetic["residues"],
        synthetic["axis_distances"],
        discovery_keys=[(339, "CRBN_DDB1")],
        config={"spring_factors": (0.8, 0.9, 1.1, 1.2)},
    )
    group = next(row for row in result["groups"] if row["group_id"] == "339:CRBN_DDB1")
    edge_ids = tuple(int(value) for value in group["edge_ids"].split(";") if value)
    columns = subject._edge_columns(
        synthetic["coords"],
        [(row["i_node"], row["j_node"]) for row in result["edges"]],
        np.array([row["weight"] for row in result["edges"]]),
    )
    direct = _direct_state_after_group(
        synthetic["hessian"],
        columns,
        edge_ids,
        factor,
        synthetic["coords"],
        synthetic["n_crbn"],
        synthetic["q_vector"],
        state_name,
    )
    fast = next(
        row
        for row in result["factor_effects"]
        if row["group_id"] == "339:CRBN_DDB1"
        and row["state"] == state_name
        and row["spring_factor"] == factor
    )
    assert fast["C_close"] == pytest.approx(direct["C_close"], rel=1e-10)
    assert fast["mean_compliance"] == pytest.approx(direct["mean_compliance"], rel=1e-10)
    assert fast["S_close"] == pytest.approx(direct["S_close"], rel=1e-10)


def test_exact_derivative_matches_small_finite_difference(synthetic):
    result = subject.analyse_contacts(
        synthetic["system"],
        synthetic["states"],
        synthetic["coords"],
        synthetic["residues"],
        synthetic["axis_distances"],
        discovery_keys=[(339, "CRBN_DDB1")],
        config={"spring_factors": (0.9999, 1.0001), "central_low_factor": 0.9999, "central_high_factor": 1.0001},
    )
    group = next(row for row in result["groups"] if row["group_id"] == "339:CRBN_DDB1")
    assert group["flexible_D_g"] == pytest.approx(group["flexible_derivative_log_S_close"], rel=5e-5)
    assert group["rigid_D_g"] == pytest.approx(group["rigid_derivative_log_S_close"], rel=5e-5)
    assert group["fixed_D_g"] == pytest.approx(group["fixed_derivative_log_S_close"], rel=5e-5)


def test_shared_edges_excluded_from_ridge_training_and_insufficient_support(synthetic):
    result = subject.analyse_contacts(
        synthetic["system"],
        synthetic["states"],
        synthetic["coords"],
        synthetic["residues"],
        synthetic["axis_distances"],
        discovery_keys=None,
        config={"spring_factors": (0.9, 1.1), "ridge_min_training": 20, "ridge_min_same_domain": 10},
    )
    assert result["ridge"]
    assert all(row["status"] == "insufficient" for row in result["ridge"])
    shared = [row for row in result["groups"] if row.get("shared_edge_group_ids")]
    assert shared
    for row in shared:
        for other in row["shared_edge_group_ids"].split(";"):
            if other:
                assert other != row["group_id"]


def test_rejects_index_mixing_between_crbn_and_axis_distances(synthetic):
    with pytest.raises(ValueError, match="same length"):
        subject.analyse_contacts(
            synthetic["system"],
            synthetic["states"],
            synthetic["coords"],
            synthetic["residues"],
            synthetic["axis_distances"][:-1],
        )


def test_accepts_canonical_pairs_weights_and_inverse_square_string(synthetic):
    pairs = synthetic["system"]["pairs"]
    weights = synthetic["system"]["weights"]
    canonical = {"cutoff": 18.0, "pairs": pairs, "weights": weights}
    explicit = subject.analyse_contacts(
        canonical,
        synthetic["states"],
        synthetic["coords"],
        synthetic["residues"],
        synthetic["axis_distances"],
        discovery_keys=[(339, "CRBN_DDB1")],
        config={"spring_factors": (0.9, 1.1)},
    )
    implicit = subject.analyse_contacts(
        {"cutoff": 18.0, "pairs": pairs, "weights": "inverse_square"},
        synthetic["states"],
        synthetic["coords"],
        synthetic["residues"],
        synthetic["axis_distances"],
        discovery_keys=[(339, "CRBN_DDB1")],
        config={
            "spring_factors": (0.9, 1.1),
            "inverse_square_reference_distance_A": 15.0,
        },
    )
    assert explicit["factor_effects"][0]["S_close"] == pytest.approx(
        implicit["factor_effects"][0]["S_close"], rel=1e-12
    )


def test_system_cutoff_overrides_config_default_for_candidate_detection(synthetic):
    low_edges, low_groups, _ = strengthen_contacts.candidate_groups(
        synthetic["coords"], synthetic["residues"], 15.0
    )
    assert (180, "CRBN_DDB1") not in low_groups
    assert low_edges
    result = subject.analyse_contacts(
        synthetic["system"],
        synthetic["states"],
        synthetic["coords"],
        synthetic["residues"],
        synthetic["axis_distances"],
        discovery_keys=[(180, "CRBN_DDB1")],
        config={"cutoff_A": 15.0, "spring_factors": (0.9, 1.1)},
    )
    group = next(row for row in result["groups"] if row["group_id"] == "180:CRBN_DDB1")
    assert group["status"] == "present"


def test_missing_pair_weight_is_hard_failure_when_pairs_are_provided(synthetic):
    pairs = np.asarray(synthetic["system"]["pairs"])
    weights = np.asarray(synthetic["system"]["weights"])
    result = subject.analyse_contacts(
        synthetic["system"],
        synthetic["states"],
        synthetic["coords"],
        synthetic["residues"],
        synthetic["axis_distances"],
        discovery_keys=[(339, "CRBN_DDB1")],
        config={"spring_factors": (0.9, 1.1)},
    )
    group = next(row for row in result["groups"] if row["group_id"] == "339:CRBN_DDB1")
    first_edge = int(group["edge_ids"].split(";")[0])
    selected = result["edges"][first_edge]
    pair = tuple(sorted((selected["i_node"], selected["j_node"])))
    keep = np.array([tuple(sorted(row)) != pair for row in pairs])
    with pytest.raises(ValueError, match="Missing weights"):
        subject.analyse_contacts(
            {**synthetic["system"], "pairs": pairs[keep], "weights": weights[keep]},
            synthetic["states"],
            synthetic["coords"],
            synthetic["residues"],
            synthetic["axis_distances"],
            discovery_keys=[(339, "CRBN_DDB1")],
            config={"spring_factors": (0.9, 1.1)},
        )


def test_edges_export_per_edge_state_derivatives(synthetic):
    result = subject.analyse_contacts(
        synthetic["system"],
        synthetic["states"],
        synthetic["coords"],
        synthetic["residues"],
        synthetic["axis_distances"],
        discovery_keys=[(339, "CRBN_DDB1")],
        config={"spring_factors": (0.9, 1.1)},
    )
    edge = next(row for row in result["edges"] if row["group_ids"])
    for state_name in ("fixed", "rigid", "flexible"):
        assert f"{state_name}_derivative_log_C_close" in edge
        assert f"{state_name}_derivative_log_mean_compliance" in edge
        assert f"{state_name}_derivative_log_S_close" in edge
    assert "delta_R_body_derivative_log_S_close" in edge
    assert "delta_R_internal_derivative_log_S_close" in edge


def test_ridge_uses_mean_squared_loss_not_sum_squared_loss():
    rows = []
    for idx, residue in enumerate([200, 210, 220, 230, 240, 250]):
        rows.append(
            {
                "group_id": f"{residue}:HB_TBD",
                "residue": residue,
                "contact_class": "HB_TBD",
                "domain": "HB",
                "edge_ids_tuple": (idx,),
                "contact_count": idx + 1,
                "joint_degree": 10 + 2 * idx,
                "axis_distance_A": float(idx),
                "flexible_derivative_log_S_close_per_edge": float(idx - 2),
            }
        )
    coords = np.array([[30.0 * idx, 0.0, 0.0] for idx in range(len(rows))])
    residues = np.array([row["residue"] for row in rows])
    ridge = subject._ridge_fit(
        rows,
        coords,
        residues,
        alpha=2.0,
        min_training=3,
        min_same_domain=3,
        neighbor_cutoff=10.0,
    )
    target = rows[0]
    train = rows[1:]
    domains = ["HB"]
    x_train = np.asarray([subject._ridge_features(row, domains) for row in train], dtype=float)
    y_train = np.asarray([row["flexible_derivative_log_S_close_per_edge"] for row in train])
    mean = x_train[:, :3].mean(axis=0)
    sd = x_train[:, :3].std(axis=0)
    sd[sd == 0.0] = 1.0
    x_train[:, :3] = (x_train[:, :3] - mean) / sd
    x_target = np.asarray(subject._ridge_features(target, domains), dtype=float)
    x_target[:3] = (x_target[:3] - mean) / sd
    design = np.column_stack([np.ones(len(train)), x_train])
    penalty = np.eye(design.shape[1]) * 2.0
    penalty[0, 0] = 0.0
    expected = np.r_[1.0, x_target] @ np.linalg.solve(
        design.T @ design / len(train) + penalty,
        design.T @ y_train / len(train),
    )
    sum_loss = np.r_[1.0, x_target] @ np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y_train,
    )
    fitted = next(row for row in ridge if row["group_id"] == "200:HB_TBD")
    assert fitted["status"] == "fit"
    assert fitted["predicted_per_edge_derivative"] == pytest.approx(expected)
    assert fitted["predicted_per_edge_derivative"] != pytest.approx(sum_loss)
    assert fitted["ridge_objective"] == "mean_squared_error_plus_alpha_l2"
