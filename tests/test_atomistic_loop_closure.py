"""Synthetic geometry and provenance tests; these are not scientific MD data."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts import atomistic_loop_closure as lc
from scripts.atomistic_caps import AtomRecord, format_atom


def place_atom(a, b, c, length, angle_deg, torsion_deg):
    axis = (c - b) / np.linalg.norm(c - b)
    normal = np.cross(b - a, axis)
    normal /= np.linalg.norm(normal)
    lateral = np.cross(normal, axis)
    angle, torsion = np.radians([angle_deg, torsion_deg])
    return c + length * (-np.cos(angle) * axis + np.sin(angle) * (np.cos(torsion) * lateral + np.sin(torsion) * normal))


def synthetic_donor(*, proline=False):
    """Six synthetic residues, ideal peptide lengths/angles and positive L sign."""
    donor = {"pdb_id": "SYNTHETIC", "label_asym_id": "D", "source_sha256": "0" * 64,
             "canonical_mapping": [], "atoms": [], "additional_bonds": []}
    n = np.array([0.0, 0.0, 0.0])
    ca = np.array([0.145, 0.0, 0.0])
    c = ca + 0.152 * np.array([np.cos(np.radians(69)), np.sin(np.radians(69)), 0.0])
    for residue in range(341, 347):
        name = "PRO" if proline and residue == 344 else "ALA"
        oxygen = place_atom(n, ca, c, 0.123, 120, 135)
        u, v = (n - ca) / np.linalg.norm(n - ca), (c - ca) / np.linalg.norm(c - ca)
        side = -0.5 * (u + v) + 0.8 * np.cross(u, v)
        cb = ca + 0.153 * side / np.linalg.norm(side)
        atoms = {"N": n, "CA": ca, "C": c, "O": oxygen, "CB": cb}
        if name == "PRO":
            atoms["CG"] = cb + np.array([-0.055, 0.06, 0.095])
            atoms["CD"] = n + np.array([0.04, -0.06, 0.12])
        identity = {"canonical_residue": residue, "label_seq_id": residue - 69, "residue_name": name}
        donor["canonical_mapping"].append(identity)
        donor["atoms"].extend({**identity, "atom_name": atom, "xyz_nm": xyz.tolist()} for atom, xyz in atoms.items())
        next_n = place_atom(n, ca, c, 0.133, 116, -45)
        next_ca = place_atom(ca, c, next_n, 0.146, 121, 180)
        next_c = place_atom(c, next_n, next_ca, 0.152, 111, -60)
        n, ca, c = next_n, next_ca, next_c
    return donor


@pytest.fixture
def peptide():
    return lc.peptide_from_input(synthetic_donor())


def target_fixture(peptide):
    target = {"chain": "B", "anchor_residues": [341, 346], "loop_residues": [342, 343, 344, 345]}
    lines, mapping = ["REMARK synthetic geometry fixture, not scientific data\r\n"], {}
    names = dict(peptide.residue_names)
    for serial, (key, xyz) in enumerate(zip(peptide.keys, peptide.coordinates_nm * 10.0, strict=True), 1):
        residue, name = key
        atom = AtomRecord("ATOM", serial, name, "", names[residue], "B", residue, "", *xyz,
                          element=name[0])
        line = format_atom(atom, serial) + "\r\n"
        lines.append(line)
        atom = lc.parse_pdb_atom_line(line)
        mapping["B", residue, name] = {"residue_name": names[residue], "xyz": atom.xyz,
                                       "source_status": "repaired_internal_missing_residue_or_atom" if residue in target["loop_residues"] else "observed_input"}
    lines.append("END\r\n")
    return target, lines, mapping


def test_rodrigues_preserves_radial_distance_and_axial_projection():
    rng = np.random.default_rng(17)
    points, origin, axis = rng.normal(size=(20, 3)), rng.normal(size=3), np.array([1.2, -0.4, 2.8])
    unit = axis / np.linalg.norm(axis)
    rotated = lc.rotate_about_axis(points, origin, axis, 0.72)
    np.testing.assert_allclose((points - origin) @ unit, (rotated - origin) @ unit, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(np.cross(points - origin, unit), axis=1),
                               np.linalg.norm(np.cross(rotated - origin, unit), axis=1), atol=1e-14)
    np.testing.assert_allclose(lc.rotate_about_axis(rotated, origin, axis, -0.72), points, atol=1e-14)
    on_axis = origin + np.outer([-3, 0, 2], unit)
    np.testing.assert_allclose(lc.rotate_about_axis(on_axis, origin, axis, 1.3), on_axis, atol=1e-14)


def test_rotation_rejects_zero_axis_and_nan():
    with pytest.raises(ValueError, match="degenerate"):
        lc.rotate_about_axis(np.ones((2, 3)), np.zeros(3), np.zeros(3), 0.3)
    with pytest.raises(ValueError, match="Nonfinite"):
        lc.rotate_about_axis(np.full((2, 3), np.nan), np.zeros(3), np.ones(3), 0.3)


def test_phi_psi_move_entire_correct_bond_component(peptide):
    torsions = {(t.residue, t.kind): t for t in lc.torsions_for(peptide)}
    phi = {peptide.keys[i] for i in torsions[343, "phi"].moving}
    psi = {peptide.keys[i] for i in torsions[343, "psi"].moving}
    assert {(343, "CB"), (343, "C"), (343, "O"), (344, "N")} <= phi
    assert (343, "N") not in phi and (343, "CA") not in phi
    assert {(343, "O"), (344, "N"), (344, "CB")} <= psi
    assert not {(343, "N"), (343, "CB"), (343, "CA"), (343, "C")} & psi


def test_arbitrary_torsions_preserve_lengths_angles_omega_chirality_pro_ring():
    peptide = lc.peptide_from_input(synthetic_donor(proline=True))
    torsions = lc.torsions_for(peptide)
    assert (344, "phi") not in {(t.residue, t.kind) for t in torsions}
    assert (344, "psi") in {(t.residue, t.kind) for t in torsions}
    with pytest.raises(ValueError, match="ring or crosslink"):
        lc.downstream_component(peptide, (peptide.index[344, "N"], peptide.index[344, "CA"]))
    transformed = lc.apply_torsions(peptide, torsions, np.random.default_rng(91).uniform(-2, 2, len(torsions)))
    report = lc.invariance_report(peptide, transformed)
    assert report["pass"], report
    ring = [peptide.index[344, atom] for atom in ("N", "CA", "CB", "CG", "CD")]
    before = peptide.coordinates_nm[ring]
    after = transformed[ring]
    np.testing.assert_allclose(np.linalg.norm(before[:, None] - before, axis=2),
                               np.linalg.norm(after[:, None] - after, axis=2), atol=2e-14)
    assert report["maximum_absolute_deltas"]["pro_n_volume"] < 1e-14


def test_extra_crosslink_cannot_be_silently_cut():
    donor = synthetic_donor()
    donor["additional_bonds"] = [[[342, "CB"], [345, "CB"]]]
    with pytest.raises(ValueError, match="ring or crosslink"):
        lc.torsions_for(lc.peptide_from_input(donor))


@pytest.mark.parametrize("defect", ["missing_atom", "nan", "duplicate", "identity", "label", "empty", "unsupported"])
def test_donor_identity_finite_and_completeness_are_required(defect):
    donor = synthetic_donor()
    if defect == "missing_atom":
        donor["atoms"].pop()
    elif defect == "nan":
        donor["atoms"][0]["xyz_nm"][0] = float("nan")
    elif defect == "duplicate":
        donor["atoms"].append(copy.deepcopy(donor["atoms"][0]))
    elif defect == "identity":
        donor["atoms"][0]["residue_name"] = "GLY"
    elif defect == "label":
        donor["canonical_mapping"][0]["label_seq_id"] = donor["canonical_mapping"][1]["label_seq_id"]
    elif defect == "empty":
        donor["canonical_mapping"] = []
    elif defect == "unsupported":
        donor["canonical_mapping"][0]["residue_name"] = "MSE"
    with pytest.raises(ValueError):
        lc.peptide_from_input(donor)


def test_solver_recovers_reachable_fixed_anchors_without_changing_internal_geometry(peptide):
    torsions = lc.torsions_for(peptide)
    perturbation = np.random.default_rng(3).normal(0, 0.08, len(torsions))
    desired = lc.apply_torsions(peptide, torsions, perturbation)
    desired = desired @ Rotation.from_rotvec([0.3, -0.2, 0.1]).as_matrix().T + [0.7, 0.2, -0.5]
    result = lc.close_loop(peptide, desired[lc.anchor_indices(peptide)], max_nfev=50)
    assert result["anchor_fit"]["pass"], result["anchor_fit"]
    assert result["donor_invariance"]["pass"]
    assert result["qualified_for_md"] is False
    assert len(result["starts"]) == 3
    assert [row["start_sd_radians"] for row in result["starts"]] == [0.0, 0.15, 0.35]


def test_graft_preserves_observed_and_all_nonloop_bytes(peptide):
    target, lines, mapping = target_fixture(peptide)
    output, report = lc.graft_lines(lines, peptide, peptide.coordinates_nm, target, mapping)
    assert report["pass"], report
    assert report["observed_atoms_preserved"] == 10
    assert report["replaced_modeled_atom_count"] == 20
    assert report["all_nonloop_lines_byte_identical"]
    assert b"\r\n" in "".join(output).encode()
    for before, after in zip(lines, output, strict=True):
        atom = lc.parse_pdb_atom_line(before)
        if atom is None or atom.resseq not in target["loop_residues"]:
            assert before.encode() == after.encode()


@pytest.mark.parametrize("defect", ["observed_loop", "unknown_loop", "changed_observed", "unobserved_anchor"])
def test_graft_refuses_observed_modification_or_unprovenanced_atoms(peptide, defect):
    target, lines, mapping = target_fixture(peptide)
    if defect == "observed_loop":
        mapping["B", 342, "CB"]["source_status"] = "observed_input"
    elif defect == "unknown_loop":
        del mapping["B", 342, "CB"]
    elif defect == "changed_observed":
        mapping["B", 341, "N"]["xyz"][0] += 0.1
    elif defect == "unobserved_anchor":
        mapping["B", 341, "O"]["source_status"] = "repaired_missing_heavy_atom"
    with pytest.raises(ValueError):
        lc.graft_lines(lines, peptide, peptide.coordinates_nm, target, mapping)


def test_fixed_endpoint_oxygen_rejects_peptide_plane_even_when_distance_passes(peptide):
    actual = peptide.coordinates_nm.copy()
    c, ca, oxygen = (peptide.index[341, name] for name in ("C", "CA", "O"))
    actual[[oxygen]] = lc.rotate_about_axis(actual[[oxygen]], actual[ca], actual[c] - actual[ca], 0.5)
    report = lc.junction_report(peptide, actual)
    assert not report["pass"]
    first = report["junctions"][0]
    assert 0.11 < first["C_N_nm"] < 0.17
    assert first["planarity_delta_from_donor_degrees"] > 20
    assert max(first["angle_delta_from_donor_degrees"]) < 1e-10


def test_loop_alpha_inversion_is_unqualified(peptide):
    coordinates = peptide.coordinates_nm.copy()
    n, ca, c, cb = (peptide.index[342, name] for name in ("N", "CA", "C", "CB"))
    normal = np.cross(coordinates[n] - coordinates[ca], coordinates[c] - coordinates[ca])
    normal /= np.linalg.norm(normal)
    coordinates[cb] -= 2 * np.dot(coordinates[cb] - coordinates[ca], normal) * normal
    report = lc.loop_geometry_report(peptide, coordinates, {342, 343, 344, 345}, serialized=False)
    assert report["chirality"]["alpha_min_nm3"] < 0
    assert not report["pass"]


def test_impossible_anchors_are_unqualified_even_with_finite_solver_result(peptide):
    target = peptide.coordinates_nm[lc.anchor_indices(peptide)].copy()
    target[4:] += [10, 0, 0]
    report = lc.close_loop(peptide, target, max_nfev=2)
    assert not report["anchor_fit"]["pass"]
    assert report["qualified_for_md"] is False


def test_cli_retains_failed_candidate_without_md_qualification(tmp_path, monkeypatch, peptide):
    target, lines, mapping = target_fixture(peptide)
    request = {"schema_version": 1, "coordinate_unit": "nm", "donor": synthetic_donor(), "target": target}
    source, pdb, config, csv_path = (tmp_path / name for name in ("donor.json", "input.pdb", "config.json", "mapping.csv"))
    source.write_text(json.dumps(request))
    pdb.write_bytes("".join(lines).encode())
    config.write_text("{}")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["chain", "residue_number", "residue_name", "atom", "source_status", "x", "y", "z"])
        writer.writeheader()
        for (chain, residue, name), row in mapping.items():
            writer.writerow({"chain": chain, "residue_number": residue, "atom": name,
                             "residue_name": row["residue_name"], "source_status": row["source_status"],
                             **dict(zip(("x", "y", "z"), row["xyz"], strict=True))})
    bad = peptide.coordinates_nm.copy()
    bad[peptide.index[342, "N"]] += [0.1, 0.1, 0.1]
    monkeypatch.setattr(lc, "close_loop", lambda *_: {"coordinates_nm": bad, "anchor_fit": {"pass": True},
                                                    "donor_invariance": {"pass": False}, "qualified_for_md": False})
    output = tmp_path / "new_output"
    result = lc.main(["--input", str(source), "--pdb", str(pdb), "--source-csv", str(csv_path),
                      "--config", str(config), "--output-dir", str(output), "--offline"])
    assert result == 2
    report = json.loads((output / "loop_closure.json").read_text())
    assert report["status"] == "unqualified_loop_geometry"
    assert report["qualified_for_md"] is False
    assert (output / "candidate_capped_heavy.pdb").is_file()
    with pytest.raises(FileExistsError):
        lc.main(["--input", str(source), "--pdb", str(pdb), "--source-csv", str(csv_path),
                 "--config", str(config), "--output-dir", str(output)])


def test_actual_explicit_donor_inventory_if_available():
    path = Path(__file__).resolve().parents[1] / "125_atomistic_response_validation_20260906/analysis/loop_closure_input_v2/loop_closure_input.json"
    if not path.is_file():
        pytest.skip("Private donor input is not distributed with portable tests")
    peptide = lc.peptide_from_input(json.loads(path.read_text())["donor"])
    assert len(peptide.keys) == 135
    assert sum(342 <= residue <= 357 for residue, _ in peptide.keys) == 120
    torsions = lc.torsions_for(peptide)
    assert len(torsions) == 34
    assert {(t.residue, t.kind) for t in torsions if t.kind == "phi"}.isdisjoint({(345, "phi"), (352, "phi")})
    changed = lc.apply_torsions(peptide, torsions, np.random.default_rng(4).normal(0, 0.3, len(torsions)))
    assert lc.invariance_report(peptide, changed)["pass"]
