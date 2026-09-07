import json
import math

import numpy as np
import pytest

from scripts import prepare_atomistic_salt as salt


def test_salt_pair_count_uses_actual_box_volume_formula():
    assert salt.salt_pair_count(3884.618) == 351
    expected = round(0.150 * 3884.618 * 1e-24 * salt.AVOGADRO)
    assert expected == 351


def test_triclinic_minimum_image_wraps_row_vectors():
    box = salt.box_vectors_from_lengths_angles([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    delta = np.array([[0.91, 0.0, 0.0], [-0.91, 0.0, 0.0]])
    wrapped = salt.minimum_image_displacements(delta, box)
    assert np.allclose(wrapped, [[-0.09, 0.0, 0.0], [0.09, 0.0, 0.0]])
    assert math.isclose(salt.box_volume_nm3(box), 1.0)


def test_minimum_image_uses_27_neighbors_for_oblique_cells():
    box = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    frac = np.array([1.34594834, -0.56450564, -0.23002065])
    delta = frac @ box
    rounded_only = (frac - np.round(frac)) @ box
    wrapped = salt.minimum_image_displacements(delta.reshape(1, 3), box)[0]
    assert np.linalg.norm(rounded_only) > 0.73
    assert np.linalg.norm(wrapped) < 0.51
    assert np.allclose(wrapped, [0.007244956, -0.451604512, -0.23002065], atol=1e-8)


def test_kdtree_replicated_matches_27_neighbor_bruteforce():
    rng = np.random.default_rng(4)
    box = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.1, 0.2, 1.1]])
    points = rng.normal(size=(40, 3))
    refs = rng.normal(size=(35, 3))
    brute = salt.min_distance_nm_27_bruteforce(points, refs, box, chunk_size=7)
    kd = salt.min_distance_nm_kdtree_replicated(points, refs, box)
    assert np.allclose(kd, brute, atol=1e-12)


def test_deterministic_selection_filters_solute_ions_and_selected_clashes():
    oxy_idx = [10, 20, 30, 40, 50, 60]
    oxy = np.array([
        [0.10, 0.10, 0.10],  # too close to solute
        [1.00, 1.00, 1.00],
        [2.00, 2.00, 2.00],
        [2.10, 2.00, 2.00],  # clashes with index 30 if selected first
        [3.00, 3.00, 3.00],
        [4.00, 4.00, 4.00],
    ])
    solute = np.array([[0.12, 0.10, 0.10]])
    ions = np.array([[4.10, 4.00, 4.00]])  # makes index 60 unavailable
    box = np.diag([10.0, 10.0, 10.0])
    na1, cl1, audit1 = salt.deterministic_water_pair_selection(
        oxy_idx, oxy, solute, ions, box, 1, seed=20260907, min_distance=0.5
    )
    na2, cl2, audit2 = salt.deterministic_water_pair_selection(
        oxy_idx, oxy, solute, ions, box, 1, seed=20260907, min_distance=0.5
    )
    assert (na1, cl1, audit1) == (na2, cl2, audit2)
    assert len(na1) == len(cl1) == 1
    assert not ({10, 60} & set(na1 + cl1))
    assert audit1["selected_count"] == 2
    selected_pos = np.array([oxy[oxy_idx.index(i)] for i in na1 + cl1])
    assert salt.min_distance_nm(selected_pos, solute, box).min() >= 0.5
    assert salt.min_distance_nm(selected_pos, ions, box).min() >= 0.5


def test_selection_fails_when_waters_insufficient():
    box = np.diag([2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="insufficient waters"):
        salt.deterministic_water_pair_selection(
            [1, 2], np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), np.empty((0, 3)), np.empty((0, 3)), box, 1, min_distance=0.5
        )

try:
    import parmed as pmd
except Exception:  # pragma: no cover - local lightweight env may not have ParmEd
    pmd = None


def _require_parmed():
    if pmd is None:
        pytest.skip("ParmEd mutation tests run where AmberTools/ParmEd is installed")


def _atom(name, element, mass, charge=0.0, epsilon=0.1, rmin=1.5):
    atom_type = pmd.topologyobjects.AtomType(name, 0, mass, element)
    atom_type.set_lj_params(epsilon, rmin)
    atom = pmd.Atom(name=name, type=name, atomic_number=element, mass=mass, charge=charge)
    atom.atom_type = atom_type
    return atom


def _add_atom(structure, atom, resname, resnum):
    structure.add_atom(atom, resname, resnum)
    return atom


def _bond(structure, a, b):
    bt = pmd.BondType(300.0, 1.0)
    structure.bond_types.append(bt)
    structure.bonds.append(pmd.Bond(a, b, type=bt))


def _angle(structure, a, b, c):
    at = pmd.AngleType(50.0, 104.5)
    structure.angle_types.append(at)
    structure.angles.append(pmd.Angle(a, b, c, type=at))


def _fixture_parm(include_na=True, include_cl=False):
    s = pmd.Structure()
    n = _add_atom(s, _atom("N", 7, 14.01), "HIE", 1)
    ca = _add_atom(s, _atom("CA", 6, 12.01), "HIE", 1)
    c = _add_atom(s, _atom("C", 6, 12.01), "HIE", 1)
    cb = _add_atom(s, _atom("CB", 6, 12.01), "HIE", 1)
    he2 = _add_atom(s, _atom("HE2", 1, 1.008), "HIE", 1)
    _bond(s, n, ca); _bond(s, ca, c); _bond(s, ca, cb); _bond(s, cb, he2)
    _angle(s, n, ca, c)
    sg = _add_atom(s, _atom("SG", 16, 32.06, charge=-0.2, epsilon=0.2, rmin=1.8), "CY1", 2)
    zn = _add_atom(s, _atom("ZN", 30, 65.4, charge=0.5, epsilon=0.01, rmin=1.2), "ZN1", 3)
    _bond(s, sg, zn)
    if include_na:
        _add_atom(s, _atom("Na+", 11, 22.99, charge=1.0, epsilon=0.0874, rmin=1.369), "Na+", 4)
    if include_cl:
        _add_atom(s, _atom("Cl-", 17, 35.45, charge=-1.0, epsilon=0.0356, rmin=2.513), "Cl-", 5)
    for i in range(4):
        o = _add_atom(s, _atom("O", 8, 16.0, charge=-0.834, epsilon=0.1521, rmin=1.768), "WAT", 10 + i)
        h1 = _add_atom(s, _atom("H1", 1, 1.008, charge=0.417), "WAT", 10 + i)
        h2 = _add_atom(s, _atom("H2", 1, 1.008, charge=0.417), "WAT", 10 + i)
        _bond(s, o, h1); _bond(s, o, h2); _angle(s, h1, o, h2)
    coords = np.arange(len(s.atoms) * 3, dtype=float).reshape(-1, 3) * 0.1
    s.coordinates = coords
    s.box = [50, 50, 50, 90, 90, 90]
    return pmd.amber.AmberParm.from_structure(s)


def _write_parm(parm, path):
    parm.save(str(path), overwrite=True)
    return path


def test_parmed_mutation_adds_missing_cl_type_and_preserves_solute_invariants(tmp_path):
    _require_parmed()
    parm = _fixture_parm(include_na=True, include_cl=False)
    cl_template = _write_parm(_fixture_parm(include_na=False, include_cl=True), tmp_path / "cl_template.prmtop")
    water_oxygen_indices = [a.idx for a in parm.atoms if a.residue.name == "WAT" and a.name == "O"]
    before_atoms = len(parm.atoms)
    audit = salt.mutate_waters_to_ions(
        parm,
        sodium_oxygen_indices=[water_oxygen_indices[0]],
        chloride_oxygen_indices=[water_oxygen_indices[1]],
        na_template_prmtop=None,
        cl_template_prmtop=cl_template,
    )
    assert audit["ion_type_sources"] == {
        "Na+": "input_existing_ion_type",
        "Cl-": "template_lj_type_added_with_parmed_AddLJType",
    }
    assert len(parm.atoms) == before_atoms - 4
    assert audit["retained_atom_count"] == len(parm.atoms)
    assert audit["retained_atom_coordinate_max_abs_error_nm"] == pytest.approx(0.0)
    assert audit["solute_nonbonded_signature_unchanged"] is True
    assert audit["solute_bonded_exclusion_signature_unchanged"] is True
    assert audit["histidine_assignments_before_after"]["before"] == audit["histidine_assignments_before_after"]["after"]
    ion_indices = set(audit["new_ion_atom_indices"])
    assert ion_indices
    for term_list in (parm.bonds, parm.angles, parm.dihedrals):
        for term in term_list:
            atoms = [getattr(term, name) for name in ("atom1", "atom2", "atom3", "atom4") if hasattr(term, name)]
            assert not any(a.idx in ion_indices for a in atoms)
    counts = {name: sum(1 for r in parm.residues if r.name == name) for name in ["Na+", "Cl-", "WAT", "HIE", "CY1", "ZN1"]}
    assert counts == {"Na+": 2, "Cl-": 1, "WAT": 2, "HIE": 1, "CY1": 1, "ZN1": 1}


def test_parmed_mutation_requires_template_when_ion_type_absent():
    _require_parmed()
    parm = _fixture_parm(include_na=True, include_cl=False)
    water_oxygen_indices = [a.idx for a in parm.atoms if a.residue.name == "WAT" and a.name == "O"]
    with pytest.raises(ValueError, match="lacks Cl-"):
        salt.mutate_waters_to_ions(
            parm,
            sodium_oxygen_indices=[water_oxygen_indices[0]],
            chloride_oxygen_indices=[water_oxygen_indices[1]],
            na_template_prmtop=None,
            cl_template_prmtop=None,
        )


def test_default_paths_from_config_has_no_private_125_fallbacks():
    paths = salt.default_paths_from_config({})
    assert paths == {
        "prmtop": None,
        "inpcrd": None,
        "mapping": None,
        "prep": None,
        "na_template_prmtop": None,
        "cl_template_prmtop": None,
    }
    with pytest.raises(ValueError, match="missing required salt input path"):
        salt.require_input_paths(paths)


def test_default_paths_from_config_uses_config_without_template_requirement(tmp_path):
    files = {}
    for key in ("prmtop", "inpcrd", "mapping", "prep"):
        path = tmp_path / f"{key}.dat"
        path.write_text(key)
        files[key] = path
    config = {"atomistic_salt": {key: str(path) for key, path in files.items()}}
    paths = salt.default_paths_from_config(config)
    salt.require_input_paths(paths)
    assert paths["prmtop"] == files["prmtop"]
    assert paths["na_template_prmtop"] is None
    assert paths["cl_template_prmtop"] is None


def test_main_rejects_nonempty_output_dir_before_mutation(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    inputs = {}
    for key in ("prmtop", "inpcrd", "mapping", "prep"):
        path = tmp_path / f"{key}.dat"
        path.write_text(key)
        inputs[key] = str(path)
    config.write_text(json.dumps({"atomistic_salt": inputs}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing.txt").write_text("do not overwrite")

    def fail_if_called(**_kwargs):
        raise AssertionError("mutation should not be reached for nonempty output directory")

    monkeypatch.setattr(salt, "apply_salt_with_parmed", fail_if_called)
    with pytest.raises(FileExistsError, match="output directory is not empty"):
        salt.main(["--config", str(config), "--output-dir", str(out), "--offline"])


def test_main_allows_input_existing_ion_types_without_template_paths(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    inputs = {}
    for key in ("prmtop", "inpcrd", "mapping", "prep"):
        path = tmp_path / f"{key}.dat"
        path.write_text(key)
        inputs[key] = str(path)
    config.write_text(json.dumps({"atomistic_salt": inputs}))
    out = tmp_path / "empty_out"

    def fake_apply(**kwargs):
        assert kwargs["na_template_prmtop"] is None
        assert kwargs["cl_template_prmtop"] is None
        salt.write_json(kwargs["mapping_out"], {"ok": True})
        kwargs["output_prmtop"].write_text("prmtop")
        kwargs["output_inpcrd"].write_text("inpcrd")
        return {
            "status": "pass",
            "salt_policy": {"added_nacl_pairs": 2},
            "outputs": {},
        }

    monkeypatch.setattr(salt, "apply_salt_with_parmed", fake_apply)
    assert salt.main(["--config", str(config), "--output-dir", str(out), "--offline"]) == 0
    printed = capsys.readouterr().out
    assert '"status": "pass"' in printed
    assert (out / "salt_preparation_report.json").exists()


def test_main_accepts_required_paths_from_cli_with_empty_config(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{}")
    inputs = {}
    for key in ("prmtop", "inpcrd", "mapping", "prep"):
        path = tmp_path / f"cli_{key}.dat"
        path.write_text(key)
        inputs[key] = path
    out = tmp_path / "cli_out"

    def fake_apply(**kwargs):
        assert kwargs["prmtop"] == inputs["prmtop"]
        assert kwargs["inpcrd"] == inputs["inpcrd"]
        assert kwargs["mapping_in"] == inputs["mapping"]
        assert kwargs["prep"] == inputs["prep"]
        assert kwargs["na_template_prmtop"] is None
        assert kwargs["cl_template_prmtop"] is None
        kwargs["output_prmtop"].write_text("prmtop")
        kwargs["output_inpcrd"].write_text("inpcrd")
        salt.write_json(kwargs["mapping_out"], {"ok": True})
        return {"status": "pass", "salt_policy": {"added_nacl_pairs": 1}}

    monkeypatch.setattr(salt, "apply_salt_with_parmed", fake_apply)
    assert salt.main([
        "--config", str(config),
        "--output-dir", str(out),
        "--offline",
        "--prmtop", str(inputs["prmtop"]),
        "--inpcrd", str(inputs["inpcrd"]),
        "--mapping", str(inputs["mapping"]),
        "--prep", str(inputs["prep"]),
    ]) == 0
    assert (out / "salt_preparation_report.json").exists()
