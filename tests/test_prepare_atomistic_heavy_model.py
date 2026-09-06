import json
from types import SimpleNamespace

import pytest

from scripts import prepare_atomistic_heavy_model as prep


class FakeResidue:
    def __init__(self, residue_id):
        self.id = str(residue_id)


class FakeChain:
    def __init__(self, chain_id, residue_ids):
        self.id = chain_id
        self._residue_ids = residue_ids

    def residues(self):
        return [FakeResidue(r) for r in self._residue_ids]


def test_filter_missing_residues_keeps_only_approved_internal_gaps_and_drops_terminals():
    chains = [FakeChain("A", [1, 2, 3, 4, 5, 545, 551, 552]), FakeChain("B", [64, 65, 341, 358, 359, 428])]
    missing = {
        (0, 0): ["MET"],
        (0, 6): ["GLY", "GLY", "GLY", "GLY", "GLY"],  # DDB1 546-550 before 551
        (1, 0): ["MET"] * 63,
        (1, 3): ["GLY"] * 16,  # CRBN 342-357 before 358
        (1, 6): ["LEU"] * 14,
        (1, 2): ["ALA"],  # unapproved gap 66
    }
    kept, decisions = prep.filter_missing_residues(missing, chains, prep.ALLOWED_GAPS)
    assert set(kept) == {(0, 6), (1, 3)}
    assert missing == kept
    by_key = {tuple(d["key"]): d for d in decisions}
    assert by_key[(1, 0)]["terminal"] is True
    assert by_key[(1, 0)]["action"] == "drop_terminal_or_unapproved_gap"
    assert by_key[(1, 3)]["residue_numbers"] == list(range(342, 358))
    assert by_key[(1, 3)]["action"] == "keep_for_internal_repair"
    assert by_key[(0, 6)]["residue_numbers"] == [546, 547, 548, 549, 550]


def test_clear_missing_terminals_removes_terminal_atoms_before_add_missing_atoms():
    fixer = SimpleNamespace(missingTerminals={"a": ["OXT"], "b": ["H1", "H2"]})
    assert prep.clear_missing_terminals(fixer) == 3
    assert fixer.missingTerminals == {}


def test_coordinate_preservation_reports_moved_or_missing_atoms():
    before = {
        ("B", 323, "SG"): prep.AtomRecord("B", 323, "SG", "CYS", 1, 2, 3, "S", "ATOM"),
        ("B", 326, "SG"): prep.AtomRecord("B", 326, "SG", "CYS", 1, 2, 3, "S", "ATOM"),
    }
    after = {("B", 323, "SG"): prep.AtomRecord("B", 323, "SG", "CYS", 1, 2, 3.01, "S", "ATOM")}
    result = prep.compare_preserved_coordinates(before, after, set(before), tol=0.001)
    assert result["preserved"] is False
    assert result["moved_atom_count"] == 1
    assert result["missing_after"] == [["B", 326, "SG"]]


def test_metal_requirements_assert_single_zn_and_four_cys_sg():
    atoms = {("B", r, "SG"): prep.AtomRecord("B", r, "SG", "CYS", 0, 0, 0, "S", "ATOM") for r in prep.PROTECTED_ZN_CYS}
    atoms[("B", 501, "ZN")] = prep.AtomRecord("B", 501, "ZN", "ZN", 0, 0, 0, "ZN", "HETATM")
    assert prep.metal_requirements(atoms)["single_zn_and_four_cys_sg"] is True
    del atoms[("B", 326, "SG")]
    assert prep.metal_requirements(atoms)["single_zn_and_four_cys_sg"] is False


def test_metal_chain_relabel_tracks_identity_but_rejects_coordinate_change():
    src = prep.AtomRecord("B", 501, "ZN", "ZN", 1, 2, 3, "ZN", "HETATM")
    dst = prep.AtomRecord("C", 501, "ZN", "ZN", 1, 2, 3, "ZN", "HETATM")
    before, changes = prep.map_single_metal_identity({src.key: src}, {dst.key: dst})
    assert changes[0]["source_key"] == ["B", 501, "ZN"]
    assert prep.compare_preserved_coordinates(before, {dst.key: dst}, set(before))["preserved"]
    moved = prep.replace(dst, z=3.1)
    assert not prep.compare_preserved_coordinates(before, {moved.key: moved}, set(before))["preserved"]


def test_config_loader_uses_root_protocol_fields(tmp_path):
    core = tmp_path / "core.csv"
    core.write_text("index,author_resnum\n0,77\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"references": ["8CVP"], "cif_dir": "data/_cif_cache", "core_residue_file": str(core), "core_position_count": 1, "models": ["isolated"]}), encoding="utf-8")
    loaded = prep.load_config(config)
    assert loaded["references"] == ["8CVP"]
    assert loaded["core_position_count"] == 1
    assert "models" in loaded["ignored_config_keys"]


def test_missing_pdbfixer_dependency_is_explicit():
    try:
        prep.require_dependencies()
    except RuntimeError as exc:
        assert "PDBFixer/OpenMM are required" in str(exc)
    else:
        pytest.skip("PDBFixer/OpenMM installed; dependency failure path not applicable")


def test_add_missing_atoms_seed_call_supports_seed_when_api_exposes_it():
    class FakeFixer:
        def __init__(self):
            self.seed = None

        def addMissingAtoms(self, seed=None):
            self.seed = seed

    fixer = FakeFixer()
    result = prep.call_add_missing_atoms(fixer, 123)
    assert result["seed_supported"] is True
    assert fixer.seed == 123


@pytest.fixture(scope="module")
def complete_heavy_fixture():
    """Exact construct inventory with artificial coordinates, never an MD model."""
    lines = ["data_synthetic", "loop_", "_pdbx_poly_seq_scheme.pdb_strand_id",
             "_pdbx_poly_seq_scheme.pdb_seq_num", "_pdbx_poly_seq_scheme.auth_seq_num",
             "_pdbx_poly_seq_scheme.mon_id"]
    atoms = []
    for chain, (start, end) in prep.EXPECTED_RESIDUE_RANGES.items():
        for residue in range(start, end+1):
            name = "GLY"
            if chain == "B" and residue in prep.PROTECTED_ZN_CYS:
                name = "CYS"
            elif (chain, residue) == ("B", 342):
                name = "LEU"
            elif (chain, residue) == ("A", 546):
                name = "ALA"
            elif (chain, residue) == ("B", 100):
                name = "LYS"
            author = "?" if residue in prep.ALLOWED_GAPS[chain] else str(residue)
            lines.append(f"{chain} {residue} {author} {name}")
            for atom in sorted(prep.STANDARD_HEAVY_ATOMS[name]):
                atoms.append(prep.AtomRecord(chain, residue, atom, name, 0, 1, 2, atom[0], "ATOM"))
    lines.append("#")
    atoms.append(prep.AtomRecord("C", 501, "ZN", "ZN", 0, 0, 0, "ZN", "HETATM"))
    return "\n".join(lines), atoms


def test_exact_construct_postcondition_passes_without_oxt(complete_heavy_fixture):
    source, atoms = complete_heavy_fixture
    result = prep.heavy_model_postconditions(source, atoms, [77])
    assert result["repair_complete"]
    assert result["expected_protein_residue_count"] == 1505
    assert result["observed_protein_residue_counts"] == {"A": 1140, "B": 365}
    assert not result["OXT_required"]
    assert len(result["requested_loop_identity_checks"]) == 21
    extra_oxt = prep.AtomRecord("B", 428, "OXT", "GLY", 0, 0, 0, "O", "ATOM")
    assert prep.heavy_model_postconditions(source, [*atoms, extra_oxt], [77])["repair_complete"]


@pytest.mark.parametrize("missing", ["requested_loop", "side_chain_atom"])
def test_preservation_of_observed_ca_does_not_imply_repair_complete(complete_heavy_fixture, missing):
    source, atoms = complete_heavy_fixture
    if missing == "requested_loop":
        after = [atom for atom in atoms if (atom.chain, atom.resseq) != ("B", 342)]
    else:
        after = [atom for atom in atoms if atom.key != ("B", 100, "NZ")]
    before = {atom.key: atom for atom in after}
    preservation = prep.compare_preserved_coordinates(before, before, set(before))
    assert preservation["preserved"]
    result = prep.heavy_model_postconditions(source, after, [77])
    assert not result["repair_complete"]
    assert result["core_ca_missing"] == []
    assert prep.preparation_status(preservation, {"single_zn_and_four_cys_sg": True}, result) == "blocked_after_repair_validation"
    if missing == "requested_loop":
        assert result["missing_residues"] == [["B", 342]]
    else:
        assert result["missing_heavy_atoms"] == [{"residue": ["B", 100], "name": "LYS", "missing": ["NZ"]}]


def test_requested_loop_identity_comes_from_cif_sequence(complete_heavy_fixture):
    source, atoms = complete_heavy_fixture
    altered = [prep.replace(atom, comp="ILE") if (atom.chain, atom.resseq) == ("B", 342) else atom for atom in atoms]
    result = prep.heavy_model_postconditions(source, altered, [77])
    assert not result["repair_complete"]
    assert result["residue_identity_mismatches"] == [{"residue": ["B", 342], "expected": "LEU", "observed": ["ILE"]}]


def test_duplicate_output_atoms_and_extra_construct_residues_cannot_pass(complete_heavy_fixture):
    source, atoms = complete_heavy_fixture
    result = prep.heavy_model_postconditions(source, [*atoms, atoms[0]], [77])
    assert not result["repair_complete"]
    assert result["duplicate_atom_keys"] == [list(atoms[0].key)]
    extra = prep.AtomRecord("B", 63, "CA", "GLY", 0, 0, 0, "C", "ATOM")
    assert prep.heavy_model_postconditions(source, [*atoms, extra], [77])["unexpected_residues"] == [["B", 63]]


def test_unsupported_source_sequence_and_duplicates_are_rejected(complete_heavy_fixture):
    source, _ = complete_heavy_fixture
    for altered in (source.replace("B 342 ? LEU", "B 342 ? UNK"), source.replace("B 342 ? LEU", "B 342 ? LEU\nB 342 ? LEU"), source.replace("B 342 ? LEU\n", "")):
        with pytest.raises(ValueError):
            prep.expected_heavy_sequence(altered)


@pytest.mark.parametrize("case", ["model", "duplicate", "altloc", "nan"])
def test_atom_parser_rejects_silent_model_or_identity_selection(case):
    headers = ["group_PDB", "type_symbol", "auth_atom_id", "auth_comp_id", "auth_asym_id", "auth_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num", "label_alt_id"]
    row = ["ATOM", "C", "CA", "GLY", "B", "77", "0", "0", "0", "1", "."]
    if case == "model":
        row[9] = "2"
    elif case == "altloc":
        row[10] = "A"
    elif case == "nan":
        row[6] = "nan"
    rows = [" ".join(row)] * (2 if case == "duplicate" else 1)
    source = "\n".join(["data_synthetic", "loop_", *("_atom_site."+key for key in headers), *rows, "#"])
    with pytest.raises(ValueError):
        prep.parse_input_atoms(source)


def test_dependency_version_fields_are_explicit():
    result = prep.dependency_versions()
    assert set(result) == {"openmm", "pdbfixer"}
    assert all(value is None or isinstance(value, str) for value in result.values())
