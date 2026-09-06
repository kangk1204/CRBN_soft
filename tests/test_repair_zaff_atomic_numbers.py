import pytest

from scripts import prepare_atomistic_amber as amber
from scripts import repair_zaff_atomic_numbers as repair


def i8(values):
    return "".join(f"{value:8d}" for value in values) + "\n"


def e16(values):
    return "".join(f"{value:16.8E}" for value in values) + "\n"


def a4(values):
    return "".join(f"{value:<4}"[:4] for value in values) + "\n"


def section(name, fmt, payload):
    return f"%FLAG {name}\n%FORMAT({fmt})\n{payload}"


def minimal_prmtop(atomic_numbers=None, masses=None, atom_types=None, atom_names=None):
    atom_names = atom_names or ["SG", "CB", "SG", "CB", "SG", "CB", "SG", "CB", "ZN"]
    atom_types = atom_types or ["S1", "CT", "S1", "CT", "S1", "CT", "S1", "CT", "ZN"]
    masses = masses or [32.06, 12.01, 32.06, 12.01, 32.06, 12.01, 32.06, 12.01, 65.4]
    atomic_numbers = atomic_numbers or [0, 6, 0, 6, 0, 6, 0, 6, -1]
    text = "%VERSION VERSION_STAMP = TEST\n"
    text += section("POINTERS", "10I8", i8([9, 5, 0, 0, 0, 0, 0, 0, 0, 0]))
    text += section("ATOM_NAME", "20a4", a4(atom_names))
    text += section("CHARGE", "5E16.8", e16([0.0] * 9))
    text += section("MASS", "5E16.8", e16(masses))
    text += section("ATOMIC_NUMBER", "10I8", i8(atomic_numbers))
    text += section("RESIDUE_LABEL", "20a4", a4(["CY1", "CY1", "CY1", "CY1", "ZN1"]))
    text += section("RESIDUE_POINTER", "10I8", i8([1, 3, 5, 7, 9]))
    text += section("BONDS_INC_HYDROGEN", "10I8", "")
    text += section("BONDS_WITHOUT_HYDROGEN", "10I8", i8([0, 24, 1, 6, 24, 1, 12, 24, 1, 18, 24, 1]))
    text += section("AMBER_ATOM_TYPE", "20a4", a4(atom_types))
    text += section("TREE_CHAIN_CLASSIFICATION", "20a4", a4(["M"] * 9))
    return text


def write_prmtop(tmp_path, text):
    path = tmp_path / "input.prmtop"
    path.write_text(text, encoding="ascii")
    return path


def outside_atomic_number(text):
    sections = repair.parse_sections(text)
    sec = sections["ATOMIC_NUMBER"]
    return text[: sec.data_start] + text[sec.data_end :]


def test_repairs_only_exact_zaff_atomic_numbers_and_preserves_other_bytes(tmp_path):
    input_path = write_prmtop(tmp_path, minimal_prmtop())
    output_path = tmp_path / "fixed.prmtop"

    report = repair.repair_file(input_path, output_path)
    before = input_path.read_text(encoding="ascii")
    after = output_path.read_text(encoding="ascii")
    atomic_numbers = repair.parse_i8(after, repair.parse_sections(after)["ATOMIC_NUMBER"])

    assert report["status"] == "complete"
    assert report["change_count"] == 5
    assert atomic_numbers == [16, 6, 16, 6, 16, 6, 16, 6, 30]
    assert len(before) == len(after)
    assert outside_atomic_number(before) == outside_atomic_number(after)
    assert report["target_counts"] == {"ZN1/ZN/typeZN": 1, "CY1/SG/typeS1": 4}
    assert len(report["zn_sg_bond_pairs"]) == 4


def test_rejects_unexpected_unknown_atomic_number(tmp_path):
    text = minimal_prmtop(atomic_numbers=[0, 0, 0, 6, 0, 6, 0, 6, -1])
    with pytest.raises(ValueError, match="unexpected_nonpositive_atomic_numbers"):
        repair.repair_file(write_prmtop(tmp_path, text), tmp_path / "fixed.prmtop")


def test_rejects_wrong_prior_positive_target_number(tmp_path):
    text = minimal_prmtop(atomic_numbers=[8, 6, 0, 6, 0, 6, 0, 6, -1])
    with pytest.raises(ValueError, match="wrong_prior_positive_atomic_numbers"):
        repair.repair_file(write_prmtop(tmp_path, text), tmp_path / "fixed.prmtop")


def test_rejects_wrong_mass_and_missing_zn_sg_bonds(tmp_path):
    text = minimal_prmtop(masses=[12.0, 12.01, 32.06, 12.01, 32.06, 12.01, 32.06, 12.01, 65.4])
    text = text.replace(section("BONDS_WITHOUT_HYDROGEN", "10I8", i8([0, 24, 1, 6, 24, 1, 12, 24, 1, 18, 24, 1])), section("BONDS_WITHOUT_HYDROGEN", "10I8", i8([0, 24, 1])))
    with pytest.raises(ValueError) as exc:
        repair.repair_file(write_prmtop(tmp_path, text), tmp_path / "fixed.prmtop")
    message = str(exc.value)
    assert "mass_failures" in message
    assert "zn_sg_bond_failure" in message


def test_builder_emits_zaff_add_atom_types_before_loading_prep(tmp_path):
    pdb = tmp_path / "input.pdb"
    prep = tmp_path / "ZAFF.prep"
    frcmod = tmp_path / "ZAFF.frcmod"
    pdb.write_text("END\n", encoding="utf-8")
    prep.write_text("", encoding="utf-8")
    frcmod.write_text("", encoding="utf-8")

    text = amber.build_tleap_input(pdb, prep, frcmod, "dry", [], solvated=False)

    assert 'addAtomTypes {' in text
    assert '{ "ZN" "Zn" "sp3" }' in text
    assert '{ "S1" "S" "sp3" }' in text
    assert text.index("addAtomTypes") < text.index("loadAmberPrep")


def test_repair_preserves_input_and_existing_output(tmp_path):
    source = write_prmtop(tmp_path, minimal_prmtop())
    original = source.read_bytes()
    with pytest.raises(ValueError, match="new output copy"):
        repair.repair_file(source, source)
    assert source.read_bytes() == original
    output = tmp_path / "earlier.prmtop"
    output.write_text("preserve earlier attempt")
    with pytest.raises(ValueError, match="new output copy"):
        repair.repair_file(source, output)
    assert output.read_text() == "preserve earlier attempt"
