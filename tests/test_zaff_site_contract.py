from pathlib import Path

import pytest

from scripts import zaff_site_contract as zaff


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "tests" / "fixtures" / "zaff_contract"


def test_retained_zaff_cy1_zn1_contract_and_tleap_prototype():
    report = zaff.validate_contract(SOURCE_DIR / "ZAFF.prep", SOURCE_DIR / "ZAFF.frcmod")

    assert report["status"] == "pass"
    assert report["actual_model_blocks"]["CY1"]["net_charge_e"] == pytest.approx(-0.63109)
    assert report["actual_model_blocks"]["ZN1"]["net_charge_e"] == pytest.approx(0.52437)
    assert report["site_net_charge_e"] == pytest.approx(-1.99999)
    assert report["required_parameters"]["bonds"]["ZN-S1"]["k_amber"] == pytest.approx(32.69)
    assert report["required_parameters"]["angles"]["CT-S1-ZN"]["theta0_deg"] == pytest.approx(101.733)
    assert report["required_parameters"]["angles"]["S1-ZN-S1"]["theta0_deg"] == pytest.approx(109.472)
    assert report["geometry_status"] == "geometry_not_qualified"
    assert report["md_direct_use"] is False
    assert report["amber_harmonic_convention"]["legacy_qm_curvature_factor_applied"] is False

    cy1_atoms = {row["name"]: row for row in report["actual_model_blocks"]["CY1"]["atoms"]}
    assert cy1_atoms["SG"]["type"] == "S1"
    assert "HG" not in cy1_atoms

    prototype = zaff.build_tleap_prototype(SOURCE_DIR / "ZAFF.prep", SOURCE_DIR / "ZAFF.frcmod")
    assert "p1 = sequence { ACE CY1 NME }" in prototype
    assert prototype.count("bond site.") == 4
    assert "saveamberparm site prototype.parm7 prototype.rst7" in prototype
    assert "geometry_not_qualified" in prototype
    assert _prototype_bond_residues(prototype) == [(2, 13), (5, 13), (8, 13), (11, 13)]


def test_wrong_cys_template_is_rejected(tmp_path):
    prep = tmp_path / "wrong.prep"
    prep.write_text(
        """\
    1    1    2
wrong

 CY4 INT 1
 CORR OMIT DU   BEG
0.00000
   1  DUMM  DU    M    0  -1  -2     0.000      .0        .0      .00000
   2  DUMM  DU    M    1   0  -1     1.449      .0        .0      .00000
   3  DUMM  DU    M    2   1   0     1.522   111.1        .0      .00000
   4  SG    S4    E    3   2   1     1.810   116.000   180.000  -0.61293
DONE
 ZN1 INT 1
 CORR OMIT DU   BEG
0.00000
   1  DUMM  DU    M    0  -1  -2     0.000      .0        .0      .00000
   4 ZN     ZN    M    1   0  -1     1.000    90.000   180.000   0.52437
DONE
""",
        encoding="utf-8",
    )
    report = zaff.validate_contract(prep, SOURCE_DIR / "ZAFF.frcmod")
    assert report["status"] == "fail"
    assert any("CY1 residue block is absent" in failure for failure in report["failures"])
    assert any("CY4" in failure and "template mixing" in failure for failure in report["failures"])


def test_wrong_charge_is_rejected(tmp_path):
    prep = tmp_path / "wrong_charge.prep"
    text = (SOURCE_DIR / "ZAFF.prep").read_text(encoding="utf-8")
    prep.write_text(text.replace("-0.43963", "-0.33963", 1), encoding="utf-8")

    report = zaff.validate_contract(prep, SOURCE_DIR / "ZAFF.frcmod")

    assert report["status"] == "fail"
    assert any("CY1 charge" in failure for failure in report["failures"])
    assert any("Four CY1 plus ZN1 charge" in failure for failure in report["failures"])


def test_missing_angle_is_rejected(tmp_path):
    frcmod = tmp_path / "missing_angle.frcmod"
    text = (SOURCE_DIR / "ZAFF.frcmod").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.strip().startswith("S1-ZN-S1")]
    frcmod.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = zaff.validate_contract(SOURCE_DIR / "ZAFF.prep", frcmod)

    assert report["status"] == "fail"
    assert any("S1-ZN-S1" in failure for failure in report["failures"])


def test_tleap_prototype_uses_post_combine_residue_numbering():
    prototype = zaff.build_tleap_prototype(SOURCE_DIR / "ZAFF.prep", SOURCE_DIR / "ZAFF.frcmod")

    assert [
        line for line in prototype.splitlines() if line.startswith(("p", "z = ", "site = "))
    ] == [
        "p1 = sequence { ACE CY1 NME }",
        "p2 = sequence { ACE CY1 NME }",
        "p3 = sequence { ACE CY1 NME }",
        "p4 = sequence { ACE CY1 NME }",
        "z = sequence { ZN1 }",
        "site = combine { p1 p2 p3 p4 z }",
    ]
    assert _prototype_bond_residues(prototype) == [(2, 13), (5, 13), (8, 13), (11, 13)]
    assert "site.1.SG" not in prototype
    assert "site.5.ZN" not in prototype


def _prototype_bond_residues(prototype: str) -> list[tuple[int, int]]:
    bonds = []
    for line in prototype.splitlines():
        if not line.startswith("bond site."):
            continue
        _, sulfur, zinc = line.split()
        sg_residue, sg_atom = sulfur.removeprefix("site.").split(".")
        zn_residue, zn_atom = zinc.removeprefix("site.").split(".")
        assert sg_atom == "SG"
        assert zn_atom == "ZN"
        bonds.append((int(sg_residue), int(zn_residue)))
    return bonds
