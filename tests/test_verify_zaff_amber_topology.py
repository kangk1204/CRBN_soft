from pathlib import Path
import sys
import types

import pytest

from scripts import verify_zaff_amber_topology as verify


def test_openmm_conversion_constants_match_amber_harmonic_contract():
    constants = verify.expected_openmm_constants()

    assert constants["ZN-S1"]["openmm_k_kj_mol_nm2"] == pytest.approx(27354.992)
    assert constants["ZN-S1"]["openmm_r0_nm"] == pytest.approx(0.2426)
    assert constants["CT-S1-ZN"]["openmm_k_kj_mol_rad2"] == pytest.approx(538.874096)
    assert constants["S1-ZN-S1"]["openmm_k_kj_mol_rad2"] == pytest.approx(298.980272)


def test_leap_warning_classifier_separates_overlap_from_parameter_warnings(tmp_path):
    log = tmp_path / "leap.log"
    log.write_text(
        """\
/env/bin/teLeap: Warning!
There is a bond of 6.018 angstroms between SG and ZN atoms:
-------  .R<CY1 2>.A<SG 8> and .R<ZN1 13>.A<ZN 1>
/env/bin/teLeap: Warning!
Could not find angle parameter: XX-YY-ZZ
Exiting LEaP: Errors = 0; Warnings = 2; Notes = 1.
""",
        encoding="utf-8",
    )

    report = verify.classify_leap_warnings(log)

    assert report["errors_total"] == 0
    assert report["warnings_total"] == 2
    assert report["categories"]["long_bond_from_overlapping_prototype_geometry"] == 1
    assert report["categories"]["parameter_warning"] == 1


def test_missing_leap_log_is_explicit():
    report = verify.classify_leap_warnings(Path("absent.log"))

    assert report["log_present"] is False
    assert report["warnings_total"] is None


def test_parameter_contract_detects_charge_type_angle_and_reversed_cb_s_zn(tmp_path, monkeypatch):
    install_fake_openmm(monkeypatch)
    prep = write_minimal_prep(tmp_path)
    system, topology, atom_types = fake_site()

    report = verify.parameter_contract(system, topology, prep_path=prep, atom_types=atom_types)

    assert report["status"] == "pass"
    assert len(report["bonded_terms"]["zn_s_bonds"]) == 4
    assert len(report["bonded_terms"]["s_zn_s_angles"]) == 6
    assert len(report["bonded_terms"]["cb_s_zn_angles"]) == 4
    assert report["charges"]["cy1_zn_site_charge_e"] == pytest.approx(-1.99999)

    bad_charge, topology, atom_types = fake_site(sg_charge_delta=0.1)
    report = verify.parameter_contract(bad_charge, topology, prep_path=prep, atom_types=atom_types)
    assert report["status"] == "fail"
    assert any("CY1 SG charge" in failure for failure in report["failures"])

    bad_type, topology, atom_types = fake_site(sg_type="S4")
    report = verify.parameter_contract(bad_type, topology, prep_path=prep, atom_types=atom_types)
    assert report["status"] == "fail"
    assert any("CY1 SG atom type S4" in failure for failure in report["failures"])

    missing_angle, topology, atom_types = fake_site(remove_cb_angle=True)
    report = verify.parameter_contract(missing_angle, topology, prep_path=prep, atom_types=atom_types)
    assert report["status"] == "fail"
    assert any("Expected exactly 4 CB-S-Zn angles" in failure for failure in report["failures"])


def test_parameter_contract_rejects_extra_zn_link(tmp_path, monkeypatch):
    install_fake_openmm(monkeypatch)
    prep = write_minimal_prep(tmp_path)
    system, topology, atom_types = fake_site(extra_zn_link=True)

    report = verify.parameter_contract(system, topology, prep_path=prep, atom_types=atom_types)

    assert report["status"] == "fail"
    assert any("Unexpected extra Zn covalent bond" in failure for failure in report["failures"])


class Quantity:
    def __init__(self, value):
        self.value = value

    def value_in_unit(self, unit):
        return self.value


class UnitValue:
    def __truediv__(self, other):
        return self

    def __pow__(self, power):
        return self


class FakeNonbondedForce:
    def __init__(self, charges, exceptions):
        self.charges = charges
        self.exceptions = exceptions

    def getParticleParameters(self, index):
        return Quantity(self.charges[index]), Quantity(0.1), Quantity(0.0)

    def getNumExceptions(self):
        return len(self.exceptions)

    def getExceptionParameters(self, index):
        a, b, charge_product, sigma, epsilon = self.exceptions[index]
        return a, b, Quantity(charge_product), Quantity(sigma), Quantity(epsilon)


class FakeBondForce:
    def __init__(self, bonds):
        self.bonds = bonds

    def getNumBonds(self):
        return len(self.bonds)

    def getBondParameters(self, index):
        a, b, length, k_value = self.bonds[index]
        return a, b, Quantity(length), Quantity(k_value)


class FakeAngleForce:
    def __init__(self, angles):
        self.angles = angles

    def getNumAngles(self):
        return len(self.angles)

    def getAngleParameters(self, index):
        a, b, c, theta, k_value = self.angles[index]
        return a, b, c, Quantity(theta), Quantity(k_value)


class FakeSystem:
    def __init__(self, particle_count, forces):
        self.particle_count = particle_count
        self.forces = forces

    def getNumParticles(self):
        return self.particle_count

    def getForces(self):
        return self.forces


class FakeResidue:
    def __init__(self, name, index):
        self.name = name
        self.index = index


class FakeAtom:
    def __init__(self, name, residue, index):
        self.name = name
        self.residue = residue
        self.index = index


class FakeTopology:
    def __init__(self, atoms, residues):
        self._atoms = atoms
        self._residues = residues

    def atoms(self):
        return iter(self._atoms)

    def getNumResidues(self):
        return len(self._residues)


def install_fake_openmm(monkeypatch):
    fake_unit = types.SimpleNamespace(
        elementary_charge=UnitValue(),
        nanometer=UnitValue(),
        kilojoules_per_mole=UnitValue(),
        radian=UnitValue(),
    )
    fake = types.SimpleNamespace(
        HarmonicAngleForce=FakeAngleForce,
        HarmonicBondForce=FakeBondForce,
        NonbondedForce=FakeNonbondedForce,
        unit=fake_unit,
    )
    monkeypatch.setitem(sys.modules, "openmm", fake)


def write_minimal_prep(tmp_path):
    prep = tmp_path / "ZAFF.prep"
    prep.write_text(
        """\
 CY1 INT 1
 CORR OMIT DU   BEG
   1 N     N     M    0   0   0     0.000   0.000   0.000  -0.46300
   2 HN    H     E    0   0   0     0.000   0.000   0.000   0.25200
   3 CA    CT    M    0   0   0     0.000   0.000   0.000   0.03500
   4 HA    H1    E    0   0   0     0.000   0.000   0.000   0.04800
   5 CB    CT    3    0   0   0     0.000   0.000   0.000  -0.54300
   6 HB3   H1    E    0   0   0     0.000   0.000   0.000   0.18377
   7 HB2   H1    E    0   0   0     0.000   0.000   0.000   0.18377
   8 SG    S1    E    0   0   0     0.000   0.000   0.000  -0.43963
   9 C     C     M    0   0   0     0.000   0.000   0.000   0.61600
  10 O     O     E    0   0   0     0.000   0.000   0.000  -0.50400
DONE
 ZN1 INT 1
 CORR OMIT DU   BEG
   1 ZN     ZN    M    0   0   0     0.000   0.000   0.000   0.52437
DONE
""",
        encoding="utf-8",
    )
    return prep


def fake_site(sg_charge_delta=0.0, sg_type="S1", remove_cb_angle=False, extra_zn_link=False):
    atoms = []
    residues = []
    charges = []
    atom_types = []
    cy1_charges = {
        "N": -0.46300,
        "HN": 0.25200,
        "CA": 0.03500,
        "HA": 0.04800,
        "CB": -0.54300,
        "HB3": 0.18377,
        "HB2": 0.18377,
        "SG": -0.43963 + sg_charge_delta,
        "C": 0.61600,
        "O": -0.50400,
    }
    cy1_types = {"N": "N", "HN": "H", "CA": "CT", "HA": "H1", "CB": "CT", "HB3": "H1", "HB2": "H1", "SG": sg_type, "C": "C", "O": "O"}
    sg_indices = []
    cb_indices = []
    for residue_index in range(4):
        residue = FakeResidue("CY1", residue_index)
        residues.append(residue)
        for atom_name in cy1_charges:
            atom = FakeAtom(atom_name, residue, len(atoms))
            atoms.append(atom)
            charges.append(cy1_charges[atom_name])
            atom_types.append(cy1_types[atom_name])
            if atom_name == "SG":
                sg_indices.append(atom.index)
            if atom_name == "CB":
                cb_indices.append(atom.index)
    zn_residue = FakeResidue("ZN1", 4)
    residues.append(zn_residue)
    zn_index = len(atoms)
    atoms.append(FakeAtom("ZN", zn_residue, zn_index))
    charges.append(0.52437)
    atom_types.append("ZN")

    constants = verify.expected_openmm_constants()
    bonds = [
        (sg, zn_index, constants["ZN-S1"]["openmm_r0_nm"], constants["ZN-S1"]["openmm_k_kj_mol_nm2"])
        for sg in sg_indices
    ]
    for cb, sg in zip(cb_indices, sg_indices):
        bonds.append((cb, sg, 0.181, 1.0))
    if extra_zn_link:
        bonds.append((zn_index, 0, constants["ZN-S1"]["openmm_r0_nm"], constants["ZN-S1"]["openmm_k_kj_mol_nm2"]))

    angles = []
    for i, first in enumerate(sg_indices):
        for second in sg_indices[i + 1 :]:
            angles.append(
                (
                    first,
                    zn_index,
                    second,
                    constants["S1-ZN-S1"]["openmm_theta0_rad"],
                    constants["S1-ZN-S1"]["openmm_k_kj_mol_rad2"],
                )
            )
    for n, (cb, sg) in enumerate(zip(cb_indices, sg_indices)):
        if remove_cb_angle and n == 0:
            continue
        atoms_order = (zn_index, sg, cb) if n == 1 else (cb, sg, zn_index)
        angles.append(
            (
                *atoms_order,
                constants["CT-S1-ZN"]["openmm_theta0_rad"],
                constants["CT-S1-ZN"]["openmm_k_kj_mol_rad2"],
            )
        )
    exceptions = []
    for sg in sg_indices:
        exceptions.append((zn_index, sg, 0.0, 0.1, 0.0))
    for i, first in enumerate(sg_indices):
        for second in sg_indices[i + 1 :]:
            exceptions.append((first, second, 0.0, 0.1, 0.0))
    for cb in cb_indices:
        exceptions.append((zn_index, cb, 0.0, 0.1, 0.0))
    system = FakeSystem(len(atoms), [FakeNonbondedForce(charges, exceptions), FakeBondForce(bonds), FakeAngleForce(angles)])
    return system, FakeTopology(atoms, residues), atom_types
