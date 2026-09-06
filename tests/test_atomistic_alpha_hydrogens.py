"""Coordinate and real Amber/ParmEd I/O tests; no Context, minimization or MD."""
import hashlib

import numpy as np
import pytest

pmd = pytest.importorskip("parmed")
from scripts.atomistic_alpha_hydrogens import repair_alpha_hydrogens, run


def fixture():
    structure = pmd.Structure()
    atoms = [("N", 7, 14.01), ("CA", 6, 12.01), ("C", 6, 12.01),
             ("CB", 6, 12.01), ("HA", 1, 1.008), ("O", 8, 16.0), ("HB", 1, 1.008)]
    for i, (name, number, mass) in enumerate(atoms):
        atom_type = pmd.topologyobjects.AtomType(name, i+1, mass, number)
        atom_type.set_lj_params(0.1, 1.5)
        atom = pmd.Atom(name=name, type=name, atomic_number=number, mass=mass, charge=0)
        atom.atom_type = atom_type
        structure.add_atom(atom, "ALA", 1)
    for i, j, req in [(1,0,1.45),(1,2,1.52),(1,3,1.53),(1,4,1.09),(2,5,1.23),(3,6,1.09)]:
        bond_type = pmd.BondType(340,req)
        structure.bond_types.append(bond_type)
        structure.bonds.append(pmd.Bond(structure.atoms[i],structure.atoms[j],type=bond_type))
    xyz = np.zeros((7,3))
    xyz[0] = np.array([-1,-1,-1])*.145/np.sqrt(3)
    xyz[2] = np.array([1,-1,1])*.152/np.sqrt(3)
    xyz[3] = np.array([-1,1,1])*.153/np.sqrt(3)
    xyz[4] = xyz[0]/np.linalg.norm(xyz[0])*.109
    xyz[5] = xyz[2]+[.1,.1,0]
    xyz[6] = xyz[3]+[.1,.1,0]
    return structure, xyz


def test_actual_bond_parameter_and_all_other_coordinates_unchanged():
    structure, xyz = fixture()
    original = xyz.copy()
    structure.bonds[3].type.req = 1.17
    result = repair_alpha_hydrogens(structure, xyz)
    np.testing.assert_array_equal(xyz, original)
    np.testing.assert_array_equal(result.positions_nm[[0,1,2,3,5,6]], original[[0,1,2,3,5,6]])
    assert np.linalg.norm(result.positions_nm[4]-result.positions_nm[1]) == pytest.approx(.117, abs=1e-14)
    row = result.metadata["rows"][0]
    assert row["reference_CA_signed_volume_nm3"] > 1e-4
    assert row["post_HA_signed_volume_nm3"] < 0
    np.testing.assert_allclose(list(row["post_HA_angles_degrees"].values()),np.degrees(np.arccos(-1/3)),atol=1e-12)
    assert result.metadata["geometry_qualified"] is False


def test_rotation_translation_equivariance():
    structure, xyz = fixture()
    rotation, _ = np.linalg.qr(np.random.default_rng(19).normal(size=(3,3)))
    if np.linalg.det(rotation)<0:
        rotation[:,0] *= -1
    offset = np.array([17.,-3.,12.])
    original = repair_alpha_hydrogens(structure,xyz)
    transformed = repair_alpha_hydrogens(structure,xyz@rotation+offset)
    np.testing.assert_allclose(transformed.positions_nm,original.positions_nm@rotation+offset,rtol=0,atol=5e-15)


@pytest.mark.parametrize("kind",["coincident","planar","near_planar","negative","missing","duplicate","wrong_element"])
def test_invalid_reference_is_rejected_before_modifying_inputs(kind):
    structure, xyz = fixture()
    if kind=="coincident":
        xyz[0]=xyz[1]
    elif kind in {"planar","near_planar"}:
        normal = np.cross(xyz[0]-xyz[1],xyz[2]-xyz[1])
        normal /= np.linalg.norm(normal)
        xyz[3] -= np.dot(xyz[3],normal)*normal
        if kind=="near_planar":
            xyz[3] += normal*1e-5
    elif kind=="negative":
        xyz[[0,2]]=xyz[[2,0]]
    elif kind=="missing":
        structure.atoms[4].name="HX"
    elif kind=="duplicate":
        structure.atoms[5].name="HA"
    else:
        structure.atoms[4].atomic_number=6
    before=xyz.copy()
    with pytest.raises(ValueError):
        repair_alpha_hydrogens(structure,xyz)
    np.testing.assert_array_equal(xyz,before)


def test_distorted_positive_heavy_geometry_is_reported_unchanged():
    structure, xyz = fixture()
    xyz[0]=[.145,0,0]
    xyz[2]=.152*np.array([np.cos(np.radians(163.6)),np.sin(np.radians(163.6)),0])
    xyz[3]=[.01,.02,.153]
    result=repair_alpha_hydrogens(structure,xyz)
    assert result.metadata["rows"][0]["unchanged_heavy_angles_degrees"]["N_CA_C"] == pytest.approx(163.6)
    np.testing.assert_array_equal(result.positions_nm[[0,1,2,3]],xyz[[0,1,2,3]])
    assert result.metadata["geometry_qualified"] is False


def test_glycine_and_nonprotein_coordinates_remain_unchanged():
    structure, xyz = fixture()
    for name in ["N","CA","C","HA2","HA3"]:
        structure.add_atom(pmd.Atom(name=name,atomic_number=1 if name.startswith("H") else 6),"GLY",2)
    structure.add_atom(pmd.Atom(name="O",atomic_number=8),"WAT",3)
    xyz=np.vstack([xyz,np.arange(18).reshape(6,3)*.1+2])
    result=repair_alpha_hydrogens(structure,xyz)
    np.testing.assert_array_equal(result.positions_nm[7:],xyz[7:])
    assert result.metadata["skipped_glycine_count"]==1


def test_actual_amber_restart_roundtrip_and_no_overwrite(tmp_path):
    pytest.importorskip("openmm")
    structure, xyz = fixture()
    structure.coordinates=xyz*10
    structure.box=[30,30,30,90,90,90]
    parm=pmd.amber.AmberParm.from_structure(structure)
    prmtop, inpcrd=tmp_path/"input.prmtop", tmp_path/"input.inpcrd"
    parm.save(str(prmtop)); parm.save(str(inpcrd))
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [prmtop,inpcrd]}
    output=tmp_path/"alpha_repaired"
    report=run(prmtop,inpcrd,output)
    assert report["repair"]["alpha_hydrogen_count"]==1
    assert report["roundtrip"]["status"]=="pass"
    assert report["inputs_unchanged"] is True
    assert report["MD_performed"] is report["minimization_performed"] is False
    assert (output/"alpha_hydrogen_positions.npy").exists()
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    with pytest.raises(FileExistsError):
        run(prmtop,inpcrd,output)
