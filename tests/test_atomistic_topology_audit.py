"""Fail-closed screens for the optional atomistic workflow."""
import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app
from scripts.atomistic_topology_audit import inspect_system


def fixture(protonated=False, displacement=0.0, charge=2.0):
    topology = app.Topology()
    chain = topology.addChain("B")
    system = openmm.System()
    nb = openmm.NonbondedForce()
    points = []
    zinc_residue = topology.addResidue("ZN", chain)
    topology.addAtom("ZN", app.element.zinc, zinc_residue)
    system.addParticle(65.38)
    nb.addParticle(charge, .2, .1)
    points.append([displacement, 0.0, 0.0])
    directions = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]) / np.sqrt(3)
    for i, direction in enumerate(directions):
        residue = topology.addResidue("CYS" if protonated else "CYM", chain, str(i+1))
        topology.addAtom("SG", app.element.sulfur, residue)
        system.addParticle(32.06)
        nb.addParticle(-.5, .3, .2)
        points.append(.23 * direction)
        if protonated:
            topology.addAtom("HG", app.element.hydrogen, residue)
            system.addParticle(1.008)
            nb.addParticle(0.0, .1, .0)
            points.append(.24 * direction)
    system.addForce(nb)
    return system, topology, np.array(points)


def test_geometry_is_not_parameter_validation():
    report = inspect_system(*fixture())
    assert report["zinc_sites"][0]["sg4_geometry_screen_pass"]
    assert abs(report["total_charge_e"]) < 1e-10
    assert report["production_ready"] is False
    assert any("validation record" in x for x in report["production_blockers"])


def test_protonated_sulfur_and_nonintegral_charge_are_reported():
    report = inspect_system(*fixture(protonated=True, charge=.842))
    assert any("retains HG" in x for x in report["production_blockers"])
    assert any("not integral" in x for x in report["production_blockers"])


def test_displaced_zinc_fails_geometry():
    report = inspect_system(*fixture(displacement=1.0))
    assert not report["zinc_sites"][0]["sg4_geometry_screen_pass"]
    assert any("four nearby" in x for x in report["production_blockers"])


def test_bad_coordinate_count_and_nan_rejected():
    system, top, xyz = fixture()
    with pytest.raises(ValueError, match="counts"):
        inspect_system(system, top, xyz[:-1])
    xyz[1,0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        inspect_system(system, top, xyz)
