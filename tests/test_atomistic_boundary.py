"""Engine-level tests on tiny fixtures, not production CRBN validation."""
from contextlib import contextmanager
from itertools import combinations
import os
from pathlib import Path

import numpy as np
import pytest

mm = pytest.importorskip("openmm")
from openmm import unit
from scripts import directional_mechanics as dm
from scripts.atomistic_boundary import (
    add_core_gauge_and_probe, make_fixed_body, make_rigid_body,
)


XYZ = np.array([
    [-.30,-.20,-.10], [.40,-.10,.20], [.05,.45,-.25],
    [-.10,.05,.55], [.30,.20,.40], [-.30,.25,.15], [2.,0.,0.],
])
MASSES = np.array([12.,16.,14.,1.,12.,2.,20.])
BODY = list(range(6))


def fixture_system(xyz=XYZ, masses=MASSES, internal_constraint=True):
    system = mm.System()
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for mass in masses:
        system.addParticle(float(mass))
        nonbonded.addParticle(0, .2, 0)
    system.addForce(nonbonded)
    if internal_constraint:
        system.addConstraint(0, 1, float(np.linalg.norm(xyz[0]-xyz[1])))
    return system


@contextmanager
def engine(system, xyz, integrator=None):
    if integrator is None:
        integrator = mm.VerletIntegrator(.001)
    integrator.setConstraintTolerance(1e-9)
    platform = mm.Platform.getPlatformByName(os.environ.get("OPENMM_TEST_PLATFORM", "Reference"))
    properties = {"Precision": "double"} if "Precision" in platform.getPropertyNames() else {}
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(np.asarray(xyz)*unit.nanometer)
    context.computeVirtualSites()
    try:
        yield context, integrator
    finally:
        del context


def energy_force(context):
    state = context.getState(getEnergy=True, getForces=True)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))


def positions(context):
    return context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)


def test_rigid_mass_center_inertia_and_reconstruction():
    original = fixture_system()
    before = mm.XmlSerializer.serialize(original)
    result = make_rigid_body(original, XYZ, BODY)
    assert mm.XmlSerializer.serialize(original) == before
    assert result.system.getNumParticles() == len(XYZ)+4
    assert result.system.getNumConstraints() == 6
    anchors = result.metadata["anchor_indices"]
    masses = np.array([result.system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(len(XYZ)+4)])
    assert masses[BODY].sum() == 0
    assert np.all(masses[anchors] == MASSES[BODY].sum()/4)
    assert masses.sum() == MASSES.sum()
    original_center = np.average(XYZ[BODY], axis=0, weights=MASSES[BODY])
    center = np.average(result.positions_nm[anchors], axis=0, weights=masses[anchors])
    np.testing.assert_allclose(center, original_center, atol=1e-14)
    tensors = []
    for x, m in [(XYZ[BODY], MASSES[BODY]), (result.positions_nm[anchors], masses[anchors])]:
        r = x-np.average(x, axis=0, weights=m)
        tensors.append(sum(w*(np.dot(v,v)*np.eye(3)-np.outer(v,v)) for v,w in zip(r,m)))
    np.testing.assert_allclose(tensors[0], tensors[1], rtol=1e-12, atol=1e-13)
    jacobian = []
    for i,j in combinations(range(4), 2):
        row = np.zeros((4,3))
        vector = result.positions_nm[anchors[i]]-result.positions_nm[anchors[j]]
        row[i] = vector; row[j] = -vector
        jacobian.append(row.ravel())
    assert np.linalg.matrix_rank(jacobian) == 6
    with engine(result.system, result.positions_nm) as (context, _):
        np.testing.assert_allclose(positions(context)[:len(XYZ)], XYZ, atol=1e-12)
    assert result.metadata["body_dof"] == 6


def test_virtual_site_force_and_torque_transfer():
    original = fixture_system()
    force = mm.CustomExternalForce("-fx*x-fy*y-fz*z")
    for key in ["fx", "fy", "fz"]:
        force.addPerParticleParameter(key)
    applied = np.array([.7, -.3, .4])
    force.addParticle(2, applied)
    original.addForce(force)
    result = make_rigid_body(original, XYZ, BODY)
    with engine(result.system, result.positions_nm) as (context, _):
        _, forces = energy_force(context)
        anchors = result.metadata["anchor_indices"]
        center = np.asarray(result.metadata["center_of_mass_nm"])
        np.testing.assert_allclose(forces[anchors].sum(axis=0), applied, atol=1e-12)
        torque = np.cross(result.positions_nm[anchors]-center, forces[anchors]).sum(axis=0)
        np.testing.assert_allclose(torque, np.cross(XYZ[2]-center, applied), atol=1e-12)


def test_bonded_nonbonded_energy_and_generalized_forces_preserved():
    original = fixture_system()
    nb = original.getForce(0)
    for i in range(len(XYZ)):
        nb.setParticleParameters(i, (-1)**i*.1, .12, .02)
    bond = mm.HarmonicBondForce(); bond.addBond(0,1,.7,100.)
    angle = mm.HarmonicAngleForce(); angle.addAngle(0,2,3,1.,10.)
    torsion = mm.PeriodicTorsionForce(); torsion.addTorsion(0,1,2,3,3,.2,1.)
    rb = mm.RBTorsionForce(); rb.addTorsion(0,1,2,4,.1,.2,.3,.1,.05,.02)
    for force in [bond, angle, torsion, rb, mm.CMMotionRemover()]:
        original.addForce(force)
    result = make_rigid_body(original, XYZ, BODY)
    with engine(original, XYZ) as (context, _):
        original_energy, original_forces = energy_force(context)
    with engine(result.system, result.positions_nm) as (context, _):
        rigid_energy, rigid_forces = energy_force(context)
        assert rigid_energy == pytest.approx(original_energy, abs=1e-10)
        anchors = result.metadata["anchor_indices"]
        center = np.asarray(result.metadata["center_of_mass_nm"])
        net = original_forces[BODY].sum(axis=0)
        torque = np.cross(XYZ[BODY]-center, original_forces[BODY]).sum(axis=0)
        np.testing.assert_allclose(rigid_forces[anchors].sum(axis=0), net, atol=1e-10)
        np.testing.assert_allclose(np.cross(result.positions_nm[anchors]-center, rigid_forces[anchors]).sum(axis=0), torque, atol=1e-10)
        eps = 1e-6
        samples = []
        for sign in [-1,1]:
            x = result.positions_nm.copy(); x[anchors,0] += sign*eps
            context.setPositions(x); context.computeVirtualSites()
            samples.append(energy_force(context)[0])
        assert -(samples[1]-samples[0])/(2*eps) == pytest.approx(net[0], abs=1e-7)


@pytest.mark.parametrize("method", ["verlet", "langevin"])
def test_rigid_ten_ps_internal_distance_invariance(method):
    result = make_rigid_body(fixture_system(), XYZ, BODY)
    integrator = mm.VerletIntegrator(.001) if method == "verlet" else mm.LangevinMiddleIntegrator(300, 1, .001)
    if method == "langevin":
        integrator.setRandomNumberSeed(712)
    anchors = result.metadata["anchor_indices"]
    center = np.asarray(result.metadata["center_of_mass_nm"])
    velocity = np.zeros_like(result.positions_nm)
    velocity[anchors] = [.01,.02,-.01] + np.cross([.04,-.06,.03], result.positions_nm[anchors]-center)
    pair = list(combinations(BODY, 2))
    target = np.array([np.linalg.norm(XYZ[i]-XYZ[j]) for i,j in pair])
    anchor_pairs = list(combinations(anchors, 2))
    anchor_target = np.array([np.linalg.norm(result.positions_nm[i]-result.positions_nm[j]) for i,j in anchor_pairs])
    with engine(result.system, result.positions_nm, integrator) as (context, integ):
        context.setVelocities(velocity)
        context.applyVelocityConstraints(1e-9)
        for _ in range(10):
            integ.step(1000)
            xyz = positions(context)
            actual = np.array([np.linalg.norm(xyz[i]-xyz[j]) for i,j in pair])
            np.testing.assert_allclose(actual, target, rtol=2e-7, atol=1e-9)
            anchor_actual = np.array([np.linalg.norm(xyz[i]-xyz[j]) for i,j in anchor_pairs])
            np.testing.assert_allclose(anchor_actual, anchor_target, rtol=2e-7, atol=1e-9)
        assert np.linalg.norm(xyz[BODY]-XYZ[BODY]) > .001  # body did move


def test_fixed_body_positions_invariant_ten_ps():
    original = fixture_system()
    before = mm.XmlSerializer.serialize(original)
    result = make_fixed_body(original, BODY)
    assert result.system.getNumConstraints() == 0
    assert mm.XmlSerializer.serialize(original) == before
    integrator = mm.LangevinMiddleIntegrator(300, 1, .001)
    integrator.setRandomNumberSeed(713)
    with engine(result.system, XYZ, integrator) as (context, integ):
        context.setVelocitiesToTemperature(300, 714)
        integ.step(10000)
        final = positions(context)
        np.testing.assert_allclose(final[BODY], XYZ[BODY], atol=1e-14)
        assert np.linalg.norm(final[6]-XYZ[6]) > .001


def test_gauge_energy_force_finite_difference_and_rank_six():
    system = fixture_system(internal_constraint=False)
    ids = list(range(4))
    U = dm.internal_basis(XYZ[ids])
    q = U[:, 0]
    result = add_core_gauge_and_probe(system, ids, XYZ, q, 19., .7)
    B = np.asarray(result.metadata["rigid_basis"])
    xyz = XYZ.copy()
    delta = np.random.default_rng(40).normal(size=(4,3))*.02
    xyz[ids] += delta
    with engine(result.system, xyz) as (context, _):
        e, forces = energy_force(context)
        g = B.T@delta.ravel()
        assert e == pytest.approx(.5*19.*np.dot(g,g)-.7*np.dot(q,delta.ravel()), abs=1e-12)
        expected = (-19.*B@g+.7*q).reshape(-1,3)
        gpu = context.getPlatform().getName() in ("OpenCL", "CUDA", "HIP")
        quantum = result.metadata["gpu_force_quantum_kj_mol_nm"]
        # Same derived buffer bound as the 269-core test: six scaled CV
        # truncations, six final gauge truncations, and one probe truncation.
        # Reference retains its 1e-12 absolute criterion; no relative cushion.
        buffer_bound = (quantum*(7+19.*np.abs(g).sum()/result.metadata["gauge_cv_scale"])
                        if gpu else 0.)
        np.testing.assert_allclose(forces[ids], expected, rtol=0., atol=1e-12+buffer_bound)
        np.testing.assert_allclose(forces[4:], 0, atol=1e-12)
        # This potential is exactly quadratic, so a large centered step has
        # no truncation error. It avoids dividing the GPU's 2**-32 force-buffer
        # quantum by 1e-6 when checking the Hessian nullspace.
        eps = 1.
        hessian = np.empty((12,12))
        for j in range(12):
            samples = []
            for sign in [-1, 1]:
                moved = xyz.copy(); moved[j//3,j%3] += sign*eps
                context.setPositions(moved)
                samples.append(energy_force(context))
            assert -(samples[1][0]-samples[0][0])/(2*eps) == pytest.approx(forces.ravel()[j], abs=2e-8)
            hessian[:,j] = -(samples[1][1][ids].ravel()-samples[0][1][ids].ravel())/(2*eps)
        assert np.linalg.matrix_rank(hessian, tol=1e-6) == 6
        np.testing.assert_allclose(hessian@U, 0, atol=2e-8)
        context.setParameter(result.metadata["gauge_parameter"], 0.)
        context.setParameter(result.metadata["probe_parameter"], -.4)
        context.setPositions(xyz)
        _, f = energy_force(context)
        # With k=0 only the single final probe-buffer truncation remains.
        np.testing.assert_allclose(f[ids].ravel(), -.4*q, rtol=0.,
                                   atol=1e-12+(quantum if gpu else 0.))


def test_rotated_translated_gauge_energy_and_force_covariance():
    ids = list(range(4)); q = dm.internal_basis(XYZ[ids])[:,1]
    axis = np.array([1.,2.,3.]); axis /= np.linalg.norm(axis)
    cross = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3)+np.sin(.7)*cross+(1-np.cos(.7))*(cross@cross)
    shift = np.array([2.,-3.,1.])
    xyz = XYZ+np.random.default_rng(41).normal(size=XYZ.shape)*.02
    a = add_core_gauge_and_probe(fixture_system(internal_constraint=False), ids, XYZ, q, 11., .2)
    b = add_core_gauge_and_probe(fixture_system(internal_constraint=False), ids, XYZ@R.T+shift, q.reshape(-1,3)@R.T, 11., .2)
    with engine(a.system, xyz) as (c, _):
        e1, f1 = energy_force(c)
    with engine(b.system, xyz@R.T+shift) as (c, _):
        e2, f2 = energy_force(c)
    assert e1 == pytest.approx(e2, abs=1e-12)
    np.testing.assert_allclose(f2, f1@R.T, atol=1e-12)


def test_gauge_then_rigid_body_context_composes():
    xyz = np.vstack([XYZ[:4]-3, XYZ[:6]+3])
    system = fixture_system(xyz, np.ones(10)*12, internal_constraint=False)
    q = dm.internal_basis(xyz[:4])[:,0]
    gauge = add_core_gauge_and_probe(system, range(4), xyz, q, 20., .2)
    result = make_rigid_body(gauge.system, xyz, range(4,10))
    with engine(result.system, result.positions_nm) as (context, integ):
        assert np.isfinite(energy_force(context)[0])
        integ.step(10)
        assert np.isfinite(positions(context)).all()


def test_frozen_269_core_double_precision_at_gauge_1e4():
    """Exact frozen core/closure data at nm scale; no atomistic force field/MD."""
    root = Path(__file__).resolve().parents[1]
    if not (root/"data/crbn_ensemble.ens.npz").is_file() or not (root/"data/pca_diffvec.npz").is_file():
        pytest.skip("Acquire the frozen CRBN data bundle for the 269-core integration check")
    with np.load(root/"data/crbn_ensemble.ens.npz") as data:
        conformations = data["_confs"]
        labels = list(data["_labels"])
    with np.load(root/"data/pca_diffvec.npz") as data:
        mask = data["open_mask"].astype(bool)
    ref = conformations[labels.index("8CVP")]/10  # frozen Angstrom -> nm
    assert ref.shape == (269, 3)
    B = dm.rigid_basis(ref)
    direction = (conformations[mask].mean(0)-conformations[~mask].mean(0)).ravel()
    q = direction-B@(B.T@direction)
    q /= np.linalg.norm(q)
    internal = np.random.default_rng(269).normal(size=807)*.01
    internal -= B@(B.T@internal)
    delta = (internal+B@np.array([.012,-.008,.017,.003,-.011,.009])).reshape(-1,3)
    axis = np.array([1.,2.,3.]); axis /= np.linalg.norm(axis)
    cross = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    rotation = np.eye(3)+np.sin(.7)*cross+(1-np.cos(.7))*(cross@cross)
    measured_energies = []
    for R, shift in [(np.eye(3), np.zeros(3)), (rotation, np.array([20.,-30.,10.]))]:
        reference = ref@R.T+shift
        xyz = (ref+delta)@R.T+shift
        probe = (q.reshape(-1,3)@R.T).ravel()
        system = mm.System()
        for _ in range(269):
            system.addParticle(12.)
        result = add_core_gauge_and_probe(system, range(269), reference, probe, 1e4, 0.)
        gauge = result.system.getForce(0)
        for i in range(6):
            cv = gauge.getCollectiveVariable(i)
            assert type(cv) is mm.CustomCompoundBondForce
            assert cv.getNumParticlesPerBond() == 1 and cv.getNumBonds() == 269
        basis = np.asarray(result.metadata["rigid_basis"])
        displacement = (xyz-reference).ravel()
        g = basis.T@displacement
        # At most one inner CV truncation and one final truncation per gauge
        # term, plus one probe truncation. OpenMM CommonKernels.cpp:247-257,
        # 2476-2487. Scaling suppresses the inner error by scale=2**16.
        quantum = result.metadata["gpu_force_quantum_kj_mol_nm"]
        buffer_bound = quantum*(7+1e4*np.abs(g).sum()/result.metadata["gauge_cv_scale"])
        energies = []
        with engine(result.system, reference) as (context, _):
            e0, f0 = energy_force(context)
            assert e0 == pytest.approx(0., abs=1e-20)
            np.testing.assert_allclose(f0, 0., rtol=0., atol=1e-12)
            context.setPositions(xyz)
            for h in (-.5, -.25, 0., .25, .5):
                context.setParameter(result.metadata["probe_parameter"], h)
                energy, force = energy_force(context)
                expected = .5*1e4*np.dot(g,g)-h*np.dot(probe,displacement)
                assert energy == pytest.approx(expected, abs=1e-12)
                exact_force = (-1e4*basis@g+h*probe).reshape(-1,3)
                arithmetic_bound = 256*np.finfo(float).eps*max(1., np.max(np.abs(exact_force)))
                atol = arithmetic_bound
                if context.getPlatform().getName() in ("OpenCL", "CUDA", "HIP"):
                    atol += buffer_bound
                np.testing.assert_allclose(force, exact_force, rtol=0., atol=atol)
                energies.append(energy)
        measured_energies.append(energies)
    np.testing.assert_allclose(*measured_energies, rtol=0., atol=1e-12)


@pytest.mark.parametrize("kind", ["cross_constraint", "unknown_force", "unknown_compound_force", "active_barostat", "body_virtual_site"])
@pytest.mark.parametrize("mode", ["fixed", "rigid"])
def test_unsupported_input_rejected_without_mutation(kind, mode):
    system = fixture_system()
    if kind == "cross_constraint":
        system.addConstraint(0, 6, float(np.linalg.norm(XYZ[0]-XYZ[6])))
    elif kind == "unknown_force":
        system.addForce(mm.CustomBondForce("0"))
    elif kind == "unknown_compound_force":
        system.addForce(mm.CustomCompoundBondForce(1, "0"))
    elif kind == "active_barostat":
        system.addForce(mm.MonteCarloBarostat(1, 300, 25))
    else:
        system.setParticleMass(5, 0)
        system.setVirtualSite(5, mm.TwoParticleAverageSite(0, 1, .5, .5))
    before = mm.XmlSerializer.serialize(system)
    with pytest.raises(ValueError):
        make_fixed_body(system, BODY) if mode == "fixed" else make_rigid_body(system, XYZ, BODY)
    assert mm.XmlSerializer.serialize(system) == before


@pytest.mark.parametrize("bad", ["nonunit", "rigid", "nan", "duplicate", "bad_index"])
def test_bad_gauge_input_rejected_without_mutation(bad):
    system = fixture_system(internal_constraint=False)
    ids = list(range(4)); q = dm.internal_basis(XYZ[ids])[:,0]
    if bad == "nonunit": q = 2*q
    elif bad == "rigid": q = dm.rigid_basis(XYZ[ids])[:,0]
    elif bad == "nan": q[0] = np.nan
    elif bad == "duplicate": ids[-1] = 0
    else: ids[-1] = 99
    before = mm.XmlSerializer.serialize(system)
    with pytest.raises(ValueError):
        add_core_gauge_and_probe(system, ids, XYZ, q, 20., .2)
    assert mm.XmlSerializer.serialize(system) == before


def test_planar_body_rejected_without_mutation():
    system = fixture_system(); xyz = XYZ.copy(); xyz[BODY,2] = 0
    before = mm.XmlSerializer.serialize(system)
    with pytest.raises(ValueError, match="nonplanar"):
        make_rigid_body(system, xyz, BODY)
    assert mm.XmlSerializer.serialize(system) == before
