"""Tiny PME engine integration and input gates; no CRBN production MD."""
import csv
from dataclasses import FrozenInstanceError
from itertools import combinations
import json
import os

import numpy as np
import pytest

mm = pytest.importorskip("openmm")
from openmm import app, unit
from scripts import directional_mechanics as dm
from scripts import run_atomistic_technical_pilot as pilot

TEST_PLATFORM = os.environ.get("OPENMM_TEST_PLATFORM", "Reference")


def tiny_fixture(isolated=False, zinc=False):
    topology = app.Topology()
    chain = topology.addChain("A")
    core = np.array([[-.30,-.20,-.10],[.40,-.10,.20],[.05,.45,-.25],[-.10,.05,.55]])+1.
    xyz = list(core)
    masses = [12.]*4
    for _i in range(4):
        residue = topology.addResidue("ALA", chain)
        topology.addAtom("CA", app.element.carbon, residue)
    body = []
    if not isolated:
        chain = topology.addChain("B")
        residue = topology.addResidue("ALA", chain)
        body = list(range(4,10))
        body_xyz = np.array([[-.30,-.20,-.10],[.40,-.10,.20],[.05,.45,-.25],[-.10,.05,.55],[.30,.20,.40],[-.30,.25,.15]])+3.3
        xyz.extend(body_xyz)
        masses.extend([12.,16.,14.,1.,12.,2.])
        for name, element in [("CA", app.element.carbon), ("O", app.element.oxygen), ("N", app.element.nitrogen),
                              ("H", app.element.hydrogen), ("C", app.element.carbon), ("CB", app.element.carbon)]:
            topology.addAtom(name, element, residue)
    water_start = len(xyz)
    residue = topology.addResidue("HOH", topology.addChain("W"))
    for name, element in [("O", app.element.oxygen), ("H1", app.element.hydrogen), ("H2", app.element.hydrogen)]:
        topology.addAtom(name, element, residue)
    theta = np.deg2rad(104.52)
    water = np.array([[0,0,0],[.09572,0,0],[.09572*np.cos(theta),.09572*np.sin(theta),0]])+[4.7,1.,1.]
    xyz.extend(water); masses.extend([15.999,1.008,1.008])
    zn_indices = None
    if zinc:
        zn_index = len(xyz)
        zn_res = topology.addResidue("ZN", topology.addChain("Z"))
        topology.addAtom("ZN", app.element.zinc, zn_res)
        center = np.array([1.,4.,4.]); xyz.append(center); masses.append(65.38)
        signs = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
        sulfur_chain = topology.addChain("S")
        for point in center+.23/np.sqrt(3)*signs:
            residue = topology.addResidue("CYS", sulfur_chain)
            topology.addAtom("SG", app.element.sulfur, residue)
            xyz.append(point); masses.append(32.06)
        zn_indices = (zn_index, list(range(zn_index+1, zn_index+5)))
    xyz = np.asarray(xyz)
    system = mm.System()
    box = np.eye(3)*6.
    system.setDefaultPeriodicBoxVectors(*box)
    topology.setPeriodicBoxVectors(tuple(mm.Vec3(*r) for r in box)*unit.nanometer)
    nb = mm.NonbondedForce(); nb.setNonbondedMethod(mm.NonbondedForce.PME); nb.setCutoffDistance(1.)
    for i, mass in enumerate(masses):
        system.addParticle(mass)
        water_index = i-water_start
        charge = (-.834,.417,.417)[water_index] if 0 <= water_index < 3 else 0.
        nb.addParticle(charge, .3, 0.)
    for i, j in [(0,1),(0,2),(1,2)]:
        a, b = water_start+i, water_start+j
        system.addConstraint(a,b,float(np.linalg.norm(xyz[a]-xyz[b])))
        nb.addException(a,b,0.,.1,0.)
    if body:
        system.addConstraint(body[0],body[1],float(np.linalg.norm(xyz[body[0]]-xyz[body[1]])))
    spring = mm.HarmonicBondForce()
    for i, j in [(0,1),(1,2),(2,3)]:
        spring.addBond(i,j,float(np.linalg.norm(core[i]-core[j]))*1.01,1000.)
    system.addForce(nb); system.addForce(spring); system.addForce(mm.CMMotionRemover())
    q = dm.internal_basis(core)[:,0]
    mapping = {"schema_version": "1.0", "core_indices": list(range(4)),
               "reference_nm": core.tolist(), "q": q.tolist(), "ddb1_atom_indices": body}
    if zn_indices:
        mapping.update(zn_atom_index=zn_indices[0], zn_sg_indices=zn_indices[1])
    model = "isolated" if isolated else "flexible"
    mapping = pilot.validate_mapping(mapping, topology, xyz, model, expected_core_count=4)
    return topology, system, xyz, mapping


@pytest.mark.parametrize("model", ["flexible", "fixed", "rigid", "isolated"])
def test_real_pme_langevin_technical_loop_and_original_atom_outputs(tmp_path, model):
    _, system, xyz, mapping = tiny_fixture(isolated=model == "isolated")
    original = mm.XmlSerializer.serialize(system)
    settings = pilot.resolve_settings({}, {"steps": 10, "benchmark_steps": 5, "report_interval_steps": 5,
                                           "minimization_max_iterations": 10, "max_wall_seconds": 60.})
    result = pilot._run_engine(system, xyz, mapping, settings, tmp_path/model, model=model,
                               platform_name=TEST_PLATFORM, synthetic_fixture=True, qualification={"chemical_review": {"status": "pending"}},
                               provenance={"role": "synthetic_engine_fixture"})
    assert result["status"] == "technical_completed", result.get("reason")
    assert result["technical_accepted"] and not result["response_converged"] and not result["production_ready"]
    assert result["chemical_review"]["status"] == "pending"
    assert result["completed_steps"] == 10
    assert result["platform"] == TEST_PLATFORM
    if TEST_PLATFORM in ("OpenCL", "CUDA", "HIP"):
        assert result["platform_properties"]["Precision"] == "double"
    assert result["benchmark"]["measured_steps"] == 5 and result["benchmark"]["steps_per_second"] > 0
    assert mm.XmlSerializer.serialize(system) == original
    expected_dof = {"flexible": 35, "fixed": 18, "rigid": 24, "isolated": 18}
    assert result["kinetic_dof"] == expected_dof[model]
    with np.load(tmp_path/model/"technical_pilot.npz") as output:
        assert output["last_positions_nm"].shape == xyz.shape
        assert output["last_anchor_positions_nm"].shape == ((4,3) if model == "rigid" else (0,3))
        assert output["core_displacement_nm"].shape == (3,4,3)
        np.testing.assert_allclose(output["time_ps"], [0.,.005,.01], atol=1e-14)
        np.testing.assert_array_equal(output["q_ambient"], mapping["q"])
        np.testing.assert_array_equal(output["reference_nm"], mapping["reference_nm"])
        np.testing.assert_allclose(output["last_positions_nm"][:4]-mapping["reference_nm"], output["core_displacement_nm"][-1])
        if model == "fixed":
            np.testing.assert_allclose(output["last_positions_nm"][mapping["ddb1_atom_indices"]], xyz[mapping["ddb1_atom_indices"]], atol=1e-14)
    rows = list(csv.DictReader((tmp_path/model/"diagnostics.csv").open()))
    assert all(np.isfinite(float(row["kinetic_temperature_K"])) for row in rows)
    assert float(rows[-1]["kinetic_temperature_K"]) > 0
    for row in rows:
        assert float(row["kinetic_temperature_K"]) == pytest.approx(2*float(row["kinetic_kj_mol"])/(expected_dof[model]*pilot.GAS_CONSTANT))


def test_long_rigid_anchor_edges_use_unwrapped_constraint_geometry(tmp_path):
    _, system, xyz, mapping = tiny_fixture()
    body = mapping["ddb1_atom_indices"]
    xyz[body] = (xyz[body]-3.3)*10.+3.3
    settings = pilot.Settings(steps=100, benchmark_steps=50, report_interval_steps=25,
                               minimization_max_iterations=5, max_wall_seconds=60.)
    prepared, positions, _, body_meta, _ = pilot._prepare_system(system, xyz, mapping, "rigid", settings)
    anchor_indices = body_meta["anchor_indices"]
    pairs = np.asarray(list(combinations(range(4), 2)))
    anchor_edges = positions[anchor_indices][pairs[:,0]]-positions[anchor_indices][pairs[:,1]]
    lengths = np.linalg.norm(anchor_edges, axis=1)
    box = np.asarray([v.value_in_unit(unit.nanometer) for v in prepared.getDefaultPeriodicBoxVectors()])
    assert np.max(lengths) > np.min(np.diag(box))/2
    wrapped_lengths = np.linalg.norm(pilot._minimum_image(anchor_edges, box), axis=1)
    assert np.max(np.abs(wrapped_lengths-lengths)/lengths) > .1
    result = pilot._run_engine(system, xyz, mapping, settings, tmp_path/"long_body", model="rigid",
                               platform_name=TEST_PLATFORM, synthetic_fixture=True,
                               qualification={"chemical_review": {"status": "pending"}}, provenance={})
    assert result["status"] == "technical_completed", result.get("reason")
    rows = list(csv.DictReader((tmp_path/"long_body"/"diagnostics.csv").open()))
    assert max(float(row["max_relative_constraint_error"]) for row in rows) <= 10*settings.constraint_tolerance
    with np.load(tmp_path/"long_body"/"technical_pilot.npz") as output:
        final_anchors = output["last_anchor_positions_nm"]
        final_lengths = np.linalg.norm(final_anchors[pairs[:,0]]-final_anchors[pairs[:,1]], axis=1)
        np.testing.assert_allclose(final_lengths, lengths, rtol=settings.constraint_tolerance, atol=0.)
        # Physical body atoms remain one coherent unwrapped rigid geometry.
        body_pairs = np.asarray(list(combinations(range(len(body)), 2)))
        before = xyz[body]
        after = output["last_positions_nm"][body]
        before_distances = np.linalg.norm(before[body_pairs[:,0]]-before[body_pairs[:,1]], axis=1)
        after_distances = np.linalg.norm(after[body_pairs[:,0]]-after[body_pairs[:,1]], axis=1)
        np.testing.assert_allclose(after_distances, before_distances, rtol=0., atol=1e-11)


def qualified_files(tmp_path):
    paths = {name: tmp_path/name for name in ("prmtop", "inpcrd", "mapping")}
    for name, path in paths.items():
        path.write_text(name)
    qualification = {"schema_version": "1.0", "status": "pass",
                     "gates": {name: "pass" for name in ("metal", "mapping", "geometry")},
                     "chemical_review": {"status": pilot.CHEMICAL_REVIEW_PASS},
                     "input_sha256": {name: pilot._sha256(path) for name, path in paths.items()}}
    qpath = tmp_path/"qualification.json"
    qpath.write_text(json.dumps(qualification))
    return paths, qpath, qualification


@pytest.mark.parametrize("failure", ["absent", "metal", "geometry", "hash"])
def test_input_qualification_is_required_before_amber_or_engine(tmp_path, failure):
    paths, qpath, qualification = qualified_files(tmp_path)
    if failure == "absent":
        qpath.unlink()
    elif failure == "hash":
        paths["mapping"].write_text("changed input")
    else:
        qualification["gates"][failure] = "fail"
        qpath.write_text(json.dumps(qualification))
    with pytest.raises((ValueError, FileNotFoundError)) as error:
        pilot.run(paths["prmtop"], paths["inpcrd"], paths["mapping"], tmp_path/"unused_config",
                  qpath, tmp_path/"output")
    assert "Amber" not in str(error.value)
    assert not (tmp_path/"output").exists()


def test_mapping_rejects_implicit_alignment_nonflat_probe_and_complex_as_isolated():
    topology, _, xyz, mapping = tiny_fixture()
    for change, model, pattern in [({"reference_nm": (np.asarray(mapping["reference_nm"])+.1).tolist()}, "flexible", "reference frame"),
                                    ({"q": np.asarray(mapping["q"]).reshape(4,3).tolist()}, "flexible", "flat ambient"),
                                    ({}, "isolated", "DDB1-removed"),
                                    ({"ddb1_atom_indices": mapping["ddb1_atom_indices"][:-1]}, "flexible", "including H")]:
        with pytest.raises(ValueError, match=pattern):
            pilot.validate_mapping({**mapping, **change}, topology, xyz, model, expected_core_count=4)
    with pytest.raises(ValueError, match="269"):
        pilot.validate_mapping(mapping, topology, xyz, "flexible")


def test_zero_available_wall_budget_is_not_technical_pass(tmp_path):
    _, system, xyz, mapping = tiny_fixture()
    settings = pilot.Settings(steps=10, max_wall_seconds=.1)
    result = pilot._run_engine(system, xyz, mapping, settings, tmp_path/"limited", model="flexible",
                               platform_name=TEST_PLATFORM, synthetic_fixture=True, qualification={"chemical_review": {"status": "pending"}},
                               provenance={}, started_at=pilot.time.monotonic()-1.)
    assert result["status"] == "budget_limited"
    assert not result["technical_accepted"] and result["completed_steps"] == 0
    assert (tmp_path/"limited"/"technical_pilot.json").is_file()


def test_measured_benchmark_survives_projected_budget_stop(tmp_path, monkeypatch):
    _, system, xyz, mapping = tiny_fixture()
    tick = iter(np.arange(0., 100., .001))
    monkeypatch.setattr(pilot.time, "monotonic", lambda: float(next(tick)))
    settings = pilot.Settings(steps=100000, benchmark_steps=5, report_interval_steps=5,
                               minimization_max_iterations=5, max_wall_seconds=1.)
    result = pilot._run_engine(system, xyz, mapping, settings, tmp_path/"benchmark", model="flexible",
                               platform_name=TEST_PLATFORM, synthetic_fixture=True, qualification={"chemical_review": {"status": "pending"}}, provenance={})
    assert result["status"] == "budget_limited" and result["completed_steps"] == 5
    assert result["benchmark"]["measured_steps"] == 5
    assert result["benchmark"]["steps_per_second"] > 0
    assert json.loads((tmp_path/"benchmark"/"benchmark.json").read_text()) == result["benchmark"]


def test_qualified_displaced_start_keeps_frozen_reference_and_probe(tmp_path):
    topology, system, xyz, mapping = tiny_fixture()
    initial = xyz.copy(); initial[:4,0] += .01
    mapping["initial_core_nm"] = initial[:4].tolist()
    mapping = pilot.validate_mapping(mapping, topology, initial, "flexible", expected_core_count=4)
    settings = pilot.Settings(steps=2, benchmark_steps=2, report_interval_steps=1,
                               minimization_max_iterations=2, max_wall_seconds=60.)
    result = pilot._run_engine(system, initial, mapping, settings, tmp_path/"displaced", model="flexible",
                               platform_name=TEST_PLATFORM, synthetic_fixture=True, qualification={"chemical_review": {"status": "pending"}}, provenance={})
    assert result["status"] == "technical_completed", result.get("reason")
    assert result["initial_vs_reference_rms_displacement_nm"] == pytest.approx(.01)
    with np.load(tmp_path/"displaced"/"technical_pilot.npz") as output:
        np.testing.assert_array_equal(output["reference_nm"], mapping["reference_nm"])
        np.testing.assert_array_equal(output["q_ambient"], mapping["q"])
        np.testing.assert_array_equal(output["initial_core_nm"], mapping["initial_core_nm"])
        np.testing.assert_allclose(output["initial_vs_reference_displacement_nm"], np.tile([.01,0,0], (4,1)), atol=1e-15)
    for invalid in [initial[:3].tolist(), (initial[:4]+.01).tolist(), np.full((4,3), np.nan).tolist()]:
        with pytest.raises(ValueError, match="initial_core_nm"):
            pilot.validate_mapping({**mapping, "initial_core_nm": invalid}, topology, initial, "flexible", expected_core_count=4)


def test_zinc_sg_screen_is_reported_and_failure_blocks_technical_acceptance(tmp_path):
    _, system, xyz, mapping = tiny_fixture(zinc=True)
    settings = pilot.Settings(steps=2, benchmark_steps=2, report_interval_steps=1,
                               minimization_max_iterations=5, max_wall_seconds=60.)
    mapping["zn_sg_distance_bounds_nm"] = [.18, .20]  # fixture distances are .23 nm
    result = pilot._run_engine(system, xyz, mapping, settings, tmp_path/"zinc", model="flexible",
                               platform_name=TEST_PLATFORM, synthetic_fixture=True, qualification={"chemical_review": {"status": "pending"}}, provenance={})
    assert result["status"] == "failed" and not result["technical_accepted"]
    assert "Zn-SG" in result["reason"]
    rows = list(csv.DictReader((tmp_path/"zinc"/"diagnostics.csv").open()))
    assert float(rows[0]["zn_sg_1_distance_nm"]) == pytest.approx(.23)


def test_settings_reject_hmr_and_unbounded_or_changed_integrator():
    for config in [{"hydrogen_mass_repartitioning": True},
                   {"timestep_fs_initial": 2.},
                   {"technical_pilot": {"minimization_max_iterations": 0}},
                   {"technical_pilot": {"max_step_batch": 1000}}]:
        with pytest.raises(ValueError):
            pilot.resolve_settings(config)


@pytest.mark.parametrize("status", ["pending", "fail", "pass", None])
def test_cli_requires_specific_chemical_preparation_pass_before_parsing(tmp_path, status):
    paths, qpath, qualification = qualified_files(tmp_path)
    qualification["chemical_review"]["status"] = status
    qpath.write_text(json.dumps(qualification))
    with pytest.raises(ValueError, match="chemical_review.status"):
        pilot.run(paths["prmtop"], paths["inpcrd"], paths["mapping"], tmp_path/"no_config",
                  qpath, tmp_path/"output")
    assert not (tmp_path/"output").exists()


def chemical_fixture(omit_peptide=False, omit_ca_cb=False, omit_ha=False):
    """Four harmonic toy residues with complete CA/CB identities and one peptide bond."""
    topology = app.Topology()
    core = np.array([[-.30,-.20,-.10],[.40,-.10,.20],[.05,.45,-.25],[-.10,.05,.55]])+1.
    xyz, indices = [], []
    for i, ca in enumerate(core):
        chain = topology.addChain(str(i))
        cap = None
        if i == 0:
            residue = topology.addResidue("ACE", chain)
            cap = topology.addAtom("C", app.element.carbon, residue)
            xyz.append(ca+[.017, 0, 0])
        resname = ("ALA", "THR", "ILE", "ALA")[i]
        residue = topology.addResidue(resname, chain)
        offsets = {"N": [.15,0,0], "CA": [0,0,0], "C": [0,.15,0], "O": [0,.25,0], "CB": [0,0,.13]}
        if not (omit_ha and i == 0):
            offsets["HA"] = [0,0,-.109]
        if resname in {"THR", "ILE"}:
            offsets["OG1" if resname == "THR" else "CG1"] = [.13,0,.13]
            offsets["CG2"] = [0,-.13,.13]
        if resname == "ILE":
            offsets["CD1"] = [.24,0,.18]
        names = {}
        for name, offset in offsets.items():
            element = app.element.nitrogen if name == "N" else (app.element.oxygen if name.startswith("O") else (app.element.hydrogen if name == "HA" else app.element.carbon))
            atom = topology.addAtom(name, element, residue)
            names[name] = atom
            xyz.append(ca+offset)
        indices.append(names["CA"].index)
        for first, second in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")):
            if omit_ca_cb and i == 0 and (first, second) == ("CA", "CB"):
                continue
            topology.addBond(names[first], names[second])
        if "HA" in names:
            topology.addBond(names["CA"], names["HA"])
        if resname in {"THR", "ILE"}:
            topology.addBond(names["CB"], names["OG1" if resname == "THR" else "CG1"])
            topology.addBond(names["CB"], names["CG2"])
        if resname == "ILE":
            topology.addBond(names["CG1"], names["CD1"])
        if cap is not None and not omit_peptide:
            topology.addBond(cap, names["N"])
    xyz = np.asarray(xyz)
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*np.eye(3)*6.)
    force = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", 5000.)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    for atom in topology.atoms():
        system.addParticle(atom.element.mass)
        force.addParticle(atom.index, xyz[atom.index])
    system.addForce(force)
    mapping = {"schema_version": "1.0", "core_indices": indices, "reference_nm": core.tolist(),
               "q": dm.internal_basis(core)[:,0].tolist(), "ddb1_atom_indices": []}
    mapping = pilot.validate_mapping(mapping, topology, xyz, "isolated", expected_core_count=4)
    return topology, system, xyz, mapping


def chemical_engine(tmp_path, *, steps=1):
    topology, system, xyz, mapping = chemical_fixture()
    settings = pilot.Settings(steps=steps, benchmark_steps=1, report_interval_steps=1,
                               minimization_max_iterations=2, max_wall_seconds=60.)
    return pilot._run_engine(system, xyz, mapping, settings, tmp_path/"chemical", model="isolated",
                              platform_name=TEST_PLATFORM, qualification={"chemical_review": {"status": pilot.CHEMICAL_REVIEW_PASS}},
                              provenance={"role": "synthetic_harmonic_chemical_fixture"},
                              chemical_geometry=pilot.derive_chemical_geometry(topology))


def test_complete_chemical_topology_is_immutable_and_mapping_cannot_hide_centers():
    topology, _, xyz, mapping = chemical_fixture()
    mapping["protein_ca_indices"] = mapping["core_indices"][:1]
    mapping["chemical_geometry"] = {"alpha_centers": [], "beta_centers": [], "peptide_pairs": []}
    definition = pilot.derive_chemical_geometry(topology)
    with pytest.raises(FrozenInstanceError):
        definition.alpha_centers = ()
    result = pilot.chemical_geometry_screen(xyz, definition)
    assert result["pass"]
    assert (result["alpha_center_count"], result["beta_center_count"], result["peptide_bond_count"]) == (4, 2, 1)
    assert result["alpha_hydrogen_center_count"] == 4
    assert result["alpha_hydrogen_maximum_signed_volume_nm3"] < -1e-4
    assert result["peptide_minimum_distance_nm"] == pytest.approx(.133)
    assert isinstance(definition.alpha_centers, tuple) and isinstance(definition.alpha_centers[0], tuple)


@pytest.mark.parametrize("failure", ["alpha", "beta", "planar_beta", "alpha_hydrogen", "planar_alpha_hydrogen", "peptide"])
def test_chemical_geometry_detects_inversion_nearplanarity_and_long_peptide(failure):
    topology, _, xyz, _ = chemical_fixture()
    geometry = pilot.derive_chemical_geometry(topology)
    if failure == "peptide":
        xyz[geometry.peptide_pairs[0][0],0] -= .1
    else:
        if "alpha_hydrogen" in failure:
            center = geometry.alpha_hydrogen_centers[0]
        else:
            center = geometry.alpha_centers[0] if failure == "alpha" else geometry.beta_centers[0]
        first, origin, second, branch = center
        normal = np.cross(xyz[first]-xyz[origin], xyz[second]-xyz[origin])
        normal /= np.linalg.norm(normal)
        vector = xyz[branch]-xyz[origin]
        if failure.startswith("planar_"):
            xyz[branch] = xyz[origin]+.001*vector
        else:
            xyz[branch] -= 2*np.dot(vector, normal)*normal
    result = pilot.chemical_geometry_screen(xyz, geometry)
    assert not result["pass"]
    assert result[failure.removeprefix("planar_")+"_failure_count"] > 0


@pytest.mark.parametrize("failure", ["missing_peptide", "missing_ca_cb", "missing_ha", "unknown_residue"])
def test_derivation_rejects_missing_bonds_and_unknown_ca_residue(failure):
    topology, _, _, _ = chemical_fixture(omit_peptide=failure == "missing_peptide", omit_ca_cb=failure == "missing_ca_cb", omit_ha=failure == "missing_ha")
    if failure == "unknown_residue":
        list(topology.residues())[1].name = "UNK"
    with pytest.raises(ValueError):
        pilot.derive_chemical_geometry(topology)


def test_healthy_toy_records_chemical_screens_at_every_stage(tmp_path):
    result = chemical_engine(tmp_path)
    assert result["technical_accepted"] and result["status"] == "technical_completed", result.get("reason")
    assert len(result["chemical_geometry_sha256"]) == 64
    assert not result["chemical_geometry_failures"]
    rows = list(csv.DictReader((tmp_path/"chemical/diagnostics.csv").open()))
    assert [r["stage"] for r in rows] == ["initial", "post_minimization", "technical", "final"]
    assert all(r["chemical_pass"] == "True" for r in rows)
    assert all(int(r["chemical_alpha_center_count"]) == 4 for r in rows)


@pytest.mark.parametrize("center", ["alpha_centers", "alpha_hydrogen_centers"])
def test_post_minimization_inversion_stops_before_dynamics(tmp_path, monkeypatch, center):
    minimize = pilot.mm.LocalEnergyMinimizer.minimize
    topology, _, _, _ = chemical_fixture()
    first, origin, second, branch = getattr(pilot.derive_chemical_geometry(topology), center)[0]

    def invert_after_minimization(context, *args):
        minimize(context, *args)
        xyz = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        normal = np.cross(xyz[first]-xyz[origin], xyz[second]-xyz[origin])
        normal /= np.linalg.norm(normal)
        xyz[branch] -= 2*np.dot(xyz[branch]-xyz[origin], normal)*normal
        context.setPositions(xyz)

    monkeypatch.setattr(pilot.mm.LocalEnergyMinimizer, "minimize", invert_after_minimization)
    result = chemical_engine(tmp_path)
    assert result["status"] == "failed" and not result["technical_accepted"]
    assert result["completed_steps"] == 0
    assert result["chemical_geometry_failures"][0]["stage"] == "post_minimization"
    assert result["chemical_geometry_failures"][0][center.removesuffix("_centers")+"_failure_count"] == 1


@pytest.mark.parametrize("center", ["beta_centers", "alpha_hydrogen_centers"])
def test_final_chemical_failure_revokes_previously_completed_technical_acceptance(tmp_path, monkeypatch, center):
    screen = pilot.chemical_geometry_screen
    calls = []

    def invert_final_sample(xyz, geometry):
        calls.append(1)
        if len(calls) == 4:  # initial, post-minimization, last MD report, final
            xyz = xyz.copy()
            first, origin, second, branch = getattr(geometry, center)[0]
            normal = np.cross(xyz[first]-xyz[origin], xyz[second]-xyz[origin])
            normal /= np.linalg.norm(normal)
            xyz[branch] -= 2*np.dot(xyz[branch]-xyz[origin], normal)*normal
        return screen(xyz, geometry)

    monkeypatch.setattr(pilot, "chemical_geometry_screen", invert_final_sample)
    result = chemical_engine(tmp_path)
    assert result["completed_steps"] == 1
    assert result["status"] == "failed" and not result["technical_accepted"]
    assert result["chemical_geometry_failures"][-1]["stage"] == "final"
    assert result["chemical_geometry_failures"][-1][center.removesuffix("_centers")+"_failure_count"] == 1
    stored = json.loads((tmp_path/"chemical/technical_pilot.json").read_text())
    assert not stored["technical_accepted"] and stored["status"] == "failed"


def test_private_engine_cannot_implicitly_skip_chemical_geometry(tmp_path):
    _, system, xyz, mapping = tiny_fixture()
    with pytest.raises(ValueError, match="synthetic_fixture=True"):
        pilot._run_engine(system, xyz, mapping, pilot.Settings(), tmp_path/"missing", model="flexible",
                          platform_name=TEST_PLATFORM, qualification={"chemical_review": {"status": "pending"}}, provenance={})


def test_direct_chemical_engine_still_requires_pass_unless_explicitly_synthetic(tmp_path):
    topology, system, xyz, mapping = chemical_fixture()
    with pytest.raises(ValueError, match="chemical_review.status"):
        pilot._run_engine(system, xyz, mapping, pilot.Settings(), tmp_path/"unqualified", model="isolated",
                          platform_name=TEST_PLATFORM, qualification={"chemical_review": {"status": "pending"}}, provenance={},
                          chemical_geometry=pilot.derive_chemical_geometry(topology))
