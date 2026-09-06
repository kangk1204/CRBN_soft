import numpy as np
import pytest

from scripts import relax_atomistic_hydrogens as relax_h


class FakeElement:
    def __init__(self, symbol):
        self.symbol = symbol


class FakeChain:
    def __init__(self, chain_id="A"):
        self.id = chain_id


class FakeResidue:
    def __init__(self, name, index, chain_id="A", residue_id=None):
        self.name = name
        self.index = index
        self.id = str(residue_id if residue_id is not None else index + 1)
        self.chain = FakeChain(chain_id)
        self._atoms = []

    def atoms(self):
        return iter(self._atoms)


class FakeAtom:
    def __init__(self, name, residue, index, element):
        self.name = name
        self.residue = residue
        self.index = index
        self.element = FakeElement(element) if element is not None else None
        residue._atoms.append(self)


class FakeTopology:
    def __init__(self, residues):
        self._residues = residues
        self._atoms = [atom for residue in residues for atom in residue._atoms]

    def atoms(self):
        return iter(self._atoms)

    def residues(self):
        return iter(self._residues)


def make_residue(name, atom_specs, chain="A", res_id=1):
    residue = FakeResidue(name, 0, chain, res_id)
    for index, (atom_name, element) in enumerate(atom_specs):
        FakeAtom(atom_name, residue, index, element)
    return residue


def test_mask_solvent_distinction_fixes_only_solute_heavy_and_rejects_unknown():
    protein = make_residue("ALA", [("N", "N"), ("CA", "C"), ("HA", "H")], res_id=1)
    water = FakeResidue("WAT", 1, "", 2)
    FakeAtom("O", water, 3, "O")
    FakeAtom("H1", water, 4, "H")
    ion = FakeResidue("Na+", 2, "", 3)
    FakeAtom("Na+", ion, 5, "Na")
    topology = FakeTopology([protein, water, ion])

    masks = relax_h.classify_mobile_and_fixed_atoms(topology)

    assert masks["fixed_solute_heavy"] == [0, 1]
    assert masks["hydrogens"] == [2]
    assert masks["solvent_or_ions"] == [3, 4, 5]
    assert masks["mobile"] == [2, 3, 4, 5]

    bad = make_residue("ALA", [("XX", None)], res_id=4)
    with pytest.raises(ValueError, match="unknown solute elements"):
        relax_h.classify_mobile_and_fixed_atoms(FakeTopology([bad]))


def test_alpha_ha_stereochemistry_passes_opposite_side_and_fails_same_side():
    residue = make_residue("VAL", [("N", "N"), ("CA", "C"), ("C", "C"), ("CB", "C"), ("HA", "H")])
    topology = FakeTopology([residue])
    initial = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.2],
        ]
    )
    assert relax_h.alpha_ha_stereochemistry(topology, initial, initial)["status"] == "pass"

    bad = initial.copy()
    bad[4] = [0.0, 0.0, 0.2]
    report = relax_h.alpha_ha_stereochemistry(topology, initial, bad)
    assert report["status"] == "fail"
    assert "not opposite CB" in report["failures"][0]


def test_alpha_ha_stereochemistry_requires_initial_heavy_l_geometry():
    residue = make_residue("PHE", [("N", "N"), ("CA", "C"), ("C", "C"), ("CB", "C"), ("HA", "H")])
    topology = FakeTopology([residue])
    initial = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.2],
            [0.0, 0.0, 0.2],
        ]
    )
    report = relax_h.alpha_ha_stereochemistry(topology, initial, initial)
    assert report["status"] == "fail"
    assert "initial heavy" in report["failures"][0]


def test_set_zero_masses_preserves_fixed_particle_under_nonzero_force_reference():
    openmm = pytest.importorskip("openmm")
    unit = openmm.unit
    system = openmm.System()
    system.addParticle(12.0)
    force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+y^2+z^2)")
    force.addGlobalParameter("k", 1000.0)
    force.addGlobalParameter("x0", 1.0)
    force.addParticle(0, [])
    system.addForce(force)

    original = relax_h.set_zero_masses(system, [0])

    assert original == [12.0]
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    try:
        context.setPositions([[0, 0, 0]] * unit.nanometer)
        openmm.LocalEnergyMinimizer.minimize(context, 1.0 * unit.kilojoules_per_mole / unit.nanometer, 100)
        pos = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    finally:
        del context
        del integrator
    assert pos[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_observed_heavy_mode_fixes_strict_subset_and_makes_modeled_heavy_mobile(tmp_path):
    residue = FakeResidue("ALA", 0, "A", 1)
    FakeAtom("N", residue, 0, "N")
    FakeAtom("CA", residue, 1, "C")
    FakeAtom("C", residue, 2, "C")
    FakeAtom("CB", residue, 3, "C")
    FakeAtom("HA", residue, 4, "H")
    topology = FakeTopology([residue])
    masks = relax_h.classify_mobile_and_fixed_atoms(topology)
    path = tmp_path / "observed.json"
    path.write_text('{"observed_heavy_indices": [0, 1]}', encoding="utf-8")

    updated = relax_h.apply_observed_heavy_mode(masks, path)

    assert updated["mode"] == "fixed_observed_heavy"
    assert updated["fixed_solute_heavy"] == [0, 1]
    assert updated["modeled_solute_heavy"] == [2, 3]
    assert updated["mobile"] == [2, 3, 4]

    path.write_text('{"observed_heavy_indices": [0, 1, 2, 3]}', encoding="utf-8")
    with pytest.raises(ValueError, match="strict subset"):
        relax_h.apply_observed_heavy_mode(masks, path)

    path.write_text('{"observed_heavy_indices": [4]}', encoding="utf-8")
    with pytest.raises(ValueError, match="solute heavy"):
        relax_h.apply_observed_heavy_mode(masks, path)


def test_alpha_gate_fails_when_movable_modeled_heavy_inverts_post_geometry():
    residue = make_residue("VAL", [("N", "N"), ("CA", "C"), ("C", "C"), ("CB", "C"), ("HA", "H")])
    topology = FakeTopology([residue])
    initial = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.2],
        ]
    )
    final = initial.copy()
    final[3] = [0.0, 0.0, -0.2]
    final[4] = [0.0, 0.0, 0.2]

    report = relax_h.alpha_ha_stereochemistry(topology, initial, final)

    assert report["status"] == "fail"
    assert any("post heavy" in failure for failure in report["failures"])


def test_beta_chirality_checks_initial_and_post_thr_ile_geometry():
    residue = make_residue("ILE", [("CA", "C"), ("CB", "C"), ("CG1", "C"), ("CG2", "C")])
    topology = FakeTopology([residue])
    initial = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.2],
        ]
    )
    assert relax_h.beta_chirality(topology, initial, initial)["status"] == "pass"

    final = initial.copy()
    final[3] = [0.0, 0.0, -0.2]
    report = relax_h.beta_chirality(topology, initial, final)
    assert report["status"] == "fail"
    assert "post THR/ILE" in report["failures"][0]
    near_flat = initial.copy()
    near_flat[3, 2] = 1e-5
    repaired = relax_h.beta_chirality(topology, near_flat, initial)
    assert repaired["status"] == "pass"
    assert repaired["examples"][0]["initial_valid"] is False
    assert relax_h.beta_chirality(topology, near_flat, near_flat)["status"] == "fail"


def test_observed_cli_inventory_binds_topology_hash(tmp_path):
    import json
    top = tmp_path / "top.prmtop"
    top.write_text("topology")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"status": "pass", "sources": {
        "prmtop": {"sha256": relax_h.sha256_file(top)}}}))
    relax_h.validate_observed_source(inventory, top)
    top.write_text("changed topology")
    with pytest.raises(ValueError, match="topology hash mismatch"):
        relax_h.validate_observed_source(inventory, top)
