import json
import sys
import types

import numpy as np
import pytest

from scripts import relax_atomistic_preparation as relax


def test_input_snapshot_retains_start_hash_and_detects_changed_or_removed_file(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"indices":[1,2]}')
    snapshot = relax.input_snapshot({"restraints": path})
    initial_digest = snapshot["restraints"]["sha256"]
    assert relax.changed_inputs(snapshot) == []
    path.write_text('{"indices":[1,3]}')
    assert relax.changed_inputs(snapshot) == ["restraints"]
    assert snapshot["restraints"]["sha256"] == initial_digest
    path.unlink()
    assert relax.changed_inputs(snapshot) == ["restraints"]


def test_load_restrain_indices_accepts_declared_json_and_rejects_bad_values(tmp_path):
    path = tmp_path / "restrain.json"
    path.write_text(json.dumps({"restrain_indices": [0, 3, 5]}), encoding="utf-8")

    assert relax.load_restrain_indices(path, 6) == [0, 3, 5]

    path.write_text(json.dumps({"restrain_indices": [0, 0]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        relax.load_restrain_indices(path, 6)

    path.write_text(json.dumps({"restrain_indices": [6]}), encoding="utf-8")
    with pytest.raises(ValueError, match="outside atom range"):
        relax.load_restrain_indices(path, 6)

    path.write_text(json.dumps({"restrain_indices": [True]}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-integer"):
        relax.load_restrain_indices(path, 6)


def test_choose_nonbonded_method_supports_periodic_and_dry_inputs():
    assert relax.choose_nonbonded_method("auto", True) == "PME"
    assert relax.choose_nonbonded_method("auto", False) == "NoCutoff"
    assert relax.choose_nonbonded_method("PME", True) == "PME"
    assert relax.choose_nonbonded_method("NoCutoff", False) == "NoCutoff"
    with pytest.raises(ValueError, match="PME requires periodic box"):
        relax.choose_nonbonded_method("PME", False)


def test_mapping_payload_preserves_reference_and_q_shape_contract():
    mapping = {
        "core_indices": [0, 2],
        "reference_nm": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
        "q": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "ddb1_atom_indices": [],
    }

    assert relax.validate_mapping_payload(mapping, 3) == {
        "core_count": 2,
        "ddb1_atom_count": 0,
        "reference_shape": [2, 3],
        "q_length": 6,
    }

    bad = dict(mapping, reference_nm=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="reference_nm"):
        relax.validate_mapping_payload(bad, 3)

    valid_two_core = dict(mapping, q=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert relax.validate_mapping_payload(valid_two_core, 3)["q_length"] == 6

    bad_q = dict(mapping, q=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"3\*len\(core_indices\)=6"):
        relax.validate_mapping_payload(bad_q, 3)

    bad_bool_index = dict(mapping, core_indices=[True, 2])
    with pytest.raises(ValueError, match="non-integer"):
        relax.validate_mapping_payload(bad_bool_index, 3)



def test_mapping_payload_accepts_actual_269_core_807_q_contract():
    core = list(range(269))
    mapping = {
        "core_indices": core,
        "reference_nm": [[float(i), 0.0, 0.0] for i in core],
        "q": [0.0] * 807,
        "ddb1_atom_indices": [],
    }

    summary = relax.validate_mapping_payload(mapping, 300)

    assert summary["core_count"] == 269
    assert summary["q_length"] == 807

    bad = dict(mapping, q=[0.0] * 806)
    with pytest.raises(ValueError, match=r"3\*len\(core_indices\)=807"):
        relax.validate_mapping_payload(bad, 300)

def test_heavy_clash_pairs_excludes_direct_bonds_only():
    topology = fake_topology(
        [
            ("C", "ALA", "C"),
            ("CA", "ALA", "C"),
            ("N", "GLY", "N"),
            ("H", "GLY", "H"),
        ],
        bonds=[(0, 1)],
    )
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.07, 0.0, 0.0],
            [0.0, 0.01, 0.0],
        ]
    )

    clashes = relax.heavy_clash_pairs(topology, positions)

    assert len(clashes) == 2
    assert [row["atoms"] for row in clashes] == [[1, 2], [0, 2]]



def brute_periodic_distance(delta, box):
    return min(
        np.linalg.norm(delta - np.array([i, j, k]) @ box)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
    )


def test_heavy_clash_pairs_periodic_cross_boundary_skew_matches_brute27():
    box = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.35, 1.0, 0.0],
            [0.1, 0.2, 1.0],
        ]
    )
    topology = fake_topology(
        [
            ("CA", "ALA", "C"),
            ("CB", "GLY", "C"),
            ("H", "GLY", "H"),
        ],
        bonds=[],
    )
    frac = np.array([[0.98, 0.50, 0.50], [0.03, 0.50, 0.50], [0.04, 0.50, 0.50]])
    positions = frac @ box

    clashes = relax.heavy_clash_pairs(topology, positions, box_vectors_nm=box)

    expected = brute_periodic_distance(positions[1] - positions[0], box)
    assert len(clashes) == 1
    assert clashes[0]["atoms"] == [0, 1]
    assert clashes[0]["distance_nm"] == pytest.approx(expected)
    assert clashes[0]["distance_nm"] < relax.HEAVY_CLASH_CUTOFF_NM


def test_periodic_clash_detection_is_lattice_translation_invariant():
    box = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.25, 1.1, 0.0],
            [0.0, 0.15, 1.2],
        ]
    )
    topology = fake_topology(
        [
            ("CA", "ALA", "C"),
            ("CB", "GLY", "C"),
        ],
        bonds=[],
    )
    positions = np.array([[0.01, 0.02, 0.03], [0.06, 0.02, 0.03]])
    shifted = positions.copy()
    shifted[1] = shifted[1] + np.array([2, -1, 1]) @ box

    base = relax.heavy_clash_pairs(topology, positions, box_vectors_nm=box)
    moved = relax.heavy_clash_pairs(topology, shifted, box_vectors_nm=box)

    assert [row["atoms"] for row in moved] == [row["atoms"] for row in base]
    assert moved[0]["distance_nm"] == pytest.approx(base[0]["distance_nm"])


def test_position_restraints_use_compound_bond_force(monkeypatch):
    created = {}

    class FakeCompoundBondForce:
        def __init__(self, particles_per_bond, expression):
            created["particles_per_bond"] = particles_per_bond
            created["expression"] = expression
            self.global_parameters = []
            self.per_bond_parameters = []
            self.bonds = []

        def addGlobalParameter(self, name, value):
            self.global_parameters.append((name, value))

        def addPerBondParameter(self, name):
            self.per_bond_parameters.append(name)

        def addBond(self, particles, parameters):
            self.bonds.append((particles, parameters))

    class FakeSystem:
        def __init__(self):
            self.forces = []

        def addForce(self, force):
            self.forces.append(force)

        def getNumForces(self):
            return len(self.forces)

    fake_openmm = types.SimpleNamespace(CustomCompoundBondForce=FakeCompoundBondForce)
    monkeypatch.setitem(sys.modules, "openmm", fake_openmm)
    system = FakeSystem()

    force_index = relax.add_position_restraints(system, [2], np.array([[0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3]]), 1000.0)

    assert force_index == 0
    assert created["particles_per_bond"] == 1
    assert "x1-x0" in created["expression"]
    assert system.forces[0].per_bond_parameters == ["x0", "y0", "z0"]
    assert system.forces[0].bonds == [([2], [0.1, 0.2, 0.3])]


def test_zn_sg_distances_use_raw_bonded_geometry_not_minimum_image():
    topology = fake_topology(
        [
            ("ZN", "ZN1", "Zn"),
            ("SG", "CY1", "S"),
            ("SG", "CY1", "S"),
            ("SG", "CY1", "S"),
            ("SG", "CY1", "S"),
        ],
        bonds=[(0, 1), (0, 2), (0, 3), (0, 4)],
    )
    box = np.diag([5.0, 5.0, 5.0])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.95, 0.0, 0.0],
            [0.23, 0.0, 0.0],
            [0.24, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ]
    )

    distances = relax.zn_sg_distances(topology, positions, box)

    assert distances[0] == pytest.approx(4.95)
    assert relax.status_from_checks(
        finite_energies=True,
        final_clashes=[],
        final_zn_sg_nm=distances,
        positions_changed=True,
    ) == "failed"


class FakePlatformForOptions:
    def __init__(self, names, values=None):
        self._names = list(names)
        self._values = dict(values or {})

    def getPropertyNames(self):
        return list(self._names)

    def getPropertyValue(self, context, name):
        return self._values[name]


class FakePlatformFactory:
    def __init__(self, platform):
        self.platform = platform
        self.requested_names = []

    def getPlatformByName(self, name):
        self.requested_names.append(name)
        return self.platform


def test_gpu_pme_requires_disable_before_context_creation():
    with pytest.raises(ValueError, match="--disable-pme-stream"):
        relax.validate_gpu_pme_request("OpenCL", "PME", False)
    with pytest.raises(ValueError, match="--disable-pme-stream"):
        relax.validate_gpu_pme_request("CUDA", "PME", False)

    relax.validate_gpu_pme_request("OpenCL", "PME", True)
    relax.validate_gpu_pme_request("OpenCL", "NoCutoff", False)
    relax.validate_gpu_pme_request("Reference", "PME", False)


def test_minimizer_platform_options_set_gpu_double_and_disable_pme_stream():
    platform = FakePlatformForOptions(["Precision", "DeviceIndex", "DisablePmeStream"])
    factory = FakePlatformFactory(platform)

    selected, properties = relax.minimizer_platform_options(
        factory,
        "OpenCL",
        "0",
        disable_pme_stream=True,
    )

    assert selected is platform
    assert factory.requested_names == ["OpenCL"]
    assert properties == {
        "Precision": "double",
        "DeviceIndex": "0",
        "DisablePmeStream": "true",
    }


def test_minimizer_platform_options_reject_missing_gpu_pme_stream_property():
    platform = FakePlatformForOptions(["Precision"])
    factory = FakePlatformFactory(platform)

    with pytest.raises(ValueError, match="DisablePmeStream"):
        relax.minimizer_platform_options(
            factory,
            "OpenCL",
            None,
            disable_pme_stream=True,
        )


def test_precision_provenance_uses_effective_context_property_readback():
    platform = FakePlatformForOptions(
        ["Precision", "DeviceIndex", "DisablePmeStream"],
        {"Precision": "double", "DeviceIndex": "0", "DisablePmeStream": "true"},
    )
    context = object()

    precision = relax.precision_provenance(
        platform,
        context,
        requested="double",
        disable_pme_stream=True,
    )

    assert precision["requested_override"] == "double"
    assert precision["effective"] == "double"
    assert precision["source"] == "Context Platform.getPropertyValue(Precision)"
    assert precision["pme_stream"] == {
        "disable_requested": True,
        "override_applied": True,
        "effective_disabled": True,
        "source": "Context Platform.getPropertyValue(DisablePmeStream)",
    }
    assert relax.effective_platform_properties(platform, context) == {
        "Precision": "double",
        "DeviceIndex": "0",
        "DisablePmeStream": "true",
    }
    relax.validate_effective_gpu_pme("OpenCL", "PME", precision)


def test_precision_provenance_does_not_claim_precision_override_when_unavailable():
    platform = FakePlatformForOptions([])

    precision = relax.precision_provenance(
        platform,
        object(),
        requested=None,
        disable_pme_stream=False,
    )

    assert precision["requested_override"] is None
    assert precision["effective"] == "unreported_platform_default"
    assert precision["pme_stream"]["effective_disabled"] is None


def test_precision_provenance_rejects_ineffective_gpu_pme_stream_disable():
    platform = FakePlatformForOptions(
        ["Precision", "DisablePmeStream"],
        {"Precision": "double", "DisablePmeStream": "false"},
    )

    with pytest.raises(ValueError, match="DisablePmeStream=true"):
        relax.precision_provenance(
            platform,
            object(),
            requested="double",
            disable_pme_stream=True,
        )

    precision = {"effective": "double", "pme_stream": {"effective_disabled": False}}
    with pytest.raises(ValueError, match="effective double precision"):
        relax.validate_effective_gpu_pme("OpenCL", "PME", precision)


def test_cli_plumbs_disable_pme_stream_to_run(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "status": "minimization_complete",
            "heavy_clashes_lt_0p8A_excluding_bonded_neighbors": {"post_count": 0},
        }

    monkeypatch.setattr(relax, "run", fake_run)

    rc = relax.main(
        [
            "--prmtop",
            str(tmp_path / "input.prmtop"),
            "--inpcrd",
            str(tmp_path / "input.inpcrd"),
            "--mapping",
            str(tmp_path / "mapping.json"),
            "--restrain-indices",
            str(tmp_path / "restraints.json"),
            "--prep",
            str(tmp_path / "zinc.prep"),
            "--output-dir",
            str(tmp_path / "out"),
            "--platform",
            "OpenCL",
            "--disable-pme-stream",
        ]
    )

    assert rc == 0
    assert captured["platform_name"] == "OpenCL"
    assert captured["disable_pme_stream"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "minimization_complete"

def test_status_requires_changed_positions_no_clashes_and_metal_bounds():
    assert (
        relax.status_from_checks(
            finite_energies=True,
            final_clashes=[],
            final_zn_sg_nm=[0.23, 0.24, 0.25, 0.26],
            positions_changed=True,
        )
        == "minimization_complete"
    )
    assert (
        relax.status_from_checks(
            finite_energies=True,
            final_clashes=[{"atoms": [1, 2]}],
            final_zn_sg_nm=[0.23, 0.24, 0.25, 0.26],
            positions_changed=True,
        )
        == "failed"
    )
    assert (
        relax.status_from_checks(
            finite_energies=True,
            final_clashes=[],
            final_zn_sg_nm=[0.18, 0.24, 0.25, 0.26],
            positions_changed=True,
        )
        == "failed"
    )
    assert (
        relax.status_from_checks(
            finite_energies=True,
            final_clashes=[],
            final_zn_sg_nm=[0.23, 0.24, 0.25, 0.26],
            positions_changed=False,
        )
        == "failed"
    )


class FakeElement:
    def __init__(self, symbol):
        self.symbol = symbol


class FakeResidue:
    def __init__(self, name, index):
        self.name = name
        self.index = index


class FakeAtom:
    def __init__(self, name, residue, index, element):
        self.name = name
        self.residue = residue
        self.index = index
        self.element = FakeElement(element)


class FakeTopology:
    def __init__(self, atoms, bonds):
        self._atoms = atoms
        self._bonds = bonds

    def atoms(self):
        return iter(self._atoms)

    def bonds(self):
        for a, b in self._bonds:
            yield self._atoms[a], self._atoms[b]


def fake_topology(spec, bonds):
    atoms = []
    residues = {}
    for index, (atom_name, residue_name, element) in enumerate(spec):
        residue = residues.setdefault(residue_name, FakeResidue(residue_name, len(residues)))
        atoms.append(FakeAtom(atom_name, residue, index, element))
    return FakeTopology(atoms, bonds)
