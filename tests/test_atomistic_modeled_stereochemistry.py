import csv
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import atomistic_modeled_stereochemistry as stereo
from scripts.atomistic_caps import AtomRecord, format_atom


def make_atoms(resname="LEU", beta=False):
    if beta:
        xyz = {"N": (1.5, 1.3, 0), "CA": (1.5, 0, 0), "C": (1.5, 0, -1.5),
               "O": (1.5, 0, -2.5), "CB": (0, 0, 0), "CG2": (.2, .2, -1.4)}
        if resname == "ILE":
            xyz.update(CG1=(0, 1.5, 0), CD1=(.5, 2.7, .4))
        else:
            xyz["OG1"] = (0, 1.5, 0)
    else:
        xyz = {"N": (1.3, 0, 0), "CA": (0, 0, 0), "C": (0, 1.5, 0),
               "O": (0, 2.5, .1), "CB": (.2, .3, -1.2), "CG": (.8, .7, -2.3),
               "CD1": (.4, 1.7, -3), "CD2": (2, .5, -2.5)}
    atoms = [AtomRecord("ATOM", i, name, "", resname, "B", 342, "", *point, element=name[0])
             for i, (name, point) in enumerate(xyz.items(), 1)]
    atoms.append(AtomRecord("HETATM", 90, "ZN", "", "ZN", "C", 501, "", 9, 8, 7, element="ZN"))
    return atoms


def mapping_for(atoms, beta=False):
    result = {}
    for atom in atoms:
        modeled = (atom.name == "CG2") if beta else (atom.name not in stereo.BACKBONE and atom.resname != "ZN")
        result[stereo.atom_key(atom)] = {"residue_name": atom.resname, "xyz": atom.xyz,
                                       "source_status": "repaired_missing_heavy_atom" if modeled else "observed_input"}
    return result


def write_inputs(tmp_path, atoms, mapping):
    pdb = tmp_path / "input.pdb"
    pdb.write_text("REMARK synthetic geometry; not an MD result\n" + "".join(format_atom(atom, i)+"\n" for i, atom in enumerate(atoms, 1)) + "END\n")
    csv_path = tmp_path / "repair.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chain", "residue_number", "residue_name", "atom", "source_status", "x", "y", "z"])
        writer.writeheader()
        for key, row in mapping.items():
            writer.writerow({"chain": key[0], "residue_number": key[1], "atom": key[2],
                             "residue_name": row["residue_name"], "source_status": row["source_status"],
                             **dict(zip(("x", "y", "z"), row["xyz"]))})
    return pdb, csv_path


def test_retained_leap_met_template_defines_positive_L_sign():
    result = stereo.validate_template(stereo.DEFAULT_TEMPLATE)
    assert result["signed_volume_A3"] > 2
    assert result["cb_out_of_backbone_plane_degrees"] > 40
    assert len(result["sha256"]) == 64


def test_alpha_reflection_flips_sign_preserves_distances_angles_and_fixed_atoms():
    atoms = make_atoms()
    before = stereo.residue_groups(atoms)[("B", 342, "LEU")]
    out, changes = stereo.corrected_atoms(atoms, mapping_for(atoms))
    after = stereo.residue_groups(out)[("B", 342, "LEU")]
    np.testing.assert_allclose(stereo.ca_geometry(after)[0], -stereo.ca_geometry(before)[0], atol=1e-12)
    assert len(changes) == 1
    assert changes[0]["floating_point_isometry"]["max_pair_distance_change_A"] < 1e-12
    assert changes[0]["floating_point_isometry"]["max_triple_angle_change_degrees"] < 1e-10
    for old, new in zip(atoms, out):
        if old.name in stereo.BACKBONE or old.resname == "ZN":
            assert old == new
        else:
            np.testing.assert_array_equal(new.xyz, old.xyz * [1, 1, -1])


@pytest.mark.parametrize("provenance", ["observed_input", "unknown", None])
def test_alpha_rejects_observed_or_unproven_side_chain(provenance):
    atoms = make_atoms()
    mapping = mapping_for(atoms)
    key = ("B", 342, "CD1")
    if provenance is None:
        del mapping[key]
    else:
        mapping[key]["source_status"] = provenance
    with pytest.raises(ValueError, match="entirely proven modeled"):
        stereo.corrected_atoms(atoms, mapping)


@pytest.mark.parametrize("resname", ["ILE", "THR", "PRO", "VAL", "CYS", "UNK"])
def test_negative_alpha_unhandled_residues_fail_closed(resname):
    atoms = make_atoms(resname)
    with pytest.raises(ValueError, match="Unhandled negative C-alpha"):
        stereo.corrected_atoms(atoms, mapping_for(atoms))


@pytest.mark.parametrize("resname", ["ILE", "THR"])
def test_beta_changes_only_modeled_terminal_CG2_preserving_alpha_and_other_branch(resname):
    atoms = make_atoms(resname, beta=True)
    before = stereo.residue_groups(atoms)[("B", 342, resname)]
    assert stereo.ca_geometry(before)[0] > 0
    assert stereo.beta_geometry(before, resname)[0] < 0
    out, changes = stereo.corrected_beta_atoms(atoms, mapping_for(atoms, beta=True))
    after = stereo.residue_groups(out)[("B", 342, resname)]
    assert stereo.ca_geometry(after) == stereo.ca_geometry(before)
    np.testing.assert_allclose(stereo.beta_geometry(after, resname)[0], -stereo.beta_geometry(before, resname)[0], atol=1e-12)
    assert changes[0]["floating_point_isometry"]["max_pair_distance_change_A"] < 1e-12
    assert changes[0]["floating_point_isometry"]["max_triple_angle_change_degrees"] < 1e-10
    for old, new in zip(atoms, out):
        if old.name != "CG2":
            assert old == new  # Includes ILE CG1 and CD1, THR OG1, backbone and Zn.
    assert changes[0]["sidechain_atom_keys"] == [["B", 342, "CG2"]]


@pytest.mark.parametrize("resname", ["ILE", "THR"])
def test_beta_rejects_observed_CG2(resname):
    atoms = make_atoms(resname, beta=True)
    mapping = mapping_for(atoms, beta=True)
    mapping[("B", 342, "CG2")]["source_status"] = "observed_input"
    with pytest.raises(ValueError, match="requires modeled CG2"):
        stereo.corrected_beta_atoms(atoms, mapping)


@pytest.mark.parametrize("beta", [False, True])
def test_pdb_repair_is_deterministic_with_exact_observed_and_backbone_records(tmp_path, beta):
    atoms = make_atoms("ILE" if beta else "LEU", beta=beta)
    pdb, mapping = write_inputs(tmp_path, atoms, mapping_for(atoms, beta=beta))
    original_bytes = pdb.read_bytes()
    report = stereo.repair(pdb, mapping, tmp_path / "out")
    again = stereo.repair(pdb, mapping, tmp_path / "out")
    assert report["output_pdb_sha256"] == again["output_pdb_sha256"]
    assert pdb.read_bytes() == original_bytes
    assert report["corrected_alpha_count"] == int(not beta)
    assert report["corrected_beta_count"] == int(beta)
    assert report["after"]["negative_count"] == report["beta_after"]["negative_count"] == 0
    assert not report["production_ready"]
    assert not report["minimization_performed"]
    assert not report["MD_performed"]
    assert report["preservation"]["observed_atoms_exact"]
    old_lines, new_lines = pdb.read_text().splitlines(), Path(report["output_pdb"]).read_text().splitlines()
    moved = {tuple(row["atom"]) for row in report["changed_atoms"]}
    for old, new in zip(old_lines, new_lines):
        atom = stereo.parse_pdb_atom_line(old)
        if atom is None or stereo.atom_key(atom) not in moved:
            assert new == old
        else:
            assert new[:30] == old[:30] and new[54:] == old[54:]


def test_near_flat_positive_beta_is_reported_without_claiming_readiness(tmp_path):
    atoms = make_atoms("ILE", beta=True)
    atoms = [replace(a, z=.001) if a.name == "CG2" else a for a in atoms]
    pdb, mapping = write_inputs(tmp_path, atoms, mapping_for(atoms, beta=True))
    result = stereo.repair(pdb, mapping, tmp_path / "out")
    assert result["corrected_residue_count"] == 0
    assert result["beta_after"]["minimum_out_of_plane_angle_degrees"] < 1
    assert len(result["beta_after"]["near_flat_residues"]) == 1
    assert not result["production_ready"]
    assert "Near-flat" in result["post_minimization_handoff"]


def test_mismatched_provenance_coordinates_and_input_overwrite_are_rejected(tmp_path):
    atoms = make_atoms()
    mapping = mapping_for(atoms)
    mapping[("B", 342, "CD1")]["xyz"] = np.array([1, 2, 3])
    pdb, csv_path = write_inputs(tmp_path, atoms, mapping)
    with pytest.raises(ValueError, match="do not match repair CSV"):
        stereo.repair(pdb, csv_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    pdb.rename(tmp_path / "capped_heavy.pdb")
    with pytest.raises(ValueError, match="separate"):
        stereo.repair(tmp_path / "capped_heavy.pdb", csv_path, tmp_path)


@pytest.mark.parametrize("case", ["hydrogen", "duplicate", "nan", "model2", "altloc"])
def test_ambiguous_or_nonheavy_input_rejected(case):
    atom = make_atoms()[0]
    if case == "hydrogen":
        atom = replace(atom, name="H", element="H")
    elif case == "nan":
        atom = replace(atom, x=float("nan"))
    elif case == "altloc":
        atom = replace(atom, altloc="A")
    lines = [format_atom(atom, 1)]
    if case == "duplicate":
        lines.append(lines[0])
    elif case == "model2":
        lines.insert(0, "MODEL        2")
    with pytest.raises(ValueError):
        stereo.strict_atoms(lines)


def test_cli_offline_generates_only_premin_geometry(tmp_path):
    atoms = make_atoms()
    pdb, mapping = write_inputs(tmp_path, atoms, mapping_for(atoms))
    result = subprocess.run([sys.executable, str(Path(stereo.__file__)), "--input", str(pdb),
                             "--repair-csv", str(mapping), "--output-dir", str(tmp_path / "cli"), "--offline"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert '"corrected_residue_count": 1' in result.stdout
    assert (tmp_path / "cli/stereochemistry.json").is_file()
