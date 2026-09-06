import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import prepare_atomistic_amber as amber


ROOT = Path(__file__).resolve().parents[1]
ACTUAL_CAPPED_PDB = Path(os.environ.get(
    "CRBN_ATOMISTIC_CAPPED_PDB", ROOT / "results/atomistic/stereochemistry/capped_heavy.pdb"
))


def atom_line(serial, name, resname, chain, resseq, x, y, z, element=None, record="ATOM"):
    element = element or name.lstrip("0123456789")[:1].upper()
    atom = amber.PdbAtom(
        record=record,
        serial=serial,
        name=name,
        altloc=" ",
        resname=resname,
        chain=chain,
        resseq=resseq,
        icode=" ",
        x=float(x),
        y=float(y),
        z=float(z),
        occupancy="  1.00",
        bfactor="  0.00",
        element=element,
        charge="  ",
        raw="",
    )
    return amber.format_atom(atom, serial)


def capped_fixture(tmp_path):
    lines = []
    serial = 1
    for resseq in range(1, 1141):
        lines.append(atom_line(serial, "CA", "ALA", "A", resseq, resseq * 1.0, 0.0, 0.0, "C"))
        serial += 1
    lines.append(atom_line(serial, "C", "ACE", "B", 63, 0.0, 1.0, 0.0, "C"))
    serial += 1
    for resseq in range(64, 429):
        resname = "CYS" if resseq in amber.ZN_CYS_RESIDUES else "ALA"
        base = resseq - 64
        lines.append(atom_line(serial, "N", resname, "B", resseq, base * 1.1, 1.0, np.sin(base / 11), "N"))
        serial += 1
        lines.append(atom_line(serial, "CA", resname, "B", resseq, base * 1.1 + 0.3, 1.7, np.cos(base / 13), "C"))
        serial += 1
        lines.append(atom_line(serial, "C", resname, "B", resseq, base * 1.1 + 0.7, 1.1, np.sin(base / 17), "C"))
        serial += 1
        lines.append(atom_line(serial, "O", resname, "B", resseq, base * 1.1 + 1.0, 0.7, np.cos(base / 19), "O"))
        serial += 1
        if resseq in amber.ZN_CYS_RESIDUES:
            lines.append(atom_line(serial, "CB", resname, "B", resseq, base * 1.1 + 0.2, 2.2, 0.4, "C"))
            serial += 1
            lines.append(atom_line(serial, "SG", resname, "B", resseq, base * 1.1 + 0.2, 2.8, 0.8, "S"))
            serial += 1
    lines.append(atom_line(serial, "N", "NME", "B", 429, 402.0, 1.0, 0.0, "N"))
    serial += 1
    lines.append(atom_line(serial, "ZN", "ZN", "C", 501, 300.0, 2.5, 1.0, "ZN", record="HETATM"))
    lines.append("END")
    out = tmp_path / "capped_heavy.pdb"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def q_inputs(tmp_path, capped_pdb):
    tmp_path.mkdir(parents=True, exist_ok=True)
    atoms = amber.read_pdb_atoms(capped_pdb)
    ca = {atom.resseq: np.asarray(atom.xyz, dtype=float) for atom in atoms if atom.chain == "B" and atom.name == "CA"}
    core_residues = list(range(64, 333))
    core_csv = tmp_path / "core.csv"
    core_csv.write_text("author_resnum\n" + "\n".join(str(i) for i in core_residues) + "\n", encoding="utf-8")
    reference = np.vstack([ca[residue] for residue in core_residues])
    delta = np.column_stack(
        [
            np.sin(np.arange(len(core_residues)) / 7),
            np.cos(np.arange(len(core_residues)) / 9),
            np.sin(np.arange(len(core_residues)) / 13),
        ]
    )
    delta /= np.linalg.norm(delta.reshape(-1))
    conformers = np.stack([reference, reference + delta])
    labels = np.asarray(["8CVP", "open"])
    ensemble = tmp_path / "ensemble.npz"
    diffvec = tmp_path / "diffvec.npz"
    np.savez(ensemble, _confs=conformers, _labels=labels)
    np.savez(diffvec, diff_vec=delta.reshape(-1), labels=labels, open_mask=np.asarray([False, True]))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "core_residue_file": str(core_csv),
                "core_position_count": len(core_residues),
                "pilot_reference": "8CVP",
            }
        ),
        encoding="utf-8",
    )
    return config, ensemble, diffvec


def metal_sources(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prep = tmp_path / "ZAFF.prep"
    frcmod = tmp_path / "ZAFF.frcmod"
    prep.write_text("synthetic prep placeholder\n", encoding="utf-8")
    frcmod.write_text("synthetic frcmod placeholder\n", encoding="utf-8")
    return prep, frcmod


def bridge_kwargs(tmp_path, capped, assembly="joint"):
    config, ensemble, diffvec = q_inputs(tmp_path, capped)
    prep, frcmod = metal_sources(tmp_path)
    return {
        "config_path": config,
        "input_pdb": capped,
        "prep": prep,
        "frcmod": frcmod,
        "output_dir": tmp_path / f"out_{assembly}",
        "offline": True,
        "tleap": None,
        "ensemble_path": ensemble,
        "diffvec_path": diffvec,
        "assembly": assembly,
    }


def test_joint_pdb_is_normalized_for_zaff_and_bonded_by_actual_residue_order(tmp_path):
    capped = capped_fixture(tmp_path)
    prepared = tmp_path / "amber_input_renamed.pdb"
    report = amber.write_normalized_pdb(capped, prepared)
    atoms = amber.read_pdb_atoms(prepared)
    commands = amber.zinc_bond_commands(atoms)

    assert report["assembly"] == "joint"
    assert report["validation"]["ddb1_residue_count"] == 1140
    assert report["validation"]["crbn_protein_residue_count"] == 365
    assert report["validation"]["crbn_cap_residues"] == [("B", 63, "ACE"), ("B", 429, "NME")]
    assert report["validation"]["zinc_atom"] == ("C", 501, "ZN")
    assert report["renamed_atom_count"] > 4
    assert {a.resname for a in atoms if a.chain == "B" and a.resseq in amber.ZN_CYS_RESIDUES} == {"CY1"}
    assert {a.resname for a in atoms if a.chain == "C" and a.resseq == 501} == {"ZN1"}
    assert commands == [
        "bond mol.1401.SG mol.1508.ZN",
        "bond mol.1404.SG mol.1508.ZN",
        "bond mol.1469.SG mol.1508.ZN",
        "bond mol.1472.SG mol.1508.ZN",
    ]
    assert [row["one_based_residue_order_command"] for row in amber.zinc_bond_selectors(atoms)] == commands


def test_isolated_pdb_retains_only_crbn_caps_crbn_and_zinc(tmp_path):
    capped = capped_fixture(tmp_path)
    prepared = tmp_path / "amber_input_isolated_renamed.pdb"
    report = amber.write_normalized_pdb(capped, prepared, assembly="isolated")
    atoms = amber.read_pdb_atoms(prepared)
    residues = amber.residue_order(atoms)

    assert report["assembly"] == "isolated"
    assert report["validation"]["ddb1_residue_count"] == 0
    assert report["validation"]["crbn_protein_residue_count"] == 365
    assert report["validation"]["residue_count_before_solvent"] == 368
    assert residues[0] == ("B", 63, "ACE")
    assert residues[-2:] == [("B", 429, "NME"), ("C", 501, "ZN1")]
    assert {chain for chain, _, _ in residues} == {"B", "C"}
    assert amber.zinc_bond_commands(atoms, assembly="isolated") == [
        "bond mol.323.SG mol.430.ZN",
        "bond mol.326.SG mol.430.ZN",
        "bond mol.391.SG mol.430.ZN",
        "bond mol.394.SG mol.430.ZN",
    ]
    assert [row["one_based_residue_order_command"] for row in amber.zinc_bond_selectors(atoms, assembly="isolated")] == [
        "bond mol.261.SG mol.368.ZN",
        "bond mol.264.SG mol.368.ZN",
        "bond mol.329.SG mol.368.ZN",
        "bond mol.332.SG mol.368.ZN",
    ]


def test_tleap_inputs_are_ff14sb_tip3p_zaff_and_neutralization_only(tmp_path):
    capped = capped_fixture(tmp_path)
    prepared = tmp_path / "amber_input_renamed.pdb"
    amber.write_normalized_pdb(capped, prepared)
    atoms = amber.read_pdb_atoms(prepared)
    prep, frcmod = metal_sources(tmp_path)
    text = amber.build_tleap_input(
        prepared,
        prep,
        frcmod,
        "solvated",
        amber.zinc_bond_commands(atoms),
        solvated=True,
    )

    assert "source leaprc.protein.ff14SB" in text
    assert "source leaprc.water.tip3p" in text
    assert "loadAmberPrep" in text and "ZAFF.prep" in text
    assert "loadAmberParams" in text and "ZAFF.frcmod" in text
    assert "solvateOct mol TIP3PBOX 12.0" in text
    assert "addIonsRand mol Na+ 0" in text
    assert "addIonsRand mol Cl- 0" in text
    assert "150 mM salt-pair placement remains pending" in text
    assert text.count("bond mol.") == 4


def test_q807_uses_frozen_axis_in_prepared_frame_and_projects_rigid_motion(tmp_path):
    capped = capped_fixture(tmp_path)
    prepared = tmp_path / "amber_input_renamed.pdb"
    amber.write_normalized_pdb(capped, prepared)
    config, ensemble, diffvec = q_inputs(tmp_path, prepared)
    core = amber.read_core_residues(config.parent / "core.csv")
    q = amber.internal_projected_q807(atoms=amber.read_pdb_atoms(prepared), core_residues=core, ensemble_path=ensemble, diffvec_path=diffvec, reference_label="8CVP")

    q807 = np.asarray(q["prepared_q"])
    reference = np.asarray(q["prepared_reference_nm"])
    assert reference.shape == (269, 3)
    assert q807.shape == (807,)
    assert np.linalg.norm(q807) == pytest.approx(1.0)
    assert q["q_source"]["stored_axis_dot_recomputed"] == pytest.approx(1.0)
    assert q["q_source"]["kabsch_reference_to_prepared_rmsd_A"] < 1e-3
    rigid = amber.rigid_basis(reference)
    assert np.linalg.norm(rigid.T @ q807) < 1e-12


def test_bridge_writes_joint_and_isolated_reports_and_mapping_pending_without_tleap(tmp_path):
    capped = capped_fixture(tmp_path)
    joint = amber.prepare_amber_bridge(**bridge_kwargs(tmp_path / "joint", capped, assembly="joint"))
    isolated = amber.prepare_amber_bridge(**bridge_kwargs(tmp_path / "isolated", capped, assembly="isolated"))

    for report, assembly, expected_ddb1, expected_bonds in [
        (joint, "joint", 1140, ["bond mol.1401.SG mol.1508.ZN", "bond mol.1404.SG mol.1508.ZN", "bond mol.1469.SG mol.1508.ZN", "bond mol.1472.SG mol.1508.ZN"]),
        (isolated, "isolated", 0, ["bond mol.261.SG mol.368.ZN", "bond mol.264.SG mol.368.ZN", "bond mol.329.SG mol.368.ZN", "bond mol.332.SG mol.368.ZN"]),
    ]:
        saved = json.loads((Path(report["tleap_inputs"]["dry"]).parent / "preparation_report.json").read_text())
        assert report["assembly"] == assembly
        assert report["status"] == "leap_inputs_ready_mapping_pending"
        assert saved["pdb_preparation"]["validation"]["ddb1_residue_count"] == expected_ddb1
        assert saved["mapping"]["status"] == "pending_amber_outputs"
        assert saved["zinc_site"]["requested_site_bond_command_count"] == 4
        if assembly == "joint":
            assert saved["zinc_site"]["bond_commands"] == expected_bonds
        else:
            assert saved["zinc_site"]["bond_commands"] == [
                "bond mol.323.SG mol.430.ZN",
                "bond mol.326.SG mol.430.ZN",
                "bond mol.391.SG mol.430.ZN",
                "bond mol.394.SG mol.430.ZN",
            ]
        assert saved["zinc_site"]["one_based_residue_order_bond_commands"] == expected_bonds
        assert saved["tleap_inputs"]["salt_policy"]["status"] == "neutralization_only"
        assert Path(saved["tleap_inputs"]["dry"]).is_file()
        assert Path(saved["tleap_inputs"]["solvated_neutral"]).is_file()


def test_cy1_hg_is_fail_closed(tmp_path):
    atoms = amber.read_pdb_atoms(capped_fixture(tmp_path))
    hg = amber.PdbAtom(
        record="ATOM",
        serial=99999,
        name="HG",
        altloc=" ",
        resname="CYS",
        chain="B",
        resseq=323,
        icode=" ",
        x=0.0,
        y=0.0,
        z=0.0,
        occupancy="  1.00",
        bfactor="  0.00",
        element="H",
        charge="  ",
        raw="",
    )
    normalized, _ = amber.normalize_residue_names([*atoms, hg])
    with pytest.raises(ValueError, match="thiol hydrogens"):
        amber.validate_prepared_atoms(normalized)


def test_uncapped_input_is_rejected(tmp_path):
    capped = capped_fixture(tmp_path)
    lines = [
        line
        for line in capped.read_text().splitlines()
        if not (line.startswith(("ATOM  ", "HETATM")) and line[21] == "B" and int(line[22:26]) in {63, 429})
    ]
    uncapped = tmp_path / "uncapped.pdb"
    uncapped.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prepared = tmp_path / "amber_input_renamed.pdb"
    with pytest.raises(ValueError, match="exact CRBN caps"):
        amber.write_normalized_pdb(uncapped, prepared)


def test_transport_q_to_actual_frame_tracks_leap_recenter():
    prepared = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    q = np.arange(12, dtype=float)
    q = q - amber.rigid_basis(prepared) @ (amber.rigid_basis(prepared).T @ q)
    q /= np.linalg.norm(q)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    actual = prepared @ rotation + np.array([10.0, -3.0, 2.0])

    mapped = amber.transport_q_to_actual_frame(prepared, q, actual)

    assert np.asarray(mapped["reference_nm"]).shape == (4, 3)
    assert np.asarray(mapped["q"]).shape == (12,)
    assert mapped["q_transport"]["prepared_to_actual_core_rmsd_nm"] < 1e-12
    assert mapped["q_transport"]["post_projection_rigid_component_norm"] < 1e-10


def test_duplicate_and_altloc_atoms_are_rejected(tmp_path):
    capped = capped_fixture(tmp_path)
    lines = capped.read_text().splitlines()
    duplicate = tmp_path / "duplicate.pdb"
    duplicate.write_text("\n".join([*lines[:-1], lines[0], "END"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate PDB atom keys"):
        amber.read_pdb_atoms(duplicate)

    altloc = tmp_path / "altloc.pdb"
    changed = lines.copy()
    changed[0] = f"{changed[0][:16]}A{changed[0][17:]}"
    altloc.write_text("\n".join(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Alternate-location"):
        amber.read_pdb_atoms(altloc)


def test_residue_name_compatibility_is_exact_except_histidine_protonation():
    assert amber.residue_name_compatible("CY1", "CY1")
    assert amber.residue_name_compatible("ZN1", "ZN1")
    assert not amber.residue_name_compatible("CY1", "CYS")
    assert not amber.residue_name_compatible("ZN1", "ZN")
    assert amber.residue_name_compatible("HIS", "HID")
    assert amber.residue_name_compatible("HIS", "HIE")
    assert amber.residue_name_compatible("HIS", "HIP")


def test_cli_requires_input_pdb_when_public_default_is_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(amber, "DEFAULT_INPUT_PDB", tmp_path / "missing_capped_heavy.pdb")

    assert amber.main(["--offline", "--output-dir", str(tmp_path / "out")]) == 2
    assert "missing required --input-pdb" in capsys.readouterr().out


def test_failed_current_tleap_blocks_stale_solvated_mapping(tmp_path, monkeypatch):
    capped = capped_fixture(tmp_path)
    kwargs = bridge_kwargs(tmp_path, capped)
    kwargs["tleap"] = "tleap"
    kwargs["output_dir"].mkdir(parents=True)
    (kwargs["output_dir"] / "solvated.prmtop").write_text("stale", encoding="utf-8")
    (kwargs["output_dir"] / "solvated.inpcrd").write_text("stale", encoding="utf-8")

    def fake_run_tleap(input_file, workdir, executable):
        return {
            "command": [executable, "-f", str(input_file)],
            "returncode": 0,
            "status": "failed",
            "log": {"combined": str(workdir / "fake.log")},
            "failures": ["LEaP reported Errors = 1"],
        }

    monkeypatch.setattr(amber, "run_tleap", fake_run_tleap)
    report = amber.prepare_amber_bridge(**kwargs)

    assert report["status"] == "tleap_failed_mapping_blocked"
    assert report["mapping"]["status"] == "blocked_by_tleap_failure"
    assert not (kwargs["output_dir"] / "atomistic_mapping.json").exists()


def test_run_tleap_treats_returncode_zero_errors_as_failure_and_uses_generic_warning_labels(tmp_path):
    fake_tleap = tmp_path / "fake_tleap"
    fake_tleap.write_text(
        "#!/bin/sh\n"
        "echo 'Welcome to LEaP'\n"
        "echo 'Warning!'\n"
        "echo 'Close contact of 0.8 angstroms between .R<CY1 1>.A<SG 1> and .R<ZN1 2>.A<ZN 1>'\n"
        "echo 'Warning!'\n"
        "echo 'The unperturbed charge of the unit: -43.000000 is not zero.'\n"
        "echo 'Exiting LEaP: Errors = 1; Warnings = 2; Notes = 0.'\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_tleap.chmod(0o755)
    input_file = tmp_path / "tleap_solvated_neutral.in"
    input_file.write_text("quit\n", encoding="utf-8")

    result = amber.run_tleap(input_file, tmp_path, str(fake_tleap))

    assert result["returncode"] == 0
    assert result["status"] == "failed"
    assert "LEaP reported Errors = 1" in result["failures"]
    assert Path(result["log"]["combined"]).is_file()
    categories = result["log"]["leap_warnings"]["categories"]
    assert categories["nonbonded_close_contact"] == 1
    assert categories["nonzero_unit_charge"] == 1
    assert "close_contact_from_overlapping_prototype_geometry" not in categories


@pytest.mark.skipif(not ACTUAL_CAPPED_PDB.exists(), reason="set CRBN_ATOMISTIC_CAPPED_PDB to a prepared capped CRBN-DDB1 input")
def test_actual_capped_input_smoke_preserves_joint_and_isolated_residue_orders(tmp_path):
    joint = tmp_path / "joint.pdb"
    isolated = tmp_path / "isolated.pdb"
    amber.write_normalized_pdb(ACTUAL_CAPPED_PDB, joint)
    amber.write_normalized_pdb(ACTUAL_CAPPED_PDB, isolated, assembly="isolated")

    joint_atoms = amber.read_pdb_atoms(joint)
    isolated_atoms = amber.read_pdb_atoms(isolated)
    assert [row["one_based_residue_order_command"] for row in amber.zinc_bond_selectors(joint_atoms)] == [
        "bond mol.1401.SG mol.1508.ZN",
        "bond mol.1404.SG mol.1508.ZN",
        "bond mol.1469.SG mol.1508.ZN",
        "bond mol.1472.SG mol.1508.ZN",
    ]
    assert amber.zinc_bond_commands(isolated_atoms, assembly="isolated") == [
        "bond mol.323.SG mol.430.ZN",
        "bond mol.326.SG mol.430.ZN",
        "bond mol.391.SG mol.430.ZN",
        "bond mol.394.SG mol.430.ZN",
    ]
    assert [row["one_based_residue_order_command"] for row in amber.zinc_bond_selectors(isolated_atoms, assembly="isolated")] == [
        "bond mol.261.SG mol.368.ZN",
        "bond mol.264.SG mol.368.ZN",
        "bond mol.329.SG mol.368.ZN",
        "bond mol.332.SG mol.368.ZN",
    ]


def test_cli_returns_nonzero_when_current_tleap_fails(tmp_path, monkeypatch, capsys):
    capped = capped_fixture(tmp_path)
    config, ensemble, diffvec = q_inputs(tmp_path / "q", capped)
    prep, frcmod = metal_sources(tmp_path / "metal")

    def fake_run_tleap(input_file, workdir, executable):
        return {
            "command": [executable, "-f", str(input_file)],
            "returncode": 0,
            "status": "failed",
            "log": {"combined": str(workdir / "fake.log")},
            "failures": ["LEaP reported Errors = 1"],
        }

    monkeypatch.setattr(amber, "run_tleap", fake_run_tleap)
    monkeypatch.setattr(amber.shutil, "which", lambda _: "/fake/tleap")
    code = amber.main(
        [
            "--offline",
            "--input-pdb",
            str(capped),
            "--config",
            str(config),
            "--prep",
            str(prep),
            "--frcmod",
            str(frcmod),
            "--ensemble",
            str(ensemble),
            "--diffvec",
            str(diffvec),
            "--output-dir",
            str(tmp_path / "out"),
            "--tleap",
            "tleap",
        ]
    )

    assert code == 2
    assert '"status": "tleap_failed_mapping_blocked"' in capsys.readouterr().out
