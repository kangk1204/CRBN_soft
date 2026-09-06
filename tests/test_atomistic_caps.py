import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts import atomistic_caps as caps

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "tests" / "fixtures" / "atomistic_caps"


def write_pdb(path: Path, atoms):
    with path.open("w", encoding="utf-8") as handle:
        for i, atom in enumerate(atoms, 1):
            handle.write(caps.format_atom(atom, i) + "\n")
        handle.write("END\n")


def transformed_middle_atoms(template_path: Path, middle_resname: str, chain: str, resseq: int):
    _, atoms = caps.read_pdb(template_path)
    _ace, mid, _nme = caps.template_residue_groups(atoms, middle_resname)
    return [caps.replace(a, chain=chain, resseq=resseq, resname=middle_resname, source="synthetic_input") for a in mid if not a.is_hydrogen]


def parse_output_atoms(path: Path):
    _, atoms = caps.read_pdb(path)
    return atoms


def atom_lookup(atoms):
    return {(a.chain, a.resseq, a.resname, a.name): a for a in atoms}


def test_leap_templates_exist_and_contain_expected_cap_and_middle_residues():
    for filename, middle in [("ace_met_nme.pdb", "MET"), ("ace_asp_nme.pdb", "ASP")]:
        path = TEMPLATE_DIR / filename
        assert path.is_file()
        _, atoms = caps.read_pdb(path)
        ace, mid, nme = caps.template_residue_groups(atoms, middle)
        assert {a.resname for a in ace} == {"ACE"}
        assert {a.resname for a in mid} == {middle}
        assert {a.resname for a in nme} == {"NME"}
        assert caps.heavy_cap_atoms(ace)
        assert caps.heavy_cap_atoms(nme)


def test_add_caps_transfers_only_template_heavy_caps_preserves_input_identity(tmp_path):
    n_template = TEMPLATE_DIR / "ace_met_nme.pdb"
    c_template = TEMPLATE_DIR / "ace_asp_nme.pdb"
    ddb1 = [
        caps.AtomRecord("ATOM", 1, "N", "", "MET", "A", 1, "", -5, 0, 0, 1, 0, "N"),
        caps.AtomRecord("ATOM", 2, "CA", "", "MET", "A", 1, "", -4, 0, 0, 1, 0, "C"),
        caps.AtomRecord("ATOM", 3, "C", "", "ASP", "A", 1140, "", -3, 0, 0, 1, 0, "C"),
        caps.AtomRecord("HETATM", 4, "ZN", "", "ZN", "C", 501, "", 9, 9, 9, 1, 0, "ZN"),
    ]
    met64 = transformed_middle_atoms(n_template, "MET", "B", 64)
    asp428 = transformed_middle_atoms(c_template, "ASP", "B", 428)
    input_atoms = ddb1 + met64 + asp428
    input_pdb = tmp_path / "heavy.pdb"
    write_pdb(input_pdb, input_atoms)

    result = caps.add_caps(input_pdb, n_template, c_template, tmp_path / "out")
    out_atoms = parse_output_atoms(Path(result["capped_pdb"]))
    lookup = atom_lookup(out_atoms)

    assert ("B", 63, "ACE", "C") in lookup
    assert ("B", 429, "NME", "N") in lookup
    assert not any(a.resname in {"ACE", "NME"} and a.chain == "A" for a in out_atoms)
    assert ("C", 501, "ZN", "ZN") in lookup
    assert not any(a.resname in {"ACE", "NME"} and a.is_hydrogen for a in out_atoms)

    original = atom_lookup(input_atoms)
    capped = {(a.chain, a.resseq, a.resname, a.name): a for a in out_atoms if a.resname not in {"ACE", "NME"}}
    for key, before in original.items():
        after = capped[key]
        np.testing.assert_allclose(after.xyz, before.xyz, atol=1e-6)

    provenance = json.loads(Path(result["provenance_json"]).read_text())
    assert provenance["terminal_model"]["hydrogens_transferred_from_templates"] is False
    assert provenance["terminal_model"]["ddb1_caps_added"] is False
    assert provenance["production_ready"] is False
    assert provenance["alignment"]["n_template_backbone_rmsd_angstrom"] < 0.15
    assert provenance["alignment"]["c_template_carbonyl_plane_rmsd_angstrom"] < 0.15
    assert 1.2 <= provenance["peptide_bond_distances_angstrom"]["ACE_C_to_residue_64_N"] <= 1.5
    assert 1.2 <= provenance["peptide_bond_distances_angstrom"]["residue_428_C_to_NME_N"] <= 1.5
    out_atoms = caps.read_pdb(Path(result["capped_pdb"]))[1]
    lookup = atom_lookup(out_atoms)
    nme_n = lookup[("B", 429, "NME", "N")]
    asp_o = lookup[("B", 428, "ASP", "O")]
    assert float(np.linalg.norm(nme_n.xyz - asp_o.xyz)) > 2.0


def test_bad_terminal_backbone_fails_rmsd_gate(tmp_path):
    n_template = TEMPLATE_DIR / "ace_met_nme.pdb"
    c_template = TEMPLATE_DIR / "ace_asp_nme.pdb"
    met64 = transformed_middle_atoms(n_template, "MET", "B", 64)
    distorted = []
    for atom in met64:
        if atom.name == "CA":
            atom = caps.replace(atom, x=atom.x + 0.8)
        distorted.append(atom)
    asp428 = transformed_middle_atoms(c_template, "ASP", "B", 428)
    input_pdb = tmp_path / "bad.pdb"
    write_pdb(input_pdb, distorted + asp428)
    with pytest.raises(ValueError, match="RMSD"):
        caps.add_caps(input_pdb, n_template, c_template, tmp_path / "out")


def test_cli_writes_expected_outputs(tmp_path):
    n_template = TEMPLATE_DIR / "ace_met_nme.pdb"
    c_template = TEMPLATE_DIR / "ace_asp_nme.pdb"
    input_atoms = transformed_middle_atoms(n_template, "MET", "B", 64) + transformed_middle_atoms(c_template, "ASP", "B", 428)
    input_pdb = tmp_path / "heavy.pdb"
    write_pdb(input_pdb, input_atoms)
    out = tmp_path / "cli_out"
    completed = subprocess.run(
        [
            "python3",
            "scripts/atomistic_caps.py",
            "--input-heavy-pdb",
            str(input_pdb),
            "--n-template-pdb",
            str(n_template),
            "--c-template-pdb",
            str(c_template),
            "--output-dir",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (out / "capped_heavy.pdb").is_file()
    assert (out / "cap_atom_mapping.csv").is_file()
    assert (out / "cap_provenance.json").is_file()
    assert json.loads(completed.stdout)["status"] in {"complete", "complete_with_warnings"}


def test_critical_cap_clash_below_one_angstrom_fails(tmp_path):
    n_template = TEMPLATE_DIR / "ace_met_nme.pdb"
    c_template = TEMPLATE_DIR / "ace_asp_nme.pdb"
    met64 = transformed_middle_atoms(n_template, "MET", "B", 64)
    asp428 = transformed_middle_atoms(c_template, "ASP", "B", 428)
    _lines, template_atoms = caps.read_pdb(n_template)
    ace, mid, _nme = caps.template_residue_groups(template_atoms, "MET")
    rot, trans, _rmsd = caps.kabsch_fit(
        caps.residue_backbone(mid, "N-template MET", caps.N_CAP_FIT_ATOMS),
        caps.residue_backbone(met64, "target B:64 MET", caps.N_CAP_FIT_ATOMS),
    )
    placed_ace_c = caps.transform_atom(caps.atom_by_name(ace, "C"), rot, trans, chain="B", resseq=63, resname="ACE", source="fixture")
    extra_clash = caps.AtomRecord("ATOM", 99, "CB", "", "ALA", "B", 100, "", placed_ace_c.x + 0.2, placed_ace_c.y, placed_ace_c.z, 1, 0, "C")
    input_pdb = tmp_path / "clash.pdb"
    write_pdb(input_pdb, met64 + [extra_clash] + asp428)
    with pytest.raises(ValueError, match="critical cap/input heavy-atom clash"):
        caps.add_caps(input_pdb, n_template, c_template, tmp_path / "out")


def test_actual_8cvp_terminal_backbones_pass_template_rmsd_bond_gates_and_avoids_carbonyl_clash(tmp_path):
    records = caps.read_pdb(TEMPLATE_DIR / "8cvp_terminal_atoms.pdb")[1]
    assert {a.resname for a in records if a.resseq == 64} == {"MET"}
    assert {a.resname for a in records if a.resseq == 428} == {"ASP"}
    input_pdb = tmp_path / "8cvp_terminals.pdb"
    write_pdb(input_pdb, records)

    result = caps.add_caps(input_pdb, TEMPLATE_DIR / "ace_met_nme.pdb", TEMPLATE_DIR / "ace_asp_nme.pdb", tmp_path / "out")
    provenance = json.loads(Path(result["provenance_json"]).read_text())
    assert provenance["alignment"]["n_template_backbone_rmsd_angstrom"] < 0.15
    assert provenance["alignment"]["c_template_carbonyl_plane_rmsd_angstrom"] < 0.15
    assert 1.2 <= provenance["peptide_bond_distances_angstrom"]["ACE_C_to_residue_64_N"] <= 1.5
    assert 1.2 <= provenance["peptide_bond_distances_angstrom"]["residue_428_C_to_NME_N"] <= 1.5
    out_atoms = caps.read_pdb(Path(result["capped_pdb"]))[1]
    lookup = atom_lookup(out_atoms)
    nme_n = lookup[("B", 429, "NME", "N")]
    asp_o = lookup[("B", 428, "ASP", "O")]
    assert float(np.linalg.norm(nme_n.xyz - asp_o.xyz)) > 2.0
