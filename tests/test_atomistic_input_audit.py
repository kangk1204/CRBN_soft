import gzip
import json
from pathlib import Path

import pytest

from scripts import atomistic_input_audit as audit


def loop(category, headers, rows):
    lines = ["loop_"]
    lines.extend(f"_{category}.{header}" for header in headers)
    lines.extend(" ".join(str(value) for value in row) for row in rows)
    lines.append("#")
    return "\n".join(lines)


def atom_row(atom_id, atom_name, comp, label_chain, entity, label_seq, auth_seq, auth_chain, x, y, z, element=None, group="ATOM", ins="?"):
    element = element or ("ZN" if atom_name == "ZN" else atom_name[0])
    return [
        group,
        atom_id,
        element,
        atom_name,
        ".",
        comp,
        label_chain,
        entity,
        label_seq,
        ins,
        x,
        y,
        z,
        "1.00",
        "10.0",
        "?",
        auth_seq,
        comp,
        auth_chain,
        atom_name,
        "1",
    ]


def synthetic_cif(poly_rows=None, atom_rows=None, struct_rows=None):
    struct_rows = struct_rows or [
        ["1", "1", "TST1", "C", "1", "?", "5", "?", audit.CRBN_ACCESSION, "77", "?", "81", "?", "77", "81"],
        ["2", "2", "TST1", "A", "1", "?", "4", "?", audit.DDB1_ACCESSION, "1", "?", "4", "?", "1", "4"],
    ]
    poly_rows = poly_rows or [
        ["C", "1", "1", "CYS", "1", "77", "77", "CYS", "CYS", "C", ".", "n"],
        ["C", "1", "2", "ALA", "2", "78", "78", "ALA", "ALA", "C", ".", "n"],
        ["C", "1", "4", "GLY", "4", "80", "80", "GLY", "GLY", "C", ".", "n"],
        ["C", "1", "5", "SER", "5", "81", "81", "SER", "SER", "C", "A", "n"],
        ["A", "2", "1", "MET", "1", "1", "1", "MET", "MET", "A", ".", "n"],
        ["A", "2", "2", "ALA", "2", "2", "2", "ALA", "ALA", "A", ".", "n"],
        ["A", "2", "4", "GLY", "4", "4", "4", "GLY", "GLY", "A", ".", "n"],
    ]
    if atom_rows is None:
        atom_rows = []
        i = 1
        for auth, comp, label_seq in [("77", "CYS", "1"), ("78", "ALA", "2"), ("80", "GLY", "4"), ("81", "SER", "5")]:
            for atom in ["N", "CA", "C", "O"]:
                atom_rows.append(atom_row(i, atom, comp, "C", "1", label_seq, auth, "C", i, i + 1, i + 2))
                i += 1
        atom_rows.append(atom_row(i, "SG", "CYS", "C", "1", "1", "77", "C", 0, 0, 0, element="S")); i += 1
        atom_rows.append(atom_row(i, "ZN", "ZN", "Z", "3", ".", "1", "Z", 2.3, 0, 0, element="ZN", group="HETATM"))
    return "\n".join(
        [
            "data_TST1",
            loop(
                "struct_ref_seq",
                [
                    "align_id", "ref_id", "pdbx_PDB_id_code", "pdbx_strand_id", "seq_align_beg", "pdbx_seq_align_beg_ins_code",
                    "seq_align_end", "pdbx_seq_align_end_ins_code", "pdbx_db_accession", "db_align_beg", "pdbx_db_align_beg_ins_code",
                    "db_align_end", "pdbx_db_align_end_ins_code", "pdbx_auth_seq_align_beg", "pdbx_auth_seq_align_end",
                ],
                struct_rows,
            ),
            loop(
                "pdbx_poly_seq_scheme",
                ["asym_id", "entity_id", "seq_id", "mon_id", "ndb_seq_num", "pdb_seq_num", "auth_seq_num", "pdb_mon_id", "auth_mon_id", "pdb_strand_id", "pdb_ins_code", "hetero"],
                poly_rows,
            ),
            loop(
                "atom_site",
                [
                    "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv", "pdbx_formal_charge", "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id", "pdbx_PDB_model_num",
                ],
                atom_rows,
            ),
        ]
    )


def write_gz(path: Path, text: str):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_audit_distinguishes_core_construct_gap_coordinate_gap_insertion_and_ddb1_deletion(tmp_path):
    cif = tmp_path / "TST1.cif.gz"
    write_gz(cif, synthetic_cif())
    record = audit.audit_structure(cif, [77, 78, 79, 80, 81])
    assert record["identified_chains"] == {"CRBN_Q96SW2": ["C"], "DDB1_Q16531": ["A"]}
    crbn = next(row for row in record["chain_summaries"] if row["accession"] == audit.CRBN_ACCESSION)
    ddb1 = next(row for row in record["chain_summaries"] if row["accession"] == audit.DDB1_ACCESSION)
    assert crbn["core269"]["core_missing_from_construct"] == [{"start": 79, "end": 79, "length": 1}]
    assert crbn["core269"]["measurement_core_coordinates_complete"] is False
    assert crbn["core269"]["gapped_core269_md_allowed"] is False
    assert crbn["insertion_or_duplicate_mapping_count"] == 1
    assert {b["code"] for b in record["md_assembly_blockers"]} >= {"core269_missing_from_construct", "do_not_build_md_from_gapped_core269_pdb", "insertion_or_duplicate_uniprot_mapping"}
    assert ddb1["construct_deletions_vs_canonical"]["internal_missing"] == [{"start": 3, "end": 3, "length": 1}]


def test_nonmonotonic_author_numbering_is_blocker(tmp_path):
    poly_rows = [
        ["C", "1", "1", "ALA", "1", "77", "20", "ALA", "ALA", "C", ".", "n"],
        ["C", "1", "2", "ALA", "2", "78", "19", "ALA", "ALA", "C", ".", "n"],
    ]
    cif = tmp_path / "TST2.cif.gz"
    write_gz(cif, synthetic_cif(poly_rows=poly_rows))
    record = audit.audit_structure(cif, [77, 78])
    assert "nonmonotonic_author_numbering" in {b["code"] for b in record["md_assembly_blockers"]}



def test_coordinate_missing_uses_observed_ca_bounds_not_construct_bounds(tmp_path):
    poly_rows = []
    atom_rows = []
    atom_id = 1
    for residue in range(1, 443):
        poly_rows.append(["C", "1", str(residue), "GLY", str(residue), str(residue), str(residue), "GLY", "GLY", "C", ".", "n"])
        if 64 <= residue <= 428 and not (342 <= residue <= 357):
            for atom in ["N", "CA", "C", "O"]:
                atom_rows.append(atom_row(atom_id, atom, "GLY", "C", "1", str(residue), str(residue), "C", atom_id, atom_id, atom_id))
                atom_id += 1
    struct_rows = [["1", "1", "TST3", "C", "1", "?", "442", "?", audit.CRBN_ACCESSION, "1", "?", "442", "?", "1", "442"]]
    cif = tmp_path / "TST3.cif.gz"
    write_gz(cif, synthetic_cif(poly_rows=poly_rows, atom_rows=atom_rows, struct_rows=struct_rows))
    record = audit.audit_structure(cif, list(range(77, 346)))
    crbn = record["chain_summaries"][0]
    assert crbn["coordinate_missing_ca"]["terminal_missing"] == [
        {"start": 1, "end": 63, "length": 63},
        {"start": 429, "end": 442, "length": 14},
    ]
    assert crbn["coordinate_missing_ca"]["internal_missing"] == [{"start": 342, "end": 357, "length": 16}]


def test_multimodel_and_altloc_are_fail_closed_blockers(tmp_path):
    rows = [
        atom_row(1, "N", "ALA", "C", "1", "1", "77", "C", 1, 1, 1),
        atom_row(2, "CA", "ALA", "C", "1", "1", "77", "C", 2, 2, 2),
        atom_row(3, "CA", "ALA", "C", "1", "1", "77", "C", 3, 3, 3),
        atom_row(4, "C", "ALA", "C", "1", "1", "77", "C", 4, 4, 4),
        atom_row(5, "O", "ALA", "C", "1", "1", "77", "C", 5, 5, 5),
    ]
    rows[1][4] = "A"
    rows[2][4] = "B"
    rows[4][-1] = "2"
    cif = tmp_path / "TST4.cif.gz"
    write_gz(cif, synthetic_cif(poly_rows=[["C", "1", "1", "ALA", "1", "77", "77", "ALA", "ALA", "C", ".", "n"]], atom_rows=rows, struct_rows=[["1", "1", "TST4", "C", "1", "?", "1", "?", audit.CRBN_ACCESSION, "77", "?", "77", "?", "77", "77"]]))
    record = audit.audit_structure(cif, [77])
    blocker = next(b for b in record["md_assembly_blockers"] if b["code"] == "unsupported_multimodel_or_altloc_atom_site")
    codes = {example["code"] for example in blocker["examples"]}
    assert {"altloc_ambiguity", "additional_model_ignored"} <= codes


def test_offline_run_fails_on_missing_retained_cif(tmp_path):
    window = tmp_path / "crbn_residue_window.csv"
    window.write_text("index,author_resnum\n" + "".join(f"{i},{77+i}\n" for i in range(269)), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"refs":["MISSING"],"cif_cache":"%s","core_window":"%s"}' % (tmp_path, window), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="offline atomistic audit missing"):
        audit.run(config, tmp_path / "out", offline=True)


def test_loader_accepts_root_protocol_config_aliases_and_records_ignored_keys(tmp_path):
    window = tmp_path / "window.csv"
    config = tmp_path / "config.json"
    config.write_text(
        '{"references":["TST1"],"cif_dir":"data/_cif_cache","core_residue_file":"%s","core_position_count":5,"production_gate":{"x":1}}' % window,
        encoding="utf-8",
    )
    loaded = audit.load_config(config)
    assert loaded["refs"] == ["TST1"]
    assert loaded["core_window"] == str(window)
    assert loaded["core_position_count"] == 5
    assert "production_gate" in loaded["ignored_config_keys"]


def inventory_cif(missing_ddb=(), with_zinc=False, with_sg=True, invalid_core_ca=False):
    """Small coordinate inventory fixture, not a chemically qualified model."""
    struct = [
        ["1", "1", "TSTI", "C", "1", "?", "1", "?", audit.CRBN_ACCESSION, "77", "?", "77", "?", "77", "77"],
        ["2", "2", "TSTI", "A", "1", "?", "3", "?", audit.DDB1_ACCESSION, "1", "?", "3", "?", "1", "3"],
    ]
    poly = [["C", "1", "1", "GLY", "1", "77", "77", "GLY", "GLY", "C", ".", "n"]]
    poly += [["A", "2", str(i), "GLY", str(i), str(i), str(i), "GLY", "GLY", "A", ".", "n"] for i in (1, 2, 3)]
    atoms = []
    residues = [("C", "1", "1", "77")]
    residues += [("A", "2", str(i), str(i)) for i in (1, 2, 3) if i not in missing_ddb]
    for chain, entity, seq, auth in residues:
        for atom in ("N", "CA", "C", "O"):
            x = "nan" if invalid_core_ca and chain == "C" and atom == "CA" else len(atoms)
            atoms.append(atom_row(len(atoms)+1, atom, "GLY", chain, entity, seq, auth, chain, x, 0, 0))
    if with_zinc:
        atoms.append(atom_row(len(atoms)+1, "ZN", "ZN", "Z", "3", ".", "1", "Z", 0, 0, 0, element="ZN", group="HETATM"))
        if with_sg:
            for i, xyz in enumerate(((2.3, 0, 0), (-2.3, 0, 0), (0, 2.3, 0), (0, 0, 2.3))):
                atoms.append(atom_row(len(atoms)+1, "SG", "CYS", "C", "1", str(300+i), str(300+i), "C", *xyz, element="S"))
    return synthetic_cif(poly_rows=poly, atom_rows=atoms, struct_rows=struct)


def test_complete_measurement_core_does_not_hide_missing_zinc_or_noncore_gap(tmp_path):
    source = tmp_path / "TSTI.cif.gz"
    write_gz(source, inventory_cif(missing_ddb=(2,)))
    record = audit.audit_structure(source, [77])
    codes = {item["code"] for item in record["md_assembly_blockers"]}
    assert {"crbn_structural_zinc_coordinate_absent", "internal_missing_coordinates_require_repair", "do_not_build_md_from_gapped_core269_pdb"} <= codes
    assert record["coordinate_repairs_required"]
    assert record["chain_summaries"][0]["core269"]["measurement_core_coordinates_complete"]
    assert not record["chain_summaries"][0]["core269"]["gapped_core269_md_allowed"]
    assert "safe_to_build_269_gapped_pdb" not in record["chain_summaries"][0]["core269"]


def test_no_repair_inventory_never_grants_md_readiness(tmp_path):
    write_gz(tmp_path / "TSTI.cif.gz", inventory_cif(with_zinc=True))
    window = tmp_path / "core.csv"
    window.write_text("author_resnum\n77\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"references": ["TSTI"], "cif_dir": str(tmp_path), "core_residue_file": str(window), "core_position_count": 1}))
    result = audit.run(config, tmp_path / "inventory", offline=True)
    assert not result["coordinate_repairs_required"]
    assert not result["md_ready_without_repair"]
    assert not result["production_ready"]
    assert not result["gapped_core269_md_allowed"]
    assert result["assessment_scope"] == "input_coordinate_inventory_only_not_chemical_completeness_or_simulation_qualification"
    assert "measurement_core_coordinates_complete" in (tmp_path / "inventory/chain_summary.csv").read_text()


def test_missing_terminals_require_construct_choice_without_forcing_all_residues_to_be_filled(tmp_path):
    source = tmp_path / "TSTI.cif.gz"
    write_gz(source, inventory_cif(missing_ddb=(3,), with_zinc=True))
    result = audit.audit_structure(source, [77])
    assert result["terminal_construct_choice_required"]
    assert not result["coordinate_repairs_required"]
    assert "unobserved_terminals_require_construct_choice" in {item["code"] for item in result["md_assembly_blockers"]}


def test_zinc_without_nearby_sg_is_retained_and_blocked(tmp_path):
    source = tmp_path / "TSTI.cif.gz"
    write_gz(source, inventory_cif(with_zinc=True, with_sg=False))
    result = audit.audit_structure(source, [77])
    assert len(result["zinc_coordination"]) == 1
    assert result["zinc_coordination"][0]["sg_count_within_3a"] == 0
    assert "crbn_zinc_not_sg4_coordinated" in {item["code"] for item in result["md_assembly_blockers"]}


def test_observed_noncore_heavy_atom_gap_requires_repair(tmp_path):
    text = inventory_cif(with_zinc=True)
    lines = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 21 and fields[0] == "ATOM" and fields[3] == "O" and fields[16] == "1" and fields[18] == "A":
            continue
        lines.append(line)
    source = tmp_path / "TSTI.cif.gz"
    write_gz(source, "\n".join(lines))
    result = audit.audit_structure(source, [77])
    ddb1 = next(item for item in result["chain_summaries"] if item["accession"] == audit.DDB1_ACCESSION)
    assert ddb1["observed_incomplete_heavy_atom_residues"] == [{"start": 1, "end": 1, "length": 1}]
    assert ddb1["coordinate_repairs_required"]
    assert "observed_residue_incomplete_heavy_atoms_require_repair" in {item["code"] for item in result["md_assembly_blockers"]}


def test_nonfinite_ca_is_not_a_complete_measurement_coordinate(tmp_path):
    source = tmp_path / "TSTI.cif.gz"
    write_gz(source, inventory_cif(with_zinc=True, invalid_core_ca=True))
    result = audit.audit_structure(source, [77])
    assert not result["chain_summaries"][0]["core269"]["measurement_core_coordinates_complete"]
    assert result["coordinate_repairs_required"]
    assert "invalid_atom_coordinates" in {item["code"] for item in result["md_assembly_blockers"]}


def test_retained_8cvp_has_complete_measurement_ca_but_needs_full_construct_repairs():
    source = audit.DEFAULT_CIF_CACHE / "8CVP.cif.gz"
    if not source.is_file() or not audit.DEFAULT_CORE_WINDOW.is_file():
        pytest.skip("Retained 8CVP/core source files are not present in this checkout")
    result = audit.audit_structure(source, audit.read_core_window(audit.DEFAULT_CORE_WINDOW))
    chains = {item["accession"]: item for item in result["chain_summaries"]}
    assert chains[audit.CRBN_ACCESSION]["core269"]["measurement_core_coordinates_complete"]
    assert chains[audit.CRBN_ACCESSION]["coordinate_missing_ca"]["internal_missing"] == [{"start": 342, "end": 357, "length": 16}]
    assert chains[audit.DDB1_ACCESSION]["coordinate_missing_ca"]["internal_missing"] == [{"start": 546, "end": 550, "length": 5}]
    assert all(item["coordinate_repairs_required"] for item in chains.values())
    assert result["terminal_construct_choice_required"]
    assert all(item["sg4_coordination_ok"] for item in result["zinc_coordination"])
