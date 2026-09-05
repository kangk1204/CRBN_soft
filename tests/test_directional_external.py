from __future__ import annotations

import csv
from pathlib import Path

from scripts import directional_external as subject


def synthetic_cif() -> str:
    return """data_9SFM
#
loop_
_struct_ref_seq.align_id
_struct_ref_seq.ref_id
_struct_ref_seq.pdbx_PDB_id_code
_struct_ref_seq.pdbx_strand_id
_struct_ref_seq.seq_align_beg
_struct_ref_seq.pdbx_seq_align_beg_ins_code
_struct_ref_seq.seq_align_end
_struct_ref_seq.pdbx_seq_align_end_ins_code
_struct_ref_seq.pdbx_db_accession
_struct_ref_seq.db_align_beg
_struct_ref_seq.pdbx_db_align_beg_ins_code
_struct_ref_seq.db_align_end
_struct_ref_seq.pdbx_db_align_end_ins_code
_struct_ref_seq.pdbx_auth_seq_align_beg
_struct_ref_seq.pdbx_auth_seq_align_end
1 1 9SFM B 1 ? 382 ? Q96SW2 46 ? 427 ? 46 427
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
ATOM 1 C CA . GLY X 1 1 ? 0.0 0.0 0.0 1.00 10.0 ? 190 GLY B CA
ATOM 2 C CB A GLY X 1 1 ? 1.0 0.0 0.0 0.50 10.0 ? 190 GLY B CB
ATOM 3 C CB B GLY X 1 1 ? 99.0 0.0 0.0 0.50 10.0 ? 190 GLY B CB
ATOM 4 O O . GLY X 1 1 ? 2.0 0.0 0.0 1.00 10.0 ? 190 GLY B O
ATOM 5 C CA . ALA X 1 2 ? 10.0 0.0 0.0 1.00 10.0 ? 326 ALA B CA
ATOM 6 C CA . SER X 1 3 ? 2.0 2.0 0.0 1.00 10.0 ? 339 SER B CA
HETATM 7 C C1 . A1CEG L 2 . ? 0.0 0.0 3.0 1.00 10.0 ? 504 A1CEG B C1
HETATM 8 N N1 . A1CEG L 2 . ? 2.0 0.0 0.0 1.00 10.0 ? 504 A1CEG B N1
HETATM 9 H H1 . A1CEG L 2 . ? 0.0 0.0 0.0 1.00 10.0 ? 504 A1CEG B H1
#
"""


def test_structure_contacts_selects_altloc_a_and_maps_chain_b_to_uniprot() -> None:
    atom_rows, residue_rows, qa = subject.structure_contacts(synthetic_cif(), 4.5)
    independent = subject.independent_contact_residue_audit(synthetic_cif(), 4.5)

    residues = {row["uniprot_residue"] for row in residue_rows}
    assert residues == {190, 339}
    assert independent["contact_residues"] == [190, 339]
    assert qa["crbn_mapping"]["exact_author_to_uniprot_identity"] is True
    assert qa["ligand_auth_chains"] == ["B"]
    assert qa["altloc_tie_count"] == 1
    assert any(row["protein_atom"] == "CB" and row["distance_A"] == 1.0 for row in atom_rows)


def test_candidate_overlap_keeps_full_universe_and_stable_flags() -> None:
    candidates = [
        {
            "residue": "190",
            "contact_class": "CRBN_DDB1",
            "discovery_rank": "1",
            "discovery_D_g": "-0.2",
            "discovery_top5": "True",
            "stable_apo_model_candidate": "False",
            "also_consistent_in_engineered_references": "False",
        },
        {
            "residue": "326",
            "contact_class": "HB_TBD",
            "discovery_rank": "2",
            "discovery_D_g": "0.1",
            "discovery_top5": "True",
            "stable_apo_model_candidate": "True",
            "also_consistent_in_engineered_references": "True",
        },
    ]
    _, residue_rows, _ = subject.structure_contacts(synthetic_cif(), 4.5)

    overlap = subject.candidate_overlap_rows(candidates, residue_rows, synthetic_cif())

    by_residue = {row["residue"]: row for row in overlap}
    assert by_residue[190]["same_residue_as_A1CEG_contact"] is True
    assert by_residue[326]["same_residue_as_A1CEG_contact"] is False
    assert by_residue[326]["stable_apo_model_candidate"] == "True"


def test_blood_variant_rows_preserve_w415x_patient_symbol_and_no_w264_claim() -> None:
    candidates = [
        {"residue": "190", "contact_class": "CRBN_DDB1", "stable_apo_model_candidate": "False"},
        {"residue": "326", "contact_class": "HB_TBD", "stable_apo_model_candidate": "False"},
        {"residue": "339", "contact_class": "CRBN_DDB1", "stable_apo_model_candidate": "True"},
    ]

    rows = subject.blood_variant_rows(candidates)
    by_id = {row["variant_id"]: row for row in rows}

    assert len(rows) == 12
    assert by_id["W415G_experiment"]["patient_reported_symbol"] == "W415X"
    assert by_id["W415G_experiment"]["experimentally_tested_symbol"] == "W415G"
    assert by_id["L190F"]["candidate_overlap"] == "same_residue"
    assert by_id["C326G"]["candidate_overlap"] == "same_residue"
    assert by_id["C326G"]["functional_endpoint_type"] == "qualitative_from_table_and_figures"
    assert by_id["C326G"]["evidence_file"] == "supplemental PDF"
    assert by_id["C326G"]["evidence_page"] == 8
    assert by_id["C326G"]["evidence_article_page"] == 2636
    assert by_id["C326G"]["evidence_table_or_figure"] == "Supplementary Table 3"
    assert by_id["C326G"]["binding_endpoint_type"] == "not_variant_resolved_in_retrieved_main_pdf"
    assert not any("W264" in str(value) for row in rows for value in row.values())


def test_oconnor_reuse_reports_no_exact_candidate_overlap(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    with (external / "oconnor_variant_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "residues", "primary_269_window", "source_tables"],
        )
        writer.writeheader()
        writer.writerow({"variant": "Q100A", "residues": "100", "primary_269_window": "inside", "source_tables": "S4"})

    rows = subject.oconnor_reuse_rows(external, [{"residue": "339", "contact_class": "CRBN_DDB1", "stable_apo_model_candidate": "True"}])

    assert rows[0]["candidate_overlap"] == "none"
    assert rows[0]["stable_apo_candidate_overlap"] is False


def test_resolve_candidate_path_prefers_staged_bundle(tmp_path: Path) -> None:
    staged = tmp_path / "data" / "directional_reference_inputs" / "candidate_universe.csv"
    staged.parent.mkdir(parents=True)
    staged.write_text("residue,contact_class\n190,CRBN_DDB1\n", encoding="utf-8")
    private = tmp_path / "private.csv"
    private.write_text("residue,contact_class\n326,HB_TBD\n", encoding="utf-8")

    resolved = subject.resolve_candidate_path(tmp_path, {"external": {"candidate_robustness_csv": str(private)}})

    assert resolved == staged


def test_load_candidate_rows_merges_staged_legacy_robustness(tmp_path: Path) -> None:
    universe = tmp_path / "candidate_universe.csv"
    universe.write_text("residue,contact_class,D_g,rank\n339,CRBN_DDB1,-0.1,1\n", encoding="utf-8")
    universe.with_name("legacy_robustness.csv").write_text(
        "residue,contact_class,stable_apo_model_candidate,also_consistent_in_engineered_references\n"
        "339,CRBN_DDB1,True,False\n",
        encoding="utf-8",
    )

    rows = subject.load_candidate_rows(universe)

    assert rows[0]["stable_apo_model_candidate"] == "True"
    assert rows[0]["D_g"] == "-0.1"


def test_copy_cached_uses_existing_destination_when_source_missing(tmp_path: Path) -> None:
    dst = tmp_path / "data" / "OConnor_118_variant_inventory.csv"
    dst.parent.mkdir(parents=True)
    dst.write_text("variant,residues\nWT,\n", encoding="utf-8")

    record = subject.copy_cached(tmp_path / "missing.csv", dst, "oconnor_118_variant_inventory")

    assert record.status == "cached_local_copy"
    assert record.bytes == dst.stat().st_size
