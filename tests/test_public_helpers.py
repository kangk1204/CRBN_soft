from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pdb_id_validation_accepts_standard_ids():
    validator = load_script("pdb_id")
    assert validator.validate_pdb_id("8cvp") == "8CVP"
    assert validator.validate_pdb_id("5FQD") == "5FQD"


def test_pdb_id_validation_rejects_paths_urls_and_bad_lengths():
    validator = load_script("pdb_id")
    bad_values = ["../8CVP", "8CVP/extra", "abc", "ABCDE", "A B1", "https://x"]
    for value in bad_values:
        with pytest.raises(ValueError, match="PDB ID"):
            validator.validate_pdb_id(value)


def test_contact_pairs_returns_upper_triangle_contacts_only():
    lib = load_script("softmode_lib")
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    i, j, distances = lib.contact_pairs(coords, cutoff=1.5)
    assert i.tolist() == [0]
    assert j.tolist() == [1]
    assert distances.tolist() == [1.0]


def test_kabsch_superposition_recovers_rotated_points():
    lib = load_script("softmode_lib")
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    moved = reference @ np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    aligned = lib.kabsch_apply(moved + np.array([5.0, -2.0, 1.0]), reference)
    assert np.allclose(aligned, reference)


def test_exact_accession_chain_contract_ignores_description_mentions():
    contracts = load_script("curation_contracts")
    entry = {
        "polymer_entities": [
            {
                "rcsb_polymer_entity": {
                    "pdbx_description": "Partner involved in binding of cereblon"
                },
                "rcsb_polymer_entity_container_identifiers": {
                    "auth_asym_ids": ["C"],
                    "reference_sequence_identifiers": [
                        {"database_accession": "Q16531"}
                    ],
                },
            },
            {
                "rcsb_polymer_entity_container_identifiers": {
                    "auth_asym_ids": ["D", "B"],
                    "reference_sequence_identifiers": [
                        {"database_accession": "Q96SW2"}
                    ],
                },
            },
        ]
    }
    assert contracts.chains_for_exact_accession(entry, "Q96SW2") == ["B", "D"]


def test_primary_chain_contract_uses_override_else_lowest_id():
    contracts = load_script("curation_contracts")
    assert contracts.choose_primary_chain(["D", "B"], "9XYZ") == "B"
    assert contracts.choose_primary_chain(["D", "B"], "9XYZ", {"9XYZ": "D"}) == "D"
    with pytest.raises(ValueError, match="not among"):
        contracts.choose_primary_chain(["D", "B"], "9XYZ", {"9XYZ": "A"})


def test_construct_contract_uses_exact_uniprot_mapping():
    contracts = load_script("curation_contracts")
    entry = {
        "polymer_entities": [
            {
                "entity_poly": {"rcsb_sample_sequence_length": 405},
                "rcsb_polymer_entity": {
                    "pdbx_description": "Protein cereblon",
                    "pdbx_mutation": None,
                },
                "rcsb_polymer_entity_container_identifiers": {
                    "auth_asym_ids": ["A"],
                    "reference_sequence_identifiers": [
                        {"database_accession": "Q96SW2"}
                    ],
                },
            }
        ]
    }
    cif = """data_example
loop_
_struct_ref_seq.pdbx_db_accession
_struct_ref_seq.db_align_beg
_struct_ref_seq.db_align_end
Q96SW2 40 442
#
"""
    flags = contracts.exact_construct_flags(entry, cif)
    assert "CRBN_UniProt_mapping:40-442" in flags
    assert "CRBN_extra_sequence_or_tag" in flags


@pytest.mark.parametrize("start,end", [(0, 442), (-5, 100), (50, 49), (1, 443)])
def test_construct_contract_rejects_invalid_uniprot_intervals(start, end):
    contracts = load_script("curation_contracts")
    cif = f"""data_example
loop_
_struct_ref_seq.pdbx_db_accession
_struct_ref_seq.db_align_beg
_struct_ref_seq.db_align_end
Q96SW2 {start} {end}
#
"""
    with pytest.raises(ValueError, match="invalid Q96SW2 struct_ref_seq interval"):
        contracts.accession_ranges(cif, "Q96SW2")


def test_construct_contract_does_not_union_complementary_entities_into_false_wild_type():
    contracts = load_script("curation_contracts")
    entry = {
        "polymer_entities": [
            {
                "entity_poly": {"rcsb_sample_sequence_length": 221},
                "rcsb_polymer_entity": {"pdbx_mutation": None},
                "rcsb_polymer_entity_container_identifiers": {
                    "auth_asym_ids": [chain],
                    "reference_sequence_identifiers": [{"database_accession": "Q96SW2"}],
                },
            }
            for chain in ("A", "B")
        ]
    }
    cif = """data_example
loop_
_struct_ref_seq.pdbx_strand_id
_struct_ref_seq.pdbx_db_accession
_struct_ref_seq.db_align_beg
_struct_ref_seq.db_align_end
A Q96SW2 1 221
B Q96SW2 222 442
#
"""
    flags = contracts.exact_construct_flags(entry, cif)
    assert "CRBN_UniProt_mapping:1-221;222-442" in flags


def test_main_builders_execute_directory_setup_before_bundle_validation(tmp_path):
    builders = [
        "build_fig1.py",
        "build_fig2.py",
        "build_fig3.py",
        "build_fig4.py",
        "build_fig5_robustness.py",
    ]
    for name in builders:
        work = tmp_path / Path(name).stem
        work.mkdir()
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / name)],
            cwd=work,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode != 0, f"{name} unexpectedly ran without its data bundle"
        assert (work / "figures" / "vector").is_dir(), name
        assert (work / "figures" / "panels").is_dir(), name
        if name == "build_fig4.py":
            assert "pymol -cq scripts/render_fig4_pocket.py" in result.stderr


def test_figure_build_helpers_create_dirs_and_fail_with_generator_command(tmp_path):
    helpers = load_script("figure_package_utils")
    figures, vector, panels = helpers.prepare_figure_dirs(tmp_path)
    assert figures.is_dir() and vector.is_dir() and panels.is_dir()
    missing = panels / "missing.png"
    with pytest.raises(FileNotFoundError, match="pymol -cq scripts/render_fig4_pocket.py"):
        helpers.require_prepared_panel(
            missing,
            "pymol -cq scripts/render_fig4_pocket.py",
        )


def test_rigid_null_consumers_require_matched_subspace_schema():
    helpers = load_script("figure_package_utils")
    stale = {
        "rigid_domain_null": {
            "two_block_internal_dim": 6,
            "two_block_capture": 0.9,
            "three_block_internal_dim": 12,
            "three_block_capture": 0.95,
            "two_block": {
                "internal_dim": 6,
                "subspace_capture_of_transition": 0.9,
                "p_empirical": 0.03,
                "z": 2.0,
            },
            "three_block": {
                "internal_dim": 12,
                "subspace_capture_of_transition": 0.95,
                "p_empirical": 0.01,
                "z": 3.0,
            },
            "bond_length_preserving_boundary": {
                "internal_dim": 5,
                "subspace_capture_of_transition": 0.9,
                "p_empirical": 0.05,
                "z": 1.7,
            },
            "equal_displacement_boundary": {
                "internal_dim": 3,
                "subspace_capture_of_transition": 0.8,
                "p_empirical": 0.16,
                "z": 1.0,
            },
        }
    }
    with pytest.raises(RuntimeError, match=r"(?i)observed_projected_mode1_overlap.*rebuild"):
        helpers.require_rigid_null_schema(stale)

    for model in (
        "two_block",
        "three_block",
        "bond_length_preserving_boundary",
        "equal_displacement_boundary",
    ):
        stale["rigid_domain_null"][model].update(
            observed_direction_cosine_in_subspace=0.8,
            observed_projected_mode1_overlap=0.7,
        )
    assert helpers.require_rigid_null_schema(stale) is stale["rigid_domain_null"]


def test_negative_control_verification_requires_complete_named_set():
    panel = load_script("control_panel")
    expected = [{"name": "one"}, {"name": "two"}]
    panel.validate_negative_control_results([{"name": "two"}, {"name": "one"}], expected)
    with pytest.raises(AssertionError, match=r"missing=\['two'\]"):
        panel.validate_negative_control_results([{"name": "one"}], expected)

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "control_panel.py"), "--verify", "--skip-negative"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "--verify requires the complete negative-control panel" in result.stderr


def test_study_group_resolution_is_fail_closed_for_missing_dois(tmp_path):
    groups = load_script("study_groups")
    overrides = tmp_path / "overrides.csv"
    overrides.write_text(
        "pdb,study_group,reason\n"
        "1AAA,no_doi_series:one,shared deposition series\n",
        encoding="utf-8",
    )
    resolved = groups.resolve_study_groups(
        {"1AAA": "NO_DOI:1AAA", "2BBB": "10.1234/example"},
        ["1AAA", "2BBB"],
        overrides,
    )
    assert resolved == {
        "1AAA": "no_doi_series:one",
        "2BBB": "10.1234/example",
    }
    with pytest.raises(ValueError, match="missing primary DOI"):
        groups.resolve_study_groups(
            {"3CCC": "NO_DOI:3CCC"},
            ["3CCC"],
            overrides,
        )
    with pytest.raises(ValueError, match="duplicate labels"):
        groups.resolve_study_groups(
            {"1AAA": "NO_DOI:1AAA"},
            ["1AAA", "1AAA"],
            overrides,
        )


def test_frozen_study_group_snapshot_resolves_to_38_curated_groups():
    groups = load_script("study_groups").load_study_groups()
    counts = Counter(groups.values())
    assert len(groups) == 70
    assert len(counts) == 38
    assert groups["9H59"] == "10.1101/2024.11.06.622079"
    assert counts["no_doi_series:9sq4_9sq6"] == 3
    assert counts["no_doi_series:9uum_9v0f"] == 4


def test_boundary_nulls_distinguish_bond_length_from_equal_displacement():
    nulls = load_script("assembly_rigid_null")
    basis = np.eye(6)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    equal = nulls.equal_displacement_subspace(basis, 0, 1)
    bond = nulls.bond_length_preserving_subspace(basis, coords, 0, 1)
    assert equal.shape == (6, 3)
    assert bond.shape == (6, 5)
    for field in bond.T:
        displacement = field.reshape(2, 3)
        assert abs(displacement[1, 0] - displacement[0, 0]) < 1e-12
    with pytest.raises(ValueError, match="distinct coordinates"):
        nulls.bond_length_preserving_subspace(
            basis,
            np.zeros((2, 3)),
            0,
            1,
        )


def test_rigid_null_producer_uses_matched_subspace_statistic():
    source = (SCRIPTS / "assembly_rigid_null.py").read_text(encoding="utf-8")
    assert "mode_unit = mode_coeff / mode_content" in source
    assert "axis_unit = axis_coeff / axis_capture" in source
    assert "observed_direction = float(abs(mode_unit @ axis_unit))" in source
    assert "p_value(null_values, observed_direction)" in source
    assert '"observed_direction_cosine_in_subspace"' in source
    assert "p_value(null_vals)" not in source


@pytest.mark.parametrize(
    "description",
    ["DDB1", "DNA damage-binding protein 1", "DDB1 (DNA damage binding protein 1)"],
)
def test_ddb1_description_census_accepts_rcsb_naming_variants(description):
    contracts = load_script("curation_contracts")
    entry = {
        "polymer_entities": [
            {"rcsb_polymer_entity": {"pdbx_description": description}}
        ]
    }
    assert contracts.describes_ddb1(entry)
    entry["polymer_entities"][0]["rcsb_polymer_entity"]["pdbx_description"] = "Cereblon"
    assert not contracts.describes_ddb1(entry)
