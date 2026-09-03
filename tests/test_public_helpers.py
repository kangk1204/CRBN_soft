from __future__ import annotations

import csv
import importlib.util
import shutil
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


def copy_scripts_only(tmp_path: Path) -> Path:
    isolated_root = tmp_path / "release_source"
    shutil.copytree(
        SCRIPTS,
        isolated_root / "scripts",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return isolated_root


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


@pytest.mark.parametrize(
    ("pdb_id", "sample_length", "deleted_residues", "expected_ranges"),
    [
        ("9DWW", 839, range(396, 700), "396-699"),
        (
            "9SAF",
            836,
            [
                *range(396, 461),
                *range(463, 686),
                *range(687, 696),
                *range(698, 702),
                *range(703, 706),
            ],
            "396-460;463-685;687-695;698-701;703-705",
        ),
    ],
)
def test_construct_contract_retains_q16531_deletion_rows(
    pdb_id, sample_length, deleted_residues, expected_ranges
):
    contracts = load_script("curation_contracts")
    entry = {
        "polymer_entities": [
            {
                "entity_poly": {"rcsb_sample_sequence_length": 442},
                "rcsb_polymer_entity": {"pdbx_mutation": None},
                "rcsb_polymer_entity_container_identifiers": {
                    "auth_asym_ids": ["B"],
                    "reference_sequence_identifiers": [
                        {"database_accession": "Q96SW2"}
                    ],
                },
            },
            {
                "entity_poly": {"rcsb_sample_sequence_length": sample_length},
                "rcsb_polymer_entity": {"pdbx_mutation": None},
                "rcsb_polymer_entity_container_identifiers": {
                    "auth_asym_ids": ["A"],
                    "reference_sequence_identifiers": [
                        {"database_accession": "Q16531"}
                    ],
                },
            },
        ]
    }
    difference_rows = "\n".join(
        f"2 Q16531 {residue} deletion" for residue in deleted_residues
    )
    cif = f"""data_{pdb_id}
loop_
_struct_ref_seq.pdbx_strand_id
_struct_ref_seq.pdbx_db_accession
_struct_ref_seq.db_align_beg
_struct_ref_seq.db_align_end
B Q96SW2 1 442
A Q16531 1 1140
#
loop_
_struct_ref_seq_dif.align_id
_struct_ref_seq_dif.pdbx_seq_db_accession_code
_struct_ref_seq_dif.pdbx_seq_db_seq_num
_struct_ref_seq_dif.details
{difference_rows}
#
"""
    flags = contracts.exact_construct_flags(entry, cif)
    assert flags == f"DDB1_struct_ref_seq_dif_deletion:{expected_ranges}"


def test_construct_contract_retains_8u15_q16531_mapping_unavailable_flag():
    contracts = load_script("curation_contracts")
    entry = {
        "polymer_entities": [
            {
                "entity_poly": {"rcsb_sample_sequence_length": 373},
                "rcsb_polymer_entity": {"pdbx_mutation": None},
                "rcsb_polymer_entity_container_identifiers": {
                    "auth_asym_ids": ["A"],
                    "reference_sequence_identifiers": [
                        {"database_accession": "Q96SW2"}
                    ],
                },
            },
            {
                "entity_poly": {"rcsb_sample_sequence_length": 836},
                "rcsb_polymer_entity": {"pdbx_mutation": None},
                "rcsb_polymer_entity_container_identifiers": {
                    "auth_asym_ids": ["B"],
                    "reference_sequence_identifiers": [
                        {"database_accession": "Q16531"}
                    ],
                },
            },
        ]
    }
    cif = """data_8U15
loop_
_struct_ref_seq.pdbx_strand_id
_struct_ref_seq.pdbx_db_accession
_struct_ref_seq.db_align_beg
_struct_ref_seq.db_align_end
A Q96SW2 70 442
B 8U15 1 1140
#
"""
    flags = contracts.exact_construct_flags(entry, cif)
    assert flags == (
        "CRBN_UniProt_mapping:70-442;DDB1_exact_Q16531_mapping_unavailable"
    )


def test_main_builders_are_root_relative_and_do_not_write_to_the_calling_directory(tmp_path):
    isolated_root = copy_scripts_only(tmp_path)
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
            [sys.executable, str(isolated_root / "scripts" / name)],
            cwd=work,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode != 0, f"{name} unexpectedly ran without its data bundle"
        assert not (work / "figures").exists(), name
        assert str(isolated_root) in result.stderr, name


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
            "n_draws": 0,
            "seed": None,
            "directional_null_note": (
                "Directional nulls are exact analytic distributions; n_draws=0 and seed=null."
            ),
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
    for model in (
        "two_block",
        "three_block",
        "bond_length_preserving_boundary",
        "equal_displacement_boundary",
    ):
        record = stale["rigid_domain_null"][model]
        record.update(
            p_exact=record["p_empirical"],
            p_empirical_note=(
                "Deprecated compatibility alias for p_exact; no empirical draws were used."
            ),
            null_method="exact_analytic_beta",
            null_distribution={
                "statistic": "absolute_direction_cosine",
                "squared_statistic": "Beta(alpha, beta)",
                "alpha": 0.5,
                "beta": (record["internal_dim"] - 1) / 2,
            },
            null_mean=0.2,
            null_sd=0.1,
            null_p95=0.5,
            null_max=1.0,
        )
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

    stale["rigid_domain_null"]["two_block"]["p_empirical"] += 0.01
    with pytest.raises(RuntimeError, match=r"p_empirical alias differs from p_exact"):
        helpers.require_rigid_null_schema(stale)
    stale["rigid_domain_null"]["two_block"]["p_empirical"] = stale[
        "rigid_domain_null"
    ]["two_block"]["p_exact"]

    stale["rigid_domain_null"]["n_draws"] = 200_000
    with pytest.raises(RuntimeError, match=r"still advertises a sampled directional null"):
        helpers.require_rigid_null_schema(stale)


def test_rigid_null_figure_and_table_consumers_use_exact_distribution():
    figure = (SCRIPTS / "build_fig5_robustness.py").read_text(encoding="utf-8")
    tables = (SCRIPTS / "build_tables.py").read_text(encoding="utf-8")
    assert '"p_exact"' in figure
    assert '"p_empirical"' not in figure
    assert "Exact-null density" in figure
    assert "exact_null_density" in figure
    # The point of this check is that every reported p value comes from the exact
    # directional distribution, which is what "p_empirical" would violate. Banning the
    # substring "n_draws" outright also caught a legitimate non-p diagnostic: the notes
    # report what fraction of draws were as chain-continuous as the observed transition,
    # which is why the unconstrained nulls are labelled upper bounds. The ban is narrowed
    # to the quantity it was aimed at.
    assert "p_empirical" not in tables
    assert "p_exact" in tables


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
    module = load_script("study_groups")
    groups = module.load_study_groups()
    counts = Counter(groups.values())
    assert len(groups) == 70
    assert len(counts) == 38
    assert groups["9H59"] == "10.1101/2024.11.06.622079"
    assert counts["10.1038/nsmb.2874"] == 1
    assert counts["10.1038/s41467-024-44698-1"] == 1
    assert counts["10.1038/s41587-026-03237-7"] == 3
    assert counts["10.1101/2025.06.08.658527"] == 4

    with module.GROUP_TABLE.open(encoding="utf-8", newline="") as handle:
        snapshot = {
            row["pdb"]: row["primary_citation_doi"]
            for row in csv.DictReader(handle)
        }
    for pdb in ["4TZ4", "8G66", "9SQ4", "9SQ5", "9SQ6", "9UUM", "9V0A", "9V0B", "9V0F"]:
        assert snapshot[pdb].startswith("no_doi:")


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


def test_exact_abs_cosine_null_matches_closed_form_in_three_dimensions():
    nulls = load_script("assembly_rigid_null")
    observed = 0.8
    result = nulls.analytic_abs_cosine_null(3, observed)
    assert result["p_exact"] == pytest.approx(1.0 - observed, abs=1e-14)
    assert result["p_empirical"] == result["p_exact"]
    assert result["null_mean"] == pytest.approx(0.5, abs=1e-14)
    assert result["null_sd"] == pytest.approx(np.sqrt(1.0 / 12.0), abs=1e-14)
    assert result["null_p95"] == pytest.approx(0.95, abs=1e-14)
    assert result["null_max"] == 1.0
    assert result["null_method"] == "exact_analytic_beta"
    assert result["null_distribution"] == {
        "statistic": "absolute_direction_cosine",
        "squared_statistic": "Beta(alpha, beta)",
        "alpha": 0.5,
        "beta": 1.0,
    }


def test_exact_abs_cosine_null_matches_closed_form_in_two_dimensions():
    nulls = load_script("assembly_rigid_null")
    observed = 0.6
    result = nulls.analytic_abs_cosine_null(2, observed)
    assert result["p_exact"] == pytest.approx(
        2.0 * np.arccos(observed) / np.pi,
        abs=1e-14,
    )
    assert result["null_mean"] == pytest.approx(2.0 / np.pi, abs=1e-14)
    assert result["null_sd"] == pytest.approx(
        np.sqrt(0.5 - (2.0 / np.pi) ** 2),
        abs=1e-14,
    )
    assert result["null_p95"] == pytest.approx(np.sin(0.95 * np.pi / 2.0), abs=1e-14)


def test_exact_abs_cosine_null_rejects_invalid_inputs():
    nulls = load_script("assembly_rigid_null")
    for dimension in (True, 1, 1.5):
        with pytest.raises((TypeError, ValueError)):
            nulls.analytic_abs_cosine_null(dimension, 0.5)
    for observed in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            nulls.analytic_abs_cosine_null(3, observed)


def test_rigid_null_producer_uses_matched_subspace_exact_beta_statistic():
    source = (SCRIPTS / "assembly_rigid_null.py").read_text(encoding="utf-8")
    assert "mode_unit = mode_coeff / mode_content" in source
    assert "axis_unit = axis_coeff / axis_capture" in source
    assert "observed_direction = float(abs(mode_unit @ axis_unit))" in source
    direction_null = source.split("def direction_null", 1)[1].split(
        "# Both parameterisations", 1
    )[0]
    assert "analytic_abs_cosine_null" in direction_null
    assert "default_rng" not in direction_null
    assert "standard_normal" not in direction_null
    assert "betaincc" in source
    assert '"p_exact"' in source
    assert '"null_method": "exact_analytic_beta"' in source
    assert '"n_draws": 0' in source
    assert '"seed": None' in source
    assert '"method": "full_space_gaussian_projection_monte_carlo"' in source
    assert '"p_random_rigid_direction_note"' in source
    assert "tol=tolerance" in source
    assert "1e-10" in source
    assert '"observed_direction_cosine_in_subspace"' in source
    assert "def p_value" not in source


def test_junction_continuity_draws_are_invariant_to_svd_basis_rotation():
    nulls = load_script("assembly_rigid_null")
    rng = np.random.default_rng(20260829)
    basis, _ = np.linalg.qr(rng.standard_normal((15, 4)))
    rotation, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    first = nulls.projected_uniform_directions(basis, 50, seed=123)
    rotated = nulls.projected_uniform_directions(basis @ rotation, 50, seed=123)
    assert np.allclose(first, rotated, atol=1e-12)


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
