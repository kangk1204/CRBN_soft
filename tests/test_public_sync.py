from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(f"public_sync_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


class Artifact(dict):
    @property
    def files(self):
        return list(self)


def valid_mobility_inputs(module):
    residues = np.array(sorted(module.DRUG + module.ZN), dtype=int)
    ensemble = Artifact(
        _confs=np.zeros((2, len(residues), 3), dtype=float),
        _labels=np.array([module.REF, "1ABC"]),
    )
    curation = [
        {"pdb": module.REF, "method": "X-ray"},
        {"pdb": "1ABC", "method": "cryo-EM"},
    ]
    fluctuations = [
        {"resnum": str(residue), "anm_sqfluct": str(index + 1.0)}
        for index, residue in enumerate(residues)
    ]
    return ensemble, residues, curation, fluctuations


def test_mobility_input_contract_requires_exact_residue_order_and_finite_coordinates():
    module = load_script("drug_loop_statistics")
    ensemble, residues, curation, fluctuations = valid_mobility_inputs(module)
    conformers, labels, checked_residues, methods, reference = module.validate_core_inputs(
        ensemble,
        residues,
        curation,
        fluctuations,
    )
    assert conformers.shape == (2, len(residues), 3)
    assert labels == [module.REF, "1ABC"]
    assert np.array_equal(checked_residues, residues)
    assert methods[module.REF] == "X-ray"
    assert np.array_equal(reference, np.arange(1.0, len(residues) + 1.0))

    reversed_rows = list(reversed(fluctuations))
    with pytest.raises(ValueError, match="residue order"):
        module.validate_core_inputs(ensemble, residues, curation, reversed_rows)

    nonfinite = Artifact(ensemble)
    nonfinite["_confs"] = ensemble["_confs"].copy()
    nonfinite["_confs"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module.validate_core_inputs(nonfinite, residues, curation, fluctuations)

    invalid_methods = [dict(row) for row in curation]
    invalid_methods[0]["method"] = "unknown"
    with pytest.raises(ValueError, match="structure method"):
        module.validate_core_inputs(
            ensemble,
            residues,
            invalid_methods,
            fluctuations,
        )


def test_mobility_output_and_rigid_null_outputs_use_strict_atomic_contracts():
    mobility = (SCRIPTS / "drug_loop_statistics.py").read_text(encoding="utf-8")
    rigid = (SCRIPTS / "assembly_rigid_null.py").read_text(encoding="utf-8")
    for source in (mobility, rigid):
        assert "atomic_write_json" in source
        assert "assert_tree_close(out, committed" in source
    assert "otherwise lowest exact-Q96SW2 auth-chain id" in mobility
    assert "otherwise highest CRBN-chain window coverage" not in mobility
    assert "B-factor profile skipped" not in mobility
    assert "L.CACHE_WRITES_ENABLED = not verify" in mobility
    assert "return R.fetch_cif(pdb)" in mobility
    assert 'expected_chain_counts = {"A": 1135, "B": 349}' in rigid
    assert rigid.index("validate_parsed_assembly(ca, window)") < rigid.index(
        "atomic_write_text(destination"
    )


def test_context_inputs_require_exact_residue_order():
    context = (SCRIPTS / "context_stats.py").read_text(encoding="utf-8")
    assert "residue fluctuation labels do not exactly match the analysis-window order" in context
    assert "open-reference coordinates do not exactly match the analysis-window order" in context


def test_rigid_null_requires_orthonormal_modes_and_an_exact_assembly_window():
    module = load_script("assembly_rigid_null")
    basis = np.eye(24)[:, :20]
    assert module.validate_mode_basis(basis, expected_rows=24).shape == (24, 20)
    nonorthogonal = basis.copy()
    nonorthogonal[:, 1] = nonorthogonal[:, 0]
    with pytest.raises(ValueError, match="orthonormal"):
        module.validate_mode_basis(nonorthogonal, expected_rows=24)

    tags = [("A", 1), ("B", 10), ("B", 11), ("B", 12)]
    assert module.ordered_window_indices(tags, np.array([10, 11, 12])) == [1, 2, 3]
    duplicate_and_missing = [("B", 10), ("B", 10), ("B", 12)]
    with pytest.raises(ValueError, match="does not exactly match"):
        module.ordered_window_indices(
            duplicate_and_missing,
            np.array([10, 11, 12]),
        )


def test_fresh_assembly_validation_runs_before_replacement():
    module = load_script("assembly_rigid_null")
    window = np.arange(10, 279)
    ca = {
        "A": {residue: np.ones(3) for residue in range(1, 1136)},
        "B": {
            **{residue: np.ones(3) for residue in window},
            **{residue: np.ones(3) for residue in range(1000, 1080)},
        },
    }
    module.validate_parsed_assembly(ca, window)
    ca["B"].pop(int(window[-1]))
    with pytest.raises(ValueError, match="chain counts"):
        module.validate_parsed_assembly(ca, window)


def test_verify_cannot_be_combined_with_assembly_replacement():
    module = load_script("assembly_rigid_null")
    with pytest.raises(SystemExit):
        module.parse_args(["--verify", "--write-assembly"])
    with pytest.raises(SystemExit):
        module.parse_args(["--unknown-option"])


def test_mobility_cli_rejects_unknown_options():
    module = load_script("drug_loop_statistics")
    args = module.parse_args(["--verify", "--no-network"])
    assert args.verify and args.no_network
    with pytest.raises(SystemExit):
        module.parse_args(["--unknown-option"])


@pytest.mark.parametrize("script", ["context_stats", "study_group_sensitivity"])
def test_analysis_clis_reject_unknown_options(script):
    module = load_script(script)
    with pytest.raises(SystemExit):
        module.parse_args(["--unknown-option"])
