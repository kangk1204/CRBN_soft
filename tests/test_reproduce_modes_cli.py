from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import warnings
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reproduce_modes.py"


def load_module():
    spec = importlib.util.spec_from_file_location("reproduce_modes_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_reference(module):
    return {
        "anm_diff_overlap": np.zeros(module.N_MODES),
        "cum_overlap": np.zeros(module.N_MODES),
        "anm_eigvals": np.ones(module.N_MODES),
        "anm_eigvecs": np.eye(3 * 269, module.N_MODES),
        "rmsip": np.asarray(0.5),
        "resnums": np.arange(1, 270),
        "overlap_anm_pca": np.zeros((10, 10)),
    }


def test_cli_defaults_and_external_source_write_guard():
    module = load_module()
    args = module.parse_args([])
    assert not args.verify and args.data_source is None
    args = module.parse_args(["--verify", "--data-source", "/tmp/example.zip"])
    assert args.verify and args.data_source == Path("/tmp/example.zip")
    with pytest.raises(SystemExit):
        module.parse_args(["--data-source", "/tmp/example.zip"])
    with pytest.raises(SystemExit):
        module.parse_args(["--verfiy"])
    with pytest.raises(SystemExit):
        module.parse_args(["--unknown-option"])


def test_help_exits_before_data_access(capsys):
    module = load_module()
    with pytest.raises(SystemExit) as error:
        module.parse_args(["--help"])
    assert error.value.code == 0
    assert "--data-source" in capsys.readouterr().out


def test_directory_source_preflight_lists_all_missing_inputs(tmp_path):
    module = load_module()
    source = module.AnalysisDataSource.open(tmp_path)
    (tmp_path / module.ENSEMBLE_NAME).write_bytes(b"placeholder")
    with pytest.raises(ValueError) as error:
        source.preflight(module.VERIFY_INPUTS)
    assert str(error.value).endswith(
        "missing required input(s): crbn_residue_window.csv, crbn_anm_modes.npz"
    )


def test_zip_source_reads_exact_members_without_extraction(tmp_path):
    module = load_module()
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in module.VERIFY_INPUTS:
            archive.writestr(f"data/{name}", name.encode())
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    before_stat = path.stat()

    source = module.AnalysisDataSource.open(path)
    source.preflight(module.VERIFY_INPUTS)
    assert source.read_text(module.WINDOW_NAME) == module.WINDOW_NAME
    assert not (tmp_path / "data").exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    after_stat = path.stat()
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


def test_zip_source_rejects_wrong_layout_duplicate_members_and_invalid_files(tmp_path):
    module = load_module()
    wrong = tmp_path / "wrong.zip"
    with zipfile.ZipFile(wrong, "w") as archive:
        for name in module.VERIFY_INPUTS:
            archive.writestr(name, b"wrong root")
    with pytest.raises(ValueError, match="data/crbn_ensemble.ens.npz"):
        module.AnalysisDataSource.open(wrong).preflight(module.VERIFY_INPUTS)

    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr(f"data/{module.ENSEMBLE_NAME}", b"first")
            archive.writestr(f"data/{module.ENSEMBLE_NAME}", b"second")
    with pytest.raises(ValueError, match="duplicate required ZIP member"):
        module.AnalysisDataSource.open(duplicate).preflight((module.ENSEMBLE_NAME,))

    invalid = tmp_path / "invalid.zip"
    invalid.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="not a readable ZIP"):
        module.AnalysisDataSource.open(invalid)


def test_reference_contract_is_complete_finite_and_shape_strict():
    module = load_module()
    resnums = np.arange(1, 270)
    reference = valid_reference(module)
    checked = module.validate_reference(reference, resnums, "test reference")
    assert checked["anm_eigvecs"].shape == (807, 20)

    missing = valid_reference(module)
    del missing["anm_diff_overlap"]
    with pytest.raises(ValueError, match="invalid numeric reference key anm_diff_overlap"):
        module.validate_reference(missing, resnums, "test reference")

    wrong_shape = valid_reference(module)
    wrong_shape["anm_eigvecs"] = np.eye(806, module.N_MODES)
    with pytest.raises(ValueError, match="anm_eigvecs shape"):
        module.validate_reference(wrong_shape, resnums, "test reference")

    nonfinite = valid_reference(module)
    nonfinite["cum_overlap"][0] = np.nan
    with pytest.raises(ValueError, match="contains non-finite"):
        module.validate_reference(nonfinite, resnums, "test reference")

    nonorthogonal = valid_reference(module)
    nonorthogonal["anm_eigvecs"][:, 1] = nonorthogonal["anm_eigvecs"][:, 0]
    with pytest.raises(ValueError, match="not an orthonormal basis"):
        module.validate_reference(nonorthogonal, resnums, "test reference")


def test_verification_does_not_use_optimizable_assert_statements():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]


def test_preflight_failure_occurs_before_numerical_work(tmp_path, monkeypatch):
    module = load_module()

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("numerical work started")

    monkeypatch.setattr(module, "pca", unexpected_call)
    with pytest.raises(SystemExit, match="missing required input"):
        module.main(["--verify", "--data-source", str(tmp_path)])
