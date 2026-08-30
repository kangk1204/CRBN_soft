from __future__ import annotations

import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

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


def load_context_stats(monkeypatch):
    library = types.ModuleType("softmode_lib")
    library.functional_residues = lambda: {"drug": [1], "zinc": [2]}
    groups = types.ModuleType("study_groups")
    groups.load_study_groups = lambda labels: {label: label for label in labels}
    monkeypatch.setitem(sys.modules, "softmode_lib", library)
    monkeypatch.setitem(sys.modules, "study_groups", groups)
    saved_path = sys.path.copy()
    try:
        return load_script("context_stats")
    finally:
        sys.path[:] = saved_path


def test_circular_shift_test_includes_identity_and_validates_inputs(monkeypatch):
    module = load_context_stats(monkeypatch)
    values = np.array([10.0, 0.0, 0.0, 0.0])
    mask = np.array([True, False, False, False])

    observed, shifted, pvalue = module.circular_shift_pvalue(values, mask)

    assert shifted.shape == values.shape
    assert shifted[0] == observed
    assert pvalue == pytest.approx(0.25)
    with pytest.raises(ValueError, match="finite one-dimensional"):
        module.circular_shift_pvalue([1.0, np.nan], [True, False])
    with pytest.raises(ValueError, match="must be boolean"):
        module.circular_shift_pvalue([1.0, 2.0], [1, 0])
    with pytest.raises(ValueError, match="proper subset"):
        module.circular_shift_pvalue([1.0, 2.0], [True, True])


def test_study_group_fetch_uses_complete_atomic_csv_payload(tmp_path, monkeypatch):
    module = load_script("study_group_sensitivity")
    module.ROOT_D = str(tmp_path) + "/"
    labels = ["2DEF", "1ABC"]
    response_payload = {
        "data": {
            "entries": [
                {"rcsb_id": "2DEF", "rcsb_primary_citation": None},
                {
                    "rcsb_id": "1ABC",
                    "rcsb_primary_citation": {"pdbx_database_id_DOI": "10.1/ABC"},
                },
            ]
        }
    }

    class Response:
        entered = False
        exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *args):
            self.exited = True

        def read(self):
            return json.dumps(response_payload).encode()

    response = Response()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: response)
    monkeypatch.setattr(module, "resolve_study_groups", lambda mapping, order: mapping)

    resolved = module.fetch_dois(labels)

    assert response.entered and response.exited
    assert resolved == {"2DEF": "no_doi:2def", "1ABC": "10.1/abc"}
    assert (tmp_path / "curation_study_groups.csv").read_text(encoding="utf-8") == (
        "pdb,primary_citation_doi\n"
        "1ABC,10.1/abc\n"
        "2DEF,no_doi:2def\n"
    )


def test_target_scripts_enforce_exact_artifact_and_atomic_output_contracts():
    context = (SCRIPTS / "context_stats.py").read_text(encoding="utf-8")
    study = (SCRIPTS / "study_group_sensitivity.py").read_text(encoding="utf-8")
    window = (SCRIPTS / "window_sensitivity.py").read_text(encoding="utf-8")

    assert "validate_ensemble_diff(ens, dd)" in context
    assert "assert_tree_close(out, reference)" in context
    assert 'atomic_write_json(Path("data/context_stats.json"), out)' in context
    assert "validate_ensemble_diff(ens, dv)" in study
    assert "assert_tree_close(out, reference)" in study
    assert "atomic_write_json" in study
    assert "assert_tree_close(normalized, reference, float_tolerance=1e-9)" in window
    assert "validate_complete_generation(" in window
    assert "prepare_artifact_payloads(out)" in window
    assert "atomic_write_text(Path(csv_path), csv_payload)" in window
    assert "atomic_write_text(Path(json_path), json_payload)" in window


def test_window_sensitivity_rejects_partial_checks_and_missing_inputs():
    source = (SCRIPTS / "window_sensitivity.py").read_text(encoding="utf-8")
    main_source = source.split("def main():", 1)[1]

    assert 'raise ValueError("--limit must be positive")' in source
    assert "--verify requires the complete inventory and rejects --limit" in source
    assert "diagnostic partial results; canonical JSON/CSV left untouched" in source
    assert "missing RCSB metadata after resolution" in source
    assert "no coordinates were parsed for exact-Q96SW2 chains" in source
    assert "resolution unavailable; cannot establish the curated-set ceiling" in source
    assert main_source.index("load_bundle_configuration()") < main_source.index(
        'inventory_path = DATA / "crbn_structure_inventory.csv"'
    )
    assert 'json_path = DATA / "window_sensitivity.json"' in main_source
    assert 'csv_path = DATA / "excluded_structures_adjudication.csv"' in main_source
    assert main_source.index("validate_complete_generation(") < main_source.index(
        "finalize_artifacts("
    )


def test_window_module_defers_bundle_loading_and_uses_explicit_root(tmp_path, monkeypatch):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    module = load_script("window_sensitivity")
    assert module.WINSET == []
    assert module.CHAIN_MAP == {}

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    residues = list(range(77, 346))
    (bundle / "crbn_residue_window.csv").write_text(
        "index,author_resnum\n"
        + "".join(f"{index},{residue}\n" for index, residue in enumerate(residues)),
        encoding="utf-8",
    )
    (bundle / "curation_chain_map.json").write_text(
        '{"1ABC": "B"}\n', encoding="utf-8"
    )

    module.load_bundle_configuration(bundle)

    assert module.WINSET == residues
    assert module.CHAIN_MAP == {"1ABC": "B"}
    assert module.NTD_R == list(range(77, 187))
    assert module.HB_R == list(range(187, 318))
    assert module.TBD_R == list(range(318, 346))

    (bundle / "crbn_residue_window.csv").write_text(
        "index,author_resnum\n"
        + "".join(f"{index},{residue}\n" for index, residue in enumerate(residues[::-1])),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        module.load_bundle_configuration(bundle)
    (bundle / "crbn_residue_window.csv").write_text(
        "index,author_resnum\n"
        + "".join(f"{index},{residue}\n" for index, residue in enumerate(residues[:-1])),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly 269"):
        module.load_bundle_configuration(bundle)


def test_window_help_does_not_require_an_input_bundle(monkeypatch, capsys):
    module = load_script("window_sensitivity")
    monkeypatch.setattr(sys, "argv", ["window_sensitivity.py", "--help"])

    assert module.main() == 0
    assert "--limit N" in capsys.readouterr().out


def test_limit_mode_is_no_write_and_incompatible_with_verification(tmp_path):
    module = load_script("window_sensitivity")
    assert module.parse_run_options(["--limit", "3"]) == (False, 3)
    with pytest.raises(ValueError, match="complete inventory"):
        module.parse_run_options(["--verify", "--limit", "3"])

    json_path = tmp_path / "window.json"
    csv_path = tmp_path / "adjudication.csv"
    json_path.write_text("json sentinel\n", encoding="utf-8")
    csv_path.write_text("csv sentinel\n", encoding="utf-8")
    partial = {"nonfinite_diagnostic": float("nan")}

    assert module.finalize_artifacts(
        partial, json_path, csv_path, verify=False, partial=True
    ) is partial
    assert json_path.read_text(encoding="utf-8") == "json sentinel\n"
    assert csv_path.read_text(encoding="utf-8") == "csv sentinel\n"
    with pytest.raises(ValueError, match="cannot use partial"):
        module.finalize_artifacts(
            partial, json_path, csv_path, verify=True, partial=True
        )


def test_complete_generation_contract_rejects_skipped_branches():
    module = load_script("window_sensitivity")
    adjudication = [{"pdb": "2DEF"}]
    variants = {"variant": {"skipped": False}}
    output = {
        "protocol": {},
        "curation_rule_rediscovered": {"matches_committed_labels": True},
        "adjudication_summary": {},
        "adjudication": adjudication,
        "ensembles": variants,
        "superposition_dependence": {
            "whole_molecule": {},
            "on_NTD": {},
            "on_HB": {},
            "on_TBD": {},
        },
        "method_subsets": {
            "X-ray": {},
            "cryo-EM": {},
            "contingency": {},
            "cos_xray_vs_cryoem_axis": 1.0,
        },
        "constructs": {},
        "ddb1_entity_census": {},
        "empty_middle": {"variant": {}},
    }
    arguments = (
        ["1ABC", "2DEF"],
        {"1ABC": {}, "2DEF": {}},
        ["1ABC"],
        ["1ABC"],
        ["2DEF"],
        adjudication,
        variants,
        ["variant"],
        output,
    )

    module.validate_complete_generation(*arguments)
    variants["variant"]["skipped"] = True
    with pytest.raises(RuntimeError, match="were skipped"):
        module.validate_complete_generation(*arguments)


def test_complete_artifacts_are_validated_before_atomic_replacement(tmp_path):
    module = load_script("window_sensitivity")
    json_path = tmp_path / "window.json"
    csv_path = tmp_path / "adjudication.csv"
    output = {"adjudication": [{"pdb": "1ABC", "score": 1.25}]}

    normalized = module.finalize_artifacts(
        output, json_path, csv_path, verify=False, partial=False
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == normalized
    assert csv_path.read_text(encoding="utf-8") == "pdb,score\n1ABC,1.25\n"
    with pytest.raises(ValueError):
        module.prepare_artifact_payloads(
            {"adjudication": [{"pdb": "1ABC", "score": float("nan")}]}
        )
    with pytest.raises(RuntimeError, match="one complete CSV schema"):
        module.prepare_artifact_payloads(
            {"adjudication": [{"pdb": "1ABC", "score": 1}, {"pdb": "2DEF"}]}
        )


def test_cif_cache_is_validated_and_atomically_reacquired(tmp_path, monkeypatch):
    module = load_script("window_sensitivity")
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "1ABC.cif.gz"
    path.write_bytes(b"truncated cache")
    os.chmod(path, 0o640)
    module.CACHE = cache
    module.CACHE_WRITES_ENABLED = True
    text = "data_1ABC\nloop_\n_atom_site.group_PDB\nATOM\n#\n"
    valid_blob = gzip.compress(text.encode())

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return self.payload

    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *args, **kwargs: Response(valid_blob)
    )

    assert module.fetch_cif("1ABC") == text
    assert path.read_bytes() == valid_blob
    assert path.stat().st_mode & 0o777 == 0o640
    assert sorted(item.name for item in cache.iterdir()) == ["1ABC.cif.gz"]

    path.write_bytes(b"still truncated")
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"not gzip")
    )
    with pytest.raises(ValueError, match="invalid gzipped"):
        module.fetch_cif("1ABC")
    assert path.read_bytes() == b"still truncated"


def test_new_binary_cache_mode_respects_current_umask(tmp_path):
    module = load_script("window_sensitivity")
    path = tmp_path / "new.cif.gz"
    previous = os.umask(0o027)
    try:
        module.atomic_write_bytes(path, b"payload")
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o640


def test_failed_binary_replace_preserves_existing_cache(tmp_path, monkeypatch):
    module = load_script("window_sensitivity")
    path = tmp_path / "existing.cif.gz"
    path.write_bytes(b"existing payload")

    def fail_replace(source, destination):
        raise OSError("controlled replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="controlled"):
        module.atomic_write_bytes(path, b"new payload")

    assert path.read_bytes() == b"existing payload"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["existing.cif.gz"]
