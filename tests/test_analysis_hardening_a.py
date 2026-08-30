from __future__ import annotations

import gzip
import importlib.util
import io
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_hardening_a", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Artifact(dict):
    @property
    def files(self):
        return list(self)


def valid_ensemble_pair():
    labels = np.asarray(["A", "B", "C", "D", "E", "F"])
    conformers = np.zeros((6, 2, 3), dtype=float)
    conformers[:5, 0, 0] = 1.0
    mask = np.asarray([True, True, True, True, True, False])
    axis = np.zeros(6)
    axis[0] = 1.0
    ensemble = Artifact(_confs=conformers, _labels=labels)
    difference = Artifact(labels=labels.copy(), open_mask=mask, diff_vec=axis)
    return ensemble, difference


def test_ensemble_difference_contract_rejects_semantic_mismatches():
    contracts = load_script("analysis_contracts")
    ensemble, difference = valid_ensemble_pair()
    returned = contracts.validate_ensemble_diff(ensemble, difference)
    assert returned[0].shape == (6, 2, 3)

    permuted_labels = Artifact(difference)
    permuted_labels["labels"] = difference["labels"][[1, 0, 2, 3, 4, 5]]
    with pytest.raises(ValueError, match="label order"):
        contracts.validate_ensemble_diff(ensemble, permuted_labels)

    integer_mask = Artifact(difference)
    integer_mask["open_mask"] = difference["open_mask"].astype(int)
    with pytest.raises(ValueError, match="must be boolean"):
        contracts.validate_ensemble_diff(ensemble, integer_mask)

    arbitrary_unit_axis = Artifact(difference)
    arbitrary_unit_axis["diff_vec"] = np.roll(difference["diff_vec"], 1)
    with pytest.raises(ValueError, match="open-minus-closed"):
        contracts.validate_ensemble_diff(ensemble, arbitrary_unit_axis)

    residue_permuted_axis = Artifact(difference)
    residue_permuted_axis["diff_vec"] = difference["diff_vec"].reshape(2, 3)[::-1].reshape(-1)
    with pytest.raises(ValueError, match="open-minus-closed"):
        contracts.validate_ensemble_diff(ensemble, residue_permuted_axis)


def test_tree_comparison_rejects_numeric_type_coercion():
    contracts = load_script("analysis_contracts")
    contracts.assert_tree_close({"count": 3, "score": 1.0}, {"count": np.int64(3), "score": 1})

    for invalid in (3.0, 3.9, "3", True):
        with pytest.raises(AssertionError, match="recomputed"):
            contracts.assert_tree_close(3, invalid)
    for invalid in ("1.0", True, None):
        with pytest.raises(AssertionError, match="not numeric"):
            contracts.assert_tree_close(1.0, invalid)


def test_atomic_text_write_preserves_or_assigns_normal_file_mode(tmp_path):
    contracts = load_script("analysis_contracts")
    existing = tmp_path / "existing.txt"
    existing.write_text("old", encoding="utf-8")
    existing.chmod(0o640)
    contracts.atomic_write_text(existing, "new")
    assert existing.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(existing.stat().st_mode) == 0o640

    current_umask = os.umask(0)
    os.umask(current_umask)
    created = tmp_path / "created.txt"
    contracts.atomic_write_text(created, "value")
    assert stat.S_IMODE(created.stat().st_mode) == 0o666 & ~current_umask


def test_strict_json_write_rejects_nonfinite_values_without_replacing(tmp_path):
    contracts = load_script("analysis_contracts")
    output = tmp_path / "result.json"
    output.write_text('{"status": "stable"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Out of range float values"):
        contracts.atomic_write_json(output, {"value": float("nan")})
    assert output.read_text(encoding="utf-8") == '{"status": "stable"}\n'


def test_pca_helpers_validate_inputs_and_are_deterministic():
    pca = load_script("pca_robustness")
    conformers = np.zeros((4, 2, 3), dtype=float)
    conformers[:, 0, 0] = np.arange(4)
    axis = np.zeros(6)
    axis[0] = 1.0
    variance, overlap = pca.pca_pc1(conformers, axis)
    assert variance == pytest.approx(1.0)
    assert overlap == pytest.approx(1.0)

    sampler = lambda rng: rng.integers(0, 4, 4)
    first = pca.bootstrap(conformers, axis, {3}, sampler, draws=8, seed=9)
    second = pca.bootstrap(conformers, axis, {3}, sampler, draws=8, seed=9)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]

    with pytest.raises(ValueError, match="at least two"):
        pca.pca_pc1(conformers[:1], axis)
    with pytest.raises(ValueError, match="unit length"):
        pca.pca_pc1(conformers, axis * 2)
    with pytest.raises(ValueError, match="positive integer"):
        pca.bootstrap(conformers, axis, {3}, sampler, draws=0)
    with pytest.raises(ValueError, match="invalid conformer index"):
        pca.bootstrap(conformers, axis, {3}, lambda _rng: [0, 1, 9], draws=1)


def test_pca_exact_check_and_atomic_archive_write(tmp_path, monkeypatch):
    pca = load_script("pca_robustness")
    output = tmp_path / "result.npz"
    monkeypatch.setattr(pca, "OUTPUT", output)
    expected = {"array": np.asarray([1.0, 2.0]), "count": 3}
    pca.atomic_save(expected)
    pca.verify_exact(expected)
    output.chmod(0o640)
    pca.atomic_save(expected)
    assert stat.S_IMODE(output.stat().st_mode) == 0o640

    with pytest.raises(AssertionError, match="key mismatch"):
        pca.verify_exact({**expected, "extra": 4})
    with pytest.raises(AssertionError, match="exact array mismatch"):
        pca.verify_exact({"array": np.asarray([1.0, 2.1]), "count": 3})
    pca.verify_exact({"array": np.asarray([1.0 + 5e-13, 2.0]), "count": 3})
    with pytest.raises(AssertionError, match="dtype mismatch"):
        pca.verify_exact({"array": np.asarray([1, 2]), "count": 3})


def test_tensor_module_imports_without_bulk_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tensor = load_script("reproduce_tensor")
    assert tensor.ROOT == ROOT
    assert tensor.WINSET == []
    assert tensor.UNSAFE_ALLOW_AMBIGUOUS_CHAIN is False


def test_tensor_window_and_optional_json_inputs_are_fail_closed(tmp_path):
    tensor = load_script("reproduce_tensor")
    missing = tmp_path / "missing.json"
    assert tensor._load_optional_json(missing) == {}

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        tensor._load_optional_json(malformed)
    nonobject = tmp_path / "nonobject.json"
    nonobject.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        tensor._load_optional_json(nonobject)

    window = tmp_path / "window.csv"
    window.write_text(
        "author_resnum\n" + "".join(f"{residue}\n" for residue in range(1, 270)),
        encoding="utf-8",
    )
    np.testing.assert_array_equal(tensor._load_window(window), np.arange(1, 270))
    window.write_text(
        "author_resnum\n" + "".join(f"{residue}\n" for residue in range(1, 269)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected 269"):
        tensor._load_window(window)


def test_tensor_chain_selection_keeps_primary_and_sensitivity_rules(monkeypatch):
    tensor = load_script("reproduce_tensor")
    residues = list(range(1, 270))
    full = {residue: (0.0, 0.0, 0.0) for residue in residues}
    shorter = dict(list(full.items())[:-10])
    metadata = {
        "8TZX": {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity_container_identifiers": {
                        "auth_asym_ids": ["A", "D"],
                        "reference_sequence_identifiers": [
                            {"database_accession": "Q96SW2"}
                        ],
                    }
                },
                {
                    "rcsb_polymer_entity_container_identifiers": {
                        "auth_asym_ids": ["Z"],
                        "reference_sequence_identifiers": [
                            {"database_accession": "P12345"}
                        ],
                    }
                },
            ]
        }
    }
    monkeypatch.setattr(tensor, "WIN", np.asarray(residues))
    monkeypatch.setattr(tensor, "WINSET", residues)
    monkeypatch.setattr(tensor, "RCSB_META", metadata)
    monkeypatch.setattr(tensor, "CHAIN_MAP", {})
    coordinates = {"A": shorter, "D": full, "Z": full}
    assert tensor.crbn_chains("8tzx") == ["A", "D"]
    assert tensor.select_chain(coordinates, "8TZX") == ("A", len(shorter))
    assert tensor.best_chain(coordinates, "8TZX") == ("D", len(full))
    with pytest.raises(RuntimeError, match="chain A is absent"):
        tensor.select_chain({"D": full, "Z": full}, "8TZX")


def test_tensor_full_check_rejects_partial_order_and_any_large_rmsd():
    tensor = load_script("reproduce_tensor")
    tensor._verify_rebuild(["1ABC", "2DEF"], ["1ABC", "2DEF"], np.asarray([0.1, 0.49]))
    with pytest.raises(AssertionError, match="exact label order"):
        tensor._verify_rebuild(["1ABC", "2DEF"], ["2DEF", "1ABC"], np.asarray([0.1, 0.1]))
    with pytest.raises(AssertionError, match="max 0.500"):
        tensor._verify_rebuild(["1ABC"], ["1ABC"], np.asarray([0.5]))
    with pytest.raises(AssertionError, match="non-finite"):
        tensor._verify_rebuild(["1ABC"], ["1ABC"], np.asarray([np.nan]))


def test_tensor_cli_rejects_partial_or_invalid_full_check():
    tensor = load_script("reproduce_tensor")
    with pytest.raises(SystemExit):
        tensor.parse_args(["--verify", "--limit", "1"])
    with pytest.raises(SystemExit):
        tensor.parse_args(["--limit", "0"])
    with pytest.raises(SystemExit):
        tensor.parse_args(["--verify", "--unsafe-allow-ambiguous-chain"])


def test_tensor_generation_validates_before_write_and_limit_never_writes(monkeypatch):
    tensor = load_script("reproduce_tensor")
    stored = np.zeros((1, 2, 3), dtype=float)
    monkeypatch.setattr(tensor, "_configure_console", lambda: None)
    monkeypatch.setattr(tensor, "_load_ensemble", lambda: (stored, ["1ABC"]))
    monkeypatch.setattr(tensor, "_require_window", lambda: np.asarray([1, 2]))
    monkeypatch.setattr(tensor, "extract", lambda _pdb: (np.ones((2, 3)), 2))
    monkeypatch.setattr(
        tensor,
        "superpose",
        lambda coordinates: (coordinates, np.zeros((2, 3))),
    )
    monkeypatch.setattr(tensor, "kabsch", lambda rebuilt, _stored: rebuilt)
    writes = []
    monkeypatch.setattr(tensor, "_atomic_save_tensor", lambda *args: writes.append(args))
    with pytest.raises(AssertionError, match="diverges"):
        tensor.main([])
    assert writes == []

    monkeypatch.setattr(tensor, "extract", lambda _pdb: (np.zeros((2, 3)), 2))
    assert tensor.main(["--limit", "1"]) == 0
    assert writes == []


def test_tensor_output_and_cache_writes_respect_no_write_mode(tmp_path, monkeypatch):
    tensor = load_script("reproduce_tensor")
    output = tmp_path / "rebuilt.npz"
    conformers = np.zeros((2, 3, 3))
    reference = np.zeros((3, 3))
    tensor._atomic_save_tensor(output, conformers, ["1ABC", "2DEF"], reference)
    output.chmod(0o640)
    tensor._atomic_save_tensor(output, conformers, ["1ABC", "2DEF"], reference)
    assert stat.S_IMODE(output.stat().st_mode) == 0o640
    with np.load(output, allow_pickle=False) as stored:
        assert set(stored.files) == {"confs", "labels", "ref"}
        assert stored["confs"].dtype == np.float32
        assert stored["ref"].dtype == np.float32
        assert stored["labels"].tolist() == ["1ABC", "2DEF"]

    cif_text = b"data_1ABC\nloop_\n_atom_site.label_atom_id\n_atom_site.auth_asym_id\n"
    payload = gzip.compress(cif_text)
    monkeypatch.setattr(tensor, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(tensor, "CACHE_WRITES_ENABLED", False)
    monkeypatch.setattr(
        tensor.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    assert tensor.fetch_cif("1abc") == cif_text.decode()
    assert not (tmp_path / "cache").exists()


def test_tensor_reacquires_semantically_invalid_gzip_cache_before_replacing(
    tmp_path, monkeypatch
):
    tensor = load_script("reproduce_tensor")
    cache = tmp_path / "cache"
    cache.mkdir()
    cached_path = cache / "1ABC.cif.gz"
    poisoned = gzip.compress(b"data_1ABC\n# atom-site loop missing\n")
    cached_path.write_bytes(poisoned)
    valid_text = b"data_1ABC\nloop_\n_atom_site.label_atom_id\n_atom_site.auth_asym_id\n"
    valid = gzip.compress(valid_text)
    monkeypatch.setattr(tensor, "CACHE", cache)
    monkeypatch.setattr(tensor, "CACHE_WRITES_ENABLED", True)
    monkeypatch.setattr(
        tensor.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(valid),
    )
    assert tensor.fetch_cif("1ABC") == valid_text.decode()
    assert cached_path.read_bytes() == valid

    cached_path.write_bytes(poisoned)
    invalid_download = gzip.compress(b"<html>temporary error</html>\n")
    monkeypatch.setattr(
        tensor.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(invalid_download),
    )
    with pytest.raises(ValueError, match="atom-site loop"):
        tensor.fetch_cif("1ABC")
    assert cached_path.read_bytes() == poisoned
