from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


def test_window_coordinates_accepts_complete_explicit_chain():
    score = load_script("score_structure")
    window = np.asarray([10, 11, 12])
    chains = {
        "A": {
            10: (1.0, 0.0, 0.0),
            11: (2.0, 0.0, 0.0),
            12: (3.0, 0.0, 0.0),
        }
    }

    chain, coords = score.window_coordinates(chains, window, "A")

    assert chain == "A"
    np.testing.assert_allclose(coords[:, 0], [1.0, 2.0, 3.0])


def test_window_coordinates_rejects_partial_window():
    score = load_script("score_structure")
    window = np.asarray([10, 11, 12])
    chains = {"A": {10: (1.0, 0.0, 0.0), 11: (2.0, 0.0, 0.0)}}

    with pytest.raises(SystemExit, match="complete window"):
        score.window_coordinates(chains, window, "A")


def test_window_coordinates_uses_exact_accession_mapping(monkeypatch):
    score = load_script("score_structure")
    window = np.asarray([10, 11])
    chains = {
        "A": {10: (1.0, 0.0, 0.0), 11: (2.0, 0.0, 0.0)},
        "B": {10: (10.0, 0.0, 0.0), 11: (20.0, 0.0, 0.0)},
    }
    monkeypatch.setattr(score, "crbn_chains", lambda pdb_id: ["B"] if pdb_id == "1ABC" else None)

    chain, coords = score.window_coordinates(chains, window, None, "1ABC")

    assert chain == "B"
    np.testing.assert_allclose(coords[:, 0], [10.0, 20.0])


def test_window_coordinates_requires_chain_when_fallback_is_ambiguous(monkeypatch):
    score = load_script("score_structure")
    window = np.asarray([10, 11])
    chains = {
        "A": {10: (1.0, 0.0, 0.0), 11: (2.0, 0.0, 0.0)},
        "B": {10: (10.0, 0.0, 0.0), 11: (20.0, 0.0, 0.0)},
    }
    monkeypatch.setattr(score, "crbn_chains", lambda _pdb_id: None)

    with pytest.raises(SystemExit, match="Pass --chain explicitly"):
        score.window_coordinates(chains, window, None, "1ABC")


def test_closure_score_keeps_projection_orientation(monkeypatch):
    score = load_script("score_structure")
    window = np.asarray([10, 11])
    mean = np.zeros((2, 3), dtype=float)
    pc1 = np.zeros(6, dtype=float)
    pc1[0] = np.sqrt(2.0)
    coords = mean.copy()
    coords[0, 0] = 0.5
    monkeypatch.setattr(score, "load_reference", lambda: (window, mean, pc1, 0.0, 1.0))
    monkeypatch.setattr(score, "kabsch_apply", lambda moving, _reference: moving)

    raw, coordinate = score.closure_score(coords)

    assert raw == pytest.approx(0.5)
    assert coordinate == pytest.approx(0.5)


def test_classify_uses_reference_bands():
    score = load_script("score_structure")

    assert score.classify(0.25) == "closed"
    assert score.classify(0.95) == "open"
    assert score.classify(0.5).startswith("intermediate")


def test_read_structure_rejects_non_file_non_pdb_target(monkeypatch):
    score = load_script("score_structure")
    monkeypatch.setattr(score, "fetch_cif", lambda _identifier: pytest.fail("fetch_cif called"))

    with pytest.raises(ValueError, match="PDB ID"):
        score.read_structure("missing/path.cif")
