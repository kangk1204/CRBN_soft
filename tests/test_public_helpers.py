from __future__ import annotations

import importlib.util
from pathlib import Path

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
