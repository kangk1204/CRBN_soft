from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "strengthen_ensemble", SCRIPTS / "strengthen_ensemble.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
        sys.modules.pop(spec.name, None)
    return module


def test_overlap_metrics_reports_rank_gaps_and_subspaces():
    mod = load_script()
    values = np.arange(1, 61, dtype=float)
    vectors = np.eye(60)
    axis = np.zeros(60)
    axis[2] = 0.8
    axis[21] = 0.6

    metrics = mod.overlap_metrics(axis, values, vectors)

    assert metrics["mode1_overlap"] == 0.0
    assert metrics["mode1_next_eigenvalue_gap"] == 1.0
    assert metrics["best20_rank"] == 3
    assert metrics["best20_overlap"] == 0.8
    assert metrics["best60_rank"] == 3
    assert metrics["top3_subspace_projection"] == 0.8
    assert metrics["top20_subspace_projection"] == 0.8


def test_primary_chain_uses_recorded_override_only_when_entity_contains_it():
    mod = load_script()

    assert mod.primary_chain(["C", "B"], "1ABC", {"1ABC": "C"}) == "C"
    assert mod.primary_chain(["C", "B"], "1ABC", {"1ABC": "A"}) == "B"
    assert mod.primary_chain([], "1ABC", {}) is None


def test_live_search_records_replay_url_and_result_sha(monkeypatch):
    mod = load_script()

    def fake_request_json(url, payload):
        assert url == mod.SEARCH_ENDPOINT
        assert payload["query"]["parameters"]["value"] == mod.CRBN_ACCESSION
        return {
            "total_count": 2,
            "result_set": [{"identifier": "9OPJ_2"}, {"identifier": "10AY_2"}],
        }

    monkeypatch.setattr(mod, "request_json", fake_request_json)

    result = mod.live_search()

    assert result["total_count"] == 2
    assert result["entities"] == ["10AY_2", "9OPJ_2"]
    assert result["endpoint_url"] == mod.SEARCH_ENDPOINT
    assert result["replay_url"].startswith(mod.SEARCH_ENDPOINT + "?json=")
    assert len(result["query_sha256"]) == 64
    assert len(result["result_sha256"]) == 64


def test_score_newer_uses_frozen_closure_function(monkeypatch):
    mod = load_script()

    monkeypatch.setattr(mod, "closure_score", lambda coords: (12.5, 1.1))
    monkeypatch.setattr(mod, "classify", lambda coordinate: "open" if coordinate >= 0.95 else "closed")
    coords = np.zeros((269, 3))
    inventory_row = {
        "pdb_id": "9OPJ",
        "release_date": "2026-07-29",
        "resolution_A": "3.200",
        "title": "example",
        "ligands": "ABC;ZN",
        "other_polymer_partners": "partner",
    }

    rows = mod.score_newer({"9OPJ_2": ("B", coords, inventory_row)})

    assert rows == [
        {
            "pdb_entity": "9OPJ_2",
            "pdb_id": "9OPJ",
            "chain": "B",
            "release_date": "2026-07-29",
            "resolution_A": "3.200",
            "n_window_positions": 269,
            "pc1_score": "12.500000",
            "closure_coordinate": "1.100000",
            "frozen_closed_band_max": 0.25,
            "frozen_open_band_min": 0.95,
            "frozen_state_call": "open",
            "title": "example",
            "ligands": "ABC;ZN",
            "other_polymer_partners": "partner",
        }
    ]


def test_default_outputs_are_neutral_results_paths():
    mod = load_script()

    assert mod.DEFAULT_OUTPUT.parts[-3:] == ("results", "strengthening", "ensemble")
    assert mod.DEFAULT_STRUCTURE_DIR.parts[-4:] == (
        "results",
        "strengthening",
        "data",
        "structures",
    )


def test_display_path_accepts_external_absolute_paths(tmp_path):
    mod = load_script()
    external = tmp_path / "artifact.txt"

    assert mod.display_path(external) == str(external.resolve())
