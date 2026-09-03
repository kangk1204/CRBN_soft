from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_fig5_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "fig5_render_contract", SCRIPTS / "build_fig5_robustness.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def write_shuffled_curation(path: Path) -> None:
    rows = [
        {"pdb": "DRUG_B", "global_state": "drug-conditioned"},
        {"pdb": "APO_C", "global_state": "genuine-apo"},
        {"pdb": "APO_A", "global_state": "genuine-apo"},
        {"pdb": "DRUG_A", "global_state": "drug-conditioned"},
        {"pdb": "APO_B", "global_state": "genuine-apo"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb", "global_state"])
        writer.writeheader()
        writer.writerows(rows)


def test_draw_cutoff_panel_encodes_open_reference_context_without_reordering(
    tmp_path, monkeypatch
):
    module = load_fig5_module()
    curation = tmp_path / "crbn_curation_log.csv"
    write_shuffled_curation(curation)
    monkeypatch.setattr(module, "CURATION_INPUT", curation)
    monkeypatch.setattr(module, "finish_axis", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "panel_label", lambda *_args, **_kwargs: None)

    cutoffs = ["10", "15", "18"]
    open_labels = ["APO_B", "DRUG_A", "APO_A", "DRUG_B", "APO_C"]
    supplied = {
        "APO_B": [0.31, 0.41, 0.51],
        "DRUG_A": [0.32, 0.42, 0.52],
        "APO_A": [0.33, 0.43, 0.53],
        "DRUG_B": [0.34, 0.44, 0.54],
        "APO_C": [0.35, 0.45, 0.55],
    }
    robustness = {
        "cutoffs": cutoffs,
        "open_set": open_labels,
        "table": {
            label: {
                cutoff: {"mode1_overlap": value}
                for cutoff, value in zip(cutoffs, supplied[label])
            }
            for label in open_labels
        },
    }

    fig, ax = plt.subplots()
    try:
        module.draw_cutoff_panel(ax, robustness)
    finally:
        plt.close(fig)

    lines = ax.get_lines()
    reference_lines = lines[:5]
    mean_line = lines[5]
    assert len(reference_lines) == 5
    assert [list(line.get_xdata()) for line in reference_lines] == [[10.0, 15.0, 18.0]] * 5
    assert [list(line.get_ydata()) for line in reference_lines] == [
        supplied[label] for label in open_labels
    ]

    for label, line in zip(open_labels, reference_lines):
        expected_drug = label.startswith("DRUG")
        assert line.get_color() == (module.ORANGE if expected_drug else module.BLUE)
        assert line.get_marker() == ("s" if expected_drug else "o")
        linestyle = line.get_linestyle()
        if expected_drug:
            assert linestyle in ("--", (0, (4.0, 1.8)))
        else:
            assert linestyle == "-"

    expected_mean = np.asarray([supplied[label] for label in open_labels]).mean(axis=0)
    np.testing.assert_allclose(mean_line.get_ydata(), expected_mean)
    assert mean_line.get_color() == module.BLUE
    assert mean_line.get_marker() == "o"

    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend_labels == ["genuine apo (3)", "drug-conditioned (2)", "mean of five"]
