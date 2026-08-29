#!/usr/bin/env python3
"""Build supplementary Fig S1 and its exact source-data and legend files."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from figure_package_utils import save_figure_set, write_legend_docx


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "crbn_curation_log.csv"
SOURCE_DATA = ROOT / "figures" / "source_data" / "FigS1_source_data.csv"
LEGEND_DOCX = ROOT / "figures" / "FigS1_legend.docx"

STATE_ORDER = ["drug-conditioned", "native-substrate", "genuine-apo"]
STATE_COLORS = {
    "drug-conditioned": "#D9792B",
    "native-substrate": "#2A9D8F",
    "genuine-apo": "#3B6EA8",
}
METHOD_ORDER = ["cryo-EM", "X-ray"]
METHOD_COLORS = ["#2A9D8F", "#3B6EA8"]
RESOLUTION_BINS = np.arange(2.0, 4.01, 0.25)


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {"pdb", "global_state", "method", "resolution"}
    missing_columns = required - set(reader.fieldnames or [])
    if missing_columns:
        raise ValueError(f"curation log lacks columns: {sorted(missing_columns)}")
    if len(rows) != 70:
        raise ValueError(f"expected 70 curated conformers, found {len(rows)}")
    if len({row["pdb"] for row in rows}) != len(rows):
        raise ValueError("curation log contains duplicate PDB identifiers")
    if any(not row[column].strip() for row in rows for column in required):
        raise ValueError("curation log contains missing plotted values")
    if set(row["global_state"] for row in rows) != set(STATE_ORDER):
        raise ValueError("unexpected global-state labels in curation log")
    if set(row["method"] for row in rows) != set(METHOD_ORDER):
        raise ValueError("unexpected experimental-method labels in curation log")
    resolutions = np.array([float(row["resolution"]) for row in rows])
    if not np.isfinite(resolutions).all() or resolutions.min() < 2.0 or resolutions.max() > 4.0:
        raise ValueError("resolution values fall outside the reported 2.0–4.0 Å range")
    return rows


def write_source_data(rows: list[dict[str, str]]) -> None:
    SOURCE_DATA.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["panel", "series", "category", "x_lower", "x_upper", "value", "unit"])
        for state in STATE_ORDER:
            values = [float(row["resolution"]) for row in rows if row["global_state"] == state]
            counts, edges = np.histogram(values, bins=RESOLUTION_BINS)
            for left, right, count in zip(edges[:-1], edges[1:], counts):
                writer.writerow(["a", state, "resolution bin", f"{left:.2f}", f"{right:.2f}", int(count), "conformers"])
        methods = Counter(row["method"] for row in rows)
        for method in METHOD_ORDER:
            writer.writerow(["b", "experimental method", method, "", "", methods[method], "conformers"])


def build_figure(rows: list[dict[str, str]]) -> None:
    state_values = {
        state: [float(row["resolution"]) for row in rows if row["global_state"] == state]
        for state in STATE_ORDER
    }
    state_counts = {state: len(values) for state, values in state_values.items()}
    methods = Counter(row["method"] for row in rows)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "crbn-FigS1",
        }
    )
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    axa.hist(
        [state_values[state] for state in STATE_ORDER],
        bins=RESOLUTION_BINS,
        stacked=True,
        color=[STATE_COLORS[state] for state in STATE_ORDER],
        label=[f"{state} (n={state_counts[state]})" for state in STATE_ORDER],
    )
    axa.set_xlabel("resolution (Å)")
    axa.set_ylabel("curated conformers")
    axa.legend(frameon=False, fontsize=7.2)
    axa.spines[["top", "right"]].set_visible(False)

    method_counts = [methods[method] for method in METHOD_ORDER]
    axb.bar(METHOD_ORDER, method_counts, color=METHOD_COLORS, width=0.8)
    for index, count in enumerate(method_counts):
        axb.text(index, count + 0.4, str(count), ha="center", fontsize=8.5)
    axb.set_ylabel("curated conformers")
    axb.spines[["top", "right"]].set_visible(False)

    for axis, label in ((axa, "a"), (axb, "b")):
        axis.text(
            -0.15,
            1.05,
            f"({label})",
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="right",
        )

    fig.tight_layout()
    save_figure_set(fig, ROOT, "FigS1")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    counts = Counter(row["global_state"] for row in rows)
    methods = Counter(row["method"] for row in rows)
    write_source_data(rows)
    build_figure(rows)
    write_legend_docx(
        LEGEND_DOCX,
        "Fig S1. Ensemble composition. (a) Resolution distribution of the curated "
        "70-conformer ensemble, stacked by global state: 66 drug-conditioned, one "
        "native-substrate-bound and three genuine-apo conformers. (b) Experimental "
        "method composition: 42 cryo-electron microscopy (cryo-EM) and 28 X-ray "
        "crystallography structures.",
    )
    print(f"FigS1 built: states {dict(counts)}; methods {dict(methods)}")


if __name__ == "__main__":
    main()
