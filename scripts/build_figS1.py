#!/usr/bin/env python3
"""Build Fig. S1 and its exact source-data file."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from figure_package_utils import save_figure_set, write_legend_docx
from figure_style import (
    BLACK,
    CLOSED,
    DARK_GREY,
    GENUINE_APO,
    GREEN,
    MAIN_WIDTH_IN,
    MID_GREY,
    apply_publication_style,
    finish_axis,
    panel_label,
    sample_size_label,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "crbn_curation_log.csv"
SOURCE_DATA = ROOT / "figures" / "source_data" / "FigS1_source_data.csv"

LEGEND = (
    "Fig S1. Ensemble composition. (a) Resolution distribution of the curated 70-conformer "
    "ensemble, stacked by ligand or substrate context: 66 drug-conditioned, one "
    "native-substrate-bound and three genuine-apo conformers. Drug-conditioned denotes a "
    "complex prepared with a drug, including compounds not modelled in the deposited "
    "coordinates; genuine-apo denotes preparation without a ligand or substrate. (b) "
    "Experimental method composition: 42 cryo-electron microscopy (cryo-EM) and 28 X-ray "
    "crystallography structures."
)

STATE_ORDER = ["drug-conditioned", "native-substrate", "genuine-apo"]
STATE_COLORS = {
    "drug-conditioned": CLOSED,
    "native-substrate": GREEN,
    "genuine-apo": GENUINE_APO,
}
STATE_HATCHES = {"drug-conditioned": "", "native-substrate": "//", "genuine-apo": ".."}
METHOD_ORDER = ["cryo-EM", "X-ray"]
METHOD_COLORS = [DARK_GREY, MID_GREY]
METHOD_HATCHES = ["", "//"]
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

    apply_publication_style("FigS1")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(MAIN_WIDTH_IN, 2.82), dpi=300)
    fig.subplots_adjust(left=0.088, right=0.988, bottom=0.18, top=0.84, wspace=0.35)

    centers = (RESOLUTION_BINS[:-1] + RESOLUTION_BINS[1:]) / 2
    widths = np.diff(RESOLUTION_BINS)
    bottom = np.zeros(len(centers), dtype=int)
    for state in STATE_ORDER:
        counts, edges = np.histogram(state_values[state], bins=RESOLUTION_BINS)
        if not np.array_equal(edges, RESOLUTION_BINS):
            raise RuntimeError("resolution histogram edges changed unexpectedly")
        axa.bar(
            centers,
            counts,
            width=widths,
            bottom=bottom,
            align="center",
            color=STATE_COLORS[state],
            edgecolor=BLACK,
            linewidth=0.45,
            hatch=STATE_HATCHES[state],
            alpha=0.84,
            label=f"{state} ({sample_size_label(state_counts[state])})",
            zorder=2,
        )
        bottom += counts
    if int(bottom.sum()) != 70:
        raise RuntimeError(f"stacked resolution histogram contains {int(bottom.sum())} entries")
    axa.set(
        xlabel="Resolution (Å)",
        ylabel="Curated conformers",
        xlim=(1.98, 4.02),
        ylim=(0, 19.2),
    )
    legend_handles, legend_labels = axa.get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        ncol=3,
        fontsize=8.0,
        labelspacing=0.25,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=0.9,
        frameon=False,
    )
    finish_axis(axa, grid="y")
    axa.tick_params(which="both", top=False, right=False)
    panel_label(axa, "a", x=-0.10, y=1.055)

    method_counts = [methods[method] for method in METHOD_ORDER]
    method_bars = axb.bar(
        METHOD_ORDER,
        method_counts,
        color=METHOD_COLORS,
        edgecolor=BLACK,
        linewidth=0.55,
        width=0.68,
        zorder=2,
    )
    for bar, hatch in zip(method_bars, METHOD_HATCHES):
        bar.set_hatch(hatch)
    for index, count in enumerate(method_counts):
        axb.text(
            index,
            count + 0.7,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=DARK_GREY,
        )
    axb.set(ylabel="Curated conformers", ylim=(0, 45))
    finish_axis(axb, grid="y")
    axb.tick_params(which="both", top=False, right=False)
    panel_label(axb, "b", x=-0.10, y=1.055)

    save_figure_set(fig, ROOT, "FigS1")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    counts = Counter(row["global_state"] for row in rows)
    methods = Counter(row["method"] for row in rows)
    write_source_data(rows)
    build_figure(rows)
    write_legend_docx(ROOT / "figures" / "FigS1_legend.docx", LEGEND)
    print(f"FigS1 built: states {dict(counts)}; methods {dict(methods)}")


if __name__ == "__main__":
    main()
