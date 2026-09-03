#!/usr/bin/env python3
"""Build main Fig. 5: robustness of the intrinsic ANM prediction.

The four panels retain the frozen endpoint, exact matched-subspace null,
contact-cutoff and leave-one-out records. This builder changes presentation
only; it does not refit a model, draw a random null, or alter a reported value.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from figure_package_utils import require_rigid_null_schema, save_figure_set
from figure_style import (
    BLACK,
    BLUE,
    CLOSED,
    DARK_GREY,
    LIGHT_GREY,
    MAGENTA,
    OPEN,
    ORANGE,
    PALE_BLUE,
    PALE_GREEN,
    PALE_ORANGE,
    PURPLE,
    apply_publication_style,
    finish_axis,
    panel_label,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_WIDTH_IN = 6.40  # allows for tight-bbox text padding below 170 mm
ROBUSTNESS_INPUT = ROOT / "data" / "anm_robustness.json"
NULL_INPUT = ROOT / "data" / "anm_null_significance.json"
RIGID_NULL_INPUT = ROOT / "data" / "assembly_rigid_null.json"
ENSEMBLE_INPUT = ROOT / "data" / "crbn_ensemble.ens.npz"
CURATION_INPUT = ROOT / "data" / "crbn_curation_log.csv"
DIFFERENCE_INPUT = ROOT / "data" / "pca_diffvec.npz"

NULL_X = np.linspace(0.0, 1.0, 1001)
MODEL_ROWS = (
    ("two_block", "2-lobe", DARK_GREY, "-"),
    ("three_block", "3-domain", PURPLE, "--"),
    ("bond_length_preserving_boundary", "bond-length", BLUE, "-."),
    ("equal_displacement_boundary", "equal displacement", MAGENTA, ":"),
)


def exact_null_density(dimension: int) -> np.ndarray:
    """Density of |cos(theta)| for a random direction in ``dimension`` D."""
    beta = 0.5 * (dimension - 1)
    normalizer = 2.0 * math.exp(
        math.lgamma(0.5 + beta) - math.lgamma(0.5) - math.lgamma(beta)
    )
    return normalizer * np.power(
        np.clip(1.0 - NULL_X**2, 0.0, None), beta - 1.0
    )


def load_and_validate() -> tuple[dict, dict, dict]:
    robustness = json.loads(ROBUSTNESS_INPUT.read_text(encoding="utf-8"))
    null = json.loads(NULL_INPUT.read_text(encoding="utf-8"))
    rigid = require_rigid_null_schema(
        json.loads(RIGID_NULL_INPUT.read_text(encoding="utf-8")),
        str(RIGID_NULL_INPUT),
    )

    # Re-derive the open set from the canonical difference-vector artifact and
    # require exact agreement with both the robustness record and ensemble.
    with np.load(ENSEMBLE_INPUT, allow_pickle=False) as ensemble:
        labels = np.asarray([str(value) for value in ensemble["_labels"]])
    with np.load(DIFFERENCE_INPUT, allow_pickle=False) as difference:
        if not {"labels", "open_mask"}.issubset(difference.files):
            raise ValueError("pca_diffvec.npz lacks labels/open_mask")
        difference_labels = np.asarray([str(value) for value in difference["labels"]])
        difference_mask = np.asarray(difference["open_mask"])

    if (
        difference_mask.dtype.kind != "b"
        or difference_mask.shape != difference_labels.shape
    ):
        raise ValueError("pca_diffvec.npz open_mask is not a matching boolean vector")
    expected_open = {
        label
        for label, is_open in zip(difference_labels, difference_mask)
        if is_open
    }
    reported_open = [str(value) for value in robustness["open_set"]]
    if len(reported_open) != len(set(reported_open)) or set(reported_open) != expected_open:
        raise ValueError(
            "open-set mismatch between robustness and difference artifacts: "
            f"{reported_open} versus {sorted(expected_open)}"
        )
    if sum(label in expected_open for label in labels) != len(expected_open):
        raise ValueError("ensemble does not contain every canonical open structure once")

    endpoints = reported_open + [str(value) for value in robustness["closed_endpoints"]]
    if len(endpoints) != 10 or len(endpoints) != len(set(endpoints)):
        raise ValueError("Fig. 5a requires ten unique endpoint structures")
    if [float(value) for value in robustness["cutoffs"]] != [10, 12, 13, 15, 16, 18]:
        raise ValueError("unexpected ANM cutoff series")
    return robustness, null, rigid


def format_p(value: float) -> str:
    if value < 0.001:
        return f"p = {value:.5f}"
    return f"p = {value:.3f}"


def draw_endpoint_panel(ax, robustness: dict) -> None:
    open_labels = robustness["open_set"]
    closed_labels = robustness["closed_endpoints"]
    labels = open_labels + closed_labels
    records = [robustness["table"][label]["15.0"] for label in labels]
    y = np.arange(len(labels))

    ax.axhspan(-0.5, len(open_labels) - 0.5, color=PALE_BLUE, zorder=0)
    ax.axhspan(len(open_labels) - 0.5, len(labels) - 0.5, color=PALE_ORANGE, zorder=0)
    for index, record in enumerate(records):
        color = OPEN if index < len(open_labels) else CLOSED
        mode1 = float(record["mode1_overlap"])
        best = float(record["best_overlap"])
        rank = int(record["best_mode_rank"])
        if rank > 1:
            ax.plot([mode1, best], [index, index], color=color, linewidth=1.2, alpha=0.58)
            ax.scatter(
                [best], [index], s=45, facecolor="white", edgecolor=color,
                linewidth=1.35, zorder=4,
            )
            ax.annotate(
                f"m{rank}", (best, index), xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=8.0, va="center", ha="left",
            )
        ax.scatter(
            [mode1], [index], s=38, color=color, edgecolor="white",
            linewidth=0.45, zorder=5,
        )

    ax.axhline(len(open_labels) - 0.5, color="white", linewidth=1.4, zorder=1)
    ax.set_yticks(y, labels)
    colors = [OPEN] * len(open_labels) + [CLOSED] * len(closed_labels)
    for tick, color in zip(ax.get_yticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.84)
    ax.set_xticks(np.arange(0.0, 0.81, 0.2))
    ax.set_xlabel("Directional overlap with open–closed axis")
    handles = [
        Patch(facecolor=PALE_BLUE, edgecolor="none", label="open endpoint"),
        Patch(facecolor=PALE_ORANGE, edgecolor="none", label="closed endpoint"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=BLACK,
               markeredgecolor="white", label="mode 1"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=BLACK, label="best higher mode"),
    ]
    ax.legend(
        handles=handles, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0, fontsize=8.0,
    )
    finish_axis(ax, grid="x")
    ax.tick_params(which="both", top=False, right=False)
    panel_label(ax, "a", x=-0.20, y=1.10)


def draw_rigid_null_panel(ax_density, ax_result, rigid: dict) -> None:
    handles: list[Line2D] = []
    for key, label, color, linestyle in MODEL_ROWS:
        record = rigid[key]
        density = exact_null_density(int(record["internal_dim"]))
        ax_density.plot(
            NULL_X, density, color=color, linestyle=linestyle,
            linewidth=1.45, zorder=3,
        )
        observed = float(record["observed_direction_cosine_in_subspace"])
        observed_density = float(np.interp(observed, NULL_X, density))
        ax_density.vlines(
            observed, 0, observed_density, color=color, linestyle=linestyle,
            linewidth=1.15, alpha=0.90, zorder=4,
        )
        handles.append(
            Line2D(
                [], [], color=color, linestyle=linestyle,
                label=f"{label}, d = {int(record['internal_dim'])}",
            )
        )

    two_density = exact_null_density(int(rigid["two_block"]["internal_dim"]))
    ax_density.fill_between(NULL_X, two_density, color=LIGHT_GREY, alpha=0.42, zorder=1)
    ax_density.set_xlim(0.0, 1.0)
    ax_density.set_ylim(bottom=0.0)
    ax_density.set_ylabel("Exact-null density")
    ax_density.tick_params(axis="x", labelbottom=False)
    ax_density.set_title(
        "Exact Beta nulls in matched subspaces", loc="left", pad=4,
        fontsize=8.0, fontweight="bold", color=DARK_GREY,
    )
    ax_density.legend(
        handles=handles, ncol=1, loc="upper right", fontsize=8.0,
        labelspacing=0.16, handlelength=1.65, borderaxespad=0.25,
    )
    finish_axis(ax_density)
    ax_density.tick_params(which="both", top=False, right=False)
    panel_label(ax_density, "b", x=-0.19, y=1.10)

    result_rows = list(MODEL_ROWS)
    y_positions = np.arange(len(result_rows))[::-1]
    for y, (key, _label, color, _linestyle) in zip(y_positions, result_rows):
        record = rigid[key]
        observed = float(record["observed_direction_cosine_in_subspace"])
        ax_result.hlines(y, 0.0, observed, color=LIGHT_GREY, linewidth=0.8, zorder=1)
        ax_result.scatter(
            [observed], [y], s=32, color=color, edgecolor="white",
            linewidth=0.45, zorder=3,
        )
        ax_result.text(
            0.02, y, format_p(float(record["p_exact"])), color=BLACK,
            fontsize=8.0, va="center", ha="left",
        )
    ax_result.set_yticks(y_positions, [row[1] for row in result_rows])
    ax_result.set_xlim(0.0, 1.0)
    ax_result.set_ylim(-0.65, len(result_rows) - 0.35)
    ax_result.set_xlabel("Observed direction cosine")
    ax_result.tick_params(axis="y", labelsize=8.0, pad=2)
    finish_axis(ax_result, grid="x")
    ax_result.tick_params(which="both", top=False, right=False)


def _open_reference_states(open_labels) -> dict[str, str]:
    """Map each open reference to its committed experimental context.

    Fig 5c contrasts the genuine-apo references with the engineered
    drug-conditioned ones, so the split must come from the curation log rather
    than from a literal list that could drift away from the census.
    """
    with CURATION_INPUT.open(encoding="utf-8", newline="") as handle:
        recorded = {row["pdb"]: row["global_state"] for row in csv.DictReader(handle)}
    missing = [label for label in open_labels if label not in recorded]
    if missing:
        raise KeyError(f"open references absent from the curation log: {missing}")
    return {label: recorded[label] for label in open_labels}


def draw_cutoff_panel(ax, robustness: dict) -> None:
    cutoffs = np.asarray([float(value) for value in robustness["cutoffs"]])
    open_labels = robustness["open_set"]
    values = np.asarray(
        [
            [
                float(robustness["table"][label][str(raw)]["mode1_overlap"])
                for raw in robustness["cutoffs"]
            ]
            for label in open_labels
        ]
    )
    # The legend contrasts the three genuine-apo references with the two engineered
    # drug-conditioned ones, so the panel has to encode that split.  The assignment is
    # read from the committed curation log rather than hard-coded here.
    states = _open_reference_states(open_labels)
    for label, row in zip(open_labels, values):
        drug_conditioned = states[label] != "genuine-apo"
        ax.plot(
            cutoffs, row,
            color=ORANGE if drug_conditioned else BLUE,
            alpha=0.55 if drug_conditioned else 0.30,
            linewidth=0.95,
            linestyle=(0, (4.0, 1.8)) if drug_conditioned else "-",
            marker="s" if drug_conditioned else "o",
            markersize=3.0, markerfacecolor="white",
            markeredgewidth=0.65,
        )
    mean = values.mean(axis=0)
    ax.plot(
        cutoffs, mean, color=BLUE, linewidth=2.1, marker="o",
        markersize=4.2, markeredgecolor="white", markeredgewidth=0.5, zorder=5,
    )
    ax.axvspan(15.0, 18.0, color=PALE_GREEN, zorder=0)
    ax.axvline(15.0, color=DARK_GREY, linestyle="--", linewidth=0.85, zorder=1)
    ax.text(
        0.98, 0.97, "15–18 Å stability range", transform=ax.transAxes,
        fontsize=8.0, color=DARK_GREY, ha="right", va="top",
    )
    ax.set_xlim(9.6, 18.4)
    ax.set_ylim(0.20, 0.82)
    ax.set_xticks(cutoffs)
    ax.set_xlabel("ANM contact cutoff (Å)")
    ax.set_ylabel("Mode-1 directional overlap")
    n_apo = sum(1 for label in open_labels if states[label] == "genuine-apo")
    n_drug = len(open_labels) - n_apo
    ax.legend(
        handles=[
            Line2D(
                [], [], color=BLUE, alpha=0.40, marker="o",
                markerfacecolor="white", label=f"genuine apo ({n_apo})",
            ),
            Line2D(
                [], [], color=ORANGE, alpha=0.65, marker="s",
                linestyle=(0, (4.0, 1.8)), markerfacecolor="white",
                label=f"drug-conditioned ({n_drug})",
            ),
            Line2D([], [], color=BLUE, linewidth=2.1, marker="o", label="mean of five"),
        ],
        ncol=1, loc="lower right", fontsize=8.0,
    )
    finish_axis(ax, grid="y")
    ax.tick_params(which="both", top=False, right=False)
    panel_label(ax, "c", x=-0.19)


def draw_leave_one_out_panel(ax, null: dict) -> None:
    closed = null["leave_one_closed_out"]
    open_ = null["leave_one_open_out"]
    records = [closed, open_]
    x = np.arange(2)
    means = np.asarray([float(record["mean"]) for record in records])
    lows = np.asarray([float(record["min"]) for record in records])
    highs = np.asarray([float(record["max"]) for record in records])
    errors = np.vstack([means - lows, highs - means])
    full = float(null["observed_mode1_overlap"])

    ax.errorbar(
        x, means, yerr=errors, fmt="o", markersize=7.0, color=BLUE,
        markeredgecolor="white", markeredgewidth=0.6, ecolor=DARK_GREY,
        elinewidth=1.2, capsize=4.0, zorder=4,
    )
    ax.axhline(full, color=CLOSED, linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        0.98, full - 0.030, f"full ensemble = {full:.3f}",
        transform=ax.get_yaxis_transform(), color=CLOSED, fontsize=8.0,
        ha="right", va="top",
    )
    for index, (mean, low, high) in enumerate(zip(means, lows, highs)):
        ax.annotate(
            f"mean {mean:.4f}\nrange {low:.4f}–{high:.4f}",
            (index, high), xytext=(0, 40 if index == 0 else 8),
            textcoords="offset points",
            fontsize=8.0, color=BLACK, ha="center", va="bottom",
        )
    ax.set_xticks(
        x,
        [
            f"drop one closed\n(n = {int(closed['n'])})",
            f"drop one open\n(n = {int(open_['n'])})",
        ],
    )
    ax.set_xlim(-0.55, 1.55)
    # Keep the complete metric domain: the narrow ranges must not look like a
    # large biological effect merely because of an expanded axis.
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.25))
    ax.set_ylabel("Mode-1 directional overlap")
    finish_axis(ax, grid="y")
    ax.tick_params(which="both", top=False, right=False)
    panel_label(ax, "d", x=-0.19)


def main() -> None:
    robustness, null, rigid = load_and_validate()
    apply_publication_style("Fig5")

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 6.10), dpi=300)
    outer = fig.add_gridspec(
        2, 2, left=0.105, right=0.985, top=0.94, bottom=0.085,
        hspace=0.39, wspace=0.37,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    null_grid = outer[0, 1].subgridspec(
        2, 1, height_ratios=[1.65, 1.0], hspace=0.06,
    )
    ax_b_density = fig.add_subplot(null_grid[0, 0])
    ax_b_result = fig.add_subplot(null_grid[1, 0], sharex=ax_b_density)
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    draw_endpoint_panel(ax_a, robustness)
    draw_rigid_null_panel(ax_b_density, ax_b_result, rigid)
    draw_cutoff_panel(ax_c, robustness)
    draw_leave_one_out_panel(ax_d, null)

    save_figure_set(fig, ROOT, "Fig5")
    plt.close(fig)
    print("Fig5 written from frozen endpoint, exact-null, cutoff and leave-one-out records")


if __name__ == "__main__":
    main()
