#!/usr/bin/env python3
"""Shared publication styling for the CRBN quantitative figures.

The style layer is presentation-only. It changes no data, filtering,
statistics, ordering, units, or missing-value behaviour.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt


# Colour-universal palette. Important categories are also separated by marker,
# line style, fill, or position so no conclusion depends on hue alone.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#D55E00"
AMBER = "#E69F00"
GREEN = "#009E73"
PURPLE = "#7B3294"
MAGENTA = "#CC79A7"
BLACK = "#202124"
DARK_GREY = "#5F6368"
MID_GREY = "#9AA0A6"
LIGHT_GREY = "#DADCE0"
PALE_BLUE = "#E8F1F8"
PALE_ORANGE = "#FBEDE7"
PALE_GREEN = "#E6F4EF"

OPEN = BLUE
CLOSED = ORANGE
DRUG_CONDITIONED = ORANGE
GENUINE_APO = BLUE
NATIVE_SUBSTRATE = GREEN
NTD = BLUE
HB = GREEN
TBD = ORANGE
ANM = BLUE
PCA = PURPLE

MAIN_WIDTH_IN = 6.62
MIN_FONT_PT = 8.0


def apply_publication_style(figure_id: str) -> None:
    """Register SciencePlots and apply deterministic CRBN overrides."""
    try:
        import scienceplots  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SciencePlots is required for figure generation; create the "
            "environment from environment.yml"
        ) from exc

    plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "legend.handlelength": 1.35,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 0.8,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.5,
            "patch.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": f"crbn-{figure_id}-figure-v2",
            "mathtext.fontset": "dejavusans",
        }
    )


def panel_label(ax, label: str, *, x: float = -0.14, y: float = 1.06) -> None:
    """Place a consistent bold lower-case panel label in axes coordinates."""
    normalized = label.strip().strip("()")
    ax.text(
        x,
        y,
        f"({normalized})",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=BLACK,
        ha="right",
        va="top",
        clip_on=False,
    )


def finish_axis(
    ax,
    *,
    grid: Literal["x", "y", "both"] | None = None,
    zero_line: bool = False,
) -> None:
    """Apply restrained axis finishing without changing limits or data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        axis = "both" if grid == "both" else grid
        ax.grid(True, axis=axis, color=LIGHT_GREY, linewidth=0.45, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
    if zero_line:
        ax.axhline(0, color=MID_GREY, linewidth=0.55, zorder=0)


def sample_size_label(n: int) -> str:
    """Use one consistent sample-size form across legends and annotations."""
    return f"n = {int(n)}"
