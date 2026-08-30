#!/usr/bin/env python3
"""Build Fig. 3 from the frozen GNM/ANM/PCA analysis artifacts.

Panel (a) is the all-20-mode GNM cross-correlation matrix. Panel (b) pairs the
ANM and ensemble-PCA residue fluctuation profiles without connecting across
unresolved sequence gaps. The isolated residue-222 ANM spike is retained and
shown on a broken y axis; it is never clipped or used as the normalization
denominator.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from figure_package_utils import save_figure_set
from figure_style import (
    ANM,
    BLACK,
    DARK_GREY,
    HB,
    LIGHT_GREY,
    MAIN_WIDTH_IN,
    NTD,
    PCA,
    TBD,
    apply_publication_style,
    finish_axis,
    panel_label,
)
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
MODES_INPUT = ROOT / "data" / "crbn_anm_modes.npz"
FLUCTUATION_INPUT = ROOT / "data" / "crbn_residue_fluctuations.csv"
HINGE_INPUT = ROOT / "data" / "hinge_geometry.json"

# UniProt/author numbering: NTD 1-186, HB 187-317, TBD 318-426.
NTD_HB = 186.5
HB_TBD = 317.5


def _load_axis_band() -> tuple[float, float, tuple[int, ...]]:
    """Load the endpoint-derived screw-axis-proximal boundary band."""
    payload = json.loads(HINGE_INPUT.read_text(encoding="utf-8"))
    residues = tuple(int(value) for value in payload["axis_proximal_boundary_residues"])
    if residues != (316, 317, 318, 319, 320):
        raise ValueError(f"unexpected screw-axis-proximal boundary residues: {residues}")
    if abs(float(payload["rotation_angle_deg"]) - 82.457) > 0.01:
        raise ValueError("axis geometry does not match the frozen endpoint transform")
    return residues[0] - 0.5, residues[-1] + 0.5, residues


def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(MODES_INPUT, allow_pickle=False) as modes:
        eigenvectors = np.asarray(modes["gnm_eigvecs"], dtype=float)
        eigenvalues = np.asarray(modes["gnm_eigvals"], dtype=float)

    with FLUCTUATION_INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    residues = np.asarray([int(row["resnum"]) for row in rows], dtype=int)
    anm_raw = np.asarray([float(row["anm_sqfluct"]) for row in rows], dtype=float)
    pca_raw = np.asarray([float(row["pca_sqfluct"]) for row in rows], dtype=float)

    if eigenvectors.shape != (len(residues), 20):
        raise ValueError(
            "GNM eigenvectors must contain 20 modes over the same residue window: "
            f"{eigenvectors.shape} versus {len(residues)} residues"
        )
    if eigenvalues.shape != (20,) or np.any(eigenvalues <= 0):
        raise ValueError(f"expected 20 positive GNM eigenvalues, found {eigenvalues.shape}")
    if len(np.unique(residues)) != len(residues) or np.any(np.diff(residues) <= 0):
        raise ValueError("residue identifiers must be unique and strictly increasing")
    if not all(np.isfinite(array).all() for array in (eigenvectors, eigenvalues, anm_raw, pca_raw)):
        raise ValueError("Fig. 3 input contains non-finite values")

    covariance = (eigenvectors / eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(diagonal, diagonal)
    if not np.allclose(correlation, correlation.T, rtol=0.0, atol=1e-12):
        raise ValueError("recomputed GNM correlation matrix is not symmetric")
    if not np.allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("recomputed GNM correlation matrix has a non-unit diagonal")

    # Exclude two represented residues on either side of each sequence gap from
    # the ANM normalization denominator only. Every observed residue remains
    # plotted, including the documented gap-flank spike at residue 222.
    gap_flank = np.zeros(len(residues), dtype=bool)
    for index in np.flatnonzero(np.diff(residues) > 1) + 1:
        gap_flank[max(0, index - 2) : index + 2] = True
    anm_reference = float(anm_raw[~gap_flank].max())
    anm = anm_raw / anm_reference
    pca = pca_raw / float(pca_raw.max())
    return residues, correlation, anm, pca


def _break_at_gaps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs so lines never interpolate across unresolved sequence gaps."""
    x_out: list[float] = [float(x[0])]
    y_out: list[float] = [float(y[0])]
    for index in range(1, len(x)):
        if x[index] - x[index - 1] > 1:
            x_out.append(float(x[index - 1]) + 0.5)
            y_out.append(np.nan)
        x_out.append(float(x[index]))
        y_out.append(float(y[index]))
    return np.asarray(x_out), np.asarray(y_out)


def _domain_strip(axis, residues: np.ndarray) -> None:
    """Add a compact domain key aligned to matrix-index space."""
    strip = axis.inset_axes([0.0, 1.012, 1.0, 0.052], transform=axis.transAxes)
    strip.set_xlim(-0.5, len(residues) - 0.5)
    strip.set_ylim(0, 1)
    domains = [
        (residues <= NTD_HB, NTD, "NTD"),
        ((residues > NTD_HB) & (residues <= HB_TBD), HB, "HB"),
        (residues > HB_TBD, TBD, "TBD"),
    ]
    for mask, color, label in domains:
        indices = np.flatnonzero(mask)
        strip.add_patch(
            Rectangle(
                (indices[0] - 0.5, 0),
                indices[-1] - indices[0] + 1,
                1,
                facecolor=color,
                edgecolor="none",
            )
        )
        strip.text(
            float(indices.mean()),
            0.5,
            label,
            color="white",
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="center",
        )
    strip.set_axis_off()


def _shade_domains(
    axis, first_residue: int, last_residue: int, axis_band: tuple[float, float]
) -> None:
    axis.axvspan(first_residue, NTD_HB, color=NTD, alpha=0.085, linewidth=0)
    axis.axvspan(NTD_HB, HB_TBD, color=HB, alpha=0.085, linewidth=0)
    axis.axvspan(HB_TBD, last_residue, color=TBD, alpha=0.085, linewidth=0)
    axis.axvspan(*axis_band, color=DARK_GREY, alpha=0.10, hatch="////", linewidth=0)


def _axis_break_marks(upper, lower) -> None:
    kwargs = {"color": BLACK, "clip_on": False, "linewidth": 0.75}
    d = 0.012
    upper.plot((-d, +d), (-d, +d), transform=upper.transAxes, **kwargs)
    upper.plot((1 - d, 1 + d), (-d, +d), transform=upper.transAxes, **kwargs)
    lower.plot((-d, +d), (1 - d, 1 + d), transform=lower.transAxes, **kwargs)
    lower.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=lower.transAxes, **kwargs)


def main() -> None:
    residues, correlation, anm, pca = _load_data()
    axis_start, axis_end, axis_residues = _load_axis_band()
    axis_band = (axis_start, axis_end)
    apply_publication_style("Fig3")

    # Tight bounding includes the external panel labels and colour bar; a
    # 0.10-in canvas allowance keeps the exported media box at about 168 mm.
    figure = plt.figure(figsize=(MAIN_WIDTH_IN - 0.10, 3.45))
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.00, 1.22),
        left=0.075,
        right=0.985,
        bottom=0.15,
        top=0.90,
        wspace=0.34,
    )

    # Panel a: all-mode GNM cross-correlation in matrix-index space. True
    # residue numbers remain the tick labels because the sequence is gapped.
    ax_corr = figure.add_subplot(outer[0, 0])
    image = ax_corr.imshow(
        correlation,
        origin="lower",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
        rasterized=True,
    )
    for boundary in (NTD_HB, HB_TBD):
        boundary_index = np.searchsorted(residues, boundary) - 0.5
        ax_corr.axhline(boundary_index, color=BLACK, linewidth=0.65, linestyle=":")
        ax_corr.axvline(boundary_index, color=BLACK, linewidth=0.65, linestyle=":")
    tick_residues = np.asarray([100, 150, 200, 250, 300, 350, 400])
    tick_indices = np.searchsorted(residues, tick_residues)
    tick_indices = np.clip(tick_indices, 0, len(residues) - 1)
    ax_corr.set_xticks(tick_indices, labels=residues[tick_indices])
    ax_corr.set_yticks(tick_indices, labels=residues[tick_indices])
    ax_corr.set_xlabel("Residue")
    ax_corr.set_ylabel("Residue")
    ax_corr.tick_params(which="both", top=False, right=False)
    _domain_strip(ax_corr, residues)
    panel_label(ax_corr, "a", x=-0.16, y=1.13)
    colorbar = figure.colorbar(image, ax=ax_corr, fraction=0.047, pad=0.035)
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    colorbar.ax.set_title("GNM r", fontsize=8.0, fontweight="bold", pad=3)
    colorbar.outline.set_linewidth(0.65)

    # Panel b: a small upper segment keeps the 1.93x residue-222 artefact
    # visible while preserving readable detail in the biologically relevant
    # 0-1 range shared by the two normalized profiles.
    mobility_grid = outer[0, 1].subgridspec(2, 1, height_ratios=(0.42, 2.30), hspace=0.055)
    ax_high = figure.add_subplot(mobility_grid[0, 0])
    ax_low = figure.add_subplot(mobility_grid[1, 0], sharex=ax_high)
    x_plot, anm_plot = _break_at_gaps(residues, anm)
    _, pca_plot = _break_at_gaps(residues, pca)
    for axis in (ax_high, ax_low):
        _shade_domains(axis, int(residues[0]), int(residues[-1]), axis_band)
        axis.plot(x_plot, anm_plot, color=ANM, linewidth=1.35, label="ANM (intrinsic)")
        axis.plot(
            x_plot,
            pca_plot,
            color=PCA,
            linewidth=1.35,
            linestyle=(0, (4.2, 2.0)),
            label="PCA (ensemble)",
        )
        axis.set_xlim(residues[0], residues[-1])
        finish_axis(axis, grid="y")
        axis.tick_params(which="both", top=False, right=False)

    ax_high.set_ylim(1.68, 2.02)
    ax_high.set_yticks([1.75, 2.00])
    ax_high.spines["bottom"].set_visible(False)
    ax_high.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_low.set_ylim(0, 1.05)
    ax_low.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax_low.spines["top"].set_visible(False)
    ax_low.set_xlabel("Residue")
    ax_low.set_ylabel("Normalized square fluctuation")
    _axis_break_marks(ax_high, ax_low)

    residue_222_index = int(np.flatnonzero(residues == 222)[0])
    ax_high.annotate(
        f"222  ({anm[residue_222_index]:.2f}×)",
        xy=(222, anm[residue_222_index]),
        xytext=(238, 1.88),
        arrowprops={"arrowstyle": "-", "color": DARK_GREY, "linewidth": 0.7},
        color=DARK_GREY,
        fontsize=8.0,
        ha="left",
        va="center",
    )
    pca_peak_index = int(np.argmax(pca))
    ax_low.annotate(
        f"PCA peak {residues[pca_peak_index]}",
        xy=(residues[pca_peak_index], pca[pca_peak_index]),
        xytext=(358, 0.83),
        arrowprops={"arrowstyle": "-", "color": PCA, "linewidth": 0.7},
        color=PCA,
        fontsize=8.0,
        ha="right",
        va="center",
    )
    ax_low.text(130, 0.94, "NTD", color=NTD, fontweight="bold", ha="center", va="top")
    ax_low.text(242, 0.94, "HB", color=HB, fontweight="bold", ha="center", va="top")
    ax_low.text(318, 0.83, "axis", color=DARK_GREY, ha="center", va="top")
    ax_low.text(374, 0.94, "TBD", color=TBD, fontweight="bold", ha="center", va="top")
    legend_handles = [
        plt.Line2D([], [], color=ANM, linewidth=1.5, label="ANM (intrinsic)"),
        plt.Line2D(
            [], [], color=PCA, linewidth=1.5, linestyle=(0, (4.2, 2.0)), label="PCA (ensemble)"
        ),
        Patch(
            facecolor=LIGHT_GREY,
            edgecolor=DARK_GREY,
            hatch="////",
            label=f"screw-axis proximity {axis_residues[0]}-{axis_residues[-1]}",
        ),
    ]
    ax_low.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.77),
        borderaxespad=0,
        labelspacing=0.35,
    )
    panel_label(ax_high, "b", x=-0.13, y=1.11)

    save_figure_set(figure, ROOT, "Fig3")
    plt.close(figure)

    ntd = residues <= NTD_HB
    tbd = residues > HB_TBD
    mean_ntd_tbd = float(correlation[np.ix_(ntd, tbd)].mean())
    print(
        "Fig3 built: "
        f"GNM range [{correlation.min():.3f}, {correlation.max():.3f}], "
        f"mean NTD-TBD correlation {mean_ntd_tbd:.3f}, "
        f"residue-222 ANM {anm[residue_222_index]:.3f}x, "
        f"PCA peak residue {residues[pca_peak_index]}, n={len(residues)}"
    )


if __name__ == "__main__":
    main()
