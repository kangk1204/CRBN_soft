#!/usr/bin/env python3
"""Build Fig. 1 from the frozen CRBN census and mode-analysis arrays.

The visual revision is presentation-only: the 70 conformers, PCA values,
ANM statistics, binning, state labels, and empty-middle definition are read
unchanged from the committed inputs. The empty-middle interval is converted
from the stored PC1 coordinate into the normalized coordinate displayed in
panel d; the previous builder omitted that coordinate conversion.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from figure_package_utils import save_figure_set
from figure_style import (
    ANM,
    BLACK,
    CLOSED,
    DARK_GREY,
    GENUINE_APO,
    GREEN,
    LIGHT_GREY,
    MAIN_WIDTH_IN,
    MID_GREY,
    PCA,
    apply_publication_style,
    finish_axis,
    panel_label,
    sample_size_label,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_inputs() -> dict[str, object]:
    """Load and cross-check the exact frozen arrays used by all four panels."""
    pca = np.load(DATA / "crbn_pca.npz", allow_pickle=False)
    anm = np.load(DATA / "crbn_anm_modes.npz", allow_pickle=False)
    diffvec = np.load(DATA / "pca_diffvec.npz", allow_pickle=False)

    pc1 = np.asarray(pca["pc1_scores"], dtype=float)
    pc2_raw = np.asarray(pca["pc2_scores"], dtype=float)
    open_mask = np.asarray(pca["open_mask"], dtype=bool)
    variance_ratio = np.asarray(pca["variance_ratio"], dtype=float)
    cumulative_overlap = np.asarray(anm["cum_overlap"], dtype=float)
    rmsip = float(anm["rmsip"])
    labels = [str(value) for value in diffvec["labels"]]

    if not len(pc1) == len(pc2_raw) == len(open_mask) == len(labels) == 70:
        raise ValueError("Fig. 1 inputs must contain the same 70 conformers")
    if len(set(labels)) != 70:
        raise ValueError("Fig. 1 labels contain duplicate PDB identifiers")
    if int(open_mask.sum()) != 5:
        raise ValueError(f"expected five open conformers, found {int(open_mask.sum())}")
    if len(variance_ratio) < 10 or len(cumulative_overlap) < 10:
        raise ValueError("Fig. 1 requires at least ten PCA and ANM values")
    if not all(
        np.isfinite(values).all()
        for values in (pc1, pc2_raw, variance_ratio, cumulative_overlap)
    ) or not np.isfinite(rmsip):
        raise ValueError("Fig. 1 inputs contain non-finite plotted values")

    # PC1 is already RMSD-scaled. Put PC2 on the same scale, exactly as in the
    # frozen builder and source-data export.
    n_ca = np.asarray(pca["mean"]).reshape(-1, 3).shape[0]
    if n_ca != 269:
        raise ValueError(f"expected 269 common C-alpha positions, found {n_ca}")
    pc2 = pc2_raw / np.sqrt(n_ca)

    with (DATA / "crbn_curation_log.csv").open(encoding="utf-8", newline="") as handle:
        curation_rows = list(csv.DictReader(handle))
    state_by_pdb = {row["pdb"].upper(): row["global_state"] for row in curation_rows}
    if len(state_by_pdb) != 70 or any(label.upper() not in state_by_pdb for label in labels):
        raise ValueError("curation log and PCA labels do not form the same 70-entry set")
    global_state = np.array([state_by_pdb[label.upper()] for label in labels])
    expected_states = {"drug-conditioned", "genuine-apo", "native-substrate"}
    if set(global_state) != expected_states:
        raise ValueError(f"unexpected global-state labels: {sorted(set(global_state))}")

    empty_middle = json.loads(
        (DATA / "window_sensitivity.json").read_text(encoding="utf-8")
    )["empty_middle"]["a_paper_rule"]
    raw_band = np.asarray(empty_middle["band_15_85_pct"], dtype=float)
    closed_mean = float(pc1[~open_mask].mean())
    open_mean = float(pc1[open_mask].mean())
    normalized = (pc1 - closed_mean) / (open_mean - closed_mean)
    normalized_band = (raw_band - closed_mean) / (open_mean - closed_mean)
    if not (
        normalized[~open_mask].max()
        < normalized_band[0]
        < normalized_band[1]
        < normalized[open_mask].min()
    ):
        raise ValueError("the normalized empty-middle interval is not empty")
    if int(empty_middle["n_occupants"]) != 0:
        raise ValueError("the frozen empty-middle definition contains occupants")

    return {
        "pc1": pc1,
        "pc2": pc2,
        "open_mask": open_mask,
        "variance_ratio": variance_ratio,
        "cumulative_overlap": cumulative_overlap,
        "rmsip": rmsip,
        "global_state": global_state,
        "normalized_coordinate": normalized,
        "normalized_band": normalized_band,
        "divider": float(np.sort(pc1)[::-1][4:6].mean()),
    }


def build_figure(values: dict[str, object]) -> None:
    pc1 = np.asarray(values["pc1"])
    pc2 = np.asarray(values["pc2"])
    open_mask = np.asarray(values["open_mask"])
    variance_ratio = np.asarray(values["variance_ratio"])
    cumulative_overlap = np.asarray(values["cumulative_overlap"])
    rmsip = float(values["rmsip"])
    global_state = np.asarray(values["global_state"])
    normalized = np.asarray(values["normalized_coordinate"])
    normalized_band = np.asarray(values["normalized_band"])
    divider = float(values["divider"])

    apply_publication_style("Fig1")
    fig = plt.figure(figsize=(MAIN_WIDTH_IN, 5.45), dpi=300)
    grid = GridSpec(
        2,
        2,
        figure=fig,
        left=0.095,
        right=0.988,
        bottom=0.092,
        top=0.972,
        wspace=0.37,
        hspace=0.43,
    )

    # a, fixed-date census in the first two PC coordinates.
    axa = fig.add_subplot(grid[0, 0])
    axa.axvline(divider, linestyle=(0, (1.5, 2.0)), color=MID_GREY, linewidth=0.8, zorder=0)
    encodings = (
        ("drug-conditioned", CLOSED, "o", 22, 0.68, "drug-conditioned"),
        ("genuine-apo", GENUINE_APO, "^", 32, 0.96, "genuine apo"),
        ("native-substrate", GREEN, "D", 30, 0.96, "native substrate"),
    )
    for state, color, marker, size, alpha, label in encodings:
        mask = global_state == state
        axa.scatter(
            pc1[mask],
            pc2[mask],
            s=size,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=alpha,
            label=label,
            zorder=3,
        )
    axa.text(
        divider + 0.10,
        0.95,
        "boundary",
        transform=axa.get_xaxis_transform(),
        color=DARK_GREY,
        fontsize=8.0,
        rotation=90,
        ha="left",
        va="top",
    )
    axa.text(
        -0.35,
        0.055,
        "closed",
        transform=axa.get_xaxis_transform(),
        color=DARK_GREY,
        fontsize=8.0,
        fontstyle="italic",
        ha="center",
    )
    axa.text(
        8.88,
        0.055,
        "open",
        transform=axa.get_xaxis_transform(),
        color=DARK_GREY,
        fontsize=8.0,
        fontstyle="italic",
        ha="center",
    )
    axa.set(
        xlabel=f"PC1 ({variance_ratio[0] * 100:.0f}% coordinate variance)",
        ylabel=f"PC2 ({variance_ratio[1] * 100:.0f}% coordinate variance)",
        xlim=(-1.35, 9.35),
        ylim=(-1.62, 1.92),
    )
    axa.legend(
        loc="center",
        bbox_to_anchor=(0.61, 0.45),
        borderaxespad=0,
        labelspacing=0.32,
        handletextpad=0.45,
    )
    finish_axis(axa)
    axa.tick_params(which="both", top=False, right=False)
    panel_label(axa, "a", x=-0.10, y=1.055)

    # b, individual and cumulative PCA coordinate variance.
    axb = fig.add_subplot(grid[0, 1])
    n_components = min(10, len(variance_ratio))
    components = np.arange(1, n_components + 1)
    individual = variance_ratio[:n_components] * 100
    cumulative = np.cumsum(variance_ratio[:n_components]) * 100
    bars = axb.bar(
        components,
        individual,
        width=0.70,
        color=PCA,
        edgecolor=PCA,
        alpha=0.30,
        linewidth=0.7,
        label="individual",
        zorder=2,
    )
    bars[0].set_alpha(0.94)
    axb.plot(
        components,
        cumulative,
        color=BLACK,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=BLACK,
        markeredgewidth=0.7,
        linewidth=1.15,
        label="cumulative",
        zorder=3,
    )
    axb.annotate(
        f"PC1  {individual[0]:.1f}%",
        xy=(1, individual[0]),
        xytext=(2.15, 74),
        color=PCA,
        fontsize=8.0,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": PCA, "linewidth": 0.7},
    )
    axb.set(
        xlabel="Principal component",
        ylabel="Coordinate variance (%)",
        xticks=components,
        ylim=(0, 102),
    )
    axb.legend(loc="lower right", ncol=1, labelspacing=0.25)
    finish_axis(axb, grid="y")
    axb.tick_params(which="both", top=False, right=False)
    panel_label(axb, "b", x=-0.10, y=1.055)

    # c, cumulative projection of the transition axis into ANM subspaces.
    axc = fig.add_subplot(grid[1, 0])
    modes = np.arange(1, 11)
    axc.plot(
        modes,
        cumulative_overlap[:10],
        color=ANM,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=ANM,
        markeredgewidth=0.8,
        linewidth=1.35,
        zorder=3,
    )
    axc.axhline(rmsip, linestyle=(0, (4, 2)), color=PCA, linewidth=1.0, zorder=2)
    axc.text(
        9.85,
        rmsip + 0.007,
        f"ANM–PCA RMSIP  {rmsip:.2f}",
        color=PCA,
        fontsize=8.0,
        ha="right",
        va="bottom",
    )
    axc.annotate(
        f"{cumulative_overlap[0]:.3f}",
        xy=(1, cumulative_overlap[0]),
        xytext=(6, -14),
        textcoords="offset points",
        fontsize=8.0,
        color=ANM,
    )
    axc.annotate(
        f"{cumulative_overlap[9]:.3f}",
        xy=(10, cumulative_overlap[9]),
        xytext=(-2, 7),
        textcoords="offset points",
        fontsize=8.0,
        color=ANM,
        ha="right",
    )
    axc.set(
        xlabel="ANM modes included",
        ylabel="Cumulative projection norm",
        xticks=modes,
        ylim=(0.60, 0.91),
    )
    finish_axis(axc, grid="y")
    axc.tick_params(which="both", top=False, right=False)
    panel_label(axc, "c", x=-0.10, y=1.055)

    # d, distribution in the normalized closed-mean to open-mean coordinate.
    axd = fig.add_subplot(grid[1, 1])
    bins = np.arange(-0.15, 1.15 + 1e-9, 0.05)
    axd.axvspan(
        normalized_band[0],
        normalized_band[1],
        color=LIGHT_GREY,
        alpha=0.58,
        linewidth=0,
        zorder=0,
    )
    for boundary in normalized_band:
        axd.axvline(boundary, color=MID_GREY, linewidth=0.55, linestyle=(0, (2, 2)), zorder=1)
    axd.hist(
        normalized[~open_mask],
        bins=bins,
        color=CLOSED,
        edgecolor=CLOSED,
        linewidth=0.55,
        alpha=0.72,
        hatch="//",
        label=f"closed ({sample_size_label((~open_mask).sum())})",
        zorder=2,
    )
    axd.hist(
        normalized[open_mask],
        bins=bins,
        color=GENUINE_APO,
        edgecolor=GENUINE_APO,
        linewidth=0.7,
        alpha=0.92,
        label=f"open ({sample_size_label(open_mask.sum())})",
        zorder=3,
    )
    axd.text(
        float(normalized_band.mean()),
        0.47,
        "no deposited\nstructures",
        transform=axd.get_xaxis_transform(),
        color=DARK_GREY,
        fontsize=8.0,
        ha="center",
        va="center",
    )
    axd.set(
        xlabel="Transition coordinate",
        ylabel="Structures",
        xlim=(-0.10, 1.08),
    )
    axd.legend(loc="upper center", ncol=1, labelspacing=0.25)
    finish_axis(axd, grid="y")
    axd.tick_params(which="both", top=False, right=False)
    panel_label(axd, "d", x=-0.10, y=1.055)

    save_figure_set(fig, ROOT, "Fig1")
    plt.close(fig)


def main() -> None:
    values = load_inputs()
    build_figure(values)
    normalized_band = np.asarray(values["normalized_band"])
    cumulative_overlap = np.asarray(values["cumulative_overlap"])
    variance_ratio = np.asarray(values["variance_ratio"])
    print(
        "Fig1 built: "
        f"PC1 {variance_ratio[0] * 100:.1f}%; "
        f"cumulative ANM mode 1 {cumulative_overlap[0]:.4f}, "
        f"mode 10 {cumulative_overlap[9]:.4f}; "
        f"RMSIP {float(values['rmsip']):.3f}; "
        f"normalized empty band {normalized_band[0]:.3f}–{normalized_band[1]:.3f}"
    )


if __name__ == "__main__":
    main()
