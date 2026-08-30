#!/usr/bin/env python3
"""Build Fig. S3: publication-group bootstrap robustness."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from figure_package_utils import save_figure_set
from figure_style import (
    BLACK,
    BLUE,
    DARK_GREY,
    MID_GREY,
    PALE_BLUE,
    PALE_ORANGE,
    PURPLE,
    apply_publication_style,
    finish_axis,
    panel_label,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_WIDTH_IN = 6.50  # allows for tight-bbox panel-label padding below 170 mm
ROBUSTNESS_INPUT = ROOT / "data" / "pca_robust.npz"
ANM_INPUT = ROOT / "data" / "crbn_anm_modes.npz"
SOURCE_DATA = ROOT / "figures" / "source_data" / "FigS3_source_data.csv"

HISTOGRAM_BINS = 28


def load_data() -> dict[str, np.ndarray | float | int]:
    with np.load(ROBUSTNESS_INPUT, allow_pickle=False) as robustness:
        values: dict[str, np.ndarray | float | int] = {
            "variance": np.asarray(robustness["vfs"], dtype=float),
            "overlap": np.asarray(robustness["ovs"], dtype=float),
            "full_variance": float(robustness["vf0"]) * 100.0,
            "full_overlap": float(robustness["ov0"]),
            "fraction_without_open": float(robustness["frac_resamples_without_open"]),
            "n_groups": int(robustness["n_groups"]),
        }
    with np.load(ANM_INPUT, allow_pickle=False) as anm:
        values["anm_mode1_overlap"] = float(anm["anm_diff_overlap"][0])

    variance = values["variance"]
    overlap = values["overlap"]
    assert isinstance(variance, np.ndarray) and isinstance(overlap, np.ndarray)
    if variance.shape != (2000,) or overlap.shape != (2000,):
        raise ValueError(f"expected 2,000 paired bootstrap records, found {variance.shape} and {overlap.shape}")
    if not np.isfinite(variance).all() or not np.isfinite(overlap).all():
        raise ValueError("bootstrap arrays contain non-finite values")
    if (variance < 0).any() or (variance > 100).any() or (overlap < 0).any() or (overlap > 1).any():
        raise ValueError("bootstrap values fall outside their defined ranges")
    if values["n_groups"] != 38:
        raise ValueError(f"expected 38 fail-closed publication groups, found {values['n_groups']}")
    return values


def write_source_data(values: dict[str, np.ndarray | float | int]) -> None:
    variance = np.asarray(values["variance"])
    overlap = np.asarray(values["overlap"])
    variance_counts, variance_edges = np.histogram(variance, bins=HISTOGRAM_BINS)
    overlap_counts, overlap_edges = np.histogram(overlap, bins=HISTOGRAM_BINS)
    variance_low, variance_high = np.percentile(variance, [2.5, 97.5])

    SOURCE_DATA.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["record_type", "panel", "index", "metric", "x_left", "x_right", "value", "unit"]
        )
        for index, value in enumerate(variance, start=1):
            writer.writerow(["bootstrap", "a", index, "PC1 variance", "", "", f"{value:.12f}", "percent"])
        for index, value in enumerate(overlap, start=1):
            writer.writerow(
                [
                    "bootstrap",
                    "b",
                    index,
                    "PC1-axis directional overlap",
                    "",
                    "",
                    f"{value:.12f}",
                    "absolute directional overlap",
                ]
            )
        for index, (left, right, count) in enumerate(
            zip(variance_edges[:-1], variance_edges[1:], variance_counts), start=1
        ):
            writer.writerow(["histogram_bin", "a", index, "PC1 variance", f"{left:.12f}", f"{right:.12f}", int(count), "resamples"])
        for index, (left, right, count) in enumerate(
            zip(overlap_edges[:-1], overlap_edges[1:], overlap_counts), start=1
        ):
            writer.writerow(
                [
                    "histogram_bin",
                    "b",
                    index,
                    "PC1-axis directional overlap",
                    f"{left:.12f}",
                    f"{right:.12f}",
                    int(count),
                    "resamples",
                ]
            )
        references = [
            ("a", "full-ensemble variance", values["full_variance"], "percent"),
            ("a", "2.5th percentile", variance_low, "percent"),
            ("a", "97.5th percentile", variance_high, "percent"),
            (
                "b",
                "full-ensemble PC1-axis directional overlap",
                values["full_overlap"],
                "absolute directional overlap",
            ),
            (
                "b",
                "single-open ANM mode-1 directional overlap",
                values["anm_mode1_overlap"],
                "absolute directional overlap",
            ),
            ("b", "resamples without an open structure", 100.0 * float(values["fraction_without_open"]), "percent"),
        ]
        for panel, metric, value, unit in references:
            writer.writerow(["reference", panel, "", metric, "", "", f"{float(value):.12f}", unit])


def build_figure(values: dict[str, np.ndarray | float | int]) -> None:
    variance = np.asarray(values["variance"])
    overlap = np.asarray(values["overlap"])
    full_variance = float(values["full_variance"])
    full_overlap = float(values["full_overlap"])
    anm_mode1_overlap = float(values["anm_mode1_overlap"])
    variance_low, variance_high = np.percentile(variance, [2.5, 97.5])
    overlap_low, overlap_high = np.percentile(overlap, [2.5, 97.5])

    apply_publication_style("FigS3")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, 3.10), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.20, wspace=0.32)

    axa.axvspan(variance_low, variance_high, color=PALE_BLUE, zorder=0)
    axa.hist(
        variance,
        bins=HISTOGRAM_BINS,
        color=BLUE,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    axa.axvline(full_variance, color=BLACK, linestyle="--", linewidth=1.25, zorder=4)
    axa.axvline(variance_low, color=DARK_GREY, linestyle=":", linewidth=1.0, zorder=3)
    axa.axvline(variance_high, color=DARK_GREY, linestyle=":", linewidth=1.0, zorder=3)
    ymax_a = axa.get_ylim()[1]
    axa.annotate(
        f"full ensemble\n{full_variance:.1f}%",
        xy=(full_variance, ymax_a * 0.96),
        xytext=(full_variance - 5.2, ymax_a * 0.78),
        fontsize=8.0,
        color=BLACK,
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "-", "linewidth": 0.65, "color": BLACK},
    )
    axa.text(
        0.02,
        0.97,
        f"95% bootstrap range\n{variance_low:.1f}–{variance_high:.1f}%",
        transform=axa.transAxes,
        fontsize=8.0,
        color=DARK_GREY,
        ha="left",
        va="top",
    )
    axa.set_xlabel("First principal component variance fraction (%)")
    axa.set_ylabel("Bootstrap resamples")
    finish_axis(axa, grid="y")
    axa.tick_params(which="both", top=False, right=False)

    # A log count axis is explicit and necessary here: 1,875 of 2,000 values
    # occupy the uppermost bin, whereas occupied tail bins contain as few as one
    # value. A linear axis would visually erase that tail.
    axb.axvspan(overlap_low, overlap_high, color=PALE_ORANGE, zorder=0)
    axb.hist(
        overlap,
        bins=HISTOGRAM_BINS,
        color=PURPLE,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    axb.set_yscale("log")
    axb.set_ylim(0.8, 3000)
    # Plain-text major labels avoid undersized mathtext exponents while
    # retaining the explicitly labelled logarithmic axis.
    axb.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    axb.tick_params(axis="y", which="minor", labelleft=False)
    axb.axvline(anm_mode1_overlap, color=BLACK, linewidth=1.35, zorder=4)
    axb.axvline(full_overlap, color=DARK_GREY, linestyle="--", linewidth=1.15, zorder=4)
    axb.axvline(overlap_low, color=MID_GREY, linestyle=":", linewidth=1.0, zorder=3)
    axb.axvline(overlap_high, color=MID_GREY, linestyle=":", linewidth=1.0, zorder=3)
    axb.annotate(
        f"ANM mode 1\n{anm_mode1_overlap:.3f}",
        xy=(anm_mode1_overlap, 55),
        xytext=(anm_mode1_overlap - 0.05, 220),
        fontsize=8.0,
        color=BLACK,
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "-", "linewidth": 0.65, "color": BLACK},
    )
    axb.annotate(
        f"full-ensemble PC1\n{full_overlap:.3f}",
        xy=(full_overlap, 1900),
        xytext=(0.96, 500),
        fontsize=8.0,
        color=DARK_GREY,
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "-", "linewidth": 0.65, "color": DARK_GREY},
    )
    axb.text(
        0.02,
        0.97,
        f"95% range {overlap_low:.3f}–{overlap_high:.3f}\n"
        f"no open structure: {100.0 * float(values['fraction_without_open']):.1f}%",
        transform=axb.transAxes,
        fontsize=8.0,
        color=DARK_GREY,
        ha="left",
        va="top",
    )
    axb.set_xlim(0.0, 1.02)
    axb.set_xlabel("Directional overlap with open–closed axis")
    axb.set_ylabel("Bootstrap resamples (log count)")
    finish_axis(axb, grid="y")
    axb.tick_params(which="both", top=False, right=False)

    panel_label(axa, "a", x=-0.18, y=1.10)
    panel_label(axb, "b", x=-0.18, y=1.10)

    save_figure_set(fig, ROOT, "FigS3")
    plt.close(fig)


def main() -> None:
    values = load_data()
    write_source_data(values)
    build_figure(values)
    variance = np.asarray(values["variance"])
    overlap = np.asarray(values["overlap"])
    variance_low, variance_high = np.percentile(variance, [2.5, 97.5])
    overlap_low, overlap_high = np.percentile(overlap, [2.5, 97.5])
    print(
        f"FigS3 built: variance {variance.mean():.0f}% [{variance_low:.0f}, {variance_high:.0f}], "
        f"overlap {overlap.mean():.3f} [{overlap_low:.3f}, {overlap_high:.3f}]"
    )


if __name__ == "__main__":
    main()
