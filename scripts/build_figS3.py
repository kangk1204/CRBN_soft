#!/usr/bin/env python3
"""Build supplementary Fig S3: publication-group bootstrap robustness."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from figure_package_utils import save_figure_set, write_legend_docx


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS_INPUT = ROOT / "data" / "pca_robust.npz"
ANM_INPUT = ROOT / "data" / "crbn_anm_modes.npz"
SOURCE_DATA = ROOT / "figures" / "source_data" / "FigS3_source_data.csv"
LEGEND_DOCX = ROOT / "figures" / "FigS3_legend.docx"

BLUE = "#3B6EA8"
ORANGE = "#D9792B"
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
    if values["n_groups"] != 43:
        raise ValueError(f"expected 43 publication groups, found {values['n_groups']}")
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
    anm_mode1_overlap = float(values["anm_mode1_overlap"])
    variance_low, variance_high = np.percentile(variance, [2.5, 97.5])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "crbn-FigS3",
        }
    )
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.20, wspace=0.30)

    axa.hist(variance, bins=HISTOGRAM_BINS, color=BLUE, alpha=0.85, linewidth=0)
    axa.axvline(full_variance, color="black", linestyle="--", linewidth=1.2)
    axa.axvline(variance_low, color="#888888", linestyle=":", linewidth=0.9)
    axa.axvline(variance_high, color="#888888", linestyle=":", linewidth=0.9)
    axa.annotate(
        f"full ensemble\n{full_variance:.0f}%",
        xy=(full_variance, axa.get_ylim()[1] * 0.97),
        xytext=(full_variance - 10, axa.get_ylim()[1] * 0.83),
        fontsize=7.5,
        ha="center",
        va="top",
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "black"},
    )
    axa.set_xlabel("First principal component variance fraction (%)")
    axa.set_ylabel("bootstrap resamples")

    axb.hist(overlap, bins=HISTOGRAM_BINS, color=ORANGE, alpha=0.85, linewidth=0)
    axb.axvline(anm_mode1_overlap, color="black", linewidth=1.4)
    axb.text(
        anm_mode1_overlap - 0.02,
        axb.get_ylim()[1] * 0.58,
        f"open-structure model\nmode 1 ({anm_mode1_overlap:.2f})",
        fontsize=7.5,
        ha="right",
        va="center",
    )
    axb.set_xlim(0.0, 1.02)
    axb.set_xlabel("Directional overlap with open–closed axis")
    axb.set_ylabel("bootstrap resamples")

    for axis, label in ((axa, "a"), (axb, "b")):
        axis.text(
            -0.14,
            1.07,
            f"({label})",
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="right",
        )

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
    write_legend_docx(
        LEGEND_DOCX,
        "Fig S3. Bootstrap stability of the Protein Data Bank (PDB)-derived coordinate. "
        f"Publication-group bootstrap resampling across {int(values['n_groups'])} groups "
        f"gave a first principal component (PC1) variance fraction of {variance.mean():.0f}% "
        f"(2.5th–97.5th percentile range {variance_low:.0f}–{variance_high:.0f}%) and "
        f"a PC1–axis directional overlap of {overlap.mean():.2f} "
        f"({overlap_low:.2f}–{overlap_high:.2f}). Five per cent of resamples contained "
        "no open structure. The black line in panel b marks the single-open-structure "
        f"anisotropic network model (ANM) mode-1 directional overlap "
        f"({float(values['anm_mode1_overlap']):.2f}). Directional overlap is the absolute "
        "normalised dot product, ranging from 0 for orthogonal directions to 1 for the same axis.",
    )
    print(
        f"FigS3 built: variance {variance.mean():.0f}% [{variance_low:.0f}, {variance_high:.0f}], "
        f"overlap {overlap.mean():.3f} [{overlap_low:.3f}, {overlap_high:.3f}]"
    )


if __name__ == "__main__":
    main()
