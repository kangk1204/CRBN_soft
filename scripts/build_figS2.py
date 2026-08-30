#!/usr/bin/env python3
"""Build Fig. S2: complete 10 x 10 ANM-PCA overlap matrix."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from figure_package_utils import save_figure_set
from figure_style import BLACK, DARK_GREY, MAIN_WIDTH_IN, apply_publication_style

ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_INPUT = ROOT / "data" / "crbn_ensemble.ens.npz"
ANM_INPUT = ROOT / "data" / "crbn_anm_modes.npz"
SOURCE_DATA = ROOT / "figures" / "source_data" / "FigS2_source_data.csv"


def load_matrix() -> tuple[np.ndarray, float]:
    with np.load(ENSEMBLE_INPUT, allow_pickle=False) as ensemble:
        conformers = np.asarray(ensemble["_confs"], dtype=float)
    with np.load(ANM_INPUT, allow_pickle=False) as anm:
        anm_modes = np.asarray(anm["anm_eigvecs"], dtype=float)
        saved_overlap = np.asarray(anm["overlap_anm_pca"], dtype=float)
        saved_rmsip = float(anm["rmsip"])

    if conformers.shape != (70, 269, 3):
        raise ValueError(f"expected a 70 x 269 x 3 ensemble, found {conformers.shape}")
    if anm_modes.shape[0] != 269 * 3 or anm_modes.shape[1] < 10:
        raise ValueError(f"ANM array cannot supply ten 807-component modes: {anm_modes.shape}")
    if not np.isfinite(conformers).all() or not np.isfinite(anm_modes[:, :10]).all():
        raise ValueError("ensemble or ANM input contains non-finite values")

    centered = (conformers - conformers.mean(axis=0)).reshape(len(conformers), -1)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    pca_modes = right_vectors.T[:, :10]
    matrix = np.abs(anm_modes[:, :10].T @ pca_modes)
    rmsip = float(np.sqrt(np.square(matrix).sum() / 10.0))

    if saved_overlap.shape != (10, 10):
        raise ValueError(f"saved ANM-PCA overlap matrix must be 10 x 10, found {saved_overlap.shape}")
    saved_difference = float(np.max(np.abs(matrix - np.abs(saved_overlap))))
    if saved_difference > 1e-10:
        raise ValueError(f"recomputed matrix conflicts with saved overlap ({saved_difference:.2e})")
    if abs(rmsip - saved_rmsip) > 1e-10:
        raise ValueError(f"recomputed RMSIP {rmsip:.12f} conflicts with saved {saved_rmsip:.12f}")
    if np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("absolute directional overlaps must lie on the closed [0, 1] interval")
    return matrix, rmsip


def write_source_data(matrix: np.ndarray, rmsip: float) -> None:
    SOURCE_DATA.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_type", "anm_mode", "principal_component", "value", "unit"])
        for mode_index in range(10):
            for component_index in range(10):
                writer.writerow(
                    [
                        "mode_pc_overlap",
                        mode_index + 1,
                        component_index + 1,
                        f"{matrix[mode_index, component_index]:.12f}",
                        "absolute directional overlap",
                    ]
                )
        writer.writerow(["summary", "", "", f"{rmsip:.12f}", "10-mode RMSIP"])


def build_figure(matrix: np.ndarray, rmsip: float) -> None:
    apply_publication_style("FigS2")
    figure, axis = plt.subplots(figsize=(MAIN_WIDTH_IN, 5.15))
    image = axis.imshow(
        matrix,
        origin="lower",
        cmap="cividis",
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
        extent=[0.5, 10.5, 0.5, 10.5],
        rasterized=True,
    )

    axis.set_xticks(range(1, 11), labels=[f"PC{index}" for index in range(1, 11)])
    axis.set_yticks(range(1, 11), labels=[f"{index}" for index in range(1, 11)])
    axis.set_xticks(np.arange(1.5, 10.5, 1.0), minor=True)
    axis.set_yticks(np.arange(1.5, 10.5, 1.0), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.45, alpha=0.48)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_xlabel("Ensemble principal component")
    axis.set_ylabel("ANM mode")
    axis.tick_params(which="both", top=False, right=False)

    # Numerical labels for the seven cells at or above 0.40 make the salient
    # matching pattern legible in greyscale and ensure no interpretation
    # depends on hue alone.
    for mode_index, component_index in np.argwhere(matrix >= 0.40):
        value = float(matrix[mode_index, component_index])
        axis.text(
            component_index + 1,
            mode_index + 1,
            f"{value:.2f}",
            color=BLACK if value >= 0.62 else "white",
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="center",
        )
    axis.add_patch(
        plt.Rectangle(
            (0.505, 0.505),
            0.99,
            0.99,
            fill=False,
            edgecolor="white",
            linewidth=1.15,
        )
    )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_label("Absolute directional overlap", labelpad=5)
    colorbar.outline.set_linewidth(0.65)
    axis.text(
        1.0,
        1.015,
        f"10-mode RMSIP = {rmsip:.2f}",
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=DARK_GREY,
        ha="right",
        va="bottom",
        clip_on=False,
    )

    figure.subplots_adjust(left=0.12, right=0.89, top=0.94, bottom=0.115)
    save_figure_set(figure, ROOT, "FigS2")
    plt.close(figure)


def main() -> None:
    matrix, rmsip = load_matrix()
    write_source_data(matrix, rmsip)
    build_figure(matrix, rmsip)
    print(
        f"FigS2 built: matrix {matrix.shape[0]}x{matrix.shape[1]}, "
        f"RMSIP {rmsip:.12f}, mode1-PC1 {matrix[0, 0]:.12f}, "
        f"labelled cells {(matrix >= 0.40).sum()}"
    )


if __name__ == "__main__":
    main()
