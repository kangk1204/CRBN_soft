#!/usr/bin/env python3
"""Build supplementary Fig S2: the complete 10 × 10 ANM–PCA overlap matrix."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from figure_package_utils import save_figure_set, write_legend_docx


ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_INPUT = ROOT / "data" / "crbn_ensemble.ens.npz"
ANM_INPUT = ROOT / "data" / "crbn_anm_modes.npz"
SOURCE_DATA = ROOT / "figures" / "source_data" / "FigS2_source_data.csv"
LEGEND_DOCX = ROOT / "figures" / "FigS2_legend.docx"


def load_matrix() -> tuple[np.ndarray, float]:
    with np.load(ENSEMBLE_INPUT, allow_pickle=False) as ensemble:
        conformers = np.asarray(ensemble["_confs"], dtype=float)
    with np.load(ANM_INPUT, allow_pickle=False) as anm:
        anm_modes = np.asarray(anm["anm_eigvecs"], dtype=float)
        saved_overlap = np.asarray(anm["overlap_anm_pca"], dtype=float)
        saved_rmsip = float(anm["rmsip"])

    if conformers.shape != (70, 269, 3):
        raise ValueError(f"expected a 70 × 269 × 3 ensemble, found {conformers.shape}")
    if anm_modes.shape[0] != 269 * 3 or anm_modes.shape[1] < 10:
        raise ValueError(f"ANM eigenvector array cannot supply ten 807-component modes: {anm_modes.shape}")
    if not np.isfinite(conformers).all() or not np.isfinite(anm_modes[:, :10]).all():
        raise ValueError("ensemble or ANM input contains non-finite values")

    centered = (conformers - conformers.mean(axis=0)).reshape(len(conformers), -1)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    pca_modes = right_vectors.T[:, :10]
    matrix = np.abs(anm_modes[:, :10].T @ pca_modes)
    rmsip = float(np.sqrt(np.square(matrix).sum() / 10.0))

    if saved_overlap.shape != (10, 10):
        raise ValueError(f"saved ANM–PCA overlap matrix must be 10 × 10, found {saved_overlap.shape}")
    saved_difference = float(np.max(np.abs(matrix - np.abs(saved_overlap))))
    if saved_difference > 1e-10:
        raise ValueError(f"recomputed matrix conflicts with the saved overlap matrix ({saved_difference:.2e})")
    if abs(rmsip - saved_rmsip) > 1e-10:
        raise ValueError(f"recomputed RMSIP {rmsip:.12f} conflicts with saved {saved_rmsip:.12f}")
    return matrix, rmsip


def write_source_data(matrix: np.ndarray, rmsip: float) -> None:
    SOURCE_DATA.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_type", "anm_mode", "principal_component", "value", "unit"])
        for mode_index in range(10):
            for pc_index in range(10):
                writer.writerow(
                    [
                        "mode_pc_overlap",
                        mode_index + 1,
                        pc_index + 1,
                        f"{matrix[mode_index, pc_index]:.12f}",
                        "absolute directional overlap",
                    ]
                )
        writer.writerow(["summary", "", "", f"{rmsip:.12f}", "10-mode RMSIP"])


def build_figure(matrix: np.ndarray) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "crbn-FigS2",
        }
    )
    fig, axis = plt.subplots(figsize=(6.8, 5.6))
    image = axis.imshow(
        matrix,
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=1,
        aspect="equal",
        extent=[0.5, 10.5, 0.5, 10.5],
    )
    axis.set_xticks(range(1, 11), labels=[f"PC{index}" for index in range(1, 11)])
    axis.set_yticks(range(1, 11))
    axis.set_xlabel("Ensemble principal component")
    axis.set_ylabel("Network-model mode")
    axis.text(
        1,
        1,
        f"{matrix[0, 0]:.2f}",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        fontweight="bold",
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Absolute directional overlap")
    fig.tight_layout()
    save_figure_set(fig, ROOT, "FigS2")
    plt.close(fig)


def main() -> None:
    matrix, rmsip = load_matrix()
    write_source_data(matrix, rmsip)
    build_figure(matrix)
    write_legend_docx(
        LEGEND_DOCX,
        "Fig S2. Overlap between anisotropic network model (ANM) modes and principal "
        "component analysis (PCA) axes. Directional overlap is the absolute normalised dot "
        "product (0 for orthogonal directions and 1 for the same axis). The heat map compares "
        "the ten lowest-frequency "
        "open-state ANM modes and the ten leading Protein Data Bank (PDB)-derived principal "
        "components. The root-mean-square inner product (RMSIP) compares the two ten-dimensional "
        f"motion subspaces on the same 0-to-1 scale and is {rmsip:.2f}; the directional overlap between "
        f"mode 1 and the first principal component (PC1) is {matrix[0, 0]:.2f}.",
    )
    print(
        f"FigS2 built: matrix {matrix.shape[0]}x{matrix.shape[1]}, "
        f"RMSIP {rmsip:.3f}, mode1-PC1 {matrix[0, 0]:.3f}"
    )


if __name__ == "__main__":
    main()
