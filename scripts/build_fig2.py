#!/usr/bin/env python3
"""Build Fig. 2 from frozen ANM/PCA arrays and structural raster panels.

Panels b and c are immutable PyMOL renders. Their SHA256 values are checked
before composition; this script only places them in the final figure. Panels a
and d are recomputed from the supplied numeric arrays without filtering.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_package_utils import save_figure_set
from figure_style import (
    ANM,
    CLOSED,
    DARK_GREY,
    HB,
    LIGHT_GREY,
    MAIN_WIDTH_IN,
    NTD,
    TBD,
    apply_publication_style,
    finish_axis,
    panel_label,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PANELS = ROOT / "figures" / "panels"
FROZEN_PANEL_SHA256 = {
    "fig2_anm3d.png": "17c8b267c8a06e2b7da5a1665a96979176992dd66ad67998973be58fd47a0de9",
    "fig2_pc13d.png": "e59077e6b8361b1aa70852de304c3f265da9f52d01ddf51b1d0dafcb2d2f2213",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_structural_rasters() -> None:
    """Fail closed if either frozen scientific image was changed or omitted."""
    for filename, expected_hash in FROZEN_PANEL_SHA256.items():
        path = PANELS / filename
        if not path.is_file():
            raise FileNotFoundError(f"required frozen structural panel is missing: {path}")
        observed_hash = sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"frozen structural panel changed: {filename}; "
                f"expected {expected_hash}, observed {observed_hash}"
            )


def anm_hessian(coords: np.ndarray, cutoff: float) -> np.ndarray:
    """Construct the same unit-spring ANM Hessian used in robustness analysis."""
    n_nodes = len(coords)
    hessian = np.zeros((3 * n_nodes, 3 * n_nodes))
    for i in range(n_nodes):
        displacement = coords - coords[i]
        distance = np.linalg.norm(displacement, axis=1)
        for j in range(i + 1, n_nodes):
            if 1e-6 < distance[j] <= cutoff:
                spring = np.outer(displacement[j], displacement[j]) / distance[j] ** 2
                hessian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -spring
                hessian[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = -spring
                hessian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += spring
                hessian[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += spring
    return hessian


def mode_overlaps(
    coords: np.ndarray,
    difference_vector: np.ndarray,
    n_modes: int = 10,
    cutoff: float = 15.0,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(anm_hessian(coords, cutoff))
    nonzero = eigenvalues > 1e-9
    modes = eigenvectors[:, nonzero][:, :n_modes]
    if modes.shape[1] != n_modes:
        raise ValueError(f"expected {n_modes} non-zero ANM modes, found {modes.shape[1]}")
    return np.array([abs(modes[:, index] @ difference_vector) for index in range(n_modes)])


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anm = np.load(DATA / "crbn_anm_modes.npz", allow_pickle=False)
    open_spectrum = np.abs(np.asarray(anm["anm_diff_overlap"][:10], dtype=float))

    ensemble = np.load(DATA / "crbn_ensemble.ens.npz", allow_pickle=False)
    conformers = np.asarray(ensemble["_confs"], dtype=float)
    labels = [str(value) for value in ensemble["_labels"]]
    diffvec = np.load(DATA / "pca_diffvec.npz", allow_pickle=False)
    open_labels = {
        str(label)
        for label, is_open in zip(diffvec["labels"], diffvec["open_mask"])
        if bool(is_open)
    }
    open_mask = np.array([label in open_labels for label in labels])
    if conformers.shape != (70, 269, 3) or len(labels) != 70:
        raise ValueError(f"unexpected ensemble shape: {conformers.shape}")
    if len(set(labels)) != 70 or int(open_mask.sum()) != 5:
        raise ValueError("ensemble labels must contain five unique open and 65 closed entries")
    if "8CVP" not in labels or "5FQD" not in labels:
        raise ValueError("Fig. 2 endpoint structures 8CVP and 5FQD are absent")

    displacement = conformers[open_mask].mean(axis=0) - conformers[~open_mask].mean(axis=0)
    difference_vector = (displacement / np.linalg.norm(displacement)).ravel()
    open_endpoint = mode_overlaps(conformers[labels.index("8CVP")], difference_vector)
    closed_endpoint = mode_overlaps(conformers[labels.index("5FQD")], difference_vector)
    for name, values in (
        ("open-state spectrum", open_spectrum),
        ("8CVP spectrum", open_endpoint),
        ("5FQD spectrum", closed_endpoint),
    ):
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError(f"invalid {name}: shape={values.shape}")
    # The archived ensemble coordinates are rounded slightly differently from
    # the values used to store the spectrum. The complete ten-mode spectra must
    # nevertheless reproduce within 1e-4 (observed maximum difference < 8e-5).
    if not np.allclose(open_spectrum, open_endpoint, rtol=0, atol=1e-4):
        delta = float(np.max(np.abs(open_spectrum - open_endpoint)))
        raise ValueError(f"stored and recomputed open spectra differ by {delta:.3g}")
    return open_spectrum, open_endpoint, closed_endpoint


def structural_panel(ax, filename: str, direct_label: str, letter: str, *, show_key: bool) -> None:
    """Compose one frozen PyMOL image without changing its pixels."""
    image = mpimg.imread(PANELS / filename)
    ax.imshow(image, interpolation="none")
    ax.set_axis_off()
    ax.text(
        0.025,
        0.975,
        direct_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=DARK_GREY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )
    if show_key:
        handles = [Line2D([0], [0], color=color, lw=3.0) for color in (NTD, HB, TBD)]
        ax.legend(
            handles,
            ["NTD", "HB", "TBD"],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=3,
            handlelength=1.7,
            borderaxespad=0,
        )
    panel_label(ax, letter, x=-0.035, y=1.025)


def build_figure(
    open_spectrum: np.ndarray,
    open_endpoint: np.ndarray,
    closed_endpoint: np.ndarray,
) -> None:
    apply_publication_style("Fig2")
    fig = plt.figure(figsize=(MAIN_WIDTH_IN, 5.80), dpi=300)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.988,
        bottom=0.083,
        top=0.974,
        wspace=0.31,
        hspace=0.27,
    )
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1])
    axc = fig.add_subplot(grid[1, 0])
    axd = fig.add_subplot(grid[1, 1])
    mode_numbers = np.arange(1, 11)

    # a, open-state ANM spectrum with mode 1 as the pre-specified focal result.
    bars = axa.bar(
        mode_numbers,
        open_spectrum,
        width=0.70,
        color=LIGHT_GREY,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    bars[0].set_facecolor(ANM)
    bars[0].set_edgecolor(ANM)
    axa.annotate(
        f"{open_spectrum[0]:.3f}",
        xy=(1, open_spectrum[0]),
        xytext=(6, -2),
        textcoords="offset points",
        color=ANM,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axa.set(
        xlabel="Open-state ANM mode",
        ylabel="Directional overlap\nwith open–closed axis",
        xticks=mode_numbers,
        ylim=(0, 0.78),
    )
    finish_axis(axa, grid="y")
    axa.tick_params(which="both", top=False, right=False)
    panel_label(axa, "a", x=-0.105, y=1.055)

    # b/c, immutable structural-vector images; one shared domain key avoids a
    # redundant second legend while retaining the mapping beside the images.
    structural_panel(axb, "fig2_anm3d.png", "Open-state ANM mode 1", "b", show_key=True)
    structural_panel(axc, "fig2_pc13d.png", "PDB-ensemble PC1", "c", show_key=False)

    # d, endpoint-specific ANM mode spectra with position plus hatch redundancy.
    width = 0.36
    axd.bar(
        mode_numbers - width / 2,
        open_endpoint,
        width,
        color=ANM,
        edgecolor=ANM,
        linewidth=0.6,
        label="from open (8CVP)",
        zorder=3,
    )
    axd.bar(
        mode_numbers + width / 2,
        closed_endpoint,
        width,
        color=CLOSED,
        edgecolor=CLOSED,
        linewidth=0.6,
        hatch="//",
        alpha=0.82,
        label="from closed (5FQD)",
        zorder=3,
    )
    axd.annotate(
        f"{open_endpoint[0]:.2f}",
        xy=(1 - width / 2, open_endpoint[0]),
        xytext=(0, 5),
        textcoords="offset points",
        color=ANM,
        fontsize=8.0,
        fontweight="bold",
        ha="center",
    )
    closed_best = int(np.argmax(closed_endpoint))
    axd.annotate(
        f"{closed_endpoint[closed_best]:.2f}",
        xy=(closed_best + 1 + width / 2, closed_endpoint[closed_best]),
        xytext=(0, 5),
        textcoords="offset points",
        color=CLOSED,
        fontsize=8.0,
        fontweight="bold",
        ha="center",
    )
    axd.set(
        xlabel="ANM mode",
        ylabel="Directional overlap\nwith open–closed axis",
        xticks=mode_numbers,
        ylim=(0, 0.80),
    )
    axd.legend(loc="upper right", labelspacing=0.28)
    finish_axis(axd, grid="y")
    axd.tick_params(which="both", top=False, right=False)
    panel_label(axd, "d", x=-0.105, y=1.055)

    save_figure_set(fig, ROOT, "Fig2")
    plt.close(fig)


def main() -> None:
    verify_structural_rasters()
    open_spectrum, open_endpoint, closed_endpoint = load_inputs()
    build_figure(open_spectrum, open_endpoint, closed_endpoint)
    print(
        "Fig2 built: "
        f"open mode 1 {open_spectrum[0]:.3f}; "
        f"8CVP best mode {int(open_endpoint.argmax()) + 1} ({open_endpoint.max():.3f}); "
        f"5FQD best mode {int(closed_endpoint.argmax()) + 1} ({closed_endpoint.max():.3f}); "
        "frozen structural raster hashes verified"
    )


if __name__ == "__main__":
    main()
