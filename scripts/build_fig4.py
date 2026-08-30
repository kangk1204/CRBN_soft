#!/usr/bin/env python3
"""Build Fig. 4: residue, pocket-definition and structural mobility views.

Panel (b) keeps the three pre-specified UniProt ligand annotations separate
from the seven 5FQD LVY heavy-atom contacts in the common analysis window.
All individual percentiles are displayed; no inferential symbol, interval or
significance bracket is added. The committed panel-(c) structural raster is
cropped only to its alpha bounding box and is otherwise unchanged.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from figure_package_utils import save_figure_set
from figure_style import (
    AMBER,
    ANM,
    BLACK,
    DARK_GREY,
    LIGHT_GREY,
    MAIN_WIDTH_IN,
    ORANGE,
    PALE_ORANGE,
    PCA,
    apply_publication_style,
    finish_axis,
    panel_label,
    sample_size_label,
)
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODES_INPUT = ROOT / "data" / "crbn_anm_modes.npz"
FLUCTUATION_INPUT = ROOT / "data" / "crbn_residue_fluctuations.csv"
STRUCTURE_INPUT = ROOT / "figures" / "panels" / "render_closed_pocket.png"
FROZEN_STRUCTURE_SHA256 = "e72b571169fe71ce6b6dc4c50cde67adce3fff9ae696ed3ef3482353f1e4f072"

ANNOTATED_RESIDUES = (378, 380, 386)
CONTACT_RESIDUES = (377, 378, 379, 380, 386, 400, 402)
ZINC_RESIDUES = (323, 326, 391, 394)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_structure_input() -> None:
    """Require the frozen reference render used by the composite figure."""
    if not STRUCTURE_INPUT.is_file():
        raise FileNotFoundError(f"required frozen structural panel is missing: {STRUCTURE_INPUT}")
    observed = _sha256(STRUCTURE_INPUT)
    if observed != FROZEN_STRUCTURE_SHA256:
        raise ValueError(
            "frozen structural panel changed: "
            f"expected {FROZEN_STRUCTURE_SHA256}, observed {observed}"
        )


def _load_anm_profile() -> tuple[np.ndarray, np.ndarray]:
    with np.load(MODES_INPUT, allow_pickle=False) as modes:
        residues = np.asarray(modes["resnums"], dtype=int)
        eigenvectors = np.asarray(modes["anm_eigvecs"], dtype=float)
        eigenvalues = np.asarray(modes["anm_eigvals"], dtype=float)

    if eigenvectors.shape[0] != 3 * len(residues) or eigenvectors.shape[1] < 10:
        raise ValueError(f"ANM modes do not match the residue window: {eigenvectors.shape}")
    if eigenvalues.shape[0] < 10 or np.any(eigenvalues[:10] <= 0):
        raise ValueError("ANM input cannot supply ten positive-eigenvalue modes")
    if len(np.unique(residues)) != len(residues) or np.any(np.diff(residues) <= 0):
        raise ValueError("ANM residue identifiers must be unique and strictly increasing")

    square_fluctuation = np.zeros(len(residues), dtype=float)
    for mode_index in range(10):
        mode = eigenvectors[:, mode_index].reshape(-1, 3)
        square_fluctuation += np.square(mode).sum(axis=1) / eigenvalues[mode_index]
    square_fluctuation /= float(square_fluctuation.max())
    if not np.isfinite(square_fluctuation).all():
        raise ValueError("ANM square-fluctuation profile contains non-finite values")
    return residues, square_fluctuation


def _load_group_percentiles() -> tuple[dict[str, dict[str, list[float]]], np.ndarray]:
    with FLUCTUATION_INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    residues = np.asarray([int(row["resnum"]) for row in rows], dtype=int)
    profiles = {
        "ANM": np.asarray([float(row["anm_sqfluct"]) for row in rows], dtype=float),
        "PCA": np.asarray([float(row["pca_sqfluct"]) for row in rows], dtype=float),
    }
    if len(rows) != 269 or len(np.unique(residues)) != len(residues):
        raise ValueError("expected 269 unique residues in the common ANM/PCA analysis window")
    if not all(np.isfinite(profile).all() for profile in profiles.values()):
        raise ValueError("mobility profile contains non-finite values")

    groups: dict[str, dict[str, list[float]]] = {}
    for method, profile in profiles.items():
        by_residue = {int(residue): float(value) for residue, value in zip(residues, profile)}
        missing = (set(CONTACT_RESIDUES) | set(ZINC_RESIDUES)) - set(by_residue)
        if missing:
            raise ValueError(f"{method} profile is missing focal residues {sorted(missing)}")

        groups[method] = {
            "annotated": [
                100.0 * float(np.mean(profile <= by_residue[residue]))
                for residue in ANNOTATED_RESIDUES
            ],
            "contact": [
                100.0 * float(np.mean(profile <= by_residue[residue]))
                for residue in CONTACT_RESIDUES
            ],
            "zinc": [
                100.0 * float(np.mean(profile <= by_residue[residue]))
                for residue in ZINC_RESIDUES
            ],
        }
    return groups, residues


def _break_at_gaps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_out: list[float] = [float(x[0])]
    y_out: list[float] = [float(y[0])]
    for index in range(1, len(x)):
        if x[index] - x[index - 1] > 1:
            x_out.append(float(x[index - 1]) + 0.5)
            y_out.append(np.nan)
        x_out.append(float(x[index]))
        y_out.append(float(y[index]))
    return np.asarray(x_out), np.asarray(y_out)


def _plot_residue_panel(axis, residues: np.ndarray, profile: np.ndarray) -> None:
    tbd_mask = (residues >= 318) & (residues <= 424)
    tbd_residues = residues[tbd_mask]
    tbd_profile = profile[tbd_mask]
    x_plot, y_plot = _break_at_gaps(tbd_residues, tbd_profile)
    lookup = {int(residue): float(value) for residue, value in zip(residues, profile)}

    axis.axvspan(318, 424, color=PALE_ORANGE, alpha=0.65, linewidth=0, zorder=0)
    axis.plot(x_plot, y_plot, color=ANM, linewidth=1.45, zorder=2, label="ANM profile")
    axis.scatter(
        CONTACT_RESIDUES,
        [lookup[residue] for residue in CONTACT_RESIDUES],
        s=30,
        marker="o",
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=0.9,
        zorder=4,
        label=f"5FQD contacts ({sample_size_label(len(CONTACT_RESIDUES))})",
    )
    axis.scatter(
        ANNOTATED_RESIDUES,
        [lookup[residue] for residue in ANNOTATED_RESIDUES],
        s=22,
        marker="o",
        facecolor=ORANGE,
        edgecolor="white",
        linewidth=0.45,
        zorder=5,
        label=f"UniProt ({sample_size_label(len(ANNOTATED_RESIDUES))})",
    )
    axis.scatter(
        ZINC_RESIDUES,
        [lookup[residue] for residue in ZINC_RESIDUES],
        s=31,
        marker="s",
        facecolor=BLACK,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
        label=f"Zn²⁺ ({sample_size_label(len(ZINC_RESIDUES))})",
    )
    label_offsets = {
        377: (-8, -12),
        378: (-8, 8),
        379: (1, -12),
        380: (2, 8),
        386: (4, -13),
        400: (-2, 8),
        402: (5, -12),
    }
    for residue, (x_offset, y_offset) in label_offsets.items():
        axis.annotate(
            str(residue),
            xy=(residue, lookup[residue]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            color=ORANGE,
            fontsize=8.0,
            fontweight="bold",
            ha="center",
        )
    axis.set_xlim(316, 426)
    axis.set_ylim(0, max(0.62, float(tbd_profile.max()) * 1.16))
    axis.set_xticks([320, 350, 380, 410])
    axis.set_xlabel("Residue (TBD)")
    axis.set_ylabel("ANM square fluctuation\n(max-normalized)")
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles=handles[1:],
        labels=labels[1:],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        borderaxespad=0,
        ncol=3,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.65,
    )
    finish_axis(axis, grid="y")
    axis.tick_params(which="both", top=False, right=False)


def _plot_group_panel(axis, groups: dict[str, dict[str, list[float]]]) -> None:
    group_centres = {"annotated": 0.0, "contact": 1.0, "zinc": 2.0}
    method_offsets = {"ANM": -0.12, "PCA": 0.12}
    method_colors = {"ANM": ANM, "PCA": PCA}
    method_markers = {"ANM": "o", "PCA": "D"}

    for method in ("ANM", "PCA"):
        for group_name in ("annotated", "contact", "zinc"):
            values = np.asarray(groups[method][group_name], dtype=float)
            centre = group_centres[group_name] + method_offsets[method]
            jitter = np.linspace(-0.037, 0.037, len(values))
            axis.scatter(
                centre + jitter,
                values,
                s=18,
                marker=method_markers[method],
                facecolor="white",
                edgecolor=method_colors[method],
                linewidth=0.75,
                zorder=3,
            )
            mean = float(values.mean())
            axis.plot(
                [centre - 0.065, centre + 0.065],
                [mean, mean],
                color=method_colors[method],
                linewidth=2.0,
                solid_capstyle="round",
                zorder=4,
            )
            axis.text(
                centre,
                max(4.0, mean - 8.0),
                f"{mean:.0f}",
                color=method_colors[method],
                fontsize=8.0,
                fontweight="bold",
                ha="center",
                va="top",
            )

    axis.axhline(50, color=LIGHT_GREY, linewidth=0.75, linestyle=":", zorder=0)
    axis.set_xlim(-0.42, 2.42)
    axis.set_ylim(0, 102)
    axis.set_xticks(
        [0, 1, 2],
        labels=[
            f"UniProt ligand\n{sample_size_label(len(ANNOTATED_RESIDUES))}",
            f"5FQD contacts\n{sample_size_label(len(CONTACT_RESIDUES))}",
            f"Zn²⁺ site\n{sample_size_label(len(ZINC_RESIDUES))}",
        ],
    )
    axis.set_ylabel("Mobility percentile")
    axis.text(
        0.5,
        0.10,
        "definitions shown separately",
        transform=axis.transAxes,
        color=DARK_GREY,
        fontsize=8.0,
        ha="center",
        va="bottom",
    )
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=ANM,
            color=ANM,
            label="ANM",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=PCA,
            color=PCA,
            label="PCA",
        ),
    ]
    axis.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        borderaxespad=0,
        ncol=2,
        handlelength=0.9,
    )
    finish_axis(axis, grid="y")
    axis.tick_params(which="both", top=False, right=False)


def _plot_structure_panel(axis) -> None:
    axis.set_axis_off()
    with Image.open(STRUCTURE_INPUT) as source:
        image = source.convert("RGBA")
        alpha_box = image.getchannel("A").getbbox()
        if alpha_box is None:
            raise ValueError("frozen structural raster has an empty alpha channel")
        image = image.crop(alpha_box)
        axis.imshow(np.asarray(image), interpolation="nearest")

    axis.text(
        0.12,
        0.98,
        "drug-binding loop",
        transform=axis.transAxes,
        fontsize=8.5,
        color=ORANGE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axis.text(
        0.12,
        0.08,
        "S-lenalidomide",
        transform=axis.transAxes,
        fontsize=8.5,
        color=AMBER,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    axis.text(
        0.97,
        0.19,
        "Zn²⁺ site",
        transform=axis.transAxes,
        fontsize=8.5,
        color=BLACK,
        fontweight="bold",
        ha="right",
        va="top",
    )


def main() -> None:
    _verify_structure_input()
    residues, anm_profile = _load_anm_profile()
    groups, group_residues = _load_group_percentiles()
    if not np.array_equal(residues, group_residues):
        raise ValueError("ANM and group-percentile inputs use different residue windows")
    apply_publication_style("Fig4")

    # Tight bounding includes the external panel labels and top keys; a
    # 0.13-in canvas allowance keeps the exported media box at about 168 mm.
    figure = plt.figure(figsize=(MAIN_WIDTH_IN - 0.13, 3.35))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.08, 1.08, 1.32),
        left=0.080,
        right=0.995,
        bottom=0.18,
        top=0.86,
        wspace=0.38,
    )
    ax_residue = figure.add_subplot(grid[0, 0])
    ax_group = figure.add_subplot(grid[0, 1])
    ax_structure = figure.add_subplot(grid[0, 2])

    _plot_residue_panel(ax_residue, residues, anm_profile)
    _plot_group_panel(ax_group, groups)
    _plot_structure_panel(ax_structure)
    panel_label(ax_residue, "a", x=-0.18, y=1.08)
    panel_label(ax_group, "b", x=-0.18, y=1.08)
    panel_label(ax_structure, "c", x=-0.08, y=1.02)

    save_figure_set(figure, ROOT, "Fig4")
    plt.close(figure)

    means = {
        method: {group: float(np.mean(values)) for group, values in by_group.items()}
        for method, by_group in groups.items()
    }
    print(
        "Fig4 built (UniProt3 and 5FQD-contact7 kept separate): "
        f"ANM annotated/contact/Zn {means['ANM']['annotated']:.3f}/"
        f"{means['ANM']['contact']:.3f}/{means['ANM']['zinc']:.3f}; "
        f"PCA annotated/contact/Zn {means['PCA']['annotated']:.3f}/"
        f"{means['PCA']['contact']:.3f}/{means['PCA']['zinc']:.3f}; "
        f"frozen raster {STRUCTURE_INPUT.name} composed without redraw"
    )


if __name__ == "__main__":
    main()
