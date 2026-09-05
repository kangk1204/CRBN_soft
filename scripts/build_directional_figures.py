#!/usr/bin/env python3
"""Build directional-mechanics manuscript figures from frozen analysis sources.

The plotting layer is intentionally read-only with respect to scientific data:
it consumes completed CSV/JSON outputs, writes figure source snapshots and
manifests, and renders figures. It does not recompute mechanics, contacts, or
external-data classifications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from figure_style import (  # type: ignore  # noqa: E402
        AMBER,
        BLACK,
        BLUE,
        DARK_GREY,
        GREEN,
        LIGHT_GREY,
        MAGENTA,
        MID_GREY,
        ORANGE,
        PURPLE,
        SKY,
        apply_publication_style,
        finish_axis,
        panel_label,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for packaged use
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

    def apply_publication_style(figure_id: str) -> None:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 8.0,
                "axes.labelsize": 8.0,
                "axes.titlesize": 8.8,
                "xtick.labelsize": 7.2,
                "ytick.labelsize": 7.2,
                "legend.fontsize": 7.0,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )

    def finish_axis(ax, *, grid: str | None = None, zero_line: bool = False) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if grid:
            ax.grid(True, axis="both" if grid == "both" else grid, color=LIGHT_GREY, linewidth=0.45)
            ax.set_axisbelow(True)
        if zero_line:
            ax.axhline(0, color=MID_GREY, linewidth=0.55, zorder=0)

    def panel_label(ax, label: str, *, x: float = -0.14, y: float = 1.06) -> None:
        ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=10, fontweight="bold")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "directional_mechanics"
FIGURE_STEMS = ("Fig3", "Fig4", "Fig5")
WIDTH_IN = 7.09  # 180 mm
MODEL_ORDER = ("isolated", "fixed", "rigid", "flexible")
MODEL_LABELS = {
    "isolated": "Isolated\nCRBN",
    "fixed": "Fixed\nDDB1",
    "rigid": "Rigid-body\nDDB1",
    "flexible": "Flexible\nDDB1",
}
REF_ORDER = ("8CVP", "8D7X", "8D7Y", "6H0F", "7U8F")
REF_COLORS = {
    "8CVP": BLUE,
    "8D7X": SKY,
    "8D7Y": PURPLE,
    "6H0F": ORANGE,
    "7U8F": AMBER,
}
CONTACT_COLORS = {
    "CRBN_DDB1": BLUE,
    "HB_TBD": GREEN,
    "NTD_HB": ORANGE,
    "NTD_TBD": PURPLE,
}


class MissingSource(RuntimeError):
    """Raised when a required completed source table is unavailable."""


@dataclass(frozen=True)
class BuildPaths:
    root: Path
    figure_dir: Path
    vector_dir: Path
    source_dir: Path
    protocol_dir: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="output root containing completed analysis sources",
    )
    return parser.parse_args(argv)


def configure_style(figure_id: str) -> None:
    apply_publication_style(figure_id)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.6,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise MissingSource(f"required source is missing: {path}")
    return path


def require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise MissingSource(f"{source} is missing columns: {', '.join(missing)}")


def read_csv(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    path = require_file(path)
    df = pd.read_csv(path)
    require_columns(df, columns, path)
    return df


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_snapshot(df: pd.DataFrame, paths: BuildPaths, filename: str) -> Path:
    paths.source_dir.mkdir(parents=True, exist_ok=True)
    output = paths.source_dir / filename
    df.to_csv(output, index=False)
    return output


def save_figure(fig: plt.Figure, paths: BuildPaths, stem: str) -> list[Path]:
    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    paths.vector_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        paths.figure_dir / f"{stem}.png",
        paths.vector_dir / f"{stem}.pdf",
        paths.vector_dir / f"{stem}.svg",
    ]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return outputs


def write_manifest(
    paths: BuildPaths,
    stem: str,
    input_paths: Sequence[Path],
    source_paths: Sequence[Path],
    output_paths: Sequence[Path],
    notes: Sequence[str],
) -> Path:
    payload = {
        "figure": stem,
        "builder": "scripts/build_directional_figures.py",
        "inputs": [
            {
                "path": relative_to_root(path, paths.root),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
        ],
        "source_snapshots": [
            {
                "path": relative_to_root(path, paths.root),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        ],
        "outputs": [
            {
                "path": relative_to_root(path, paths.root),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
        "notes": list(notes),
    }
    manifest = paths.source_dir / f"{stem}_input_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def compact_residue_label(row: pd.Series) -> str:
    return f"{int(row['residue'])} {row['contact_class'].replace('_', '-')}"


def ordered_refs(df: pd.DataFrame) -> list[str]:
    present = set(df["pdb"].astype(str))
    ordered = [pdb for pdb in REF_ORDER if pdb in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def primary_mechanics(paths: BuildPaths) -> tuple[pd.DataFrame, Path]:
    source = paths.root / "analysis" / "mechanics" / "models_all.csv"
    df = read_csv(
        source,
        [
            "pdb",
            "cutoff_A",
            "weighting",
            "reference_type",
            "model",
            "S_close",
            "C_close",
            "mean_compliance",
            "best_mode20",
            "best_overlap20",
        ],
    )
    primary = df[(df["cutoff_A"] == 15.0) & (df["weighting"] == "uniform")].copy()
    if primary.empty:
        raise MissingSource("models_all.csv has no 15 A uniform rows")
    return primary, source


def primary_comparisons(paths: BuildPaths) -> tuple[pd.DataFrame, Path]:
    source = paths.root / "analysis" / "mechanics" / "comparisons_all.csv"
    df = read_csv(
        source,
        [
            "pdb",
            "cutoff_A",
            "weighting",
            "reference_type",
            "role",
            "target",
            "effect",
            "rotational_percentile",
            "null_q95",
            "finite_tangent_overlap",
        ],
    )
    primary = df[(df["cutoff_A"] == 15.0) & (df["weighting"] == "uniform")].copy()
    if primary.empty:
        raise MissingSource("comparisons_all.csv has no 15 A uniform rows")
    return primary, source


def load_mode_path(paths: BuildPaths) -> tuple[pd.DataFrame, Path]:
    source = require_file(
        paths.root
        / "analysis"
        / "mode_paths"
        / "8CVP_15A_uniform"
        / "directional_modes_summary.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("alpha_summaries", [])
    if not rows:
        raise MissingSource(f"mode path summary has no alpha_summaries: {source}")
    df = pd.DataFrame(rows)
    require_columns(
        df,
        [
            "interface_alpha",
            "best_mode",
            "best_crbn_directional_overlap",
            "internal_best_mode",
            "internal_best_overlap",
            "tracked_rank",
            "cluster_projection",
            "identity_interpretable",
            "low_overlap_flag",
        ],
        source,
    )
    return df, source


def build_fig3(paths: BuildPaths) -> dict[str, Any]:
    configure_style("Fig3")
    models, models_path = primary_mechanics(paths)
    comparisons, comparisons_path = primary_comparisons(paths)
    mode_path, mode_path_source = load_mode_path(paths)

    source_models = write_snapshot(models, paths, "Fig3_models_primary_15A_uniform.csv")
    source_comparisons = write_snapshot(comparisons, paths, "Fig3_comparisons_primary_15A_uniform.csv")
    source_mode = write_snapshot(mode_path, paths, "Fig3_8CVP_alpha_mode_path.csv")

    fig = plt.figure(figsize=(WIDTH_IN, 6.25), constrained_layout=False)
    grid = GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.32)

    ax = fig.add_subplot(grid[0, 0])
    refs = ordered_refs(models)
    x = np.arange(len(MODEL_ORDER))
    for pdb in refs:
        sub = models[models["pdb"] == pdb]
        values = [
            float(sub.loc[sub["model"] == model, "S_close"].iloc[0])
            if (sub["model"] == model).any()
            else np.nan
            for model in MODEL_ORDER
        ]
        ref_type = str(sub["reference_type"].iloc[0])
        marker = "o" if ref_type == "apo" else "s"
        ax.plot(x, np.log10(values), marker=marker, color=REF_COLORS.get(pdb, DARK_GREY), label=pdb, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel(r"$\log_{10} S_{\mathrm{close}}$")
    ax.set_title("Relative closure compliance")
    finish_axis(ax, grid="y")
    panel_label(ax, "a")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), handlelength=1.0)

    ax = fig.add_subplot(grid[0, 1])
    finite = comparisons[(comparisons["target"] == "finite") & (comparisons["role"].isin(["R_body", "R_internal", "M"]))]
    width = 0.62
    xpos = np.arange(len(refs))
    r_body = []
    r_internal = []
    m_values = []
    for pdb in refs:
        sub = finite[finite["pdb"] == pdb]
        r_body.append(float(sub.loc[sub["role"] == "R_body", "effect"].iloc[0]))
        r_internal.append(float(sub.loc[sub["role"] == "R_internal", "effect"].iloc[0]))
        m_values.append(float(sub.loc[sub["role"] == "M", "effect"].iloc[0]))
    ax.bar(xpos, r_body, width=width, color=BLUE, label="DDB1 body")
    positive_internal_labeled = False
    negative_internal_labeled = False
    for idx, (body, internal) in enumerate(zip(r_body, r_internal)):
        if internal >= 0:
            ax.bar(
                idx,
                internal,
                width=width,
                bottom=body,
                color=GREEN,
                label="DDB1 internal" if not positive_internal_labeled else None,
            )
            positive_internal_labeled = True
        else:
            ax.bar(
                idx,
                abs(internal),
                width=width,
                bottom=body + internal,
                color=GREEN,
                hatch="\\\\\\",
                edgecolor=BLACK,
                linewidth=0.25,
                label="negative internal" if not negative_internal_labeled else None,
            )
            negative_internal_labeled = True
        ax.plot([idx - width / 2, idx + width / 2], [body + internal, body + internal], color=BLACK, linewidth=0.55)
    ax.scatter(xpos, m_values, marker="D", s=22, color=BLACK, zorder=4, label="Flexible - isolated")
    ax.set_xticks(xpos)
    ax.set_xticklabels(refs)
    ax.set_ylabel(r"$\Delta\ln S_{\mathrm{close}}$")
    ymin = min(min(m_values), min(r_body), min(np.asarray(r_body) + np.asarray(r_internal)), 0.0) - 0.06
    ymax = max(max(m_values), max(r_body), max(np.asarray(r_body) + np.asarray(r_internal)), 0.0) + 0.08
    ax.set_ylim(ymin, ymax)
    ax.set_title("DDB1 response components")
    finish_axis(ax, grid="y", zero_line=True)
    panel_label(ax, "b")
    ax.legend(loc="best")

    ax = fig.add_subplot(grid[1, 0])
    tangent = comparisons[
        (comparisons["target"] == "tangent") & (comparisons["role"].isin(["R_body", "R_internal", "M"]))
    ].copy()
    role_order = ["R_body", "R_internal", "M"]
    role_labels = ["Body", "Internal", "Total vs isolated"]
    for idx, role in enumerate(role_order):
        sub = tangent[tangent["role"] == role]
        jitter = np.linspace(-0.17, 0.17, max(len(sub), 1))
        for offset, (_, row) in zip(jitter, sub.iterrows()):
            marker = "o" if row["reference_type"] == "apo" else "s"
            ax.scatter(
                idx + offset,
                row["rotational_percentile"],
                marker=marker,
                color=REF_COLORS.get(str(row["pdb"]), DARK_GREY),
                edgecolor=BLACK,
                linewidth=0.25,
                s=24,
                zorder=3,
            )
    ax.axhline(95, color=ORANGE, linestyle="--", linewidth=0.9, label="95 Percentile reference")
    ax.set_xticks(np.arange(len(role_order)))
    ax.set_xticklabels(role_labels)
    ax.set_ylim(-3, 103)
    ax.set_ylabel("Rotation-null percentile")
    ax.set_title("Rotation-null comparison")
    finish_axis(ax, grid="y")
    panel_label(ax, "c")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(grid[1, 1])
    alpha = mode_path["interface_alpha"].astype(float).to_numpy()
    internal_best = mode_path["internal_best_mode"].astype(float).to_numpy()
    raw_best_column = "raw_best_mode" if "raw_best_mode" in mode_path.columns else "best_mode"
    raw_best = mode_path[raw_best_column].astype(float).to_numpy()
    tracked = mode_path["tracked_rank"].astype(float).to_numpy()
    internal_overlap_column = (
        "best_crbn_internal_overlap" if "best_crbn_internal_overlap" in mode_path.columns else "internal_best_overlap"
    )
    overlap = mode_path[internal_overlap_column].astype(float).to_numpy()
    cluster = mode_path["cluster_projection"].astype(float).to_numpy()
    ax.step(alpha, tracked, where="mid", color=BLUE, label="tracked branch rank")
    ax.step(alpha, internal_best, where="mid", color=ORANGE, linestyle="--", label="best internal rank")
    ax.step(alpha, raw_best, where="mid", color=MAGENTA, linestyle=":", label="best raw rank")
    ax.set_xlabel("Interface strength alpha (α)")
    ax.set_ylabel("Mode rank")
    ax.set_ylim(max(np.nanmax([internal_best, raw_best, tracked]) + 0.6, 6), 0.4)
    finish_axis(ax, grid="both")
    twin = ax.twinx()
    twin.plot(alpha, overlap, color=BLACK, linewidth=1.05, label="Best internal overlap")
    twin.plot(alpha, cluster, color=GREEN, linewidth=1.0, linestyle=":", label="Cluster projection")
    twin.set_ylabel("Directional overlap")
    twin.set_ylim(0, 1.02)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower left")
    ax.set_title("8CVP internal and raw ranks")
    panel_label(ax, "d")

    outputs = save_figure(fig, paths, "Fig3")
    manifest = write_manifest(
        paths,
        "Fig3",
        [models_path, comparisons_path, mode_path_source],
        [source_models, source_comparisons, source_mode],
        outputs,
        [
            "Primary rows are restricted to 15 A uniform mechanics.",
            "Primary mode rank uses CRBN internal projection; raw joint-vector rank is shown separately when available.",
            "Mode branch identities use Hungarian tracking of full joint vectors from the completed mode-path summary.",
        ],
    )
    return {"figure": "Fig3", "outputs": [str(p) for p in outputs], "manifest": str(manifest)}


def load_contact_sources(paths: BuildPaths) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    groups_path = paths.root / "analysis" / "contact_roles" / "8CVP_15A_uniform" / "groups.csv"
    legacy_path = paths.root / "data" / "directional_reference_inputs" / "legacy_robustness.csv"
    groups = read_csv(
        groups_path,
        [
            "residue",
            "contact_class",
            "domain",
            "contact_count",
            "shared_edge_group_ids",
            "flexible_D_g",
            "flexible_derivative_log_C_close",
            "flexible_derivative_log_mean_compliance",
            "delta_R_body_derivative_log_S_close",
            "delta_R_internal_derivative_log_S_close",
            "flexible_rank",
        ],
    )
    legacy = read_csv(
        legacy_path,
        [
            "residue",
            "contact_class",
            "discovery_rank",
            "discovery_top5",
            "stable_apo_model_candidate",
            "also_consistent_in_engineered_references",
            "condition_results",
        ],
    )
    merged = groups.merge(legacy, on=["residue", "contact_class"], how="left", validate="one_to_one")
    return merged, legacy, groups_path, legacy_path


def selected_contact_rows(merged: pd.DataFrame, limit_new: int = 5) -> pd.DataFrame:
    stable = merged[merged["stable_apo_model_candidate"].fillna(False)].copy()
    top_new = merged[~merged["stable_apo_model_candidate"].fillna(False)].copy()
    top_new = top_new.assign(abs_effect=top_new["flexible_D_g"].abs()).sort_values(
        ["abs_effect", "residue"], ascending=[False, True]
    )
    selected = pd.concat([stable, top_new.head(limit_new)], ignore_index=True)
    selected = selected.drop_duplicates(["residue", "contact_class"]).copy()
    selected["class_label"] = np.where(selected["stable_apo_model_candidate"], "prespecified stable", "new high-effect")
    selected["engineered_label"] = np.where(
        selected["also_consistent_in_engineered_references"].fillna(False),
        "engineered-consistent",
        "apo-only or not stable",
    )
    selected["abs_effect"] = selected["flexible_D_g"].abs()
    return selected.sort_values(["class_label", "abs_effect", "residue"], ascending=[False, False, True])


def condition_matrix(selected: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    condition_order = ["8CVP:13", "8CVP:15", "8CVP:18", "8D7X:15", "8D7Y:15", "6H0F:15", "7U8F:15"]
    matrix = np.full((len(selected), len(condition_order)), np.nan)
    for row_idx, (_, row) in enumerate(selected.iterrows()):
        entries = {}
        for token in str(row.get("condition_results", "")).split(";"):
            if "=" in token:
                key, value = token.split("=", 1)
                entries[key.strip()] = value.strip()
        for col_idx, condition in enumerate(condition_order):
            value = entries.get(condition)
            if value == "pass":
                matrix[row_idx, col_idx] = 1
            elif value == "fail":
                matrix[row_idx, col_idx] = 0
            elif value == "absent":
                matrix[row_idx, col_idx] = -1
    return matrix, condition_order


def build_fig4(paths: BuildPaths) -> dict[str, Any]:
    configure_style("Fig4")
    merged, legacy, groups_path, legacy_path = load_contact_sources(paths)
    selected = selected_contact_rows(merged)
    if selected.empty:
        raise MissingSource("no contact candidates were available for Fig4")
    source_all = write_snapshot(merged, paths, "Fig4_8CVP_contact_roles_with_legacy.csv")
    source_selected = write_snapshot(selected, paths, "Fig4_selected_contact_candidates.csv")
    source_legacy = write_snapshot(legacy, paths, "Fig4_legacy_robustness.csv")

    fig = plt.figure(figsize=(WIDTH_IN, 6.45), constrained_layout=False)
    grid = GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    plot_rows = selected.sort_values("flexible_D_g")
    y = np.arange(len(plot_rows))
    colors = [CONTACT_COLORS.get(str(cls), DARK_GREY) for cls in plot_rows["contact_class"]]
    hatches = ["///" if stable else "" for stable in plot_rows["stable_apo_model_candidate"].fillna(False)]
    bars = ax.barh(y, plot_rows["flexible_D_g"], color=colors, edgecolor=BLACK, linewidth=0.25)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    labels = [compact_residue_label(row) for _, row in plot_rows.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Derivative of $\ln S_{\mathrm{close}}$")
    ax.set_title("Candidate contact effects")
    finish_axis(ax, grid="x", zero_line=True)
    panel_label(ax, "a")
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, edgecolor=BLACK, label="CRBN-DDB1"),
            Patch(facecolor=GREEN, edgecolor=BLACK, label="HB-TBD"),
            Patch(facecolor="white", edgecolor=BLACK, hatch="///", label="Prespecified stable"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.18),
        ncol=3,
    )

    ax = fig.add_subplot(grid[0, 1])
    for _, row in selected.iterrows():
        marker = "o" if row["stable_apo_model_candidate"] else "s"
        face = CONTACT_COLORS.get(str(row["contact_class"]), DARK_GREY)
        edge = BLACK if row["also_consistent_in_engineered_references"] else "white"
        ax.scatter(
            row["flexible_derivative_log_C_close"],
            row["flexible_derivative_log_mean_compliance"],
            marker=marker,
            s=34,
            color=face,
            edgecolor=edge,
            linewidth=0.7,
            zorder=3,
        )
    lims = np.array(
        [
            selected["flexible_derivative_log_C_close"].min(),
            selected["flexible_derivative_log_C_close"].max(),
            selected["flexible_derivative_log_mean_compliance"].min(),
            selected["flexible_derivative_log_mean_compliance"].max(),
        ]
    )
    pad = max(0.001, float(np.nanmax(lims) - np.nanmin(lims)) * 0.08)
    lo, hi = float(np.nanmin(lims) - pad), float(np.nanmax(lims) + pad)
    ax.plot([lo, hi], [lo, hi], color=MID_GREY, linewidth=0.8, linestyle=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Derivative of $\ln C_{\mathrm{close}}$")
    ax.set_ylabel(r"Derivative of $\ln \overline{C}$")
    ax.set_title("Compliance decomposition")
    finish_axis(ax, grid="both")
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[1, 0])
    for _, row in selected.iterrows():
        marker = "o" if row["stable_apo_model_candidate"] else "s"
        face = CONTACT_COLORS.get(str(row["contact_class"]), DARK_GREY)
        ax.scatter(
            row["delta_R_body_derivative_log_S_close"],
            row["delta_R_internal_derivative_log_S_close"],
            marker=marker,
            s=36,
            color=face,
            edgecolor=BLACK,
            linewidth=0.35,
            zorder=3,
        )
    ax.set_xlabel(r"Derivative of $R_{\mathrm{body}}$")
    ax.set_ylabel(r"Derivative of $R_{\mathrm{internal}}$")
    ax.set_title("DDB1 role derivatives")
    finish_axis(ax, grid="both", zero_line=True)
    ax.axvline(0, color=MID_GREY, linewidth=0.55, zorder=0)
    panel_label(ax, "c", x=-0.10)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=DARK_GREY, markeredgecolor=BLACK, label="stable"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=DARK_GREY, markeredgecolor=BLACK, label="high-effect"),
        ],
        loc="best",
    )

    ax = fig.add_subplot(grid[1, 1])
    matrix, conditions = condition_matrix(selected)
    cmap = matplotlib.colors.ListedColormap(["#EFEFEF", "#F2B6A0", "#9AD4B3"])
    norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels([compact_residue_label(row) for _, row in selected.iterrows()])
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Robustness rules")
    ax.tick_params(length=0)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            token = "P" if value == 1 else ("F" if value == 0 else "A")
            ax.text(col_idx, row_idx, token, ha="center", va="center", fontsize=6.0, color=BLACK)
    ax.text(
        0.0,
        -0.30,
        "P: same sign/top 20; F: failed gate; A: absent",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=DARK_GREY,
    )
    panel_label(ax, "d")

    outputs = save_figure(fig, paths, "Fig4")
    manifest = write_manifest(
        paths,
        "Fig4",
        [groups_path, legacy_path],
        [source_all, source_selected, source_legacy],
        outputs,
        [
            "Contact effects are read from completed contact_roles outputs for 8CVP 15 A uniform.",
            "P/F/A in the robustness matrix means pass, fail, or absent under the prespecified legacy conditions.",
        ],
    )
    return {"figure": "Fig4", "outputs": [str(p) for p in outputs], "manifest": str(manifest)}


def read_external(paths: BuildPaths) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    external = paths.root / "analysis" / "external"
    source_map = {
        "site": external / "candidate_9sfm_spatial_correspondence.csv",
        "blood": external / "blood_2025_variant_observations.csv",
        "saxs": external / "saxs_guinier_refits.csv",
        "oconnor": external / "oconnor_compound9_mutant_case_comparison.csv",
    }
    data = {
        "site": read_csv(
            source_map["site"],
            [
                "residue",
                "contact_class",
                "discovery_rank",
                "stable_apo_model_candidate",
                "also_consistent_in_engineered_references",
                "min_distance_to_A1CEG_A",
                "same_residue_as_A1CEG_contact",
            ],
        ),
        "blood": read_csv(
            source_map["blood"],
            [
                "variant_id",
                "primary_269_window",
                "binding_endpoint",
                "abundance_or_folding_endpoint",
                "degradation_endpoint",
                "cell_response_endpoint",
                "candidate_overlap",
                "stable_apo_candidate_overlap",
                "functional_endpoint_type",
                "binding_endpoint_type",
                "abundance_endpoint_type",
                "degradation_endpoint_type",
                "evidence_quote",
            ],
        ),
        "saxs": read_csv(
            source_map["saxs"],
            ["accession", "condition", "refit_rg_nm", "refit_rg_nm_stderr", "low_q_qa"],
        ),
        "oconnor": read_csv(
            source_map["oconnor"],
            [
                "variant",
                "primary_269_window",
                "dsf_delta_delta_tm_vs_wt_degC",
                "saxs_delta_rg_vs_wt_angstrom",
                "binding_note",
            ],
        ),
    }
    return data, list(source_map.values())


def overlap_code(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "false", "0"}:
        return 0
    return 1


def table3_function_code(value: Any) -> int:
    text = str(value).lower()
    if "similar to ev" in text or "loss" in text:
        return 4
    if "partial" in text:
        return 3
    if "wild type" in text or "wt" in text:
        return 2
    return 0


def availability_code(value: Any) -> int:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return 0
    if "not_measured" in text or "not measured" in text:
        return 0
    if "not_separated" in text or "not separated" in text or "not separately" in text:
        return 0
    if "not_variant_resolved" in text:
        return 0
    if "qualitative" in text:
        return 1
    return 1


def short_variant_label(value: Any) -> str:
    text = str(value)
    return text.replace("_experiment", " exp").replace(" ", "\n")


def build_fig5(paths: BuildPaths) -> dict[str, Any]:
    configure_style("Fig5")
    data, input_paths = read_external(paths)
    snapshots = [write_snapshot(df, paths, f"Fig5_{name}.csv") for name, df in data.items()]

    fig = plt.figure(figsize=(WIDTH_IN, 6.85), constrained_layout=False)
    grid = GridSpec(2, 2, figure=fig, hspace=0.92, wspace=0.42)

    site = data["site"].copy()
    ax = fig.add_subplot(grid[0, 0])
    stable = site["stable_apo_model_candidate"].fillna(False).astype(bool)
    engineered = site["also_consistent_in_engineered_references"].fillna(False).astype(bool)
    same = site["same_residue_as_A1CEG_contact"].fillna(False).astype(bool)
    colors = np.where(stable, BLUE, np.where(site["discovery_rank"] <= 5, ORANGE, LIGHT_GREY))
    sizes = np.where(engineered, 48, 26)
    ax.scatter(
        site["discovery_rank"],
        site["min_distance_to_A1CEG_A"],
        c=colors,
        s=sizes,
        marker="o",
        edgecolor=np.where(same, BLACK, "white"),
        linewidth=0.35,
        alpha=0.95,
    )
    label_positions = {264: (27, 7), 289: (18, 13), 339: (8, 17)}
    for residue in (264, 289, 339):
        hit = site[(site["residue"] == residue) & (stable)]
        if not hit.empty:
            row = hit.iloc[0]
            ax.annotate(
                str(residue),
                (row["discovery_rank"], row["min_distance_to_A1CEG_A"]),
                xytext=label_positions[residue],
                textcoords="data",
                arrowprops={"arrowstyle": "-", "color": MID_GREY, "linewidth": 0.5},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
                fontsize=6.5,
                color=BLACK,
            )
    ax.axhline(4.5, color=ORANGE, linestyle="--", linewidth=0.9)
    ax.set_xlabel("Rank within contact class")
    ax.set_ylabel("Nearest A1CEG distance (Å)")
    ax.set_title("9SFM ligand-site relation")
    finish_axis(ax, grid="both")
    panel_label(ax, "a")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLACK, label="stable apo"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="white", label="top 5"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=LIGHT_GREY, markeredgecolor="white", label="other"),
        ],
        loc="upper right",
    )

    blood = data["blood"].copy()
    ax = fig.add_subplot(grid[0, 1])
    endpoint_cols = [
        "evidence_quote",
        "binding_endpoint_type",
        "abundance_endpoint_type",
        "degradation_endpoint_type",
        "candidate_overlap",
        "stable_apo_candidate_overlap",
    ]
    matrix = np.array(
        [
            [
                table3_function_code(row[col])
                if col == "evidence_quote"
                else (
                    overlap_code(row[col])
                    if col in {"candidate_overlap", "stable_apo_candidate_overlap"}
                    else availability_code(row[col])
                )
                for col in endpoint_cols
            ]
            for _, row in blood.iterrows()
        ],
        dtype=float,
    )
    cmap = matplotlib.colors.ListedColormap(["#F4F4F4", "#B9D8F0", "#9AD4B3", "#F4D17A", "#D98973"])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(blood)))
    ax.set_yticklabels([short_variant_label(value) for value in blood["variant_id"]])
    ax.set_xticks(np.arange(len(endpoint_cols)))
    ax.set_xticklabels(
        ["Table S3\nclass", "binding\navailable", "abundance\nseparate", "function\nqual.", "candidate\noverlap", "stable\noverlap"],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_title("Blood variant evidence classes")
    ax.tick_params(length=0)
    panel_label(ax, "b")
    ax.legend(
        handles=[
            Patch(facecolor="#9AD4B3", edgecolor=BLACK, label="WT-like"),
            Patch(facecolor="#F4D17A", edgecolor=BLACK, label="partial"),
            Patch(facecolor="#D98973", edgecolor=BLACK, label="EV-like/loss"),
            Patch(facecolor="#B9D8F0", edgecolor=BLACK, label="available/overlap"),
            Patch(facecolor="#F4F4F4", edgecolor=BLACK, label="not available/none"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        ncol=1,
        fontsize=6.1,
    )

    saxs = data["saxs"].copy()
    ax = fig.add_subplot(grid[1, 0])
    order = ["apo", "lenalidomide", "pomalidomide", "iberdomide", "mezigdomide"]
    saxs["_order"] = saxs["condition"].map({value: idx for idx, value in enumerate(order)}).fillna(99)
    saxs = saxs.sort_values(["_order", "condition"])
    x = np.arange(len(saxs))
    ax.errorbar(
        x,
        saxs["refit_rg_nm"],
        yerr=saxs["refit_rg_nm_stderr"],
        marker="o",
        color=BLUE,
        ecolor=DARK_GREY,
        capsize=2.5,
        linewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(saxs["condition"], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"Refit $R_g$ (nm)")
    ax.set_title(r"CRBNmidi SAXS $R_g$")
    finish_axis(ax, grid="y")
    panel_label(ax, "c")
    ax.legend(
        handles=[Line2D([0], [0], marker="o", color=BLUE, label=r"Guinier SE; $qR_g\leq 1.3$")],
        loc="best",
    )

    oconnor = data["oconnor"].copy()
    rows = oconnor[oconnor["variant"] != "WT"].copy()
    subgrid = GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[1, 1], hspace=0.25)
    ax_top = fig.add_subplot(subgrid[0, 0])
    ax_bottom = fig.add_subplot(subgrid[1, 0], sharex=ax_top)
    x = np.arange(len(rows))
    ax_top.bar(x, rows["dsf_delta_delta_tm_vs_wt_degC"], width=0.62, color=PURPLE)
    ax_bottom.bar(x, rows["saxs_delta_rg_vs_wt_angstrom"], width=0.62, color=GREEN)
    ax_top.set_ylabel(r"$\Delta\Delta T_m$ (°C)")
    ax_bottom.set_ylabel(r"$\Delta R_g$ (Å)")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(
        [short_variant_label(value) for value in rows["variant"]],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title("O'Connor compound 9 cases")
    finish_axis(ax_top, grid="y", zero_line=True)
    finish_axis(ax_bottom, grid="y", zero_line=True)
    panel_label(ax_top, "d", x=-0.10)

    outputs = save_figure(fig, paths, "Fig5")
    manifest = write_manifest(
        paths,
        "Fig5",
        input_paths,
        snapshots,
        outputs,
        [
            "External panels are retrospective comparisons only.",
            "Blood endpoint categories are plotted as source-coded classes, not as quantitative effect sizes.",
        ],
    )
    return {"figure": "Fig5", "outputs": [str(p) for p in outputs], "manifest": str(manifest)}


def write_legends(paths: BuildPaths) -> Path:
    text = r"""# Directional-mechanics figure legends

## Fig3

Fig. 3. DDB1 separates complex mode ordering from the CRBN closure response. (a) Relative closure compliance, $\log_{10} S_{\mathrm{close}}$, is shown for five open references under isolated, fixed-DDB1, rigid-body-DDB1, and flexible-DDB1 models. (b) Natural-log differences in relative compliance for the finite closure direction separate the contribution from DDB1 body motion and the additional contribution from DDB1 internal relaxation; hatched internal segments indicate negative values and diamonds show flexible minus isolated CRBN. (c) The tangent-direction response is compared with the prespecified local-rotation null distribution, with the 95th percentile reference shown. (d) For 8CVP, mode ranks across interface strength alpha show the Hungarian-tracked branch that starts from the isolated CRBN lowest internal mode, the best internal CRBN match, and the raw CRBN-vector match when available.

## Fig4

Fig. 4. Contact groups have separable effects on closure compliance, mean compliance, and DDB1-mediated response terms. (a) Frozen stable candidates and additional high-effect 8CVP contact groups are ranked by the finite-contrast derivative of $\ln S_{\mathrm{close}}$. Hatched bars mark groups that met the prespecified stable apo rule. (b) The same groups are decomposed into exact derivatives of log closure compliance and log mean compliance; the diagonal marks equal changes in both terms, so horizontal displacement from the diagonal reflects change in relative closure compliance. Circles mark prespecified stable candidates and squares mark additional high-effect groups. (c) The axes show derivatives of $R_{\mathrm{body}}$ and $R_{\mathrm{internal}}$ for the same contact groups. (d) The robustness matrix reports P, F, or A: P means the condition was observed with the same sign and remained in the top 20 percent under the prespecified rule; F means the gate failed; A means the contact group was absent.

## Fig5

Fig. 5. Public experimental observations provide retrospective context for the contact candidates. (a) All 142 candidate groups are compared with the A1CEG allosteric-ligand site in 9SFM; the x-axis is the rank within each contact class, same-residue site contacts are outlined, and residues 264, 289, and 339 are labelled when present. (b) The 12 Blood 2025 variants are shown by the qualitative Supplementary Table 3 functional class and by endpoint availability or candidate-overlap status; light grey indicates an unavailable separate endpoint or no candidate overlap. (c) Refit SAXS Guinier $R_g$ values compare apo and ligand-bound CRBNmidi conditions; error bars are fitting standard errors from the Guinier fits and the plotted fits use the $qR_g\leq 1.3$ criterion. (d) The O'Connor compound-9 mutant cases are split into separate axes for $\Delta\Delta T_m$ in °C and $\Delta R_g$ in Å.
"""
    paths.source_dir.mkdir(parents=True, exist_ok=True)
    output = paths.source_dir / "LEGENDS.md"
    output.write_text(text, encoding="utf-8")
    return output


def write_contract(paths: BuildPaths) -> Path:
    text = """# Directional-mechanics figure contract

## Scope

This contract covers Fig3, Fig4, and Fig5 generated by `scripts/build_directional_figures.py` for a CRBN directional-mechanics package. The builder uses completed source CSV/JSON files in the selected output root only. It does not recompute frozen mechanics, contact perturbations, null distributions, SAXS refits, or external-data classifications.

## Fig3

Claim: DDB1 changes the ordering of modes in the assembly while the closure-directed CRBN response can be evaluated in a common CRBN coordinate system.

Evidence: `analysis/mechanics/models_all.csv`, `analysis/mechanics/comparisons_all.csv`, and `analysis/mode_paths/8CVP_15A_uniform/directional_modes_summary.json`.

Panel roles: paired relative closure compliance across four DDB1 models; finite response decomposition with negative internal segments distinguished; local-rotation percentile comparison; 8CVP alpha mode branch history.

Limit: branch identity is reported from completed Hungarian tracking and is not reassigned by the closure vector during plotting.

## Fig4

Claim: selected residue-contact groups have distinct effects on relative closure compliance, global compliance, and the DDB1 body/internal response terms.

Evidence: `analysis/contact_roles/8CVP_15A_uniform/groups.csv` and `data/directional_reference_inputs/legacy_robustness.csv`.

Panel roles: candidate ranking; closure versus mean compliance decomposition with a diagonal reference; derivatives of R_body and R_internal; prespecified robustness rules with P/F/A meanings.

Limit: plotted derivatives are spring-model sensitivities and are not converted into mutation or ligand effect sizes.

## Fig5

Claim: public experimental data provide external context but do not by themselves validate a predictive model.

Evidence: 9SFM ligand-site correspondence, Blood 2025 variant observations, SAXS Guinier refits, and O'Connor compound-9 cases under `analysis/external/`.

Panel roles: all-candidate spatial relation to 9SFM A1CEG contacts using within-class rank; Blood qualitative functional class plus endpoint availability; SAXS Rg refits with fitting SE; O'Connor DeltaDeltaTm and DeltaRg on separated axes.

Limit: external panels are retrospective and preserve negative or non-overlapping cases.
"""
    paths.protocol_dir.mkdir(parents=True, exist_ok=True)
    output = paths.protocol_dir / "figure_contract.md"
    output.write_text(text, encoding="utf-8")
    return output


def build(output: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(output)
    paths = BuildPaths(
        root=root,
        figure_dir=root / "manuscript" / "figures",
        vector_dir=root / "manuscript" / "figures" / "vector",
        source_dir=root / "analysis" / "directional_figure_sources",
        protocol_dir=root / "protocol",
    )
    paths.source_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "figures": [build_fig3(paths), build_fig4(paths), build_fig5(paths)],
        "legends": str(write_legends(paths)),
        "contract": str(write_contract(paths)),
    }
    summary_path = paths.source_dir / "figure_build_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results["summary"] = str(summary_path)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build(args.output)
    except MissingSource as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
