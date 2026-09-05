#!/usr/bin/env python3
"""Build CRBN CSBJ strengthening figures from available real analysis outputs.

The builder is intentionally fail-closed at the panel level: it only renders
panels whose required input files are present and schema-valid. Missing upstream
analysis products are reported in figure_readiness.json and never plotted as
zeros or placeholders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from figure_style import (  # type: ignore  # noqa: E402
        AMBER,
        BLACK,
        BLUE,
        CLOSED,
        DARK_GREY,
        GREEN,
        LIGHT_GREY,
        MAGENTA,
        MAIN_WIDTH_IN,
        MID_GREY,
        OPEN,
        ORANGE,
        PALE_BLUE,
        PALE_GREEN,
        PURPLE,
        SKY,
        apply_publication_style,
        finish_axis,
        panel_label,
    )
except ModuleNotFoundError:
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
    PALE_GREEN = "#E6F4EF"
    OPEN = BLUE
    CLOSED = ORANGE
    MAIN_WIDTH_IN = 6.62

    def apply_publication_style(figure_id: str) -> None:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 8.5,
                "axes.labelsize": 8.5,
                "axes.titlesize": 9.0,
                "axes.titleweight": "semibold",
                "axes.linewidth": 0.7,
                "xtick.labelsize": 8.0,
                "ytick.labelsize": 8.0,
                "legend.fontsize": 8.0,
                "legend.frameon": False,
                "savefig.dpi": 300,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
                "svg.hashsalt": f"crbn-{figure_id}-strengthening",
            }
        )

    def finish_axis(ax, *, grid: str | None = None, zero_line: bool = False) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if grid:
            ax.grid(True, axis="both" if grid == "both" else grid, color=LIGHT_GREY, linewidth=0.45, alpha=0.7)
            ax.set_axisbelow(True)
        if zero_line:
            ax.axhline(0, color=MID_GREY, linewidth=0.55, zorder=0)

    def panel_label(ax, label: str, *, x: float = -0.14, y: float = 1.06) -> None:
        ax.text(
            x,
            y,
            f"({label.strip().strip('()')})",
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=BLACK,
            ha="right",
            va="top",
            clip_on=False,
        )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "results" / "strengthening"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "manuscript" / "figures"
DEFAULT_SOURCE_DIR = DEFAULT_INPUT_ROOT / "analysis" / "figure_sources"
DATA = ROOT / "data"

MAIN_FIGURE_STEMS = ("Fig1", "Fig2", "Fig3", "Fig4", "Fig5")
SUPPLEMENT_FIGURE_STEMS = ("FigS1", "FigS2", "FigS3", "FigS4", "FigS5", "FigS6")
FIGURE_STEMS = (*MAIN_FIGURE_STEMS, *SUPPLEMENT_FIGURE_STEMS)
SUPPLEMENT_BUILDER_SCRIPTS = {"FigS1": "build_figS1.py", "FigS2": "build_figS2.py", "FigS3": "build_figS3.py"}
CONTACT_CLASSES = ("CRBN_DDB1", "HB_TBD", "NTD_HB", "NTD_TBD")
CONTACT_COLORS = {
    "CRBN_DDB1": BLUE,
    "HB_TBD": GREEN,
    "NTD_HB": ORANGE,
    "NTD_TBD": PURPLE,
}
STATE_LABELS = {
    "drug-conditioned": "drug-conditioned",
    "genuine-apo": "genuine apo",
    "native-substrate": "native substrate",
}
STATE_MARKERS = {
    "drug-conditioned": "o",
    "genuine-apo": "^",
    "native-substrate": "D",
}
STATE_COLORS = {
    "drug-conditioned": CLOSED,
    "genuine-apo": OPEN,
    "native-substrate": GREEN,
}


class NotReady(RuntimeError):
    """Raised when a figure or panel is missing required upstream evidence."""


@dataclass(frozen=True)
class BuildContext:
    input_root: Path
    output_dir: Path
    source_dir: Path
    require_all: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="exit non-zero unless all five main figures can be rendered",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise NotReady(f"{description} is missing: {repo_rel(path)}")
    return path


def read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    require_file(path, "required CSV")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = [field for field in required if field not in fields]
        if missing:
            raise NotReady(f"{repo_rel(path)} missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise NotReady(f"{repo_rel(path)} has no rows")
    return rows


def read_json(path: Path) -> Any:
    require_file(path, "required JSON")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> Path:
    if not rows:
        raise NotReady(f"no source-data rows to write for {repo_rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for field in row.keys():
                if field not in seen:
                    seen.add(field)
                    fieldnames.append(field)
    else:
        fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def float_value(row: Mapping[str, Any], key: str) -> float:
    value = row[key]
    if value in ("", None):
        raise NotReady(f"missing numeric value for {key}")
    number = float(value)
    if not math.isfinite(number):
        raise NotReady(f"non-finite numeric value for {key}")
    return number


def bool_value(row: Mapping[str, Any], key: str) -> bool:
    value = str(row[key]).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise NotReady(f"invalid boolean value for {key}: {row[key]!r}")


def median(values: Iterable[float]) -> float:
    numbers = sorted(values)
    if not numbers:
        raise NotReady("cannot compute median of an empty sequence")
    mid = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[mid]
    return 0.5 * (numbers[mid - 1] + numbers[mid])


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise NotReady(f"{label} mismatch: expected {expected!r}, observed {actual!r}")


def assert_close(actual: float, expected: float, label: str, *, tol: float = 5e-4) -> None:
    if abs(actual - expected) > tol:
        raise NotReady(f"{label} mismatch: expected {expected:.6g}, observed {actual:.6g}")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = output_dir / "vector"
    vector_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = vector_dir / f"{stem}.pdf"
    svg = vector_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata={"Title": stem, "Creator": "scripts/build_strengthening_figures.py"},
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata={"Title": stem, "Date": None},
    )
    clean_svg(svg)
    plt.close(fig)
    for path in (png, pdf, svg):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"failed to write {repo_rel(path)}")
    return {"png": repo_rel(png), "pdf": repo_rel(pdf), "svg": repo_rel(svg)}


def build_existing_supplement_figure_set(stem: str, ctx: BuildContext) -> dict[str, Any]:
    script_source = require_file(ROOT / "scripts" / SUPPLEMENT_BUILDER_SCRIPTS[stem], f"{stem} builder script")
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    (ctx.output_dir / "vector").mkdir(parents=True, exist_ok=True)
    copied_png = ctx.output_dir / f"{stem}.png"
    copied_pdf = ctx.output_dir / "vector" / f"{stem}.pdf"
    copied_svg = ctx.output_dir / "vector" / f"{stem}.svg"
    copied_csv = ctx.source_dir / f"{stem}_source.csv"
    copied_legend_docx = ctx.source_dir / f"{stem}_source_legend.docx"

    spec = importlib.util.spec_from_file_location(f"crbn_{stem}_builder", script_source)
    if spec is None or spec.loader is None:
        raise NotReady(f"cannot import {repo_rel(script_source)}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix=f"crbn_{stem}_", dir=ctx.source_dir) as temporary:
        temporary_root = Path(temporary)
        module.ROOT = temporary_root
        module.SOURCE_DATA = temporary_root / "figures" / "source_data" / f"{stem}_source_data.csv"
        module.main()
        png_source = require_file(temporary_root / "figures" / f"{stem}.png", f"rebuilt {stem} PNG")
        pdf_source = require_file(temporary_root / "figures" / "vector" / f"{stem}.pdf", f"rebuilt {stem} PDF")
        svg_source = require_file(temporary_root / "figures" / "vector" / f"{stem}.svg", f"rebuilt {stem} SVG")
        csv_source = require_file(module.SOURCE_DATA, f"rebuilt {stem} source data")
        legend_source = temporary_root / "figures" / f"{stem}_legend.docx"
        for source, destination in (
            (png_source, copied_png),
            (pdf_source, copied_pdf),
            (svg_source, copied_svg),
            (csv_source, copied_csv),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if legend_source.is_file():
            shutil.copy2(legend_source, copied_legend_docx)

    input_attrs = ("INPUT", "ENSEMBLE_INPUT", "ANM_INPUT", "ROBUSTNESS_INPUT")
    inputs = [script_source]
    for attr in input_attrs:
        value = getattr(module, attr, None)
        if isinstance(value, Path) and value.is_file():
            inputs.append(value)
    provenance = {
        "figure": stem,
        "rebuilt_at_utc": now_utc(),
        "builder_script": repo_rel(script_source),
        "copied_files": {
            repo_rel(copied_png): {"sha256": sha256_file(copied_png)},
            repo_rel(copied_pdf): {"sha256": sha256_file(copied_pdf)},
            repo_rel(copied_svg): {"sha256": sha256_file(copied_svg)},
            repo_rel(copied_csv): {"sha256": sha256_file(copied_csv)},
        },
        "note": "Supplemental figure regenerated from the existing scientific builder in an isolated temporary output root.",
    }
    provenance_path = ctx.source_dir / f"{stem}_legacy_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "png": repo_rel(copied_png),
        "pdf": repo_rel(copied_pdf),
        "svg": repo_rel(copied_svg),
        "source_data": repo_rel(copied_csv),
        "inputs": [repo_rel(path) for path in inputs],
        "provenance": repo_rel(provenance_path),
    }


def clean_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    cleaned: list[str] = []
    skipping_doctype = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<!DOCTYPE"):
            skipping_doctype = not stripped.endswith(">")
            continue
        if skipping_doctype:
            if stripped.endswith(">"):
                skipping_doctype = False
            continue
        cleaned.append(line.rstrip())
    path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def load_frozen_figure1() -> dict[str, Any]:
    pca_path = require_file(DATA / "crbn_pca.npz", "frozen PCA array")
    anm_path = require_file(DATA / "crbn_anm_modes.npz", "frozen ANM array")
    diff_path = require_file(DATA / "pca_diffvec.npz", "frozen PCA-difference array")
    curation_path = require_file(DATA / "crbn_curation_log.csv", "curation log")
    window_path = require_file(DATA / "window_sensitivity.json", "window sensitivity JSON")

    pca = np.load(pca_path, allow_pickle=False)
    anm = np.load(anm_path, allow_pickle=False)
    diff = np.load(diff_path, allow_pickle=False)
    pc1 = np.asarray(pca["pc1_scores"], dtype=float)
    pc2_raw = np.asarray(pca["pc2_scores"], dtype=float)
    variance = np.asarray(pca["variance_ratio"], dtype=float)
    cumulative_overlap = np.asarray(anm["cum_overlap"], dtype=float)
    rmsip = float(anm["rmsip"])
    labels = [str(value).upper() for value in diff["labels"]]
    open_mask = np.asarray(diff["open_mask"], dtype=bool)
    n_ca = np.asarray(pca["mean"]).reshape(-1, 3).shape[0]
    if pc1.shape != (70,) or pc2_raw.shape != (70,) or open_mask.shape != (70,) or n_ca != 269:
        raise NotReady("frozen Fig. 1 arrays do not contain 70 structures on 269 C-alpha nodes")
    if int(open_mask.sum()) != 5:
        raise NotReady("frozen Fig. 1 open-set mask does not contain five structures")
    rows = read_csv(curation_path, ["pdb", "global_state"])
    state_by_pdb = {row["pdb"].upper(): row["global_state"] for row in rows}
    if set(labels) - set(state_by_pdb):
        raise NotReady("curation log does not cover all frozen PCA labels")
    pc2 = pc2_raw / np.sqrt(n_ca)
    empty_middle = read_json(window_path)["empty_middle"]["a_paper_rule"]
    raw_band = np.asarray(empty_middle["band_15_85_pct"], dtype=float)
    closed_mean = float(pc1[~open_mask].mean())
    open_mean = float(pc1[open_mask].mean())
    normalized = (pc1 - closed_mean) / (open_mean - closed_mean)
    normalized_band = (raw_band - closed_mean) / (open_mean - closed_mean)
    return {
        "pc1": pc1,
        "pc2": pc2,
        "variance": variance,
        "cumulative_overlap": cumulative_overlap,
        "rmsip": rmsip,
        "labels": labels,
        "open_mask": open_mask,
        "states": np.asarray([state_by_pdb[label] for label in labels]),
        "normalized": normalized,
        "normalized_band": normalized_band,
        "input_paths": [pca_path, anm_path, diff_path, curation_path, window_path],
    }


def build_fig1(ctx: BuildContext) -> dict[str, Any]:
    values = load_frozen_figure1()
    source_rows: list[dict[str, Any]] = []
    for idx, label in enumerate(values["labels"]):
        source_rows.append(
            {
                "panel": "a,d",
                "pdb": label,
                "state": values["states"][idx],
                "pc1_rmsd_scaled_A": values["pc1"][idx],
                "pc2_rmsd_scaled_A": values["pc2"][idx],
                "normalized_pc1_closed0_open1": values["normalized"][idx],
                "open_mask": bool(values["open_mask"][idx]),
            }
        )
    for idx, (var, overlap) in enumerate(zip(values["variance"][:10], values["cumulative_overlap"][:10]), start=1):
        source_rows.append(
            {
                "panel": "b,c",
                "pdb": "",
                "state": "",
                "component_or_mode": idx,
                "pca_individual_variance_fraction": float(var),
                "anm_cumulative_directional_overlap": float(overlap),
                "rmsip_global_metric": values["rmsip"] if idx == 1 else "",
            }
        )
    source = write_csv(ctx.source_dir / "Fig1_source.csv", source_rows)

    apply_publication_style("Fig1_strengthening")
    fig = plt.figure(figsize=(MAIN_WIDTH_IN, 5.65), dpi=300)
    grid = GridSpec(2, 2, figure=fig, left=0.10, right=0.99, bottom=0.09, top=0.97, wspace=0.38, hspace=0.46)
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1])
    axc = fig.add_subplot(grid[1, 0])
    axd = fig.add_subplot(grid[1, 1])

    for state in STATE_LABELS:
        mask = values["states"] == state
        axa.scatter(
            values["pc1"][mask],
            values["pc2"][mask],
            s=28 if state == "drug-conditioned" else 36,
            marker=STATE_MARKERS[state],
            color=STATE_COLORS[state],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.75 if state == "drug-conditioned" else 0.98,
            label=STATE_LABELS[state],
        )
    axa.set(
        xlabel=f"PC1 ({values['variance'][0] * 100:.1f}% variance)",
        ylabel=f"PC2 ({values['variance'][1] * 100:.1f}% variance)",
    )
    axa.legend(loc="best", labelspacing=0.25)
    finish_axis(axa)
    panel_label(axa, "a")

    components = np.arange(1, 11)
    axb.bar(components, values["variance"][:10] * 100, color=PURPLE, alpha=0.32, width=0.72)
    axb.plot(components, np.cumsum(values["variance"][:10]) * 100, color=BLACK, marker="o", markersize=3.5)
    axb.set(xlabel="PCA component", ylabel="Coordinate variance (%)", xticks=components)
    finish_axis(axb, grid="y")
    panel_label(axb, "b")

    modes = np.arange(1, 11)
    axc.plot(modes, values["cumulative_overlap"][:10], color=BLUE, marker="o", markersize=3.8)
    axc.set(xlabel="ANM modes included", ylabel="Cumulative directional overlap", xticks=modes, ylim=(0, 1.02))
    finish_axis(axc, grid="y")
    inset = axc.inset_axes([0.58, 0.18, 0.34, 0.34])
    inset.bar([0], [values["rmsip"]], color=AMBER, width=0.58)
    inset.set_ylim(0, 1)
    inset.set_xticks([0], ["RMSIP"])
    inset.set_ylabel("global", fontsize=7.0)
    inset.tick_params(axis="both", labelsize=7.0)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.text(0, values["rmsip"] + 0.03, f"{values['rmsip']:.2f}", ha="center", va="bottom", fontsize=7.0)
    panel_label(axc, "c")

    order = np.argsort(values["normalized"])
    y = np.arange(len(order))
    colors = [OPEN if values["open_mask"][idx] else CLOSED for idx in order]
    axd.scatter(values["normalized"][order], y, c=colors, s=12, alpha=0.8, edgecolor="none")
    band = values["normalized_band"]
    axd.axvspan(float(band[0]), float(band[1]), color=LIGHT_GREY, alpha=0.55, zorder=0)
    axd.set(xlabel="Normalized PC1", ylabel="Frozen conformers", yticks=[])
    axd.text(float(np.mean(band)), len(order) * 0.82, "empty middle", ha="center", va="center", fontsize=8.0, color=DARK_GREY)
    finish_axis(axd, grid="x")
    panel_label(axd, "d")

    outputs = save_figure(fig, ctx.output_dir, "Fig1")
    return {
        "figure": "Fig1",
        "status": "rendered",
        "outputs": outputs,
        "source_data": repo_rel(source),
        "inputs": [repo_rel(path) for path in values["input_paths"]],
        "panel_mapping": {
            "a": "frozen 70-structure PCA state census",
            "b": "PCA variance distribution",
            "c": "ANM cumulative overlap with separate RMSIP inset",
            "d": "normalized open-closed PC1 gap with closed mean set to 0 and open mean set to 1",
        },
    }


def anm_robustness_rows() -> tuple[list[dict[str, Any]], list[Path]]:
    robust_path = require_file(DATA / "anm_robustness.json", "committed ANM robustness JSON")
    payload = read_json(robust_path)
    rows = []
    for pdb, by_cutoff in payload["table"].items():
        for cutoff, record in by_cutoff.items():
            rows.append(
                {
                    "pdb": pdb,
                    "cutoff_A": float(cutoff),
                    "mode1_overlap": float(record["mode1_overlap"]),
                    "best_overlap": float(record["best_overlap"]),
                    "best_mode_rank": int(record.get("best_mode_rank", record.get("best_rank"))),
                    "cum_top10": float(record["cum_top10"]),
                    "state_group": "open" if pdb in payload.get("open_set", []) else "closed_endpoint",
                }
            )
    for pdb, record in payload.get("closed_all_15A", {}).items():
        rows.append(
            {
                "pdb": pdb,
                "cutoff_A": 15.0,
                "mode1_overlap": float(record["mode1_overlap"]),
                "best_overlap": float(record["best_overlap"]),
                "best_mode_rank": int(record["best_mode_rank"]),
                "cum_top10": "",
                "state_group": "closed_all70",
            }
        )
    if not rows:
        raise NotReady("ANM robustness JSON contains no table rows")
    return rows, [robust_path]


def build_fig2(ctx: BuildContext) -> dict[str, Any]:
    rows, inputs = anm_robustness_rows()
    ensemble_dir = ctx.input_root / "analysis" / "ensemble"
    pair_path = ensemble_dir / "open_closed_pair_basis_comparison.csv"
    summary_path = ensemble_dir / "summary.json"
    pair_rows: list[dict[str, str]] = []
    missing_panels: list[str] = []
    if pair_path.is_file():
        pair_rows = read_csv(pair_path, ["basis", "open_pdb", "closed_pdb", "mode1_overlap", "best20_rank", "top3_subspace_projection"])
        inputs.append(pair_path)
        if summary_path.is_file():
            inputs.append(summary_path)
    else:
        missing_panels.append(f"panel c,d need {repo_rel(pair_path)}")

    source_rows = [{**row, "panel": "a,b"} for row in rows]
    for row in pair_rows:
        source_rows.append(
            {
                "panel": "c,d",
                "basis": row["basis"],
                "open_pdb": row["open_pdb"],
                "closed_pdb": row["closed_pdb"],
                "mode1_overlap": row["mode1_overlap"],
                "best20_rank": row["best20_rank"],
                "top3_subspace_projection": row["top3_subspace_projection"],
            }
        )
    source = write_csv(ctx.source_dir / "Fig2_source.csv", source_rows)

    apply_publication_style("Fig2_strengthening")
    has_pairs = bool(pair_rows)
    fig = plt.figure(figsize=(MAIN_WIDTH_IN, 3.35 if not has_pairs else 5.55), dpi=300)
    grid = fig.add_gridspec(2 if has_pairs else 1, 2, left=0.10, right=0.99, bottom=0.13 if not has_pairs else 0.08, top=0.95, wspace=0.36, hspace=0.48)
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1])

    primary = [row for row in rows if abs(row["cutoff_A"] - 15.0) < 1e-9]
    ordered = sorted(primary, key=lambda r: (r["state_group"] != "open", r["best_mode_rank"], -r["mode1_overlap"], r["pdb"]))
    x = np.arange(len(ordered))
    axa.scatter(
        x,
        [row["mode1_overlap"] for row in ordered],
        c=[OPEN if row["state_group"] == "open" else MID_GREY for row in ordered],
        s=[38 if row["state_group"] == "open" else 15 for row in ordered],
        alpha=[0.98 if row["state_group"] == "open" else 0.50 for row in ordered],
    )
    axa.set(xlabel="70 frozen structures ordered by rank/state", ylabel="Mode-1 overlap at 15 A", xticks=[])
    finish_axis(axa, grid="y")
    panel_label(axa, "a")

    open_rows = sorted([row for row in rows if row["state_group"] == "open"], key=lambda r: (r["pdb"], r["cutoff_A"]))
    for pdb in sorted({row["pdb"] for row in open_rows}):
        subset = [row for row in open_rows if row["pdb"] == pdb]
        axb.plot([row["cutoff_A"] for row in subset], [row["mode1_overlap"] for row in subset], marker="o", linewidth=1.0, label=pdb)
    axb.set(xlabel="ANM cutoff (A)", ylabel="Open-reference mode-1 overlap", ylim=(0, 1.0))
    axb.legend(ncol=2, loc="lower left")
    finish_axis(axb, grid="y")
    panel_label(axb, "b")

    panel_mapping = {
        "a": "all 70 frozen structures at primary 15 A cutoff",
        "b": "five open references across cutoff sensitivity",
    }
    if has_pairs:
        axc = fig.add_subplot(grid[1, 0])
        axd = fig.add_subplot(grid[1, 1])
        by_basis: dict[str, list[dict[str, str]]] = {}
        for row in pair_rows:
            by_basis.setdefault(row["basis"], []).append(row)
        labels = []
        values = []
        for basis in sorted(by_basis):
            labels.append(basis.replace("_", " "))
            values.append([float_value(row, "mode1_overlap") for row in by_basis[basis]])
        axc.boxplot(values, tick_labels=labels, patch_artist=True, boxprops={"facecolor": PALE_BLUE, "edgecolor": BLUE}, medianprops={"color": BLACK})
        axc.set(ylabel="Pair-axis mode-1 overlap")
        axc.tick_params(axis="x", rotation=18)
        finish_axis(axc, grid="y")
        panel_label(axc, "c")
        rank1 = []
        for basis in sorted(by_basis):
            basis_rows = by_basis[basis]
            rank1.append(sum(int(float_value(row, "best20_rank")) == 1 for row in basis_rows) / len(basis_rows))
        axd.bar(labels, rank1, color=[BLUE, GREEN][: len(rank1)])
        axd.set(ylabel="Fraction with best rank = 1", ylim=(0, 1.0))
        axd.tick_params(axis="x", rotation=18)
        finish_axis(axd, grid="y")
        panel_label(axd, "d")
        panel_mapping["c"] = "5 x 65 pair-axis mode-1 overlap by fixed versus own-open basis"
        panel_mapping["d"] = "rank-1 recovery fraction by basis"

    outputs = save_figure(fig, ctx.output_dir, "Fig2")
    return {
        "figure": "Fig2",
        "status": "rendered_partial" if missing_panels else "rendered",
        "outputs": outputs,
        "source_data": repo_rel(source),
        "inputs": [repo_rel(path) for path in inputs],
        "panel_mapping": panel_mapping,
        "not_ready": missing_panels,
    }


def build_fig3(ctx: BuildContext) -> dict[str, Any]:
    ddb1_dir = ctx.input_root / "analysis" / "ddb1"
    pilot_dir = ctx.input_root / "analysis" / "ddb1_pilot"
    summary_path = ddb1_dir / "model_summary.csv"
    modes_path = ddb1_dir / "modes.csv"
    json_path = ddb1_dir / "model_summary.json"
    if not summary_path.is_file() and pilot_dir.is_dir():
        raise NotReady(
            f"full DDB1 table is missing: {repo_rel(summary_path)}; "
            f"pilot output at {repo_rel(pilot_dir)} was not used for a main figure"
        )
    rows = read_csv(
        summary_path,
        [
            "pdb",
            "cutoff_A",
            "interface_alpha",
            "model",
            "best_mode",
            "best_crbn_directional_overlap",
            "best_crbn_amplitude",
            "internal_best_overlap",
            "internal_best_mode",
        ],
    )
    inputs = [summary_path]
    for optional in (modes_path, json_path):
        if optional.is_file():
            inputs.append(optional)
    source = write_csv(ctx.source_dir / "Fig3_source.csv", rows)
    primary = [row for row in rows if abs(float_value(row, "cutoff_A") - 15.0) < 1e-9]
    if not primary:
        raise NotReady(f"{repo_rel(summary_path)} has no 15 A primary rows")

    apply_publication_style("Fig3_strengthening")
    fig = plt.figure(figsize=(MAIN_WIDTH_IN, 5.65), dpi=300)
    grid = fig.add_gridspec(2, 2, left=0.10, right=0.99, bottom=0.08, top=0.96, wspace=0.38, hspace=0.46)
    pdbs = sorted({row["pdb"] for row in primary})
    models = ["isolated", "joint", "schur_static", "fixed_partner"]
    model_labels = ["isolated", "joint", "Schur", "fixed"]
    colors = [MID_GREY, BLUE, GREEN, ORANGE]

    axa = fig.add_subplot(grid[0, 0])
    width = 0.18
    x = np.arange(len(pdbs))
    for idx, model in enumerate(models):
        vals = []
        for pdb in pdbs:
            candidates = [row for row in primary if row["pdb"] == pdb and row["model"] == model and abs(float_value(row, "interface_alpha") - (1.0 if model != "isolated" else 0.0)) < 1e-9]
            vals.append(float_value(candidates[0], "best_crbn_directional_overlap") if candidates else np.nan)
        axa.bar(x + (idx - 1.5) * width, vals, width=width, color=colors[idx], label=model_labels[idx])
    axa.set(ylabel="Best CRBN directional overlap", xticks=x, xticklabels=pdbs, ylim=(0, 1.0))
    axa.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.52, 1.16), fontsize=6.8)
    finish_axis(axa, grid="y")
    panel_label(axa, "a")

    axb = fig.add_subplot(grid[0, 1])
    for model, color in zip(("joint", "schur_static", "fixed_partner"), (BLUE, GREEN, ORANGE)):
        vals = []
        for pdb in pdbs:
            candidates = [row for row in primary if row["pdb"] == pdb and row["model"] == model and abs(float_value(row, "interface_alpha") - 1.0) < 1e-9]
            vals.append(float_value(candidates[0], "best_crbn_amplitude") if candidates else np.nan)
        axb.plot(x, vals, marker="o", color=color, label=model.replace("_", " "))
    axb.set(ylabel="CRBN amplitude in selected mode", xticks=x, xticklabels=pdbs, ylim=(0, 1.05))
    axb.legend(loc="best")
    finish_axis(axb, grid="y")
    panel_label(axb, "b")

    axc = fig.add_subplot(grid[1, 0])
    for model, color in zip(("isolated", "joint", "schur_static", "fixed_partner"), (MID_GREY, BLUE, GREEN, ORANGE)):
        vals = []
        for pdb in pdbs:
            target_alpha = 0.0 if model == "isolated" else 1.0
            candidates = [row for row in primary if row["pdb"] == pdb and row["model"] == model and abs(float_value(row, "interface_alpha") - target_alpha) < 1e-9]
            vals.append(float_value(candidates[0], "internal_best_overlap") if candidates else np.nan)
        axc.plot(x, vals, marker="o", color=color, label=model.replace("_", " "))
    axc.set(ylabel="Rigid-removed internal overlap", xticks=x, xticklabels=pdbs, ylim=(0, 1.0))
    axc.legend(loc="best")
    finish_axis(axc, grid="y")
    panel_label(axc, "c")

    axd = fig.add_subplot(grid[1, 1])
    ranks = []
    labels = []
    for model in ("isolated", "joint", "schur_static", "fixed_partner"):
        vals = []
        for row in primary:
            alpha_ok = abs(float_value(row, "interface_alpha") - (0.0 if model == "isolated" else 1.0)) < 1e-9
            if row["model"] == model and alpha_ok:
                vals.append(float_value(row, "best_mode"))
        if vals:
            labels.append(model.replace("_", " "))
            ranks.append(vals)
    axd.boxplot(ranks, tick_labels=labels, patch_artist=True, boxprops={"facecolor": PALE_GREEN, "edgecolor": GREEN}, medianprops={"color": BLACK})
    axd.set(ylabel="Best-mode rank")
    axd.tick_params(axis="x", rotation=18)
    finish_axis(axd, grid="y")
    panel_label(axd, "d")

    outputs = save_figure(fig, ctx.output_dir, "Fig3")
    return {
        "figure": "Fig3",
        "status": "rendered",
        "outputs": outputs,
        "source_data": repo_rel(source),
        "inputs": [repo_rel(path) for path in inputs],
        "panel_mapping": {
            "a": "primary 15 A CRBN directional overlap by DDB1 model definition",
            "b": "CRBN amplitude in joint/interface-aware modes",
            "c": "joint CRBN-internal and other rigid-removed overlaps kept distinct from amplitude",
            "d": "best-mode rank distribution by model definition",
        },
    }


def parse_condition_results(text: str) -> tuple[int, int]:
    if not text:
        return 0, 0
    observed = 0
    passed = 0
    for item in text.split(";"):
        if not item:
            continue
        observed += 1
        if item.endswith("=pass"):
            passed += 1
    return passed, observed


def load_8cvp_trace(residue_effects_path: Path | None = None) -> list[dict[str, Any]]:
    ensemble_path = require_file(DATA / "crbn_ensemble.ens.npz", "frozen ensemble coordinates")
    window_path = require_file(DATA / "crbn_residue_window.csv", "CRBN residue window")
    with np.load(ensemble_path, allow_pickle=False) as ensemble:
        labels = [str(label).upper() for label in ensemble["_labels"]]
        coords = np.asarray(ensemble["_confs"], dtype=float)
    if "8CVP" not in labels:
        raise NotReady("8CVP is absent from frozen ensemble labels")
    with window_path.open(encoding="utf-8", newline="") as handle:
        residues = [int(row["author_resnum"]) for row in csv.DictReader(handle)]
    if coords.shape[1:] != (len(residues), 3):
        raise NotReady("8CVP coordinate array does not match the residue window")
    domain_by_residue: dict[int, str] = {}
    if residue_effects_path and residue_effects_path.is_file():
        for row in read_csv(residue_effects_path, ["resnum", "domain"]):
            domain_by_residue[int(row["resnum"])] = row["domain"]
    xyz = coords[labels.index("8CVP")]
    rows = []
    for residue, point in zip(residues, xyz):
        rows.append(
            {
                "residue": residue,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "domain": domain_by_residue.get(
                    residue,
                    "NTD" if residue <= 186 else "HB" if residue <= 317 else "TBD",
                ),
            }
        )
    return rows


def build_fig4(ctx: BuildContext) -> dict[str, Any]:
    contacts = ctx.input_root / "analysis" / "contacts"
    controls = ctx.input_root / "analysis" / "controls"
    discovery = contacts / "discovery_top5.csv"
    robustness = contacts / "candidate_robustness.csv"
    residue_effects_path = controls / "residue_effects.csv"
    summary_files = sorted(contacts.glob("*_*A/summary.json"))
    if not summary_files:
        raise NotReady(f"no contact condition summaries found under {repo_rel(contacts)}")
    top_rows = read_csv(
        discovery,
        [
            "residue",
            "contact_class",
            "discovery_D_g",
            "discovery_rank",
            "discovery_top5",
            "all_required_conditions_observed",
            "stable_apo_model_candidate",
            "condition_results",
        ],
    )
    robustness_rows = read_csv(
        robustness,
        [
            "residue",
            "contact_class",
            "discovery_D_g",
            "discovery_rank",
            "discovery_top5",
            "stable_apo_model_candidate",
            "also_consistent_in_engineered_references",
            "condition_results",
        ],
    )
    inputs = [discovery, robustness, *summary_files]
    if residue_effects_path.is_file():
        inputs.append(residue_effects_path)

    condition_order = ["8CVP:13", "8CVP:15", "8CVP:18", "8D7X:15", "8D7Y:15", "6H0F:15", "7U8F:15"]
    condition_labels = ["8CVP\n13A", "8CVP\n15A", "8CVP\n18A", "8D7X\n15A", "8D7Y\n15A", "6H0F\n15A", "7U8F\n15A"]

    top5_ids = {(row["contact_class"], row["residue"]) for row in top_rows if row.get("discovery_top5") == "True"}
    stable_ids = {
        (row["contact_class"], row["residue"])
        for row in robustness_rows
        if row.get("stable_apo_model_candidate") == "True"
    }
    selected_ids = top5_ids | stable_ids
    selected_rows = [row for row in robustness_rows if (row["contact_class"], row["residue"]) in selected_ids]
    selected_rows.sort(key=lambda row: (CONTACT_CLASSES.index(row["contact_class"]), int(row["discovery_rank"])))

    source_rows = []
    for row in robustness_rows:
        passed, observed = parse_condition_results(row.get("condition_results", ""))
        source_rows.append(
            {
                **row,
                "panel": "a,b,c",
                "conditions_passed": passed,
                "conditions_observed": observed,
                "selected_for_heatmap": int((row["contact_class"], row["residue"]) in selected_ids),
                "selected_for_stable_map": int(row.get("stable_apo_model_candidate") == "True"),
            }
        )
    summaries = []
    for path in summary_files:
        payload = read_json(path)
        summaries.append(
            {
                "panel": "d",
                "condition": path.parent.name,
                "pdb": payload["pdb"],
                "cutoff_A": payload["cutoff_A"],
                "S_close": payload["S_close"],
                "n_candidate_groups": payload["n_candidate_groups"],
                "n_unique_perturbed_edges": payload["n_unique_perturbed_edges"],
                "minimum_internal_eigenvalue": payload["minimum_internal_eigenvalue"],
                "target_internal_norm": payload["target_internal_norm"],
            }
        )
    summaries.sort(key=lambda row: condition_order.index(f"{row['pdb']}:{int(float(row['cutoff_A']))}"))

    trace_rows = load_8cvp_trace(residue_effects_path if residue_effects_path.is_file() else None)
    stable_residues = {int(residue) for _, residue in stable_ids}
    structural_rows = [
        {
            "panel": "c",
            **row,
            "stable_contact_candidate": int(int(row["residue"]) in stable_residues),
        }
        for row in trace_rows
    ]
    source = write_csv(ctx.source_dir / "Fig4_source.csv", [*source_rows, *summaries, *structural_rows])

    apply_publication_style("Fig4_strengthening")
    fig = plt.figure(figsize=(MAIN_WIDTH_IN, 6.55), dpi=300)
    grid = fig.add_gridspec(2, 2, left=0.14, right=0.99, bottom=0.09, top=0.96, wspace=0.36, hspace=0.42)
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1])
    axc = fig.add_subplot(grid[1, 0], projection="3d")
    axd = fig.add_subplot(grid[1, 1])

    y_labels = []
    x_vals = []
    colors = []
    for cls in CONTACT_CLASSES:
        cls_rows = sorted([row for row in top_rows if row["contact_class"] == cls], key=lambda r: int(r["discovery_rank"]))[:5]
        for row in cls_rows:
            y_labels.append(f"{row['residue']} {cls.replace('_', '-')}")
            x_vals.append(float_value(row, "discovery_D_g"))
            colors.append(CONTACT_COLORS.get(cls, MID_GREY))
    y = np.arange(len(y_labels))
    axa.barh(y, x_vals, color=colors)
    axa.axvline(0, color=BLACK, linewidth=0.6)
    axa.set_yticks(y, y_labels, fontsize=6.5)
    axa.set_xlabel("Signed D_g")
    axa.set_title("Top 5 per class", fontsize=8.8, pad=5)
    axa.invert_yaxis()
    finish_axis(axa, grid="x")
    panel_label(axa, "a", x=-0.15)

    status_to_value = {"absent": -1.0, "fail": 0.0, "pass": 1.0}
    heat = np.full((len(selected_rows), len(condition_order)), np.nan)
    for i, row in enumerate(selected_rows):
        results = {}
        for item in row.get("condition_results", "").split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                results[key] = value
        for j, condition in enumerate(condition_order):
            heat[i, j] = status_to_value.get(results.get(condition, "absent"), np.nan)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    cmap = ListedColormap(["#E6E6E6", "#D55E00", "#0072B2"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    axb.imshow(heat, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    axb.set_xticks(np.arange(len(condition_labels)), condition_labels, fontsize=6.5)
    axb.set_yticks(
        np.arange(len(selected_rows)),
        [f"{row['residue']} {row['contact_class'].replace('_', '-')}" for row in selected_rows],
        fontsize=6.0,
    )
    axb.tick_params(length=0)
    axb.set_title(f"Top10 + stable candidates ({len(selected_rows)} candidates)", fontsize=8.8, pad=5)
    axb.legend(
        handles=[Patch(facecolor="#0072B2", label="pass"), Patch(facecolor="#D55E00", label="fail"), Patch(facecolor="#E6E6E6", label="absent")],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.18),
        ncols=3,
        fontsize=6.5,
    )
    panel_label(axb, "b", x=-0.16, y=1.04)

    by_domain = {domain: [row for row in trace_rows if row["domain"] == domain] for domain in ("NTD", "HB", "TBD")}
    domain_colors = {"NTD": BLUE, "HB": GREEN, "TBD": ORANGE}
    for domain, domain_rows in by_domain.items():
        if not domain_rows:
            continue
        axc.plot(
            [row["x"] for row in domain_rows],
            [row["y"] for row in domain_rows],
            [row["z"] for row in domain_rows],
            color=domain_colors[domain],
            linewidth=1.4,
            alpha=0.88,
            label=domain,
        )
    stable_points = [row for row in trace_rows if int(row["residue"]) in stable_residues]
    if stable_points:
        axc.scatter(
            [row["x"] for row in stable_points],
            [row["y"] for row in stable_points],
            [row["z"] for row in stable_points],
            color=BLACK,
            s=24,
            depthshade=False,
            label="stable candidates",
        )
        label_offsets = {
            186: (28, -32, -10),
            221: (-14, 14, 10),
            222: (18, 16, 10),
            262: (-48, -30, 22),
            264: (30, 18, -22),
            289: (25, -22, -18),
            339: (-58, 58, 28),
            422: (-26, -12, -18),
        }
        for row in stable_points:
            dx, dy, dz = label_offsets.get(int(row["residue"]), (7, -6, 3))
            lx, ly, lz = row["x"] + dx, row["y"] + dy, row["z"] + dz
            axc.plot([row["x"], lx], [row["y"], ly], [row["z"], lz], color=MID_GREY, linewidth=0.45)
            axc.text(lx, ly, lz, str(row["residue"]), fontsize=6.5, color=BLACK, bbox={"facecolor": "white", "alpha": 0.70, "edgecolor": "none", "pad": 0.35})
    axc.view_init(elev=16, azim=-60)
    axc.set_axis_off()
    axc.legend(loc="upper left", bbox_to_anchor=(-0.03, 1.03), ncols=2, fontsize=6.8)
    axc.text2D(-0.08, 1.03, "(c)", transform=axc.transAxes, fontsize=10.5, fontweight="bold", color=BLACK)

    x = np.arange(len(summaries))
    axd.scatter(x, [float(row["S_close"]) for row in summaries], color=BLUE, s=42, edgecolor="white", linewidth=0.45)
    axd.plot(x, [float(row["S_close"]) for row in summaries], color=PALE_BLUE, linewidth=1.0, zorder=0)
    axd.set_xticks(x, [row["condition"].replace("_", "\n") for row in summaries], fontsize=6.4)
    axd.set_ylabel("Baseline specificity S_close")
    axd.set_ylim(bottom=0)
    finish_axis(axd, grid="y")
    panel_label(axd, "d", x=-0.10)

    outputs = save_figure(fig, ctx.output_dir, "Fig4")
    return {
        "figure": "Fig4",
        "status": "rendered",
        "outputs": outputs,
        "source_data": repo_rel(source),
        "inputs": [repo_rel(path) for path in inputs],
        "panel_mapping": {
            "a": "top five signed D_g contact-perturbation candidates for each discovered class",
            "b": "pass, fail, or absent status for the union of top-ten and stable apo-model candidates across five apo and two engineered conditions",
            "c": "observed 8CVP C-alpha trace with stable contact-candidate residue locations",
            "d": "condition-level baseline specificity S_close across the seven contact runs",
        },
        "candidate_counts": {
            "discovery_groups": len(robustness_rows),
            "top5_union": len(top5_ids),
            "stable_apo_candidates": len(stable_ids),
            "heatmap_union": len(selected_rows),
        },
    }


def build_fig5(ctx: BuildContext) -> dict[str, Any]:
    external = ctx.input_root / "analysis" / "external"
    guinier_path = external / "saxs_guinier_refits.csv"
    curve_path = external / "saxs_plot_arrays.csv"
    compound_path = external / "oconnor_compound9_mutant_case_comparison.csv"
    measurement_path = external / "oconnor_pdf_quantitative_measurements.csv"
    variant_inventory_path = external / "oconnor_variant_inventory.csv"
    summary_path = external / "external_strengthening_summary.json"
    guinier = read_csv(
        guinier_path,
        [
            "accession",
            "condition",
            "published_rg_nm_context_only",
            "refit_rg_nm",
            "refit_rg_nm_stderr",
            "n_points",
            "qrg_max",
            "low_q_qa",
            "q_unit",
        ],
    )
    curves = read_csv(
        curve_path,
        [
            "accession",
            "condition",
            "point_index",
            "q_inverse_angstrom",
            "intensity",
            "error",
            "ln_intensity",
            "guinier_fit_ln_intensity",
            "in_primary_guinier_fit",
            "qrg_primary",
            "kratky_x_qrg",
            "kratky_y",
        ],
    )
    compound_rows = read_csv(
        compound_path,
        [
            "variant",
            "residues",
            "primary_269_window",
            "dsf_delta_delta_tm_vs_wt_degC",
            "saxs_delta_rg_vs_wt_angstrom",
            "comparison_scope",
        ],
    )
    inputs = [guinier_path, curve_path]
    for optional in (compound_path, measurement_path, variant_inventory_path, summary_path):
        if optional.is_file():
            inputs.append(optional)
    source_rows = [{**row, "panel": "a"} for row in guinier]
    source_rows.extend({**row, "panel": "b,c"} for row in curves)
    source_rows.extend({**row, "panel": "d"} for row in compound_rows)
    source = write_csv(ctx.source_dir / "Fig5_source.csv", source_rows)

    apply_publication_style("Fig5_strengthening")
    has_compound = bool(compound_rows)
    fig = plt.figure(figsize=(MAIN_WIDTH_IN, 5.35 if has_compound else 3.35), dpi=300)
    grid = fig.add_gridspec(2 if has_compound else 1, 2, left=0.10, right=0.99, bottom=0.08 if has_compound else 0.12, top=0.95, wspace=0.36, hspace=0.50)
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1])

    labels = [row["condition"] for row in guinier]
    accessions = [row["accession"] for row in guinier]
    saxs_colors = dict(zip(accessions, [BLUE, ORANGE, GREEN, PURPLE, MAGENTA]))
    x = np.arange(len(labels))
    refit = np.array([float_value(row, "refit_rg_nm") for row in guinier])
    err = np.array([float_value(row, "refit_rg_nm_stderr") for row in guinier])
    published = np.array([float_value(row, "published_rg_nm_context_only") for row in guinier])
    axa.errorbar(x - 0.08, refit, yerr=err, fmt="o", color=BLUE, capsize=2.0, label="Guinier refit")
    axa.scatter(x + 0.08, published, marker="s", color=AMBER, label="published context")
    axa.set(ylabel="Rg (nm)", xticks=x, xticklabels=labels)
    axa.tick_params(axis="x", rotation=22)
    axa.legend(loc="best")
    finish_axis(axa, grid="y")
    panel_label(axa, "a")

    for accession in accessions:
        subset = sorted(
            [row for row in curves if row["accession"] == accession and bool_value(row, "in_primary_guinier_fit")],
            key=lambda row: int(float_value(row, "point_index")),
        )
        if not subset:
            continue
        q2 = np.array([float_value(row, "q_inverse_angstrom") ** 2 for row in subset])
        ln_i = np.array([float_value(row, "ln_intensity") for row in subset])
        ln_fit = np.array([float_value(row, "guinier_fit_ln_intensity") for row in subset])
        axb.plot(q2, ln_i, color=saxs_colors[accession], linewidth=0.55, alpha=0.62)
        axb.plot(q2, ln_fit, color=saxs_colors[accession], linewidth=1.35, alpha=0.95, label=accession)
    axb.set(xlabel="q² (Å⁻²)", ylabel="ln I(q), low-q Guinier")
    axb.legend(ncols=1, loc="best", fontsize=6.5, title="weighted fits", title_fontsize=6.8)
    finish_axis(axb, grid="both")
    panel_label(axb, "b")

    panel_mapping = {
        "a": "SAXS Rg refit compared with published context values",
        "b": "primary low-q Guinier ln I versus q-squared curves with weighted fitted lines",
    }
    if has_compound:
        axc = fig.add_subplot(grid[1, 0])
        axd = fig.add_subplot(grid[1, 1])
        for accession in accessions:
            subset = sorted(
                [
                    row for row in curves
                    if row["accession"] == accession and row.get("kratky_y") not in ("", None) and float_value(row, "kratky_x_qrg") <= 4.0
                ],
                key=lambda row: int(float_value(row, "point_index")),
            )
            if not subset:
                continue
            axc.plot(
                [float_value(row, "kratky_x_qrg") for row in subset],
                [float_value(row, "kratky_y") for row in subset],
                color=saxs_colors[accession],
                linewidth=1.0,
                alpha=0.9,
                label=accession,
            )
        axc.set(xlabel="qRg", ylabel="Dimensionless Kratky\n(qRg)² I(q)/I(0)", xlim=(0, 4.0))
        axc.legend(ncols=1, loc="best", fontsize=6.5)
        finish_axis(axc, grid="both")
        panel_label(axc, "c")

        mutants = [row for row in compound_rows if row["variant"] != "WT"]
        if not mutants:
            raise NotReady(f"{repo_rel(compound_path)} has no mutant rows for panel d")
        window_colors = {"inside": GREEN, "outside": ORANGE, "mixed": PURPLE}
        axd.axhline(0, color=LIGHT_GREY, linewidth=0.7, zorder=0)
        axd.axvline(0, color=LIGHT_GREY, linewidth=0.7, zorder=0)
        axd.scatter(
            [float_value(row, "dsf_delta_delta_tm_vs_wt_degC") for row in mutants],
            [float_value(row, "saxs_delta_rg_vs_wt_angstrom") for row in mutants],
            c=[window_colors.get(row["primary_269_window"], MID_GREY) for row in mutants],
            s=48,
            edgecolor="white",
            linewidth=0.45,
        )
        label_offsets = {
            "H378N": (4, 3),
            "H378A": (-35, 5),
            "Q100A": (-34, 3),
            "L60A": (4, 3),
            "L60A H378A": (4, -8),
        }
        for row in mutants:
            axd.annotate(
                row["variant"],
                (
                    float_value(row, "dsf_delta_delta_tm_vs_wt_degC"),
                    float_value(row, "saxs_delta_rg_vs_wt_angstrom"),
                ),
                xytext=label_offsets.get(row["variant"], (3, 2)),
                textcoords="offset points",
                fontsize=6.8,
            )
        xvals = [float_value(row, "dsf_delta_delta_tm_vs_wt_degC") for row in mutants]
        yvals = [float_value(row, "saxs_delta_rg_vs_wt_angstrom") for row in mutants]
        axd.set(
            xlabel="Compound 9 ΔΔTm vs WT (°C)",
            ylabel="Compound 9 ΔRg vs WT (A)",
            xlim=(min(xvals) - 0.55, max(xvals) + 0.55),
            ylim=(min(yvals) - 0.25, max(yvals) + 0.25),
        )
        finish_axis(axd, grid="both")
        panel_label(axd, "d")
        panel_mapping["c"] = "dimensionless Kratky profiles from downloaded SASBDB arrays, displayed only over qRg 0-4"
        panel_mapping["d"] = "O'Connor Compound 9 mutant DSF and SAXS case comparison"

    outputs = save_figure(fig, ctx.output_dir, "Fig5")
    return {
        "figure": "Fig5",
        "status": "rendered",
        "outputs": outputs,
        "source_data": repo_rel(source),
        "inputs": [repo_rel(path) for path in inputs],
        "panel_mapping": panel_mapping,
    }

def build_legacy_supplement(ctx: BuildContext, stem: str) -> dict[str, Any]:
    outputs = build_existing_supplement_figure_set(stem, ctx)
    return {
        "figure": stem,
        "status": "rendered",
        "outputs": {"png": outputs["png"], "pdf": outputs["pdf"], "svg": outputs["svg"]},
        "source_data": outputs["source_data"],
        "inputs": [*outputs["inputs"], outputs["provenance"]],
        "panel_mapping": {"source": f"{stem} rebuilt from the existing scientific supplemental builder"},
        "provenance": outputs["provenance"],
    }


def build_figs4(ctx: BuildContext) -> dict[str, Any]:
    controls = ctx.input_root / "analysis" / "controls"
    control_path = controls / "control_panel_comparison.csv"
    tangent_path = controls / "finite_chord_tangent.csv"
    endpoint_path = controls / "endpoint_scores.csv"
    summary_path = controls / "strengthen_controls_summary.json"
    control_rows = read_csv(
        control_path,
        ["name", "source", "motion_class", "mode1_overlap", "best20_overlap", "best20_rank", "top3_projection"],
    )
    tangent = read_csv(
        tangent_path,
        [
            "resnum",
            "domain",
            "axis_distance_A",
            "observed_finite_chord_displacement_A",
            "local_rotation_tangent_A",
            "observed_chord_vs_tangent_cosine",
        ],
    )
    endpoints = read_csv(endpoint_path, ["reference_role", "mode1_overlap", "top3_projection", "best20_rank"])
    inputs = [control_path, tangent_path, endpoint_path]
    if summary_path.is_file():
        inputs.append(summary_path)

    pair_rows = [row for row in endpoints if row["reference_role"] == "committed_8CVP_basis_pair_axis"]
    if len(pair_rows) != 325:
        raise NotReady(f"expected 325 fixed-8CVP pair-axis rows, observed {len(pair_rows)}")
    control_plot = [
        row for row in control_rows
        if row["source"] in {"existing_external_primary_control_panel", "crbn_primary_pair", "new_pair_level_crbn_primary_pair"}
    ]
    source = write_csv(
        ctx.source_dir / "FigS4_source.csv",
        [{**row, "panel": "a,b"} for row in control_plot]
        + [{**row, "panel": "c"} for row in tangent]
        + [{**row, "panel": "d"} for row in pair_rows],
    )

    apply_publication_style("FigS4_strengthening")
    fig = plt.figure(figsize=(MAIN_WIDTH_IN, 6.15), dpi=300)
    grid = fig.add_gridspec(2, 2, left=0.10, right=0.99, bottom=0.08, top=0.96, wspace=0.38, hspace=0.48)
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1])
    axc = fig.add_subplot(grid[1, 0])
    axd = fig.add_subplot(grid[1, 1])

    external = [row for row in control_plot if row["source"] == "existing_external_primary_control_panel"]
    observed = [row for row in control_plot if row["source"] in {"crbn_primary_pair", "new_pair_level_crbn_primary_pair"}]
    axa.scatter(
        [float_value(row, "mode1_overlap") for row in external],
        [float_value(row, "top3_projection") for row in external],
        color=MID_GREY,
        alpha=0.65,
        s=24,
        label="external controls",
    )
    if observed:
        axa.scatter(
            [float_value(row, "mode1_overlap") for row in observed],
            [float_value(row, "top3_projection") for row in observed],
            color=BLUE,
            s=58,
            edgecolor="white",
            linewidth=0.5,
            label="CRBN primary pair",
        )
    axa.set(xlabel="Mode-1 overlap", ylabel="Top-3 projection", xlim=(0, 1.0), ylim=(0, 1.0))
    axa.legend(loc="lower right")
    finish_axis(axa, grid="both")
    panel_label(axa, "a")

    rank_counts: dict[int, int] = {}
    for row in external:
        rank = int(float_value(row, "best20_rank"))
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    ranks = sorted(rank_counts)
    axb.bar([str(rank) for rank in ranks], [rank_counts[rank] for rank in ranks], color=PALE_BLUE, edgecolor=BLUE)
    if observed:
        axb.axvline(str(int(float_value(observed[0], "best20_rank"))), color=ORANGE, linewidth=1.4, label="CRBN rank")
        axb.legend(loc="best")
    axb.set(xlabel="Best mode rank among top 20", ylabel="External controls")
    finish_axis(axb, grid="y")
    panel_label(axb, "b")

    domain_colors = {"NTD": BLUE, "HB": GREEN, "TBD": ORANGE}
    axc.scatter(
        [float_value(row, "axis_distance_A") for row in tangent],
        [float_value(row, "observed_finite_chord_displacement_A") for row in tangent],
        c=[domain_colors.get(row["domain"], MID_GREY) for row in tangent],
        s=18,
        alpha=0.78,
        edgecolor="none",
    )
    axc.set(xlabel="Distance to screw axis (A)", ylabel="Finite chord displacement (A)")
    finish_axis(axc, grid="both")
    panel_label(axc, "c")

    axd.hist([float_value(row, "mode1_overlap") for row in pair_rows], bins=18, color=PALE_GREEN, edgecolor=GREEN)
    axd.set(xlabel="Fixed 8CVP mode-1 overlap", ylabel="Pairs")
    finish_axis(axd, grid="y")
    panel_label(axd, "d")

    outputs = save_figure(fig, ctx.output_dir, "FigS4")
    return {
        "figure": "FigS4",
        "status": "rendered",
        "outputs": outputs,
        "source_data": repo_rel(source),
        "inputs": [repo_rel(path) for path in inputs],
        "panel_mapping": {
            "a": "CRBN primary pair plotted against external primary controls",
            "b": "external-control best-rank distribution with CRBN rank marker",
            "c": "finite chord displacement versus screw-axis distance",
            "d": "325 fixed-8CVP-basis pair-axis mode-1 overlap distribution",
        },
    }


def build_figs5(ctx: BuildContext) -> dict[str, Any]:
    controls = ctx.input_root / "analysis" / "controls"
    set_path = controls / "residue_set_effects.csv"
    residue_path = controls / "residue_effects.csv"
    summary_path = controls / "strengthen_controls_summary.json"
    set_rows = read_csv(
        set_path,
        [
            "definition",
            "profile",
            "status",
            "selected_mean",
            "zinc_mean",
            "difference_selected_minus_zinc",
            "p_exact_two_sided_exploratory",
            "rank_biserial",
            "minimum_attainable_two_sided_p",
        ],
    )
    residues = read_csv(
        residue_path,
        [
            "resnum",
            "domain",
            "axis_distance_A",
            "endpoint_displacement_A",
            "pca_sqfluct_top10",
            "pca_tbd_axis_adjusted_residual",
            "tbd_internal_ensemble_variance_A2",
            "membership",
        ],
    )
    summary = read_json(summary_path)
    inputs = [set_path, residue_path, summary_path]

    profile_order = ["pca_sqfluct_top10", "pca_tbd_axis_adjusted_residual", "tbd_internal_ensemble_variance_A2"]
    definition_order = ["uniprot_ligand_annotations", "5fqd_4.5A_contact_shell_common_window"]
    plot_rows = [
        row
        for row in set_rows
        if row["status"] == "ok" and row["profile"] in profile_order and row["definition"] in definition_order
    ]
    if not plot_rows:
        raise NotReady("pocket source rows are absent after profile/definition filtering")
    source = write_csv(
        ctx.source_dir / "FigS5_source.csv",
        [{**row, "panel": "a,b,c"} for row in plot_rows]
        + [{**row, "panel": "d"} for row in residues if row.get("membership")],
    )

    profile_labels = {
        "pca_sqfluct_top10": "PCA coord.\nvariance",
        "pca_tbd_axis_adjusted_residual": "TBD-axis\nresidual",
        "tbd_internal_ensemble_variance_A2": "TBD internal\nvariance",
    }
    definition_labels = {
        "uniprot_ligand_annotations": "UniProt ligand",
        "5fqd_4.5A_contact_shell_common_window": "5FQD contact shell",
    }
    def row_for(definition: str, profile: str) -> Mapping[str, str] | None:
        for row in plot_rows:
            if row["definition"] == definition and row["profile"] == profile:
                return row
        return None

    apply_publication_style("FigS5_strengthening")
    fig = plt.figure(figsize=(MAIN_WIDTH_IN, 6.35), dpi=300)
    grid = fig.add_gridspec(2, 2, left=0.10, right=0.99, bottom=0.10, top=0.96, wspace=0.42, hspace=0.52)
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1])
    axc = fig.add_subplot(grid[1, 0])
    axd = fig.add_subplot(grid[1, 1])

    x = np.arange(len(profile_order))
    width = 0.34
    for offset, definition, color in [(-width / 2, definition_order[0], BLUE), (width / 2, definition_order[1], GREEN)]:
        selected_vals = []
        zinc_vals = []
        for profile in profile_order:
            row = row_for(definition, profile)
            selected_vals.append(float_value(row, "selected_mean") if row else np.nan)
            zinc_vals.append(float_value(row, "zinc_mean") if row else np.nan)
        axa.scatter(x + offset, selected_vals, marker="o", color=color, s=36, label=f"{definition_labels[definition]} selected")
        axa.scatter(x + offset, zinc_vals, marker="x", color=color, s=36, label=f"{definition_labels[definition]} zinc")
        for xpos, selected, zinc in zip(x + offset, selected_vals, zinc_vals):
            if math.isfinite(selected) and math.isfinite(zinc):
                axa.plot([xpos, xpos], [zinc, selected], color=color, linewidth=0.9, alpha=0.75)
    axa.set_xticks(x, [profile_labels[p] for p in profile_order])
    axa.set_ylabel("Mean variance/residual (Å²)")
    axa.legend(loc="upper right", fontsize=6.4, ncols=1)
    finish_axis(axa, grid="y")
    panel_label(axa, "a")

    diff_positions: list[float] = []
    diff_values: list[float] = []
    diff_colors: list[str] = []
    diff_labels: list[str] = []
    for i, profile in enumerate(profile_order):
        for j, definition in enumerate(definition_order):
            row = row_for(definition, profile)
            if row:
                diff_positions.append(i + (j - 0.5) * width)
                diff_values.append(float_value(row, "difference_selected_minus_zinc"))
                diff_colors.append(BLUE if j == 0 else GREEN)
                diff_labels.append(definition_labels[definition] if i == 0 else "")
    bars = axb.bar(diff_positions, diff_values, width=width * 0.92, color=diff_colors)
    for bar, label in zip(bars, diff_labels):
        if label:
            bar.set_label(label)
    axb.set_xticks(x, [profile_labels[p] for p in profile_order])
    axb.set_ylabel("Selected - zinc (Å²)")
    axb.legend(loc="best", fontsize=6.8)
    finish_axis(axb, grid="y", zero_line=True)
    panel_label(axb, "b")

    pvals = [float_value(row, "p_exact_two_sided_exploratory") for row in plot_rows]
    min_p = [float_value(row, "minimum_attainable_two_sided_p") for row in plot_rows]
    ranks = [float_value(row, "rank_biserial") for row in plot_rows]
    colors = [BLUE if row["definition"] == definition_order[0] else GREEN for row in plot_rows]
    axc.scatter(min_p, pvals, c=colors, s=[28 + 30 * abs(r) for r in ranks], alpha=0.82, edgecolor="white", linewidth=0.45)
    lim_max = min(1.05, max(max(pvals), max(min_p)) * 1.18)
    axc.plot([0, lim_max], [0, lim_max], color=MID_GREY, linewidth=0.7, linestyle="--")
    axc.set(xlabel="Minimum attainable exact p", ylabel="Exploratory exact p", xlim=(0, lim_max), ylim=(0, lim_max))
    finish_axis(axc, grid="both")
    panel_label(axc, "c")

    domain_colors = {"NTD": BLUE, "HB": GREEN, "TBD": ORANGE}
    residue_plot = [row for row in residues if row.get("pca_sqfluct_top10") not in ("", None)]
    axd.scatter(
        [float_value(row, "resnum") for row in residue_plot],
        [float_value(row, "pca_sqfluct_top10") for row in residue_plot],
        c=[domain_colors.get(row["domain"], MID_GREY) for row in residue_plot],
        s=12,
        alpha=0.55,
        edgecolor="none",
    )
    pocket_members = [row for row in residues if row.get("membership")]
    for row in pocket_members:
        axd.scatter(
            float_value(row, "resnum"),
            float_value(row, "pca_sqfluct_top10"),
            color=MAGENTA if "5fqd" in row["membership"].lower() else BLACK,
            s=32,
            marker="D" if "zinc" not in row["membership"].lower() else "s",
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    axd.set(xlabel="CRBN residue", ylabel="PCA coordinate variance (Å²)")
    from matplotlib.lines import Line2D

    axd.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="none", markersize=5.0, label="NTD"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor="none", markersize=5.0, label="HB"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="none", markersize=5.0, label="TBD"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=MAGENTA, markeredgecolor="white", markersize=5.0, label="pocket"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=BLACK, markeredgecolor="white", markersize=5.0, label="zinc"),
        ],
        loc="upper left",
        fontsize=6.2,
        ncols=2,
    )
    finish_axis(axd, grid="both")
    panel_label(axd, "d")

    outputs = save_figure(fig, ctx.output_dir, "FigS5")
    return {
        "figure": "FigS5",
        "status": "rendered",
        "outputs": outputs,
        "source_data": repo_rel(source),
        "inputs": [repo_rel(path) for path in inputs],
        "panel_mapping": {
            "a": "pocket-selected and zinc residue means for PCA coordinate variance, TBD-axis residual, and TBD internal variance",
            "b": "selected-minus-zinc signed differences in Å² for the same metrics",
            "c": "exploratory exact p-values against their minimum attainable limits",
            "d": "residue-level PCA coordinate variance with annotated pocket/zinc memberships",
        },
        "summary_highlights": summary.get("pocket_effect_highlights", {}),
    }


def build_figs6(ctx: BuildContext) -> dict[str, Any]:
    rigid_path = DATA / "assembly_rigid_null.json"
    sensitivity_path = DATA / "anm_sensitivity_ext.json"
    window_path = DATA / "window_sensitivity.json"
    null_data = read_json(rigid_path)
    sensitivity = read_json(sensitivity_path)
    window = read_json(window_path)
    inputs = [rigid_path, sensitivity_path, window_path]

    rigid = null_data["rigid_domain_null"]
    model_names = ["two_block", "three_block", "equal_displacement_boundary", "bond_length_preserving_boundary"]
    null_rows = []
    for name in model_names:
        model = rigid.get(name)
        if not isinstance(model, dict):
            continue
        null_rows.append(
            {
                "panel": "a,b",
                "model": name,
                "internal_dim": model.get("internal_dim", rigid.get(f"{name}_internal_dim", "")),
                "observed_direction_cosine_in_subspace": model.get("observed_direction_cosine_in_subspace", ""),
                "observed_projected_mode1_overlap": model.get("observed_projected_mode1_overlap", ""),
                "subspace_capture_of_transition": model.get("subspace_capture_of_transition", ""),
                "null_mean": model.get("null_mean", rigid.get("null_mean", "")),
                "null_sd": model.get("null_sd", ""),
                "null_p95": model.get("null_p95", rigid.get("null_p95", "")),
                "p_exact": model.get("p_exact", ""),
                "z": model.get("z", ""),
            }
        )
    if not null_rows:
        raise NotReady("rigid-domain null model rows are absent")

    cutoff_rows = []
    for pdb, rows_by_cutoff in sensitivity["cutoff_scan"].items():
        if not isinstance(rows_by_cutoff, dict):
            raise NotReady(f"unexpected cutoff_scan schema for {pdb}")
        for cutoff, row in rows_by_cutoff.items():
            cutoff_rows.append({"panel": "c", "pdb": pdb, "cutoff_A": cutoff, **row})
    if not cutoff_rows:
        raise NotReady("cutoff sensitivity rows are absent")

    empty_rows = []
    empty_middle = window.get("empty_middle", {})
    for rule, payload in empty_middle.items():
        if isinstance(payload, dict):
            empty_rows.append(
                {
                    "panel": "d",
                    "rule": rule,
                    "n_open": payload.get("n_open", payload.get("open_n", payload.get("open_count", ""))),
                    "n_closed": payload.get("n_closed", payload.get("closed_n", payload.get("closed_count", ""))),
                    "gap_A": payload.get("gap_A", payload.get("empty_middle_width_A", payload.get("gap_width_A", ""))),
                    "open_min_A": payload.get("open_min_A", payload.get("open_lower_A", "")),
                    "closed_max_A": payload.get("closed_max_A", payload.get("closed_upper_A", "")),
                }
            )
    source = write_csv(ctx.source_dir / "FigS6_source.csv", null_rows + cutoff_rows + empty_rows)

    apply_publication_style("FigS6_strengthening")
    fig = plt.figure(figsize=(MAIN_WIDTH_IN, 6.15), dpi=300)
    grid = fig.add_gridspec(2, 2, left=0.11, right=0.99, bottom=0.10, top=0.91, wspace=0.42, hspace=0.58)
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1])
    axc = fig.add_subplot(grid[1, 0])
    axd = fig.add_subplot(grid[1, 1])

    labels = [row["model"].replace("_", "\n") for row in null_rows]
    observed = [float_value(row, "observed_direction_cosine_in_subspace") for row in null_rows]
    null_means = [float_value(row, "null_mean") for row in null_rows]
    null_p95 = [float_value(row, "null_p95") for row in null_rows]
    xpos = np.arange(len(labels))
    axa.bar(xpos - 0.18, null_means, width=0.34, color=LIGHT_GREY, edgecolor=MID_GREY, label="null mean")
    axa.bar(xpos + 0.18, observed, width=0.34, color=BLUE, label="observed")
    axa.scatter(xpos - 0.18, null_p95, marker="_", color=ORANGE, s=90, label="null p95")
    axa.set_xticks(xpos, labels)
    axa.tick_params(axis="x", rotation=0)
    axa.set_ylabel("Matched direction cosine")
    axa.legend(loc="upper center", bbox_to_anchor=(0.52, 1.30), ncols=3, fontsize=6.3)
    finish_axis(axa, grid="y")
    panel_label(axa, "a")

    subspace = [float_value(row, "subspace_capture_of_transition") for row in null_rows]
    dims = [float_value(row, "internal_dim") for row in null_rows]
    axb.scatter(dims, subspace, color=GREEN, s=45, edgecolor="white", linewidth=0.45)
    label_offsets = {
        "equal_displacement_boundary": (-5, -30),
        "bond_length_preserving_boundary": (-25, 20),
        "two_block": (10, -25),
        "three_block": (0, -25),
    }
    label_text = {
        "two_block": "two",
        "three_block": "three",
        "equal_displacement_boundary": "equal",
        "bond_length_preserving_boundary": "bond",
    }
    for row, dim, cap in zip(null_rows, dims, subspace):
        dx, dy = label_offsets.get(row["model"], (4, 4))
        axb.annotate(
            label_text.get(row["model"], row["model"].split("_")[0]),
            (dim, cap),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5,
            ha="right" if dx < 0 else "left",
            va="top" if dy < 0 else "bottom",
            arrowprops={"arrowstyle": "-", "color": MID_GREY, "linewidth": 0.45, "shrinkA": 1.5, "shrinkB": 2.0},
        )
    axb.set(xlabel="Internal dimension", ylabel="Transition captured by subspace", xlim=(0, 14), ylim=(0, 1.08))
    finish_axis(axb, grid="both")
    panel_label(axb, "b")

    for pdb in sorted({row["pdb"] for row in cutoff_rows}):
        rows = sorted([row for row in cutoff_rows if row["pdb"] == pdb], key=lambda r: float_value(r, "cutoff_A"))
        color = ORANGE if pdb == "8CVP" else MID_GREY
        alpha = 0.95 if pdb == "8CVP" else 0.45
        lw = 1.5 if pdb == "8CVP" else 0.8
        axc.plot(
            [float_value(row, "cutoff_A") for row in rows],
            [float_value(row, "mode1_overlap") for row in rows],
            marker="o" if pdb == "8CVP" else None,
            markersize=3.2,
            linewidth=lw,
            color=color,
            alpha=alpha,
            label=pdb if pdb == "8CVP" else None,
        )
    axc.axvline(15, color=BLACK, linewidth=0.7, linestyle="--")
    axc.set(xlabel="ANM cutoff (A)", ylabel="Mode-1 overlap", ylim=(0, 1.0))
    axc.legend(loc="best")
    finish_axis(axc, grid="both")
    panel_label(axc, "c")

    display_rules = [row for row in empty_rows if row.get("gap_A") not in ("", None)]
    if display_rules:
        display_rules = display_rules[:7]
        rule_labels = {
            "a_paper_rule": "paper",
            "b_coverage_90": "cov90",
            "b_coverage_95": "cov95",
            "c_drop_424_no_resolution_ceiling": "drop424",
            "d_terminal_gaps_only": "terminal",
            "e_best_covered_chain": "best chain",
            "f_no_resolution_ceiling": "no res",
            "g_coverage95_no_res_ceiling": "cov95 no res",
        }
        axd.barh(
            [rule_labels.get(row["rule"], row["rule"]) for row in display_rules],
            [float_value(row, "gap_A") for row in display_rules],
            color=PALE_BLUE,
            edgecolor=BLUE,
        )
        axd.set_xlabel("Empty-middle gap (A)")
    else:
        meta = sensitivity.get("meta", {})
        gap_resnums = meta.get("gap_flanking_resnums", [])
        values = [float(v) for v in gap_resnums] if gap_resnums else []
        if not values:
            values = [float(meta.get("n_gaps", 0))]
        axd.bar(["chain-gap\nflanks"], [len(values)], color=PALE_BLUE, edgecolor=BLUE)
        axd.set_ylabel("Count")
    finish_axis(axd, grid="x")
    panel_label(axd, "d")

    outputs = save_figure(fig, ctx.output_dir, "FigS6")
    return {
        "figure": "FigS6",
        "status": "rendered",
        "outputs": outputs,
        "source_data": repo_rel(source),
        "inputs": [repo_rel(path) for path in inputs],
        "panel_mapping": {
            "a": "observed matched direction cosine against analytic rigid-domain null summaries",
            "b": "rigid-domain subspace dimension versus transition capture",
            "c": "ANM cutoff sensitivity across the five open references, highlighting 8CVP",
            "d": "window-rule empty-middle gap summary or gap-flank diagnostic when no gap width is available",
        },
    }

def source_manifest(records: Sequence[Mapping[str, Any]], ctx: BuildContext) -> Path:
    paths: dict[str, str] = {}
    for record in records:
        for path_text in record.get("inputs", []):
            path = ROOT / path_text if not os.path.isabs(path_text) else Path(path_text)
            if path.is_file():
                paths[path_text] = sha256_file(path)
        source_text = record.get("source_data")
        if source_text:
            path = ROOT / source_text if not os.path.isabs(source_text) else Path(source_text)
            if path.is_file():
                paths[source_text] = sha256_file(path)
        for output in record.get("outputs", {}).values():
            path = ROOT / output if not os.path.isabs(output) else Path(output)
            if path.is_file():
                paths[output] = sha256_file(path)
    rows = [{"path": path, "sha256": digest} for path, digest in sorted(paths.items())]
    return write_csv(ctx.source_dir / "manifest_sha256.csv", rows, ["path", "sha256"])


def fmt3(value: float) -> str:
    return f"{value:.3f}"


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def rows_for(ctx: BuildContext, name: str, required: Iterable[str]) -> list[dict[str, str]]:
    return read_csv(ctx.source_dir / name, required)


def collect_legend_facts(ctx: BuildContext) -> dict[str, Any]:
    facts: dict[str, Any] = {}

    fig1 = rows_for(
        ctx,
        "Fig1_source.csv",
        ["panel", "state", "open_mask", "component_or_mode", "pca_individual_variance_fraction", "anm_cumulative_directional_overlap", "rmsip_global_metric"],
    )
    fig1_structures = [row for row in fig1 if row["panel"] == "a,d"]
    state_counts = {state: sum(row["state"] == state for row in fig1_structures) for state in STATE_LABELS}
    open_count = sum(bool_value(row, "open_mask") for row in fig1_structures)
    closed_count = len(fig1_structures) - open_count
    assert_equal(len(fig1_structures), 70, "Fig1 structure count")
    assert_equal(open_count, 5, "Fig1 open-state count")
    assert_equal(closed_count, 65, "Fig1 closed-state count")
    assert_equal(state_counts, {"drug-conditioned": 66, "genuine-apo": 3, "native-substrate": 1}, "Fig1 condition counts")
    native = [row for row in fig1_structures if row["state"] == "native-substrate"]
    assert_equal(len(native), 1, "Fig1 native-substrate count")
    assert_equal(bool_value(native[0], "open_mask"), False, "Fig1 native-substrate open mask")
    pc1 = [float_value(row, "pca_individual_variance_fraction") for row in fig1 if row["component_or_mode"] == "1" and row["pca_individual_variance_fraction"]]
    anm1 = [float_value(row, "anm_cumulative_directional_overlap") for row in fig1 if row["component_or_mode"] == "1" and row["anm_cumulative_directional_overlap"]]
    anm10 = [float_value(row, "anm_cumulative_directional_overlap") for row in fig1 if row["component_or_mode"] == "10" and row["anm_cumulative_directional_overlap"]]
    rmsip = [float_value(row, "rmsip_global_metric") for row in fig1 if row["rmsip_global_metric"]]
    assert_close(pc1[0], 0.8833604808205999, "Fig1 PC1 variance")
    assert_close(anm1[0], 0.7442023669393273, "Fig1 ANM mode-1 cumulative overlap")
    assert_close(anm10[0], 0.8805245009129341, "Fig1 ANM mode-10 cumulative overlap")
    assert_close(rmsip[0], 0.6414534130944979, "Fig1 RMSIP")
    facts["fig1"] = {"state_counts": state_counts, "open": open_count, "closed": closed_count, "pc1": pc1[0], "anm1": anm1[0], "anm10": anm10[0], "rmsip": rmsip[0]}

    fig2 = rows_for(ctx, "Fig2_source.csv", ["panel", "cutoff_A", "state_group", "pdb", "mode1_overlap", "best_mode_rank", "basis", "best20_rank", "top3_subspace_projection"])
    primary = [row for row in fig2 if row["panel"] == "a,b" and abs(float_value(row, "cutoff_A") - 15.0) < 1e-9]
    open15 = [row for row in primary if row["state_group"] == "open"]
    closed15 = [row for row in primary if row["state_group"] == "closed_all70"]
    assert_equal(len(open15), 5, "Fig2 15 A open rows")
    assert_equal(len(closed15), 65, "Fig2 15 A closed rows")
    assert_equal(sum(int(float_value(row, "best_mode_rank")) == 1 for row in open15), 5, "Fig2 15 A open rank-1 rows")
    assert_equal(sum(int(float_value(row, "best_mode_rank")) == 1 for row in closed15), 3, "Fig2 15 A closed rank-1 rows")
    open13 = [row for row in fig2 if row["panel"] == "a,b" and row["state_group"] == "open" and abs(float_value(row, "cutoff_A") - 13.0) < 1e-9]
    assert_equal(len(open13), 5, "Fig2 13 A open rows")
    rank_loss13 = sorted(row["pdb"] for row in open13 if int(float_value(row, "best_mode_rank")) != 1)
    assert_equal(rank_loss13, ["6H0F", "7U8F"], "Fig2 13 A engineered rank-loss references")
    fixed = [row for row in fig2 if row["panel"] == "c,d" and row["basis"] == "fixed_8CVP"]
    own = [row for row in fig2 if row["panel"] == "c,d" and row["basis"] == "own_open"]
    assert_equal(len(fixed), 325, "Fig2 fixed-basis pair count")
    assert_equal(len(own), 325, "Fig2 own-open pair count")
    facts["fig2"] = {
        "open15_median": median(float_value(row, "mode1_overlap") for row in open15),
        "closed15_median": median(float_value(row, "mode1_overlap") for row in closed15),
        "fixed_median": median(float_value(row, "mode1_overlap") for row in fixed),
        "own_median": median(float_value(row, "mode1_overlap") for row in own),
        "fixed_rank1": sum(int(float_value(row, "best20_rank")) == 1 for row in fixed),
        "own_rank1": sum(int(float_value(row, "best20_rank")) == 1 for row in own),
        "rank_loss13": rank_loss13,
    }

    fig3 = rows_for(ctx, "Fig3_source.csv", ["pdb", "cutoff_A", "interface_alpha", "model", "best_mode", "best_crbn_directional_overlap", "best_crbn_amplitude", "internal_best_overlap", "internal_best_mode"])
    joint = [row for row in fig3 if row["model"] == "joint" and abs(float_value(row, "cutoff_A") - 15.0) < 1e-9 and abs(float_value(row, "interface_alpha") - 1.0) < 1e-9]
    assert_equal(len(joint), 5, "Fig3 joint alpha=1 primary rows")
    ranks = [int(float_value(row, "best_mode")) for row in joint]
    overlaps = [float_value(row, "best_crbn_directional_overlap") for row in joint]
    amplitudes = [float_value(row, "best_crbn_amplitude") for row in joint]
    assert_equal(ranks, [5, 6, 7, 4, 4], "Fig3 joint alpha=1 best-mode ranks")
    for observed, expected, label in zip(overlaps, [0.7265, 0.6466, 0.6715, 0.4509, 0.4205], [row["pdb"] for row in joint]):
        assert_close(observed, expected, f"Fig3 joint alpha=1 overlap {label}", tol=5e-4)
    assert_close(median(overlaps), 0.6466332988192851, "Fig3 joint alpha=1 median overlap")
    assert_close(median(amplitudes), 0.934328523588207, "Fig3 joint alpha=1 median amplitude")
    internal_overlaps = [float_value(row, "internal_best_overlap") for row in joint]
    internal_ranks = [int(float_value(row, "internal_best_mode")) for row in joint]
    for observed, expected, label in zip(internal_overlaps, [0.79646, 0.7397, 0.7539, 0.5667, 0.6058], [row["pdb"] for row in joint]):
        assert_close(observed, expected, f"Fig3 joint alpha=1 internal overlap {label}", tol=5e-4)
    assert_equal(internal_ranks, [5, 6, 7, 2, 3], "Fig3 joint alpha=1 internal best-mode ranks")
    facts["fig3"] = {"pdbs": [row["pdb"] for row in joint], "ranks": ranks, "overlaps": overlaps, "amplitudes": amplitudes, "internal_overlaps": internal_overlaps, "internal_ranks": internal_ranks, "internal_median": median(internal_overlaps)}

    fig4 = rows_for(ctx, "Fig4_source.csv", ["panel", "residue", "contact_class", "discovery_rank", "stable_apo_model_candidate", "selected_for_heatmap", "selected_for_stable_map", "S_close", "condition"])
    discovery = [row for row in fig4 if row["panel"] == "a,b,c" and row["contact_class"]]
    stable = {(row["residue"], row["contact_class"]) for row in discovery if row["selected_for_stable_map"] == "1"}
    heat = {(row["residue"], row["contact_class"]) for row in discovery if row["selected_for_heatmap"] == "1"}
    summaries = [row for row in fig4 if row["panel"] == "d"]
    primary_sclose = [float_value(row, "S_close") for row in summaries if row["condition"] == "8CVP_15A"]
    assert_equal(len(discovery), 142, "Fig4 discovery group rows")
    assert_equal(len(stable), 8, "Fig4 stable apo-model candidate count")
    assert_equal(len(heat), 14, "Fig4 heatmap union count")
    assert_close(primary_sclose[0], 54.4579, "Fig4 primary S_close", tol=5e-4)
    facts["fig4"] = {"discovery": len(discovery), "stable": len(stable), "heat": len(heat), "primary_sclose": primary_sclose[0]}

    fig5 = rows_for(
        ctx,
        "Fig5_source.csv",
        [
            "panel",
            "accession",
            "condition",
            "refit_rg_nm",
            "q_unit",
            "low_q_qa",
            "in_primary_guinier_fit",
            "kratky_x_qrg",
            "variant",
            "primary_269_window",
            "dsf_delta_delta_tm_vs_wt_degC",
            "saxs_delta_rg_vs_wt_angstrom",
        ],
    )
    guinier = [row for row in fig5 if row["panel"] == "a"]
    curves = [row for row in fig5 if row["panel"] == "b,c"]
    accessions = sorted({row["accession"] for row in guinier})
    curve_accessions = sorted({row["accession"] for row in curves})
    guinier_fit_points = [row for row in curves if bool_value(row, "in_primary_guinier_fit")]
    kratky_display = [row for row in curves if row.get("kratky_y") not in ("", None) and float_value(row, "kratky_x_qrg") <= 4.0]
    compound = [row for row in fig5 if row["panel"] == "d"]
    mutants = [row for row in compound if row["variant"] != "WT"]
    assert_equal(len(guinier), 5, "Fig5 SAXS refit rows")
    assert_equal(curve_accessions, accessions, "Fig5 raw curve accession set")
    assert_equal(sorted({row["accession"] for row in guinier_fit_points}), accessions, "Fig5 Guinier curve accession set")
    assert_equal(sorted({row["accession"] for row in kratky_display}), accessions, "Fig5 displayed Kratky accession set")
    assert_equal(len(compound), 6, "Fig5 Compound 9 case rows")
    assert_equal(len(mutants), 5, "Fig5 Compound 9 mutant rows")
    assert_equal(sum(row["primary_269_window"] == "outside" for row in mutants), 1, "Fig5 outside mutant count")
    assert_equal(sum(row["primary_269_window"] == "inside" for row in mutants), 3, "Fig5 inside mutant count")
    assert_equal(sum(row["primary_269_window"] == "mixed" for row in mutants), 1, "Fig5 mixed mutant count")
    l60a = [row for row in mutants if row["variant"] == "L60A"][0]
    double = [row for row in mutants if row["variant"] == "L60A H378A"][0]
    facts["fig5"] = {
        "rg": [float_value(row, "refit_rg_nm") for row in guinier],
        "qa_pass": sum(row["low_q_qa"] == "pass" for row in guinier),
        "q_assumed": [row["accession"] for row in guinier if row["q_unit"] == "assumed_inverse_angstrom"],
        "q_explicit": [row["accession"] for row in guinier if row["q_unit"] == "inverse_angstrom"],
        "curve_points": len(curves),
        "guinier_fit_points": len(guinier_fit_points),
        "kratky_display_points": len(kratky_display),
        "l60a_ddtm": float_value(l60a, "dsf_delta_delta_tm_vs_wt_degC"),
        "l60a_drg": float_value(l60a, "saxs_delta_rg_vs_wt_angstrom"),
        "double_ddtm": float_value(double, "dsf_delta_delta_tm_vs_wt_degC"),
        "double_drg": float_value(double, "saxs_delta_rg_vs_wt_angstrom"),
    }

    figs4 = rows_for(ctx, "FigS4_source.csv", ["panel", "source", "mode1_overlap", "best20_rank", "top3_projection", "observed_finite_chord_displacement_A", "observed_chord_vs_tangent_cosine", "reference_role"])
    ext = [row for row in figs4 if row["panel"] == "a,b" and row["source"] == "existing_external_primary_control_panel"]
    observed = [row for row in figs4 if row["panel"] == "a,b" and row["source"] in {"crbn_primary_pair", "new_pair_level_crbn_primary_pair"}]
    tangent = [row for row in figs4 if row["panel"] == "c"]
    pair_rows = [row for row in figs4 if row["panel"] == "d"]
    assert_equal(len(ext), 18, "FigS4 external controls")
    assert_equal(len(pair_rows), 325, "FigS4 fixed pair-axis rows")
    facts["figs4"] = {
        "crbn_mode1": float_value(observed[0], "mode1_overlap"),
        "crbn_rank": int(float_value(observed[0], "best20_rank")),
        "crbn_top3": float_value(observed[0], "top3_projection"),
        "chord_min": min(float_value(row, "observed_finite_chord_displacement_A") for row in tangent),
        "chord_max": max(float_value(row, "observed_finite_chord_displacement_A") for row in tangent),
        "cos_min": min(float_value(row, "observed_chord_vs_tangent_cosine") for row in tangent),
        "cos_max": max(float_value(row, "observed_chord_vs_tangent_cosine") for row in tangent),
        "pair_median": median(float_value(row, "mode1_overlap") for row in pair_rows),
    }

    figs5 = rows_for(ctx, "FigS5_source.csv", ["panel", "definition", "profile", "difference_selected_minus_zinc", "p_exact_two_sided_exploratory", "minimum_attainable_two_sided_p"])
    summary = [row for row in figs5 if row["panel"] == "a,b,c"]
    uni = [row for row in summary if row["definition"] == "uniprot_ligand_annotations"]
    shell = [row for row in summary if row["definition"] == "5fqd_4.5A_contact_shell_common_window"]
    order = ["pca_sqfluct_top10", "pca_tbd_axis_adjusted_residual", "tbd_internal_ensemble_variance_A2"]
    uni = sorted(uni, key=lambda row: order.index(row["profile"]))
    shell = sorted(shell, key=lambda row: order.index(row["profile"]))
    assert_equal(len(uni), 3, "FigS5 UniProt ligand summary rows")
    assert_equal(len(shell), 3, "FigS5 5FQD shell summary rows")
    facts["figs5"] = {
        "uni_deltas": [float_value(row, "difference_selected_minus_zinc") for row in uni],
        "shell_deltas": [float_value(row, "difference_selected_minus_zinc") for row in shell],
        "min_p": min(float_value(row, "p_exact_two_sided_exploratory") for row in summary if row["p_exact_two_sided_exploratory"]),
        "min_attainable": min(float_value(row, "minimum_attainable_two_sided_p") for row in summary if row["minimum_attainable_two_sided_p"]),
    }

    figs6 = rows_for(ctx, "FigS6_source.csv", ["panel", "model", "observed_direction_cosine_in_subspace", "observed_projected_mode1_overlap", "null_mean", "null_p95", "p_exact", "pdb", "cutoff_A", "mode1_overlap", "rule", "gap_A"])
    nulls = [row for row in figs6 if row["panel"] == "a,b"]
    model_names = [row["model"] for row in nulls]
    assert_equal(model_names, ["two_block", "three_block", "equal_displacement_boundary", "bond_length_preserving_boundary"], "FigS6 null model order")
    cosines = [float_value(row, "observed_direction_cosine_in_subspace") for row in nulls]
    p95 = [float_value(row, "null_p95") for row in nulls]
    p_exact = [float_value(row, "p_exact") for row in nulls]
    for observed, expected, model in zip(cosines, [0.8106, 0.8041, 0.9332, 0.8326], model_names):
        assert_close(observed, expected, f"FigS6 matched cosine {model}", tol=5e-4)
    cvp_cutoffs = [row for row in figs6 if row["panel"] == "c" and row["pdb"] == "8CVP"]
    gap_rows = [row for row in figs6 if row["panel"] == "d" and row["gap_A"]]
    facts["figs6"] = {"models": model_names, "cosines": cosines, "p95": p95, "p_exact": p_exact, "cvp_min": min(float_value(row, "mode1_overlap") for row in cvp_cutoffs), "cvp_max": max(float_value(row, "mode1_overlap") for row in cvp_cutoffs), "gap_min": min(float_value(row, "gap_A") for row in gap_rows), "gap_max": max(float_value(row, "gap_A") for row in gap_rows), "gap_n": len(gap_rows)}

    return facts


def write_legends(records: Sequence[Mapping[str, Any]], ctx: BuildContext) -> Path:
    rendered = {
        record["figure"]: record
        for record in records
        if str(record.get("status", "")).startswith("rendered") or record.get("status") == "copied_legacy"
    }
    facts = collect_legend_facts(ctx)
    f1 = facts["fig1"]
    f2 = facts["fig2"]
    f3 = facts["fig3"]
    f4 = facts["fig4"]
    f5 = facts["fig5"]
    fs4 = facts["figs4"]
    fs5 = facts["figs5"]
    fs6 = facts["figs6"]
    f3_pairs = ", ".join(f"{pdb} rank {rank}, overlap {fmt4(overlap)}" for pdb, rank, overlap in zip(f3["pdbs"], f3["ranks"], f3["overlaps"]))
    f3_internal_pairs = ", ".join(f"{pdb} rank {rank}, overlap {fmt4(overlap)}" for pdb, rank, overlap in zip(f3["pdbs"], f3["internal_ranks"], f3["internal_overlaps"]))
    fs6_pairs = ", ".join(f"{model.replace('_', ' ')} {fmt3(cos)}/p95 {fmt3(p95)}" for model, cos, p95 in zip(fs6["models"], fs6["cosines"], fs6["p95"]))
    legend_text = {
        "Fig1": f"Fig. 1. Frozen 70-structure CRBN conformational census and mode relationship. (a) PCA projection of the 269 C-alpha common residue window separates {f1['open']} open and {f1['closed']} closed structures; ligand-condition counts are {f1['state_counts']['drug-conditioned']} drug-conditioned, {f1['state_counts']['genuine-apo']} genuine apo, and {f1['state_counts']['native-substrate']} native-substrate structures, with the native-substrate structure classified as closed on PC1. (b) PCA variance spectrum shows PC1 explaining {100*f1['pc1']:.1f}% of coordinate variance. (c) CRBN 8CVP ANM cumulative directional overlap reaches {fmt3(f1['anm1'])} at mode 1 and {fmt3(f1['anm10'])} by mode 10; RMSIP = {fmt3(f1['rmsip'])} is shown as a separate ANM/PCA subspace metric and is not a horizontal overlap threshold. (d) Normalized PC1 rescales the PC1 coordinate so the closed mean is 0 and the open mean is 1.",
        "Fig2": f"Fig. 2. CRBN ANM robustness in the full frozen ensemble. (a) At the primary 15 A cutoff, the five open references have median mode-1 overlap {fmt3(f2['open15_median'])} and all five recover the transition at rank 1, while the 65 closed structures have median mode-1 overlap {fmt3(f2['closed15_median'])} and 3/65 rank-1 cases. (b) Cutoff sensitivity shows that rank and overlap remain distinct: at 13 A, engineered open references {', '.join(f2['rank_loss13'])} shift to best rank 2, whereas all five open references are rank 1 at 15 A. (c) The 650 open-closed pair comparison separates fixed-8CVP-basis and own-open-basis calculations across 325 pairs per basis; median mode-1 overlaps are {fmt3(f2['fixed_median'])} and {fmt3(f2['own_median'])}, respectively. (d) Rank-1 recovery is {f2['fixed_rank1']}/325 for the fixed basis and {f2['own_rank1']}/325 for the own-open basis, showing that amplitude, rank, and basis choice are separable checks of the same transition vector.",
        "Fig3": f"Fig. 3. Matched CRBN-DDB1 elastic-network definitions across five open references and the 269-residue CRBN window. (a) Best CRBN directional overlap is shown for isolated CRBN and DDB1-aware models at the primary 15 A cutoff. For the five joint alpha = 1 rows, the best-mode ranks and directional overlaps are {f3_pairs}; the median overlap is {fmt4(median(f3['overlaps']))}. (b) CRBN amplitude reports how much of the selected mode lies on CRBN rather than the partner; median joint alpha = 1 amplitude is {fmt4(median(f3['amplitudes']))}. (c) Joint CRBN-internal best directional overlaps are included with the isolated, Schur-static, and fixed-partner internal comparisons; joint internal rows are {f3_internal_pairs}, with median {fmt3(f3['internal_median'])}. (d) Best-mode rank distinguishes rank from overlap magnitude across isolated, joint, Schur-static, and fixed-partner definitions.",
        "Fig4": f"Fig. 4. Contact-perturbation candidates from the seven-condition contact analysis. (a) The top five signed D_g candidates are shown for each discovered class; D_g is the signed change in the contact perturbation objective, not a thermodynamic free energy. (b) The heat map shows pass, fail, or absent status for the union of top-ten and stable apo-model candidates across five apo conditions and two engineered references, giving {f4['heat']} unique residue-class candidates from {f4['discovery']} discovery groups. (c) The observed 8CVP C-alpha trace locates the {f4['stable']} stable apo-model candidates on the NTD, HB, and TBD domain map. (d) Baseline specificity S_close is shown for each condition; the primary 8CVP 15 A run has S_close = {fmt4(f4['primary_sclose'])}.",
        "Fig5": f"Fig. 5. Solution compaction and published mutant responses in engineered CRBN constructs. (a) Guinier refits for SASDU52, SASDU62, SASDU72, SASDU82, and SASDU92 give Rg values of {', '.join(fmt3(v) for v in f5['rg'])} nm, respectively; {', '.join(f5['q_explicit'])} have explicit inverse-A q units, while {', '.join(f5['q_assumed'])} are marked as assumed inverse-A from source context. (b) Weighted low-q Guinier displays plot ln I(q) versus q² for the five profiles using {f5['guinier_fit_points']} primary-fit points from the recorded curve arrays. (c) Dimensionless Kratky profiles are drawn from the same recorded SAXS points and displayed over qRg 0-4 using {f5['kratky_display_points']} curve points. (d) The O'Connor Compound 9 sparse case comparison plots five mutants with measured DSF ΔΔTm and SAXS ΔRg versus WT. The outside-window L60A mutant has ΔΔTm {f5['l60a_ddtm']:.2f} °C and ΔRg {f5['l60a_drg']:.2f} A; the mixed L60A H378A double mutant has ΔΔTm {f5['double_ddtm']:.2f} °C and ΔRg {f5['double_drg']:.2f} A. The mutant panel is retrospective and is not interpreted as contact-candidate overlap enrichment.",
        "FigS1": "Fig. S1. Frozen ensemble composition. (a) Resolution-bin counts are shown for the 70 frozen structures: 66 drug-conditioned, three genuine-apo, and one native-substrate entry. (b) Experimental-method counts show 42 cryo-EM and 28 X-ray structures. The panel preserves the frozen source table while reporting the actual composition used for the 269-residue common-window analyses.",
        "FigS2": f"Fig. S2. ANM/PCA mode-overlap matrix. The 10 x 10 heat map reports absolute directional overlaps between the first ten CRBN 8CVP ANM modes and the first ten ensemble PCs. ANM mode 1 overlaps PC1 by 0.745, and the ten-mode RMSIP is {fmt3(f1['rmsip'])}, matching Fig. 1c as a subspace metric rather than a threshold line.",
        "FigS3": "Fig. S3. Bootstrap stability of the frozen PC1 axis. (a) Across 2,000 resamples, the PC1 variance 95% interval is 48.2-94.2%, with the full-ensemble value at 88.3%. (b) The PC1-axis directional-overlap 95% interval is 0.754-1.000; 4.3% of resamples contain no open structure, and the 8CVP ANM mode-1 reference overlap is 0.744.",
        "FigS4": f"Fig. S4. Controls and finite-geometry diagnostics. (a) CRBN 8CVP-5FQD is plotted against 18 external primary controls; CRBN mode-1 overlap is {fmt3(fs4['crbn_mode1'])}, best rank is {fs4['crbn_rank']}, and top-3 projection is {fmt3(fs4['crbn_top3'])}. (b) The external-control best-rank distribution is shown with the CRBN rank marker. (c) Repaired finite-chord geometry is shown for 269 residues; observed finite-chord displacement ranges from {fmt3(fs4['chord_min'])} to {fmt3(fs4['chord_max'])} A and chord-versus-tangent cosine ranges from {fmt3(fs4['cos_min'])} to {fmt3(fs4['cos_max'])}. (d) The fixed 8CVP mode-1 overlap distribution contains 325 fixed-basis open-closed pair axes with median overlap {fmt3(fs4['pair_median'])}.",
        "FigS5": f"Fig. S5. Pocket PCA-variance, axis-residual, and internal-variance checks. (a) Pocket-selected and zinc residue means are shown for PCA coordinate variance, TBD-axis residual, and TBD internal ensemble variance; all three metrics are plotted in Å². (b) Signed selected-minus-zinc differences in Å² are {', '.join(f'{v:+.3f}' for v in fs5['uni_deltas'])} for UniProt ligand residues, and {', '.join(f'{v:+.3f}' for v in fs5['shell_deltas'])} for the 5FQD 4.5 A contact shell. (c) Exact exploratory p-values are plotted against their minimum attainable limits; the smallest current value is p = {fs5['min_p']:.3f}, with minimum attainable p = {fs5['min_attainable']:.3f}. (d) Residue-level PCA coordinate variance is shown with domain colors; diamonds mark pocket memberships and squares mark zinc memberships.",
        "FigS6": f"Fig. S6. Geometric-null and sensitivity diagnostics. (a) Analytic rigid-domain null summaries compare the observed matched direction cosine with the null mean and p95 for the stated null statistic; observed/p95 pairs are {fs6_pairs}. Exact p-values for the four nulls are {', '.join(f'{p:.3g}' for p in fs6['p_exact'])}. (b) Subspace dimension is plotted against transition capture. (c) ANM cutoff sensitivity across the five open references highlights 8CVP, whose mode-1 overlap ranges from {fmt3(fs6['cvp_min'])} to {fmt3(fs6['cvp_max'])} across scanned cutoffs. (d) Window-rule sensitivity preserves an empty-middle gap of {fmt3(fs6['gap_min'])} to {fmt3(fs6['gap_max'])} A across {fs6['gap_n']} curation rules.",
    }
    lines = ["# CRBN strengthening figure legends", ""]
    for stem in FIGURE_STEMS:
        record = rendered.get(stem)
        if record:
            lines.extend([f"## {stem}", "", legend_text[stem], "", f"Source data: `{record['source_data']}`", ""])
    path = ctx.source_dir / "LEGENDS.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path

def write_readiness(records: Sequence[Mapping[str, Any]], ctx: BuildContext) -> Path:
    rendered = [record["figure"] for record in records if str(record.get("status", "")).startswith("rendered")]
    copied = [record["figure"] for record in records if record.get("status") == "copied_legacy"]
    missing = [
        {"figure": record["figure"], "reason": record.get("reason", ""), "required": record.get("required", [])}
        for record in records
        if record.get("status") == "not_ready"
    ]
    payload = {
        "generated_at_utc": now_utc(),
        "input_root": repo_rel(ctx.input_root),
        "output_dir": repo_rel(ctx.output_dir),
        "source_dir": repo_rel(ctx.source_dir),
        "rendered_figures": rendered,
        "copied_legacy_figures": copied,
        "not_ready": missing,
        "panel_mapping": {record["figure"]: record.get("panel_mapping", {}) for record in records},
        "records": records,
        "notes": [
            "No unavailable values were encoded as zero.",
            "Fig3 requires the new strengthen_ddb1 outputs before it is rendered.",
            "Fig2 pair-basis panels are included when strengthen_ensemble outputs are present.",
            "FigS4 uses the owner-repaired finite-chord/tangent fields from strengthen_controls.",
            "FigS5 reports pocket comparisons as descriptive/exploratory exact tests with their attainable p-value limits.",
            "FigS6 uses frozen geometric-null and sensitivity JSON inputs under data/.",
        ],
    }
    path = ctx.source_dir / "figure_readiness.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root.resolve()
    output_dir = (args.output_dir or (input_root / "manuscript" / "figures")).resolve()
    source_dir = (args.source_dir or (input_root / "analysis" / "figure_sources")).resolve()
    ctx = BuildContext(
        input_root=input_root,
        output_dir=output_dir,
        source_dir=source_dir,
        require_all=bool(args.require_all),
    )
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.source_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "Fig1": build_fig1,
        "Fig2": build_fig2,
        "Fig3": build_fig3,
        "Fig4": build_fig4,
        "Fig5": build_fig5,
        "FigS1": lambda build_ctx: build_legacy_supplement(build_ctx, "FigS1"),
        "FigS2": lambda build_ctx: build_legacy_supplement(build_ctx, "FigS2"),
        "FigS3": lambda build_ctx: build_legacy_supplement(build_ctx, "FigS3"),
        "FigS4": build_figs4,
        "FigS5": build_figs5,
        "FigS6": build_figs6,
    }
    records: list[dict[str, Any]] = []
    for figure in FIGURE_STEMS:
        try:
            record = builders[figure](ctx)
            print(f"{figure}: {record['status']} -> {record['outputs']['png']}")
            records.append(record)
        except NotReady as exc:
            print(f"{figure}: not ready: {exc}", file=sys.stderr)
            required: list[str] = []
            if figure == "Fig3":
                required = [
                    repo_rel(ctx.input_root / "analysis/ddb1/model_summary.csv"),
                    repo_rel(ctx.input_root / "analysis/ddb1/modes.csv"),
                    repo_rel(ctx.input_root / "analysis/ddb1/model_summary.json"),
                ]
            records.append({"figure": figure, "status": "not_ready", "reason": str(exc), "required": required})

    manifest = source_manifest(records, ctx)
    legends = write_legends(records, ctx)
    readiness = write_readiness(records, ctx)
    print(f"manifest: {repo_rel(manifest)}")
    print(f"legends: {repo_rel(legends)}")
    print(f"readiness: {repo_rel(readiness)}")
    complete = all(
        str(record.get("status", "")).startswith("rendered")
        for record in records
        if record.get("figure") in MAIN_FIGURE_STEMS
    )
    if ctx.require_all and not complete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
