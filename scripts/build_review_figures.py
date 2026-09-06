#!/usr/bin/env python3
"""Render reviewed CRBN figures from completed, provenance-tracked source tables.

This module changes presentation only. Existing mechanics and contact values are
read from frozen tables; observed-window and external comparisons must have been
computed and validated by their respective analysis stages before plotting.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Rectangle
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_directional_figures import (  # noqa: E402
    BLACK, BLUE, DARK_GREY, GREEN, MID_GREY, ORANGE,
    REF_COLORS, REF_ORDER, MODEL_ORDER, MODEL_LABELS, CONTACT_COLORS,
    configure_style, finish_axis, panel_label, condition_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 7.09  # 180 mm; final drawing is not resized after export.
AA = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV",
))
FROZEN_TABLES = {
    "models": "Fig3_models_primary_15A_uniform.csv",
    "comparisons": "Fig3_comparisons_primary_15A_uniform.csv",
    "modes": "Fig3_8CVP_alpha_mode_path.csv",
    "contacts": "Fig4_selected_contact_candidates.csv",
    "all_contacts": "Fig4_8CVP_contact_roles_with_legacy.csv",
    "robustness": "Fig4_legacy_robustness.csv",
}


class FigureSourceError(ValueError):
    """Completed figure inputs do not meet the figure's data contract."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv(path: Path, columns: Iterable[str] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise FigureSourceError(f"Missing completed figure input: {path}")
    data = pd.read_csv(path)
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise FigureSourceError(f"Missing columns in {path.name}: {missing}")
    return data


def finite(data: pd.DataFrame, columns: Sequence[str]) -> None:
    if not np.isfinite(data[list(columns)].to_numpy(dtype=float)).all():
        raise FigureSourceError(f"Non-finite plotted values in {list(columns)}")


def source_path(config: dict, key: str, default: Path) -> Path:
    value = config.get("figure_inputs", {}).get(key)
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT / path


def read_residue_labels(path: Path, chain: str = "B") -> dict[int, str]:
    """Read observed C-alpha identities from the same mmCIF used for 8CVP."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        lines = iter(handle)
        for line in lines:
            if not line.startswith("_atom_site."):
                continue
            headers = [line.strip().split(".", 1)[1]]
            line = next(lines)
            while line.startswith("_atom_site."):
                headers.append(line.strip().split(".", 1)[1])
                line = next(lines)
            ix = {key: idx for idx, key in enumerate(headers)}
            required = {"group_PDB", "label_atom_id", "auth_asym_id", "auth_seq_id", "label_comp_id"}
            if not required.issubset(ix):
                raise FigureSourceError("CIF atom table does not supply residue identities")
            labels: dict[int, str] = {}
            while not line.startswith("#"):
                fields = shlex.split(line)
                if len(fields) != len(headers):
                    raise FigureSourceError("Unsupported multiline CIF atom row")
                if (fields[ix["group_PDB"]] == "ATOM"
                        and fields[ix["label_atom_id"]] == "CA"
                        and fields[ix["auth_asym_id"]] == chain):
                    residue = int(fields[ix["auth_seq_id"]])
                    name = fields[ix["label_comp_id"]]
                    if name not in AA:
                        raise FigureSourceError(f"Unknown residue identity {name} at {residue}")
                    labels.setdefault(residue, f"{AA[name]}{residue}")
                line = next(lines)
            return labels
    raise FigureSourceError("CIF atom table not found")


def style(stem: str) -> None:
    configure_style(stem)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": None,
        "figure.constrained_layout.use": False,
        "legend.frameon": False,
        "axes.labelsize": 9.3,
        "axes.titlesize": 8.4,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.1,
        "svg.hashsalt": f"crbn-review-{stem}",
    })


def _record(path: Path, base: Path) -> dict:
    try:
        label = path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        # A user-provided external cache can live outside the repository.
        label = path.name
    return {"path": label, "sha256": digest(path), "bytes": path.stat().st_size}


def snapshot(source: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    target = output / source.name
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return target


def export(fig: plt.Figure, stem: str, output: Path, inputs: Sequence[Path],
           sources: Sequence[Path], trace: list[dict], notes: Sequence[str]) -> dict:
    figure_dir = output.parents[1] / "manuscript" / "figures"
    vector_dir = figure_dir / "vector"
    vector_dir.mkdir(parents=True, exist_ok=True)
    products = [figure_dir / f"{stem}.png", vector_dir / f"{stem}.pdf", vector_dir / f"{stem}.svg"]
    fig.savefig(products[0], dpi=300, metadata={"Software": ""}, bbox_inches=None)
    fig.savefig(products[1], metadata={"Creator": None, "Producer": None,
                                     "CreationDate": None, "ModDate": None}, bbox_inches=None)
    fig.savefig(products[2], metadata={"Creator": None, "Date": None}, bbox_inches=None)
    plt.close(fig)
    trace_path = output / f"{stem}_plotted_values.csv"
    pd.DataFrame(trace).to_csv(trace_path, index=False, float_format="%.17g")
    manifest = {
        "figure": stem,
        "builder": "scripts/build_review_figures.py",
        "builder_sha256": digest(Path(__file__)),
        "size_inches": list(fig.get_size_inches()),
        "raster_dpi": 300,
        "inputs": [_record(p, ROOT) for p in inputs],
        "source_snapshots": [_record(p, output) for p in [*sources, trace_path]],
        "outputs": [_record(p, output.parents[1]) for p in products],
        "notes": list(notes),
    }
    manifest_path = output / f"{stem}_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"figure": stem, "manifest": str(manifest_path), "outputs": [str(p) for p in products]}


def model_strip(fig: plt.Figure) -> None:
    """Draw only permitted freedoms; the shapes are explanatory, not structures."""
    description = {
        "isolated": ("Isolated CRBN", "DDB1 absent"),
        "fixed": ("Fixed DDB1", "No DDB1 motion"),
        "rigid": ("Rigid-body DDB1", "Whole-body translation / rotation"),
        "flexible": ("Flexible DDB1", "Body motion + internal deformation"),
    }
    for idx, model in enumerate(MODEL_ORDER):
        ax = fig.add_axes([0.06 + 0.237 * idx, 0.843, 0.213, 0.124])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.5, 1.0, description[model][0], ha="center", va="top", fontsize=7.8, weight="bold")
        ax.add_patch(Ellipse((0.23, 0.54), 0.28, 0.36, angle=15, facecolor="#CBE4F3", edgecolor=BLUE, lw=0.8))
        ax.text(0.23, 0.54, "CRBN", ha="center", va="center", fontsize=7.2)
        ax.annotate("", (0.38, 0.43), (0.38, 0.68), arrowprops={"arrowstyle": "<->", "color": BLUE, "lw": 0.7})
        if model != "isolated":
            ax.plot([0.35, 0.56], [0.56, 0.57], color=MID_GREY, lw=1.1)
            ax.add_patch(Ellipse((0.73, 0.57), 0.34, 0.37, facecolor="#ECEDEC", edgecolor=DARK_GREY, lw=0.8))
            ax.text(0.73, 0.57, "DDB1", ha="center", va="center", fontsize=7.2)
        if model == "fixed":
            ax.plot([0.63, 0.83], [0.31, 0.31], color=DARK_GREY, lw=0.8)
            for x in np.linspace(0.63, 0.83, 5):
                ax.plot([x, x - 0.035], [0.31, 0.25], color=DARK_GREY, lw=0.6)
        if model in {"rigid", "flexible"}:
            ax.annotate("", (0.92, 0.83), (0.56, 0.83), arrowprops={"arrowstyle": "<->", "color": DARK_GREY, "lw": 0.7})
            ax.annotate("", (0.9, 0.40), (0.94, 0.71), arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-0.6", "color": DARK_GREY, "lw": 0.7})
        if model == "flexible":
            ax.plot([0.65, 0.68, 0.73, 0.77, 0.81], [0.39, 0.43, 0.39, 0.43, 0.39], color=GREEN, lw=1.1)
        ax.text(0.5, 0.05, description[model][1], ha="center", va="bottom", fontsize=6.8)
    fig.text(0.52, 0.818, "Same CRBN internal response; whole-CRBN translation and rotation constrained in all four static models",
             ha="center", va="center", fontsize=6.8, color=DARK_GREY)


def fig3(tables: dict[str, Path], output: Path) -> dict:
    style("Fig3")
    models = csv(tables["models"], ["pdb", "model", "S_close", "reference_type"])
    comparisons = csv(tables["comparisons"], ["pdb", "target", "role", "effect", "rotational_percentile"])
    modes = csv(tables["modes"], ["interface_alpha", "tracked_rank", "internal_best_mode", "raw_best_mode"])
    finite(models, ["S_close"])
    if (models.S_close <= 0).any():
        raise FigureSourceError("Relative compliance must be positive before logarithmic plotting")
    refs = [ref for ref in REF_ORDER if ref in set(models.pdb)]
    if len(refs) != 5 or models.duplicated(["pdb", "model"]).any() or len(models) != 20:
        raise FigureSourceError("Figure 3 requires all five references and four unique model conditions")
    fig = plt.figure(figsize=(WIDTH, 7.0))
    model_strip(fig)
    axes = [fig.add_axes(pos) for pos in ((0.09, .49, .365, .263), (.57, .49, .365, .263),
                                        (.09, .084, .365, .263), (.57, .084, .365, .263))]
    trace: list[dict] = []
    ax = axes[0]
    for ref in refs:
        sub = models.loc[models.pdb == ref].set_index("model").loc[list(MODEL_ORDER)]
        values = np.log10(sub.S_close.to_numpy(float))
        ax.plot(range(4), values, marker="o" if sub.reference_type.iloc[0] == "apo" else "s", color=REF_COLORS[ref], label=ref)
        trace.extend({"panel": "a", "identity": f"{ref}:{model}", "x": idx, "y": value,
                      "source_table": tables["models"].name, "source_column": "S_close", "transform": "log10"}
                     for idx, (model, value) in enumerate(zip(MODEL_ORDER, values)))
    ax.set_xticks(range(4), [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel(r"Relative compliance, $\log_{10} S$")
    ax.set_title("Closure response relative to mean response")
    finish_axis(ax, grid="y"); panel_label(ax, "a")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.22), handlelength=1.0, fontsize=7.6)

    ax = axes[1]
    vals = comparisons.query("target == 'finite'").set_index(["pdb", "role"])
    body = [float(vals.loc[(p, "R_body"), "effect"]) for p in refs]
    internal = [float(vals.loc[(p, "R_internal"), "effect"]) for p in refs]
    total = [float(vals.loc[(p, "M"), "effect"]) for p in refs]
    for idx, (ref, rb, ri, m) in enumerate(zip(refs, body, internal, total)):
        ax.bar(idx, rb, width=.62, color=BLUE)
        ax.bar(idx, abs(ri), width=.62, bottom=rb if ri >= 0 else rb + ri,
               color=GREEN, hatch="///" if ri < 0 else "", edgecolor=BLACK if ri < 0 else GREEN, linewidth=.3)
        ax.plot([idx-.31, idx+.31], [rb+ri]*2, color=BLACK, lw=.55)
        ax.scatter(idx, m, color=BLACK, marker="D", s=18, zorder=4)
        for role, value in (("R_body", rb), ("R_internal", ri), ("M", m)):
            trace.append({"panel": "b", "identity": f"{ref}:{role}", "x": idx, "y": value,
                          "source_table": tables["comparisons"].name, "source_column": "effect", "transform": "identity"})
    ax.set_xticks(range(5), refs); ax.set_ylabel(r"Relative compliance change, $\Delta\ln S$")
    ax.set_title("DDB1 body and internal contributions")
    ax.set_ylim(min(total)-.065, max(np.array(body)+np.array(internal))+.12)
    finish_axis(ax, grid="y", zero_line=True); panel_label(ax, "b")
    ax.legend(handles=[Patch(color=BLUE, label="Body"), Patch(color=GREEN, label="Internal"),
                       Patch(facecolor=GREEN, hatch="///", label="Negative internal"),
                       Line2D([], [], ls="none", marker="D", color=BLACK, label="Flexible − isolated")],
              loc="upper left", fontsize=6.8, handlelength=1.0)

    ax = axes[2]
    for idx, role in enumerate(("R_body", "R_internal", "M")):
        sub = comparisons[(comparisons.target == "tangent") & (comparisons.role == role)].set_index("pdb")
        for offset, ref in zip(np.linspace(-.17,.17,5), refs):
            row = sub.loc[ref]
            ax.scatter(idx+offset, row.rotational_percentile, color=REF_COLORS[ref],
                       marker="o" if row.reference_type == "apo" else "s", edgecolor=BLACK, linewidth=.3, s=22)
            trace.append({"panel": "c", "identity": f"{ref}:{role}:tangent", "x": idx+offset,
                          "y": float(row.rotational_percentile), "source_table": tables["comparisons"].name,
                          "source_column": "rotational_percentile", "transform": "identity"})
    ax.axhline(95, color=ORANGE, ls="--", lw=.8, label="95th percentile")
    ax.set_xticks(range(3), ["Body", "Internal", "Flexible − isolated"])
    ax.set_ylim(-3,103); ax.set_ylabel("Percentile among local rotations")
    ax.set_title("Directional comparison with local rotations")
    finish_axis(ax, grid="y"); panel_label(ax, "c"); ax.legend(loc="lower right")

    ax = axes[3]
    alpha = modes.interface_alpha.to_numpy(float)
    for column, label, color, dash in (
        ("tracked_rank", "Tracked branch rank", BLUE, "-"),
        ("internal_best_mode", "Best internal rank", ORANGE, "--"),
        ("raw_best_mode", "Best raw rank", "#CC79A7", ":"),
    ):
        ax.step(alpha, modes[column], where="mid", color=color, ls=dash, label=label)
        trace.extend({"panel": "d", "identity": column, "x": x, "y": y,
                      "source_table": tables["modes"].name, "source_column": column, "transform": "identity"}
                     for x,y in zip(alpha,modes[column]))
    ax.set_ylim(6.6,.4); ax.set_yticks(range(1,7)); ax.set_xlabel(r"Interface spring strength, $\alpha$")
    ax.set_ylabel("Complex mode rank"); finish_axis(ax, grid="both")
    twin = ax.twinx()
    for column, label, color, dash in (("best_crbn_internal_overlap", "Internal overlap", BLACK, "-"),
                                       ("cluster_projection", "Cluster projection", GREEN, ":")):
        twin.plot(alpha, modes[column], color=color, ls=dash, lw=.9, label=label)
        trace.extend({"panel": "d", "identity": column, "x": x, "y": y,
                      "source_table": tables["modes"].name, "source_column": column, "transform": "identity"}
                     for x,y in zip(alpha,modes[column]))
    twin.set_ylim(0,1.02); twin.set_ylabel("Directional overlap")
    lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines+lines2, labels+labels2, loc="center left", bbox_to_anchor=(.34,.46), fontsize=6.8)
    ax.set_title("8CVP mode ordering as coupling changes"); panel_label(ax, "d")
    used = [tables[k] for k in ("models","comparisons","modes")]
    snaps = [snapshot(p,output) for p in used]
    return export(fig, "Fig3", output, used, snaps, trace, [
        "Frozen numeric data retained: 15 Å, uniform springs, five open references.",
        "The unlettered strip shows allowed freedoms, not atomic coordinates or a predicted motion trajectory.",
        "CRBN whole-body translation and rotation are constrained before inversion in every static model.",
        "Stack heights and diamonds are log S differences, not fractions of energy or actual motion.",
    ])


def contact_trace(data: pd.DataFrame, labels: dict[int, str], table_name: str) -> list[dict]:
    required = ["group_id", "residue", "flexible_D_g", "flexible_derivative_log_C_close",
                "flexible_derivative_log_mean_compliance", "delta_R_body_derivative_log_S_close",
                "delta_R_internal_derivative_log_S_close"]
    finite(data, required[2:])
    if data.group_id.duplicated().any():
        raise FigureSourceError("Contact group identities must be unique")
    trace = []
    for _, row in data.iterrows():
        residue = int(row.residue)
        if residue not in labels:
            raise FigureSourceError(f"No observed residue identity for {residue}")
        for panel, xkey, ykey in (("b", required[3], required[4]), ("c", required[5], required[6])):
            trace.append({"panel": panel, "identity": row.group_id, "label": labels[residue],
                          "x": float(row[xkey]), "y": float(row[ykey]), "x_source_column": xkey,
                          "y_source_column": ykey, "source_table": table_name, "transform": "identity"})
    return trace


def _contact_points(ax, data: pd.DataFrame, xcol: str, ycol: str) -> None:
    for _, row in data.iterrows():
        ax.scatter(row[xcol], row[ycol], s=24, marker="o" if row.stable_apo_model_candidate else "s",
                   facecolor=CONTACT_COLORS[row.contact_class], edgecolor=BLACK, linewidth=.45, zorder=5)


def _labels(ax, data: pd.DataFrame, labels: dict[int,str], xcol: str, ycol: str,
            offsets: dict[int, tuple[float,float]], fontsize: float=7.6) -> None:
    for _, row in data.iterrows():
        dx,dy = offsets[int(row.residue)]
        ax.annotate(labels[int(row.residue)], (row[xcol],row[ycol]), xytext=(dx,dy),
                    textcoords="offset points", ha="left" if dx >= 0 else "right", va="center",
                    color=BLACK, fontsize=fontsize,
                    arrowprops={"arrowstyle":"-", "lw":.45, "color":MID_GREY, "shrinkA":1, "shrinkB":3}, zorder=8)


def fig4(tables: dict[str,Path], cif_path: Path, output: Path) -> dict:
    style("Fig4")
    data = csv(tables["contacts"], ["group_id", "residue", "contact_class", "condition_results"])
    labels = read_residue_labels(cif_path)
    trace = contact_trace(data,labels,tables["contacts"].name)
    if len(data) != 13 or not {221,222,339}.issubset(data.residue):
        raise FigureSourceError("Figure 4 must retain the 13 frozen display groups, including Y221, K222 and S339")
    fig = plt.figure(figsize=(WIDTH,7.0))
    fig.text(.54,.965,"8CVP · 15 Å · uniform springs", ha="center", fontsize=8.0, weight="bold")
    fig.text(.54,.943,"Cα network springs; the same 13 groups are identified in panels a–d", ha="center", fontsize=7.2, color=DARK_GREY)
    axa = fig.add_axes((.16,.56,.32,.31)); axb = fig.add_axes((.62,.56,.34,.31))
    axc = fig.add_axes((.12,.12,.36,.31)); axd = fig.add_axes((.68,.12,.28,.31))
    rows = data.sort_values("flexible_D_g")
    bars = axa.barh(range(len(rows)), rows.flexible_D_g,
                     color=[CONTACT_COLORS[x] for x in rows.contact_class], edgecolor=BLACK, linewidth=.25)
    for bar, (_, row) in zip(bars,rows.iterrows()):
        if row.stable_apo_model_candidate: bar.set_hatch("///")
    class_labels = {"CRBN_DDB1":"DDB1", "HB_TBD":"HB–TBD"}
    axa.set_yticks(range(len(rows)), [f"{labels[int(r.residue)]} {class_labels[r.contact_class]}" for _,r in rows.iterrows()])
    axa.set_xlabel(r"Relative closure response, $D_g$")
    axa.set_title("Effects of spring strengthening")
    axa.axvline(0,color=MID_GREY,lw=.6); finish_axis(axa,grid="x"); panel_label(axa,"a",x=-.22)
    axa.legend(handles=[Patch(color=BLUE,label="CRBN–DDB1"),Patch(color=GREEN,label="HB–TBD"),
                         Patch(facecolor="white",edgecolor=BLACK,hatch="///",label="Original stable group")],
               loc="upper left",bbox_to_anchor=(-.32,-.16),ncol=2,fontsize=6.8)
    trace.extend({"panel":"a","identity":r.group_id,"label":labels[int(r.residue)],
                  "x":float(r.flexible_D_g),"y":idx,"source_table":tables["contacts"].name,
                  "x_source_column":"flexible_D_g","transform":"identity"} for idx,(_,r) in enumerate(rows.iterrows()))

    xc,yc = "flexible_derivative_log_C_close","flexible_derivative_log_mean_compliance"
    _contact_points(axb,data,xc,yc)
    axb.plot([-.062,.008],[-.062,.008],ls=":",color=MID_GREY,lw=.8)
    axb.set_xlim(-.062,.008); axb.set_ylim(-.058,.008)
    axb.set_xlabel(r"Closure compliance: $\partial\ln C_{close}$")
    axb.set_ylabel(r"Mean compliance: $\partial\ln\overline{C}$")
    axb.set_title("Closure versus mean response (flexible)")
    finish_axis(axb,grid="both"); panel_label(axb,"b",x=-.20)
    zoom_b = {261,263,264,289}
    _labels(axb,data[~data.residue.isin(zoom_b)],labels,xc,yc,
            {339:(3,-13),420:(-3,14),262:(-3,-16),422:(5,17),341:(13,-2),287:(3,-29),186:(-6,18),221:(12,5),222:(12,-11)})
    inset_b = axb.inset_axes([.40,.09,.57,.45])
    cluster = data[data.residue.isin(zoom_b)]
    _contact_points(inset_b,cluster,xc,yc)
    inset_b.set_xlim(-.032,-.0260); inset_b.set_ylim(-.0102,-.0061)
    _labels(inset_b,cluster,labels,xc,yc,{289:(-13,-12),263:(6,3),261:(-18,14),264:(5,11)},fontsize=7.6)
    inset_b.tick_params(labelsize=6.8,length=2); inset_b.set_xticks([-.032,-.029,-.026])
    inset_b.set_yticks([-.010,-.008,-.006]); inset_b.set_title("HB–TBD cluster",fontsize=6.8,pad=3)
    finish_axis(inset_b,grid="both")
    axb.add_patch(Rectangle((-.032,-.0102),.006,.0041,fill=False,ec=MID_GREY,lw=.7))

    xr,yr = "delta_R_body_derivative_log_S_close","delta_R_internal_derivative_log_S_close"
    _contact_points(axc,data,xr,yr)
    axc.set_xlim(-.054,.004); axc.set_ylim(-.0034,.0108)
    axc.set_xlabel(r"Body contribution: $\partial R_{body}$")
    axc.set_ylabel(r"Internal contribution: $\partial R_{internal}$")
    axc.set_title("Effects on DDB1 response components")
    finish_axis(axc,grid="both",zero_line=True); axc.axvline(0,color=MID_GREY,lw=.6)
    panel_label(axc,"c",x=-.16)
    zoom_c = set(data.residue).difference({339,341,420})
    _labels(axc,data[~data.residue.isin(zoom_c)],labels,xr,yr,{339:(7,5),341:(-22,-8),420:(-5,-13)})
    inset_c = axc.inset_axes([.39,.42,.59,.53])
    cluster = data[data.residue.isin(zoom_c)]
    _contact_points(inset_c,cluster,xr,yr)
    inset_c.set_xlim(-.007,.004); inset_c.set_ylim(-.0022,.0024)
    _labels(inset_c,cluster,labels,xr,yr,{287:(-8,-14),422:(-10,-14),289:(2,-12),264:(-23,-2),
                                             221:(26,9),222:(20,-9),186:(-16,10),262:(-16,20),261:(3,12),263:(8,14)},fontsize=7.6)
    inset_c.set_xticks([-.006,-.003,0]); inset_c.set_yticks([-.001,0,.001])
    inset_c.tick_params(labelsize=6.8,length=2); inset_c.set_title("Groups near zero",fontsize=6.8,pad=3)
    inset_c.axhline(0,color=MID_GREY,lw=.45); inset_c.axvline(0,color=MID_GREY,lw=.45)
    finish_axis(inset_c,grid="both")
    axc.add_patch(Rectangle((-.0066,-.00185),.0094,.0037,fill=False,ec=MID_GREY,lw=.7))

    matrix,conditions = condition_matrix(data)
    cmap = matplotlib.colors.ListedColormap(["#EEEEEE","#EBC0AF","#A4D2B6"])
    norm = matplotlib.colors.BoundaryNorm([-1.5,-.5,.5,1.5],3)
    axd.imshow(matrix,aspect="auto",cmap=cmap,norm=norm)
    axd.set_yticks(range(len(data)), [f"{labels[int(r.residue)]} {class_labels[r.contact_class]}" for _,r in data.iterrows()])
    axd.set_xticks(range(len(conditions)),conditions,rotation=45,ha="right",rotation_mode="anchor")
    axd.tick_params(length=0,labelsize=7.2); axd.set_title("Original consistency criteria")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isfinite(matrix[i,j]):
                raise FigureSourceError("Unclassified robustness result in frozen display groups")
            token = {1:"P",0:"F",-1:"A"}[int(matrix[i,j])]
            axd.text(j,i,token,ha="center",va="center",fontsize=7.6)
            trace.append({"panel":"d","identity":data.iloc[i].group_id,"label":labels[int(data.iloc[i].residue)],
                          "x":j,"y":i,"condition":conditions[j],"value":int(matrix[i,j]),"display":token,
                          "source_table":tables["contacts"].name,"source_column":"condition_results"})
    panel_label(axd,"d",x=-.30)
    fig.text(.54,.032,"P: same sign and top 20%; F: criterion not met; A: absent. Circles: original stable groups; squares: other displayed groups.",
             ha="center",va="center",fontsize=6.8,color=DARK_GREY)
    label_path=output/"Fig4_residue_identities.csv"
    pd.DataFrame([{"pdb":"8CVP","chain":"B","residue":int(r.residue),"group_id":r.group_id,
                   "residue_label":labels[int(r.residue)],"cif_sha256":digest(cif_path)} for _,r in data.iterrows()]).to_csv(label_path,index=False)
    used=[tables[k] for k in ("contacts","all_contacts","robustness")]
    snaps=[snapshot(p,output) for p in used]
    return export(fig,"Fig4",output,[*used,cif_path],[*snaps,label_path],trace,[
        "All 13 groups, all point coordinates and all original robustness cells are retained.",
        "Panels a–c: 8CVP, 15 Å, uniform springs. Panels a–b use flexible DDB1.",
        "Full residue identities are read from the same 8CVP atom table; label offsets never alter point coordinates.",
        "Insets repeat clustered points to make identities legible; no extra observations are added.",
        "Network spring counts are C-alpha distance links and are not a count of chemical bonds or direct heavy-atom contacts.",
    ])


FIG3_LEGEND = r"""Fig. 3. DDB1 mobility changes the predicted CRBN closure response and complex mode ordering. All panels retain the original 269-position CRBN representation; the observed-residue expansion is evaluated separately in Fig. S8. The upper diagrams define the four static conditions: DDB1 is absent, fixed, allowed whole-body translation and rotation, or also allowed internal deformation. CRBN whole-body translation and rotation are constrained before inversion in all four conditions; the measured response is internal CRBN deformation. Panels a–c use five open references at 15 Å with uniform springs; circles denote ligand-free references and squares engineered references. (a) Relative closure compliance, $S=C_{\mathrm{close}}/\overline C$, compares the response along the frozen closure direction with the mean response over 801 CRBN internal directions. Values are shown as $\log_{10}S$. (b) Natural-log changes for finite closure separate $R_{\mathrm{body}}=\ln S_{\mathrm{rigid}}-\ln S_{\mathrm{fixed}}$ and $R_{\mathrm{internal}}=\ln S_{\mathrm{flexible}}-\ln S_{\mathrm{rigid}}$. Hatched internal segments are negative; diamonds show $M=\ln S_{\mathrm{flexible}}-\ln S_{\mathrm{isolated}}$. These terms measure changes in relative compliance, not motion amplitude or energy. (c) Tangent-direction responses are located within the prespecified distribution of 5,000 local rotations; the dashed line marks its 95th percentile. (d) At 8CVP, 15 Å and uniform springs, mode ranks and directional overlaps are shown across interface strength $\alpha$. The tracked branch starts from the isolated CRBN lowest internal mode. The internal match removes whole-CRBN translation and rotation before comparing directions; the raw CRBN-vector match retains them. At zero interface strength the disconnected complex contains the modes of both proteins, so the isolated CRBN lowest internal mode need not be first in the combined ordering."""

FIG4_LEGEND = r"""Fig. 4. Residue–contact groups affect closure compliance, mean compliance and the response to DDB1 mobility differently. All panels retain the original 269-position CRBN network and its original spring groups. Panels a–c use 8CVP, a 15 Å Cα cutoff and uniform springs. The same 13 previously displayed groups are retained: eight groups meeting the original ligand-free consistency criteria and five additional high-effect groups. Labels give the observed amino-acid identity and residue number; CRBN–DDB1 and HB–TBD denote spring classes. These network links are distinct from direct heavy-atom contacts. (a) Under flexible DDB1, $D_g=[\ln S(1.1)-\ln S(0.9)]/0.2$ ranks the signed effect of strengthening each group; positive values indicate increased relative closure compliance. Hatched bars identify the original stable groups. (b) Under flexible DDB1, exact derivatives of log closure compliance and log mean compliance separate the two contributions to relative compliance. Both axes differentiate with respect to log group spring strength. The diagonal marks equal changes; its horizontal displacement equals the derivative of $\ln S$. The inset identifies overlapping HB–TBD points. (c) Exact derivatives of $R_{\mathrm{body}}$ and $R_{\mathrm{internal}}$ show how a group changes the response to DDB1 body motion and internal deformation. The inset identifies groups close to zero; it repeats those observations without changing their values. In b and c, circles denote original stable groups and squares the other displayed groups. (d) Original uniform-spring consistency results remain separate from the expanded-window analysis in Fig. S8. P indicates an observed group retaining the discovery sign and a top-20% rank; F indicates that the criterion was not met; A indicates absence. Shared spring groups are listed in the source data."""


WINDOW_FILES = ("coverage.csv", "models.csv", "comparisons.csv", "group_effects.csv",
                "robustness.csv", "summary.json")
WINDOW_POLICIES = ("original_edges", "expanded_incident_edges")
COVERAGE_COLORS = {"core": BLUE, "observed_added": GREEN, "unobserved": "#C7C7C7",
                   "outside_construct": "#F5F5F5"}


def boolean_column(data: pd.DataFrame, column: str) -> pd.Series:
    converted = data[column].astype(str).str.lower().map({"true": True, "false": False})
    if converted.isna().any():
        raise FigureSourceError(f"Unclassified boolean source values in {column}")
    return converted.astype(bool)


def validated_window_sources(directory: Path, config: dict) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    """Require the completed 30-condition analysis, not partial pilot tables."""
    paths = [directory / name for name in WINDOW_FILES]
    for path in paths:
        if not path.is_file():
            raise FigureSourceError(f"Missing completed observed-window figure input: {path}")
    summary = json.loads(paths[-1].read_text())
    if (summary.get("status") != "complete" or summary.get("condition_count") != 30
            or summary.get("all_verification_pass") is not True):
        raise FigureSourceError("Observed-window analysis has not passed the complete 30-condition gate")
    required = {
        "coverage": ["pdb", "reference_type", "residue", "domain", "status"],
        "models": ["pdb", "cutoff_A", "weighting", "window", "model", "n_core", "n_added",
                   "C_close", "mean_compliance", "S_close"],
        "comparisons": ["pdb", "cutoff_A", "weighting", "window", "role", "target", "effect"],
        "group_effects": ["pdb", "cutoff_A", "weighting", "window", "policy", "group_id",
                          "residue", "contact_class", "status", "flexible_D_g"],
        "robustness": ["group_id", "residue", "contact_class", "policy", "legacy_stable_apo",
                       "legacy_engineered", "expanded_apo_stable", "expanded_engineered_consistent",
                       "expanded_all_weighted_stable", "condition_results"],
    }
    tables = {path.stem: csv(path, required[path.stem]) for path in paths[:-1]}
    refs = config.get("references", list(REF_ORDER))
    expected_conditions = {(p, float(c), w) for p in refs
                           for c in config.get("cutoffs_A", [13, 15, 18])
                           for w in config.get("weightings", ["uniform", "inverse_square"])}
    if len(expected_conditions) != 30 or set(refs) != set(REF_ORDER):
        raise FigureSourceError("The figure requires the frozen five-reference, 30-condition comparison")
    models = tables["models"]
    model_keys = ["pdb", "cutoff_A", "weighting", "window", "model"]
    expected_models = {(*key, window, model) for key in expected_conditions
                       for window in ("original", "expanded") for model in MODEL_ORDER}
    if models.duplicated(model_keys).any() or set(models[model_keys].itertuples(index=False, name=None)) != expected_models:
        raise FigureSourceError("Observed-window models do not contain every matched model/window condition")
    finite(models, ["C_close", "mean_compliance", "S_close"])
    if (models.n_core != 269).any() or (models[["C_close", "mean_compliance", "S_close"]] <= 0).any().any():
        raise FigureSourceError("Observed-window response does not retain a positive fixed-269 observable")
    if not np.allclose(models.C_close / models.mean_compliance, models.S_close, rtol=1e-9, atol=1e-11):
        raise FigureSourceError("Supplied relative compliance differs from its numerator and denominator")
    coverage = tables["coverage"]
    if set(coverage.pdb) != set(refs) or not set(coverage.status).issubset(COVERAGE_COLORS):
        raise FigureSourceError("Incomplete or unclassified residue coverage")
    for pdb, data in coverage.groupby("pdb"):
        if sorted(data.residue.tolist()) != list(range(1, 443)) or (data.status == "core").sum() != 269:
            raise FigureSourceError(f"Coverage does not resolve all canonical positions and the fixed core: {pdb}")
        count = (data.status == "observed_added").sum()
        if (models.loc[(models.pdb == pdb) & (models.window == "expanded"), "n_added"] != count).any():
            raise FigureSourceError(f"Observed-residue counts disagree with the expanded model: {pdb}")
    comparisons = tables["comparisons"]
    comparison_keys = ["pdb", "cutoff_A", "weighting", "window", "role", "target"]
    expected_comparisons = {(*key, window, role, target) for key in expected_conditions
                            for window in ("original", "expanded")
                            for role in ("R_body", "R_internal", "R_total", "M")
                            for target in ("finite", "tangent")}
    if (comparisons.duplicated(comparison_keys).any()
            or set(comparisons[comparison_keys].itertuples(index=False, name=None)) != expected_comparisons):
        raise FigureSourceError("Observed-window comparison terms are incomplete")
    finite(comparisons, ["effect"])
    condition_window = ["pdb", "cutoff_A", "weighting", "window"]
    model_values = models.pivot(index=condition_window, columns="model", values="S_close")
    differences = {"R_body": ("rigid", "fixed"), "R_internal": ("flexible", "rigid"),
                   "R_total": ("flexible", "fixed"), "M": ("flexible", "isolated")}
    for role, (first, second) in differences.items():
        supplied = comparisons[(comparisons.role == role) & (comparisons.target == "finite")].set_index(condition_window)
        calculated = np.log(model_values[first] / model_values[second]).reindex(supplied.index)
        if not np.allclose(supplied.effect, calculated, rtol=1e-8, atol=1e-10):
            raise FigureSourceError(f"The supplied {role} terms disagree with the matched model responses")
    robustness = tables["robustness"]
    if robustness.duplicated(["group_id", "policy"]).any() or set(robustness.policy) != set(WINDOW_POLICIES):
        raise FigureSourceError("Observed-window contact policies are missing or duplicated")
    candidate_count = config.get("contact", {}).get("candidate_count", 142)
    if any(group.group_id.nunique() != candidate_count for _, group in robustness.groupby("policy")):
        raise FigureSourceError("Observed-window stability does not retain the frozen candidate universe")
    candidate_sets = [set(group.group_id) for _, group in robustness.groupby("policy")]
    if candidate_sets[0] != candidate_sets[1]:
        raise FigureSourceError("Perturbation policies use different candidate identities")
    for column in ("legacy_stable_apo", "legacy_engineered", "expanded_apo_stable",
                   "expanded_engineered_consistent", "expanded_all_weighted_stable"):
        robustness[column] = boolean_column(robustness, column)
    if not robustness.groupby("group_id")[["legacy_stable_apo", "legacy_engineered"]].nunique().eq(1).all().all():
        raise FigureSourceError("Legacy flags differ between perturbation policies")
    effects = tables["group_effects"]
    effect_keys = ["pdb", "cutoff_A", "weighting", "window", "policy"]
    expected_effects = {(*key, window, policy) for key in expected_conditions
                        for window, policy in (("original", "original_edges"),
                                               ("expanded", "original_edges"),
                                               ("expanded", "expanded_incident_edges"))}
    if (effects.duplicated([*effect_keys, "group_id"]).any()
            or set(effects[effect_keys].itertuples(index=False, name=None)) != expected_effects
            or any(set(group.group_id) != candidate_sets[0]
                   for _, group in effects.groupby(effect_keys))):
        raise FigureSourceError("Contact effects do not contain all fixed candidates and window/policy conditions")
    return tables, paths


def primary(data: pd.DataFrame) -> pd.DataFrame:
    return data[(data.cutoff_A == 15) & (data.weighting == "uniform")].copy()


def fig_s8(directory: Path, config: dict, cif_path: Path, output: Path) -> dict:
    tables, paths = validated_window_sources(directory, config)
    style("FigS8")
    fig = plt.figure(figsize=(WIDTH, 7.0))
    axa = fig.add_axes([0.13, 0.805, 0.83, 0.105])
    axb = fig.add_axes([0.12, 0.515, 0.35, 0.17])
    axc = fig.add_axes([0.61, 0.515, 0.35, 0.17])
    axd = fig.add_axes([0.12, 0.185, 0.35, 0.19])
    axe = fig.add_axes([0.70, 0.185, 0.26, 0.19])
    trace = []
    fig.text(0.53, 0.985, "Testing the effect of additional observed CRBN residues", ha="center", va="top", fontsize=8.7, weight="bold")
    coverage = tables["coverage"]
    codes = {key: index for index, key in enumerate(COVERAGE_COLORS)}
    matrix = []
    coverage_labels = []
    for i, pdb in enumerate(REF_ORDER):
        rows = coverage[coverage.pdb == pdb].sort_values("residue")
        matrix.append(rows.status.map(codes).to_numpy())
        added = int((rows.status == "observed_added").sum())
        coverage_labels.append(f"{pdb}  +{added}")
        for _, row in rows.iterrows():
            trace.append({"panel": "a", "identity": f"{pdb}:{row.residue}", "label": row.status,
                          "x": row.residue, "y": i, "value": codes[row.status],
                          "source_table": "coverage.csv", "source_column": "status"})
    axa.imshow(matrix, origin="upper", aspect="auto", interpolation="nearest",
               cmap=matplotlib.colors.ListedColormap(list(COVERAGE_COLORS.values())),
               vmin=0, vmax=len(codes)-1, extent=[0.5, 442.5, 4.5, -0.5])
    axa.set(yticks=range(5), yticklabels=coverage_labels, xticks=[1, 100, 187, 250, 318, 400, 442],
            xlabel="Canonical human CRBN residue")
    fig.text(0.545, 0.951, "Observed positions added to the common core", ha="center", va="center",
             fontsize=8.4, weight="bold")
    axa.tick_params(length=2, labelsize=7.6)
    for boundary in (186.5, 317.5):
        axa.axvline(boundary, color=BLACK, lw=0.7)
    for x, label in ((94, "NTD"), (252, "HB"), (380, "TBD")):
        axa.text(x, -0.63, label, ha="center", va="bottom", fontsize=7.1)
    panel_label(axa, "a", x=-0.125, y=1.3)
    coverage_names = {"core": "Common core (269)", "observed_added": "Observed, added",
                      "unobserved": "Unobserved", "outside_construct": "Outside construct"}
    fig.legend(handles=[Patch(fc=color, ec=MID_GREY, lw=0.4, label=coverage_names[key])
                        for key, color in COVERAGE_COLORS.items()],
               loc="center", bbox_to_anchor=(0.55, 0.746), ncol=4, fontsize=7.1, columnspacing=1.1)
    models = primary(tables["models"])
    for pdb in REF_ORDER:
        local = models[models.pdb == pdb].pivot(index="model", columns="window", values="S_close").loc[list(MODEL_ORDER)]
        effect = np.log(local.expanded / local.original)
        axb.plot(range(4), effect, color=REF_COLORS[pdb], marker="o" if pdb.startswith("8") else "s", ms=3.6, lw=0.7, label=pdb)
        for index, model in enumerate(MODEL_ORDER):
            trace.append({"panel": "b", "identity": f"{pdb}:{model}", "label": pdb, "x": index,
                          "y": effect.iloc[index], "original_S": local.original.iloc[index],
                          "expanded_S": local.expanded.iloc[index], "source_table": "models.csv",
                          "source_column": "S_close", "transform": "ln(expanded/original)"})
    axb.set(xticks=range(4), xticklabels=[MODEL_LABELS[m] for m in MODEL_ORDER],
            ylabel=r"$\ln(S_{\mathrm{expanded}}/S_{\mathrm{original}})$", title="Effect of added residues on relative compliance")
    axb.tick_params(axis="x", labelsize=7.1)
    finish_axis(axb, grid="y", zero_line=True); panel_label(axb, "b", x=-0.23)
    fig.legend(*axb.get_legend_handles_labels(), loc="center", bbox_to_anchor=(0.298, 0.725),
               ncol=5, fontsize=7.0, columnspacing=0.7, handlelength=0.9, handletextpad=0.35)
    comparisons = primary(tables["comparisons"])
    comparisons = comparisons[comparisons.target == "finite"]
    role_specs = [("R_body", BLUE, "o", -0.21), ("R_internal", GREEN, "s", 0), ("M", BLACK, "D", 0.21)]
    for i, pdb in enumerate(REF_ORDER):
        for role, color, marker, offset in role_specs:
            local = comparisons[(comparisons.pdb == pdb) & (comparisons.role == role)].set_index("window")
            xs = [i + offset - 0.035, i + offset + 0.035]
            ys = local.loc[["original", "expanded"], "effect"].to_numpy()
            axc.plot(xs, ys, color=color, lw=0.7)
            for x, y, window in zip(xs, ys, ["original", "expanded"]):
                axc.scatter(x, y, marker=marker, s=17, ec=color,
                            fc="white" if window == "original" else color, lw=0.7, zorder=3)
                trace.append({"panel": "c", "identity": f"{pdb}:{role}:{window}", "label": role,
                              "x": x, "y": y, "window": window, "source_table": "comparisons.csv",
                              "source_column": "effect", "target": "finite"})
    axc.set(xticks=range(5), xticklabels=list(REF_ORDER), ylabel="Natural-log change in S",
            title="Body and internal relaxation terms")
    axc.tick_params(axis="x", labelsize=7.2)
    finish_axis(axc, grid="y", zero_line=True); panel_label(axc, "c", x=-0.24)
    axc.legend(handles=[Line2D([], [], color=color, marker=marker, ms=3, lw=0,
                               label={"R_body": "Body", "R_internal": "Internal", "M": "M"}[role])
                        for role, color, marker, _ in role_specs],
               loc="upper center", bbox_to_anchor=(0.51, 1.30), ncol=3, fontsize=7.0, columnspacing=1.0)
    axc.text(0.5, -0.32, "Open symbols: original; filled: expanded", transform=axc.transAxes,
             ha="center", fontsize=7.1, color=DARK_GREY)
    robustness = tables["robustness"]
    legacy = robustness[robustness.policy == "original_edges"].set_index("group_id")
    displayed = legacy[legacy.legacy_stable_apo].sort_values(["residue", "contact_class"])
    if len(displayed) != 8:
        raise FigureSourceError("The fixed original candidate display no longer contains eight groups")
    effects = primary(tables["group_effects"])
    effects = effects[(effects.pdb == "8CVP") & (effects.policy == "original_edges")]
    if effects.duplicated(["group_id", "window"]).any():
        raise FigureSourceError("Duplicated contact effects for the window comparison")
    selected = effects.pivot(index="group_id", columns="window", values="flexible_D_g")
    if set(selected.index) != set(legacy.index):
        raise FigureSourceError("The contact effect display changes the frozen candidate universe")
    absent = effects[effects.status.isin(["absent", "missing", "unevaluable"])].group_id.unique()
    available = selected.drop(index=absent)
    finite(available, ["original", "expanded"])
    for contact_class, color in CONTACT_COLORS.items():
        ids = legacy[legacy.contact_class == contact_class].index.intersection(available.index)
        if not len(ids):
            continue
        group = available.loc[ids]
        axd.scatter(group.original, group.expanded, c=color, s=13, alpha=0.7, ec="none",
                    label=contact_class.replace("_", "–"))
    stable_available = available.loc[available.index.intersection(displayed.index)]
    axd.scatter(stable_available.original, stable_available.expanded, fc="none", ec=BLACK, s=33, lw=0.65)
    limits = np.array([available[["original", "expanded"]].min().min(), available[["original", "expanded"]].max().max()])
    pad = max((limits[1] - limits[0]) * 0.06, 1e-5)
    limits += np.array([-pad, pad])
    axd.plot(limits, limits, color=MID_GREY, lw=0.7, ls="--", zorder=0)
    axd.set(xlim=limits, ylim=limits, xlabel=r"Original network $D_g$", ylabel=r"Expanded network $D_g$",
            title="Same original spring groups")
    finish_axis(axd, grid="both"); panel_label(axd, "d", x=-0.23)
    axd.legend(loc="upper center", bbox_to_anchor=(0.5, -0.29), ncol=2, fontsize=7.0)
    axd.text(0.5, 0.97, f"8CVP: {len(available)} evaluated; {len(absent)} absent", transform=axd.transAxes,
             ha="center", va="top", fontsize=7.0, color=DARK_GREY)
    for group_id, row in available.iterrows():
        trace.append({"panel": "d", "identity": group_id, "label": group_id, "x": row.original,
                      "y": row.expanded, "source_table": "group_effects.csv", "source_column": "flexible_D_g",
                      "policy": "original_edges", "pdb": "8CVP", "cutoff_A": 15, "weighting": "uniform"})
    for group_id in absent:
        trace.append({"panel": "d", "identity": group_id, "label": group_id, "status": "absent",
                      "source_table": "group_effects.csv", "source_column": "status",
                      "policy": "original_edges", "pdb": "8CVP", "cutoff_A": 15, "weighting": "uniform"})
    labels = read_residue_labels(cif_path)
    choices = [("original_edges", "legacy_stable_apo", "Legacy\napo"),
               ("original_edges", "legacy_engineered", "Legacy\neng."),
               ("original_edges", "expanded_apo_stable", "Original\nedges\napo"),
               ("original_edges", "expanded_engineered_consistent", "Original\nedges\neng."),
               ("expanded_incident_edges", "expanded_apo_stable", "All\nincident\napo"),
               ("expanded_incident_edges", "expanded_engineered_consistent", "All\nincident\neng.")]
    stability = []
    indexed = robustness.set_index(["group_id", "policy"])
    for i, (group_id, row) in enumerate(displayed.iterrows()):
        cells = []
        for j, (policy, column, _label) in enumerate(choices):
            value = int(indexed.loc[(group_id, policy), column])
            cells.append(value)
            trace.append({"panel": "e", "identity": group_id, "label": labels[int(row.residue)],
                          "x": j, "y": i, "value": value, "display": "P" if value else "F",
                          "source_table": "robustness.csv", "source_column": column, "policy": policy})
        stability.append(cells)
    axe.imshow(stability, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#EBC0AF", "#A4D2B6"]), vmin=0, vmax=1)
    axe.set(yticks=range(len(displayed)),
            yticklabels=[f"{labels[int(r.residue)]} {'DDB1' if r.contact_class == 'CRBN_DDB1' else 'HB–TBD'}" for _, r in displayed.iterrows()],
            xticks=range(len(choices)), xticklabels=["apo", "eng."]*3,
            title="Original eight: consistency")
    axe.tick_params(length=0, labelsize=7.1)
    axe.tick_params(axis="y", labelsize=7.6)
    for x, label in ((1/6, "Original\nnetwork"), (3/6, "Expanded:\noriginal edges"),
                     (5/6, "Expanded:\nall incident")):
        axe.text(x, -0.17, label, transform=axe.transAxes, ha="center", va="top", fontsize=7.1)
    for i, row in enumerate(stability):
        for j, value in enumerate(row):
            axe.text(j, i, "P" if value else "F", ha="center", va="center", fontsize=7.6)
    panel_label(axe, "e", x=-0.45)
    counts = robustness.groupby("policy")[["expanded_apo_stable", "expanded_all_weighted_stable"]].sum()
    weighted_names = {}
    for policy in WINDOW_POLICIES:
        selected_ids = [name for name in displayed.index
                        if indexed.loc[(name, policy), "expanded_all_weighted_stable"]]
        weighted_names[policy] = ", ".join(labels[int(displayed.loc[name, "residue"])] for name in selected_ids) or "None"
        for name in displayed.index:
            trace.append({"panel": "e", "identity": name, "label": labels[int(displayed.loc[name, "residue"])],
                          "value": int(indexed.loc[(name, policy), "expanded_all_weighted_stable"]),
                          "source_table": "robustness.csv", "source_column": "expanded_all_weighted_stable",
                          "policy": policy, "display_role": "weighted-original-eight annotation"})
    fig.text(0.80, 0.092,
             f"All 142: apo {int(counts.loc['original_edges', 'expanded_apo_stable'])} / {int(counts.loc['expanded_incident_edges', 'expanded_apo_stable'])}; "
             f"weighted {int(counts.loc['original_edges', 'expanded_all_weighted_stable'])} / {int(counts.loc['expanded_incident_edges', 'expanded_all_weighted_stable'])}\n"
             f"Original eight, weighted: {weighted_names['original_edges']} / {weighted_names['expanded_incident_edges']}\n"
             "Original edges / all incident", ha="center", va="top", fontsize=7.6, color=DARK_GREY)
    for policy, values in counts.iterrows():
        for column, value in values.items():
            trace.append({"panel": "e", "identity": f"count:{policy}:{column}", "label": policy,
                          "value": int(value), "source_table": "robustness.csv", "source_column": column,
                          "transform": "sum of supplied boolean flags over all 142 groups"})
    fig.text(0.53, 0.02, "Panels b–d: 15 Å, uniform springs. Added residues relax freely; closure and mean responses use the same 801 core directions.",
             ha="center", va="center", fontsize=7.0, color=DARK_GREY)
    snaps = [snapshot(path, output) for path in paths]
    return export(fig, "FigS8", output, [*paths, cif_path], snaps, trace, [
        "All 30 conditions and numerical verification must pass before this figure can be built.",
        "Missing and outside-construct residues are distinct from observed added residues; no coordinates are imputed.",
        "The same 269-position observable is used in both network windows; added nodes receive no test force and relax freely.",
        "Panel d retains the original 142-group candidate universe and perturbs only the original spring sets.",
        "Panel e distinguishes legacy flags, original-edge perturbations and all-incident perturbations; it does not preserve a target candidate count.",
        "All 30 conditions, both perturbation policies and every candidate are retained in the source snapshots.",
    ])


FIGS8_LEGEND = r"""Fig. S8. Sensitivity to observed CRBN residues outside the common 269-position core. Additional observed human CRBN residues are included without filling unobserved coordinates. The closure direction, core rigid-motion constraint and 801-dimensional measured internal response remain fixed; added residues receive no test force and relax freely. (a) Canonical residue coverage for each open reference. Labels give the number of observed positions added to the original core. Unobserved positions and positions outside the construct are separate categories. (b) Natural-log change in relative closure compliance, $\ln(S_{\mathrm{expanded}}/S_{\mathrm{original}})$, for each DDB1 condition at 15 Å with uniform springs. Positive values indicate increased closure compliance relative to the mean core response, not increased absolute compliance. (c) Paired original and expanded values of $R_{\mathrm{body}}$, $R_{\mathrm{internal}}$ and $M$ for the finite closure direction under the same primary conditions. These are changes in log relative compliance, not fractions of displacement or energy. The accompanying comparison table reports finite-displacement and rotational-tangent effects and their percentiles among 5,000 prescribed local rotations for all 30 model conditions. (d) Flexible-DDB1 group effects $D_g$ at 8CVP, 15 Å and uniform springs. The same original springs are perturbed in both windows. The dashed line denotes equal effects, and outlined points identify the original eight stable groups. Explicitly absent groups are excluded from the scatter and counted, not assigned zero. (e) Original stability flags and expanded-window flags for the original eight groups are shown separately. Original edges changes only the frozen group spring set; all incident also changes added springs of the same contact class. P and F denote satisfaction or failure of the relevant prespecified consistency criterion. Apo uses the five required ligand-free uniform-spring conditions; eng. also requires both engineered references at 15 Å with uniform springs. Counts below the matrix summarize all 142 candidates under each perturbation policy. The weighted extension additionally requires all 15 distance-weighted models, giving 22 conditions in this consistency flag. The remaining uniform-spring sensitivity conditions are reported separately and do not change that flag. Full responses, their numerator and denominator, all 30 model conditions, both perturbation policies and all candidate results are retained in the source data."""


PATH_FILES = ("zenodo_predicted_path_metrics.csv", "zenodo_predicted_path_summary.json",
              "zenodo_predicted_path_mapping.json")
PATH_LABELS = {"autoencoder_raw": "Predicted path", "rosetta_relaxed": "After Rosetta relaxation",
               "final_refined": "Final refinement"}
PATH_ADOPTION = "descriptive_computational_structure_comparison"
TRAJECTORY_FILES = ("zenodo_trajectory_metric_summary.csv", "zenodo_trajectory_inventory.json",
                    "zenodo_trajectory_inventory.csv", "zenodo_frame_metrics.csv.gz",
                    "zenodo_comparison_summary.json")
TRAJECTORY_METRICS = ("closure_coordinate", "DDB1_body_translation_A",
                      "DDB1_body_rotation_deg", "DDB1_internal_RMSD_A")
TRAJECTORY_STYLES = {
    "three_cv": ("Three-CV simulation", BLUE, 0),
    "path_cv": ("Path-CV simulation", GREEN, 1),
    "path_selected": ("Path-selected frames", ORANGE, 2),
    "protein_only": ("Protein-only file", REF_COLORS["8D7Y"], 3),
    "relaxation": ("Relaxation stage", DARK_GREY, 4),
}


def trajectory_display(condition: str) -> dict[str, Any]:
    """Keep source processing stages separate from simulation and ligand labels."""
    stages = {
        "three-CV meta-eABF": "three_cv",
        "path-CV meta-eABF": "path_cv",
        "path-selected coordinates from path-CV meta-eABF": "path_selected",
        "processed protein-only coordinates from path-CV meta-eABF directory": "protein_only",
    }
    relaxation = "; relaxation stage (bias flag for this execution not established)"
    if condition.endswith(relaxation):
        stage, ligand = "relaxation", condition[:-len(relaxation)]
    else:
        parts = condition.split("; ", 1)
        if len(parts) != 2 or parts[0] not in stages:
            raise FigureSourceError(f"Trajectory source condition needs an explicit display label: {condition}")
        stage, ligand = stages[parts[0]], parts[1]
    if ligand == "apo":
        ligand_label, marker, ligand_order = "Apo", "o", 0
    elif re.fullmatch(r"ligand-containing \([^()]+\)", ligand):
        ligand_label, marker, ligand_order = "Ligand-containing", "^", 1
    else:
        raise FigureSourceError(f"Unresolved trajectory ligand condition: {condition}")
    label, color, order = TRAJECTORY_STYLES[stage]
    return {"source_stage": stage, "stage_label": label, "color": color, "marker": marker,
            "ligand_label": ligand_label, "display_order": 2 * order + ligand_order}


def validated_path_sources(directory: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Adopt only complete, retained, mapped CRBN-only path representations."""
    paths = [directory / name for name in PATH_FILES]
    for path in paths:
        if not path.is_file():
            raise FigureSourceError(f"Missing adopted path comparison: {path}")
    data = csv(paths[0], ["trajectory_id", "frame", "source_member", "source_model",
                         "closure_coordinate", "NTD_TBD_centroid_distance_A",
                         "TBD_body_rotation_deg", "TBD_internal_RMSD_A",
                         "DDB1_observed_mapped_CA_count", "quantitative_adoption",
                         "CRBN_adjacent_CA_below_2p5A", "CRBN_adjacent_CA_above_4p5A"])
    summary = json.loads(paths[1].read_text())
    mapping = json.loads(paths[2].read_text())
    if not summary.get("raw_sources_retained") or not summary.get("offline_raw_PDB_recomputation"):
        raise FigureSourceError("Path source coordinates are not retained for recomputation")
    if len(data) != summary.get("frames") or data.duplicated(["trajectory_id", "frame"]).any():
        raise FigureSourceError("Path frame count or identities disagree with the completed summary")
    groups = {row["trajectory_id"]: row for row in summary.get("groups", [])}
    maps = {row["trajectory_id"]: row for row in mapping}
    if set(data.trajectory_id) != set(PATH_LABELS) or set(groups) != set(PATH_LABELS):
        raise FigureSourceError("The three sequential path representations are incomplete")
    for name, group in data.groupby("trajectory_id", sort=False):
        record = groups[name]
        if record.get("status") != "eligible" or len(group) != record.get("frames_analyzed") or len(group) != 20:
            raise FigureSourceError(f"Path representation not eligible or incomplete: {name}")
        if sorted(group.frame.tolist()) != list(range(1, len(group) + 1)):
            raise FigureSourceError(f"Nonconsecutive or missing path frames: {name}")
        if maps.get(name, {}).get("core_count") != 269:
            raise FigureSourceError(f"Path representation lacks the fixed 269-position mapping: {name}")
        if maps[name].get("DDB1_mapped_count") != 0 or (group.DDB1_observed_mapped_CA_count != 0).any():
            raise FigureSourceError("This path figure accepts only the documented CRBN-only coordinates")
    finite(data, ["frame", "closure_coordinate", "NTD_TBD_centroid_distance_A",
                  "TBD_body_rotation_deg", "TBD_internal_RMSD_A"])
    adopted = data[data.quantitative_adoption == PATH_ADOPTION]
    if set(adopted.trajectory_id) != {"final_refined"} or len(adopted) != 20:
        raise FigureSourceError("The complete final-refined path did not pass the external adoption gate")
    if (adopted[["CRBN_adjacent_CA_below_2p5A", "CRBN_adjacent_CA_above_4p5A"]] != 0).any().any():
        raise FigureSourceError("Adopted path frames violate the recorded consecutive C-alpha geometry gate")
    return data, paths


def validated_trajectory_sources(directory: Path) -> tuple[pd.DataFrame, list[Path], dict]:
    """Use complete per-trajectory summaries only after all archive members are resolved."""
    paths = [directory / name for name in TRAJECTORY_FILES]
    for path in paths:
        if not path.is_file():
            raise FigureSourceError(f"Missing completed trajectory comparison: {path}")
    completed = json.loads(paths[-1].read_text())
    if (completed.get("analysis_complete") is not True
            or completed.get("all_XTC_members_accounted_for") is not True):
        raise FigureSourceError("The complete XTC acquisition and adoption gate has not passed")
    required_hashed = set(TRAJECTORY_FILES[:2]) | {TRAJECTORY_FILES[3]}
    hashes = {row["file"]: row["sha256"] for row in completed.get("retained_output_hashes", [])}
    if not required_hashed.issubset(hashes):
        raise FigureSourceError("Trajectory summary lacks required retained-source hashes")
    for name in hashes:
        if Path(name).name != name or not (directory / name).is_file():
            raise FigureSourceError(f"Missing or invalid retained trajectory source: {name}")
        if digest(directory / name) != hashes[name]:
            raise FigureSourceError(f"Trajectory completion hash differs from the source: {name}")
    paths.extend(directory / name for name in hashes if name not in TRAJECTORY_FILES)
    data = csv(paths[0], ["trajectory_id", "coordinate_role", "source_condition", "quantitative_adoption",
                         "frames_analyzed", "DDB1_mapped_positions", "metric", "finite_frame_count",
                         "median", "p05", "p95", "minimum", "maximum"])
    inventory = json.loads(paths[1].read_text())
    xtc_inventory = {row["trajectory_id"]: row for row in inventory if row["trajectory_id"].endswith(".xtc")}
    if len(xtc_inventory) != completed.get("xtc_archive_member_count"):
        raise FigureSourceError("Archive member accounting differs from the trajectory inventory")
    if any(row.get("status") != "all_frames_analyzed" and not row.get("status", "").startswith("excluded_")
           for row in xtc_inventory.values()):
        raise FigureSourceError("An XTC member remains unresolved")
    adopted = data[(data.trajectory_id.str.endswith(".xtc"))
                   & (data.quantitative_adoption == PATH_ADOPTION)
                   & data.metric.isin(TRAJECTORY_METRICS)].copy()
    if adopted.trajectory_id.nunique() != completed.get("xtc_analyzed_comparison_count") or adopted.empty:
        raise FigureSourceError("Adopted trajectory count differs from the completed source")
    if adopted.duplicated(["trajectory_id", "metric"]).any():
        raise FigureSourceError("Repeated trajectory/metric summary identities")
    finite(adopted, ["frames_analyzed", "DDB1_mapped_positions", "finite_frame_count",
                     "median", "p05", "p95", "minimum", "maximum"])
    for identity, group in adopted.groupby("trajectory_id"):
        record = xtc_inventory.get(identity, {})
        if (set(group.metric) != set(TRAJECTORY_METRICS)
                or (group.finite_frame_count != group.frames_analyzed).any()
                or (group.frames_analyzed != record.get("frames_analyzed")).any()
                or (group.DDB1_mapped_positions != record.get("DDB1_mapped_positions")).any()
                or record.get("status") != "all_frames_analyzed"
                or record.get("zip_crc32_verified") is not True
                or record.get("core_positions") != 269):
            raise FigureSourceError(f"Incomplete or mismatched full-frame trajectory summary: {identity}")
        if group.source_condition.nunique() != 1 or group.coordinate_role.nunique() != 1:
            raise FigureSourceError(f"A trajectory pools unlike source conditions: {identity}")
    if (adopted.DDB1_mapped_positions != 830).any():
        raise FigureSourceError("The displayed DDB1 construct differs from the documented 830-position construct")
    if ((adopted.minimum > adopted.p05) | (adopted.p05 > adopted["median"])
            | (adopted["median"] > adopted.p95) | (adopted.p95 > adopted.maximum)).any():
        raise FigureSourceError("Trajectory distribution quantiles are not ordered")
    return adopted, paths, xtc_inventory


def fig_s9(directory: Path, output: Path) -> dict:
    path_data, path_sources = validated_path_sources(directory)
    data = path_data[path_data.quantitative_adoption == PATH_ADOPTION].sort_values("frame")
    metrics, trajectory_sources, inventory = validated_trajectory_sources(directory)
    style("FigS9")
    fig = plt.figure(figsize=(WIDTH, 7.0))
    axa = fig.add_axes([0.11, 0.585, 0.35, 0.275])
    axb = fig.add_axes([0.69, 0.585, 0.27, 0.275])
    axes = [fig.add_axes([left, 0.15, 0.22, 0.285]) for left in (0.11, 0.425, 0.74)]
    trace = []
    axa.plot(data.frame, data.closure_coordinate, color=BLUE, marker="o", ms=3.2, lw=0.9)
    axa.set(xlabel="Path frame (ordered, not time)", ylabel="Frozen normalized PC1", title="Final-refined CRBN path",
            xticks=[1, 5, 10, 15, 20], ylim=(-0.10, 1.12), yticks=[0, 0.25, 0.5, 0.75, 1])
    finish_axis(axa, grid="y"); panel_label(axa, "a", x=-0.25)
    for _, row in data.iterrows():
        trace.append({"panel": "a", "identity": f"{row.trajectory_id}:{int(row.frame)}",
                      "label": "Final-refined path", "trajectory_id": row.trajectory_id,
                      "frame": int(row.frame), "x": row.frame, "y": row.closure_coordinate,
                      "source_table": path_sources[0].name, "x_source_column": "frame",
                      "y_source_column": "closure_coordinate", "source_member": row.source_member,
                      "source_model": row.source_model})
    for value, label in [(0, "Deposited closed mean"), (1, "Deposited open mean")]:
        axa.axhline(value, color=MID_GREY, ls="--", lw=0.65, zorder=1)
        axa.text(10.4 if value == 0 else 1.2, value + 0.018, label,
                  color=DARK_GREY, fontsize=7.0, va="bottom")
        trace.append({"panel": "a", "identity": label, "label": label, "y": value,
                      "source_table": "frozen_coordinate_definition", "y_source_column": "normalized_reference_mean"})
    axa.text(0.03, 0.71, "Increasing coordinate → open", transform=axa.transAxes, fontsize=7.0, color=DARK_GREY)
    conditions = {condition: trajectory_display(condition) for condition in set(metrics.source_condition)}
    identities = metrics.drop_duplicates("trajectory_id").copy()
    identities["display_order"] = identities.source_condition.map(lambda value: conditions[value]["display_order"])
    identities = identities.sort_values(["display_order", "trajectory_id"]).reset_index(drop=True)
    identities["display_id"] = [f"T{number + 1:02d}" for number in range(len(identities))]
    identity_columns = ["display_id", "trajectory_id", "source_condition", "coordinate_role",
                        "frames_analyzed", "DDB1_mapped_positions", "quantitative_adoption"]
    identity_table = identities[identity_columns].copy()
    identity_table["source_stage"] = identity_table.source_condition.map(lambda value: conditions[value]["source_stage"])
    identity_table["ligand_label"] = identity_table.source_condition.map(lambda value: conditions[value]["ligand_label"])
    identity_table["topology_member"] = identity_table.trajectory_id.map(lambda name: inventory[name].get("topology_member", ""))
    identity_table["source_sha256"] = identity_table.trajectory_id.map(lambda name: inventory[name].get("source_sha256", ""))
    identity_path = output / "FigS9_trajectory_identities.csv"
    identity_table.to_csv(identity_path, index=False)
    specs = [(axb, "b", "closure_coordinate", "Frozen normalized PC1", "Closure coordinate by series"),
             (axes[0], "c", "DDB1_body_translation_A", "Translation (Å)", "DDB1 translation"),
             (axes[1], "d", "DDB1_body_rotation_deg", "Rotation (°)", "DDB1 rotation"),
             (axes[2], "e", "DDB1_internal_RMSD_A", "Internal Cα RMSD (Å)", "DDB1 internal deformation")]
    for ax, panel, metric, xlabel, title in specs:
        for number, identity in identities.iterrows():
            row = metrics[(metrics.trajectory_id == identity.trajectory_id) & (metrics.metric == metric)].iloc[0]
            display = conditions[row.source_condition]
            color, marker = display["color"], display["marker"]
            ax.plot([row.p05, row.p95], [number, number], color=color, lw=1.0, zorder=2)
            ax.scatter(row["median"], number, color=color, marker=marker, s=17, ec="white", lw=0.3, zorder=3)
            trace.append({"panel": panel, "identity": identity.trajectory_id, "label": identity.display_id,
                          "trajectory_id": identity.trajectory_id, "x": row["median"], "y": number,
                          "lower": row.p05, "upper": row.p95, "minimum": row.minimum, "maximum": row.maximum,
                          "frames_analyzed": int(row.frames_analyzed), "metric": metric,
                          "source_condition": row.source_condition, "source_stage": display["source_stage"],
                          "source_table": TRAJECTORY_FILES[0], "source_column": "median;p05;p95"})
        ax.set(yticks=range(len(identities)), yticklabels=identities.display_id if panel in {"b", "c"} else [],
                ylim=(len(identities)-0.5, -0.5), xlabel=xlabel, title=title)
        ax.tick_params(labelsize=7.6, axis="both", length=2)
        finish_axis(ax, grid="y"); panel_label(ax, panel, x=-0.29 if panel != "b" else -0.33)
    axb.axvline(0, color=MID_GREY, lw=0.6, ls="--"); axb.axvline(1, color=MID_GREY, lw=0.6, ls="--")
    axb.set_xlim(min(-0.1, float(metrics.loc[metrics.metric == "closure_coordinate", "p05"].min()) - 0.04),
                  max(1.05, float(metrics.loc[metrics.metric == "closure_coordinate", "p95"].max()) + 0.04))
    displayed_stages = {entry["source_stage"] for entry in conditions.values()}
    handles = [Line2D([], [], color=color, lw=1.4, label=label)
               for stage, (label, color, _) in TRAJECTORY_STYLES.items() if stage in displayed_stages]
    displayed_ligands = {entry["ligand_label"] for entry in conditions.values()}
    handles.extend(Line2D([], [], color=BLACK, marker=marker, lw=0, ms=3.5, label=label)
                   for label, marker in [("Apo", "o"), ("Ligand-containing", "^")] if label in displayed_ligands)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.953),
               ncol=3, fontsize=7.3, columnspacing=2.0)
    fig.text(0.53, 0.99, "Public computational coordinates in the frozen CRBN structural frame", ha="center", va="top", fontsize=8.7, weight="bold")
    fig.text(0.535, 0.496, "DDB1 relative to each series' first frame after HB alignment; observed construct omits residues 396–705",
             ha="center", va="center", fontsize=7.0, color=DARK_GREY)
    fig.text(0.535, 0.057,
             f"{len(identities)} coordinate series. Points: frame median; bars: 5th–95th frame percentiles. All frames retained; no pooling across series.",
             ha="center", va="center", fontsize=7.0, color=DARK_GREY)
    paths = [*path_sources, *trajectory_sources]
    snaps = [snapshot(path, output) for path in paths]
    return export(fig, "FigS9", output, paths, [*snaps, identity_path], trace, [
        "The 20 final-cleanup path frames appear only in panel a; all other panels use individually listed coordinate series, without treating frames as independent replicates.",
        "All archive XTC members must be accounted for and every plotted metric must use all frames of its own trajectory.",
        "Frame medians and 5th–95th percentiles are supplied descriptive distributions, not confidence intervals or equilibrium populations; source processing roles and unresolved relaxation-stage bias are retained.",
        "DDB1 metrics use the 830-position construct omitting 396–705; rigid alignment sets an observational frame and does not apply the static force constraints.",
        "Full per-frame metrics are copied as a compressed source companion; plotted summaries and trajectory labels trace to exact CSV rows.",
        "The two earlier path-processing stages remain as geometry diagnostics; DDB1 is absent from all three CRBN-only path coordinate sets.",
    ])


FIGS9_LEGEND = r"""Fig. S9. Public refined path and CRBN–DDB1 trajectory coordinates compared in the frozen structural frame. (a) All 20 frames from the source workflow's final-cleanup CRBN path in Zenodo record 16459122, projected onto the unchanged PCA axis and normalized to the deposited closed and open means at 0 and 1. Frame index denotes supplied order, not time. The maximum coordinate is 0.53, below the original open region at or above 0.95. Earlier processing-stage geometries remain in the source data; DDB1 is absent from these path files. (b) Frozen closure-coordinate distributions for each adopted public coordinate series. (c–e) DDB1 centroid translation, best-fit rigid-body rotation and residual Cα RMSD after its own rigid-body fit, respectively. Each metric is relative to the first frame of its own series after aligning CRBN HB residues 187–317 to the frozen mean. The measured DDB1 construct contains 830 mapped Cα positions and omits native residues 396–705. HB alignment sets an observational coordinate frame; it does not impose the force constraints used in the static models. Points mark the median and horizontal bars the 5th–95th percentiles of all frames within each series. Line colours identify the source stage and symbols indicate the source ligand condition. Simulation outputs, selected path coordinates, processed protein-only coordinates and relaxation-stage coordinates retain their distinct source roles; the relaxation-stage bias setting is unconfirmed. These descriptive ranges are not confidence intervals or equilibrium populations. Frames are not pooled across series. Row identifiers link to the full source-file names, conditions and checksums in the source table; all frame metrics are retained as a compressed companion."""


def update_legends(baseline: Path, output: Path, extra: dict[str,str] | None = None) -> Path:
    text = baseline.read_text()
    updates = {"Fig3":FIG3_LEGEND, "Fig4":FIG4_LEGEND, **(extra or {})}
    for stem,legend in updates.items():
        pattern = rf"(^## {stem}\n).*?(?=^## |\Z)"
        replacement = f"## {stem}\n\n{legend}\n\n"
        if re.search(pattern,text,re.M|re.S):
            text=re.sub(pattern,lambda _, replacement=replacement:replacement,text,flags=re.M|re.S)
        else:
            text=text.rstrip()+"\n\n"+replacement
    target=output/"LEGENDS.md"
    target.write_text(text)
    return target


def build(config_path: Path | str, output_dir: Path | str,
          figures: Sequence[str] | None = None, offline: bool = False) -> dict[str,Any]:
    """Build the requested figures from completed local tables; never fetch data."""
    config_path=Path(config_path); output=Path(output_dir)
    config=json.loads(config_path.read_text())
    output.mkdir(parents=True,exist_ok=True)
    frozen=source_path(config,"directional_sources",output.resolve())
    baseline=source_path(config,"baseline_legends",output.resolve()/"BASELINE_LEGENDS.md")
    cif_path=source_path(config,"8cvp_cif",ROOT/"data/_cif_cache/8CVP.cif.gz")
    external=source_path(config,"external_sources",output.resolve().parent/"external")
    window=source_path(config,"window_sources",output.resolve().parent/"window")
    tables={key:frozen/value for key,value in FROZEN_TABLES.items()}
    requested=tuple(figures) if figures is not None else ("Fig3","Fig4","FigS8")
    optional_external={"status":"not_requested"}
    if figures is None:
        try:
            validated_path_sources(external)
            validated_trajectory_sources(external)
        except FigureSourceError as error:
            optional_external={"status":"not_adopted", "reason":str(error)}
        else:
            requested=(*requested,"FigS9")
            optional_external={"status":"adopted", "role":"refined computational coordinate comparison"}
    unknown=set(requested)-{"Fig3","Fig4","FigS8","FigS9"}
    if unknown:raise FigureSourceError(f"Unknown figures: {sorted(unknown)}")
    result=[]; extra={}
    for stem in requested:
        if stem=="Fig3":result.append(fig3(tables,output))
        elif stem=="Fig4":result.append(fig4(tables,cif_path,output))
        elif stem=="FigS8":
            result.append(fig_s8(window,config,cif_path,output));extra[stem]=FIGS8_LEGEND
        elif stem=="FigS9":
            result.append(fig_s9(external,output));extra[stem]=FIGS9_LEGEND
            optional_external={"status":"adopted", "role":"refined computational coordinate comparison"}
        else:raise FigureSourceError(f"{stem} requires completed observed-window or external source tables")
    if figures is None and optional_external["status"] == "not_adopted":
        figure_dir=output.parents[1]/"manuscript"/"figures"
        for path in (figure_dir/"FigS9.png", figure_dir/"vector/FigS9.pdf", figure_dir/"vector/FigS9.svg",
                     output/"FigS9_input_manifest.json", output/"FigS9_plotted_values.csv"):
            path.unlink(missing_ok=True)
    legend_input=output/"LEGENDS.md" if figures is not None and (output/"LEGENDS.md").is_file() else baseline
    legend=update_legends(legend_input,output,extra)
    summary={"status":"complete" if figures is None else "requested_subset_complete",
             "offline":True,"figures":result,"legend":str(legend),"config_sha256":digest(config_path),
             "optional_external_figure":optional_external}
    (output/"review_figure_build_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--figures",nargs="+",choices=("Fig3","Fig4","FigS8","FigS9"))
    args=parser.parse_args(argv)
    print(json.dumps(build(args.config,args.output_dir,args.figures,args.offline),indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
