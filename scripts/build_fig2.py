#!/usr/bin/env python3
"""Build Fig 2 (the open-to-closed transition aligns with a low-frequency open-state mode) from canonical data.

(a) Overlap of the first ten ANM modes (built on the open state alone) with the
    open->closed difference vector, from data/crbn_anm_modes.npz (anm_diff_overlap);
    mode 1 highlighted at the canonical 0.744.
(b) ANM mode-1 porcupine on the open Ca trace, from data/crbn_anm_mode1.nmd.
(c) Archive-derived PC1 porcupine on the same trace, from data/crbn_pca_modes.nmd.
(d) Per-mode overlap with the transition axis built from the open (8CVP) vs the
    closed (5FQD) structure, recomputed from data/crbn_ensemble.ens.npz with the
    same ANM used in scripts/anm_robustness.py (open recovers the axis at mode 1;
    closed only at a higher mode).
Domain colour code NTD #3b6ea5 / HB #4bab8c / TBD #e07b39. Panel labels (a)-(d).
Panel d uses the same open/closed encoding as Fig 5: blue = open, orange = closed.
"""
from figure_package_utils import prepare_figure_dirs

FIGURES, VECTOR, _ = prepare_figure_dirs()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

NTD_C = "#3b6ea5"; HB_C = "#4bab8c"; TBD_C = "#e07b39"
FOCAL = "#e07b39"; GREY = "#b8b8b8"
# open/closed encoding for panel d, kept identical to Fig 5 (scripts/build_fig5_robustness.py)
C_OPEN = "#3b6ea5"; C_CLOSED = "#e07b39"


def clean_svg(path):
    """Normalise generated SVG whitespace so release diffs stay clean."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(line.rstrip() for line in text.splitlines()) + "\n")

# domain boundaries in author numbering: NTD 1-186, HB 187-317, TBD 318-426
def domain_color(resnum):
    if resnum <= 186:
        return NTD_C
    if resnum <= 317:
        return HB_C
    return TBD_C


def parse_nmd(path):
    coords = None; mode = None; resnums = None
    for ln in open(path):
        tok = ln.split()
        if not tok:
            continue
        if tok[0] == "coordinates":
            coords = np.array(tok[1:], float).reshape(-1, 3)
        elif tok[0] == "mode" and mode is None:
            mode = np.array(tok[3:], float).reshape(-1, 3)   # skip id + scale
        elif tok[0] in ("resnums", "resids"):
            resnums = np.array(tok[1:], int)
    if resnums is None:                                       # fall back to committed author numbering
        import csv
        resnums = np.array([int(r["resnum"]) for r in
                            csv.DictReader(open("data/crbn_residue_fluctuations.csv"))])
    return coords, mode, resnums


# --- ANM helper (identical formalism to scripts/anm_robustness.py) ---
def anm_hessian(coords, cutoff):
    n = len(coords); H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        d = coords - coords[i]; r = np.linalg.norm(d, axis=1)
        for j in range(i + 1, n):
            if 1e-6 < r[j] <= cutoff:
                k = np.outer(d[j], d[j]) / r[j] ** 2
                H[3*i:3*i+3, 3*j:3*j+3] = -k; H[3*j:3*j+3, 3*i:3*i+3] = -k
                H[3*i:3*i+3, 3*i:3*i+3] += k; H[3*j:3*j+3, 3*j:3*j+3] += k
    return H


def mode_overlaps(coords, dvec, k=10, cutoff=15.0):
    w, v = np.linalg.eigh(anm_hessian(coords, cutoff))
    nz = w > 1e-9; v = v[:, nz][:, :k]
    return np.array([abs(v[:, m] @ dvec) for m in range(v.shape[1])])


anm = np.load("data/crbn_anm_modes.npz")
open_spec = np.abs(anm["anm_diff_overlap"][:10])          # 8CVP open mode spectrum

# panel d: recompute open(8CVP) and closed(5FQD) spectra from the committed ensemble
ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
confs = ens["_confs"]; labels = [str(x) for x in ens["_labels"]]
dv = np.load("data/pca_diffvec.npz")
OPEN = sorted(str(l) for l, m in zip(dv["labels"], dv["open_mask"]) if m)
open_mask = np.array([l in OPEN for l in labels])
diff = confs[open_mask].mean(0) - confs[~open_mask].mean(0)
dvec = (diff / np.linalg.norm(diff)).ravel()
spec_open = mode_overlaps(confs[labels.index("8CVP")], dvec)
spec_closed = mode_overlaps(confs[labels.index("5FQD")], dvec)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 12, "axes.linewidth": 0.8, "savefig.dpi": 300,
})

fig = plt.figure(figsize=(13.5, 12.0))
gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.20)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[1, 0])
axd = fig.add_subplot(gs[1, 1])

# panel a: overlap spectrum, open state
modes_x = np.arange(1, 11)
bars = axa.bar(modes_x, open_spec, color=GREY)
bars[0].set_color(FOCAL)
axa.annotate(f"{open_spec[0]:.2f}", (1, open_spec[0]), xytext=(14, -2),
             textcoords="offset points", color=FOCAL, fontsize=13, fontweight="bold")
axa.set_xticks(modes_x); axa.set_xlabel("ANM mode (from open state)")
axa.set_ylabel("Directional overlap with open–closed axis")
# panel descriptions live in the figure legend (figure style), not in the figure
axa.spines[["top", "right"]].set_visible(False)
axa.text(-0.14, 1.05, "(a)", transform=axa.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")


def panel_png(ax, png, title, letter):
    """Display a committed PyMOL porcupine render (scripts/render_fig2_3d.py)."""
    import os
    import matplotlib.image as mpimg
    from matplotlib.lines import Line2D
    if not os.path.exists(png):
        ax.text(0.5, 0.5, "run scripts/render_fig2_3d.py\nto build " + os.path.basename(png),
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#999999")
    else:
        ax.imshow(mpimg.imread(png))
    ax.set_axis_off()
    pass  # panel title moved to legend
    handles = [Line2D([0], [0], color=c, lw=3) for c in (NTD_C, HB_C, TBD_C)]
    ax.legend(handles, ["NTD", "HB", "TBD"], frameon=False, fontsize=9,
              loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    ax.text(-0.14, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")


panel_png(axb, "figures/panels/fig2_anm3d.png", "ANM mode 1 (from open)", "(b)")
panel_png(axc, "figures/panels/fig2_pc13d.png", "Archive-derived PC1", "(c)")

# panel d: overlap by starting state
w = 0.4
axd.bar(modes_x - w/2, spec_open, w, color=C_OPEN, label="from open (8CVP)")
axd.bar(modes_x + w/2, spec_closed, w, color=C_CLOSED, label="from closed (5FQD)")
axd.set_xticks(modes_x); axd.set_xlabel("ANM mode")
axd.set_ylabel("Directional overlap with open–closed axis")
pass  # panel title moved to legend
axd.legend(frameon=False, fontsize=9.5, loc="upper right")
axd.spines[["top", "right"]].set_visible(False)
axd.text(-0.14, 1.05, "(d)", transform=axd.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")

fig.savefig(FIGURES / "Fig2.png", dpi=300, bbox_inches="tight")
fig.savefig(VECTOR / "Fig2.pdf", bbox_inches="tight")
fig.savefig(VECTOR / "Fig2.svg", bbox_inches="tight")
clean_svg(VECTOR / "Fig2.svg")
print(f"Fig2 built: open mode1 {open_spec[0]:.3f}; panel-d open best mode "
      f"{int(spec_open.argmax())+1} ({spec_open.max():.2f}), "
      f"closed best mode {int(spec_closed.argmax())+1} ({spec_closed.max():.2f})")
