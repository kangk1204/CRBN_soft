#!/usr/bin/env python3
"""Fig 4 - mobility map of the drug pocket.
Reproducible from the matching archived data/render bundle (relative paths, no hardcoded values).
Panel a: ANM square-fluctuation across the TBD (data/crbn_anm_modes.npz).
Panel b: drug vs Zn group-mean mobility percentile for ANM, PCA (data/crbn_residue_fluctuations.csv)
         The MD profile is deliberately NOT shown here: data/crbn_md_rmsf.csv has no
         committed generating script, so it cannot sit in a main figure beside two
         reproducible profiles. It is described in the Supplementary Information.
Panel c: closed-state TBD coloured by ANM mobility, with bound S-lenalidomide and the
         structural Zn (figures/panels/render_closed_pocket.png; scripts/render_fig4_pocket.py).
"""
from figure_package_utils import prepare_figure_dirs, require_prepared_panel

FIGURES, VECTOR, PANELS = prepare_figure_dirs()
POCKET_PANEL = require_prepared_panel(
    PANELS / "render_closed_pocket.png",
    "pymol -cq scripts/render_fig4_pocket.py",
)

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

DOM = {"NTD": "#3b6ea5", "HB": "#4bab8c", "TBD": "#e07b39"}
FOCAL = "#6f4e9c"; GREY = "#9aa0a6"  # purple avoids a red/green pairing
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})

def set_frame(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
def panel_letter(ax, s, x=-0.13, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="right")
def clean_svg(path):
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(line.rstrip() for line in txt.splitlines()) + "\n")

drug_res = [378, 380, 386]; zn_res = [323, 326, 391, 394]

# ---- panel a data: ANM square-fluctuation from committed npz ----
am = np.load("data/crbn_anm_modes.npz")
resnums = am["resnums"]; ev = am["anm_eigvecs"]; ew = am["anm_eigvals"]
anm_sqf = np.zeros(len(resnums))
for m in range(10):
    v = ev[:, m].reshape(-1, 3); anm_sqf += (v**2).sum(1) / ew[m]
anm_sqf /= anm_sqf.max()

# ---- panel b data: group-mean percentiles from committed CSVs ----
def load2(path):
    rows = list(csv.DictReader(open(path)))
    cols = rows[0].keys(); rn = np.array([int(r["resnum"]) for r in rows])
    return rn, {c: np.array([float(r[c]) for r in rows]) for c in cols if c != "resnum"}
def res_pct(rn, arr, res):
    # percentile of each residue's value within the 269-residue analysis window, which is
    # the whole of `arr`: both profiles plotted here are computed on that same window.
    return [100.0 * np.mean(arr <= arr[rn == r][0]) for r in res if (rn == r).any()]

frn, fcols = load2("data/crbn_residue_fluctuations.csv")   # 269-residue analysis window
# Both profiles are already computed on this window, so the percentile denominator is the
# same for each without further pooling (Fig 4b).
indiv = {
    "ANM": {"drug": res_pct(frn, fcols["anm_sqfluct"], drug_res), "Zn": res_pct(frn, fcols["anm_sqfluct"], zn_res)},
    "PCA": {"drug": res_pct(frn, fcols["pca_sqfluct"], drug_res), "Zn": res_pct(frn, fcols["pca_sqfluct"], zn_res)},
}
conc = {m: {g: float(np.mean(v)) for g, v in d.items()} for m, d in indiv.items()}
methods = ["ANM", "PCA"]; x = np.arange(2); w = 0.38
drugv = [conc[m]["drug"] for m in methods]; znv = [conc[m]["Zn"] for m in methods]

fig = plt.figure(figsize=(8.2, 3.5))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.12, 0.78, 1.35], wspace=0.42,
              left=0.075, right=0.99, top=0.90, bottom=0.16)

# panel a
axa = fig.add_subplot(gs[0, 0])
tbd = (resnums >= 318) & (resnums <= 424)
# the analysis window is non-contiguous; break the trace at sequence gaps so the plot
# does not interpolate across unresolved segments (as in Fig 3b)
def _break_at_gaps(x, y):
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if x[i] - x[i - 1] > 1:
            xs.append(x[i - 1] + 0.5); ys.append(np.nan)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs, float), np.array(ys, float)

tx, ty = _break_at_gaps(resnums[tbd], anm_sqf[tbd])
axa.plot(tx, ty, color=DOM["TBD"], lw=1.1, zorder=2)
idx = list(resnums)
axa.scatter(drug_res, [anm_sqf[idx.index(r)] for r in drug_res], s=42, c=FOCAL,
            edgecolor='w', lw=0.5, zorder=4, label="thalidomide-binding")
axa.scatter(zn_res, [anm_sqf[idx.index(r)] for r in zn_res], marker='s', s=34, c='#333',
            edgecolor='w', lw=0.5, zorder=4, label="Zn\u00b2\u207a-coordinating")
for r in drug_res:
    axa.annotate(str(r), (r, anm_sqf[idx.index(r)]), textcoords="offset points",
                 xytext=(0, 6), fontsize=7.5, color=FOCAL, ha='center')
axa.set_xlabel("residue (TBD)"); axa.set_ylabel("ANM square fluctuation")

axa.legend(frameon=False, fontsize=7.5, loc='upper left', handletextpad=0.3)
axa.set_ylim(0, 1.18); set_frame(axa)

# panel b
axb = fig.add_subplot(gs[0, 1])
axb.bar(x - w/2, drugv, w, color=FOCAL, zorder=3, label="drug-binding")
axb.bar(x + w/2, znv, w, color='#555', zorder=3, label="zinc site")
axb.axhline(50, ls=':', color=GREY, lw=0.8)
for xi, (d_, z_) in enumerate(zip(drugv, znv)):
    axb.text(xi - w/2, 2.5, f"{d_:.0f}", ha='center', va='bottom', fontsize=7.5, color='w')
    axb.text(xi + w/2, 2.5, f"{z_:.0f}", ha='center', va='bottom', fontsize=7.5, color='w')
# individual residue percentiles behind each group mean, so the overlap between the
# n=3 drug-binding and n=4 zinc groups is visible
for xi, m in enumerate(methods):
    for off, grp in [(-w/2, "drug"), (w/2, "Zn")]:
        v = indiv[m][grp]
        xs = xi + off + np.linspace(-0.32, 0.32, len(v)) * w
        axb.scatter(xs, v, s=7, facecolor='w', edgecolor='#222', lw=0.45, zorder=5, clip_on=False)
axb.set_xticks(x); axb.set_xticklabels(methods)
axb.set_ylabel("mean mobility percentile")
axb.set_ylim(0, 100)
# The exact n=3 versus n=4 limitation and non-significant rank test are stated in
# the caption. Avoid a bracket here because it can visually imply an inferential
# comparison even when its label says otherwise.
axb.legend(frameon=False, fontsize=7.5, loc='lower center', ncol=2,
           handlelength=0.9, handletextpad=0.3, columnspacing=0.8,
           borderaxespad=0.0, bbox_to_anchor=(0.5, 1.03))
set_frame(axb)

# panel c
axc = fig.add_subplot(gs[0, 2]); axc.axis('off')
axc.imshow(Image.open(POCKET_PANEL))

axc.text(0.12, 1.02, "drug-binding loop", transform=axc.transAxes,
         fontsize=7.5, color=FOCAL, fontweight='bold', ha='left', va='bottom')
axc.text(0.06, -0.02, "lenalidomide", transform=axc.transAxes,
         fontsize=7.5, color='#8a7a00', ha='left', va='bottom')
axc.text(0.98, 0.18, "Zn\u00b2\u207a site", transform=axc.transAxes,
         fontsize=7.5, color='#111', fontweight='bold', ha='right', va='top')

for ax, l in [(axa, 'a'), (axb, 'b'), (axc, 'c')]:
    panel_letter(ax, f"({l})")
fig.savefig(FIGURES / "Fig4.png", dpi=300, bbox_inches="tight")
fig.savefig(VECTOR / "Fig4.pdf", bbox_inches="tight"); fig.savefig(VECTOR / "Fig4.svg", bbox_inches="tight")
clean_svg(VECTOR / "Fig4.svg")
print(f"Fig4 rebuilt. "
      f"ANM {conc['ANM']['drug']:.1f}/{conc['ANM']['Zn']:.1f}, PCA {conc['PCA']['drug']:.1f}/{conc['PCA']['Zn']:.1f}")
