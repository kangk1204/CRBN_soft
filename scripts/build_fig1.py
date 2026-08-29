#!/usr/bin/env python3
"""Build Fig 1 (archive-wide PCA and cumulative ANM capture) from canonical data.

All four panels are sourced from the committed canonical arrays on the
five-open/65-closed axis:
  data/crbn_pca.npz         PC scores, variance ratio, open mask (0.744 axis)
  data/crbn_anm_modes.npz   cumulative ANM axis projection and ANM-PCA RMSIP
Panel c cumulative curve starts at the canonical mode-1 overlap 0.744.
"""
import json
from figure_package_utils import prepare_figure_dirs

FIGURES, VECTOR, _ = prepare_figure_dirs()

import numpy as np, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

C_APO   = "#3b6ea5"      # genuine-apo — blue
C_DRUG  = "#e07b39"      # drug-conditioned — orange
C_NATIVE= "#4bab8c"      # native-substrate (9NR3) — green
FOCAL   = "#6f4e9c"      # purple focal accent; avoids a red/green pairing
GREY    = "#9aa0a6"


def clean_svg(path):
    """Normalise generated SVG whitespace so release diffs stay clean."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(line.rstrip() for line in text.splitlines()) + "\n")

pca = np.load("data/crbn_pca.npz")
anm = np.load("data/crbn_anm_modes.npz")

pc1 = pca["pc1_scores"]; pc2 = pca["pc2_scores"]
open_mask = pca["open_mask"].astype(bool)
# pc1_scores are stored divided by sqrt(N_Ca) (RMSD-scaled, Angstrom); pc2_scores are
# raw projections. Put PC2 on the same RMSD scale so the scatter axes share units.
_NCA = pca["mean"].reshape(-1, 3).shape[0]        # 269 common Ca
pc2 = pc2 / np.sqrt(_NCA)
vr = pca["variance_ratio"]
cum = anm["cum_overlap"]; rmsip = float(anm["rmsip"])

# global-state labels (drug-conditioned / genuine-apo / native-substrate) aligned to pca_diffvec order
dv = np.load("data/pca_diffvec.npz")
labels = [str(x) for x in dv["labels"]]
log = {r["pdb"].upper(): r["global_state"] for r in csv.DictReader(open("data/crbn_curation_log.csv"))}
gstate = np.array([log.get(l.upper(), "drug-conditioned") for l in labels])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
    "axes.titlelocation": "left", "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig = plt.figure(figsize=(7.9, 2.9), dpi=300)
gs = GridSpec(1, 4, figure=fig, wspace=0.52, left=0.065, right=0.995,
              top=0.88, bottom=0.26)

# panel a: PCA scatter, colored by global ligand/substrate state
axa = fig.add_subplot(gs[0, 0])
axa.axvline(np.sort(pc1)[::-1][4:6].mean(), ls=":", color="#bbbbbb", lw=0.8)
m_drug = gstate == "drug-conditioned"
m_apo  = gstate == "genuine-apo"
m_nat  = gstate == "native-substrate"
axa.scatter(pc1[m_drug], pc2[m_drug], color=C_DRUG, alpha=0.75, s=20, lw=0, label="drug-conditioned")
axa.scatter(pc1[m_apo],  pc2[m_apo],  color=C_APO,  alpha=0.95, s=26, lw=0, label="genuine apo")
axa.scatter(pc1[m_nat],  pc2[m_nat],  color=C_NATIVE, alpha=0.95, s=26, lw=0,
            marker="D", label="native substrate")
axa.text(0.30, 0.04, "closed", transform=axa.transAxes, color="#999999", fontsize=7.5, style="italic", ha="center")
axa.text(0.86, 0.04, "open", transform=axa.transAxes, color="#999999", fontsize=7.5, style="italic", ha="center")
axa.set_xlabel(f"PC1 ({vr[0]*100:.0f}% coordinate variance)")
axa.set_ylabel(f"PC2 ({vr[1]*100:.0f}% coordinate variance)")
# panel descriptions live in the figure legend (figure style), not in the figure
axa.legend(loc="center", fontsize=7.5, borderpad=0.2, labelspacing=0.3,
           handletextpad=0.4, bbox_to_anchor=(0.76, 0.70), frameon=True,
           facecolor="white", edgecolor="none", framealpha=0.90)

# panel b: variance spectrum
axb = fig.add_subplot(gs[0, 1])
npc = min(10, len(vr)); modes = np.arange(1, npc+1)
axb.bar(modes, vr[:npc]*100, color=[FOCAL if i==0 else "#3b6ea5" for i in range(npc)], width=0.72)
axb.plot(modes, np.cumsum(vr[:npc])*100, "-o", color=FOCAL, ms=3, lw=1.0)
axb.annotate(f"PC1 = {vr[0]*100:.0f}%", xy=(1, vr[0]*100), xytext=(2.3, vr[0]*100-16),
             fontsize=8, color="#3b6ea5",
             arrowprops=dict(arrowstyle="-", color="#3b6ea5", lw=0.8))
axb.set_xlabel("principal component"); axb.set_ylabel("Coordinate variance (%)")
axb.set_xticks(modes)

# panel c: projection norm of the open–closed axis in successive ANM subspaces;
# the RMSIP reference line is a separate statistic (ANM vs PCA subspaces)
axc = fig.add_subplot(gs[0, 2])
k = np.arange(1, len(cum[:10])+1)
axc.plot(k, cum[:10], "-o", color=FOCAL, ms=3.5, lw=1.1)
axc.axhline(rmsip, ls="--", color="#3b6ea5", lw=1.0)
axc.text(1.2, rmsip+0.008, f"ANM\u2013PCA RMSIP = {rmsip:.2f}", color="#3b6ea5", fontsize=7.5, va="bottom")
axc.set_xlabel("ANM modes included"); axc.set_ylabel("Axis projection norm")
pass  # panel title moved to legend
axc.set_xticks(k); axc.set_ylim(0.60, 0.92)

# panel d: bimodality of the transition coordinate (closed vs open, empty middle)
axd = fig.add_subplot(gs[0, 3])
_cm = pc1[~open_mask].mean(); _om = pc1[open_mask].mean()
_norm = (pc1 - _cm) / (_om - _cm)          # 0 = closed mean, 1 = open mean
bins = np.arange(-0.15, 1.15 + 1e-9, 0.05)
axd.hist(_norm[~open_mask], bins=bins, color=C_DRUG, alpha=0.85, label=f"closed (n={int((~open_mask).sum())})")
axd.hist(_norm[open_mask],  bins=bins, color=C_APO,  alpha=0.95, label=f"open (n={int(open_mask.sum())})")
# The band is the artifact's own 15-85% window between the clusters, not a hardcoded
# interval. A band chosen inside the largest gap is empty by construction; this one is
# derived from the data so the emptiness is a statement about the data.
_em = json.load(open("data/window_sensitivity.json", encoding="utf-8"))["empty_middle"]["a_paper_rule"]
axd.axvspan(*_em["band_15_85_pct"], color="#dddddd", alpha=0.5, lw=0)
axd.text(0.6, 0.45, "no structures\n(empty middle)", transform=axd.get_xaxis_transform(),
         ha="center", va="center", fontsize=7.5, color="#888888")
axd.set_xlabel("transition coordinate")
axd.set_ylabel("structures")
pass  # panel title moved to legend
axd.legend(loc="upper center", fontsize=7.5)
axd.set_xlim(-0.2, 1.15)

for ax, l in [(axa,"a"),(axb,"b"),(axc,"c"),(axd,"d")]:
    ax.text(-0.14, 1.06, f"({l})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="right")

fig.savefig(FIGURES / "Fig1.png", dpi=300, bbox_inches="tight")
fig.savefig(VECTOR / "Fig1.pdf", bbox_inches="tight")
fig.savefig(VECTOR / "Fig1.svg", bbox_inches="tight")
clean_svg(VECTOR / "Fig1.svg")
print(f"Fig1 built: PC1 {vr[0]*100:.1f}%, cum[0]={cum[0]:.4f}, cum[9]={cum[9]:.4f}, RMSIP {rmsip:.3f}")
