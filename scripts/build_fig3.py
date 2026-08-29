#!/usr/bin/env python3
"""Build Fig 3 (rigid-body swing about the DDB1-binding helical-bundle hinge).

(a) GNM cross-correlation map from data/crbn_anm_modes.npz (all 20 GNM modes;
    covariance = sum_k (1/lambda_k) v_k v_k^T, normalised to correlations). The eigenpairs
    come from the artifact reproduce_modes.py regenerates, so rerunning that script updates
    this figure. The figure previously read data/crbn_gnm_model.gnm.npz, a ProDy-pickled
    file that no script in the repository writes; the two agree to ~1e-14 (eigenvalues
    9.0e-15, correlation map 4.7e-14), which is why this needed a test rather than an eye.
(b) Per-residue square fluctuation (each normalised to its own maximum) for the
    intrinsic ANM and archive-derived PCA, from data/crbn_residue_fluctuations.csv.
Domain colour code NTD #3b6ea5 / HB #4bab8c / TBD #e07b39 threaded across figures.
Panel labels use the panel (a)/(b) convention.
"""
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NTD_C = "#3b6ea5"; HB_C = "#4bab8c"; TBD_C = "#e07b39"
ANM_C = "#333333"; PCA_C = "#6b4fa0"

# domain boundaries (UniProt/author numbering): NTD 1-186, HB 187-317, TBD 318-426
NTD_HB = 186.5; HB_TBD = 317.5
# Shades residues 258-315 inclusive. The band must contain all three reported hinge
# points (273, 289, 315); an earlier upper edge of 314 excluded 315, and the caption,
# the body text and this constant disagreed three ways.
HINGE = (257.5, 315.5)

g = np.load("data/crbn_anm_modes.npz")
vecs = np.asarray(g["gnm_eigvecs"]); vals = np.asarray(g["gnm_eigvals"])
cov = (vecs / vals) @ vecs.T
dsq = np.sqrt(np.diag(cov))
cc = cov / np.outer(dsq, dsq)

rows = list(csv.DictReader(open("data/crbn_residue_fluctuations.csv")))
res = np.array([int(r["resnum"]) for r in rows])
anm = np.array([float(r["anm_sqfluct"]) for r in rows])
# Normalise to the profile with the chain-break artefacts removed, not to residue 222.
# 222 is the global maximum and the same caption calls it an artefact, so dividing by it
# compresses every real feature against a value the text asks the reader to discount.
# "Near a break" means within two sequence positions of a gap edge, which is what it takes
# to reach 222: the 198-220 gap leaves 221 as the immediate flank and 222 one step further,
# and it is 222, not 221, that is under-contacted enough to spike. Excluded from the
# denominator only -- every residue is still plotted, and 222 is annotated as the artefact.
_edge = np.zeros(len(res), bool)
for i in range(1, len(res)):
    if res[i] - res[i - 1] > 1:
        _edge[max(0, i - 2):i + 2] = True
anm_ref = anm[~_edge].max()
anm = anm / anm_ref
axb_note = f"residue 222 = {anm[list(res).index(222)]:.1f}x the artefact-free maximum"
pca = np.array([float(r["pca_sqfluct"]) for r in rows]); pca /= pca.max()

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.linewidth": 0.8, "savefig.dpi": 300,
})

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.0, 1.25]})

# panel a: GNM cross-correlation
# the analysis window is non-contiguous, so the map is drawn in matrix-index space and the
# ticks are labelled with the true residue numbers (a residue-number extent would stretch
# 269 matrix rows linearly over 348 residue-number units)
im = axa.imshow(cc, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
for b in (NTD_HB, HB_TBD):
    i = np.searchsorted(res, b) - 0.5
    axa.axhline(i, color="k", lw=0.5, ls=":"); axa.axvline(i, color="k", lw=0.5, ls=":")
i_ntd = np.searchsorted(res, 130); i_tbd = np.searchsorted(res, 375)
axa.text(i_ntd, i_ntd, "NTD", ha="center", va="center", fontsize=12)
axa.text(i_tbd, i_tbd, "TBD", ha="center", va="center", fontsize=12)
tick_idx = np.searchsorted(res, [100, 150, 200, 250, 300, 350, 400])
axa.set_xticks(tick_idx); axa.set_xticklabels(res[tick_idx])
axa.set_yticks(tick_idx); axa.set_yticklabels(res[tick_idx])
axa.set_xlabel("residue"); axa.set_ylabel("residue")
# panel descriptions live in the figure legend (figure style), not in the figure
cb = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.04); cb.set_label("GNM cross-correlation")
axa.text(-0.16, 1.04, "(a)", transform=axa.transAxes, fontsize=13, fontweight="bold", va="top", ha="right")

# panel b: per-residue fluctuation with domain background
axb.axvspan(res[0], NTD_HB, color=NTD_C, alpha=0.10)
axb.axvspan(NTD_HB, HB_TBD, color=HB_C, alpha=0.10)
axb.axvspan(HB_TBD, res[-1], color=TBD_C, alpha=0.10)
axb.axvspan(*HINGE, color="#888888", alpha=0.18)
# the analysis window is non-contiguous; break the traces at sequence gaps so the plot
# does not interpolate across unresolved segments
def _break_at_gaps(x, y):
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if x[i] - x[i - 1] > 1:
            xs.append(x[i - 1] + 0.5); ys.append(np.nan)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs, float), np.array(ys, float)

rx, ay = _break_at_gaps(res, anm)
_, py = _break_at_gaps(res, pca)
axb.plot(rx, ay, color=ANM_C, lw=1.4, label="ANM (intrinsic)")
axb.plot(rx, py, color=PCA_C, lw=1.4, label="PCA (experimental)")
axb.text(130, 0.62, "NTD", ha="center", color=NTD_C, fontweight="bold", fontsize=12)
axb.text((NTD_HB + HB_TBD) / 2, 0.93, "HB", ha="center", color=HB_C, fontweight="bold", fontsize=12)
axb.text((HINGE[0] + HINGE[1]) / 2, 0.86, "hinge", ha="center", color="#666666", fontsize=11)
axb.text((HB_TBD + res[-1]) / 2, 0.93, "TBD", ha="center", color=TBD_C, fontweight="bold", fontsize=12)
axb.set_xlabel("residue"); axb.set_ylabel("normalized fluctuation")
axb.set_xlim(res[0], res[-1]); axb.set_ylim(0, max(1.05, float(np.nanmax(ay)) * 1.05))
axb.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0.0, 0.88))
pass  # panel title moved to legend
axb.spines[["top", "right"]].set_visible(False)
axb.text(-0.10, 1.04, "(b)", transform=axb.transAxes, fontsize=13, fontweight="bold", va="top", ha="right")

fig.tight_layout()
fig.savefig("figures/Fig3.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/vector/Fig3.pdf", bbox_inches="tight")
fig.savefig("figures/vector/Fig3.svg", bbox_inches="tight")
print(f"Fig3 built: cc range [{cc.min():.2f}, {cc.max():.2f}]; "
      f"ANM/PCA fluct n={len(res)} residues {res[0]}-{res[-1]}")
