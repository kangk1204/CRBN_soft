#!/usr/bin/env python3
"""Build main Fig 5: ANM intrinsic-prediction robustness.
Panels: (a) endpoint sweep, (b) rigid-domain null, (c) cutoff stability, (d) leave-one-out.

Panel (b) shows the null that can fail. The isotropic null it replaces is passed by
any structured collective direction (closed-form tail 2e-143) and is kept only in
data/anm_null_significance.json; the transition is a rigid interdomain swing, so the
question worth asking is whether mode 1 picks the right axis INSIDE that space.

Inputs: data/anm_robustness.json, data/anm_null_significance.json,
        data/assembly_rigid_null.json, data/crbn_ensemble.ens.npz
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rob=json.load(open("data/anm_robustness.json")); nul=json.load(open("data/anm_null_significance.json"))
arn=json.load(open("data/assembly_rigid_null.json"))
ens=np.load("data/crbn_ensemble.ens.npz",allow_pickle=False)
confs=ens["_confs"]; labels=[str(x) for x in ens["_labels"]]
# Canonical open set: derived from data/pca_diffvec.npz (built by reproduce_modes.py),
# the same source the robustness scripts use; literal list kept only as an asserted fallback.
_OPEN_FALLBACK=["8CVP","8D7X","8D7Y","6H0F","7U8F"]
def _canonical_open():
    import os
    p=os.path.join("data","pca_diffvec.npz")
    if os.path.exists(p):
        d=np.load(p)
        s=sorted(str(l) for l,m in zip(d["labels"],d["open_mask"]) if m)
        assert s==sorted(_OPEN_FALLBACK), f"open set drift: {s} vs {_OPEN_FALLBACK}"
        # fixed canonical ORDER so Fig 5a x-axis is byte-stable across runs
        return list(_OPEN_FALLBACK)
    return list(_OPEN_FALLBACK)
OPENL=_canonical_open()
mask=np.array([l in OPENL for l in labels])
dvec=(confs[mask].mean(0)-confs[~mask].mean(0)).reshape(-1); dvec/=np.linalg.norm(dvec)
# Panel b null: random directions INSIDE the rigid interdomain subspace, reconstructed
# from the committed capture and the observed value so the figure needs no eigenvectors.
_rd=arn["rigid_domain_null"]
rng=np.random.default_rng(20260720)
def _null(dim,capture):
    r=rng.standard_normal((_rd["n_draws"],dim)); r/=np.linalg.norm(r,axis=1,keepdims=True)
    return np.abs(r[:,0])*capture
def _clean_svg(path):
    with open(path, encoding="utf-8") as f:
        txt=f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(line.rstrip() for line in txt.splitlines()) + "\n")
# All reported rigid nulls are drawn. The continuity-constrained null is the hardest
# calibration because it keeps the chain joined at the HB-TBD boundary; it must be as
# visible as the two significant nulls so the panel does not imply universal significance.
null_rigid=_null(_rd["two_block_internal_dim"],_rd["two_block_capture"])
null_rigid3=_null(_rd["three_block_internal_dim"],_rd["three_block_capture"])
_cc=_rd["connectivity_constrained"]
null_continuity=_null(_cc["internal_dim"],_cc["subspace_capture_of_transition"])

C_OPEN="#3b6ea5"; C_CLOSED="#e07b39"; C_NULL="#b8b8b8"; C_OBS="#c0392b"; META="#555555"; C_CONT="#4bab8c"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.linewidth":0.8,
    "xtick.major.width":0.8,"ytick.major.width":0.8,"axes.spines.top":False,"axes.spines.right":False,
    "pdf.fonttype":42,"ps.fonttype":42})
fig,axs=plt.subplots(2,2,figsize=(8.2,6.6)); axA,axB,axC,axD=axs.flat
OPENs=rob["open_set"]; CLOSEDs=rob["closed_endpoints"]; labs=OPENs+CLOSEDs
m1=[rob["table"][l]["15.0"]["mode1_overlap"] for l in labs]
best=[rob["table"][l]["15.0"]["best_overlap"] for l in labs]
rank=[rob["table"][l]["15.0"]["best_mode_rank"] for l in labs]
x=np.arange(len(labs)); cols=[C_OPEN]*len(OPENs)+[C_CLOSED]*len(CLOSEDs)
axA.scatter(x,m1,s=46,c=cols,zorder=3); axA.scatter(x,best,s=46,facecolors="none",edgecolors=cols,linewidths=1.4,zorder=3)
for xi,b,r in zip(x,best,rank):
    if r>1: axA.annotate(f"m{r}",(xi,b),textcoords="offset points",xytext=(0,6),ha="center",fontsize=7.5,color=META)
for xi,a,b,r in zip(x,m1,best,rank):
    if r>1: axA.plot([xi,xi],[a,b],color=C_CLOSED,lw=0.8,alpha=0.5,zorder=2)
axA.axhspan(0.73,0.77,color=C_OPEN,alpha=0.10,zorder=0)
axA.set_xticks(x); axA.set_xticklabels(labs,rotation=45,ha="right",fontsize=7.5)
axA.set_ylabel("Directional overlap with open–closed axis"); axA.set_ylim(0,0.85)

axA.legend(handles=[Line2D([],[],marker='o',ls='',mfc=C_OPEN,mec=C_OPEN,label='open endpoint'),
    Line2D([],[],marker='o',ls='',mfc=C_CLOSED,mec=C_CLOSED,label='closed endpoint'),
    Line2D([],[],marker='o',ls='',mfc='w',mec=META,label='best-mode directional overlap')],fontsize=7.5,frameon=False,loc="lower left")
_obs=_rd["observed_mode1_overlap"]
_p2=_rd["two_block"]["p_empirical"]; _p3=_rd["three_block"]["p_empirical"]; _pc=_cc["p_empirical"]
axB.hist(null_rigid,bins=60,color=C_NULL,edgecolor="none",alpha=0.85,zorder=2,
         label=f"2-lobe null (p={_p2:.3f})")
axB.hist(null_rigid3,bins=60,histtype="step",color=META,lw=1.0,zorder=3,
         label=f"3-domain null (p={_p3:.4f})")
axB.hist(null_continuity,bins=60,histtype="step",color=C_CONT,lw=1.3,zorder=4,
         label=f"joined-chain null (p={_pc:.2f})")
axB.axvline(_rd["two_block_capture"],color=META,lw=1.2,ls=":",zorder=3)
axB.annotate(f"rigid-subspace\nprojection norm {_rd['two_block_capture']:.2f}",(_rd["two_block_capture"],axB.get_ylim()[1]*0.34),
    ha="right",va="top",xytext=(-16,-6),textcoords="offset points",fontsize=7.5,color=META)
axB.axvline(_cc["subspace_capture_of_transition"],color=C_CONT,lw=1.2,ls=":",zorder=3)
axB.annotate("joined-chain subspace\nprojection norm",(_cc["subspace_capture_of_transition"],axB.get_ylim()[1]*0.72),
    ha="right",va="top",xytext=(-6,0),textcoords="offset points",fontsize=7.5,color=C_CONT)
axB.axvline(_obs,color=C_OBS,lw=1.6,zorder=4)
axB.annotate(f"ANM mode 1\n{_obs:.2f}",(_obs,axB.get_ylim()[1]*0.55),
    ha="right",va="center",xytext=(-6,0),textcoords="offset points",fontsize=7.5,color=C_OBS)
axB.legend(fontsize=7.5,frameon=True,facecolor="white",edgecolor="none",framealpha=0.92,
    loc="upper left",bbox_to_anchor=(0.0,0.99),borderpad=0.25,labelspacing=0.25)
axB.set_xlabel("Directional overlap of a random rigid interdomain motion\nwith the open–closed axis")
axB.set_ylabel(f"Count ({_rd['n_draws']:,} draws)"); axB.set_xlim(0,1.03)
cuts=[float(c) for c in rob["cutoffs"]]
for l in OPENs:
    axC.plot(cuts,[rob["table"][l][str(c)]["mode1_overlap"] for c in rob["cutoffs"]],"-o",color=C_OPEN,alpha=0.55,ms=3.5,lw=1.0)
axC.plot(cuts,[np.mean([rob["table"][l][str(c)]["mode1_overlap"] for l in OPENs]) for c in rob["cutoffs"]],"-",color=C_OBS,lw=1.8,zorder=4,label="open mean")
axC.axvspan(15,18,color=C_OPEN,alpha=0.08); axC.set_xlabel("ANM contact cutoff (Å)"); axC.set_ylabel("Mode-1 directional overlap")
axC.set_ylim(0.2,0.85); axC.legend(fontsize=7.5,frameon=False,loc="lower right")
loo_c=nul["leave_one_closed_out"]; loo_o=nul["leave_one_open_out"]; xd=np.arange(2)
means=[loo_c["mean"],loo_o["mean"]]; los=[loo_c["min"],loo_o["min"]]; his=[loo_c["max"],loo_o["max"]]
axD.errorbar(xd,means,yerr=[np.array(means)-np.array(los),np.array(his)-np.array(means)],fmt='o',ms=8,color=C_OPEN,ecolor=META,elinewidth=1.2,capsize=4,zorder=3)
axD.axhline(0.744,color=C_OBS,lw=1.0,ls="--",alpha=0.7,zorder=2)
axD.annotate("full-ensemble 0.744",(1.4,0.744),fontsize=7.5,color=C_OBS,va="bottom",ha="right")
axD.set_xticks(xd); axD.set_xticklabels(["drop one\nclosed (n=65)","drop one\nopen (n=5)"],fontsize=7.5)
axD.set_xlim(-0.5,1.5); axD.set_ylim(0.70,0.78); axD.set_ylabel("Mode-1 directional overlap")
for ax,l in zip([axA,axB,axC,axD],"abcd"):
    ax.text(-0.14,1.06,f"({l})",transform=ax.transAxes,fontweight="bold",fontsize=11,va="top",ha="right")
# left margin widened so the right-aligned (a)/(c) letters clear the figure edge
fig.subplots_adjust(left=0.105,right=0.985,top=0.94,bottom=0.10,hspace=0.42,wspace=0.30)
fig.savefig("figures/Fig5.png",dpi=300); fig.savefig("figures/vector/Fig5.pdf"); fig.savefig("figures/vector/Fig5.svg")
_clean_svg("figures/vector/Fig5.svg")
print("Fig5 written")
