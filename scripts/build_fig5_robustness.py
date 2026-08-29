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
from figure_package_utils import prepare_figure_dirs, require_rigid_null_schema

FIGURES, VECTOR, _ = prepare_figure_dirs()

import json, math, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rob=json.load(open("data/anm_robustness.json")); nul=json.load(open("data/anm_null_significance.json"))
arn=json.load(open("data/assembly_rigid_null.json"))
ens=np.load("data/crbn_ensemble.ens.npz",allow_pickle=False)
confs=ens["_confs"]; labels=[str(x) for x in ens["_labels"]]
# Derive the open set from the canonical difference artifact and require exact agreement
# with the robustness artifact. Missing or stale classification data must not silently fall
# back to a literal list.
dv=np.load("data/pca_diffvec.npz",allow_pickle=False)
if not {"labels","open_mask"}.issubset(dv.files):
    raise ValueError("data/pca_diffvec.npz lacks labels/open_mask; rerun reproduce_modes.py")
dv_labels=np.asarray([str(value) for value in dv["labels"]])
dv_mask=np.asarray(dv["open_mask"])
if dv_mask.dtype.kind != "b" or dv_mask.shape != dv_labels.shape:
    raise ValueError("data/pca_diffvec.npz open_mask is not a matching boolean vector")
OPENL=[str(value) for value in rob["open_set"]]
derived_open={label for label,is_open in zip(dv_labels,dv_mask) if is_open}
if len(OPENL) != len(set(OPENL)) or set(OPENL) != derived_open:
    raise ValueError(f"open-set mismatch between robustness and difference artifacts: {OPENL} vs {sorted(derived_open)}")
mask=np.array([l in OPENL for l in labels])
if int(mask.sum()) != len(OPENL):
    raise ValueError("ensemble does not contain each canonical open structure exactly once")
dvec=(confs[mask].mean(0)-confs[~mask].mean(0)).reshape(-1); dvec/=np.linalg.norm(dvec)
# Panel b null: exact absolute-directional-cosine distributions INSIDE each
# rigid-motion subspace. If C is the absolute cosine between a fixed direction
# and a uniformly random unit direction in d dimensions, then
# C^2 ~ Beta(1/2, (d-1)/2). Plotting the closed-form density keeps the figure
# deterministic and avoids presenting exact inference as a finite simulation.
_rd=require_rigid_null_schema(arn)
_null_x=np.linspace(0.0,1.0,1001)
def _null_density(dim):
    b=0.5*(dim-1)
    normaliser=2.0*math.exp(math.lgamma(0.5+b)-math.lgamma(0.5)-math.lgamma(b))
    return normaliser*np.power(np.clip(1.0-_null_x**2,0.0,None),b-1.0)
def _clean_svg(path):
    with open(path, encoding="utf-8") as f:
        txt=f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(line.rstrip() for line in txt.splitlines()) + "\n")
# All reported rigid nulls are drawn. Bond-length preservation is the literal first-order
# connectivity condition; equal boundary displacement is the stronger orientation-freezing
# sensitivity. Both remain visible so the panel does not imply universal significance.
_bond=_rd["bond_length_preserving_boundary"]
_equal=_rd["equal_displacement_boundary"]
_two=_rd["two_block"]
_three=_rd["three_block"]
null_rigid=_null_density(_two["internal_dim"])
null_rigid3=_null_density(_three["internal_dim"])
null_bond=_null_density(_bond["internal_dim"])
null_equal=_null_density(_equal["internal_dim"])

C_OPEN="#3b6ea5"; C_CLOSED="#e07b39"; C_NULL="#b8b8b8"; C_OBS="#D55E00"; META="#555555"; C_BOND="#0072B2"; C_EQUAL="#CC79A7"
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
_p2=_two["p_exact"]; _p3=_three["p_exact"]
_pb=_bond["p_exact"]; _pe=_equal["p_exact"]
_o2=_two["observed_direction_cosine_in_subspace"]
_o3=_three["observed_direction_cosine_in_subspace"]
_ob=_bond["observed_direction_cosine_in_subspace"]
_oe=_equal["observed_direction_cosine_in_subspace"]
axB.fill_between(_null_x,null_rigid,color=C_NULL,alpha=0.65,zorder=2,
         label=f"2-lobe (obs={_o2:.2f}; p={_p2:.3f})")
axB.plot(_null_x,null_rigid3,color=META,lw=1.0,zorder=3,
         label=f"3-domain (obs={_o3:.2f}; p={_p3:.3f})")
axB.plot(_null_x,null_bond,color=C_BOND,lw=1.3,zorder=4,
         label=f"bond-length (obs={_ob:.2f}; p={_pb:.3f})")
axB.plot(_null_x,null_equal,color=C_EQUAL,lw=1.3,zorder=4,
         label=f"equal displacement (obs={_oe:.2f}; p={_pe:.3f})")
for observed,color,style in ((_o2,C_OBS,"-"),(_o3,META,"--"),
                             (_ob,C_BOND,"-."),(_oe,C_EQUAL,":")):
    axB.axvline(observed,color=color,lw=1.5,ls=style,zorder=5)
axB.legend(fontsize=7.5,frameon=True,facecolor="white",edgecolor="none",framealpha=0.92,
    loc="upper left",bbox_to_anchor=(0.0,0.99),borderpad=0.25,labelspacing=0.25)
axB.set_xlabel("Matched-subspace mode-1 direction cosine")
axB.set_ylabel("Probability density (exact null)"); axB.set_xlim(0,1.03)
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
fig.savefig(FIGURES / "Fig5.png",dpi=300); fig.savefig(VECTOR / "Fig5.pdf"); fig.savefig(VECTOR / "Fig5.svg")
_clean_svg(VECTOR / "Fig5.svg")
print("Fig5 written")
