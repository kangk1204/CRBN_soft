#!/usr/bin/env python3
"""Fig 4c: closed-state drug pocket, backbone coloured by ANM mobility.

Reproducible from committed data only:
  render/closed_5fqd_lig.pdb    5FQD chain B: protein + S-lenalidomide (LVY) + Zn
  data/crbn_anm_modes.npz       ANM eigenvectors/eigenvalues -> per-residue mobility

Mobility is the raw ANM square fluctuation normalised over the rendered TBD. Normalising
over the whole chain would let the helical-bundle peak (residue 222, outside this view)
compress the pocket into one end of the ramp and hide the drug-loop / zinc-site contrast
that panels a and b quantify.

Bound S-lenalidomide is drawn as yellow sticks so the pocket is identifiable; the
structural Zn is a grey sphere. Run from the repo root:
  pymol -cq scripts/render_fig4_pocket.py
"""
import os
import numpy as np
from pymol import cmd

ROOT = os.environ.get("FIG4_ROOT", os.getcwd())
PDB = os.path.join(ROOT, "render", "closed_5fqd_lig.pdb")
NPZ = os.path.join(ROOT, "data", "crbn_anm_modes.npz")
OUT = os.path.join(ROOT, "figures", "panels", "render_closed_pocket.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

DRUG = [378, 380, 386]          # thalidomide-binding residues (Fig 4a,b)
ZN = [323, 326, 391, 394]       # structural Zn-coordinating cysteines
TBD_LO, TBD_HI = 318, 424

# ---- per-residue ANM square fluctuation over the 10 slowest modes (same as Fig 4a) ----
am = np.load(NPZ)
resnums = am["resnums"]
ev, ew = am["anm_eigvecs"], am["anm_eigvals"]
sqf = np.zeros(len(resnums))
for m in range(10):
    v = ev[:, m].reshape(-1, 3)
    sqf += (v ** 2).sum(1) / ew[m]

# normalise the raw square fluctuation over the rendered region (TBD). Using the raw value
# (not a rank) keeps the colour proportional to the quantity Fig 4a plots; normalising over
# the TBD rather than the whole chain avoids the helical-bundle peak at residue 222
# flattening the whole pocket into one end of the ramp.
tbd_mask = (resnums >= TBD_LO) & (resnums <= TBD_HI)
mob = sqf / float(sqf[tbd_mask].max())

cmd.reinitialize()
cmd.bg_color("white")
cmd.set("ray_opaque_background", 0)
cmd.load(PDB, "cl")
cmd.remove("cl and not chain B")
cmd.remove("cl and hydro")

# render only the TBD (plus its bound ligand and Zn)
cmd.create("pocket", "cl and polymer and resi %d-%d" % (TBD_LO, TBD_HI))
cmd.create("lig", "cl and resn LVY")
cmd.create("zn", "cl and resn ZN")
cmd.delete("cl")

cmd.hide("everything")
cmd.show("cartoon", "pocket")
cmd.set("cartoon_transparency", 0.0)

# colour the backbone by within-TBD mobility percentile
cmd.alter("pocket", "b=0.0")
for rn, mv in zip(resnums, mob):
    if TBD_LO <= int(rn) <= TBD_HI:
        cmd.alter("pocket and resi %d" % int(rn), "b=%f" % float(mv))
cmd.sort()
cmd.spectrum("b", "blue_white_red", "pocket and name CA", minimum=0.0, maximum=1.0)

# functional-site side chains
sel_drug = "pocket and resi " + "+".join(map(str, DRUG)) + " and not name N+C+O"
sel_zn = "pocket and resi " + "+".join(map(str, ZN)) + " and not name N+C+O"
cmd.show("sticks", sel_drug); cmd.color("orange", sel_drug)
cmd.show("sticks", sel_zn);   cmd.color("gray20", sel_zn)

# bound ligand + structural zinc
cmd.show("sticks", "lig"); cmd.color("yellow", "lig"); cmd.util.cnc("lig")
cmd.show("spheres", "zn"); cmd.color("gray50", "zn"); cmd.set("sphere_scale", 0.45, "zn")
cmd.set("stick_radius", 0.22)

# ---- claim-driven camera ----------------------------------------------------------------
# Fig 4c must show BOTH functional sites, separated, neither hidden behind the fold. Choose
# the view axis that maximises the in-plane separation of the drug-site and Zn-site
# centroids while keeping both toward the viewer, then build the PyMOL view matrix from it.
def _centroid(sel):
    m = cmd.get_model(sel)
    if not m.atom:
        raise RuntimeError("empty selection: " + sel)
    return np.mean([a.coord for a in m.atom], axis=0)

c_drug = _centroid("pocket and resi " + "+".join(map(str, DRUG)))
c_zn = _centroid("pocket and resi " + "+".join(map(str, ZN)))
com = _centroid("pocket")

rng = np.random.default_rng(0)          # fixed seed -> deterministic, reproducible view
best = None
for _ in range(20000):
    v = rng.normal(size=3); v /= np.linalg.norm(v)      # camera viewing direction
    up0 = np.array([0.0, 0.0, 1.0])
    if abs(float(v @ up0)) > 0.9:
        up0 = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(v, up0); e1 /= np.linalg.norm(e1)
    e2 = np.cross(v, e1)
    d = np.array([(c_drug - com) @ e1, (c_drug - com) @ e2]) - \
        np.array([(c_zn - com) @ e1, (c_zn - com) @ e2])
    sep = float(np.linalg.norm(d))
    front = min(float(-(c_drug - com) @ v), float(-(c_zn - com) @ v))
    score = sep + 0.6 * front
    if best is None or score > best[0]:
        best = (score, sep, v, e1, e2)
_, sep, v, e1, e2 = best

# PyMOL view matrix rows are the camera axes (right, up, backward) in model space.
right = e1 / np.linalg.norm(e1)
up = e2 / np.linalg.norm(e2)
back = -v / np.linalg.norm(v)
up = np.cross(back, right); up /= np.linalg.norm(up)     # re-orthogonalise
R = np.vstack([right, up, back])
view = list(R.flatten()) + [0.0, 0.0, -250.0] + list(com) + [50.0, 450.0, -20.0]
cmd.set_view(view)
cmd.zoom("pocket or lig or zn", buffer=3.0, complete=1)
print("view chosen: in-plane site separation %.1f A" % sep)

cmd.set("ray_trace_mode", 1)
cmd.set("ray_trace_color", "black")
cmd.set("antialias", 2)
cmd.ray(1500, 1250)
cmd.png(OUT, dpi=300)

# PyMOL leaves a wide transparent margin; crop to rendered content so the panel is filled.
try:
    from PIL import Image
    im = Image.open(OUT).convert("RGBA")
    bbox = im.split()[-1].getbbox()
    if bbox:
        pad = 12
        x0, y0, x1, y1 = bbox
        im.crop((max(0, x0 - pad), max(0, y0 - pad),
                 min(im.width, x1 + pad), min(im.height, y1 + pad))).save(OUT)
except Exception as e:
    print("crop skipped:", e)
print("wrote", OUT)
