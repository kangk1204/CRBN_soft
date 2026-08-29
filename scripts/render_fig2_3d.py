#!/usr/bin/env python3
"""Render the Fig 2 b/c 3D porcupine panels (PyMOL) from committed data only.

  ANM mode 1 (from open) -> figures/panels/fig2_anm3d.png
  Archive-derived PC1    -> figures/panels/fig2_pc13d.png

Ribbon = open 8CVP chain B (render/open_8cvp.pdb), domain-coloured
(NTD blue / HB green / TBD orange). Arrows = mode-vector porcupine on every
2nd Ca. Mode vectors are read from the committed NMD files
(data/crbn_anm_mode1.nmd, data/crbn_pca_modes.nmd), which live in the
ensemble-superposed frame, and are Kabsch-rotated into the 8CVP coordinate
frame so the arrows sit on the ribbon. Run inside the `crbn` env (PyMOL):

  pymol -cq scripts/render_fig2_3d.py
"""
import os
import numpy as np
from pymol import cmd, cgo

# pymol -cq rebinds __file__ to its exec context, so prefer an explicit repo root:
# FIG2_ROOT env var, else the cwd (run from the repo root).
ROOT = os.environ.get("FIG2_ROOT", os.getcwd())
PDB = os.path.join(ROOT, "render", "open_8cvp.pdb")
OUT = os.path.join(ROOT, "figures", "panels")
os.makedirs(OUT, exist_ok=True)

DOM = {"NTD": (0x3b/255, 0x6e/255, 0xa5/255),
       "HB":  (0x4b/255, 0xab/255, 0x8c/255),
       "TBD": (0xe0/255, 0x7b/255, 0x39/255)}
ARROW = (0.15, 0.15, 0.15)


def parse_nmd(path):
    coords = mode = resids = None
    for ln in open(path):
        t = ln.split()
        if not t:
            continue
        if t[0] == "coordinates":
            coords = np.array(t[1:], float).reshape(-1, 3)
        elif t[0] == "mode" and mode is None:
            mode = np.array(t[3:], float).reshape(-1, 3)
        elif t[0] in ("resnums", "resids"):
            resids = np.array(t[1:], int)
    return coords, mode, resids


def pdb_ca(resids):
    d = {}
    for ln in open(PDB):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == "B":
            d[int(ln[22:26])] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
    return np.array([d[r] for r in resids])


def kabsch(mob, ref):
    """rotation mapping mob-frame vectors into ref frame (least-squares on Ca)."""
    Pc = ref - ref.mean(0); Qc = mob - mob.mean(0)
    H = Qc.T @ Pc
    U, S, Vt = np.linalg.svd(H)
    Dg = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, Dg]) @ U.T


def draw(vec, ca, outfile, viewfile, save_view):
    cmd.reinitialize()
    cmd.load(PDB, "m")
    cmd.remove("not chain B"); cmd.remove("not polymer")
    cmd.hide("everything"); cmd.show("cartoon", "m")
    cmd.set("cartoon_transparency", 0.35); cmd.color("grey80", "m")
    cmd.set("bg_rgb", [1, 1, 1]); cmd.set("ray_opaque_background", 0)
    cmd.set("cartoon_fancy_helices", 1)
    for dn, (r, g, b) in DOM.items():
        cmd.set_color("c_" + dn, [r, g, b])
    cmd.color("c_NTD", "m and resi 77-186")
    cmd.color("c_HB",  "m and resi 187-317")
    cmd.color("c_TBD", "m and resi 318-424")
    lens = np.linalg.norm(vec, axis=1)
    scale = 9.0 / np.percentile(lens, 90)
    obj = []
    for i in range(0, len(ca), 2):
        p = ca[i]; v = vec[i] * scale
        if np.linalg.norm(v) < 0.6:
            continue
        q = p + v
        obj += [cgo.CYLINDER, *p, *(p + 0.75*v), 0.18, *ARROW, *ARROW]
        obj += [cgo.CONE, *(p + 0.72*v), *q, 0.45, 0.0, *ARROW, *ARROW, 1.0, 1.0]
    cmd.load_cgo(obj, "arrows")
    if save_view:
        cmd.orient("m"); cmd.turn("y", 20); cmd.turn("x", -10)
        open(viewfile, "w").write("\n".join(str(x) for x in cmd.get_view()))
    else:
        cmd.set_view([float(x) for x in open(viewfile).read().split("\n")])
    cmd.set("ray_trace_mode", 1); cmd.set("ray_trace_color", "black")
    cmd.ray(1500, 1400); cmd.png(outfile, dpi=300)
    print("wrote", outfile)


ca_nmd, anm, res = parse_nmd(os.path.join(ROOT, "data", "crbn_anm_mode1.nmd"))
_, pc1, _ = parse_nmd(os.path.join(ROOT, "data", "crbn_pca_modes.nmd"))
ca = pdb_ca(res)
R = kabsch(ca_nmd, ca)                       # ensemble frame -> 8CVP frame
anm_r = (R @ anm.T).T
pc1_r = (R @ pc1.T).T
# consistent sign convention: net displacement points +x in the 8CVP frame
if anm_r.sum(0)[0] < 0:
    anm_r = -anm_r
if pc1_r.sum(0)[0] < 0:
    pc1_r = -pc1_r

VIEW = os.path.join(OUT, "fig2_view.txt")
draw(anm_r, ca, os.path.join(OUT, "fig2_anm3d.png"), VIEW, True)
draw(pc1_r, ca, os.path.join(OUT, "fig2_pc13d.png"), VIEW, False)
print("Fig2 3D porcupine renders done (ANM mode 1 + archive-derived PC1)")
