#!/usr/bin/env python3
"""Domain-coloured open|closed overview cartoons.

Renders the open (8CVP) and closed (5FQD) chain-B structures as cartoons
coloured by domain (NTD blue / HB green / TBD orange), with the closed
structure superposed onto the open one over the shared NTD so the two are
shown in a matched orientation and the TBD swing is visible. Committed data
only (render/open_8cvp.pdb, render/closed_5fqd.pdb). Run from the repo root:

  pymol -cq scripts/render_open_closed_overview.py
"""
import os
from pymol import cmd

ROOT = os.environ.get("FIG1_ROOT", os.getcwd())
OPEN = os.path.join(ROOT, "render", "open_8cvp.pdb")
CLOSED = os.path.join(ROOT, "render", "closed_5fqd.pdb")
OUT = os.path.join(ROOT, "figures", "panels")
os.makedirs(OUT, exist_ok=True)

DOM = {"NTD": ("resi 1-186",  [0x3b/255, 0x6e/255, 0xa5/255]),
       "HB":  ("resi 187-317",[0x4b/255, 0xab/255, 0x8c/255]),
       "TBD": ("resi 318-500",[0xe0/255, 0x7b/255, 0x39/255])}


def style(obj):
    cmd.hide("everything", obj); cmd.show("cartoon", obj)
    cmd.set("cartoon_transparency", 0.0, obj)
    for dn, (sel, rgb) in DOM.items():
        cmd.set_color("d_" + dn, rgb)
        cmd.color("d_" + dn, f"{obj} and {sel}")


def render_one(pdb, name, outfile, ref=None):
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.load(pdb, name)
    cmd.remove(f"{name} and not chain B")
    cmd.remove(f"{name} and not polymer")
    style(name)
    cmd.orient(name)
    cmd.turn("y", 20); cmd.turn("x", -10)
    cmd.set("ray_trace_mode", 1); cmd.set("ray_trace_color", "black")
    cmd.ray(1400, 1300); cmd.png(outfile, dpi=300)
    print("wrote", outfile)


render_one(OPEN, "op", os.path.join(OUT, "overview_open.png"))
render_one(CLOSED, "cl", os.path.join(OUT, "overview_closed.png"))
print("open/closed domain-coloured overview done")
