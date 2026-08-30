#!/usr/bin/env python3
"""Post hoc proximity of the kinematic CRBN hinge to the reported allosteric site.

The axis-proximal residues reported here (316-320) come from the endpoint Kabsch screw-axis
analysis in ``data/hinge_geometry.json``.  That analysis uses no ligand coordinates.  The
distances below are a post hoc structural description in CRBN-DDB1 structure 9SFM, which
contains the site-binding compound SB-405483 (10.1038/s41586-025-09994-w).

The earlier GNM sign-change candidates are not used: one crossed an unresolved sequence gap,
and scalar correlation nodes do not define a three-dimensional rotation axis.

Reported for contrast: lenalidomide, bound in the same crystal at the canonical TBD pocket,
which should be far from the hinge if the two sites are distinct.

Usage
  python scripts/hinge_allosteric_site.py [--verify]
Input   RCSB 9SFM mmCIF (cached under data/_cif_cache outside --verify)
Output  data/hinge_allosteric_site.json
"""
import gzip
import json
import os
import sys
import urllib.request

import numpy as np

try:
    from pdb_id import validate_pdb_id
except ModuleNotFoundError:  # imported by path from the repository root
    from scripts.pdb_id import validate_pdb_id

HINGE = (316, 317, 318, 319, 320)  # <=2.5 A from the endpoint-derived screw axis
JUNCTION = (187, 320)              # helical bundle plus immediate TBD boundary
ALLOSTERIC = "A1CEG"           # SB-405483
ORTHOSTERIC = "LVY"            # lenalidomide, canonical TBD pocket
PDB = "9SFM"
CRBN_CHAIN = "B"
CACHE = os.environ.get("CRBN_CIF_CACHE", "data/_cif_cache")
CACHE_WRITES_ENABLED = True


def fetch_cif(pdb):
    pdb = validate_pdb_id(pdb)
    p = os.path.join(CACHE, f"{pdb}.cif.gz")
    if os.path.exists(p):
        with gzip.open(p, "rt") as fh:
            return fh.read()
    with urllib.request.urlopen(
            f"https://files.rcsb.org/download/{pdb}.cif.gz", timeout=120) as fh:
        blob = fh.read()
    text = gzip.decompress(blob).decode("utf-8")
    if CACHE_WRITES_ENABLED:
        os.makedirs(CACHE, exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(blob)
    return text


def atom_site(cif):
    """Yield (group, comp, chain, resnum, element, xyz) for every atom record."""
    lines = cif.splitlines()
    hdr, start = [], None
    for i, l in enumerate(lines):
        if l.strip() == "loop_":
            j, h = i + 1, []
            while j < len(lines) and lines[j].lstrip().startswith("_atom_site."):
                h.append(lines[j].strip()); j += 1
            if h:
                hdr, start = h, j
                break
    col = {h.split(".")[1]: k for k, h in enumerate(hdr)}
    for l in lines[start:]:
        if l.startswith(("#", "_")) or l.strip() in ("", "loop_"):
            break
        f = l.split()
        if len(f) < len(hdr):
            continue
        try:
            xyz = (float(f[col["Cartn_x"]]), float(f[col["Cartn_y"]]), float(f[col["Cartn_z"]]))
        except ValueError:
            continue
        rn = f[col["auth_seq_id"]]
        yield (f[col["group_PDB"]], f[col["label_comp_id"]], f[col["auth_asym_id"]],
               int(rn) if rn.lstrip("-").isdigit() else None,
               f[col["type_symbol"]], xyz)


def main():
    global CACHE_WRITES_ENABLED
    verify = "--verify" in sys.argv
    CACHE_WRITES_ENABLED = not verify
    cif = fetch_cif(PDB)
    prot, lig = {}, {}
    for group, comp, chain, rn, el, xyz in atom_site(cif):
        if el == "H":
            continue                                   # heavy-atom distances only
        if group == "ATOM" and chain == CRBN_CHAIN and rn is not None:
            prot.setdefault(rn, []).append(xyz)
        elif group == "HETATM" and comp in (ALLOSTERIC, ORTHOSTERIC):
            lig.setdefault(comp, []).append(xyz)

    assert prot, "no CRBN protein atoms found"
    for code in (ALLOSTERIC, ORTHOSTERIC):
        assert code in lig, f"{code} not present in {PDB}"

    resnums = np.array(sorted(prot))
    coords = {r: np.array(v) for r, v in prot.items()}

    def min_dist(code):
        L = np.array(lig[code])
        return np.array([float(np.sqrt(((coords[r][:, None, :] - L[None, :, :]) ** 2)
                                       .sum(-1)).min()) for r in resnums])

    d_allo = min_dist(ALLOSTERIC)
    d_ortho = min_dist(ORTHOSTERIC)
    junction = (resnums >= JUNCTION[0]) & (resnums <= JUNCTION[1])
    idx = {int(r): i for i, r in enumerate(resnums)}

    # The comparison distribution is the local HB/TBD junction rather than the whole fold.
    hinge_d = {r: float(d_allo[idx[r]]) for r in HINGE if r in idx}
    junction_d = d_allo[junction]
    n_close = int((junction_d <= 4.5).sum())

    out = {
        "structure": PDB,
        "note": ("Axis-proximal boundary residues were defined by the endpoint Kabsch screw "
                 "axis without ligand coordinates. Ligand distances in 9SFM are a post hoc "
                 "structural description, not independent validation."),
        "citation_doi": "10.1038/s41586-025-09994-w",
        "allosteric_ligand": ALLOSTERIC,
        "orthosteric_ligand": ORTHOSTERIC,
        "axis_proximal_boundary_residues": list(HINGE),
        "junction_comparison_region": list(JUNCTION),
        "n_junction_residues_resolved": int(junction.sum()),
        "hinge_min_heavy_atom_distance_A": hinge_d,
        "junction_distance_distribution_A": {
            "median": float(np.median(junction_d)),
            "min": float(junction_d.min()),
            "p05": float(np.percentile(junction_d, 5)),
            "n_within_4.5A": n_close,
        },
        "hinge_percentile_within_junction": {
            r: float((junction_d < d).mean() * 100) for r, d in hinge_d.items()
        },
        "orthosteric_contrast": {
            "hinge_min_heavy_atom_distance_A": {
                r: float(d_ortho[idx[r]]) for r in HINGE if r in idx},
            "note": ("Lenalidomide sits in the canonical TBD pocket in the same crystal and is "
                     "far from the same boundary residues; this is a within-structure distance "
                     "contrast and does not establish allosteric causality."),
        },
    }

    if not verify:
        with open("data/hinge_allosteric_site.json", "w") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
        print("wrote data/hinge_allosteric_site.json")

    print(f"{PDB}: {ALLOSTERIC} vs the {int(junction.sum())} resolved junction residues")
    for r, d in hinge_d.items():
        print(f"  axis-proximal {r}: {d:5.2f} A  "
              f"({out['hinge_percentile_within_junction'][r]:4.1f}th "
              f"percentile of the junction)")
    print(f"  junction median {np.median(junction_d):.1f} A; "
          f"{n_close} of {int(junction.sum())} "
          f"residues within 4.5 A")
    print(f"  lenalidomide to the same hinge residues: "
          f"{', '.join(f'{v:.1f}' for v in out['orthosteric_contrast']['hinge_min_heavy_atom_distance_A'].values())} A")

    if verify:
        assert set(hinge_d) == set(HINGE), hinge_d
        assert all(d < 7.1 for d in hinge_d.values()), hinge_d
        assert np.median(junction_d) > 10.0, float(np.median(junction_d))
        assert all(v > 20.0 for v in
                   out["orthosteric_contrast"]["hinge_min_heavy_atom_distance_A"].values())
        print("verify OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
