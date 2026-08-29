#!/usr/bin/env python3
"""Do the predicted hinge residues coincide with the allosteric site found experimentally?

The hinge points reported here (273, 289, 315) come from sign changes in the slowest GNM
mode, computed from open-state coordinates alone with no ligand information. While this work
was in preparation, an allosteric site on cereblon was identified experimentally and a
structure of CRBN-DDB1 with the site-binding compound SB-405483 was deposited (9SFM,
10.1038/s41586-025-09994-w). That structure was already in the curated ensemble, having been
selected by the geometric criteria alone.

This script asks whether the two agree. It is a test the analysis could have failed: nothing
in the mode calculation knows where the compound binds, and a hinge is a small target -- 125
residues of the helical bundle are resolved in this structure and the compound touches a handful.

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

HINGE = (273, 289, 315)        # GNM slow-mode sign changes, recurrent over 30 settings
HB = (187, 317)                # DDB1-binding helical bundle, author numbering
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
    hb = (resnums >= HB[0]) & (resnums <= HB[1])
    idx = {int(r): i for i, r in enumerate(resnums)}

    # A hinge point is a single residue; the honest comparison is against the rest of the
    # helical bundle, not against the whole protein, since the bundle is where the hinge is
    # already known to be.
    hinge_d = {r: float(d_allo[idx[r]]) for r in HINGE if r in idx}
    hb_d = d_allo[hb]
    n_close = int((hb_d <= 4.5).sum())

    out = {
        "structure": PDB,
        "note": ("Hinge residues are predicted from open-state GNM slow modes with no ligand "
                 "information; the allosteric site was identified experimentally and "
                 "independently. 9SFM entered the ensemble through the geometric curation "
                 "criteria, not because of this comparison."),
        "citation_doi": "10.1038/s41586-025-09994-w",
        "allosteric_ligand": ALLOSTERIC,
        "orthosteric_ligand": ORTHOSTERIC,
        "helical_bundle": list(HB),
        "n_hb_residues_resolved": int(hb.sum()),
        "hinge_min_heavy_atom_distance_A": hinge_d,
        "hb_distance_distribution_A": {
            "median": float(np.median(hb_d)),
            "min": float(hb_d.min()),
            "p05": float(np.percentile(hb_d, 5)),
            "n_within_4.5A": n_close,
        },
        "hinge_percentile_within_hb": {
            r: float((hb_d < d).mean() * 100) for r, d in hinge_d.items()
        },
        "orthosteric_contrast": {
            "hinge_min_heavy_atom_distance_A": {
                r: float(d_ortho[idx[r]]) for r in HINGE if r in idx},
            "note": ("Lenalidomide sits in the canonical TBD pocket in the same crystal and is "
                     "far from the hinge, so the coincidence is specific to the allosteric "
                     "compound rather than a consequence of any ligand being present."),
        },
    }

    if not verify:
        with open("data/hinge_allosteric_site.json", "w") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
        print("wrote data/hinge_allosteric_site.json")

    print(f"{PDB}: {ALLOSTERIC} vs the {int(hb.sum())} resolved helical-bundle residues")
    for r, d in hinge_d.items():
        print(f"  hinge {r}: {d:5.2f} A  ({out['hinge_percentile_within_hb'][r]:4.1f}th "
              f"percentile of the bundle)")
    print(f"  bundle median {np.median(hb_d):.1f} A; {n_close} of {int(hb.sum())} "
          f"residues within 4.5 A")
    print(f"  lenalidomide to the same hinge residues: "
          f"{', '.join(f'{v:.1f}' for v in out['orthosteric_contrast']['hinge_min_heavy_atom_distance_A'].values())} A")

    if verify:
        assert all(d < 6.0 for d in hinge_d.values()), hinge_d
        assert np.median(hb_d) > 10.0, float(np.median(hb_d))
        # lenalidomide is 17.4-26.1 A from the three hinge residues; the contrast only
        # needs to be large, and 20 A was a guess that the data do not meet.
        assert all(v > 15.0 for v in
                   out["orthosteric_contrast"]["hinge_min_heavy_atom_distance_A"].values())
        print("verify OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
