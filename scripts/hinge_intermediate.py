#!/usr/bin/env python3
"""Hinge stability across structures/cutoffs and near-boundary depositions.

Two robustness checks for the structure-based mechanistic interpretation:

  1. GNM NODAL STABILITY. Sign changes of the slowest GNM eigenvector identify
     scalar correlation nodes, not a three-dimensional rotation axis. We
     recompute them (a) on several open structures and (b) over GNM cutoffs
     7-12 A. Only consecutive author residue numbers are eligible: array
     neighbours separated by an unresolved sequence gap are not covalent
     neighbours and cannot define a residue-level sign-change site.

  2. INTERMEDIATE PROJECTION. We project every ensemble member onto experimental
     PC1 and report where selected recent depositions (9NFQ, 9NFR, 9Y7D) and other
     near-boundary structures fall, showing the axis is a continuous open<->closed
     coordinate rather than a two-state switch. This is descriptive: we do not
     claim a populated free-energy intermediate, nor a state assignment for any
     individual deposition beyond its measured PC1 coordinate.

Outputs  data/hinge_intermediate.json
Usage:   python scripts/hinge_intermediate.py [--verify]
"""
import sys, json, csv
import numpy as np

# Canonical open set: derived from the data (widest-gap PC1 cut) and stored in
# data/pca_diffvec.npz by scripts/reproduce_modes.py. Loaded here so this script
# and the figures share one source; the literal list is a documented fallback and
# is asserted to match when the npz is present.
_OPEN_FALLBACK = ["8CVP", "8D7X", "8D7Y", "6H0F", "7U8F"]
def _canonical_open():
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "data", "pca_diffvec.npz")
    if os.path.exists(p):
        d = np.load(p)
        s = sorted(str(l) for l, m in zip(d["labels"], d["open_mask"]) if m)
        assert s == sorted(_OPEN_FALLBACK), f"open set drift: {s} vs {_OPEN_FALLBACK}"
        # Return the fixed canonical ORDER (not npz label order) so regenerated
        # products (e.g. Fig 5a x-axis) are byte-stable across runs.
        return list(_OPEN_FALLBACK)
    return list(_OPEN_FALLBACK)
OPEN = _canonical_open()
GNM_CUTOFFS = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

def gnm_kirchhoff(coords, cutoff):
    n = len(coords); K = np.zeros((n, n))
    for i in range(n):
        r = np.linalg.norm(coords - coords[i], axis=1)
        for j in range(i+1, n):
            if 1e-6 < r[j] <= cutoff:
                K[i, j] = K[j, i] = -1; K[i, i] += 1; K[j, j] += 1
    return K

def gnm_slow_mode(coords, cutoff):
    K = gnm_kirchhoff(coords, cutoff)
    w, v = np.linalg.eigh(K)
    nz = np.where(w > 1e-9)[0]
    return v[:, nz[0]]                  # slowest nonzero GNM mode

def hinge_residues(mode, resnums):
    """Consecutive residues at a sign change of the slow GNM mode."""
    mode = np.asarray(mode)
    resnums = np.asarray(resnums)
    if mode.ndim != 1 or resnums.ndim != 1 or mode.shape != resnums.shape:
        raise ValueError("mode and residue numbers must be matching one-dimensional arrays")
    if not np.isfinite(mode).all() or np.any(np.diff(resnums) <= 0):
        raise ValueError("mode must be finite and residue numbers strictly increasing")
    s = np.sign(mode); out = []
    for i in range(len(s)-1):
        if (resnums[i + 1] == resnums[i] + 1 and s[i] != 0 and s[i+1] != 0
                and s[i] != s[i+1]):
            out.append(int(resnums[i+1]))
    return out


def gap_crossing_sign_changes(mode, resnums):
    """Diagnostic sign changes that cross an unresolved author-number gap."""
    mode = np.asarray(mode)
    resnums = np.asarray(resnums)
    if mode.ndim != 1 or resnums.ndim != 1 or mode.shape != resnums.shape:
        raise ValueError("mode and residue numbers must be matching one-dimensional arrays")
    s = np.sign(mode); out = []
    for i in range(len(s) - 1):
        if (resnums[i + 1] > resnums[i] + 1 and s[i] != 0 and s[i + 1] != 0
                and s[i] != s[i + 1]):
            out.append({"left": int(resnums[i]), "right": int(resnums[i + 1])})
    return out

def main():
    verify = "--verify" in sys.argv
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    confs = ens["_confs"]; labels = [str(x) for x in ens["_labels"]]
    # 269-residue analysis window from the committed plain-text input, same
    # source reproduce_modes.py uses -- not the mode artifact.
    resnums = np.array([int(r["author_resnum"]) for r in
                        csv.DictReader(open("data/crbn_residue_window.csv"))])

    # 1. hinge stability across open structures x cutoffs
    from collections import Counter
    votes = Counter(); gap_votes = Counter(); settings = 0
    for lab in OPEN:
        coords = confs[labels.index(lab)]
        for co in GNM_CUTOFFS:
            m = gnm_slow_mode(coords, co)
            for r in hinge_residues(m, resnums):
                votes[r] += 1
            for gap in gap_crossing_sign_changes(m, resnums):
                gap_votes[(gap["left"], gap["right"])] += 1
            settings += 1
    # hinge points appearing in a majority of settings, grouped into contiguous bands
    majority = sorted(r for r, c in votes.items() if c >= settings/2)
    bands = []
    if majority:
        start = prev = majority[0]
        for r in majority[1:]:
            if r - prev <= 3: prev = r
            else: bands.append((start, prev)); start = prev = r
        bands.append((start, prev))

    # 2. intermediate projection onto archive-derived PC1
    X = (confs - confs.mean(0)).reshape(len(confs), -1)
    w, v = np.linalg.eigh(np.cov(X.T)); pc1 = v[:, np.argsort(w)[::-1][0]]
    s1 = X @ pc1
    if abs(s1.min()) > abs(s1.max()): s1 = -s1
    s1 = s1 / np.sqrt(confs.shape[1])
    proj = {labels[i]: float(s1[i]) for i in range(len(labels))}
    open_mean = np.mean([proj[o] for o in OPEN])
    closed_mean = np.mean([proj[l] for l in labels if l not in OPEN])
    # normalised open->closed coordinate: 0 = closed mean, 1 = open mean
    def coord(x): return (x - closed_mean) / (open_mean - closed_mean)
    intermediates = {p: {"pc1": proj[p], "open_closed_coord": coord(proj[p])}
                     for p in ["9NFQ", "9NFR", "9Y7D"] if p in proj}
    # near-boundary members (coord between 0.15 and 0.6): candidate partial states
    near = sorted(((coord(proj[l]), l) for l in labels if l not in OPEN
                   and 0.12 < coord(proj[l]) < 0.7), reverse=True)

    out = {"gnm_nodal_sites": {"bands": bands, "settings": settings,
                     "top_votes": votes.most_common(10),
                     "gap_crossing_votes_excluded": [
                         {"left": left, "right": right, "votes": count}
                         for (left, right), count in gap_votes.most_common()
                     ],
                     "interpretation": (
                         "Scalar GNM sign-change nodes; not a kinematic rotation axis. "
                         "See data/hinge_geometry.json for the screw-axis analysis."
                     )},
           "pc1_open_mean": open_mean, "pc1_closed_mean": closed_mean,
           "intermediates": intermediates,
           "near_boundary": [{"pdb": l, "coord": round(c, 3)} for c, l in near]}
    if not verify:
        with open("data/hinge_intermediate.json", "w", encoding="utf-8") as _fh:
            json.dump(out, _fh, indent=1)

    print(f"GNM nodal bands (majority of {settings} settings): {bands}")
    print(f"excluded gap-crossing sign changes: {gap_votes.most_common()}")
    print(f"PC1 open mean {open_mean:.2f}, closed mean {closed_mean:.2f}")
    for p, d in intermediates.items():
        print(f"  {p}: PC1 {d['pc1']:.3f}, open->closed coord {d['open_closed_coord']:.2f}")
    print(f"near-boundary (partial) members: "
          f"{[(d['pdb'], d['coord']) for d in out['near_boundary'][:6]]}")

    if verify:
        assert bands, "no stable GNM nodal site found"
        assert bands == [(289, 289), (315, 315)], bands
        assert gap_votes[(264, 273)] == 27, gap_votes
        assert intermediates, "no intermediate projected"
        print(f"verify OK: gap-aware stable GNM nodal bands {bands}; "
              f"9NFQ at open->closed coord "
              f"{intermediates.get('9NFQ',{}).get('open_closed_coord',float('nan')):.2f}")

if __name__ == "__main__":
    main()
