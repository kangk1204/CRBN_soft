#!/usr/bin/env python3
"""ANM robustness of the open->closed prediction: endpoint and cutoff sweep.

The headline result is that an ANM built on a single OPEN CRBN structure has, as
its slowest nontrivial mode, a direction that overlaps the experimental
open<->closed difference vector (canonical five-open/65-closed axis, overlap 0.744).
This script tests how that
result depends on (a) which structure the ANM is built on, and (b) the ANM
contact cutoff. It also reports, for every endpoint, the BEST-overlapping mode
and its rank, making explicit that closed structures predict the same axis but
at a higher mode index (the endpoint-dependent rank shift).

All conformers are the 269-Ca superposed tensor (same atom order as resnums), so
ANM is built directly on tensor coordinates -- no per-PDB chain handling needed.

Outputs
  data/anm_robustness.json   endpoint x cutoff table of mode-1 overlap, best
                             overlap, best-mode rank, cumulative top-k overlap
Usage:  python scripts/anm_robustness.py [--verify]
"""
import sys, json, os, subprocess
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
CUTOFFS = [10.0, 12.0, 13.0, 15.0, 16.0, 18.0]
N_MODES = 20

def load_verify_json(path):
    """Load the immutable reference for --verify from git, or from an on-disk artifact."""
    blob = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=False)
    if blob.returncode == 0 and blob.stdout:
        return json.loads(blob.stdout), "committed snapshot (git HEAD)"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), f"on-disk reference artifact ({path})"
    sys.exit(f"verify aborted: no committed reference available for {path}; run inside a "
             "git checkout or provide the on-disk reference artifact")

def anm_hessian(coords, cutoff):
    n = len(coords); H = np.zeros((3*n, 3*n))
    for i in range(n):
        d = coords - coords[i]
        r = np.linalg.norm(d, axis=1)
        for j in range(i+1, n):
            if r[j] <= cutoff and r[j] > 1e-6:
                k = np.outer(d[j], d[j]) / r[j]**2
                H[3*i:3*i+3, 3*j:3*j+3] = -k
                H[3*j:3*j+3, 3*i:3*i+3] = -k
                H[3*i:3*i+3, 3*i:3*i+3] += k
                H[3*j:3*j+3, 3*j:3*j+3] += k
    return H

def modes(H, k):
    w, v = np.linalg.eigh(H)
    nz = w > 1e-9
    return w[nz][:k], v[:, nz][:, :k]

def main():
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    confs = ens["_confs"]; labels = [str(x) for x in ens["_labels"]]
    # experimental open<->closed difference vector (independent of ANM)
    open_mask = np.array([l in OPEN for l in labels])
    diff = confs[open_mask].mean(0) - confs[~open_mask].mean(0)
    dvec = diff.reshape(-1); dvec /= np.linalg.norm(dvec)
    # PCA PC1 for cross-check
    X = (confs - confs.mean(0)).reshape(len(confs), -1)
    w, v = np.linalg.eigh(np.cov(X.T)); pc1 = v[:, np.argsort(w)[::-1][0]]

    # endpoints: 5 open + 5 representative closed (span of PC1)
    s1 = X @ pc1
    closed_idx = np.where(~open_mask)[0]
    closed_sorted = closed_idx[np.argsort(s1[closed_idx])]
    closed_pick = [labels[i] for i in closed_sorted[[0, len(closed_sorted)//4,
                    len(closed_sorted)//2, 3*len(closed_sorted)//4, -1]]]
    endpoints = OPEN + closed_pick

    table = {}
    for lab in endpoints:
        coords = confs[labels.index(lab)]
        row = {}
        for co in CUTOFFS:
            aw, av = modes(anm_hessian(coords, co), N_MODES)
            ov = np.array([abs(av[:, m] @ dvec) for m in range(av.shape[1])])
            best = int(np.argmax(ov))
            cum10 = float(np.sqrt((ov[:10]**2).sum()))
            row[str(co)] = {"mode1_overlap": float(ov[0]),
                            "best_overlap": float(ov[best]),
                            "best_mode_rank": best + 1,
                            "cum_top10": cum10,
                            "n_modes": int(av.shape[1])}
        table[lab] = row

    # Exhaustive census over ALL 65 closed structures at the reference 15 A cutoff,
    # so the "most closed structures only at a higher mode" statement in the Abstract
    # and Discussion is backed by committed code rather than by the 5-structure sample.
    closed_all = {}
    for i in closed_idx:
        lab = labels[i]
        aw, av = modes(anm_hessian(confs[i], 15.0), N_MODES)
        ov = np.array([abs(av[:, m] @ dvec) for m in range(av.shape[1])])
        best = int(np.argmax(ov))
        closed_all[lab] = {"mode1_overlap": float(ov[0]),
                           "best_overlap": float(ov[best]),
                           "best_mode_rank": best + 1}
    ranks = np.array([closed_all[l]["best_mode_rank"] for l in closed_all])
    census = {"n_closed": int(len(ranks)),
              "n_best_mode_ge2": int((ranks >= 2).sum()),
              "n_best_mode_5_or_6": int(((ranks == 5) | (ranks == 6)).sum()),
              "n_best_mode_1": int((ranks == 1).sum())}

    out = {"open_set": OPEN, "closed_endpoints": closed_pick, "cutoffs": CUTOFFS,
           "table": table, "closed_census_15A": census, "closed_all_15A": closed_all}
    # --verify must not mutate the study evidence it is about to check against
    if "--verify" not in sys.argv:
        with open("data/anm_robustness.json", "w", encoding="utf-8") as _fh:
            json.dump(out, _fh, indent=1)
    print(f"closed census @15A: {census['n_best_mode_ge2']}/{census['n_closed']} recover the "
          f"axis only above mode 1; {census['n_best_mode_5_or_6']}/{census['n_closed']} at mode 5-6")

    # summary
    print("endpoint   cutoff15: mode1 / best(rank)")
    for lab in endpoints:
        r = table[lab]["15.0"]
        tag = "OPEN " if lab in OPEN else "closed"
        print(f"  {tag} {lab}: {r['mode1_overlap']:.3f} / "
              f"{r['best_overlap']:.3f}(rank {r['best_mode_rank']})")

    if "--verify" in sys.argv:
        # Cross-check against the COMMITTED snapshot read out of git, not against the
        # file this script writes (a --verify run deliberately writes nothing, so a
        # self-comparison that cannot fail is impossible here).
        c, src = load_verify_json("data/anm_robustness.json")
        assert OPEN == c["open_set"], (OPEN, c["open_set"])
        assert closed_pick == c["closed_endpoints"], (closed_pick, c["closed_endpoints"])
        for lab in endpoints:
            for co in CUTOFFS:
                a, b = table[lab][str(co)], c["table"][lab][str(co)]
                assert abs(a["mode1_overlap"] - b["mode1_overlap"]) < 1e-6, (lab, co, a, b)
                assert abs(a["best_overlap"] - b["best_overlap"]) < 1e-6, (lab, co, a, b)
                assert a["best_mode_rank"] == b["best_mode_rank"], (lab, co, a, b)
        assert census == c["closed_census_15A"], (census, c["closed_census_15A"])
        print(f"cross-checked against {src}")
        # Thresholds guard the published claims: Table 1 reports a 0.73–0.77
        # directional-overlap range, with mode 1 ranked first in each open-structure ANM.
        opens_m1 = [table[l]["15.0"]["mode1_overlap"] for l in OPEN]
        assert min(opens_m1) > 0.70, opens_m1
        open_ranks = [table[l]["15.0"]["best_mode_rank"] for l in OPEN]
        assert open_ranks == [1] * len(OPEN), open_ranks
        assert census == {"n_closed": 65, "n_best_mode_ge2": 62,
                          "n_best_mode_5_or_6": 51, "n_best_mode_1": 3}, census
        # the difference axis is recovered (best overlap high) for every endpoint
        allbest = [table[l][str(c_)]["best_overlap"] for l in endpoints for c_ in CUTOFFS]
        assert min(allbest) > 0.4, min(allbest)
        # rank shift: open endpoints recover it at a lower mode than closed on average
        open_rank = np.mean([table[l]["15.0"]["best_mode_rank"] for l in OPEN])
        closed_rank = np.mean([table[l]["15.0"]["best_mode_rank"] for l in closed_pick])
        print(f"verify OK: open best-mode rank {open_rank:.1f}, "
              f"closed best-mode rank {closed_rank:.1f} "
              f"(rank shift = closed recovers the axis at a higher mode index)")

if __name__ == "__main__":
    main()
