#!/usr/bin/env python3
"""Significance of the ANM open–closed directional overlap against a null model, plus
leave-one-structure-out robustness.

Two questions the reviewers raised:
  1. Is overlap 0.744 between a single ANM mode and the experimental difference
     vector larger than chance? A 269-Ca mode lives in a 807-d space; random unit
     vectors have small but nonzero overlap with any fixed direction. We build a
     null distribution two ways and report a z-score and empirical p.
       (a) isotropic: random unit vectors in 807-d.
       (b) structural: ANM modes of the SAME open structure but mode indices
           4..20 (higher modes = the "background" of real ANM directions).
  2. Is the difference vector driven by one over-represented deposition cluster?
     We recompute the open<->closed difference vector under leave-one-structure-out
     (drop each closed structure in turn, and each open structure in turn) and report
     the spread of mode-1 overlap. Grouping by study is not attempted: the deposition
     metadata needed to define redundancy groups is not part of the committed data.

Outputs  data/anm_null_significance.json
Usage:   python scripts/anm_null_significance.py [--verify]
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
CUTOFF = 15.0; N = 20; SEED = 20260720

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
        d = coords - coords[i]; r = np.linalg.norm(d, axis=1)
        for j in range(i+1, n):
            if 1e-6 < r[j] <= cutoff:
                k = np.outer(d[j], d[j]) / r[j]**2
                H[3*i:3*i+3, 3*j:3*j+3] = -k; H[3*j:3*j+3, 3*i:3*i+3] = -k
                H[3*i:3*i+3, 3*i:3*i+3] += k; H[3*j:3*j+3, 3*j:3*j+3] += k
    return H

def modes(H, k):
    w, v = np.linalg.eigh(H); nz = w > 1e-9
    return w[nz][:k], v[:, nz][:, :k]

def diffvec(confs, mask):
    d = confs[mask].mean(0) - confs[~mask].mean(0)
    v = d.reshape(-1); return v / np.linalg.norm(v)

def main():
    rng = np.random.default_rng(SEED)
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    confs = ens["_confs"]; labels = [str(x) for x in ens["_labels"]]
    mask = np.array([l in OPEN for l in labels])
    dvec = diffvec(confs, mask)
    dim = dvec.size

    aw, av = modes(anm_hessian(confs[labels.index("8CVP")], CUTOFF), N)
    obs = abs(av[:, 0] @ dvec)

    # null (a): isotropic random unit vectors
    R = rng.standard_normal((20000, dim)); R /= np.linalg.norm(R, axis=1, keepdims=True)
    null_iso = np.abs(R @ dvec)
    z_iso = (obs - null_iso.mean()) / null_iso.std()
    n_exceed = int((null_iso >= obs).sum())
    n_draw = int(null_iso.size)
    # zero exceedances does not mean p = 0: report the add-one (Laplace) estimate,
    # which is the smallest value this many draws can support.
    p_iso = (n_exceed + 1) / (n_draw + 1)

    # null (b): higher ANM modes (structural background)
    null_struct = np.array([abs(av[:, m] @ dvec) for m in range(3, N)])
    z_struct = (obs - null_struct.mean()) / null_struct.std()

    # cumulative subspace overlap (how much of dvec lies in top-k ANM modes)
    cum = [float(np.sqrt(sum((av[:, m] @ dvec)**2 for m in range(k)))) for k in (1,2,3,5,10)]

    # leave-one-closed-out: spread of mode-1 overlap when each closed structure dropped
    closed_idx = [i for i, l in enumerate(labels) if l not in OPEN]
    loo = []
    for drop in closed_idx:
        keep = np.ones(len(labels), bool); keep[drop] = False
        dv = diffvec(confs[keep], mask[keep])
        loo.append(abs(av[:, 0] @ dv))
    loo = np.array(loo)

    # leave-one-open-out: drop each open structure from the open group
    open_idx = [i for i, l in enumerate(labels) if l in OPEN]
    looo = []
    for drop in open_idx:
        keep = np.ones(len(labels), bool); keep[drop] = False
        dv = diffvec(confs[keep], mask[keep])
        looo.append(abs(av[:, 0] @ dv))
    looo = np.array(looo)

    out = {"observed_mode1_overlap": float(obs),
           "null_isotropic": {"mean": float(null_iso.mean()), "std": float(null_iso.std()),
                              "z": float(z_iso), "p_empirical": p_iso,
                              "n_exceedances": n_exceed, "n_draws": n_draw,
                              "n": 20000},
           "null_higher_modes": {"mean": float(null_struct.mean()),
                                 "std": float(null_struct.std()), "z": float(z_struct)},
           "cumulative_overlap_topk": {"k1":cum[0],"k2":cum[1],"k3":cum[2],"k5":cum[3],"k10":cum[4]},
           "leave_one_closed_out": {"mean": float(loo.mean()), "min": float(loo.min()),
                                    "max": float(loo.max()), "n": len(loo)},
           "leave_one_open_out": {"mean": float(looo.mean()), "min": float(looo.min()),
                                  "max": float(looo.max()), "n": len(looo)}}
    # --verify must not mutate the study evidence it is about to check against
    if "--verify" not in sys.argv:
        with open("data/anm_null_significance.json", "w", encoding="utf-8") as _fh:
            json.dump(out, _fh, indent=1)
    print(f"observed mode-1 overlap: {obs:.3f}")
    print(f"isotropic null: mean {null_iso.mean():.4f} sd {null_iso.std():.4f} "
          f"-> z {z_iso:.1f}, p {p_iso:.1e}")
    print(f"higher-mode null: mean {null_struct.mean():.3f} -> z {z_struct:.1f}")
    print(f"cumulative overlap top-1/3/10: {cum[0]:.3f}/{cum[2]:.3f}/{cum[4]:.3f}")
    print(f"leave-one-closed-out mode-1: {loo.mean():.3f} [{loo.min():.3f},{loo.max():.3f}]")
    print(f"leave-one-open-out  mode-1: {looo.mean():.3f} [{looo.min():.3f},{looo.max():.3f}]")

    if "--verify" in sys.argv:
        # Cross-check against the COMMITTED snapshot read out of git, not against the
        # file this script writes (a --verify run deliberately writes nothing, so a
        # self-comparison that cannot fail is impossible here).
        c, src = load_verify_json("data/anm_null_significance.json")
        assert abs(obs - c["observed_mode1_overlap"]) < 1e-6, (obs, c["observed_mode1_overlap"])
        assert abs(z_iso - c["null_isotropic"]["z"]) < 1e-6, (z_iso, c["null_isotropic"]["z"])
        assert n_exceed == c["null_isotropic"]["n_exceedances"], n_exceed
        assert n_draw == c["null_isotropic"]["n_draws"], n_draw
        assert abs(z_struct - c["null_higher_modes"]["z"]) < 1e-6, (z_struct, c["null_higher_modes"]["z"])
        for key, arr in (("leave_one_closed_out", loo), ("leave_one_open_out", looo)):
            assert len(arr) == c[key]["n"], (key, len(arr))
            assert abs(arr.min() - c[key]["min"]) < 1e-6, (key, arr.min(), c[key]["min"])
            assert abs(arr.mean() - c[key]["mean"]) < 1e-6, (key, arr.mean(), c[key]["mean"])
        print(f"cross-checked against {src}")
        # thresholds guard the published claims: z = 34 vs the isotropic null with zero
        # exceedances in 20000 draws, higher-mode z = 8.4, leave-one-out floors 0.744/0.734
        assert z_iso > 30, z_iso
        assert n_exceed == 0, n_exceed
        assert p_iso <= 1.1 / (n_draw + 1), p_iso
        assert loo.min() > 0.74 and looo.min() > 0.73, (loo.min(), looo.min())
        assert z_struct > 8, z_struct
        print(f"verify OK: overlap {obs:.3f} is {z_iso:.0f} sd above isotropic null "
              f"({n_exceed}/{n_draw} exceedances; add-one p = {p_iso:.1e}); "
              f"higher-mode z {z_struct:.1f}; leave-one-out floors "
              f"{loo.min():.3f}/{looo.min():.3f}")

if __name__ == "__main__":
    main()
