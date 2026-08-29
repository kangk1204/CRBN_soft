#!/usr/bin/env python3
"""Is the result an artefact of the 65-closed / 5-open deposition imbalance?

Two objections are addressed, both without any free parameter.

(A) "PC1 = 88% only says the two clusters are far apart, and the ensemble is
     mostly closed."  The variance fraction is recomputed (i) closed-only,
     (ii) with each cluster's own mean removed, and (iii) on balanced
     5-open + 5-closed subsamples.  If the imbalance inflated 88%, balancing
     must lower it.

(B) "The 0.744 overlap is scored against an axis built from cluster means, so
     it inherits the ensemble composition."  The ANM mode from the open
     reference is instead scored against every INDIVIDUAL open-closed
     structure pair, which uses no mean and no ensemble weighting.

Inputs   data/crbn_ensemble.ens.npz, data/pca_diffvec.npz (open mask),
         data/crbn_anm_modes.npz (ANM eigenvectors on the open reference)
Outputs  data/pairwise_sensitivity.json
Usage    python scripts/pairwise_sensitivity.py [--verify]
"""
import json, sys
import numpy as np

SEED, NDRAW = 42, 2000


def pc1_fraction(M):
    ev = np.linalg.svd(M - M.mean(0), full_matrices=False)[1] ** 2
    return float(ev[0] / ev.sum())


def main():
    verify = "--verify" in sys.argv
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    X = ens["_confs"].reshape(len(ens["_confs"]), -1)
    labels = [str(l)[:4] for l in ens["_labels"]]
    om = np.load("data/pca_diffvec.npz")["open_mask"].astype(bool)
    oi, ci = np.where(om)[0], np.where(~om)[0]
    n_ca = ens["_confs"].shape[1]

    # ---- (A) does the imbalance inflate the variance fraction? ----------------
    Xg = X.copy()
    Xg[om] -= X[om].mean(0)
    Xg[~om] -= X[~om].mean(0)
    rng = np.random.default_rng(SEED)
    bal = np.array([pc1_fraction(np.vstack([X[oi], X[rng.choice(ci, len(oi), False)]]))
                    for _ in range(NDRAW)])

    # ---- (B) per-pair overlap, no ensemble mean anywhere ---------------------
    V = np.load("data/crbn_anm_modes.npz")["anm_eigvecs"]
    V = V / np.linalg.norm(V, axis=0)
    per, rank = [], []
    for i in oi:
        for j in ci:
            a = X[i] - X[j]
            a /= np.linalg.norm(a)
            o = np.abs(V.T @ a)
            per.append(float(o[0]))
            rank.append(int(np.argmax(o)) + 1)
    per, rank = np.array(per), np.array(rank)
    worst = int(np.argmin(per))

    # axis stability: rebuild the target axis from only 5 closed structures
    ax = X[om].mean(0) - X[~om].mean(0)
    ax /= np.linalg.norm(ax)
    cos = []
    for _ in range(NDRAW):
        a = X[om].mean(0) - X[rng.choice(ci, len(oi), False)].mean(0)
        a /= np.linalg.norm(a)
        cos.append(abs(float(a @ ax)))
    cos = np.array(cos)

    out = {
        "n_open": int(om.sum()), "n_closed": int((~om).sum()), "n_ca": int(n_ca),
        "variance_fraction": {
            "full_ensemble": pc1_fraction(X),
            "closed_only": pc1_fraction(X[~om]),
            "open_only": pc1_fraction(X[om]),
            "cluster_means_removed": pc1_fraction(Xg),
            "balanced_mean": float(bal.mean()),
            "balanced_ci95": [float(np.percentile(bal, 2.5)), float(np.percentile(bal, 97.5))],
        },
        "geometry_A": {
            "centroid_separation": float(np.linalg.norm(X[om].mean(0) - X[~om].mean(0)) / np.sqrt(n_ca)),
            "within_closed_rms": float(np.sqrt(((X[~om] - X[~om].mean(0)) ** 2).sum(1).mean() / n_ca)),
            "within_open_rms": float(np.sqrt(((X[om] - X[om].mean(0)) ** 2).sum(1).mean() / n_ca)),
        },
        "per_pair_mode1_overlap": {
            "n_pairs": int(per.size), "mean": float(per.mean()), "median": float(np.median(per)),
            "min": float(per.min()), "max": float(per.max()),
            "frac_above_0.6": float((per > 0.6).mean()),
            "frac_mode1_is_best": float((rank == 1).mean()),
            "worst_pair": [labels[oi[worst // ci.size]], labels[ci[worst % ci.size]]],
        },
        "axis_from_5_closed_vs_full_abscos": {
            "mean": float(cos.mean()),
            "ci95": [float(np.percentile(cos, 2.5)), float(np.percentile(cos, 97.5))],
        },
        "seed": SEED, "n_draws": NDRAW,
    }
    if not verify:
        with open("data/pairwise_sensitivity.json", "w") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")

    v, p = out["variance_fraction"], out["per_pair_mode1_overlap"]
    print(f"PC1 variance: full {100*v['full_ensemble']:.1f}%  closed-only {100*v['closed_only']:.1f}%  "
          f"cluster-means-removed {100*v['cluster_means_removed']:.1f}%  "
          f"balanced(5+5) {100*v['balanced_mean']:.1f}%")
    print(f"per-pair ANM mode-1 overlap over {p['n_pairs']} pairs: mean {p['mean']:.3f} "
          f"min {p['min']:.3f}; mode 1 best in {100*p['frac_mode1_is_best']:.0f}% of pairs")

    if verify:
        assert v["balanced_mean"] > v["full_ensemble"], "balancing should not lower PC1"
        assert p["frac_mode1_is_best"] == 1.0, p["frac_mode1_is_best"]
        assert p["min"] > 0.5, p["min"]
        assert out["axis_from_5_closed_vs_full_abscos"]["mean"] > 0.99
        print("verify OK: the imbalance does not create the result; mode 1 is the best-matching "
              "mode for every individual open-closed pair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
