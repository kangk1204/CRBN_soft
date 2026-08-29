#!/usr/bin/env python3
"""Bootstrap and leave-one-open-out robustness of the ensemble soft mode.

Regenerates data/pca_robust.npz, the grouped-bootstrap source array. PC1
"overlap" is measured against the
STRUCTURAL open->closed difference vector (data/pca_diffvec.npz, the canonical
five-open/65-closed axis), NOT against the ensemble's own PC1 -- the latter
would be near-circular. The grouped bootstrap gives PC1 variance 86% [48,94]
and open->closed overlap 0.98 [0.75,1.00] across 38 fail-closed publication groups
(fixed seed 42, 2000 resamples). The entry-level bootstrap, 88% [73,93] and 0.99 [0.97,1.00], is
computed alongside it and reported only as a within-study comparison.

Inputs (committed, small):
  data/crbn_ensemble.ens.npz   70 conformers x 269 Ca (_confs)
  data/pca_diffvec.npz         open->closed diff_vec, per-conformer labels, open mask
Output:
  data/pca_robust.npz          vfs, ovs, vf0, ov0, open_labels, vf_closed
"""
import sys

import numpy as np
ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
confs = ens["_confs"]                       # (70, 269, 3)
n = confs.shape[0]
dv = np.load("data/pca_diffvec.npz")
diff = dv["diff_vec"]                        # unit vector, open->closed (3*269,)
labels = dv["labels"]
open_idx = np.where(dv["open_mask"])[0]      # the 5 open structures

def pca_pc1(X, diff):
    """Return (PC1 variance fraction, |PC1 . open-closed axis|) for a coord stack.

    With m samples in d dimensions and m << d (here 70 vs 807), the covariance
    eigenproblem is solved through the m x m Gram matrix instead of the d x d
    covariance: the two share their non-zero spectrum, and the leading covariance
    eigenvector is recovered as Xc^T u / ||Xc^T u||. Identical to within numerical
    noise, and fast enough to bootstrap.
    """
    Xf = X.reshape(X.shape[0], -1); Xc = Xf - Xf.mean(0)
    m = Xc.shape[0]
    gram = Xc @ Xc.T / (m - 1)                      # m x m, same non-zero eigenvalues
    w, u = np.linalg.eigh(gram)
    o = np.argsort(w)[::-1]
    total = w[w > 0].sum()
    pc1 = Xc.T @ u[:, o[0]]
    nrm = np.linalg.norm(pc1)
    if nrm < 1e-12:
        return 0.0, 0.0
    return w[o][0] / total, abs((pc1 / nrm) @ diff)

vf0, ov0 = pca_pc1(confs, diff)
print(f"Full ensemble: PC1 var {vf0*100:.1f}%  overlap {ov0:.3f}")

# --- bootstrap -------------------------------------------------------------------------
# Entries are not independent: many come from one study, often the same construct re-solved
# with a different ligand. Resampling entries treats 70 correlated depositions as 70 draws
# and gives an interval that is too narrow to describe the archive. The primary interval is
# therefore a CLUSTER bootstrap over publication groups; the entry-level interval is kept
# for comparison and labelled as a within-study interval, which is what it measures.
_lab = [str(x).split("_")[0].split()[0][:4] for x in labels]
try:
    from study_groups import load_study_groups
except ModuleNotFoundError:
    from scripts.study_groups import load_study_groups
_grp = load_study_groups(_lab)
groups = {}
for i, p in enumerate(_lab):
    groups.setdefault(_grp[p], []).append(i)
gkeys = sorted(groups)
print(f"study groups: {len(gkeys)} over {n} conformers "
      f"(largest {max(len(v) for v in groups.values())})")

def boot(sampler, ndraw=2000, seed=42):
    rng = np.random.default_rng(seed)
    vfs, ovs, n_open_zero = [], [], 0
    for _ in range(ndraw):
        idx = sampler(rng)
        if len(idx) < 3:
            continue
        if not any(j in set(open_idx.tolist()) for j in idx):
            n_open_zero += 1                 # resample contains no open structure
        vf, ov = pca_pc1(confs[idx], diff)
        vfs.append(vf * 100); ovs.append(ov)
    return np.array(vfs), np.array(ovs), n_open_zero

vfs_e, ovs_e, _ = boot(lambda r: r.integers(0, n, n))
vfs, ovs, zero_open = boot(
    lambda r: [i for g in r.choice(len(gkeys), len(gkeys)) for i in groups[gkeys[g]]])

print(f"Entry bootstrap (within-study) var {vfs_e.mean():.0f}% "
      f"[{np.percentile(vfs_e,2.5):.0f},{np.percentile(vfs_e,97.5):.0f}]  "
      f"overlap {ovs_e.mean():.3f} "
      f"[{np.percentile(ovs_e,2.5):.2f},{np.percentile(ovs_e,97.5):.2f}]")
print(f"Cluster bootstrap ({len(gkeys)} groups)  var {vfs.mean():.0f}% "
      f"[{np.percentile(vfs,2.5):.0f},{np.percentile(vfs,97.5):.0f}]  "
      f"overlap {ovs.mean():.3f} "
      f"[{np.percentile(ovs,2.5):.2f},{np.percentile(ovs,97.5):.2f}]")
print(f"  {zero_open} of {len(vfs)} cluster resamples ({zero_open/len(vfs)*100:.1f}%) "
      f"contain no open structure")

# --- leave-one-open-out jackknife ---
print("Leave-one-open-out:")
for i in open_idx:
    keep = [j for j in range(n) if j != i]
    vf, ov = pca_pc1(confs[keep], diff)
    print(f"  drop {labels[i]}: PC1 var {vf*100:.1f}%  overlap {ov:.3f}")

# --- drop ALL open structures ---
keep = [j for j in range(n) if j not in open_idx]
vf_c, ov_c = pca_pc1(confs[keep], diff)
print(f"Drop all {len(open_idx)} open: PC1 var {vf_c*100:.1f}%  overlap {ov_c:.3f} (n={len(keep)})")

# --- save the array consumed by build_figS3.py ---
# vfs/ovs are the CLUSTER bootstrap, which is what Fig S3 and the Limitations now quote.
# The entry-level arrays travel with them so the difference stays visible.
payload = {
    "vfs": vfs,
    "ovs": ovs,
    "vf0": vf0,
    "ov0": ov0,
    "open_labels": labels[open_idx],
    "vf_closed": vf_c,
    "vfs_entry": vfs_e,
    "ovs_entry": ovs_e,
    "n_groups": len(gkeys),
    "frac_resamples_without_open": zero_open / len(vfs),
}
if "--verify" in sys.argv:
    with np.load("data/pca_robust.npz", allow_pickle=False) as current:
        missing = sorted(set(payload) - set(current.files))
        if missing:
            raise AssertionError(f"pca_robust.npz is missing fields: {missing}")
        for key, expected in payload.items():
            actual = current[key]
            if np.issubdtype(np.asarray(expected).dtype, np.number):
                if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
                    raise AssertionError(f"pca_robust.npz field differs: {key}")
            elif not np.array_equal(actual, expected):
                raise AssertionError(f"pca_robust.npz field differs: {key}")
    print("verify OK: pca_robust.npz matches the 38-group fail-closed bootstrap")
else:
    np.savez("data/pca_robust.npz", **payload)
    print("wrote data/pca_robust.npz")
