#!/usr/bin/env python3
"""Regenerate data/closed_subpca.csv: a secondary PCA within the closed sub-ensemble.

Pipeline (fully reproducible from the committed tensor):
  1. Load the 70-conformer x 269-Ca tensor (data/crbn_ensemble.ens.npz).
  2. Select the 65 closed structures (state == 'closed' in data/crbn_pc_projections.csv,
     which reproduce_modes.py derives geometrically from PC1).
  3. Mean-center and run PCA (SVD) on the 65 x 807 closed coordinate block.
  4. Sign convention (removes the arbitrary eigenvector sign): each sub-PC is oriented so
     that its score is POSITIVE for the alphabetically-first closed PDB id. This is a fixed,
     data-independent rule so the CSV is byte-reproducible across runs and platforms.

Writes data/closed_subpca.csv (pdb, sub_PC1, sub_PC2, sub_PC3), sorted by PDB id.
This sub-PCA is retained only as exploratory repository provenance, not for any central claim.
"""
import csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ens = np.load(DATA / "crbn_ensemble.ens.npz", allow_pickle=False)
confs = ens["_confs"]; labels = [str(x) for x in ens["_labels"]]
state = {r["pdb"]: r["state"] for r in csv.DictReader(open(DATA / "crbn_pc_projections.csv"))}

closed = [l for l in labels if state.get(l) == "closed"]
idx = [labels.index(l) for l in closed]
X = confs[idx].reshape(len(idx), -1)
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
scores = U * S                                   # (65, k) principal-component scores

ref = sorted(closed)[0]                           # alphabetically-first closed PDB
ri = closed.index(ref)
for k in range(scores.shape[1]):
    sgn = np.sign(scores[ri, k]) or 1.0           # orient so ref is positive
    scores[:, k] *= sgn

rows = sorted(
    ({"pdb": closed[i],
      "sub_PC1": f"{scores[i, 0]:.4f}",
      "sub_PC2": f"{scores[i, 1]:.4f}",
      "sub_PC3": f"{scores[i, 2]:.4f}"} for i in range(len(closed))),
    key=lambda r: r["pdb"])

with open(DATA / "closed_subpca.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["pdb", "sub_PC1", "sub_PC2", "sub_PC3"], lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print(f"wrote closed_subpca.csv: {len(rows)} closed structures; "
      f"sign convention = positive for {ref}")
