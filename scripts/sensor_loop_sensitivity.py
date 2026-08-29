#!/usr/bin/env python3
"""What happens to the prediction if the analysis window keeps the sensor loop?

The 269-residue window is the intersection of what all 70 conformers resolve, and its
largest C-terminal gap (342-357) is part of the sensor loop that carries His353. That gap
is forced by exactly three conformers -- 8CVP, 8D7X, 8D7Y -- which are the three
genuine-apo open structures and the ones that define the open end of the axis
(see the window-composition workflow). The analysis treats this as a conditional on the
result, so the obvious question is what the number becomes if the loop is retained.

It cannot be answered by simply widening the window: the three apo structures resolve
none of the loop's 16 positions, so retaining it removes them. The largest ensemble that
resolves the sensor loop is 67 conformers whose open end is the two drug-conditioned
ternaries (6H0F, 7U8F) alone -- and the ANM reference must be one of those two, because
8CVP cannot supply coordinates it does not have. That is the trade the window encodes,
and the point of this script is to quantify both sides of it rather than assert one.

Method: the pipeline's own parser and chain rule (reproduce_tensor.py), a window defined
as the residues every retained conformer resolves, iterative Kabsch superposition, PCA,
and the same 15 A ANM scored against the open->closed difference of cluster means. The
comparison ensemble drops the same three structures but keeps the primary 269-residue
window, which separates the effect of losing the apo trio from the effect of the window.

Usage
  python scripts/sensor_loop_sensitivity.py [--verify]
Output  data/sensor_loop_sensitivity.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce_tensor as R          # noqa: E402  (parser + chain rule of record)

SENSOR_LOOP = (341, 361)
CUTOFF_ANM = 15.0
N_MODES = 20
APO = ("8CVP", "8D7X", "8D7Y")


def kabsch_apply(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (Rm @ (P - pc).T).T + qc


def superpose(X, n_iter=10):
    ref = X[0]
    for _ in range(n_iter):
        X = np.array([kabsch_apply(x, ref) for x in X])
        new = X.mean(0)
        if np.linalg.norm(new - ref) < 1e-6:
            return X, new
        ref = new
    return X, ref


def anm_modes(coords, cutoff=CUTOFF_ANM, k=N_MODES):
    n = len(coords)
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        d = coords - coords[i]
        r2 = (d ** 2).sum(1)
        for j in np.where((r2 <= cutoff ** 2) & (r2 > 1e-9))[0]:
            b = np.outer(d[j], d[j]) / r2[j]
            H[3*i:3*i+3, 3*j:3*j+3] = -b
            H[3*i:3*i+3, 3*i:3*i+3] += b
    w, v = np.linalg.eigh(H)
    nz = w > 1e-9
    return w[nz][:k], v[:, nz][:, :k]


def build(resolved, members, window):
    """Superposed tensor + PCA + open/closed axis for `members` over `window`."""
    X = np.array([[resolved[p]["ca"][r] for r in window] for p in members], float)
    X, mean = superpose(X)
    F = X.reshape(len(X), -1)
    Fc = F - F.mean(0)
    _, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    return X, mean, Vt.T, var


def axis(X, open_mask):
    d = (X[open_mask].mean(0) - X[~open_mask].mean(0)).reshape(-1)
    return d / np.linalg.norm(d)


def score(X, members, ref_pdb, dvec):
    w, v = anm_modes(X[members.index(ref_pdb)])
    ov = np.abs(v.T @ dvec)
    return {"reference": ref_pdb, "mode1_overlap": float(ov[0]),
            "best_mode_rank": int(ov.argmax()) + 1, "best_overlap": float(ov.max()),
            "cum_top10": float(np.sqrt((ov[:10] ** 2).sum()))}


def main():
    verify = "--verify" in sys.argv
    R.CACHE_WRITES_ENABLED = not verify
    curated = [r["pdb"] for r in csv.DictReader(
        open(ROOT / "data" / "crbn_curation_log.csv", encoding="utf-8"))]
    state = {r["pdb"]: r["conformation"] for r in csv.DictReader(
        open(ROOT / "data" / "ens_classified.csv", encoding="utf-8"))}
    paper_window = [int(x) for x in R.WIN]
    lo, hi = min(paper_window), max(paper_window)

    resolved = {}
    for pdb in curated:
        ca = R.parse_ca(R.fetch_cif(pdb))
        ch, _ = R.select_chain(ca, pdb)
        resolved[pdb] = {"chain": ch, "ca": ca[ch]}

    # who fails the sensor loop, from the pipeline's own parse
    loop = list(range(SENSOR_LOOP[0], SENSOR_LOOP[1] + 1))
    fails = sorted(p for p in curated
                   if any(r not in resolved[p]["ca"] for r in loop))
    kept = [p for p in curated if p not in fails]

    # (1) sensor-loop-retaining ensemble: widest window the 67 all resolve
    win_ext = sorted(r for r in range(lo, hi + 1)
                     if all(r in resolved[p]["ca"] for p in kept))
    Xe, _, _, var_e = build(resolved, kept, win_ext)
    open_e = np.array([state[p] == "open" for p in kept])
    d_e = axis(Xe, open_e)

    # (2) same 67 structures, but the primary 269-residue window: isolates the effect of
    # losing the apo trio from the effect of changing the window
    Xc, _, _, var_c = build(resolved, kept, paper_window)
    d_c = axis(Xc, open_e)

    out = {
        "question": ("The 269-residue window drops the 342-357 sensor-loop segment because "
                     "the three genuine-apo open structures resolve none of it. Retaining "
                     "the loop therefore costs those three structures, and the open end "
                     "becomes the two drug-conditioned ternaries alone."),
        "sensor_loop": list(SENSOR_LOOP),
        "conformers_failing_the_loop": fails,
        "n_retained": len(kept),
        "n_open_retained": int(open_e.sum()),
        "open_retained": sorted(np.array(kept)[open_e].tolist()),
        "loop_retaining_window": {
            "n_residues": len(win_ext),
            "includes_sensor_loop": all(r in win_ext for r in range(342, 358)),
            "pc1_variance_fraction": float(var_e[0]),
            "anm": [score(Xe, kept, p, d_e) for p in sorted(np.array(kept)[open_e].tolist())],
        },
        "same_structures_paper_window": {
            "n_residues": len(paper_window),
            "pc1_variance_fraction": float(var_c[0]),
            "anm": [score(Xc, kept, p, d_c) for p in sorted(np.array(kept)[open_e].tolist())],
        },
    }
    a_ext = out["loop_retaining_window"]["anm"]
    a_pap = out["same_structures_paper_window"]["anm"]
    out["interpretation"] = (
        "Comparing the two rows isolates the window from the sample: both use the same 67 "
        "conformers and the same two open references, and differ only in whether the sensor "
        "loop is in the node set. Mode-1 overlap goes from "
        + ", ".join(f"{a['reference']} {a['mode1_overlap']:.3f} (rank {a['best_mode_rank']})"
                    for a in a_pap)
        + " on the primary window to "
        + ", ".join(f"{a['reference']} {a['mode1_overlap']:.3f} (rank {a['best_mode_rank']})"
                    for a in a_ext)
        + " when the loop is retained.")

    if not verify:
        with open(ROOT / "data" / "sensor_loop_sensitivity.json", "w",
                  encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
        print("wrote data/sensor_loop_sensitivity.json")

    print(f"sensor loop {SENSOR_LOOP[0]}-{SENSOR_LOOP[1]} unresolved in: {fails}")
    print(f"retained {len(kept)} conformers, {int(open_e.sum())} open "
          f"({', '.join(sorted(np.array(kept)[open_e].tolist()))})")
    print(f"loop-retaining window: {len(win_ext)} residues, PC1 {var_e[0]*100:.1f}%")
    for a in a_ext:
        print(f"    ANM {a['reference']}: mode-1 {a['mode1_overlap']:.3f}, "
              f"best mode {a['best_mode_rank']} ({a['best_overlap']:.3f})")
    print(f"same 67 on the primary {len(paper_window)}-residue window: PC1 {var_c[0]*100:.1f}%")
    for a in a_pap:
        print(f"    ANM {a['reference']}: mode-1 {a['mode1_overlap']:.3f}, "
              f"best mode {a['best_mode_rank']} ({a['best_overlap']:.3f})")

    if verify:
        assert set(fails) == set(APO), fails
        assert len(kept) == 67 and int(open_e.sum()) == 2
        assert all(r in win_ext for r in range(342, 358)), "sensor loop not retained"
        print("verify OK: the sensor-loop gap is forced by the apo trio alone, and the "
              "loop-retaining ensemble is the 67 conformers with a two-structure open end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
