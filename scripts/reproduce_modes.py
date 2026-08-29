#!/usr/bin/env python3
"""Reproduce the CRBN dynamics analysis: PCA of the curated ensemble, an
anisotropic network model (ANM) built on the open reference structure, a
Gaussian network model (GNM), and the overlap/RMSIP cross-checks.

Numpy-only (no ProDy) so it is portable and matches the committed arrays.
The 269-residue window is read from data/crbn_residue_window.csv.

Inputs
  data/crbn_ensemble.ens.npz     70 x 269 x 3 curated Cα (superposed)
  render/open_8cvp.pdb           open reference for the non-circular ANM
Outputs (regenerated; --verify cross-checks them against the committed
snapshot read out of git HEAD)
  crbn_pca.npz                   pc1..3, variance ratios, PC1 scores, diff vector
  crbn_pc_projections.csv        per-structure PC1/PC2 with open/closed label
  crbn_residue_fluctuations.csv  per-residue ANM & PCA square fluctuation

Headline numbers reproduced: PC1 coordinate-variance fraction 88.3%, PC1-axis directional
overlap 0.9996, ANM(open) mode-1 directional overlap 0.744, and ANM-PCA RMSIP 0.641.

Usage:  python scripts/reproduce_modes.py [--verify]
"""
import sys, csv, io, os, subprocess
import numpy as np

CUTOFF_ANM = 15.0   # Å
CUTOFF_GNM = 10.0   # Å  (GNM contact cutoff; Fig 3a cross-correlation map)
N_MODES = 20

def load_verify_npz(path):
    """Load the immutable reference for --verify from git, or from an on-disk artifact."""
    blob = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=False)
    if blob.returncode == 0 and blob.stdout:
        return np.load(io.BytesIO(blob.stdout)), "committed snapshot (git HEAD)"
    if os.path.exists(path):
        return np.load(path), f"on-disk reference artifact ({path})"
    sys.exit(f"verify aborted: no committed reference available for {path}; run inside a "
             "git checkout or provide the on-disk reference artifact")

def pca(confs):
    n, m, _ = confs.shape
    mean = confs.mean(0)
    X = (confs - mean).reshape(n, -1)
    C = np.cov(X.T)
    w, v = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    vr = w / w.sum()
    scores = X @ v
    return mean, v, w, vr, scores

def gnm_kirchhoff(coords, cutoff):
    n = len(coords); K = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[j] - coords[i]) <= cutoff:
                K[i, j] = K[j, i] = -1; K[i, i] += 1; K[j, j] += 1
    return K

def anm_hessian(coords, cutoff):
    n = len(coords); H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            d = coords[j] - coords[i]; r = np.linalg.norm(d)
            if r <= cutoff:
                k = np.outer(d, d) / r**2
                H[3*i:3*i+3, 3*j:3*j+3] = -k
                H[3*j:3*j+3, 3*i:3*i+3] = -k
                H[3*i:3*i+3, 3*i:3*i+3] += k
                H[3*j:3*j+3, 3*j:3*j+3] += k
    return H

def modes_from(H, k):
    w, v = np.linalg.eigh(H)
    nz = w > 1e-9
    return w[nz][:k], v[:, nz][:, :k]

def read_ca(pdb, resnums, chain="B"):
    want = set(int(r) for r in resnums); got = {}
    for ln in open(pdb):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
            ri = int(ln[22:26])
            if ri in want and ri not in got:
                got[ri] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
    # Fail loudly on a short read. Filtering silently would return a coordinate array
    # shorter than the window and misalign every downstream index; the matmul against the
    # 807-component difference vector would then fail with a shape error far from the cause.
    absent = [int(r) for r in resnums if int(r) not in got]
    if absent:
        raise ValueError(f"{pdb} chain {chain} is missing {len(absent)} window residues "
                         f"(first: {absent[:5]}); the ANM node set would not match the axis")
    return np.array([got[int(r)] for r in resnums])

def main():
    verify = "--verify" in sys.argv
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    confs = ens["_confs"]
    # Residue numbering comes from the committed plain-text window
    # (data/crbn_residue_window.csv, written by reproduce_ensemble.py --write-window),
    # not from the mode artifact this script writes (which would be self-seeding) and
    # not from the pickled ProDy AtomGroup inside the npz (which would require ProDy).
    resnums = np.array([int(r["author_resnum"]) for r in
                        csv.DictReader(open("data/crbn_residue_window.csv"))])
    assert len(resnums) == confs.shape[1], (len(resnums), confs.shape)
    labels = [str(l).split("_")[0].split()[0][:4] for l in ens["_labels"]]

    mean, pcv, pcw, vr, scores = pca(confs)
    pc1 = pcv[:, 0]
    # normalise PC1 scores to the RMSD-like convention used in the committed data
    # (score = projection / sqrt(n_atoms)) so open cluster lands near 8-9, closed near 0
    s1 = scores[:, 0] / np.sqrt(confs.shape[1])
    if abs(s1.min()) > abs(s1.max()):   # sign so the open cluster is positive
        s1 = -s1; pc1 = -pc1; pcv[:, 0] = -pcv[:, 0]; scores[:, 0] = -scores[:, 0]
    # open/closed split by the natural gap: the 5 fully-open structures are cleanly
    # separated from the rest (a large jump in sorted PC1). Cut at the widest gap in
    # the top of the distribution.
    srt = np.sort(s1)[::-1]
    gaps = srt[:-1] - srt[1:]
    ncut = int(np.argmax(gaps[:15])) + 1     # index of the widest gap among the leaders
    thresh = (srt[ncut-1] + srt[ncut]) / 2
    open_mask = s1 >= thresh
    scores[:, 0] = s1                         # store normalised PC1
    diff = confs[open_mask].mean(0) - confs[~open_mask].mean(0)
    dvec = diff.reshape(-1); dvec /= np.linalg.norm(dvec)
    ov_pc1_diff = abs(pc1 @ dvec)

    # ANM on the OPEN reference (non-circular test)
    open_ca = read_ca("render/open_8cvp.pdb", resnums)
    Ha = anm_hessian(open_ca, CUTOFF_ANM)
    aw, av = modes_from(Ha, N_MODES)
    anm_diff_overlap = np.array([abs(av[:, m] @ dvec) for m in range(N_MODES)])
    # secondary axis using only the three apo open structures (8CVP/8D7X/8D7Y) as the
    # open end-state; reported in the text as a sensitivity check (0.77) alongside the
    # canonical five-open/65-closed axis (0.744).
    apo_labels = {"8CVP", "8D7X", "8D7Y"}
    apo_mask = np.array([lab in apo_labels for lab in labels])
    if apo_mask.sum() == 3:
        diff_apo = confs[apo_mask].mean(0) - confs[~open_mask].mean(0)
        dvec_apo = diff_apo.reshape(-1); dvec_apo /= np.linalg.norm(dvec_apo)
        anm_apo_overlap = abs(av[:, 0] @ dvec_apo)
    else:
        anm_apo_overlap = float("nan")
    # ANM-PCA RMSIP over 10x10
    pca_modes = pcv[:, :10]
    ov = np.array([[abs(av[:, i] @ pca_modes[:, j]) for j in range(10)] for i in range(10)])
    rmsip = np.sqrt((ov[:10, :10]**2).sum() / 10)

    print(f"PC1 variance {vr[0]*100:.1f}%  PC2 {vr[1]*100:.1f}%  PC3 {vr[2]*100:.1f}%")
    print(f"PC1-difference overlap {ov_pc1_diff:.3f}")
    print(f"ANM(open) mode-1 overlap {anm_diff_overlap[0]:.3f}  top-10 cum "
          f"{np.sqrt((anm_diff_overlap[:10]**2).sum()):.3f}")
    print(f"ANM mode-1 overlap, three-apo-only axis {anm_apo_overlap:.3f}")
    print(f"ANM-PCA RMSIP {rmsip:.3f}")

    # GNM modes (Fig 3a) and the combined mode artifact consumed by the figure builders
    Kg = gnm_kirchhoff(open_ca, CUTOFF_GNM)
    gw_all, gv_all = np.linalg.eigh(Kg)
    keep_g = gw_all > 1e-8
    gw, gv = gw_all[keep_g][:20], gv_all[:, keep_g][:, :20]
    if not verify:
        np.savez("data/crbn_anm_modes.npz",
                 anm_eigvals=aw[:20], anm_eigvecs=av[:, :20],
                 gnm_eigvals=gw, gnm_eigvecs=gv,
                 # keep all ten PC columns: the RMSIP is a 10x10 quantity, so a
                 # truncated matrix cannot be recomputed from the saved artifact
                 overlap_anm_pca=ov[:, :10], anm_diff_overlap=anm_diff_overlap,
                 cum_overlap=np.sqrt(np.cumsum(anm_diff_overlap**2)),
                 rmsip=rmsip, resnums=resnums)
        np.savez("data/crbn_pca.npz", mean=mean, pcs=pcv[:, :3], eigvals=pcw[:3],
                 variance_ratio=vr[:10], pc1_scores=scores[:, 0], pc2_scores=scores[:, 1],
                 diff_vec=dvec, open_mask=open_mask)
        # pca_diffvec.npz: the canonical open–closed axis (sign arbitrary) plus the
        # data-derived open set. This is
        # the source consumed by build_fig1.py, pca_robustness.py, anm_robustness.py,
        # anm_null_significance.py and hinge_intermediate.py (the latter three assert their
        # fallback list against it). The open set here is NOT hardcoded -- it is the
        # widest-gap cut of the PC1 distribution computed above.
        np.savez("data/pca_diffvec.npz", diff_vec=dvec, labels=np.array(labels),
                 pc1_scores=scores[:, 0], open_mask=open_mask)
        with open("data/crbn_pc_projections.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n"); w.writerow(["pdb", "PC1", "PC2", "state"])
            for i, lab in enumerate(labels):
                w.writerow([lab, f"{scores[i,0]:.3f}", f"{scores[i,1]:.3f}",
                            "open" if open_mask[i] else "closed"])
    # per-residue square fluctuations (ANM slow modes, PCA)
    anm_sqf = np.zeros(len(resnums)); pca_sqf = np.zeros(len(resnums))
    for m in range(10):
        anm_sqf += (av[:, m].reshape(-1, 3)**2).sum(1) / aw[m]
    for m in range(10):
        pca_sqf += (pcv[:, m].reshape(-1, 3)**2).sum(1) * pcw[m]
    if not verify:
        with open("data/crbn_residue_fluctuations.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n"); w.writerow(["resnum", "anm_sqfluct", "pca_sqfluct"])
            for i, r in enumerate(resnums):
                w.writerow([int(r), f"{anm_sqf[i]:.6f}", f"{pca_sqf[i]:.6f}"])
        print("wrote data/crbn_pca.npz, pca_diffvec.npz, crbn_pc_projections.csv, "
              "crbn_residue_fluctuations.csv")
    else:
        print("verify mode: tracked mode/PCA output files left untouched")

    if verify:
        # Cross-check against the COMMITTED snapshot read out of git, not against the
        # file this run just wrote (that would be a self-comparison that cannot fail).
        # Tolerances allow small numerical differences between this numpy
        # implementation and the committed arrays (CA extraction order, eigensolver
        # conventions, and the ProDy origin of the first committed version); the
        # science reproduces to <0.03 on every metric.
        c, src = load_verify_npz("data/crbn_anm_modes.npz")
        assert abs(vr[0]*100 - 88.3) < 0.5, vr[0]*100
        assert ov_pc1_diff > 0.98, ov_pc1_diff
        assert abs(anm_diff_overlap[0] - 0.744) < 0.01, anm_diff_overlap[0]
        # Verify the complete matrix in the current artifact. In verify mode this script does
        # not write outputs, so this is an independent recomputation rather than a self-check.
        with np.load("data/crbn_anm_modes.npz") as current:
            saved_ov = np.asarray(current["overlap_anm_pca"], float)
        assert saved_ov.shape == (10, 10), saved_ov.shape
        matrix_dmax = float(np.max(np.abs(saved_ov - ov)))
        assert matrix_dmax < 2e-3, matrix_dmax

        # Compare the primary ANM arrays against the immutable reference, not just the
        # RMSIP scalar. Eigenvectors are sign-arbitrary and are checked by absolute cosine.
        drift = []
        if "anm_diff_overlap" in getattr(c, "files", []):
            for key, got in (("anm_diff_overlap", anm_diff_overlap),
                             ("cum_overlap", np.sqrt(np.cumsum(anm_diff_overlap**2))),
                             ("anm_eigvals", aw[:20]), ("resnums", resnums)):
                want = np.asarray(c[key], float)
                if want.shape != np.asarray(got, float).shape:
                    drift.append(f"{key}: shape {np.asarray(got).shape} vs {want.shape}")
                else:
                    dmax = float(np.max(np.abs(np.asarray(got, float) - want)))
                    if dmax > 2e-3:
                        drift.append(f"{key}: max |diff| {dmax:.2e}")
            # eigenvectors are sign-arbitrary, so compare them up to a per-column sign
            ev = np.asarray(c["anm_eigvecs"], float)
            if ev.shape == av[:, :20].shape:
                cos = np.abs((av[:, :20] * ev).sum(0))
                if float(cos.min()) < 0.999:
                    drift.append(f"anm_eigvecs: worst |cos| {float(cos.min()):.4f}")
        assert abs(rmsip - float(c["rmsip"])) < 0.02, (rmsip, float(c["rmsip"]))
        assert not drift, f"recomputed arrays disagree with {src}:\n  " + "\n  ".join(drift)
        print(f"cross-checked primary ANM arrays against {src}; complete 10x10 overlap matrix verified")
        assert int(open_mask.sum()) == 5, int(open_mask.sum())
        assert abs(anm_apo_overlap - 0.77) < 0.02, anm_apo_overlap
        print(f"verify OK: PC1 {vr[0]*100:.1f}%, PC1-diff overlap {ov_pc1_diff:.3f}, "
              f"ANM mode-1 {anm_diff_overlap[0]:.3f}, RMSIP {rmsip:.3f}, 5 open "
              f"(88.3% / 0.9996 / 0.744 / 0.641)")

if __name__ == "__main__":
    main()
