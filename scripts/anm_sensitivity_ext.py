#!/usr/bin/env python3
"""Elastic-network modelling choices: cutoff, springs, chain breaks, degeneracy, subspace.

Five network-model questions, answered on the committed
ensemble. All calculations are frame-matched: the ANM is built on ensemble-frame
coordinates, so its eigenvectors and the committed open->closed difference vector share
a reference frame (building the ANM from RAW deposited coordinates and scoring against
the ensemble-frame difference vector is a frame mismatch and gives spuriously low
overlaps).

(9)  CUTOFF AND SPRING VARIANTS. The reference analysis uses a 15 A Ca cutoff with uniform
     springs. 15 A is longer-ranged than the conventional 13-15 A and makes the network
     nearly fully connected within a domain, so the cutoff dependence must be shown.
     Reported per open structure at 10-18 A, and for distance-weighted springs
     (k ~ r^-2, r^-6) which are the parameter-free alternatives.

(10) CHAIN-BREAK HANDLING. Ten sequence gaps fall inside the 269-residue window. The
     network connects across gaps by 3D proximity only, so gap-flanking residues have
     anomalous connectivity. Three variants: as published; with explicit backbone
     springs across each gap; and with gap-flanking residues removed from the NETWORK
     (not merely from the reported profile).

(11) EIGENVALUE SPACING AND MODE ROBUSTNESS. If modes 1 and 2 are near-degenerate,
     "mode 1" is not a well-defined direction and the result must be stated as a
     subspace overlap. Reported as eigenvalue ratios for the monomer.

(ii) SUBSPACE FORMULATION. Cumulative overlap over the k slowest modes and RMSIP
     against the PCA subspace, so the headline result can be expressed as a subspace
     property rather than a single eigenvector.

Inputs   data/crbn_ensemble.ens.npz, data/pca_diffvec.npz, data/crbn_pca.npz,
         data/crbn_residue_window.csv
Outputs  data/anm_sensitivity_ext.json
Usage    python scripts/anm_sensitivity_ext.py [--verify]
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmode_lib as L

NMODES = 20
SEED = 20260720
APO_OPEN = ("8CVP", "8D7X", "8D7Y")            # genuine-apo cryo-EM
TERNARY_OPEN = ("6H0F", "7U8F")                # drug + degron, X-ray
CUTOFFS = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
DEGEN_RATIO = 1.20


def load():
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    confs = ens["_confs"]
    labels = [str(x)[:4] for x in ens["_labels"]]
    resnums = np.array([int(r["author_resnum"]) for r in
                        csv.DictReader(open("data/crbn_residue_window.csv"))])
    d = np.load("data/pca_diffvec.npz")
    d_labels = [str(x)[:4] for x in d["labels"]]
    if d_labels != labels:
        raise ValueError("pca_diffvec labels do not align with the ensemble labels")
    open_mask = d["open_mask"].astype(bool)
    open_labels = [label for label, is_open in zip(d_labels, open_mask) if is_open]
    open_labels.sort(key=lambda label: (label not in APO_OPEN, label))
    dvec = d["diff_vec"] / np.linalg.norm(d["diff_vec"])
    return confs, labels, resnums, dvec, open_mask, open_labels


def gap_flanking(resnums):
    """Indices flanking each sequence gap in the window."""
    gaps, flank = [], set()
    for k in range(len(resnums) - 1):
        if resnums[k + 1] != resnums[k] + 1:
            gaps.append((int(resnums[k]), int(resnums[k + 1])))
            flank.add(k)
            flank.add(k + 1)
    return gaps, sorted(flank)


def backbone_pairs(resnums):
    """Consecutive-in-window index pairs, including those spanning a gap: the springs a
    backbone-connectivity restraint would add."""
    i = np.arange(len(resnums) - 1)
    return i, i + 1


def main():
    verify = "--verify" in sys.argv
    confs, labels, resnums, dvec, open_mask, open_labels = load()
    gaps, flank = gap_flanking(resnums)
    out = {"meta": {"n_modes": NMODES, "seed": SEED, "n_ca": int(confs.shape[1]),
                    "n_gaps": len(gaps), "gaps": gaps,
                    "n_gap_flanking_residues": len(flank),
                    "gap_flanking_resnums": [int(resnums[k]) for k in flank],
                    "degeneracy_criterion": f"lambda2/lambda1 < {DEGEN_RATIO}"}}

    # ---------------------------------------------------- (9) cutoff scan ---
    print("cutoff scan (mode-1 overlap / best rank), per open structure:")
    scan = {}
    for pdb in open_labels:
        ref = confs[labels.index(pdb)]
        row = {}
        for cut in CUTOFFS:
            w, V = L.modes(L.anm_hessian(ref, cut), NMODES)
            ov = L.mode_overlaps(V, dvec)
            row[f"{cut:g}"] = {"mode1_overlap": float(ov[0]),
                               "best_overlap": float(ov.max()),
                               "best_rank": int(np.argmax(ov) + 1),
                               "cum_top3": L.cumulative_overlap(V, dvec, 3),
                               "cum_top10": L.cumulative_overlap(V, dvec, 10),
                               "eigval_ratio_2_1": float(w[1] / w[0]),
                               "n_contacts": int(len(L.contact_pairs(ref, cut)[0]))}
        scan[pdb] = row
        print(f"  {pdb}: " + " ".join(
            f"{c:g}A {row[f'{c:g}']['mode1_overlap']:.3f}@{row[f'{c:g}']['best_rank']}"
            for c in (10, 13, 15, 18)))
    out["cutoff_scan"] = scan

    # the specific disclosure: which structures fail at the conventional 13 A
    at13 = {p: scan[p]["13"] for p in open_labels}
    at15 = {p: scan[p]["15"] for p in open_labels}
    drops = [p for p in open_labels
             if at13[p]["best_rank"] != 1 or at13[p]["mode1_overlap"] < 0.6]
    out["cutoff_disclosure"] = {
        "conventional_cutoff": 13.0, "reported_cutoff": 15.0,
        "structures_failing_at_13A": drops,
        "at_13A": {p: {"mode1_overlap": at13[p]["mode1_overlap"],
                       "best_rank": at13[p]["best_rank"],
                       "best_overlap": at13[p]["best_overlap"]} for p in open_labels},
        "at_15A": {p: {"mode1_overlap": at15[p]["mode1_overlap"],
                       "best_rank": at15[p]["best_rank"]} for p in open_labels},
        "apo_open_stable_at_13A": all(at13[p]["best_rank"] == 1 and
                                      at13[p]["mode1_overlap"] > 0.7 for p in APO_OPEN),
        "note": "The two structures that lose rank 1 at the conventional 13 A cutoff are "
                "precisely the two drug-conditioned ternary complexes (6H0F, 7U8F); the "
                "three genuine-apo cryo-EM structures are stable from 13 to 18 A. This "
                "must be stated explicitly with the structures named.",
    }

    # ---------------------------------------------- (9b) spring variants ---
    print("spring variants (8CVP):")
    spring = {}
    ref = confs[labels.index("8CVP")]
    for cut in (13.0, 15.0):
        for g in ("uniform", "r2", "r6"):
            w, V = L.modes(L.anm_hessian(ref, cut, gamma=g), NMODES)
            ov = L.mode_overlaps(V, dvec)
            spring[f"{g}_{cut:g}A"] = {"mode1_overlap": float(ov[0]),
                                       "best_overlap": float(ov.max()),
                                       "best_rank": int(np.argmax(ov) + 1),
                                       "cum_top3": L.cumulative_overlap(V, dvec, 3),
                                       "cum_top10": L.cumulative_overlap(V, dvec, 10),
                                       "eigval_ratio_2_1": float(w[1] / w[0])}
            print(f"  {g:8s} {cut:g}A: mode1 {ov[0]:.3f} best {ov.max():.3f}"
                  f"@{np.argmax(ov)+1}")
    out["spring_variants"] = spring

    # ------------------------------------------- (10) chain-break handling ---
    print("chain-break handling (8CVP, 15 A):")
    cb = {}
    w, V = L.modes(L.anm_hessian(ref, 15.0), NMODES)
    ov = L.mode_overlaps(V, dvec)
    cb["as_published"] = {"mode1_overlap": float(ov[0]), "best_rank": int(np.argmax(ov) + 1),
                          "n_ca": int(len(ref))}
    bi, bj = backbone_pairs(resnums)
    w2, V2 = L.modes(L.anm_hessian(ref, 15.0, extra_pairs=(bi, bj)), NMODES)
    ov2 = L.mode_overlaps(V2, dvec)
    cb["backbone_restraints_across_gaps"] = {
        "mode1_overlap": float(ov2[0]), "best_rank": int(np.argmax(ov2) + 1),
        "n_extra_springs": int(len(bi)), "n_ca": int(len(ref))}
    keep = np.array([k for k in range(len(resnums)) if k not in flank])
    sub = ref[keep]
    dsub = dvec.reshape(-1, 3)[keep].reshape(-1)
    dsub = dsub / np.linalg.norm(dsub)
    w3, V3 = L.modes(L.anm_hessian(sub, 15.0), NMODES)
    ov3 = L.mode_overlaps(V3, dsub)
    cb["gap_flanking_excluded_from_network"] = {
        "mode1_overlap": float(ov3[0]), "best_rank": int(np.argmax(ov3) + 1),
        "n_ca": int(len(sub)), "n_removed": int(len(flank)),
        "note": "difference vector restricted to the same residues and renormalised"}
    for k, v in cb.items():
        print(f"  {k[:44]:44s} mode1 {v['mode1_overlap']:.3f} rank {v['best_rank']}")
    out["chain_break_handling"] = cb

    # -------------------------------------------- (11) eigenvalue spacing ---
    eig = {}
    for cut in (13.0, 15.0):
        w, V = L.modes(L.anm_hessian(ref, cut), NMODES)
        ov = L.mode_overlaps(V, dvec)
        eig[f"{cut:g}A"] = {
            "eigenvalues_1_8": [float(x) for x in w[:8]],
            "ratio_2_1": float(w[1] / w[0]), "ratio_3_2": float(w[2] / w[1]),
            "ratio_4_3": float(w[3] / w[2]),
            "overlaps_1_6": [float(x) for x in ov[:6]],
            "mode1_well_separated": bool(w[1] / w[0] >= DEGEN_RATIO),
            "cumulative": {f"k{k}": L.cumulative_overlap(V, dvec, k)
                           for k in (1, 2, 3, 5, 10, 20)}}
    out["eigenvalue_spacing_monomer"] = eig
    print(f"eigenvalue ratios (monomer): 15 A l2/l1 = {eig['15A']['ratio_2_1']:.3f}, "
          f"13 A l2/l1 = {eig['13A']['ratio_2_1']:.3f} "
          f"-> mode 1 {'IS' if eig['15A']['mode1_well_separated'] else 'is NOT'} "
          f"well separated")

    # ------------------------------------------ (ii) subspace formulation ---
    # PCA modes: the committed artifact stores only the first three PCs
    # (data/crbn_pca.npz 'pcs', 807 x 3), so the ensemble PCA is recomputed here to
    # get a 10-mode subspace for RMSIP. The recomputed PC1 is checked against the
    # committed variance ratio.
    pca = np.load("data/crbn_pca.npz")
    P_full, var, _ = L.ensemble_pca(confs)
    assert abs(var[0] - float(pca["variance_ratio"][0])) < 1e-6, \
        (var[0], pca["variance_ratio"][0])
    assert abs(abs(P_full[:, 0] @ (pca["pcs"][:, 0] /
                                   np.linalg.norm(pca["pcs"][:, 0]))) - 1) < 1e-6
    w, V = L.modes(L.anm_hessian(ref, 15.0), NMODES)
    sub = {"cumulative_overlap_monomer_15A":
           {f"k{k}": L.cumulative_overlap(V, dvec, k) for k in (1, 2, 3, 5, 10, 20)},
           "pca_variance_ratio_top3": [float(x) for x in var[:3]],
           "rmsip_anm_pca": {f"k{k}": L.rmsip(V, P_full, k) for k in (3, 5, 10)},
           "n_modes_to_reach_0p8": int(next(
               (k for k in range(1, NMODES + 1)
                if L.cumulative_overlap(V, dvec, k) >= 0.8), -1)),
           "n_modes_to_reach_0p9": int(next(
               (k for k in range(1, NMODES + 1)
                if L.cumulative_overlap(V, dvec, k) >= 0.9), -1))}
    # expected cumulative overlap of k random directions, for reference
    dim = 3 * confs.shape[1]
    sub["random_expectation_cumulative"] = {f"k{k}": float(np.sqrt(k / dim))
                                            for k in (1, 3, 10, 20)}
    out["subspace_formulation"] = sub
    print(f"subspace: cumulative overlap reaches 0.8 at k = {sub['n_modes_to_reach_0p8']} "
          f"modes (random expectation at k=10 is "
          f"{sub['random_expectation_cumulative']['k10']:.3f})")

    if not verify:
        with open("data/anm_sensitivity_ext.json", "w", encoding="utf-8") as _fh:
            json.dump(out, _fh, indent=1)

    if verify:
        d = out["cutoff_disclosure"]
        assert set(open_labels) == set(APO_OPEN) | set(TERNARY_OPEN), open_labels
        assert set(d["structures_failing_at_13A"]) == set(TERNARY_OPEN), \
            d["structures_failing_at_13A"]
        assert d["apo_open_stable_at_13A"], "apo open structures must hold at 13 A"
        assert 0.45 < d["at_13A"]["6H0F"]["mode1_overlap"] < 0.55, d["at_13A"]["6H0F"]
        assert 0.45 < d["at_13A"]["7U8F"]["mode1_overlap"] < 0.56, d["at_13A"]["7U8F"]
        assert eig["15A"]["mode1_well_separated"] and eig["13A"]["mode1_well_separated"]
        assert eig["15A"]["ratio_2_1"] > 1.5, eig["15A"]["ratio_2_1"]
        assert all(v["best_rank"] == 1 for v in spring.values()), \
            {k: v["best_rank"] for k, v in spring.items()}
        assert cb["backbone_restraints_across_gaps"]["best_rank"] == 1
        assert cb["gap_flanking_excluded_from_network"]["best_rank"] == 1
        assert out["meta"]["n_gaps"] == 10, out["meta"]["n_gaps"]
        print(f"verify OK: 6H0F/7U8F drop to "
              f"{d['at_13A']['6H0F']['mode1_overlap']:.3f}/"
              f"{d['at_13A']['7U8F']['mode1_overlap']:.3f} at 13 A while the three apo "
              f"open structures hold; monomer mode 1 well separated "
              f"(l2/l1 = {eig['15A']['ratio_2_1']:.2f}); rank 1 survives every spring "
              f"variant and both chain-break treatments")


if __name__ == "__main__":
    main()
