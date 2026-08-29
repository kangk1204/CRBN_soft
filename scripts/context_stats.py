#!/usr/bin/env python3
"""Context statistics for the numbers quoted in the text but not produced by a figure script.

Eight blocks, all recomputed from committed data:

 1  random-subspace RMSIP null  -- the missing baseline for RMSIP = 0.641 (two random
    orthonormal 10-D subspaces of R^807 already share sqrt(k/d) = 0.111)
 2  RMSIP decomposition         -- how much of RMSIP^2 is PC1, and how little variance
    the PC2-PC10 half of the comparison carries
 3  exact isotropic-null tail   -- cos^2 of a random unit vector with a fixed direction is
    Beta(1/2, (d-1)/2), so the two-sided tail is closed-form and replaces the
    Monte-Carlo resolution floor p = 5e-5 of data/anm_null_significance.json
 4  single-cluster variance null-- the balanced 5+5 comparison is confounded by sample
    size; the n-matched null draws the same n from ONE cluster
 5  openness vs mode rank       -- among the 65 closed structures, how far open a
    structure sits vs the ANM mode index that recovers the transition axis
 6  pair-vector parallelism     -- how far the 325 open-closed pair vectors are from
    being 325 independent observations
 7  autocorrelation-aware residue statistics -- 269 spatially autocorrelated residues are
    not 269 independent ones; contact number, n_eff-corrected Spearman p, and a
    circular-shift p for the gap-flanking comparison
 8  exact drug-vs-zinc tests    -- 3 vs 4 residues cannot reach p < 0.057, so the
    percentiles and the effect size carry the claim, not the p-value

Inputs   data/crbn_anm_modes.npz, data/crbn_ensemble.ens.npz, data/pca_diffvec.npz,
         data/anm_null_significance.json, data/anm_robustness.json,
         data/ens_classified.csv, data/crbn_residue_fluctuations.csv,
         render/open_8cvp.pdb
Outputs  data/context_stats.json
Usage    python scripts/context_stats.py [--verify]
"""
import json, os, sys, csv
import numpy as np
from scipy import stats

SEED, NDRAW = 42, 2000
CUTOFF_ANM = 15.0        # A; same contact cutoff as the ANM in reproduce_modes.py
K = 10                   # subspace dimension of the RMSIP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmode_lib as _L                                              # noqa: E402
from study_groups import load_study_groups                             # noqa: E402
_FEAT = _L.functional_residues()          # data/crbn_features.json
DRUG, ZINC = _FEAT["drug"], _FEAT["zinc"]
TBD_START = 318          # thalidomide-binding domain


def pc1_fraction(M):
    """PC1 variance fraction via the n x n Gram matrix (n << 807, so this is cheap)."""
    Y = M - M.mean(0)
    ev = np.linalg.eigvalsh(Y @ Y.T)
    return float(ev[-1] / ev.sum())


def lag1(x):
    x = x - x.mean()
    return float((x[:-1] * x[1:]).sum() / (x * x).sum())


def read_ca(pdb, resnums, chain="B"):
    want = set(int(r) for r in resnums); got = {}
    for ln in open(pdb):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
            ri = int(ln[22:26])
            if ri in want and ri not in got:
                got[ri] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
    return np.array([got[int(r)] for r in resnums if int(r) in got])


def pct_rank(x):
    return stats.rankdata(x) / len(x) * 100


def anm_modes(coords, cutoff, k):
    """Slowest k non-trivial ANM modes; same construction as reproduce_modes.py."""
    n = len(coords)
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            dxyz = coords[j] - coords[i]
            r = np.linalg.norm(dxyz)
            if r <= cutoff:
                blk = np.outer(dxyz, dxyz) / r ** 2
                H[3*i:3*i+3, 3*j:3*j+3] = -blk
                H[3*j:3*j+3, 3*i:3*i+3] = -blk
                H[3*i:3*i+3, 3*i:3*i+3] += blk
                H[3*j:3*j+3, 3*j:3*j+3] += blk
    ew, ev = np.linalg.eigh(H)
    nz = ew > 1e-9
    return ew[nz][:k], ev[:, nz][:, :k]


def main():
    verify = "--verify" in sys.argv
    rng = np.random.default_rng(SEED)
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    confs = ens["_confs"]
    X = confs.reshape(len(confs), -1)
    dd = np.load("data/pca_diffvec.npz")
    om = dd["open_mask"].astype(bool)
    dvec = dd["diff_vec"]
    oi, ci = np.where(om)[0], np.where(~om)[0]
    M = np.load("data/crbn_anm_modes.npz")
    V = M["anm_eigvecs"]
    d = V.shape[0]

    # ---- (1) what RMSIP do two UNRELATED 10-D subspaces of R^807 already share? -------
    rs = []
    for _ in range(NDRAW):
        A = np.linalg.qr(rng.standard_normal((d, K)))[0]
        B = np.linalg.qr(rng.standard_normal((d, K)))[0]
        rs.append(np.sqrt(((A.T @ B) ** 2).sum() / K))
    rs = np.array(rs)

    # ---- (2) where does RMSIP^2 come from, and what variance backs PC2-PC10? ----------
    # the committed npz stores only 3 PCs, so the 10 PCs are recomputed here
    Xc = (confs - confs.mean(0)).reshape(len(confs), -1)
    w, v = np.linalg.eigh(np.cov(Xc.T))
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    vr = w / w.sum()
    ov = np.abs(V[:, :K].T @ v[:, :K])
    tot = float((ov ** 2).sum())          # = RMSIP^2 * K
    pc1_share = float((ov[:, 0] ** 2).sum() / tot)

    # ---- (3) exact tail of the isotropic null, no Monte-Carlo floor ------------------
    obs = json.load(open("data/anm_null_significance.json"))["observed_mode1_overlap"]
    p_exact = float(stats.beta.sf(obs ** 2, 0.5, (d - 1) / 2))

    # ---- (4) n-matched single-cluster null for the PC1 variance fraction -------------
    single = {}
    for n in (10, 5):
        f = np.array([pc1_fraction(X[rng.choice(ci, n, False)]) for _ in range(NDRAW)])
        single[f"n{n}_closed_only"] = {
            "mean": float(f.mean()),
            "ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))],
        }

    # ---- (5) among the closed set, openness vs the mode that recovers the axis -------
    rob = json.load(open("data/anm_robustness.json"))["closed_all_15A"]
    pc1_of = {r["pdb"]: float(r["global_PC1"]) for r in
              csv.DictReader(open("data/ens_classified.csv")) if r["conformation"] == "closed"}
    pdbs = [p for p in rob if p in pc1_of]
    gp = np.array([pc1_of[p] for p in pdbs])
    rk = np.array([rob[p]["best_mode_rank"] for p in pdbs])
    rho_rank, p_rank = stats.spearmanr(gp, rk)
    rank1 = sorted([p for p in pdbs if rob[p]["best_mode_rank"] == 1],
                   key=lambda p: -pc1_of[p])

    # ---- (6) are the 325 pair vectors 325 independent observations? ------------------
    P = np.array([X[i] - X[j] for i in oi for j in ci])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    cos = np.abs(P @ dvec)
    sv = np.linalg.svd(P, compute_uv=False)

    # ---- (7) residue statistics with the spatial autocorrelation accounted for -------
    rows = list(csv.DictReader(open("data/crbn_residue_fluctuations.csv")))
    res = np.array([int(r["resnum"]) for r in rows])
    anm = np.array([float(r["anm_sqfluct"]) for r in rows])
    pcaf = np.array([float(r["pca_sqfluct"]) for r in rows])
    ca = read_ca("render/open_8cvp.pdb", res)
    D = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    cn = (D <= CUTOFF_ANM).sum(1) - 1
    rho_cn, p_cn = stats.spearmanr(cn, anm)
    r1, r2 = lag1(cn.astype(float)), lag1(anm)
    n = len(res)
    n_eff = n * (1 - r1 * r2) / (1 + r1 * r2)
    t = rho_cn * np.sqrt((n_eff - 2) / (1 - rho_cn ** 2))
    p_cn_eff = float(2 * stats.t.sf(abs(t), n_eff - 2))
    # gap-flanking = the two residues on either side of a break in author numbering
    flank = np.zeros(n, bool)
    for i in np.where(np.diff(res) > 1)[0]:
        flank[i] = flank[i + 1] = True
    p_flank = float(stats.mannwhitneyu(anm[flank], anm[~flank], alternative="two-sided")[1])
    obs_diff = float(anm[flank].mean() - anm[~flank].mean())
    # circular shift keeps the autocorrelation of the profile and destroys only the
    # alignment between the profile and the gap positions
    shifted = np.array([np.roll(anm, s)[flank].mean() - np.roll(anm, s)[~flank].mean()
                        for s in range(1, n)])
    p_shift = float((np.abs(shifted) >= abs(obs_diff)).mean())

    # ---- (8) exact 3-vs-4 tests, and the smallest p they could possibly reach --------
    idx = {r: i for i, r in enumerate(res)}
    tbd = res >= TBD_START
    dg, zn = [idx[r] for r in DRUG], [idx[r] for r in ZINC]
    functional = {}
    for name, prof in (("anm", anm), ("pca", pcaf)):
        pw = pct_rank(prof)
        pt = np.full(n, np.nan); pt[tbd] = pct_rank(prof[tbd])
        u, pv = stats.mannwhitneyu(prof[dg], prof[zn], alternative="two-sided", method="exact")
        functional[name] = {
            "drug_percentile_window": {str(r): float(pw[idx[r]]) for r in DRUG},
            "zinc_percentile_window": {str(r): float(pw[idx[r]]) for r in ZINC},
            "drug_percentile_tbd": {str(r): float(pt[idx[r]]) for r in DRUG},
            "zinc_percentile_tbd": {str(r): float(pt[idx[r]]) for r in ZINC},
            "drug_mean": float(prof[dg].mean()), "zinc_mean": float(prof[zn].mean()),
            "drug_mean_percentile_tbd": float(pt[dg].mean()),
            "zinc_mean_percentile_tbd": float(pt[zn].mean()),
            "mannwhitney_u": float(u), "p_exact": float(pv),
            "rank_biserial": float(2 * u / (len(dg) * len(zn)) - 1),
        }
    p_floor = float(stats.mannwhitneyu([1, 2, 3], [4, 5, 6, 7],
                                       alternative="two-sided", method="exact")[1])

    # ---- (9) node set: the 269-residue window is fixed by the CLOSED structures -------
    # An ANM on every Ca the open reference itself resolves is the genuinely open-only
    # network. Its mode 1 is projected onto the 269 window to score it against the axis.
    res_all, X_all = [], []
    seen = set()
    for ln in open("render/open_8cvp.pdb"):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == "B":
            ri = int(ln[22:26])
            if ri in seen:
                continue
            seen.add(ri); res_all.append(ri)
            X_all.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    res_all, X_all = np.array(res_all), np.array(X_all)
    window = np.array([int(r["author_resnum"]) for r in
                       csv.DictReader(open("data/crbn_residue_window.csv"))])
    w_full, v_full = anm_modes(X_all, CUTOFF_ANM, 20)
    keep = np.where(np.isin(res_all, window))[0]
    proj = v_full.reshape(len(res_all), 3, 20)[keep].reshape(-1, 20)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    ov_full = np.abs(proj.T @ dvec)
    node_set = {
        "n_ca_open_reference": int(len(res_all)),
        "n_ca_common_window": int(len(window)),
        "mode1_overlap": float(ov_full[0]),
        "best_mode_rank": int(ov_full.argmax()) + 1,
        "best_overlap": float(ov_full.max()),
        "mode1_overlap_common_window": float(np.abs(M["anm_eigvecs"][:, 0] @ dvec)),
    }

    # ---- (9b) is the drug-vs-zinc contrast just lever-arm distance from the hinge? ----
    # A Ca-level square fluctuation dominated by one hinge mode grows with distance from
    # that hinge, so a group further out is more mobile for geometric reasons alone. The
    # test is whether the contrast survives removing the distance trend.
    X8 = confs[[str(l) for l in ens["_labels"]].index("8CVP")]   # open reference, 269 x 3
    hinge_cen = X8[(window >= 258) & (window <= 315)].mean(0)
    dist = np.linalg.norm(X8 - hinge_cen, axis=1)
    lever = {}
    for nm, prof in (("anm", anm), ("pca", pcaf)):
        for scope, msk in (("window", np.ones(len(window), bool)), ("tbd", window >= TBD_START)):
            f, dd, rr = prof[msk], dist[msk], window[msk]
            sl, ic = np.polyfit(dd, f, 1)
            resid = f - (sl * dd + ic)
            di = [int(np.where(rr == r)[0][0]) for r in DRUG]
            zi = [int(np.where(rr == r)[0][0]) for r in ZINC]
            lever[f"{nm}_{scope}"] = {
                "spearman_distance_vs_fluctuation": float(stats.spearmanr(dd, f)[0]),
                "raw_difference": float(f[di].mean() - f[zi].mean()),
                "raw_p_exact": float(stats.mannwhitneyu(f[di], f[zi], alternative="two-sided",
                                                        method="exact")[1]),
                "residual_difference": float(resid[di].mean() - resid[zi].mean()),
                "residual_p_exact": float(stats.mannwhitneyu(resid[di], resid[zi],
                                                             alternative="two-sided",
                                                             method="exact")[1]),
            }
    lever["hinge_distance_A"] = {"drug_mean": float(dist[[int(np.where(window == r)[0][0])
                                                          for r in DRUG]].mean()),
                                 "zinc_mean": float(dist[[int(np.where(window == r)[0][0])
                                                          for r in ZINC]].mean())}

    # ---- (9c) does RMSIP depend on experimental method or resolution? -----------------
    log = {r["pdb"]: r for r in csv.DictReader(open("data/crbn_curation_log.csv"))}
    def _rmsip(idx):
        Y = X[idx] - X[idx].mean(0)
        P = np.linalg.svd(Y, full_matrices=False)[2][:K].T
        return float(np.sqrt((np.abs(V[:, :K].T @ P) ** 2).sum() / K))
    labels = [str(l) for l in ens["_labels"]]
    med_res = float(np.median([float(log[l]["resolution"]) for l in labels]))
    rmsip_splits = {"all": _rmsip(np.arange(len(labels)))}
    for nm, keep in (("xray", lambda l: log[l]["method"] == "X-ray"),
                     ("cryoem", lambda l: log[l]["method"] == "cryo-EM"),
                     ("resolution_better_than_median", lambda l: float(log[l]["resolution"]) < med_res),
                     ("resolution_worse", lambda l: float(log[l]["resolution"]) >= med_res)):
        idx = np.array([i for i, l in enumerate(labels) if keep(l)])
        rmsip_splits[nm] = {"n": int(len(idx)), "rmsip": _rmsip(idx)}
    rmsip_splits["median_resolution_A"] = med_res

    # ---- (10) why a study-level ligand/state Fisher table is not estimable -------------
    log = [r for r in csv.DictReader(open("data/crbn_curation_log.csv"))
           if r["global_state"] in ("drug-conditioned", "genuine-apo")]
    groups = load_study_groups(labels)
    open_set = {str(l) for l in np.load("data/pca_diffvec.npz")["labels"][om]}
    by_study = {}
    for r in log:
        study = groups[r["pdb"]]
        rec = by_study.setdefault(study, {"states": set(), "open": False})
        rec["states"].add(r["global_state"])
        rec["open"] = rec["open"] or (r["pdb"] in open_set)
    apo_studies = sorted({groups[r["pdb"]] for r in log
                          if r["global_state"] == "genuine-apo"})
    cross_arm_studies = sorted(study for study, rec in by_study.items()
                               if len(rec["states"]) > 1)
    fisher_study = {
        "status": "not_estimable",
        "p": None,
        "reason": (
            "Only one independent publication contributes genuine-apo structures, and "
            "that publication also contributes drug-conditioned structures; duplicating "
            "it across Fisher-table arms would violate independence."
        ),
        "n_studies_contributing_genuine_apo": len(apo_studies),
        "genuine_apo_studies": apo_studies,
        "studies_contributing_both_arms": cross_arm_studies,
    }

    out = {
        "rmsip_random_subspace_null": {
            "d": int(d), "k": K, "mean": float(rs.mean()), "sd": float(rs.std(ddof=1)),
            "analytic_sqrt_k_over_d": float(np.sqrt(K / d)),
            "observed_rmsip": float(M["rmsip"]),
        },
        "rmsip_decomposition": {
            "rmsip": float(np.sqrt(tot / K)), "rmsip_sq_times_k": tot,
            "pc1_column_fraction": pc1_share,
            "pc1_variance_fraction": float(vr[0]),
            "pc2_to_pc10_variance_fraction": float(vr[1:K].sum()),
        },
        "isotropic_null_exact_tail": {
            "observed_mode1_overlap": float(obs), "d": int(d), "p_exact": p_exact,
            "p_monte_carlo_floor": 4.999750012499375e-05,
        },
        "single_cluster_variance_fraction_null": single,
        "openness_vs_mode_rank": {
            "spearman_rho": float(rho_rank), "p": float(p_rank), "n": len(pdbs),
            "best_mode_rank1": {p: pc1_of[p] for p in rank1},
        },
        "pair_vector_parallelism": {
            "n_pairs": int(len(P)), "abscos_min": float(cos.min()),
            "abscos_mean": float(cos.mean()),
            "leading_direction_variance_fraction": float(sv[0] ** 2 / (sv ** 2).sum()),
        },
        "residue_autocorrelation": {
            "n_residues": n, "contact_cutoff_A": CUTOFF_ANM,
            "contact_number_mean": float(cn.mean()),
            "contact_number_min": int(cn.min()),
            "contact_number_min_resnum": int(res[cn.argmin()]),
            "spearman_rho": float(rho_cn), "p_naive": float(p_cn),
            "lag1_contact_number": r1, "lag1_anm_sqfluct": r2,
            "n_eff": float(n_eff), "p_neff_corrected": p_cn_eff,
            "n_flank": int(flank.sum()), "n_other": int((~flank).sum()),
            "flank_resnums": [int(r) for r in res[flank]],
            "flank_anm_mean": float(anm[flank].mean()),
            "other_anm_mean": float(anm[~flank].mean()),
            "flank_contact_mean": float(cn[flank].mean()),
            "other_contact_mean": float(cn[~flank].mean()),
            "p_mannwhitney_naive": p_flank,
            "n_shifts": int(shifted.size), "p_circular_shift": p_shift,
        },
        "open_only_node_set": node_set,
        "lever_arm_control": lever,
        "rmsip_by_method_and_resolution": rmsip_splits,
        "fisher_study_level": fisher_study,
        "drug_vs_zinc": dict(functional, p_exact_floor_3v4=p_floor,
                             drug_residues=DRUG, zinc_residues=ZINC,
                             tbd_start=TBD_START, n_tbd=int(tbd.sum())),
        "seed": SEED, "n_draws": NDRAW,
    }
    if not verify:
        with open("data/context_stats.json", "w") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")

    r, dcp, s5, pp = (out["rmsip_random_subspace_null"], out["rmsip_decomposition"],
                      out["openness_vs_mode_rank"], out["pair_vector_parallelism"])
    ra = out["residue_autocorrelation"]
    print(f"random 10-D subspace RMSIP {r['mean']:.3f} +/- {r['sd']:.3f} "
          f"(analytic {r['analytic_sqrt_k_over_d']:.4f}); observed {r['observed_rmsip']:.3f}")
    print(f"RMSIP^2: PC1 column {100*dcp['pc1_column_fraction']:.1f}%; "
          f"PC2-PC10 carry {100*dcp['pc2_to_pc10_variance_fraction']:.1f}% of the variance")
    print(f"isotropic null exact tail p = {out['isotropic_null_exact_tail']['p_exact']:.1e} "
          f"(Monte-Carlo floor 5e-5)")
    for k_, s in single.items():
        print(f"single-cluster PC1 fraction, {k_}: {s['mean']:.2f} "
              f"[{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]")
    print(f"openness vs best ANM mode rank (closed set): rho {s5['spearman_rho']:.3f} "
          f"p {s5['p']:.1e} n {s5['n']}; rank 1 = {', '.join(s5['best_mode_rank1'])}")
    print(f"{pp['n_pairs']} pair vectors vs mean axis: |cos| min {pp['abscos_min']:.3f} "
          f"mean {pp['abscos_mean']:.3f}; leading direction "
          f"{100*pp['leading_direction_variance_fraction']:.0f}% of the set variance")
    print(f"contact number {ra['contact_number_mean']:.2f} mean, min {ra['contact_number_min']} "
          f"at residue {ra['contact_number_min_resnum']}; Spearman {ra['spearman_rho']:.2f} "
          f"p {ra['p_naive']:.0e} -> n_eff {ra['n_eff']:.1f}, p {ra['p_neff_corrected']:.1e}")
    print(f"gap-flanking ANM sqfluct {ra['flank_anm_mean']:.3f} vs {ra['other_anm_mean']:.3f}: "
          f"Mann-Whitney p {ra['p_mannwhitney_naive']:.5f}, circular-shift p {ra['p_circular_shift']:.3f}")
    print(f"drug vs zinc (3 vs 4): ANM p {functional['anm']['p_exact']:.4f}, "
          f"PCA p {functional['pca']['p_exact']:.4f}; smallest attainable p {p_floor:.5f}")
    print(f"open-only node set: {node_set['n_ca_open_reference']} Ca of the open reference "
          f"-> mode-1 overlap {node_set['mode1_overlap']:.3f} at rank {node_set['best_mode_rank']} "
          f"(vs {node_set['mode1_overlap_common_window']:.3f} on the "
          f"{node_set['n_ca_common_window']}-residue common window)")
    la = lever["anm_window"]; lp = lever["pca_window"]
    print(f"lever arm: Spearman(hinge distance, ANM sqfluct) = {la['spearman_distance_vs_fluctuation']:.3f}; "
          f"drug-zinc {la['raw_difference']:+.3f} raw -> {la['residual_difference']:+.3f} after removing "
          f"the distance trend (exact p {la['raw_p_exact']:.3f} -> {la['residual_p_exact']:.3f})")
    print(f"           PCA: {lp['raw_difference']:+.1f} -> {lp['residual_difference']:+.1f} "
          f"(p {lp['raw_p_exact']:.3f} -> {lp['residual_p_exact']:.3f}); hinge distance "
          f"drug {lever['hinge_distance_A']['drug_mean']:.1f} A vs zinc "
          f"{lever['hinge_distance_A']['zinc_mean']:.1f} A")
    print("RMSIP by subset: " + ", ".join(
        f"{k} {v['rmsip']:.3f} (n={v['n']})" for k, v in rmsip_splits.items() if isinstance(v, dict)))
    print("ligand/state association at study level: not estimable "
          f"(entry-level tabulation p = 0.0002); genuine-apo comes from "
          f"{len(apo_studies)} study and {len(cross_arm_studies)} study spans both states")

    if verify:
        assert abs(r["mean"] - 0.111) < 0.005, r["mean"]
        assert abs(r["mean"] - r["analytic_sqrt_k_over_d"]) < 0.005, r["mean"]
        assert abs(dcp["pc1_column_fraction"] - 0.188) < 0.01, dcp["pc1_column_fraction"]
        assert abs(dcp["pc2_to_pc10_variance_fraction"] - 0.096) < 0.005
        assert abs(dcp["rmsip"] - float(M["rmsip"])) < 1e-6, dcp["rmsip"]
        assert p_exact < 1e-100, p_exact
        assert abs(single["n10_closed_only"]["mean"] - 0.51) < 0.05
        assert single["n10_closed_only"]["ci95"][1] < 0.88, "n-matched null must sit below 88%"
        assert abs(s5["spearman_rho"] + 0.813) < 0.01, s5["spearman_rho"]
        assert s5["n"] == 65 and set(s5["best_mode_rank1"]) == {"9DJT", "9NGT", "9DJX"}
        assert pp["n_pairs"] == 325 and pp["abscos_min"] > 0.94, pp["abscos_min"]
        assert pp["leading_direction_variance_fraction"] > 0.95
        assert abs(ra["contact_number_mean"] - 42.63) < 0.01, ra["contact_number_mean"]
        assert (ra["contact_number_min"], ra["contact_number_min_resnum"]) == (12, 222)
        assert abs(ra["n_eff"] - 53.5) < 0.5, ra["n_eff"]
        assert 1e-16 < ra["p_neff_corrected"] < 1e-12, ra["p_neff_corrected"]
        assert ra["n_flank"] == 20 and 0.02 < ra["p_circular_shift"] < 0.10
        assert abs(functional["anm"]["p_exact"] - 0.2286) < 1e-3
        assert abs(functional["pca"]["p_exact"] - p_floor) < 1e-9
        assert node_set["n_ca_open_reference"] == 349, node_set["n_ca_open_reference"]
        assert node_set["best_mode_rank"] == 1, node_set["best_mode_rank"]
        assert abs(node_set["mode1_overlap"] - 0.61) < 0.02, node_set["mode1_overlap"]
        assert abs(node_set["mode1_overlap_common_window"] - 0.744) < 5e-3
        assert lever["anm_window"]["spearman_distance_vs_fluctuation"] > 0.75
        assert lever["anm_window"]["residual_p_exact"] > 0.9, "ANM contrast should not survive"
        assert lever["anm_tbd"]["residual_p_exact"] > 0.9
        assert lever["pca_window"]["residual_difference"] > 0
        assert 0.45 < rmsip_splits["cryoem"]["rmsip"] < 0.70
        assert 0.55 < rmsip_splits["xray"]["rmsip"] < 0.75
        assert len(apo_studies) == 1, apo_studies
        assert cross_arm_studies, cross_arm_studies
        assert fisher_study["status"] == "not_estimable" and fisher_study["p"] is None
        print("verify OK: RMSIP has a 0.111 floor, its p is exactly 2e-143 not 5e-5, the "
              "n-matched single-cluster null is 0.51 not 0.93, the 325 pairs are one "
              "direction, 269 residues are ~54 independent ones, and 3-vs-4 cannot beat 0.057")
    return 0


if __name__ == "__main__":
    sys.exit(main())
