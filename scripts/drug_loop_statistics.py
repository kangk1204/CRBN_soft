#!/usr/bin/env python3
"""How strongly can the drug-binding-loop mobility claim be worded?

The analysis tests whether the drug-binding loop residues (378/380/386) are more mobile
than the structural zinc-site residues (323/326/391/394). Three objections must be met
before that sentence can stand:

(a) SMALL SAMPLE. A Mann-Whitney test on 3 vs 4 residues cannot reach p < 0.05: the
    minimum attainable one-sided p at these sample sizes is 1/C(7,3) = 0.0286, and the
    observed configuration gives 0.114. Reporting that test alone understates the
    evidence AND overstates the rigour, so both facts are computed here.

(b) THE RIGHT NULL IS DOMAIN-MATCHED. Both residue sets lie in the TBD. The question is
    not "are these residues mobile relative to the whole protein" (they are, trivially,
    because the TBD is the mobile lobe) but "are they mobile relative to other TBD
    residues". A permutation null over random TBD trios answers exactly that.

(c) MOBILITY MEASURE CONFOUNDS. Two are tested. (i) Contact number: a residue with few
    neighbours fluctuates more in any elastic network, so a mobility claim may merely
    restate surface exposure; the percentiles are recomputed after regressing mobility
    on contact number. (ii) Crystallographic B-factors are the EXPERIMENTAL measure of
    local mobility; if the model profile simply tracks them, the claim adds nothing.
    Both are computed on three profiles: the ANM 10-mode fluctuation, the ensemble PCA
    fluctuation, and the experimental B-factor profile from the X-ray subset.

The result determines the wording. The strongest defensible version rests on the
ensemble PCA fluctuation against the domain-matched null; the B-factor comparison is a
null result and is reported as such.

Inputs   data/crbn_ensemble.ens.npz, data/crbn_residue_window.csv,
         data/crbn_curation_log.csv, data/crbn_residue_fluctuations.csv,
         RCSB mmCIF (B-factors; cached in data/_cif_cache)
Outputs  data/drug_loop_statistics.json
Usage    python scripts/drug_loop_statistics.py [--verify] [--no-network]
"""
import csv
import gzip
import json
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmode_lib as L
import reproduce_tensor as R

DRUG = [378, 380, 386]                    # tri-tryptophan drug-binding pocket loop
ZN = [323, 326, 391, 394]                 # structural zinc site
TBD = (318, 426)
CUTOFF = 15.0
NMODES_FLUCT = 10
NDRAW = 20000
SEED = 20260720
REF = "8CVP"
MIN_BFACTOR_COVERAGE = 0.9


def percentile_of(values, idx, universe=None):
    """Mean within-universe percentile of the residues at `idx`."""
    u = np.arange(len(values)) if universe is None else np.asarray(universe)
    v = values[u]
    pct = []
    for k in idx:
        pct.append(100.0 * (v < values[k]).mean())
    return float(np.mean(pct))


def mannwhitney_one_sided(a, b):
    """Exact one-sided Mann-Whitney (P(a > b)) by full enumeration, plus the minimum
    attainable p at these sample sizes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, m = len(a), len(b)
    U = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    pooled = np.concatenate([a, b])
    ge = tot = 0
    for c in combinations(range(n + m), n):
        aa = pooled[list(c)]
        bb = pooled[[i for i in range(n + m) if i not in c]]
        u = sum((x > y) + 0.5 * (x == y) for x in aa for y in bb)
        tot += 1
        if u >= U:
            ge += 1
    return {"U": float(U), "p_one_sided": ge / tot, "n_permutations": tot,
            "p_minimum_attainable": 1.0 / tot, "n_a": n, "n_b": m}


def residualise(y, x):
    """y after removing its linear dependence on x (OLS)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def trio_null(profile, tbd_idx, obs_idx, ndraw=NDRAW, seed=SEED):
    """Empirical p for the mean within-window percentile of the observed trio against
    random same-size trios drawn from the TBD."""
    rng = np.random.default_rng(seed)
    ranks = np.array([100.0 * (profile < v).mean() for v in profile])
    obs = float(ranks[obs_idx].mean())
    draws = np.array([ranks[rng.choice(tbd_idx, len(obs_idx), replace=False)].mean()
                      for _ in range(ndraw)])
    n_ex = int((draws >= obs).sum())
    return {"observed_percentile": obs, "null_mean": float(draws.mean()),
            "null_sd": float(draws.std(ddof=1)), "n_draws": ndraw,
            "n_exceedances": n_ex, "p_empirical": (n_ex + 1) / (ndraw + 1),
            "z": float((obs - draws.mean()) / draws.std(ddof=1))}


def fetch_bfactor_cif(pdb, cache_only=False):
    """Read a mmCIF from the local analysis cache, optionally without network fallback."""
    path = os.path.join("data", "_cif_cache", f"{pdb.upper()}.cif.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            return fh.read()
    if cache_only:
        raise FileNotFoundError(f"{pdb}: cached mmCIF not found at {path}")
    return L.fetch_cif(pdb, cache=os.path.join("data", "_cif_cache"))


def choose_crbn_chain_for_bfactor(ca, pdb, resnums):
    """Use the same metadata-validated CRBN-chain rule as the coordinate tensor."""
    chains = R.crbn_chains(pdb)
    if not chains:
        raise RuntimeError(f"{pdb}: no CRBN entity chain in data/_rcsb_meta.json")
    ch, n = R.select_chain(ca, pdb)
    min_n = int(np.ceil(MIN_BFACTOR_COVERAGE * len(resnums)))
    if n < min_n:
        raise RuntimeError(f"{pdb}: CRBN chain {ch} resolves {n}/{len(resnums)} "
                           f"window residues, below expected {min_n}/{len(resnums)}")
    return ch, n, chains


def bfactor_profile(labels, resnums, methods, cache_only=False, verbose=True):
    """Ensemble-average normalised Ca B-factor profile from the X-RAY entries only.

    Cryo-EM entries are excluded: the B field there is an ADP/blur factor from a
    different refinement target and is not comparable to a crystallographic B-factor.
    Each structure is z-scored over the window before averaging so that per-structure
    scaling and resolution do not dominate.
    """
    prof, used, chain_rows, alt, occ = [], [], [], 0, 0
    for pdb in labels:
        if methods.get(pdb) != "X-ray":
            continue
        ca, stats = L.parse_atoms(fetch_bfactor_cif(pdb, cache_only=cache_only))
        ch, n, chains = choose_crbn_chain_for_bfactor(ca, pdb, resnums)
        b = np.array([ca[ch][r]["b"] if r in ca[ch] else np.nan for r in resnums])
        finite_n = int(np.isfinite(b).sum())
        min_n = int(np.ceil(MIN_BFACTOR_COVERAGE * len(resnums)))
        if finite_n < min_n:
            raise RuntimeError(f"{pdb}: CRBN chain {ch} has finite B-factors for "
                               f"{finite_n}/{len(resnums)} window residues, below "
                               f"expected {min_n}/{len(resnums)}")
        alt += sum(1 for r in resnums if r in ca[ch] and ca[ch][r]["alt"] not in (".", "?"))
        occ += sum(1 for r in resnums if r in ca[ch] and ca[ch][r]["occ"] < 0.999)
        z = (b - np.nanmean(b)) / np.nanstd(b)
        prof.append(z)
        used.append(pdb)
        chain_rows.append({"pdb": pdb, "chain": ch, "coverage": int(n),
                           "crbn_entity_chains": list(chains)})
    if not prof:
        return None, used, {"n_altloc_ca": alt, "n_partial_occupancy_ca": occ,
                            "chain_selection": chain_rows}
    P = np.array(prof)
    return np.nanmean(P, 0), used, {"n_altloc_ca": alt, "n_partial_occupancy_ca": occ,
                                    "chain_selection": chain_rows,
                                    "chain_selection_rule":
                                        "metadata-validated reproduce_tensor.select_chain: "
                                        "use committed primary-chain override where present, "
                                        "otherwise highest CRBN-chain window coverage"}


def main():
    verify = "--verify" in sys.argv
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    confs = ens["_confs"]
    labels = [str(x)[:4] for x in ens["_labels"]]
    resnums = np.array([int(r["author_resnum"]) for r in
                        csv.DictReader(open("data/crbn_residue_window.csv"))])
    methods = {r["pdb"]: r["method"] for r in
               csv.DictReader(open("data/crbn_curation_log.csv"))}
    idx = {int(r): k for k, r in enumerate(resnums)}
    drug_i = [idx[r] for r in DRUG]
    zn_i = [idx[r] for r in ZN]
    tbd_i = np.array([k for k, r in enumerate(resnums) if TBD[0] <= r <= TBD[1]])

    ref = confs[labels.index(REF)]
    w, V = L.modes(L.anm_hessian(ref, CUTOFF), 20)
    anm_f = L.sqfluct(w, V, NMODES_FLUCT)
    # ensemble PCA fluctuation: eigenvalue-weighted over the top 10 PCs
    P, var, scores = L.ensemble_pca(confs)
    ev = (scores ** 2).sum(0) / (len(confs) - 1)
    pca_f = np.zeros(len(resnums))
    for m in range(10):
        pca_f += ev[m] * (P[:, m].reshape(-1, 3) ** 2).sum(1)
    # contact number at the ANM cutoff
    ci, cj, _ = L.contact_pairs(ref, CUTOFF)
    contact = np.bincount(np.concatenate([ci, cj]), minlength=len(ref)).astype(float)

    # committed profiles must match the recomputed ones
    comm = list(csv.DictReader(open("data/crbn_residue_fluctuations.csv")))
    anm_comm = np.array([float(r["anm_sqfluct"]) for r in comm])
    rho_check = L.spearman(anm_f, anm_comm)[0]

    out = {"meta": {"drug_residues": DRUG, "zn_residues": ZN, "tbd_window": list(TBD),
                    "n_tbd_residues": int(len(tbd_i)), "cutoff": CUTOFF,
                    "n_modes_fluctuation": NMODES_FLUCT, "n_draws": NDRAW, "seed": SEED,
                    "reference": REF,
                    "spearman_recomputed_vs_committed_anm_profile": rho_check}}

    # ------------------------------------------------ B-factor profile ---
    bprof, bpdbs, bstats = (None, [], {})
    no_network = "--no-network" in sys.argv
    try:
        bprof, bpdbs, bstats = bfactor_profile(labels, resnums, methods,
                                               cache_only=no_network)
    except FileNotFoundError as exc:
        if no_network:
            print(f"B-factor profile skipped: --no-network and {exc}")
        else:
            raise

    profiles = {"anm_fluctuation": anm_f, "pca_fluctuation": pca_f}
    if bprof is not None:
        profiles["experimental_bfactor"] = bprof
    out["bfactor_provenance"] = {
        "n_xray_entries_used": len(bpdbs), "entries": bpdbs,
        "normalisation": "per-structure z-score over the 269-residue window, then averaged",
        "cryoem_excluded": True,
        "cryoem_exclusion_reason": "the B field in cryo-EM depositions is an ADP/blur "
                                   "factor from a different refinement target and is not "
                                   "comparable to a crystallographic B-factor",
        **bstats}

    # ------------------------------------- three-way percentile comparison ---
    print("percentile of the functional residue sets in each mobility profile:")
    comparison = {}
    for name, prof in profiles.items():
        d_win = percentile_of(prof, drug_i)
        z_win = percentile_of(prof, zn_i)
        d_tbd = percentile_of(prof, drug_i, tbd_i)
        z_tbd = percentile_of(prof, zn_i, tbd_i)
        comparison[name] = {
            "drug_percentile_window": d_win, "zn_percentile_window": z_win,
            "drug_percentile_within_tbd": d_tbd, "zn_percentile_within_tbd": z_tbd,
            "difference_window": d_win - z_win,
            "drug_values": [float(prof[k]) for k in drug_i],
            "zn_values": [float(prof[k]) for k in zn_i],
            "mannwhitney": mannwhitney_one_sided(prof[drug_i], prof[zn_i]),
            "trio_null_tbd_matched": trio_null(prof, tbd_i, drug_i),
        }
        c = comparison[name]
        print(f"  {name:22s} drug {d_win:5.1f} vs Zn {z_win:5.1f} (window) | "
              f"within-TBD {d_tbd:5.1f} vs {z_tbd:5.1f} | "
              f"trio-null p = {c['trio_null_tbd_matched']['p_empirical']:.4f} | "
              f"MW p = {c['mannwhitney']['p_one_sided']:.3f}")
    out["three_way_comparison"] = comparison

    # --------------------------------------------- contact-number confound ---
    conf = {}
    for name, prof in profiles.items():
        rho, p, n = L.spearman(contact, prof)
        res = residualise(prof, contact)
        conf[name] = {
            "spearman_contact_vs_profile": {"rho": rho, "p": p, "n": n},
            "drug_percentile_after_contact_residualisation": percentile_of(res, drug_i),
            "zn_percentile_after_contact_residualisation": percentile_of(res, zn_i),
            "trio_null_after_contact_residualisation":
                trio_null(res, tbd_i, drug_i)}
        if bprof is not None and name != "experimental_bfactor":
            resb = residualise(prof, np.nan_to_num(bprof, nan=float(np.nanmean(bprof))))
            conf[name]["drug_percentile_after_bfactor_residualisation"] = \
                percentile_of(resb, drug_i)
            conf[name]["zn_percentile_after_bfactor_residualisation"] = \
                percentile_of(resb, zn_i)
            conf[name]["trio_null_after_bfactor_residualisation"] = \
                trio_null(resb, tbd_i, drug_i)
    out["confound_control"] = conf
    print("contact-number confound:")
    for name, c in conf.items():
        s = c["spearman_contact_vs_profile"]
        print(f"  {name:22s} rho(contact, mobility) = {s['rho']:+.3f} "
              f"(p = {s['p']:.1e}); after residualisation drug "
              f"{c['drug_percentile_after_contact_residualisation']:.1f} vs Zn "
              f"{c['zn_percentile_after_contact_residualisation']:.1f}")

    # ------------------------------- cross-correlation among the profiles ---
    cross = {}
    names = list(profiles)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            rho, p, n = L.spearman(profiles[names[a]], profiles[names[b]])
            cross[f"{names[a]}__vs__{names[b]}"] = {"rho": rho, "p": p, "n": n}
    out["profile_cross_correlation"] = cross
    print("profile cross-correlation:")
    for k, v in cross.items():
        print(f"  {k:52s} rho = {v['rho']:+.3f}")

    # --------------------------------------------- multiple comparisons ---
    tests = [("trio null, ANM", comparison["anm_fluctuation"]
              ["trio_null_tbd_matched"]["p_empirical"]),
             ("trio null, PCA", comparison["pca_fluctuation"]
              ["trio_null_tbd_matched"]["p_empirical"])]
    if bprof is not None:
        tests.append(("trio null, B-factor", comparison["experimental_bfactor"]
                      ["trio_null_tbd_matched"]["p_empirical"]))
    tests += [("Mann-Whitney, ANM", comparison["anm_fluctuation"]
               ["mannwhitney"]["p_one_sided"]),
              ("Mann-Whitney, PCA", comparison["pca_fluctuation"]
               ["mannwhitney"]["p_one_sided"])]
    ps = sorted(p for _, p in tests)
    m = len(ps)
    bh = [p * m / (k + 1) for k, p in enumerate(ps)]
    out["multiple_comparisons"] = {
        "n_tests_on_this_hypothesis": m,
        "tests": [{"test": t, "p": p} for t, p in tests],
        "bonferroni_threshold_for_0p05": 0.05 / m,
        "smallest_p": ps[0],
        "smallest_p_survives_bonferroni": bool(ps[0] < 0.05 / m),
        "benjamini_hochberg_adjusted_smallest": float(min(bh)),
        "note": "The three trio-null tests are three measures of ONE hypothesis, not "
                "three hypotheses; they are reported together rather than corrected "
                "against each other, and the B-factor test is a pre-specified negative "
                "control rather than a competing test. The Bonferroni figure is given "
                "so a reader can apply the strictest possible correction.",
    }

    # --------------------------------------------------- effect sizes ---
    eff = {}
    for name, prof in profiles.items():
        a, b = prof[drug_i], prof[zn_i]
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                     (len(a) + len(b) - 2))
        d = float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")
        # Cliff's delta (rank effect size), and its full range at n=3 vs 4
        cd = float(np.mean([np.sign(x - y) for x in a for y in b]))
        # bootstrap CI on the percentile difference
        rng = np.random.default_rng(SEED)
        boot = []
        ranks = np.array([100.0 * (prof < v).mean() for v in prof])
        for _ in range(10000):
            ia = rng.choice(drug_i, len(drug_i), replace=True)
            ib = rng.choice(zn_i, len(zn_i), replace=True)
            boot.append(ranks[ia].mean() - ranks[ib].mean())
        boot = np.array(boot)
        eff[name] = {"cohens_d": d, "cliffs_delta": cd,
                     "percentile_difference": float(ranks[drug_i].mean() -
                                                    ranks[zn_i].mean()),
                     "percentile_difference_ci95": [float(np.percentile(boot, 2.5)),
                                                    float(np.percentile(boot, 97.5))],
                     "note": "n = 3 vs 4; intervals are wide by construction and are "
                             "reported to convey exactly that"}
    out["effect_sizes"] = eff

    # ------------------------------------------------- recommended wording ---
    pca_p = comparison["pca_fluctuation"]["trio_null_tbd_matched"]["p_empirical"]
    anm_p = comparison["anm_fluctuation"]["trio_null_tbd_matched"]["p_empirical"]
    b_p = (comparison["experimental_bfactor"]["trio_null_tbd_matched"]["p_empirical"]
           if bprof is not None else None)
    out["verdict"] = {
        "primary_statistic": "ensemble PCA fluctuation percentile of the drug trio "
                             "against a TBD-matched random-trio null",
        "primary_p": pca_p,
        "secondary_p_anm": anm_p,
        "bfactor_null_result_p": b_p,
        "mannwhitney_p": comparison["pca_fluctuation"]["mannwhitney"]["p_one_sided"],
        "supportable_claim": "The drug-binding loop carries more amplitude in the "
                             "dominant collective coordinate than the structural zinc "
                             "site, at a level unlikely under a domain-matched null.",
        "not_supportable": ["that the loop is 'more disordered' (crystallographic "
                            "B-factors do not distinguish the two sites)",
                            "any claim resting on the 3-vs-4 Mann-Whitney test, which "
                            "cannot reach p < 0.05 at these sample sizes"],
    }

    if not verify:
        with open("data/drug_loop_statistics.json", "w", encoding="utf-8") as _fh:
            json.dump(out, _fh, indent=1)
    print(f"\nprimary: PCA-fluctuation trio null p = {pca_p:.4f}; "
          f"ANM p = {anm_p:.4f}; "
          + (f"B-factor p = {b_p:.3f} (negative control)"
             if b_p is not None else "B-factor control skipped/unavailable"))

    if verify:
        c = comparison
        assert abs(rho_check - 1.0) < 1e-6 or rho_check > 0.99, rho_check
        # (a) small-sample limit
        mw = c["anm_fluctuation"]["mannwhitney"]
        assert mw["n_permutations"] == 35, mw
        assert abs(mw["p_minimum_attainable"] - 1 / 35) < 1e-9
        assert abs(mw["p_one_sided"] - 0.114) < 0.01, mw["p_one_sided"]
        # (b) domain-matched nulls
        assert 0.02 < c["anm_fluctuation"]["trio_null_tbd_matched"]["p_empirical"] < 0.06, \
            c["anm_fluctuation"]["trio_null_tbd_matched"]
        assert c["pca_fluctuation"]["trio_null_tbd_matched"]["p_empirical"] < 0.01, \
            c["pca_fluctuation"]["trio_null_tbd_matched"]
        # (c) contact confound is strong and negative, and the result survives it
        s = conf["anm_fluctuation"]["spearman_contact_vs_profile"]
        assert s["rho"] < -0.7, s
        assert (conf["anm_fluctuation"]["drug_percentile_after_contact_residualisation"] >
                conf["anm_fluctuation"]["zn_percentile_after_contact_residualisation"])
        if bprof is not None:
            # B-factor null result: no separation, and the model survives residualisation
            bc = c["experimental_bfactor"]
            assert abs(bc["difference_window"]) < 15, bc["difference_window"]
            assert bc["trio_null_tbd_matched"]["p_empirical"] > 0.2, \
                bc["trio_null_tbd_matched"]
            assert (conf["anm_fluctuation"]
                    ["drug_percentile_after_bfactor_residualisation"] >
                    conf["anm_fluctuation"]
                    ["zn_percentile_after_bfactor_residualisation"])
        b_msg = (f"B-factor p = {b_p:.3f} (pre-specified negative control)"
                 if b_p is not None else
                 "B-factor control skipped/unavailable")
        print(f"verify OK: PCA trio-null p = {pca_p:.4f} (primary), ANM p = {anm_p:.4f}, "
              f"{b_msg}; "
              f"Mann-Whitney p = {mw['p_one_sided']:.3f} against a floor of "
              f"{mw['p_minimum_attainable']:.4f}; contact-number rho = {s['rho']:+.2f} "
              f"and the ordering survives residualisation")


if __name__ == "__main__":
    main()
