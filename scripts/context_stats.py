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
 8  pocket-definition tests    -- the three UniProt ligand annotations are kept
    separate from an objective 5FQD ligand-contact shell, preventing the three
    annotated residues from being over-generalised to the whole pocket

Inputs   data/crbn_anm_modes.npz, data/crbn_ensemble.ens.npz, data/pca_diffvec.npz,
         data/anm_null_significance.json, data/anm_robustness.json,
         data/ens_classified.csv, data/crbn_residue_fluctuations.csv,
         render/open_8cvp.pdb
Outputs  data/context_stats.json
Usage    python scripts/context_stats.py [--verify]
"""
import argparse, json, os, sys, csv
from pathlib import Path
import numpy as np
from scipy import stats

SEED, NDRAW = 42, 2000
CUTOFF_ANM = 15.0        # A; same contact cutoff as the ANM in reproduce_modes.py
K = 10                   # subspace dimension of the RMSIP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmode_lib as _L                                              # noqa: E402
from study_groups import load_study_groups                             # noqa: E402
from analysis_contracts import (                                       # noqa: E402
    assert_tree_close,
    atomic_write_json,
    validate_ensemble_diff,
)
TBD_START = 318          # thalidomide-binding domain
POCKET_CONTACT_CUTOFF_A = 4.5
POCKET_STRUCTURE = "5FQD"
POCKET_CHAIN = "B"
POCKET_LIGAND = "LVY"


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


def ligand_contact_shell(pdb, *, chain, ligand, cutoff_A, common_residues):
    """Return resolved protein residues contacting a ligand by heavy atoms.

    The PDB fixed-column parser deliberately accepts only the named chain and
    primary/blank alternate locations.  Hydrogen atoms are excluded.  Both the
    complete resolved shell and its intersection with the common analysis
    window are returned so missing sensor-loop residues remain explicit.
    """
    protein = {}
    ligand_atoms = []
    for line in open(pdb, encoding="utf-8"):
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"} or len(line) < 54:
            continue
        if line[21].strip() != chain or line[16:17] not in {" ", "A"}:
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            element = "".join(character for character in line[12:16] if character.isalpha())[:1]
        if element == "H":
            continue
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        residue_name = line[17:20].strip()
        if record == "ATOM":
            try:
                residue_number = int(line[22:26])
            except ValueError:
                continue
            protein.setdefault(residue_number, []).append(xyz)
        elif residue_name == ligand:
            ligand_atoms.append(xyz)
    if not protein or not ligand_atoms:
        raise ValueError(f"{pdb} lacks chain {chain} protein atoms or ligand {ligand}")
    ligand_array = np.asarray(ligand_atoms, dtype=float)
    distances = {}
    for residue, atoms in protein.items():
        atom_array = np.asarray(atoms, dtype=float)
        distances[residue] = float(
            np.linalg.norm(atom_array[:, None, :] - ligand_array[None, :, :], axis=2).min()
        )
    all_contacts = sorted(residue for residue, distance in distances.items() if distance <= cutoff_A)
    common_set = {int(residue) for residue in common_residues}
    common_contacts = [residue for residue in all_contacts if residue in common_set]
    return all_contacts, common_contacts, distances


def pct_rank(x):
    return stats.rankdata(x) / len(x) * 100


def circular_shift_pvalue(profile, group_mask):
    """Two-sided exact circular-shift test, including the identity shift."""
    values = np.asarray(profile, dtype=float)
    mask = np.asarray(group_mask)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("profile must be a finite one-dimensional array with at least two values")
    if mask.shape != values.shape or mask.dtype.kind != "b":
        raise ValueError("group mask must be boolean and match the profile shape")
    if not mask.any() or mask.all():
        raise ValueError("group mask must select a nonempty proper subset")
    observed = float(values[mask].mean() - values[~mask].mean())
    shifted = np.array([
        np.roll(values, shift)[mask].mean() - np.roll(values, shift)[~mask].mean()
        for shift in range(values.size)
    ])
    pvalue = float((np.abs(shifted) >= abs(observed)).mean())
    return observed, shifted, pvalue


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="recompute and compare without writing")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verify = bool(args.verify)
    features = _L.functional_residues()
    drug_residues, zinc_residues = features["drug"], features["zinc"]
    DRUG, ZINC = drug_residues, zinc_residues
    rng = np.random.default_rng(SEED)
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    dd = np.load("data/pca_diffvec.npz", allow_pickle=False)
    confs, label_array, om, dvec = validate_ensemble_diff(ens, dd)
    labels = label_array.tolist()
    X = confs.reshape(len(confs), -1)
    oi, ci = np.where(om)[0], np.where(~om)[0]
    M = np.load("data/crbn_anm_modes.npz", allow_pickle=False)
    required_modes = {"anm_eigvals", "anm_eigvecs", "rmsip", "resnums"}
    if not required_modes.issubset(M.files):
        raise ValueError(f"mode artifact missing keys {sorted(required_modes - set(M.files))}")
    V = M["anm_eigvecs"]
    eigvals = np.asarray(M["anm_eigvals"], dtype=float)
    rmsip_value = np.asarray(M["rmsip"], dtype=float)
    window = np.array([int(r["author_resnum"]) for r in
                       csv.DictReader(open("data/crbn_residue_window.csv"))])
    if V.ndim != 2 or V.shape[0] != dvec.size or V.shape[1] < K or not np.isfinite(V).all():
        raise ValueError(f"invalid ANM eigenvector matrix {V.shape}")
    if eigvals.shape != (V.shape[1],) or eigvals.size < K or not np.isfinite(eigvals).all():
        raise ValueError(f"invalid ANM eigenvalue array {eigvals.shape}")
    if not (eigvals > 0).all():
        raise ValueError("ANM eigenvalues must be positive")
    if rmsip_value.shape != () or not np.isfinite(rmsip_value):
        raise ValueError("RMSIP artifact must contain one finite scalar")
    if not 0.0 <= float(rmsip_value) <= 1.0:
        raise ValueError("RMSIP artifact must lie in [0, 1]")
    if not np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-8, rtol=0.0):
        raise ValueError("ANM eigenvectors are not an orthonormal finite basis")
    if M["resnums"].shape != window.shape or not np.array_equal(M["resnums"], window):
        raise ValueError("ANM residue labels do not exactly match the analysis-window order")
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
    if len(set(res.tolist())) != len(res) or not np.isfinite(anm).all() or not np.isfinite(pcaf).all():
        raise ValueError("residue fluctuation table must have unique labels and finite values")
    if not np.array_equal(res, window):
        raise ValueError(
            "residue fluctuation labels do not exactly match the analysis-window order"
        )
    ca = read_ca("render/open_8cvp.pdb", res)
    if ca.shape != (len(res), 3) or not np.isfinite(ca).all():
        raise ValueError(
            f"open-reference coordinates do not exactly cover residue-table order: {ca.shape}"
        )
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
    # circular shift keeps the autocorrelation of the profile and destroys only the
    # alignment between the profile and the gap positions
    _, shifted, p_shift = circular_shift_pvalue(anm, flank)

    # ---- (8) functional annotation versus structure-defined ligand-contact shell -----
    idx = {r: i for i, r in enumerate(res)}
    tbd = res >= TBD_START
    all_contacts, contact_shell, contact_distances = ligand_contact_shell(
        "render/closed_5fqd_lig.pdb",
        chain=POCKET_CHAIN,
        ligand=POCKET_LIGAND,
        cutoff_A=POCKET_CONTACT_CUTOFF_A,
        common_residues=res,
    )
    cage_plus = sorted(set(DRUG) | {400, 402})
    expected_contact_shell = [377, 378, 379, 380, 386, 400, 402]
    if contact_shell != expected_contact_shell:
        raise ValueError(
            f"{POCKET_STRUCTURE} {POCKET_LIGAND} contact-shell drift: {contact_shell}"
        )
    if not set(cage_plus).issubset(contact_shell):
        raise ValueError("canonical annotated/cage residues are not all in the 5FQD contact shell")
    definitions = {
        "uniprot_ligand_annotations": list(DRUG),
        "annotated_plus_W400_F402": cage_plus,
        "5fqd_4.5A_contact_shell_common_window": contact_shell,
    }
    zn = [idx[r] for r in ZINC]

    def functional_summary(profile, selected):
        selected_indices = [idx[r] for r in selected]
        pw = pct_rank(profile)
        pt = np.full(n, np.nan); pt[tbd] = pct_rank(profile[tbd])
        u, pv = stats.mannwhitneyu(
            profile[selected_indices], profile[zn], alternative="two-sided", method="exact"
        )
        return {
            "residues": list(selected),
            "group_percentile_window": {
                str(r): float(pw[idx[r]]) for r in selected
            },
            "zinc_percentile_window": {str(r): float(pw[idx[r]]) for r in ZINC},
            "group_percentile_tbd": {str(r): float(pt[idx[r]]) for r in selected},
            "zinc_percentile_tbd": {str(r): float(pt[idx[r]]) for r in ZINC},
            "group_mean": float(profile[selected_indices].mean()),
            "zinc_mean": float(profile[zn].mean()),
            "group_mean_percentile_window": float(pw[selected_indices].mean()),
            "zinc_mean_percentile_window": float(pw[zn].mean()),
            "group_mean_percentile_tbd": float(pt[selected_indices].mean()),
            "zinc_mean_percentile_tbd": float(pt[zn].mean()),
            "mannwhitney_u": float(u),
            "p_exact": float(pv),
            "rank_biserial": float(2 * u / (len(selected_indices) * len(zn)) - 1),
        }

    functional_by_definition = {
        definition: {
            name: functional_summary(profile, selected)
            for name, profile in (("anm", anm), ("pca", pcaf))
        }
        for definition, selected in definitions.items()
    }
    # Backward-compatible aliases retain the pre-specified UniProt comparison while the
    # full object makes clear that it is not synonymous with the structural pocket.
    functional = {
        name: functional_by_definition["uniprot_ligand_annotations"][name]
        for name in ("anm", "pca")
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
    w_full, v_full = anm_modes(X_all, CUTOFF_ANM, 20)
    keep = np.where(np.isin(res_all, window))[0]
    if not np.array_equal(res_all[keep], window):
        raise ValueError(
            "open-reference coordinates do not exactly match the analysis-window order"
        )
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

    # ---- (9b) is the pocket-vs-zinc contrast just lever-arm geometry? -----------------
    # For a rigid rotation, square displacement grows with squared perpendicular distance
    # from the rotation axis.  The axis is derived independently from the mean endpoint
    # geometry after anchoring NTD+HB (scripts/hinge_geometry.py), not from the former
    # 258-315 centroid.  The within-TBD regression is the primary control because all
    # functional groups lie in that domain; the full-window result is retained only as a
    # transparent sensitivity calculation across domains.
    from hinge_geometry import compute_geometry

    hinge_geometry, axis_distance, _endpoint_displacement = compute_geometry(
        confs, om, window
    )
    axis_distance_sq = axis_distance ** 2
    lever = {
        "geometry_source": "endpoint Kabsch screw axis; data/hinge_geometry.json",
        "distance_metric": "squared perpendicular C-alpha distance to screw axis",
        "primary_scope": "TBD residues >=318",
        "rotation_angle_deg": hinge_geometry["rotation_angle_deg"],
    }
    for nm, prof in (("anm", anm), ("pca", pcaf)):
        for scope, msk in (("window", np.ones(len(window), bool)), ("tbd", window >= TBD_START)):
            f, dd, dd2, rr = prof[msk], axis_distance[msk], axis_distance_sq[msk], window[msk]
            sl, ic = np.polyfit(dd2, f, 1)
            resid = f - (sl * dd2 + ic)
            zi = [int(np.where(rr == r)[0][0]) for r in ZINC]
            by_definition = {}
            for definition, selected in definitions.items():
                di = [int(np.where(rr == r)[0][0]) for r in selected]
                by_definition[definition] = {
                    "residues": list(selected),
                    "group_mean_axis_distance_A": float(dd[di].mean()),
                    "zinc_mean_axis_distance_A": float(dd[zi].mean()),
                    "raw_difference": float(f[di].mean() - f[zi].mean()),
                    "raw_p_exact": float(stats.mannwhitneyu(
                        f[di], f[zi], alternative="two-sided", method="exact"
                    )[1]),
                    "residual_difference": float(resid[di].mean() - resid[zi].mean()),
                    "residual_p_exact": float(stats.mannwhitneyu(
                        resid[di], resid[zi], alternative="two-sided", method="exact"
                    )[1]),
                }
            primary = by_definition["uniprot_ligand_annotations"]
            lever[f"{nm}_{scope}"] = {
                "spearman_axis_distance_vs_fluctuation": float(stats.spearmanr(dd, f)[0]),
                "pearson_axis_distance_squared_vs_fluctuation": float(
                    stats.pearsonr(dd2, f)[0]
                ),
                "linear_slope_per_A2": float(sl),
                "raw_difference": primary["raw_difference"],
                "raw_p_exact": primary["raw_p_exact"],
                "residual_difference": primary["residual_difference"],
                "residual_p_exact": primary["residual_p_exact"],
                "by_definition": by_definition,
            }
    lever["axis_distance_A"] = {
        definition: {
            "group_mean": float(axis_distance[[idx[r] for r in selected]].mean()),
            "zinc_mean": float(axis_distance[[idx[r] for r in ZINC]].mean()),
        }
        for definition, selected in definitions.items()
    }

    # ---- (9c) does RMSIP depend on experimental method or resolution? -----------------
    log = {r["pdb"]: r for r in csv.DictReader(open("data/crbn_curation_log.csv"))}
    def _rmsip(idx):
        Y = X[idx] - X[idx].mean(0)
        P = np.linalg.svd(Y, full_matrices=False)[2][:K].T
        return float(np.sqrt((np.abs(V[:, :K].T @ P) ** 2).sum() / K))
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
    open_set = {labels[index] for index in np.where(om)[0]}
    by_study = {}
    for r in log:
        key = (groups[r["pdb"]], r["global_state"])
        by_study[key] = by_study.get(key, False) or (r["pdb"] in open_set)
    tab = [[0, 0], [0, 0]]
    for (_, state), is_open in by_study.items():
        tab[0 if state == "drug-conditioned" else 1][1 if is_open else 0] += 1
    apo_studies = sorted({groups[r["pdb"]] for r in log
                          if r["global_state"] == "genuine-apo"})
    drug_studies = sorted({groups[r["pdb"]] for r in log
                           if r["global_state"] == "drug-conditioned"})
    shared_arm_studies = sorted(set(apo_studies) & set(drug_studies))

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
        "fisher_study_level": {
            "status": "not_estimable", "p": None,
            "reason": ("Only one independent publication contributes genuine-apo "
                       "structures, and it also contributes drug-conditioned structures; "
                       "splitting it by arm would create pseudo-independent observations."),
            "descriptive_table_drug_then_apo_closed_open": tab,
            "n_studies_contributing_genuine_apo": len(apo_studies),
            "genuine_apo_studies": apo_studies,
            "studies_contributing_both_arms": shared_arm_studies,
        },
        "drug_vs_zinc": dict(
            functional,
            p_exact_floor_3v4=p_floor,
            drug_residues=DRUG,
            zinc_residues=ZINC,
            tbd_start=TBD_START,
            n_tbd=int(tbd.sum()),
            definitions=functional_by_definition,
            structural_contact_shell={
                "structure": POCKET_STRUCTURE,
                "chain": POCKET_CHAIN,
                "ligand": POCKET_LIGAND,
                "heavy_atom_cutoff_A": POCKET_CONTACT_CUTOFF_A,
                "all_resolved_contacts": all_contacts,
                "common_window_contacts": contact_shell,
                "minimum_heavy_atom_distance_A": {
                    str(residue): contact_distances[residue] for residue in all_contacts
                },
            },
        ),
        "seed": SEED, "n_draws": NDRAW,
    }
    if not verify:
        atomic_write_json(Path("data/context_stats.json"), out)

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
    la = lever["anm_tbd"]; lp = lever["pca_tbd"]
    contact_pca = lp["by_definition"]["5fqd_4.5A_contact_shell_common_window"]
    print(f"lever arm within TBD: Spearman(axis distance, ANM sqfluct) = "
          f"{la['spearman_axis_distance_vs_fluctuation']:.3f}; "
          f"UniProt3-zinc {la['raw_difference']:+.3f} raw -> "
          f"{la['residual_difference']:+.3f} after squared-axis-distance regression "
          f"(exact p {la['raw_p_exact']:.3f} -> {la['residual_p_exact']:.3f})")
    print(f"           PCA: UniProt3 {lp['raw_difference']:+.1f} -> "
          f"{lp['residual_difference']:+.1f} (p {lp['raw_p_exact']:.3f} -> "
          f"{lp['residual_p_exact']:.3f}); 5FQD contact7 "
          f"{contact_pca['raw_difference']:+.1f} -> {contact_pca['residual_difference']:+.1f} "
          f"(p {contact_pca['raw_p_exact']:.3f} -> "
          f"{contact_pca['residual_p_exact']:.3f})")
    print("RMSIP by subset: " + ", ".join(
        f"{k} {v['rmsip']:.3f} (n={v['n']})" for k, v in rmsip_splits.items() if isinstance(v, dict)))
    print("ligand/state association at study level: not estimable "
          f"(entry-level tabulation p=0.0002); {len(apo_studies)} apo publication, "
          f"also represented in the drug-conditioned arm={shared_arm_studies}")

    if verify:
        reference = json.loads(Path("data/context_stats.json").read_text(encoding="utf-8"))
        assert_tree_close(out, reference)
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
        assert ra["n_flank"] == 20 and ra["n_shifts"] == 269
        assert abs(ra["p_circular_shift"] - 15 / 269) < 1e-15
        assert abs(functional["anm"]["p_exact"] - 0.2286) < 1e-3
        assert abs(functional["pca"]["p_exact"] - p_floor) < 1e-9
        assert node_set["n_ca_open_reference"] == 349, node_set["n_ca_open_reference"]
        assert node_set["best_mode_rank"] == 1, node_set["best_mode_rank"]
        assert abs(node_set["mode1_overlap"] - 0.61) < 0.02, node_set["mode1_overlap"]
        assert abs(node_set["mode1_overlap_common_window"] - 0.744) < 5e-3
        assert 0.5 < lever["anm_tbd"]["spearman_axis_distance_vs_fluctuation"] < 0.7
        assert lever["anm_tbd"]["pearson_axis_distance_squared_vs_fluctuation"] > 0.7
        assert lever["pca_tbd"]["pearson_axis_distance_squared_vs_fluctuation"] > 0.9
        assert lever["anm_tbd"]["residual_p_exact"] > 0.1
        assert lever["pca_tbd"]["residual_p_exact"] > 0.1
        contact_tbd = lever["pca_tbd"]["by_definition"][
            "5fqd_4.5A_contact_shell_common_window"
        ]
        assert contact_tbd["raw_p_exact"] < 0.05
        assert contact_tbd["residual_p_exact"] > 0.1
        assert contact_shell == expected_contact_shell
        assert 0.45 < rmsip_splits["cryoem"]["rmsip"] < 0.70
        assert 0.55 < rmsip_splits["xray"]["rmsip"] < 0.75
        assert len(apo_studies) == 1, apo_studies
        assert shared_arm_studies == apo_studies, (shared_arm_studies, apo_studies)
        print("verify OK: exact committed artifact matches; RMSIP has a 0.111 floor, its "
              "p is exactly 2e-143 not 5e-5, the "
              "n-matched single-cluster null is 0.51 not 0.93, the 325 pairs are one "
              "direction, 269 residues are ~54 independent ones, and 3-vs-4 cannot beat 0.057")
    return 0


if __name__ == "__main__":
    sys.exit(main())
