#!/usr/bin/env python3
"""Two calibration analyses for the primary result: molecular context and a matched null.

Both answer objections that the isotropic and higher-mode nulls in
anm_null_significance.py cannot address, because both of those nulls are passed by
construction for any two-lobed molecule.

 1  ASSEMBLY CONTEXT. Most conformers in the ensemble include DDB1, but the ANM that
    gives the headline overlap is built on the isolated CRBN monomer. This
    rebuilds the network on 8CVP as deposited (chain A DDB1 + chain B CRBN, 1484 Ca),
    projects each mode onto the 269-residue CRBN window and scores it against the same
    open–closed axis. The axis is not mode 1 of the assembly: four modes are slower. The
    decomposition below shows that modes 1–3 are mainly relative motion of the two bodies,
    whereas mode 4 deforms DDB1 more than CRBN.

 2  RIGID-DOMAIN NULL. The transition is a rigid swing of the TBD against the NTD+HB, so
    the informative question is not "is mode 1 unlike a random 807-dimensional direction"
    (it is, trivially) but "inside the space of rigid interdomain motions, does mode 1
    pick the right axis". The 6-dimensional internal rigid subspace of the two-block
    partition captures the transition at projection norm 0.93 -- above the projection
    norm in the ten-mode ANM subspace -- so the ANM's contribution is axis selection within that
    space, not description of the motion. Modes 1-3 are all essentially rigid interdomain
    motions; only mode 1 points along the transition.

Inputs   render/open_8cvp_assembly.pdb   (Ca-only 8CVP as deposited; --write-assembly
                                          regenerates it from the RCSB mmCIF)
         data/crbn_ensemble.ens.npz, data/pca_diffvec.npz,
         data/crbn_anm_modes.npz, data/crbn_residue_window.csv
Outputs  data/assembly_rigid_null.json
Usage    python scripts/assembly_rigid_null.py [--verify] [--write-assembly]

The assembly eigendecomposition is a 4452 x 4452 problem and takes a few minutes.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

CUTOFFS = (12.0, 15.0, 18.0)
CUTOFF_ANM = 15.0          # monomer reference cutoff
NMODE = 20
HB_TBD = 318               # TBD starts here; NTD+HB is everything below
NTD_HB_BOUNDARY = 187      # NTD | helical bundle, for the three-block variant
NDRAW = 200_000
SEED = 0


def read_ca_pdb(path):
    """Ca coordinates from a Ca-only PDB, keyed by (chain, author resnum), in file order."""
    tag, xyz = [], []
    for ln in open(path):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            tag.append((ln[21], int(ln[22:26])))
            xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return tag, np.array(xyz)


def anm_slow(X, cutoff, k):
    """The k slowest non-trivial ANM modes. Same Hessian as reproduce_modes.py."""
    n = len(X)
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        d = X - X[i]
        r2 = (d ** 2).sum(1)
        for j in np.where((r2 <= cutoff ** 2) & (r2 > 1e-9))[0]:
            b = np.outer(d[j], d[j]) / r2[j]
            H[3*i:3*i+3, 3*j:3*j+3] = -b
            H[3*i:3*i+3, 3*i:3*i+3] += b
    w, v = np.linalg.eigh(H)
    nz = w > 1e-8
    return w[nz][:k], v[:, nz][:, :k]


def rigid_dof(X, mask):
    """Six linearised rigid-body directions (3 translations, 3 rotations) for a subset."""
    n = len(X)
    cen = X[mask].mean(0)
    cols = []
    for k in range(3):
        t = np.zeros((n, 3)); t[mask, k] = 1.0; cols.append(t.ravel())
    for k in range(3):
        ax = np.zeros(3); ax[k] = 1.0
        r = np.zeros((n, 3)); r[mask] = np.cross(ax, X[mask] - cen); cols.append(r.ravel())
    return np.array(cols).T


def onb(M, tol=1e-8):
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, S > tol * S.max()]


def internal_rigid_subspace(X, blocks):
    """Rigid motions of the blocks with the whole molecule's rigid motions removed."""
    glob = onb(rigid_dof(X, np.ones(len(X), bool)))
    both = onb(np.hstack([rigid_dof(X, b) for b in blocks]))
    return onb(both - glob @ (glob.T @ both))


def connectivity_constrained_subspace(basis, i, j, tol=1e-8):
    """The sub-subspace of `basis` whose draws leave the chain joined at atoms i and j.

    The two-lobe null concedes rigidity inside each block but nothing holds the blocks
    together, so almost every draw tears the chain apart at the domain boundary (the
    junction-continuity figures below). Requiring continuity is a LINEAR condition on the
    coefficient vector c: for a draw v = basis @ c the mismatch at the boundary is
    v_i - v_j = M c with M = basis[3i:3i+3] - basis[3j:3j+3], a 3 x r matrix. Its null
    space is the set of internal rigid motions that displace the two boundary atoms
    identically.

    For the two-block partition this leaves exactly 3 of the 6 dimensions: fixing the
    NTD+HB block, a TBD motion that keeps residue 318 attached is a rotation about an
    axis through that residue, and only the axis direction is free. The null therefore
    concedes rigidity AND connectivity and leaves only the choice of hinge axis, which is
    precisely what the elastic-network model is being credited with selecting.

    It is a strictly harder test than the unconstrained one: the surviving directions are
    the ones that most resemble the observed transition, so the null becomes more
    competitive rather than less.
    """
    M = basis[3 * i:3 * i + 3, :] - basis[3 * j:3 * j + 3, :]
    _, S, Vt = np.linalg.svd(M)
    rank = int((S > tol * S.max()).sum()) if S.size and S.max() > 0 else 0
    return basis @ Vt[rank:].T


def kabsch(P, Q):
    """Rotation taking centred P onto centred Q."""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def write_assembly():
    """Regenerate render/open_8cvp_assembly.pdb from the RCSB mmCIF (needs network)."""
    import gzip, io, urllib.request, importlib.util
    spec = importlib.util.spec_from_file_location("rt", "scripts/reproduce_tensor.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/reproduce_tensor.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    raw = urllib.request.urlopen("https://files.rcsb.org/download/8CVP.cif.gz", timeout=120).read()
    ca = rt.parse_ca(gzip.open(io.BytesIO(raw), "rt").read())
    out = ["REMARK   CRBN-DDB1 open assembly 8CVP, Ca only, as deposited (author numbering).",
           "REMARK   Chain A = DDB1 (1135 Ca), chain B = CRBN (349 Ca).",
           "REMARK   Written by scripts/assembly_rigid_null.py --write-assembly from the RCSB mmCIF."]
    n = 0
    for c in sorted(ca):
        for r in sorted(ca[c]):
            n += 1
            x, y, z = ca[c][r]
            out.append("ATOM  %5d  CA  ALA %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C"
                       % (n, c, r, x, y, z))
        out.append("TER")
    out.append("END")
    destination = Path("render/open_8cvp_assembly.pdb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {destination} with {n} CA")


def main():
    if "--write-assembly" in sys.argv:
        write_assembly()

    window = np.array([int(r["author_resnum"]) for r in
                       csv.DictReader(open("data/crbn_residue_window.csv"))])
    dvec = np.load("data/pca_diffvec.npz")["diff_vec"]
    dvec = dvec / np.linalg.norm(dvec)
    V = np.load("data/crbn_anm_modes.npz")["anm_eigvecs"]
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    labels = [str(x) for x in ens["_labels"]]
    Xmono = ens["_confs"][labels.index("8CVP")]           # 269 Ca, ensemble frame

    # ---- (1) the same network, built on the assembly that was actually deposited ------
    tag, Xa = read_ca_pdb("render/open_8cvp_assembly.pdb")
    wset = set(window.tolist())
    sel = [i for i, (c, r) in enumerate(tag) if c == "B" and r in wset]
    sel = [sel[i] for i in np.argsort([tag[i][1] for i in sel])]
    assert len(sel) == len(window), f"CRBN window incomplete in the assembly: {len(sel)}"

    # bring the assembly into the ensemble frame using the CRBN window only, so the mode
    # projections are scored against the axis in the frame the axis is defined in
    P = Xa[sel] - Xa[sel].mean(0)
    Q = Xmono - Xmono.mean(0)
    R = kabsch(P, Q)
    fit_rmsd = float(np.sqrt((((P @ R.T) - Q) ** 2).sum(1).mean()))
    Xa = (Xa - Xa[sel].mean(0)) @ R.T

    # What ARE the modes below the transition in the assembly? Earlier notes asserted an
    # answer twice before computing one, and was wrong both times: first that all four are
    # the partner's own motions, then that none of them is. The decomposition below is the
    # quantity that settles it -- modes 1-3 are two-body wobble, mode 4 is the partner
    # deforming, and the partner deforms about twice as much as CRBN throughout.
    crbn_mask = np.zeros(len(tag), bool); crbn_mask[sel] = True
    ddb1_mask = np.array([c == "A" for c, _ in tag])
    two_body = onb(np.hstack([rigid_dof(Xa, crbn_mask), rigid_dof(Xa, ddb1_mask)]))

    def mode_character(vec):
        """How a joint mode divides into two-body rigid wobble vs internal deformation.

        NOTE ON BASIS. crbn_mask covers only the 269-residue analysis window, while
        ddb1_mask covers all 1135 deposited DDB1 Ca. The two masks therefore do not
        partition the assembly: the 80 CRBN Ca outside the window belong to neither, so
        amplitude_on_crbn**2 + amplitude_on_ddb1**2 < 1 (0.77 for mode 1). Each amplitude
        is the fraction of the mode's norm carried by that subset, and the two internal
        deformation measures are taken on those same different bases. The window is used
        for CRBN because that is the node set the transition axis is defined on.
        """
        f = vec / np.linalg.norm(vec)
        rigid_frac = float(np.linalg.norm(two_body.T @ f))
        crbn_part = f.reshape(-1, 3)[crbn_mask].ravel()
        ddb1_part = f.reshape(-1, 3)[ddb1_mask].ravel()
        # internal deformation of CRBN = what remains after its own best rigid fit
        def internal(part, mask):
            gl = onb(rigid_dof(Xa[mask], np.ones(int(mask.sum()), bool)))
            r = part - gl @ (gl.T @ part)
            return float(np.linalg.norm(r) / np.linalg.norm(part))

        return {
            "two_body_rigid_content": rigid_frac,
            "crbn_internal_deformation": internal(crbn_part, crbn_mask),
            # Symmetric quantity for the partner. Without it the analysis cannot show what the
            # sub-transition modes ARE, only what they are not -- which is how an unsupported
            # claim about them survived in the first place.
            "ddb1_internal_deformation": internal(ddb1_part, ddb1_mask),
            "amplitude_on_crbn": float(np.linalg.norm(crbn_part)),
            "amplitude_on_ddb1": float(np.linalg.norm(ddb1_part)),
        }

    assembly = {}
    for cut in CUTOFFS:
        _, v = anm_slow(Xa, cut, NMODE)
        ov = []
        for m in range(NMODE):
            sub = v[:, m].reshape(-1, 3)[sel].ravel()
            ov.append(float(abs((sub / np.linalg.norm(sub)) @ dvec)))
        ov = np.array(ov)
        if abs(cut - 15.0) < 1e-9:
            slow_character = [dict(mode=m + 1, **mode_character(v[:, m])) for m in range(6)]
        assembly[f"{cut:.1f}"] = {
            "mode1_overlap": float(ov[0]),
            "best_mode_rank": int(ov.argmax()) + 1,
            "best_overlap": float(ov.max()),
            "overlaps": [round(float(x), 4) for x in ov],
        }
        print(f"assembly {cut:4.1f} A: mode 1 = {ov[0]:.3f}, best = mode {ov.argmax()+1} "
              f"({ov.max():.3f})")

    # ---- (2) the rigid-domain null, on the monomer used by the ANM ----------
    two = internal_rigid_subspace(Xmono, [window < HB_TBD, window >= HB_TBD])
    three = internal_rigid_subspace(Xmono, [window < NTD_HB_BOUNDARY,
                                            (window >= NTD_HB_BOUNDARY) & (window < HB_TBD),
                                            window >= HB_TBD])
    cap2, cap3 = float(np.linalg.norm(two.T @ dvec)), float(np.linalg.norm(three.T @ dvec))
    cum10 = float(np.sqrt((np.abs(V[:, :10].T @ dvec) ** 2).sum()))

    obs = float(abs(V[:, 0] @ dvec))

    def draw_null(basis, seed):
        """Uniform directions on the unit sphere of an internal rigid subspace.

        The subspace already has the whole-molecule translations and rotations projected
        out (see internal_rigid_subspace), so every draw is a genuine internal hinge
        motion rather than a rotation of the entire protein. Leakage onto the global
        block is ~1e-16.
        """
        proj = basis.T @ dvec
        rng = np.random.default_rng(seed)
        r = rng.standard_normal((NDRAW, basis.shape[1]))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        vals = np.abs(r @ proj)
        return proj, vals

    def p_value(null_vals):
        """Add-one empirical tail, matching softmode_lib.null_summary.

        The two conventions were mixed across the repository: this file used the raw
        exceedance fraction while the shared library used (k+1)/(n+1). At 200,000 draws the
        two agree far beyond the reported precision, but the raw form can report p = 0 from
        a finite sample, which is never the right claim. Use one convention everywhere.
        """
        return float((int((null_vals >= obs).sum()) + 1) / (null_vals.size + 1))

    # Both parameterisations are reported. They answer different questions and they do not
    # agree: how surprising the alignment looks depends on how many hinges the null is
    # allowed. Quoting only the more favourable one would be the same error this analysis
    # was introduced to correct.
    dproj, null = draw_null(two, SEED)
    dhat = dproj / np.linalg.norm(dproj)
    p_rigid = p_value(null)
    _, null3 = draw_null(three, SEED + 1)
    p_rigid3 = p_value(null3)

    def zscore(v):
        return float((obs - v.mean()) / v.std(ddof=1))

    # The third null: rigid blocks that also stay joined at the domain boundary. The
    # Earlier notes stated that such a null "would be stricter still" and that we had
    # not constructed one. It is constructed here, and it is stricter: p rises rather
    # than falls, because constraining the draws to continuous motions leaves only the
    # directions that already resemble the transition.
    junction = np.argsort(np.abs(window - HB_TBD))[:2]
    cont = connectivity_constrained_subspace(two, int(junction[0]), int(junction[1]))
    cap_cont = float(np.linalg.norm(cont.T @ dvec))
    _, null_cont = draw_null(cont, SEED + 3)
    p_cont = p_value(null_cont)
    mode1_cont = float(np.linalg.norm(cont.T @ V[:, 0]))

    # A random draw in either subspace is a rigid motion of each block, but nothing forces
    # the blocks to stay in contact: most draws pull the domains apart at the boundary. The
    # observed transition does not. Reporting this keeps the null honest -- it is a hinge
    # null only in the sense that block interiors stay rigid, not in the sense that the
    # protein stays connected.
    def discontinuity(field):
        f = field.reshape(-1, 3)
        d = f[junction[0]] - f[junction[1]]
        return float(np.linalg.norm(d) / (np.linalg.norm(f) / np.sqrt(len(f))))

    obs_disc = discontinuity(dvec)
    rng_d = np.random.default_rng(SEED + 2)
    rd = rng_d.standard_normal((2000, two.shape[1]))
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    null_disc = np.array([discontinuity(two @ c) for c in rd])
    frac_as_continuous = float((null_disc <= obs_disc).mean())

    per_mode = []
    for m in range(6):
        p = two.T @ V[:, m]
        rc = float(np.linalg.norm(p))
        per_mode.append({"mode": m + 1, "rigid_content": rc,
                         "direction_cosine_in_rigid_subspace": float(abs(p @ dhat) / rc) if rc > 1e-9 else 0.0,
                         "overlap_with_axis": float(abs(V[:, m] @ dvec))})
    print(f"\nrigid interdomain subspace captures the transition at projection norm {cap2:.3f} "
          f"(3 blocks {cap3:.3f}); ANM top-10 cumulative = {cum10:.3f}")
    print(f"two-lobe   (dim {two.shape[1]:2d}): observed {obs:.3f}, p = {p_rigid:.4f}, "
          f"z = {zscore(null):.2f} (null mean {null.mean():.3f}, "
          f"p95 {np.percentile(null, 95):.3f})")
    print(f"three-body (dim {three.shape[1]:2d}): observed {obs:.3f}, p = {p_rigid3:.4f}, "
          f"z = {zscore(null3):.2f} (null mean {null3.mean():.3f}, "
          f"p95 {np.percentile(null3, 95):.3f})")
    print(f"continuity-constrained (dim {cont.shape[1]:2d}): observed {obs:.3f}, "
          f"p = {p_cont:.4f}, z = {zscore(null_cont):.2f} (subspace captures the "
          f"transition at {cap_cont:.3f}; null mean {null_cont.mean():.3f})")
    print(f"junction continuity: observed {obs_disc:.4f}, null median "
          f"{np.median(null_disc):.4f}; {frac_as_continuous*100:.2f}% of draws are as "
          f"continuous as the observed transition")
    print("mode  rigid-content  direction-cosine  overlap")
    for d in per_mode:
        print(f"  {d['mode']}      {d['rigid_content']:.3f}          "
              f"{d['direction_cosine_in_rigid_subspace']:.3f}         {d['overlap_with_axis']:.3f}")

    out = {
        "assembly": {
            "structure": "8CVP as deposited (chain A DDB1 + chain B CRBN)",
            "n_ca": len(tag), "n_ca_crbn_window": len(sel),
            "window_fit_rmsd_A": fit_rmsd,
            "by_cutoff": assembly,
            "slow_mode_character_15A": slow_character,
        },
        "rigid_domain_null": {
            "two_block_capture": cap2, "three_block_capture": cap3,
            "two_block_internal_dim": int(two.shape[1]),
            "three_block_internal_dim": int(three.shape[1]),
            "anm_top10_cumulative": cum10,
            "observed_mode1_overlap": obs,
            # Both parameterisations, because the answer depends on which one is used.
            "two_block": {
                "internal_dim": int(two.shape[1]),
                "p_empirical": p_rigid, "z": zscore(null),
                "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                "null_p95": float(np.percentile(null, 95)),
                "null_max": float(null.max()),
            },
            "three_block": {
                "internal_dim": int(three.shape[1]),
                "p_empirical": p_rigid3, "z": zscore(null3),
                "null_mean": float(null3.mean()), "null_sd": float(null3.std(ddof=1)),
                "null_p95": float(np.percentile(null3, 95)),
                "null_max": float(null3.max()),
            },
            # The strictest of the three: rigid blocks AND a chain that stays joined.
            "connectivity_constrained": {
                "internal_dim": int(cont.shape[1]),
                "subspace_capture_of_transition": cap_cont,
                "anm_mode1_content": mode1_cont,
                "p_empirical": p_cont, "z": zscore(null_cont),
                "null_mean": float(null_cont.mean()),
                "null_sd": float(null_cont.std(ddof=1)),
                "null_p95": float(np.percentile(null_cont, 95)),
                "null_max": float(null_cont.max()),
                "note": ("Draws are rigid within each block and displace the two residues "
                         "flanking the HB-TBD boundary identically, so they are rotations "
                         "of the TBD about an axis through the junction and only the axis "
                         "direction is free. This concedes everything the elastic-network "
                         "model is credited with except axis selection, and it is a harder "
                         "test than the unconstrained two-lobe null, not an easier one."),
            },
            "junction_continuity": {
                "observed": obs_disc,
                "null_median": float(np.median(null_disc)),
                "fraction_of_draws_as_continuous": frac_as_continuous,
                "n_draws": int(len(null_disc)),
                "note": ("A draw is rigid within each block but nothing holds the blocks "
                         "together, so most draws separate the domains at the boundary "
                         "while the observed transition does not. The null therefore "
                         "concedes block rigidity, not connectivity."),
            },
            # kept for backward compatibility with the two-lobe-only reporting
            "p_random_rigid_direction": p_rigid,
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "n_draws": NDRAW, "seed": SEED,
            "per_mode": per_mode,
        },
    }
    if "--verify" not in sys.argv:
        with open("data/assembly_rigid_null.json", "w") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")

    if "--verify" in sys.argv:
        # Cross-check the RECOMPUTED values against the committed artifact, the way
        # reproduce_modes.py --verify does. Asserting only hardcoded ranges would let the
        # committed JSON drift away from the code that claims to produce it -- and this
        # file carries the primary calibration, so it needs the stronger check.
        import math
        with open("data/assembly_rigid_null.json", encoding="utf-8") as fh:
            committed = json.load(fh)
        crd, cas = committed["rigid_domain_null"], committed["assembly"]
        drift = []

        def same(label, got, want, tol=2e-3):
            if want is None or not math.isfinite(float(got)):
                drift.append(f"{label}: missing in the committed artifact")
            elif abs(float(got) - float(want)) > tol:
                drift.append(f"{label}: recomputed {float(got):.4f} vs committed {float(want):.4f}")

        same("two_block_capture", cap2, crd.get("two_block_capture"))
        same("three_block_capture", cap3, crd.get("three_block_capture"))
        same("observed_mode1_overlap", obs, crd.get("observed_mode1_overlap"))
        same("anm_top10_cumulative", cum10, crd.get("anm_top10_cumulative"))
        same("two_block p", p_rigid, crd.get("two_block", {}).get("p_empirical"))
        same("three_block p", p_rigid3, crd.get("three_block", {}).get("p_empirical"))
        same("connectivity p", p_cont,
             crd.get("connectivity_constrained", {}).get("p_empirical"))
        same("connectivity capture", cap_cont,
             crd.get("connectivity_constrained", {}).get("subspace_capture_of_transition"))
        for m, d in enumerate(crd.get("per_mode", [])):
            same(f"mode {m+1} direction cosine",
                 per_mode[m]["direction_cosine_in_rigid_subspace"],
                 d.get("direction_cosine_in_rigid_subspace"))
        for cut in CUTOFFS:
            k = f"{cut:.1f}"
            same(f"assembly {k} best overlap", assembly[k]["best_overlap"],
                 cas.get("by_cutoff", {}).get(k, {}).get("best_overlap"))
            if assembly[k]["best_mode_rank"] != cas.get("by_cutoff", {}).get(k, {}).get("best_mode_rank"):
                drift.append(f"assembly {k} best mode rank differs from the committed artifact")
        assert not drift, "recomputed values disagree with data/assembly_rigid_null.json:\n  " \
                          + "\n  ".join(drift)
        print(f"cross-checked {len(CUTOFFS) * 2 + 14} values against "
              "data/assembly_rigid_null.json")

        a15 = assembly["15.0"]
        assert a15["best_mode_rank"] >= 4, a15["best_mode_rank"]
        assert a15["mode1_overlap"] < 0.2, a15["mode1_overlap"]
        assert 0.5 < a15["best_overlap"] < 0.75, a15["best_overlap"]
        for cut in CUTOFFS:
            assert assembly[f"{cut:.1f}"]["best_mode_rank"] >= 4, cut
        assert fit_rmsd < 0.01, fit_rmsd
        assert cap2 > cum10, (cap2, cum10)          # the rigid model beats ten ANM modes
        assert 0.9 < cap2 < 0.95, cap2
        assert 0.01 < p_rigid < 0.06, p_rigid
        assert cont.shape[1] == 3, cont.shape          # rotation axis through the junction
        assert p_cont > p_rigid, (p_cont, p_rigid)     # the constrained null is harder
        assert per_mode[0]["direction_cosine_in_rigid_subspace"] > 0.75
        assert max(d["direction_cosine_in_rigid_subspace"] for d in per_mode[1:]) < 0.35
        assert min(d["rigid_content"] for d in per_mode[:3]) > 0.7
        print("\nverify OK: in the deposited assembly the axis is not mode 1; the rigid "
              "interdomain subspace outperforms ten ANM modes; and within it only mode 1 "
              "points along the transition. The unconstrained two-lobe null gives "
              f"p = {p_rigid:.3f}, but the strict joined-chain null gives "
              f"p = {p_cont:.3f}; the claim rests on mode ordering, not significance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
