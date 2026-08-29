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
    motions; only mode 1 points along the transition. The uniform-direction null is evaluated
    exactly because the squared absolute direction cosine follows Beta(1/2, (d-1)/2).

Inputs   render/open_8cvp_assembly.pdb   (Ca-only 8CVP as deposited; --write-assembly
                                          regenerates it from the RCSB mmCIF)
         data/crbn_ensemble.ens.npz, data/pca_diffvec.npz,
         data/crbn_anm_modes.npz, data/crbn_residue_window.csv
Outputs  data/assembly_rigid_null.json
Usage    python scripts/assembly_rigid_null.py [--verify] [--write-assembly]

The assembly eigendecomposition is a 4452 x 4452 problem and takes a few minutes.
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.special import betaincc, betaincinv, gammaln

CUTOFFS = (12.0, 15.0, 18.0)
CUTOFF_ANM = 15.0          # monomer reference cutoff
NMODE = 20
HB_TBD = 318               # TBD starts here; NTD+HB is everything below
NTD_HB_BOUNDARY = 187      # NTD | helical bundle, for the three-block variant
CONTINUITY_NDRAW = 2_000
CONTINUITY_SEED = 2


def analytic_abs_cosine_null(internal_dim, observed):
    """Exact null for ``|u . a|`` with ``u`` uniform on a unit sphere.

    For a fixed unit vector ``a`` and a uniformly random unit direction ``u``
    in ``d`` dimensions, ``|u . a|**2`` follows
    ``Beta(1/2, (d-1)/2)``.  The returned tail probability, moments and 95th
    percentile are therefore analytic and do not depend on a sampled basis,
    random seed, BLAS implementation or finite-draw correction.

    ``p_empirical`` is retained only as a compatibility alias for older
    consumers.  It is exactly equal to ``p_exact`` and is not empirical.
    """
    if isinstance(internal_dim, bool) or not isinstance(internal_dim, (int, np.integer)):
        raise TypeError("internal_dim must be an integer")
    internal_dim = int(internal_dim)
    if internal_dim < 2:
        raise ValueError("internal_dim must be at least 2")
    observed = float(observed)
    if not math.isfinite(observed) or observed < -1e-12 or observed > 1.0 + 1e-12:
        raise ValueError("observed absolute direction cosine must be finite and in [0, 1]")
    observed = float(np.clip(observed, 0.0, 1.0))

    alpha = 0.5
    beta = 0.5 * (internal_dim - 1)
    p_exact = float(betaincc(alpha, beta, observed * observed))
    null_mean = float(math.exp(
        gammaln(internal_dim / 2.0)
        - 0.5 * math.log(math.pi)
        - gammaln((internal_dim + 1.0) / 2.0)
    ))
    null_variance = max(0.0, 1.0 / internal_dim - null_mean * null_mean)
    null_sd = float(math.sqrt(null_variance))
    null_p95 = float(math.sqrt(betaincinv(alpha, beta, 0.95)))
    z = float((observed - null_mean) / null_sd)

    return {
        "internal_dim": internal_dim,
        "p_exact": p_exact,
        "p_empirical": p_exact,
        "p_empirical_note": (
            "Deprecated compatibility alias for p_exact; no empirical draws were used."
        ),
        "null_method": "exact_analytic_beta",
        "null_distribution": {
            "statistic": "absolute_direction_cosine",
            "squared_statistic": "Beta(alpha, beta)",
            "alpha": alpha,
            "beta": beta,
        },
        "z": z,
        "z_definition": (
            "(observed_direction_cosine_in_subspace - null_mean) / null_sd using exact "
            "population moments"
        ),
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_p95": null_p95,
        "null_max": 1.0,
        "null_max_note": "Theoretical upper support bound, not an observed sample maximum.",
    }


def projected_uniform_directions(basis, n_draws, seed):
    """Return reproducible uniform directions in a subspace, basis-orientation invariant.

    Gaussian vectors are generated in the fixed full-coordinate frame and only then
    projected into the subspace. Rotating or sign-flipping an orthonormal basis therefore
    gives the same realised full-space directions, up to floating-point roundoff.
    """
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2 or basis.shape[1] < 1 or not np.isfinite(basis).all():
        raise ValueError("basis must be a finite two-dimensional matrix with at least one column")
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10, rtol=0.0):
        raise ValueError("basis columns must be orthonormal")
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)):
        raise TypeError("n_draws must be an integer")
    if int(n_draws) < 1:
        raise ValueError("n_draws must be positive")
    rng = np.random.default_rng(seed)
    full_draws = rng.standard_normal((int(n_draws), basis.shape[0]))
    coefficients = full_draws @ basis
    norms = np.linalg.norm(coefficients, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("a projected Gaussian direction had zero norm")
    coefficients /= norms
    return coefficients @ basis.T


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


def equal_displacement_subspace(basis, i, j, tol=1e-8):
    """Subspace whose draws give the two boundary nodes identical displacement.

    For ``v = basis @ c``, equality is the three-component linear condition
    ``v_i - v_j = 0``. It leaves three of the six two-block dimensions. This is
    intentionally named for what it imposes: it preserves boundary bond length and
    freezes boundary bond orientation to first order. It is therefore stronger than
    the one-component bond-length condition below.
    """
    M = basis[3 * i:3 * i + 3, :] - basis[3 * j:3 * j + 3, :]
    _, S, Vt = np.linalg.svd(M)
    rank = int((S > tol * S.max()).sum()) if S.size and S.max() > 0 else 0
    return basis @ Vt[rank:].T


def bond_length_preserving_subspace(basis, coords, i, j, tol=1e-8):
    """Subspace preserving the 317-318 bond length to first order.

    For boundary unit vector ``e``, first-order bond extension is
    ``e . (u_j - u_i)``. Setting this scalar to zero leaves five of the six
    two-block internal degrees of freedom. This is the literal first-order bond-length
    constraint; equal displacement is a stronger three-component condition
    that also freezes the bond's instantaneous orientation.
    """
    e = coords[j] - coords[i]
    distance = np.linalg.norm(e)
    if distance <= tol:
        raise ValueError("boundary nodes must have distinct coordinates")
    e = e / distance
    M = e.reshape(1, 3) @ (
        basis[3 * j:3 * j + 3, :] - basis[3 * i:3 * i + 3, :]
    )
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
    diff_artifact = np.load("data/pca_diffvec.npz", allow_pickle=False)
    mode_artifact = np.load("data/crbn_anm_modes.npz", allow_pickle=False)
    V = np.asarray(mode_artifact["anm_eigvecs"])
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    required_ensemble = {"_confs", "_labels"}
    required_difference = {"labels", "open_mask", "diff_vec"}
    if not required_ensemble.issubset(ens.files) or not required_difference.issubset(diff_artifact.files):
        raise ValueError("ensemble or difference artifact is missing required arrays")
    conformers = np.asarray(ens["_confs"])
    ensemble_labels = np.asarray(ens["_labels"])
    difference_labels = np.asarray(diff_artifact["labels"])
    if conformers.ndim != 3 or conformers.shape[2] != 3:
        raise ValueError(f"invalid ensemble coordinate shape: {conformers.shape}")
    if ensemble_labels.shape != (conformers.shape[0],) or not np.array_equal(
        ensemble_labels, difference_labels
    ):
        raise ValueError("difference-artifact labels do not match ensemble label order")
    open_mask = np.asarray(diff_artifact["open_mask"])
    if open_mask.dtype.kind != "b" or open_mask.shape != ensemble_labels.shape:
        raise ValueError("difference-artifact open_mask is not a matching boolean vector")
    if int(open_mask.sum()) != 5 or not np.isfinite(conformers).all():
        raise ValueError("ensemble must contain finite coordinates and exactly five open conformers")
    dvec = np.asarray(diff_artifact["diff_vec"], dtype=float)
    if dvec.shape != (conformers.shape[1] * 3,) or not np.isfinite(dvec).all():
        raise ValueError(f"invalid difference-axis shape: {dvec.shape}")
    dnorm = float(np.linalg.norm(dvec))
    if not np.isclose(dnorm, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"difference axis is not unit length: {dnorm}")
    if V.shape[0] != dvec.size or V.shape[1] < NMODE or not np.isfinite(V).all():
        raise ValueError(f"ANM basis is incompatible with the difference axis: {V.shape}")
    if "resnums" not in mode_artifact.files or not np.array_equal(mode_artifact["resnums"], window):
        raise ValueError("ANM residue labels do not match the analysis-window order")
    labels = [str(value) for value in ensemble_labels]
    if len(labels) != len(set(labels)) or labels.count("8CVP") != 1:
        raise ValueError("ensemble labels must be unique and contain exactly one 8CVP")
    Xmono = conformers[labels.index("8CVP")]               # 269 Ca, ensemble frame

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

    # Characterise both proteins on comparable full-chain node sets. The 269-node window
    # remains the scoring basis because that is where the transition axis is defined, but
    # internal-deformation and amplitude fractions use all deposited Ca atoms of CRBN and
    # DDB1 and therefore partition this two-chain assembly exactly.
    crbn_mask = np.array([c == "B" for c, _ in tag])
    ddb1_mask = np.array([c == "A" for c, _ in tag])
    assert np.all(crbn_mask | ddb1_mask), "8CVP reference must contain only CRBN and DDB1"
    two_body = onb(np.hstack([rigid_dof(Xa, crbn_mask), rigid_dof(Xa, ddb1_mask)]))

    def mode_character(vec):
        """How a joint mode divides into two-body rigid wobble vs internal deformation.

        Both masks cover the complete deposited chains (349 CRBN and 1,135 DDB1 Ca), so
        amplitudes and internal-deformation fractions are directly comparable and the
        squared amplitudes sum to one up to floating-point error.
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

    def direction_null(basis):
        """Compare projected vectors using the exact unit-sphere cosine distribution."""
        mode_coeff = basis.T @ V[:, 0]
        axis_coeff = basis.T @ dvec
        mode_content = float(np.linalg.norm(mode_coeff))
        axis_capture = float(np.linalg.norm(axis_coeff))
        if mode_content <= 1e-12 or axis_capture <= 1e-12:
            raise ValueError("mode or transition has zero projection in a null subspace")
        mode_unit = mode_coeff / mode_content
        axis_unit = axis_coeff / axis_capture
        observed_direction = float(abs(mode_unit @ axis_unit))
        projected_overlap = float(abs((basis @ mode_unit) @ dvec))
        result = dict(analytic_abs_cosine_null(basis.shape[1], observed_direction), **{
            "observed_direction_cosine_in_subspace": observed_direction,
            "observed_projected_mode1_overlap": projected_overlap,
            "anm_mode1_content": mode_content,
            "subspace_capture_of_transition": axis_capture,
        })
        return result, axis_unit

    # Both parameterisations are reported. They answer different questions and they do not
    # agree: how surprising the alignment looks depends on how many hinges the null is
    # allowed. Quoting only the more favourable one would be the same error this analysis
    # was introduced to correct.
    two_null, two_axis_unit = direction_null(two)
    three_null, _ = direction_null(three)
    p_rigid = two_null["p_exact"]
    p_rigid3 = three_null["p_exact"]

    # The third null: rigid blocks that also stay joined at the domain boundary. The
    # Earlier notes stated that such a null "would be stricter still" and that we had
    # not constructed one. It is constructed here, and it is stricter: p rises rather
    # than falls, because constraining the draws to continuous motions leaves only the
    # directions that already resemble the transition.
    index_317 = np.where(window == 317)[0]
    index_318 = np.where(window == 318)[0]
    if index_317.size != 1 or index_318.size != 1:
        raise ValueError("analysis window must contain unique boundary residues 317 and 318")
    junction = np.array([int(index_317[0]), int(index_318[0])])
    cont = equal_displacement_subspace(two, int(junction[0]), int(junction[1]))
    equal_null, _ = direction_null(cont)
    cap_cont = equal_null["subspace_capture_of_transition"]
    p_cont = equal_null["p_exact"]
    bond = bond_length_preserving_subspace(
        two, Xmono, int(junction[0]), int(junction[1])
    )
    bond_null, _ = direction_null(bond)
    cap_bond = bond_null["subspace_capture_of_transition"]
    p_bond = bond_null["p_exact"]

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
    # Generate fixed Gaussian vectors in the 807-dimensional coordinate frame before
    # projection. This makes each realised field invariant to arbitrary rotations or
    # sign flips of the SVD basis spanning the same rigid-motion subspace.
    continuity_fields = projected_uniform_directions(
        two,
        CONTINUITY_NDRAW,
        CONTINUITY_SEED,
    )
    null_disc = np.array([discontinuity(field) for field in continuity_fields])
    frac_as_continuous = float((null_disc <= obs_disc).mean())

    per_mode = []
    for m in range(6):
        p = two.T @ V[:, m]
        rc = float(np.linalg.norm(p))
        per_mode.append({"mode": m + 1, "rigid_content": rc,
                         "direction_cosine_in_rigid_subspace":
                             float(abs(p @ two_axis_unit) / rc) if rc > 1e-9 else 0.0,
                         "overlap_with_axis": float(abs(V[:, m] @ dvec))})
    print(f"\nrigid interdomain subspace captures the transition at projection norm {cap2:.3f} "
          f"(3 blocks {cap3:.3f}); ANM top-10 cumulative = {cum10:.3f}")
    print(f"two-lobe   (dim {two.shape[1]:2d}): in-subspace cosine "
          f"{two_null['observed_direction_cosine_in_subspace']:.3f}, p = {p_rigid:.4f}, "
          f"z = {two_null['z']:.2f} (exact null mean {two_null['null_mean']:.3f}, "
          f"p95 {two_null['null_p95']:.3f})")
    print(f"three-body (dim {three.shape[1]:2d}): in-subspace cosine "
          f"{three_null['observed_direction_cosine_in_subspace']:.3f}, p = {p_rigid3:.4f}, "
          f"z = {three_null['z']:.2f} (exact null mean {three_null['null_mean']:.3f}, "
          f"p95 {three_null['null_p95']:.3f})")
    print(f"equal-displacement boundary (dim {cont.shape[1]:2d}): in-subspace cosine "
          f"{equal_null['observed_direction_cosine_in_subspace']:.3f}, p = {p_cont:.4f}, "
          f"z = {equal_null['z']:.2f} (subspace captures the "
          f"transition at {cap_cont:.3f}; exact null mean {equal_null['null_mean']:.3f})")
    print(f"bond-length-preserving boundary (dim {bond.shape[1]:2d}): in-subspace cosine "
          f"{bond_null['observed_direction_cosine_in_subspace']:.3f}, p = {p_bond:.4f}, "
          f"z = {bond_null['z']:.2f} (subspace captures the "
          f"transition at {cap_bond:.3f}; exact null mean {bond_null['null_mean']:.3f})")
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
            "n_ca_crbn_full": int(crbn_mask.sum()),
            "n_ca_ddb1_full": int(ddb1_mask.sum()),
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
            "two_block": two_null,
            "three_block": three_null,
            # A strong three-component boundary condition: equal displacement freezes
            # both first-order bond extension and instantaneous bond reorientation.
            "equal_displacement_boundary": dict(equal_null, **{
                "note": ("This imposes u317 = u318 (three scalar constraints). It is "
                         "stronger than preserving chain connectivity: it also "
                         "freezes the boundary bond orientation to first order."),
            }),
            # Literal first-order bond-length sensitivity: one scalar constraint.
            "bond_length_preserving_boundary": dict(bond_null, **{
                "note": ("This imposes e317-318 dot (u318-u317) = 0 (one scalar "
                         "constraint), preserving boundary bond length to first order "
                         "while allowing bond reorientation."),
            }),
            "junction_continuity": {
                "observed": obs_disc,
                "null_median": float(np.median(null_disc)),
                "fraction_of_draws_as_continuous": frac_as_continuous,
                "n_draws": int(len(null_disc)),
                "seed": CONTINUITY_SEED,
                "method": "full_space_gaussian_projection_monte_carlo",
                "note": ("Fixed full-coordinate Gaussian draws are projected into the "
                         "rigid subspace, making the seeded diagnostic invariant to SVD "
                         "basis signs and rotations. A draw is rigid within each block but "
                         "nothing holds the blocks together, so most draws separate the "
                         "domains at the boundary while the observed transition does not. "
                         "The null therefore concedes block rigidity, not connectivity."),
            },
            # kept for backward compatibility with the two-lobe-only reporting
            "p_random_rigid_direction": p_rigid,
            "p_random_rigid_direction_note": (
                "Compatibility alias for two_block.p_exact."
            ),
            "null_mean": two_null["null_mean"],
            "null_p95": two_null["null_p95"],
            "n_draws": 0,
            "seed": None,
            "directional_null_note": (
                "Directional nulls are exact analytic distributions; n_draws=0 and seed=null."
            ),
            "per_mode": per_mode,
        },
    }
    # Compatibility alias for older figure/table readers. New reporting must use the
    # explicit equal-displacement name above rather than calling it merely joined.
    out["rigid_domain_null"]["connectivity_constrained"] = dict(
        out["rigid_domain_null"]["equal_displacement_boundary"]
    )
    out["rigid_domain_null"]["connectivity_constrained"]["deprecated_alias"] = True
    if "--verify" not in sys.argv:
        with open("data/assembly_rigid_null.json", "w") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")

    if "--verify" in sys.argv:
        # Cross-check the RECOMPUTED values against the committed artifact, the way
        # reproduce_modes.py --verify does. Asserting only hardcoded ranges would let the
        # committed JSON drift away from the code that claims to produce it -- and this
        # file carries the primary calibration, so it needs the stronger check.
        with open("data/assembly_rigid_null.json", encoding="utf-8") as fh:
            committed = json.load(fh)
        crd, cas = committed["rigid_domain_null"], committed["assembly"]
        drift = []
        checked = 0

        def same(label, got, want, tol=2e-3):
            nonlocal checked
            checked += 1
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
        same("equal-displacement p", p_cont,
             crd.get("equal_displacement_boundary", {}).get("p_empirical"))
        same("equal-displacement capture", cap_cont,
             crd.get("equal_displacement_boundary", {}).get("subspace_capture_of_transition"))
        same("bond-length-preserving p", p_bond,
             crd.get("bond_length_preserving_boundary", {}).get("p_empirical"))
        same("bond-length-preserving capture", cap_bond,
             crd.get("bond_length_preserving_boundary", {}).get("subspace_capture_of_transition"))
        for model_name, recomputed in (
            ("two_block", two_null),
            ("three_block", three_null),
            ("bond_length_preserving_boundary", bond_null),
            ("equal_displacement_boundary", equal_null),
        ):
            committed_model = crd.get(model_name, {})
            for field in (
                "observed_direction_cosine_in_subspace",
                "observed_projected_mode1_overlap",
                "anm_mode1_content",
                "subspace_capture_of_transition",
                "p_exact",
                "p_empirical",
                "z",
                "null_mean",
                "null_sd",
                "null_p95",
                "null_max",
            ):
                tolerance = 1e-10 if field in {
                    "p_exact", "p_empirical", "z", "null_mean", "null_sd",
                    "null_p95", "null_max",
                } else 2e-3
                same(
                    f"{model_name} {field}",
                    recomputed[field],
                    committed_model.get(field),
                    tol=tolerance,
                )
            expected_distribution = recomputed["null_distribution"]
            for field in ("null_method", "p_empirical_note", "z_definition", "null_max_note"):
                checked += 1
                if committed_model.get(field) != recomputed[field]:
                    drift.append(f"{model_name} {field}: committed metadata differs")
            checked += 1
            if committed_model.get("null_distribution") != expected_distribution:
                drift.append(f"{model_name} null_distribution: committed metadata differs")
        for m, d in enumerate(crd.get("per_mode", [])):
            same(f"mode {m+1} direction cosine",
                 per_mode[m]["direction_cosine_in_rigid_subspace"],
                 d.get("direction_cosine_in_rigid_subspace"))
        for cut in CUTOFFS:
            k = f"{cut:.1f}"
            same(f"assembly {k} best overlap", assembly[k]["best_overlap"],
                 cas.get("by_cutoff", {}).get(k, {}).get("best_overlap"))
            checked += 1
            if assembly[k]["best_mode_rank"] != cas.get("by_cutoff", {}).get(k, {}).get("best_mode_rank"):
                drift.append(f"assembly {k} best mode rank differs from the committed artifact")
        checked += 4
        if crd.get("n_draws") != 0:
            drift.append("directional null n_draws must be zero for exact analytic inference")
        if crd.get("seed") is not None:
            drift.append("directional null seed must be null for exact analytic inference")
        if crd.get("directional_null_note") != out["rigid_domain_null"]["directional_null_note"]:
            drift.append("directional null metadata differs from the committed artifact")
        if crd.get("p_random_rigid_direction_note") != out["rigid_domain_null"]["p_random_rigid_direction_note"]:
            drift.append("directional-null compatibility note differs from the committed artifact")
        committed_continuity = crd.get("junction_continuity", {})
        recomputed_continuity = out["rigid_domain_null"]["junction_continuity"]
        for field in ("observed", "null_median", "fraction_of_draws_as_continuous"):
            same(
                f"junction_continuity {field}",
                recomputed_continuity[field],
                committed_continuity.get(field),
            )
        for field in ("n_draws", "seed", "method", "note"):
            checked += 1
            if committed_continuity.get(field) != recomputed_continuity[field]:
                drift.append(f"junction_continuity {field}: committed metadata differs")
        assert not drift, "recomputed values disagree with data/assembly_rigid_null.json:\n  " \
                          + "\n  ".join(drift)
        print(f"cross-checked {checked} values and metadata fields against "
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
        assert bond.shape[1] == 5, bond.shape          # one scalar bond-extension constraint
        assert p_cont > p_rigid, (p_cont, p_rigid)     # the constrained null is harder
        assert p_rigid < p_bond < p_cont, (p_rigid, p_bond, p_cont)
        assert per_mode[0]["direction_cosine_in_rigid_subspace"] > 0.75
        assert max(d["direction_cosine_in_rigid_subspace"] for d in per_mode[1:]) < 0.35
        assert min(d["rigid_content"] for d in per_mode[:3]) > 0.7
        print("\nverify OK: in the deposited assembly the axis is not mode 1; the rigid "
              "interdomain subspace outperforms ten ANM modes; and within it only mode 1 "
              "points along the transition. The unconstrained two-lobe null gives "
              f"p = {p_rigid:.3f}; the bond-length-preserving boundary gives "
              f"p = {p_bond:.3f}; the stronger equal-displacement boundary gives "
              f"p = {p_cont:.3f}. The conclusion rests on mode ordering, not significance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
