#!/usr/bin/env python3
"""Is overlap 0.744 large under a null that already knows CRBN is a multi-domain protein?

the primary published null is isotropic: random unit vectors in the full 807-d
Cartesian space (data/anm_null_significance.json, z = 33.6).  That null is
over-permissive.  A reader's objection is concrete: CRBN is a three-domain protein,
the open->closed transition is largely a domain hinge, and ANM mode 1 of any such fold
is largely a domain hinge as well.  A null of random directions in 807-d therefore tests
a hypothesis nobody holds -- it asks whether the mode is better than a random
*conformational scramble*, not whether it is better than a random *domain motion*.

This script replaces it with nulls that concede the domain architecture and ask what is
left:

  NULL A  3-domain rigid-body subspace (18-d): random unit directions inside the span of
          3 translations + 3 rotations for each of NTD / HB / TBD.
  NULL B  2-lobe hinge subspace (12-d): the same for (NTD+HB) vs TBD, i.e. the single
          hinge a reader would propose as the trivial explanation.
  NULL C  degree-preserving contact-map rewiring (this work): the real Ca coordinates are
          kept, so every residue keeps its exact contact NUMBER, but the contact
          TOPOLOGY is randomised by double-edge swaps.  Mode 1 of each rewired network is
          scored against the transition.  This isolates the contribution of the native
          contact pattern from that of contact density and molecular shape -- neither the
          isotropic nor the rigid-body null does that.
  NULL D  higher ANM modes 4..20 of the same structure (the primary structural null,
          recomputed here for a common reporting frame).

It also DECOMPOSES the transition, because that is the substantive result: how much of
the open->closed difference vector, and of each ANM mode, is rigid-body domain motion,
and how well a plain per-domain Kabsch fit reproduces the transition.

Inputs   data/crbn_ensemble.ens.npz, data/pca_diffvec.npz, data/crbn_residue_window.csv
Outputs  data/rigidbody_null.json
Usage    python scripts/rigidbody_null.py [--verify] [--ndraw N]
"""
import csv
import json
import os
import sys

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmode_lib as L

CUTOFF = 15.0
NMODES = 20
SEED = 20260720
NDRAW = 20000
NREWIRE = 200
REF = "8CVP"                      # open reference The analysis builds the ANM on
# Domain boundaries as published (author numbering); the window ends at 424.
DOMAINS_3 = {"NTD": (77, 186), "HB": (187, 317), "TBD": (318, 426)}
DOMAINS_2 = {"NTD+HB": (77, 317), "TBD": (318, 426)}


def sparse_anm_hessian_from_pairs(coords, i, j):
    """Sparse ANM Hessian for an existing contact topology."""
    n = len(coords)
    dv = coords[j] - coords[i]
    r2 = (dv * dv).sum(axis=1)
    blocks = dv[:, :, None] * dv[:, None, :] / r2[:, None, None]
    rows, cols, data = [], [], []
    for a, b, B in zip(i, j, blocks):
        ia = 3 * int(a)
        ib = 3 * int(b)
        for u in range(3):
            for v in range(3):
                val = float(B[u, v])
                rows.extend((ia + u, ib + u, ia + u, ib + u))
                cols.extend((ia + v, ib + v, ib + v, ia + v))
                data.extend((val, val, -val, -val))
    return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(3 * n, 3 * n)).tocsr()


def low_anm_modes(H, k=6, tol=1e-9, extra=14):
    """Lowest `k` non-trivial ANM modes using a bounded eigensolve.

    Contact-rewired verification evaluates hundreds of 807x807 Hessians. A full
    dense eigensolve per draw dominates runtime, while the statistic only needs
    mode 1 and its rank among the first few low modes. ANM Hessians should have
    six rigid-body zero modes, but rewiring can occasionally add near-zero modes,
    so expand the requested subset until enough non-trivial modes are available.
    """
    n = H.shape[0]
    if scipy.sparse.issparse(H):
        hi = min(n - 2, k + 6 + extra)
        v0 = np.linspace(-1.0, 1.0, n)
        v0 /= np.linalg.norm(v0)
        while True:
            w, v = scipy.sparse.linalg.eigsh(
                H, k=hi + 1, which="SM", tol=1e-8, ncv=min(n, max(2 * (hi + 1) + 1, 40)),
                v0=v0,
            )
            order = np.argsort(w)
            w = w[order]
            v = v[:, order]
            nz = w > tol
            if int(nz.sum()) >= k or hi >= n - 2:
                return w[nz][:k], v[:, nz][:, :k]
            hi = min(n - 2, max(hi + extra, 2 * hi + 1))

    hi = min(n - 1, k + 6 + extra - 1)
    while True:
        w, v = scipy.linalg.eigh(H, subset_by_index=(0, hi), check_finite=False)
        nz = w > tol
        if int(nz.sum()) >= k or hi == n - 1:
            return w[nz][:k], v[:, nz][:, :k]
        hi = min(n - 1, max(hi + extra, 2 * hi + 1))


def domain_indices(resnums, spans):
    return [np.where((resnums >= a) & (resnums <= b))[0] for a, b in spans.values()]


def main():
    verify = "--verify" in sys.argv
    ndraw = NDRAW
    if "--ndraw" in sys.argv:
        ndraw = int(sys.argv[sys.argv.index("--ndraw") + 1])

    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    confs = ens["_confs"]
    labels = [str(x)[:4] for x in ens["_labels"]]
    resnums = np.array([int(r["author_resnum"]) for r in
                        csv.DictReader(open("data/crbn_residue_window.csv"))])
    assert len(resnums) == confs.shape[1], (len(resnums), confs.shape)

    d = np.load("data/pca_diffvec.npz")
    dvec = d["diff_vec"] / np.linalg.norm(d["diff_vec"])
    open_mask = d["open_mask"].astype(bool)

    ref = confs[labels.index(REF)]
    w, V = L.modes(L.anm_hessian(ref, CUTOFF), NMODES)
    ov = L.mode_overlaps(V, dvec)
    obs = float(ov[0])
    rank = int(np.argmax(ov) + 1)

    idx3 = domain_indices(resnums, DOMAINS_3)
    idx2 = domain_indices(resnums, DOMAINS_2)
    # This script is superseded by assembly_rigid_null.py, which uses the INTERNAL
    # subspace. Pin internal=False so it keeps reproducing its own committed JSON
    # instead of silently changing meaning when the library default changed.
    B3 = L.rigid_body_basis(ref, idx3, internal=False)
    B2 = L.rigid_body_basis(ref, idx2, internal=False)

    # ---------------------------------------------------------------- nulls ---
    nulls = {}
    nulls["isotropic_807d"] = L.null_summary(
        L.isotropic_null(dvec, ndraw=ndraw, seed=SEED), obs, "isotropic 807-d")
    nulls["rigid_body_3domain_18d"] = L.null_summary(
        L.rigid_body_null(dvec, B3, ndraw=ndraw, seed=SEED), obs,
        "random rigid-body motions of NTD/HB/TBD")
    nulls["rigid_body_2lobe_12d"] = L.null_summary(
        L.rigid_body_null(dvec, B2, ndraw=ndraw, seed=SEED), obs,
        "random rigid-body motions of (NTD+HB)/TBD")

    # NULL C: degree-preserving contact rewiring
    rng = np.random.default_rng(SEED)
    ci, cj, _ = L.contact_pairs(ref, CUTOFF)
    deg = np.bincount(np.concatenate([ci, cj]), minlength=len(ref))
    rew_ov, rew_rank, deg_ok = [], [], True
    for _ in range(NREWIRE):
        ri, rj = L.rewire_contacts(ci, cj, len(ref), rng)
        deg_new = np.bincount(np.concatenate([ri, rj]), minlength=len(ref))
        if not np.array_equal(deg, deg_new):
            deg_ok = False
        Hr = sparse_anm_hessian_from_pairs(ref, ri, rj)
        wr, Vr = low_anm_modes(Hr, 6)
        if Vr.shape[1] == 0:
            continue
        o = L.mode_overlaps(Vr, dvec)
        rew_ov.append(float(o[0]))
        rew_rank.append(int(np.argmax(o) + 1))
    nulls["contact_rewired_mode1"] = L.null_summary(np.array(rew_ov), obs,
                                                   "mode 1 of degree-preserving rewired networks")
    nulls["contact_rewired_mode1"]["degree_sequence_preserved"] = bool(deg_ok)
    nulls["contact_rewired_mode1"]["n_contacts"] = int(len(ci))
    nulls["contact_rewired_mode1"]["mean_contact_number"] = float(deg.mean())

    nulls["higher_anm_modes_4_20"] = L.null_summary(
        np.abs([V[:, m] @ dvec for m in range(3, NMODES)]), obs, "ANM modes 4-20")

    # ------------------------------------------------ transition decomposition ---
    rb_dv_3 = L.rigid_body_content(dvec, B3)
    rb_dv_2 = L.rigid_body_content(dvec, B2)
    mode_rb = {f"mode{m+1}": {"rb_3domain": L.rigid_body_content(V[:, m], B3),
                              "rb_2lobe": L.rigid_body_content(V[:, m], B2),
                              "overlap_with_transition": float(ov[m]),
                              "eigenvalue": float(w[m])}
               for m in range(6)}

    open_c = confs[open_mask].mean(0)
    closed_c = confs[~open_mask].mean(0)
    field3, dom_rmsd3 = L.rigid_body_fit_field(open_c, closed_c, idx3)
    field2, dom_rmsd2 = L.rigid_body_fit_field(open_c, closed_c, idx2)
    # sign convention: transition vector points open -> closed
    dv_oc = (closed_c - open_c).reshape(-1)
    dv_oc /= np.linalg.norm(dv_oc)
    fit_overlap_3 = float(abs(field3 @ dv_oc))
    fit_overlap_2 = float(abs(field2 @ dv_oc))
    mode1_vs_field3 = float(abs(V[:, 0] @ field3))

    # internal (non-rigid) residual of the transition
    resid3 = float(np.sqrt(max(0.0, 1 - rb_dv_3 ** 2)))
    # amplitude of the transition, and of its internal part, in Angstrom
    amp = float(np.linalg.norm((closed_c - open_c).reshape(-1)))
    rmsd_oc = float(np.sqrt(((closed_c - open_c) ** 2).sum(1).mean()))

    out = {
        "meta": {"reference": REF, "cutoff": CUTOFF, "n_modes": NMODES, "seed": SEED,
                 "n_draws": ndraw, "n_rewire": NREWIRE,
                 "domains_3": {k: list(v) for k, v in DOMAINS_3.items()},
                 "domains_2": {k: list(v) for k, v in DOMAINS_2.items()},
                 "subspace_rank_3domain": int(B3.shape[1]),
                 "subspace_rank_2lobe": int(B2.shape[1]),
                 # Disambiguation, because these labels collide with the primary.
                 "parameterisation": "internal=False (whole-molecule block retained)",
                 "SUPERSEDED_BY": "data/assembly_rigid_null.json",
                 "WARNING": (
                     "These nulls keep the six whole-molecule translations and rotations "
                     "inside the subspace, so the two-lobe basis here has rank 12 and the "
                     "three-domain basis rank 18. the primary nulls are the INTERNAL "
                     "ones in data/assembly_rigid_null.json, of rank 6 (two lobes, "
                     "p = 0.030, z = 2.05), rank 12 (three domains, p = 0.0012, z = 3.38) "
                     "and rank 3 (two lobes with the chain kept joined, p = 0.16, "
                     "z = 1.17). Do not read 'rigid_body_2lobe_12d' below as the primary "
                     "two-lobe null: it is the earlier global-included variant, retained "
                     "only so this file keeps reproducing itself. Spending probability "
                     "mass on six directions a superposed difference vector cannot occupy "
                     "inflates the apparent significance.")},
        "observed": {"mode1_overlap": obs, "best_mode_rank": rank,
                     "cumulative_overlap": {f"k{k}": L.cumulative_overlap(V, dvec, k)
                                            for k in (1, 2, 3, 5, 10, 20)}},
        "nulls": nulls,
        "decomposition": {
            "transition_rigid_body_amplitude_3domain": rb_dv_3,
            "transition_rigid_body_variance_3domain": rb_dv_3 ** 2,
            "transition_rigid_body_amplitude_2lobe": rb_dv_2,
            "transition_rigid_body_variance_2lobe": rb_dv_2 ** 2,
            "transition_internal_amplitude_3domain": resid3,
            "transition_ca_rmsd_open_to_closed": rmsd_oc,
            "transition_norm": amp,
            "anm_modes_rigid_body_content": mode_rb,
            "per_domain_kabsch_fit": {
                "field_vs_transition_overlap_3domain": fit_overlap_3,
                "field_vs_transition_overlap_2lobe": fit_overlap_2,
                "anm_mode1_vs_fit_field_3domain": mode1_vs_field3,
                "per_domain_internal_rmsd_3domain": dict(zip(DOMAINS_3, dom_rmsd3)),
                "per_domain_internal_rmsd_2lobe": dict(zip(DOMAINS_2, dom_rmsd2)),
            },
        },
    }
    if not verify:
        with open("data/rigidbody_null.json", "w", encoding="utf-8") as _fh:
            json.dump(out, _fh, indent=1)

    print(f"observed mode-1 overlap {obs:.4f} (rank {rank})")
    for k, v in nulls.items():
        print(f"  {k:28s} mean {v['mean']:.3f} sd {v['sd']:.3f} max {v['max']:.3f} "
              f"z {v['z']:+.1f} p {v['p_empirical']:.2g}"
              f"{'  [NULL MAX EXCEEDS OBSERVED]' if v['null_max_exceeds_observed'] else ''}")
    print(f"transition rigid-body content: {rb_dv_3:.3f} amplitude "
          f"({100*rb_dv_3**2:.0f}% of squared amplitude), 2-lobe {rb_dv_2:.3f}")
    print(f"ANM mode 1 rigid-body content: {mode_rb['mode1']['rb_3domain']:.3f}")
    print(f"per-domain Kabsch field vs transition: {fit_overlap_3:.4f}; "
          f"ANM mode 1 vs that field: {mode1_vs_field3:.4f}")

    if verify:
        a = out["nulls"]
        assert abs(obs - 0.744) < 5e-3, obs
        assert rank == 1, rank
        assert a["isotropic_807d"]["z"] > 30, a["isotropic_807d"]["z"]
        assert 3.5 < a["rigid_body_3domain_18d"]["z"] < 5.5, a["rigid_body_3domain_18d"]["z"]
        assert 2.5 < a["rigid_body_2lobe_12d"]["z"] < 4.5, a["rigid_body_2lobe_12d"]["z"]
        assert a["rigid_body_2lobe_12d"]["null_max_exceeds_observed"], \
            "the 12-d null max must be reported even though it exceeds the observation"
        assert a["contact_rewired_mode1"]["degree_sequence_preserved"]
        assert a["contact_rewired_mode1"]["z"] > 3, a["contact_rewired_mode1"]["z"]
        assert 0.90 < rb_dv_3 < 0.95, rb_dv_3
        assert 0.84 < rb_dv_3 ** 2 < 0.88, rb_dv_3 ** 2
        assert mode_rb["mode1"]["rb_3domain"] > 0.98, mode_rb["mode1"]["rb_3domain"]
        assert fit_overlap_3 > 0.97, fit_overlap_3
        print("verify OK: rigid-body-subspace nulls give z = "
              f"{a['rigid_body_3domain_18d']['z']:.1f} (18-d) and "
              f"{a['rigid_body_2lobe_12d']['z']:.1f} (12-d), not "
              f"{a['isotropic_807d']['z']:.0f}; transition is "
              f"{100*rb_dv_3**2:.0f}% rigid-body domain motion")


if __name__ == "__main__":
    main()
