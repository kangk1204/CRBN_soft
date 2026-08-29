#!/usr/bin/env python3
"""Shared numerics for the control-panel / null-model analyses.

Everything here is numpy-only and deterministic. It is imported by
  scripts/rigidbody_null.py      (rigid-body subspace nulls + transition decomposition)
  scripts/control_panel.py       (positive/negative control benchmark)
  scripts/anm_sensitivity_ext.py (cutoff, spring, chain-break, degeneracy, subspace)
  scripts/drug_loop_statistics.py

Design notes that matter for correctness:

* FRAME.  ANM eigenvectors live in the reference frame of the coordinates the
  Hessian was built from.  The committed open->closed difference vector
  (data/pca_diffvec.npz) lives in the frame of the superposed ensemble
  (data/crbn_ensemble.ens.npz).  Scoring modes from RAW deposited coordinates
  against that vector is a frame mismatch and gives wrong (too low) overlaps.
  Use `kabsch_apply` to rotate raw coordinates onto the corresponding ensemble
  member before scoring, or rotate the difference vector instead.

* RIGID-BODY SUBSPACE.  For a domain decomposition, the infinitesimal
  rigid-body motions are 3 translations + 3 rotations per domain, evaluated at
  the domain centroid, assembled as columns in the full 3N space and
  QR-orthonormalised.  A k-domain decomposition gives 6k columns (rank 6k for
  domains with >= 3 non-collinear atoms).
"""
import gzip
import json
import os
import urllib.parse
import urllib.request

import numpy as np

try:
    from pdb_id import validate_pdb_id, validate_polymer_entity_id
except ModuleNotFoundError:  # imported by path from the repository root
    from scripts.pdb_id import validate_pdb_id, validate_polymer_entity_id

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "_cif_cache")
META_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "_rcsb_meta")
# Verification callers disable this process-local switch. Existing caches remain
# readable, but a cache miss is served in memory rather than mutating the worktree.
CACHE_WRITES_ENABLED = True


# ---------------------------------------------------------------- ANM / GNM ---
def contact_pairs(coords, cutoff):
    """Indices (i<j) of Ca pairs within `cutoff`. Vectorised."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    iu = np.triu_indices(len(coords), 1)
    sel = (d[iu] > 1e-6) & (d[iu] <= cutoff)
    return iu[0][sel], iu[1][sel], d[iu][sel]


def anm_hessian(coords, cutoff=15.0, gamma="uniform", extra_pairs=None):
    """Anisotropic network model Hessian (3N x 3N), vectorised over contacts.

    gamma: 'uniform'            k = 1                     (the reference setting)
           'r2' / 'r6'          k = (cutoff/r)**2 or **6  (parameter-free, distance-weighted)
           float                uniform with that spring constant
    extra_pairs: optional (i, j) arrays of additional springs to add regardless of
           cutoff -- used for backbone connectivity restraints across chain breaks.
    """
    i, j, r = contact_pairs(coords, cutoff)
    if extra_pairs is not None:
        ei, ej = extra_pairs
        if len(ei):
            er = np.linalg.norm(coords[ei] - coords[ej], axis=1)
            keep = er > 1e-6
            i = np.concatenate([i, ei[keep]])
            j = np.concatenate([j, ej[keep]])
            r = np.concatenate([r, er[keep]])
            # de-duplicate (a backbone pair may already be inside the cutoff)
            key = i.astype(np.int64) * (len(coords) + 1) + j
            _, uniq = np.unique(key, return_index=True)
            i, j, r = i[uniq], j[uniq], r[uniq]
    if gamma == "uniform":
        k = np.ones_like(r)
    elif gamma == "r2":
        k = (cutoff / r) ** 2
    elif gamma == "r6":
        k = (cutoff / r) ** 6
    else:
        k = np.full_like(r, float(gamma))

    n = len(coords)
    H = np.zeros((3 * n, 3 * n))
    dv = coords[j] - coords[i]
    # outer products scaled by k / r^2
    blocks = (k / r ** 2)[:, None, None] * (dv[:, :, None] * dv[:, None, :])
    for a, b, B in zip(i, j, blocks):
        H[3 * a:3 * a + 3, 3 * b:3 * b + 3] -= B
        H[3 * b:3 * b + 3, 3 * a:3 * a + 3] -= B
        H[3 * a:3 * a + 3, 3 * a:3 * a + 3] += B
        H[3 * b:3 * b + 3, 3 * b:3 * b + 3] += B
    return H


def gnm_kirchhoff(coords, cutoff=7.3):
    i, j, _ = contact_pairs(coords, cutoff)
    n = len(coords)
    K = np.zeros((n, n))
    K[i, j] -= 1.0
    K[j, i] -= 1.0
    np.add.at(K, (i, i), 1.0)
    np.add.at(K, (j, j), 1.0)
    return K


def modes(H, k=20, tol=1e-9):
    """Lowest `k` non-trivial eigenpairs (ascending). Zero modes dropped by tolerance."""
    w, v = np.linalg.eigh(H)
    nz = w > tol
    return w[nz][:k], v[:, nz][:, :k]


def sqfluct(w, v, nmodes=10):
    """Per-residue square fluctuation from the `nmodes` slowest modes, 1/lambda weighted."""
    n = v.shape[0] // 3
    f = np.zeros(n)
    for m in range(min(nmodes, v.shape[1])):
        f += (v[:, m].reshape(n, 3) ** 2).sum(1) / w[m]
    return f


# ------------------------------------------------------------ superposition ---
def kabsch(P, Q):
    """Rotation+translation taking P onto Q (both (n,3)). Returns (R, t, rmsd)."""
    pc, qc = P.mean(0), Q.mean(0)
    A = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    Pr = (R @ (P - pc).T).T + qc
    return R, qc - (R @ pc), float(np.sqrt(((Pr - Q) ** 2).sum(1).mean()))


def kabsch_apply(P, Q):
    """Superpose P onto Q and return the transformed P (internal geometry unchanged)."""
    R, t, _ = kabsch(P, Q)
    return (R @ P.T).T + t


def superpose_rmsd(P, Q):
    return kabsch(P, Q)[2]


# ------------------------------------------------------- rigid-body subspace ---
def rigid_body_basis(coords, domains, internal=True):
    """Orthonormal basis (3N x r) of infinitesimal rigid-body motions.

    domains: list of index arrays partitioning (or covering) the atoms.
    Columns per domain: 3 translations + 3 rotations about the domain centroid.

    internal=True (default) removes the six whole-molecule translations and rotations from
    the span, leaving only motions of the domains RELATIVE to each other. That is almost
    always what is wanted. A superposed difference vector has no global component, so a
    null that keeps the global block spends probability mass on six directions the observed
    statistic cannot occupy, which inflates the apparent significance: for the CRBN
    transition the two-block null moves from p = 0.030 to a substantially smaller value if
    the global block is left in. Pass internal=False only to reproduce that earlier
    behaviour deliberately.
    """
    n = len(coords)
    cols = []
    for idx in domains:
        idx = np.asarray(idx)
        c = coords[idx].mean(0)
        rel = coords[idx] - c
        for ax in range(3):
            t = np.zeros((n, 3))
            t[idx, ax] = 1.0
            cols.append(t.reshape(-1))
        for ax in range(3):
            e = np.zeros(3)
            e[ax] = 1.0
            rot = np.zeros((n, 3))
            rot[idx] = np.cross(e, rel)
            cols.append(rot.reshape(-1))
    B = np.column_stack(cols)
    Q, R = np.linalg.qr(B)
    keep = np.abs(np.diag(R)) > 1e-8
    Q = Q[:, keep]
    if internal:
        whole = [np.arange(n)]
        G = rigid_body_basis(coords, whole, internal=False)
        Q = Q - G @ (G.T @ Q)
        U, S, _ = np.linalg.svd(Q, full_matrices=False)
        Q = U[:, S > 1e-8 * S.max()]
    return Q


def rigid_body_content(vec, basis):
    """Fraction of |vec| inside the subspace (amplitude). Square it for variance share."""
    v = np.asarray(vec, float).reshape(-1)
    v = v / np.linalg.norm(v)
    return float(np.linalg.norm(basis.T @ v))


def rigid_body_null(dvec, basis, ndraw=20000, seed=20260720):
    """Overlaps of random unit vectors DRAWN INSIDE the rigid-body subspace with dvec.

    Because the basis is orthonormal, a uniform direction in the subspace is
    Q @ c with c uniform on the unit sphere of dimension r; the resulting vector
    is already unit-norm in the full 3N space.
    """
    rng = np.random.default_rng(seed)
    r = basis.shape[1]
    C = rng.standard_normal((ndraw, r))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    p = basis.T @ (np.asarray(dvec, float).reshape(-1) / np.linalg.norm(dvec))
    return np.abs(C @ p)


def isotropic_null(dvec, dim=None, ndraw=20000, seed=20260720):
    rng = np.random.default_rng(seed)
    d = np.asarray(dvec, float).reshape(-1)
    d = d / np.linalg.norm(d)
    dim = dim or d.size
    R = rng.standard_normal((ndraw, dim))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return np.abs(R @ d)


def null_summary(null, obs, label=""):
    null = np.asarray(null, float)
    n_ex = int((null >= obs).sum())
    return {
        "label": label,
        "n_draws": int(null.size),
        "mean": float(null.mean()),
        "sd": float(null.std(ddof=1)),
        "p50": float(np.percentile(null, 50)),
        "p95": float(np.percentile(null, 95)),
        "p99": float(np.percentile(null, 99)),
        "p999": float(np.percentile(null, 99.9)),
        "max": float(null.max()),
        "observed": float(obs),
        "z": float((obs - null.mean()) / null.std(ddof=1)),
        "n_exceedances": n_ex,
        "p_empirical": (n_ex + 1) / (null.size + 1),   # add-one; 0 exceedances != p=0
        "null_max_exceeds_observed": bool(null.max() > obs),
    }


def rigid_body_fit_field(open_c, closed_c, domains):
    """Displacement field produced by fitting each domain of `open_c` onto `closed_c`
    rigidly (per-domain Kabsch), expressed in the same frame as open_c.
    Returns (unit vector 3N, per-domain internal rmsd list)."""
    pred = open_c.copy()
    dom_rmsd = []
    for idx in domains:
        idx = np.asarray(idx)
        R, t, rms = kabsch(open_c[idx], closed_c[idx])
        pred[idx] = (R @ open_c[idx].T).T + t
        dom_rmsd.append(float(rms))
    f = (pred - open_c).reshape(-1)
    return f / np.linalg.norm(f), dom_rmsd


# --------------------------------------------------- degree-preserving rewire ---
def rewire_contacts(i, j, n, rng, nswap_mult=5):
    """Degree-preserving double-edge-swap randomisation of an undirected simple graph.

    Repeatedly picks two edges (a,b), (c,d) with four distinct endpoints and replaces
    them by (a,c), (b,d) when neither replacement already exists. This is the standard
    configuration-model randomisation: every node's degree is exactly invariant, while
    the contact TOPOLOGY is scrambled. `nswap_mult` x |E| accepted swaps are performed.
    Returns new (i, j) index arrays (i < j).
    """
    edges = {(int(min(a, b)), int(max(a, b))) for a, b in zip(i, j)}
    E = list(edges)
    m = len(E)
    target = int(nswap_mult * m)
    done = attempts = 0
    max_attempts = 40 * target
    randint = rng.integers
    while done < target and attempts < max_attempts:
        attempts += 1
        x = int(randint(m)); y = int(randint(m))
        if x == y:
            continue
        a, b = E[x]
        c, d = E[y]
        if randint(2):
            c, d = d, c
        if a == c or a == d or b == c or b == d:
            continue
        e1 = (a, c) if a < c else (c, a)
        e2 = (b, d) if b < d else (d, b)
        if e1 in edges or e2 in edges:
            continue
        edges.discard(E[x]); edges.discard(E[y])
        edges.add(e1); edges.add(e2)
        E[x] = e1; E[y] = e2
        done += 1
    arr = np.array(sorted(edges), dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def anm_hessian_from_pairs(coords, i, j, gamma="uniform", cutoff=15.0):
    """Hessian for an explicit contact list (used by the rewired-contact null)."""
    n = len(coords)
    H = np.zeros((3 * n, 3 * n))
    dv = coords[j] - coords[i]
    r = np.linalg.norm(dv, axis=1)
    ok = r > 1e-6
    i, j, dv, r = i[ok], j[ok], dv[ok], r[ok]
    if gamma == "uniform":
        k = np.ones_like(r)
    elif gamma == "r2":
        k = (cutoff / r) ** 2
    else:
        k = np.full_like(r, float(gamma))
    blocks = (k / r ** 2)[:, None, None] * (dv[:, :, None] * dv[:, None, :])
    for a, b, B in zip(i, j, blocks):
        H[3 * a:3 * a + 3, 3 * b:3 * b + 3] -= B
        H[3 * b:3 * b + 3, 3 * a:3 * a + 3] -= B
        H[3 * a:3 * a + 3, 3 * a:3 * a + 3] += B
        H[3 * b:3 * b + 3, 3 * b:3 * b + 3] += B
    return H


# ------------------------------------------------------------ overlap metrics ---
def mode_overlaps(V, dvec):
    d = np.asarray(dvec, float).reshape(-1)
    d = d / np.linalg.norm(d)
    Vn = V / np.linalg.norm(V, axis=0, keepdims=True)
    return np.abs(Vn.T @ d)


def cumulative_overlap(V, dvec, k):
    o = mode_overlaps(V, dvec)
    return float(np.sqrt((o[:k] ** 2).sum()))


def rmsip(A, B, k=10):
    """Root mean square inner product between the first k columns of two mode sets."""
    a = A[:, :k] / np.linalg.norm(A[:, :k], axis=0, keepdims=True)
    b = B[:, :k] / np.linalg.norm(B[:, :k], axis=0, keepdims=True)
    return float(np.sqrt(((a.T @ b) ** 2).sum() / k))


def subspace_overlap(V, dvec, k):
    """|projection of dvec onto span(first k modes)| -- identical to cumulative_overlap
    for orthonormal V, kept as a named concept for the subspace formulation."""
    return cumulative_overlap(V, dvec, k)


# ----------------------------------------------------------------- structures ---
def fetch_cif(pdb, cache=None):
    pdb = validate_pdb_id(pdb)
    cache = cache or CACHE
    p = os.path.join(cache, f"{pdb.upper()}.cif.gz")
    if os.path.exists(p):
        with gzip.open(p, "rt") as fh:
            return fh.read()
    with urllib.request.urlopen(
            f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz", timeout=120) as fh:
        blob = fh.read()
    text = gzip.decompress(blob).decode("utf-8")
    if CACHE_WRITES_ENABLED:
        os.makedirs(cache, exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(blob)
    return text


def parse_atoms(cif, atom="CA"):
    """Parse mmCIF ATOM records for one atom name.

    Returns {chain: {auth_resnum: dict(xyz, b, occ, altloc, comp)}}, keeping the
    FIRST occurrence of each (chain, resnum) -- the repo parser's convention -- and
    separately counting altloc-labelled and partial-occupancy records so the
    crystallographic bookkeeping can be reported rather than silently dropped.
    """
    out, stats = {}, {"n_altloc": 0, "n_partial_occ": 0, "n_records": 0}
    lines = cif.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            hdr = []
            while j < len(lines) and lines[j].lstrip().startswith("_atom_site."):
                hdr.append(lines[j].strip())
                j += 1
            if hdr:
                col = {h.split(".", 1)[1]: k for k, h in enumerate(hdr)}
                need = ["label_atom_id", "auth_asym_id", "auth_seq_id",
                        "Cartn_x", "Cartn_y", "Cartn_z", "group_PDB"]
                if all(c in col for c in need):
                    k = j
                    while k < len(lines):
                        ln = lines[k]
                        s = ln.strip()
                        if s.startswith("#") or s == "" or s == "loop_" or ln.startswith("_"):
                            break
                        f = ln.split()
                        if len(f) < len(hdr):
                            k += 1
                            continue
                        if f[col["group_PDB"]] != "ATOM" or \
                           f[col["label_atom_id"]].strip('"') != atom:
                            k += 1
                            continue
                        ch = f[col["auth_asym_id"]]
                        try:
                            ri = int(f[col["auth_seq_id"]])
                        except ValueError:
                            k += 1
                            continue
                        alt = f[col["label_alt_id"]] if "label_alt_id" in col else "."
                        try:
                            occ = float(f[col["occupancy"]]) if "occupancy" in col else 1.0
                        except ValueError:
                            occ = 1.0
                        try:
                            b = float(f[col["B_iso_or_equiv"]]) if "B_iso_or_equiv" in col else np.nan
                        except ValueError:
                            b = np.nan
                        stats["n_records"] += 1
                        if alt not in (".", "?", "A"):
                            stats["n_altloc"] += 1
                        elif alt == "A":
                            stats["n_altloc"] += 1
                        if occ < 0.999:
                            stats["n_partial_occ"] += 1
                        rec = out.setdefault(ch, {})
                        if ri not in rec:
                            rec[ri] = {
                                "xyz": [float(f[col["Cartn_x"]]), float(f[col["Cartn_y"]]),
                                        float(f[col["Cartn_z"]])],
                                "b": b, "occ": occ, "alt": alt,
                                "comp": f[col["label_comp_id"]] if "label_comp_id" in col else "",
                            }
                        k += 1
                    i = k
                    continue
        i += 1
    return out, stats


def rcsb_meta(pdb, cache=None):
    """Deposition quality metadata from the RCSB data API (cached to disk)."""
    cache = cache or META_CACHE
    p = os.path.join(cache, f"{pdb.upper()}.json")
    if os.path.exists(p):
        d = json.load(open(p))
    else:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb.upper()}"
        with urllib.request.urlopen(url, timeout=60) as fh:
            d = json.load(fh)
        if CACHE_WRITES_ENABLED:
            os.makedirs(cache, exist_ok=True)
            with open(p, "w", encoding="utf-8") as _fh:
                json.dump(d, _fh)
    info = d.get("rcsb_entry_info", {})
    res = info.get("resolution_combined") or []
    method = (d.get("exptl") or [{}])[0].get("method", "")
    refine = d.get("refine") or [{}]
    rfree = refine[0].get("ls_R_factor_R_free")
    rwork = refine[0].get("ls_R_factor_R_work") or refine[0].get("ls_R_factor_obs")
    em = (d.get("em_3d_reconstruction") or [{}])[0].get("resolution")
    cell = d.get("cell", {})
    sg = (d.get("symmetry") or {}).get("space_group_name_hm")
    return {
        "pdb": pdb.upper(),
        "method": method,
        "resolution": float(res[0]) if res else (float(em) if em else None),
        "em_resolution": float(em) if em else None,
        "r_free": float(rfree) if rfree is not None else None,
        "r_work": float(rwork) if rwork is not None else None,
        "space_group": sg,
        "cell": [cell.get("length_a"), cell.get("length_b"), cell.get("length_c")],
        "title": (d.get("struct") or {}).get("title", ""),
        "deposited_polymer_entities": info.get("polymer_entity_count_protein"),
    }


def isotropic_null_exact(dim, ndraw=20000, seed=20260720):
    """Exact marginal distribution of |cos| between a uniform unit vector in `dim`
    dimensions and any fixed direction, sampled in O(ndraw).

    For z ~ N(0, I_dim), the component along the fixed axis is a ~ N(0,1) and the
    squared norm of the orthogonal complement is b ~ chi2(dim-1), so
    |cos| = |a| / sqrt(a^2 + b). This avoids materialising an (ndraw x dim) matrix
    and is used for the larger control-panel proteins.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(ndraw)
    b = rng.chisquare(dim - 1, ndraw)
    return np.abs(a) / np.sqrt(a ** 2 + b)


def dynamic_domains(open_c, closed_c, k=2, seed=20260720, n_init=10):
    """Displacement-based ('dynamic') domain decomposition, k-means on the per-residue
    displacement vector after whole-molecule superposition.

    This is the DynDom-style operational definition of a rigid domain: residues whose
    displacement vectors cluster together move as a unit. Deriving the decomposition
    FROM the transition makes any rigid-body-subspace null maximally conservative --
    it hands the null the best domain definition available -- which is the point when
    the null is being used as a sceptical control.

    Returns list of index arrays (one per domain).
    """
    rng = np.random.default_rng(seed)
    fitted = kabsch_apply(closed_c, open_c)
    D = fitted - open_c
    scale = np.linalg.norm(D, axis=1).mean() or 1.0
    X = D / scale
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        C = X[rng.choice(len(X), k, replace=False)]
        lab = np.zeros(len(X), dtype=int)
        for _ in range(100):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, lab):
                break
            lab = new
            for c in range(k):
                if (lab == c).any():
                    C[c] = X[lab == c].mean(0)
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best_inertia and len(np.unique(lab)) == k:
            best_inertia, best = inertia, lab.copy()
    if best is None:
        return [np.arange(len(X))]
    return [np.where(best == c)[0] for c in range(k)]


# --------------------------------------------------- ensemble construction ---
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def rcsb_entities_for_uniprot(acc, max_resolution=None, require_rfree=False):
    """Polymer entity ids ('1ABC_1') mapped to a UniProt accession, optionally
    restricted by resolution. Returns a sorted list."""
    nodes = [{"type": "terminal", "service": "text", "parameters": {
        "attribute": "rcsb_polymer_entity_container_identifiers."
                     "reference_sequence_identifiers.database_accession",
        "operator": "exact_match", "value": acc}}]
    if max_resolution is not None:
        nodes.append({"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "less_or_equal", "value": float(max_resolution)}})
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes}
         if len(nodes) > 1 else nodes[0],
         "return_type": "polymer_entity",
         "request_options": {"return_all_hits": True,
                             "results_content_type": ["experimental"]}}
    url = SEARCH_URL + "?json=" + urllib.parse.quote(json.dumps(q))
    with urllib.request.urlopen(url, timeout=120) as fh:
        d = json.load(fh)
    return sorted(x["identifier"] for x in d.get("result_set", []))


def rcsb_entity_meta(entity_id, cache=None):
    """Polymer-entity record ('1ABC_2') from the RCSB data API, cached to disk."""
    pdb, ent = validate_polymer_entity_id(entity_id)
    entity_id = f"{pdb}_{ent}"
    cache = cache or META_CACHE
    p = os.path.join(cache, f"entity_{entity_id.upper()}.json")
    if os.path.exists(p):
        return json.load(open(p))
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{ent}"
    with urllib.request.urlopen(url, timeout=60) as fh:
        d = json.load(fh)
    if CACHE_WRITES_ENABLED:
        os.makedirs(cache, exist_ok=True)
        with open(p, "w", encoding="utf-8") as _fh:
            json.dump(d, _fh)
    return d


def uniprot_chain_map(entity_ids, max_workers=8):
    """{pdb: [auth chain ids]} for the entities that carry the target UniProt accession.

    Chain identity must come from the deposition's own entity->chain mapping, not from
    a "chain with the most residues" heuristic: in ternary-complex depositions (VHL-
    EloB/EloC-target, CRBN-DDB1-degron) a heuristic silently selects a partner chain
    whose author numbering happens to span the analysis window, which produces
    nonsensical 15-20 A outliers in the ensemble.
    """
    import concurrent.futures as cf

    def one(eid):
        try:
            d = rcsb_entity_meta(eid)
            ids = d.get("rcsb_polymer_entity_container_identifiers", {})
            return eid.split("_")[0], list(ids.get("auth_asym_ids") or [])
        except Exception:
            return eid.split("_")[0], []

    out = {}
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for pdb, chains in ex.map(one, entity_ids):
            if chains:
                out.setdefault(pdb, [])
                out[pdb] = sorted(set(out[pdb]) | set(chains))
    return out


def build_ca_ensemble(pdb_ids, coverage=0.95, min_window=40, quality=None,
                      max_workers=8, verbose=False, chain_map=None,
                      outlier_rmsd=None):
    """Superposed Ca ensemble from a list of PDB ids, one chain per entry.

    Window rule (matches the CRBN pipeline in spirit): take the author residue
    numbers present in at least `coverage` of the candidate chains, require every
    retained chain to resolve the full window, then iteratively superpose onto the
    running mean.

    chain_map: {pdb: [auth chain ids]} restricting which chains may be selected --
    supply this from uniprot_chain_map() so the target protein's chain is chosen by
    deposition metadata rather than by a residue-count heuristic.

    outlier_rmsd: if set, structures further than this Ca RMSD from the ensemble mean
    are removed (one pass, after the first superposition) and reported. Used only to
    quarantine chain-assignment failures, never to prune genuine conformational spread;
    the removed labels are returned so the exclusion can be disclosed.

    quality: optional dict pdb -> metadata used only for reporting (not filtering;
    filter the id list before calling).

    Returns dict with confs (n, w, 3), labels, resnums, chains, and per-entry stats.
    """
    import concurrent.futures as cf

    def one(pdb):
        try:
            cif = fetch_cif(pdb)
            ca, stats = parse_atoms(cif, "CA")
            return pdb, ca, stats
        except Exception as exc:                                  # network / parse
            return pdb, None, {"error": f"{type(exc).__name__}: {exc}"}

    got = {}
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for pdb, ca, stats in ex.map(one, pdb_ids):
            if ca:
                got[pdb] = (ca, stats)
    # best chain per entry = most residues (ties -> lowest chain id)
    best = {}
    for pdb, (ca, _) in got.items():
        allowed = [c for c in sorted(ca) if chain_map is None
                   or c in chain_map.get(pdb, [])]
        if not allowed:
            continue
        ch = max(allowed, key=lambda c: len(ca[c]))
        best[pdb] = (ch, ca[ch])
    # window: residues present in >= coverage of chains
    from collections import Counter
    cnt = Counter()
    for _, rec in best.values():
        cnt.update(rec.keys())
    need = coverage * len(best)
    win = sorted(r for r, c in cnt.items() if c >= need)
    if len(win) < min_window:
        raise ValueError(f"window too small: {len(win)} residues from {len(best)} chains")
    labels, chains, coords, bfac = [], [], [], []
    for pdb, (ch, rec) in sorted(best.items()):
        if all(r in rec for r in win):
            labels.append(pdb)
            chains.append(ch)
            coords.append([rec[r]["xyz"] for r in win])
            bfac.append([rec[r]["b"] for r in win])
    X = np.array(coords, float)
    if len(X) < 3:
        raise ValueError(f"too few complete chains: {len(X)}")
    # iterative superposition onto the running mean
    ref = X[0]
    for _ in range(10):
        X = np.array([kabsch_apply(x, ref) for x in X])
        new = X.mean(0)
        if np.linalg.norm(new - ref) < 1e-6:
            ref = new
            break
        ref = new
    removed = []
    if outlier_rmsd is not None and len(X) > 5:
        mean = X.mean(0)
        dev = np.array([np.sqrt(((x - mean) ** 2).sum(1).mean()) for x in X])
        keep = dev <= outlier_rmsd
        if not keep.all():
            removed = [(labels[i], float(dev[i])) for i in np.where(~keep)[0]]
            X, labels = X[keep], [l for l, k in zip(labels, keep) if k]
            chains = [c for c, k in zip(chains, keep) if k]
            bfac = np.asarray(bfac, float)[keep]
            ref = X[0]
            for _ in range(10):
                X = np.array([kabsch_apply(x, ref) for x in X])
                new = X.mean(0)
                if np.linalg.norm(new - ref) < 1e-6:
                    ref = new
                    break
                ref = new
    return {"confs": X, "labels": labels, "chains": chains, "removed_outliers": removed,
            "resnums": np.array(win), "bfactors": np.array(bfac, float),
            "n_candidates": len(pdb_ids), "n_used": len(labels),
            "parse_stats": {p: st for p, (_, st) in got.items()}}


def ensemble_pca(confs):
    X = confs.reshape(len(confs), -1)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    return Vt.T, var, U * S


def pairwise_rmsd_stats(confs):
    n = len(confs)
    vals = []
    for a in range(n):
        for b in range(a + 1, n):
            vals.append(np.sqrt(((confs[a] - confs[b]) ** 2).sum(1).mean()))
    v = np.array(vals)
    return {"median": float(np.median(v)), "max": float(v.max()),
            "mean": float(v.mean()), "n_pairs": int(v.size)}


FEATURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "crbn_features.json")


def functional_residues(path=None):
    """UniProt-annotated residue groups from data/crbn_features.json.

    The drug-binding and zinc-coordinating lists drive Fig 4, the mobility percentiles and
    the lever-arm control, and they were duplicated as literals in five scripts. Reading
    them from the deposited annotation table gives one source of truth and makes the
    repository data file load-bearing rather than decorative: if it changes, every
    analysis that depends on it changes with it.

    Returns {'drug': [...], 'zinc': [...], 'mutagenesis': [...], 'variant': [...]}.
    """
    with open(path or FEATURES, encoding="utf-8") as fh:
        d = json.load(fh)
    return {
        "drug": list(d["drug_binding_thalidomide"]),
        "zinc": list(d["zinc_coordinating"]),
        "mutagenesis": list(d.get("mutagenesis_thalidomide_binding_abolishing", [])),
        "variant": list(d.get("natural_variant", [])),
    }


def spearman(x, y):
    """Spearman rho and a two-sided p from the t approximation (no scipy dependency)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        r[order] = np.arange(1, len(a) + 1)
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        for k in np.where(cnt > 1)[0]:
            m = inv == k
            r[m] = r[m].mean()
        return r

    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0, n
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    # exact two-sided p from the Student-t approximation to the null of rho.
    # scipy is used only for the CDF; the fallback keeps the module numpy-only.
    try:
        from scipy.stats import t as _t
        p = float(2 * _t.sf(abs(t), n - 2))
    except ImportError:
        from math import erf, sqrt
        p = float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))
    return rho, p, n
