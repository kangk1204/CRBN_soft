#!/usr/bin/env python3
"""CRBN-DDB1 interface-strengthening analysis for the CSBJ revision package.

This script evaluates a matched 269-CRBN-residue elastic-network model across the
five open CRBN-DDB1 references used for the manuscript response:

  8CVP B/A, 8D7X B/A, 8D7Y B/A, 6H0F B/A, and 7U8F A/B.

For each reference, cutoff, and interface strength alpha, it reports five
definitions that must not be averaged together:

  isolated       CRBN-only ANM on the 269-node analysis window.
  zero_interface CRBN and DDB1 in one coordinate set with no CRBN-DDB1 springs.
  joint          full CRBN+DDB1 modes with interface springs scaled by alpha.
  schur_static   CRBN block after statically relaxing DDB1, H_eff=A-B D^+ B^T.
  fixed_partner  CRBN block A with DDB1 held fixed.

Schur eigenvalues are quasi-static stiffness values in unit-spring ANM units, not
physical dynamic frequencies. The joint model reports normalized CRBN directional
overlap separately from CRBN amplitude because joint eigenvectors are normalized
over CRBN plus DDB1. Offline mode reads the frozen `data/_cif_cache/*.cif.gz`
files and does not write cache data.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import urllib.request

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, norm as sparse_norm, splu

try:
    from pdb_id import validate_pdb_id
except ModuleNotFoundError:
    from scripts.pdb_id import validate_pdb_id


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CIF_CACHE = DATA / "_cif_cache"
DEFAULT_OUT = ROOT / "results" / "strengthening" / "ddb1"
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

CASES = (
    ("8CVP", "B", "A", "apo cryo-EM, manuscript open reference"),
    ("8D7X", "B", "A", "apo cryo-EM"),
    ("8D7Y", "B", "A", "apo cryo-EM, DDB1 twisted conformation"),
    ("6H0F", "B", "A", "pomalidomide + IKZF1 ternary, open"),
    ("7U8F", "A", "B", "DKY709 + IKZF2 ternary, open"),
)
PRIMARY_CUTOFF = 15.0
SENSITIVITY_CUTOFFS = (13.0, 18.0)
DEFAULT_ALPHAS = (0.0, 0.5, 1.0, 2.0)
NEAR_DEGENERATE_RATIO = 1.20
ZERO_TOL = 1e-9
SPARSE_EIGEN_SHIFT = -1e-4
SPARSE_EIGEN_TOL = 1e-10
SPARSE_RESIDUAL_TOL = 1e-7


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--offline", action="store_true", help="use only local data/_cif_cache files")
    parser.add_argument("--primary-cutoff", type=float, default=PRIMARY_CUTOFF)
    parser.add_argument("--sensitivity-cutoffs", type=float, nargs="*", default=list(SENSITIVITY_CUTOFFS))
    parser.add_argument("--alphas", type=float, nargs="*", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--primary-modes", type=int, default=20)
    parser.add_argument("--sensitivity-modes", type=int, default=60)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="write outputs, then fail the run if numerical consistency checks do not pass",
    )
    return parser.parse_args(argv)


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def read_source_bytes(path: Path) -> bytes:
    return path.read_bytes()


def fetch_cif_blob(pdb_id: str, offline: bool) -> tuple[bytes, str]:
    pdb_id = validate_pdb_id(pdb_id)
    local = CIF_CACHE / f"{pdb_id}.cif.gz"
    if local.exists():
        return local.read_bytes(), str(local.relative_to(ROOT))
    if offline:
        raise FileNotFoundError(f"{pdb_id}: missing {local.relative_to(ROOT)} in offline mode")
    with urllib.request.urlopen(RCSB_URL.format(pdb_id=pdb_id), timeout=180) as handle:
        return handle.read(), RCSB_URL.format(pdb_id=pdb_id)


def cif_text_from_blob(blob: bytes) -> str:
    try:
        return gzip.decompress(blob).decode("utf-8", errors="replace")
    except gzip.BadGzipFile:
        return blob.decode("utf-8", errors="replace")


def ca_coords_from_cif(text: str, chain: str) -> dict[int, list[float]]:
    lines = text.splitlines()
    out: dict[int, list[float]] = {}
    i = 0
    while i < len(lines):
        if lines[i].startswith("_atom_site."):
            cols = []
            j = i
            while j < len(lines) and lines[j].startswith("_atom_site."):
                cols.append(lines[j].strip().split(".")[1])
                j += 1
            ix = {col: pos for pos, col in enumerate(cols)}
            required = ("group_PDB", "label_atom_id", "auth_asym_id", "auth_seq_id",
                        "Cartn_x", "Cartn_y", "Cartn_z")
            missing = [key for key in required if key not in ix]
            if missing:
                raise ValueError(f"mmCIF atom_site loop missing columns: {missing}")
            while j < len(lines) and not lines[j].startswith("#"):
                fields = lines[j].split()
                if (
                    len(fields) >= len(cols)
                    and fields[ix["group_PDB"]] == "ATOM"
                    and fields[ix["label_atom_id"]] == "CA"
                    and fields[ix["auth_asym_id"]] == chain
                ):
                    try:
                        num = int(fields[ix["auth_seq_id"]])
                    except ValueError:
                        j += 1
                        continue
                    out.setdefault(
                        num,
                        [
                            float(fields[ix["Cartn_x"]]),
                            float(fields[ix["Cartn_y"]]),
                            float(fields[ix["Cartn_z"]]),
                        ],
                    )
                j += 1
            break
        i += 1
    return out


def load_reference() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    ens = np.load(DATA / "crbn_ensemble.ens.npz", allow_pickle=False)
    confs = np.asarray(ens["_confs"], dtype=float)
    labels = [str(label)[:4] for label in ens["_labels"]]
    with (DATA / "crbn_residue_window.csv").open(newline="") as handle:
        window = np.array([int(row["author_resnum"]) for row in csv.DictReader(handle)], dtype=int)
    diff = np.load(DATA / "pca_diffvec.npz", allow_pickle=False)
    if "diff_vec" in diff:
        diff_axis = np.asarray(diff["diff_vec"], dtype=float).reshape(-1)
        diff_axis /= np.linalg.norm(diff_axis)
    else:
        open_mask = diff["open_mask"].astype(bool)
        diff_axis = (confs[open_mask].mean(0) - confs[~open_mask].mean(0)).reshape(-1)
        diff_axis /= np.linalg.norm(diff_axis)
    open_mask = diff["open_mask"].astype(bool)
    mean_axis = (confs[open_mask].mean(0) - confs[~open_mask].mean(0)).reshape(-1)
    mean_axis /= np.linalg.norm(mean_axis)
    return confs, labels, window, diff_axis, mean_axis


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(0)
    target_center = target.mean(0)
    u, _, vt = np.linalg.svd((mobile - mobile_center).T @ (target - target_center))
    sign = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, sign]) @ vt, mobile_center, target_center


def rigid_basis(coords: np.ndarray) -> np.ndarray:
    cols = []
    center = coords.mean(0)
    for axis in range(3):
        trans = np.zeros_like(coords)
        trans[:, axis] = 1.0
        cols.append(trans.ravel())
    for axis in range(3):
        unit = np.zeros(3)
        unit[axis] = 1.0
        rot = np.cross(unit, coords - center)
        cols.append(rot.ravel())
    q, _ = np.linalg.qr(np.array(cols).T)
    return q[:, :6]


def remove_rigid(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return vector - basis @ (basis.T @ vector)


def normalized_overlap(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= ZERO_TOL or nb <= ZERO_TOL:
        return 0.0
    return float(abs(a @ b) / (na * nb))


def anm_hessian(coords: np.ndarray, cutoff: float) -> np.ndarray:
    n = len(coords)
    hessian = np.zeros((3 * n, 3 * n), dtype=float)
    dist = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    ii, jj = np.where((dist <= cutoff) & (dist > 1e-6))
    for i, j in zip(ii[ii < jj], jj[ii < jj]):
        delta = coords[j] - coords[i]
        spring = np.outer(delta, delta) / dist[i, j] ** 2
        add_pair_block(hessian, i, j, spring, 1.0)
    return hessian


def slow_modes(hessian: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    dim = hessian.shape[0]
    hi = min(dim - 1, n_modes + 24)
    while True:
        values, vectors = eigh(hessian, subset_by_index=(0, hi), check_finite=False)
        keep = values > ZERO_TOL
        if int(keep.sum()) >= n_modes or hi == dim - 1:
            return values[keep][:n_modes], vectors[:, keep][:, :n_modes]
        hi = min(dim - 1, max(2 * hi + 1, hi + n_modes + 24))


def deterministic_start_vector(dim: int) -> np.ndarray:
    """Return a fixed nonzero start vector for ARPACK iterations."""

    vector = np.sin(np.arange(dim, dtype=float) + 1.0)
    return vector / np.linalg.norm(vector)


def eigenpair_residuals(
    hessian: sparse.spmatrix | np.ndarray,
    values: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    residual = hessian @ vectors - vectors * values
    return np.linalg.norm(residual, axis=0) / np.maximum(1.0, np.abs(values))


def slow_modes_sparse(hessian: sparse.spmatrix, n_modes: int, rigid_modes: int) -> tuple[np.ndarray, np.ndarray]:
    dim = hessian.shape[0]
    request = min(dim - 2, n_modes + rigid_modes + 12)
    while True:
        values, vectors = eigsh(
            hessian,
            k=request,
            sigma=SPARSE_EIGEN_SHIFT,
            which="LM",
            tol=SPARSE_EIGEN_TOL,
            maxiter=max(6000, dim * 4),
            v0=deterministic_start_vector(dim),
        )
        order = np.argsort(values)
        values = values[order]
        vectors = vectors[:, order]
        keep = values > ZERO_TOL
        if int(keep.sum()) >= n_modes or request >= dim - 2:
            values = values[keep][:n_modes]
            vectors = vectors[:, keep][:, :n_modes]
            residuals = eigenpair_residuals(hessian, values, vectors)
            if float(residuals.max(initial=0.0)) > SPARSE_RESIDUAL_TOL:
                values, vectors = eigsh(
                    hessian,
                    k=request,
                    sigma=10.0 * SPARSE_EIGEN_SHIFT,
                    which="LM",
                    tol=SPARSE_EIGEN_TOL,
                    maxiter=max(8000, dim * 6),
                    v0=deterministic_start_vector(dim),
                )
                order = np.argsort(values)
                values = values[order]
                vectors = vectors[:, order]
                keep = values > ZERO_TOL
                values = values[keep][:n_modes]
                vectors = vectors[:, keep][:, :n_modes]
                residuals = eigenpair_residuals(hessian, values, vectors)
                if float(residuals.max(initial=0.0)) > SPARSE_RESIDUAL_TOL:
                    raise RuntimeError(
                        "sparse slow-mode solver did not converge to residual "
                        f"{SPARSE_RESIDUAL_TOL:g}; max residual {float(residuals.max()):.3g}"
                    )
            return values, vectors
        request = min(dim - 2, request + n_modes + 12)


def add_pair_block(block: np.ndarray, i: int, j: int, spring: np.ndarray, weight: float) -> None:
    scaled = weight * spring
    si = slice(3 * i, 3 * i + 3)
    sj = slice(3 * j, 3 * j + 3)
    block[si, si] += scaled
    block[sj, sj] += scaled
    block[si, sj] -= scaled
    block[sj, si] -= scaled


def decompose_hessian(crbn_xyz: np.ndarray, ddb1_xyz: np.ndarray, cutoff: float) -> dict[str, np.ndarray | int]:
    n_c = len(crbn_xyz)
    n_d = len(ddb1_xyz)
    joint = np.vstack([crbn_xyz, ddb1_xyz])
    dist = np.linalg.norm(joint[:, None] - joint[None], axis=2)
    h_crbn = np.zeros((3 * n_c, 3 * n_c), dtype=float)
    h_ddb1 = np.zeros((3 * n_d, 3 * n_d), dtype=float)
    h_interface = np.zeros((3 * (n_c + n_d), 3 * (n_c + n_d)), dtype=float)
    interface_pairs = 0
    contact_count = 0

    ii, jj = np.where((dist <= cutoff) & (dist > 1e-6))
    for i, j in zip(ii[ii < jj], jj[ii < jj]):
        delta = joint[j] - joint[i]
        spring = np.outer(delta, delta) / dist[i, j] ** 2
        if i < n_c and j < n_c:
            add_pair_block(h_crbn, i, j, spring, 1.0)
        elif i >= n_c and j >= n_c:
            add_pair_block(h_ddb1, i - n_c, j - n_c, spring, 1.0)
        else:
            add_pair_block(h_interface, i, j, spring, 1.0)
            interface_pairs += 1
            contact_count += 2
    return {
        "h_crbn": h_crbn,
        "h_ddb1": h_ddb1,
        "h_interface": h_interface,
        "n_interface_pairs": interface_pairs,
        "n_directed_interface_contacts": contact_count,
    }


def compose_joint(parts: dict[str, np.ndarray | int], alpha: float) -> np.ndarray:
    h_crbn = np.asarray(parts["h_crbn"])
    h_ddb1 = np.asarray(parts["h_ddb1"])
    joint = np.zeros(
        (h_crbn.shape[0] + h_ddb1.shape[0], h_crbn.shape[1] + h_ddb1.shape[1]),
        dtype=float,
    )
    joint[: h_crbn.shape[0], : h_crbn.shape[1]] = h_crbn
    joint[h_crbn.shape[0] :, h_crbn.shape[1] :] = h_ddb1
    joint += alpha * np.asarray(parts["h_interface"])
    return joint


def compose_joint_sparse(parts: dict[str, np.ndarray | int], alpha: float) -> sparse.csr_matrix:
    h_crbn = sparse.csr_matrix(np.asarray(parts["h_crbn"]))
    h_ddb1 = sparse.csr_matrix(np.asarray(parts["h_ddb1"]))
    joint = sparse.block_diag((h_crbn, h_ddb1), format="csr")
    if alpha != 0.0:
        joint = joint + alpha * sparse.csr_matrix(np.asarray(parts["h_interface"]))
    return joint.tocsr()


def crbn_metrics(
    model: str,
    values: np.ndarray,
    vectors: np.ndarray,
    crbn_dim: int,
    axis: np.ndarray,
    rigid: np.ndarray,
    alpha: float,
    cutoff: float,
    pdb_id: str,
    n_modes_requested: int,
    primary_limit: int = 20,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    overlaps = []
    internal_overlaps = []
    amplitudes = []
    internal_amplitudes = []
    for idx in range(vectors.shape[1]):
        crbn_vec = np.asarray(vectors[:crbn_dim, idx], dtype=float)
        amp = float(np.linalg.norm(crbn_vec))
        internal_vec = remove_rigid(crbn_vec, rigid)
        internal_axis = remove_rigid(axis, rigid)
        overlap = normalized_overlap(crbn_vec, axis)
        internal_overlap = normalized_overlap(internal_vec, internal_axis)
        internal_amp = float(np.linalg.norm(internal_vec))
        overlaps.append(overlap)
        internal_overlaps.append(internal_overlap)
        amplitudes.append(amp)
        internal_amplitudes.append(internal_amp)
        rows.append(
            {
                "pdb": pdb_id,
                "cutoff_A": cutoff,
                "interface_alpha": alpha,
                "model": model,
                "mode": idx + 1,
                "eigenvalue": float(values[idx]),
                "crbn_directional_overlap": overlap,
                "crbn_amplitude": amp,
                "internal_rigid_removed_overlap": internal_overlap,
                "internal_rigid_removed_amplitude": internal_amp,
            }
        )

    primary_count = min(primary_limit, len(overlaps))
    sensitivity_count = len(overlaps)
    best_idx = int(np.argmax(overlaps[:primary_count]))
    internal_best_idx = int(np.argmax(internal_overlaps[:primary_count]))
    best60_idx = int(np.argmax(overlaps[:sensitivity_count]))
    internal60_idx = int(np.argmax(internal_overlaps[:sensitivity_count]))
    best_value = float(values[best_idx])
    near = [
        i
        for i, value in enumerate(values)
        if max(float(value), best_value) / max(min(float(value), best_value), ZERO_TOL)
        <= NEAR_DEGENERATE_RATIO
    ]
    q_vectors = []
    internal_axis = remove_rigid(axis, rigid)
    for idx in near:
        vec = remove_rigid(np.asarray(vectors[:crbn_dim, idx], dtype=float), rigid)
        if np.linalg.norm(vec) > ZERO_TOL:
            q_vectors.append(vec)
    if q_vectors:
        basis_input = np.column_stack(q_vectors)
        u, singular_values, _ = np.linalg.svd(basis_input, full_matrices=False)
        threshold = max(basis_input.shape) * float(singular_values[0]) * 1e-10
        retained = singular_values > threshold
        q = u[:, retained]
        subspace = float(np.linalg.norm(q.T @ (internal_axis / np.linalg.norm(internal_axis))))
        subspace = max(0.0, min(1.0, subspace))
        subspace_rank = int(retained.sum())
    else:
        subspace = 0.0
        subspace_rank = 0
    summary = {
        "pdb": pdb_id,
        "cutoff_A": cutoff,
        "interface_alpha": alpha,
        "model": model,
        "n_modes_requested": n_modes_requested,
        "n_modes_returned": int(vectors.shape[1]),
        "primary_best_limit": int(primary_count),
        "best_mode": best_idx + 1,
        "best_crbn_directional_overlap": float(overlaps[best_idx]),
        "best_crbn_amplitude": float(amplitudes[best_idx]),
        "best_eigenvalue": best_value,
        "mode1_crbn_directional_overlap": float(overlaps[0]),
        "mode1_crbn_amplitude": float(amplitudes[0]),
        "mode1_internal_rigid_removed_overlap": float(internal_overlaps[0]),
        "mode1_internal_rigid_removed_amplitude": float(internal_amplitudes[0]),
        "internal_best_mode": internal_best_idx + 1,
        "internal_best_overlap": float(internal_overlaps[internal_best_idx]),
        "internal_best_amplitude": float(internal_amplitudes[internal_best_idx]),
        "sensitivity_best_limit": int(sensitivity_count),
        "best60_mode": best60_idx + 1,
        "best60_crbn_directional_overlap": float(overlaps[best60_idx]),
        "best60_crbn_amplitude": float(amplitudes[best60_idx]),
        "best60_eigenvalue": float(values[best60_idx]),
        "internal_best60_mode": internal60_idx + 1,
        "internal_best60_overlap": float(internal_overlaps[internal60_idx]),
        "internal_best60_amplitude": float(internal_amplitudes[internal60_idx]),
        "higher_mode_21_60_changes_sensitivity_best": bool(best60_idx >= primary_count),
        "near_degenerate_ratio": NEAR_DEGENERATE_RATIO,
        "near_degenerate_modes": [i + 1 for i in near],
        "near_degenerate_internal_subspace_overlap": subspace,
        "near_degenerate_internal_subspace_rank": subspace_rank,
    }
    return rows, summary


def block_diagonal_modes(
    crbn_values: np.ndarray,
    crbn_vectors: np.ndarray,
    ddb1_values: np.ndarray,
    ddb1_vectors: np.ndarray,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    crbn_dim = crbn_vectors.shape[0]
    ddb1_dim = ddb1_vectors.shape[0]
    entries = [("crbn", i, float(value)) for i, value in enumerate(crbn_values)]
    entries.extend(("ddb1", i, float(value)) for i, value in enumerate(ddb1_values))
    entries.sort(key=lambda item: item[2])
    values = []
    vectors = np.zeros((crbn_dim + ddb1_dim, min(n_modes, len(entries))), dtype=float)
    for out_idx, (block, in_idx, value) in enumerate(entries[:n_modes]):
        values.append(value)
        if block == "crbn":
            vectors[:crbn_dim, out_idx] = crbn_vectors[:, in_idx]
        else:
            vectors[crbn_dim:, out_idx] = ddb1_vectors[:, in_idx]
    return np.array(values, dtype=float), vectors


def static_response(hessian: np.ndarray, force: np.ndarray, coords: np.ndarray) -> np.ndarray:
    basis = rigid_basis(coords)
    clean_force = remove_rigid(force, basis)
    return np.linalg.pinv(hessian, rcond=1e-10) @ clean_force


def constrained_solver(matrix: sparse.spmatrix, rigid: np.ndarray):
    dim = matrix.shape[0]
    constraints = sparse.csc_matrix(rigid)
    zeros = sparse.csc_matrix((rigid.shape[1], rigid.shape[1]))
    augmented = sparse.bmat(
        [[matrix.tocsc(), constraints], [constraints.T, zeros]],
        format="csc",
    )
    factor = splu(augmented)

    def solve(rhs: np.ndarray) -> np.ndarray:
        rhs_2d = np.asarray(rhs, dtype=float)
        squeeze = rhs_2d.ndim == 1
        if squeeze:
            rhs_2d = rhs_2d[:, None]
        padded = np.vstack([rhs_2d, np.zeros((rigid.shape[1], rhs_2d.shape[1]))])
        solved = factor.solve(padded)[:dim]
        return solved[:, 0] if squeeze else solved

    return solve


def schur_from_sparse(
    joint: sparse.spmatrix,
    crbn_dim: int,
    ddb1_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, sparse.spmatrix, sparse.spmatrix, object]:
    a_sparse = joint[:crbn_dim, :crbn_dim].tocsc()
    b_sparse = joint[:crbn_dim, crbn_dim:].tocsc()
    d_sparse = joint[crbn_dim:, crbn_dim:].tocsc()
    _ = ddb1_xyz
    factor = splu(d_sparse)

    def solve_d(rhs: np.ndarray) -> np.ndarray:
        rhs_2d = np.asarray(rhs, dtype=float)
        squeeze = rhs_2d.ndim == 1
        if squeeze:
            rhs_2d = rhs_2d[:, None]
        solved = factor.solve(rhs_2d)
        return solved[:, 0] if squeeze else solved

    x = solve_d(b_sparse.T.toarray())
    a = a_sparse.toarray()
    schur = a - b_sparse @ x
    return np.asarray(schur), a, b_sparse, d_sparse, solve_d


def schur_from_joint(joint: np.ndarray, crbn_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = joint[:crbn_dim, :crbn_dim]
    b = joint[:crbn_dim, crbn_dim:]
    d = joint[crbn_dim:, crbn_dim:]
    d_pinv = np.linalg.pinv(d, rcond=1e-10)
    return a - b @ d_pinv @ b.T, a, b, d


def balanced_force(axis: np.ndarray, crbn_xyz: np.ndarray) -> np.ndarray:
    return remove_rigid(axis, rigid_basis(crbn_xyz))


def schur_full_static_response_check(
    joint: sparse.spmatrix,
    h_schur: np.ndarray,
    crbn_dim: int,
    crbn_xyz: np.ndarray,
    ddb1_xyz: np.ndarray,
    force: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compare Schur response to an independently solved full-joint response."""

    schur_response = static_response(h_schur, force, crbn_xyz)
    full_force = np.zeros(joint.shape[0], dtype=float)
    full_force[:crbn_dim] = force
    full_coords = np.vstack([crbn_xyz, ddb1_xyz])
    full_response = constrained_solver(joint, rigid_basis(full_coords))(full_force)
    rigid = rigid_basis(crbn_xyz)
    full_internal = remove_rigid(full_response[:crbn_dim], rigid)
    schur_internal = remove_rigid(schur_response, rigid)
    denom = max(np.linalg.norm(schur_internal), ZERO_TOL)
    rel_error = float(np.linalg.norm(full_internal - schur_internal) / denom)
    return rel_error, schur_response, full_response[:crbn_dim]


def analyse_case(
    pdb_id: str,
    crbn_chain: str,
    ddb1_chain: str,
    note: str,
    cutoffs: list[float],
    primary_cutoff: float,
    alphas: list[float],
    primary_modes: int,
    sensitivity_modes: int,
    confs: np.ndarray,
    labels: list[str],
    window: np.ndarray,
    axis: np.ndarray,
    offline: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, str]]:
    blob, source = fetch_cif_blob(pdb_id, offline)
    text = cif_text_from_blob(blob)
    crbn = ca_coords_from_cif(text, crbn_chain)
    ddb1 = ca_coords_from_cif(text, ddb1_chain)
    missing = [int(resnum) for resnum in window if int(resnum) not in crbn]
    if missing:
        raise ValueError(f"{pdb_id} chain {crbn_chain} misses window residues {missing[:8]}")
    if not ddb1:
        raise ValueError(f"{pdb_id} chain {ddb1_chain} has no C-alpha coordinates")

    raw = np.array([crbn[int(resnum)] for resnum in window], dtype=float)
    ref = confs[labels.index(pdb_id)]
    rotation, mobile_center, target_center = kabsch(raw, ref)

    def rotate(coords: np.ndarray) -> np.ndarray:
        return (coords - mobile_center) @ rotation + target_center

    crbn_xyz = rotate(raw)
    ddb1_nums = sorted(ddb1)
    ddb1_xyz = rotate(np.array([ddb1[num] for num in ddb1_nums], dtype=float))
    frame_rmsd = float(np.sqrt(((crbn_xyz - ref) ** 2).sum() / len(ref)))
    if frame_rmsd >= 1e-6:
        raise ValueError(f"{pdb_id}: frame match failed, RMSD {frame_rmsd:.6g} A")

    modes_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    checks: dict[str, object] = {
        "pdb": pdb_id,
        "crbn_chain": crbn_chain,
        "ddb1_chain": ddb1_chain,
        "note": note,
        "n_crbn_nodes": int(len(crbn_xyz)),
        "n_ddb1_nodes": int(len(ddb1_xyz)),
        "frame_match_rmsd_A": frame_rmsd,
        "cutoffs": {},
    }
    source_hashes = {f"rcsb_mmcif_{pdb_id}": sha256_bytes(blob), f"rcsb_mmcif_{pdb_id}_source": source}
    crbn_dim = 3 * len(crbn_xyz)
    rigid = rigid_basis(crbn_xyz)
    axis_internal = remove_rigid(axis, rigid)
    if np.linalg.norm(axis_internal) <= ZERO_TOL:
        raise ValueError(f"{pdb_id}: open-closed axis collapsed after rigid removal")

    for cutoff in cutoffs:
        started = time.perf_counter()
        n_modes = sensitivity_modes
        parts = decompose_hessian(crbn_xyz, ddb1_xyz, cutoff)
        cutoff_checks = {
            "n_interface_pairs": int(parts["n_interface_pairs"]),
            "n_modes": n_modes,
            "primary_best_limit": primary_modes,
            "models": {},
            "seconds": None,
        }
        h_iso = np.asarray(parts["h_crbn"])
        iso_values, iso_vectors = slow_modes(h_iso, n_modes)
        iso_rows, iso_summary = crbn_metrics(
            "isolated", iso_values, iso_vectors, crbn_dim, axis, rigid, 0.0, cutoff, pdb_id, n_modes, primary_modes
        )
        modes_rows.extend(iso_rows)
        summary_rows.append(iso_summary)

        ddb1_values, ddb1_vectors = slow_modes_sparse(
            sparse.csr_matrix(np.asarray(parts["h_ddb1"])), n_modes, rigid_modes=6
        )
        zero_values, zero_vectors = block_diagonal_modes(
            iso_values, iso_vectors, ddb1_values, ddb1_vectors, n_modes
        )
        zero_rows, zero_summary = crbn_metrics(
            "zero_interface", zero_values, zero_vectors, crbn_dim, axis, rigid, 0.0, cutoff, pdb_id, n_modes, primary_modes
        )
        modes_rows.extend(zero_rows)
        summary_rows.append(zero_summary)

        for alpha in alphas:
            if alpha == 0.0:
                joint_rows, joint_summary = crbn_metrics(
                    "joint", zero_values, zero_vectors, crbn_dim, axis, rigid, alpha, cutoff, pdb_id, n_modes, primary_modes
                )
                modes_rows.extend(joint_rows)
                summary_rows.append(joint_summary)
                schur_rows, schur_summary = crbn_metrics(
                    "schur_static", iso_values, iso_vectors, crbn_dim, axis, rigid, alpha, cutoff, pdb_id, n_modes, primary_modes
                )
                modes_rows.extend(schur_rows)
                schur_summary["unit_note"] = "quasi_static_unit_spring_stiffness_not_dynamic_frequency"
                summary_rows.append(schur_summary)
                fixed_rows, fixed_summary = crbn_metrics(
                    "fixed_partner", iso_values, iso_vectors, crbn_dim, axis, rigid, alpha, cutoff, pdb_id, n_modes, primary_modes
                )
                modes_rows.extend(fixed_rows)
                summary_rows.append(fixed_summary)
                cutoff_checks["models"][f"alpha_{alpha:g}"] = {
                    "schur_full_static_response_relative_error": 0.0,
                    "full_block_static_equilibrium_relative_error": 0.0,
                    "schur_rank": int(schur_summary["best_mode"]),
                    "fixed_partner_rank": int(fixed_summary["best_mode"]),
                    "joint_rank": int(joint_summary["best_mode"]),
                    "joint_best_crbn_amplitude": float(joint_summary["best_crbn_amplitude"]),
                    "dblock_rank_estimate": int(3 * len(ddb1_xyz) - 6),
                    "dblock_dimension": int(3 * len(ddb1_xyz)),
                    "b_norm": 0.0,
                }
                continue

            joint_sparse = compose_joint_sparse(parts, alpha)
            joint_values, joint_vectors = slow_modes_sparse(joint_sparse, n_modes, rigid_modes=6)
            joint_rows, joint_summary = crbn_metrics(
                "joint", joint_values, joint_vectors, crbn_dim, axis, rigid, alpha, cutoff,
                pdb_id, n_modes, primary_modes
            )
            modes_rows.extend(joint_rows)
            summary_rows.append(joint_summary)

            h_schur, h_fixed, b_sparse, d_sparse, solve_d = schur_from_sparse(
                joint_sparse, crbn_dim, ddb1_xyz
            )
            schur_values, schur_vectors = slow_modes(h_schur, n_modes)
            schur_rows, schur_summary = crbn_metrics(
                "schur_static", schur_values, schur_vectors, crbn_dim, axis, rigid, alpha,
                cutoff, pdb_id, n_modes, primary_modes
            )
            modes_rows.extend(schur_rows)
            schur_summary["unit_note"] = "quasi_static_unit_spring_stiffness_not_dynamic_frequency"
            summary_rows.append(schur_summary)

            fixed_values, fixed_vectors = slow_modes(h_fixed, n_modes)
            fixed_rows, fixed_summary = crbn_metrics(
                "fixed_partner", fixed_values, fixed_vectors, crbn_dim, axis, rigid, alpha,
                cutoff, pdb_id, n_modes, primary_modes
            )
            modes_rows.extend(fixed_rows)
            summary_rows.append(fixed_summary)

            force = balanced_force(axis, crbn_xyz)
            response_rel_error, schur_response, _ = schur_full_static_response_check(
                joint_sparse, h_schur, crbn_dim, crbn_xyz, ddb1_xyz, force
            )
            partner_response = -solve_d(b_sparse.T @ schur_response)
            crbn_residual = h_fixed @ schur_response + b_sparse @ partner_response - force
            partner_residual = b_sparse.T @ schur_response + d_sparse @ partner_response
            equilibrium_error = float(
                (np.linalg.norm(remove_rigid(crbn_residual, rigid)) + np.linalg.norm(partner_residual))
                / max(np.linalg.norm(force), ZERO_TOL)
            )
            cutoff_checks["models"][f"alpha_{alpha:g}"] = {
                "schur_full_static_response_relative_error": response_rel_error,
                "full_block_static_equilibrium_relative_error": equilibrium_error,
                "schur_rank": int(schur_summary["best_mode"]),
                "fixed_partner_rank": int(fixed_summary["best_mode"]),
                "joint_rank": int(joint_summary["best_mode"]),
                "joint_best_crbn_amplitude": float(joint_summary["best_crbn_amplitude"]),
                "dblock_rank_estimate": int(d_sparse.shape[0]),
                "dblock_dimension": int(d_sparse.shape[0]),
                "b_norm": float(sparse_norm(b_sparse)),
            }
        cutoff_checks["seconds"] = round(time.perf_counter() - started, 3)
        checks["cutoffs"][f"{cutoff:g}"] = cutoff_checks
    return modes_rows, summary_rows, checks, source_hashes


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def observed_8cvp_reconciliation(
    summary_rows: list[dict[str, object]],
    confs: np.ndarray,
    labels: list[str],
    window: np.ndarray,
    axis: np.ndarray,
    offline: bool,
) -> dict[str, object]:
    rows = [
        row for row in summary_rows
        if row["pdb"] == "8CVP" and row["cutoff_A"] == PRIMARY_CUTOFF and row["interface_alpha"] == 0.0
    ]
    by_model = {str(row["model"]): row for row in rows}
    blob, source = fetch_cif_blob("8CVP", offline)
    crbn = ca_coords_from_cif(cif_text_from_blob(blob), "B")
    raw_common = np.array([crbn[int(resnum)] for resnum in window], dtype=float)
    ref = confs[labels.index("8CVP")]
    rotation, mobile_center, target_center = kabsch(raw_common, ref)
    rotate = lambda coords: (coords - mobile_center) @ rotation + target_center
    all_resnums = np.array(sorted(crbn), dtype=int)
    all_xyz = rotate(np.array([crbn[int(resnum)] for resnum in all_resnums], dtype=float))
    values, vectors = slow_modes(anm_hessian(all_xyz, PRIMARY_CUTOFF), 60)
    keep = np.where(np.isin(all_resnums, window))[0]
    if not np.array_equal(all_resnums[keep], window):
        raise ValueError("8CVP all-resolved CRBN coordinates do not contain the common window in order")
    projected = vectors.reshape(len(all_resnums), 3, 60)[keep].reshape(-1, 60)
    projected_amplitude = np.linalg.norm(projected, axis=0)
    projected_normalized_overlap = np.array([
        normalized_overlap(projected[:, idx], axis) for idx in range(projected.shape[1])
    ])
    projected_raw_dot = np.abs(projected.T @ axis)
    rigid = rigid_basis(ref)
    internal_axis = remove_rigid(axis, rigid)
    projected_internal_overlap = np.array([
        normalized_overlap(remove_rigid(projected[:, idx], rigid), internal_axis)
        for idx in range(projected.shape[1])
    ])
    return {
        "scope": "8CVP primary cutoff, alpha 0 where applicable",
        "observed_crbn349_note": (
            "0.744 is the isolated 269-node common-window ANM mode-1 overlap. 0.613 is "
            "reproduced here by building the ANM on all 349 resolved 8CVP CRBN C-alpha nodes, "
            "cropping mode 1 back to the 269 common-window residues, and normalizing that "
            "cropped direction before scoring it. The 0.598 value is not reproduced by the "
            "current frozen mmCIF/code path as full-joint, CRBN-normalized, cropped-349, "
            "cropped-349 raw-dot, or rigid-removed scoring; treat it as a stale legacy variant "
            "unless an older artifact defines it."
        ),
        "matched_269_isolated_mode1": by_model.get("isolated", {}).get("mode1_crbn_directional_overlap"),
        "matched_269_internal_rigid_removed_mode1": by_model.get("isolated", {}).get(
            "mode1_internal_rigid_removed_overlap"
        ),
        "zero_interface_joint_mode1_crbn_slice": by_model.get("zero_interface", {}).get("mode1_crbn_directional_overlap"),
        "zero_interface_joint_mode1_crbn_amplitude": by_model.get("zero_interface", {}).get("mode1_crbn_amplitude"),
        "all_resolved_crbn_source": source,
        "all_resolved_crbn_sha256": sha256_bytes(blob),
        "all_resolved_crbn_nodes": int(len(all_resnums)),
        "all_resolved_common_window_nodes": int(len(window)),
        "all_resolved_mode1_projected_normalized_overlap": float(projected_normalized_overlap[0]),
        "all_resolved_mode1_projected_internal_rigid_removed_overlap": float(projected_internal_overlap[0]),
        "all_resolved_mode1_common_window_amplitude": float(projected_amplitude[0]),
        "all_resolved_mode1_projected_raw_dot": float(projected_raw_dot[0]),
        "all_resolved_best60_projected_mode": int(projected_normalized_overlap.argmax()) + 1,
        "all_resolved_best60_projected_normalized_overlap": float(projected_normalized_overlap.max()),
        "all_resolved_eigenvalue_mode1": float(values[0]),
        "definitions_not_averaged": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    cutoffs = [float(args.primary_cutoff)] + [float(value) for value in args.sensitivity_cutoffs]
    cutoffs = list(dict.fromkeys(cutoffs))
    alphas = [float(value) for value in args.alphas]
    confs, labels, window, axis, mean_axis = load_reference()

    config = {
        "script": "scripts/strengthen_ddb1.py",
        "output_dir": str(output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir),
        "offline": bool(args.offline),
        "cases": [
            {"pdb": pdb_id, "crbn_chain": crbn, "ddb1_chain": ddb1, "note": note}
            for pdb_id, crbn, ddb1, note in CASES
        ],
        "primary_cutoff_A": float(args.primary_cutoff),
        "sensitivity_cutoffs_A": [float(value) for value in args.sensitivity_cutoffs],
        "interface_alphas": alphas,
        "primary_modes": int(args.primary_modes),
        "mode_rows_per_condition": int(args.sensitivity_modes),
        "sensitivity_modes": int(args.sensitivity_modes),
        "best_fields": "best_mode and internal_best_mode are best within primary_modes; best60_* fields are best within sensitivity_modes",
        "near_degenerate_ratio": NEAR_DEGENERATE_RATIO,
        "blas_thread_env": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }

    source_hashes = {
        "data/crbn_ensemble.ens.npz": sha256_bytes(read_source_bytes(DATA / "crbn_ensemble.ens.npz")),
        "data/crbn_residue_window.csv": sha256_bytes(read_source_bytes(DATA / "crbn_residue_window.csv")),
        "data/pca_diffvec.npz": sha256_bytes(read_source_bytes(DATA / "pca_diffvec.npz")),
        "scripts/strengthen_ddb1.py": sha256_bytes(read_source_bytes(Path(__file__))),
    }

    all_modes: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    for pdb_id, crbn_chain, ddb1_chain, note in CASES:
        print(f"{pdb_id}: loading chains {crbn_chain}/{ddb1_chain} and running {len(cutoffs)} cutoffs")
        modes, summaries, case_checks, case_hashes = analyse_case(
            pdb_id, crbn_chain, ddb1_chain, note, cutoffs, float(args.primary_cutoff), alphas,
            int(args.primary_modes), int(args.sensitivity_modes), confs, labels, window, axis, bool(args.offline)
        )
        all_modes.extend(modes)
        all_summary.extend(summaries)
        cases.append(case_checks)
        source_hashes.update(case_hashes)
        primary = [
            row for row in summaries
            if row["cutoff_A"] == float(args.primary_cutoff)
            and row["model"] in {"isolated", "joint", "schur_static", "fixed_partner"}
            and row["interface_alpha"] in {0.0, 1.0}
        ]
        for row in primary:
            print(
                f"  {row['model']:13s} alpha {row['interface_alpha']:.1f}: "
                f"best mode {row['best_mode']} overlap {row['best_crbn_directional_overlap']:.3f} "
                f"amp {row['best_crbn_amplitude']:.3f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "modes.csv", all_modes)
    write_csv(output_dir / "model_summary.csv", all_summary)

    primary_rows = [
        row for row in all_summary
        if row["cutoff_A"] == float(args.primary_cutoff) and row["model"] == "schur_static"
    ]
    primary_alpha1 = [row for row in primary_rows if row["interface_alpha"] == 1.0]
    response_errors = [
        model["schur_full_static_response_relative_error"]
        for case in cases
        for cutoff in case["cutoffs"].values()
        for model in cutoff["models"].values()
    ]
    equilibrium_errors = [
        model["full_block_static_equilibrium_relative_error"]
        for case in cases
        for cutoff in case["cutoffs"].values()
        for model in cutoff["models"].values()
    ]
    payload = {
        "config": config,
        "provenance": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": os.popen("git rev-parse HEAD").read().strip(),
            "git_branch": os.popen("git branch --show-current").read().strip(),
            "source_hash_file": "source_hashes.json",
            "models_file": "modes.csv",
            "model_summary_file": "model_summary.csv",
        },
        "model_definitions": {
            "isolated": "CRBN-only 269-node ANM",
            "zero_interface": (
                "CRBN+DDB1 block diagonal Hessian; exact zero CRBN-DDB1 springs. "
                "A zero CRBN amplitude row is an explicit scoring convention for a DDB1-only "
                "joint eigenvector, not evidence that a biological CRBN motion is orthogonal."
            ),
            "joint": "CRBN+DDB1 Hessian with CRBN-DDB1 interface springs scaled by alpha",
            "schur_static": "A-B D^+ B^T after static DDB1 relaxation; stiffness only, not dynamic frequency",
            "fixed_partner": "CRBN block A with DDB1 fixed",
        },
        "summary": {
            "n_mode_rows": len(all_modes),
            "n_model_summary_rows": len(all_summary),
            "primary_schur_alpha1_best20_rank_range": [
                int(min(row["best_mode"] for row in primary_alpha1)),
                int(max(row["best_mode"] for row in primary_alpha1)),
            ],
            "primary_schur_alpha1_best20_overlap_range": [
                round(float(min(row["best_crbn_directional_overlap"] for row in primary_alpha1)), 3),
                round(float(max(row["best_crbn_directional_overlap"] for row in primary_alpha1)), 3),
            ],
            "primary_schur_alpha1_internal_best20_rank_range": [
                int(min(row["internal_best_mode"] for row in primary_alpha1)),
                int(max(row["internal_best_mode"] for row in primary_alpha1)),
            ],
            "primary_schur_alpha1_internal_best20_overlap_range": [
                round(float(min(row["internal_best_overlap"] for row in primary_alpha1)), 3),
                round(float(max(row["internal_best_overlap"] for row in primary_alpha1)), 3),
            ],
            "primary_schur_alpha1_best60_rank_range": [
                int(min(row["best60_mode"] for row in primary_alpha1)),
                int(max(row["best60_mode"] for row in primary_alpha1)),
            ],
            "primary_schur_alpha1_best60_overlap_range": [
                round(float(min(row["best60_crbn_directional_overlap"] for row in primary_alpha1)), 3),
                round(float(max(row["best60_crbn_directional_overlap"] for row in primary_alpha1)), 3),
            ],
            "primary_schur_alpha1_internal_best60_rank_range": [
                int(min(row["internal_best60_mode"] for row in primary_alpha1)),
                int(max(row["internal_best60_mode"] for row in primary_alpha1)),
            ],
            "primary_schur_alpha1_internal_best60_overlap_range": [
                round(float(min(row["internal_best60_overlap"] for row in primary_alpha1)), 3),
                round(float(max(row["internal_best60_overlap"] for row in primary_alpha1)), 3),
            ],
            "primary_schur_alpha1_best_rank_range": [
                int(min(row["best_mode"] for row in primary_alpha1)),
                int(max(row["best_mode"] for row in primary_alpha1)),
            ],
            "primary_schur_alpha1_overlap_range": [
                round(float(min(row["best_crbn_directional_overlap"] for row in primary_alpha1)), 3),
                round(float(max(row["best_crbn_directional_overlap"] for row in primary_alpha1)), 3),
            ],
            "max_schur_full_static_response_relative_error": float(max(response_errors)),
            "max_full_block_static_equilibrium_relative_error": float(max(equilibrium_errors)),
            "observed_8cvp_reconciliation": observed_8cvp_reconciliation(
                all_summary, confs, labels, window, axis, bool(args.offline)
            ),
        },
        "cases": cases,
    }
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "source_hashes.json").open("w") as handle:
        json.dump(source_hashes, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "model_summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    timings = {
        "wall_seconds": round(time.perf_counter() - started, 3),
        "user_cpu_seconds": round(float(usage.ru_utime), 3),
        "system_cpu_seconds": round(float(usage.ru_stime), 3),
        "max_rss": int(usage.ru_maxrss),
        "max_rss_unit": "bytes_on_macos_kilobytes_elsewhere",
        "case_cutoff_seconds": {
            case["pdb"]: {cutoff: value["seconds"] for cutoff, value in case["cutoffs"].items()}
            for case in cases
        },
    }
    with (output_dir / "timings.json").open("w") as handle:
        json.dump(timings, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.verify:
        if payload["summary"]["max_schur_full_static_response_relative_error"] > 1e-6:
            raise RuntimeError(payload["summary"]["max_schur_full_static_response_relative_error"])
        if payload["summary"]["max_full_block_static_equilibrium_relative_error"] > 1e-6:
            raise RuntimeError(payload["summary"]["max_full_block_static_equilibrium_relative_error"])
        if any(case["n_crbn_nodes"] != 269 for case in cases):
            raise RuntimeError("not every case used 269 CRBN nodes")
        if any(value["n_interface_pairs"] <= 0 for case in cases for value in case["cutoffs"].values()):
            raise RuntimeError("an interface had zero pairs at a requested cutoff")
        print("verify OK: matched 269-node CRBN models, nonzero interfaces, and Schur/full static responses agree")

    print(f"wrote {output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir}")
    print(
        "summary: primary Schur alpha=1 rank "
        f"{payload['summary']['primary_schur_alpha1_best_rank_range']}, overlap "
        f"{payload['summary']['primary_schur_alpha1_overlap_range']}, max static-response error "
        f"{payload['summary']['max_schur_full_static_response_relative_error']:.2e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
