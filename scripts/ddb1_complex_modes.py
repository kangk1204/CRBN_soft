#!/usr/bin/env python3
"""Does the open->closed soft mode survive when the obligate adaptor DDB1 is in
the network?

The manuscript's central claim is built on an anisotropic network model (ANM) of
the CRBN monomer. A structural biologist will object that the biological unit is
not the monomer: CRBN is an obligate partner of DDB1 within CRL4^CRBN, and every
open structure in the ensemble is deposited as a CRBN-DDB1 complex. If the
transition axis is only the softest mode of the isolated fold and stops being so
once the adaptor is present, the claim is about an object that does not exist in
the cell.

This script answers that with four calculations per structure.

(1) MONOMER BASELINE reproduces the manuscript's per-structure value, so the rest
    is anchored to a number the reader can check.

(2) JOINT SPECTRUM builds one ANM over CRBN plus DDB1 (about 1400 C-alpha) and
    scores every mode against the transition axis embedded with zeros on DDB1.
    This is the calculation the objection asks for, and the axis is no longer the
    slowest mode: it appears at ranks 4-7 depending on the structure.

(3) ZERO-CONTACT DECOY translates DDB1 500 A away, leaving the residue count and
    therefore the mode count unchanged while removing every interface contact.
    Any rank shift here is a pure density-of-states effect from the added degrees
    of freedom. This control is what makes (2) interpretable: the decoy shift is
    0-3 ranks and is exactly 0 for the two ternary structures, so the observed
    4-7 cannot be attributed to counting alone. Modes in (2) with no counterpart
    in the decoy spectrum (mode-mode overlap < 0.5) are interface-created; the
    number of those varies by structure (1 to 3 of the modes below the axis), and
    the remaining displacing modes are monomer modes that the interface stiffness
    reorders rather than new ones. Both mechanisms contribute and their mixture is
    not constant across structures, so this script reports the decomposition
    rather than asserting a single mechanism.

(4) GUYAN / SCHUR-COMPLEMENT REDUCTION condenses the DDB1 coordinates out
    exactly, H_eff = A - B D^+ B^T on the CRBN block. What remains is CRBN's own
    static response after DDB1 is allowed to relax, without counting DDB1's own
    modes as CRBN modes. This is a quasi-static stiffness calculation, not a
    physical frequency calculation for the complex. It asks whether the adaptor
    statically stiffens this hinge, or merely adds slower joint modes of its own.
    In this legacy calculation the reduction reproduces the free-monomer rank
    exactly (10 of 10 calculations) with the overlap essentially unchanged
    (mean +0.017), while the unit-spring Schur eigenvalue rises by a factor of
    1.0-2.0. The claim is static rank preservation within the reduced CRBN block,
    not preservation of actual dynamic frequencies and not rank 1 everywhere:
    for the two ternary structures at 13 A the free monomer itself places the axis
    at rank 2, and the reduction reproduces that.

Reference frames are load-bearing here. ANM eigenvectors are expressed in the
frame of the input coordinates, while the committed difference vector lives in
the frame of the superposed ensemble. Scoring raw-mmCIF modes against it without
a Kabsch rotation is a frame mismatch that silently produces wrong overlaps (it
gave 0.145 for 6H0F, against the correct 0.761). Every structure here is rotated
onto its ensemble member before any mode is scored, and the script asserts the
residual RMSD is zero so a future reader sees the check.

Inputs
  data/crbn_ensemble.ens.npz     superposed ensemble (defines the frame)
  data/crbn_residue_window.csv   the 269 analysis positions
  data/pca_diffvec.npz           open/closed masks
  RCSB mmCIF                     downloaded on demand, cached under data/cif_cache/

Output
  data/ddb1_complex_modes.json

Usage
  python scripts/ddb1_complex_modes.py
  python scripts/ddb1_complex_modes.py --verify
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import urllib.request

import numpy as np
from scipy.linalg import eigh
from scipy.stats import mannwhitneyu

try:
    from pdb_id import validate_pdb_id
except ModuleNotFoundError:  # imported by path from the repository root
    from scripts.pdb_id import validate_pdb_id

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
CIF_CACHE = os.path.join(DATA, "cif_cache")
OUT = os.path.join(DATA, "ddb1_complex_modes.json")
CACHE_WRITES_ENABLED = True

# CRBN chain and DDB1 chain for each structure carrying a resolved 269-residue window.
# 7U8F deposits CRBN as chain A and DDB1 as chain B; the others are the reverse.
CASES = [
    ("8CVP", "B", "A", "apo cryo-EM, ANM reference of the manuscript"),
    ("8D7X", "B", "A", "apo cryo-EM"),
    ("8D7Y", "B", "A", "apo cryo-EM, DDB1 twisted conformation"),
    ("6H0F", "B", "A", "pomalidomide + IKZF1 ternary, open"),
    ("7U8F", "A", "B", "DKY709 + IKZF2 ternary, open"),
]
CUTOFFS = (15.0, 13.0)
N_JOINT_MODES = 60
DECOY_OFFSET = 500.0          # A; far beyond any cutoff, so coupling is exactly zero
NEW_MODE_THRESHOLD = 0.5      # max |overlap| with any decoy mode below which a mode is "new"

# Domain boundaries used only for the two-body rigid-body diagnostic.
NTD = (77, 186)
HB = (187, 317)
TBD = (318, 424)


def fetch_cif(pdb_id: str) -> str:
    """Return mmCIF text, caching only during generation (never verification)."""
    pdb_id = validate_pdb_id(pdb_id)
    path = os.path.join(CIF_CACHE, f"{pdb_id}.cif.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt", errors="replace") as fh:
            return fh.read()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    with urllib.request.urlopen(url, timeout=120) as fh:
        blob = fh.read()
    text = gzip.decompress(blob).decode("utf-8", errors="replace")
    if CACHE_WRITES_ENABLED:
        os.makedirs(CIF_CACHE, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(blob)
    return text


def ca_coords(pdb_id: str, chain: str) -> dict:
    """C-alpha coordinates of one chain, keyed by author residue number.

    Takes the first occurrence of each residue number, which is the repository's
    convention for alternate conformations and matches the ensemble builder.
    """
    lines = fetch_cif(pdb_id).splitlines()
    out = {}
    i = 0
    while i < len(lines):
        if lines[i].startswith("_atom_site."):
            cols = []
            j = i
            while lines[j].startswith("_atom_site."):
                cols.append(lines[j].strip().split(".")[1])
                j += 1
            ix = {c: k for k, c in enumerate(cols)}
            while j < len(lines) and not lines[j].startswith("#"):
                f = lines[j].split()
                if (len(f) >= len(cols)
                        and f[ix["group_PDB"]] == "ATOM"
                        and f[ix["label_atom_id"]] == "CA"
                        and f[ix["auth_asym_id"]] == chain):
                    try:
                        num = int(f[ix["auth_seq_id"]])
                    except ValueError:
                        j += 1
                        continue
                    out.setdefault(num, [float(f[ix["Cartn_x"]]),
                                         float(f[ix["Cartn_y"]]),
                                         float(f[ix["Cartn_z"]])])
                j += 1
            break
        i += 1
    return out


def anm_hessian(coords: np.ndarray, cutoff: float) -> np.ndarray:
    n = len(coords)
    H = np.zeros((3 * n, 3 * n))
    D = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    ii, jj = np.where((D <= cutoff) & (D > 1e-6))
    upper = ii < jj
    for i, j in zip(ii[upper], jj[upper]):
        d = coords[j] - coords[i]
        k = np.outer(d, d) / D[i, j] ** 2
        H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = -k
        H[3 * j:3 * j + 3, 3 * i:3 * i + 3] = -k
        H[3 * i:3 * i + 3, 3 * i:3 * i + 3] += k
        H[3 * j:3 * j + 3, 3 * j:3 * j + 3] += k
    return H


def slow_modes(H: np.ndarray, k: int):
    n = H.shape[0]
    hi = min(n - 1, k + 24)
    while True:
        w, v = eigh(H, subset_by_index=(0, hi), check_finite=False)
        nz = w > 1e-9
        if int(nz.sum()) >= k or hi == n - 1:
            return w[nz][:k], v[:, nz][:, :k]
        hi = min(n - 1, max(2 * hi + 1, hi + k + 24))


def kabsch(mobile: np.ndarray, target: np.ndarray):
    """Rotation and centroids taking `mobile` onto `target`."""
    pc = mobile.mean(0)
    qc = target.mean(0)
    U, _, Vt = np.linalg.svd((mobile - pc).T @ (target - qc))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return R, pc, qc


def rigid_basis(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the 6 rigid-body dof of the masked subset, in 3N space."""
    n = len(coords)
    cols = []
    centre = coords[mask].mean(0)
    for axis in range(3):
        t = np.zeros((n, 3))
        t[mask, axis] = 1.0
        cols.append(t.ravel())
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        t = np.zeros((n, 3))
        t[mask] = np.cross(e, coords[mask] - centre)
        cols.append(t.ravel())
    return np.linalg.qr(np.array(cols).T)[0]


def load_reference():
    ens = np.load(os.path.join(DATA, "crbn_ensemble.ens.npz"), allow_pickle=False)
    confs = ens["_confs"]
    labels = [str(lab)[:4] for lab in ens["_labels"]]
    with open(os.path.join(DATA, "crbn_residue_window.csv")) as fh:
        window = np.array([int(r["author_resnum"]) for r in csv.DictReader(fh)])
    diff = np.load(os.path.join(DATA, "pca_diffvec.npz"))
    open_mask = diff["open_mask"].astype(bool)
    axis = (confs[open_mask].mean(0) - confs[~open_mask].mean(0)).reshape(-1)
    axis /= np.linalg.norm(axis)
    return confs, labels, window, axis


def analyse(pdb_id, crbn_chain, ddb1_chain, cutoff, confs, labels, window, axis):
    crbn = ca_coords(pdb_id, crbn_chain)
    ddb1 = ca_coords(pdb_id, ddb1_chain)
    missing = [int(r) for r in window if int(r) not in crbn]
    if missing:
        raise RuntimeError(f"{pdb_id} chain {crbn_chain} misses window residues {missing[:5]}")

    raw = np.array([crbn[int(r)] for r in window])
    ref = confs[labels.index(pdb_id)]
    R, pc, qc = kabsch(raw, ref)
    rotate = lambda P: (P - pc) @ R + qc

    crbn_xyz = rotate(raw)
    frame_rmsd = float(np.sqrt(((crbn_xyz - ref) ** 2).sum() / len(ref)))
    assert frame_rmsd < 1e-6, (
        f"{pdb_id}: frame match failed (RMSD {frame_rmsd:.4f} A). Modes scored against "
        "the ensemble-frame axis would be meaningless.")

    ddb1_nums = sorted(ddb1)
    ddb1_xyz = rotate(np.array([ddb1[r] for r in ddb1_nums]))
    n_c = len(crbn_xyz)
    n_d = len(ddb1_xyz)

    # (1) monomer baseline
    w_mono, v_mono = slow_modes(anm_hessian(crbn_xyz, cutoff), 20)
    ov_mono = np.array([abs(v_mono[:, m] @ axis) for m in range(20)])

    # (2) joint spectrum
    joint_xyz = np.vstack([crbn_xyz, ddb1_xyz])
    H_joint = anm_hessian(joint_xyz, cutoff)
    w_joint, v_joint = slow_modes(H_joint, N_JOINT_MODES)
    embedded = np.zeros(3 * (n_c + n_d))
    embedded[:3 * n_c] = axis
    embedded /= np.linalg.norm(embedded)
    ov_joint = np.array([abs(v_joint[:, m] @ embedded) for m in range(N_JOINT_MODES)])
    aligned = int(ov_joint.argmax())

    # (3) zero-contact decoy: same residue count, no interface
    decoy_xyz = np.vstack([crbn_xyz, ddb1_xyz + np.array([DECOY_OFFSET, 0.0, 0.0])])
    w_decoy, v_decoy = slow_modes(anm_hessian(decoy_xyz, cutoff), N_JOINT_MODES)
    ov_decoy = np.array([abs(v_decoy[:, m] @ embedded) for m in range(N_JOINT_MODES)])
    decoy_rank = int(ov_decoy.argmax()) + 1

    # which displacing modes have no counterpart in the decoy spectrum?
    two_body = np.zeros(len(joint_xyz), dtype=bool)
    two_body[:n_c] = True
    Q2 = np.linalg.qr(np.hstack([rigid_basis(joint_xyz, two_body),
                                 rigid_basis(joint_xyz, ~two_body)]))[0]
    displacers = []
    for m in range(aligned):
        match = float(np.abs(v_decoy.T @ v_joint[:, m]).max())
        displacers.append(dict(
            mode=m + 1,
            best_decoy_match=match,
            interface_created=bool(match < NEW_MODE_THRESHOLD),
            two_body_rigid=float(np.linalg.norm(Q2.T @ v_joint[:, m])),
            crbn_amplitude=float(np.linalg.norm(v_joint[:3 * n_c, m])),
            eigenvalue=float(w_joint[m]),
        ))
    n_new = sum(d["interface_created"] for d in displacers)

    # (4) Guyan reduction: condense DDB1 out exactly
    A = H_joint[:3 * n_c, :3 * n_c]
    B = H_joint[:3 * n_c, 3 * n_c:]
    Dblock = H_joint[3 * n_c:, 3 * n_c:]
    H_eff = A - B @ np.linalg.pinv(Dblock, rcond=1e-10) @ B.T
    w_eff, v_eff = slow_modes(H_eff, 20)
    ov_eff = np.array([abs(v_eff[:, m] @ axis) for m in range(20)])

    return dict(
        pdb=pdb_id, crbn_chain=crbn_chain, ddb1_chain=ddb1_chain,
        cutoff=cutoff, n_crbn=n_c, n_ddb1=n_d, frame_match_rmsd=frame_rmsd,
        monomer=dict(mode1_overlap=float(ov_mono[0]),
                     best_overlap=float(ov_mono.max()),
                     best_rank=int(ov_mono.argmax()) + 1,
                     eigenvalue_mode1=float(w_mono[0])),
        joint=dict(aligned_rank=aligned + 1,
                   aligned_overlap=float(ov_joint[aligned]),
                   mode1_overlap=float(ov_joint[0]),
                   cumulative_top10=float(np.sqrt((ov_joint[:10] ** 2).sum())),
                   eigenvalue_aligned=float(w_joint[aligned]),
                   aligned_two_body_rigid=float(np.linalg.norm(Q2.T @ v_joint[:, aligned]))),
        decoy=dict(aligned_rank=decoy_rank,
                   aligned_overlap=float(ov_decoy.max()),
                   size_effect_ranks=decoy_rank - (int(ov_mono.argmax()) + 1),
                   contact_effect_ranks=(aligned + 1) - decoy_rank),
        displacing_modes=displacers,
        n_interface_created=int(n_new),
        effective=dict(mode1_overlap=float(ov_eff[0]),
                       best_overlap=float(ov_eff.max()),
                       best_rank=int(ov_eff.argmax()) + 1,
                       eigenvalue_mode1=float(w_eff[0]),
                       stiffening_factor=float(w_eff[0] / w_mono[0])),
    )


def summarise(records):
    rank_preserved = [r for r in records
                      if r["effective"]["best_rank"] == r["monomer"]["best_rank"]]
    stiff = np.array([r["effective"]["stiffening_factor"] for r in records])
    eff_ov = np.array([r["effective"]["mode1_overlap"] for r in records])
    mono_ov = np.array([r["monomer"]["mode1_overlap"] for r in records])
    joint_rank = np.array([r["joint"]["aligned_rank"] for r in records])
    decoy_rank = np.array([r["decoy"]["aligned_rank"] for r in records])
    size = np.array([r["decoy"]["size_effect_ranks"] for r in records])
    contact = np.array([r["decoy"]["contact_effect_ranks"] for r in records])
    n_new = np.array([r["n_interface_created"] for r in records])
    new_rigid = [d["two_body_rigid"] for r in records for d in r["displacing_modes"]
                 if d["interface_created"]]
    old_rigid = [d["two_body_rigid"] for r in records for d in r["displacing_modes"]
                 if not d["interface_created"]]
    aligned_rigid = [r["joint"]["aligned_two_body_rigid"] for r in records]
    return dict(
        n_calculations=len(records),
        joint_aligned_rank_range=[int(joint_rank.min()), int(joint_rank.max())],
        decoy_aligned_rank_range=[int(decoy_rank.min()), int(decoy_rank.max())],
        rank_shift_decomposition=dict(
            size_effect_mean=float(size.mean()), size_effect_range=[int(size.min()), int(size.max())],
            contact_effect_mean=float(contact.mean()),
            contact_effect_range=[int(contact.min()), int(contact.max())],
            contact_share_of_total=float(contact.sum() / (size.sum() + contact.sum())),
            note=("Both mechanisms contribute. Contact dominates on average but the mixture "
                  "varies: the size effect is 0 for the two ternary structures and up to 3 "
                  "for 8D7Y.")),
        interface_created_modes=dict(
            count_per_structure_range=[int(n_new.min()), int(n_new.max())],
            n_interface_created=len(new_rigid),
            n_reordered_monomer=len(old_rigid),
            two_body_rigid_median=round(float(np.median(new_rigid)), 3) if new_rigid else None,
            two_body_rigid_range=([round(min(new_rigid), 3), round(max(new_rigid), 3)]
                                  if new_rigid else None),
            reordered_monomer_modes_two_body_rigid_median=(
                round(float(np.median(old_rigid)), 3) if old_rigid else None),
            reordered_monomer_modes_two_body_rigid_range=(
                [round(min(old_rigid), 3), round(max(old_rigid), 3)] if old_rigid else None),
            aligned_mode_two_body_rigid_median=round(float(np.median(aligned_rigid)), 3),
            aligned_mode_two_body_rigid_range=[round(min(aligned_rigid), 3),
                                               round(max(aligned_rigid), 3)],
            mannwhitney_p_new_greater=(
                float(mannwhitneyu(new_rigid, old_rigid, alternative="greater").pvalue)
                if new_rigid and old_rigid else None),
            note=("Modes with no decoy counterpart are CRBN-DDB1 relative rigid-body wobble and "
                  "carry higher two-body rigid content (median 0.89) than the remaining "
                  "displacing modes, which are monomer modes reordered by interface stiffness "
                  "(median 0.56); the two distributions separate significantly but their RANGES "
                  "OVERLAP (0.66-0.95 against 0.25-0.84), so quote medians, not range floors, and "
                  "do not treat the classification as clean per mode. The aligned transition mode "
                  "itself has much lower two-body rigid content, confirming it is internal CRBN "
                  "motion rather than relative wobble. The displacement is therefore a mixture of "
                  "two mechanisms whose proportions vary by structure.")),
        effective_hessian=dict(
            rank_preserved_fraction=len(rank_preserved) / len(records),
            rank_preserved=len(rank_preserved) == len(records),
            mode1_overlap_range=[round(float(eff_ov.min()), 3), round(float(eff_ov.max()), 3)],
            monomer_overlap_range=[round(float(mono_ov.min()), 3), round(float(mono_ov.max()), 3)],
            mean_overlap_change=round(float((eff_ov - mono_ov).mean()), 3),
            stiffening_range=[round(float(stiff.min()), 2), round(float(stiff.max()), 2)],
            note=("Condensing DDB1 out exactly reproduces the free-monomer rank in every "
                  "calculation (rank 1 at both cutoffs for the three apo structures and at "
                  "15 A for the two ternary ones; rank 2 for the two ternary structures at "
                  "13 A, matching their free-monomer rank there). Overlap is essentially "
                  "unchanged (mean +0.017, range -0.008 to +0.046) while the unit-spring "
                  "Schur eigenvalue rises 1.0-2.0x. This is a quasi-static stiffness result "
                  "for the reduced CRBN block, not a preserved physical dynamic frequency for "
                  "the complex. DDB1 stiffens the hinge without changing where it sits in "
                  "CRBN's own static reduced spectrum; the joint-spectrum displacement comes "
                  "from DDB1's added modes.")),
        caveats=[
            "The biological unit is larger still: CRL4^CRBN also contains CUL4A, RBX1 and NEDD8. "
            "scripts/crl4_assembly_modes.py shows these make zero C-alpha contacts with CRBN "
            "inside the 15 A cutoff in all four complete assemblies (nearest approach "
            "22.9-28.4 A), so they add no spring to CRBN directly - but they are not absent from "
            "its condensed Hessian. H_eff = A - B D^+ B^T has B zero on their rows while D is "
            "not, and they stiffen DDB1 and the degron; adding them on top of DDB1+degron+E2 "
            "moves the closed-endpoint mode-1 overlap by up to 0.193 and the mode-1 eigenvalue "
            "by 1.32-1.45x. On the two open ternaries the degron leaves overlap and rank "
            "unchanged.",
            "DDB1 itself is conformationally variable; each structure contributes its own "
            "DDB1 conformer (8D7Y is deposited in the twisted state).",
            "Guyan reduction is quasi-static: it assumes DDB1 relaxes instantaneously, so it "
            "answers whether the adaptor stiffens this hinge, not how the two couple "
            "dynamically.",
            "Interface springs use the same force constant as intramolecular contacts, the "
            "ANM convention but a coarse approximation at a protein-protein interface.",
            "The two ternary structures show almost no stiffening (1.09-1.10x) against "
            "1.38-1.96x for the apo structures; this is unexplained.",
        ],
    )


def main() -> int:
    global CACHE_WRITES_ENABLED
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="check the committed conclusions still hold")
    args = ap.parse_args()
    CACHE_WRITES_ENABLED = not args.verify

    confs, labels, window, axis = load_reference()
    records = []
    for pdb_id, cc, dc, _note in CASES:
        for cutoff in CUTOFFS:
            rec = analyse(pdb_id, cc, dc, cutoff, confs, labels, window, axis)
            records.append(rec)
            print(f"{pdb_id} {cutoff:.0f} A: monomer {rec['monomer']['mode1_overlap']:.3f} "
                  f"(rank {rec['monomer']['best_rank']}) | joint rank "
                  f"{rec['joint']['aligned_rank']} ({rec['joint']['aligned_overlap']:.3f}) | "
                  f"decoy rank {rec['decoy']['aligned_rank']} | DDB1 condensed "
                  f"{rec['effective']['mode1_overlap']:.3f} (rank "
                  f"{rec['effective']['best_rank']}), stiffening "
                  f"{rec['effective']['stiffening_factor']:.2f}x")

    summary = summarise(records)
    payload = dict(
        provenance=dict(
            script="scripts/ddb1_complex_modes.py",
            cases=[dict(pdb=p, crbn_chain=c, ddb1_chain=d, note=n) for p, c, d, n in CASES],
            cutoffs=list(CUTOFFS), n_joint_modes=N_JOINT_MODES,
            decoy_offset_angstrom=DECOY_OFFSET,
            new_mode_threshold=NEW_MODE_THRESHOLD),
        summary=summary,
        records=records,
    )
    if not args.verify:
        with open(OUT, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"wrote {os.path.relpath(OUT, ROOT)}")

    eff = summary["effective_hessian"]
    dec = summary["rank_shift_decomposition"]
    print(f"\nsummary: joint rank {summary['joint_aligned_rank_range']}, decoy rank "
          f"{summary['decoy_aligned_rank_range']}; rank shift is "
          f"{100 * dec['contact_share_of_total']:.0f}% contact-driven "
          f"(size effect {dec['size_effect_range']}, contact {dec['contact_effect_range']}); "
          f"static Schur reduction of DDB1 reproduces the free-monomer rank in "
          f"{eff['rank_preserved_fraction']:.0%} of calculations, overlap change "
          f"{eff['mean_overlap_change']:+.3f}, stiffening {eff['stiffening_range']}x")

    if args.verify:
        assert eff["rank_preserved"], (
            "condensing DDB1 out did not reproduce the free-monomer rank")
        assert abs(eff["mean_overlap_change"]) < 0.05, eff["mean_overlap_change"]
        assert eff["stiffening_range"][0] >= 1.0, eff["stiffening_range"]
        assert summary["joint_aligned_rank_range"][0] >= 4, summary["joint_aligned_rank_range"]
        assert dec["contact_share_of_total"] > 0.5, dec["contact_share_of_total"]
        assert dec["size_effect_range"][0] == 0, dec["size_effect_range"]
        assert summary["interface_created_modes"]["count_per_structure_range"][0] >= 1, \
            summary["interface_created_modes"]
        print("verify OK: the joint-spectrum rank shift is majority contact-driven "
              f"({100 * dec['contact_share_of_total']:.0f}%), not a mode-count artefact; "
              "static Schur reduction of DDB1 reproduces the free-monomer rank in every calculation "
              f"({eff['rank_preserved_fraction']:.0%}) with overlap unchanged "
              f"({eff['mean_overlap_change']:+.3f}) and 1.0-2.0x stiffening")
    return 0


if __name__ == "__main__":
    sys.exit(main())
