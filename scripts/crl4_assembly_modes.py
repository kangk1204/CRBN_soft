#!/usr/bin/env python3
"""Close the "biological unit" objection: is the soft mode a property of CRBN alone,
of CRBN-DDB1, or of the whole cullin-RING ligase?

`ddb1_complex_modes.py` answered the objection one step out from the monomer and
left a caveat: CUL4A and RBX1 are not in the network. That caveat was written
because no calculation had been done, not because one was impossible -- four
depositions carry the complete neddylated assembly (9UUM, 9V0A, 9V0B, 9V0F:
CRBN + DDB1 + CUL4A + RBX1 + NEDD8 + UBE2D1 + ubiquitin + an IKZF3 degron), and
all four are already curated conformers of the ensemble. This script tests the
caveat instead of restating it.

The caveat turns out to be half right. The cullin arm touches nothing in CRBN, but
"no contact" is not the same as "no influence", and the difference is measurable.

(1) THE CULLIN ARM MAKES NO DIRECT CONTACT, YET STILL REACHES CRBN. In all four
    assemblies the nearest CUL4A C-alpha sits 27.1-28.4 A from any CRBN C-alpha,
    RBX1 22.9-24.3 A and NEDD8 23.8-25.3 A, so at the 15 A cutoff used throughout
    these subunits contribute no springs to CRBN: the coupling block B is exactly
    zero on their rows. That is where the earlier reasoning stopped, and it was
    wrong to stop there. Condensation gives H_eff = A - B D^+ B^T, and although B
    is zero on the cullin rows, D is not: RBX1 lies 7.3 A from the degron and 4.9 A
    from UBE2D1, and CUL4A packs against DDB1 over more than a thousand C-alpha
    pairs. The scaffold therefore stiffens DDB1 and the degron, and that stiffening
    propagates into CRBN's quasi-static condensed Hessian. Measured on the ladder, adding
    CUL4A+RBX1+NEDD8 on top of DDB1+degron+E2 shifts the mode-1 overlap by up to
    0.193 (9V0A, 0.543 to 0.736) and raises the mode-1 Schur stiffness eigenvalue
    1.32-1.45x. This does not mean the physical assembly frequency is preserved or
    measured; the "biological unit is larger" objection is answered by static
    stiffness measurement, not by a distance argument.

(2) WHAT DOES TOUCH CRBN IS THE DEGRON AND E2. The degron lies 5.1-6.1 A away with
    360-410 C-alpha pairs inside 15 A, and UBE2D1 5.7-6.3 A with 138-153 pairs.
    Neither was in any network in the manuscript. On the open structures the claim
    actually rests on -- 6H0F and 7U8F, the two open ternaries -- including the
    degron leaves the result unchanged: mode-1 overlap 0.761 to 0.766 for 6H0F and
    0.767 to 0.757 for 7U8F, rank 1 throughout, stiffening only 1.10-1.12x. The
    degron is bound at the TBD surface, not across the hinge, so it does not
    restrain the interdomain coordinate.

(3) A SIDE RESULT WORTH REPORTING. On the four closed assemblies the same nesting
    changes the picture substantially: monomer mode-1 overlap is 0.011-0.097
    (closed structures are poor ANM references for this transition, which is the
    manuscript's existing endpoint-asymmetry result), and adding DDB1 plus the
    degron raises it to 0.362-0.599, with everything that contacts CRBN included
    reaching 0.392-0.736. So assembly context partially rescues the closed
    endpoint. This does not affect the open-reference claim and should be reported
    as a separate observation, not folded into it.

Reference frames are load-bearing, as in `ddb1_complex_modes.py`: ANM eigenvectors
live in the frame of the input coordinates while the difference vector lives in the
ensemble frame, so every structure is Kabsch-rotated onto its ensemble member first
and the residual RMSD is asserted to be zero.

Inputs
  data/crbn_ensemble.ens.npz     superposed ensemble (defines the frame)
  data/crbn_residue_window.csv   the 269 analysis positions
  data/pca_diffvec.npz           open/closed masks
  RCSB mmCIF                     downloaded on demand, cached under data/cif_cache/

Output
  data/crl4_assembly_modes.json

Usage
  python scripts/crl4_assembly_modes.py
  python scripts/crl4_assembly_modes.py --verify
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

try:
    from pdb_id import validate_pdb_id
except ModuleNotFoundError:  # imported by path from the repository root
    from scripts.pdb_id import validate_pdb_id

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
CIF_CACHE = os.path.join(DATA, "cif_cache")
OUT = os.path.join(DATA, "crl4_assembly_modes.json")
CACHE_WRITES_ENABLED = True

CUTOFF = 15.0
N_MODES = 20

# Complete neddylated CRL4-CRBN assemblies. chain roles verified from the RCSB
# polymer-entity records; all four are closed, glue-bound IKZF3 ternaries.
ASSEMBLIES = {
    "9UUM": dict(crbn="C", ddb1="B", cul4a="A", rbx1="R", nedd8="N", e2="D",
                 degron="I", ubiquitin="U", glue="mezigdomide"),
    "9V0A": dict(crbn="C", ddb1="B", cul4a="A", rbx1="R", nedd8="N", e2="D",
                 degron="I", ubiquitin="U", glue="pomalidomide"),
    "9V0B": dict(crbn="C", ddb1="B", cul4a="A", rbx1="R", nedd8="N", e2="D",
                 degron="I", ubiquitin="U", glue="avadomide"),
    "9V0F": dict(crbn="C", ddb1="B", cul4a="A", rbx1="R", nedd8="N", e2="D",
                 degron="I", ubiquitin="U", glue="cemsidomide"),
}

# The two OPEN ternary structures: the only open depositions carrying a degron, so the
# only ones that can test degron inclusion against the manuscript's actual claim.
OPEN_TERNARIES = {
    "6H0F": dict(crbn="B", ddb1="A", degron="C", glue="pomalidomide"),
    "7U8F": dict(crbn="A", ddb1="B", degron="C", glue="DKY709"),
}

# Nesting ladders, evaluated by condensing every non-CRBN chain out of the Hessian.
CLOSED_LADDER = [
    ("monomer", []),
    ("DDB1", ["ddb1"]),
    ("DDB1+degron", ["ddb1", "degron"]),
    ("DDB1+degron+E2", ["ddb1", "degron", "e2"]),
    ("full_assembly", ["ddb1", "degron", "e2", "cul4a", "rbx1", "nedd8", "ubiquitin"]),
]
OPEN_LADDER = [
    ("monomer", []),
    ("DDB1", ["ddb1"]),
    ("degron_only", ["degron"]),
    ("DDB1+degron", ["ddb1", "degron"]),
]


def fetch_cif(pdb_id: str) -> str:
    """Return mmCIF text, caching only during generation (never verification)."""
    pdb_id = validate_pdb_id(pdb_id)
    path = os.path.join(CIF_CACHE, f"{pdb_id}.cif.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt", errors="replace") as fh:
            return fh.read()
    with urllib.request.urlopen(
            f"https://files.rcsb.org/download/{pdb_id}.cif.gz", timeout=180) as fh:
        blob = fh.read()
    text = gzip.decompress(blob).decode("utf-8", errors="replace")
    if CACHE_WRITES_ENABLED:
        os.makedirs(CIF_CACHE, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(blob)
    return text


def ca_by_chain(pdb_id: str) -> dict:
    """C-alpha coordinates per auth chain, keyed by author residue number.

    First occurrence of each residue wins, matching the ensemble builder's handling
    of alternate conformations.
    """
    lines = fetch_cif(pdb_id).splitlines()
    out: dict = {}
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
                if (len(f) >= len(cols) and f[ix["group_PDB"]] == "ATOM"
                        and f[ix["label_atom_id"]] == "CA"):
                    try:
                        num = int(f[ix["auth_seq_id"]])
                    except ValueError:
                        j += 1
                        continue
                    out.setdefault(f[ix["auth_asym_id"]], {}).setdefault(
                        num, [float(f[ix["Cartn_x"]]), float(f[ix["Cartn_y"]]),
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
    pc = mobile.mean(0)
    qc = target.mean(0)
    U, _, Vt = np.linalg.svd((mobile - pc).T @ (target - qc))
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt, pc, qc


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


def condensed_spectrum(crbn_xyz, partner_blocks, axis, cutoff=CUTOFF):
    """Slowest CRBN modes with partners condensed out exactly (Schur complement).

    With no partners this is the free monomer. With partners it is CRBN's
    quasi-static reduced stiffness after partner relaxation; partner modes are not
    counted as CRBN modes, and the returned eigenvalues are not physical assembly
    dynamic frequencies.
    """
    n_c = len(crbn_xyz)
    if partner_blocks:
        joint = np.vstack([crbn_xyz] + partner_blocks)
        H = anm_hessian(joint, cutoff)
        A = H[:3 * n_c, :3 * n_c]
        B = H[:3 * n_c, 3 * n_c:]
        D = H[3 * n_c:, 3 * n_c:]
        H = A - B @ np.linalg.pinv(D, rcond=1e-10) @ B.T
    else:
        H = anm_hessian(crbn_xyz, cutoff)
    w, v = slow_modes(H, N_MODES)
    ov = np.array([abs(v[:, m] @ axis) for m in range(N_MODES)])
    return dict(mode1_overlap=float(ov[0]), best_overlap=float(ov.max()),
                best_rank=int(ov.argmax()) + 1, eigenvalue_mode1=float(w[0]))


def analyse(pdb_id, roles, ladder, confs, labels, window, axis):
    chains = ca_by_chain(pdb_id)
    crbn_ch = roles["crbn"]
    missing = [int(r) for r in window if int(r) not in chains.get(crbn_ch, {})]
    if missing:
        raise RuntimeError(f"{pdb_id} chain {crbn_ch} misses window residues {missing[:5]}")

    raw = np.array([chains[crbn_ch][int(r)] for r in window])
    ref = confs[labels.index(pdb_id)]
    R, pc, qc = kabsch(raw, ref)
    rotate = lambda P: (P - pc) @ R + qc
    crbn_xyz = rotate(raw)
    frame_rmsd = float(np.sqrt(((crbn_xyz - ref) ** 2).sum() / len(ref)))
    assert frame_rmsd < 1e-6, f"{pdb_id}: frame match failed ({frame_rmsd:.4f} A)"

    def block(role):
        ch = roles.get(role)
        if ch is None or ch not in chains:
            return None
        return rotate(np.array([chains[ch][r] for r in sorted(chains[ch])]))

    # geometric contact census: which subunits can reach CRBN at all?
    contacts = {}
    for role in ("ddb1", "cul4a", "rbx1", "nedd8", "e2", "degron", "ubiquitin"):
        blk = block(role)
        if blk is None:
            continue
        D = np.linalg.norm(crbn_xyz[:, None] - blk[None], axis=2)
        contacts[role] = dict(n_ca=int(len(blk)), min_distance=float(D.min()),
                              n_pairs_within_cutoff=int((D <= CUTOFF).sum()),
                              reaches_crbn=bool(D.min() <= CUTOFF))

    steps = {}
    base_eig = None
    for tag, roles_in in ladder:
        blocks = [b for b in (block(r) for r in roles_in) if b is not None]
        spec = condensed_spectrum(crbn_xyz, blocks, axis)
        if base_eig is None:
            base_eig = spec["eigenvalue_mode1"]
        spec["stiffening_factor"] = spec["eigenvalue_mode1"] / base_eig
        spec["n_partner_ca"] = int(sum(len(b) for b in blocks))
        steps[tag] = spec

    return dict(pdb=pdb_id, glue=roles.get("glue"), frame_match_rmsd=frame_rmsd,
                n_crbn_ca=len(crbn_xyz), contacts=contacts, ladder=steps)


def summarise(closed, open_):
    remote = ("cul4a", "rbx1", "nedd8")
    remote_min = [r["contacts"][k]["min_distance"] for r in closed for k in remote
                  if k in r["contacts"]]
    remote_pairs = [r["contacts"][k]["n_pairs_within_cutoff"] for r in closed for k in remote
                    if k in r["contacts"]]
    touching = ("ddb1", "degron", "e2")
    touch_min = {k: [r["contacts"][k]["min_distance"] for r in closed if k in r["contacts"]]
                 for k in touching}
    # does adding the remote scaffold change anything beyond DDB1+degron+E2?
    deltas = [abs(r["ladder"]["full_assembly"]["mode1_overlap"]
                  - r["ladder"]["DDB1+degron+E2"]["mode1_overlap"]) for r in closed]
    # DDB1-only against DDB1+CUL4A+RBX1+NEDD8 is the direct test of remoteness; the ladder's
    # full step also adds degron/E2, so compare against the step that has them
    open_delta = [abs(r["ladder"]["DDB1+degron"]["mode1_overlap"]
                      - r["ladder"]["monomer"]["mode1_overlap"]) for r in open_]
    return dict(
        cullin_scaffold_makes_no_direct_contact=dict(
            subunits=list(remote),
            min_distance_range=[round(min(remote_min), 1), round(max(remote_min), 1)],
            max_pairs_within_cutoff=int(max(remote_pairs)),
            cutoff=CUTOFF,
            conclusion=("CUL4A, RBX1 and NEDD8 make no C-alpha pair within the 15 A cutoff in any "
                        "of the four complete assemblies (nearest approach 22.9-28.4 A), so they "
                        "add no spring directly to CRBN. They still enter its quasi-static "
                        "condensed Hessian "
                        "indirectly: H_eff = A - B D^+ B^T has B zero on the cullin rows but D is "
                        "not, and the scaffold stiffens DDB1 and the degron. Including them on top "
                        "of DDB1+degron+E2 changes the closed-endpoint mode-1 overlap by up to "
                        "0.193 and the mode-1 Schur stiffness eigenvalue by 1.32-1.45x. This is "
                        "an indirect static stiffness effect, not a preserved physical assembly "
                        "frequency.")),
        subunits_that_do_contact_crbn=dict(
            min_distance_range={k: [round(min(v), 1), round(max(v), 1)]
                                for k, v in touch_min.items() if v},
            note=("The degron and E2 touch CRBN and were absent from every network in the "
                  "manuscript; DDB1 was the only partner previously tested.")),
        open_reference_claim_unaffected=dict(
            structures=[r["pdb"] for r in open_],
            monomer_mode1=[round(r["ladder"]["monomer"]["mode1_overlap"], 3) for r in open_],
            with_ddb1_and_degron=[round(r["ladder"]["DDB1+degron"]["mode1_overlap"], 3)
                                  for r in open_],
            max_change=round(max(open_delta), 3),
            all_rank1=all(r["ladder"][t]["best_rank"] == 1 for r in open_
                          for t in ("monomer", "DDB1", "degron_only", "DDB1+degron")),
            stiffening_range=[round(min(r["ladder"]["DDB1+degron"]["stiffening_factor"]
                                        for r in open_), 2),
                              round(max(r["ladder"]["DDB1+degron"]["stiffening_factor"]
                                        for r in open_), 2)],
            conclusion=("On 6H0F and 7U8F, the only open depositions carrying a degron, including "
                        "it leaves mode-1 overlap and rank unchanged (largest change 0.010) with "
                        "stiffening 1.10-1.12x. The degron binds the TBD surface, not across the "
                        "hinge, so it does not restrain the interdomain coordinate.")),
        assembly_partially_rescues_the_closed_endpoint=dict(
            structures=[r["pdb"] for r in closed],
            monomer_mode1=[round(r["ladder"]["monomer"]["mode1_overlap"], 3) for r in closed],
            with_ddb1_and_degron=[round(r["ladder"]["DDB1+degron"]["mode1_overlap"], 3)
                                  for r in closed],
            full_assembly=[round(r["ladder"]["full_assembly"]["mode1_overlap"], 3)
                           for r in closed],
            remote_scaffold_delta_max=round(max(deltas), 3),
            note=("Closed structures are poor ANM references for this transition on their own "
                  "(mode-1 overlap 0.011-0.097), consistent with the endpoint asymmetry already "
                  "reported. Adding the contacting partners raises it substantially. Report this "
                  "as a separate observation about assembly context; it does not bear on the "
                  "open-reference claim.")),
        caveats=[
            "All four complete assemblies are closed, glue-bound IKZF3 ternaries, so they cannot "
            "test an open-reference prediction directly; the open-structure test uses 6H0F and "
            "7U8F, the only open depositions with a degron.",
            "Ubiquitin and the E2 active site are positioned for transfer in these structures, a "
            "specific catalytic pose rather than a general assembly state.",
            "Condensation is quasi-static, so it reports how partners stiffen CRBN rather than "
            "how they couple dynamically.",
            "Interface springs use the same force constant as intramolecular contacts.",
            "A cullin arm that is remote in these deposited poses is not necessarily remote in "
            "every state of the ligase; cullin rotation is documented for CRL4 and no open-state "
            "full assembly exists to test.",
        ],
    )


def main() -> int:
    global CACHE_WRITES_ENABLED
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    CACHE_WRITES_ENABLED = not args.verify

    confs, labels, window, axis = load_reference()

    closed = []
    for pdb_id, roles in ASSEMBLIES.items():
        rec = analyse(pdb_id, roles, CLOSED_LADDER, confs, labels, window, axis)
        closed.append(rec)
        c = rec["contacts"]
        print(f"{pdb_id} ({roles['glue']}): CUL4A {c['cul4a']['min_distance']:.1f} A / "
              f"{c['cul4a']['n_pairs_within_cutoff']} pairs, RBX1 "
              f"{c['rbx1']['min_distance']:.1f} A, degron "
              f"{c['degron']['min_distance']:.1f} A / {c['degron']['n_pairs_within_cutoff']} pairs")
        for tag, _ in CLOSED_LADDER:
            s = rec["ladder"][tag]
            print(f"    {tag:18s} m1 {s['mode1_overlap']:.3f} rank {s['best_rank']} "
                  f"stiffening {s['stiffening_factor']:.2f}x")

    open_ = []
    for pdb_id, roles in OPEN_TERNARIES.items():
        rec = analyse(pdb_id, roles, OPEN_LADDER, confs, labels, window, axis)
        open_.append(rec)
        print(f"{pdb_id} (OPEN, {roles['glue']}): degron "
              f"{rec['contacts']['degron']['min_distance']:.1f} A / "
              f"{rec['contacts']['degron']['n_pairs_within_cutoff']} pairs")
        for tag, _ in OPEN_LADDER:
            s = rec["ladder"][tag]
            print(f"    {tag:18s} m1 {s['mode1_overlap']:.3f} rank {s['best_rank']} "
                  f"stiffening {s['stiffening_factor']:.2f}x")

    summary = summarise(closed, open_)
    payload = dict(
        provenance=dict(script="scripts/crl4_assembly_modes.py", cutoff=CUTOFF,
                        n_modes=N_MODES, assemblies=ASSEMBLIES, open_ternaries=OPEN_TERNARIES),
        summary=summary, closed_assemblies=closed, open_ternaries=open_)
    if not args.verify:
        with open(OUT, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"wrote {os.path.relpath(OUT, ROOT)}")

    rem = summary["cullin_scaffold_makes_no_direct_contact"]
    oc = summary["open_reference_claim_unaffected"]
    rescue = summary["assembly_partially_rescues_the_closed_endpoint"]
    print(f"\nsummary: cullin scaffold {rem['min_distance_range']} A from CRBN with "
          f"{rem['max_pairs_within_cutoff']} pairs inside {CUTOFF} A; on the open ternaries the "
          f"degron changes mode-1 overlap by at most {oc['max_change']} and every step stays "
          f"rank 1 (stiffening {oc['stiffening_range']}x); on the closed assemblies the "
          f"non-contacting scaffold still moves mode-1 overlap by up to "
          f"{rescue['remote_scaffold_delta_max']}")

    if args.verify:
        assert rem["max_pairs_within_cutoff"] == 0, rem
        assert rem["min_distance_range"][0] > CUTOFF, rem
        assert oc["all_rank1"], oc
        assert oc["max_change"] < 0.05, oc
        assert oc["stiffening_range"][1] < 1.3, oc
        # The scaffold contributes no direct spring but is NOT inert: it stiffens DDB1 and the
        # degron, and that propagates through D^+ into CRBN's static condensed Hessian. This bound is
        # the measured effect, not a claim that the effect is zero -- an earlier version of this
        # script asserted the spectrum was unchanged "to all reported digits" without ever
        # running that comparison, and the number below is what the comparison actually gives.
        delta = rescue["remote_scaffold_delta_max"]
        assert 0.15 < delta < 0.25, delta
        assert len(summary["caveats"]) >= 5
        print(f"verify OK: CUL4A/RBX1/NEDD8 make zero direct contacts with CRBN inside the "
              f"cutoff, yet still shift the closed-endpoint mode-1 overlap by {delta} through "
              f"DDB1; including the degron on the open ternaries leaves the overlap and rank "
              f"unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
