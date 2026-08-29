#!/usr/bin/env python3
"""Recompute the post-freeze CRBN sensitivity from raw mmCIF coordinates.

The primary 2026-07-20 ensemble remains fixed. This script verifies the separate
2026-08-28 audit by extracting the recorded primary CRBN chain for every later
entity, re-adjudicating 269-residue coverage, and adding only same-rule-eligible
entries to the frozen tensor before recomputing PCA and open-reference ANM.

Usage:
  python scripts/post_freeze_sensitivity.py --verify [--no-network]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reproduce_modes as M
import reproduce_tensor as R


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "data" / "post_freeze_rcsb_audit.json"
CACHE = ROOT / "data" / "_cif_cache"


def cif_text(pdb_id: str, *, no_network: bool) -> str:
    cached = CACHE / f"{pdb_id.upper()}.cif.gz"
    if cached.exists():
        with gzip.open(cached, "rt", encoding="utf-8") as fh:
            return fh.read()
    if no_network:
        raise FileNotFoundError(f"{pdb_id}: no cached mmCIF at {cached}")
    with urllib.request.urlopen(
        f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz", timeout=120
    ) as response:
        return gzip.decompress(response.read()).decode("utf-8")


def reextract(audit: dict, *, no_network: bool) -> tuple[list[np.ndarray], list[str]]:
    """Return eligible raw coordinates after checking every recorded coverage call."""
    eligible_coords: list[np.ndarray] = []
    eligible_ids: list[str] = []
    window = [int(r) for r in R.WINSET]
    for entry in audit["post_freeze_entities"]:
        pdb = entry["pdb_id"]
        chain = entry["primary_chain"]
        ca = R.parse_ca(cif_text(pdb, no_network=no_network))
        if chain not in ca:
            raise RuntimeError(f"{pdb}: recorded primary CRBN chain {chain} is absent")
        missing = [r for r in window if r not in ca[chain]]
        resolved = len(window) - len(missing)
        assert resolved == entry["window_resolved"], (pdb, resolved, entry["window_resolved"])
        assert missing == entry["missing_residues"], (pdb, missing, entry["missing_residues"])
        expected_call = (
            "eligible_post_freeze"
            if resolved == len(window) and float(entry["resolution_A"]) <= 4.0
            else "excluded_post_freeze"
        )
        assert entry["paper_rule_call"] == expected_call, (pdb, expected_call)
        if expected_call == "eligible_post_freeze":
            eligible_coords.append(np.array([ca[chain][r] for r in window], dtype=float))
            eligible_ids.append(pdb)
    return eligible_coords, eligible_ids


def recompute(audit: dict, *, no_network: bool) -> dict:
    new_coords, new_ids = reextract(audit, no_network=no_network)
    ens = np.load(ROOT / "data" / "crbn_ensemble.ens.npz", allow_pickle=False)
    frozen = np.asarray(ens["_confs"], dtype=float)
    labels = [str(label) for label in ens["_labels"]] + new_ids

    # Re-superpose the combined set so the temporal sensitivity follows the same
    # iterative Kabsch convention as the primary tensor.
    combined, _ = R.superpose(np.concatenate([frozen, np.asarray(new_coords)], axis=0))
    _, pcv, _, variance_ratio, scores = M.pca(combined)
    pc1 = pcv[:, 0]
    pc1_scores = scores[:, 0] / np.sqrt(combined.shape[1])
    if abs(pc1_scores.min()) > abs(pc1_scores.max()):
        pc1_scores = -pc1_scores
        pc1 = -pc1
    ordered = np.sort(pc1_scores)[::-1]
    split = int(np.argmax(ordered[:15][:-1] - ordered[:15][1:])) + 1
    threshold = 0.5 * (ordered[split - 1] + ordered[split])
    open_mask = pc1_scores >= threshold

    difference = combined[open_mask].mean(0) - combined[~open_mask].mean(0)
    dvec = difference.reshape(-1).copy()
    dvec /= np.linalg.norm(dvec)
    ref = combined[labels.index("8CVP")]
    _, anm_modes = M.modes_from(M.anm_hessian(ref, M.CUTOFF_ANM), M.N_MODES)
    overlaps = np.abs(anm_modes.T @ dvec)
    pca10 = pcv[:, :10]
    rmsip = float(np.sqrt(np.square(anm_modes[:, :10].T @ pca10).sum() / 10.0))

    return {
        "eligible_entities_added": [next(
            e["pdb_entity"] for e in audit["post_freeze_entities"] if e["pdb_id"] == pdb
        ) for pdb in new_ids],
        "n_conformers": int(combined.shape[0]),
        "n_residues": int(combined.shape[1]),
        "n_open": int(open_mask.sum()),
        "n_closed": int((~open_mask).sum()),
        "open_members": sorted(label for label, is_open in zip(labels, open_mask) if is_open),
        "pc1_variance_fraction": float(variance_ratio[0]),
        "anm_mode1_overlap": float(overlaps[0]),
        "anm_best_overlap": float(overlaps.max()),
        "anm_best_rank": int(np.argmax(overlaps) + 1),
        "rmsip_anm_pca": rmsip,
        # Separation along PC1 in the RMSD-like score convention used by the
        # primary analysis, not the slightly larger full-vector centroid RMSD.
        "open_closed_separation_A": float(
            abs(pc1 @ difference.reshape(-1)) / np.sqrt(combined.shape[1])
        ),
    }


def verify(actual: dict, expected: dict) -> None:
    exact = [
        "eligible_entities_added", "n_conformers", "n_residues", "n_open", "n_closed",
        "open_members", "anm_best_rank",
    ]
    for key in exact:
        assert actual[key] == expected[key], (key, actual[key], expected[key])
    numeric = [
        "pc1_variance_fraction", "anm_mode1_overlap", "anm_best_overlap",
        "rmsip_anm_pca", "open_closed_separation_A",
    ]
    for key in numeric:
        assert abs(actual[key] - expected[key]) < 1e-6, (key, actual[key], expected[key])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-network", action="store_true")
    args = parser.parse_args(argv)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    actual = recompute(audit, no_network=args.no_network)
    if args.verify:
        verify(actual, audit["sensitivity_if_eligible_entries_are_added"])
        print(
            "verify OK: raw post-freeze mmCIF -> "
            f"{actual['n_conformers']}x{actual['n_residues']}, open={actual['n_open']}, "
            f"PC1 {100 * actual['pc1_variance_fraction']:.1f}%, ANM m1 "
            f"{actual['anm_mode1_overlap']:.3f} rank {actual['anm_best_rank']}, "
            f"RMSIP {actual['rmsip_anm_pca']:.3f}"
        )
    else:
        print(json.dumps(actual, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
