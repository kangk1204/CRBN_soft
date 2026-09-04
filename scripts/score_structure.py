#!/usr/bin/env python3
"""Place a human CRBN structure on the open-to-closed PCA coordinate.

The coordinate is defined from a fixed 70-structure reference set: the first
principal component over a common 269-Calpha window, rescaled so that the closed
reference mean is 0 and the open reference mean is 1.

The projection uses the stored PCA frame. Nothing is refitted: the mean
coordinates, PC1 vector, residue window, and two rescaling constants are read
from data/crbn_pca.npz and data/crbn_residue_window.csv.

Usage
-----
    python scripts/score_structure.py 9SFM
    python scripts/score_structure.py path/to/structure.cif --chain B
    python scripts/score_structure.py --self-test

Chain selection first uses the exact Q96SW2 accession mapping for known PDB
identifiers. For local files or identifiers without a mapping, exactly one chain
must resolve the full residue window, or --chain must be supplied. A structure
must resolve every one of the 269 window positions in the selected chain.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdb_id import validate_pdb_id  # noqa: E402
from reproduce_tensor import crbn_chains, fetch_cif, parse_ca  # noqa: E402
from softmode_lib import kabsch_apply  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PCA_INPUT = DATA / "crbn_pca.npz"
WINDOW_INPUT = DATA / "crbn_residue_window.csv"
CACHE = DATA / "_cif_cache"
ACCESSION = "Q96SW2"

# Reference-set thresholds: every curated closed structure scored <= 0.25 and
# every curated open structure scored >= 0.95, with no deposited structure in between.
CLOSED_MAX = 0.25
OPEN_MIN = 0.95


def load_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return the stored frame: window, mean coordinates, PC1, and the 0/1 scale."""
    with WINDOW_INPUT.open(encoding="utf-8", newline="") as handle:
        window = np.asarray(
            [int(row["author_resnum"]) for row in csv.DictReader(handle)], dtype=int
        )
    with np.load(PCA_INPUT, allow_pickle=False) as pca:
        mean = np.asarray(pca["mean"], dtype=float)
        pc1 = np.asarray(pca["pcs"], dtype=float)[:, 0]
        scores = np.asarray(pca["pc1_scores"], dtype=float)
        open_mask = np.asarray(pca["open_mask"], dtype=bool)
    if mean.shape != (len(window), 3) or pc1.shape != (3 * len(window),):
        raise ValueError("committed PCA artifact does not match the residue window")
    closed_mean = float(scores[~open_mask].mean())
    open_mean = float(scores[open_mask].mean())
    return window, mean, pc1, closed_mean, open_mean


def window_coordinates(
    chains: dict[str, dict[int, tuple[float, float, float]]],
    window: np.ndarray,
    chain: str | None,
    pdb_id: str | None = None,
) -> tuple[str, np.ndarray]:
    """Select the CRBN chain that resolves the whole window and return its Calpha block.

    Author residue numbering is not unique across chains: in several CRBN
    complexes a partner chain also carries numbers 77-424 and would resolve the
    window while describing a different protein entirely. Scoring that chain
    returns a confident, meaningless number. Candidate chains are therefore
    restricted to entities mapped to the exact CRBN accession whenever the
    identifier is known. Without a mapping, the fallback accepts exactly one
    chain that resolves the full window; otherwise an explicit --chain is required.
    """
    needed = set(int(value) for value in window)
    if chain is not None:
        candidates = [chain]
        if chain not in chains:
            raise SystemExit(f"chain {chain!r} is absent; present chains: {sorted(chains)}")
    else:
        mapped = crbn_chains(pdb_id) if pdb_id else None
        if mapped:
            candidates = [name for name in mapped if name in chains]
            if not candidates:
                raise SystemExit(
                    f"{pdb_id}: chains mapped to {ACCESSION} ({', '.join(mapped)}) are absent "
                    f"from the coordinate records ({', '.join(sorted(chains))})"
                )
        else:
            resolving = [name for name in sorted(chains) if needed <= set(chains[name])]
            if len(resolving) != 1:
                raise SystemExit(
                    f"cannot identify the CRBN chain without the {ACCESSION} mapping: "
                    f"{len(resolving)} chains resolve the window "
                    f"({', '.join(resolving) or 'none'}). Pass --chain explicitly."
                )
            candidates = resolving

    complete = [name for name in candidates if needed <= set(chains[name])]
    if not complete:
        best = max(
            candidates, key=lambda name: len(needed & set(chains[name],)), default=None
        )
        resolved = len(needed & set(chains[best])) if best else 0
        raise SystemExit(
            f"no candidate chain resolves all {len(needed)} window positions "
            f"(best: chain {best} with {resolved}). The coordinate is defined only on "
            "the complete window, so this structure cannot be scored."
        )
    selected = complete[0]
    coords = np.asarray(
        [chains[selected][int(value)] for value in window], dtype=float
    )
    return selected, coords


def closure_score(coords: np.ndarray) -> tuple[float, float]:
    """Return the raw PC1 score and the rescaled closure coordinate."""
    window, mean, pc1, closed_mean, open_mean = load_reference()
    if coords.shape != mean.shape:
        raise ValueError(f"expected {mean.shape} coordinates, received {coords.shape}")
    # Superpose onto the stored mean, exactly as the reference ensemble was aligned.
    aligned = kabsch_apply(coords, mean)
    raw = float(((aligned - mean).ravel() @ pc1) / np.sqrt(len(window)))
    return raw, (raw - closed_mean) / (open_mean - closed_mean)


def classify(coordinate: float) -> str:
    if coordinate <= CLOSED_MAX:
        return "closed"
    if coordinate >= OPEN_MIN:
        return "open"
    return "intermediate (no curated deposited structure occupies this band)"


def read_structure(target: str) -> str:
    path = Path(target)
    if path.is_file():
        blob = path.read_bytes()
        if path.suffix == ".gz":
            return gzip.decompress(blob).decode("utf-8", errors="replace")
        return blob.decode("utf-8", errors="replace")
    identifier = validate_pdb_id(target)
    cached = CACHE / f"{identifier}.cif.gz"
    if cached.is_file():
        return gzip.decompress(cached.read_bytes()).decode("utf-8", errors="replace")
    return fetch_cif(identifier)


def self_test() -> int:
    """Re-score census members and require the committed coordinate to come back."""
    window, mean, pc1, closed_mean, open_mean = load_reference()
    with np.load(DATA / "crbn_ensemble.ens.npz", allow_pickle=False) as ensemble:
        confs = np.asarray(ensemble["_confs"], dtype=float)
        labels = [str(value) for value in ensemble["_labels"]]
    with np.load(PCA_INPUT, allow_pickle=False) as pca:
        recorded = np.asarray(pca["pc1_scores"], dtype=float)

    worst = 0.0
    for index in range(len(labels)):
        raw, _ = closure_score(confs[index])
        worst = max(worst, abs(raw - recorded[index]))
    print(f"self-test: {len(labels)} census members rescored, max |PC1 difference| = {worst:.2e}")
    if worst > 1e-8:
        print("FAIL: rescoring does not reproduce the committed PC1 scores")
        return 1

    for label, expected in (("8CVP", "open"), ("5FQD", "closed"), ("9DJT", "closed")):
        index = labels.index(label)
        _, coordinate = closure_score(confs[index])
        state = classify(coordinate)
        print(f"  {label}: closure coordinate {coordinate:.3f} -> {state}")
        if not state.startswith(expected):
            print(f"FAIL: {label} classified as {state}, expected {expected}")
            return 1
    print("self-test OK")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", nargs="?", help="PDB identifier, or a path to an mmCIF (.cif/.cif.gz) file"
    )
    parser.add_argument("--chain", help="author chain identifier for the CRBN copy")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable record")
    parser.add_argument(
        "--self-test", action="store_true", help="rescore the census and verify the frame"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.target:
        raise SystemExit("provide a PDB identifier or an mmCIF path, or use --self-test")

    window, *_ = load_reference()
    chains = parse_ca(read_structure(args.target))
    if not chains:
        raise SystemExit(f"no C-alpha records parsed from {args.target}")
    pdb_id = None if Path(args.target).is_file() else args.target.upper()
    chain, coords = window_coordinates(chains, window, args.chain, pdb_id)
    raw, coordinate = closure_score(coords)
    state = classify(coordinate)

    if args.json:
        print(
            json.dumps(
                {
                    "target": args.target,
                    "chain": chain,
                    "n_window_positions": int(len(window)),
                    "pc1_score": raw,
                    "closure_coordinate": coordinate,
                    "state": state,
                    "closed_band_max": CLOSED_MAX,
                    "open_band_min": OPEN_MIN,
                },
                indent=2,
            )
        )
    else:
        print(f"{args.target} chain {chain}: {len(window)} window positions resolved")
        print(f"  PC1 score            {raw:8.3f}")
        print(f"  closure coordinate   {coordinate:8.3f}   (0 = closed mean, 1 = open mean)")
        print(f"  assignment           {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
