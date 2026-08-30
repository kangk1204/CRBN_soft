#!/usr/bin/env python3
"""Rebuild the 70 x 269 x 3 C-alpha coordinate tensor from raw mmCIF.

For every structure identifier in the matching input bundle, this workflow
downloads or reads cached mmCIF, selects the exact-accession CRBN chain,
extracts the 269-residue analysis window, and iteratively superposes all
conformers. ``--verify`` requires a complete rebuild in the exact input order,
compares every conformer with the stored tensor, and leaves files untouched.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import stat
import sys
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from curation_contracts import chains_for_exact_accession, choose_primary_chain
    from pdb_id import validate_pdb_id
except ModuleNotFoundError:
    from scripts.curation_contracts import chains_for_exact_accession, choose_primary_chain
    from scripts.pdb_id import validate_pdb_id


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WINDOW_PATH = DATA / "crbn_residue_window.csv"
CHAIN_MAP_PATH = DATA / "curation_chain_map.json"
RCSB_META_PATH = DATA / "_rcsb_meta.json"
OUTPUT = DATA / "crbn_ensemble_rebuilt.npz"
CACHE: str | Path = DATA / "_cif_cache"
CACHE_WRITES_ENABLED = True
UNSAFE_ALLOW_AMBIGUOUS_CHAIN = False
EXPECTED_WINDOW_SIZE = 269


def _configure_console() -> None:
    """Use UTF-8 output when the active console supports reconfiguration."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _load_window(path: Path = WINDOW_PATH) -> np.ndarray:
    """Load and validate the ordered author-residue analysis window."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "author_resnum" not in reader.fieldnames:
            raise ValueError(f"{path}: missing author_resnum column")
        values: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            raw = row.get("author_resnum")
            try:
                value = int(raw) if raw is not None else None
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid author_resnum {raw!r}") from exc
            if value is None:
                raise ValueError(f"{path}:{line_number}: missing author_resnum")
            values.append(value)
    if len(values) != EXPECTED_WINDOW_SIZE:
        raise ValueError(
            f"{path}: expected {EXPECTED_WINDOW_SIZE} ordered residues, found {len(values)}"
        )
    if len(set(values)) != len(values) or any(
        right <= left for left, right in zip(values, values[1:])
    ):
        raise ValueError(f"{path}: author_resnum values must be unique and strictly increasing")
    return np.asarray(values, dtype=int)


def _load_optional_json(path: str | Path) -> dict[str, Any]:
    """Load an optional object sidecar; absence is allowed, malformed input is not."""

    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


try:
    WIN = _load_window()
except FileNotFoundError:
    # The public code checkout is importable before its matching input bundle is added.
    WIN = np.asarray([], dtype=int)
WINSET = [int(value) for value in WIN]

# Recorded primary-chain overrides used by the curation workflow. Entity metadata is still
# required and each override is checked against it; the sidecar cannot authorize a partner
# chain. Entries without an override use the lowest auth chain id among exact-Q96SW2 chains.
try:
    with CHAIN_MAP_PATH.open(encoding="utf-8") as handle:
        CHAIN_MAP = json.load(handle)
except FileNotFoundError:
    CHAIN_MAP = {}
if not isinstance(CHAIN_MAP, dict):
    raise ValueError(f"{CHAIN_MAP_PATH}: expected a JSON object")

# Committed RCSB entity metadata: which auth chains carry the CRBN entity. This is what
# makes chain selection safe without the sidecar; see crbn_chains().
try:
    with RCSB_META_PATH.open(encoding="utf-8") as handle:
        RCSB_META = json.load(handle)
except FileNotFoundError:
    RCSB_META = {}
if not isinstance(RCSB_META, dict):
    raise ValueError(f"{RCSB_META_PATH}: expected a JSON object")


def _require_window() -> list[int]:
    """Return the validated window, loading it if it appeared after module import."""

    global WIN, WINSET
    if len(WINSET) != EXPECTED_WINDOW_SIZE:
        WIN = _load_window()
        WINSET = [int(value) for value in WIN]
    return WINSET


def _replacement_mode(path: Path) -> int:
    """Return an existing access mode or the normal creation mode under umask."""

    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        return 0o666 & ~current_umask


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace a binary file after its complete payload is staged."""

    path.parent.mkdir(parents=True, exist_ok=True)
    target_mode = _replacement_mode(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, target_mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _decode_cif_blob(blob: bytes, pdb: str, source: str) -> str:
    """Decode gzip bytes and require a minimal coordinate-bearing mmCIF structure."""

    try:
        text = gzip.decompress(blob).decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise ValueError(f"{pdb}: {source} is not valid UTF-8 gzip mmCIF") from exc
    stripped = text.lstrip()
    required_markers = ("_atom_site.label_atom_id", "_atom_site.auth_asym_id")
    if not stripped.startswith("data_") or any(marker not in text for marker in required_markers):
        raise ValueError(f"{pdb}: {source} lacks a coordinate-bearing mmCIF atom-site loop")
    return text


def fetch_cif(pdb: str) -> str:
    """Return validated mmCIF text from cache or the canonical download endpoint."""

    pdb = validate_pdb_id(pdb)
    cache_path = Path(CACHE) / f"{pdb}.cif.gz"
    if cache_path.exists():
        try:
            return _decode_cif_blob(cache_path.read_bytes(), pdb, "cached payload")
        except (OSError, ValueError):
            # Generation replaces a corrupt cache entry. Verification reads a clean
            # network copy in memory and never mutates the cache.
            pass
    with urllib.request.urlopen(
        f"https://files.rcsb.org/download/{pdb}.cif.gz", timeout=120
    ) as handle:
        blob = handle.read()
    text = _decode_cif_blob(blob, pdb, "downloaded payload")
    if CACHE_WRITES_ENABLED:
        _atomic_write_bytes(cache_path, blob)
    return text


def parse_ca(cif: str) -> dict[str, dict[int, tuple[float, float, float]]]:
    """Return author-chain C-alpha coordinates from the scalar atom-site loop."""

    if not isinstance(cif, str) or not cif.strip():
        raise ValueError("mmCIF input must be nonempty text")
    lines = cif.splitlines()
    coordinates: dict[str, dict[int, tuple[float, float, float]]] = {}
    index = 0
    while index < len(lines):
        if lines[index].strip() == "loop_":
            cursor = index + 1
            headers: list[str] = []
            while cursor < len(lines) and lines[cursor].lstrip().startswith("_atom_site."):
                headers.append(lines[cursor].strip())
                cursor += 1
            if headers:
                columns = {header.split(".", 1)[1]: offset for offset, header in enumerate(headers)}
                required = {
                    "label_atom_id",
                    "auth_asym_id",
                    "auth_seq_id",
                    "Cartn_x",
                    "Cartn_y",
                    "Cartn_z",
                    "group_PDB",
                }
                if required <= set(columns):
                    row_index = cursor
                    while row_index < len(lines):
                        raw_line = lines[row_index]
                        stripped = raw_line.strip()
                        if not stripped or stripped == "loop_" or stripped.startswith("#"):
                            break
                        if raw_line.startswith("_"):
                            break
                        fields = raw_line.split()
                        if len(fields) < len(headers):
                            row_index += 1
                            continue
                        if fields[columns["group_PDB"]] != "ATOM":
                            row_index += 1
                            continue
                        if fields[columns["label_atom_id"]].strip('"') != "CA":
                            row_index += 1
                            continue
                        chain = fields[columns["auth_asym_id"]]
                        try:
                            residue = int(fields[columns["auth_seq_id"]])
                            xyz = (
                                float(fields[columns["Cartn_x"]]),
                                float(fields[columns["Cartn_y"]]),
                                float(fields[columns["Cartn_z"]]),
                            )
                        except ValueError:
                            row_index += 1
                            continue
                        if not np.isfinite(xyz).all():
                            raise ValueError(
                                f"atom-site loop contains non-finite coordinates for {chain}:{residue}"
                            )
                        coordinates.setdefault(chain, {})
                        coordinates[chain].setdefault(residue, xyz)
                        row_index += 1
                    index = row_index
                    continue
        index += 1
    return coordinates


def crbn_chains(pdb: str) -> list[str] | None:
    """Return auth-chain identifiers mapped to the exact Q96SW2 accession."""

    pdb = validate_pdb_id(pdb)
    entry = RCSB_META.get(pdb)
    if not entry:
        return None
    if not isinstance(entry, Mapping):
        raise ValueError(f"{pdb}: entity metadata must be an object")
    return chains_for_exact_accession(entry, "Q96SW2") or None


def best_chain(
    coordinates: Mapping[str, Mapping[int, tuple[float, float, float]]],
    pdb: str | None = None,
) -> tuple[str, int]:
    """Return the best-covered exact-accession chain for sensitivity checks."""

    window = _require_window()
    candidates = crbn_chains(pdb) if pdb else None
    if candidates is None:
        if UNSAFE_ALLOW_AMBIGUOUS_CHAIN:
            pool = sorted(coordinates)
            print(
                f"WARNING: {pdb or '<unknown>'} has no CRBN chain metadata; "
                "using all chains because --unsafe-allow-ambiguous-chain was passed",
                file=sys.stderr,
            )
        else:
            raise RuntimeError(
                f"{pdb or '<unknown>'}: missing CRBN chain metadata; refusing "
                "to rank all chains. Pass --unsafe-allow-ambiguous-chain to use the "
                "historical fail-open fallback."
            )
    else:
        pool = [chain for chain in sorted(coordinates) if chain in candidates]
        if not pool:
            if UNSAFE_ALLOW_AMBIGUOUS_CHAIN:
                pool = sorted(coordinates)
                print(
                    f"WARNING: {pdb or '<unknown>'} metadata lists CRBN chains "
                    f"{candidates}, but none were found in coordinates; using all "
                    "chains because --unsafe-allow-ambiguous-chain was passed",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(
                    f"{pdb or '<unknown>'}: CRBN chain metadata lists {candidates}, "
                    "but none are present in the parsed coordinates; refusing "
                    "ambiguous fallback."
                )
    if not pool:
        raise RuntimeError(f"{pdb or '<unknown>'}: no coordinate chains are available")
    chosen = max(pool, key=lambda chain: sum(residue in coordinates[chain] for residue in window))
    coverage = sum(residue in coordinates[chosen] for residue in window)
    return chosen, coverage


def select_chain(
    coordinates: Mapping[str, Mapping[int, tuple[float, float, float]]], pdb: str
) -> tuple[str, int]:
    """Select the frozen primary chain and enforce exact entity provenance."""

    window = _require_window()
    pdb = validate_pdb_id(pdb)
    candidates = crbn_chains(pdb)
    problems: list[str] = []
    if candidates is None:
        problems.append("missing CRBN chain metadata")

    try:
        chosen = choose_primary_chain(candidates or [], pdb, CHAIN_MAP)
    except ValueError as exc:
        fallback = CHAIN_MAP.get(pdb)
        chosen = str(fallback) if fallback is not None else None
        problems.append(str(exc))
    if chosen is not None and chosen not in coordinates:
        problems.append(f"recorded chain {chosen} is absent from parsed coordinates")

    if problems:
        detail = "; ".join(problems)
        if not UNSAFE_ALLOW_AMBIGUOUS_CHAIN:
            raise RuntimeError(f"{pdb}: {detail}; refusing to bypass chain provenance")
        print(
            f"WARNING: {pdb}: {detail}; using the best-coverage fallback because "
            "--unsafe-allow-ambiguous-chain was passed",
            file=sys.stderr,
        )
        return best_chain(coordinates, pdb)

    if chosen is None:
        raise RuntimeError(f"{pdb}: no CRBN chain could be selected")
    coverage = sum(residue in coordinates[chosen] for residue in window)
    return chosen, coverage


def extract(pdb: str) -> tuple[np.ndarray | None, int]:
    """Extract a complete ordered CRBN coordinate window for one structure."""

    window = _require_window()
    coordinates = parse_ca(fetch_cif(pdb))
    if not coordinates:
        return None, 0
    chain, coverage = select_chain(coordinates, pdb)
    if coverage < len(window):
        return None, coverage
    extracted = np.asarray([coordinates[chain][residue] for residue in window], dtype=float)
    return extracted, coverage


def kabsch(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Center and optimally rotate one coordinate set onto another."""

    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if points.shape != reference.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Kabsch inputs must have one matching n x 3 shape, found "
            f"{points.shape} and {reference.shape}"
        )
    if points.shape[0] < 3 or not np.isfinite(points).all() or not np.isfinite(reference).all():
        raise ValueError("Kabsch inputs require at least three finite coordinate rows")
    centered_points = points - points.mean(axis=0)
    centered_reference = reference - reference.mean(axis=0)
    covariance = centered_points.T @ centered_reference
    left, _, right_transpose = np.linalg.svd(covariance)
    determinant = np.linalg.det(right_transpose.T @ left.T)
    correction = np.diag([1.0, 1.0, -1.0 if determinant < 0 else 1.0])
    rotation = right_transpose.T @ correction @ left.T
    return (rotation @ centered_points.T).T


def superpose(conformers: np.ndarray, iterations: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively align conformers to their running mean."""

    conformers = np.asarray(conformers, dtype=float)
    if conformers.ndim != 3 or conformers.shape[0] == 0 or conformers.shape[2] != 3:
        raise ValueError(f"conformers must have nonempty n x residues x 3 shape: {conformers.shape}")
    if conformers.shape[1] < 3 or not np.isfinite(conformers).all():
        raise ValueError("conformers must contain at least three finite coordinate rows")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")

    reference = conformers[0] - conformers[0].mean(axis=0)
    aligned = conformers.copy()
    for _ in range(iterations):
        aligned = np.asarray([kabsch(conformer, reference) for conformer in conformers])
        new_reference = aligned.mean(axis=0)
        if np.allclose(new_reference, reference, atol=1e-4):
            reference = new_reference
            break
        reference, conformers = new_reference, aligned
    return aligned, reference


def _load_ensemble(path: Path = DATA / "crbn_ensemble.ens.npz") -> tuple[np.ndarray, list[str]]:
    """Load and validate the stored coordinate tensor and ordered identifiers."""

    window = _require_window()
    with np.load(path, allow_pickle=False) as ensemble:
        missing = {"_confs", "_labels"} - set(ensemble.files)
        if missing:
            raise ValueError(f"{path}: missing required arrays {sorted(missing)}")
        conformers = np.asarray(ensemble["_confs"])
        raw_labels = np.asarray(ensemble["_labels"])
    if conformers.ndim != 3 or conformers.shape[2] != 3:
        raise ValueError(f"{path}: coordinates must have n x residues x 3 shape")
    if conformers.shape[1] != len(window) or not np.isfinite(conformers).all():
        raise ValueError(
            f"{path}: coordinates must contain {len(window)} finite residues per conformer"
        )
    if raw_labels.shape != (conformers.shape[0],):
        raise ValueError(f"{path}: labels do not match the conformer count")
    labels = [str(value) for value in raw_labels]
    if any(not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError(f"{path}: labels must be nonempty and unique")
    normalized = [validate_pdb_id(label) for label in labels]
    if normalized != labels:
        raise ValueError(f"{path}: labels must use canonical uppercase PDB identifiers")
    return conformers, labels


def _atomic_save_tensor(
    path: Path, conformers: np.ndarray, labels: Sequence[str], reference: np.ndarray
) -> None:
    """Atomically store the complete rebuilt tensor archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    target_mode = _replacement_mode(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        np.savez(
            temporary,
            confs=np.asarray(conformers, dtype=np.float32),
            labels=np.asarray(labels),
            ref=np.asarray(reference, dtype=np.float32),
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, target_mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_rebuild(
    requested: Sequence[str], rebuilt: Sequence[str], rmsd: np.ndarray
) -> None:
    """Require exact membership, order, finite comparisons, and a full RMSD bound."""

    if list(rebuilt) != list(requested):
        raise AssertionError(
            f"expected exact label order {list(requested)}, rebuilt {list(rebuilt)}"
        )
    rmsd = np.asarray(rmsd, dtype=float)
    if rmsd.shape != (len(requested),):
        raise AssertionError(
            f"expected {len(requested)} per-conformer RMSD values, found {rmsd.shape}"
        )
    if not np.isfinite(rmsd).all():
        raise AssertionError("rebuilt tensor produced non-finite per-conformer RMSD")
    if float(rmsd.max()) >= 0.5:
        raise AssertionError(f"rebuilt tensor diverges: max {float(rmsd.max()):.3f}")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="fully verify without writing")
    parser.add_argument("--limit", type=_positive_int, help="rebuild only the first N entries")
    parser.add_argument(
        "--unsafe-allow-ambiguous-chain",
        action="store_true",
        help="allow the historical all-chain fallback",
    )
    args = parser.parse_args(argv)
    if args.verify and args.limit is not None:
        parser.error("full --verify rejects --limit; verify every stored conformer")
    if args.verify and args.unsafe_allow_ambiguous_chain:
        parser.error("full --verify rejects unsafe ambiguous-chain fallback")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    global CACHE_WRITES_ENABLED, UNSAFE_ALLOW_AMBIGUOUS_CHAIN

    _configure_console()
    args = parse_args(argv)
    verify = bool(args.verify)
    CACHE_WRITES_ENABLED = not verify
    UNSAFE_ALLOW_AMBIGUOUS_CHAIN = bool(args.unsafe_allow_ambiguous_chain)
    stored_conformers, labels = _load_ensemble()
    requested = labels[: args.limit] if args.limit is not None else labels

    extracted: list[np.ndarray] = []
    rebuilt_labels: list[str] = []
    dropped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    window_size = len(_require_window())
    for count, pdb in enumerate(requested, start=1):
        try:
            coordinates, coverage = extract(pdb)
        except Exception as exc:
            failed.append((pdb, f"{type(exc).__name__}: {exc}"))
            continue
        if coordinates is None:
            dropped.append((pdb, f"window {coverage}/{window_size}"))
            continue
        extracted.append(coordinates)
        rebuilt_labels.append(pdb)
        if count % 10 == 0:
            print(f"  ...{count}/{len(requested)}", flush=True)

    print(
        f"extracted {len(rebuilt_labels)} conformers; dropped {len(dropped)}: {dropped[:8]}"
    )
    if failed:
        raise RuntimeError(
            f"aborted: {len(failed)} deposition(s) could not be fetched or parsed: {failed[:8]}"
        )
    if dropped:
        raise RuntimeError(f"input entries no longer satisfy the full window: {dropped}")

    aligned, reference = superpose(np.asarray(extracted))

    stored_index = {label: index for index, label in enumerate(labels)}
    rmsd: list[float] = []
    for index, pdb in enumerate(rebuilt_labels):
        rebuilt = aligned[index]
        stored = stored_conformers[stored_index[pdb]]
        centered_stored = stored - stored.mean(axis=0)
        matched = kabsch(rebuilt, centered_stored)
        rmsd.append(float(np.sqrt(np.square(matched - centered_stored).sum(axis=-1).mean())))
    rmsd_array = np.asarray(rmsd)
    print(
        "rebuilt-vs-stored per-conformer C-alpha RMSD: "
        f"median {np.median(rmsd_array):.3f} A, max {rmsd_array.max():.3f} A, "
        f"n={len(rmsd_array)}"
    )
    _verify_rebuild(requested, rebuilt_labels, rmsd_array)

    if verify:
        print(
            f"verify OK: {len(rebuilt_labels)} conformers rebuilt from raw mmCIF, "
            f"maximum C-alpha RMSD to stored tensor {rmsd_array.max():.3f} A (<0.5)"
        )
    elif args.limit is not None:
        print(
            f"partial rebuild OK: {len(rebuilt_labels)} conformers checked in memory; "
            f"--limit left canonical output {OUTPUT} untouched"
        )
    else:
        _atomic_save_tensor(OUTPUT, aligned, rebuilt_labels, reference)
        print(f"validated rebuilt tensor {aligned.shape} -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
