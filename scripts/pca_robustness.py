#!/usr/bin/env python3
"""Bootstrap and leave-one-open-out robustness of the ensemble soft mode.

The primary interval is a cluster bootstrap over 38 source-study groups (fixed
seed 42, 2,000 resamples). ``--verify`` recomputes every stored array without
writing.
"""

from __future__ import annotations

import argparse
import os
import stat
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

try:
    from analysis_contracts import validate_ensemble_diff
    from study_groups import load_study_groups
except ModuleNotFoundError:
    from scripts.analysis_contracts import validate_ensemble_diff
    from scripts.study_groups import load_study_groups


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "pca_robust.npz"
NDRAW = 2_000
SEED = 42
IndexSampler = Callable[[np.random.Generator], Sequence[int] | np.ndarray]


def pca_pc1(coordinates: np.ndarray, difference_axis: np.ndarray) -> tuple[float, float]:
    """Return PC1 variance fraction and absolute overlap with a fixed unit axis."""

    coordinates = np.asarray(coordinates, dtype=float)
    difference_axis = np.asarray(difference_axis, dtype=float)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise ValueError(
            f"coordinates must have shape n x residues x 3, found {coordinates.shape}"
        )
    if coordinates.shape[0] < 2:
        raise ValueError("PCA requires at least two conformers")
    expected_axis_shape = (coordinates.shape[1] * 3,)
    if difference_axis.shape != expected_axis_shape:
        raise ValueError(
            f"difference axis must have shape {expected_axis_shape}, "
            f"found {difference_axis.shape}"
        )
    if not np.isfinite(coordinates).all() or not np.isfinite(difference_axis).all():
        raise ValueError("PCA inputs must contain only finite values")
    axis_norm = float(np.linalg.norm(difference_axis))
    if not np.isclose(axis_norm, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"difference axis must be unit length, found norm={axis_norm:.16g}")

    flattened = coordinates.reshape(coordinates.shape[0], -1)
    centered = flattened - flattened.mean(axis=0)
    sample_count = centered.shape[0]
    gram = centered @ centered.T / (sample_count - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive_total = float(eigenvalues[eigenvalues > 0].sum())
    pc1 = centered.T @ eigenvectors[:, order[0]]
    norm = float(np.linalg.norm(pc1))
    if norm < 1e-12 or positive_total <= 0:
        return 0.0, 0.0
    return float(eigenvalues[order[0]] / positive_total), float(
        abs((pc1 / norm) @ difference_axis)
    )


def bootstrap(
    conformers: np.ndarray,
    difference_axis: np.ndarray,
    open_indices: set[int],
    sampler: IndexSampler,
    *,
    draws: int = NDRAW,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run a deterministic bootstrap for an index sampler."""

    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    conformers = np.asarray(conformers)
    if conformers.ndim != 3 or conformers.shape[2] != 3:
        raise ValueError(
            f"conformers must have shape n x residues x 3, found {conformers.shape}"
        )
    if any(
        isinstance(index, bool)
        or not isinstance(index, (int, np.integer))
        or not 0 <= int(index) < conformers.shape[0]
        for index in open_indices
    ):
        raise ValueError("open_indices contains an out-of-range or non-integral index")

    rng = np.random.default_rng(seed)
    variance_fractions: list[float] = []
    overlaps: list[float] = []
    no_open = 0
    for _ in range(draws):
        indices = list(sampler(rng))
        if len(indices) < 3:
            raise ValueError("bootstrap sampler returned fewer than three conformers")
        if any(
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or not 0 <= int(index) < conformers.shape[0]
            for index in indices
        ):
            raise ValueError("bootstrap sampler returned an invalid conformer index")
        normalized_indices = [int(index) for index in indices]
        if not open_indices.intersection(normalized_indices):
            no_open += 1
        variance, overlap = pca_pc1(conformers[normalized_indices], difference_axis)
        variance_fractions.append(100.0 * variance)
        overlaps.append(overlap)
    return np.asarray(variance_fractions), np.asarray(overlaps), no_open


def build_results() -> dict[str, object]:
    """Recompute the complete deterministic robustness payload."""

    with (
        np.load(ROOT / "data" / "crbn_ensemble.ens.npz", allow_pickle=False) as ensemble,
        np.load(ROOT / "data" / "pca_diffvec.npz", allow_pickle=False) as difference,
    ):
        conformers, labels, open_mask, difference_axis = validate_ensemble_diff(
            ensemble, difference
        )

    n_conformers = conformers.shape[0]
    label_list = labels.tolist()
    open_indices = set(np.where(open_mask)[0].tolist())
    variance_full, overlap_full = pca_pc1(conformers, difference_axis)
    print(f"Full ensemble: PC1 var {variance_full * 100:.1f}%  overlap {overlap_full:.3f}")

    study_for = load_study_groups(label_list)
    grouped_indices: dict[str, list[int]] = {}
    for index, pdb_id in enumerate(label_list):
        grouped_indices.setdefault(study_for[pdb_id], []).append(index)
    # Sort by the actual partition, not by arbitrary source-study labels. If a
    # study label is renamed while the same conformers remain grouped together,
    # the seeded cluster bootstrap should remain bit-identical.
    group_members = sorted(
        (sorted(members) for members in grouped_indices.values()),
        key=lambda members: members,
    )
    if not group_members:
        raise ValueError("no source-study groups are available for bootstrap resampling")
    n_groups = len(group_members)
    print(
        f"study groups: {n_groups} over {n_conformers} conformers "
        f"(largest {max(len(value) for value in grouped_indices.values())})"
    )

    entry_variance, entry_overlap, _ = bootstrap(
        conformers,
        difference_axis,
        open_indices,
        lambda rng: rng.integers(0, n_conformers, n_conformers),
    )
    cluster_variance, cluster_overlap, no_open = bootstrap(
        conformers,
        difference_axis,
        open_indices,
        lambda rng: [
            index
            for group_index in rng.choice(n_groups, n_groups)
            for index in group_members[int(group_index)]
        ],
    )

    print(
        f"Entry bootstrap (within-study) var {entry_variance.mean():.0f}% "
        f"[{np.percentile(entry_variance, 2.5):.0f},"
        f"{np.percentile(entry_variance, 97.5):.0f}]  "
        f"overlap {entry_overlap.mean():.3f} "
        f"[{np.percentile(entry_overlap, 2.5):.2f},"
        f"{np.percentile(entry_overlap, 97.5):.2f}]"
    )
    print(
        f"Cluster bootstrap ({n_groups} groups) var {cluster_variance.mean():.0f}% "
        f"[{np.percentile(cluster_variance, 2.5):.0f},"
        f"{np.percentile(cluster_variance, 97.5):.0f}]  "
        f"overlap {cluster_overlap.mean():.3f} "
        f"[{np.percentile(cluster_overlap, 2.5):.2f},"
        f"{np.percentile(cluster_overlap, 97.5):.2f}]"
    )
    print(
        f"  {no_open} of {len(cluster_variance)} cluster resamples "
        f"({no_open / len(cluster_variance) * 100:.1f}%) contain no open structure"
    )

    print("Leave-one-open-out:")
    for index in sorted(open_indices):
        keep = [candidate for candidate in range(n_conformers) if candidate != index]
        variance, overlap = pca_pc1(conformers[keep], difference_axis)
        print(f"  drop {labels[index]}: PC1 var {variance * 100:.1f}%  overlap {overlap:.3f}")

    closed_indices = [index for index in range(n_conformers) if index not in open_indices]
    closed_variance, closed_overlap = pca_pc1(conformers[closed_indices], difference_axis)
    print(
        f"Drop all {len(open_indices)} open: PC1 var {closed_variance * 100:.1f}%  "
        f"overlap {closed_overlap:.3f} (n={len(closed_indices)})"
    )

    return {
        "vfs": cluster_variance,
        "ovs": cluster_overlap,
        "vf0": variance_full,
        "ov0": overlap_full,
        "open_labels": labels[open_mask],
        "vf_closed": closed_variance,
        "vfs_entry": entry_variance,
        "ovs_entry": entry_overlap,
        "n_groups": n_groups,
        "frac_resamples_without_open": no_open / len(cluster_variance),
    }


def verify_exact(results: dict[str, object]) -> None:
    """Require exact schema/dtypes and a portable tight match for float arrays."""

    with np.load(OUTPUT, allow_pickle=False) as committed:
        if set(committed.files) != set(results):
            raise AssertionError(
                f"{OUTPUT}: key mismatch; expected={sorted(results)}, "
                f"found={sorted(committed.files)}"
            )
        for key, recomputed in results.items():
            candidate = np.asarray(recomputed)
            stored = committed[key]
            if candidate.shape != stored.shape:
                raise AssertionError(
                    f"{OUTPUT}: shape mismatch for {key}: "
                    f"recomputed={candidate.shape}, stored={stored.shape}"
                )
            if candidate.dtype != stored.dtype:
                raise AssertionError(
                    f"{OUTPUT}: dtype mismatch for {key}: "
                    f"recomputed={candidate.dtype}, stored={stored.dtype}"
                )
            if candidate.dtype.kind in "fc":
                matches = (
                    np.isfinite(candidate).all()
                    and np.isfinite(stored).all()
                    and np.allclose(candidate, stored, rtol=1e-12, atol=1e-12)
                )
            else:
                matches = np.array_equal(candidate, stored)
            if not matches:
                raise AssertionError(f"{OUTPUT}: exact array mismatch for {key}")


def atomic_save(results: dict[str, object]) -> None:
    """Atomically replace the stored result after writing the full archive."""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_mode = stat.S_IMODE(OUTPUT.stat().st_mode)
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        target_mode = 0o666 & ~current_umask
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=OUTPUT.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        np.savez(temporary, **results)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, target_mode)
        os.replace(temporary, OUTPUT)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="recompute and compare without writing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = build_results()
    if args.verify:
        verify_exact(results)
        print(
            f"verify OK: schema and dtypes match {OUTPUT}; "
            "finite float arrays agree within 1e-12"
        )
    else:
        atomic_save(results)
        print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
