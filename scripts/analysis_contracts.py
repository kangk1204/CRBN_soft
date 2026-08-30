"""Shared fail-closed contracts for deterministic scientific artifacts."""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def validate_ensemble_diff(
    ensemble: np.lib.npyio.NpzFile,
    difference: np.lib.npyio.NpzFile,
    *,
    expected_open: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return coordinates, labels, open mask, and unit difference axis.

    The mask and difference vector are meaningful only in the exact conformer and
    residue ordering of the ensemble. Shape compatibility alone is insufficient.
    """

    required_ensemble = {"_confs", "_labels"}
    required_difference = {"labels", "open_mask", "diff_vec"}
    missing_ensemble = required_ensemble - set(ensemble.files)
    missing_difference = required_difference - set(difference.files)
    if missing_ensemble or missing_difference:
        raise ValueError(
            "incomplete ensemble/difference artifacts: "
            f"ensemble_missing={sorted(missing_ensemble)}, "
            f"difference_missing={sorted(missing_difference)}"
        )

    conformers = np.asarray(ensemble["_confs"])
    ensemble_labels = np.asarray(ensemble["_labels"])
    difference_labels = np.asarray(difference["labels"])
    if conformers.ndim != 3 or conformers.shape[2] != 3:
        raise ValueError(
            f"ensemble coordinates must be n x residues x 3, found {conformers.shape}"
        )
    if ensemble_labels.shape != (conformers.shape[0],):
        raise ValueError(
            f"ensemble labels must have shape ({conformers.shape[0]},), "
            f"found {ensemble_labels.shape}"
        )
    labels = np.asarray([str(value) for value in ensemble_labels])
    if any(not label.strip() for label in labels) or len(set(labels.tolist())) != len(labels):
        raise ValueError("ensemble labels must be nonempty and unique")
    if difference_labels.shape != ensemble_labels.shape or not np.array_equal(
        difference_labels, ensemble_labels
    ):
        raise ValueError("difference-artifact labels do not exactly match ensemble label order")
    if not np.isfinite(conformers).all():
        raise ValueError("ensemble coordinates contain non-finite values")

    raw_mask = np.asarray(difference["open_mask"])
    if raw_mask.shape != (conformers.shape[0],) or raw_mask.dtype.kind != "b":
        raise ValueError(
            f"open_mask must be boolean with shape ({conformers.shape[0]},), "
            f"found dtype={raw_mask.dtype}, shape={raw_mask.shape}"
        )
    open_mask = raw_mask.astype(bool, copy=False)
    if int(open_mask.sum()) != expected_open:
        raise ValueError(f"expected {expected_open} open conformers, found {int(open_mask.sum())}")

    axis = np.asarray(difference["diff_vec"], dtype=float)
    expected_axis_shape = (conformers.shape[1] * 3,)
    if axis.shape != expected_axis_shape or not np.isfinite(axis).all():
        raise ValueError(
            f"diff_vec must be finite with shape {expected_axis_shape}, found {axis.shape}"
        )
    norm = float(np.linalg.norm(axis))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"diff_vec must be unit length, found norm={norm:.16g}")
    flattened = conformers.reshape(conformers.shape[0], -1)
    closed_mask = ~open_mask
    if not closed_mask.any():
        raise ValueError("at least one closed conformer is required to define diff_vec")
    recomputed_axis = flattened[open_mask].mean(axis=0) - flattened[closed_mask].mean(axis=0)
    recomputed_norm = float(np.linalg.norm(recomputed_axis))
    if not math.isfinite(recomputed_norm) or recomputed_norm <= 1e-12:
        raise ValueError("open-minus-closed mean difference cannot define a stable axis")
    recomputed_axis /= recomputed_norm
    if not np.allclose(axis, recomputed_axis, rtol=0.0, atol=1e-12):
        raise ValueError("diff_vec does not match the normalized open-minus-closed mean")
    return conformers, labels, open_mask, axis


def assert_tree_close(
    recomputed: Any,
    committed: Any,
    *,
    path: str = "root",
    float_tolerance: float = 1e-10,
) -> None:
    """Require identical schema/cardinality and tolerance-aware finite numbers."""

    if isinstance(recomputed, dict):
        if not isinstance(committed, dict):
            raise AssertionError(f"{path}: expected object, found {type(committed).__name__}")
        if set(recomputed) != set(committed):
            raise AssertionError(
                f"{path}: key mismatch; missing={sorted(set(recomputed) - set(committed))}, "
                f"extra={sorted(set(committed) - set(recomputed))}"
            )
        for key in recomputed:
            assert_tree_close(
                recomputed[key],
                committed[key],
                path=f"{path}.{key}",
                float_tolerance=float_tolerance,
            )
        return
    if isinstance(recomputed, list):
        if not isinstance(committed, list) or len(recomputed) != len(committed):
            got = len(committed) if isinstance(committed, list) else type(committed).__name__
            raise AssertionError(
                f"{path}: list cardinality mismatch; expected {len(recomputed)}, got {got}"
            )
        for index, (left, right) in enumerate(zip(recomputed, committed)):
            assert_tree_close(
                left,
                right,
                path=f"{path}[{index}]",
                float_tolerance=float_tolerance,
            )
        return
    if isinstance(recomputed, bool) or recomputed is None or isinstance(recomputed, str):
        if type(committed) is not type(recomputed) or committed != recomputed:
            raise AssertionError(f"{path}: recomputed {recomputed!r} != committed {committed!r}")
        return
    if isinstance(recomputed, (int, np.integer)):
        if (
            isinstance(committed, bool)
            or not isinstance(committed, (int, np.integer))
            or int(committed) != int(recomputed)
        ):
            raise AssertionError(f"{path}: recomputed {recomputed!r} != committed {committed!r}")
        return
    if isinstance(recomputed, (float, np.floating)):
        if isinstance(committed, bool) or not isinstance(
            committed, (int, float, np.integer, np.floating)
        ):
            raise AssertionError(f"{path}: committed value is not numeric: {committed!r}")
        left, right = float(recomputed), float(committed)
        if not math.isfinite(left) or not math.isfinite(right):
            raise AssertionError(f"{path}: non-finite numeric value")
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=float_tolerance):
            raise AssertionError(f"{path}: recomputed {left:.16g} != committed {right:.16g}")
        return
    if recomputed != committed:
        raise AssertionError(f"{path}: recomputed {recomputed!r} != committed {committed!r}")


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text artifact only after its complete payload is staged."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        target_mode = 0o666 & ~current_umask
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, target_mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, *, sort_keys: bool = False) -> None:
    """Serialize strict JSON (no NaN/Infinity) and atomically replace ``path``."""

    payload = json.dumps(value, indent=1, sort_keys=sort_keys, allow_nan=False) + "\n"
    atomic_write_text(path, payload)
