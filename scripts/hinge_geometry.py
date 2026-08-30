#!/usr/bin/env python3
"""Kinematic hinge geometry of the deposited CRBN open-to-closed displacement.

The scalar sign changes of a Gaussian network model identify correlation nodes,
not a three-dimensional rotation axis.  This analysis instead derives the
finite rigid transform of the thalidomide-binding domain (TBD) after anchoring
the N-terminal domain plus helical bundle (NTD+HB).  The corresponding screw
axis gives a geometric, residue-level description of the interdomain hinge.

Inputs   data/crbn_ensemble.ens.npz, data/pca_diffvec.npz,
         data/crbn_residue_window.csv
Output   data/hinge_geometry.json
Usage    python scripts/hinge_geometry.py [--verify]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis_contracts import assert_tree_close, atomic_write_json, validate_ensemble_diff
from softmode_lib import kabsch


DATA = Path("data")
HB_TBD_BOUNDARY = 317
BOUNDARY_RESIDUES = tuple(range(315, 321))
AXIS_PROXIMITY_A = 2.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="compare with the committed JSON")
    return parser.parse_args(argv)


def load_residue_window(path: Path = DATA / "crbn_residue_window.csv") -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        residues = np.asarray(
            [int(row["author_resnum"]) for row in csv.DictReader(handle)], dtype=int
        )
    if residues.ndim != 1 or residues.size < 2:
        raise ValueError("residue window must be a one-dimensional nonempty sequence")
    if len(np.unique(residues)) != len(residues) or np.any(np.diff(residues) <= 0):
        raise ValueError("residue window must contain unique, increasing author residue numbers")
    return residues


def screw_axis(rotation: np.ndarray, translation: np.ndarray) -> dict[str, np.ndarray | float]:
    """Return the unique perpendicular-origin representation of a rigid screw axis.

    The transform convention is ``x_closed = R @ x_open + t``.  The returned
    axis point is constrained to be perpendicular to the unit axis, removing
    the otherwise arbitrary translation of the point along the same line.
    """

    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("rigid transform must contain a 3x3 rotation and 3-vector")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("rigid transform contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
        raise ValueError("rotation matrix is not orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("rotation matrix must be proper")

    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    if angle < 1e-6:
        raise ValueError("near-zero rotation does not define a stable screw axis")

    eigenvalues, eigenvectors = np.linalg.eig(rotation)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    axis = np.real(eigenvectors[:, index])
    axis /= np.linalg.norm(axis)
    # Fix the arbitrary eigenvector sign for byte-stable output.
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis

    rise = float(axis @ translation)
    perpendicular_translation = translation - rise * axis
    design = np.vstack((np.eye(3) - rotation, axis))
    target = np.concatenate((perpendicular_translation, [0.0]))
    point = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = np.linalg.norm((np.eye(3) - rotation) @ point - perpendicular_translation)
    if residual > 1e-8 or abs(float(axis @ point)) > 1e-8:
        raise ValueError("failed to determine a stable screw-axis point")
    return {
        "angle_deg": angle,
        "axis_unit_vector": axis,
        "axis_point_A": point,
        "rise_A": rise,
    }


def distance_to_axis(coordinates: np.ndarray, point: np.ndarray, axis: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    point = np.asarray(point, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape n x 3")
    if point.shape != (3,) or axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1.0):
        raise ValueError("axis must contain a point and a unit direction")
    return np.linalg.norm(np.cross(coordinates - point, axis), axis=1)


def compute_geometry(
    conformers: np.ndarray, open_mask: np.ndarray, residues: np.ndarray
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    conformers = np.asarray(conformers, dtype=float)
    open_mask = np.asarray(open_mask, dtype=bool)
    residues = np.asarray(residues, dtype=int)
    if conformers.shape != (open_mask.size, residues.size, 3):
        raise ValueError("ensemble, state mask and residue window have incompatible shapes")
    if not open_mask.any() or open_mask.all():
        raise ValueError("both open and closed conformers are required")

    open_mean = conformers[open_mask].mean(axis=0)
    closed_mean = conformers[~open_mask].mean(axis=0)
    anchor = residues <= HB_TBD_BOUNDARY
    moving = residues > HB_TBD_BOUNDARY
    if anchor.sum() < 3 or moving.sum() < 3:
        raise ValueError("both anchored and moving domains require at least three residues")

    anchor_rotation, anchor_translation, anchor_rmsd = kabsch(
        open_mean[anchor], closed_mean[anchor]
    )
    anchored_open = (anchor_rotation @ open_mean.T).T + anchor_translation
    tbd_rotation, tbd_translation, tbd_rmsd = kabsch(
        anchored_open[moving], closed_mean[moving]
    )
    screw = screw_axis(tbd_rotation, tbd_translation)
    distances = distance_to_axis(
        anchored_open,
        np.asarray(screw["axis_point_A"]),
        np.asarray(screw["axis_unit_vector"]),
    )
    endpoint_displacements = np.linalg.norm(anchored_open - closed_mean, axis=1)
    lookup = {int(residue): index for index, residue in enumerate(residues)}
    missing_boundary = set(BOUNDARY_RESIDUES) - set(lookup)
    if missing_boundary:
        raise ValueError(f"analysis window is missing boundary residues {sorted(missing_boundary)}")

    boundary_distances = {
        str(residue): float(distances[lookup[residue]]) for residue in BOUNDARY_RESIDUES
    }
    boundary_displacements = {
        str(residue): float(endpoint_displacements[lookup[residue]])
        for residue in BOUNDARY_RESIDUES
    }
    proximal_boundary = [
        residue
        for residue in BOUNDARY_RESIDUES
        if distances[lookup[residue]] <= AXIS_PROXIMITY_A
    ]
    out: dict[str, object] = {
        "definition": (
            "Mean-open CRBN was Kabsch-aligned to mean-closed CRBN on residues <=317; "
            "the finite Kabsch transform of residues >=318 then defined the TBD screw axis."
        ),
        "n_open": int(open_mask.sum()),
        "n_closed": int((~open_mask).sum()),
        "anchor_residue_range": [int(residues[anchor].min()), HB_TBD_BOUNDARY],
        "moving_residue_range": [HB_TBD_BOUNDARY + 1, int(residues[moving].max())],
        "anchor_kabsch_rmsd_A": float(anchor_rmsd),
        "tbd_rigid_fit_rmsd_A": float(tbd_rmsd),
        "rotation_angle_deg": float(screw["angle_deg"]),
        "screw_rise_A": float(screw["rise_A"]),
        "axis_unit_vector": [float(value) for value in np.asarray(screw["axis_unit_vector"])],
        "axis_point_A": [float(value) for value in np.asarray(screw["axis_point_A"])],
        "axis_proximity_threshold_A": AXIS_PROXIMITY_A,
        "boundary_axis_distance_A": boundary_distances,
        "boundary_endpoint_displacement_A": boundary_displacements,
        "axis_proximal_boundary_residues": proximal_boundary,
        "diagnostic_residues": {
            str(residue): {
                "axis_distance_A": float(distances[lookup[residue]]),
                "endpoint_displacement_A": float(endpoint_displacements[lookup[residue]]),
            }
            for residue in (273, 289, 315)
        },
    }
    return out, distances, endpoint_displacements


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensemble = np.load(DATA / "crbn_ensemble.ens.npz", allow_pickle=False)
    difference = np.load(DATA / "pca_diffvec.npz", allow_pickle=False)
    conformers, _labels, open_mask, _axis = validate_ensemble_diff(ensemble, difference)
    residues = load_residue_window()
    if conformers.shape[1] != residues.size:
        raise ValueError("residue window does not match ensemble coordinates")
    out, _distances, _displacements = compute_geometry(conformers, open_mask, residues)
    output = DATA / "hinge_geometry.json"

    if args.verify:
        committed = json.loads(output.read_text(encoding="utf-8"))
        assert_tree_close(out, committed)
    else:
        atomic_write_json(output, out)

    print(
        f"TBD screw axis: {out['rotation_angle_deg']:.3f} deg; "
        f"boundary residues within {AXIS_PROXIMITY_A:.1f} A: "
        f"{out['axis_proximal_boundary_residues']}"
    )
    diagnostic = out["diagnostic_residues"]
    print(
        "legacy GNM sites: "
        + ", ".join(
            f"{residue} axis {values['axis_distance_A']:.2f} A / "
            f"endpoint {values['endpoint_displacement_A']:.2f} A"
            for residue, values in diagnostic.items()
        )
    )
    if args.verify:
        assert abs(float(out["rotation_angle_deg"]) - 82.457) < 0.01
        assert out["axis_proximal_boundary_residues"] == [316, 317, 318, 319, 320]
        assert float(diagnostic["273"]["axis_distance_A"]) > 20.0
        assert float(diagnostic["289"]["axis_distance_A"]) > 10.0
        print("verify OK: committed kinematic hinge geometry matches recomputation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
