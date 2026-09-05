#!/usr/bin/env python3
"""Strengthening controls for the CRBN soft-mode manuscript revision.

This script is intentionally offline: it reads committed local artifacts and writes
review-source tables under a caller-selected output directory.  It does not update the
frozen manuscript package or the canonical data files.

Outputs:
  strengthen_controls_summary.json
  endpoint_scores.csv
  control_panel_comparison.csv
  residue_effects.csv
  residue_set_effects.csv
  finite_chord_tangent.csv
  INTERPRETATION.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmode_lib as L  # noqa: E402
import control_panel as CP  # noqa: E402
from analysis_contracts import (  # noqa: E402
    assert_tree_close,
    atomic_write_json,
    atomic_write_text,
    validate_ensemble_diff,
)
from hinge_geometry import compute_geometry, distance_to_axis, screw_axis  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "strengthening" / "controls"

DEFAULT_CONFIG = {
    "cutoff_A": 15.0,
    "n_modes": 20,
    "primary_open": "8CVP",
    "primary_closed": "5FQD",
    "domains": {
        "NTD": [77, 186],
        "HB": [187, 317],
        "TBD": [318, 426],
    },
    "pocket_definitions_common269": {
        "uniprot_ligand_annotations": [378, 380, 386],
        "annotated_plus_W400_F402": [378, 380, 386, 400, 402],
        "5fqd_4.5A_contact_shell_common_window": [377, 378, 379, 380, 386, 400, 402],
    },
    "zinc_residues": [323, 326, 391, 394],
    "seed": 20260905,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="optional JSON config overriding defaults")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="assert offline mode; retained for explicit run provenance",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="recompute and compare with existing JSON/CSV/Markdown outputs without writing",
    )
    return parser.parse_args(argv)


def load_config(path: Path | None) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return config
    user = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(user, dict):
        raise ValueError("config JSON must be an object")
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def read_residue_window(path: Path = DATA / "crbn_residue_window.csv") -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        residues = np.asarray(
            [int(row["author_resnum"]) for row in csv.DictReader(handle)], dtype=int
        )
    if residues.shape != (269,):
        raise ValueError(f"CRBN common C-alpha window must contain 269 residues, found {residues.size}")
    if len(np.unique(residues)) != residues.size or np.any(np.diff(residues) <= 0):
        raise ValueError("CRBN residue window must be unique and strictly increasing")
    return residues


def domain_indices(residues: np.ndarray, domains: dict[str, list[int]]) -> dict[str, np.ndarray]:
    out = {}
    for name, (lo, hi) in domains.items():
        idx = np.where((residues >= int(lo)) & (residues <= int(hi)))[0]
        if idx.size < 3:
            raise ValueError(f"domain {name} maps to fewer than three residues")
        out[name] = idx
    covered = np.concatenate(list(out.values()))
    if len(np.unique(covered)) != len(covered):
        raise ValueError("domain definitions overlap in the common residue window")
    return out


def unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return vector / norm


def anm_modes_subset(coords: np.ndarray, cutoff_A: float, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Lowest nontrivial ANM modes using a subset eigensolve for endpoint scans."""
    hessian = L.anm_hessian(coords, cutoff_A)
    dim = hessian.shape[0]
    upper = min(dim - 1, n_modes + 24)
    while True:
        eigenvalues, eigenvectors = linalg.eigh(
            hessian,
            subset_by_index=[0, upper],
            check_finite=False,
        )
        keep = eigenvalues > 1e-9
        if int(keep.sum()) >= n_modes or upper >= dim - 1:
            eigenvalues = eigenvalues[keep][:n_modes]
            eigenvectors = eigenvectors[:, keep][:, :n_modes]
            if eigenvectors.shape[1] < n_modes:
                raise ValueError(
                    f"ANM eigensolve returned {eigenvectors.shape[1]} nontrivial modes"
                )
            return eigenvalues, eigenvectors
        upper = min(dim - 1, upper + n_modes + 24)


def sparse_anm_hessian(coords: np.ndarray, cutoff_A: float) -> sparse.csr_matrix:
    """Sparse uniform ANM Hessian matching softmode_lib.anm_hessian."""
    i, j, _distance = L.contact_pairs(coords, cutoff_A)
    n = len(coords)
    dxyz = coords[j] - coords[i]
    r2 = np.sum(dxyz * dxyz, axis=1)
    blocks = dxyz[:, :, None] * dxyz[:, None, :] / r2[:, None, None]
    rows = []
    cols = []
    data = []
    for a, b, block in zip(i, j, blocks):
        ia = 3 * int(a)
        ib = 3 * int(b)
        for u in range(3):
            for v in range(3):
                value = float(block[u, v])
                rows.extend((ia + u, ib + u, ia + u, ib + u))
                cols.extend((ia + v, ib + v, ib + v, ia + v))
                data.extend((value, value, -value, -value))
    return sparse.coo_matrix((data, (rows, cols)), shape=(3 * n, 3 * n)).tocsr()


def score_anm_endpoint(
    reference_coords: np.ndarray | None,
    displacement: np.ndarray,
    *,
    cutoff_A: float,
    n_modes: int,
    eigensystem: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, object]:
    if eigensystem is None:
        if reference_coords is None:
            raise ValueError("reference coordinates are required when eigensystem is absent")
        eigensystem = anm_modes_subset(reference_coords, cutoff_A, n_modes)
    eigenvalues, modes = eigensystem
    if modes.shape[1] < n_modes:
        raise ValueError(f"ANM returned {modes.shape[1]} modes, expected at least {n_modes}")
    direction = unit_vector(displacement)
    overlaps = L.mode_overlaps(modes[:, :n_modes], direction)
    return {
        "mode1_overlap": float(overlaps[0]),
        "best20_overlap": float(overlaps.max()),
        "best20_rank": int(np.argmax(overlaps) + 1),
        "top3_projection": float(np.sqrt(np.sum(overlaps[:3] ** 2))),
        "top10_projection": float(np.sqrt(np.sum(overlaps[:10] ** 2))),
        "eigval_ratio_2_1": float(eigenvalues[1] / eigenvalues[0]),
        "overlaps": [float(value) for value in overlaps],
    }


def endpoint_rows(
    conformers: np.ndarray,
    labels: np.ndarray,
    open_mask: np.ndarray,
    *,
    cutoff_A: float,
    n_modes: int,
) -> list[dict[str, object]]:
    rows = []
    open_indices = np.where(open_mask)[0]
    closed_indices = np.where(~open_mask)[0]
    committed_modes = np.load(DATA / "crbn_anm_modes.npz", allow_pickle=False)["anm_eigvecs"][
        :, :n_modes
    ]
    for open_index in open_indices:
        for closed_index in closed_indices:
            open_coords = conformers[open_index]
            closed_coords = conformers[closed_index]
            displacement = (closed_coords - open_coords).reshape(-1)
            rmsd = float(np.sqrt(np.mean(np.sum((closed_coords - open_coords) ** 2, axis=1))))
            overlaps = L.mode_overlaps(committed_modes, displacement)
            rows.append(
                {
                    "open_pdb": str(labels[open_index]),
                    "closed_pdb": str(labels[closed_index]),
                    "reference_role": "committed_8CVP_basis_pair_axis",
                    "reference_pdb": "8CVP",
                    "n_ca_common_window": int(conformers.shape[1]),
                    "pair_ca_rmsd_A": rmsd,
                    "mode1_overlap": float(overlaps[0]),
                    "best20_overlap": float(overlaps.max()),
                    "best20_rank": int(np.argmax(overlaps) + 1),
                    "top3_projection": float(np.sqrt(np.sum(overlaps[:3] ** 2))),
                    "top10_projection": float(np.sqrt(np.sum(overlaps[:10] ** 2))),
                    "eigval_ratio_2_1": None,
                }
            )
    return rows


def add_primary_pair_own_basis_rows(
    rows: list[dict[str, object]],
    conformers: np.ndarray,
    labels: np.ndarray,
    config: dict,
) -> None:
    open_matches = np.where(labels == config["primary_open"])[0]
    closed_matches = np.where(labels == config["primary_closed"])[0]
    if open_matches.size != 1 or closed_matches.size != 1:
        raise ValueError("primary open/closed endpoints must each occur exactly once")
    open_index = int(open_matches[0])
    closed_index = int(closed_matches[0])
    open_coords = conformers[open_index]
    closed_coords = conformers[closed_index]
    displacement = (closed_coords - open_coords).reshape(-1)
    rmsd = float(np.sqrt(np.mean(np.sum((closed_coords - open_coords) ** 2, axis=1))))
    for reference_role, reference_index, reference_coords in (
        ("open_endpoint_own_basis_primary_pair", open_index, open_coords),
        ("closed_endpoint_own_basis_primary_pair", closed_index, closed_coords),
    ):
        metrics = score_anm_endpoint(
            reference_coords,
            displacement,
            cutoff_A=float(config["cutoff_A"]),
            n_modes=int(config["n_modes"]),
        )
        rows.append(
            {
                "open_pdb": str(labels[open_index]),
                "closed_pdb": str(labels[closed_index]),
                "reference_role": reference_role,
                "reference_pdb": str(labels[reference_index]),
                "n_ca_common_window": int(conformers.shape[1]),
                "pair_ca_rmsd_A": rmsd,
                "mode1_overlap": metrics["mode1_overlap"],
                "best20_overlap": metrics["best20_overlap"],
                "best20_rank": metrics["best20_rank"],
                "top3_projection": metrics["top3_projection"],
                "top10_projection": metrics["top10_projection"],
                "eigval_ratio_2_1": metrics["eigval_ratio_2_1"],
            }
        )


def add_committed_endpoint_sensitivity_rows(
    rows: list[dict[str, object]],
    labels: np.ndarray,
    open_mask: np.ndarray,
    n_ca: int,
) -> None:
    robustness = json.loads((DATA / "anm_robustness.json").read_text(encoding="utf-8"))
    closed_lookup = robustness["closed_all_15A"]
    for label in labels[open_mask]:
        row = robustness["table"][str(label)]["15.0"]
        rows.append(
            {
                "open_pdb": str(label),
                "closed_pdb": "ensemble_mean_65_closed",
                "reference_role": "committed_open_endpoint_vs_mean_axis",
                "reference_pdb": str(label),
                "n_ca_common_window": n_ca,
                "pair_ca_rmsd_A": None,
                "mode1_overlap": float(row["mode1_overlap"]),
                "best20_overlap": float(row["best_overlap"]),
                "best20_rank": int(row["best_mode_rank"]),
                "top3_projection": None,
                "top10_projection": float(row["cum_top10"]),
                "eigval_ratio_2_1": None,
            }
        )
    for label in labels[~open_mask]:
        row = closed_lookup[str(label)]
        rows.append(
            {
                "open_pdb": "mean_open_axis",
                "closed_pdb": str(label),
                "reference_role": "committed_closed_endpoint_vs_mean_axis",
                "reference_pdb": str(label),
                "n_ca_common_window": n_ca,
                "pair_ca_rmsd_A": None,
                "mode1_overlap": float(row["mode1_overlap"]),
                "best20_overlap": float(row["best_overlap"]),
                "best20_rank": int(row["best_mode_rank"]),
                "top3_projection": None,
                "top10_projection": None,
                "eigval_ratio_2_1": None,
            }
        )


def percentile_less(values: Iterable[float], observed: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan")
    return float(100.0 * np.mean(array < float(observed)))


def control_panel_rows(primary_row: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    with (DATA / "positive_controls.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["name"] == "CRBN (this work)":
                continue
            if row["passes_quality"] != "True" or row["substantial_transition"] != "True":
                continue
            rows.append(
                {
                    "name": row["name"],
                    "source": "existing_external_primary_control_panel",
                    "motion_class": row["motion_class"],
                    "mode1_overlap": float(row["mode1_overlap_15A"]),
                    "best20_overlap": float(row["best_overlap_15A"]),
                    "best20_rank": int(row["best_rank_15A"]),
                    "top3_projection": float(row["cum_overlap_top3_15A"]),
                    "panel_percentile_scope": "18 external primary controls; not a proteome-wide percentile",
                }
            )
    if len(rows) != 18:
        raise ValueError(f"expected 18 external primary controls, found {len(rows)}")
    observed = {
        "name": "CRBN 8CVP-5FQD",
        "source": "new_pair_level_crbn_primary_pair",
        "motion_class": "interdomain hinge",
        "mode1_overlap": float(primary_row["mode1_overlap"]),
        "best20_overlap": float(primary_row["best20_overlap"]),
        "best20_rank": int(primary_row["best20_rank"]),
        "top3_projection": float(primary_row["top3_projection"]),
        "panel_percentile_scope": "18 external primary controls; not a proteome-wide percentile",
    }
    rows_with_observed = rows + [observed]
    percentiles = {
        metric: percentile_less([row[metric] for row in rows], observed[metric])
        for metric in ("mode1_overlap", "best20_overlap", "top3_projection")
    }
    return rows_with_observed, percentiles


def pca_fluctuation_top10(conformers: np.ndarray) -> np.ndarray:
    modes, _variance, scores = L.ensemble_pca(conformers)
    eigenvalues = (scores ** 2).sum(axis=0) / (len(conformers) - 1)
    out = np.zeros(conformers.shape[1], dtype=float)
    for mode_index in range(10):
        out += eigenvalues[mode_index] * (
            modes[:, mode_index].reshape(conformers.shape[1], 3) ** 2
        ).sum(axis=1)
    return out


def tbd_internal_variance(
    conformers: np.ndarray,
    domain_map: dict[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    tbd = domain_map["TBD"]
    mean_tbd = conformers[:, tbd].mean(axis=0)
    residuals = []
    rmsds = []
    for coords in conformers[:, tbd]:
        rotation, translation, rmsd = L.kabsch(coords, mean_tbd)
        fitted = (rotation @ coords.T).T + translation
        residuals.append(fitted - mean_tbd)
        rmsds.append(rmsd)
    residuals = np.asarray(residuals, dtype=float)
    per_residue = np.var(residuals, axis=0, ddof=1).sum(axis=1)
    full = np.full(conformers.shape[1], np.nan, dtype=float)
    full[tbd] = per_residue
    return full, float(np.mean(rmsds))


def profile_set_effect(
    profile: np.ndarray,
    selected_indices: list[int],
    zinc_indices: list[int],
    *,
    selected_name: str,
    profile_name: str,
) -> dict[str, object]:
    selected = np.asarray(profile[selected_indices], dtype=float)
    zinc = np.asarray(profile[zinc_indices], dtype=float)
    if not np.isfinite(selected).all() or not np.isfinite(zinc).all():
        return {
            "definition": selected_name,
            "profile": profile_name,
            "status": "not_applicable",
            "reason": "profile is only defined for a subset that does not cover both groups",
        }
    u, p = stats.mannwhitneyu(selected, zinc, alternative="two-sided", method="exact")
    return {
        "definition": selected_name,
        "profile": profile_name,
        "status": "ok",
        "selected_mean": float(selected.mean()),
        "zinc_mean": float(zinc.mean()),
        "difference_selected_minus_zinc": float(selected.mean() - zinc.mean()),
        "selected_values": [float(value) for value in selected],
        "zinc_values": [float(value) for value in zinc],
        "mannwhitney_u": float(u),
        "p_exact_two_sided_exploratory": float(p),
        "rank_biserial": float(2 * u / (selected.size * zinc.size) - 1),
        "minimum_attainable_two_sided_p": float(
            stats.mannwhitneyu(
                np.arange(selected.size),
                np.arange(selected.size, selected.size + zinc.size),
                alternative="two-sided",
                method="exact",
            )[1]
        ),
    }


def residualize_against_axis_distance(
    profile: np.ndarray,
    axis_distance: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    x = axis_distance[mask] ** 2
    y = profile[mask]
    slope, intercept = np.polyfit(x, y, 1)
    residual = np.full(profile.shape, np.nan, dtype=float)
    residual[mask] = y - (slope * x + intercept)
    return residual


def csv_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def cached_cif_path(pdb_id: str) -> Path:
    return DATA / "_controls_cif_cache" / f"{pdb_id.upper()}.cif.gz"


def fetch_control_cif(pdb_id: str, offline: bool, manifest: list[dict[str, object]]) -> str:
    path = cached_cif_path(pdb_id)
    cached_before = path.is_file()
    if offline and not cached_before:
        raise FileNotFoundError(
            f"offline control endpoint analysis requires cached mmCIF: {path}"
        )
    text = L.fetch_cif(pdb_id, cache=str(path.parent))
    manifest.append(
        {
            "pdb": pdb_id.upper(),
            "cache_path": str(path.relative_to(ROOT)),
            "cache_present": path.is_file(),
            "url": f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_authority": "official RCSB PDB mmCIF",
            "retrieval_policy": "read local cache; download official RCSB mmCIF only when cache is missing and offline mode is not requested",
        }
    )
    return text


def primary_external_control_names() -> set[str]:
    names = set()
    with (DATA / "positive_controls.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["name"] == "CRBN (this work)":
                continue
            if csv_bool(row["passes_quality"]) and csv_bool(row["substantial_transition"]):
                names.add(row["name"])
    if len(names) != 18:
        raise ValueError(f"expected 18 external primary controls in positive_controls.csv, found {len(names)}")
    return names


def legacy_primary_control_rows() -> dict[str, dict[str, str]]:
    rows = {}
    with (DATA / "positive_controls.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["name"] == "CRBN (this work)":
                continue
            if csv_bool(row["passes_quality"]) and csv_bool(row["substantial_transition"]):
                rows[row["name"]] = row
    if len(rows) != 18:
        raise ValueError(f"expected 18 legacy primary controls, found {len(rows)}")
    return rows


def select_control_pair_coordinates(
    rec: dict[str, object], offline: bool, manifest: list[dict[str, object]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    ca_open, _ = L.parse_atoms(fetch_control_cif(str(rec["open"]), offline, manifest))
    ca_closed, _ = L.parse_atoms(fetch_control_cif(str(rec["closed"]), offline, manifest))
    best = None
    for chain_open in sorted(ca_open):
        for chain_closed in sorted(ca_closed):
            n_shared = len(set(ca_open[chain_open]) & set(ca_closed[chain_closed]))
            if best is None or n_shared > best[0]:
                best = (n_shared, chain_open, chain_closed)
    if best is None:
        raise ValueError(f"{rec['name']}: no chain pair found")
    _, chain_open, chain_closed = best
    open_coords, closed_coords, resnums = CP._window_pair(ca_open[chain_open], ca_closed[chain_closed])
    if len(open_coords) < 40:
        raise ValueError(f"{rec['name']}: only {len(open_coords)} shared residues")
    return open_coords, closed_coords, resnums, chain_open, chain_closed


def score_endpoint_basis(
    basis_coords: np.ndarray, target_coords: np.ndarray, cutoff_A: float, n_modes: int
) -> dict[str, object]:
    target_in_basis_frame = L.kabsch_apply(target_coords, basis_coords)
    displacement = (target_in_basis_frame - basis_coords).reshape(-1)
    displacement = displacement / np.linalg.norm(displacement)
    eigenvalues, modes = L.modes(L.anm_hessian(basis_coords, cutoff_A), n_modes)
    overlaps = L.mode_overlaps(modes, displacement)
    return {
        "mode1_overlap": float(overlaps[0]),
        "best20_overlap": float(overlaps.max()),
        "best20_rank": int(np.argmax(overlaps) + 1),
        "top3_projection": float(L.cumulative_overlap(modes, displacement, 3)),
        "top10_projection": float(L.cumulative_overlap(modes, displacement, 10)),
        "eigval_ratio_2_1": float(eigenvalues[1] / eigenvalues[0]),
    }


def control_endpoint_rankings(
    config: dict, offline: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    cutoff_A = float(config["cutoff_A"])
    n_modes = int(config["n_modes"])
    primary_names = primary_external_control_names()
    legacy_rows = legacy_primary_control_rows()
    manifest: list[dict[str, object]] = []
    rankings: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    for rec in CP.PAIRS:
        if rec["name"] not in primary_names:
            continue
        open_coords, closed_coords, resnums, chain_open, chain_closed = select_control_pair_coordinates(
            rec, offline, manifest
        )
        transition_rmsd = float(np.sqrt(((L.kabsch_apply(closed_coords, open_coords) - open_coords) ** 2).sum(1).mean()))
        scores_by_state = {
            "open": score_endpoint_basis(open_coords, closed_coords, cutoff_A, n_modes),
            "closed": score_endpoint_basis(closed_coords, open_coords, cutoff_A, n_modes),
        }
        legacy = legacy_rows[str(rec["name"])]
        for state, scores in scores_by_state.items():
            row = {
                "name": rec["name"],
                "motion_class": rec["motion_class"],
                "pdb_open": rec["open"],
                "pdb_closed": rec["closed"],
                "chain_open": chain_open,
                "chain_closed": chain_closed,
                "n_shared_ca": int(len(resnums)),
                "first_shared_residue": int(resnums[0]),
                "last_shared_residue": int(resnums[-1]),
                "transition_ca_rmsd_A": transition_rmsd,
                "endpoint_basis_state": state,
                "endpoint_basis_pdb": rec["open"] if state == "open" else rec["closed"],
                "target_state": "closed" if state == "open" else "open",
                "target_pdb": rec["closed"] if state == "open" else rec["open"],
                "mode1_overlap": scores["mode1_overlap"],
                "best20_overlap": scores["best20_overlap"],
                "best20_rank": scores["best20_rank"],
                "top3_projection": scores["top3_projection"],
                "top10_projection": scores["top10_projection"],
                "eigval_ratio_2_1": scores["eigval_ratio_2_1"],
                "legacy_open_mode1_overlap": float(legacy["mode1_overlap_15A"]) if state == "open" else None,
                "legacy_open_best20_overlap": float(legacy["best_overlap_15A"]) if state == "open" else None,
                "legacy_open_best20_rank": int(legacy["best_rank_15A"]) if state == "open" else None,
                "legacy_open_top3_projection": float(legacy["cum_overlap_top3_15A"]) if state == "open" else None,
                "legacy_open_mode1_abs_delta": abs(scores["mode1_overlap"] - float(legacy["mode1_overlap_15A"])) if state == "open" else None,
                "legacy_open_best20_abs_delta": abs(scores["best20_overlap"] - float(legacy["best_overlap_15A"])) if state == "open" else None,
                "legacy_open_top3_abs_delta": abs(scores["top3_projection"] - float(legacy["cum_overlap_top3_15A"])) if state == "open" else None,
                "coordinate_source": "mmCIF local cache or official RCSB download manifest",
                "residue_pairing_rule": "max shared-chain pair; common author residue-number intersection; no imputation",
            }
            rankings.append(row)
        open_score = scores_by_state["open"]
        closed_score = scores_by_state["closed"]
        paired.append(
            {
                "name": rec["name"],
                "motion_class": rec["motion_class"],
                "pdb_open": rec["open"],
                "pdb_closed": rec["closed"],
                "n_shared_ca": int(len(resnums)),
                "transition_ca_rmsd_A": transition_rmsd,
                "open_basis_mode1_overlap": open_score["mode1_overlap"],
                "closed_basis_mode1_overlap": closed_score["mode1_overlap"],
                "closed_minus_open_mode1_overlap": closed_score["mode1_overlap"] - open_score["mode1_overlap"],
                "open_basis_best20_overlap": open_score["best20_overlap"],
                "closed_basis_best20_overlap": closed_score["best20_overlap"],
                "closed_minus_open_best20_overlap": closed_score["best20_overlap"] - open_score["best20_overlap"],
                "open_basis_best20_rank": open_score["best20_rank"],
                "closed_basis_best20_rank": closed_score["best20_rank"],
                "open_basis_top3_projection": open_score["top3_projection"],
                "closed_basis_top3_projection": closed_score["top3_projection"],
                "closed_minus_open_top3_projection": closed_score["top3_projection"] - open_score["top3_projection"],
            }
        )
    if len(rankings) != 36 or len(paired) != 18:
        raise ValueError(f"expected 36 ranking rows and 18 paired rows, found {len(rankings)} and {len(paired)}")
    open_deltas = [row["legacy_open_mode1_abs_delta"] for row in rankings if row["endpoint_basis_state"] == "open"]
    summary = {
        "n_external_primary_controls": 18,
        "n_endpoint_owned_basis_rows": len(rankings),
        "n_paired_state_rows": len(paired),
        "coordinate_rule": "original open/closed PDB pairs from control_panel.PAIRS; chain pair with maximal shared author residues; common author residue-number intersection; no imputation",
        "cutoff_A": cutoff_A,
        "n_modes": n_modes,
        "open_basis_legacy_max_abs_delta_mode1": float(max(open_deltas)),
        "open_basis_legacy_allclose_atol_1e_8": bool(max(open_deltas) <= 1e-8),
        "open_basis_mode1_median": float(np.median([row["open_basis_mode1_overlap"] for row in paired])),
        "closed_basis_mode1_median": float(np.median([row["closed_basis_mode1_overlap"] for row in paired])),
        "closed_minus_open_mode1_median": float(np.median([row["closed_minus_open_mode1_overlap"] for row in paired])),
        "open_basis_rank1_fraction": float(np.mean([row["open_basis_best20_rank"] == 1 for row in paired])),
        "closed_basis_rank1_fraction": float(np.mean([row["closed_basis_best20_rank"] == 1 for row in paired])),
        "coordinate_manifest": manifest,
    }
    return rankings, paired, summary


def residue_outputs(
    conformers: np.ndarray,
    open_mask: np.ndarray,
    residues: np.ndarray,
    axis: np.ndarray,
    config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    domain_map = domain_indices(residues, config["domains"])
    reference = conformers[0]
    labels = np.load(DATA / "pca_diffvec.npz", allow_pickle=False)["labels"].astype(str)
    reference = conformers[list(labels).index(config["primary_open"])]
    eigenvalues, modes = L.modes(L.anm_hessian(reference, config["cutoff_A"]), config["n_modes"])
    anm_fluct = L.sqfluct(eigenvalues, modes, 10)
    pca_fluct = pca_fluctuation_top10(conformers)
    tbd_internal, mean_tbd_internal_rmsd = tbd_internal_variance(conformers, domain_map)
    hinge, axis_distance, endpoint_displacement = compute_geometry(conformers, open_mask, residues)
    tbd_mask = residues >= int(config["domains"]["TBD"][0])
    pca_tbd_axis_residual = residualize_against_axis_distance(pca_fluct, axis_distance, tbd_mask)
    anm_tbd_axis_residual = residualize_against_axis_distance(anm_fluct, axis_distance, tbd_mask)

    definitions = {
        key: [int(value) for value in values]
        for key, values in config["pocket_definitions_common269"].items()
    }
    zinc = [int(value) for value in config["zinc_residues"]]
    residue_to_index = {int(residue): index for index, residue in enumerate(residues)}
    for residue in sorted({*zinc, *[r for values in definitions.values() for r in values]}):
        if residue not in residue_to_index:
            raise ValueError(f"functional residue {residue} is absent from the 269-residue window")

    effect_rows = []
    profiles = {
        "anm_sqfluct_top10": anm_fluct,
        "pca_sqfluct_top10": pca_fluct,
        "anm_tbd_axis_adjusted_residual": anm_tbd_axis_residual,
        "pca_tbd_axis_adjusted_residual": pca_tbd_axis_residual,
        "tbd_internal_ensemble_variance_A2": tbd_internal,
    }
    zinc_indices = [residue_to_index[residue] for residue in zinc]
    for definition, selected_residues in definitions.items():
        selected_indices = [residue_to_index[residue] for residue in selected_residues]
        for profile_name, profile in profiles.items():
            effect_rows.append(
                profile_set_effect(
                    profile,
                    selected_indices,
                    zinc_indices,
                    selected_name=definition,
                    profile_name=profile_name,
                )
            )

    residue_rows = []
    for index, residue in enumerate(residues):
        memberships = [
            name for name, values in definitions.items() if int(residue) in set(values)
        ]
        if int(residue) in zinc:
            memberships.append("zinc_coordinating")
        residue_rows.append(
            {
                "resnum": int(residue),
                "domain": domain_name_for_residue(int(residue), config["domains"]),
                "axis_distance_A": float(axis_distance[index]),
                "endpoint_displacement_A": float(endpoint_displacement[index]),
                "anm_sqfluct_top10": float(anm_fluct[index]),
                "pca_sqfluct_top10": float(pca_fluct[index]),
                "anm_tbd_axis_adjusted_residual": none_if_nan(anm_tbd_axis_residual[index]),
                "pca_tbd_axis_adjusted_residual": none_if_nan(pca_tbd_axis_residual[index]),
                "tbd_internal_ensemble_variance_A2": none_if_nan(tbd_internal[index]),
                "membership": ";".join(memberships),
            }
        )
    summary = {
        "hinge_geometry": {
            "rotation_angle_deg": float(hinge["rotation_angle_deg"]),
            "axis_proximal_boundary_residues": hinge["axis_proximal_boundary_residues"],
        },
        "mean_tbd_internal_alignment_rmsd_A": mean_tbd_internal_rmsd,
        "effect_note": (
            "Exact residue-set p values are exploratory and have small minimum attainable "
            "values; nonsignificant results are not evidence of absence."
        ),
    }
    return residue_rows, effect_rows, summary


def domain_name_for_residue(residue: int, domains: dict[str, list[int]]) -> str:
    for name, (lo, hi) in domains.items():
        if int(lo) <= residue <= int(hi):
            return name
    return "outside_defined_domains"


def none_if_nan(value: float) -> float | None:
    value = float(value)
    return None if math.isnan(value) else value


def signed_rotation_angle(rotation: np.ndarray, axis_unit: np.ndarray) -> float:
    """Return the signed rotation angle around the supplied unit axis.

    The screw-axis eigenvector has an arbitrary sign.  Using the skew part of the
    row/column-consistent rotation matrix fixes the sign for the tangent field and
    keeps the result invariant if the axis vector is flipped.
    """
    axis_unit = unit_vector(np.asarray(axis_unit, dtype=float))
    skew_vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    sin_component = float(axis_unit @ skew_vector) / 2.0
    cos_component = float((np.trace(rotation) - 1.0) / 2.0)
    return float(math.atan2(sin_component, cos_component))


def local_screw_tangent(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_unit: np.ndarray,
    rotation: np.ndarray,
    rise_A: float,
) -> np.ndarray:
    signed_angle = signed_rotation_angle(rotation, axis_unit)
    axis_unit = unit_vector(np.asarray(axis_unit, dtype=float))
    return signed_angle * np.cross(axis_unit, points - axis_point) + float(rise_A) * axis_unit


def embedded_unit_vector(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    embedded = np.zeros_like(field, dtype=float)
    embedded[mask] = field[mask]
    return unit_vector(embedded.reshape(-1))


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def finite_chord_tangent_rows(
    conformers: np.ndarray,
    open_mask: np.ndarray,
    residues: np.ndarray,
    axis: np.ndarray,
    config: dict,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    open_mean = conformers[open_mask].mean(axis=0)
    closed_mean = conformers[~open_mask].mean(axis=0)
    anchor = residues <= int(config["domains"]["HB"][1])
    moving = residues >= int(config["domains"]["TBD"][0])
    anchor_rotation, anchor_translation, anchor_rmsd = L.kabsch(open_mean[anchor], closed_mean[anchor])
    anchored_open = (anchor_rotation @ open_mean.T).T + anchor_translation
    tbd_rotation, tbd_translation, tbd_rmsd = L.kabsch(anchored_open[moving], closed_mean[moving])
    screw = screw_axis(tbd_rotation, tbd_translation)
    axis_unit = np.asarray(screw["axis_unit_vector"], dtype=float)
    axis_point = np.asarray(screw["axis_point_A"], dtype=float)
    signed_angle_rad = signed_rotation_angle(tbd_rotation, axis_unit)
    rise = float(screw["rise_A"])

    observed_chord = closed_mean - anchored_open
    rigid_tbd_closed = anchored_open.copy()
    rigid_tbd_closed[moving] = (tbd_rotation @ anchored_open[moving].T).T + tbd_translation
    rigid_tbd_chord = rigid_tbd_closed - anchored_open
    tangent = local_screw_tangent(anchored_open, axis_point, axis_unit, tbd_rotation, rise)
    distance = distance_to_axis(anchored_open, axis_point, axis_unit)

    rows = []
    for index, residue in enumerate(residues):
        observed_chord_norm = float(np.linalg.norm(observed_chord[index]))
        rigid_chord_norm = float(np.linalg.norm(rigid_tbd_chord[index]))
        tangent_norm = float(np.linalg.norm(tangent[index]))
        observed_cosine = safe_cosine(observed_chord[index], tangent[index])
        rigid_cosine = safe_cosine(rigid_tbd_chord[index], tangent[index]) if moving[index] else None
        rows.append(
            {
                "resnum": int(residue),
                "domain": domain_name_for_residue(int(residue), config["domains"]),
                "axis_distance_A": float(distance[index]),
                "observed_finite_chord_displacement_A": observed_chord_norm,
                "tbd_rigidfit_finite_chord_displacement_A": none_if_nan(rigid_chord_norm if moving[index] else math.nan),
                "local_rotation_tangent_A": tangent_norm,
                "observed_chord_vs_tangent_cosine": none_if_nan(observed_cosine if observed_cosine is not None else math.nan),
                "tbd_rigidfit_chord_vs_tangent_cosine": none_if_nan(rigid_cosine if rigid_cosine is not None else math.nan),
            }
        )

    observed_chord_unit = unit_vector(observed_chord.reshape(-1))
    tbd_observed_chord_unit = embedded_unit_vector(observed_chord, moving)
    tbd_rigid_chord_unit = embedded_unit_vector(rigid_tbd_chord, moving)
    tbd_tangent_unit = embedded_unit_vector(tangent, moving)
    modes = np.load(DATA / "crbn_anm_modes.npz", allow_pickle=False)["anm_eigvecs"]
    tbd_observed_cosine = float(abs(tbd_observed_chord_unit @ tbd_tangent_unit))
    tbd_rigid_cosine = float(abs(tbd_rigid_chord_unit @ tbd_tangent_unit))
    summary = {
        "anchor_kabsch_rmsd_A": float(anchor_rmsd),
        "tbd_rigid_fit_rmsd_A": float(tbd_rmsd),
        "rotation_angle_deg": float(screw["angle_deg"]),
        "signed_rotation_angle_deg": math.degrees(signed_angle_rad),
        "ideal_zero_rise_rigid_rotation_cos_theta_over_2": float(math.cos(abs(signed_angle_rad) / 2.0)),
        "screw_rise_A": rise,
        "observed_all_window_chord_vs_tbd_tangent_cosine": float(abs(observed_chord_unit @ tbd_tangent_unit)),
        "observed_tbd_chord_vs_tangent_cosine": tbd_observed_cosine,
        "tbd_rigidfit_chord_vs_tangent_cosine": tbd_rigid_cosine,
        "mode1_overlap_with_observed_all_window_finite_chord": float(abs(modes[:, 0] @ observed_chord_unit)),
        "mode1_overlap_with_observed_tbd_finite_chord_embedded": float(abs(modes[:, 0] @ tbd_observed_chord_unit)),
        "mode1_overlap_with_tbd_rigidfit_finite_chord_embedded": float(abs(modes[:, 0] @ tbd_rigid_chord_unit)),
        "mode1_overlap_with_tbd_local_rotation_tangent_embedded": float(abs(modes[:, 0] @ tbd_tangent_unit)),
        "diagnosis": (
            "The observed open-to-closed displacement is the finite endpoint chord. "
            "The tangent field is a signed local screw-rotation diagnostic for the "
            "TBD after anchoring NTD/HB, and is reported separately from the total "
            "global endpoint displacement."
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fieldnames = list(rows[0])
    lines = []
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    lines.append(buffer.getvalue())
    atomic_write_text(path, "".join(lines))


def csv_payload(rows: list[dict[str, object]]) -> str:
    if not rows:
        raise ValueError("empty CSV payload")
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main_pair_row(endpoint_score_rows: list[dict[str, object]], config: dict) -> dict[str, object]:
    matches = [
        row
        for row in endpoint_score_rows
        if row["open_pdb"] == config["primary_open"]
        and row["closed_pdb"] == config["primary_closed"]
        and row["reference_role"] == "open_endpoint_own_basis_primary_pair"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one primary pair row, found {len(matches)}")
    return matches[0]


def build_report(summary: dict[str, object]) -> str:
    primary = summary["primary_pair"]
    pct = summary["control_panel_percentiles"]
    pocket = summary["pocket_effect_highlights"]
    controls = summary["control_endpoint_state_effects"]
    chord = summary["finite_chord_vs_local_tangent"]
    return (
        "# CRBN Strengthening Controls\n\n"
        "## Scope\n"
        "This offline analysis adds pair-level endpoint controls for the CRBN "
        "8CVP-5FQD transition, compares it with the existing 18 external primary "
        "control panel, separates mode-1, best-of-20 and top-3 projection statistics, "
        "and keeps the resulting percentile scoped to this selected panel rather than "
        "to the proteome.\n\n"
        "## Primary Pair\n"
        f"- Pair: {primary['open_pdb']} open endpoint to {primary['closed_pdb']} closed endpoint.\n"
        f"- Common residue membership: {primary['n_ca_common_window']} C-alpha residues in the "
        "fixed 269-residue window.\n"
        f"- Open-endpoint ANM basis: mode 1 overlap {primary['mode1_overlap']:.3f}; "
        f"best-of-20 {primary['best20_overlap']:.3f} at rank {primary['best20_rank']}; "
        f"top-3 projection {primary['top3_projection']:.3f}.\n"
        f"- External primary-panel percentile: mode 1 {pct['mode1_overlap']:.1f}%, "
        f"best-of-20 {pct['best20_overlap']:.1f}%, top-3 projection "
        f"{pct['top3_projection']:.1f}%.\n\n"
        "## External Control Endpoint Bases\n"
        f"The 18 external primary controls were recomputed from their original open/closed "
        f"PDB coordinate pairs using the original common-author-residue intersection rule "
        f"and no imputation. Each control has an open-endpoint ANM basis and a closed-endpoint "
        f"ANM basis in control_endpoint_rankings.csv ({controls['n_endpoint_owned_basis_rows']} "
        "rows total). "
        f"The recomputed open-basis mode-1 values agree with the legacy panel with maximum "
        f"absolute delta {controls['open_basis_legacy_max_abs_delta_mode1']:.3g}. "
        f"Across paired controls, median mode-1 overlap is "
        f"{controls['open_basis_mode1_median']:.3f} on the open basis and "
        f"{controls['closed_basis_mode1_median']:.3f} on the closed basis; the median "
        f"closed-minus-open mode-1 shift is "
        f"{controls['closed_minus_open_mode1_median']:.3f}. Rank-1 recovery fractions are "
        f"{controls['open_basis_rank1_fraction']:.3f} for open bases and "
        f"{controls['closed_basis_rank1_fraction']:.3f} for closed bases.\n\n"
        "## Endpoint Sensitivity\n"
        f"All 5 open x 65 closed CRBN endpoint pairs were scored against the committed "
        f"8CVP ANM basis ({summary['endpoint_sensitivity']['pair_axis_projection_rows']} rows), "
        "and the 8CVP-5FQD primary pair was rescored on both endpoint-owned bases. "
        f"Committed endpoint-vs-mean sensitivity adds "
        f"{summary['endpoint_sensitivity']['committed_open_endpoint_rows']} open-endpoint and "
        f"{summary['endpoint_sensitivity']['committed_closed_endpoint_rows']} closed-endpoint rows. "
        f"The primary-pair open own-basis mode-1 overlap is "
        f"{summary['endpoint_sensitivity']['primary_open_own_basis_mode1']:.3f}; the closed "
        f"own-basis value is {summary['endpoint_sensitivity']['primary_closed_own_basis_mode1']:.3f}. "
        "This supports a robust endpoint-level signal but leaves biological success "
        "defined by geometry and recovery, not by any arbitrary p-value cutoff.\n\n"
        "## Pocket And Residue Definitions\n"
        f"UniProt3 PCA raw effect is {pocket['uniprot_pca_raw_difference']:.3f}; after "
        f"TBD lever-arm adjustment it is {pocket['uniprot_pca_axis_adjusted_difference']:.3f}. "
        f"The 5FQD 4.5 A contact-shell PCA raw effect is "
        f"{pocket['contact_shell_pca_raw_difference']:.3f}; after adjustment it is "
        f"{pocket['contact_shell_pca_axis_adjusted_difference']:.3f}. "
        "The 5FQD contact shell is a structural pocket definition in the common window; "
        "sensor-loop contacts outside the common 269 residues remain excluded from this "
        "particular residue-level test.\n\n"
        "## Domain-Internal PCA\n"
        f"After removing per-conformer TBD rigid motions, the UniProt3 versus zinc "
        f"domain-internal variance effect is "
        f"{pocket['uniprot_tbd_internal_difference']:.3f} A^2. This is reported separately "
        "from ANM fluctuation, because it measures residual ensemble variance after "
        "TBD rigid-body alignment rather than an intrinsic-mode fluctuation profile.\n\n"
        "## Finite Chord Versus Tangent\n"
        f"The finite screw rotation angle is {chord['rotation_angle_deg']:.3f} degrees "
        f"(signed angle {chord['signed_rotation_angle_deg']:.3f} degrees in the selected "
        "axis orientation). "
        f"For a pure zero-rise rigid rotation, the expected chord/tangent cosine is "
        f"cos(theta/2) = {chord['ideal_zero_rise_rigid_rotation_cos_theta_over_2']:.3f}. "
        f"The observed TBD-only chord/tangent cosine is "
        f"{chord['observed_tbd_chord_vs_tangent_cosine']:.3f}; the fitted pure-TBD "
        f"rigid-field cosine is {chord['tbd_rigidfit_chord_vs_tangent_cosine']:.3f}. "
        f"ANM mode 1 overlaps the total observed all-window finite chord by "
        f"{chord['mode1_overlap_with_observed_all_window_finite_chord']:.3f}, the "
        f"observed TBD-only chord by "
        f"{chord['mode1_overlap_with_observed_tbd_finite_chord_embedded']:.3f}, the "
        f"pure-TBD rigid chord by "
        f"{chord['mode1_overlap_with_tbd_rigidfit_finite_chord_embedded']:.3f}, and "
        f"the pure-TBD local tangent by "
        f"{chord['mode1_overlap_with_tbd_local_rotation_tangent_embedded']:.3f}. "
        "The tangent is a diagnostic for the local rotation model, not a replacement "
        "for the finite endpoint displacement being scored.\n\n"
        "## Interpretation Boundary\n"
        "Exact residue-set p values are exploratory because the residue sets are tiny "
        "and spatially dependent. Nonsignificant residue tests should be written as "
        "inconclusive rather than as evidence that an effect is absent.\n"
    )


def summarize_endpoint_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    pair_rows = [row for row in rows if row["reference_role"] == "committed_8CVP_basis_pair_axis"]
    primary_open = [
        row for row in rows if row["reference_role"] == "open_endpoint_own_basis_primary_pair"
    ]
    primary_closed = [
        row for row in rows if row["reference_role"] == "closed_endpoint_own_basis_primary_pair"
    ]
    committed_open = [
        row for row in rows if row["reference_role"] == "committed_open_endpoint_vs_mean_axis"
    ]
    committed_closed = [
        row for row in rows if row["reference_role"] == "committed_closed_endpoint_vs_mean_axis"
    ]
    if len(primary_open) != 1 or len(primary_closed) != 1:
        raise ValueError("primary pair own-basis rows are incomplete")
    return {
        "n_rows": len(rows),
        "pair_axis_projection_rows": len(pair_rows),
        "committed_open_endpoint_rows": len(committed_open),
        "committed_closed_endpoint_rows": len(committed_closed),
        "primary_open_own_basis_mode1": float(primary_open[0]["mode1_overlap"]),
        "primary_closed_own_basis_mode1": float(primary_closed[0]["mode1_overlap"]),
        "pair_axis_mode1_median": float(np.median([row["mode1_overlap"] for row in pair_rows])),
        "pair_axis_rank1_fraction": float(np.mean([row["best20_rank"] == 1 for row in pair_rows])),
        "pair_axis_top3_projection_min": float(min(row["top3_projection"] for row in pair_rows)),
        "pair_axis_top3_projection_median": float(
            np.median([row["top3_projection"] for row in pair_rows])
        ),
        "committed_open_endpoint_rank1_fraction": float(
            np.mean([row["best20_rank"] == 1 for row in committed_open])
        ),
        "committed_closed_endpoint_rank1_fraction": float(
            np.mean([row["best20_rank"] == 1 for row in committed_closed])
        ),
    }


def effect_lookup(effect_rows: list[dict[str, object]], definition: str, profile: str) -> dict[str, object]:
    matches = [
        row for row in effect_rows if row["definition"] == definition and row["profile"] == profile
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one effect for {definition}/{profile}, found {len(matches)}")
    return matches[0]


def build_outputs(config: dict, offline: bool = True) -> dict[str, object]:
    residues = read_residue_window()
    ensemble = np.load(DATA / "crbn_ensemble.ens.npz", allow_pickle=False)
    difference = np.load(DATA / "pca_diffvec.npz", allow_pickle=False)
    conformers, labels, open_mask, axis = validate_ensemble_diff(ensemble, difference)
    if not np.array_equal(labels, difference["labels"].astype(str)):
        raise ValueError("ensemble and difference labels drifted")

    endpoint_score_rows = endpoint_rows(
        conformers,
        labels,
        open_mask,
        cutoff_A=float(config["cutoff_A"]),
        n_modes=int(config["n_modes"]),
    )
    add_primary_pair_own_basis_rows(endpoint_score_rows, conformers, labels, config)
    add_committed_endpoint_sensitivity_rows(
        endpoint_score_rows, labels, open_mask, int(conformers.shape[1])
    )
    primary = main_pair_row(endpoint_score_rows, config)
    control_rows, panel_percentiles = control_panel_rows(primary)
    control_endpoint_rows, control_state_rows, control_endpoint_summary = control_endpoint_rankings(
        config, offline=offline
    )
    residue_rows, effect_rows, residue_summary = residue_outputs(
        conformers, open_mask, residues, axis, config
    )
    chord_rows, chord_summary = finite_chord_tangent_rows(
        conformers, open_mask, residues, axis, config
    )

    uniprot_pca_raw = effect_lookup(
        effect_rows, "uniprot_ligand_annotations", "pca_sqfluct_top10"
    )
    uniprot_pca_adj = effect_lookup(
        effect_rows, "uniprot_ligand_annotations", "pca_tbd_axis_adjusted_residual"
    )
    contact_pca_raw = effect_lookup(
        effect_rows, "5fqd_4.5A_contact_shell_common_window", "pca_sqfluct_top10"
    )
    contact_pca_adj = effect_lookup(
        effect_rows,
        "5fqd_4.5A_contact_shell_common_window",
        "pca_tbd_axis_adjusted_residual",
    )
    uniprot_internal = effect_lookup(
        effect_rows, "uniprot_ligand_annotations", "tbd_internal_ensemble_variance_A2"
    )
    summary = {
        "meta": {
            "script": "scripts/strengthen_controls.py",
            "offline": True,
            "cutoff_A": float(config["cutoff_A"]),
            "n_modes": int(config["n_modes"]),
            "primary_pair": [config["primary_open"], config["primary_closed"]],
            "inputs": [
                "data/crbn_ensemble.ens.npz",
                "data/pca_diffvec.npz",
                "data/crbn_residue_window.csv",
                "data/positive_controls.csv",
                "data/control_panel_summary.json",
                "data/_cif_cache/*.cif.gz for the 18 external primary controls",
                "data/crbn_anm_modes.npz",
                "render/open_8cvp.pdb",
            ],
        },
        "primary_pair": primary,
        "control_panel_percentiles": panel_percentiles,
        "control_endpoint_state_effects": control_endpoint_summary,
        "endpoint_sensitivity": summarize_endpoint_rows(endpoint_score_rows),
        "residue_membership": {
            "n_ca_common_window": int(residues.size),
            "first_residue": int(residues[0]),
            "last_residue": int(residues[-1]),
            "rule": "fixed common C-alpha window; missing loops/sidechains are not added as residue-level observations",
        },
        "residue_effects": residue_summary,
        "pocket_definitions": {
            "uniprot_ligand_annotations": "pre-specified UniProt (S)-thalidomide ligand annotations",
            "annotated_plus_W400_F402": "UniProt ligand annotations plus local cage residues W400/F402",
            "5fqd_4.5A_contact_shell_common_window": (
                "5FQD LVY heavy-atom contacts within 4.5 A intersected with the common 269-residue window"
            ),
            "zinc_residues": config["zinc_residues"],
        },
        "pocket_effect_highlights": {
            "uniprot_pca_raw_difference": uniprot_pca_raw["difference_selected_minus_zinc"],
            "uniprot_pca_raw_p_exact": uniprot_pca_raw["p_exact_two_sided_exploratory"],
            "uniprot_pca_axis_adjusted_difference": uniprot_pca_adj[
                "difference_selected_minus_zinc"
            ],
            "uniprot_pca_axis_adjusted_p_exact": uniprot_pca_adj[
                "p_exact_two_sided_exploratory"
            ],
            "contact_shell_pca_raw_difference": contact_pca_raw[
                "difference_selected_minus_zinc"
            ],
            "contact_shell_pca_raw_p_exact": contact_pca_raw[
                "p_exact_two_sided_exploratory"
            ],
            "contact_shell_pca_axis_adjusted_difference": contact_pca_adj[
                "difference_selected_minus_zinc"
            ],
            "contact_shell_pca_axis_adjusted_p_exact": contact_pca_adj[
                "p_exact_two_sided_exploratory"
            ],
            "uniprot_tbd_internal_difference": uniprot_internal[
                "difference_selected_minus_zinc"
            ],
            "uniprot_tbd_internal_p_exact": uniprot_internal[
                "p_exact_two_sided_exploratory"
            ],
        },
        "finite_chord_vs_local_tangent": chord_summary,
        "interpretation_boundaries": [
            "The selected-panel percentile is not a proteome-wide percentile.",
            "Mode 1, best-of-20 and top-3 projection are distinct statistics.",
            "Exact residue-set p values are exploratory; nonsignificant is not evidence of absence.",
            "Domain-internal PCA variance after TBD rigid-motion removal is separate from ANM fluctuation.",
        ],
    }
    return {
        "summary": summary,
        "endpoint_scores": endpoint_score_rows,
        "control_panel_comparison": control_rows,
        "control_endpoint_rankings": control_endpoint_rows,
        "control_paired_state_effect_summary": control_state_rows,
        "residue_effects": residue_rows,
        "residue_set_effects": effect_rows,
        "finite_chord_tangent": chord_rows,
        "interpretation": build_report(summary),
    }


def verify_outputs(output_dir: Path, outputs: dict[str, object]) -> None:
    expected = {
        "strengthen_controls_summary.json": json.dumps(
            outputs["summary"], indent=1, allow_nan=False
        )
        + "\n",
        "endpoint_scores.csv": csv_payload(outputs["endpoint_scores"]),
        "control_panel_comparison.csv": csv_payload(outputs["control_panel_comparison"]),
        "control_endpoint_rankings.csv": csv_payload(outputs["control_endpoint_rankings"]),
        "control_paired_state_effect_summary.csv": csv_payload(outputs["control_paired_state_effect_summary"]),
        "residue_effects.csv": csv_payload(outputs["residue_effects"]),
        "residue_set_effects.csv": csv_payload(outputs["residue_set_effects"]),
        "finite_chord_tangent.csv": csv_payload(outputs["finite_chord_tangent"]),
        "INTERPRETATION.md": outputs["interpretation"],
    }
    for filename, payload in expected.items():
        path = output_dir / filename
        if not path.exists():
            raise AssertionError(f"missing output for verification: {path}")
        actual = path.read_text(encoding="utf-8")
        if filename.endswith(".json"):
            assert_tree_close(json.loads(payload), json.loads(actual), float_tolerance=1e-8)
        elif filename.endswith(".csv"):
            assert_csv_payload_close(payload, actual, path)
        elif actual != payload:
            raise AssertionError(f"{path} does not match recomputation")


def assert_csv_payload_close(expected: str, actual: str, path: Path) -> None:
    import io

    expected_rows = list(csv.DictReader(io.StringIO(expected)))
    actual_rows = list(csv.DictReader(io.StringIO(actual)))
    if len(expected_rows) != len(actual_rows):
        raise AssertionError(f"{path}: row count mismatch")
    if expected_rows and list(expected_rows[0]) != list(actual_rows[0]):
        raise AssertionError(f"{path}: header mismatch")
    for row_index, (expected_row, actual_row) in enumerate(zip(expected_rows, actual_rows), start=1):
        for key, expected_value in expected_row.items():
            actual_value = actual_row[key]
            if expected_value == actual_value:
                continue
            if expected_value == "" or actual_value == "":
                raise AssertionError(
                    f"{path}: row {row_index} column {key} mismatch "
                    f"{expected_value!r} != {actual_value!r}"
                )
            try:
                expected_float = float(expected_value)
                actual_float = float(actual_value)
            except ValueError as exc:
                raise AssertionError(
                    f"{path}: row {row_index} column {key} mismatch "
                    f"{expected_value!r} != {actual_value!r}"
                ) from exc
            if not math.isclose(expected_float, actual_float, rel_tol=0.0, abs_tol=1e-8):
                raise AssertionError(
                    f"{path}: row {row_index} column {key} numeric mismatch "
                    f"{expected_float:.16g} != {actual_float:.16g}"
                )


def write_outputs(output_dir: Path, outputs: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "strengthen_controls_summary.json", outputs["summary"])
    write_csv(output_dir / "endpoint_scores.csv", outputs["endpoint_scores"])
    write_csv(output_dir / "control_panel_comparison.csv", outputs["control_panel_comparison"])
    write_csv(output_dir / "control_endpoint_rankings.csv", outputs["control_endpoint_rankings"])
    write_csv(output_dir / "control_paired_state_effect_summary.csv", outputs["control_paired_state_effect_summary"])
    write_csv(output_dir / "residue_effects.csv", outputs["residue_effects"])
    write_csv(output_dir / "residue_set_effects.csv", outputs["residue_set_effects"])
    write_csv(output_dir / "finite_chord_tangent.csv", outputs["finite_chord_tangent"])
    atomic_write_text(output_dir / "INTERPRETATION.md", outputs["interpretation"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.verify:
        L.CACHE_WRITES_ENABLED = False
    outputs = build_outputs(config, offline=args.offline or args.verify)
    output_dir = args.output_dir.resolve()
    if args.verify:
        verify_outputs(output_dir, outputs)
        print(f"verify OK: strengthening controls match {output_dir}")
    else:
        write_outputs(output_dir, outputs)
        summary = outputs["summary"]
        primary = summary["primary_pair"]
        print(
            "wrote strengthening controls: "
            f"{primary['open_pdb']}-{primary['closed_pdb']} mode1 "
            f"{primary['mode1_overlap']:.3f}, best20 {primary['best20_overlap']:.3f}"
            f"@{primary['best20_rank']}, top3 {primary['top3_projection']:.3f}"
        )
        print(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
