#!/usr/bin/env python3
"""Export exact tabular source data for CRBN Figures 1-5.

The calculations intentionally mirror the corresponding figure builders.  The
script writes only CSV records and does not modify any upstream analysis file.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from figure_package_utils import require_rigid_null_schema


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "source_data"
HEADER = [
    "panel",
    "record_type",
    "series",
    "id",
    "index",
    "x",
    "y",
    "z",
    "dx",
    "dy",
    "dz",
    "value",
    "unit",
    "notes",
]


def write_rows(stem: str, rows: list[list[object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}_source_data.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(f"{stem}: wrote {len(rows):,} source-data records to {path.relative_to(ROOT)}")


def row(
    panel: str,
    record_type: str,
    series: str = "",
    identifier: object = "",
    index: object = "",
    x: object = "",
    y: object = "",
    z: object = "",
    dx: object = "",
    dy: object = "",
    dz: object = "",
    value: object = "",
    unit: str = "",
    notes: str = "",
) -> list[object]:
    return [
        panel,
        record_type,
        series,
        identifier,
        index,
        x,
        y,
        z,
        dx,
        dy,
        dz,
        value,
        unit,
        notes,
    ]


def export_fig1() -> None:
    pca = np.load(ROOT / "data" / "crbn_pca.npz", allow_pickle=False)
    anm = np.load(ROOT / "data" / "crbn_anm_modes.npz", allow_pickle=False)
    dv = np.load(ROOT / "data" / "pca_diffvec.npz", allow_pickle=False)
    labels = [str(value) for value in dv["labels"]]
    log = {
        record["pdb"].upper(): record["global_state"]
        for record in csv.DictReader((ROOT / "data" / "crbn_curation_log.csv").open())
    }
    pc1 = np.asarray(pca["pc1_scores"], dtype=float)
    n_ca = pca["mean"].reshape(-1, 3).shape[0]
    pc2 = np.asarray(pca["pc2_scores"], dtype=float) / np.sqrt(n_ca)
    open_mask = np.asarray(pca["open_mask"], dtype=bool)
    variance = np.asarray(pca["variance_ratio"], dtype=float)
    cumulative = np.asarray(anm["cum_overlap"], dtype=float)
    rmsip = float(anm["rmsip"])
    if not (len(labels) == len(pc1) == len(pc2) == len(open_mask)):
        raise ValueError("Fig1 PCA labels, scores and open mask differ in length")

    rows: list[list[object]] = []
    for pdb, score1, score2, is_open in zip(labels, pc1, pc2, open_mask):
        rows.append(
            row(
                "a",
                "PCA score",
                log[pdb.upper()],
                pdb,
                x=f"{score1:.12f}",
                y=f"{score2:.12f}",
                unit="Angstrom RMSD-scaled coordinate",
                notes="open" if is_open else "closed",
            )
        )
    threshold = float(np.sort(pc1)[::-1][4:6].mean())
    rows.append(row("a", "open-closed visual divider", x=f"{threshold:.12f}", unit="PC1 coordinate"))
    for index, fraction in enumerate(variance[:10], start=1):
        rows.append(
            row(
                "b",
                "variance spectrum",
                "individual",
                index=index,
                x=index,
                value=f"{100.0 * fraction:.12f}",
                unit="percent coordinate variance",
            )
        )
        rows.append(
            row(
                "b",
                "variance spectrum",
                "cumulative",
                index=index,
                x=index,
                value=f"{100.0 * variance[:index].sum():.12f}",
                unit="percent coordinate variance",
            )
        )
    for index, value in enumerate(cumulative[:10], start=1):
        rows.append(
            row(
                "c",
                "cumulative ANM projection",
                index=index,
                x=index,
                value=f"{value:.12f}",
                unit="projection norm",
            )
        )
    rows.append(row("c", "ANM-PCA RMSIP reference", value=f"{rmsip:.12f}", unit="RMSIP"))
    closed_mean = float(pc1[~open_mask].mean())
    open_mean = float(pc1[open_mask].mean())
    transition = (pc1 - closed_mean) / (open_mean - closed_mean)
    for pdb, value, is_open in zip(labels, transition, open_mask):
        rows.append(
            row(
                "d",
                "transition coordinate",
                "open" if is_open else "closed",
                pdb,
                value=f"{value:.12f}",
                unit="closed-mean to open-mean coordinate",
            )
        )
    band_raw = json.loads((ROOT / "data" / "window_sensitivity.json").read_text(encoding="utf-8"))[
        "empty_middle"
    ]["a_paper_rule"]["band_15_85_pct"]
    band = [
        (float(bound) - closed_mean) / (open_mean - closed_mean)
        for bound in band_raw
    ]
    rows.append(
        row(
            "d",
            "empty-middle band",
            x=f"{float(band[0]):.12f}",
            y=f"{float(band[1]):.12f}",
            unit="transition coordinate",
        )
    )
    write_rows("Fig1", rows)


def anm_hessian(coords: np.ndarray, cutoff: float) -> np.ndarray:
    n = len(coords)
    hessian = np.zeros((3 * n, 3 * n))
    for i in range(n):
        displacement = coords - coords[i]
        distances = np.linalg.norm(displacement, axis=1)
        for j in range(i + 1, n):
            if 1e-6 < distances[j] <= cutoff:
                block = np.outer(displacement[j], displacement[j]) / distances[j] ** 2
                hessian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -block
                hessian[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = -block
                hessian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += block
                hessian[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += block
    return hessian


def mode_overlaps(coords: np.ndarray, difference: np.ndarray, cutoff: float = 15.0) -> np.ndarray:
    values, vectors = np.linalg.eigh(anm_hessian(coords, cutoff))
    vectors = vectors[:, values > 1e-9][:, :10]
    return np.asarray([abs(vectors[:, mode] @ difference) for mode in range(vectors.shape[1])])


def parse_nmd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = mode = residue_numbers = None
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "coordinates":
            coordinates = np.asarray(tokens[1:], dtype=float).reshape(-1, 3)
        elif tokens[0] == "mode" and mode is None:
            mode = np.asarray(tokens[3:], dtype=float).reshape(-1, 3)
        elif tokens[0] in ("resnums", "resids"):
            residue_numbers = np.asarray(tokens[1:], dtype=int)
    if coordinates is None or mode is None:
        raise ValueError(f"missing coordinates or first mode in {path}")
    if residue_numbers is None:
        residue_numbers = np.asarray(
            [
                int(record["resnum"])
                for record in csv.DictReader((ROOT / "data" / "crbn_residue_fluctuations.csv").open())
            ]
        )
    return coordinates, mode, residue_numbers


def export_fig2() -> None:
    anm = np.load(ROOT / "data" / "crbn_anm_modes.npz", allow_pickle=False)
    open_spectrum = np.abs(np.asarray(anm["anm_diff_overlap"], dtype=float)[:10])
    ensemble = np.load(ROOT / "data" / "crbn_ensemble.ens.npz", allow_pickle=False)
    conformers = np.asarray(ensemble["_confs"], dtype=float)
    labels = [str(value) for value in ensemble["_labels"]]
    dv = np.load(ROOT / "data" / "pca_diffvec.npz", allow_pickle=False)
    open_ids = {str(label) for label, state in zip(dv["labels"], dv["open_mask"]) if state}
    mask = np.asarray([label in open_ids for label in labels])
    difference = conformers[mask].mean(0) - conformers[~mask].mean(0)
    difference = (difference / np.linalg.norm(difference)).ravel()
    open_endpoint = mode_overlaps(conformers[labels.index("8CVP")], difference)
    closed_endpoint = mode_overlaps(conformers[labels.index("5FQD")], difference)

    rows: list[list[object]] = []
    for index, value in enumerate(open_spectrum, start=1):
        rows.append(
            row(
                "a",
                "open-state ANM mode spectrum",
                index=index,
                x=index,
                value=f"{value:.12f}",
                unit="absolute directional overlap",
            )
        )
    for panel, stem in (("b", "crbn_anm_mode1.nmd"), ("c", "crbn_pca_modes.nmd")):
        coordinates, mode, residue_numbers = parse_nmd(ROOT / "data" / stem)
        for index, (residue, coord, vector) in enumerate(
            zip(residue_numbers, coordinates, mode), start=1
        ):
            rows.append(
                row(
                    panel,
                    "porcupine coordinate and vector",
                    "ANM mode 1" if panel == "b" else "PCA PC1",
                    int(residue),
                    index,
                    *(f"{value:.12f}" for value in coord),
                    *(f"{value:.12f}" for value in vector),
                    unit="NMD coordinate/vector units",
                )
            )
    for series, values in (("from open (8CVP)", open_endpoint), ("from closed (5FQD)", closed_endpoint)):
        for index, value in enumerate(values, start=1):
            rows.append(
                row(
                    "d",
                    "endpoint ANM mode spectrum",
                    series,
                    index=index,
                    x=index,
                    value=f"{value:.12f}",
                    unit="absolute directional overlap",
                )
            )
    write_rows("Fig2", rows)


def export_fig3() -> None:
    anm_data = np.load(ROOT / "data" / "crbn_anm_modes.npz", allow_pickle=False)
    vectors = np.asarray(anm_data["gnm_eigvecs"], dtype=float)
    values = np.asarray(anm_data["gnm_eigvals"], dtype=float)
    covariance = (vectors / values) @ vectors.T
    diagonal = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(diagonal, diagonal)
    records = list(csv.DictReader((ROOT / "data" / "crbn_residue_fluctuations.csv").open()))
    residues = np.asarray([int(record["resnum"]) for record in records])
    intrinsic = np.asarray([float(record["anm_sqfluct"]) for record in records])
    edges = np.zeros(len(residues), dtype=bool)
    for index in range(1, len(residues)):
        if residues[index] - residues[index - 1] > 1:
            edges[max(0, index - 2) : index + 2] = True
    intrinsic = intrinsic / intrinsic[~edges].max()
    experimental = np.asarray([float(record["pca_sqfluct"]) for record in records])
    experimental = experimental / experimental.max()

    rows: list[list[object]] = []
    for row_index, residue_i in enumerate(residues):
        for column_index, residue_j in enumerate(residues):
            rows.append(
                row(
                    "a",
                    "GNM cross-correlation",
                    identifier=int(residue_i),
                    index=int(residue_j),
                    x=int(residue_i),
                    y=int(residue_j),
                    value=f"{correlation[row_index, column_index]:.12f}",
                    unit="correlation",
                )
            )
    for residue, value in zip(residues, intrinsic):
        rows.append(
            row(
                "b",
                "normalized square fluctuation",
                "ANM intrinsic",
                int(residue),
                x=int(residue),
                value=f"{value:.12f}",
                unit="artefact-free maximum normalized",
            )
        )
    for residue, value in zip(residues, experimental):
        rows.append(
            row(
                "b",
                "normalized square fluctuation",
                "PCA experimental",
                int(residue),
                x=int(residue),
                value=f"{value:.12f}",
                unit="maximum normalized",
            )
        )
    geometry = json.loads((ROOT / "data" / "hinge_geometry.json").read_text(encoding="utf-8"))
    for residue in geometry["axis_proximal_boundary_residues"]:
        rows.append(
            row(
                "b",
                "screw-axis-proximal boundary residue",
                "endpoint Kabsch geometry",
                int(residue),
                x=int(residue),
                value=f"{geometry['boundary_axis_distance_A'][str(residue)]:.12f}",
                unit="Angstrom from screw axis",
            )
        )
    write_rows("Fig3", rows)


def percentile(residues: np.ndarray, values: np.ndarray, selected: list[int]) -> list[float]:
    return [100.0 * np.mean(values <= values[residues == residue][0]) for residue in selected]


def export_fig4() -> None:
    annotated_residues = [378, 380, 386]
    contact_residues = [377, 378, 379, 380, 386, 400, 402]
    zinc_residues = [323, 326, 391, 394]
    modes = np.load(ROOT / "data" / "crbn_anm_modes.npz", allow_pickle=False)
    residues = np.asarray(modes["resnums"])
    eigenvectors = np.asarray(modes["anm_eigvecs"], dtype=float)
    eigenvalues = np.asarray(modes["anm_eigvals"], dtype=float)
    square_fluctuation = np.zeros(len(residues))
    for index in range(10):
        vector = eigenvectors[:, index].reshape(-1, 3)
        square_fluctuation += (vector**2).sum(1) / eigenvalues[index]
    square_fluctuation /= square_fluctuation.max()
    records = list(csv.DictReader((ROOT / "data" / "crbn_residue_fluctuations.csv").open()))
    analysis_residues = np.asarray([int(record["resnum"]) for record in records])
    profiles = {
        "ANM": np.asarray([float(record["anm_sqfluct"]) for record in records]),
        "PCA": np.asarray([float(record["pca_sqfluct"]) for record in records]),
    }

    rows: list[list[object]] = []
    for residue, value in zip(residues, square_fluctuation):
        if 318 <= residue <= 424:
            rows.append(
                row(
                    "a",
                    "TBD ANM square fluctuation",
                    identifier=int(residue),
                    x=int(residue),
                    value=f"{value:.12f}",
                    unit="maximum normalized",
                )
            )
    for method, values in profiles.items():
        for group, selected in (
            ("UniProt ligand annotations", annotated_residues),
            ("5FQD LVY contacts <=4.5 A", contact_residues),
            ("zinc site", zinc_residues),
        ):
            group_values = percentile(analysis_residues, values, selected)
            for residue, value in zip(selected, group_values):
                rows.append(
                    row(
                        "b",
                        "mobility percentile",
                        f"{method} {group}",
                        residue,
                        x=method,
                        value=f"{value:.12f}",
                        unit="percentile among 269 residues",
                    )
                )
            rows.append(
                row(
                    "b",
                    "group mean mobility percentile",
                    f"{method} {group}",
                    x=method,
                    value=f"{np.mean(group_values):.12f}",
                    unit="mean percentile",
                )
            )
    rows.append(
        row(
            "c",
            "structural raster input",
            value="figures/panels/render_closed_pocket.png",
            unit="file path",
            notes=(
                "closed-state TBD colored by ANM mobility with lenalidomide, the three "
                "UniProt ligand annotations, and zinc"
            ),
        )
    )
    write_rows("Fig4", rows)


def exact_null_density(x: np.ndarray, dimension: int) -> np.ndarray:
    beta = 0.5 * (dimension - 1)
    normalizer = 2.0 * math.exp(
        math.lgamma(0.5 + beta) - math.lgamma(0.5) - math.lgamma(beta)
    )
    return normalizer * np.power(np.clip(1.0 - x**2, 0.0, None), beta - 1.0)


def export_fig5() -> None:
    robustness = json.loads((ROOT / "data" / "anm_robustness.json").read_text(encoding="utf-8"))
    null = json.loads((ROOT / "data" / "anm_null_significance.json").read_text(encoding="utf-8"))
    rigid = require_rigid_null_schema(
        json.loads((ROOT / "data" / "assembly_rigid_null.json").read_text(encoding="utf-8"))
    )
    rows: list[list[object]] = []
    endpoints = robustness["open_set"] + robustness["closed_endpoints"]
    for endpoint in endpoints:
        record = robustness["table"][endpoint]["15.0"]
        rows.append(
            row(
                "a",
                "endpoint overlap",
                "open" if endpoint in robustness["open_set"] else "closed",
                endpoint,
                value=f"{float(record['mode1_overlap']):.12f}",
                unit="absolute directional overlap",
                notes=f"best={float(record['best_overlap']):.12f}; best_mode={int(record['best_mode_rank'])}",
            )
        )
    x_values = np.linspace(0.0, 1.0, 1001)
    for key, label in (
        ("two_block", "2-lobe"),
        ("three_block", "3-domain"),
        ("bond_length_preserving_boundary", "bond-length"),
        ("equal_displacement_boundary", "equal displacement"),
    ):
        record = rigid[key]
        density = exact_null_density(x_values, int(record["internal_dim"]))
        for index, (x_value, density_value) in enumerate(zip(x_values, density), start=1):
            rows.append(
                row(
                    "b",
                    "matched-subspace exact null density",
                    label,
                    index=index,
                    x=f"{x_value:.12f}",
                    value=f"{density_value:.12f}",
                    unit="probability density",
                    notes=(
                        f"observed={float(record['observed_direction_cosine_in_subspace']):.12f}; "
                        f"p_exact={float(record['p_exact']):.12f}; dim={int(record['internal_dim'])}"
                    ),
                )
            )
    cutoffs = [float(value) for value in robustness["cutoffs"]]
    for endpoint in robustness["open_set"]:
        for cutoff_raw, cutoff in zip(robustness["cutoffs"], cutoffs):
            rows.append(
                row(
                    "c",
                    "cutoff sensitivity",
                    endpoint,
                    x=f"{cutoff:.1f}",
                    value=f"{float(robustness['table'][endpoint][str(cutoff_raw)]['mode1_overlap']):.12f}",
                    unit="absolute directional overlap",
                )
            )
    for cutoff_raw, cutoff in zip(robustness["cutoffs"], cutoffs):
        mean_value = np.mean(
            [
                robustness["table"][endpoint][str(cutoff_raw)]["mode1_overlap"]
                for endpoint in robustness["open_set"]
            ]
        )
        rows.append(
            row(
                "c",
                "cutoff sensitivity",
                "open mean",
                x=f"{cutoff:.1f}",
                value=f"{float(mean_value):.12f}",
                unit="absolute directional overlap",
            )
        )
    for series, record in (
        ("drop one closed", null["leave_one_closed_out"]),
        ("drop one open", null["leave_one_open_out"]),
    ):
        rows.append(
            row(
                "d",
                "leave-one-out summary",
                series,
                value=f"{float(record['mean']):.12f}",
                unit="absolute directional overlap",
                notes=f"min={float(record['min']):.12f}; max={float(record['max']):.12f}",
            )
        )
    rows.append(
        row(
            "d",
            "full-ensemble reference",
            value=f"{float(null['observed_mode1_overlap']):.12f}",
            unit="absolute directional overlap",
        )
    )
    write_rows("Fig5", rows)


def main() -> None:
    export_fig1()
    export_fig2()
    export_fig3()
    export_fig4()
    export_fig5()


if __name__ == "__main__":
    main()
