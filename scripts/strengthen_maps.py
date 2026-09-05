#!/usr/bin/env python3
"""Acquire and inspect CRBN cryo-EM maps for the CSBJ strengthening analysis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from typing import Any
import urllib.request

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analysis_contracts import atomic_write_json, atomic_write_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "results" / "strengthening"
OUT_ROOT = DEFAULT_OUT_ROOT
DATA_ROOT = OUT_ROOT / "data" / "maps"
ANALYSIS_ROOT = OUT_ROOT / "analysis" / "maps"
API_TEMPLATE = "https://www.ebi.ac.uk/emdb/api/entry/EMD-{emdb_id}"
FTP_TEMPLATE = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_id}/{folder}/{file_name}"
DIPPON_NATURE_URL = "https://www.nature.com/articles/s41586-025-09994-w"
CHIMERAX_DOWNLOAD_URL = "https://www.cgl.ucsf.edu/chimerax/download.html"


@dataclass(frozen=True)
class EntrySpec:
    emdb_id: str
    state: str
    role: str
    analysis_scope: str
    primary_source: str


DIPPON_ENTRIES = (
    EntrySpec("70776", "open", "open-consensus-refine", "metadata+maps", DIPPON_NATURE_URL),
    EntrySpec("70777", "open", "open-focused-refine", "metadata+maps", DIPPON_NATURE_URL),
    EntrySpec("70778", "open", "open-lenalidomide", "metadata+maps", DIPPON_NATURE_URL),
    EntrySpec("70781", "intermediate", "intermediate-nu", "metadata+maps", DIPPON_NATURE_URL),
    EntrySpec("70782", "closed", "closed-consensus-refine", "metadata+maps", DIPPON_NATURE_URL),
    EntrySpec("70783", "closed", "closed-focused-refine", "metadata+maps", DIPPON_NATURE_URL),
    EntrySpec("70784", "closed", "closed-composite", "metadata+maps", DIPPON_NATURE_URL),
)

WATSON_ENTRIES = tuple(
    EntrySpec(emdb_id, "historical", "watson-historical", "metadata+quality-only", "EMDB API")
    for emdb_id in (
        "27012",
        "27234",
        "27235",
        "27236",
        "27237",
        "27238",
        "27239",
        "27240",
        "27241",
        "27242",
    )
)

ENTRY_SPECS = {spec.emdb_id: spec for spec in (*DIPPON_ENTRIES, *WATSON_ENTRIES)}
DEFAULT_METADATA_IDS = tuple(ENTRY_SPECS)
DEFAULT_DOWNLOAD_IDS = ("70781",)
PLAN_D_FIT_IDS = ("70776", "70781", "70782")
MAP_KINDS = {"primary_map", "additional_map", "half_map", "mask"}
PLAN_D_SEED = 20260905
PLAN_D_SEARCH_PLACEMENTS = 100
PLAN_D_LOW_FREQUENCY_MODE_COUNT = 10
PLAN_D_FIT_RESOLUTION_ANGSTROM = {"70776": 2.52, "70781": 2.71, "70782": 2.59}
PLAN_D_DOMAINS = {"NTD+HB": (77, 317), "TBD": (318, 426)}
PLAN_D_WINDOW_CSV = ROOT / "data" / "crbn_residue_window.csv"
PLAN_D_PCA_NPZ = ROOT / "data" / "crbn_pca.npz"
PLAN_D_MODES_NPZ = ROOT / "data" / "crbn_anm_modes.npz"
PLAN_D_TEMPLATES = {
    "open_8cvp_crbn": ROOT / "render" / "open_8cvp_assembly.pdb",
    "closed_5fqd_crbn": ROOT / "render" / "closed_5fqd.pdb",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def strict_json_loads(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("EMDB API response must be a JSON object")
    return value


def fetch_json(url: str, *, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return strict_json_loads(response.read())


def load_or_fetch_metadata(
    emdb_id: str,
    *,
    output_dir: Path = DATA_ROOT,
    offline: bool = False,
) -> dict[str, Any]:
    path = output_dir / f"EMD-{emdb_id}" / "metadata_api.json"
    if offline:
        return strict_json_loads(path.read_bytes())
    value = fetch_json(API_TEMPLATE.format(emdb_id=emdb_id))
    atomic_write_json(path, value, sort_keys=True)
    return value


def resolved_output_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def value_of(value: Any) -> Any:
    if isinstance(value, dict) and "valueOf_" in value:
        return value["valueOf_"]
    return value


def float_value(value: Any) -> float | None:
    raw = value_of(value)
    if raw is None:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def artifact_url(emdb_id: str, kind: str, file_name: str) -> str:
    if kind == "primary_map":
        folder = "map"
    elif kind in {"additional_map", "half_map"}:
        folder = "other"
    elif kind == "mask":
        folder = "masks"
    else:
        raise ValueError(f"unsupported artifact kind: {kind}")
    return FTP_TEMPLATE.format(emdb_id=emdb_id, folder=folder, file_name=file_name)


def discover_artifacts(emdb_id: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    primary = metadata.get("map")
    if isinstance(primary, dict) and primary.get("file"):
        artifacts.append(
            {
                "kind": "primary_map",
                "emdb_id": f"EMD-{emdb_id}",
                "file_name": str(primary["file"]),
                "annotation": primary.get("annotation_details"),
                "pixel_spacing_angstrom": pixel_spacing(primary),
                "url": artifact_url(emdb_id, "primary_map", str(primary["file"])),
            }
        )

    interpretation = metadata.get("interpretation", {})
    if not isinstance(interpretation, dict):
        interpretation = {}
    for item in ensure_list(interpretation.get("additional_map_list", {}).get("additional_map")):
        if isinstance(item, dict) and item.get("file"):
            file_name = str(item["file"])
            artifacts.append(
                {
                    "kind": "additional_map",
                    "emdb_id": f"EMD-{emdb_id}",
                    "file_name": file_name,
                    "annotation": item.get("annotation_details"),
                    "pixel_spacing_angstrom": pixel_spacing(item),
                    "url": artifact_url(emdb_id, "additional_map", file_name),
                }
            )
    for index, item in enumerate(
        ensure_list(interpretation.get("half_map_list", {}).get("half_map")), start=1
    ):
        if isinstance(item, dict) and item.get("file"):
            file_name = str(item["file"])
            artifacts.append(
                {
                    "kind": "half_map",
                    "emdb_id": f"EMD-{emdb_id}",
                    "part": index,
                    "file_name": file_name,
                    "annotation": item.get("annotation_details"),
                    "pixel_spacing_angstrom": pixel_spacing(item),
                    "url": artifact_url(emdb_id, "half_map", file_name),
                }
            )
    for index, item in enumerate(
        ensure_list(interpretation.get("segmentation_list", {}).get("segmentation")), start=1
    ):
        if isinstance(item, dict) and item.get("file"):
            file_name = str(item["file"])
            artifacts.append(
                {
                    "kind": "mask",
                    "emdb_id": f"EMD-{emdb_id}",
                    "part": index,
                    "file_name": file_name,
                    "annotation": item.get("details") or item.get("annotation_details"),
                    "url": artifact_url(emdb_id, "mask", file_name),
                }
            )
    return artifacts


def pixel_spacing(map_item: dict[str, Any]) -> dict[str, float | None]:
    spacing = map_item.get("pixel_spacing", {})
    if not isinstance(spacing, dict):
        return {"x": None, "y": None, "z": None}
    return {axis: float_value(spacing.get(axis)) for axis in ("x", "y", "z")}


def entry_summary(emdb_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    admin = metadata.get("admin", {})
    if not isinstance(admin, dict):
        admin = {}
    reconstruction = final_reconstruction(metadata)
    revision = admin.get("revision_history", {}).get("revision", [])
    revisions = ensure_list(revision)
    latest_version = None
    if revisions and isinstance(revisions[-1], dict):
        latest_version = revisions[-1].get("version")
    spec = ENTRY_SPECS[emdb_id]
    return {
        "emdb_id": f"EMD-{emdb_id}",
        "state": spec.state,
        "role": spec.role,
        "analysis_scope": spec.analysis_scope,
        "primary_source": spec.primary_source,
        "api_url": API_TEMPLATE.format(emdb_id=emdb_id),
        "title": admin.get("title"),
        "keywords": admin.get("keywords"),
        "status": admin.get("current_status", {}).get("code", {}).get("valueOf_"),
        "status_date": admin.get("current_status", {}).get("date"),
        "deposition_date": admin.get("key_dates", {}).get("deposition"),
        "map_release": admin.get("key_dates", {}).get("map_release"),
        "latest_revision_version": latest_version,
        "latest_revision_date": revisions[-1].get("date") if revisions and isinstance(revisions[-1], dict) else None,
        "resolution_angstrom": float_value(reconstruction.get("resolution")),
        "resolution_method": reconstruction.get("resolution_method"),
        "map_dimensions": metadata.get("map", {}).get("dimensions"),
        "pixel_spacing_angstrom": pixel_spacing(metadata.get("map", {})),
        "artifacts": discover_artifacts(emdb_id, metadata),
    }


def final_reconstruction(metadata: dict[str, Any]) -> dict[str, Any]:
    determinations = metadata.get("structure_determination_list", {}).get("structure_determination")
    for determination in ensure_list(determinations):
        if not isinstance(determination, dict):
            continue
        for processing in ensure_list(determination.get("image_processing")):
            if isinstance(processing, dict) and isinstance(
                processing.get("final_reconstruction"), dict
            ):
                return processing["final_reconstruction"]
    return {}


def download_artifact(
    artifact: dict[str, Any],
    destination: Path,
    *,
    timeout: int = 180,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = str(artifact["url"])
    temporary: Path | None = None
    hasher = hashlib.sha256()
    bytes_written = 0
    started = utc_now()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    handle.write(chunk)
                    bytes_written += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        **artifact,
        "local_path": str(destination.relative_to(ROOT)),
        "download_started_utc": started,
        "download_finished_utc": utc_now(),
        "bytes": bytes_written,
        "sha256": hasher.hexdigest(),
    }


def existing_artifact_record(artifact: dict[str, Any], destination: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {
        **artifact,
        "local_path": str(destination.relative_to(ROOT)),
        "download_started_utc": None,
        "download_finished_utc": None,
        "bytes": destination.stat().st_size,
        "sha256": hasher.hexdigest(),
        "cached": True,
    }


def opener(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def read_mrc_header(path: Path) -> dict[str, Any]:
    with opener(path) as handle:
        header = handle.read(1024)
    if len(header) < 1024:
        raise ValueError(f"{path} is too short to contain an MRC header")
    nx, ny, nz, mode = struct.unpack("<4i", header[:16])
    mx, my, mz = struct.unpack("<3i", header[28:40])
    cella = struct.unpack("<3f", header[40:52])
    cellb = struct.unpack("<3f", header[52:64])
    dmin, dmax, dmean = struct.unpack("<3f", header[76:88])
    nsymbt = struct.unpack("<i", header[92:96])[0]
    origin = struct.unpack("<3f", header[196:208])
    stamp = header[208:212].decode("ascii", errors="replace")
    rms = struct.unpack("<f", header[216:220])[0]
    nlabels = max(0, min(struct.unpack("<i", header[220:224])[0], 10))
    labels = [
        header[224 + index * 80 : 304 + index * 80].decode("ascii", errors="replace").strip()
        for index in range(nlabels)
    ]
    dimensions = [int(nx), int(ny), int(nz)]
    mode_name = {0: "int8", 1: "int16", 2: "float32", 6: "uint16"}.get(mode, f"mode-{mode}")
    voxel = [
        float(cella[index]) / [mx, my, mz][index] if [mx, my, mz][index] else None
        for index in range(3)
    ]
    return {
        "dimensions": dimensions,
        "mode": int(mode),
        "mode_name": mode_name,
        "grid": [int(mx), int(my), int(mz)],
        "cell_angstrom": [float(value) for value in cella],
        "cell_angles_deg": [float(value) for value in cellb],
        "voxel_spacing_angstrom": voxel,
        "header_density_min": finite_or_none(dmin),
        "header_density_max": finite_or_none(dmax),
        "header_density_mean": finite_or_none(dmean),
        "header_density_rms": finite_or_none(rms),
        "symmetry_bytes": int(nsymbt),
        "origin": [float(value) for value in origin],
        "map_stamp": stamp,
        "labels": labels,
    }


def finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def dtype_for_mode(mode: int) -> np.dtype:
    mapping = {
        0: np.dtype("<i1"),
        1: np.dtype("<i2"),
        2: np.dtype("<f4"),
        6: np.dtype("<u2"),
    }
    if mode not in mapping:
        raise ValueError(f"unsupported MRC mode for density scan: {mode}")
    return mapping[mode]


def read_mrc_array(path: Path) -> np.ndarray:
    header = read_mrc_header(path)
    shape = tuple(int(value) for value in header["dimensions"])
    dtype = dtype_for_mode(int(header["mode"]))
    skip = 1024 + int(header["symmetry_bytes"])
    with opener(path) as handle:
        raw = handle.read()
    payload = raw[skip:]
    expected = int(np.prod(shape)) * dtype.itemsize
    if len(payload) < expected:
        raise ValueError(f"{path} contains {len(payload)} data bytes; expected at least {expected}")
    array = np.frombuffer(payload[:expected], dtype=dtype)
    return array.reshape((shape[2], shape[1], shape[0]))


def density_summary(path: Path) -> dict[str, Any]:
    header = read_mrc_header(path)
    values = read_mrc_array(path).astype(np.float64, copy=False).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{path} contains no finite density values")
    quantiles = np.quantile(finite, [0.001, 0.01, 0.5, 0.99, 0.999])
    absolute = np.abs(finite)
    return {
        "header": header,
        "computed_voxels": int(finite.size),
        "computed_min": float(finite.min()),
        "computed_max": float(finite.max()),
        "computed_mean": float(finite.mean()),
        "computed_std": float(finite.std()),
        "computed_abs_mean": float(absolute.mean()),
        "computed_nonzero_fraction": float(np.count_nonzero(finite) / finite.size),
        "computed_quantiles": {
            "q0.001": float(quantiles[0]),
            "q0.01": float(quantiles[1]),
            "q0.5": float(quantiles[2]),
            "q0.99": float(quantiles[3]),
            "q0.999": float(quantiles[4]),
        },
        "header_mean_delta": delta(header["header_density_mean"], float(finite.mean())),
        "header_rms_delta": delta(header["header_density_rms"], float(finite.std())),
    }


def delta(left: float | None, right: float) -> float | None:
    if left is None:
        return None
    return float(abs(left - right))


def sampled_half_correlation(first: Path, second: Path, *, max_points: int = 2_000_000) -> dict[str, Any]:
    left = read_mrc_array(first).astype(np.float32, copy=False).ravel()
    right = read_mrc_array(second).astype(np.float32, copy=False).ravel()
    if left.shape != right.shape:
        raise ValueError(f"half-map shape mismatch: {left.shape} != {right.shape}")
    stride = max(1, left.size // max_points)
    left_sample = left[::stride].astype(np.float64, copy=False)
    right_sample = right[::stride].astype(np.float64, copy=False)
    mask = np.isfinite(left_sample) & np.isfinite(right_sample)
    if int(mask.sum()) < 3:
        raise ValueError("not enough finite sampled half-map voxels for correlation")
    left_centered = left_sample[mask] - float(left_sample[mask].mean())
    right_centered = right_sample[mask] - float(right_sample[mask].mean())
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    corr = float(np.dot(left_centered, right_centered) / denom) if denom else None
    return {
        "sampled_points": int(mask.sum()),
        "stride": int(stride),
        "pearson_r": corr,
    }


def find_chimerax() -> dict[str, Any]:
    candidates = [
        shutil.which("ChimeraX"),
        shutil.which("chimerax"),
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
        "/Applications/UCSF ChimeraX.app/Contents/MacOS/ChimeraX",
    ]
    existing = [str(Path(path)) for path in candidates if path and Path(path).exists()]
    return {
        "available": bool(existing),
        "candidates": existing,
        "install_status": "installed" if existing else "blocked_by_license_acceptance",
        "license_acceptance_required": not bool(existing),
        "supported_installation_url": CHIMERAX_DOWNLOAD_URL,
        "note": (
            "ChimeraX rigid-fit analysis requires UCSF ChimeraX from its supported "
            "distribution. The UCSF macOS download endpoint presents an explicit "
            "license-acceptance form before the DMG download."
        ),
    }


def fit_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    orientation = metrics.get("relative_orientation_deg")
    coordinate = metrics.get("normalized_structural_coordinate")
    density = metrics.get("density_support")
    if not isinstance(orientation, (int, float)) or not isinstance(coordinate, (int, float)):
        return {
            "quantitative_fit_allowed": False,
            "decision": "pending",
            "reasons": [
                "relative orientation and normalized structural coordinate require a completed fit"
            ],
        }
    stable = (
        float(orientation) <= 10.0
        and abs(float(coordinate)) <= 0.1
        and density == "supported"
    )
    reasons = []
    if float(orientation) > 10.0:
        reasons.append("relative orientation differs by more than 10 degrees")
    if abs(float(coordinate)) > 0.1:
        reasons.append("normalized structural coordinate differs by more than 0.1")
    if density != "supported":
        reasons.append("density support is not sufficient for quantitative fitting")
    return {
        "quantitative_fit_allowed": stable,
        "decision": "quantitative" if stable else "qualitative-only",
        "reasons": reasons,
    }


def quality_decision(
    *,
    chimerax: dict[str, Any],
    density_checks: dict[str, Any],
    half_map_correlations: dict[str, Any],
    focus_emdb_id: str = "EMD-70781",
) -> dict[str, Any]:
    reasons: list[str] = []
    global_map_qa = "supported"
    focus_checks = density_checks.get(focus_emdb_id, {})
    primary = focus_checks.get("primary_map") if isinstance(focus_checks, dict) else None
    if not isinstance(primary, dict):
        global_map_qa = "not-tested"
        reasons.append("primary map density was not inspected")
    else:
        std = primary.get("computed_std")
        nonzero = primary.get("computed_nonzero_fraction")
        if not isinstance(std, (int, float)) or float(std) <= 0:
            global_map_qa = "weak"
            reasons.append("primary map has non-positive computed density standard deviation")
        if not isinstance(nonzero, (int, float)) or float(nonzero) < 0.01:
            global_map_qa = "weak"
            reasons.append("primary map has too few nonzero voxels")
    focus_half_correlation = half_map_correlations.get(focus_emdb_id)
    if focus_half_correlation is None:
        reasons.append(f"{focus_emdb_id} half-map correlation was not measured")
    elif focus_half_correlation.get("pearson_r") is None:
        reasons.append(f"{focus_emdb_id} half-map correlation is undefined")
    reasons.append(
        "global map and whole-volume half-map QA do not establish local CRBN density support"
    )
    if not chimerax.get("available"):
        reasons.append(
            "ChimeraX installation is pending because the official download requires "
            "explicit license acceptance"
        )
    gate = fit_gate(
        {
            "relative_orientation_deg": None,
            "normalized_structural_coordinate": None,
            "density_support": "not-assessed",
        }
    )
    return {
        "density_support": "not-assessed",
        "global_map_qa": global_map_qa,
        "crbn_local_density_support": "not-assessed-pending-domain-fit",
        "focus_emdb_id": focus_emdb_id,
        "half_map_correlation": focus_half_correlation,
        "half_map_correlations": half_map_correlations,
        "chimerax": chimerax,
        "fit_stability_gate": gate,
        "overall_use": "fit-pending" if gate["decision"] == "pending" else gate["decision"],
        "reasons": reasons + gate["reasons"],
        "claim_boundary": (
            "Functional allosteric-site mapping remains retrospective; fitted coordinates "
            "must not be described as experimental atomic models unless the ChimeraX half-map, "
            "local CRBN density, and template-stability gates pass."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolved_output_root(args.output_dir)
    data_root = output_root / "data" / "maps"
    analysis_root = output_root / "analysis" / "maps"
    data_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    retrieved_at = utc_now()
    summaries = []
    metadata: dict[str, dict[str, Any]] = {}
    for emdb_id in args.metadata_ids:
        entry = load_or_fetch_metadata(emdb_id, output_dir=data_root, offline=args.offline)
        metadata[emdb_id] = entry
        summary = entry_summary(emdb_id, entry)
        summary["retrieval_utc"] = retrieved_at
        summaries.append(summary)

    downloaded: list[dict[str, Any]] = []
    selected_kinds = set(args.artifact_kinds)
    for emdb_id in args.download_ids:
        entry_dir = data_root / f"EMD-{emdb_id}"
        for artifact in discover_artifacts(emdb_id, metadata[emdb_id]):
            if artifact["kind"] not in selected_kinds:
                continue
            destination = entry_dir / str(artifact["file_name"])
            if args.offline or (destination.exists() and not args.refresh_downloads):
                if not destination.exists():
                    continue
                downloaded.append(existing_artifact_record(artifact, destination))
            else:
                downloaded.append(download_artifact(artifact, destination))

    density_checks: dict[str, dict[str, Any]] = {}
    for record in downloaded:
        if record["kind"] not in MAP_KINDS:
            continue
        local = ROOT / record["local_path"]
        key = record["kind"]
        if record["kind"] == "half_map":
            key = f"half_map_{record.get('part')}"
        elif record["kind"] == "mask":
            key = f"mask_{record.get('part')}"
        emdb_key = str(record["emdb_id"])
        density_checks.setdefault(emdb_key, {})[key] = density_summary(local)

    half_map_correlations = {}
    for emdb_key in sorted({str(record["emdb_id"]) for record in downloaded}):
        half_records = [
            record
            for record in downloaded
            if record["kind"] == "half_map" and record["emdb_id"] == emdb_key
        ]
        half_records.sort(key=lambda record: int(record.get("part") or 0))
        if len(half_records) >= 2:
            half_map_correlations[emdb_key] = sampled_half_correlation(
                ROOT / half_records[0]["local_path"],
                ROOT / half_records[1]["local_path"],
            )

    chimerax = find_chimerax()
    result = {
        "analysis": "CRBN cryoEM acquisition and analysis D",
        "output_root": display_path(output_root),
        "retrieval_utc": retrieved_at,
        "offline": bool(args.offline),
        "source_urls": {
            "dippon_nature": DIPPON_NATURE_URL,
            "emdb_api_template": API_TEMPLATE,
            "emdb_ftp_template": FTP_TEMPLATE,
        },
        "entries": summaries,
        "downloaded_artifacts": downloaded,
        "density_checks": density_checks,
        "quality_decision": quality_decision(
            chimerax=chimerax,
            density_checks=density_checks,
            half_map_correlations=half_map_correlations,
        ),
    }
    atomic_write_json(analysis_root / "strengthen_maps_summary.json", result, sort_keys=True)
    atomic_write_text(analysis_root / "strengthen_maps_report.md", markdown_report(result))
    write_chimerax_assets(output_root, analysis_root)
    return result


CHIMERAX_PLAN_D_RUNNER = '"""Run CRBN Plan D domain rigid fitting inside UCSF ChimeraX.\n\nGenerated by scripts/strengthen_maps.py. The fitting path uses\nchimerax.map_fit.fitcmd.fitmap(..., metric=\'correlation\', search=N) with fixed\nper-entry resolution. Each fit uses one half-map; the reverse half-map is scored\nat the fitted pose with shift=False, rotate=False, search=0.\n"""\n\nfrom __future__ import annotations\n\nimport csv\nimport json\nimport math\nimport random\nfrom collections import defaultdict\nfrom pathlib import Path\n\nimport numpy as np\n\n\ndef _resolve(root: Path, value: str, *, base: Path | None = None) -> Path:\n    path = Path(value)\n    if path.is_absolute():\n        return path\n    if base is not None:\n        candidate = base / path\n        if candidate.exists() or not (root / path).exists():\n            return candidate\n    return root / path\n\n\ndef _quote(path: Path) -> str:\n    return \'"\' + str(path).replace(\'"\', \'\\\\"\') + \'"\'\n\n\ndef _read_window(path: Path) -> list[int]:\n    with path.open(newline=\'\', encoding=\'utf-8\') as handle:\n        return [int(row[\'author_resnum\']) for row in csv.DictReader(handle)]\n\n\ndef _read_ca_pdb(path: Path, chain_id: str) -> dict[int, tuple[str, np.ndarray]]:\n    atoms = {}\n    for line in path.read_text(encoding=\'utf-8\', errors=\'replace\').splitlines():\n        if not line.startswith((\'ATOM  \', \'HETATM\')):\n            continue\n        if line[12:16].strip() != \'CA\' or line[21].strip() != chain_id:\n            continue\n        resnum = int(line[22:26])\n        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)\n        atoms[resnum] = (line[17:20].strip() or \'ALA\', xyz)\n    return atoms\n\n\ndef _write_ca_pdb(source: dict[int, tuple[str, np.ndarray]], residues: list[int], path: Path) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    lines = []\n    for serial, resnum in enumerate(residues, start=1):\n        resname, xyz = source[resnum]\n        lines.append(\n            f"ATOM  {serial:5d}  CA  {resname:>3s} B{resnum:4d}    "\n            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C"\n        )\n    path.write_text(\'\\n\'.join([*lines, \'TER\', \'END\']) + \'\\n\', encoding=\'utf-8\')\n\n\ndef _kabsch_rotation(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:\n    mob_cent = mobile.mean(axis=0)\n    ref_cent = reference.mean(axis=0)\n    cov = (mobile - mob_cent).T @ (reference - ref_cent)\n    u, _, vt = np.linalg.svd(cov)\n    rot = u @ vt\n    if np.linalg.det(rot) < 0:\n        vt[-1, :] *= -1\n        rot = u @ vt\n    return rot\n\n\ndef _kabsch_apply(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:\n    rot = _kabsch_rotation(mobile, reference)\n    return (mobile - mobile.mean(axis=0)) @ rot + reference.mean(axis=0)\n\n\ndef _rotation_angle_deg(left: np.ndarray, right: np.ndarray) -> float:\n    rel = left.T @ right\n    u, _, vt = np.linalg.svd(rel)\n    rot = u @ vt\n    if np.linalg.det(rot) < 0:\n        u[:, -1] *= -1\n        rot = u @ vt\n    skew = np.array([\n        rot[2, 1] - rot[1, 2],\n        rot[0, 2] - rot[2, 0],\n        rot[1, 0] - rot[0, 1],\n    ])\n    sine = 0.5 * float(np.linalg.norm(skew))\n    cosine = 0.5 * (float(np.trace(rot)) - 1.0)\n    if sine <= 1e-12 and abs(cosine - 1.0) <= 1e-12:\n        return 0.0\n    return float(math.degrees(math.atan2(sine, cosine)))\n\n\ndef _coordinate(mobile: np.ndarray, mean: np.ndarray, pc1: np.ndarray, closed_mean: float, open_mean: float) -> tuple[float, float]:\n    aligned = _kabsch_apply(mobile, mean)\n    raw = float(((aligned - mean).ravel() @ pc1) / math.sqrt(mean.shape[0]))\n    if open_mean == closed_mean:\n        raise RuntimeError(\'open/closed score scale has zero width\')\n    return raw, float((raw - closed_mean) / (open_mean - closed_mean))\n\n\ndef _rigid_body_basis(coords: np.ndarray, domain_indices: list[np.ndarray]) -> np.ndarray:\n    cols = []\n    axes = np.eye(3)\n    for idx in domain_indices:\n        centered = coords[idx] - coords[idx].mean(axis=0)\n        for axis in axes:\n            disp = np.zeros_like(coords)\n            disp[idx] = axis\n            cols.append(disp.reshape(-1))\n        for axis in axes:\n            disp = np.zeros_like(coords)\n            disp[idx] = np.cross(axis, centered)\n            cols.append(disp.reshape(-1))\n    mat = np.column_stack(cols)\n    q, _ = np.linalg.qr(mat)\n    return q\n\n\ndef _postfit_projection(\n    displacement: np.ndarray,\n    mode1: np.ndarray,\n    low_frequency_basis: np.ndarray,\n    coords: np.ndarray,\n    domain_indices: list[np.ndarray],\n) -> dict[str, float | None]:\n    vector = displacement.reshape(-1)\n    norm = float(np.linalg.norm(vector))\n    if norm == 0:\n        return {\n            "mode1_abs_cosine": None,\n            "low_frequency_subspace_fraction": None,\n            "rigid_two_domain_subspace_fraction": None,\n        }\n    unit = vector / norm\n    mode_unit = mode1 / float(np.linalg.norm(mode1))\n    low_projected = low_frequency_basis @ (low_frequency_basis.T @ unit)\n    rigid_basis = _rigid_body_basis(coords, domain_indices)\n    rigid_projected = rigid_basis @ (rigid_basis.T @ unit)\n    return {\n        "mode1_abs_cosine": float(abs(unit @ mode_unit)),\n        "low_frequency_subspace_fraction": float(np.linalg.norm(low_projected)),\n        "rigid_two_domain_subspace_fraction": float(np.linalg.norm(rigid_projected)),\n    }\n\n\ndef _load_score_reference(root: Path, config: dict, coord_residues: list[int], *, base: Path | None = None) -> dict[str, object]:\n    pca = np.load(_resolve(root, config[\'pca_npz\'], base=base), allow_pickle=False)\n    modes = np.load(_resolve(root, config[\'modes_npz\'], base=base), allow_pickle=False)\n    window = _read_window(_resolve(root, config[\'window_csv\'], base=base))\n    if window != coord_residues:\n        raise RuntimeError(f"coordinate residues do not match frozen 269 window: {len(coord_residues)}")\n    mean = np.asarray(pca[\'mean\'], dtype=float)\n    pc1 = np.asarray(pca[\'pcs\'][:, 0], dtype=float)\n    scores = np.asarray(pca[\'pc1_scores\'], dtype=float)\n    open_mask = np.asarray(pca[\'open_mask\'], dtype=bool)\n    low_frequency_count = int(config.get(\'low_frequency_mode_count\', 10))\n    low_frequency_modes = np.asarray(modes[\'anm_eigvecs\'][:, :low_frequency_count], dtype=float)\n    mode1 = low_frequency_modes[:, 0]\n    q, _ = np.linalg.qr(low_frequency_modes)\n    return {\n        "mean": mean,\n        "pc1": pc1,\n        "closed_mean": float(scores[~open_mask].mean()),\n        "open_mean": float(scores[open_mask].mean()),\n        "mode1": mode1,\n        "low_frequency_basis": q,\n        "low_frequency_mode_count": low_frequency_count,\n    }\n\n\ndef _atoms_by_residue(model) -> dict[int, object]:\n    result = {}\n    for residue in model.residues:\n        try:\n            resnum = int(residue.number)\n        except Exception:\n            continue\n        atom = residue.find_atom(\'CA\')\n        if atom is not None:\n            result[resnum] = atom\n    return result\n\n\ndef _atoms_for_residues(model, residues: list[int]):\n    from chimerax.atomic import Atoms\n\n    by_residue = _atoms_by_residue(model)\n    atoms = [by_residue[resnum] for resnum in residues if resnum in by_residue]\n    if len(atoms) != len(residues):\n        missing = sorted(set(residues) - set(by_residue))\n        raise RuntimeError(f"model {model.name} is missing CA residues {missing[:20]}")\n    return Atoms(atoms)\n\n\ndef _matrix(place) -> list[list[float]]:\n    return [[float(value) for value in row] for row in place.matrix]\n\n\ndef _fit_domain(session, atoms, train_map, *, resolution: float, search: int, seed: int):\n    from chimerax.map_fit.fitcmd import fitmap\n\n    fits = fitmap(\n        session,\n        atoms,\n        train_map,\n        metric=\'correlation\',\n        resolution=resolution,\n        search=search,\n        seed=seed,\n    )\n    if not fits:\n        raise RuntimeError(\'fitmap returned no fits\')\n    best = max(fits, key=lambda fit: ((fit.correlation() or -999.0), (fit.average_map_value() or -999.0)))\n    best.place_models(session, frames=0)\n    return best\n\n\ndef _score_heldout(session, atoms, heldout_map, *, resolution: float):\n    from chimerax.map_fit.fitcmd import fitmap\n\n    fits = fitmap(\n        session,\n        atoms,\n        heldout_map,\n        metric=\'correlation\',\n        resolution=resolution,\n        shift=False,\n        rotate=False,\n        search=0,\n    )\n    if not fits:\n        raise RuntimeError(\'heldout fitmap scoring returned no metrics\')\n    fit = fits[0]\n    return {\n        "heldout_correlation": fit.correlation(),\n        "heldout_average_map_value": fit.average_map_value(),\n        "heldout_points_inside_contour": fit.points_inside_contour(),\n    }\n\n\ndef _save_pdb(session, model, path: Path, rel_model=None) -> None:\n    from chimerax.pdb import save_pdb\n\n    path.parent.mkdir(parents=True, exist_ok=True)\n    kwargs = {"models": [model]}\n    if rel_model is not None:\n        kwargs["rel_model"] = rel_model\n    save_pdb(session, str(path), **kwargs)\n\n\ndef _open_one(session, path: Path):\n    from chimerax.core.commands import run\n\n    models = run(session, f"open {_quote(path)}")\n    if not models:\n        raise RuntimeError(f"opening {path} produced no model")\n    return models[0]\n\n\ndef _close_model(session, model) -> None:\n    from chimerax.core.commands import run\n\n    run(session, f"close #{model.id_string}")\n\n\ndef _prepare_templates(root: Path, config: dict, output_dir: Path, *, base: Path | None = None):\n    window = _read_window(_resolve(root, config[\'window_csv\'], base=base))\n    chain_id = config[\'template_chain\']\n    template_ca = {name: _read_ca_pdb(_resolve(root, path, base=base), chain_id) for name, path in config[\'templates\'].items()}\n    common = sorted(set(window).intersection(*(set(coords) for coords in template_ca.values())))\n    domains = {}\n    domain_indices = []\n    template_paths = {}\n    template_domain_refs = defaultdict(dict)\n    coord_residues = [resnum for resnum in common if any(span[0] <= resnum <= span[1] for span in config[\'domains\'].values())]\n    for domain_name, (start, end) in config[\'domains\'].items():\n        residues = [resnum for resnum in coord_residues if start <= resnum <= end]\n        domains[domain_name] = residues\n        domain_indices.append(np.array([coord_residues.index(resnum) for resnum in residues], dtype=int))\n        for template_name, coords in template_ca.items():\n            out = output_dir / \'matched_templates\' / f"{template_name}_{domain_name.replace(\'+\', \'plus\')}.pdb"\n            _write_ca_pdb(coords, residues, out)\n            template_paths[(template_name, domain_name)] = out\n            template_domain_refs[template_name][domain_name] = np.array([coords[resnum][1] for resnum in residues], dtype=float)\n    return domains, domain_indices, template_paths, coord_residues, template_domain_refs\n\n\ndef _state_stability(rows: list[dict], thresholds: dict) -> dict[str, dict]:\n    grouped = defaultdict(list)\n    for row in rows:\n        grouped[row[\'state\']].append(row)\n    out = {}\n    for state, state_rows in sorted(grouped.items()):\n        orientations = [float(row[\'relative_orientation_deg\']) for row in state_rows]\n        coords = [float(row[\'normalized_structural_coordinate\']) for row in state_rows]\n        orientation_range = max(orientations) - min(orientations)\n        coord_range = max(coords) - min(coords)\n        out[state] = {\n            "n_fits": len(state_rows),\n            "relative_orientation_range_deg": orientation_range,\n            "normalized_structural_coordinate_range": coord_range,\n            "relative_orientation_pass": orientation_range <= thresholds[\'relative_orientation_max_deg\'],\n            "normalized_structural_coordinate_pass": coord_range <= thresholds[\'normalized_structural_coordinate_max_abs_delta\'],\n            "state_gate_pass": orientation_range <= thresholds[\'relative_orientation_max_deg\'] and coord_range <= thresholds[\'normalized_structural_coordinate_max_abs_delta\'],\n        }\n    return out\n\n\ndef _accepted_state_alignment(rows: list[dict], state_stability: dict, score_ref: dict, domain_indices: list[np.ndarray]) -> dict[str, object]:\n    accepted = [state for state, gate in state_stability.items() if gate[\'state_gate_pass\']]\n    by_state = {}\n    for state in accepted:\n        coords = [np.array(row[\'coordinates\'], dtype=float) for row in rows if row[\'state\'] == state]\n        if coords:\n            by_state[state] = np.mean(coords, axis=0)\n    pairs = []\n    order = [\'open\', \'intermediate\', \'closed\']\n    for left, right in zip(order, order[1:]):\n        if left not in by_state or right not in by_state:\n            continue\n        left_aligned = _kabsch_apply(by_state[left], score_ref[\'mean\'])\n        right_aligned = _kabsch_apply(by_state[right], score_ref[\'mean\'])\n        metrics = _postfit_projection(\n            right_aligned - left_aligned,\n            score_ref[\'mode1\'],\n            score_ref[\'low_frequency_basis\'],\n            score_ref[\'mean\'],\n            domain_indices,\n        )\n        pairs.append({\n            "from_state": left,\n            "to_state": right,\n            "low_frequency_mode_count": score_ref[\'low_frequency_mode_count\'],\n            **metrics,\n        })\n    return {"accepted_states": accepted, "state_displacements": pairs}\n\n\ndef run_plan_d(session, config_path: str):\n    config_file = Path(config_path).resolve()\n    config = json.loads(config_file.read_text(encoding=\'utf-8\'))\n    root = Path(config[\'repository_root\']).resolve()\n    output_dir = _resolve(root, config[\'output_dir\'])\n    output_dir.mkdir(parents=True, exist_ok=True)\n    random.seed(int(config[\'seed\']))\n    np.random.seed(int(config[\'seed\']))\n\n    config_base = config_file.parent\n    domains, domain_indices, template_paths, coord_residues, template_domain_refs = _prepare_templates(root, config, output_dir, base=config_base)\n    score_ref = _load_score_reference(root, config, coord_residues, base=config_base)\n    rows = []\n\n    for entry in config[\'fit_entries\']:\n        maps = {key: _open_one(session, _resolve(root, path, base=config_base)) for key, path in entry[\'half_maps\'].items()}\n        try:\n            for pair in config[\'train_heldout_pairs\']:\n                train_key = pair[\'train\']\n                heldout_key = pair[\'heldout\']\n                train_map = maps[train_key]\n                heldout_map = maps[heldout_key]\n                for template_name in config[\'templates\']:\n                    domain_records = {}\n                    coord_by_residue = {}\n                    domain_rotations = {}\n                    for domain_name, residues in domains.items():\n                        model = _open_one(session, template_paths[(template_name, domain_name)])\n                        try:\n                            atoms = _atoms_for_residues(model, residues)\n                            best = _fit_domain(\n                                session,\n                                atoms,\n                                train_map,\n                                resolution=float(entry[\'fit_resolution_angstrom\']),\n                                search=int(config[\'search_placements\']),\n                                seed=int(config[\'seed\']),\n                            )\n                            heldout = _score_heldout(\n                                session,\n                                atoms,\n                                heldout_map,\n                                resolution=float(entry[\'fit_resolution_angstrom\']),\n                            )\n                            by_residue = _atoms_by_residue(model)\n                            fitted_domain = np.array([by_residue[resnum].scene_coord for resnum in residues], dtype=float)\n                            domain_rotations[domain_name] = _kabsch_rotation(\n                                template_domain_refs[template_name][domain_name], fitted_domain\n                            )\n                            coord_by_residue.update({str(resnum): [float(x) for x in by_residue[resnum].scene_coord] for resnum in residues})\n                            pose_path = output_dir / \'poses\' / entry[\'emdb_id\'] / train_key / template_name / f"{domain_name.replace(\'+\', \'plus\')}.pdb"\n                            _save_pdb(session, model, pose_path, rel_model=train_map)\n                            domain_records[domain_name] = {\n                                "residue_count": len(residues),\n                                "fit_resolution_angstrom": float(entry[\'fit_resolution_angstrom\']),\n                                "train_correlation": best.correlation(),\n                                "train_average_map_value": best.average_map_value(),\n                                "train_points_inside_contour": best.points_inside_contour(),\n                                "heldout_score": heldout,\n                                "transform_matrix": _matrix(model.position),\n                                "reference_to_fitted_rotation": domain_rotations[domain_name].tolist(),\n                                "pose_pdb": str(pose_path),\n                            }\n                        finally:\n                            _close_model(session, model)\n                    fitted = np.array([coord_by_residue[str(resnum)] for resnum in coord_residues], dtype=float)\n                    raw_coordinate, normalized_coordinate = _coordinate(\n                        fitted,\n                        score_ref[\'mean\'],\n                        score_ref[\'pc1\'],\n                        score_ref[\'closed_mean\'],\n                        score_ref[\'open_mean\'],\n                    )\n                    orientation = _rotation_angle_deg(domain_rotations[\'NTD+HB\'], domain_rotations[\'TBD\'])\n                    coord_path = output_dir / \'coordinates\' / f"{entry[\'emdb_id\']}_{train_key}_fit_{heldout_key}_score_{template_name}_269ca.json"\n                    coord_path.parent.mkdir(parents=True, exist_ok=True)\n                    coordinates = [coord_by_residue[str(resnum)] for resnum in coord_residues]\n                    coord_payload = {\n                        "emdb_id": entry[\'emdb_id\'],\n                        "state": entry[\'state\'],\n                        "template": template_name,\n                        "train_half": train_key,\n                        "heldout_half": heldout_key,\n                        "residues": coord_residues,\n                        "coordinates": coordinates,\n                    }\n                    coord_path.write_text(json.dumps(coord_payload, indent=1, allow_nan=False) + \'\\n\', encoding=\'utf-8\')\n                    rows.append({\n                        "emdb_id": entry[\'emdb_id\'],\n                        "state": entry[\'state\'],\n                        "template": template_name,\n                        "train_half": train_key,\n                        "heldout_half": heldout_key,\n                        "coordinate_residue_count": len(coord_residues),\n                        "domain_metrics": domain_records,\n                        "relative_orientation_deg": orientation,\n                        "raw_pc1_coordinate": raw_coordinate,\n                        "normalized_structural_coordinate": normalized_coordinate,\n                        "coordinate_json": str(coord_path),\n                        "coordinates": coordinates,\n                    })\n        finally:\n            for model in maps.values():\n                _close_model(session, model)\n\n    state_stability = _state_stability(rows, config[\'stability_gate\'])\n    alignment = _accepted_state_alignment(rows, state_stability, score_ref, domain_indices)\n    summary = {\n        "config": str(config_file),\n        "seed": config[\'seed\'],\n        "search_placements": config[\'search_placements\'],\n        "results": rows,\n        "state_stability": state_stability,\n        "accepted_state_alignment": alignment,\n    }\n    summary_path = output_dir / \'plan_d_fit_summary.json\'\n    summary_path.write_text(json.dumps(summary, indent=1, allow_nan=False) + \'\\n\', encoding=\'utf-8\')\n    session.logger.info(f"CRBN Plan D fit summary written to {summary_path}")\n\n\ndef register_command(session):\n    from chimerax.core.commands import CmdDesc, OpenFileNameArg, register\n\n    desc = CmdDesc(required=[(\'config_path\', OpenFileNameArg)], synopsis=\'Run CRBN Plan D half-map domain rigid fitting\')\n    register(\'crbnpland\', desc, run_plan_d, logger=session.logger)\n\n\nif \'session\' in globals():\n    register_command(globals()[\'session\'])\n'

def path_from_config_dir(path: Path, analysis_root: Path) -> str:
    return os.path.relpath(path, analysis_root)


def plan_d_reference_sources() -> dict[str, Path]:
    return {
        "crbn_residue_window.csv": PLAN_D_WINDOW_CSV,
        "crbn_pca.npz": PLAN_D_PCA_NPZ,
        "crbn_anm_modes.npz": PLAN_D_MODES_NPZ,
        "open_8cvp_assembly.pdb": PLAN_D_TEMPLATES["open_8cvp_crbn"],
        "closed_5fqd.pdb": PLAN_D_TEMPLATES["closed_5fqd_crbn"],
    }


def write_plan_d_reference_inputs(analysis_root: Path) -> dict[str, Any]:
    reference_dir = analysis_root / "reference_inputs"
    reference_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    missing: list[dict[str, str]] = []
    for name, source in plan_d_reference_sources().items():
        destination = reference_dir / name
        if source.exists():
            shutil.copyfile(source, destination)
            copied[name] = path_from_config_dir(destination, analysis_root)
        else:
            missing.append({"name": name, "source": display_path(source)})
    return {
        "available": not missing,
        "copied": copied,
        "missing": missing,
        "reference_dir": path_from_config_dir(reference_dir, analysis_root),
    }


def plan_d_config(
    output_root: Path,
    analysis_root: Path,
    *,
    reference_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    targets = []
    for emdb_id in PLAN_D_FIT_IDS:
        entry_dir = output_root / "data" / "maps" / f"EMD-{emdb_id}"
        targets.append(
            {
                "emdb_id": f"EMD-{emdb_id}",
                "state": ENTRY_SPECS[emdb_id].state,
                "role": ENTRY_SPECS[emdb_id].role,
                "primary_map": path_from_config_dir(entry_dir / f"emd_{emdb_id}.map.gz", analysis_root),
                "half_maps": {
                    "A": path_from_config_dir(entry_dir / f"emd_{emdb_id}_half_map_1.map.gz", analysis_root),
                    "B": path_from_config_dir(entry_dir / f"emd_{emdb_id}_half_map_2.map.gz", analysis_root),
                },
                "mask": path_from_config_dir(entry_dir / f"emd_{emdb_id}_msk_1.map", analysis_root),
                "fit_resolution_angstrom": PLAN_D_FIT_RESOLUTION_ANGSTROM[emdb_id],
            }
        )
    reference_inputs = {
        "crbn_residue_window.csv": "reference_inputs/crbn_residue_window.csv",
        "crbn_pca.npz": "reference_inputs/crbn_pca.npz",
        "crbn_anm_modes.npz": "reference_inputs/crbn_anm_modes.npz",
        "open_8cvp_assembly.pdb": "reference_inputs/open_8cvp_assembly.pdb",
        "closed_5fqd.pdb": "reference_inputs/closed_5fqd.pdb",
    }
    if reference_status is None:
        reference_status = {"available": None, "missing": [], "copied": {}}
    return {
        "plan": "CRBN cryoEM acquisition and analysis D",
        "seed": PLAN_D_SEED,
        "search_placements": PLAN_D_SEARCH_PLACEMENTS,
        "low_frequency_mode_count": PLAN_D_LOW_FREQUENCY_MODE_COUNT,
        "fit_entries": targets,
        "repository_root": str(ROOT),
        "reference_inputs": reference_inputs,
        "reference_inputs_available": reference_status.get("available"),
        "missing_reference_inputs": reference_status.get("missing", []),
        "source_reference_inputs": {name: display_path(path) for name, path in plan_d_reference_sources().items()},
        "templates": {
            "open_8cvp_crbn": reference_inputs["open_8cvp_assembly.pdb"],
            "closed_5fqd_crbn": reference_inputs["closed_5fqd.pdb"],
        },
        "template_chain": "B",
        "window_csv": reference_inputs["crbn_residue_window.csv"],
        "pca_npz": reference_inputs["crbn_pca.npz"],
        "modes_npz": reference_inputs["crbn_anm_modes.npz"],
        "domains": {name: list(span) for name, span in PLAN_D_DOMAINS.items()},
        "train_heldout_pairs": [{"train": "A", "heldout": "B"}, {"train": "B", "heldout": "A"}],
        "stability_gate": {
            "relative_orientation_max_deg": 10.0,
            "normalized_structural_coordinate_max_abs_delta": 0.1,
        },
        "output_dir": display_path(analysis_root / "chimerax_plan_d"),
        "method_boundary": (
            "Rigid-body domain fitting is evaluated against independent half maps. "
            "Modal coordinates are recorded only after fitting, not used as fit objectives "
            "or constraints."
        ),
    }


def write_chimerax_assets(output_root: Path, analysis_root: Path) -> None:
    """Write the approved ChimeraX Plan D recipe without accepting/installing ChimeraX."""

    reference_status = write_plan_d_reference_inputs(analysis_root)
    config = plan_d_config(output_root, analysis_root, reference_status=reference_status)
    config_path = analysis_root / "chimerax_plan_d_config.json"
    runner_py = analysis_root / "chimerax_plan_d_runner.py"
    run_cxc = analysis_root / "run_chimerax_plan_d.cxc"
    install_md = analysis_root / "chimerax_install_ready.md"
    obsolete_assets = [
        analysis_root / "run_chimerax_emd70781.cxc",
        analysis_root / "chimerax_fit_search.py",
    ]
    for obsolete in obsolete_assets:
        if obsolete.exists():
            obsolete.unlink()
    atomic_write_json(config_path, config, sort_keys=True)
    atomic_write_text(runner_py, CHIMERAX_PLAN_D_RUNNER)
    atomic_write_text(
        run_cxc,
        f"""# ChimeraX Plan D run script for CRBN domain rigid fitting.
# Requires UCSF ChimeraX installed after user/licensee acceptance of the UCSF license.
# Run example:
#   ChimeraX --nogui --cmd \"open {display_path(run_cxc)}\"

open {display_path(runner_py)}
crbnpland {display_path(config_path)}
log save {display_path(analysis_root / 'chimerax_plan_d_log.html')}
exit
""",
    )
    atomic_write_text(
        install_md,
        """# ChimeraX installation gate

Official page: https://www.cgl.ucsf.edu/chimerax/download.html

Apple Silicon production build from UCSF release metadata:

- Version: 1.12
- Release date: 11 June 2026
- Package route: chimerax-get.py?file=1.12/mac_arm64/ChimeraX-1.12.dmg
- Exact gated endpoint: https://www.cgl.ucsf.edu/chimerax/cgi-bin/secure/chimerax-get.py?file=1.12/mac_arm64/ChimeraX-1.12.dmg
- SHA-256: b944ee9df84af518091ca14d4e3184725c6d3475d46391e9b4d034dd332b5a26

The UCSF download route displays the ChimeraX Non-Commercial Software
License Agreement and requires pressing Accept before the DMG download. This
agent did not press Accept and did not install ChimeraX. The Plan D runner also
requires staged reference inputs under analysis/maps/reference_inputs;
chimerax_plan_d_config.json records missing_reference_inputs if a public checkout
does not contain those frozen inputs. Once the license is accepted by the
user/licensee and reference_inputs_available is true, run
run_chimerax_plan_d.cxc with ChimeraX.
""",
    )


def markdown_report(result: dict[str, Any]) -> str:
    decision = result["quality_decision"]
    lines = [
        "# CRBN cryoEM acquisition and analysis D",
        "",
        f"- Retrieval UTC: {result['retrieval_utc']}",
        f"- Dippon source: {result['source_urls']['dippon_nature']}",
        f"- Overall use decision: {decision['overall_use']}",
        f"- Global map QA: {decision.get('global_map_qa', 'not-tested')}",
        f"- CRBN-local density support: {decision.get('crbn_local_density_support', decision['density_support'])}",
        f"- ChimeraX install status: {decision['chimerax']['install_status']}",
        "",
        "## Entry inventory",
        "",
        "| EMDB | State | Role | Scope | Resolution (A) | Artifacts |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for entry in result["entries"]:
        lines.append(
            "| {emdb_id} | {state} | {role} | {analysis_scope} | {resolution} | {artifacts} |".format(
                emdb_id=entry["emdb_id"],
                state=entry["state"],
                role=entry["role"],
                analysis_scope=entry["analysis_scope"],
                resolution=entry["resolution_angstrom"] or "",
                artifacts=len(entry["artifacts"]),
            )
        )
    lines.extend(["", "## Downloaded artifacts", ""])
    if result["downloaded_artifacts"]:
        lines.extend(
            [
                "| File | Kind | Bytes | SHA-256 |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for artifact in result["downloaded_artifacts"]:
            lines.append(
                f"| {artifact['local_path']} | {artifact['kind']} | "
                f"{artifact['bytes']} | `{artifact['sha256']}` |"
            )
    else:
        lines.append("No map artifacts were downloaded in this run.")
    lines.extend(["", "## Density checks", ""])
    if result["density_checks"]:
        lines.extend(
            [
                "| EMDB | Artifact | Dimensions | Mode | Mean | Std | Nonzero fraction |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for emdb_id, checks in sorted(result["density_checks"].items()):
            for artifact_key, check in sorted(checks.items()):
                header = check["header"]
                lines.append(
                    f"| {emdb_id} | {artifact_key} | "
                    f"{' x '.join(str(value) for value in header['dimensions'])} | "
                    f"{header['mode_name']} | {check['computed_mean']:.6g} | "
                    f"{check['computed_std']:.6g} | "
                    f"{check['computed_nonzero_fraction']:.6g} |"
                )
    else:
        lines.append("No downloaded map density was inspected in this run.")
    lines.extend(["", "## Half-map correlations", ""])
    correlations = decision["half_map_correlations"]
    if correlations:
        lines.extend(
            [
                "| EMDB | Pearson r | Sampled voxels | Stride |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for emdb_id, correlation in sorted(correlations.items()):
            lines.append(
                f"| {emdb_id} | {correlation['pearson_r']:.6g} | "
                f"{correlation['sampled_points']} | {correlation['stride']} |"
            )
    else:
        lines.append("No two-half-map pairs were available for correlation measurement.")
    lines.extend(["", "## Quality decision", ""])
    for reason in decision["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(["", decision["claim_boundary"], ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_ROOT.relative_to(ROOT),
        help=(
            "Output root containing data/maps and analysis/maps. "
            "Defaults to results/strengthening."
        ),
    )
    parser.add_argument("--offline", action="store_true", help="Use cached metadata/maps only.")
    parser.add_argument(
        "--metadata-ids",
        nargs="+",
        default=list(DEFAULT_METADATA_IDS),
        choices=sorted(ENTRY_SPECS),
        help="EMDB numeric ids for metadata capture.",
    )
    parser.add_argument(
        "--download-ids",
        nargs="*",
        default=list(DEFAULT_DOWNLOAD_IDS),
        choices=sorted(ENTRY_SPECS),
        help="EMDB numeric ids for map artifact download/QA.",
    )
    parser.add_argument(
        "--artifact-kinds",
        nargs="+",
        default=["primary_map", "half_map"],
        choices=sorted(MAP_KINDS),
        help="Artifact classes to download/analyze for selected download ids.",
    )
    parser.add_argument(
        "--refresh-downloads",
        action="store_true",
        help="Re-download map files even when a cached local file exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
