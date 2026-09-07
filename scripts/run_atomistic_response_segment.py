#!/usr/bin/env python3
"""Bounded NVT segment runner for atomistic CRBN response sampling.

The runner executes one restartable local segment.  A completed segment is not
an equilibrium, production-readiness, or response-convergence claim.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import openmm as mm
from openmm import app, unit

try:
    from . import run_atomistic_technical_pilot as pilot
except ImportError:
    import run_atomistic_technical_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
PHASES = ("zero_equilibration", "zero_calibration", "force_equilibration", "force_sampling")
RESPONSE_COLUMNS = ("model", "replicate", "force_kj_mol_nm", "time_ps", "closure_nm")
SEGMENT_SCHEMA_VERSION = "1.0"
CERT_SCHEMA_VERSION = "1.0"
GENERATION_ARTIFACTS = ("segment.chk", "segment_state.xml", "response_observations.csv", "segment_observables.npz")
DATA_INPUT_HASH_KEYS = ("prmtop", "inpcrd", "mapping")
IDENTITY_BINDING_KEYS = (
    "data_input_hashes",
    "stage_provenance_hashes",
    "source_provenance",
    "topology",
    "mapping_sha256",
    "core_indices_sha256",
    "reference_nm_sha256",
    "q_ambient_sha256",
    "system_xml_sha256",
    "gauge_sha256",
    "body_sha256",
    "measurement_identity",
)


@dataclass(frozen=True)
class SegmentSettings:
    steps: int = 1000
    timestep_fs: float = 1.0
    temperature_K: float = 300.0
    friction_per_ps: float = 1.0
    gauge_k: float = 1000.0
    master_seed: int = 0
    velocity_seed: int = 0
    thermostat_seed: int = 0
    report_interval_steps: int = 100
    max_step_batch: int = 100
    max_wall_seconds: float = 300.0
    constraint_tolerance: float = 1e-6
    disable_pme_stream: bool = False


class AdmissionError(ValueError):
    """Raised before Context construction for a non-admissible segment."""


class ResumeMismatch(ValueError):
    """Raised when requested inputs do not match a checkpoint lineage."""


class _BudgetStop(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def _topology_fingerprint(topology) -> dict[str, Any]:
    if topology is None:
        return {"provided": False, "sha256": "not_supplied"}
    atoms = [
        {
            "index": atom.index,
            "name": atom.name,
            "element": atom.element.symbol if atom.element is not None else None,
            "residue_index": atom.residue.index,
            "residue_name": atom.residue.name,
            "residue_id": atom.residue.id,
            "chain_index": atom.residue.chain.index,
            "chain_id": atom.residue.chain.id,
        }
        for atom in topology.atoms()
    ]
    bonds = sorted(tuple(sorted((a.index, b.index))) for a, b in topology.bonds())
    payload = {"provided": True, "atom_count": len(atoms), "atoms": atoms, "bonds": bonds}
    payload["sha256"] = _canonical_hash({"atoms": atoms, "bonds": bonds})
    return payload


def _derive_seed(master_seed: int, *, model: str, replicate_id: str, phase: str, force_h: float, stream: str) -> int:
    payload = {
        "master_seed": int(master_seed),
        "model": model,
        "replicate_id": str(replicate_id),
        "phase": phase,
        "force_kj_mol_nm": float(force_h),
        "stream": stream,
    }
    value = int.from_bytes(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).digest()[:4], "big")
    return (value & 0x7FFFFFFF) or 1


def _resolved_settings(settings: SegmentSettings, *, model: str, replicate_id: str, phase: str, force_h: float) -> SegmentSettings:
    if isinstance(settings.master_seed, bool) or not isinstance(settings.master_seed, int) or settings.master_seed <= 0:
        raise ValueError("SegmentSettings.master_seed must be an explicit positive independent-replica seed")
    values = asdict(settings)
    values["velocity_seed"] = settings.velocity_seed or _derive_seed(
        settings.master_seed, model=model, replicate_id=replicate_id, phase=phase, force_h=force_h, stream="velocity"
    )
    values["thermostat_seed"] = settings.thermostat_seed or _derive_seed(
        settings.master_seed, model=model, replicate_id=replicate_id, phase=phase, force_h=force_h, stream="thermostat"
    )
    return SegmentSettings(**values)


def resolve_segment_settings(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> SegmentSettings:
    values = asdict(SegmentSettings())
    values.update({"temperature_K": config.get("temperature_K", values["temperature_K"]),
                   "timestep_fs": config.get("timestep_fs_initial", values["timestep_fs"])})
    section = config.get("response_segment", {})
    if not isinstance(section, dict):
        raise ValueError("response_segment must be an object")
    unknown = set(section) - set(values)
    if unknown:
        raise ValueError(f"Unknown response_segment setting(s): {sorted(unknown)}")
    values.update(section)
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    integer_keys = ("steps", "master_seed", "velocity_seed", "thermostat_seed", "report_interval_steps", "max_step_batch")
    boolean_keys = ("disable_pme_stream",)
    for key in boolean_keys:
        if not isinstance(values[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key, value in values.items():
        if key in boolean_keys:
            continue
        if key in ("velocity_seed", "thermostat_seed") and value == 0:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
        if key in integer_keys and (int(value) != value or value > 2**31 - 1):
            raise ValueError(f"{key} must be a positive 32-bit integer")
    if values["timestep_fs"] != 1.0 or values["friction_per_ps"] != 1.0:
        raise ValueError("Response segments require 1 fs and friction 1/ps until separately validated")
    if values["max_step_batch"] > values["report_interval_steps"]:
        raise ValueError("max_step_batch cannot exceed report_interval_steps")
    if values["report_interval_steps"] > values["steps"]:
        raise ValueError("report_interval_steps cannot exceed steps")
    if values["steps"] % values["report_interval_steps"]:
        raise ValueError("steps must be an integer multiple of report_interval_steps")
    if not 0 < values["constraint_tolerance"] <= 1e-5:
        raise ValueError("constraint_tolerance must lie in (0, 1e-5]")
    for key in integer_keys:
        values[key] = int(values[key])
    return SegmentSettings(**values)


def _response_row(model: str, replicate_id: str, force_h: float, time_ps: float, closure_nm: float) -> dict[str, Any]:
    return {
        "model": model,
        "replicate": replicate_id,
        "force_kj_mol_nm": float(force_h),
        "time_ps": float(time_ps),
        "closure_nm": float(closure_nm),
    }


def _system_without_barostat(system: mm.System) -> mm.System:
    for force in system.getForces():
        if "Barostat" in type(force).__name__:
            raise ValueError("NVT response segment input must not contain a barostat")
    return system


def _identity(system_xml: str, mapping: Mapping[str, Any], topology_hash: Mapping[str, Any], input_hashes: Mapping[str, Any]) -> dict[str, Any]:
    core_indices = list(map(int, mapping["core_indices"]))
    reference_nm = np.asarray(mapping["reference_nm"], dtype=float)
    q = np.asarray(mapping["q"], dtype=float).reshape(-1)
    data_hashes = {key: input_hashes[key] for key in DATA_INPUT_HASH_KEYS if key in input_hashes}
    stage_hashes = {key: value for key, value in input_hashes.items() if key not in DATA_INPUT_HASH_KEYS and key != "measurement_identity"}
    measurement_identity = dict(input_hashes.get("measurement_identity", {"provided": False})) if isinstance(input_hashes.get("measurement_identity"), Mapping) else {"provided": False}
    return {
        "input_hashes": dict(input_hashes),
        "data_input_hashes": data_hashes,
        "stage_provenance_hashes": stage_hashes,
        "source_provenance": {
            "response_runner_sha256": _sha256(ROOT / "scripts" / "run_atomistic_response_segment.py"),
            "technical_pilot_sha256": _sha256(ROOT / "scripts" / "run_atomistic_technical_pilot.py"),
            "atomistic_boundary_sha256": _sha256(ROOT / "scripts" / "atomistic_boundary.py"),
        },
        "topology": dict(topology_hash),
        "mapping_sha256": _canonical_hash(mapping),
        "core_indices_sha256": _canonical_hash(core_indices),
        "reference_nm_sha256": _canonical_hash(reference_nm.tolist()),
        "q_ambient_sha256": _canonical_hash(q.tolist()),
        "q_norm": float(np.linalg.norm(q)),
        "system_xml_sha256": _hash_bytes(system_xml.encode()),
        "measurement_identity": measurement_identity,
    }


def _array_sha256(array: np.ndarray) -> str:
    return _hash_bytes(np.ascontiguousarray(array).tobytes())


def _load_density_handoff_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path / "npt_to_nvt_handoff.npz") as payload:
        return np.asarray(payload["positions_nm"], dtype=float), np.asarray(payload["box_vectors_nm"], dtype=float)


def _density_handoff_record(path: Path) -> dict[str, Any]:
    summary_path = path / "density_preparation.json"
    handoff_json_path = path / "npt_to_nvt_handoff.json"
    handoff_npz_path = path / "npt_to_nvt_handoff.npz"
    manifest_path = path / "artifact_manifest.json"
    summary = _read_json(summary_path)
    handoff = _read_json(handoff_json_path)
    artifact_manifest = _read_json(manifest_path)
    if summary.get("status") != "density_completed" or not summary.get("density_preparation_completed"):
        raise AdmissionError("Density handoff parent must be a completed density preparation")
    if summary.get("equilibrium_certified") or summary.get("production_ready") or handoff.get("equilibrium_certified"):
        raise AdmissionError("Density handoff must not self-certify equilibrium or production readiness")
    if handoff.get("role") != "npt_density_to_nvt_coordinate_box_handoff_only":
        raise AdmissionError("Density handoff role must be coordinate/box handoff only")
    salt_gate = summary.get("salt_protonation_gate", {})
    if not summary.get("synthetic_fixture"):
        if not isinstance(salt_gate, dict) or not salt_gate.get("production_prep_label_allowed") or not salt_gate.get("salt_protonation_manifest"):
            raise AdmissionError("Production density handoff requires salt/protonation preparation provenance")
    expected = summary.get("handoff_sha256", {})
    actual_json_sha = _sha256(handoff_json_path)
    actual_npz_sha = _sha256(handoff_npz_path)
    if expected.get("npt_to_nvt_handoff.json") != actual_json_sha:
        raise AdmissionError("Density handoff JSON hash mismatch")
    if expected.get("npt_to_nvt_handoff.npz") != actual_npz_sha:
        raise AdmissionError("Density handoff NPZ hash mismatch")
    artifacts = artifact_manifest.get("artifacts", {})
    if artifacts.get("density_preparation.json", {}).get("sha256") != _sha256(summary_path):
        raise AdmissionError("Density artifact manifest summary hash mismatch")
    positions, box = _load_density_handoff_arrays(path)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise AdmissionError("Density handoff positions_nm must be finite Nx3 coordinates")
    if box.shape != (3, 3) or not np.isfinite(box).all() or np.linalg.det(box) <= 0:
        raise AdmissionError("Density handoff box_vectors_nm must be a finite positive-volume 3x3 box")
    return {
        "kind": "density_handoff",
        "path": str(path.resolve()),
        "density_manifest_sha256": _sha256(summary_path),
        "artifact_manifest_sha256": _sha256(manifest_path),
        "handoff_json_sha256": actual_json_sha,
        "handoff_npz_sha256": actual_npz_sha,
        "positions_nm_sha256": _array_sha256(positions),
        "box_vectors_nm_sha256": _array_sha256(box),
        "positions_shape": list(positions.shape),
        "box_volume_nm3": float(np.linalg.det(box)),
        "state_xml_sha256": actual_npz_sha,
        "phase": "density_handoff",
        "force_kj_mol_nm": 0.0,
        "model": summary.get("model"),
        "input_hashes": dict(summary.get("input_hashes_verified", {})),
        "data_input_hashes": {
            key: summary.get("input_hashes_verified", {})[key]
            for key in DATA_INPUT_HASH_KEYS
            if key in summary.get("input_hashes_verified", {})
        },
        "core_indices_sha256": _canonical_hash(summary.get("gauge", {}).get("core_indices")),
        "reference_nm_sha256": _canonical_hash(summary.get("gauge", {}).get("reference_core_nm")),
        "q_ambient_sha256": _canonical_hash(summary.get("gauge", {}).get("q_ambient")),
        "density_gauge_sha256": _canonical_hash(summary.get("gauge")),
        "salt_protonation_gate": salt_gate,
        "synthetic_fixture": bool(summary.get("synthetic_fixture")),
        "admission_role": "preparation_provenance_only_no_equilibrium_claim",
    }



def _validate_common_measurement_rows(rows: Any, *, synthetic_fixture: bool) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise AdmissionError("Measurement identity requires ordered common_identity_rows")
    if not synthetic_fixture and len(rows) != 269:
        raise AdmissionError("Production CRBN measurement identity requires 269 ordered residues")
    normalized = []
    seen = set()
    for expected_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise AdmissionError("Measurement identity rows must be JSON objects")
        item = {
            "order_index": row.get("order_index"),
            "uniprot_accession": row.get("uniprot_accession"),
            "author_resnum": row.get("author_resnum"),
            "canonical_resname": row.get("canonical_resname"),
            "atom_name": row.get("atom_name"),
        }
        if item["order_index"] != expected_index or item["uniprot_accession"] != "Q96SW2" or item["atom_name"] != "CA":
            raise AdmissionError("Measurement identity common rows must be Q96SW2 ordered CA residues")
        if isinstance(item["author_resnum"], bool) or not isinstance(item["author_resnum"], int):
            raise AdmissionError("Measurement identity common rows require integer author_resnum")
        if not isinstance(item["canonical_resname"], str) or len(item["canonical_resname"]) != 3:
            raise AdmissionError("Measurement identity common rows require canonical residue types")
        key = (item["author_resnum"], item["canonical_resname"])
        if key in seen:
            raise AdmissionError("Measurement identity common rows must be unique canonical residues")
        seen.add(key)
        normalized.append(item)
    return normalized


def _validate_core_residue_csv(record: Mapping[str, Any], common_rows: list[Mapping[str, Any]], base: Path) -> dict[str, Any]:
    path_value = record.get("path") if isinstance(record, Mapping) else None
    sha_value = record.get("sha256") if isinstance(record, Mapping) else None
    if not path_value or not sha_value:
        raise AdmissionError("Measurement identity requires hash-bound core_residue_file")
    path = Path(path_value)
    if not path.is_absolute():
        path = base / path
    if _sha256(path) != sha_value:
        raise AdmissionError("Measurement identity core_residue_file hash mismatch")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        author_resnums = [int(row["author_resnum"]) for row in reader]
    expected = [int(row["author_resnum"]) for row in common_rows]
    if author_resnums != expected:
        raise AdmissionError("Measurement identity common rows do not match data/crbn_residue_window.csv order")
    return {"path": str(path.resolve()), "sha256": sha_value}


def _topology_atoms_by_index(topology) -> dict[int, Any]:
    if topology is None:
        raise AdmissionError("Measurement identity validation requires topology")
    return {atom.index: atom for atom in topology.atoms()}


def _assert_residue_compatible(actual: str, expected: str) -> None:
    aliases = {"CY1": "CYS", "CY2": "CYS", "CY3": "CYS", "CY4": "CYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS"}
    if aliases.get(actual, actual) != aliases.get(expected, expected):
        raise AdmissionError("Measurement identity endpoint topology residue type mismatch")


def _validate_measurement_identity(
    path: Path,
    *,
    mapping: Mapping[str, Any],
    topology,
    model: str,
    expected_sha256: str | None = None,
    synthetic_fixture: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    artifact_sha = _sha256(path)
    if expected_sha256 is not None and artifact_sha != expected_sha256:
        raise AdmissionError("Measurement identity artifact hash mismatch")
    data = _read_json(path)
    if data.get("schema_version") != "1.0" or data.get("status") != "pass":
        raise AdmissionError("Measurement identity artifact must have schema_version 1.0 and status pass")
    if data.get("response_bridge", {}).get("status") != "ready_for_engine_binding_not_production_admission":
        raise AdmissionError("Measurement identity bridge status is not engine-bindable")
    if data.get("response_bridge", {}).get("trajectory_refit_or_new_q_projection") is not False:
        raise AdmissionError("Measurement identity bridge must prohibit trajectory refit or new q projection")
    common_rows = _validate_common_measurement_rows(data.get("common_identity_rows"), synthetic_fixture=synthetic_fixture)
    common_sha = _canonical_hash(common_rows)
    if data.get("common_identity_sha256") != common_sha:
        raise AdmissionError("Measurement identity common_identity_sha256 mismatch")
    core_residue_file = _validate_core_residue_csv(data.get("core_residue_file", {}), common_rows, path.parent)
    endpoint = "isolated" if model == "isolated" else "joint"
    endpoint_data = data.get("endpoints", {}).get(endpoint)
    if not isinstance(endpoint_data, Mapping):
        raise AdmissionError(f"Measurement identity missing {endpoint} endpoint")
    if endpoint_data.get("identity_rows") != common_rows:
        raise AdmissionError("Measurement identity endpoint common rows differ from shared canonical identity")
    endpoint_rows = endpoint_data.get("endpoint_rows")
    if not isinstance(endpoint_rows, list) or len(endpoint_rows) != len(common_rows):
        raise AdmissionError("Measurement identity endpoint rows must match common identity length")
    core_indices = list(map(int, mapping["core_indices"]))
    if len(core_indices) != len(common_rows):
        raise AdmissionError("Measurement identity core count does not match current mapping")
    atoms = _topology_atoms_by_index(topology)
    for expected_index, (common, row, core_index) in enumerate(zip(common_rows, endpoint_rows, core_indices)):
        for key in ("order_index", "uniprot_accession", "author_resnum", "canonical_resname", "atom_name"):
            if row.get(key) != common[key]:
                raise AdmissionError(f"Measurement identity endpoint row canonical mismatch: {key}")
        if row.get("endpoint") != endpoint:
            raise AdmissionError("Measurement identity endpoint row label mismatch")
        if row.get("atom_index_zero_based") != core_index or row.get("order_index") != expected_index:
            raise AdmissionError("Measurement identity endpoint atom order does not match current mapping core_indices")
        atom = atoms.get(core_index)
        if atom is None or atom.name != "CA":
            raise AdmissionError("Measurement identity endpoint topology core atom is not CA")
        if row.get("amber_residue_index_zero_based") != atom.residue.index:
            raise AdmissionError("Measurement identity endpoint topology residue index mismatch")
        _assert_residue_compatible(atom.residue.name, row.get("amber_residue_name", common["canonical_resname"]))
    endpoint_summary = endpoint_data.get("summary", {})
    if endpoint_summary.get("core_count") != len(core_indices):
        raise AdmissionError("Measurement identity endpoint core_count mismatch")
    if endpoint_summary.get("first_core_atom_index") != core_indices[0] or endpoint_summary.get("last_core_atom_index") != core_indices[-1]:
        raise AdmissionError("Measurement identity endpoint first/last core atom mismatch")
    if endpoint_summary.get("mapping_core_indices_sha256") != _canonical_hash(core_indices):
        raise AdmissionError("Measurement identity endpoint core_indices hash mismatch")
    endpoint_identity_sha = _canonical_hash(endpoint_rows)
    if endpoint_summary.get("endpoint_identity_sha256") != endpoint_identity_sha:
        raise AdmissionError("Measurement identity endpoint_identity_sha256 mismatch")
    if endpoint == "isolated":
        transport = endpoint_data.get("transport_from_joint", {})
        if transport.get("status") != "pass" or transport.get("no_new_q_projection") is not True:
            raise AdmissionError("Isolated measurement identity transport must pass with no new q projection")
        mapping_transport = mapping.get("q_transport")
        expected_transport = transport.get("mapping_q_transport", {})
        if not isinstance(mapping_transport, Mapping) or mapping_transport.get("no_new_q_projection") is not True:
            raise AdmissionError("Isolated mapping requires q_transport with no_new_q_projection=True")
        for key in ("rotation_matrix", "source_center_nm", "target_center_nm"):
            if key not in mapping_transport:
                raise AdmissionError(f"Isolated mapping q_transport missing {key}")
            if not np.allclose(np.asarray(mapping_transport[key], dtype=float), np.asarray(expected_transport.get(key), dtype=float), rtol=0.0, atol=1e-12):
                raise AdmissionError(f"Isolated mapping q_transport mismatch: {key}")
        rotation = np.asarray(mapping_transport["rotation_matrix"], dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise AdmissionError("Isolated q_transport rotation must be finite 3x3")
        orth_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant = float(np.linalg.det(rotation))
        if orth_error > 1e-10 or abs(determinant - 1.0) > 1e-10:
            raise AdmissionError("Isolated q_transport rotation must be proper orthogonal det+1")
        joint_mapping_record = data.get("joint_reference_q", {}).get("mapping", {})
        joint_mapping_path = _bound_file_record(path.parent, joint_mapping_record, "measurement_identity joint mapping")
        joint_mapping = _read_json(joint_mapping_path)
        joint_reference = np.asarray(joint_mapping["reference_nm"], dtype=float)
        joint_q_flat = np.asarray(joint_mapping["q"], dtype=float).reshape(-1)
        if _canonical_hash(joint_reference.tolist()) != data.get("joint_reference_q", {}).get("reference_nm_sha256"):
            raise AdmissionError("Measurement identity declared joint reference hash does not match loaded joint mapping")
        if _canonical_hash(joint_q_flat.tolist()) != data.get("joint_reference_q", {}).get("q_sha256"):
            raise AdmissionError("Measurement identity declared joint q hash does not match loaded joint mapping")
        joint_q = joint_q_flat.reshape(-1, 3)
        source_center = np.asarray(mapping_transport["source_center_nm"], dtype=float)
        target_center = np.asarray(mapping_transport["target_center_nm"], dtype=float)
        expected_reference = (joint_reference - source_center) @ rotation + target_center
        expected_q = (joint_q @ rotation).reshape(-1)
        current_reference = np.asarray(mapping["reference_nm"], dtype=float)
        current_q = np.asarray(mapping["q"], dtype=float).reshape(-1)
        if not np.allclose(current_reference, expected_reference, rtol=0.0, atol=1e-12):
            raise AdmissionError("Isolated reference_nm does not match proper-rigid transported joint reference")
        if not np.allclose(current_q, expected_q, rtol=0.0, atol=1e-12):
            raise AdmissionError("Isolated q does not match proper-rigid transported joint q")
        if _canonical_hash(current_reference.tolist()) != transport.get("isolated_reference_nm_sha256"):
            raise AdmissionError("Isolated reference hash does not match measurement bridge")
        if _canonical_hash(current_q.tolist()) != transport.get("isolated_q_sha256"):
            raise AdmissionError("Isolated q hash does not match measurement bridge")
        bridge_record = {
            "endpoint": endpoint,
            "joint_reference_nm_sha256": data.get("joint_reference_q", {}).get("reference_nm_sha256"),
            "joint_q_ambient_sha256": data.get("joint_reference_q", {}).get("q_sha256"),
            "isolated_reference_nm_sha256": transport.get("isolated_reference_nm_sha256"),
            "isolated_q_ambient_sha256": transport.get("isolated_q_sha256"),
            "q_transport_sha256": _canonical_hash({key: mapping_transport[key] for key in ("rotation_matrix", "source_center_nm", "target_center_nm", "no_new_q_projection")}),
            "rotation_determinant": determinant,
            "rotation_max_orthogonality_error": orth_error,
            "formula": transport.get("formula"),
        }
    else:
        current_reference = np.asarray(mapping["reference_nm"], dtype=float)
        current_q = np.asarray(mapping["q"], dtype=float).reshape(-1)
        if _canonical_hash(current_reference.tolist()) != data.get("joint_reference_q", {}).get("reference_nm_sha256"):
            raise AdmissionError("Joint reference hash does not match measurement identity")
        if _canonical_hash(current_q.tolist()) != data.get("joint_reference_q", {}).get("q_sha256"):
            raise AdmissionError("Joint q hash does not match measurement identity")
        bridge_record = {
            "endpoint": endpoint,
            "joint_reference_nm_sha256": data.get("joint_reference_q", {}).get("reference_nm_sha256"),
            "joint_q_ambient_sha256": data.get("joint_reference_q", {}).get("q_sha256"),
        }
    return {
        "provided": True,
        "path": str(path.resolve()),
        "sha256": artifact_sha,
        "schema_version": data.get("schema_version"),
        "measurement_kind": data.get("measurement_kind"),
        "common_identity_sha256": common_sha,
        "common_identity_count": len(common_rows),
        "core_residue_file": core_residue_file,
        "endpoint_identity_sha256": endpoint_identity_sha,
        "endpoint_mapping_core_indices_sha256": endpoint_summary.get("mapping_core_indices_sha256"),
        "bridge": bridge_record,
    }

def _parent_record(path: Path) -> dict[str, Any]:
    if path.is_dir():
        if (path / "segment_manifest.json").exists():
            manifest = _read_json(path / "segment_manifest.json")
            state = path / manifest["current_generation"] / "segment_state.xml"
            generation_sha256 = manifest.get("generation_sha256", {})
            missing = set(GENERATION_ARTIFACTS) - set(generation_sha256)
            if missing:
                raise ResumeMismatch(f"Parent segment generation hash map missing: {', '.join(sorted(missing))}")
            if _sha256(state) != generation_sha256["segment_state.xml"]:
                raise ResumeMismatch("Parent segment StateXML hash mismatch")
            return {
                "kind": "segment_state",
                "path": str(path.resolve()),
                "manifest_sha256": _sha256(path / "segment_manifest.json"),
                "state_xml": str(state.resolve()),
                "state_xml_sha256": _sha256(state),
                "system_xml_sha256": _sha256(path / "segment_system.xml"),
                "last_committed_step": manifest["last_committed_step"],
                "last_committed_time_ps": manifest["last_committed_time_ps"],
                "phase": manifest.get("phase"),
                "force_kj_mol_nm": manifest.get("force_kj_mol_nm"),
                "model": manifest.get("model"),
                "identity": {key: manifest.get(key) for key in IDENTITY_BINDING_KEYS},
                "gauge_sha256": _canonical_hash(manifest.get("gauge")),
                "body_sha256": _canonical_hash(manifest.get("body")),
                "body_kind": manifest.get("body", {}).get("kind"),
                "settings_gauge_k": manifest.get("settings", {}).get("gauge_k"),
            }
        if (path / "density_preparation.json").exists() and (path / "npt_to_nvt_handoff.npz").exists():
            return _density_handoff_record(path)
        raise AdmissionError("Parent directory is neither a segment state nor a density handoff")
    return {"kind": "state_xml", "path": str(path.resolve()), "state_xml": str(path.resolve()), "state_xml_sha256": _sha256(path)}


def _certificate(
    path: Path,
    *,
    expected_kind: str,
    phase: str,
    model: str,
    force_h: float,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    data = _read_json(path)
    if data.get("schema_version") != CERT_SCHEMA_VERSION or data.get("status") != "pass":
        raise AdmissionError(f"{expected_kind} certificate must have schema_version 1.0 and status pass")
    if data.get("certificate_kind") != expected_kind:
        raise AdmissionError(f"Expected {expected_kind} certificate")
    for key, value in {"phase": phase, "model": model, "force_kj_mol_nm": float(force_h)}.items():
        if data.get(key) != value:
            raise AdmissionError(f"{expected_kind} certificate is not bound to {key}")
    for key, value in binding.items():
        if data.get("binding", {}).get(key) != value:
            raise AdmissionError(f"{expected_kind} certificate binding mismatch: {key}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "payload_sha256": _canonical_hash(data)}


def _bound_file_record(base: Path, record: Mapping[str, Any], label: str) -> Path:
    if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
        raise AdmissionError(f"{label} requires path and sha256")
    path = Path(record["path"])
    if not path.is_absolute():
        path = base / path
    if _sha256(path) != record["sha256"]:
        raise AdmissionError(f"{label} hash mismatch")
    return path


def _npz_calibration_artifact(path: Path, binding: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    try:
        with np.load(path) as payload:
            required = {"core_displacement_nm", "time_ps", "reference_nm", "q_ambient", "core_indices", "force_kj_mol_nm"}
            missing = required - set(payload.files)
            if missing:
                raise AdmissionError(f"zero_calibration_artifact missing engine export key(s): {', '.join(sorted(missing))}")
            core_displacement = np.asarray(payload["core_displacement_nm"], dtype=float)
            times = np.asarray(payload["time_ps"], dtype=float)
            reference_nm = np.asarray(payload["reference_nm"], dtype=float)
            q = np.asarray(payload["q_ambient"], dtype=float).reshape(-1)
            core_indices = np.asarray(payload["core_indices"], dtype=int)
            force_array = np.asarray(payload["force_kj_mol_nm"], dtype=float).reshape(-1)
            force = float(force_array[0])
    except OSError as exc:
        raise AdmissionError("zero_calibration_artifact must be a readable NPZ file") from exc
    if core_displacement.ndim != 3 or core_displacement.shape[1:] != reference_nm.shape or reference_nm.ndim != 2 or reference_nm.shape[1] != 3:
        raise AdmissionError("zero_calibration_artifact core_displacement/reference shape mismatch")
    if times.ndim != 1 or len(times) != core_displacement.shape[0] or len(times) < 2:
        raise AdmissionError("zero_calibration_artifact requires at least two time-matched frames")
    if not np.isfinite(core_displacement).all() or not np.isfinite(times).all() or not np.isfinite(reference_nm).all() or not np.isfinite(q).all():
        raise AdmissionError("zero_calibration_artifact contains non-finite values")
    if abs(force) > 1e-12:
        raise AdmissionError("zero_calibration_artifact must be zero-force h=0 data")
    if q.size != core_displacement.shape[1] * 3:
        raise AdmissionError("zero_calibration_artifact q_ambient length does not match core displacement")
    if _canonical_hash(core_indices.astype(int).tolist()) != binding.get("core_indices_sha256"):
        raise AdmissionError("zero_calibration_artifact core_indices binding mismatch")
    if _canonical_hash(reference_nm.tolist()) != binding.get("reference_nm_sha256"):
        raise AdmissionError("zero_calibration_artifact reference_nm binding mismatch")
    if _canonical_hash(q.tolist()) != binding.get("q_ambient_sha256"):
        raise AdmissionError("zero_calibration_artifact q_ambient binding mismatch")
    arrays = {
        "core_displacement_nm": core_displacement,
        "time_ps": times,
        "reference_nm": reference_nm,
        "q_ambient": q,
        "core_indices": core_indices.astype(np.int64),
        "force_kj_mol_nm": force_array.astype(float),
    }
    content = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": _hash_bytes(np.ascontiguousarray(array).tobytes()),
        }
        for name, array in arrays.items()
    }
    content_sha256 = _canonical_hash({
        "schema_version": SEGMENT_SCHEMA_VERSION,
        "artifact_kind": "zero_calibration_engine_npz_scientific_arrays",
        "arrays": content,
    })
    return core_displacement.reshape(core_displacement.shape[0], -1) @ q, content_sha256


def _npz_q_observable(path: Path, binding: Mapping[str, Any]) -> np.ndarray:
    return _npz_calibration_artifact(path, binding)[0]


def _validate_calibration_manifest(path: Path, row: Mapping[str, Any], binding: Mapping[str, Any], artifact_sha256: str) -> Mapping[str, Any]:
    manifest = _read_json(path)
    if manifest.get("schema_version") != SEGMENT_SCHEMA_VERSION or manifest.get("status") != "segment_complete":
        raise AdmissionError("zero_calibration_manifest must be a completed response segment manifest")
    for key, expected in {
        "phase": "zero_calibration",
        "model": row.get("model"),
        "replicate_id": row.get("replicate_id"),
        "force_kj_mol_nm": 0.0,
    }.items():
        if manifest.get(key) != expected:
            raise AdmissionError(f"zero_calibration_manifest mismatch: {key}")
    if manifest.get("current_generation") is None or manifest.get("generation_sha256", {}).get("segment_observables.npz") != artifact_sha256:
        raise AdmissionError("zero_calibration_manifest is not bound to the declared NPZ artifact")
    for key in IDENTITY_BINDING_KEYS:
        if manifest.get(key) != binding.get(key):
            raise AdmissionError(f"zero_calibration_manifest identity mismatch: {key}")
    if manifest.get("seed_lineage") != row.get("initialization_lineage"):
        raise AdmissionError("Variance row initialization_lineage does not match zero_calibration_manifest seed_lineage")
    return manifest


def _effective_initialization_record(lineage: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    if "thermostat_seed" not in lineage:
        raise AdmissionError("Variance row initialization_lineage missing thermostat_seed")
    actual_parent_sha = manifest.get("admission", {}).get("parent_state", {}).get("state_xml_sha256")
    declared_parent_sha = lineage.get("parent_state_sha256")
    initial_positions_sha = lineage.get("initial_positions_sha256")
    initial_velocities_sha = lineage.get("initial_velocities_sha256")
    # A calibration segment that starts from a parent StateXML inherits actual
    # coordinates and velocities from manifest.admission.parent_state.  Declared
    # initial_* fields or master/replicate labels cannot make that start
    # independent, and row lineage may not override the actual parent hash.
    if actual_parent_sha is not None:
        if declared_parent_sha is not None and declared_parent_sha != actual_parent_sha:
            raise AdmissionError("Variance row parent_state_sha256 does not match manifest admission parent")
        return {
            "thermostat_seed": lineage["thermostat_seed"],
            "parent_state_sha256": actual_parent_sha,
        }
    if declared_parent_sha is not None:
        raise AdmissionError("Variance row parent_state_sha256 is not bound by zero_calibration_manifest admission")
    if initial_positions_sha is None or initial_velocities_sha is None:
        raise AdmissionError("Variance row initialization_lineage must bind actual initial positions and velocities")
    return {
        "thermostat_seed": lineage["thermostat_seed"],
        "velocity_seed": lineage.get("velocity_seed"),
        "initial_positions_sha256": initial_positions_sha,
        "initial_velocities_sha256": initial_velocities_sha,
    }


def _calibration_rows_digest(records: list[Mapping[str, Any]]) -> str:
    return _canonical_hash({
        "schema_version": SEGMENT_SCHEMA_VERSION,
        "observable": "q_ambient_dot_core_displacement_nm",
        "variance_ddof": 1,
        "accepted_zero_calibration_rows": sorted(records, key=lambda row: (row["model"], row["replicate_id"])),
    })


def _measurement_endpoint_expected_hashes(identity: Mapping[str, Any]) -> tuple[str | None, str | None]:
    bridge = identity.get("bridge", {})
    if not isinstance(bridge, Mapping):
        return None, None
    endpoint = bridge.get("endpoint")
    if endpoint == "isolated":
        return bridge.get("isolated_reference_nm_sha256"), bridge.get("isolated_q_ambient_sha256")
    if endpoint == "joint":
        return bridge.get("joint_reference_nm_sha256"), bridge.get("joint_q_ambient_sha256")
    return None, None


def _validate_binding_matches_measurement_endpoint(binding: Mapping[str, Any], identity: Mapping[str, Any], label: str) -> None:
    expected_reference, expected_q = _measurement_endpoint_expected_hashes(identity)
    if not expected_reference or not expected_q:
        raise AdmissionError(f"{label} measurement identity missing endpoint reference/q hashes")
    if binding.get("reference_nm_sha256") != expected_reference:
        raise AdmissionError(f"{label} measurement identity binding mismatch: reference_nm_sha256")
    if binding.get("q_ambient_sha256") != expected_q:
        raise AdmissionError(f"{label} measurement identity binding mismatch: q_ambient_sha256")


def _shared_measurement_identity_matches(row_binding: Mapping[str, Any], current_binding: Mapping[str, Any]) -> bool:
    row_identity = row_binding.get("measurement_identity")
    current_identity = current_binding.get("measurement_identity")
    if not isinstance(row_identity, Mapping) or not isinstance(current_identity, Mapping):
        return False
    if not row_identity.get("provided") or not current_identity.get("provided"):
        return False
    if row_identity.get("common_identity_sha256") != current_identity.get("common_identity_sha256"):
        return False
    row_bridge = row_identity.get("bridge", {})
    current_bridge = current_identity.get("bridge", {})
    if not isinstance(row_bridge, Mapping) or not isinstance(current_bridge, Mapping):
        return False
    if row_bridge.get("endpoint") != "joint":
        raise AdmissionError("Variance row measurement identity endpoint must be joint")
    _validate_binding_matches_measurement_endpoint(row_binding, row_identity, "Variance row")
    _validate_binding_matches_measurement_endpoint(current_binding, current_identity, "Current segment")
    return (
        bool(row_bridge.get("joint_reference_nm_sha256"))
        and bool(row_bridge.get("joint_q_ambient_sha256"))
        and row_bridge.get("joint_reference_nm_sha256") == current_bridge.get("joint_reference_nm_sha256")
        and row_bridge.get("joint_q_ambient_sha256") == current_bridge.get("joint_q_ambient_sha256")
    )


def _validate_variance_rows(base: Path, rows: Any, binding: Mapping[str, Any]) -> tuple[float, str]:
    if not isinstance(rows, list) or len(rows) != 9:
        raise AdmissionError("Locked force plan requires nine accepted zero-calibration variance rows")
    required_models = {"flexible", "fixed", "rigid"}
    by_model: dict[str, set[str]] = {model: set() for model in required_models}
    lineages_by_model: dict[str, set[str]] = {model: set() for model in required_models}
    artifact_hashes_by_model: dict[str, set[str]] = {model: set() for model in required_models}
    artifact_content_hashes_by_model: dict[str, set[str]] = {model: set() for model in required_models}
    variances = []
    digest_records: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise AdmissionError("Variance rows must be JSON objects")
        model = row.get("model")
        replicate_id = row.get("replicate_id")
        if model not in required_models or not isinstance(replicate_id, str) or not replicate_id:
            raise AdmissionError("Variance row requires complex model and replicate_id")
        if replicate_id in by_model[model]:
            raise AdmissionError("Variance rows require three distinct replicate IDs per model")
        by_model[model].add(replicate_id)
        if not isinstance(row.get("initialization_lineage"), Mapping) or not row["initialization_lineage"]:
            raise AdmissionError("Variance row requires initialization_lineage")
        row_binding = row.get("binding")
        if not isinstance(row_binding, Mapping):
            raise AdmissionError("Variance row requires binding")
        # Calibration rows are bound to their own completed zero-calibration
        # segment manifests.  Do not compare endpoint data/topology hashes to the
        # current segment here; only the shared measurement definition is common.
        for key in IDENTITY_BINDING_KEYS:
            if key not in row_binding:
                raise AdmissionError(f"Variance row binding missing: {key}")
        if not _shared_measurement_identity_matches(row_binding, binding):
            for key in ("reference_nm_sha256", "q_ambient_sha256"):
                if row_binding.get(key) != binding.get(key):
                    raise AdmissionError(f"Variance row common measurement mismatch: {key}")
        artifact_record = row.get("zero_calibration_artifact", {})
        artifact_path = _bound_file_record(base, artifact_record, "zero_calibration_artifact")
        artifact_sha = _sha256(artifact_path)
        artifact_hashes_by_model[model].add(artifact_sha)
        q_observable, artifact_content_sha = _npz_calibration_artifact(artifact_path, row_binding)
        if artifact_content_sha in artifact_content_hashes_by_model[model]:
            raise AdmissionError("Variance rows require distinct calibration NPZ scientific array content within each model")
        artifact_content_hashes_by_model[model].add(artifact_content_sha)
        actual_variance = float(np.var(q_observable, ddof=1))
        variance = row.get("sample_variance_Q_nm2")
        if isinstance(variance, bool) or not isinstance(variance, (int, float)) or not np.isfinite(variance) or variance <= 0:
            raise AdmissionError("Variance row requires positive finite sample_variance_Q_nm2")
        if abs(float(variance) - actual_variance) > max(1e-12, 1e-9 * abs(actual_variance)):
            raise AdmissionError("Variance row sample_variance_Q_nm2 does not match hash-bound calibration NPZ ddof=1 variance")
        variances.append(actual_variance)
        manifest_record = row.get("zero_calibration_manifest", {})
        manifest_path = _bound_file_record(base, manifest_record, "zero_calibration_manifest")
        calibration_manifest = _validate_calibration_manifest(manifest_path, row, row_binding, artifact_sha)
        effective_lineage = _effective_initialization_record(row["initialization_lineage"], calibration_manifest)
        lineage_hash = _canonical_hash(effective_lineage)
        if lineage_hash in lineages_by_model[model]:
            raise AdmissionError("Variance rows require three distinct effective initialization lineages per model")
        lineages_by_model[model].add(lineage_hash)
        stationarity_record = row.get("stationarity_certificate", {})
        stationarity_path = _bound_file_record(base, stationarity_record, "stationarity_certificate")
        stationarity = _read_json(stationarity_path)
        if stationarity.get("schema_version") != CERT_SCHEMA_VERSION or stationarity.get("status") != "pass":
            raise AdmissionError("Stationarity certificate must have schema_version 1.0 and status pass")
        if stationarity.get("certificate_kind") != "zero_calibration_stationarity_acceptance":
            raise AdmissionError("Stationarity certificate kind mismatch")
        if stationarity.get("model") != model or stationarity.get("replicate_id") != replicate_id:
            raise AdmissionError("Stationarity certificate model/replicate mismatch")
        if stationarity.get("artifact_sha256") != artifact_sha:
            raise AdmissionError("Stationarity certificate artifact hash mismatch")
        artifact_selection = stationarity.get("artifact_selection")
        if artifact_selection != {"observable": "q_ambient_dot_core_displacement_nm", "variance_ddof": 1}:
            raise AdmissionError("Stationarity certificate must bind the calibration data selection and ddof")
        for key in ("data_input_hashes", "core_indices_sha256", "reference_nm_sha256", "q_ambient_sha256"):
            if stationarity.get("binding", {}).get(key) != row_binding.get(key):
                raise AdmissionError(f"Stationarity certificate binding mismatch: {key}")
        digest_records.append({
            "model": model,
            "replicate_id": replicate_id,
            "sample_variance_Q_nm2": actual_variance,
            "effective_initialization_lineage": effective_lineage,
            "binding_sha256": _canonical_hash(row_binding),
            "artifact_sha256": artifact_sha,
            "artifact_content_sha256": artifact_content_sha,
            "manifest_sha256": _sha256(manifest_path),
            "stationarity_certificate_sha256": _sha256(stationarity_path),
            "artifact_selection": artifact_selection,
        })
    if {model: len(replicates) for model, replicates in by_model.items()} != {model: 3 for model in required_models}:
        raise AdmissionError("Variance rows require three accepted independent replicas for each complex model")
    if {model: len(lineages) for model, lineages in lineages_by_model.items()} != {model: 3 for model in required_models}:
        raise AdmissionError("Variance rows require three accepted independent seed lineages for each complex model")
    if {model: len(artifacts) for model, artifacts in artifact_hashes_by_model.items()} != {model: 3 for model in required_models}:
        raise AdmissionError("Variance rows require three distinct calibration artifact files for each complex model")
    if {model: len(artifacts) for model, artifacts in artifact_content_hashes_by_model.items()} != {model: 3 for model in required_models}:
        raise AdmissionError("Variance rows require three distinct calibration NPZ scientific array contents for each complex model")
    return float(np.sqrt(max(variances))), _calibration_rows_digest(digest_records)


def _validate_locked_force_plan(
    path: Path,
    *,
    model: str,
    force_h: float,
    binding: Mapping[str, Any],
    temperature_K: float,
) -> dict[str, Any]:
    data = _read_json(path)
    if data.get("schema_version") != CERT_SCHEMA_VERSION or data.get("status") != "locked":
        raise AdmissionError("Locked force plan must have schema_version 1.0 and status locked")
    if data.get("model") != model:
        raise AdmissionError("Locked force plan model mismatch")
    grid = data.get("force_grid_kj_mol_nm")
    if not isinstance(grid, list) or not grid:
        raise AdmissionError("Locked force plan requires a finite force grid")
    try:
        grid_values = [float(value) for value in grid]
    except (TypeError, ValueError) as exc:
        raise AdmissionError("Locked force plan force grid must be numeric") from exc
    if not np.isfinite(grid_values).all() or len(set(grid_values)) != len(grid_values) or 0.0 not in grid_values:
        raise AdmissionError("Locked force plan force grid must be finite, unique, and include zero")
    rounded = {round(value, 12) for value in grid_values}
    f0 = data.get("f0_kj_mol_nm")
    if isinstance(f0, bool) or not isinstance(f0, (int, float)) or not np.isfinite(f0) or f0 <= 0:
        raise AdmissionError("Locked force plan requires positive finite f0_kj_mol_nm")
    if rounded != {round(-2 * float(f0), 12), round(-float(f0), 12), 0.0, round(float(f0), 12), round(2 * float(f0), 12)}:
        raise AdmissionError("Locked force plan force grid must be exactly [0, ±F0, ±2F0]")
    if round(float(force_h), 12) not in rounded:
        raise AdmissionError("Requested force is not in locked F0 grid")
    multipliers = data.get("force_multipliers")
    if multipliers != [-2, -1, 0, 1, 2]:
        raise AdmissionError("Locked force plan force_multipliers must be [-2, -1, 0, 1, 2]")
    sigma_max = data.get("sigma_max_nm")
    if isinstance(sigma_max, bool) or not isinstance(sigma_max, (int, float)) or not np.isfinite(sigma_max) or sigma_max <= 0:
        raise AdmissionError("Locked force plan requires positive finite sigma_max_nm")
    expected_f0 = 0.25 * pilot.GAS_CONSTANT * float(temperature_K) / float(sigma_max)
    if abs(float(f0) - expected_f0) > max(1e-12, 1e-9 * abs(expected_f0)):
        raise AdmissionError("Locked force plan f0_kj_mol_nm does not match 0.25RT/sigma_max")
    if data.get("locked_f0_rule") != "0.25RT_over_sigma_max":
        raise AdmissionError("Locked force plan must declare locked_f0_rule=0.25RT_over_sigma_max")
    sigma_from_rows, calibration_data_sha256 = _validate_variance_rows(path.parent, data.get("accepted_zero_calibration_variance_rows"), binding)
    if abs(float(sigma_max) - sigma_from_rows) > max(1e-12, 1e-9 * abs(sigma_from_rows)):
        raise AdmissionError("Locked force plan sigma_max_nm does not match accepted variance aggregation")
    source = data.get("source_calibration_certificate")
    if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
        raise AdmissionError("Locked force plan requires a hash-bound source_calibration_certificate")
    source_path = Path(source["path"])
    if not source_path.is_absolute():
        source_path = path.parent / source_path
    if _sha256(source_path) != source["sha256"]:
        raise AdmissionError("Locked force plan source calibration certificate hash mismatch")
    source_payload = _read_json(source_path)
    if source_payload.get("schema_version") != CERT_SCHEMA_VERSION or source_payload.get("status") != "pass":
        raise AdmissionError("Source calibration certificate must have schema_version 1.0 and status pass")
    if source_payload.get("certificate_kind") != source.get("certificate_kind") or source_payload.get("certificate_kind") != "zero_force_calibration":
        raise AdmissionError("Locked force plan source must be a zero_force_calibration certificate")
    if source_payload.get("model") != model:
        raise AdmissionError("Source calibration certificate model mismatch")
    for key, value in binding.items():
        if source_payload.get("binding", {}).get(key) != value:
            raise AdmissionError(f"Source calibration certificate binding mismatch: {key}")
    if data.get("zero_calibration_data_sha256") != calibration_data_sha256:
        raise AdmissionError("Locked force plan zero_calibration_data_sha256 does not match hash-bound row artifacts")
    if source_payload.get("zero_calibration_data_sha256") != calibration_data_sha256:
        raise AdmissionError("Source calibration certificate zero_calibration_data_sha256 mismatch")
    for key, value in binding.items():
        if data.get("binding", {}).get(key) != value:
            raise AdmissionError(f"Locked force plan binding mismatch: {key}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "source_calibration_certificate": {"path": str(source_path.resolve()), "sha256": source["sha256"]},
        "zero_calibration_data_sha256": calibration_data_sha256,
        "f0_kj_mol_nm": float(f0),
        "sigma_max_nm": float(sigma_max),
    }


def _validate_admission(
    *,
    phase: str,
    model: str,
    force_h: float,
    identity: Mapping[str, Any],
    gauge_meta: Mapping[str, Any],
    body_meta: Mapping[str, Any],
    settings: SegmentSettings,
    synthetic_fixture: bool,
    locked_force_plan: Path | None,
    parent_state: Path | None,
    equilibration_certificate: Path | None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"phase must be one of {', '.join(PHASES)}")
    if not np.isfinite(force_h):
        raise AdmissionError("force_kj_mol_nm must be finite")
    if phase in ("zero_equilibration", "zero_calibration") and force_h != 0.0:
        raise AdmissionError(f"{phase} requires force_kj_mol_nm=0")
    binding = {key: identity[key] for key in IDENTITY_BINDING_KEYS}
    evidence: dict[str, Any] = {
        "phase": phase,
        "force_kj_mol_nm": float(force_h),
        "equilibrium_or_response_pass": False,
        "quality_certificate_evaluated_by_engine": "schema_status_and_binding_only",
    }
    if parent_state is not None:
        evidence["parent_state"] = _parent_record(Path(parent_state))
        if evidence["parent_state"]["kind"] == "segment_state":
            parent = evidence["parent_state"]
            if parent.get("model") != model:
                raise AdmissionError("Parent segment model does not match requested response model")
            mismatches = [
                key for key in IDENTITY_BINDING_KEYS
                if parent.get("identity", {}).get(key) != identity.get(key)
            ]
            if mismatches:
                raise AdmissionError(f"Parent segment identity mismatch: {', '.join(mismatches)}")
            if parent.get("gauge_sha256") != _canonical_hash(gauge_meta):
                raise AdmissionError("Parent segment gauge identity mismatch")
            if parent.get("body_sha256") != _canonical_hash(body_meta):
                raise AdmissionError("Parent segment body identity mismatch")
            if float(parent.get("settings_gauge_k")) != float(settings.gauge_k):
                raise AdmissionError("Parent segment gauge_k mismatch")
        binding["parent_state_sha256"] = evidence["parent_state"]["state_xml_sha256"]
        for key in ("positions_nm_sha256", "box_vectors_nm_sha256", "handoff_json_sha256", "handoff_npz_sha256", "density_manifest_sha256"):
            if key in evidence["parent_state"]:
                binding[key] = evidence["parent_state"][key]
    parent_kind = evidence.get("parent_state", {}).get("kind")
    parent_phase = evidence.get("parent_state", {}).get("phase")
    parent_force = evidence.get("parent_state", {}).get("force_kj_mol_nm")
    if parent_kind == "density_handoff":
        parent = evidence["parent_state"]
        if parent.get("synthetic_fixture") and not synthetic_fixture:
            raise AdmissionError("Synthetic density handoff cannot seed a production response segment")
        for key in ("data_input_hashes", "core_indices_sha256", "reference_nm_sha256", "q_ambient_sha256"):
            if parent.get(key) != identity.get(key):
                raise AdmissionError(f"Density handoff identity mismatch: {key}")
        if phase != "zero_equilibration" or force_h != 0.0:
            raise AdmissionError("Density NPT-to-NVT handoff may only seed zero_equilibration")
        if equilibration_certificate is not None:
            raise AdmissionError("Density handoff zero_equilibration accepts preparation provenance only; do not attach a scientific equilibrium certificate")
        evidence["parent_state"]["accepted_as"] = "zero_equilibration_starting_coordinates_and_box_only"
        evidence["parent_state"]["scientific_equilibrium_claim"] = False
        evidence["parent_state"]["requires_later_equilibrium_certificate_for_calibration_or_force"] = True
    if phase == "zero_calibration":
        if equilibration_certificate is None:
            raise AdmissionError("zero_calibration requires a zero-force equilibrium certificate")
        evidence["equilibration_certificate"] = _certificate(
            Path(equilibration_certificate),
            expected_kind="zero_force_equilibrium",
            phase="zero_calibration",
            model=model,
            force_h=0.0,
            binding=binding,
        )
    if force_h != 0.0:
        if model == "isolated" and not synthetic_fixture:
            measurement_identity = identity.get("measurement_identity", {})
            if not isinstance(measurement_identity, Mapping) or not measurement_identity.get("provided"):
                raise AdmissionError("Production isolated nonzero force requires a validated proper-rigid q_transport bridge")
            evidence["measurement_identity"] = measurement_identity
        if locked_force_plan is None:
            raise AdmissionError("Nonzero force requires a locked force plan")
        if parent_state is None:
            raise AdmissionError("Nonzero force requires a parent state")
        if parent_phase is None:
            raise AdmissionError("Nonzero force requires a hash-bound segment parent with recorded phase")
        if phase == "force_equilibration" and (parent_phase not in ("zero_equilibration", "zero_calibration") or float(parent_force) != 0.0):
            raise AdmissionError("force_equilibration must start from a zero-force parent segment")
        if phase == "force_sampling" and (parent_phase != "force_equilibration" or float(parent_force) != float(force_h)):
            raise AdmissionError("force_sampling must start from a matching force_equilibration parent segment")
        evidence["locked_force_plan"] = _validate_locked_force_plan(
            Path(locked_force_plan), model=model, force_h=force_h, binding=binding, temperature_K=settings.temperature_K
        )
    if phase == "force_sampling":
        if equilibration_certificate is None:
            raise AdmissionError("force_sampling requires an external force-equilibration certificate")
        evidence["equilibration_certificate"] = _certificate(
            Path(equilibration_certificate),
            expected_kind="force_equilibration",
            phase="force_sampling",
            model=model,
            force_h=force_h,
            binding=binding,
        )
    elif phase == "force_equilibration":
        if force_h != 0.0 and equilibration_certificate is None:
            raise AdmissionError("force_equilibration requires an external zero-force equilibrium certificate")
        if equilibration_certificate is not None:
            evidence["source_certificate"] = _certificate(
                Path(equilibration_certificate),
                expected_kind="zero_force_equilibrium",
                phase="force_equilibration",
                model=model,
                force_h=0.0,
                binding=binding,
            )
    return evidence


def _load_parent_state(
    context: mm.Context,
    parent: Mapping[str, Any],
    *,
    expected_system_hash: str,
    settings: SegmentSettings,
    prepared_positions_nm: np.ndarray | None = None,
) -> None:
    if parent.get("kind") == "density_handoff":
        with np.load(Path(parent["path"]) / "npt_to_nvt_handoff.npz") as payload:
            positions = np.asarray(payload["positions_nm"], dtype=float)
            box = np.asarray(payload["box_vectors_nm"], dtype=float)
        if _array_sha256(positions) != parent["positions_nm_sha256"] or _array_sha256(box) != parent["box_vectors_nm_sha256"]:
            raise AdmissionError("Density handoff coordinate or box hash mismatch")
        if prepared_positions_nm is None:
            prepared_positions_nm = positions
        prepared_positions_nm = np.asarray(prepared_positions_nm, dtype=float)
        if len(positions) > context.getSystem().getNumParticles() or len(prepared_positions_nm) != context.getSystem().getNumParticles():
            raise AdmissionError("Density handoff particle count does not match this NVT System")
        if not np.allclose(prepared_positions_nm[:len(positions)], positions, rtol=0.0, atol=0.0):
            raise AdmissionError("Prepared boundary positions are not derived from the density handoff")
        context.setPeriodicBoxVectors(*box)
        context.setPositions(prepared_positions_nm)
        context.computeVirtualSites()
        context.setVelocitiesToTemperature(settings.temperature_K, settings.velocity_seed)
        context.applyConstraints(settings.constraint_tolerance)
        context.applyVelocityConstraints(settings.constraint_tolerance)
        context.setTime(0.0)
        return
    if parent.get("system_xml_sha256") not in (None, expected_system_hash):
        raise AdmissionError("Parent state was produced by a different serialized System")
    state = mm.XmlSerializer.deserialize(Path(parent["state_xml"]).read_text())
    positions = state.getPositions(asNumpy=True)
    if len(positions) != context.getSystem().getNumParticles():
        raise AdmissionError("Parent state particle count does not match this boundary System")
    context.setPositions(positions)
    velocities = state.getVelocities(asNumpy=True)
    if velocities is not None:
        context.setVelocities(velocities)
    context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    context.setTime(state.getTime())
    context.applyConstraints(settings.constraint_tolerance)
    context.applyVelocityConstraints(settings.constraint_tolerance)


def _build_manifest(
    *,
    phase: str,
    model: str,
    replicate_id: str,
    force_h: float,
    settings: SegmentSettings,
    identity: Mapping[str, Any],
    gauge_meta: Mapping[str, Any],
    body_meta: Mapping[str, Any],
    precision: Mapping[str, Any],
    platform_name: str,
    platform_properties: Mapping[str, Any],
    admission: Mapping[str, Any],
    synthetic_fixture: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SEGMENT_SCHEMA_VERSION,
        "status": "started",
        "segment_complete": False,
        "response_converged": False,
        "production_ready": False,
        "synthetic_fixture": bool(synthetic_fixture),
        "phase": phase,
        "model": model,
        "replicate_id": str(replicate_id),
        "force_kj_mol_nm": float(force_h),
        "settings": asdict(settings),
        "seed_lineage": {
            "master_seed": settings.master_seed,
            "velocity_seed": settings.velocity_seed,
            "thermostat_seed": settings.thermostat_seed,
            "derivation": "sha256(master_seed, model, replicate_id, phase, force, stream)",
        },
        "openmm_version": mm.__version__,
        "platform": platform_name,
        "platform_properties": dict(platform_properties),
        "precision": dict(precision),
        **dict(identity),
        "gauge": dict(gauge_meta),
        "body": dict(body_meta),
        "admission": dict(admission),
        "current_generation": None,
        "phase_start_step": 0,
        "phase_start_time_ps": 0.0,
        "requested_phase_steps": settings.steps,
        "target_global_step": settings.steps,
        "phase_completed_steps": 0,
        "last_committed_step": 0,
        "last_committed_time_ps": 0.0,
        "generation_sha256": {},
        "runtime_diagnostics": [],
        "completion_definition": "segment_complete means requested NVT steps reached with finite recorded states; it is not equilibrium, response, or production pass",
    }


def _compare_manifest(current: Mapping[str, Any], previous: Mapping[str, Any]) -> None:
    keys = (
        "phase", "model", "replicate_id", "force_kj_mol_nm", "settings", "seed_lineage",
        "synthetic_fixture", "input_hashes", "data_input_hashes", "stage_provenance_hashes", "source_provenance",
        "topology", "mapping_sha256", "core_indices_sha256", "reference_nm_sha256",
        "q_ambient_sha256", "system_xml_sha256", "gauge_sha256", "body_sha256",
        "openmm_version", "platform", "platform_properties", "precision", "admission",
    )
    mismatches = [key for key in keys if current.get(key) != previous.get(key)]
    if mismatches:
        raise ResumeMismatch(f"Resume manifest mismatch: {', '.join(mismatches)}")


def _read_generation(output_dir: Path, manifest: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]], list[np.ndarray], list[float]]:
    generation = manifest.get("current_generation")
    if not generation:
        return output_dir, [], [], []
    path = output_dir / generation
    generation_sha256 = manifest.get("generation_sha256", {})
    missing = set(GENERATION_ARTIFACTS) - set(generation_sha256)
    if missing:
        raise ResumeMismatch(f"Generation hash map missing: {', '.join(sorted(missing))}")
    for name, digest in generation_sha256.items():
        if _sha256(path / name) != digest:
            raise ResumeMismatch(f"Generation artifact hash mismatch: {name}")
    with (path / "response_observations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    old = np.load(path / "segment_observables.npz")
    frames = [frame.copy() for frame in old["core_displacement_nm"]]
    times = [float(value) for value in old["time_ps"]]
    if len(rows) != len(times) or any(abs(float(row["time_ps"]) - value) > 1e-12 for row, value in zip(rows, times)):
        raise ResumeMismatch("CSV/NPZ frame grid mismatch")
    return path, rows, frames, times


def _state_step(state: mm.State, timestep_fs: float) -> int:
    return int(round(state.getTime().value_in_unit(unit.picosecond) / (timestep_fs / 1000.0)))


def _kinetic_dof(system: mm.System) -> int:
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())])
    constraints = [system.getConstraintParameters(i) for i in range(system.getNumConstraints())]
    pairs = [tuple(sorted((i, j))) for i, j, _ in constraints]
    if len(set(pairs)) != len(pairs) or any(masses[i] <= 0 or masses[j] <= 0 for i, j in pairs):
        raise ValueError("Temperature counting requires distinct constraints on massive particles")
    dof = int(3 * np.count_nonzero(masses > 0) - len(constraints))
    if dof <= 0:
        raise ValueError("No positive kinetic degrees of freedom")
    return dof


def _platform_options(platform_name: str, precision: str, device_index: str | None, disable_pme_stream: bool):
    if disable_pme_stream and precision != "double":
        raise ValueError("disable_pme_stream is admitted only with double precision")
    signature = inspect.signature(pilot.platform_options)
    if "disable_pme_stream" in signature.parameters:
        return pilot.platform_options(platform_name, precision, device_index, disable_pme_stream=disable_pme_stream)
    if disable_pme_stream:
        raise ValueError("disable_pme_stream requires the shared pilot platform_options helper")
    return pilot.platform_options(platform_name, precision, device_index)


def _precision_record(platform, context, precision: str, disable_pme_stream: bool):
    signature = inspect.signature(pilot.precision_record)
    if "disable_pme_stream" in signature.parameters:
        return pilot.precision_record(platform, context, precision, disable_pme_stream=disable_pme_stream)
    if disable_pme_stream:
        raise ValueError("disable_pme_stream requires the shared pilot precision_record helper")
    return pilot.precision_record(platform, context, precision)


def _constraint_diagnostics(system: mm.System, xyz: np.ndarray, tolerance: float) -> dict[str, Any]:
    if system.getNumConstraints() == 0:
        return {"constraint_count": 0, "max_relative_constraint_error": 0.0, "pass": True}
    errors = []
    for index in range(system.getNumConstraints()):
        i, j, length = system.getConstraintParameters(index)
        target = length.value_in_unit(unit.nanometer)
        errors.append(abs(np.linalg.norm(xyz[i] - xyz[j]) - target) / target)
    max_error = float(max(errors))
    return {"constraint_count": system.getNumConstraints(), "max_relative_constraint_error": max_error, "pass": max_error <= max(10 * tolerance, 1e-5)}


def _rigid_fit_residual_nm(reference_xyz: np.ndarray, current_xyz: np.ndarray) -> float:
    reference_center = reference_xyz.mean(axis=0)
    current_center = current_xyz.mean(axis=0)
    ref0 = reference_xyz - reference_center
    cur0 = current_xyz - current_center
    cov = ref0.T @ cur0
    u, _s, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    fitted = ref0 @ rotation.T + current_center
    return float(np.max(np.linalg.norm(current_xyz - fitted, axis=1)))


def _runtime_screen(
    *,
    stage: str,
    state: mm.State,
    system: mm.System,
    mapping: Mapping[str, Any],
    model: str,
    original_xyz: np.ndarray,
    chemical_geometry,
    synthetic_fixture: bool,
    settings: SegmentSettings,
    gauge_meta: Mapping[str, Any],
    body_meta: Mapping[str, Any],
    kinetic_dof: int,
) -> tuple[dict[str, Any], np.ndarray, float]:
    xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(xyz).all() or not np.isfinite([potential, kinetic]).all():
        raise FloatingPointError("Nonfinite segment state")
    constraints = _constraint_diagnostics(system, xyz, settings.constraint_tolerance)
    if not constraints["pass"]:
        raise ValueError("Distance constraints failed response segment screen")
    core = list(map(int, mapping["core_indices"]))
    reference_nm = np.asarray(mapping["reference_nm"], dtype=float)
    delta = xyz[core] - reference_nm
    rigid_basis = np.asarray(gauge_meta.get("rigid_basis"), dtype=float)
    if rigid_basis.shape == (3 * len(core), 6):
        flat_delta = delta.ravel()
        rigid_coordinates = rigid_basis.T @ flat_delta
        internal_delta = flat_delta - rigid_basis @ rigid_coordinates
        gauge_norm = float(np.linalg.norm(rigid_coordinates))
        internal_rmsd = float(np.linalg.norm(internal_delta) / np.sqrt(len(core)))
    else:
        gauge_norm = float("nan")
        internal_rmsd = float("nan")
    row: dict[str, Any] = {
        "stage": stage,
        "potential_kj_mol": potential,
        "kinetic_kj_mol": kinetic,
        "kinetic_temperature_K": float(2 * kinetic / (kinetic_dof * pilot.GAS_CONSTANT)),
        "kinetic_dof": kinetic_dof,
        "kinetic_dof_rule": "3*npositiveMass - nnonredundantDistanceConstraints; no COM or harmonic-gauge subtraction",
        "core_gauge_norm_nm": gauge_norm,
        "core_internal_rmsd_nm": internal_rmsd,
        **constraints,
    }
    if model == "fixed" and mapping.get("ddb1_atom_indices"):
        movement = float(np.max(np.linalg.norm(xyz[mapping["ddb1_atom_indices"]] - original_xyz[mapping["ddb1_atom_indices"]], axis=1)))
        row["fixed_body_max_displacement_nm"] = movement
        if movement > 1e-10:
            raise ValueError("Fixed DDB1 moved")
    if model == "rigid" and mapping.get("ddb1_atom_indices"):
        body = list(map(int, mapping["ddb1_atom_indices"]))
        rigid_error = _rigid_fit_residual_nm(original_xyz[body], xyz[body]) if len(body) >= 3 else 0.0
        row["rigid_ddb1_max_allatom_fit_residual_nm"] = rigid_error
        row["rigid_ddb1_geometry_pass"] = rigid_error <= max(10 * settings.constraint_tolerance, 1e-5)
        row["rigid_anchor_indices"] = list(map(int, body_meta.get("anchor_indices", [])))
        row["rigid_ddb1_geometry_algorithm"] = "O(Nbody) Kabsch max all-atom residual; no dense pair matrix"
        if not row["rigid_ddb1_geometry_pass"]:
            raise ValueError("Rigid DDB1 geometry failed response segment screen")
    if "zn_atom_index" in mapping:
        distances = np.linalg.norm(pilot._minimum_image(xyz[mapping["zn_sg_indices"]] - xyz[mapping["zn_atom_index"]], box), axis=1)
        bounds = mapping.get("zn_sg_distance_bounds_nm", [0.18, 0.30])
        row["zn_sg_screen_pass"] = bool(np.all((distances >= bounds[0]) & (distances <= bounds[1])))
        row["zn_sg_distances_nm"] = [float(value) for value in distances]
        if not row["zn_sg_screen_pass"]:
            raise ValueError("Zn-SG distances failed the declared physical screen")
    if not synthetic_fixture:
        chemistry = pilot.chemical_geometry_screen(xyz, chemical_geometry)
        row["chemical_geometry_pass"] = chemistry["pass"]
        if not chemistry["pass"]:
            raise ValueError(f"Chemical geometry screen failed: {chemistry['failures'][:10]}")
    return row, xyz, state.getTime().value_in_unit(unit.picosecond)


def _write_generation(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    context: mm.Context,
    rows: list[dict[str, Any]],
    frames: list[np.ndarray],
    frame_times: list[float],
    reference_nm: np.ndarray,
    q: np.ndarray,
    core_indices: list[int],
    force_h: float,
) -> None:
    generation = f"generation_{manifest['last_committed_step']:012d}_{time.time_ns()}"
    tmp_dir = output_dir / (generation + ".tmp")
    gen_dir = output_dir / generation
    tmp_dir.mkdir(parents=True)
    (tmp_dir / "segment.chk").write_bytes(context.createCheckpoint())
    state_xml = mm.XmlSerializer.serialize(context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=False))
    (tmp_dir / "segment_state.xml").write_text(state_xml)
    with (tmp_dir / "response_observations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (tmp_dir / "segment_observables.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            core_displacement_nm=np.asarray(frames).reshape(-1, len(core_indices), 3),
            time_ps=np.asarray(frame_times, dtype=float),
            reference_nm=reference_nm,
            q_ambient=q,
            core_indices=np.asarray(core_indices, dtype=int),
            force_kj_mol_nm=float(force_h),
        )
    tmp_dir.replace(gen_dir)
    manifest["current_generation"] = generation
    manifest["generation_sha256"] = {name: _sha256(gen_dir / name) for name in GENERATION_ARTIFACTS}
    _atomic_json(output_dir / "segment_manifest.json", manifest)


def run_segment(
    *,
    system: mm.System,
    xyz: np.ndarray,
    mapping: Mapping[str, Any],
    settings: SegmentSettings,
    output_dir: Path,
    phase: str,
    model: str,
    replicate_id: str,
    force_h: float,
    platform_name: str = "Reference",
    precision: str = "double",
    device_index: str | None = None,
    topology=None,
    input_hashes: Mapping[str, Any] | None = None,
    qualification: Mapping[str, Any] | None = None,
    chemical_geometry=None,
    synthetic_fixture: bool = False,
    resume: bool = False,
    locked_force_plan: Path | None = None,
    parent_state: Path | None = None,
    equilibration_certificate: Path | None = None,
    measurement_identity: Path | None = None,
    measurement_identity_sha256: str | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    if model not in pilot.MODELS:
        raise ValueError("Unknown DDB1 response model")
    if qualification is not None and not synthetic_fixture and qualification.get("chemical_review", {}).get("status") != pilot.CHEMICAL_REVIEW_PASS:
        raise ValueError(f"chemical_review.status must be {pilot.CHEMICAL_REVIEW_PASS}")
    if not synthetic_fixture:
        if topology is None:
            raise ValueError("Production response segments require topology for strict mapping validation")
        if chemical_geometry is None:
            raise ValueError("Full topology-derived chemical geometry is required unless synthetic_fixture=True")
        mapping = pilot.validate_mapping(mapping, topology, xyz, model)
    settings = _resolved_settings(settings, model=model, replicate_id=replicate_id, phase=phase, force_h=force_h)
    if not synthetic_fixture and platform_name in {"OpenCL", "CUDA"} and not settings.disable_pme_stream:
        raise AdmissionError("Non-synthetic GPU response segments require --disable-pme-stream after the PME stream force-consistency screen")
    deadline = start + settings.max_wall_seconds
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topology_hash = _topology_fingerprint(topology)
    input_hashes = dict(input_hashes or {})
    if measurement_identity is not None:
        input_hashes["measurement_identity"] = _validate_measurement_identity(
            Path(measurement_identity),
            mapping=mapping,
            topology=topology,
            model=model,
            expected_sha256=measurement_identity_sha256,
            synthetic_fixture=synthetic_fixture,
        )
    original_xyz = np.asarray(xyz, dtype=float).copy()
    boundary_xyz = original_xyz
    parent_path = Path(parent_state) if parent_state is not None else None
    density_parent_for_build = None
    if (
        not resume
        and parent_path is not None
        and parent_path.is_dir()
        and (parent_path / "density_preparation.json").exists()
        and (parent_path / "npt_to_nvt_handoff.npz").exists()
    ):
        density_parent_for_build = _density_handoff_record(parent_path)
        handoff_positions, _handoff_box = _load_density_handoff_arrays(parent_path)
        if _array_sha256(handoff_positions) != density_parent_for_build["positions_nm_sha256"]:
            raise AdmissionError("Density handoff coordinate hash mismatch")
        if handoff_positions.shape != original_xyz.shape:
            raise AdmissionError("Density handoff coordinates do not match the unprepared topology particle count")
        boundary_xyz = handoff_positions.copy()
    previous = _read_json(output_dir / "segment_manifest.json") if resume else None
    system_xml_to_publish: bytes | None = None
    prepared_start_xyz: np.ndarray | None = None

    if resume:
        system_xml = (output_dir / "segment_system.xml").read_text()
        prepared_system = mm.XmlSerializer.deserialize(system_xml)
        gauge_meta, body_meta = previous["gauge"], previous["body"]
    elif parent_state is not None and Path(parent_state).is_dir() and (Path(parent_state) / "segment_system.xml").exists():
        if any(output_dir.iterdir()):
            raise ValueError("Fresh segment output directory must be empty")
        parent_manifest = _read_json(Path(parent_state) / "segment_manifest.json")
        system_xml = (Path(parent_state) / "segment_system.xml").read_text()
        prepared_system = mm.XmlSerializer.deserialize(system_xml)
        gauge_meta, body_meta = parent_manifest["gauge"], parent_manifest["body"]
        system_xml_to_publish = system_xml.encode()
    else:
        if any(output_dir.iterdir()):
            raise ValueError("Fresh segment output directory must be empty")
        prepared_system, prepared_xyz, gauge_meta, body_meta, _ = pilot._prepare_system(
            _system_without_barostat(system), boundary_xyz, mapping, model, settings
        )
        prepared_start_xyz = np.asarray(prepared_xyz, dtype=float)
        system_xml = mm.XmlSerializer.serialize(prepared_system)
        system_xml_to_publish = system_xml.encode()

    identity = _identity(system_xml, mapping, topology_hash, input_hashes)
    identity["gauge_sha256"] = _canonical_hash(gauge_meta)
    identity["body_sha256"] = _canonical_hash(body_meta)
    admission = _validate_admission(
        phase=phase,
        model=model,
        force_h=float(force_h),
        identity=identity,
        gauge_meta=gauge_meta,
        body_meta=body_meta,
        settings=settings,
        synthetic_fixture=synthetic_fixture,
        locked_force_plan=locked_force_plan,
        parent_state=parent_state,
        equilibration_certificate=equilibration_certificate,
    )
    if system_xml_to_publish is not None:
        _atomic_write(output_dir / "segment_system.xml", system_xml_to_publish)
    platform, properties = _platform_options(platform_name, precision, device_index, settings.disable_pme_stream)
    integrator = mm.LangevinMiddleIntegrator(settings.temperature_K, settings.friction_per_ps, settings.timestep_fs / 1000.0)
    integrator.setRandomNumberSeed(settings.thermostat_seed)
    integrator.setConstraintTolerance(settings.constraint_tolerance)
    context = mm.Context(prepared_system, integrator, platform, properties)
    try:
        precision_meta = _precision_record(platform, context, precision, settings.disable_pme_stream)
        manifest = _build_manifest(
            phase=phase,
            model=model,
            replicate_id=replicate_id,
            force_h=force_h,
            settings=settings,
            identity=identity,
            gauge_meta=gauge_meta,
            body_meta=body_meta,
            precision=precision_meta,
            platform_name=platform.getName(),
            platform_properties={name: platform.getPropertyValue(context, name) for name in platform.getPropertyNames()},
            admission=admission,
            synthetic_fixture=synthetic_fixture,
        )
        if resume:
            _compare_manifest(manifest, previous)
            generation_dir, rows, frames, frame_times = _read_generation(output_dir, previous)
            context.loadCheckpoint((generation_dir / "segment.chk").read_bytes())
            checkpoint_state = context.getState(getEnergy=True)
            if _state_step(checkpoint_state, settings.timestep_fs) != previous["last_committed_step"]:
                raise ResumeMismatch("Checkpoint step does not match manifest")
            if abs(checkpoint_state.getTime().value_in_unit(unit.picosecond) - previous["last_committed_time_ps"]) > 1e-9:
                raise ResumeMismatch("Checkpoint time does not match manifest")
            if abs(context.getParameter("core_force_h") - float(force_h)) > 1e-12:
                raise ResumeMismatch("Checkpoint force parameter does not match requested segment force")
            if abs(context.getParameter("core_gauge_k") - float(settings.gauge_k)) > 1e-12:
                raise ResumeMismatch("Checkpoint gauge parameter does not match manifest settings")
            manifest.update(previous)
            if manifest.get("requested_phase_steps") != settings.steps:
                raise ResumeMismatch("Requested phase steps do not match resumed segment")
            if manifest.get("target_global_step") != manifest.get("phase_start_step", 0) + settings.steps:
                raise ResumeMismatch("Resume target step is inconsistent with phase start and requested steps")
        else:
            if parent_state is not None:
                _load_parent_state(
                    context,
                    admission["parent_state"],
                    expected_system_hash=identity["system_xml_sha256"],
                    settings=settings,
                    prepared_positions_nm=prepared_start_xyz,
                )
            else:
                context.setPositions(prepared_start_xyz if prepared_start_xyz is not None else original_xyz)
                context.computeVirtualSites()
                context.setVelocitiesToTemperature(settings.temperature_K, settings.velocity_seed)
                context.applyVelocityConstraints(settings.constraint_tolerance)
            rows, frames, frame_times = [], [], []
        kinetic_dof = _kinetic_dof(prepared_system)
        manifest["kinetic_dof"] = kinetic_dof
        manifest["kinetic_dof_rule"] = "3*npositiveMass - nnonredundantDistanceConstraints; no COM or harmonic-gauge subtraction"
        if not resume:
            context.setParameter("core_force_h", float(force_h))
        if abs(context.getParameter("core_force_h") - float(force_h)) > 1e-12:
            raise ValueError("Context did not retain requested force parameter")
        if abs(context.getParameter("core_gauge_k") - float(settings.gauge_k)) > 1e-12:
            raise ValueError("Context gauge parameter does not match segment settings")
        core = list(map(int, mapping["core_indices"]))
        reference_nm = np.asarray(mapping["reference_nm"], dtype=float)
        q = np.asarray(mapping["q"], dtype=float).reshape(-1)
        last_recorded_time = frame_times[-1] if frame_times else None
        runtime_reference_xyz = prepared_start_xyz if prepared_start_xyz is not None else original_xyz
        current_state = context.getState(getEnergy=True)
        current_step = _state_step(current_state, settings.timestep_fs)
        current_time_ps = current_state.getTime().value_in_unit(unit.picosecond)
        if resume:
            phase_start_step = int(manifest["phase_start_step"])
            target_global_step = int(manifest["target_global_step"])
        else:
            phase_start_step = current_step
            target_global_step = phase_start_step + settings.steps
            manifest["phase_start_step"] = phase_start_step
            manifest["phase_start_time_ps"] = float(current_time_ps)
            manifest["requested_phase_steps"] = settings.steps
            manifest["target_global_step"] = target_global_step
            manifest["phase_completed_steps"] = 0
        if current_step < phase_start_step or current_step > target_global_step:
            raise ResumeMismatch("Current step is inconsistent with this segment phase counters")
        if parent_state is not None and prepared_start_xyz is None:
            runtime_reference_xyz = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        elif resume and prepared_start_xyz is None:
            runtime_reference_xyz = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        def sample_and_commit(stage: str) -> None:
            nonlocal last_recorded_time
            state = context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=False)
            diagnostic, xyz_now, time_ps = _runtime_screen(
                stage=stage,
                state=state,
                system=prepared_system,
                mapping=mapping,
                model=model,
                original_xyz=runtime_reference_xyz,
                chemical_geometry=chemical_geometry,
                synthetic_fixture=synthetic_fixture,
                settings=settings,
                gauge_meta=gauge_meta,
                body_meta=body_meta,
                kinetic_dof=kinetic_dof,
            )
            delta = xyz_now[core] - reference_nm
            if last_recorded_time is None or time_ps > last_recorded_time + 1e-12:
                rows.append(_response_row(model, replicate_id, force_h, time_ps, float(q @ delta.ravel())))
                frames.append(delta.copy())
                frame_times.append(float(time_ps))
                last_recorded_time = float(time_ps)
            manifest["last_committed_step"] = _state_step(state, settings.timestep_fs)
            manifest["last_committed_time_ps"] = float(time_ps)
            manifest["phase_completed_steps"] = manifest["last_committed_step"] - int(manifest["phase_start_step"])
            if manifest["phase_completed_steps"] < 0 or manifest["last_committed_step"] > int(manifest["target_global_step"]):
                raise ResumeMismatch("Committed phase counters are inconsistent with target step")
            manifest["runtime_diagnostics"].append({**diagnostic, "step": manifest["last_committed_step"], "time_ps": float(time_ps)})
            _write_generation(
                output_dir=output_dir,
                manifest=manifest,
                context=context,
                rows=rows,
                frames=frames,
                frame_times=frame_times,
                reference_nm=reference_nm,
                q=q,
                core_indices=core,
                force_h=force_h,
            )

        if not rows:
            sample_and_commit("initial")
        current_step = _state_step(context.getState(getEnergy=True), settings.timestep_fs)
        if current_step > target_global_step:
            raise ResumeMismatch("Checkpoint is beyond requested segment target step")
        while current_step < target_global_step:
            if time.monotonic() >= deadline:
                manifest["status"] = "budget_limited"
                manifest["segment_complete"] = False
                manifest["response_converged"] = False
                manifest["production_ready"] = False
                manifest["elapsed_wall_seconds"] = time.monotonic() - start
                _atomic_json(output_dir / "segment_manifest.json", manifest)
                raise _BudgetStop("Wall budget reached at committed block boundary")
            phase_done = current_step - phase_start_step
            amount = min(
                settings.max_step_batch,
                settings.report_interval_steps - phase_done % settings.report_interval_steps,
                target_global_step - current_step,
            )
            integrator.step(int(amount))
            current_step += int(amount)
            phase_done = current_step - phase_start_step
            if phase_done % settings.report_interval_steps == 0:
                sample_and_commit("report" if current_step < target_global_step else "final")
        manifest["status"] = "segment_complete"
        manifest["segment_complete"] = True
        manifest["phase_completed_steps"] = int(manifest["last_committed_step"]) - int(manifest["phase_start_step"])
        if manifest["phase_completed_steps"] != settings.steps or int(manifest["last_committed_step"]) != target_global_step:
            raise ResumeMismatch("Segment did not reach the requested new phase steps")
        manifest["response_converged"] = False
        manifest["production_ready"] = False
        manifest["elapsed_wall_seconds"] = time.monotonic() - start
        _atomic_json(output_dir / "segment_manifest.json", manifest)
        return manifest
    except _BudgetStop:
        return _read_json(output_dir / "segment_manifest.json")


def run_from_qualified(
    *,
    prmtop: Path,
    inpcrd: Path,
    mapping_path: Path,
    config_path: Path,
    qualification_path: Path,
    output_dir: Path,
    phase: str,
    model: str,
    replicate_id: str,
    force_h: float,
    platform_name: str,
    precision: str,
    device_index: str | None,
    resume: bool,
    overrides: Mapping[str, Any],
    locked_force_plan: Path | None,
    parent_state: Path | None,
    equilibration_certificate: Path | None,
    measurement_identity: Path | None = None,
    measurement_identity_sha256: str | None = None,
) -> dict[str, Any]:
    config = _read_json(config_path)
    settings = resolve_segment_settings(config, overrides)
    loaded = dict(pilot.load_qualified_inputs(prmtop, inpcrd, mapping_path, config_path, qualification_path, model=model))
    loader_provenance = loaded.pop("provenance", {})
    loaded["settings"] = settings
    amber = app.AmberPrmtopFile(str(prmtop), periodicBoxVectors=app.AmberInpcrdFile(str(inpcrd)).boxVectors)
    input_hashes = {
        "prmtop": _sha256(prmtop),
        "inpcrd": _sha256(inpcrd),
        "mapping": _sha256(mapping_path),
        "qualification": _sha256(qualification_path),
        "config": _sha256(config_path),
        "source_provenance_sha256": _canonical_hash(loader_provenance),
    }
    return run_segment(
        **loaded,
        output_dir=output_dir,
        phase=phase,
        model=model,
        replicate_id=replicate_id,
        force_h=force_h,
        platform_name=platform_name,
        precision=precision,
        device_index=device_index,
        topology=amber.topology,
        input_hashes=input_hashes,
        resume=resume,
        locked_force_plan=locked_force_plan,
        parent_state=parent_state,
        equilibration_certificate=equilibration_certificate,
        measurement_identity=measurement_identity,
        measurement_identity_sha256=measurement_identity_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prmtop", "inpcrd", "mapping", "qualification"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/atomistic_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--model", choices=pilot.MODELS, required=True)
    parser.add_argument("--replicate-id", required=True)
    parser.add_argument("--force-kj-mol-nm", type=float, required=True)
    parser.add_argument("--platform", choices=(*pilot.GPU_PLATFORMS, "Reference", "CPU"), default="OpenCL")
    parser.add_argument("--precision", choices=("double", "mixed"), default="double")
    parser.add_argument("--device-index")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-pme-stream", action="store_true", default=None)
    parser.add_argument("--locked-force-plan", type=Path)
    parser.add_argument("--parent-state", type=Path)
    parser.add_argument("--equilibration-certificate", type=Path)
    parser.add_argument("--measurement-identity", type=Path)
    parser.add_argument("--measurement-identity-sha256")
    parser.add_argument("--offline", action="store_true", help="Explicit local-only intent; this runner has no network operations")
    for name, kind in (
        ("steps", int),
        ("master-seed", int),
        ("velocity-seed", int),
        ("thermostat-seed", int),
        ("gauge-k", float),
        ("report-interval-steps", int),
        ("max-step-batch", int),
        ("max-wall-seconds", float),
    ):
        parser.add_argument("--" + name, type=kind)
    args = parser.parse_args(argv)
    overrides = {
        "steps": args.steps,
        "master_seed": args.master_seed,
        "velocity_seed": args.velocity_seed,
        "thermostat_seed": args.thermostat_seed,
        "gauge_k": args.gauge_k,
        "report_interval_steps": args.report_interval_steps,
        "max_step_batch": args.max_step_batch,
        "max_wall_seconds": args.max_wall_seconds,
        "disable_pme_stream": args.disable_pme_stream,
    }
    try:
        result = run_from_qualified(
            prmtop=args.prmtop,
            inpcrd=args.inpcrd,
            mapping_path=args.mapping,
            config_path=args.config,
            qualification_path=args.qualification,
            output_dir=args.output_dir,
            phase=args.phase,
            model=args.model,
            replicate_id=args.replicate_id,
            force_h=args.force_kj_mol_nm,
            platform_name=args.platform,
            precision=args.precision,
            device_index=args.device_index,
            resume=args.resume,
            overrides=overrides,
            locked_force_plan=args.locked_force_plan,
            parent_state=args.parent_state,
            equilibration_certificate=args.equilibration_certificate,
            measurement_identity=args.measurement_identity,
            measurement_identity_sha256=args.measurement_identity_sha256,
        )
    except Exception as exc:
        print(json.dumps({"status": "rejected_or_failed", "reason": f"{type(exc).__name__}: {exc}"}))
        return 2
    print(json.dumps({
        "status": result["status"],
        "segment_complete": result["segment_complete"],
        "response_converged": result["response_converged"],
        "last_committed_step": result["last_committed_step"],
    }))
    return 0 if result["segment_complete"] else (3 if result["status"] == "budget_limited" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
