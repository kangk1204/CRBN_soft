#!/usr/bin/env python3
"""Qualified-input, zero-force NVT engine pilot; never a response-convergence claim.

The CLI requires a hash-bound qualification JSON and exactly 269 core C-alpha
atoms. Isolated input must already lack DDB1. All positions/observables use the
fixed supplied reference frame, with no trajectory fitting before projection.
Wall limits are cooperative: setup or one OpenMM kernel call can overrun them.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import openmm as mm
from openmm import app, unit

try:
    from . import atomistic_boundary as boundary
except ImportError:
    import atomistic_boundary as boundary


ROOT = Path(__file__).resolve().parents[1]
GAS_CONSTANT = 0.00831446261815324  # kJ mol^-1 K^-1
MODELS = ("flexible", "fixed", "rigid", "isolated")
GPU_PLATFORMS = ("OpenCL", "CUDA", "HIP")
CHEMICAL_REVIEW_PASS = "technical_chemistry_preparation_pass"
MIN_SIGNED_VOLUME_NM3 = 1e-4
PEPTIDE_BOUNDS_NM = (.11, .17)
PROTEIN_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "CY1", "CYM", "CYX", "HID", "HIE", "HIP", "ASH", "GLH", "LYN",
})


@dataclass(frozen=True)
class ChemicalGeometry:
    """Immutable complete topology selections, independent of mapping lists."""
    particle_count: int
    alpha_centers: tuple[tuple[int, int, int, int], ...]
    alpha_labels: tuple[str, ...]
    alpha_hydrogen_centers: tuple[tuple[int, int, int, int], ...]
    beta_centers: tuple[tuple[int, int, int, int], ...]
    beta_labels: tuple[str, ...]
    peptide_pairs: tuple[tuple[int, int], ...]
    peptide_labels: tuple[str, ...]


def derive_chemical_geometry(topology):
    """Derive every protein center and peptide bond from the full Amber topology.

    Atom names, elements and required bonds are checked before making immutable
    index tuples. Mapping fields cannot select away a failing residue or bond.
    """
    atoms = list(topology.atoms())
    bonds = {tuple(sorted((a.index, b.index))) for a, b in topology.bonds()}
    groups = {}
    for residue in topology.residues():
        names = {}
        for atom in residue.atoms():
            if atom.name in names:
                raise ValueError(f"Duplicate atom name in chemical residue {residue.index}:{residue.name}")
            names[atom.name] = atom
        if residue.name not in PROTEIN_RESIDUES and "CA" in names and names["CA"].element == app.element.carbon:
            raise ValueError(f"Unhandled C-alpha residue: {residue.index}:{residue.name}")
        groups[residue.index] = names

    def label(residue):
        return f"chain{residue.chain.index}:{residue.chain.id}/res{residue.index}:{residue.id}:{residue.name}"

    def required(residue, names, center):
        group = groups[residue.index]
        for name in names:
            expected = {"N": app.element.nitrogen, "OG1": app.element.oxygen,
                        "HA": app.element.hydrogen}.get(name, app.element.carbon)
            if name not in group or group[name].element != expected:
                raise ValueError(f"Missing or wrong-element {name} in chemical residue {label(residue)}")
        for name in names:
            if name != center and tuple(sorted((group[center].index, group[name].index))) not in bonds:
                raise ValueError(f"Missing chemical bond {center}-{name} in {label(residue)}")
        return tuple(group[name].index for name in names)

    alpha, alpha_labels, alpha_hydrogens, beta, beta_labels = [], [], [], [], []
    for residue in topology.residues():
        if residue.name not in PROTEIN_RESIDUES:
            continue
        if residue.name == "GLY":
            required(residue, ("N", "CA", "C"), "CA")
            if "CB" in groups[residue.index]:
                raise ValueError(f"GLY unexpectedly has a C-beta: {label(residue)}")
            continue
        alpha.append(required(residue, ("N", "CA", "C", "CB"), "CA"))
        alpha_labels.append(label(residue))
        alpha_hydrogens.append(required(residue, ("N", "CA", "C", "HA"), "CA"))
        if residue.name in {"ILE", "THR"}:
            x = "CG1" if residue.name == "ILE" else "OG1"
            beta.append(required(residue, ("CA", "CB", x, "CG2"), "CB"))
            beta_labels.append(label(residue))
    if not alpha:
        raise ValueError("Chemical pilot topology has no complete protein C-alpha centers")

    peptide = {}
    for a, b in topology.bonds():
        if a.name == "N" and b.name == "C":
            a, b = b, a
        if a.name != "C" or b.name != "N" or a.residue == b.residue:
            continue
        if a.residue.name not in PROTEIN_RESIDUES | {"ACE"} or b.residue.name not in PROTEIN_RESIDUES | {"NME"}:
            continue
        if a.element != app.element.carbon or b.element != app.element.nitrogen:
            raise ValueError("Peptide C-N bond has incorrect elements")
        if a.residue.chain != b.residue.chain:
            raise ValueError("Cross-chain peptide C-N bond is unsupported")
        peptide[(a.index, b.index)] = f"{label(a.residue)}->{label(b.residue)}"
    # A missing bond between consecutive protein/cap residues must not make its
    # geometric check disappear. ACE/NME delimit separate capped constructs.
    for chain in topology.chains():
        residues = list(chain.residues())
        for left, right in zip(residues, residues[1:]):
            if left.name not in PROTEIN_RESIDUES | {"ACE"} or right.name not in PROTEIN_RESIDUES | {"NME"}:
                continue
            c, n = groups[left.index].get("C"), groups[right.index].get("N")
            if c is None or n is None or (c.index, n.index) not in peptide:
                raise ValueError(f"Missing consecutive peptide bond: {label(left)}->{label(right)}")
    pairs = tuple(sorted(peptide))
    return ChemicalGeometry(len(atoms), tuple(alpha), tuple(alpha_labels), tuple(alpha_hydrogens), tuple(beta), tuple(beta_labels),
                            pairs, tuple(peptide[pair] for pair in pairs))


def chemical_geometry_screen(xyz, geometry):
    """Use direct bonded differences in the engine's unwrapped coordinates."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < geometry.particle_count or not np.isfinite(xyz).all():
        raise ValueError("Invalid coordinates for complete chemical geometry screen")
    result = {"pass": True, "failures": []}
    for kind, centers, labels, sign in (("alpha", geometry.alpha_centers, geometry.alpha_labels, 1),
                                        ("alpha_hydrogen", geometry.alpha_hydrogen_centers, geometry.alpha_labels, -1),
                                        ("beta", geometry.beta_centers, geometry.beta_labels, 1)):
        indices = np.asarray(centers, dtype=int).reshape(-1, 4)
        points = xyz[indices]
        values = np.einsum("ij,ij->i", points[:, 0]-points[:, 1],
                           np.cross(points[:, 2]-points[:, 1], points[:, 3]-points[:, 1]))
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite signed volume in chemical geometry screen")
        failed = sign*values <= MIN_SIGNED_VOLUME_NM3
        result[f"{kind}_center_count"] = len(values)
        result[f"{kind}_minimum_signed_volume_nm3"] = float(values.min()) if len(values) else None
        result[f"{kind}_maximum_signed_volume_nm3"] = float(values.max()) if len(values) else None
        result[f"{kind}_failure_count"] = int(failed.sum())
        result["failures"].extend({"kind": kind, "residue": labels[i], "signed_volume_nm3": float(values[i])}
                                  for i in np.flatnonzero(failed))
    pairs = np.asarray(geometry.peptide_pairs, dtype=int).reshape(-1, 2)
    lengths = np.linalg.norm(xyz[pairs[:, 0]]-xyz[pairs[:, 1]], axis=1)
    if not np.isfinite(lengths).all():
        raise FloatingPointError("Nonfinite peptide length in chemical geometry screen")
    failed = (lengths < PEPTIDE_BOUNDS_NM[0]) | (lengths > PEPTIDE_BOUNDS_NM[1])
    result.update(peptide_bond_count=len(lengths), peptide_minimum_distance_nm=float(lengths.min()) if len(lengths) else None,
                  peptide_maximum_distance_nm=float(lengths.max()) if len(lengths) else None,
                  peptide_failure_count=int(failed.sum()))
    result["failures"].extend({"kind": "peptide", "bond": geometry.peptide_labels[i], "distance_nm": float(lengths[i])}
                              for i in np.flatnonzero(failed))
    result["pass"] = not result["failures"]
    return result


@dataclass(frozen=True)
class Settings:
    steps: int = 20000
    timestep_fs: float = 1.
    temperature_K: float = 300.
    friction_per_ps: float = 1.
    gauge_k: float = 1000.
    seed: int = 20260907
    max_wall_seconds: float = 300.
    report_interval_steps: int = 100
    benchmark_steps: int = 1000
    max_step_batch: int = 100
    minimization_max_iterations: int = 500
    minimization_tolerance: float = 10.
    constraint_tolerance: float = 1e-6
    nonbonded_cutoff_nm: float = 1.


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def resolve_settings(config, overrides=None):
    if config.get("hydrogen_mass_repartitioning", False) is not False:
        raise ValueError("This pilot does not permit hydrogen mass repartitioning")
    values = asdict(Settings())
    values.update({"temperature_K": config.get("temperature_K", 300.),
                   "timestep_fs": config.get("timestep_fs_initial", 1.),
                   "seed": config.get("seed", 20260907)})
    pilot = config.get("technical_pilot", {})
    if not isinstance(pilot, dict):
        raise ValueError("technical_pilot must be an object")
    unknown = set(pilot)-set(values)
    if unknown:
        raise ValueError(f"Unknown technical_pilot setting(s): {sorted(unknown)}")
    values.update(pilot)
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    integer_keys = ("steps", "seed", "report_interval_steps", "benchmark_steps",
                    "max_step_batch", "minimization_max_iterations")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
        if key in integer_keys and (int(value) != value or value > 2**31-1):
            raise ValueError(f"{key} must be a positive 32-bit integer")
    if values["timestep_fs"] != 1. or values["friction_per_ps"] != 1.:
        raise ValueError("This technical protocol requires 1 fs and friction 1/ps")
    if values["max_step_batch"] > 100:
        raise ValueError("max_step_batch cannot exceed 100 for cooperative wall checks")
    if not 0 < values["constraint_tolerance"] <= 1e-5:
        raise ValueError("constraint_tolerance must lie in (0, 1e-5]")
    for key in integer_keys:
        values[key] = int(values[key])
    return Settings(**values)


def qualify_inputs(prmtop, inpcrd, mapping_path, qualification_path):
    """Fail before Amber parsing/Context construction if a declared gate fails."""
    qualification = _read_json(qualification_path)
    if qualification.get("schema_version") != "1.0" or qualification.get("status") != "pass":
        raise ValueError("Input qualification must have schema_version 1.0 and status pass")
    if any(qualification.get("gates", {}).get(key) != "pass" for key in ("metal", "mapping", "geometry")):
        raise ValueError("Input qualification requires metal, mapping, and geometry gates to pass")
    review = qualification.get("chemical_review")
    if not isinstance(review, dict) or review.get("status") != CHEMICAL_REVIEW_PASS:
        raise ValueError(f"chemical_review.status must be {CHEMICAL_REVIEW_PASS}")
    sources = {"prmtop": Path(prmtop), "inpcrd": Path(inpcrd), "mapping": Path(mapping_path)}
    hashes = {key: _sha256(path) for key, path in sources.items()}
    for key, digest in hashes.items():
        if qualification.get("input_sha256", {}).get(key) != digest:
            raise ValueError(f"Qualified input hash mismatch: {key}")
    return qualification, hashes


def _ids(values, count, label, empty=False):
    if not isinstance(values, list) or (not values and not empty):
        raise ValueError(f"{label} must be a {'possibly empty ' if empty else 'nonempty '}list")
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in values):
        raise ValueError(f"Invalid zero-based particle index in {label}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate indices in {label}")
    return values


def validate_mapping(mapping, topology, xyz, model, expected_core_count=269):
    if mapping.get("schema_version") != "1.0":
        raise ValueError("Mapping schema_version must be 1.0")
    count = topology.getNumAtoms()
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (count, 3) or not np.isfinite(xyz).all():
        raise ValueError("Coordinates do not match the finite original topology")
    core = _ids(mapping.get("core_indices"), count, "core_indices")
    if len(core) != expected_core_count:
        raise ValueError(f"Expected exactly {expected_core_count} core atoms")
    ref = np.asarray(mapping.get("reference_nm"), dtype=float)
    q = np.asarray(mapping.get("q"), dtype=float)
    if ref.shape != (len(core), 3) or not np.isfinite(ref).all():
        raise ValueError("reference_nm must contain ordered finite core coordinates in nm")
    if q.shape != (3*len(core),) or not np.isfinite(q).all():
        raise ValueError("q must be a flat ambient vector with 3*core_count coefficients")
    B = boundary.dm.rigid_basis(ref)
    if abs(np.linalg.norm(q)-1) > 1e-10 or np.linalg.norm(B.T@q) > 1e-10:
        raise ValueError("q must already be unit norm and internal in the reference frame")
    expected_initial = np.asarray(mapping.get("initial_core_nm", ref), dtype=float)
    if expected_initial.shape != ref.shape or not np.isfinite(expected_initial).all():
        raise ValueError("initial_core_nm must contain ordered finite core coordinates in nm")
    if not np.allclose(xyz[core], expected_initial, rtol=0., atol=1e-6):
        target = "initial_core_nm" if "initial_core_nm" in mapping else "reference frame"
        raise ValueError(f"Initial core does not match {target} within 1e-6 nm; no implicit alignment")
    body = _ids(mapping.get("ddb1_atom_indices"), count, "ddb1_atom_indices", empty=True)
    if set(core).intersection(body):
        raise ValueError("Core and DDB1 indices overlap")
    if model == "isolated" and body:
        raise ValueError("Isolated requires a separately qualified DDB1-removed topology")
    if model != "isolated" and not body:
        raise ValueError("Complex models require all DDB1 atom indices including hydrogen")
    atoms = list(topology.atoms())
    for index in core:
        if atoms[index].name != "CA" or atoms[index].element != app.element.carbon:
            raise ValueError("Every measured core atom must be a protein C-alpha")
    selected = set(body)
    for residue in topology.residues():
        residue_ids = {a.index for a in residue.atoms()}
        if selected.intersection(residue_ids) and not residue_ids.issubset(selected):
            raise ValueError("DDB1 selection must include every atom of each selected residue, including H")
    ca = mapping.get("protein_ca_indices", [a.index for a in atoms if a.name == "CA" and a.element == app.element.carbon])
    _ids(ca, count, "protein_ca_indices")
    if any(atoms[i].name != "CA" or atoms[i].element != app.element.carbon for i in ca):
        raise ValueError("protein_ca_indices contains an atom that is not a C-alpha")
    if ("zn_atom_index" in mapping) != ("zn_sg_indices" in mapping):
        raise ValueError("Supply both zn_atom_index and zn_sg_indices")
    if "zn_atom_index" in mapping:
        zinc = _ids([mapping["zn_atom_index"]], count, "zn_atom_index")[0]
        sulfur = _ids(mapping["zn_sg_indices"], count, "zn_sg_indices")
        if len(sulfur) != 4 or zinc in sulfur:
            raise ValueError("The Zn screen requires four distinct coordinating sulfurs")
        if atoms[zinc].element != app.element.zinc or any(atoms[i].element != app.element.sulfur for i in sulfur):
            raise ValueError("Zn/SG mapping does not match topology elements")
        bounds = np.asarray(mapping.get("zn_sg_distance_bounds_nm", [.18, .30]), dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("Invalid Zn-SG distance screen bounds in nm")
    return {**mapping, "protein_ca_indices": ca}


def _prepare_system(system, xyz, mapping, model, settings):
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    for index in reversed(range(system.getNumForces())):
        force = system.getForce(index)
        if type(force) is mm.CMMotionRemover:
            system.removeForce(index)
        elif "Barostat" in type(force).__name__:
            raise ValueError("NVT pilot input must not contain a barostat")
    gauge = boundary.add_core_gauge_and_probe(system, mapping["core_indices"], mapping["reference_nm"],
                                               mapping["q"], settings.gauge_k, 0.)
    system = gauge.system
    body_meta = {"kind": model, "body_indices": mapping["ddb1_atom_indices"]}
    if model == "fixed":
        body = boundary.make_fixed_body(system, mapping["ddb1_atom_indices"])
        system, body_meta = body.system, body.metadata
    elif model == "rigid":
        body = boundary.make_rigid_body(system, xyz, mapping["ddb1_atom_indices"])
        system, xyz, body_meta = body.system, body.positions_nm, body.metadata
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())])
    constraints = [system.getConstraintParameters(i) for i in range(system.getNumConstraints())]
    pairs = [tuple(sorted((i, j))) for i, j, _ in constraints]
    if len(set(pairs)) != len(pairs) or any(masses[i] <= 0 or masses[j] <= 0 for i, j in pairs):
        raise ValueError("Temperature counting requires distinct constraints on massive particles")
    dof = int(3*np.count_nonzero(masses > 0)-len(constraints))
    if dof <= 0:
        raise ValueError("No positive kinetic degrees of freedom")
    return system, np.asarray(xyz), gauge.metadata, body_meta, dof


def _minimum_image(vectors, box):
    return (vectors@np.linalg.inv(box)-np.rint(vectors@np.linalg.inv(box)))@box


def _fit_rmsd(x, reference):
    a, b = x-x.mean(0), reference-reference.mean(0)
    u, _, vt = np.linalg.svd(a.T@b)
    correction = np.diag([1., 1., np.linalg.det(u@vt)])
    return float(np.sqrt(np.mean(np.sum((a@u@correction@vt-b)**2, axis=1))))


class _BudgetStop(RuntimeError):
    pass


class _DeadlineReporter(mm.MinimizationReporter):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline
        self.calls = 0

    def report(self, iteration, x, grad, args):
        self.calls += 1
        if time.monotonic() >= self.deadline:
            # Returning True can trigger another constrained minimization.
            # Raising aborts the Python operation instead of requesting a retry.
            raise _BudgetStop("Wall budget reached during minimization")
        return False


def platform_options(platform_name, precision="double", device_index=None, *, disable_pme_stream=False):
    """Explicit opt-in mixed precision; never change a global platform default."""
    if precision not in ("double", "mixed"):
        raise ValueError("Precision must be double or mixed")
    if precision == "mixed" and platform_name not in GPU_PLATFORMS:
        raise ValueError("Mixed precision requires an OpenCL, CUDA or HIP GPU platform")
    platform = mm.Platform.getPlatformByName(platform_name)
    names = set(platform.getPropertyNames())
    if platform_name in GPU_PLATFORMS and "Precision" not in names:
        raise ValueError("Selected GPU platform does not expose a Precision property")
    properties = {"Precision": precision} if "Precision" in names else {}
    if device_index is not None:
        if "DeviceIndex" not in names:
            raise ValueError("Selected platform has no DeviceIndex property")
        properties["DeviceIndex"] = str(device_index)
    if disable_pme_stream:
        if platform_name not in GPU_PLATFORMS or "DisablePmeStream" not in names:
            raise ValueError("Selected GPU platform does not expose DisablePmeStream")
        properties["DisablePmeStream"] = "true"
    return platform, properties


def precision_record(platform, context, requested, *, disable_pme_stream=False):
    if "Precision" in platform.getPropertyNames():
        effective = platform.getPropertyValue(context, "Precision")
        if effective != requested:
            raise ValueError(f"Requested precision {requested} but Context reports {effective}")
        source = "Context Platform.getPropertyValue(Precision)"
    else:
        effective, source = "unreported_platform_default", "platform exposes no Precision property; no precision override applied"
    pme_value = (platform.getPropertyValue(context, "DisablePmeStream")
                 if "DisablePmeStream" in platform.getPropertyNames() else None)
    if disable_pme_stream and pme_value != "true":
        raise ValueError(f"Requested DisablePmeStream=true but Context reports {pme_value}")
    return {"requested": requested, "effective": effective, "source": source,
            "pme_stream": {"disable_requested": bool(disable_pme_stream),
                           "override_applied": bool(disable_pme_stream),
                           "effective_disabled": None if pme_value is None else pme_value == "true",
                           "source": "Context Platform.getPropertyValue(DisablePmeStream)" if pme_value is not None else "property unavailable"},
            "default_changed": False, "mixed_requires_independent_precision_validation": requested == "mixed"}


def _run_engine(system, xyz, mapping, settings, output_dir, *, model, platform_name,
                qualification, provenance, device_index=None, started_at=None,
                chemical_geometry=None, synthetic_fixture=False, precision="double", disable_pme_stream=False):
    """Execute an already qualified System; directly exercised by tiny engine fixtures."""
    start = time.monotonic() if started_at is None else started_at
    if (not synthetic_fixture and platform_name in ("OpenCL", "CUDA")
            and (precision != "double" or not disable_pme_stream)):
        raise ValueError("Real GPU technical pilots require --precision double and --disable-pme-stream; use the separate precision benchmark for diagnostic comparisons")
    if chemical_geometry is None and not synthetic_fixture:
        raise ValueError("Full topology-derived chemical geometry is required; toy calls must explicitly declare synthetic_fixture=True")
    if not synthetic_fixture and qualification.get("chemical_review", {}).get("status") != CHEMICAL_REVIEW_PASS:
        raise ValueError(f"chemical_review.status must be {CHEMICAL_REVIEW_PASS}")
    if chemical_geometry is not None and not isinstance(chemical_geometry, ChemicalGeometry):
        raise ValueError("Chemical geometry must be an immutable topology-derived ChemicalGeometry")
    platform, properties = platform_options(platform_name, precision, device_index, disable_pme_stream=disable_pme_stream)
    geometry_payload = asdict(chemical_geometry) if chemical_geometry is not None else None
    geometry_sha = hashlib.sha256(json.dumps(geometry_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if geometry_payload is not None else None
    deadline = start+settings.max_wall_seconds
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty; previous evidence is not overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    original_xyz = np.asarray(xyz, dtype=float).copy()
    original_count = len(original_xyz)
    if chemical_geometry is not None and chemical_geometry.particle_count != original_count:
        raise ValueError("Chemical geometry particle count does not match the original System coordinates")
    system, xyz, gauge_meta, body_meta, dof = _prepare_system(system, original_xyz, mapping, model, settings)
    core = mapping["core_indices"]
    ca = mapping["protein_ca_indices"]
    ref, q, B = (np.asarray(mapping["reference_nm"]), np.asarray(mapping["q"]), np.asarray(gauge_meta["rigid_basis"]))
    centered = ref-ref.mean(0)
    rotation_columns = np.column_stack([np.cross(axis, centered).ravel() for axis in np.eye(3)])
    constraint_data = [system.getConstraintParameters(i) for i in range(system.getNumConstraints())]
    constraint_i = np.array([i for i, _, _ in constraint_data], dtype=int)
    constraint_j = np.array([j for _, j, _ in constraint_data], dtype=int)
    constraint_lengths = np.array([d.value_in_unit(unit.nanometer) for _, _, d in constraint_data])
    summary = {"status": "started", "technical_accepted": False, "response_converged": False,
               "production_ready": False, "role": "zero_force_technical_engine_pilot_only",
               "model": model, "settings": asdict(settings), "force_h_kj_mol_nm": 0.,
               "precision": {"requested": precision, "effective": None, "source": "Context not yet created", "default_changed": False},
               "chemical_review": qualification["chemical_review"], "input_provenance": provenance,
               "synthetic_fixture": bool(synthetic_fixture),
               "chemical_geometry_scope": "explicit_synthetic_fixture_without_protein_checks" if chemical_geometry is None else "all_protein_CA_HA_CB_centers_and_peptide_bonds_from_full_topology",
               "chemical_geometry_definition": geometry_payload, "chemical_geometry_sha256": geometry_sha,
               "chemical_geometry_thresholds": {"minimum_signed_volume_nm3_exclusive": MIN_SIGNED_VOLUME_NM3,
                                                 "maximum_HA_signed_volume_nm3_exclusive": -MIN_SIGNED_VOLUME_NM3,
                                                 "peptide_distance_bounds_nm_inclusive": list(PEPTIDE_BOUNDS_NM)},
               "chemical_geometry_failures": [],
               "gauge": gauge_meta, "body": body_meta, "original_particle_count": original_count,
               "engine_particle_count": system.getNumParticles(), "constraint_count": system.getNumConstraints(),
               "positive_mass_particle_count": (dof+system.getNumConstraints())//3,
               "zero_mass_particle_count": system.getNumParticles()-(dof+system.getNumConstraints())//3,
               "kinetic_dof": dof, "kinetic_dof_rule": "3*npositiveMass - nnonredundantDistanceConstraints; no extra COM or harmonic-gauge subtraction",
               "initial_coordinate_rule": "Check initial_core_nm if supplied, otherwise reference_nm; frozen reference_nm/q/gauge are never refitted or changed",
               "initial_vs_reference_rms_displacement_nm": float(np.sqrt(np.mean(np.sum((original_xyz[core]-ref)**2, axis=1)))),
               "initial_vs_reference_max_displacement_nm": float(np.max(np.linalg.norm(original_xyz[core]-ref, axis=1))),
               "wall_budget_rule": "Cooperative at setup/minimizer callback/at most 100 steps; a single engine call can overrun. Cumulative multi-job budget is tracked by coordinator.",
               "openmm_version": mm.__version__, "completed_steps": 0}
    frames, times, rows = [], [], []
    last_xyz, last_box = xyz.copy(), None
    context = None
    benchmark = None
    with (output_dir/"diagnostics.csv").open("w", newline="") as handle:
        writer = None

        def sample(stage, store_frame):
            nonlocal last_xyz, last_box, writer
            state = context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=False)
            last_xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            last_box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            if not np.isfinite(last_xyz).all() or not np.isfinite([potential, kinetic]).all():
                raise FloatingPointError("Nonfinite energy or position")
            delta = last_xyz[core]-ref
            measured_time_ps = state.getTime().value_in_unit(unit.picosecond)
            summary["completed_steps"] = int(round(measured_time_ps/(settings.timestep_fs/1000)))
            g = B.T@delta.ravel()
            translation = delta.mean(0)
            omega = np.linalg.solve(rotation_columns.T@rotation_columns, rotation_columns.T@delta.ravel())
            relative_constraint_error = 0.
            if len(constraint_data):
                # OpenMM constraints use unwrapped coordinate differences.
                # A rigid-anchor edge can exceed half a periodic box dimension.
                # OpenMM 8.5.0 integrationUtilities.cc:535-539 and
                # ReferenceCCMAAlgorithm.cpp:225,241 use direct differences.
                distances = np.linalg.norm(last_xyz[constraint_i]-last_xyz[constraint_j], axis=1)
                relative_constraint_error = float(np.max(np.abs(distances-constraint_lengths)/constraint_lengths))
            row = {"stage": stage, "step": summary["completed_steps"],
                   "time_ps": measured_time_ps,
                   "elapsed_wall_seconds": time.monotonic()-start,
                   "potential_kj_mol": potential, "kinetic_kj_mol": kinetic,
                   "kinetic_temperature_K": 2*kinetic/(dof*GAS_CONSTANT),
                   "kinetic_dof": dof, "closure_nm": float(q@delta.ravel()),
                   "core_rms_displacement_nm": float(np.sqrt(np.mean(np.sum(delta*delta, axis=1)))),
                   "gauge_norm_nm": float(np.linalg.norm(g)), "gauge_energy_kj_mol": float(.5*settings.gauge_k*(g@g)),
                   "translation_norm_nm": float(np.linalg.norm(translation)),
                   "infinitesimal_rotation_norm_rad": float(np.linalg.norm(omega)),
                   "protein_ca_unfitted_rmsd_nm": float(np.sqrt(np.mean(np.sum((last_xyz[ca]-original_xyz[ca])**2, axis=1)))),
                   "protein_ca_fitted_rmsd_nm": _fit_rmsd(last_xyz[ca], original_xyz[ca]),
                   "max_relative_constraint_error": relative_constraint_error}
            row.update({f"gauge_coordinate_{i}_nm": float(value) for i, value in enumerate(g)})
            for axis, t, w in zip("xyz", translation, omega):
                row[f"translation_{axis}_nm"], row[f"infinitesimal_rotation_{axis}_rad"] = float(t), float(w)
            if model == "fixed":
                row["fixed_body_max_displacement_nm"] = float(np.max(np.linalg.norm(last_xyz[mapping["ddb1_atom_indices"]]-original_xyz[mapping["ddb1_atom_indices"]], axis=1)))
            if "zn_atom_index" in mapping:
                distances = np.linalg.norm(_minimum_image(last_xyz[mapping["zn_sg_indices"]]-last_xyz[mapping["zn_atom_index"]], last_box), axis=1)
                row.update({f"zn_sg_{i+1}_distance_nm": float(d) for i, d in enumerate(distances)})
                bounds = mapping.get("zn_sg_distance_bounds_nm", [.18, .30])
                row["zn_sg_screen_pass"] = bool(np.all((distances >= bounds[0]) & (distances <= bounds[1])))
            chemistry = None
            if chemical_geometry is not None:
                chemistry = chemical_geometry_screen(last_xyz, chemical_geometry)
                row.update({f"chemical_{key}": value for key, value in chemistry.items() if key != "failures"})
                summary["chemical_geometry_last_screen"] = {"stage": stage, "step": row["step"], **chemistry}
                if not chemistry["pass"]:
                    summary["chemical_geometry_failures"].append({"stage": stage, "step": row["step"], **chemistry})
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row); handle.flush()
            rows.append(row)
            if store_frame and (not times or row["time_ps"] != times[-1]):
                frames.append(delta.copy()); times.append(row["time_ps"])
            if chemistry is not None and not chemistry["pass"]:
                raise ValueError(f"Chemical geometry screen failed at {stage}: {chemistry['failures'][:10]}")
            if stage != "initial":
                if relative_constraint_error > max(10*settings.constraint_tolerance, 1e-5):
                    raise ValueError("Distance constraints failed the technical geometry screen")
                if row.get("fixed_body_max_displacement_nm", 0.) > 1e-10:
                    raise ValueError("Fixed DDB1 moved")
                if not row.get("zn_sg_screen_pass", True):
                    raise ValueError("Zn-SG distances failed the declared physical screen")
            return row

        try:
            if time.monotonic() >= deadline:
                raise _BudgetStop("Wall budget exhausted before Context creation")
            integrator = mm.LangevinMiddleIntegrator(settings.temperature_K, settings.friction_per_ps, settings.timestep_fs/1000)
            integrator.setRandomNumberSeed(settings.seed)
            integrator.setConstraintTolerance(settings.constraint_tolerance)
            context = mm.Context(system, integrator, platform, properties)
            summary["precision"] = precision_record(platform, context, precision, disable_pme_stream=disable_pme_stream)
            summary["platform"] = platform.getName()
            summary["platform_properties"] = {name: platform.getPropertyValue(context, name) for name in platform.getPropertyNames()}
            context.setPositions(xyz); context.computeVirtualSites()
            sample("initial", False)
            if time.monotonic() >= deadline:
                raise _BudgetStop("Wall budget exhausted during Context setup")
            reporter = _DeadlineReporter(deadline)
            mm.LocalEnergyMinimizer.minimize(context, settings.minimization_tolerance,
                                           settings.minimization_max_iterations, reporter)
            summary["minimization_report_calls"] = reporter.calls
            context.computeVirtualSites()
            context.setVelocitiesToTemperature(settings.temperature_K, settings.seed+1)
            context.applyVelocityConstraints(settings.constraint_tolerance)
            sample("post_minimization", True)
            if time.monotonic() >= deadline:
                raise _BudgetStop("Wall budget exhausted after minimization")
            md_start = time.monotonic()
            benchmark_target = min(settings.benchmark_steps, settings.steps)
            while summary["completed_steps"] < settings.steps:
                if time.monotonic() >= deadline:
                    raise _BudgetStop("Wall budget reached at integration batch boundary")
                step = summary["completed_steps"]
                until_report = settings.report_interval_steps-step % settings.report_interval_steps
                amount = min(settings.max_step_batch, until_report, settings.steps-step)
                if benchmark is None:
                    amount = min(amount, benchmark_target-step)
                integrator.step(amount)
                summary["completed_steps"] += amount
                step = summary["completed_steps"]
                if step % settings.report_interval_steps == 0 or step == settings.steps or step == benchmark_target:
                    sample("technical", True)
                if benchmark is None and step == benchmark_target:
                    elapsed = time.monotonic()-md_start
                    benchmark = {"measured_steps": step, "measured_wall_seconds": elapsed,
                                 "steps_per_second": step/elapsed,
                                 "ns_per_day": step*settings.timestep_fs/1e6/elapsed*86400,
                                 "projected_remaining_seconds": (settings.steps-step)*elapsed/step}
                    _write_json(output_dir/"benchmark.json", benchmark)
                    if benchmark["projected_remaining_seconds"] > deadline-time.monotonic():
                        raise _BudgetStop("Measured benchmark projects completion beyond remaining wall budget")
            summary.update(status="technical_completed", technical_accepted=True)
        except _BudgetStop as error:
            summary.update(status="budget_limited", technical_accepted=False, reason=str(error))
        except Exception as error:
            summary.update(status="failed", technical_accepted=False, reason=f"{type(error).__name__}: {error}")
        finally:
            if context is not None:
                try:
                    sample("final", True)
                except Exception as error:
                    summary.update(status="failed", technical_accepted=False, reason=f"Final state: {type(error).__name__}: {error}")
                del context
    summary["elapsed_wall_seconds"] = time.monotonic()-start
    summary["benchmark"] = benchmark
    summary["trajectory_frames"] = len(times)
    summary["diagnostic_rows"] = len(rows)
    summary["technical_accepted_definition"] = "Requested steps completed with finite states and constraint/fixed-body/optional Zn-SG screens plus topology-derived CA/HA/CB/peptide geometry at initial, post-minimization, report and final states; explicit synthetic fixtures may omit protein checks. No equilibrium or response convergence claim."
    summary["fit_rule"] = "Protein CA fitting is diagnostic only; core_displacement and q projection always use the unchanged supplied frame"
    np.savez_compressed(output_dir/"technical_pilot.npz",
                        last_positions_nm=last_xyz[:original_count],
                        last_anchor_positions_nm=last_xyz[original_count:],
                        last_box_vectors_nm=np.asarray(last_box) if last_box is not None else np.empty((0,3)),
                        core_displacement_nm=np.asarray(frames).reshape(-1, len(core), 3),
                        time_ps=np.asarray(times), reference_nm=ref, q_ambient=q,
                        initial_core_nm=original_xyz[core],
                        initial_vs_reference_displacement_nm=original_xyz[core]-ref,
                        core_indices=np.asarray(core), rigid_basis=B)
    summary["artifact_sha256"] = {name: _sha256(output_dir/name) for name in ("technical_pilot.npz", "diagnostics.csv")}
    _write_json(output_dir/"technical_pilot.json", summary)
    return summary


def load_qualified_inputs(prmtop, inpcrd, mapping_path, config_path, qualification_path, *,
                          model="flexible", overrides=None):
    """Shared validated loading path; no Context, minimization or dynamics."""
    if model not in MODELS:
        raise ValueError("Unknown boundary model")
    qualification, hashes = qualify_inputs(prmtop, inpcrd, mapping_path, qualification_path)
    config = _read_json(config_path)
    settings = resolve_settings(config, overrides)
    coordinates = app.AmberInpcrdFile(str(inpcrd))
    if coordinates.boxVectors is None:
        raise ValueError("Explicit-solvent PME input requires periodic box vectors")
    box = np.asarray(coordinates.boxVectors.value_in_unit(unit.nanometer))
    volume = float(np.linalg.det(box))
    heights = volume/np.linalg.norm(np.cross(box[[1,2,0]], box[[2,0,1]]), axis=1)
    if not np.isfinite(heights).all() or np.min(heights) <= 2*settings.nonbonded_cutoff_nm:
        raise ValueError("Periodic box is invalid or too small for the PME cutoff")
    amber = app.AmberPrmtopFile(str(prmtop), periodicBoxVectors=coordinates.boxVectors)
    xyz = coordinates.positions.value_in_unit(unit.nanometer)
    mapping = validate_mapping(_read_json(mapping_path), amber.topology, xyz, model)
    chemical_geometry = derive_chemical_geometry(amber.topology)
    # Reject a stale/tampered input before Context creation. Geometry is derived
    # from the qualified complete topology, never from a mapping selection.
    if hashes != {key: _sha256(path) for key, path in
                  (("prmtop", prmtop), ("inpcrd", inpcrd), ("mapping", mapping_path))}:
        raise ValueError("Qualified inputs changed during parsing")
    system = amber.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=settings.nonbonded_cutoff_nm*unit.nanometer,
                                constraints=app.HBonds, rigidWater=True, removeCMMotion=False,
                                hydrogenMass=None, ewaldErrorTolerance=5e-4)
    for atom in amber.topology.atoms():
        if atom.element == app.element.hydrogen:
            mass = system.getParticleMass(atom.index).value_in_unit(unit.dalton)
            if not .9 < mass < 1.1:
                raise ValueError("Input hydrogen mass is not natural-abundance H; no HMR/deuteration in this protocol")
    sources = {"input_sha256": hashes, "config_sha256": _sha256(config_path),
               "qualification_sha256": _sha256(qualification_path),
               "source_sha256": {name: _sha256(ROOT/"scripts"/name) for name in
                                  ("run_atomistic_technical_pilot.py", "atomistic_boundary.py", "directional_mechanics.py")},
               "cumulative_budget_gpu_hours": config.get("initial_technical_pilot_gpu_hour_cap", 2.)}
    return {"system": system, "xyz": xyz, "mapping": mapping, "settings": settings,
            "qualification": qualification, "provenance": sources, "chemical_geometry": chemical_geometry}


def run(prmtop, inpcrd, mapping_path, config_path, qualification_path, output_dir, *,
        model="flexible", platform_name="OpenCL", device_index=None, overrides=None, precision="double", disable_pme_stream=False):
    started = time.monotonic()
    loaded = load_qualified_inputs(prmtop, inpcrd, mapping_path, config_path, qualification_path,
                                   model=model, overrides=overrides)
    return _run_engine(**loaded, output_dir=output_dir, model=model, platform_name=platform_name,
                       device_index=device_index, started_at=started, precision=precision, disable_pme_stream=disable_pme_stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prmtop", "inpcrd", "mapping", "qualification"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"scripts/atomistic_config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/atomistic/technical_pilot")
    parser.add_argument("--offline", action="store_true", help="Explicit local-only intent; this runner has no network operations")
    parser.add_argument("--model", choices=MODELS, default="flexible")
    parser.add_argument("--platform", choices=(*GPU_PLATFORMS, "Reference", "CPU"), default="OpenCL")
    parser.add_argument("--device-index")
    parser.add_argument("--precision", choices=("double", "mixed"), default="double")
    parser.add_argument("--disable-pme-stream", action="store_true", help="Required for real OpenCL/CUDA technical pilots; runtime readback required")
    for name, kind in (("steps", int), ("seed", int), ("gauge-k", float), ("max-wall-seconds", float),
                       ("report-interval-steps", int), ("benchmark-steps", int), ("minimization-max-iterations", int)):
        parser.add_argument("--"+name, type=kind)
    args = parser.parse_args(argv)
    overrides = {name: getattr(args, name) for name in ("steps", "seed", "gauge_k", "max_wall_seconds",
                                                        "report_interval_steps", "benchmark_steps", "minimization_max_iterations")}
    try:
        result = run(args.prmtop, args.inpcrd, args.mapping, args.config, args.qualification, args.output_dir,
                      model=args.model, platform_name=args.platform, device_index=args.device_index, overrides=overrides,
                      precision=args.precision, disable_pme_stream=args.disable_pme_stream)
    except Exception as error:
        print(json.dumps({"status": "rejected_before_pilot", "reason": f"{type(error).__name__}: {error}"}))
        return 2
    print(json.dumps({key: result[key] for key in ("status", "technical_accepted", "response_converged", "completed_steps", "elapsed_wall_seconds")}))
    return 0 if result["technical_accepted"] else (3 if result["status"] == "budget_limited" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
