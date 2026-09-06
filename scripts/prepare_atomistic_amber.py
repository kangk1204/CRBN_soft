#!/usr/bin/env python3
"""Prepare an Amber bridge for the CRBN-DDB1 atomistic feasibility pilot.

The script rewrites only residue identities needed by the chosen ZAFF Cys4
model, writes deterministic LEaP inputs, optionally runs LEaP, and writes the
atom-index/q-vector mapping needed by the response pilot when Amber outputs are
available. It does not run MD and does not mark the system production-ready.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from verify_zaff_amber_topology import classify_leap_warnings, parameter_contract
except ImportError:  # pragma: no cover
    from scripts.verify_zaff_amber_topology import classify_leap_warnings, parameter_contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts/atomistic_config.json"
DEFAULT_INPUT_PDB = ROOT / "results/atomistic/capped_heavy.pdb"
DEFAULT_OUTPUT = ROOT / "results/atomistic/amber"
ASSEMBLIES = {"joint", "isolated"}
DEFAULT_PREP = ROOT / "data/metal_sources/ZAFF.prep"
DEFAULT_FRCMOD = ROOT / "data/metal_sources/ZAFF.frcmod"
DEFAULT_CORE_RESIDUES = ROOT / "data/crbn_residue_window.csv"
DEFAULT_ENSEMBLE = ROOT / "data/crbn_ensemble.ens.npz"
DEFAULT_DIFFVEC = ROOT / "data/pca_diffvec.npz"

CRBN_CHAIN = "B"
DDB1_CHAIN = "A"
ZN_CHAIN = "C"
ZN_RESSEQ = 501
ZN_CYS_RESIDUES = (323, 326, 391, 394)
CY1_NAME = "CY1"
ZN1_NAME = "ZN1"
PROTEIN_CAP_RESNAMES = {"ACE", "NME"}
HIS_PROTONATION_NAMES = {"HID", "HIE", "HIP"}
LEAP_LOG_ERROR_PATTERNS = (
    "Could not find bond parameter",
    "Could not find angle parameter",
    "Could not find torsion parameter",
    "Could not find vdW",
    "Unknown residue",
    "FATAL",
)


GENERIC_WARNING_LABELS = {
    "close_contact_from_overlapping_prototype_geometry": "nonbonded_close_contact",
    "long_bond_from_overlapping_prototype_geometry": "long_bond",
    "nonzero_unit_charge_expected_for_cys4_zn_site": "nonzero_unit_charge",
}


def generic_actual_warning_labels(value: object) -> object:
    if isinstance(value, dict):
        return {GENERIC_WARNING_LABELS.get(str(key), key): generic_actual_warning_labels(item) for key, item in value.items()}
    if isinstance(value, list):
        return [generic_actual_warning_labels(item) for item in value]
    if isinstance(value, str):
        for old, new in GENERIC_WARNING_LABELS.items():
            value = value.replace(old, new)
        return value
    return value


@dataclass(frozen=True)
class PdbAtom:
    record: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: str
    bfactor: str
    element: str
    charge: str
    raw: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.name)

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path | None) -> dict[str, object]:
    config: dict[str, object] = {
        "core_residue_file": str(DEFAULT_CORE_RESIDUES),
        "core_position_count": 269,
        "pilot_reference": "8CVP",
    }
    if path is None:
        if DEFAULT_CONFIG.is_file():
            path = DEFAULT_CONFIG
    if path is not None:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        for key in ("core_residue_file", "core_position_count", "pilot_reference"):
            if key in loaded:
                config[key] = loaded[key]
        config["config_file"] = str(path.resolve())
        config["ignored_config_keys"] = sorted(
            set(loaded) - {"core_residue_file", "core_position_count", "pilot_reference"}
        )
    else:
        config["config_file"] = None
        config["ignored_config_keys"] = []
    config["core_residue_file"] = str(resolve_path(str(config["core_residue_file"])).resolve())
    return config


def parse_pdb_atom(line: str) -> PdbAtom:
    if len(line) < 54:
        raise ValueError(f"Malformed PDB atom line: {line!r}")
    resseq = line[22:26].strip()
    if not resseq.lstrip("-").isdigit():
        raise ValueError(f"Noninteger PDB residue number: {line!r}")
    return PdbAtom(
        record=line[0:6].strip(),
        serial=int(line[6:11]),
        name=line[12:16].strip(),
        altloc=line[16:17],
        resname=line[17:20].strip(),
        chain=line[21:22].strip(),
        resseq=int(resseq),
        icode=line[26:27],
        x=float(line[30:38]),
        y=float(line[38:46]),
        z=float(line[46:54]),
        occupancy=line[54:60] if len(line) >= 60 else "  1.00",
        bfactor=line[60:66] if len(line) >= 66 else "  0.00",
        element=(line[76:78].strip() if len(line) >= 78 else ""),
        charge=(line[78:80] if len(line) >= 80 else "  "),
        raw=line.rstrip("\n"),
    )


def format_atom(atom: PdbAtom, serial: int) -> str:
    element = atom.element or atom.name.lstrip("0123456789")[:1].upper()
    if len(atom.name) < 4 and len(element) == 1 and not atom.name[0].isdigit():
        atom_name = f" {atom.name:<3}"
    else:
        atom_name = f"{atom.name:<4}"
    resname = atom.resname[:3]
    return (
        f"{atom.record:<6}{serial:5d} {atom_name}{atom.altloc[:1]}"
        f"{resname:>3} {atom.chain[:1]}{atom.resseq:4d}{atom.icode[:1]}   "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{atom.occupancy[:6]:>6}{atom.bfactor[:6]:>6}          "
        f"{element:>2}{atom.charge[:2]:>2}"
    )


def read_pdb_atoms(path: Path) -> list[PdbAtom]:
    atoms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atoms.append(parse_pdb_atom(line))
    if not atoms:
        raise ValueError(f"No atoms found in PDB: {path}")
    seen: set[tuple[str, int, str]] = set()
    duplicates: list[tuple[str, int, str]] = []
    altlocs: list[tuple[str, int, str, str]] = []
    nonfinite: list[tuple[str, int, str]] = []
    for atom in atoms:
        if atom.key in seen:
            duplicates.append(atom.key)
        seen.add(atom.key)
        if atom.altloc.strip():
            altlocs.append((*atom.key, atom.altloc.strip()))
        if not np.isfinite([atom.x, atom.y, atom.z]).all():
            nonfinite.append(atom.key)
    if duplicates:
        raise ValueError(f"Duplicate PDB atom keys are not allowed: {duplicates[:10]}")
    if altlocs:
        raise ValueError(f"Alternate-location PDB atoms are not allowed: {altlocs[:10]}")
    if nonfinite:
        raise ValueError(f"Nonfinite PDB coordinates are not allowed: {nonfinite[:10]}")
    return atoms


def residue_order(atoms: Sequence[PdbAtom]) -> list[tuple[str, int, str]]:
    order: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for atom in atoms:
        key = (atom.chain, atom.resseq, atom.resname)
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def normalize_residue_names(atoms: Sequence[PdbAtom]) -> tuple[list[PdbAtom], list[dict[str, object]]]:
    normalized: list[PdbAtom] = []
    changes: list[dict[str, object]] = []
    for atom in atoms:
        new_name = atom.resname
        if atom.chain == CRBN_CHAIN and atom.resseq in ZN_CYS_RESIDUES:
            if atom.resname not in {"CYS", CY1_NAME}:
                raise ValueError(
                    f"Zn ligand residue {atom.chain}{atom.resseq} is {atom.resname}, expected CYS/CY1"
                )
            new_name = CY1_NAME
        elif atom.chain == ZN_CHAIN and atom.resseq == ZN_RESSEQ:
            if atom.name.upper() != "ZN":
                raise ValueError(f"Zn residue {ZN_CHAIN}{ZN_RESSEQ} contains non-ZN atom {atom.name}")
            new_name = ZN1_NAME
        if new_name != atom.resname:
            changes.append(
                {
                    "chain": atom.chain,
                    "resseq": atom.resseq,
                    "atom": atom.name,
                    "from": atom.resname,
                    "to": new_name,
                }
            )
            atom = replace(atom, resname=new_name)
        normalized.append(atom)
    return normalized, changes


def select_assembly_atoms(atoms: Sequence[PdbAtom], assembly: str) -> list[PdbAtom]:
    if assembly not in ASSEMBLIES:
        raise ValueError(f"Unknown assembly {assembly!r}; expected one of {sorted(ASSEMBLIES)}")
    if assembly == "joint":
        return list(atoms)
    selected = [
        atom
        for atom in atoms
        if (atom.chain == CRBN_CHAIN and 63 <= atom.resseq <= 429)
        or (atom.chain == ZN_CHAIN and atom.resseq == ZN_RESSEQ)
    ]
    if not selected:
        raise ValueError("Isolated assembly selection retained no atoms")
    return selected


def validate_prepared_atoms(atoms: Sequence[PdbAtom], assembly: str = "joint") -> dict[str, object]:
    residues = residue_order(atoms)
    if not residues:
        raise ValueError("No residues found")
    ddb1 = sorted({a.resseq for a in atoms if a.chain == DDB1_CHAIN})
    crbn = sorted({a.resseq for a in atoms if a.chain == CRBN_CHAIN and a.resname not in PROTEIN_CAP_RESNAMES})
    caps = [(chain, resseq, name) for chain, resseq, name in residues if chain == CRBN_CHAIN and name in PROTEIN_CAP_RESNAMES]
    zn_atoms = [a for a in atoms if a.chain == ZN_CHAIN and a.resseq == ZN_RESSEQ and a.name.upper() == "ZN"]
    site = [(a.chain, a.resseq, a.resname, a.name) for a in atoms if a.chain == CRBN_CHAIN and a.resseq in ZN_CYS_RESIDUES]
    hg_atoms = [a.key for a in atoms if a.chain == CRBN_CHAIN and a.resseq in ZN_CYS_RESIDUES and a.name.upper() in {"HG", "HSG"}]
    if assembly == "joint":
        if ddb1 != list(range(1, 1141)):
            raise ValueError("DDB1 chain A must contain residues 1-1140 in the prepared PDB")
    elif assembly == "isolated":
        if ddb1:
            raise ValueError("Isolated assembly must not contain DDB1 chain A residues")
    else:
        raise ValueError(f"Unknown assembly {assembly!r}; expected one of {sorted(ASSEMBLIES)}")
    if crbn != list(range(64, 429)):
        raise ValueError("CRBN chain B must contain protein residues 64-428 in the prepared PDB")
    if len(zn_atoms) != 1:
        raise ValueError("Prepared PDB must contain exactly one Zn atom at chain C residue 501")
    if caps != [(CRBN_CHAIN, 63, "ACE"), (CRBN_CHAIN, 429, "NME")]:
        raise ValueError("Prepared PDB must contain exact CRBN caps B63 ACE and B429 NME")
    for residue in ZN_CYS_RESIDUES:
        atoms_here = [a for a in atoms if a.chain == CRBN_CHAIN and a.resseq == residue]
        if not atoms_here or any(a.resname != CY1_NAME for a in atoms_here):
            raise ValueError(f"CRBN Zn ligand residue B{residue} must be renamed to CY1")
        if not any(a.name == "SG" for a in atoms_here):
            raise ValueError(f"CRBN Zn ligand residue B{residue} lacks SG")
    if hg_atoms:
        raise ValueError(f"CY1 ligand residues retain thiol hydrogens: {hg_atoms}")
    return {
        "assembly": assembly,
        "residue_count_before_solvent": len(residues),
        "ddb1_residue_count": len(ddb1),
        "crbn_protein_residue_count": len(crbn),
        "crbn_cap_residues": caps,
        "zinc_atom": zn_atoms[0].key,
        "site_atom_rows": site,
    }


def write_normalized_pdb(input_pdb: Path, output_pdb: Path, assembly: str = "joint") -> dict[str, object]:
    atoms = read_pdb_atoms(input_pdb)
    normalized, changes = normalize_residue_names(atoms)
    selected = select_assembly_atoms(normalized, assembly)
    validation = validate_prepared_atoms(selected, assembly)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "REMARK   1 CRBN atomistic Amber bridge input; residue identities normalized only."
    ]
    previous: tuple[str, int, str] | None = None
    serial = 1
    for atom in selected:
        current = (atom.chain, atom.resseq, atom.resname)
        if previous is not None and current != previous and current[0] != previous[0]:
            lines.append(f"TER   {serial:5d}      {previous[2]:>3} {previous[0]}{previous[1]:4d}")
            serial += 1
        lines.append(format_atom(atom, serial))
        previous = current
        serial += 1
    if previous is not None:
        lines.append(f"TER   {serial:5d}      {previous[2]:>3} {previous[0]}{previous[1]:4d}")
    lines.append("END")
    output_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "input_pdb": str(input_pdb),
        "output_pdb": str(output_pdb),
        "assembly": assembly,
        "input_sha256": sha256_file(input_pdb),
        "output_sha256": sha256_file(output_pdb),
        "renamed_atom_count": len(changes),
        "renamed_examples": changes[:20],
        "validation": validation,
    }


def read_core_residues(path: Path, expected_count: int = 269) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "author_resnum" not in rows[0]:
        raise ValueError(f"core residue file lacks author_resnum: {path}")
    residues = [int(row["author_resnum"]) for row in rows]
    if len(residues) != expected_count:
        raise ValueError(f"core residue count {len(residues)} != expected {expected_count}")
    return residues


def kabsch_rows(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Kabsch inputs must have matching n x 3 shape")
    pc = source.mean(axis=0)
    qc = target.mean(axis=0)
    covariance = (source - pc).T @ (target - qc)
    u, _, vh = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(u @ vh))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vh
    return rotation, pc, qc


def rigid_basis(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    centered = coords - coords.mean(axis=0)
    columns = [np.tile(axis, (len(coords), 1)).reshape(-1) for axis in np.eye(3)]
    columns.extend(np.cross(axis, centered).reshape(-1) for axis in np.eye(3))
    u, s, _ = np.linalg.svd(np.column_stack(columns), full_matrices=False)
    if len(s) != 6 or s[-1] <= 1e-10 * s[0]:
        raise ValueError("core coordinates do not define six independent rigid motions")
    return u[:, :6]


def internal_projected_q807(
    atoms: Sequence[PdbAtom],
    core_residues: Sequence[int],
    ensemble_path: Path,
    diffvec_path: Path,
    reference_label: str,
) -> dict[str, object]:
    ca_by_residue = {
        atom.resseq: np.asarray(atom.xyz, dtype=float)
        for atom in atoms
        if atom.chain == CRBN_CHAIN and atom.name == "CA"
    }
    missing = [residue for residue in core_residues if residue not in ca_by_residue]
    if missing:
        raise ValueError(f"Prepared PDB lacks core C-alpha residues: {missing[:10]}")
    prepared_a = np.vstack([ca_by_residue[residue] for residue in core_residues])
    prepared_nm = prepared_a / 10.0

    with np.load(ensemble_path, allow_pickle=False) as ensemble:
        conformers_a = np.asarray(ensemble["_confs"], dtype=float)
        labels = [str(value) for value in ensemble["_labels"]]
    with np.load(diffvec_path, allow_pickle=False) as diff:
        open_mask = np.asarray(diff["open_mask"], dtype=bool)
        diff_labels = [str(value) for value in diff["labels"]]
        stored_axis = np.asarray(diff["diff_vec"], dtype=float)
    if labels != diff_labels:
        raise ValueError("ensemble labels and pca_diffvec labels differ")
    if conformers_a.shape != (len(labels), len(core_residues), 3):
        raise ValueError("ensemble conformer dimensions do not match core residues")
    if reference_label not in labels:
        raise ValueError(f"Reference label {reference_label} not found in ensemble")
    axis_a = (conformers_a[open_mask].mean(axis=0) - conformers_a[~open_mask].mean(axis=0)).reshape(-1)
    axis_a /= np.linalg.norm(axis_a)
    stored_axis = stored_axis / np.linalg.norm(stored_axis)
    dot = float(np.dot(axis_a, stored_axis))
    if dot < 0.999999:
        raise ValueError(f"Stored diff_vec sign/order does not match mean-open minus mean-closed: dot={dot}")

    reference_a = conformers_a[labels.index(reference_label)]
    rotation, pc, qc = kabsch_rows(reference_a, prepared_a)
    fitted = (reference_a - pc) @ rotation + qc
    rmsd_a = float(np.sqrt(np.mean(np.sum((fitted - prepared_a) ** 2, axis=1))))
    if rmsd_a >= 1e-3:
        raise ValueError(f"Prepared core does not match frozen {reference_label} core: RMSD {rmsd_a:.6g} A")

    axis_prepared = (axis_a.reshape(-1, 3) @ rotation).reshape(-1)
    axis_prepared /= np.linalg.norm(axis_prepared)
    rigid = rigid_basis(prepared_nm)
    q807 = axis_prepared - rigid @ (rigid.T @ axis_prepared)
    q807 /= np.linalg.norm(q807)
    q_internal_801 = np.linalg.qr(rigid, mode="complete")[0][:, 6:].T @ q807
    return {
        "prepared_reference_nm": prepared_nm.tolist(),
        "prepared_q": q807.tolist(),
        "q_internal_801": q_internal_801.tolist(),
        "q_source": {
            "axis_definition": "unit(mean_open - mean_closed) from data/pca_diffvec.npz labels/open_mask",
            "reference_label": reference_label,
            "stored_axis_dot_recomputed": dot,
            "kabsch_reference_to_prepared_rmsd_A": rmsd_a,
            "no_md_frame_alignment": True,
            "sign_policy": "retains stored mean-open-minus-mean-closed sign",
        },
    }


def transport_q_to_actual_frame(
    prepared_reference_nm: Sequence[Sequence[float]],
    prepared_q: Sequence[float],
    actual_core_nm: Sequence[Sequence[float]],
    *,
    rmsd_tolerance_nm: float = 1e-4,
) -> dict[str, object]:
    prepared = np.asarray(prepared_reference_nm, dtype=float)
    actual = np.asarray(actual_core_nm, dtype=float)
    q = np.asarray(prepared_q, dtype=float)
    if prepared.shape != actual.shape or prepared.ndim != 2 or prepared.shape[1] != 3:
        raise ValueError("prepared and actual core coordinates must have matching n x 3 shape")
    if q.shape != (3 * len(prepared),):
        raise ValueError("q length must be three times the core coordinate count")
    if not np.isfinite(prepared).all() or not np.isfinite(actual).all() or not np.isfinite(q).all():
        raise ValueError("prepared reference, actual reference and q must be finite")
    rotation, pc, qc = kabsch_rows(prepared, actual)
    fitted = (prepared - pc) @ rotation + qc
    rmsd_nm = float(np.sqrt(np.mean(np.sum((fitted - actual) ** 2, axis=1))))
    if rmsd_nm >= rmsd_tolerance_nm:
        raise ValueError(f"Amber core coordinates differ from prepared input: RMSD {rmsd_nm:.6g} nm")
    q_rotated = (q.reshape(-1, 3) @ rotation).reshape(-1)
    q_rotated /= np.linalg.norm(q_rotated)
    rigid = rigid_basis(actual)
    rigid_component = float(np.linalg.norm(rigid.T @ q_rotated))
    q_projected = q_rotated - rigid @ (rigid.T @ q_rotated)
    q_projected /= np.linalg.norm(q_projected)
    gauge_component = float(np.linalg.norm(rigid.T @ q_projected))
    if gauge_component >= 1e-10:
        raise ValueError(f"Actual-frame q retains rigid-body component {gauge_component:.3e}")
    return {
        "reference_nm": actual.tolist(),
        "q": q_projected.tolist(),
        "q_transport": {
            "prepared_to_actual_core_rmsd_nm": rmsd_nm,
            "prepared_to_actual_core_rmsd_tolerance_nm": rmsd_tolerance_nm,
            "pre_projection_rigid_component_norm": rigid_component,
            "post_projection_rigid_component_norm": gauge_component,
            "method": "Kabsch prepared core coordinates to actual Amber core coordinates",
        },
    }


def leap_residue_indices(atoms: Sequence[PdbAtom]) -> dict[tuple[str, int, str], int]:
    return {key: idx for idx, key in enumerate(residue_order(atoms), start=1)}


def zinc_bond_selectors(atoms: Sequence[PdbAtom], assembly: str = "joint") -> list[dict[str, object]]:
    indices = leap_residue_indices(atoms)
    zn_key = next(
        (key for key in indices if key[0] == ZN_CHAIN and key[1] == ZN_RESSEQ and key[2] == ZN1_NAME),
        None,
    )
    if zn_key is None:
        raise ValueError("Cannot locate ZN1 residue for LEaP bonding")
    if assembly == "joint":
        zn_leap_sequence_number = indices[zn_key]
    elif assembly == "isolated":
        zn_leap_sequence_number = max(
            resseq for chain, resseq, _ in indices if chain == CRBN_CHAIN and resseq <= 429
        ) + 1
    else:
        raise ValueError(f"Unknown assembly {assembly!r}; expected one of {sorted(ASSEMBLIES)}")
    selectors = []
    for residue in ZN_CYS_RESIDUES:
        key = (CRBN_CHAIN, residue, CY1_NAME)
        if key not in indices:
            raise ValueError(f"Cannot locate CY1 B{residue} for LEaP bonding")
        if assembly == "joint":
            command = f"bond mol.{indices[key]}.SG mol.{indices[zn_key]}.ZN"
            selector_policy = "one_based_residue_order"
        else:
            command = f"bond mol.{residue}.SG mol.{zn_leap_sequence_number}.ZN"
            selector_policy = "isolated_preserved_crbn_resseq_with_post_nme_zn"
        selectors.append(
            {
                "command": command,
                "selector_policy": selector_policy,
                "one_based_residue_order_command": f"bond mol.{indices[key]}.SG mol.{indices[zn_key]}.ZN",
                "sg_residue_order_index": indices[key],
                "zn_residue_order_index": indices[zn_key],
                "sg_leap_sequence_number": indices[key] if assembly == "joint" else residue,
                "zn_leap_sequence_number": zn_leap_sequence_number,
                "sg_residue": {"chain": CRBN_CHAIN, "resseq": residue, "resname": CY1_NAME},
                "zn_residue": {"chain": ZN_CHAIN, "resseq": ZN_RESSEQ, "resname": ZN1_NAME},
            }
        )
    return selectors


def zinc_bond_commands(atoms: Sequence[PdbAtom], assembly: str = "joint") -> list[str]:
    return [str(row["command"]) for row in zinc_bond_selectors(atoms, assembly=assembly)]


def residue_name_compatible(expected_name: str, actual_name: str) -> bool:
    if expected_name == "HIS" and actual_name in HIS_PROTONATION_NAMES:
        return True
    return expected_name == actual_name


def build_tleap_input(
    pdb: Path,
    prep: Path,
    frcmod: Path,
    output_prefix: str,
    bond_commands: Sequence[str],
    *,
    solvated: bool,
) -> str:
    lines = [
        "# Generated by scripts/prepare_atomistic_amber.py",
        "# Technical pilot only: no MD is run by this script.",
        "source leaprc.protein.ff14SB",
        "source leaprc.water.tip3p",
        "# Annotate ZAFF custom atom types so generated prmtops carry element metadata.",
        "addAtomTypes {",
        '    { "ZN" "Zn" "sp3" }',
        '    { "S1" "S" "sp3" }',
        "}",
        f"loadAmberPrep {prep.resolve()}",
        f"loadAmberParams {frcmod.resolve()}",
        f"mol = loadPdb {pdb.resolve()}",
        *bond_commands,
        "check mol",
    ]
    if solvated:
        lines.extend(
            [
                "solvateOct mol TIP3PBOX 12.0",
                "# Neutralization-only pilot. 150 mM salt-pair placement remains pending.",
                "addIonsRand mol Na+ 0",
                "addIonsRand mol Cl- 0",
            ]
        )
    lines.extend(
        [
            f"saveAmberParm mol {output_prefix}.prmtop {output_prefix}.inpcrd",
            f"savePdb mol {output_prefix}.pdb",
            "quit",
            "",
        ]
    )
    return "\n".join(lines)


def run_tleap(input_file: Path, workdir: Path, executable: str) -> dict[str, object]:
    command = [executable, "-f", str(input_file)]
    label = input_file.stem
    stdout_path = workdir / f"{label}.stdout.log"
    stderr_path = workdir / f"{label}.stderr.log"
    combined_path = workdir / f"{label}.combined.log"
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return {
            "command": command,
            "returncode": None,
            "status": "not_run_missing_executable",
            "log": None,
            "failures": [f"{executable!r} not found"],
        }
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    combined = completed.stdout + "\n" + completed.stderr
    combined_path.write_text(combined, encoding="utf-8")
    warning_report = generic_actual_warning_labels(classify_leap_warnings(combined_path))
    failures: list[str] = []
    if completed.returncode != 0:
        failures.append(f"LEaP returned nonzero exit code {completed.returncode}")
    if warning_report.get("errors_total") not in (0, None):
        failures.append(f"LEaP reported Errors = {warning_report['errors_total']}")
    for pattern in LEAP_LOG_ERROR_PATTERNS:
        if pattern.lower() in combined.lower():
            failures.append(f"LEaP log contains fatal/parameter text: {pattern}")
    return {
        "command": command,
        "returncode": completed.returncode,
        "status": "complete" if not failures else "failed",
        "log": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "combined": str(combined_path),
            "leap_warnings": warning_report,
        },
        "failures": failures,
    }


def topology_mapping(
    prmtop: Path,
    inpcrd: Path,
    prepared_atoms: Sequence[PdbAtom],
    core_residues: Sequence[int],
    q_payload: dict[str, object],
    prep: Path,
    assembly: str = "joint",
) -> dict[str, object]:
    try:
        from openmm import app, unit
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenMM is required to build the Amber atom-index mapping") from exc

    amber_top = app.AmberPrmtopFile(str(prmtop))
    amber_crd = app.AmberInpcrdFile(str(inpcrd))
    positions_nm = np.asarray(amber_crd.positions.value_in_unit(unit.nanometer), dtype=float)
    atoms = list(amber_top.topology.atoms())
    if positions_nm.shape != (len(atoms), 3):
        raise ValueError("Amber coordinate count does not match topology atom count")
    system = amber_top.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
    zaff_contract = generic_actual_warning_labels(
        parameter_contract(
            system,
            amber_top.topology,
            prep_path=prep,
            atom_types=amber_top._prmtop.getAtomTypes(),
        )
    )
    if zaff_contract["status"] != "pass":
        raise ValueError(f"ZAFF topology contract failed: {zaff_contract['failures']}")

    prepared_residues = residue_order(prepared_atoms)
    top_residues = list(amber_top.topology.residues())
    if len(top_residues) < len(prepared_residues):
        raise ValueError("Amber topology has fewer residues than the prepared PDB")
    residue_map: dict[tuple[str, int, str], object] = {}
    for key, residue in zip(prepared_residues, top_residues):
        expected_name = key[2]
        actual_name = residue.name
        if not residue_name_compatible(expected_name, actual_name):
            raise ValueError(f"Residue order mismatch at {key}: Amber has {actual_name}")
        residue_map[key] = residue

    def atom_index(key: tuple[str, int, str], atom_name: str) -> int:
        residue = residue_map[key]
        matches = [atom.index for atom in residue.atoms() if atom.name == atom_name]
        if len(matches) != 1:
            raise ValueError(f"Expected one atom {atom_name} in residue {key}, found {len(matches)}")
        return int(matches[0])

    core_indices = [atom_index((CRBN_CHAIN, residue, residue_map_name(residue_map, CRBN_CHAIN, residue)), "CA") for residue in core_residues]
    ddb1_atom_indices = [
        int(atom.index)
        for key, residue in residue_map.items()
        if key[0] == DDB1_CHAIN
        for atom in residue.atoms()
    ]
    protein_ca_indices = [
        int(atom.index)
        for key, residue in residue_map.items()
        if key[0] in {DDB1_CHAIN, CRBN_CHAIN} and key[2] not in PROTEIN_CAP_RESNAMES
        for atom in residue.atoms()
        if atom.name == "CA"
    ]
    zn_atom_index = atom_index((ZN_CHAIN, ZN_RESSEQ, ZN1_NAME), "ZN")
    zn_sg_indices = [atom_index((CRBN_CHAIN, residue, CY1_NAME), "SG") for residue in ZN_CYS_RESIDUES]
    actual_core_nm = positions_nm[core_indices]
    actual_q = transport_q_to_actual_frame(
        q_payload["prepared_reference_nm"],
        q_payload["prepared_q"],
        actual_core_nm,
    )
    return {
        "schema_version": "1.0",
        "assembly": assembly,
        "topology_files": {"prmtop": str(prmtop), "inpcrd": str(inpcrd)},
        "atom_count": len(atoms),
        "residue_count": len(top_residues),
        "core_indices": core_indices,
        "ddb1_atom_indices": ddb1_atom_indices,
        "protein_ca_indices": protein_ca_indices,
        "zn_atom_index": zn_atom_index,
        "zn_sg_indices": zn_sg_indices,
        "topology_checks": {
            "core_count": len(core_indices),
            "ddb1_atom_count_including_h": len(ddb1_atom_indices),
            "ddb1_expected_empty": assembly == "isolated",
            "protein_ca_count": len(protein_ca_indices),
            "zaff_parameter_contract_status": zaff_contract["status"],
            "allowed_his_protonation_mapping": sorted(HIS_PROTONATION_NAMES),
        },
        "zaff_parameter_contract": zaff_contract,
        "q_source": q_payload["q_source"],
        **actual_q,
    }


def residue_map_name(residue_map: dict[tuple[str, int, str], object], chain: str, resseq: int) -> str:
    matches = [name for c, r, name in residue_map if c == chain and r == resseq]
    if len(matches) != 1:
        raise ValueError(f"Expected one residue for {chain}{resseq}, found {len(matches)}")
    return matches[0]


def write_tleap_files(
    prepared_pdb: Path,
    prep: Path,
    frcmod: Path,
    output_dir: Path,
    bond_commands: Sequence[str],
) -> dict[str, object]:
    dry_input = output_dir / "tleap_dry.in"
    solvated_input = output_dir / "tleap_solvated_neutral.in"
    dry_input.write_text(
        build_tleap_input(prepared_pdb, prep, frcmod, "dry", bond_commands, solvated=False),
        encoding="utf-8",
    )
    solvated_input.write_text(
        build_tleap_input(prepared_pdb, prep, frcmod, "solvated", bond_commands, solvated=True),
        encoding="utf-8",
    )
    return {
        "dry": str(dry_input),
        "solvated_neutral": str(solvated_input),
        "salt_policy": {
            "status": "neutralization_only",
            "reason": "150 mM salt-pair count is not inferred without a validated solvated box/water count pass",
        },
    }


def prepare_amber_bridge(
    *,
    config_path: Path | None,
    input_pdb: Path,
    prep: Path,
    frcmod: Path,
    output_dir: Path,
    offline: bool,
    tleap: str | None,
    ensemble_path: Path,
    diffvec_path: Path,
    assembly: str = "joint",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    core_residues = read_core_residues(Path(str(config["core_residue_file"])), int(config["core_position_count"]))
    if assembly not in ASSEMBLIES:
        raise ValueError(f"Unknown assembly {assembly!r}; expected one of {sorted(ASSEMBLIES)}")
    prepared_pdb = output_dir / f"amber_input_{assembly}_renamed.pdb"
    pdb_report = write_normalized_pdb(input_pdb, prepared_pdb, assembly=assembly)
    prepared_atoms = read_pdb_atoms(prepared_pdb)
    bond_selectors = zinc_bond_selectors(prepared_atoms, assembly=assembly)
    bond_commands = [str(row["command"]) for row in bond_selectors]
    tleap_inputs = write_tleap_files(prepared_pdb, prep, frcmod, output_dir, bond_commands)
    q_payload = internal_projected_q807(
        prepared_atoms,
        core_residues,
        ensemble_path,
        diffvec_path,
        str(config.get("pilot_reference", "8CVP")),
    )

    leap_runs: dict[str, object] = {}
    leap_failures: list[str] = []
    if tleap:
        for label, input_file in (
            ("dry", output_dir / "tleap_dry.in"),
            ("solvated_neutral", output_dir / "tleap_solvated_neutral.in"),
        ):
            leap_runs[label] = run_tleap(input_file, output_dir, tleap)
            leap_failures.extend(f"{label}: {failure}" for failure in leap_runs[label]["failures"])
    else:
        leap_runs["status"] = "not_run"
        leap_runs["reason"] = "Pass --tleap to run AmberTools; local input generation only."

    mapping: dict[str, object] | None = None
    mapping_status: dict[str, object]
    solvated_prmtop = output_dir / "solvated.prmtop"
    solvated_inpcrd = output_dir / "solvated.inpcrd"
    dry_outputs = {"prmtop": str(output_dir / "dry.prmtop"), "inpcrd": str(output_dir / "dry.inpcrd")}
    if tleap and leap_failures:
        mapping_status = {
            "status": "blocked_by_tleap_failure",
            "failures": leap_failures,
            "stale_output_policy": "No mapping is built after a failed current LEaP run.",
        }
    elif tleap and solvated_prmtop.is_file() and solvated_inpcrd.is_file():
        mapping = topology_mapping(
            solvated_prmtop,
            solvated_inpcrd,
            prepared_atoms,
            core_residues,
            q_payload,
            prep,
            assembly=assembly,
        )
        write_json(output_dir / "atomistic_mapping.json", mapping)
        mapping_status = {
            "status": "complete",
            "path": str(output_dir / "atomistic_mapping.json"),
            "mapped_system": "solvated",
            "dry_outputs": dry_outputs,
        }
    else:
        mapping_status = {
            "status": "pending_amber_outputs",
            "required": [str(solvated_prmtop), str(solvated_inpcrd)],
            "stale_output_policy": "Existing Amber outputs are ignored unless produced by a successful current --tleap run.",
            "prepared_q_precheck": {
                "prepared_reference_nm_count": len(q_payload["prepared_reference_nm"]),
                "prepared_q_length": len(q_payload["prepared_q"]),
            },
        }

    source_files = {
        "input_pdb": {"path": str(input_pdb), "sha256": sha256_file(input_pdb)},
        "prep": {"path": str(prep), "sha256": sha256_file(prep)},
        "frcmod": {"path": str(frcmod), "sha256": sha256_file(frcmod)},
        "ensemble": {"path": str(ensemble_path), "sha256": sha256_file(ensemble_path)},
        "diffvec": {"path": str(diffvec_path), "sha256": sha256_file(diffvec_path)},
    }
    report = {
        "schema_version": "1.0",
        "status": (
            "amber_outputs_mapped"
            if mapping is not None
            else "tleap_failed_mapping_blocked"
            if leap_failures
            else "leap_inputs_ready_mapping_pending"
        ),
        "assembly": assembly,
        "offline": offline,
        "production_ready": False,
        "production_ready_reason": (
            "Technical pilot bridge only: neutralization-only LEaP input, no 150 mM salt validation, "
            "no MD, and no production qualification."
        ),
        "config": config,
        "pdb_preparation": pdb_report,
        "zinc_site": {
            "residue_model": "four CY1 thiolate residues plus one ZN1 center",
            "bond_commands": list(bond_commands),
            "bond_selectors": bond_selectors,
            "one_based_residue_order_bond_commands": [
                row["one_based_residue_order_command"] for row in bond_selectors
            ],
            "requested_site_bond_command_count": len(bond_commands),
            "actual_site_bond_validation": "performed by verify_zaff_amber_topology.parameter_contract after solvated topology exists",
        },
        "tleap_inputs": tleap_inputs,
        "tleap_runs": leap_runs,
        "mapping": mapping_status,
        "source_files": source_files,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / "preparation_report.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--assembly", choices=sorted(ASSEMBLIES), default="joint")
    parser.add_argument(
        "--input-pdb",
        type=Path,
        default=None,
        help="Capped heavy PDB. Defaults only to results/atomistic/capped_heavy.pdb when present.",
    )
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--frcmod", type=Path, default=DEFAULT_FRCMOD)
    parser.add_argument("--tleap", nargs="?", const="tleap", help="Optional tleap executable; omit to only write inputs")
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--diffvec", type=Path, default=DEFAULT_DIFFVEC)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_pdb = args.input_pdb
    if input_pdb is None:
        if DEFAULT_INPUT_PDB.is_file():
            input_pdb = DEFAULT_INPUT_PDB
        else:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "reason": "missing required --input-pdb; default results/atomistic/capped_heavy.pdb is absent",
                    },
                    indent=2,
                )
            )
            return 2
    if args.tleap and shutil.which(args.tleap) is None and not Path(args.tleap).is_file():
        print(json.dumps({"status": "error", "reason": f"tleap executable not found: {args.tleap}"}, indent=2))
        return 2
    report = prepare_amber_bridge(
        config_path=args.config,
        input_pdb=input_pdb.resolve(),
        prep=args.prep.resolve(),
        frcmod=args.frcmod.resolve(),
        output_dir=args.output_dir.resolve(),
        offline=args.offline,
        tleap=args.tleap,
        ensemble_path=args.ensemble.resolve(),
        diffvec_path=args.diffvec.resolve(),
        assembly=args.assembly,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "production_ready": report["production_ready"],
                "assembly": report["assembly"],
                "output_dir": str(args.output_dir),
                "mapping": report["mapping"]["status"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] in {"amber_outputs_mapped", "leap_inputs_ready_mapping_pending"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
