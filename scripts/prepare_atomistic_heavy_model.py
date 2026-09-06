#!/usr/bin/env python3
"""Prepare a heavy-atom-only 8CVP CRBN-DDB1 input model.

This script performs the first repair step needed before any atomistic response
simulation. It uses PDBFixer only to add prespecified internal missing residues
and missing heavy atoms. It deliberately does not add hydrogens, water, ions,
membranes, restraints, caps, or an OpenMM System.

Primary PDBFixer API basis: the OpenMM/PDBFixer manual describes the sequence
findMissingResidues(), edit fixer.missingResidues if needed, findMissingAtoms(),
and addMissingAtoms(); findMissingAtoms() populates missingAtoms and
missingTerminals, which can be edited before addMissingAtoms().
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
from importlib import metadata
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

try:
    from curation_contracts import cif_loop_rows
    from atomistic_input_audit import STANDARD_HEAVY_ATOMS
except ImportError:  # pragma: no cover
    from scripts.curation_contracts import cif_loop_rows
    from scripts.atomistic_input_audit import STANDARD_HEAVY_ATOMS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts" / "atomistic_config.json"
DEFAULT_OUTPUT = ROOT / "results" / "atomistic" / "heavy_model"
DEFAULT_REFERENCE = "8CVP"
CRBN_CHAIN = "B"
DDB1_CHAIN = "A"
CRBN_ALLOWED_INTERNAL_GAP = set(range(342, 358))
DDB1_ALLOWED_INTERNAL_GAP = set(range(546, 551))
ALLOWED_GAPS = {CRBN_CHAIN: CRBN_ALLOWED_INTERNAL_GAP, DDB1_CHAIN: DDB1_ALLOWED_INTERNAL_GAP}
PROTECTED_ZN_CYS = {323, 326, 391, 394}
ANGSTROM_TOL = 1e-3
EXPECTED_RESIDUE_RANGES = {DDB1_CHAIN: (1, 1140), CRBN_CHAIN: (64, 428)}


@dataclass(frozen=True)
class AtomRecord:
    chain: str
    resseq: int
    atom: str
    comp: str
    x: float
    y: float
    z: float
    element: str
    group: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.atom)

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text_auto(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    return payload.decode("utf-8")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path | None) -> dict[str, Any]:
    base = {
        "references": [DEFAULT_REFERENCE],
        "cif_dir": "data/_cif_cache",
        "core_residue_file": "data/crbn_residue_window.csv",
        "core_position_count": 269,
    }
    if path is None:
        path = DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"config not found: {path}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        base.update({k: loaded[k] for k in ("references", "cif_dir", "core_residue_file", "core_position_count") if k in loaded})
        base["config_file"] = str(path.resolve())
        base["ignored_config_keys"] = sorted(set(loaded) - {"references", "cif_dir", "core_residue_file", "core_position_count"})
    else:
        base["config_file"] = None
        base["ignored_config_keys"] = []
    base["cif_dir"] = str(resolve_repo_path(base["cif_dir"]).resolve())
    base["core_residue_file"] = str(resolve_repo_path(base["core_residue_file"]).resolve())
    return base


def read_core_positions(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "author_resnum" not in rows[0]:
        raise ValueError(f"core residue file lacks author_resnum: {path}")
    return [int(row["author_resnum"]) for row in rows]


def parse_input_atoms(cif_text: str, chains: set[str] | None = None) -> dict[tuple[str, int, str], AtomRecord]:
    """Parse one unambiguous heavy-atom model without silent key overwrites."""
    atoms: dict[tuple[str, int, str], AtomRecord] = {}
    for row in cif_loop_rows(cif_text, "atom_site"):
        model = row.get("pdbx_PDB_model_num", "1")
        if model not in {"1", ".", "?"}:
            raise ValueError(f"Unsupported atom_site model {model}; select one model explicitly")
        chain = row.get("auth_asym_id", "")
        if chains and chain not in chains:
            continue
        element = row.get("type_symbol", "").upper()
        if element == "H":
            continue
        if row.get("label_alt_id", ".") not in {".", "?", ""} or row.get("pdbx_PDB_ins_code", ".") not in {".", "?", ""}:
            raise ValueError("Unsupported alternate location or insertion code in heavy-model input")
        auth_seq = row.get("auth_seq_id", "")
        if not str(auth_seq).lstrip("-").isdigit():
            raise ValueError("Heavy-model atom_site requires integer author residue IDs")
        atom = row.get("auth_atom_id") or row.get("label_atom_id") or ""
        if not atom:
            raise ValueError("Heavy-model atom_site contains an unnamed atom")
        try:
            record = AtomRecord(
                chain=chain,
                resseq=int(auth_seq),
                atom=atom,
                comp=row.get("auth_comp_id") or row.get("label_comp_id") or "UNK",
                x=float(row["Cartn_x"]),
                y=float(row["Cartn_y"]),
                z=float(row["Cartn_z"]),
                element=element,
                group=row.get("group_PDB", ""),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid heavy-model atom coordinates") from exc
        if not all(math.isfinite(value) for value in record.xyz):
            raise ValueError("Heavy-model atom coordinates must be finite")
        if record.key in atoms:
            raise ValueError(f"Duplicate atom identity in heavy-model input: {record.key}")
        atoms[record.key] = record
    return atoms


def infer_missing_numbers(chain_residue_ids: list[int], insertion_index: int, names: list[str]) -> list[int]:
    count = len(names)
    if count == 0:
        return []
    if insertion_index <= 0:
        if not chain_residue_ids:
            return list(range(1, count + 1))
        end = chain_residue_ids[0] - 1
        return list(range(end - count + 1, end + 1))
    if insertion_index >= len(chain_residue_ids):
        start = chain_residue_ids[-1] + 1
        return list(range(start, start + count))
    prev_id = chain_residue_ids[insertion_index - 1]
    return list(range(prev_id + 1, prev_id + 1 + count))


def filter_missing_residues(missing_residues: dict[Any, list[str]], chains: list[Any], allowed_gaps: dict[str, set[int]]) -> tuple[dict[Any, list[str]], list[dict[str, Any]]]:
    kept: dict[Any, list[str]] = {}
    decisions: list[dict[str, Any]] = []
    for key, names in list(missing_residues.items()):
        chain_index, insertion_index = key
        chain = chains[chain_index]
        chain_id = getattr(chain, "id", str(chain_index))
        residue_ids = [int(res.id) for res in chain.residues() if str(res.id).lstrip("-").isdigit()]
        numbers = infer_missing_numbers(residue_ids, insertion_index, list(names))
        is_terminal = bool(numbers) and (numbers[-1] < min(residue_ids, default=numbers[-1]) or numbers[0] > max(residue_ids, default=numbers[0]))
        allowed = allowed_gaps.get(chain_id, set())
        if numbers and set(numbers) <= allowed and not is_terminal:
            kept[key] = names
            action = "keep_for_internal_repair"
        else:
            action = "drop_terminal_or_unapproved_gap"
        decisions.append({"chain": chain_id, "key": [chain_index, insertion_index], "residue_numbers": numbers, "residue_names": list(names), "terminal": is_terminal, "action": action})
    missing_residues.clear()
    missing_residues.update(kept)
    return kept, decisions


def clear_missing_terminals(fixer: Any) -> int:
    missing = getattr(fixer, "missingTerminals", {})
    count = sum(len(v) for v in missing.values()) if isinstance(missing, dict) else 0
    if isinstance(missing, dict):
        missing.clear()
    return count


def distance(a: Iterable[float], b: Iterable[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def topology_atoms(topology: Any, positions: Any) -> dict[tuple[str, int, str], AtomRecord]:
    try:
        from openmm import unit
        coords = positions.value_in_unit(unit.angstrom)
    except Exception:
        coords = positions
    out: dict[tuple[str, int, str], AtomRecord] = {}
    seen_residues = set()
    index = 0
    for chain in topology.chains():
        for residue in chain.residues():
            residue_key = (chain.id, residue.id)
            if residue_key in seen_residues:
                raise ValueError(f"Duplicate residue identity in repaired topology: {residue_key}")
            seen_residues.add(residue_key)
            if not str(residue.id).lstrip("-").isdigit():
                raise ValueError(f"Unsupported noninteger residue ID in repaired topology: {residue_key}")
            for atom in residue.atoms():
                element = getattr(getattr(atom, "element", None), "symbol", "") or ""
                if element.upper() == "H":
                    index += 1
                    continue
                xyz = coords[index]
                record = AtomRecord(chain=chain.id, resseq=int(residue.id), atom=atom.name, comp=residue.name, x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]), element=element.upper(), group="")
                if record.key in out:
                    raise ValueError(f"Duplicate atom identity in repaired topology: {record.key}")
                if not all(math.isfinite(value) for value in record.xyz):
                    raise ValueError(f"Nonfinite repaired coordinates: {record.key}")
                out[record.key] = record
                index += 1
    return out


def expected_heavy_sequence(cif_text: str) -> dict[tuple[str, int], str]:
    """Read exact 8CVP construct identities, including unobserved loop residues."""
    expected = {}
    for row in cif_loop_rows(cif_text, "pdbx_poly_seq_scheme"):
        chain = row.get("pdb_strand_id")
        if chain not in EXPECTED_RESIDUE_RANGES:
            continue
        try:
            residue = int(row["pdb_seq_num"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Sequence scheme must provide integer pdb_seq_num") from exc
        start, end = EXPECTED_RESIDUE_RANGES[chain]
        if not start <= residue <= end:
            continue
        if row.get("pdb_ins_code", ".") not in {".", "?", ""} or row.get("hetero", "n").lower() != "n":
            raise ValueError("Unsupported insertion or microheterogeneity in construct sequence")
        author = row.get("auth_seq_num", "?")
        if author not in {".", "?", ""} and author != str(residue):
            raise ValueError("8CVP construct author and PDB sequence numbering must agree")
        key = (chain, residue)
        if key in expected:
            raise ValueError(f"Duplicate construct sequence identity: {key}")
        name = row.get("mon_id", "").upper()
        if name not in STANDARD_HEAVY_ATOMS:
            raise ValueError(f"Unsupported heavy-atom residue definition: {key} {name}")
        expected[key] = name
    required = {(chain, residue) for chain, (start, end) in EXPECTED_RESIDUE_RANGES.items() for residue in range(start, end+1)}
    missing = sorted(required - set(expected))
    if missing:
        raise ValueError(f"Source sequence lacks {len(missing)} required construct residues; examples: {missing[:10]}")
    return expected


def heavy_model_postconditions(cif_text: str, atoms: Iterable[AtomRecord], core_positions: Iterable[int]) -> dict[str, Any]:
    """Check repair inventory independently of preservation of observed atoms.

    Exact A1-1140/B64-428 identities come from the source sequence scheme,
    including the two requested loops. OXT is optional because terminal
    chemistry is handled separately. Passing is not a chemical/MD qualification.
    """
    expected = expected_heavy_sequence(cif_text)
    groups: dict[tuple[str, int], list[AtomRecord]] = {}
    seen, duplicate_keys, nonfinite_keys = set(), [], []
    for atom in atoms:
        if atom.key in seen:
            duplicate_keys.append(list(atom.key))
        seen.add(atom.key)
        if not all(math.isfinite(value) for value in atom.xyz):
            nonfinite_keys.append(list(atom.key))
        if atom.element.upper() == "H":
            continue
        if atom.element.upper() == "ZN" and atom.atom.upper() == "ZN":
            continue
        groups.setdefault((atom.chain, atom.resseq), []).append(atom)
    missing_residues = sorted(set(expected) - set(groups))
    unexpected_residues = sorted(set(groups) - set(expected))
    missing_atoms, unexpected_atoms, mismatches = [], [], []
    for key in sorted(set(expected) & set(groups)):
        names = {atom.comp for atom in groups[key]}
        if names != {expected[key]}:
            mismatches.append({"residue": list(key), "expected": expected[key], "observed": sorted(names)})
        observed = {atom.atom for atom in groups[key]}
        required = STANDARD_HEAVY_ATOMS[expected[key]]
        absent = sorted(required - observed)
        extra = sorted(observed - required - {"OXT"})
        if absent:
            missing_atoms.append({"residue": list(key), "name": expected[key], "missing": absent})
        if extra:
            unexpected_atoms.append({"residue": list(key), "unexpected": extra})
    core = list(core_positions)
    if len(core) != len(set(core)) or any((CRBN_CHAIN, residue) not in expected for residue in core):
        raise ValueError("Core IDs must be unique positions within the selected CRBN construct")
    core_ca_missing = [residue for residue in core if (CRBN_CHAIN, residue, "CA") not in seen]
    complete = not any((missing_residues, unexpected_residues, missing_atoms, unexpected_atoms,
                        mismatches, duplicate_keys, nonfinite_keys, core_ca_missing))
    return {
        "repair_complete": complete,
        "assessment_scope": "prespecified_heavy_atom_inventory_not_chemical_or_MD_qualification",
        "expected_residue_ranges": {chain: list(bounds) for chain, bounds in EXPECTED_RESIDUE_RANGES.items()},
        "expected_protein_residue_count": len(expected),
        "observed_protein_residue_counts": {chain: sum(key[0] == chain for key in groups) for chain in EXPECTED_RESIDUE_RANGES},
        "expected_heavy_atom_count": sum(len(STANDARD_HEAVY_ATOMS[name]) for name in expected.values()),
        "missing_residues": [list(key) for key in missing_residues],
        "unexpected_residues": [list(key) for key in unexpected_residues],
        "missing_heavy_atoms": missing_atoms, "unexpected_heavy_atoms": unexpected_atoms,
        "residue_identity_mismatches": mismatches,
        "duplicate_atom_keys": duplicate_keys, "nonfinite_atom_keys": nonfinite_keys,
        "core_ca_count_expected": len(core), "core_ca_missing": core_ca_missing,
        "OXT_required": False,
        "requested_loop_identity_checks": [
            {"chain": chain, "residue": residue, "expected": expected[(chain, residue)],
             "observed": sorted({atom.comp for atom in groups.get((chain, residue), [])})}
            for chain in sorted(ALLOWED_GAPS) for residue in sorted(ALLOWED_GAPS[chain])
        ],
    }


def preparation_status(preservation: dict, metal: dict, postconditions: dict) -> str:
    if preservation["preserved"] and metal["single_zn_and_four_cys_sg"] and postconditions["repair_complete"]:
        return "complete"
    return "blocked_after_repair_validation"


def dependency_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("openmm", "pdbfixer"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def compare_preserved_coordinates(before: dict[tuple[str, int, str], AtomRecord], after: dict[tuple[str, int, str], AtomRecord], required_keys: set[tuple[str, int, str]], tol: float = ANGSTROM_TOL) -> dict[str, Any]:
    missing_after = []
    moved = []
    max_delta = 0.0
    for key in sorted(required_keys):
        b = before.get(key)
        a = after.get(key)
        if b is None:
            continue
        if a is None:
            missing_after.append(key)
            continue
        delta = distance(b.xyz, a.xyz)
        max_delta = max(max_delta, delta)
        if delta > tol:
            moved.append({"key": list(key), "delta_a": delta})
    return {"checked_atom_count": len(required_keys), "missing_after": [list(k) for k in missing_after], "moved_atom_examples": moved[:20], "moved_atom_count": len(moved), "max_delta_a": max_delta, "preserved": not missing_after and not moved}


def metal_requirements(before: dict[tuple[str, int, str], AtomRecord], after: dict[tuple[str, int, str], AtomRecord] | None = None) -> dict[str, Any]:
    source = after or before
    zn = [atom for atom in source.values() if atom.element.upper() == "ZN" or atom.atom.upper() == "ZN"]
    cys_sg = [source.get((CRBN_CHAIN, resnum, "SG")) for resnum in sorted(PROTECTED_ZN_CYS)]
    cys_sg_present = [atom for atom in cys_sg if atom is not None]
    return {"zn_count": len(zn), "crbn_cys_sg_residues_present": [atom.resseq for atom in cys_sg_present], "single_zn_and_four_cys_sg": len(zn) == 1 and len(cys_sg_present) == 4}


def map_single_metal_identity(before, after):
    """Track PDBx's distinct metal-chain label without accepting a moved metal."""
    original = [a for a in before.values() if a.element == "ZN"]
    repaired = [a for a in after.values() if a.element == "ZN"]
    mapped = dict(before)
    changes = []
    if len(original) == len(repaired) == 1 and original[0].key != repaired[0].key:
        src, dst = original[0], repaired[0]
        if dst.key in mapped:
            raise ValueError("Metal identity mapping would overwrite another input atom")
        del mapped[src.key]
        mapped[dst.key] = replace(src, chain=dst.chain, resseq=dst.resseq, atom=dst.atom)
        changes.append({"source_key": list(src.key), "output_key": list(dst.key),
                        "coordinate_difference_A": distance(src.xyz, dst.xyz),
                        "reason": "PDBx/PDBFixer assigns the nonpolymer a distinct chain label"})
    return mapped, changes


def write_mapping_csv(path: Path, atoms: dict[tuple[str, int, str], AtomRecord], before: dict[tuple[str, int, str], AtomRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["chain", "residue_number", "residue_name", "atom", "element", "source_status", "x", "y", "z"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, atom in sorted(atoms.items()):
            gap = ALLOWED_GAPS.get(atom.chain, set())
            if key in before:
                status = "observed_input"
            elif atom.resseq in gap:
                status = "repaired_internal_missing_residue_or_atom"
            else:
                status = "repaired_missing_heavy_atom"
            writer.writerow({"chain": atom.chain, "residue_number": atom.resseq, "residue_name": atom.comp, "atom": atom.atom, "element": atom.element, "source_status": status, "x": f"{atom.x:.6f}", "y": f"{atom.y:.6f}", "z": f"{atom.z:.6f}"})


def require_dependencies() -> tuple[Any, Any, Any]:
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        try:
            from openmm.app import PDBxFile
        except ImportError:  # pragma: no cover
            PDBxFile = None
    except ImportError as exc:  # pragma: no cover - local env may not include PDBFixer
        raise RuntimeError("PDBFixer/OpenMM are required for heavy-model preparation; no repair was performed") from exc
    return PDBFixer, PDBFile, PDBxFile


def call_add_missing_atoms(fixer: Any, seed: int) -> dict[str, Any]:
    signature = inspect.signature(fixer.addMissingAtoms)
    if "seed" in signature.parameters:
        fixer.addMissingAtoms(seed=seed)
        return {"called": "addMissingAtoms(seed=seed)", "seed_used": seed, "seed_supported": True}
    fixer.addMissingAtoms()
    return {"called": "addMissingAtoms()", "seed_used": None, "seed_supported": False, "requested_seed": seed}


def write_structure_file(writer: Any, topology: Any, positions: Any, handle: Any) -> dict[str, Any]:
    signature = inspect.signature(writer.writeFile)
    if "keepIds" in signature.parameters:
        writer.writeFile(topology, positions, handle, keepIds=True)
        return {"keepIds_requested": True, "keepIds_supported": True}
    writer.writeFile(topology, positions, handle)
    return {"keepIds_requested": True, "keepIds_supported": False}


def prepare_heavy_model(config_path: Path | None, output_dir: Path, reference: str, seed: int, offline: bool) -> dict[str, Any]:
    config = load_config(config_path)
    reference = reference.upper()
    cif_path = Path(config["cif_dir"]) / f"{reference}.cif.gz"
    if not cif_path.is_file():
        message = f"missing retained CIF input: {cif_path}"
        if offline:
            raise FileNotFoundError(message)
        raise FileNotFoundError(message + "; downloader is intentionally not implemented")
    core_positions = read_core_positions(Path(config["core_residue_file"]))
    if len(core_positions) != int(config.get("core_position_count", 269)):
        raise ValueError("core residue count does not match protocol config")
    if reference != DEFAULT_REFERENCE:
        raise ValueError("heavy-model preparation is currently protocol-scoped to 8CVP only")

    PDBFixer, PDBFile, PDBxFile = require_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_text = read_text_auto(cif_path)
    before_atoms = parse_input_atoms(cif_text, {DDB1_CHAIN, CRBN_CHAIN})
    expected_heavy_sequence(cif_text)  # Validate source identities before running a repair.
    before_metal = metal_requirements(before_atoms)

    # PDBFixer reads ordinary .cif; write a local decompressed copy for reproducibility.
    input_cif = output_dir / f"{reference}_input.cif"
    input_cif.write_text(cif_text, encoding="utf-8")
    fixer = PDBFixer(filename=str(input_cif))
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    kept_missing, missing_decisions = filter_missing_residues(fixer.missingResidues, chains, ALLOWED_GAPS)
    fixer.findMissingAtoms()
    dropped_terminal_atom_count = clear_missing_terminals(fixer)
    missing_atoms_before_add = {
        f"{res.chain.id}:{res.id}:{res.name}": [atom.name for atom in atoms]
        for res, atoms in getattr(fixer, "missingAtoms", {}).items()
    }
    seed_call = call_add_missing_atoms(fixer, seed)

    repaired_pdb = output_dir / "repaired_heavy.pdb"
    writer_calls = {}
    with repaired_pdb.open("w", encoding="utf-8") as handle:
        writer_calls["pdb"] = write_structure_file(PDBFile, fixer.topology, fixer.positions, handle)
    repaired_cif = output_dir / "repaired_heavy.cif"
    if PDBxFile is not None:
        with repaired_cif.open("w", encoding="utf-8") as handle:
            writer_calls["cif"] = write_structure_file(PDBxFile, fixer.topology, fixer.positions, handle)
    else:
        repaired_cif.write_text("# PDBxFile unavailable in this OpenMM installation; see repaired_heavy.pdb\n", encoding="utf-8")
        writer_calls["cif"] = {"available": False}

    after_atoms = topology_atoms(fixer.topology, fixer.positions)
    postconditions = heavy_model_postconditions(cif_text, after_atoms.values(), core_positions)
    mapped_before, identity_changes = map_single_metal_identity(before_atoms, after_atoms)
    protected_keys = set(mapped_before)
    preservation = compare_preserved_coordinates(mapped_before, after_atoms, protected_keys)
    after_metal = metal_requirements(before_atoms, after_atoms)
    if not after_metal["single_zn_and_four_cys_sg"]:
        preservation["metal_restore_required"] = True
        preservation["metal_restore_note"] = "PDBFixer output did not preserve the single Zn and four observed CRBN Cys SG atoms; exact metal restoration is required before any downstream use."

    write_mapping_csv(output_dir / "atom_residue_mapping.csv", after_atoms, mapped_before)
    report = {
        "status": "complete",
        "script_sha256": sha256_file(Path(__file__)),
        "dependency_versions": dependency_versions(),
        "reference": reference,
        "seed": seed,
        "offline": offline,
        "config": config,
        "input_cif": {"path": str(cif_path), "sha256": sha256_file(cif_path), "bytes": cif_path.stat().st_size},
        "core_residue_file": {"path": config["core_residue_file"], "sha256": sha256_file(Path(config["core_residue_file"])), "count": len(core_positions)},
        "allowed_missing_residue_repair": {"DDB1_A": [546, 547, 548, 549, 550], "CRBN_B": [342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357]},
        "missing_residue_decisions": missing_decisions,
        "kept_missing_residue_entries": len(kept_missing),
        "missing_atoms_before_add": missing_atoms_before_add,
        "dropped_missing_terminal_atom_count": dropped_terminal_atom_count,
        "add_missing_atoms_call": seed_call,
        "coordinate_preservation": preservation,
        "heavy_atom_postconditions": postconditions,
        "repair_complete": postconditions["repair_complete"],
        "atom_identity_changes": identity_changes,
        "metal_before": before_metal,
        "metal_after": after_metal,
        "writer_calls": writer_calls,
        "outputs": {"repaired_heavy_pdb": str(repaired_pdb), "repaired_heavy_cif": str(repaired_cif), "atom_residue_mapping_csv": str(output_dir / "atom_residue_mapping.csv")},
        "production_ready": False,
        "production_ready_reason": "heavy-atom repair only; metal parameterization, terminal chemistry/caps, protonation/hydrogens, solvent/ions, and simulation gauge are not qualified",
        "policy": "No hydrogens, water, ions, caps, topology/System construction, or dynamics were added/performed.",
    }
    report["status"] = preparation_status(preservation, after_metal, postconditions)
    (output_dir / "preparation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Protocol JSON; defaults to scripts/atomistic_config.json when present.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = prepare_heavy_model(args.config, args.output_dir.resolve(), args.reference, args.seed, args.offline)
    print(json.dumps({"status": report["status"], "production_ready": report["production_ready"], "output_dir": str(args.output_dir), "reference": report["reference"]}, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
