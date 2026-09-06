#!/usr/bin/env python3
"""Audit atomistic-model inputs before any structure repair or MD assembly.

The audit is deliberately read-only: it parses retained mmCIF files, identifies
CRBN/DDB1 chains by UniProt mapping, records observed residue/atom completeness,
and inventories coordinate repairs and unresolved construct choices. It cannot
qualify chemical completeness, topology, metal parameters, or simulation gauges.
It does not create peptide topology, fill missing residues, or run dynamics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:  # allow both `python scripts/foo.py` and module imports in tests
    from curation_contracts import CRBN_ACCESSION, DDB1_ACCESSION, cif_loop_rows
except ImportError:  # pragma: no cover
    from scripts.curation_contracts import CRBN_ACCESSION, DDB1_ACCESSION, cif_loop_rows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts" / "atomistic_config.json"
DEFAULT_CIF_CACHE = ROOT / "data" / "_cif_cache"
DEFAULT_CORE_WINDOW = ROOT / "data" / "crbn_residue_window.csv"
DEFAULT_OUTPUT = ROOT / "results" / "atomistic" / "input_audit"
DEFAULT_REFS = ("8CVP", "8D7X", "8D7Y")
CANONICAL_LENGTHS = {CRBN_ACCESSION: 442, DDB1_ACCESSION: 1140}
BACKBONE_HEAVY = {"N", "CA", "C", "O"}
STANDARD_HEAVY_ATOMS = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
    "MSE": {"N", "CA", "C", "O", "CB", "CG", "SE", "CE"},
}


@dataclass(frozen=True)
class ChainMap:
    accession: str
    chain: str
    align_id: str
    seq_begin: int
    seq_end: int
    db_begin: int
    db_end: int
    auth_begin: str
    auth_end: str

    def seq_to_uniprot(self, seq_id: int) -> int | None:
        if self.seq_begin <= seq_id <= self.seq_end:
            return self.db_begin + (seq_id - self.seq_begin)
        return None


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


def read_core_window(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "author_resnum" not in rows[0]:
        raise ValueError(f"core window must contain author_resnum: {path}")
    residues = [int(row["author_resnum"]) for row in rows]
    if len(residues) != len(set(residues)):
        raise ValueError("core window contains duplicate author_resnum values")
    return residues


def as_int(value: str | None) -> int | None:
    if value in (None, "", ".", "?"):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def chain_maps(cif_text: str, accession: str) -> list[ChainMap]:
    maps: list[ChainMap] = []
    for row in cif_loop_rows(cif_text, "struct_ref_seq"):
        if row.get("pdbx_db_accession", "").upper() != accession.upper():
            continue
        seq_begin = as_int(row.get("seq_align_beg"))
        seq_end = as_int(row.get("seq_align_end"))
        db_begin = as_int(row.get("db_align_beg"))
        db_end = as_int(row.get("db_align_end"))
        if None in (seq_begin, seq_end, db_begin, db_end):
            raise ValueError(f"invalid struct_ref_seq mapping for {accession}: {row}")
        for chain in row.get("pdbx_strand_id", "").split(","):
            chain = chain.strip()
            if chain and chain not in {".", "?"}:
                maps.append(
                    ChainMap(
                        accession=accession.upper(),
                        chain=chain,
                        align_id=row.get("align_id", ""),
                        seq_begin=int(seq_begin),
                        seq_end=int(seq_end),
                        db_begin=int(db_begin),
                        db_end=int(db_end),
                        auth_begin=row.get("pdbx_auth_seq_align_beg", "?"),
                        auth_end=row.get("pdbx_auth_seq_align_end", "?"),
                    )
                )
    return sorted(maps, key=lambda item: item.chain)


def mapped_poly_scheme(cif_text: str, mapping: ChainMap) -> list[dict[str, Any]]:
    rows = []
    for row in cif_loop_rows(cif_text, "pdbx_poly_seq_scheme"):
        if row.get("pdb_strand_id") != mapping.chain:
            continue
        seq_id = as_int(row.get("seq_id"))
        if seq_id is None:
            continue
        uniprot = mapping.seq_to_uniprot(seq_id)
        if uniprot is None:
            continue
        ins = row.get("pdb_ins_code") or row.get("pdbx_PDB_ins_code") or "."
        rows.append(
            {
                "chain": mapping.chain,
                "seq_id": seq_id,
                "uniprot_resnum": uniprot,
                "auth_seq_num": row.get("auth_seq_num", "?"),
                "pdb_seq_num": row.get("pdb_seq_num", "?"),
                "ins_code": ins,
                "mon_id": row.get("mon_id") or row.get("auth_mon_id") or "UNK",
                "pdb_mon_id": row.get("pdb_mon_id", "?"),
                "auth_mon_id": row.get("auth_mon_id", "?"),
                "hetero": row.get("hetero", "?"),
            }
        )
    rows.sort(key=lambda item: (item["uniprot_resnum"], item["seq_id"], item["auth_seq_num"], item["ins_code"]))
    return rows


def atom_site_by_residue(cif_text: str, chain: str) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    residues: dict[tuple[str, str], dict[str, Any]] = {}
    atom_variants: dict[tuple[str, str, str], set[str]] = {}
    unsupported: list[dict[str, Any]] = []
    for row in cif_loop_rows(cif_text, "atom_site"):
        model = row.get("pdbx_PDB_model_num", "1")
        if model not in {"1", ".", "?"}:
            unsupported.append({"code": "additional_model_ignored", "model": model, "chain": row.get("auth_asym_id", "?")})
            continue
        if row.get("group_PDB") != "ATOM" or row.get("auth_asym_id") != chain:
            continue
        element = row.get("type_symbol", "").upper()
        if element == "H":
            continue
        auth_seq = row.get("auth_seq_id", "?")
        ins = row.get("pdbx_PDB_ins_code") or "."
        if ins == "?":
            ins = "."
        atom = row.get("auth_atom_id") or row.get("label_atom_id")
        if not atom:
            continue
        altloc = row.get("label_alt_id", ".") or "."
        if altloc == "?":
            altloc = "."
        atom_key = (auth_seq, ins, atom)
        atom_variants.setdefault(atom_key, set()).add(altloc)
        if len(atom_variants[atom_key] - {"."}) > 1 or ("." in atom_variants[atom_key] and len(atom_variants[atom_key]) > 1):
            unsupported.append({"code": "altloc_ambiguity", "auth_seq_num": auth_seq, "ins_code": ins, "atom": atom, "altlocs": sorted(atom_variants[atom_key])})
        if altloc not in {".", "A"}:
            continue
        key = (auth_seq, ins)
        rec = residues.setdefault(
            key,
            {
                "auth_seq_num": auth_seq,
                "ins_code": ins,
                "comp_id": row.get("auth_comp_id") or row.get("label_comp_id") or "UNK",
                "atoms": set(),
                "coords": {},
            },
        )
        rec["atoms"].add(atom)
        try:
            xyz = (float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"]))
            if not all(math.isfinite(value) for value in xyz):
                raise ValueError("nonfinite coordinates")
            rec["coords"][atom] = xyz
        except (KeyError, ValueError):
            unsupported.append({"code": "invalid_atom_coordinates", "auth_seq_num": auth_seq, "ins_code": ins, "atom": atom})
    dedup = []
    seen = set()
    for item in unsupported:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    return residues, dedup


def ranges(values: Iterable[int]) -> list[dict[str, int]]:
    ordered = sorted(set(values))
    if not ordered:
        return []
    out = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        out.append({"start": start, "end": prev, "length": prev - start + 1})
        start = prev = value
    out.append({"start": start, "end": prev, "length": prev - start + 1})
    return out


def split_terminal_internal(missing: set[int], reference_positions: set[int]) -> dict[str, list[dict[str, int]]]:
    if not missing:
        return {"terminal_missing": [], "internal_missing": []}
    if not reference_positions:
        return {"terminal_missing": ranges(missing), "internal_missing": []}
    lo, hi = min(reference_positions), max(reference_positions)
    terminal = {v for v in missing if v < lo or v > hi}
    internal = set(missing) - terminal
    return {"terminal_missing": ranges(terminal), "internal_missing": ranges(internal)}


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def zn_coordination(cif_text: str, crbn_chain: str) -> list[dict[str, Any]]:
    atoms = cif_loop_rows(cif_text, "atom_site")
    zn_atoms = []
    sg_atoms = []
    for row in atoms:
        if row.get("pdbx_PDB_model_num", "1") not in {"1", ".", "?"} or row.get("label_alt_id", ".") not in {".", "?", "A"}:
            continue
        try:
            xyz = (float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"]))
            if not all(math.isfinite(value) for value in xyz):
                continue
        except (KeyError, ValueError):
            continue
        element = row.get("type_symbol", "").upper()
        atom_name = row.get("auth_atom_id") or row.get("label_atom_id") or ""
        if element == "ZN" or atom_name.upper() == "ZN":
            zn_atoms.append({"chain": row.get("auth_asym_id", "?"), "auth_seq_id": row.get("auth_seq_id", "?"), "comp_id": row.get("auth_comp_id") or row.get("label_comp_id"), "xyz": xyz})
        if row.get("auth_asym_id") == crbn_chain and atom_name == "SG":
            sg_atoms.append({"auth_seq_id": row.get("auth_seq_id", "?"), "comp_id": row.get("auth_comp_id") or row.get("label_comp_id"), "xyz": xyz})
    out = []
    for zn in zn_atoms:
        neighbors = []
        for sg in sg_atoms:
            d = distance(zn["xyz"], sg["xyz"])
            if d <= 3.0:
                neighbors.append({"auth_seq_id": sg["auth_seq_id"], "comp_id": sg["comp_id"], "distance_a": round(d, 3)})
        neighbors.sort(key=lambda item: item["distance_a"])
        out.append({"zn": {k: v for k, v in zn.items() if k != "xyz"}, "sg_neighbors_within_3a": neighbors, "sg_count_within_3a": len(neighbors), "sg4_coordination_ok": len(neighbors) == 4})
    return out


def audit_chain(cif_text: str, pdb_id: str, mapping: ChainMap, core_positions: list[int] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scheme = mapped_poly_scheme(cif_text, mapping)
    atoms, atom_site_unsupported = atom_site_by_residue(cif_text, mapping.chain)
    residue_rows = []
    blockers = []
    construct_positions = {row["uniprot_resnum"] for row in scheme}
    canonical_len = CANONICAL_LENGTHS.get(mapping.accession)
    construct_deletions = set(range(1, canonical_len + 1)) - construct_positions if canonical_len else set()
    coord_missing_ca = set()
    incomplete_heavy = set()
    observed_incomplete_heavy = set()
    insertion_like = []
    seen_uniprot = set()
    nonmonotonic_auth = []
    prev_auth_int: int | None = None
    for row in scheme:
        auth = row["auth_seq_num"]
        ins = row["ins_code"] if row["ins_code"] not in {"?", ""} else "."
        atom_rec = atoms.get((auth, ins)) or atoms.get((auth, "."))
        observed_atoms = set(atom_rec["atoms"]) if atom_rec else set()
        expected_atoms = STANDARD_HEAVY_ATOMS.get(str(row["mon_id"]).upper())
        missing_expected = sorted(expected_atoms - observed_atoms) if expected_atoms else []
        has_ca = atom_rec is not None and "CA" in atom_rec["coords"]
        if not has_ca:
            coord_missing_ca.add(row["uniprot_resnum"])
        if missing_expected:
            incomplete_heavy.add(row["uniprot_resnum"])
            if observed_atoms:
                observed_incomplete_heavy.add(row["uniprot_resnum"])
        auth_int = as_int(auth)
        if auth_int is not None and prev_auth_int is not None and auth_int < prev_auth_int:
            nonmonotonic_auth.append({"uniprot_resnum": row["uniprot_resnum"], "auth_seq_num": auth, "previous_auth_seq_num": prev_auth_int})
        if auth_int is not None:
            prev_auth_int = auth_int
        if row["uniprot_resnum"] in seen_uniprot or ins not in {".", "?"}:
            insertion_like.append({"uniprot_resnum": row["uniprot_resnum"], "auth_seq_num": auth, "ins_code": ins})
        seen_uniprot.add(row["uniprot_resnum"])
        residue_rows.append(
            {
                "pdb_id": pdb_id,
                "accession": mapping.accession,
                "chain": mapping.chain,
                "seq_id": row["seq_id"],
                "uniprot_resnum": row["uniprot_resnum"],
                "auth_seq_num": auth,
                "ins_code": ins,
                "mon_id": row["mon_id"],
                "observed_ca": has_ca,
                "observed_heavy_atom_count": len(observed_atoms),
                "expected_heavy_atom_count": len(expected_atoms) if expected_atoms else None,
                "missing_expected_heavy_atoms": ";".join(missing_expected),
                "full_expected_heavy_atoms": expected_atoms is not None and not missing_expected,
                "in_core269": bool(core_positions and row["uniprot_resnum"] in set(core_positions)),
            }
        )
    observed_ca_positions = construct_positions - coord_missing_ca
    coord_split = split_terminal_internal(coord_missing_ca, observed_ca_positions)
    deletion_split = split_terminal_internal(construct_deletions, construct_positions)
    invalid_coordinates = [item for item in atom_site_unsupported if item["code"] == "invalid_atom_coordinates"]
    other_unsupported = [item for item in atom_site_unsupported if item["code"] != "invalid_atom_coordinates"]
    if coord_split["internal_missing"]:
        blockers.append({"severity": "blocker", "code": "internal_missing_coordinates_require_repair", "ranges": coord_split["internal_missing"]})
    if observed_incomplete_heavy:
        blockers.append({"severity": "blocker", "code": "observed_residue_incomplete_heavy_atoms_require_repair", "ranges": ranges(observed_incomplete_heavy)})
    if coord_split["terminal_missing"]:
        blockers.append({"severity": "review", "code": "unobserved_terminals_require_construct_choice", "ranges": coord_split["terminal_missing"], "message": "Choose and document the simulated construct and terminal chemistry; these ranges are not an instruction to fill every terminal residue."})
    if deletion_split["internal_missing"]:
        blockers.append({"severity": "review", "code": "internal_construct_deletion_requires_topology_definition", "ranges": deletion_split["internal_missing"], "message": "A construct deletion is not an unobserved loop; define its sequence and connectivity before assembly."})
    if invalid_coordinates:
        blockers.append({"severity": "blocker", "code": "invalid_atom_coordinates", "examples": invalid_coordinates[:20], "count": len(invalid_coordinates)})
    core_summary: dict[str, Any] | None = None
    if core_positions is not None and mapping.accession == CRBN_ACCESSION:
        core_set = set(core_positions)
        missing_from_construct = core_set - construct_positions
        core_missing_ca = core_set & coord_missing_ca
        core_incomplete = core_set & incomplete_heavy
        core_summary = {
            "core_count": len(core_positions),
            "core_missing_from_construct": ranges(missing_from_construct),
            "core_missing_ca": ranges(core_missing_ca),
            "core_incomplete_expected_heavy_atoms": ranges(core_incomplete),
            "measurement_core_coordinates_complete": not (missing_from_construct or core_missing_ca),
            "gapped_core269_md_allowed": False,
        }
        if missing_from_construct:
            blockers.append({"severity": "blocker", "code": "core269_missing_from_construct", "ranges": ranges(missing_from_construct)})
        if core_missing_ca:
            blockers.append({"severity": "blocker", "code": "core269_missing_ca_coordinates", "ranges": ranges(core_missing_ca)})
        if core_incomplete:
            blockers.append({"severity": "warning", "code": "core269_incomplete_expected_heavy_atoms", "ranges": ranges(core_incomplete)})
        blockers.append({"severity": "policy", "code": "do_not_build_md_from_gapped_core269_pdb", "message": "The 269 positions define the measurement window only. A core-only gapped PDB is prohibited as the simulated construct even when every measurement coordinate is available."})
    if nonmonotonic_auth:
        blockers.append({"severity": "blocker", "code": "nonmonotonic_author_numbering", "examples": nonmonotonic_auth[:10]})
    if insertion_like:
        blockers.append({"severity": "review", "code": "insertion_or_duplicate_uniprot_mapping", "examples": insertion_like[:10]})
    if other_unsupported:
        blockers.append({"severity": "blocker", "code": "unsupported_multimodel_or_altloc_atom_site", "examples": other_unsupported[:20], "count": len(other_unsupported)})
    summary = {
        "pdb_id": pdb_id,
        "accession": mapping.accession,
        "chain": mapping.chain,
        "align_id": mapping.align_id,
        "auth_range": [mapping.auth_begin, mapping.auth_end],
        "uniprot_range": [mapping.db_begin, mapping.db_end],
        "construct_residue_count": len(construct_positions),
        "observed_ca_residue_count": len(observed_ca_positions),
        "coordinate_missing_ca": coord_split,
        "construct_deletions_vs_canonical": deletion_split,
        "incomplete_expected_heavy_atom_residue_count": len(incomplete_heavy),
        "observed_incomplete_heavy_atom_residues": ranges(observed_incomplete_heavy),
        "coordinate_repairs_required": bool(coord_split["internal_missing"] or observed_incomplete_heavy or invalid_coordinates),
        "terminal_construct_choice_required": bool(coord_split["terminal_missing"]),
        "insertion_or_duplicate_mapping_count": len(insertion_like),
        "nonmonotonic_author_numbering_count": len(nonmonotonic_auth),
        "core269": core_summary,
    }
    return summary, residue_rows, blockers


def audit_structure(path: Path, core_positions: list[int]) -> dict[str, Any]:
    pdb_id = path.name.split(".")[0].upper()
    text = read_text_auto(path)
    input_hash = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    crbn_maps = chain_maps(text, CRBN_ACCESSION)
    ddb1_maps = chain_maps(text, DDB1_ACCESSION)
    chain_summaries = []
    residue_rows = []
    blockers = []
    for mapping in crbn_maps + ddb1_maps:
        summary, rows, chain_blockers = audit_chain(text, pdb_id, mapping, core_positions if mapping.accession == CRBN_ACCESSION else None)
        chain_summaries.append(summary)
        residue_rows.extend(rows)
        for item in chain_blockers:
            blockers.append({"pdb_id": pdb_id, "chain": mapping.chain, "accession": mapping.accession, **item})
    if not crbn_maps:
        blockers.append({"pdb_id": pdb_id, "severity": "blocker", "code": "crbn_q96sw2_chain_not_identified"})
    if not ddb1_maps:
        blockers.append({"pdb_id": pdb_id, "severity": "blocker", "code": "ddb1_q16531_chain_not_identified"})
    zn = []
    for mapping in crbn_maps:
        zn.extend(zn_coordination(text, mapping.chain))
    if not zn:
        blockers.append({"pdb_id": pdb_id, "severity": "blocker", "code": "crbn_structural_zinc_coordinate_absent"})
    elif not all(item["sg4_coordination_ok"] for item in zn):
        blockers.append({"pdb_id": pdb_id, "severity": "blocker", "code": "crbn_zinc_not_sg4_coordinated", "zn_sites": zn})
    return {
        "pdb_id": pdb_id,
        "input_file": input_hash,
        "identified_chains": {"CRBN_Q96SW2": [m.chain for m in crbn_maps], "DDB1_Q16531": [m.chain for m in ddb1_maps]},
        "chain_summaries": chain_summaries,
        "residue_rows": residue_rows,
        "zinc_coordination": zn,
        "coordinate_repairs_required": any(chain["coordinate_repairs_required"] for chain in chain_summaries) or not zn,
        "terminal_construct_choice_required": any(chain["terminal_construct_choice_required"] for chain in chain_summaries),
        "md_assembly_blockers": blockers,
    }


def resolve_repo_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def default_config_path() -> Path | None:
    return DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None


def load_config(path: Path | None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "refs": list(DEFAULT_REFS),
        "cif_cache": str(DEFAULT_CIF_CACHE),
        "core_window": str(DEFAULT_CORE_WINDOW),
        "core_position_count": 269,
        "config_file": None,
        "ignored_config_keys": [],
    }
    if path is None:
        path = default_config_path()
    if path and path.is_file():
        user = json.loads(path.read_text(encoding="utf-8"))
        refs = user.get("refs", user.get("references", config["refs"]))
        cif_cache = user.get("cif_cache", user.get("cif_dir", config["cif_cache"]))
        core_window = user.get("core_window", user.get("core_residue_file", config["core_window"]))
        relevant = {"refs", "references", "cif_cache", "cif_dir", "core_window", "core_residue_file", "core_position_count"}
        config.update(
            {
                "refs": refs,
                "cif_cache": resolve_repo_path(cif_cache),
                "core_window": resolve_repo_path(core_window),
                "core_position_count": int(user.get("core_position_count", config["core_position_count"])),
                "config_file": str(path.resolve()),
                "ignored_config_keys": sorted(set(user) - relevant),
            }
        )
    elif path:
        raise FileNotFoundError(f"config not found: {path}")
    else:
        config["cif_cache"] = resolve_repo_path(config["cif_cache"])
        config["core_window"] = resolve_repo_path(config["core_window"])
    return config


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(config_path: Path | None = None, output_dir: Path = DEFAULT_OUTPUT, offline: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    refs = [str(ref).upper() for ref in config["refs"]]
    cif_cache = Path(config["cif_cache"]).resolve()
    core_window_path = Path(config["core_window"]).resolve()
    core_positions = read_core_window(core_window_path)
    expected_core_count = int(config.get("core_position_count", 269))
    if len(core_positions) != expected_core_count:
        raise ValueError(f"expected {expected_core_count} core positions, found {len(core_positions)} in {core_window_path}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    structures = []
    residue_rows = []
    blocker_rows = []
    missing_inputs = []
    for ref in refs:
        path = cif_cache / f"{ref}.cif.gz"
        if not path.is_file():
            missing_inputs.append(str(path))
            continue
        record = audit_structure(path, core_positions)
        structures.append({k: v for k, v in record.items() if k != "residue_rows"})
        residue_rows.extend(record["residue_rows"])
        for blocker in record["md_assembly_blockers"]:
            blocker_rows.append({"pdb_id": blocker.get("pdb_id", record["pdb_id"]), "chain": blocker.get("chain", ""), "accession": blocker.get("accession", ""), "severity": blocker.get("severity", ""), "code": blocker.get("code", ""), "details_json": json.dumps({k: v for k, v in blocker.items() if k not in {"pdb_id", "chain", "accession", "severity", "code"}}, sort_keys=True)})
    if missing_inputs:
        if offline:
            raise FileNotFoundError(f"offline atomistic audit missing retained CIF input(s): {missing_inputs}")
        raise FileNotFoundError(f"missing retained CIF input(s); downloader is intentionally not implemented: {missing_inputs}")
    summary_rows = []
    for structure in structures:
        for chain in structure["chain_summaries"]:
            core = chain.get("core269") or {}
            summary_rows.append({
                "pdb_id": structure["pdb_id"],
                "accession": chain["accession"],
                "chain": chain["chain"],
                "construct_residue_count": chain["construct_residue_count"],
                "observed_ca_residue_count": chain["observed_ca_residue_count"],
                "incomplete_expected_heavy_atom_residue_count": chain["incomplete_expected_heavy_atom_residue_count"],
                "measurement_core_coordinates_complete": core.get("measurement_core_coordinates_complete", ""),
                "coordinate_repairs_required": chain["coordinate_repairs_required"],
                "terminal_construct_choice_required": chain["terminal_construct_choice_required"],
                "core_missing_from_construct": json.dumps(core.get("core_missing_from_construct", []), sort_keys=True),
                "core_missing_ca": json.dumps(core.get("core_missing_ca", []), sort_keys=True),
            })
    report = {
        "status": "complete",
        "offline": offline,
        "config": config,
        "core_window": {"path": str(core_window_path), "sha256": sha256_file(core_window_path), "count": len(core_positions)},
        "structures": structures,
        "blocker_count": len(blocker_rows),
        "md_ready_without_repair": False,
        "production_ready": False,
        "coordinate_repairs_required": any(structure["coordinate_repairs_required"] for structure in structures),
        "terminal_construct_choice_required": any(structure["terminal_construct_choice_required"] for structure in structures),
        "gapped_core269_md_allowed": False,
        "assessment_scope": "input_coordinate_inventory_only_not_chemical_completeness_or_simulation_qualification",
        "readiness_not_assessed": ["chemical_completeness", "terminal_chemistry", "metal_parameters", "topology_connectivity", "simulation_gauge"],
        "script_sha256": sha256_file(Path(__file__)),
        "policy": "read-only inventory; gapped core269-only MD is always prohibited; no peptide topology, residue filling, PDBFixer/OpenMM repair, or dynamics performed",
    }
    (output_dir / "atomistic_input_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / "chain_summary.csv", summary_rows)
    write_csv(output_dir / "residue_completeness.csv", residue_rows)
    write_csv(output_dir / "md_assembly_blockers.csv", blocker_rows)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON config with references/cif_dir/core_residue_file or refs/cif_cache/core_window; defaults to scripts/atomistic_config.json when present.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="Require retained local CIF inputs; no downloads are attempted.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config
    report = run(config_path=config_path, output_dir=args.output_dir, offline=args.offline)
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir), "blocker_count": report["blocker_count"], "md_ready_without_repair": report["md_ready_without_repair"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
