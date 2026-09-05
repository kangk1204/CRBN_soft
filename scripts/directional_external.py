#!/usr/bin/env python3
"""External structural and functional comparisons for CRBN directional mechanics.

The outputs are retrospective comparisons. They map public structures and
reported variants to the frozen contact-candidate universe without treating
the comparison as functional validation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "directional_mechanics"
DEFAULT_CONFIG = ROOT / "scripts" / "directional_config.json"
DEFAULT_REFERENCE_INPUTS = DEFAULT_OUTPUT / "data" / "directional_reference_inputs"
ROOT_REFERENCE_INPUTS = ROOT / "data" / "directional_reference_inputs"
DEFAULT_STAGED_CANDIDATES = DEFAULT_REFERENCE_INPUTS / "candidate_universe.csv"
DEFAULT_CANDIDATES = ROOT / "results" / "strengthening" / "analysis" / "contacts" / "candidate_robustness.csv"
DEFAULT_OConnor_118 = ROOT / "results" / "strengthening" / "analysis" / "external"

PDB_ID = "9SFM"
CRBN_ACCESSION = "Q96SW2"
CRBN_CHAIN = "B"
ALLOSTERIC_LIGAND = "A1CEG"
ALLOSTERIC_CUTOFF_A = 4.5
RCSB_CIF_URL = f"https://files.rcsb.org/download/{PDB_ID}.cif.gz"
RCSB_ENTRY_URL = f"https://data.rcsb.org/rest/v1/core/entry/{PDB_ID}"
RCSB_STRUCTURE_URL = f"https://www.rcsb.org/structure/{PDB_ID}"
BLOOD_PDF_URL = "https://repository.icr.ac.uk/server/api/core/bitstreams/87360e29-79d0-4f89-9077-892963bd343e/content"
BLOOD_PMC_ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12824623/"
BLOOD_EUROPEPMC_SUPPLEMENT_ZIP_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12824623/supplementaryFiles"
BLOOD_SUPPLEMENT_URL = "https://pmc.ncbi.nlm.nih.gov/articles/instance/12824623/bin/BLOOD_BLD-2024-025861-mmc1.pdf"
BLOOD_NCBI_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC12824623"
BLOOD_SUPPLEMENT_MEMBER = "BLOOD_BLD-2024-025861-mmc1.pdf"
BLOOD_DOI = "10.1182/blood.2024025861"
BLOOD_PMID = "39841463"
OCONNOR_DOI = "10.64898/2025.12.19.695617"


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    url: str
    local_path: str
    retrieved_utc: str
    status: str
    bytes: int | None
    sha256: str | None
    version: str = ""
    error: str = ""
    current_run_status: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_url(url: str, timeout: int = 120) -> tuple[int | None, bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "crbn-directional-external/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return getattr(response, "status", None), response.read(), ""
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except OSError:
            body = b""
        return exc.code, body, f"HTTP {exc.code}"
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        return None, b"", str(exc)


def load_previous_sources(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(rows, list):
        return {}
    return {str(row.get("source_id")): row for row in rows if isinstance(row, dict) and row.get("source_id")}


def preserve_previous_record(previous: Mapping[str, Any] | None, fallback: SourceRecord, current_status: str) -> SourceRecord:
    if previous is None:
        return SourceRecord(**{**fallback.__dict__, "current_run_status": current_status})
    if fallback.sha256 and previous.get("sha256") and fallback.sha256 != previous.get("sha256"):
        raise RuntimeError(f"cached source {fallback.source_id} differs from previous registry")
    return SourceRecord(
        source_id=str(previous.get("source_id", fallback.source_id)),
        url=str(previous.get("url", fallback.url)),
        local_path=fallback.local_path,
        retrieved_utc=str(previous.get("retrieved_utc", fallback.retrieved_utc)),
        status=str(previous.get("status", fallback.status)),
        bytes=previous.get("bytes", fallback.bytes),
        sha256=previous.get("sha256", fallback.sha256),
        version=str(previous.get("version", fallback.version)),
        error=str(previous.get("error", fallback.error) or ""),
        current_run_status=current_status,
    )


def acquire(
    url: str,
    path: Path,
    source_id: str,
    offline: bool,
    previous: Mapping[str, Any] | None = None,
) -> SourceRecord:
    retrieved = utc_now()
    if offline:
        if not path.is_file():
            fallback = SourceRecord(source_id, url, str(path), retrieved, "missing_offline", None, None, error="not present in cache")
            return preserve_previous_record(previous, fallback, "missing_offline")
        fallback = SourceRecord(source_id, url, str(path), retrieved, "cached", path.stat().st_size, sha256_file(path))
        return preserve_previous_record(previous, fallback, "cached")
    status, payload, error = fetch_url(url)
    if status is not None and 200 <= status < 300 and payload:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return SourceRecord(source_id, url, str(path), retrieved, f"http_{status}", len(payload), sha256_bytes(payload))
    if path.is_file():
        return SourceRecord(
            source_id,
            url,
            str(path),
            retrieved,
            f"cached_after_http_{status}" if status else "cached_after_fetch_failure",
            path.stat().st_size,
            sha256_file(path),
            error=error,
        )
    return SourceRecord(source_id, url, str(path), retrieved, f"http_{status}" if status else "failed", None, None, error=error)


def acquire_pdf_or_response(
    url: str,
    pdf_path: Path,
    response_path: Path,
    source_id: str,
    offline: bool,
    previous: Mapping[str, Any] | None = None,
) -> SourceRecord:
    retrieved = utc_now()
    if offline:
        if pdf_path.is_file() and pdf_path.read_bytes().startswith(b"%PDF"):
            fallback = SourceRecord(source_id, url, str(pdf_path), retrieved, "cached_pdf", pdf_path.stat().st_size, sha256_file(pdf_path))
            return preserve_previous_record(previous, fallback, "cached_pdf")
        if response_path.is_file():
            fallback = SourceRecord(
                source_id,
                url,
                str(response_path),
                retrieved,
                "cached_non_pdf_response",
                response_path.stat().st_size,
                sha256_file(response_path),
                error="cached response is not a PDF",
            )
            return preserve_previous_record(previous, fallback, "cached_non_pdf_response")
        fallback = SourceRecord(source_id, url, str(pdf_path), retrieved, "missing_offline", None, None, error="not present in cache")
        return preserve_previous_record(previous, fallback, "missing_offline")
    status, payload, error = fetch_url(url)
    if status is not None and 200 <= status < 300 and payload and payload.startswith(b"%PDF"):
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(payload)
        return SourceRecord(source_id, url, str(pdf_path), retrieved, f"http_{status}_pdf", len(payload), sha256_bytes(payload))
    if payload:
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_bytes(payload)
        return SourceRecord(
            source_id,
            url,
            str(response_path),
            retrieved,
            f"http_{status}_non_pdf" if status else "non_pdf_fetch_response",
            len(payload),
            sha256_bytes(payload),
            error=error or "response did not start with %PDF",
        )
    return SourceRecord(source_id, url, str(pdf_path), retrieved, f"http_{status}" if status else "failed", None, None, error=error)


def extract_pdf_from_zip(
    zip_path: Path,
    member_name: str,
    pdf_path: Path,
    source_id: str,
    previous: Mapping[str, Any] | None = None,
) -> SourceRecord:
    retrieved = utc_now()
    if not zip_path.is_file():
        fallback = SourceRecord(source_id, f"extracted_from:{zip_path.as_posix()}:{member_name}", str(pdf_path), retrieved, "zip_missing", None, None, error="ZIP source missing")
        return preserve_previous_record(previous, fallback, "zip_missing") if previous else fallback
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            if member_name not in names:
                fallback = SourceRecord(
                    source_id,
                    f"extracted_from:{zip_path.as_posix()}:{member_name}",
                    str(pdf_path),
                    retrieved,
                    "member_missing",
                    None,
                    None,
                    error=f"member absent; available={';'.join(names)}",
                )
                return preserve_previous_record(previous, fallback, "member_missing") if previous else fallback
            payload = archive.read(member_name)
    except zipfile.BadZipFile as exc:
        fallback = SourceRecord(source_id, f"extracted_from:{zip_path.as_posix()}:{member_name}", str(pdf_path), retrieved, "bad_zip", None, None, error=str(exc))
        return preserve_previous_record(previous, fallback, "bad_zip") if previous else fallback
    if not payload.startswith(b"%PDF"):
        fallback = SourceRecord(
            source_id,
            f"extracted_from:{zip_path.as_posix()}:{member_name}",
            str(pdf_path),
            retrieved,
            "member_not_pdf",
            len(payload),
            sha256_bytes(payload),
            error="member did not start with %PDF",
        )
        return preserve_previous_record(previous, fallback, "member_not_pdf") if previous else fallback
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(payload)
    fallback = SourceRecord(
        source_id,
        f"extracted_from:{zip_path.as_posix()}:{member_name}",
        str(pdf_path),
        retrieved,
        "extracted_pdf",
        len(payload),
        sha256_bytes(payload),
        version=f"zip_member={member_name}",
    )
    return preserve_previous_record(previous, fallback, "extracted_pdf") if previous else fallback


def copy_cached(src: Path, dst: Path, source_id: str, previous: Mapping[str, Any] | None = None) -> SourceRecord:
    retrieved = utc_now()
    if not src.is_file():
        if dst.is_file():
            fallback = SourceRecord(
                source_id,
                f"copied_from:{src.as_posix()}",
                str(dst),
                retrieved,
                "cached_local_copy",
                dst.stat().st_size,
                sha256_file(dst),
                error="source file absent; used existing local cache",
            )
            return preserve_previous_record(previous, fallback, "cached_local_copy") if previous else fallback
        fallback = SourceRecord(source_id, str(src), str(dst), retrieved, "source_missing", None, None, error="source file absent")
        return preserve_previous_record(previous, fallback, "source_missing") if previous else fallback
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    fallback = SourceRecord(
        source_id,
        f"copied_from:{src.as_posix()}",
        str(dst),
        retrieved,
        "copied_local_cache",
        dst.stat().st_size,
        sha256_file(dst),
    )
    return preserve_previous_record(previous, fallback, "copied_local_cache") if previous else fallback


def load_cif_text(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix == ".gz":
        return gzip.decompress(payload).decode("utf-8")
    return payload.decode("utf-8")


def cif_loop_rows(text: str, category: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    prefix = f"_{category}."
    out: list[dict[str, str]] = []
    for index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        headers: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].startswith(prefix):
            headers.append(lines[cursor].strip())
            cursor += 1
        if not headers:
            continue
        names = [header.split(".", 1)[1] for header in headers]
        while cursor < len(lines):
            current = lines[cursor].strip()
            if not current:
                cursor += 1
                continue
            if current == "#" or current == "loop_" or current.startswith("_"):
                break
            fields = shlex.split(current)
            if len(fields) >= len(names):
                out.append({name: fields[pos] for pos, name in enumerate(names)})
            cursor += 1
        return out
    return out


def chain_uniprot_mapping(cif_text: str, accession: str, chain: str) -> dict[str, Any]:
    for row in cif_loop_rows(cif_text, "struct_ref_seq"):
        if row.get("pdbx_db_accession") != accession:
            continue
        chains = [item.strip() for item in row.get("pdbx_strand_id", "").split(",")]
        if chain not in chains:
            continue
        auth_beg = int(row["pdbx_auth_seq_align_beg"])
        auth_end = int(row["pdbx_auth_seq_align_end"])
        db_beg = int(row["db_align_beg"])
        db_end = int(row["db_align_end"])
        offset = db_beg - auth_beg
        return {
            "accession": accession,
            "chain": chain,
            "auth_seq_begin": auth_beg,
            "auth_seq_end": auth_end,
            "uniprot_begin": db_beg,
            "uniprot_end": db_end,
            "auth_to_uniprot_offset": offset,
            "exact_author_to_uniprot_identity": auth_beg == db_beg and auth_end == db_end,
        }
    raise ValueError(f"{PDB_ID}: no {accession} struct_ref_seq row for chain {chain}")


def _atom_key(row: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("group_PDB", ""),
        row.get("auth_asym_id", ""),
        row.get("auth_seq_id", ""),
        row.get("pdbx_PDB_ins_code", ""),
        row.get("auth_atom_id") or row.get("label_atom_id", ""),
    )


def selected_atoms(cif_text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = cif_loop_rows(cif_text, "atom_site")
    selected: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    altloc_ties: list[str] = []
    for row in raw:
        element = row.get("type_symbol", "")
        if element.upper() == "H":
            continue
        key = _atom_key(row)
        try:
            occ = float(row.get("occupancy", "1") or "1")
            xyz = np.array([float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])], dtype=float)
        except (KeyError, ValueError):
            continue
        alt = row.get("label_alt_id", ".")
        record = {**row, "occupancy_float": occ, "xyz_array": xyz}
        previous = selected.get(key)
        if previous is None:
            selected[key] = record
            continue
        prev_alt = previous.get("label_alt_id", ".")
        prev_occ = float(previous["occupancy_float"])
        keep = occ > prev_occ or (math.isclose(occ, prev_occ) and (alt == "A" and prev_alt != "A"))
        if math.isclose(occ, prev_occ) and alt != prev_alt:
            altloc_ties.append(f"{key}:{prev_alt}/{alt}@{occ:g}")
        if keep:
            selected[key] = record
    qa = {
        "raw_atom_site_rows": len(raw),
        "selected_heavy_atom_rows": len(selected),
        "altloc_tie_rule": "highest occupancy; if tied, altloc A is preferred; otherwise first encountered",
        "altloc_tie_examples": altloc_ties[:20],
        "altloc_tie_count": len(altloc_ties),
    }
    return list(selected.values()), qa


def independent_contact_residue_audit(cif_text: str, cutoff_a: float) -> dict[str, Any]:
    atoms, _ = selected_atoms(cif_text)
    mapping = chain_uniprot_mapping(cif_text, CRBN_ACCESSION, CRBN_CHAIN)
    ligand_atoms = [
        atom for atom in reversed(atoms)
        if atom.get("group_PDB") == "HETATM"
        and atom.get("label_comp_id") == ALLOSTERIC_LIGAND
        and atom.get("auth_asym_id") == CRBN_CHAIN
    ]
    protein_atoms = [
        atom for atom in reversed(atoms)
        if atom.get("group_PDB") == "ATOM"
        and atom.get("auth_asym_id") == CRBN_CHAIN
        and str(atom.get("auth_seq_id", "")).lstrip("-").isdigit()
    ]
    contact_residues: set[int] = set()
    for protein in protein_atoms:
        px, py, pz = (float(value) for value in protein["xyz_array"])
        auth_residue = int(protein["auth_seq_id"])
        uniprot_residue = auth_residue + int(mapping["auth_to_uniprot_offset"])
        for ligand in ligand_atoms:
            lx, ly, lz = (float(value) for value in ligand["xyz_array"])
            distance = math.sqrt((px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2)
            if distance <= cutoff_a:
                contact_residues.add(uniprot_residue)
                break
    return {
        "method": "independent math.sqrt pairwise distances over reversed selected atom order",
        "contact_residue_count": len(contact_residues),
        "contact_residues": sorted(contact_residues),
    }


def structure_contacts(cif_text: str, cutoff_a: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    atoms, qa = selected_atoms(cif_text)
    mapping = chain_uniprot_mapping(cif_text, CRBN_ACCESSION, CRBN_CHAIN)
    ligand_atoms = [
        atom for atom in atoms
        if atom.get("group_PDB") == "HETATM"
        and atom.get("label_comp_id") == ALLOSTERIC_LIGAND
        and atom.get("auth_asym_id") == CRBN_CHAIN
    ]
    protein_atoms = [
        atom for atom in atoms
        if atom.get("group_PDB") == "ATOM"
        and atom.get("auth_asym_id") == CRBN_CHAIN
        and str(atom.get("auth_seq_id", "")).lstrip("-").isdigit()
    ]
    if not ligand_atoms:
        raise ValueError(f"{PDB_ID}: no {ALLOSTERIC_LIGAND} heavy atoms found on chain {CRBN_CHAIN}")
    if not protein_atoms:
        raise ValueError(f"{PDB_ID}: no CRBN protein atoms found on chain {CRBN_CHAIN}")
    ligand_xyz = np.vstack([atom["xyz_array"] for atom in ligand_atoms])
    per_residue: dict[int, dict[str, Any]] = {}
    atom_contacts: list[dict[str, Any]] = []
    for atom in protein_atoms:
        auth_residue = int(atom["auth_seq_id"])
        uniprot_residue = auth_residue + int(mapping["auth_to_uniprot_offset"])
        distances = np.sqrt(np.sum((ligand_xyz - atom["xyz_array"]) ** 2, axis=1))
        idx = int(np.argmin(distances))
        min_dist = float(distances[idx])
        entry = per_residue.setdefault(
            uniprot_residue,
            {
                "pdb": PDB_ID,
                "chain": CRBN_CHAIN,
                "auth_residue": auth_residue,
                "uniprot_residue": uniprot_residue,
                "residue_name": atom.get("auth_comp_id") or atom.get("label_comp_id"),
                "min_distance_A": min_dist,
                "nearest_protein_atom": atom.get("auth_atom_id") or atom.get("label_atom_id"),
                "nearest_ligand_atom": ligand_atoms[idx].get("auth_atom_id") or ligand_atoms[idx].get("label_atom_id"),
                "within_4p5A": min_dist <= cutoff_a,
                "protein_atom_contacts_within_cutoff": 0,
            },
        )
        if min_dist < float(entry["min_distance_A"]):
            entry.update(
                {
                    "min_distance_A": min_dist,
                    "nearest_protein_atom": atom.get("auth_atom_id") or atom.get("label_atom_id"),
                    "nearest_ligand_atom": ligand_atoms[idx].get("auth_atom_id") or ligand_atoms[idx].get("label_atom_id"),
                    "within_4p5A": min_dist <= cutoff_a,
                }
            )
        if min_dist <= cutoff_a:
            entry["protein_atom_contacts_within_cutoff"] = int(entry["protein_atom_contacts_within_cutoff"]) + 1
            atom_contacts.append(
                {
                    "pdb": PDB_ID,
                    "ligand": ALLOSTERIC_LIGAND,
                    "chain": CRBN_CHAIN,
                    "auth_residue": auth_residue,
                    "uniprot_residue": uniprot_residue,
                    "residue_name": atom.get("auth_comp_id") or atom.get("label_comp_id"),
                    "protein_atom": atom.get("auth_atom_id") or atom.get("label_atom_id"),
                    "ligand_atom": ligand_atoms[idx].get("auth_atom_id") or ligand_atoms[idx].get("label_atom_id"),
                    "distance_A": min_dist,
                }
            )
    residue_contacts = [row for row in per_residue.values() if row["within_4p5A"]]
    residue_contacts.sort(key=lambda row: (row["uniprot_residue"], row["min_distance_A"]))
    atom_contacts.sort(key=lambda row: (row["uniprot_residue"], row["distance_A"], row["protein_atom"]))
    qa.update(
        {
            "crbn_mapping": mapping,
            "ligand_heavy_atoms": len(ligand_atoms),
            "ligand_auth_chains": sorted({str(atom.get("auth_asym_id", "")) for atom in atoms if atom.get("label_comp_id") == ALLOSTERIC_LIGAND}),
            "protein_heavy_atoms_chain_B": len(protein_atoms),
            "contact_cutoff_A": cutoff_a,
            "residue_contacts_within_cutoff": len(residue_contacts),
            "allosteric_ligand_selection": f"{ALLOSTERIC_LIGAND} HETATM on auth chain {CRBN_CHAIN}",
        }
    )
    return atom_contacts, residue_contacts, qa


def load_candidate_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    robustness_path = path.with_name("legacy_robustness.csv")
    if robustness_path.is_file():
        with robustness_path.open(newline="", encoding="utf-8") as handle:
            robustness = {
                (row.get("residue", ""), row.get("contact_class", "")): row
                for row in csv.DictReader(handle)
            }
        merged: list[dict[str, str]] = []
        for row in rows:
            key = (row.get("residue", ""), row.get("contact_class", ""))
            extra = robustness.get(key, {})
            merged.append({**extra, **row})
        rows = merged
    return rows


def resolve_candidate_path(output_dir: Path, cfg: Mapping[str, Any]) -> Path:
    """Resolve the frozen candidate universe from staged inputs or a supplied path."""
    explicit = cfg.get("external", {}).get("candidate_robustness_csv")
    for staged in (
        output_dir / "data" / "directional_reference_inputs" / "candidate_universe.csv",
        ROOT_REFERENCE_INPUTS / "candidate_universe.csv",
        DEFAULT_STAGED_CANDIDATES,
    ):
        if staged.is_file():
            return staged
    if explicit:
        return Path(explicit)
    return DEFAULT_CANDIDATES


def candidate_overlap_rows(
    candidate_rows: Sequence[Mapping[str, str]],
    residue_contacts: Sequence[Mapping[str, Any]],
    cif_text: str,
) -> list[dict[str, Any]]:
    contact_by_residue = {int(row["uniprot_residue"]): row for row in residue_contacts}
    all_atoms, _ = selected_atoms(cif_text)
    ligand_atoms = [
        atom for atom in all_atoms
        if atom.get("group_PDB") == "HETATM"
        and atom.get("label_comp_id") == ALLOSTERIC_LIGAND
        and atom.get("auth_asym_id") == CRBN_CHAIN
    ]
    protein_atoms_by_residue: dict[int, list[dict[str, Any]]] = {}
    mapping = chain_uniprot_mapping(cif_text, CRBN_ACCESSION, CRBN_CHAIN)
    for atom in all_atoms:
        if atom.get("group_PDB") != "ATOM" or atom.get("auth_asym_id") != CRBN_CHAIN:
            continue
        seq = str(atom.get("auth_seq_id", ""))
        if not seq.lstrip("-").isdigit():
            continue
        protein_atoms_by_residue.setdefault(int(seq) + int(mapping["auth_to_uniprot_offset"]), []).append(atom)
    ligand_xyz = np.vstack([atom["xyz_array"] for atom in ligand_atoms])
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        residue = int(row["residue"])
        atoms = protein_atoms_by_residue.get(residue, [])
        min_distance = ""
        nearest_atom = ""
        nearest_lig = ""
        if atoms:
            best = (float("inf"), "", "")
            for atom in atoms:
                distances = np.sqrt(np.sum((ligand_xyz - atom["xyz_array"]) ** 2, axis=1))
                idx = int(np.argmin(distances))
                candidate = (
                    float(distances[idx]),
                    atom.get("auth_atom_id") or atom.get("label_atom_id", ""),
                    ligand_atoms[idx].get("auth_atom_id") or ligand_atoms[idx].get("label_atom_id", ""),
                )
                if candidate[0] < best[0]:
                    best = candidate
            min_distance = best[0]
            nearest_atom = best[1]
            nearest_lig = best[2]
        rows.append(
            {
                "residue": residue,
                "contact_class": row.get("contact_class", ""),
                "discovery_rank": row.get("discovery_rank", ""),
                "discovery_D_g": row.get("discovery_D_g", ""),
                "discovery_top5": row.get("discovery_top5", ""),
                "stable_apo_model_candidate": row.get("stable_apo_model_candidate", ""),
                "also_consistent_in_engineered_references": row.get("also_consistent_in_engineered_references", ""),
                "observed_in_9SFM_chain_B": bool(atoms),
                "min_distance_to_A1CEG_A": min_distance,
                "nearest_protein_atom": nearest_atom,
                "nearest_ligand_atom": nearest_lig,
                "same_residue_as_A1CEG_contact": residue in contact_by_residue,
                "comparison_role": "spatial overlap only; not functional validation",
            }
        )
    rows.sort(key=lambda item: (not bool(item["same_residue_as_A1CEG_contact"]), abs(float(item["discovery_D_g"] or 0.0)) * -1, item["residue"]))
    return rows


def contact_candidate_rows(
    residue_contacts: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    by_residue: dict[int, list[Mapping[str, str]]] = {}
    for candidate in candidate_rows:
        by_residue.setdefault(int(candidate["residue"]), []).append(candidate)
    rows: list[dict[str, Any]] = []
    for contact in residue_contacts:
        residue = int(contact["uniprot_residue"])
        candidates = by_residue.get(residue, [])
        rows.append(
            {
                **contact,
                "candidate_classes": ";".join(sorted({row.get("contact_class", "") for row in candidates if row.get("contact_class")})),
                "candidate_count_at_residue": len(candidates),
                "any_stable_apo_candidate": any(row.get("stable_apo_model_candidate") == "True" for row in candidates),
                "any_engineered_consistent_candidate": any(row.get("also_consistent_in_engineered_references") == "True" for row in candidates),
                "comparison_role": "all allosteric-ligand contact residues, including residues outside the 269-position analysis window",
            }
        )
    return rows


BLOOD_VARIANTS = [
    ("D50H", "D50H", [50], "outside_269_window", "N-terminal belt / pre-Lon", "no effect on CRBN function", "not_variant_resolved", "not separately measured", "no degradation/cell-response defect reported in the extracted functional class"),
    ("A143V", "A143V", [143], "inside_269_window", "Lon / NTD", "no effect", "not_variant_resolved", "not separately measured", "no degradation/cell-response defect reported in the extracted functional class"),
    ("L190F", "L190F", [190], "inside_269_window", "HB, CRBN-DDB1 interface", "no effect", "not_variant_resolved", "not separately measured", "no degradation/cell-response defect reported despite interface location"),
    ("R283K", "R283K", [283], "inside_269_window", "HB", "no effect", "not_variant_resolved", "not separately measured", "no degradation/cell-response defect reported in the extracted functional class"),
    ("C326G", "C326G", [326], "inside_269_window", "TBD, Zn-finger ligand", "agent-dependent: IMiD-resistant, CELMoD-rescuable", "not_variant_resolved", "abundance/folding not separated from cellular function here", "degradation/cell response depends on modulator class"),
    ("A347V", "A347V", [347], "inside_269_window", "TBD, buried/sensor-loop-adjacent", "tolerated", "not_variant_resolved", "not separately measured", "no degradation/cell-response defect reported in the extracted functional class"),
    ("P352S", "P352S", [352], "inside_269_window", "TBD, direct IMiD contact", "agent-dependent: Len/Pom/Iber inactive", "not_variant_resolved", "abundance/folding not separated from cellular function here", "degradation/cell response depends on modulator class"),
    ("C366Y", "C366Y", [366], "inside_269_window", "TBD, sensor-loop region", "agent-dependent partial effect", "not_variant_resolved", "abundance/folding not separated from cellular function here", "partial degradation/cell-response effect reported"),
    ("F381S", "F381S", [381], "inside_269_window", "TBD core adjacent to 3-Trp pocket", "agent-dependent", "not_variant_resolved", "reported effect may include stability/abundance; not converted into a separate mechanical test", "degradation/cell response depends on modulator class"),
    ("W386A", "W386A", [386], "inside_269_window", "TBD, 3-Trp pocket", "complete loss of function positive control", "not_variant_resolved", "not separately measured", "loss of degradation/cell response across tested modulators"),
    ("H397Y", "H397Y", [397], "inside_269_window", "TBD, scaffolds W400", "complete loss of function for IMiDs and CELMoDs", "not_variant_resolved", "not separately measured", "loss of degradation/cell response across tested modulators"),
    ("W415G_experiment", "W415G", [415], "inside_269_window", "TBD, buried", "tolerated", "not_variant_resolved", "not separately measured", "no degradation/cell-response defect reported for W415G"),
]


COMMON_BLOOD_SOURCE = "Chrisochoidou et al. Blood 2025 DOI 10.1182/blood.2024025861; PMID 39841463"
BLOOD_EVIDENCE = {
    "D50H": ("supplemental PDF", 8, 2636, "Supplementary Table 3", "Behaviour similar to wild type CRBN", "qualitative_from_table_and_figures"),
    "A143V": ("supplemental PDF", 8, 2636, "Supplementary Table 3", "Behaviour similar to wild type CRBN", "qualitative_from_table_and_figures"),
    "L190F": ("supplemental PDF", 8, 2636, "Supplementary Table 3", "Behaviour similar to wild type CRBN", "qualitative_from_table_and_figures"),
    "R283K": ("supplemental PDF", 8, 2636, "Supplementary Table 3", "Behaviour similar to wild type CRBN", "qualitative_from_table_and_figures"),
    "C326G": ("supplemental PDF", 8, 2636, "Supplementary Table 3", "Partial CRBN function observed", "qualitative_from_table_and_figures"),
    "A347V": ("supplemental PDF", 8, 2636, "Supplementary Table 3", "Behaviour similar to wild type CRBN", "qualitative_from_table_and_figures"),
    "P352S": ("supplemental PDF", 8, 2636, "Supplementary Table 3", "Partial CRBN function observed", "qualitative_from_table_and_figures"),
    "C366Y": ("supplemental PDF", 8, 2638, "Supplementary Table 3", "Partial CRBN function observed", "qualitative_from_table_and_figures"),
    "F381S": ("supplemental PDF", 8, 2636, "Supplementary Table 3", "Partial CRBN function observed", "qualitative_from_table_and_figures"),
    "W386A": ("supplemental PDF", 9, 2636, "Supplementary Table 3", "Behaviour similar to EV", "qualitative_from_table_and_figures"),
    "H397Y": ("supplemental PDF", 9, 2636, "Supplementary Table 3", "Behaviour similar to EV", "qualitative_from_table_and_figures"),
    "W415G": ("supplemental PDF", 9, 2636, "Supplementary Table 3", "Behaviour similar to wild type CRBN", "qualitative_from_table_and_figures"),
}


def blood_variant_rows(candidate_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    candidates_by_residue: dict[int, list[Mapping[str, str]]] = {}
    stable = 0
    for row in candidate_rows:
        residue = int(row["residue"])
        candidates_by_residue.setdefault(residue, []).append(row)
        if row.get("stable_apo_model_candidate") == "True":
            stable += 1
    rows: list[dict[str, Any]] = []
    for variant_id, mutation, residues, window_status, location, functional_effect, binding, abundance, degradation in BLOOD_VARIANTS:
        overlapping = [row for residue in residues for row in candidates_by_residue.get(residue, [])]
        evidence = BLOOD_EVIDENCE[mutation]
        rows.append(
            {
                "variant_id": variant_id,
                "reported_mutation": mutation,
                "patient_reported_symbol": "W415X" if variant_id == "W415G_experiment" else "",
                "experimentally_tested_symbol": mutation,
                "residues": ";".join(str(residue) for residue in residues),
                "primary_269_window": window_status,
                "structural_location": location,
                "binding_endpoint": binding,
                "abundance_or_folding_endpoint": abundance,
                "degradation_endpoint": degradation,
                "cell_response_endpoint": functional_effect,
                "candidate_overlap": "same_residue" if overlapping else "none",
                "candidate_contact_classes": ";".join(sorted({row.get("contact_class", "") for row in overlapping if row.get("contact_class")})),
                "stable_apo_candidate_overlap": any(row.get("stable_apo_model_candidate") == "True" for row in overlapping),
                "functional_endpoint_type": evidence[5],
                "binding_endpoint_type": "not_variant_resolved_in_retrieved_main_pdf",
                "abundance_endpoint_type": "not_separated_from_functional_assays" if "not separately" not in abundance else "not_measured_as_separate_endpoint",
                "degradation_endpoint_type": evidence[5],
                "evidence_file": evidence[0],
                "evidence_page": evidence[1],
                "evidence_article_page": evidence[2],
                "evidence_table_or_figure": evidence[3],
                "evidence_quote": evidence[4],
                "source": COMMON_BLOOD_SOURCE,
                "source_location": "row-level page/figure evidence is recorded in evidence_* fields; no quantitative variant table was acquired",
                "comparison_role": "retrospective case comparison; not used to select candidate groups",
            }
        )
    return rows


def oconnor_reuse_rows(inventory_source: Path, candidate_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    candidates = {int(row["residue"]): row for row in candidate_rows}
    rows: list[dict[str, Any]] = []
    inventory = inventory_source / "oconnor_variant_inventory.csv" if inventory_source.is_dir() else inventory_source
    if not inventory.is_file():
        return rows
    with inventory.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            residues = [int(value) for value in str(row.get("residues", "")).split(";") if value]
            overlap = [candidates[residue] for residue in residues if residue in candidates]
            rows.append(
                {
                    "variant": row.get("variant", ""),
                    "residues": row.get("residues", ""),
                    "primary_269_window": row.get("primary_269_window", ""),
                    "source_tables": row.get("source_tables", ""),
                    "candidate_overlap": "same_residue" if overlap else "none",
                    "candidate_contact_classes": ";".join(sorted({item.get("contact_class", "") for item in overlap if item.get("contact_class")})),
                    "stable_apo_candidate_overlap": any(item.get("stable_apo_model_candidate") == "True" for item in overlap),
                    "interpretation": "reused 118 inventory; no exact 142-candidate overlap if candidate_overlap is none",
                    "source": f"O'Connor preprint DOI {OCONNOR_DOI}; 118 external inventory",
                }
            )
    return rows


def extract_pdf_text(pdf_path: Path, text_path: Path) -> str:
    tool = shutil.which("pdftotext")
    if not pdf_path.is_file():
        return "pdf_missing"
    if tool is None:
        return "pdftotext_unavailable"
    try:
        subprocess.run([tool, str(pdf_path), str(text_path)], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "pdftotext_failed"
    return "extracted" if text_path.is_file() else "no_output"


def rcsb_version_from_entry(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    info = entry.get("rcsb_accession_info", {})
    fields = [
        ("initial_release_date", info.get("initial_release_date")),
        ("revision_date", info.get("revision_date")),
        ("deposit_date", info.get("deposit_date")),
        ("major_revision", info.get("major_revision")),
        ("minor_revision", info.get("minor_revision")),
    ]
    return ";".join(f"{key}={value}" for key, value in fields if value not in (None, ""))


def report_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Directional-mechanics external evidence",
        "",
        f"Generated UTC: {summary['generated_utc']}",
        "",
        "## 9SFM allosteric-ligand contact mapping",
        "",
        (
            f"The official RCSB 9SFM mmCIF was parsed on auth chain {CRBN_CHAIN}. "
            f"The {ALLOSTERIC_LIGAND} heavy-atom contact cutoff was {ALLOSTERIC_CUTOFF_A} A. "
            f"{summary['allosteric_contact_residue_count']} CRBN residues contacted the ligand at this cutoff."
        ),
        (
            "An independent pairwise distance audit over reversed atom order recovered "
            f"{summary['allosteric_contact_independent_audit_count']} contact residues "
            f"(match={summary['allosteric_contact_independent_audit_matches']})."
        ),
        (
            f"{summary['candidate_same_residue_contact_count']} of {summary['candidate_count']} frozen candidate "
            "residue-contact groups share a residue with those ligand contacts. This is a spatial correspondence only."
        ),
        "",
        "## Blood 2025 mutation panel",
        "",
        (
            "All 12 experimentally tested variants were retained, with W415X kept as the patient-reported symbol and "
            "W415G as the experimentally tested construct. Variant-level binding was not treated as measured unless a "
            "separate source table was available."
        ),
        (
            "Each variant row records the evidence file, page, figure/table location, a short evidence phrase and "
            "whether the endpoint was directly measured, qualitative, or unavailable in the retrieved files."
        ),
        (
            f"{summary['blood_variant_same_residue_candidate_count']} variants overlapped the frozen candidate universe "
            f"by residue; {summary['blood_variant_stable_candidate_count']} overlapped the stable apo-model set."
        ),
        "",
        "## O'Connor inventory reuse",
        "",
        (
            "The O'Connor mutant inventory was copied from the 118 verified external table and re-compared to the "
            "candidate universe. Rows with no exact residue overlap remain unevaluable for candidate validation."
        ),
        "",
        "## Source boundaries",
        "",
        "- RCSB 9SFM: official mmCIF and entry metadata; atom distances recomputed here.",
        "- Chrisochoidou et al. Blood 2025: DOI 10.1182/blood.2024025861, PMID 39841463; PDF cached when available.",
        f"- Chrisochoidou PMC article HTML: retrieval status was {summary['blood_pmc_article_status']}; it lists the supplemental PDF.",
        f"- Europe PMC supplementaryFiles ZIP: retrieval status was {summary['blood_europepmc_supplement_zip_status']}; supplemental PDF extraction status was {summary['blood_supplement_status']}.",
        f"- Chrisochoidou supplemental PDF text extraction status was {summary['blood_supplement_pdf_text_status']}.",
        f"- PMC direct supplemental PDF endpoint status was {summary['blood_pmc_direct_supplement_status']}.",
        f"- NCBI PMC OA API route status was {summary['blood_ncbi_oa_status']}; no OA FTP package link was available from that route in this run.",
        "- O'Connor et al. 2025: DOI 10.64898/2025.12.19.695617; 118 verified inventory reused without changing its criteria.",
        "- No W264 functional validation is claimed in these outputs.",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: Path | str = DEFAULT_OUTPUT, offline: bool = False, config: Path | str | None = DEFAULT_CONFIG) -> dict[str, Any]:
    output = Path(output_dir)
    data_dir = output / "data" / "external"
    analysis_dir = output / "analysis" / "external"
    verification_dir = output / "verification" / "external"
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    verification_dir.mkdir(parents=True, exist_ok=True)
    previous_sources = load_previous_sources(analysis_dir / "source_registry.json") if offline else {}
    cfg: dict[str, Any] = {}
    if config is not None and Path(config).is_file():
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    cutoff = float(cfg.get("external", {}).get("heavy_atom_contact_cutoff_A", ALLOSTERIC_CUTOFF_A))
    ligand = str(cfg.get("external", {}).get("allosteric_ligand", ALLOSTERIC_LIGAND))
    if ligand != ALLOSTERIC_LIGAND:
        raise ValueError(f"this implementation is fixed to {ALLOSTERIC_LIGAND}, got {ligand}")

    source_records: list[SourceRecord] = []
    cif_record = acquire(
        RCSB_CIF_URL,
        data_dir / f"{PDB_ID}.cif.gz",
        "rcsb_9sfm_mmcif_gz",
        offline,
        previous_sources.get("rcsb_9sfm_mmcif_gz"),
    )
    source_records.append(cif_record)
    entry_record = acquire(
        RCSB_ENTRY_URL,
        data_dir / f"{PDB_ID}_rcsb_entry.json",
        "rcsb_9sfm_entry_json",
        offline,
        previous_sources.get("rcsb_9sfm_entry_json"),
    )
    if entry_record.status.startswith("http_") or entry_record.status.startswith("cached"):
        entry_record = SourceRecord(
            entry_record.source_id,
            entry_record.url,
            entry_record.local_path,
            entry_record.retrieved_utc,
            entry_record.status,
            entry_record.bytes,
            entry_record.sha256,
            version=rcsb_version_from_entry(Path(entry_record.local_path)),
            error=entry_record.error,
            current_run_status=entry_record.current_run_status,
        )
    source_records.append(entry_record)
    source_records.append(
        acquire(
            BLOOD_PDF_URL,
            data_dir / "Chrisochoidou_2025_Blood.pdf",
            "blood_2025_pdf",
            offline,
            previous_sources.get("blood_2025_pdf"),
        )
    )
    source_records.append(
        acquire(
            BLOOD_PMC_ARTICLE_URL,
            verification_dir / "PMC12824623_article.html",
            "blood_2025_pmc_article_html",
            offline,
            previous_sources.get("blood_2025_pmc_article_html"),
        )
    )
    europepmc_zip_record = acquire(
        BLOOD_EUROPEPMC_SUPPLEMENT_ZIP_URL,
        data_dir / "PMC12824623_supplementaryFiles.zip",
        "blood_2025_europepmc_supplement_zip",
        offline,
        previous_sources.get("blood_2025_europepmc_supplement_zip"),
    )
    source_records.append(europepmc_zip_record)
    supplement_pdf_record = extract_pdf_from_zip(
        Path(europepmc_zip_record.local_path),
        BLOOD_SUPPLEMENT_MEMBER,
        data_dir / BLOOD_SUPPLEMENT_MEMBER,
        "blood_2025_supplement_pdf",
        previous_sources.get("blood_2025_supplement_pdf"),
    )
    source_records.append(supplement_pdf_record)
    source_records.append(
        acquire_pdf_or_response(
            BLOOD_SUPPLEMENT_URL,
            data_dir / "BLOOD_BLD-2024-025861-mmc1_direct.pdf",
            verification_dir / "BLOOD_BLD-2024-025861-mmc1_download_response.html",
            "blood_2025_pmc_direct_supplement_pdf",
            offline,
            previous_sources.get("blood_2025_pmc_direct_supplement_pdf"),
        )
    )
    source_records.append(
        acquire(
            BLOOD_NCBI_OA_URL,
            verification_dir / "NCBI_PMC_OA_PMC12824623_response.xml",
            "blood_2025_ncbi_oa_api",
            offline,
            previous_sources.get("blood_2025_ncbi_oa_api"),
        )
    )
    oconnor_record = copy_cached(
        DEFAULT_OConnor_118 / "oconnor_variant_inventory.csv",
        data_dir / "OConnor_118_variant_inventory.csv",
        "oconnor_118_variant_inventory",
        previous_sources.get("oconnor_118_variant_inventory"),
    )
    source_records.append(oconnor_record)

    write_json(analysis_dir / "source_registry.json", [record.__dict__ for record in source_records])
    write_csv(
        analysis_dir / "source_registry.csv",
        [record.__dict__ for record in source_records],
        ["source_id", "url", "local_path", "retrieved_utc", "status", "bytes", "sha256", "version", "error", "current_run_status"],
    )

    if not Path(cif_record.local_path).is_file():
        raise FileNotFoundError(f"9SFM mmCIF unavailable: {cif_record.local_path}")
    cif_text = load_cif_text(Path(cif_record.local_path))
    atom_contacts, residue_contacts, qa = structure_contacts(cif_text, cutoff)
    independent_audit = independent_contact_residue_audit(cif_text, cutoff)
    qa["independent_distance_audit"] = independent_audit
    qa["independent_distance_audit_matches_primary"] = independent_audit["contact_residues"] == [int(row["uniprot_residue"]) for row in residue_contacts]

    candidate_path = resolve_candidate_path(output, cfg)
    candidate_rows = load_candidate_rows(candidate_path)
    candidate_overlap = candidate_overlap_rows(candidate_rows, residue_contacts, cif_text)
    contact_candidate = contact_candidate_rows(residue_contacts, candidate_rows)
    blood_rows = blood_variant_rows(candidate_rows)
    oconnor_rows = oconnor_reuse_rows(Path(oconnor_record.local_path), candidate_rows)

    write_csv(
        analysis_dir / "9sfm_a1ceg_atom_contacts.csv",
        atom_contacts,
        ["pdb", "ligand", "chain", "auth_residue", "uniprot_residue", "residue_name", "protein_atom", "ligand_atom", "distance_A"],
    )
    write_csv(
        analysis_dir / "9sfm_a1ceg_residue_contacts.csv",
        residue_contacts,
        [
            "pdb",
            "chain",
            "auth_residue",
            "uniprot_residue",
            "residue_name",
            "min_distance_A",
            "nearest_protein_atom",
            "nearest_ligand_atom",
            "within_4p5A",
            "protein_atom_contacts_within_cutoff",
        ],
    )
    write_csv(
        analysis_dir / "9sfm_a1ceg_contact_candidate_overlap.csv",
        contact_candidate,
        [
            "pdb",
            "chain",
            "auth_residue",
            "uniprot_residue",
            "residue_name",
            "min_distance_A",
            "nearest_protein_atom",
            "nearest_ligand_atom",
            "within_4p5A",
            "protein_atom_contacts_within_cutoff",
            "candidate_classes",
            "candidate_count_at_residue",
            "any_stable_apo_candidate",
            "any_engineered_consistent_candidate",
            "comparison_role",
        ],
    )
    write_csv(
        analysis_dir / "candidate_9sfm_spatial_correspondence.csv",
        candidate_overlap,
        [
            "residue",
            "contact_class",
            "discovery_rank",
            "discovery_D_g",
            "discovery_top5",
            "stable_apo_model_candidate",
            "also_consistent_in_engineered_references",
            "observed_in_9SFM_chain_B",
            "min_distance_to_A1CEG_A",
            "nearest_protein_atom",
            "nearest_ligand_atom",
            "same_residue_as_A1CEG_contact",
            "comparison_role",
        ],
    )
    write_csv(
        analysis_dir / "blood_2025_variant_observations.csv",
        blood_rows,
        [
            "variant_id",
            "reported_mutation",
            "patient_reported_symbol",
            "experimentally_tested_symbol",
            "residues",
            "primary_269_window",
            "structural_location",
            "binding_endpoint",
            "abundance_or_folding_endpoint",
            "degradation_endpoint",
            "cell_response_endpoint",
            "candidate_overlap",
            "candidate_contact_classes",
            "stable_apo_candidate_overlap",
            "functional_endpoint_type",
            "binding_endpoint_type",
            "abundance_endpoint_type",
            "degradation_endpoint_type",
            "evidence_file",
            "evidence_page",
            "evidence_article_page",
            "evidence_table_or_figure",
            "evidence_quote",
            "source",
            "source_location",
            "comparison_role",
        ],
    )
    write_csv(
        analysis_dir / "oconnor_2025_variant_candidate_reuse.csv",
        oconnor_rows,
        ["variant", "residues", "primary_269_window", "source_tables", "candidate_overlap", "candidate_contact_classes", "stable_apo_candidate_overlap", "interpretation", "source"],
    )
    pdf_text_status = extract_pdf_text(data_dir / "Chrisochoidou_2025_Blood.pdf", verification_dir / "Chrisochoidou_2025_Blood.txt")
    supplement_pdf_text_status = extract_pdf_text(data_dir / BLOOD_SUPPLEMENT_MEMBER, verification_dir / "BLOOD_BLD-2024-025861-mmc1.txt")
    write_json(analysis_dir / "9sfm_atom_selection_qa.json", qa)

    same_contact = [row for row in candidate_overlap if row["same_residue_as_A1CEG_contact"]]
    blood_overlap = [row for row in blood_rows if row["candidate_overlap"] == "same_residue"]
    blood_stable = [row for row in blood_rows if row["stable_apo_candidate_overlap"]]
    summary = {
        "generated_utc": utc_now(),
        "offline": offline,
        "config": str(config) if config is not None else "",
        "candidate_csv": str(candidate_path),
        "candidate_count": len(candidate_rows),
        "stable_apo_candidate_count": sum(1 for row in candidate_rows if row.get("stable_apo_model_candidate") == "True"),
        "engineered_consistent_candidate_count": sum(1 for row in candidate_rows if row.get("also_consistent_in_engineered_references") == "True"),
        "allosteric_structure": PDB_ID,
        "allosteric_structure_url": RCSB_STRUCTURE_URL,
        "allosteric_ligand": ALLOSTERIC_LIGAND,
        "allosteric_contact_cutoff_A": cutoff,
        "allosteric_contact_residue_count": len(residue_contacts),
        "allosteric_contact_independent_audit_count": independent_audit["contact_residue_count"],
        "allosteric_contact_independent_audit_matches": qa["independent_distance_audit_matches_primary"],
        "candidate_same_residue_contact_count": len(same_contact),
        "candidate_same_residue_contact_residues": sorted({int(row["residue"]) for row in same_contact}),
        "blood_variant_count": len(blood_rows),
        "blood_variant_same_residue_candidate_count": len(blood_overlap),
        "blood_variant_same_residue_candidate_variants": [row["reported_mutation"] for row in blood_overlap],
        "blood_variant_stable_candidate_count": len(blood_stable),
        "oconnor_variant_rows": len(oconnor_rows),
        "oconnor_exact_candidate_overlap_count": sum(1 for row in oconnor_rows if row["candidate_overlap"] == "same_residue"),
        "blood_pdf_text_status": pdf_text_status,
        "blood_supplement_pdf_text_status": supplement_pdf_text_status,
        "blood_pmc_article_status": next(record.status for record in source_records if record.source_id == "blood_2025_pmc_article_html"),
        "blood_europepmc_supplement_zip_status": next(record.status for record in source_records if record.source_id == "blood_2025_europepmc_supplement_zip"),
        "blood_supplement_status": next(record.status for record in source_records if record.source_id == "blood_2025_supplement_pdf"),
        "blood_pmc_direct_supplement_status": next(record.status for record in source_records if record.source_id == "blood_2025_pmc_direct_supplement_pdf"),
        "blood_ncbi_oa_status": next(record.status for record in source_records if record.source_id == "blood_2025_ncbi_oa_api"),
        "no_w264_functional_validation_claim": True,
    }
    write_json(analysis_dir / "external_directional_summary.json", summary)
    (analysis_dir / "external_directional_report.md").write_text(report_text(summary), encoding="utf-8")
    verification_payload = {
        "script": "scripts/directional_external.py",
        "script_sha256": sha256_file(Path(__file__)),
        "outputs": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(analysis_dir.glob("*"))
            if path.is_file()
        },
    }
    write_json(verification_dir / "output_hashes.json", verification_payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.output_dir, offline=args.offline, config=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
