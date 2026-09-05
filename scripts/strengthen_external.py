#!/usr/bin/env python3
"""Acquire and refit external CRBN SAXS and functional evidence.

The script is intentionally conservative: it records source availability and
within-construct measurements, but it does not compare external observations
against discovery candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "strengthening"
DEFAULT_DATA_DIR = DEFAULT_OUTPUT / "data" / "external"
DEFAULT_ANALYSIS_DIR = DEFAULT_OUTPUT / "analysis" / "external"

KROUPOVA_ARTICLE = "https://www.nature.com/articles/s41467-024-52871-9"
KROUPOVA_SOURCE_XLSX = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41467-024-52871-9/MediaObjects/"
    "41467_2024_52871_MOESM6_ESM.xlsx"
)
KROUPOVA_SUPP_ZIP = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41467-024-52871-9/MediaObjects/"
    "41467_2024_52871_MOESM4_ESM.zip"
)
SASBDB_PROJECT = "https://www.sasbdb.org/project/2221/"
SASBDB_HELP = "https://www.sasbdb.org/help/"

OCONNOR_PMC = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12767645/"
OCONNOR_SUPPLEMENT = "https://pmc.ncbi.nlm.nih.gov/articles/instance/12767645/bin/media-1.pdf"
OCONNOR_BIORXIV_HTML = "https://www.biorxiv.org/content/10.64898/2025.12.19.695617v1"
OCONNOR_BIORXIV_PDF = "https://www.biorxiv.org/content/10.64898/2025.12.19.695617v1.full.pdf"
OCONNOR_BIORXIV_SUPPLEMENT = (
    "https://www.biorxiv.org/content/biorxiv/early/2025/12/22/"
    "2025.12.19.695617/DC1/embed/media-1.pdf?download=true"
)
OCONNOR_DOI = "10.64898/2025.12.19.695617"
OCONNOR_TITLE = "Tuning the open-close equilibrium of Cereblon with small molecules influences protein degradation"
OCONNOR_PUBMED_ID = "41497597"
OCONNOR_PUBMED_ESUMMARY = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
    f"db=pubmed&retmode=json&id={OCONNOR_PUBMED_ID}"
)
OCONNOR_CROSSREF = f"https://api.crossref.org/works/{OCONNOR_DOI}"
OCONNOR_EUROPEPMC = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
    f"query={urllib.parse.quote('DOI:' + OCONNOR_DOI)}&format=json"
)
OCONNOR_BIORXIV_SUPP_TAB = f"{OCONNOR_BIORXIV_HTML}.supplementary-material"

SASBDB_ACCESSIONS = {
    "SASDU52": {"condition": "apo", "published_rg_nm": 2.7},
    "SASDU62": {"condition": "mezigdomide", "published_rg_nm": 2.3},
    "SASDU72": {"condition": "pomalidomide", "published_rg_nm": 2.3},
    "SASDU82": {"condition": "iberdomide", "published_rg_nm": 2.4},
    "SASDU92": {"condition": "lenalidomide", "published_rg_nm": 2.2},
}

KROUPOVA_PDB_IDS = ("8RQ1", "8RQ8", "8RQ9", "8RQA", "8RQC", "9GAO")
OCONNOR_PDB_IDS = ("9SUN", "9SVG", "9SVH", "9SVI")
OCONNOR_EMDB_IDS = (
    "EMD-55257",
    "EMD-54624",
    "EMD-54644",
    "EMD-55258",
    "EMD-55057",
    "EMD-55058",
    "EMD-55259",
    "EMD-55059",
    "EMD-55060",
)

PRIMARY_WINDOW_PATH = ROOT / "data" / "crbn_residue_window.csv"


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    url: str
    local_path: str | None
    retrieved_utc: str
    status: str
    bytes: int | None = None
    sha256: str | None = None
    error: str | None = None
    current_run_status: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_url(url: str, *, timeout: int = 45) -> tuple[int | None, bytes, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "crbn-softmode-external-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            return status, response.read(), None
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        return exc.code, body, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, b"", str(exc)


def load_previous_source_manifest(path: Path) -> dict[str, SourceRecord]:
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    records: dict[str, SourceRecord] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            source_id = str(row["source_id"])
            records[source_id] = SourceRecord(
                source_id=source_id,
                url=str(row.get("url", "")),
                local_path=row.get("local_path"),
                retrieved_utc=str(row.get("retrieved_utc", "")),
                status=str(row.get("status", "")),
                bytes=row.get("bytes"),
                sha256=row.get("sha256"),
                error=row.get("error"),
                current_run_status=row.get("current_run_status"),
            )
        except KeyError:
            continue
    return records


def offline_record_from_snapshot(
    previous: SourceRecord | None,
    fallback: SourceRecord,
    *,
    current_run_status: str,
) -> SourceRecord:
    if previous is None:
        return SourceRecord(
            fallback.source_id,
            fallback.url,
            fallback.local_path,
            fallback.retrieved_utc,
            fallback.status,
            fallback.bytes,
            fallback.sha256,
            fallback.error,
            current_run_status,
        )
    if current_run_status == "cached":
        if previous.bytes is not None and fallback.bytes != previous.bytes:
            raise RuntimeError(
                f"cached source {previous.source_id} differs from frozen manifest: "
                f"bytes {fallback.bytes} != {previous.bytes}"
            )
        if previous.sha256 is not None and fallback.sha256 != previous.sha256:
            raise RuntimeError(
                f"cached source {previous.source_id} differs from frozen manifest: "
                f"sha256 {fallback.sha256} != {previous.sha256}"
            )
    return SourceRecord(
        previous.source_id,
        previous.url or fallback.url,
        fallback.local_path if fallback.local_path is not None else previous.local_path,
        previous.retrieved_utc,
        previous.status,
        previous.bytes,
        previous.sha256,
        previous.error,
        current_run_status,
    )


def acquire(
    source_id: str,
    url: str,
    data_dir: Path,
    filename: str,
    *,
    offline: bool,
    previous: SourceRecord | None = None,
) -> SourceRecord:
    path = data_dir / filename
    retrieved = utc_now()
    if offline:
        if path.is_file():
            data = path.read_bytes()
            fallback = SourceRecord(source_id, url, str(path), retrieved, "cached", len(data), sha256_bytes(data))
            return offline_record_from_snapshot(previous, fallback, current_run_status="cached")
        fallback = SourceRecord(source_id, url, str(path), retrieved, "missing_offline", error="not present in cache")
        return offline_record_from_snapshot(previous, fallback, current_run_status="missing_offline")

    status, data, error = fetch_url(url)
    if status is not None and 200 <= status < 300 and data:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return SourceRecord(source_id, url, str(path), retrieved, f"http_{status}", len(data), sha256_bytes(data))
    if path.is_file():
        cached = path.read_bytes()
        status_label = f"cached_after_http_{status}" if status else "cached_after_fetch_failure"
        return SourceRecord(source_id, url, str(path), retrieved, status_label, len(cached), sha256_bytes(cached), error)
    return SourceRecord(source_id, url, str(path), retrieved, f"http_{status}" if status else "failed", error=error)


def check_url(
    source_id: str,
    url: str,
    *,
    offline: bool,
    previous: SourceRecord | None = None,
) -> SourceRecord:
    retrieved = utc_now()
    if offline:
        fallback = SourceRecord(source_id, url, None, retrieved, "not_checked_offline")
        return offline_record_from_snapshot(previous, fallback, current_run_status="not_checked_offline")
    status, data, error = fetch_url(url)
    sha = sha256_bytes(data) if data else None
    return SourceRecord(
        source_id,
        url,
        None,
        retrieved,
        f"http_{status}" if status else "failed",
        len(data) if data else None,
        sha,
        error,
    )


def infer_saxs_unit_evidence(headers: Sequence[str]) -> str:
    joined = " | ".join(headers)
    lowered = joined.lower()
    if "q : defined in inverse angstroms" in lowered or '"momentum_transfer_units":"inverse angstroms' in lowered:
        return "explicit_dat_header_inverse_angstroms"
    if "reciprocal space rg" in lowered or "real space rg" in lowered or "angstrom" in lowered:
        return "zip_or_dat_metadata_reports_rg_in_angstroms_no_explicit_q_unit_header"
    return "not_found_in_dat_header"


def extract_json_metadata(headers: Sequence[str]) -> dict[str, object] | None:
    for line in headers:
        stripped = line.strip()
        if stripped.startswith("{") and "sas_result" in stripped:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    return None


def gnom_angular_range_from_zip(zip_path: Path) -> str:
    if not zip_path.is_file():
        return ""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if not name.endswith(".out"):
                    continue
                text = archive.read(name).decode(errors="replace")
                for line in text.splitlines():
                    lower = line.lower()
                    if "angular range" in lower or "angular" in lower and "range" in lower:
                        return f"{name}: {line.strip()}"
    except (OSError, zipfile.BadZipFile):
        return ""
    return ""


def guinier_interval_inventory_row(
    accession: str,
    condition: str,
    headers: Sequence[str],
    data_dir: Path,
) -> dict[str, object]:
    metadata = extract_json_metadata(headers)
    supplied = False
    candidate_metadata_present = False
    metadata_fields = ""
    raw_metadata = ""
    status = "no supplied Guinier range identified"
    reason = "No public entry/source-data field found that gives a usable Guinier fitting interval."
    if metadata:
        sas_result = metadata.get("sas_result")
        if isinstance(sas_result, dict):
            point_min = sas_result.get("guinier_point_min", sas_result.get("quinier_point_min"))
            point_max = sas_result.get("guinier_point_max")
            if point_min is not None or point_max is not None:
                candidate_metadata_present = True
                metadata_fields = "sas_result.guinier_point_min/sas_result.guinier_point_max"
                raw_metadata = json.dumps(
                    {"guinier_point_min": point_min, "guinier_point_max": point_max},
                    sort_keys=True,
                )
                try:
                    point_count = int(point_max) - int(point_min) + 1
                except (TypeError, ValueError):
                    point_count = 0
                status = "Guinier point metadata present but not used"
                reason = (
                    f"Point metadata yields {point_count} point(s), below the 8-point refit minimum; "
                    "not treated as a supplied reproducible fitting interval."
                )
    gnom = gnom_angular_range_from_zip(data_dir / f"{accession}.zip")
    if gnom:
        reason += " GNOM/PR angular range is recorded separately and is not used as a Guinier fitting interval."
    return {
        "accession": accession,
        "condition": condition,
        "supplied_guinier_range_identified": supplied,
        "candidate_guinier_point_metadata_present": candidate_metadata_present,
        "used_for_refit": False,
        "fit_interval_status": status,
        "source_file": f"{accession}.dat",
        "metadata_fields": metadata_fields,
        "raw_metadata": raw_metadata,
        "gnom_angular_range_present": bool(gnom),
        "gnom_angular_range_source": gnom,
        "refit_strategy": "low-q scan over positive points constrained to qRg<=max_qrg; qRg<=1.0 sensitivity also reported",
        "reason": reason,
    }

def parse_sas_dat(text: str) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    header: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        try:
            nums = [float(field) for field in fields[:3]]
        except ValueError:
            header.append(stripped)
            continue
        if len(nums) >= 3 and all(math.isfinite(value) for value in nums):
            rows.append(nums[:3])
    if not rows:
        raise ValueError("SASBDB dat file contains no numeric q/I/error rows")
    arr = np.asarray(rows, dtype=float)
    if np.any(arr[:, 0] <= 0) or np.any(arr[:, 2] <= 0):
        raise ValueError("SASBDB dat file contains non-positive q or error values")
    return arr, header


def weighted_guinier_fit(
    q: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    *,
    max_qrg: float = 1.3,
    sensitivity_qrg: float = 1.0,
    min_points: int = 8,
) -> dict[str, float | int | str]:
    usable = np.isfinite(q) & np.isfinite(intensity) & np.isfinite(sigma)
    usable &= (q > 0) & (intensity > 0) & (sigma > 0)
    q = q[usable]
    intensity = intensity[usable]
    sigma = sigma[usable]
    if q.size < min_points:
        raise ValueError("not enough positive finite SAXS points for Guinier fitting")

    best: dict[str, float | int | str] | None = None
    ln_i = np.log(intensity)
    sigma_ln = sigma / intensity
    for end in range(min_points, q.size + 1):
        x = q[:end] ** 2
        y = ln_i[:end]
        weights = 1.0 / np.square(sigma_ln[:end])
        coef, cov = np.polyfit(x, y, 1, w=np.sqrt(weights), cov=True)
        slope = float(coef[0])
        intercept = float(coef[1])
        if slope >= 0:
            continue
        rg = math.sqrt(-3.0 * slope)
        qrg_max = float(q[end - 1] * rg)
        if qrg_max > max_qrg:
            break
        fitted = slope * x + intercept
        resid = (y - fitted) / sigma_ln[:end]
        dof = max(1, end - 2)
        chi2_red = float(np.sum(np.square(resid)) / dof)
        stderr_rg = float("nan")
        if cov.shape == (2, 2) and cov[0, 0] > 0:
            stderr_rg = math.sqrt(cov[0, 0]) * 3.0 / (2.0 * rg)
        best = {
            "n_points": end,
            "q_min": float(q[0]),
            "q_max": float(q[end - 1]),
            "rg_nm": rg,
            "rg_nm_stderr": stderr_rg,
            "i0": math.exp(intercept),
            "slope": slope,
            "intercept": intercept,
            "qrg_max": qrg_max,
            "reduced_chi2": chi2_red,
        }
    if best is None:
        raise ValueError("no negative-slope Guinier region satisfied qRg constraint")

    mask_sens = q * float(best["rg_nm"]) <= sensitivity_qrg
    best["n_points_qrg_le_1_0"] = int(np.sum(mask_sens))
    best["low_q_qa"] = "pass" if int(best["n_points_qrg_le_1_0"]) >= min_points else "sparse_qrg_le_1.0"
    return best


def kratky_summary(q: np.ndarray, intensity: np.ndarray, rg_nm: float, i0: float) -> dict[str, float]:
    x = q * rg_nm
    y = (q * rg_nm) ** 2 * (intensity / i0)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return {"kratky_peak_qrg": float("nan"), "kratky_peak_y": float("nan")}
    idx = int(np.nanargmax(y[finite]))
    xf = x[finite]
    yf = y[finite]
    return {"kratky_peak_qrg": float(xf[idx]), "kratky_peak_y": float(yf[idx])}


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_primary_window(path: Path | None = None) -> set[int]:
    path = PRIMARY_WINDOW_PATH if path is None else path
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "author_resnum" not in reader.fieldnames:
            raise ValueError(f"{path}: missing author_resnum column")
        return {int(row["author_resnum"]) for row in reader if row.get("author_resnum")}


def residue_window_classification(
    residues: Iterable[int], primary_window: set[int] | None = None
) -> list[dict[str, int | str]]:
    if primary_window is None:
        primary_window = load_primary_window()
    rows = []
    for residue in residues:
        rows.append(
            {
                "residue": residue,
                "primary_269_window": "inside" if residue in primary_window else "outside",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")



def extract_pdf_text_if_available(pdf_path: Path, text_path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if not pdf_path.is_file():
        return "pdf_missing"
    if not pdftotext:
        return "pdftotext_unavailable"
    try:
        subprocess.run([pdftotext, str(pdf_path), str(text_path)], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "pdftotext_failed"
    return "extracted" if text_path.is_file() else "pdftotext_no_output"


def supplement_table_inventory(text_path: Path) -> list[dict[str, object]]:
    if not text_path.is_file():
        return []
    rows: list[dict[str, object]] = []
    table_re = re.compile(r"^(Table S\d+\.?\s+.*)")
    seen: set[str] = set()
    for line_no, line in enumerate(text_path.read_text(errors="replace").splitlines(), start=1):
        stripped = line.strip()
        match = table_re.match(stripped)
        if not match:
            continue
        title = match.group(1)
        table_id = title.split()[1].rstrip(".") if len(title.split()) > 1 else title
        if table_id in seen:
            continue
        seen.add(table_id)
        lower = title.lower()
        assay_type = ""
        if "spr" in lower:
            assay_type = "binding_spr"
        elif "itc" in lower:
            assay_type = "binding_itc"
        elif "dsf" in lower or "thermal" in lower:
            assay_type = "folding_dsf"
        elif "saxs" in lower or "scattering" in lower:
            assay_type = "saxs"
        elif "lc-ms" in lower or "mutant" in lower or "mutagenesis" in lower:
            assay_type = "mutation_construct_qc"
        elif "cryo-em" in lower or "crystallographic" in lower:
            assay_type = "structure_statistics"
        rows.append({"table_id": table_id, "title": title, "assay_type": assay_type, "text_line": line_no})
    return rows

def extract_pmc_preprint_status(html: str) -> dict[str, str | bool | None]:
    return {
        "journal": _meta_content(html, "citation_journal_title"),
        "doi": _meta_content(html, "citation_doi"),
        "version": "1" if "[Version 1]" in html or "695617v1" in html else None,
        "pmc_preprint_flag": 'name="ncbi_pcid" content="preprint"' in html,
        "not_peer_reviewed_banner": "It has not yet been peer reviewed by a journal." in html,
    }


def _meta_content(html: str, name: str) -> str | None:
    match = re.search(rf'<meta name="{re.escape(name)}" content="([^"]*)"', html)
    return match.group(1) if match else None


def run(args: argparse.Namespace) -> int:
    data_dir = args.output_dir / "data" / "external"
    analysis_dir = args.output_dir / "analysis" / "external"
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    previous_sources = load_previous_source_manifest(analysis_dir / "sources_manifest.json") if args.offline else {}

    source_records: list[SourceRecord] = []
    source_records.append(
        acquire(
            "kroupova_article_html",
            KROUPOVA_ARTICLE,
            data_dir,
            "Kroupova_2024_article.html",
            offline=args.offline,
            previous=previous_sources.get("kroupova_article_html"),
        )
    )
    source_records.append(
        acquire(
            "kroupova_source_data_xlsx",
            KROUPOVA_SOURCE_XLSX,
            data_dir,
            "Kroupova_2024_Source_Data.xlsx",
            offline=args.offline,
            previous=previous_sources.get("kroupova_source_data_xlsx"),
        )
    )
    source_records.append(
        acquire(
            "kroupova_supplementary_data_zip",
            KROUPOVA_SUPP_ZIP,
            data_dir,
            "Kroupova_2024_Supplementary_Data_1.zip",
            offline=args.offline,
            previous=previous_sources.get("kroupova_supplementary_data_zip"),
        )
    )
    source_records.append(
        acquire(
            "sasbdb_project_2221_html",
            SASBDB_PROJECT,
            data_dir,
            "SASBDB_project_2221.html",
            offline=args.offline,
            previous=previous_sources.get("sasbdb_project_2221_html"),
        )
    )
    source_records.append(
        acquire(
            "sasbdb_help_html",
            SASBDB_HELP,
            data_dir,
            "SASBDB_help.html",
            offline=args.offline,
            previous=previous_sources.get("sasbdb_help_html"),
        )
    )
    source_records.append(
        acquire(
            "oconnor_pmc_html",
            OCONNOR_PMC,
            data_dir,
            "OConnor_2025_PMC12767645.html",
            offline=args.offline,
            previous=previous_sources.get("oconnor_pmc_html"),
        )
    )
    source_records.append(
        acquire(
            "oconnor_biorxiv_html",
            OCONNOR_BIORXIV_HTML,
            data_dir,
            "OConnor_2025_biorxiv.html",
            offline=args.offline,
            previous=previous_sources.get("oconnor_biorxiv_html"),
        )
    )
    source_records.append(
        acquire(
            "oconnor_biorxiv_supplement_tab_html",
            OCONNOR_BIORXIV_SUPP_TAB,
            data_dir,
            "OConnor_2025_biorxiv_supplement_tab.html",
            offline=args.offline,
            previous=previous_sources.get("oconnor_biorxiv_supplement_tab_html"),
        )
    )
    source_records.append(
        acquire(
            "oconnor_biorxiv_preprint_pdf",
            OCONNOR_BIORXIV_PDF,
            data_dir,
            "OConnor_2025_preprint.pdf",
            offline=args.offline,
            previous=previous_sources.get("oconnor_biorxiv_preprint_pdf"),
        )
    )
    source_records.append(
        acquire(
            "oconnor_pubmed_esummary_json",
            OCONNOR_PUBMED_ESUMMARY,
            data_dir,
            "OConnor_2025_pubmed_esummary.json",
            offline=args.offline,
            previous=previous_sources.get("oconnor_pubmed_esummary_json"),
        )
    )
    source_records.append(
        acquire(
            "oconnor_crossref_json",
            OCONNOR_CROSSREF,
            data_dir,
            "OConnor_2025_crossref.json",
            offline=args.offline,
            previous=previous_sources.get("oconnor_crossref_json"),
        )
    )
    source_records.append(
        acquire(
            "oconnor_europepmc_json",
            OCONNOR_EUROPEPMC,
            data_dir,
            "OConnor_2025_europepmc.json",
            offline=args.offline,
            previous=previous_sources.get("oconnor_europepmc_json"),
        )
    )
    source_records.append(
        acquire(
            "oconnor_supplement_pdf",
            OCONNOR_BIORXIV_SUPPLEMENT,
            data_dir,
            "OConnor_2025_media-1_supplement.pdf",
            offline=args.offline,
            previous=previous_sources.get("oconnor_supplement_pdf"),
        )
    )

    for accession in SASBDB_ACCESSIONS:
        source_records.append(
            acquire(
                f"sasbdb_{accession}_dat",
                f"https://www.sasbdb.org/media/intensities_files/{accession}.dat",
                data_dir,
                f"{accession}.dat",
                offline=args.offline,
                previous=previous_sources.get(f"sasbdb_{accession}_dat"),
            )
        )
        source_records.append(
            acquire(
                f"sasbdb_{accession}_zip",
                f"https://www.sasbdb.org/media/zip_directories/{accession}.zip",
                data_dir,
                f"{accession}.zip",
                offline=args.offline,
                previous=previous_sources.get(f"sasbdb_{accession}_zip"),
            )
        )

    for pdb_id in KROUPOVA_PDB_IDS + OCONNOR_PDB_IDS:
        source_records.append(
            check_url(
                f"pdb_{pdb_id.lower()}_cif",
                f"https://files.rcsb.org/download/{pdb_id}.cif",
                offline=args.offline,
                previous=previous_sources.get(f"pdb_{pdb_id.lower()}_cif"),
            )
        )
    for emdb_id in OCONNOR_EMDB_IDS:
        bare = emdb_id.replace("EMD-", "")
        source_records.append(
            check_url(
                f"emdb_{emdb_id.lower()}",
                f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{bare}/header/emd-{bare}.xml",
                offline=args.offline,
                previous=previous_sources.get(f"emdb_{emdb_id.lower()}"),
            )
        )

    manifest = [record.__dict__ for record in source_records]
    write_json(analysis_dir / "sources_manifest.json", manifest)

    guinier_rows: list[dict[str, object]] = []
    interval_inventory_rows: list[dict[str, object]] = []
    kratky_rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for accession, meta in SASBDB_ACCESSIONS.items():
        path = data_dir / f"{accession}.dat"
        if not path.is_file():
            missing_rows.append(
                {
                    "source_id": f"sasbdb_{accession}_dat",
                    "expected_path": str(path),
                    "reason": "missing dat curve",
                }
            )
            continue
        arr, headers = parse_sas_dat(path.read_text(errors="replace"))
        q, intensity, sigma = arr[:, 0], arr[:, 1], arr[:, 2]
        interval_inventory = guinier_interval_inventory_row(accession, meta["condition"], headers, data_dir)
        interval_inventory_rows.append(interval_inventory)
        fit = weighted_guinier_fit(q, intensity, sigma, max_qrg=args.max_qrg)
        sensitivity = weighted_guinier_fit(q, intensity, sigma, max_qrg=1.0)
        unit_evidence = infer_saxs_unit_evidence(headers)
        q_unit = "inverse_angstrom"
        if unit_evidence == "not_found_in_dat_header":
            q_unit = "assumed_inverse_angstrom"
            unit_evidence = (
                "assumed_for_nm_conversion; dat header lacks explicit q unit; "
                "SASBDB help confirms q/I/error columns but not units; "
                "supplement SAXS tables report q range in Angstrom^-1"
            )
        rg_angstrom = float(fit["rg_nm"])
        rg_nm = rg_angstrom / 10.0
        rg_nm_stderr = float(fit["rg_nm_stderr"]) / 10.0
        kratky = kratky_summary(q, intensity, rg_angstrom, float(fit["i0"]))
        fit_n = int(fit["n_points"])
        slope = float(fit["slope"])
        intercept = float(fit["intercept"])
        guinier_rows.append(
            {
                "accession": accession,
                "condition": meta["condition"],
                "published_rg_nm_context_only": meta["published_rg_nm"],
                "refit_rg_nm": rg_nm,
                "refit_rg_nm_stderr": rg_nm_stderr,
                "sensitivity_qrg_le_1_0_rg_nm": float(sensitivity["rg_nm"]) / 10.0,
                "sensitivity_qrg_le_1_0_n_points": sensitivity["n_points"],
                "provided_fit_range_used": False,
                "provided_fit_range_status": interval_inventory["fit_interval_status"],
                "fit_range_strategy": "no supplied Guinier range used; fallback_low_q_scan_qrg_le_" + str(args.max_qrg),
                "i0": fit["i0"],
                "n_points": fit["n_points"],
                "q_min": fit["q_min"],
                "q_max": fit["q_max"],
                "qrg_max": fit["qrg_max"],
                "reduced_chi2": fit["reduced_chi2"],
                "n_points_qrg_le_1_0": fit["n_points_qrg_le_1_0"],
                "low_q_qa": fit["low_q_qa"],
                "q_unit": q_unit,
                "q_unit_evidence": unit_evidence,
                "header": " | ".join(headers[:3]),
            }
        )
        positive = (q > 0) & (intensity > 0) & (sigma > 0)
        positive_index = 0
        for raw_index, (qv, iv, sv, is_positive) in enumerate(zip(q, intensity, sigma, positive), start=1):
            in_fit = bool(is_positive and positive_index < fit_n)
            if is_positive:
                positive_index += 1
            ln_i = math.log(float(iv)) if iv > 0 else float("nan")
            plot_rows.append(
                {
                    "accession": accession,
                    "condition": meta["condition"],
                    "point_index": raw_index,
                    "q_inverse_angstrom": float(qv),
                    "intensity": float(iv),
                    "error": float(sv),
                    "ln_intensity": ln_i,
                    "guinier_fit_ln_intensity": slope * float(qv) ** 2 + intercept if in_fit else "",
                    "in_primary_guinier_fit": in_fit,
                    "qrg_primary": float(qv) * rg_angstrom,
                    "qrg_sensitivity": float(qv) * float(sensitivity["rg_nm"]),
                    "kratky_x_qrg": float(qv) * rg_angstrom,
                    "kratky_y": (float(qv) * rg_angstrom) ** 2 * (float(iv) / float(fit["i0"]))
                    if iv > 0
                    else "",
                }
            )
        kratky_rows.append(
            {
                "accession": accession,
                "condition": meta["condition"],
                "kratky_peak_qrg": kratky["kratky_peak_qrg"],
                "kratky_peak_y": kratky["kratky_peak_y"],
            }
        )

    write_csv(
        analysis_dir / "saxs_guinier_refits.csv",
        guinier_rows,
        [
            "accession",
            "condition",
            "published_rg_nm_context_only",
            "refit_rg_nm",
            "refit_rg_nm_stderr",
            "sensitivity_qrg_le_1_0_rg_nm",
            "sensitivity_qrg_le_1_0_n_points",
            "provided_fit_range_used",
            "provided_fit_range_status",
            "fit_range_strategy",
            "i0",
            "n_points",
            "q_min",
            "q_max",
            "qrg_max",
            "reduced_chi2",
            "n_points_qrg_le_1_0",
            "low_q_qa",
            "q_unit",
            "q_unit_evidence",
            "header",
        ],
    )

    write_csv(
        analysis_dir / "saxs_guinier_interval_inventory.csv",
        interval_inventory_rows,
        [
            "accession",
            "condition",
            "supplied_guinier_range_identified",
            "candidate_guinier_point_metadata_present",
            "used_for_refit",
            "fit_interval_status",
            "source_file",
            "metadata_fields",
            "raw_metadata",
            "gnom_angular_range_present",
            "gnom_angular_range_source",
            "refit_strategy",
            "reason",
        ],
    )

    write_csv(
        analysis_dir / "saxs_kratky_summary.csv",
        kratky_rows,
        ["accession", "condition", "kratky_peak_qrg", "kratky_peak_y"],
    )
    write_csv(
        analysis_dir / "saxs_plot_arrays.csv",
        plot_rows,
        [
            "accession",
            "condition",
            "point_index",
            "q_inverse_angstrom",
            "intensity",
            "error",
            "ln_intensity",
            "guinier_fit_ln_intensity",
            "in_primary_guinier_fit",
            "qrg_primary",
            "qrg_sensitivity",
            "kratky_x_qrg",
            "kratky_y",
        ],
    )

    construct_metadata = {
        "construct": "CRBNmidi",
        "source": KROUPOVA_ARTICLE,
        "interpretation_boundary": (
            "Engineered CRBNmidi measurements are retained as within-construct "
            "solution compaction evidence only; they are not native full-length "
            "or CRBN-DDB1 validation."
        ),
        "residue_segments": ["41-187", "249-426"],
        "deleted_region": "188-248",
        "linker": "Gly-Ser-Gly",
        "engineered_mutations": [
            "C78I",
            "I92V",
            "K116N",
            "Q134E",
            "R283W",
            "C287N",
            "V293S",
            "G302D",
            "L342R",
            "C343E",
            "T359I",
            "L423I",
        ],
        "primary_269_window_rule": f"loaded from {display_path(PRIMARY_WINDOW_PATH)}",
    }
    write_json(analysis_dir / "kroupova_crbnmidi_construct_metadata.json", construct_metadata)

    write_csv(
        analysis_dir / "oconnor_variant_window_classification.csv",
        residue_window_classification([59, 60, 100, 156, 269, 350, 351, 378, 380]),
        ["residue", "primary_269_window"],
    )

    quantitative_rows = oconnor_quantitative_rows()
    write_csv(
        analysis_dir / "oconnor_pdf_quantitative_measurements.csv",
        quantitative_rows,
        [
            "table_id",
            "pdf_page",
            "assay_type",
            "construct",
            "mutation",
            "compound",
            "compound_class",
            "measurement",
            "replicate",
            "value",
            "unit",
            "error_value",
            "error_unit",
            "censoring",
            "source_extraction",
        ],
    )
    mutagenesis_rows = oconnor_mutagenesis_qc_rows()
    write_csv(
        analysis_dir / "oconnor_pdf_mutagenesis_qc.csv",
        mutagenesis_rows,
        ["table_id", "pdf_page", "record_type", "name", "sequence", "measurement", "value", "unit", "source_extraction"],
    )
    variant_inventory = oconnor_variant_inventory_rows()
    write_csv(
        analysis_dir / "oconnor_variant_inventory.csv",
        variant_inventory,
        ["variant", "residues", "primary_269_window", "source_tables", "inventory_source", "inventory_scope"],
    )
    case_rows = oconnor_compound9_case_rows()
    write_csv(
        analysis_dir / "oconnor_compound9_mutant_case_comparison.csv",
        case_rows,
        [
            "compound",
            "compound_class",
            "variant",
            "residues",
            "primary_269_window",
            "dsf_tm_degC",
            "dsf_delta_tm_degC",
            "dsf_delta_delta_tm_vs_wt_degC",
            "saxs_guinier_rg_angstrom",
            "saxs_delta_rg_vs_wt_angstrom",
            "binding_note",
            "comparison_scope",
        ],
    )

    candidate_path = args.output_dir / "analysis" / "contacts" / "candidate_robustness.csv"
    candidate_comparison_performed = candidate_path.is_file()
    write_candidate_variant_comparison(
        candidate_path,
        analysis_dir / "oconnor_variant_candidate_sparse_comparison.csv",
    )

    html_path = data_dir / "OConnor_2025_PMC12767645.html"
    oconnor_status = extract_pmc_preprint_status(html_path.read_text(errors="replace")) if html_path.is_file() else {}
    pubmed_path = data_dir / "OConnor_2025_pubmed_esummary.json"
    if pubmed_path.is_file():
        try:
            payload = json.loads(pubmed_path.read_text())
            item = payload["result"][OCONNOR_PUBMED_ID]
            oconnor_status["pubmed_source"] = item.get("source")
            oconnor_status["pubmed_pubtypes"] = item.get("pubtype")
            oconnor_status["pubmed_fulljournalname"] = item.get("fulljournalname")
        except (KeyError, TypeError, json.JSONDecodeError):
            oconnor_status["pubmed_parse_error"] = True
    crossref_path = data_dir / "OConnor_2025_crossref.json"
    if crossref_path.is_file():
        try:
            message = json.loads(crossref_path.read_text())["message"]
            oconnor_status["crossref_publisher"] = message.get("publisher")
            oconnor_status["crossref_institution"] = [
                item.get("name") for item in message.get("institution", []) if item.get("name")
            ]
        except (KeyError, TypeError, json.JSONDecodeError):
            oconnor_status["crossref_parse_error"] = True
    europepmc_path = data_dir / "OConnor_2025_europepmc.json"
    if europepmc_path.is_file():
        try:
            result = json.loads(europepmc_path.read_text())["resultList"]["result"][0]
            oconnor_status["europepmc_pubtype"] = result.get("pubType")
            oconnor_status["europepmc_source"] = result.get("source")
            oconnor_status["europepmc_publisher"] = (
                result.get("bookOrReportDetails", {}) or {}
            ).get("publisher")
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            oconnor_status["europepmc_parse_error"] = True
    write_json(analysis_dir / "oconnor_version_status.json", oconnor_status)

    supplement_text_status = extract_pdf_text_if_available(
        data_dir / "OConnor_2025_media-1_supplement.pdf",
        analysis_dir / "OConnor_2025_supplement_tables_text.txt",
    )
    supplement_inventory = supplement_table_inventory(analysis_dir / "OConnor_2025_supplement_tables_text.txt")
    write_csv(
        analysis_dir / "oconnor_supplement_table_inventory.csv",
        supplement_inventory,
        ["table_id", "title", "assay_type", "text_line"],
    )

    if not (data_dir / "OConnor_2025_source_data.xlsx").is_file():
        missing_rows.append(
            {
                "source_id": "oconnor_machine_readable_source_tables",
                "expected_path": str(data_dir / "OConnor_2025_source_data.xlsx"),
                "reason": "no machine-readable XLSX/CSV source workbook linked from PMC or bioRxiv; preprint PDF and PDF supplement were downloaded",
            }
        )
    for record in source_records:
        if record.status.startswith("http_4") or record.status.startswith("http_5") or record.status in {
            "failed",
            "missing_offline",
        }:
            missing_rows.append(
                {
                    "source_id": record.source_id,
                    "expected_path": record.local_path or "",
                    "reason": record.error or record.status,
                }
            )
    write_csv(analysis_dir / "missing_files_report.csv", missing_rows, ["source_id", "expected_path", "reason"])

    summary = {
        "generated_utc": utc_now(),
        "offline": args.offline,
        "saxs_refit_rows": len(guinier_rows),
        "saxs_guinier_interval_inventory_rows": len(interval_inventory_rows),
        "saxs_plot_array_rows": len(plot_rows),
        "sources_checked": len(source_records),
        "missing_or_unavailable": len(missing_rows),
        "candidate_comparison_performed": candidate_comparison_performed,
        "oconnor_pdf_quantitative_rows": len(quantitative_rows),
        "oconnor_pdf_mutagenesis_qc_rows": len(mutagenesis_rows),
        "oconnor_variant_inventory_rows": len(variant_inventory),
        "oconnor_compound9_case_rows": len(case_rows),
        "oconnor_supplement_text_status": supplement_text_status,
        "oconnor_supplement_table_inventory_rows": len(supplement_inventory),
        "candidate_comparison_note": (
            "Sparse residue-level comparison to frozen candidate_robustness.csv only; "
            "no classifier, selected-positive filter, or percentile claim was applied."
            if candidate_comparison_performed
            else "Frozen discovery candidates were not compared to external measurements."
        ),
        "assay_boundary": "SAXS, structure availability, construct metadata, and O'Connor assay tables are kept separate.",
    }
    write_json(analysis_dir / "external_strengthening_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0



def oconnor_quantitative_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(
        table_id: str,
        page: int,
        assay_type: str,
        construct: str,
        mutation: str,
        compound: str,
        compound_class: str,
        measurement: str,
        value: object,
        unit: str,
        *,
        error_value: object = "",
        error_unit: str = "",
        replicate: str = "",
        censoring: str = "",
        extraction: str = "pdf_text_layout_verified_by_render",
    ) -> None:
        rows.append(
            {
                "table_id": table_id,
                "pdf_page": page,
                "assay_type": assay_type,
                "construct": construct,
                "mutation": mutation,
                "compound": compound,
                "compound_class": compound_class,
                "measurement": measurement,
                "replicate": replicate,
                "value": value,
                "unit": unit,
                "error_value": error_value,
                "error_unit": error_unit,
                "censoring": censoring,
                "source_extraction": extraction,
            }
        )

    spr = [
        ("Lenalidomide", "IMiD", [135, 371, 379], 295.0, 138.6, 500),
        ("Compound 9", "DHU", [97.3, 107, 119], 107.8, 10.9, 500),
        ("Compound 10", "Cyclimid", [32100, 33800, 32300], 32733, 929.16, 100000),
        ("Compound 11", "Phenylglutarimide", [2560, 2470, 2100], 2376.7, 243.8, 10000),
        ("Compound 12", "Aminoglutarimide", [2530, 2510, 2520], 2520.0, 10, 100000),
        ("Compound 8", "5,4-Spiro", [187, 194, 183], 188.0, 5.57, 1000),
        ("Compound 4", "Morpholinone", [280, 270, 244], 264.3, 19.1, 1000),
        ("Compound S1", "Morpholinone, negative control", [18100, 19200, 19300], 18900, 665.8, 100000),
    ]
    for compound, cls, reps, mean, sd, top in spr:
        for idx, value in enumerate(reps, start=1):
            add("S1", 12, "binding_spr", "CRBNmidi", "WT", compound, cls, "Kd", value, "nM", replicate=str(idx))
        add("S1", 12, "binding_spr", "CRBNmidi", "WT", compound, cls, "Kd_mean", mean, "nM", error_value=sd, error_unit="SD")
        add("S1", 12, "binding_spr", "CRBNmidi", "WT", compound, cls, "assay_top_concentration", top, "nM")

    itc = [
        ("Lenalidomide", "IMiD", 0.223, 0.025, 1.78, 10.8, 0.074, -48.8, "exothermic"),
        ("4", "Morpholinone", 0.899, 0.055, 1.66, 29.8, 0.186, -4.74, "endothermic"),
        ("8", "5,4-Spiro", 2.02, 0.102, 1.24, -28.8, 0.238, -3.74, "endothermic"),
        ("9", "DHU", 0.039, 0.007, 1.08, 9.77, 0.123, -52.1, "exothermic"),
        ("11", "Phenylglutarimide", 8.51, 0.59, 1.42, 19.3, 0.405, -48.3, "exothermic"),
    ]
    for compound, cls, kd, kd_err, n, dh, dh_err, tds, clsfn in itc:
        add("S2", 12, "binding_itc", "CRBN:DDB1deltaBPB", "WT", compound, cls, "KD", kd, "uM", error_value=kd_err, error_unit="plus_minus")
        add("S2", 12, "binding_itc", "CRBN:DDB1deltaBPB", "WT", compound, cls, "stoichiometry_N", n, "unitless")
        add("S2", 12, "binding_itc", "CRBN:DDB1deltaBPB", "WT", compound, cls, "delta_H", dh, "kJ/mol", error_value=dh_err, error_unit="plus_minus")
        add("S2", 12, "binding_itc", "CRBN:DDB1deltaBPB", "WT", compound, cls, "minus_T_delta_S", tds, "kJ/mol", censoring=clsfn)

    dsf_s3 = [
        ("Apo", "/", [44.07, 44.13, 44.05], 44.08, 0.04, "", ""),
        ("Lenalidomide", "IMiD", [58.03, 58.04, 58.11], 58.05, 0.05, 13.97, 3.6),
        ("4", "Morpholinone", [41.15, 41.16, 41.21], 41.17, 0.03, -2.91, 7.4),
        ("8", "5,4-Spiro", [47.18, 47.38, 47.29], 47.29, 0.10, 3.21, 6.1),
        ("9", "Dihydrouracil", [58.61, 58.65, 58.67], 58.65, 0.03, 14.57, 5.1),
        ("10", "Cyclimid", [55.00, 54.92, 54.95], 54.96, 0.04, 10.88, 0.6),
        ("11", "Phenylglutarimide", [55.96, 56.14, 56.01], 56.03, 0.09, 11.95, 2.6),
        ("12", "Aminoglutarimide", [55.47, 55.58, 55.41], 55.49, 0.08, 11.41, "n.d."),
    ]
    for compound, cls, reps, mean, sd, delta_mid, delta_tbd in dsf_s3:
        for idx, value in enumerate(reps, start=1):
            add("S3", 13, "folding_dsf", "CRBNmidi", "WT", compound, cls, "Tm_turbidity", value, "degC", replicate=str(idx))
        add("S3", 13, "folding_dsf", "CRBNmidi", "WT", compound, cls, "Tm_turbidity_mean", mean, "degC", error_value=sd, error_unit="SD")
        if delta_mid != "":
            add("S3", 13, "folding_dsf", "CRBNmidi", "WT", compound, cls, "delta_Tm_CRBNmidi", delta_mid, "degC")
        if delta_tbd != "":
            censor = "not_determined" if delta_tbd == "n.d." else ""
            value = "" if delta_tbd == "n.d." else delta_tbd
            add("S3", 13, "folding_dsf", "CRBNTBD", "WT", compound, cls, "average_delta_Tm_CRBNTBD", value, "degC", censoring=censor)

    dsf_s4 = [
        ("WT", "Apo", "/", 44.08, ""), ("WT", "Lenalidomide", "Imid", 58.05, 13.97), ("WT", "Compound 9", "DHU", 58.65, 14.57), ("WT", "Compound 8", "5,4-Spiro", 47.29, 3.21), ("WT", "Compound 4", "Morpholinone", 41.17, -2.91),
        ("H378N", "Apo", "/", 43.00, ""), ("H378N", "Lenalidomide", "Imid", 55.49, 12.49), ("H378N", "Compound 9", "DHU", 56.14, 13.14), ("H378N", "Compound 8", "5,4-Spiro", 44.70, 1.70), ("H378N", "Compound 4", "Morpholinone", 41.65, -1.35),
        ("H378A", "Apo", "/", 43.05, ""), ("H378A", "Lenalidomide", "Imid", 53.99, 10.94), ("H378A", "Compound 9", "DHU", 53.91, 10.86), ("H378A", "Compound 8", "5,4-Spiro", 47.65, 4.6), ("H378A", "Compound 4", "Morpholinone", 41.81, -1.24),
        ("Q100A", "Apo", "/", 44.00, ""), ("Q100A", "Lenalidomide", "Imid", 58.00, 14.00), ("Q100A", "Compound 9", "DHU", 58.75, 14.75), ("Q100A", "Compound 8", "5,4-Spiro", 47.17, 3.17), ("Q100A", "Compound 4", "Morpholinone", 41.37, -2.63),
        ("L60A", "Apo", "/", 44.31, ""), ("L60A", "Lenalidomide", "Imid", 51.38, 7.07), ("L60A", "Compound 9", "DHU", 49.53, 5.22), ("L60A", "Compound 8", "5,4-Spiro", 46.47, 2.16), ("L60A", "Compound 4", "Morpholinone", 41.62, -2.69),
        ("L60A H378A", "Apo", "/", 42.80, ""), ("L60A H378A", "Lenalidomide", "Imid", 49.23, 6.43), ("L60A H378A", "Compound 9", "DHU", 47.45, 4.65), ("L60A H378A", "Compound 8", "5,4-Spiro", 43.79, 0.99), ("L60A H378A", "Compound 4", "Morpholinone", 39.83, -2.97),
    ]
    for mutation, compound, cls, tm, dtm in dsf_s4:
        add("S4", 13, "folding_dsf", "CRBNmidi", mutation, compound, cls, "Tm_turbidity", tm, "degC", extraction="manual_from_rendered_page_13")
        if dtm != "":
            add("S4", 13, "folding_dsf", "CRBNmidi", mutation, compound, cls, "delta_Tm", dtm, "degC", extraction="manual_from_rendered_page_13")

    s5_samples = ["CRBNmidi", "CRBNmidi bound to lenalidomide", "CRBNmidi bound to Compound 9", "CRBNmidi bound to Compound 11", "CRBNmidi bound to Compound 8", "CRBNmidi bound to Compound 4"]
    s5_compounds = ["Apo", "Lenalidomide", "Compound 9", "Compound 11", "Compound 8", "Compound 4"]
    s5_classes = ["/", "IMiD", "DHU", "Phenylglutarimide", "5,4-Spiro", "Morpholinone"]
    s5 = [
        (37.4, 0.017, "3.4e-5", 26.83, 0.09, 1.28, 0.016, "2.7e-5", 26.26, 0.051, 154.26, "0.0126-0.2982", 0.87, 70306.90),
        (37.7, 0.017, "2.4e-5", 22.99, 0.06, 1.30, 0.017, "1.9e-5", 22.56, 0.035, 137.92, "0.0089-0.3400", 0.86, 59262.70),
        (37.7, 0.017, "2.8e-5", 23.43, 0.07, 1.30, 0.017, "2.8e-5", 23.11, 0.054, 141.94, "0.0093-0.3400", 0.81, 61505.60),
        (37.7, 0.012, "1.6e-5", 22.80, 0.06, 1.30, 0.012, "1.5e-5", 22.55, 0.046, 141.42, "0.0111-0.3400", 0.78, 63949),
        (37.7, 0.021, "2.6e-5", 26.04, 0.06, 1.30, 0.021, "2.5e-5", 26.05, 0.0049, 161.52, "0.0101-0.3071", 0.83, 60447.10),
        (37.7, 0.0096, "2.4e-5", 25.72, 0.11, 1.30, 0.0095, "2.2e-5", 25.43, 0.060, 145.16, "0.0126-0.3110", 0.85, 55533.10),
    ]
    for sample, compound, cls, vals in zip(s5_samples, s5_compounds, s5_classes, s5):
        mass, i0g, i0g_err, rgg, rgg_err, qrg, i0p, i0p_err, rgp, rgp_err, dmax, qrange, chi2, porod = vals
        add("S5", 14, "saxs", sample, "WT", compound, cls, "expected_molecular_mass", mass, "kDa")
        add("S5", 14, "saxs", sample, "WT", compound, cls, "guinier_I0", i0g, "cm^-1", error_value=i0g_err, error_unit="plus_minus")
        add("S5", 14, "saxs", sample, "WT", compound, cls, "guinier_Rg", rgg, "Angstrom", error_value=rgg_err, error_unit="plus_minus")
        add("S5", 14, "saxs", sample, "WT", compound, cls, "guinier_qRg_max", qrg, "unitless")
        add("S5", 14, "saxs", sample, "WT", compound, cls, "pr_I0", i0p, "cm^-1", error_value=i0p_err, error_unit="plus_minus")
        add("S5", 14, "saxs", sample, "WT", compound, cls, "pr_Rg", rgp, "Angstrom", error_value=rgp_err, error_unit="plus_minus")
        add("S5", 14, "saxs", sample, "WT", compound, cls, "dmax", dmax, "Angstrom")
        add("S5", 14, "saxs", sample, "WT", compound, cls, "q_range", qrange, "Angstrom^-1")
        add("S5", 14, "saxs", sample, "WT", compound, cls, "gnom_chi2", chi2, "unitless")
        add("S5", 15, "saxs", sample, "WT", compound, cls, "porod_volume", porod, "Angstrom^-3")

    s6 = [
        ("H378A", 37.4, 0.012, "1.6e-5", 22.88, 0.06, 1.30, 0.011, "1.5e-5", 22.35, 0.033, 124.76, "0.0083-0.3400", 0.87, 54576),
        ("H378N", 37.7, 0.013, "2.9e-5", 23.80, 0.10, 1.30, 0.012, "2.3e-5", 22.83, 0.041, 123.82, "0.0091-0.3361", 0.85, 64098),
        ("L60A", 37.7, 0.012, "1.8e-5", 24.84, 0.07, 1.29, 0.012, "1.8e-5", 24.56, 0.054, 145.48, "0.0121-0.3220", 0.82, 61883),
        ("L60A H378A", 37.6, 0.014, "2.0e-5", 25.18, 0.06, 1.30, 0.014, "1.7e-5", 25.05, 0.039, 149.88, "0.0090-0.3177", 0.86, 63723),
        ("Q100A", 37.7, 0.011, "1.7e-5", 23.07, 0.07, 1.30, 0.011, "1.3e-5", 22.38, 0.032, 125.78, "0.0082-0.3400", 0.86, 56333),
    ]
    for mutation, mass, i0g, i0g_err, rgg, rgg_err, qrg, i0p, i0p_err, rgp, rgp_err, dmax, qrange, chi2, porod in s6:
        construct = f"CRBNmidi {mutation} bound to compound 9"
        add("S6", 16, "saxs", construct, mutation, "Compound 9", "DHU", "expected_molecular_mass", mass, "kDa")
        add("S6", 16, "saxs", construct, mutation, "Compound 9", "DHU", "guinier_I0", i0g, "cm^-1", error_value=i0g_err, error_unit="plus_minus")
        add("S6", 16, "saxs", construct, mutation, "Compound 9", "DHU", "guinier_Rg", rgg, "Angstrom", error_value=rgg_err, error_unit="plus_minus")
        add("S6", 16, "saxs", construct, mutation, "Compound 9", "DHU", "guinier_qRg_max", qrg, "unitless")
        add("S6", 16, "saxs", construct, mutation, "Compound 9", "DHU", "pr_I0", i0p, "cm^-1", error_value=i0p_err, error_unit="plus_minus")
        add("S6", 16, "saxs", construct, mutation, "Compound 9", "DHU", "pr_Rg", rgp, "Angstrom", error_value=rgp_err, error_unit="plus_minus")
        add("S6", 16, "saxs", construct, mutation, "Compound 9", "DHU", "dmax", dmax, "Angstrom")
        add("S6", 16, "saxs", construct, mutation, "Compound 9", "DHU", "q_range", qrange, "Angstrom^-1")
        add("S6", 16, "saxs", construct, mutation, "Compound 9", "DHU", "gnom_chi2", chi2, "unitless")
        add("S6", 16, "saxs", construct, mutation, "Compound 9", "DHU", "porod_volume", porod, "Angstrom^-3")
    return rows


def oconnor_mutagenesis_qc_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    primers = [
        ("S10", 20, "L60A", "CCGACATCACACACCTACGCGGGTGCCGATATGGAAG", "Mutate Leu60 to Ala in CRBNmidi"),
        ("S10", 20, "Q100A", "CAGACCTTACCGCTGGCGCTGTTTCACCCGCAGGAAG", "Mutate Gln100 to Ala in CRBNmidi"),
        ("S10", 20, "H378A", "CGCCCGTCTACCGAAGCCAGCTGGTTTCCAGGGTATG", "Mutate His378 to Ala in CRBNmidi and CRBNmidiL60A"),
        ("S10", 20, "H378N", "CGCCCGTCTACCGAAAACAGCTGGTTTCCAGGGTATG", "Mutate His378 to Asn in CRBNmidi"),
    ]
    for table_id, page, name, seq, introduced in primers:
        rows.append({"table_id": table_id, "pdf_page": page, "record_type": "primer", "name": name, "sequence": seq, "measurement": "mutation_introduced", "value": introduced, "unit": "", "source_extraction": "manual_from_rendered_page_20"})
    cycles = [("1", 1, 95, 1), ("2a", 15, 95, 0.5), ("2b", 15, 55, 1), ("2c", 15, 68, 8.5)]
    for seg, cyc, temp, mins in cycles:
        rows.append({"table_id": "S11", "pdf_page": 20, "record_type": "cycling_parameter", "name": seg, "sequence": "", "measurement": "cycles", "value": cyc, "unit": "cycles", "source_extraction": "pdf_text_layout_verified_by_render"})
        rows.append({"table_id": "S11", "pdf_page": 20, "record_type": "cycling_parameter", "name": seg, "sequence": "", "measurement": "temperature", "value": temp, "unit": "degC", "source_extraction": "pdf_text_layout_verified_by_render"})
        rows.append({"table_id": "S11", "pdf_page": 20, "record_type": "cycling_parameter", "name": seg, "sequence": "", "measurement": "time", "value": mins, "unit": "minutes", "source_extraction": "pdf_text_layout_verified_by_render"})
    masses = [("WT", 37435.88, 37428.77), ("H378N", 37412.84, 37406.20), ("H378A", 37369.82, 37362.47), ("Q100A", 37378.83, 37370.89), ("L60A", 37393.80, 37386.22), ("L60A H378A", 37327.74, 37318.07)]
    for mutant, pred, exp in masses:
        rows.append({"table_id": "S12", "pdf_page": 20, "record_type": "lc_ms", "name": mutant, "sequence": "", "measurement": "predicted_mass", "value": pred, "unit": "Da", "source_extraction": "pdf_text_layout_verified_by_render"})
        rows.append({"table_id": "S12", "pdf_page": 20, "record_type": "lc_ms", "name": mutant, "sequence": "", "measurement": "experimental_mass", "value": exp, "unit": "Da", "source_extraction": "pdf_text_layout_verified_by_render"})
    return rows


def mutation_residue_status(mutation: str) -> tuple[str, str]:
    mapping = {"L60A": 60, "Q100A": 100, "H378A": 378, "H378N": 378}
    residues: list[int] = []
    for token, residue in mapping.items():
        if token in mutation:
            residues.append(residue)
    window = load_primary_window()
    statuses = ["inside" if residue in window else "outside" for residue in residues]
    if not residues:
        return "", "not_applicable"
    if len(set(statuses)) == 1:
        return ";".join(str(r) for r in residues), statuses[0]
    return ";".join(str(r) for r in residues), "mixed"


def oconnor_variant_inventory_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    variants = {
        "WT": {"tables": ["S3", "S4", "S5", "S12"], "source": "baseline construct"},
        "H378N": {"tables": ["S4", "S10", "S12"], "source": "mutagenesis plus DSF and LC-MS"},
        "H378A": {"tables": ["S4", "S6", "S10", "S12"], "source": "mutagenesis plus DSF, SAXS, and LC-MS"},
        "Q100A": {"tables": ["S4", "S6", "S10", "S12"], "source": "mutagenesis plus DSF, SAXS, and LC-MS"},
        "L60A": {"tables": ["S4", "S6", "S10", "S12"], "source": "mutagenesis plus DSF, SAXS, and LC-MS"},
        "L60A H378A": {"tables": ["S4", "S6", "S10", "S12"], "source": "double mutant plus DSF, SAXS, and LC-MS"},
    }
    for variant, meta in variants.items():
        residues, status = mutation_residue_status(variant)
        rows.append({"variant": variant, "residues": residues, "primary_269_window": status, "source_tables": ";".join(meta["tables"]), "inventory_source": meta["source"], "inventory_scope": "complete for mutant constructs visible in Tables S4, S6, S10, and S12"})
    return rows


def oconnor_compound9_case_rows() -> list[dict[str, object]]:
    dsf_delta = {"WT": 14.57, "H378N": 13.14, "H378A": 10.86, "Q100A": 14.75, "L60A": 5.22, "L60A H378A": 4.65}
    dsf_tm = {"WT": 58.65, "H378N": 56.14, "H378A": 53.91, "Q100A": 58.75, "L60A": 49.53, "L60A H378A": 47.45}
    saxs_rg = {"WT": 23.43, "H378A": 22.88, "H378N": 23.80, "L60A": 24.84, "L60A H378A": 25.18, "Q100A": 23.07}
    rows: list[dict[str, object]] = []
    for variant in ["WT", "H378N", "H378A", "Q100A", "L60A", "L60A H378A"]:
        residues, status = mutation_residue_status(variant)
        rows.append({
            "compound": "Compound 9",
            "compound_class": "DHU",
            "variant": variant,
            "residues": residues,
            "primary_269_window": status,
            "dsf_tm_degC": dsf_tm.get(variant, ""),
            "dsf_delta_tm_degC": dsf_delta.get(variant, ""),
            "dsf_delta_delta_tm_vs_wt_degC": "" if variant == "WT" else dsf_delta[variant] - dsf_delta["WT"],
            "saxs_guinier_rg_angstrom": saxs_rg.get(variant, ""),
            "saxs_delta_rg_vs_wt_angstrom": "" if variant == "WT" else saxs_rg[variant] - saxs_rg["WT"],
            "binding_note": "WT CRBNmidi SPR Kd mean 107.8 nM; no mutant-resolved binding table in S1-S6/S10-S12",
            "comparison_scope": "within-assay sparse case comparison from PDF supplement; no candidate classifier or validation claim",
        })
    return rows


def residue_status_for_values(residues: list[int], window: set[int]) -> str:
    if not residues:
        return "not_applicable"
    statuses = ["inside" if residue in window else "outside" for residue in residues]
    if len(set(statuses)) == 1:
        return statuses[0]
    return "mixed"

def write_candidate_variant_comparison(candidate_path: Path, output_path: Path) -> None:
    variants = [
        {"variant": row["variant"], "residues": row["residues"], "source_assay": "O'Connor PDF supplement mutant inventory"}
        for row in oconnor_variant_inventory_rows()
        if row["variant"] != "WT"
    ]
    candidates: dict[int, list[dict[str, str]]] = {}
    if candidate_path.is_file():
        with candidate_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    residue = int(row["residue"])
                except (KeyError, TypeError, ValueError):
                    continue
                candidates.setdefault(residue, []).append(row)

    window = load_primary_window()
    rows: list[dict[str, object]] = []
    for variant in variants:
        residue_values = [int(value) for value in str(variant["residues"]).split(";") if value]
        matches = [match for residue in residue_values for match in candidates.get(residue, [])]
        if not matches:
            rows.append(
                {
                    "variant": variant["variant"],
                    "residue": ";".join(str(value) for value in residue_values),
                    "source_assay": variant["source_assay"],
                    "primary_269_window": variant.get("primary_269_window") or residue_status_for_values(residue_values, window),
                    "candidate_overlap": "none",
                    "contact_class": "",
                    "discovery_rank": "",
                    "discovery_top5": "",
                    "stable_apo_model_candidate": "",
                    "also_consistent_in_engineered_references": "",
                    "interpretation": "no overlap with frozen contact-candidate table",
                }
            )
            continue
        for match in matches:
            rows.append(
                {
                    "variant": variant["variant"],
                    "residue": ";".join(str(value) for value in residue_values),
                    "source_assay": variant["source_assay"],
                    "primary_269_window": variant.get("primary_269_window") or residue_status_for_values(residue_values, window),
                    "candidate_overlap": "same_residue",
                    "contact_class": match.get("contact_class", ""),
                    "discovery_rank": match.get("discovery_rank", ""),
                    "discovery_top5": match.get("discovery_top5", ""),
                    "stable_apo_model_candidate": match.get("stable_apo_model_candidate", ""),
                    "also_consistent_in_engineered_references": match.get(
                        "also_consistent_in_engineered_references", ""
                    ),
                    "interpretation": "sparse residue-level overlap only; no classifier applied",
                }
            )
    write_csv(
        output_path,
        rows,
        [
            "variant",
            "residue",
            "source_assay",
            "primary_269_window",
            "candidate_overlap",
            "contact_class",
            "discovery_rank",
            "discovery_top5",
            "stable_apo_model_candidate",
            "also_consistent_in_engineered_references",
            "interpretation",
        ],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="analysis root containing data/external and analysis/external",
    )
    parser.add_argument("--offline", action="store_true", help="use cached downloads only")
    parser.add_argument(
        "--max-qrg",
        type=float,
        default=1.3,
        help="maximum q*Rg for the primary Guinier fit range",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
