from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import strengthen_external


def test_parse_sas_dat_and_weighted_guinier_fit_recover_rg() -> None:
    rg = 2.35
    i0 = 0.42
    q = np.linspace(0.004, 0.12, 80)
    intensity = i0 * np.exp(-(q**2) * (rg**2) / 3.0)
    sigma = np.full_like(q, 0.001)
    text = "\n".join(
        ["Sample description: synthetic"]
        + [f"{qv:.8e} {iv:.8e} {sv:.8e}" for qv, iv, sv in zip(q, intensity, sigma)]
    )

    arr, header = strengthen_external.parse_sas_dat(text)
    fit = strengthen_external.weighted_guinier_fit(arr[:, 0], arr[:, 1], arr[:, 2])

    assert header == ["Sample description: synthetic"]
    assert fit["n_points"] >= 8
    assert fit["low_q_qa"] == "pass"
    assert abs(float(fit["rg_nm"]) - rg) < 1e-6
    assert abs(float(fit["i0"]) - i0) < 1e-6


def test_residue_window_classification_keeps_outside_primary_residues() -> None:
    primary_window = {100, 156, 378, 380}
    rows = strengthen_external.residue_window_classification(
        [59, 60, 100, 156, 350, 351, 378, 380],
        primary_window,
    )
    status = {row["residue"]: row["primary_269_window"] for row in rows}

    assert status[59] == "outside"
    assert status[60] == "outside"
    assert status[350] == "outside"
    assert status[351] == "outside"
    assert status[100] == "inside"
    assert status[156] == "inside"
    assert status[378] == "inside"
    assert status[380] == "inside"


def test_infer_saxs_unit_evidence_prefers_explicit_dat_metadata() -> None:
    explicit = [
        'REMARK 265          q : defined in inverse Angstroms',
        '{"sas_scan":{"momentum_transfer_units":"inverse Angstroms"}}',
    ]
    assert strengthen_external.infer_saxs_unit_evidence(explicit) == "explicit_dat_header_inverse_angstroms"
    assert strengthen_external.infer_saxs_unit_evidence(["Sample description: chromixs"]) == "not_found_in_dat_header"


def test_candidate_variant_comparison_writes_all_evaluable_variants(tmp_path: Path, monkeypatch) -> None:
    candidate = tmp_path / "candidate_robustness.csv"
    candidate.write_text(
        "residue,contact_class,discovery_rank,discovery_top5,"
        "stable_apo_model_candidate,also_consistent_in_engineered_references\n"
        "100,interface,12,false,true,false\n"
    )
    out = tmp_path / "comparison.csv"
    window = tmp_path / "crbn_residue_window.csv"
    residues = list(range(100, 368)) + [378]
    window.write_text(
        "index,author_resnum\n" + "".join(f"{i},{residue}\n" for i, residue in enumerate(residues)),
        encoding="utf-8",
    )
    monkeypatch.setattr(strengthen_external, "PRIMARY_WINDOW_PATH", window)

    strengthen_external.write_candidate_variant_comparison(candidate, out)

    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    variants = {row["variant"] for row in rows}
    assert variants == {"H378N", "H378A", "Q100A", "L60A", "L60A H378A"}
    assert "residue_156" not in variants
    row100 = next(row for row in rows if row["variant"] == "Q100A")
    assert row100["candidate_overlap"] == "same_residue"
    assert row100["contact_class"] == "interface"
    assert next(row for row in rows if row["variant"] == "L60A")["primary_269_window"] == "outside"
    assert next(row for row in rows if row["variant"] == "L60A H378A")["primary_269_window"] == "mixed"



def test_guinier_interval_inventory_does_not_use_gnom_or_incomplete_point_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path
    zip_path = data_dir / "SASX.zip"
    import zipfile

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("SASX/pddf/SASX.out", "Angular range: 0.01 to 0.30\n")
    headers = [
        '{"sas_result":{"quinier_point_min":0,"guinier_point_max":3}}',
    ]

    row = strengthen_external.guinier_interval_inventory_row("SASX", "test", headers, data_dir)

    assert row["supplied_guinier_range_identified"] is False
    assert row["candidate_guinier_point_metadata_present"] is True
    assert row["used_for_refit"] is False
    assert "below the 8-point refit minimum" in row["reason"]
    assert row["gnom_angular_range_present"] is True
    assert "not used as a Guinier fitting interval" in row["reason"]

def test_offline_run_reports_missing_sources_without_candidate_comparison(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "run"
    args = argparse.Namespace(output_dir=out, offline=True, max_qrg=1.3)
    window = tmp_path / "crbn_residue_window.csv"
    residues = list(range(100, 368)) + [378]
    window.write_text(
        "index,author_resnum\n" + "".join(f"{i},{residue}\n" for i, residue in enumerate(residues)),
        encoding="utf-8",
    )
    monkeypatch.setattr(strengthen_external, "PRIMARY_WINDOW_PATH", window)

    assert strengthen_external.run(args) == 0

    summary = strengthen_external.json.loads(
        (out / "analysis" / "external" / "external_strengthening_summary.json").read_text()
    )
    assert summary["candidate_comparison_performed"] is False
    assert summary["saxs_refit_rows"] == 0

    with (out / "analysis" / "external" / "missing_files_report.csv").open() as handle:
        missing = list(csv.DictReader(handle))
    assert any(row["source_id"] == "oconnor_machine_readable_source_tables" for row in missing)
    assert any(row["source_id"] == "sasbdb_SASDU52_dat" for row in missing)


def test_offline_run_preserves_previous_availability_snapshot(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "run"
    analysis_dir = out / "analysis" / "external"
    data_dir = out / "data" / "external"
    analysis_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    cached_html = b"<html>cached</html>"
    (data_dir / "Kroupova_2024_article.html").write_bytes(cached_html)
    previous_manifest = [
        {
            "source_id": "kroupova_article_html",
            "url": strengthen_external.KROUPOVA_ARTICLE,
            "local_path": "/old/Kroupova_2024_article.html",
            "retrieved_utc": "2026-09-05T00:00:00Z",
            "status": "http_200",
            "bytes": len(cached_html),
            "sha256": hashlib.sha256(cached_html).hexdigest(),
            "error": None,
        },
        {
            "source_id": "pdb_9sun_cif",
            "url": "https://files.rcsb.org/download/9SUN.cif",
            "local_path": None,
            "retrieved_utc": "2026-09-05T00:00:00Z",
            "status": "http_404",
            "bytes": None,
            "sha256": None,
            "error": "HTTP 404",
        },
    ]
    (analysis_dir / "sources_manifest.json").write_text(json.dumps(previous_manifest), encoding="utf-8")
    window = tmp_path / "crbn_residue_window.csv"
    residues = list(range(100, 368)) + [378]
    window.write_text(
        "index,author_resnum\n" + "".join(f"{i},{residue}\n" for i, residue in enumerate(residues)),
        encoding="utf-8",
    )
    monkeypatch.setattr(strengthen_external, "PRIMARY_WINDOW_PATH", window)

    args = argparse.Namespace(output_dir=out, offline=True, max_qrg=1.3)
    assert strengthen_external.run(args) == 0

    manifest = {
        row["source_id"]: row
        for row in json.loads((analysis_dir / "sources_manifest.json").read_text())
    }
    assert manifest["pdb_9sun_cif"]["status"] == "http_404"
    assert manifest["pdb_9sun_cif"]["retrieved_utc"] == "2026-09-05T00:00:00Z"
    assert manifest["pdb_9sun_cif"]["current_run_status"] == "not_checked_offline"
    assert manifest["kroupova_article_html"]["status"] == "http_200"
    assert manifest["kroupova_article_html"]["current_run_status"] == "cached"

    with (analysis_dir / "missing_files_report.csv").open() as handle:
        missing = list(csv.DictReader(handle))
    pdb_row = next(row for row in missing if row["source_id"] == "pdb_9sun_cif")
    assert pdb_row["reason"] == "HTTP 404"


def test_offline_run_rejects_tampered_cached_file(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "run"
    analysis_dir = out / "analysis" / "external"
    data_dir = out / "data" / "external"
    analysis_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    original_html = b"<html>original</html>"
    (data_dir / "Kroupova_2024_article.html").write_text("<html>tampered</html>", encoding="utf-8")
    previous_manifest = [
        {
            "source_id": "kroupova_article_html",
            "url": strengthen_external.KROUPOVA_ARTICLE,
            "local_path": "/old/Kroupova_2024_article.html",
            "retrieved_utc": "2026-09-05T00:00:00Z",
            "status": "http_200",
            "bytes": len(original_html),
            "sha256": hashlib.sha256(original_html).hexdigest(),
            "error": None,
        },
    ]
    (analysis_dir / "sources_manifest.json").write_text(json.dumps(previous_manifest), encoding="utf-8")
    window = tmp_path / "crbn_residue_window.csv"
    residues = list(range(100, 368)) + [378]
    window.write_text(
        "index,author_resnum\n" + "".join(f"{i},{residue}\n" for i, residue in enumerate(residues)),
        encoding="utf-8",
    )
    monkeypatch.setattr(strengthen_external, "PRIMARY_WINDOW_PATH", window)

    args = argparse.Namespace(output_dir=out, offline=True, max_qrg=1.3)
    with pytest.raises(RuntimeError, match="cached source kroupova_article_html differs from frozen manifest"):
        strengthen_external.run(args)
