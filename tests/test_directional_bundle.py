from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, *, required: bool = True):
    path = ROOT / "scripts" / f"{name}.py"
    if not path.is_file():
        if required:
            pytest.fail(f"Required script is missing: {path}")
        pytest.skip(f"{name}.py is not present in this checkout")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(path: str, raw: bytes) -> dict[str, object]:
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write_bundle(path: Path, files: dict[str, bytes], *, aliases=None, extra: dict[str, bytes] | None = None) -> None:
    aliases = aliases or []
    manifest = {
        "public_commit": "1" * 40,
        "files": [record(name, raw) for name, raw in files.items()],
        "aliases": aliases,
        "excluded_generated_mode_vectors": [],
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
        for name, raw in (extra or {}).items():
            archive.writestr(name, raw)
        archive.writestr("README.md", "guide\n")
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest))


def test_directional_stage_rejects_extra_and_traversal_before_write(tmp_path):
    stage = load_script("stage_directional_bundle")
    bundle = tmp_path / "bundle.zip"
    write_bundle(bundle, {"directional/analysis/table.csv": b"a\n"}, extra={"../escape": b"bad"})

    with pytest.raises(ValueError):
        stage.stage(bundle, tmp_path / "repo", tmp_path / "out")

    assert not (tmp_path / "repo").exists()
    assert not (tmp_path / "out").exists()


def test_directional_stage_rejects_hash_mismatch_before_write(tmp_path):
    stage = load_script("stage_directional_bundle")
    bundle = tmp_path / "bundle.zip"
    raw = b"a\n"
    manifest = {
        "public_commit": "1" * 40,
        "files": [{"path": "data/example.csv", "bytes": len(raw), "sha256": "bad"}],
        "aliases": [],
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("data/example.csv", raw)
        archive.writestr("README.md", "guide\n")
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest))

    with pytest.raises(ValueError, match="Bundle hash mismatch"):
        stage.stage(bundle, tmp_path / "repo", tmp_path / "out")

    assert not (tmp_path / "repo").exists()


def test_directional_stage_restores_reference_inputs_and_legacy_cif_alias(tmp_path):
    stage = load_script("stage_directional_bundle")
    bundle = tmp_path / "bundle.zip"
    cif = b"cif bytes\n"
    files = {
        "data/directional_reference_inputs/candidate_universe.csv": b"id\n1\n",
        "data/_cif_cache/9SFM.cif.gz": cif,
        "directional/analysis/mechanics/models_all.csv": b"model\nflexible\n",
        "public_code/scripts/run_directional_mechanics.py": b"print('ok')\n",
    }
    aliases = [
        {
            "path": "directional/data/external/9SFM.cif.gz",
            "source": "data/_cif_cache/9SFM.cif.gz",
            "sha256": hashlib.sha256(cif).hexdigest(),
        }
    ]
    write_bundle(bundle, files, aliases=aliases)

    result = stage.stage(bundle, tmp_path / "repo", tmp_path / "out")

    assert result["aliases"] == 1
    assert (tmp_path / "repo/data/directional_reference_inputs/candidate_universe.csv").is_file()
    assert (tmp_path / "repo/data/_cif_cache/9SFM.cif.gz").read_bytes() == cif
    assert (tmp_path / "out/data/external/9SFM.cif.gz").read_bytes() == cif
    assert (tmp_path / "out/analysis/mechanics/models_all.csv").is_file()
    assert not (tmp_path / "repo/public_code").exists()


def make_minimal_package(tmp_path: Path, build) -> Path:
    package = tmp_path / "119_pkg"
    manuscript = package / "manuscript/manuscript"
    manuscript.mkdir(parents=True)
    commit = "2" * 40
    url = f"https://github.com/kangk1204/CRBN_soft/tree/{commit}\n"
    for name in ("CRBN_manuscript.md", "S_supporting_information.md", "CRBN_cover_letter.md"):
        (manuscript / name).write_text(url, encoding="utf-8")
    for name in ("CRBN_manuscript.docx", "S_supporting_information.docx", "CRBN_cover_letter.docx"):
        (manuscript / name).write_bytes(b"docx\n")
    for name in ("CRBN_manuscript.pdf", "S_supporting_information.pdf", "CRBN_cover_letter.pdf"):
        (manuscript / name).write_bytes(b"%PDF-1.4\n%%EOF\n")
    (package / "manuscript").mkdir(exist_ok=True)
    (package / "manuscript/CRBN_supplementary_tables.xlsx").write_bytes(b"xlsx\n")
    for folder in (package / "manuscript/figures", package / "manuscript/figures/vector"):
        folder.mkdir(parents=True, exist_ok=True)
    for stem in [*(f"Fig{i}" for i in range(1, 6)), *(f"FigS{i}" for i in range(1, 7))]:
        (package / "manuscript/figures" / f"{stem}.png").write_bytes(b"png\n")
        (package / "manuscript/figures/vector" / f"{stem}.pdf").write_bytes(b"pdf\n")
        (package / "manuscript/figures/vector" / f"{stem}.svg").write_text("<svg/>\n", encoding="utf-8")
    (package / "protocol").mkdir(parents=True)
    (package / "protocol/baseline_manifest.json").write_text(
        json.dumps({"preserved_tracked_118_files": [], "config_sha256": None}), encoding="utf-8"
    )
    (package / "protocol/frozen_config.json").write_text("{}\n", encoding="utf-8")
    (package / "protocol/REPRODUCE_BUNDLE.md").write_text("# Guide\n", encoding="utf-8")
    code_zip = package / "protocol/public_code_snapshot.zip"
    with zipfile.ZipFile(code_zip, "w") as archive:
        archive.writestr("scripts/run_directional_mechanics.py", "print('ok')\n")
    public = {
        "repository": "https://github.com/kangk1204/CRBN_soft.git",
        "commit": commit,
        "path": "protocol/public_code_snapshot.zip",
        "sha256": build.digest(code_zip),
    }
    (package / "protocol/public_snapshot.json").write_text(json.dumps(public), encoding="utf-8")
    mode = package / "analysis/mode_paths/8CVP_15A_uniform"
    mode.mkdir(parents=True)
    (mode / "alpha_0.npz").write_bytes(b"large vector\n")
    (mode / "directional_mode_scores.csv").write_text("mode,overlap\n1,0.7\n", encoding="utf-8")
    (mode / "completion.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    (package / "analysis/mechanics").mkdir(parents=True)
    (package / "analysis/mechanics/claim_gates.json").write_text('{"gate":"pass"}\n', encoding="utf-8")
    (package / "analysis/figure_sources").mkdir(parents=True)
    (package / "analysis/figure_sources/Fig3_source.csv").write_text("panel,value\na,1\n", encoding="utf-8")
    (package / "data/directional_reference_inputs").mkdir(parents=True)
    (package / "data/directional_reference_inputs/manifest.json").write_text('{"role":"test"}\n', encoding="utf-8")
    return package


def test_directional_builder_excludes_mode_npz_and_keeps_csv_gates(tmp_path, monkeypatch):
    build = load_script("build_directional_package", required=False)
    package = make_minimal_package(tmp_path, build)
    root_inputs = tmp_path / "repo_root/data/directional_reference_inputs"
    root_inputs.mkdir(parents=True)
    (root_inputs / "candidate_universe.csv").write_text("id\n1\n", encoding="utf-8")
    monkeypatch.setattr(build, "ROOT", tmp_path / "repo_root")

    build.main(["--package", str(package), "--no-frozen-bundle"])

    bundle = package / "submission" / build.BUNDLE_NAME
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))

    assert "directional/analysis/mode_paths/8CVP_15A_uniform/alpha_0.npz" not in names
    assert "directional/analysis/mode_paths/mode_vector_inventory.csv" in names
    assert "directional/analysis/mode_paths/8CVP_15A_uniform/directional_mode_scores.csv" in names
    assert "directional/analysis/mechanics/claim_gates.json" in names
    assert "data/directional_reference_inputs/candidate_universe.csv" in names
    assert manifest["excluded_generated_mode_vectors"][0]["path"].endswith("alpha_0.npz")
    assert manifest["stage_output_dir"] == "results/directional_mechanics"

    stage = load_script("stage_directional_bundle")
    result = stage.stage(bundle, tmp_path / "clean_repo", tmp_path / "clean_results")
    assert result["excluded_mode_vectors"] == 1
    assert (tmp_path / "clean_results/analysis/mode_paths/mode_vector_inventory.csv").is_file()
    assert (tmp_path / "clean_results/analysis/mode_paths/8CVP_15A_uniform/directional_mode_scores.csv").is_file()
    assert not (tmp_path / "clean_results/analysis/mode_paths/8CVP_15A_uniform/alpha_0.npz").exists()
    assert (tmp_path / "clean_repo/data/directional_reference_inputs/candidate_universe.csv").is_file()


def test_rebuild_directional_documents_merges_legends_and_registers_sources(tmp_path):
    rebuild = load_script("rebuild_directional_documents", required=False)
    package = tmp_path / "119_pkg"
    (package / "protocol").mkdir(parents=True)
    (package / "protocol/workbook_sources.json").write_text(
        json.dumps(
            [
                {
                    "sheet": "Existing",
                    "path": "analysis/existing.csv",
                    "description": "existing retained workbook table",
                }
            ]
        ),
        encoding="utf-8",
    )
    (package / "analysis/existing.csv").parent.mkdir(parents=True)
    (package / "analysis/existing.csv").write_text("a\n1\n", encoding="utf-8")
    mechanics = package / "analysis/mechanics"
    mechanics.mkdir(parents=True)
    (mechanics / "models_all.csv").write_text("model,value\nm,1\n", encoding="utf-8")
    (mechanics / "comparisons_all.csv").write_text("model,value\nm,2\n", encoding="utf-8")
    (mechanics / "claim_gates.json").write_text('{"gate":{"passed":true}}\n', encoding="utf-8")
    contact = package / "analysis/contact_roles"
    contact.mkdir(parents=True)
    for name in ("groups_all.csv", "ridge_all.csv", "candidate_summary.csv"):
        (contact / name).write_text("id,value\n1,0.1\n", encoding="utf-8")
    condition = contact / "8CVP_15A_uniform"
    condition.mkdir()
    for name in ("role_factor_effects.csv", "factor_effects.csv", "edges.csv"):
        (condition / name).write_text("id,value\n1,0.2\n", encoding="utf-8")
    mode = package / "analysis/mode_paths/8CVP_15A_uniform"
    mode.mkdir(parents=True)
    (mode / "directional_mode_scores.csv").write_text("mode,overlap\n1,0.7\n", encoding="utf-8")
    (mode / "alpha_0.npz").write_bytes(b"vector\n")
    external = package / "analysis/external"
    external.mkdir(parents=True)
    for name in (
        "candidate_9sfm_spatial_correspondence.csv",
        "9sfm_a1ceg_contact_candidate_overlap.csv",
        "blood_2025_variant_observations.csv",
        "oconnor_2025_variant_candidate_reuse.csv",
    ):
        (external / name).write_text("id,value\n1,0.3\n", encoding="utf-8")
    legacy = package / "analysis/figure_sources"
    directional = package / "analysis/directional_figure_sources"
    legacy.mkdir(parents=True)
    directional.mkdir(parents=True)
    legacy_blocks = ["# Old", ""]
    for stem in [*(f"Fig{i}" for i in range(1, 6)), *(f"FigS{i}" for i in range(1, 7))]:
        legacy_blocks.extend([f"## {stem}", "", f"old {stem}", ""])
    (legacy / "LEGENDS.md").write_text("\n".join(legacy_blocks), encoding="utf-8")
    (directional / "LEGENDS.md").write_text(
        "# New\n\n## Fig3\n\nnew Fig3\n\n## Fig4\n\nnew Fig4\n\n## Fig5\n\nnew Fig5\n",
        encoding="utf-8",
    )

    rebuild.build_workbook_sources(package)
    rebuild.ensure_registry(package)
    rebuild.merge_directional_legends(package)

    registry = json.loads((package / "protocol/workbook_sources.json").read_text(encoding="utf-8"))
    sheets = {row["sheet"] for row in registry}
    assert "Existing" in sheets
    assert {row["sheet"] for row in rebuild.DIRECTIONAL_WORKBOOK_ENTRIES}.issubset(sheets)
    for row in registry:
        assert (package / row["path"]).is_file()
    inventory = (package / "analysis/workbook_sources/directional_mode_vector_inventory.csv").read_text(encoding="utf-8")
    assert "excluded_generated_mode_vector_npz" in inventory
    merged = (legacy / "LEGENDS.md").read_text(encoding="utf-8")
    assert merged.count("\n## Fig") == 11
    assert "new Fig3" in merged and "old Fig2" in merged and "old FigS6" in merged
