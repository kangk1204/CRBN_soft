from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_strengthening_figures.py"
CACHED_READY_CANDIDATES = (
    ROOT / "results" / "strengthening" / "analysis" / "figure_sources" / "figure_readiness.json",
    ROOT / "118_csbj_strengthening_20260905" / "analysis" / "figure_sources" / "figure_readiness.json",
)
READY = next((path for path in CACHED_READY_CANDIDATES if path.is_file()), CACHED_READY_CANDIDATES[0])
FIXED_TEST_TIMESTAMP = "2026-09-05T00:00:00+00:00"


def tiny_png_bytes() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_scanline = b"\x00\xff\xff\xff"
    return header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw_scanline)) + chunk(b"IEND", b"")


def load_builder_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("build_strengthening_figures_manifest_test", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "scripts"))
        sys.modules.pop("build_strengthening_figures_manifest_test", None)


def write_synthetic_package_files(base: Path) -> dict[str, str]:
    inputs = base / "inputs"
    figures = base / "figures"
    sources = base / "sources"
    inputs.mkdir()
    figures.mkdir()
    sources.mkdir()
    (inputs / "analysis.json").write_text('{"n": 2}\n', encoding="utf-8")
    (sources / "FigX_source.csv").write_text("panel,value\na,1\nb,2\n", encoding="utf-8")
    (figures / "FigX.png").write_bytes(tiny_png_bytes())
    (figures / "FigX.pdf").write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    (figures / "FigX.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><path d="M0 0h1v1z"/></svg>\n',
        encoding="utf-8",
    )
    return {
        "input": str(inputs / "analysis.json"),
        "source": str(sources / "FigX_source.csv"),
        "png": str(figures / "FigX.png"),
        "pdf": str(figures / "FigX.pdf"),
        "svg": str(figures / "FigX.svg"),
    }


def synthetic_record(paths: dict[str, str]) -> dict[str, object]:
    return {
        "figure": "FigX",
        "status": "rendered",
        "inputs": [paths["input"]],
        "source_data": paths["source"],
        "outputs": {"png": paths["png"], "pdf": paths["pdf"], "svg": paths["svg"]},
        "panel_mapping": {"a": "synthetic manifest emission panel"},
    }


def synthetic_context(module, tmp_path: Path):
    return module.BuildContext(
        input_root=tmp_path / "input_root",
        output_dir=tmp_path / "figures",
        source_dir=tmp_path / "manifest_out",
        require_all=True,
    )


def test_package_manifest_emission_is_always_run_with_synthetic_files(tmp_path):
    module = load_builder_module()
    paths = write_synthetic_package_files(tmp_path)
    ctx = synthetic_context(module, tmp_path)

    first = module.package_manifests([synthetic_record(paths)], ctx, generated_at=FIXED_TEST_TIMESTAMP)
    first_bytes = first[0].read_bytes()
    second = module.package_manifests([synthetic_record(paths)], ctx, generated_at=FIXED_TEST_TIMESTAMP)
    assert second[0].read_bytes() == first_bytes

    manifest = json.loads(first[0].read_text(encoding="utf-8"))
    assert manifest["figure_id"] == "FigX"
    assert manifest["generated_at"] == FIXED_TEST_TIMESTAMP
    assert manifest["validation"]["status"] == "structural_pass"
    assert manifest["validation"]["scientific_validation"] == "not_assessed"
    assert manifest["known_limitations"] == [module.MANIFEST_LIMITATION]
    assert manifest["source_data"][0]["table"] == {"rows": 2, "columns": 2, "header": ["panel", "value"]}
    assert {Path(output["path"]).suffix for output in manifest["outputs"]} == {".png", ".pdf", ".svg"}
    assert any(output.get("png_width") == 1 and output.get("png_height") == 1 for output in manifest["outputs"])
    assert any(output.get("pdf_check") == "header_and_eof" for output in manifest["outputs"])
    assert any(output.get("svg_check") == "parsed_no_active_content" for output in manifest["outputs"])


def test_package_manifest_rejects_tampered_png(tmp_path):
    module = load_builder_module()
    paths = write_synthetic_package_files(tmp_path)
    Path(paths["png"]).write_bytes(b"not a png\n")

    with pytest.raises(module.NotReady, match="invalid PNG signature"):
        module.package_manifests([synthetic_record(paths)], synthetic_context(module, tmp_path), generated_at=FIXED_TEST_TIMESTAMP)


def test_package_manifest_rejects_tampered_source_csv(tmp_path):
    module = load_builder_module()
    paths = write_synthetic_package_files(tmp_path)
    Path(paths["source"]).write_text("panel,value\na,1\nb\n", encoding="utf-8")

    with pytest.raises(module.NotReady, match="inconsistent delimited columns"):
        module.package_manifests([synthetic_record(paths)], synthetic_context(module, tmp_path), generated_at=FIXED_TEST_TIMESTAMP)


@pytest.mark.skipif(not READY.is_file(), reason="cached strengthening figure readiness is not available")
def test_package_manifests_are_emitted_from_cached_readiness_records(tmp_path):
    module = load_builder_module()
    readiness = json.loads(READY.read_text(encoding="utf-8"))
    records = readiness["records"]
    cached_root = READY.parents[3]
    ctx = module.BuildContext(
        input_root=cached_root,
        output_dir=cached_root / "manuscript" / "figures",
        source_dir=tmp_path,
        require_all=True,
    )

    first = module.package_manifests(records, ctx, generated_at=FIXED_TEST_TIMESTAMP)
    first_bytes = {path.name: path.read_bytes() for path in first}
    second = module.package_manifests(records, ctx, generated_at=FIXED_TEST_TIMESTAMP)
    second_bytes = {path.name: path.read_bytes() for path in second}

    assert [path.name for path in first] == [f"{stem}_package_manifest.json" for stem in module.FIGURE_STEMS]
    assert first_bytes == second_bytes

    fig5 = json.loads((tmp_path / "Fig5_package_manifest.json").read_text(encoding="utf-8"))
    assert fig5["generated_at"] == FIXED_TEST_TIMESTAMP
    assert fig5["validation"] == {
        "status": "structural_pass",
        "scientific_validation": "not_assessed",
        "commands": [module.figure_manifest_command(next(record for record in records if record["figure"] == "Fig5"), ctx)],
    }
    assert fig5["source_data"][0]["path"].endswith("Fig5_source.csv")
    assert fig5["source_data"][0]["sha256"] == module.sha256_file(ROOT / fig5["source_data"][0]["path"])
    assert fig5["source_data"][0]["table"]["rows"] == 12871
    assert {Path(output["path"]).suffix for output in fig5["outputs"]} == {".png", ".pdf", ".svg"}
    assert any(output.get("png_width") and output.get("png_height") for output in fig5["outputs"])
    assert any(output.get("pdf_check") == "header_and_eof" for output in fig5["outputs"])
    assert any(output.get("svg_check") == "parsed_no_active_content" for output in fig5["outputs"])

    figs6 = json.loads((tmp_path / "FigS6_package_manifest.json").read_text(encoding="utf-8"))
    assert any("rigid-domain subspace dimension" in claim for claim in figs6["panel_claims"])
    assert figs6["known_limitations"] == [module.MANIFEST_LIMITATION]
