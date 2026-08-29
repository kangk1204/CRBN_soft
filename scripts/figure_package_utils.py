#!/usr/bin/env python3
"""Small deterministic-output helpers for supplementary figure packages."""

from __future__ import annotations

import html
import math
from numbers import Real
import zipfile
from pathlib import Path


_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def prepare_figure_dirs(root: Path | str = ".") -> tuple[Path, Path, Path]:
    """Create and return the main, vector, and prepared-panel directories."""
    figures = Path(root) / "figures"
    vector = figures / "vector"
    panels = figures / "panels"
    vector.mkdir(parents=True, exist_ok=True)
    panels.mkdir(parents=True, exist_ok=True)
    return figures, vector, panels


def require_prepared_panel(path: Path | str, generator: str) -> Path:
    """Fail with the exact command needed to create a required raster panel."""
    panel = Path(path)
    if not panel.is_file():
        raise FileNotFoundError(
            f"Required prepared panel is missing: {panel}. Generate it first with: {generator}"
        )
    return panel


def require_rigid_null_schema(payload: object, source: str = "data/assembly_rigid_null.json") -> dict:
    """Require matched-subspace statistics for every reported rigid null."""
    if not isinstance(payload, dict) or not isinstance(payload.get("rigid_domain_null"), dict):
        raise RuntimeError(
            f"{source} has no rigid_domain_null object. Rebuild it with: "
            "python scripts/assembly_rigid_null.py"
        )
    rigid = payload["rigid_domain_null"]
    required_model_fields = (
        "internal_dim",
        "subspace_capture_of_transition",
        "p_exact",
        "z",
        "null_mean",
        "null_sd",
        "null_p95",
        "null_max",
        "observed_direction_cosine_in_subspace",
        "observed_projected_mode1_overlap",
    )
    model_fields = {
        model: required_model_fields
        for model in (
            "two_block",
            "three_block",
            "bond_length_preserving_boundary",
            "equal_displacement_boundary",
        )
    }
    missing = []
    for model, fields in model_fields.items():
        record = rigid.get(model)
        if not isinstance(record, dict):
            missing.append(model)
            continue
        missing.extend(f"{model}.{field}" for field in fields if field not in record)
        if record.get("null_method") != "exact_analytic_beta":
            missing.append(f"{model}.null_method=exact_analytic_beta")
        if not isinstance(record.get("null_distribution"), dict):
            missing.append(f"{model}.null_distribution")
        if "p_empirical" in record and not math.isclose(
            record["p_empirical"], record["p_exact"], rel_tol=0.0, abs_tol=1e-15
        ):
            missing.append(f"{model}.p_empirical alias differs from p_exact")
    if missing:
        raise RuntimeError(
            f"{source} uses an obsolete or incomplete rigid-null schema; missing "
            f"{', '.join(missing)}. Rebuild it with: python scripts/assembly_rigid_null.py"
        )
    numeric_fields = [
        rigid[model][field]
        for model, fields in model_fields.items()
        for field in fields
    ]
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in numeric_fields):
        raise RuntimeError(
            f"{source} has non-numeric matched-subspace fields. Rebuild it with: "
            "python scripts/assembly_rigid_null.py"
        )
    if rigid.get("n_draws") != 0 or rigid.get("seed") is not None:
        raise RuntimeError(
            f"{source} still advertises a sampled directional null; rebuild it with: "
            "python scripts/assembly_rigid_null.py"
        )
    return rigid


def clean_svg(path: Path) -> None:
    """Normalize trailing whitespace while preserving the generated SVG."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    cleaned: list[str] = []
    skipping_doctype = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<!DOCTYPE"):
            skipping_doctype = not stripped.endswith(">")
            continue
        if skipping_doctype:
            if stripped.endswith(">"):
                skipping_doctype = False
            continue
        cleaned.append(line.rstrip())
    path.write_text(
        "\n".join(cleaned) + "\n",
        encoding="utf-8",
    )


def save_figure_set(fig, root: Path, stem: str) -> tuple[Path, Path, Path]:
    """Write matching PNG, PDF and SVG outputs with stable metadata."""
    figures = root / "figures"
    vector = figures / "vector"
    vector.mkdir(parents=True, exist_ok=True)

    png = figures / f"{stem}.png"
    pdf = vector / f"{stem}.pdf"
    svg = vector / f"{stem}.svg"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "CRBN supplementary figure builder"},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": stem,
            "Creator": "CRBN supplementary figure builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": stem, "Date": None},
    )
    clean_svg(svg)
    return png, pdf, svg


def _zip_member(name: str, text: str) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info, text.encode("utf-8")


def write_legend_docx(path: Path, legend: str) -> None:
    """Write a minimal, deterministic DOCX containing one figure legend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = f"<w:p><w:r><w:t>{html.escape(legend)}</w:t></w:r></w:p>"
    members = [
        _zip_member(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
""",
        ),
        _zip_member(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
""",
        ),
        _zip_member(
            "word/document.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {document}
    <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
  </w:body>
</w:document>
""",
        ),
    ]
    with zipfile.ZipFile(path, "w") as docx:
        for info, payload in members:
            docx.writestr(info, payload)
