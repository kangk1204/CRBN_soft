#!/usr/bin/env python3
"""Small deterministic-output helpers for supplementary figure packages."""

from __future__ import annotations

import html
import zipfile
from pathlib import Path


_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


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
