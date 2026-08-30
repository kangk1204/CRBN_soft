#!/usr/bin/env python3
"""Fail-closed publication grouping for the curated CRBN ensemble.

RCSB primary-citation DOIs are the default group identifiers. A missing DOI is
not treated as evidence that a deposition is an independent study. Every such
entry must be covered by the committed manual map in
``data/curation_study_overrides.csv``. The map can record a DOI verified from the
associated report or an explicit series identifier when no DOI can be verified.
This prevents incomplete citation metadata from silently increasing the
effective sample size.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUP_TABLE = ROOT / "data" / "curation_study_groups.csv"
OVERRIDE_TABLE = ROOT / "data" / "curation_study_overrides.csv"
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


def _normalise(value: str) -> str:
    return value.strip().lower()


def load_overrides(path: Path = OVERRIDE_TABLE) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    required = {"pdb", "study_group", "reason"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path}: expected columns {sorted(required)}")
    out: dict[str, str] = {}
    for row in rows:
        pdb = row["pdb"].strip().upper()
        group = _normalise(row["study_group"])
        if not pdb or pdb in out:
            raise ValueError(f"{path}: blank or duplicate PDB identifier {pdb!r}")
        if not row["reason"].strip():
            raise ValueError(f"{path}: {pdb} has no grouping rationale")
        if not (DOI_RE.match(group) or group.startswith("no_doi_series:")):
            raise ValueError(f"{path}: invalid study-group identifier {group!r}")
        out[pdb] = group
    return out


def load_study_groups(
    labels: list[str] | None = None,
    table: Path = GROUP_TABLE,
    overrides_path: Path = OVERRIDE_TABLE,
) -> dict[str, str]:
    """Return PDB-to-study mappings and reject incomplete or ambiguous metadata."""
    overrides = load_overrides(overrides_path)
    rows = list(csv.DictReader(table.open(encoding="utf-8", newline="")))
    required = {"pdb", "primary_citation_doi"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{table}: expected columns {sorted(required)}")

    raw: dict[str, str] = {}
    for row in rows:
        pdb = row["pdb"].strip().upper()
        value = _normalise(row["primary_citation_doi"])
        if not pdb or pdb in raw:
            raise ValueError(f"{table}: blank or duplicate PDB identifier {pdb!r}")
        raw[pdb] = value

    if labels is not None:
        normalized_labels = [str(x).upper() for x in labels]
        duplicates = sorted(
            {value for value in normalized_labels if normalized_labels.count(value) > 1}
        )
        if duplicates:
            raise ValueError(f"requested ensemble contains duplicate labels: {duplicates}")
    requested = set(raw if labels is None else normalized_labels)
    missing_rows = sorted(requested - set(raw))
    extra_rows = sorted(set(raw) - requested) if labels is not None else []
    if missing_rows or extra_rows:
        raise ValueError(
            "study metadata does not exactly match requested ensemble; "
            f"missing={missing_rows}, extra={extra_rows}"
        )

    groups = resolve_study_groups(raw, sorted(requested), overrides_path)
    unused = sorted(set(overrides) - requested)
    if labels is not None and unused:
        raise ValueError(f"manual study-group map contains out-of-ensemble entries: {unused}")
    return groups


def resolve_study_groups(
    raw: dict[str, str],
    labels: list[str],
    overrides_path: Path = OVERRIDE_TABLE,
) -> dict[str, str]:
    """Resolve an in-memory PDB-to-study mapping using committed overrides."""
    overrides = load_overrides(overrides_path)
    normalized_labels = [str(x).upper() for x in labels]
    duplicates = sorted(
        {value for value in normalized_labels if normalized_labels.count(value) > 1}
    )
    if duplicates:
        raise ValueError(f"requested ensemble contains duplicate labels: {duplicates}")
    requested = set(normalized_labels)
    normalised = {str(key).upper(): _normalise(str(value)) for key, value in raw.items()}
    if set(normalised) != requested:
        raise ValueError(
            "study metadata does not exactly match requested ensemble; "
            f"missing={sorted(requested - set(normalised))}, "
            f"extra={sorted(set(normalised) - requested)}"
        )

    groups: dict[str, str] = {}
    unresolved: list[str] = []
    for pdb in sorted(requested):
        value = normalised[pdb]
        if pdb in overrides:
            groups[pdb] = overrides[pdb]
        elif DOI_RE.match(value):
            groups[pdb] = value
        else:
            unresolved.append(pdb)
    if unresolved:
        raise ValueError(
            "missing primary DOI without a committed manual study group: "
            + ", ".join(unresolved)
        )
    return groups
