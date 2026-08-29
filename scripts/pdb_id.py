"""Strict identifiers for RCSB cache paths and API URLs."""

from __future__ import annotations

import re


_PDB_ID = re.compile(r"[A-Za-z0-9]{4}\Z")
_ENTITY_ID = re.compile(r"([A-Za-z0-9]{4})_([1-9][0-9]*)\Z")


def validate_pdb_id(value: str) -> str:
    """Return an uppercase four-character PDB ID or reject the input."""
    if not isinstance(value, str) or _PDB_ID.fullmatch(value) is None:
        raise ValueError(f"invalid PDB ID: {value!r}")
    return value.upper()


def validate_polymer_entity_id(value: str) -> tuple[str, str]:
    """Return validated ``(PDB ID, entity number)`` components."""
    if not isinstance(value, str):
        raise ValueError(f"invalid RCSB polymer entity ID: {value!r}")
    match = _ENTITY_ID.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid RCSB polymer entity ID: {value!r}")
    return validate_pdb_id(match.group(1)), match.group(2)
