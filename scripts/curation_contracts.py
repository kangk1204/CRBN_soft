"""Pure curation-contract helpers shared by coordinate rebuild workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import shlex


CRBN_ACCESSION = "Q96SW2"
CRBN_CANONICAL_LENGTH = 442
DDB1_ACCESSION = "Q16531"
DDB1_CANONICAL_LENGTH = 1140
CANONICAL_LENGTHS = {
    CRBN_ACCESSION: CRBN_CANONICAL_LENGTH,
    DDB1_ACCESSION: DDB1_CANONICAL_LENGTH,
}


def reference_accessions(polymer_entity: Mapping) -> set[str]:
    """Return normalized external sequence accessions for one polymer entity."""
    identifiers = polymer_entity.get("rcsb_polymer_entity_container_identifiers") or {}
    return {
        str(record.get("database_accession", "")).upper()
        for record in identifiers.get("reference_sequence_identifiers") or []
        if record.get("database_accession")
    }


def chains_for_exact_accession(entry: Mapping, accession: str) -> list[str]:
    """Return auth-chain ids only from entities exactly mapped to ``accession``."""
    target = accession.upper()
    chains: set[str] = set()
    for entity in entry.get("polymer_entities") or []:
        if target not in reference_accessions(entity):
            continue
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        chains.update(str(chain) for chain in identifiers.get("auth_asym_ids") or [])
    return sorted(chains)


def describes_ddb1(entry: Mapping) -> bool:
    """Detect the acronym and standard RCSB long name independently of accession."""
    descriptions = [
        ((entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "").lower()
        for entity in entry.get("polymer_entities") or []
    ]
    return any(
        "ddb1" in description or "dna damage-binding protein 1" in description
        for description in descriptions
    )


def choose_primary_chain(
    chains: Sequence[str],
    pdb_id: str,
    overrides: Mapping[str, str] | None = None,
) -> str:
    """Choose a recorded override, otherwise the lowest exact-entity auth-chain id."""
    candidates = sorted(set(chains))
    if not candidates:
        raise ValueError(f"{pdb_id}: no exact-accession CRBN chain is available")
    chosen = (overrides or {}).get(pdb_id.upper(), candidates[0])
    if chosen not in candidates:
        raise ValueError(
            f"{pdb_id}: recorded chain {chosen} is not among exact-accession CRBN "
            f"chains {candidates}"
        )
    return chosen


def cif_loop_rows(text: str, category: str) -> list[dict[str, str]]:
    """Parse a scalar mmCIF loop such as ``struct_ref_seq``."""
    prefix = f"_{category}."
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        cursor = index + 1
        headers: list[str] = []
        while cursor < len(lines) and lines[cursor].lstrip().startswith(prefix):
            headers.append(lines[cursor].strip().split(".", 1)[1])
            cursor += 1
        if not headers:
            continue
        tokens: list[str] = []
        rows: list[dict[str, str]] = []
        while cursor < len(lines):
            raw = lines[cursor]
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or stripped == "loop_" or (
                stripped.startswith("_") and not tokens
            ):
                break
            if raw.startswith(";"):
                raise ValueError(f"unexpected multiline field in {category} loop")
            tokens.extend(shlex.split(raw, posix=True))
            while len(tokens) >= len(headers):
                record, tokens = tokens[: len(headers)], tokens[len(headers) :]
                rows.append(dict(zip(headers, record)))
            cursor += 1
        if tokens:
            raise ValueError(f"incomplete {category} loop row ({len(tokens)} tokens)")
        return rows
    raise ValueError(f"mmCIF has no {category} loop")


def accession_ranges(
    cif_text: str,
    accession: str,
    *,
    required: bool = True,
) -> list[tuple[int, int]]:
    """Return unique UniProt mapping intervals from the raw ``struct_ref_seq`` loop."""
    groups = accession_range_groups(cif_text, accession, required=required)
    ranges = {interval for group in groups for interval in group}
    if not ranges and required:
        raise ValueError(f"mmCIF has no struct_ref_seq mapping for {accession}")
    return sorted(ranges)


def accession_range_groups(
    cif_text: str,
    accession: str,
    *,
    required: bool = True,
) -> list[tuple[tuple[int, int], ...]]:
    """Return validated mapping intervals grouped by author-chain/entity mapping.

    Complementary fragments from different entities must never be unioned into a
    fictitious full-length construct. Rows without an entity/strand key are kept
    separate, which is the conservative interpretation of incomplete metadata.
    """
    target = accession.upper()
    canonical_length = CANONICAL_LENGTHS.get(target)
    grouped: dict[tuple[str, str], set[tuple[int, int]]] = {}
    matched = 0
    for row_index, row in enumerate(cif_loop_rows(cif_text, "struct_ref_seq")):
        if row.get("pdbx_db_accession", "").upper() != target:
            continue
        matched += 1
        try:
            start = int(row["db_align_beg"])
            end = int(row["db_align_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid {target} struct_ref_seq interval on row {row_index + 1}") from exc
        if start < 1 or end < start or (canonical_length is not None and end > canonical_length):
            ceiling = f"{canonical_length}" if canonical_length is not None else "the accession length"
            raise ValueError(
                f"invalid {target} struct_ref_seq interval {start}-{end}; expected "
                f"1 <= start <= end <= {ceiling}"
            )
        strand = row.get("pdbx_strand_id", "").strip()
        if strand not in {"", ".", "?"}:
            key = ("strand", ",".join(sorted(part.strip() for part in strand.split(","))))
        else:
            key = ("unassociated_row", str(row_index))
        grouped.setdefault(key, set()).add((start, end))
    if not matched and required:
        raise ValueError(f"mmCIF has no struct_ref_seq mapping for {accession}")
    return sorted(tuple(sorted(intervals)) for intervals in grouped.values())


def _covered_positions(ranges: Sequence[tuple[int, int]]) -> set[int]:
    return {residue for start, end in ranges for residue in range(start, end + 1)}


def _mapping_is_canonical(
    groups: Sequence[Sequence[tuple[int, int]]], canonical_length: int
) -> bool:
    canonical = set(range(1, canonical_length + 1))
    return bool(groups) and all(_covered_positions(group) == canonical for group in groups)


def _maximum_mapping_coverage(groups: Sequence[Sequence[tuple[int, int]]]) -> int:
    return max((len(_covered_positions(group)) for group in groups), default=0)


def _entity_summary(entry: Mapping, accession: str) -> tuple[list[int], list[str]]:
    lengths: set[int] = set()
    mutations: set[str] = set()
    for entity in entry.get("polymer_entities") or []:
        if accession.upper() not in reference_accessions(entity):
            continue
        length = (entity.get("entity_poly") or {}).get("rcsb_sample_sequence_length")
        if length is not None:
            lengths.add(int(length))
        mutation = (entity.get("rcsb_polymer_entity") or {}).get("pdbx_mutation")
        if mutation and mutation not in ("?", "."):
            parts = [part.strip() for part in str(mutation).split(",") if part.strip()]
            mutations.update(parts)
    return sorted(lengths), sorted(mutations)


def _range_text(ranges: Sequence[tuple[int, int]]) -> str:
    return ";".join(f"{start}-{end}" for start, end in ranges)


def exact_construct_flags(entry: Mapping, cif_text: str) -> str:
    """Classify construct changes from exact UniProt mappings and entity metadata."""
    crbn_groups = accession_range_groups(cif_text, CRBN_ACCESSION)
    crbn_ranges = accession_ranges(cif_text, CRBN_ACCESSION)
    crbn_coverage = _maximum_mapping_coverage(crbn_groups)
    crbn_lengths, crbn_mutations = _entity_summary(entry, CRBN_ACCESSION)
    crbn_full = _mapping_is_canonical(crbn_groups, CRBN_CANONICAL_LENGTH)

    ddb1_present = bool(chains_for_exact_accession(entry, DDB1_ACCESSION))
    ddb1_groups = accession_range_groups(
        cif_text,
        DDB1_ACCESSION,
        required=False,
    ) if ddb1_present else []
    ddb1_ranges = accession_ranges(
        cif_text,
        DDB1_ACCESSION,
        required=False,
    ) if ddb1_present else []
    ddb1_coverage = _maximum_mapping_coverage(ddb1_groups)
    ddb1_lengths, ddb1_mutations = _entity_summary(entry, DDB1_ACCESSION)
    ddb1_full = _mapping_is_canonical(ddb1_groups, DDB1_CANONICAL_LENGTH)

    flags: list[str] = []
    if not crbn_full:
        flags.append(f"CRBN_UniProt_mapping:{_range_text(crbn_ranges)}")
    if crbn_mutations:
        flags.append("CRBN_mutation:" + " | ".join(crbn_mutations))
    if any(length > crbn_coverage for length in crbn_lengths):
        flags.append("CRBN_extra_sequence_or_tag")
    if ddb1_ranges and not ddb1_full:
        flags.append(f"DDB1_UniProt_mapping:{_range_text(ddb1_ranges)}")
    if ddb1_mutations:
        flags.append("DDB1_mutation:" + " | ".join(ddb1_mutations))
    if ddb1_ranges and any(length > ddb1_coverage for length in ddb1_lengths):
        flags.append("DDB1_extra_sequence_or_tag")
    return ";".join(flags)
