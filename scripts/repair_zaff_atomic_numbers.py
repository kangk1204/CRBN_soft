#!/usr/bin/env python3
"""Repair ZAFF CY1/ZN1 atomic-number metadata in an Amber prmtop copy.

Only the fixed-width values in %FLAG ATOMIC_NUMBER are changed. The utility is
intentionally narrow: it accepts exactly one ZN1/ZN atom with Amber type ZN and
four CY1/SG atoms with Amber type S1, validates their masses and Zn-SG bonds,
and rejects all other nonpositive atomic numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ZN_MASS = 65.4
SG_MASS = 32.06
MASS_TOLERANCE = 0.02
TARGETS = {
    ("ZN1", "ZN", "ZN"): {"atomic_number": 30, "mass": ZN_MASS, "count": 1},
    ("CY1", "SG", "S1"): {"atomic_number": 16, "mass": SG_MASS, "count": 4},
}


@dataclass(frozen=True)
class Section:
    name: str
    start: int
    end: int
    format_line: str
    data_start: int
    data_end: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sections(text: str) -> dict[str, Section]:
    starts: list[tuple[str, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("%FLAG "):
            starts.append((line.split()[1], offset))
        offset += len(line)
    sections: dict[str, Section] = {}
    for index, (name, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(text)
        block = text[start:end]
        lines = block.splitlines(keepends=True)
        if len(lines) < 2 or not lines[1].startswith("%FORMAT"):
            raise ValueError(f"prmtop section {name} lacks %FORMAT line")
        data_start = start + len(lines[0]) + len(lines[1])
        sections[name] = Section(name, start, end, lines[1].rstrip("\n"), data_start, end)
    return sections


def require_sections(sections: dict[str, Section], names: Iterable[str]) -> None:
    missing = [name for name in names if name not in sections]
    if missing:
        raise ValueError(f"prmtop lacks required sections: {missing}")


def parse_i8(text: str, section: Section) -> list[int]:
    values = []
    payload = text[section.data_start : section.data_end]
    for line in payload.splitlines():
        if not line:
            continue
        for start in range(0, len(line), 8):
            field = line[start : start + 8]
            if field.strip():
                values.append(int(field))
    return values


def parse_e16(text: str, section: Section) -> list[float]:
    values = []
    payload = text[section.data_start : section.data_end]
    for line in payload.splitlines():
        if not line:
            continue
        for start in range(0, len(line), 16):
            field = line[start : start + 16]
            if field.strip():
                values.append(float(field.replace("D", "E")))
    return values


def parse_a4(text: str, section: Section) -> list[str]:
    values = []
    payload = text[section.data_start : section.data_end]
    for line in payload.splitlines():
        for start in range(0, len(line), 4):
            field = line[start : start + 4]
            if field:
                values.append(field.strip())
    return values


def format_i8_like(section_text: str, values: list[int]) -> str:
    lines = section_text.splitlines(keepends=True)
    output: list[str] = []
    value_index = 0
    for line in lines:
        has_newline = line.endswith("\n")
        raw = line[:-1] if has_newline else line
        if not raw:
            output.append(line)
            continue
        rebuilt = []
        for start in range(0, len(raw), 8):
            field = raw[start : start + 8]
            if field.strip():
                if value_index >= len(values):
                    raise ValueError("too many ATOMIC_NUMBER fields while formatting")
                rebuilt.append(f"{values[value_index]:8d}")
                value_index += 1
            else:
                rebuilt.append(field)
        output.append("".join(rebuilt) + ("\n" if has_newline else ""))
    if value_index != len(values):
        raise ValueError("not all ATOMIC_NUMBER values were formatted")
    return "".join(output)


def atom_residue_labels(atom_count: int, residue_labels: list[str], residue_pointers: list[int]) -> list[str]:
    labels = [None] * atom_count
    starts = [pointer - 1 for pointer in residue_pointers]
    ends = [*starts[1:], atom_count]
    for label, start, end in zip(residue_labels, starts, ends):
        if start < 0 or end > atom_count or start >= end:
            raise ValueError("invalid RESIDUE_POINTER layout")
        for index in range(start, end):
            labels[index] = label
    if any(label is None for label in labels):
        raise ValueError("could not assign every atom to a residue")
    return [str(label) for label in labels]


def bond_pairs(values: list[int]) -> set[tuple[int, int]]:
    if len(values) % 3 != 0:
        raise ValueError("bond section length is not a multiple of 3")
    pairs = set()
    for i in range(0, len(values), 3):
        a = values[i] // 3
        b = values[i + 1] // 3
        pairs.add(tuple(sorted((a, b))))
    return pairs


def repair_text(text: str) -> tuple[str, dict[str, object]]:
    sections = parse_sections(text)
    require_sections(
        sections,
        [
            "POINTERS",
            "ATOM_NAME",
            "AMBER_ATOM_TYPE",
            "MASS",
            "ATOMIC_NUMBER",
            "RESIDUE_LABEL",
            "RESIDUE_POINTER",
            "BONDS_INC_HYDROGEN",
            "BONDS_WITHOUT_HYDROGEN",
        ],
    )
    pointers = parse_i8(text, sections["POINTERS"])
    atom_count = pointers[0]
    names = parse_a4(text, sections["ATOM_NAME"])
    types = parse_a4(text, sections["AMBER_ATOM_TYPE"])
    masses = parse_e16(text, sections["MASS"])
    atomic_numbers = parse_i8(text, sections["ATOMIC_NUMBER"])
    residue_labels = parse_a4(text, sections["RESIDUE_LABEL"])
    residue_pointers = parse_i8(text, sections["RESIDUE_POINTER"])
    if not (len(names) == len(types) == len(masses) == len(atomic_numbers) == atom_count):
        raise ValueError("atom-aligned prmtop sections do not match NATOM")
    labels = atom_residue_labels(atom_count, residue_labels, residue_pointers)

    target_indices: dict[tuple[str, str, str], list[int]] = {key: [] for key in TARGETS}
    unknown_nonpositive = []
    wrong_positive = []
    mass_failures = []
    repaired = atomic_numbers.copy()
    changes = []
    for index, (residue, atom_name, atom_type, mass, atomic_number) in enumerate(
        zip(labels, names, types, masses, atomic_numbers)
    ):
        key = (residue, atom_name, atom_type)
        if key in TARGETS:
            spec = TARGETS[key]
            expected_number = int(spec["atomic_number"])
            expected_mass = float(spec["mass"])
            target_indices[key].append(index)
            if abs(mass - expected_mass) > MASS_TOLERANCE:
                mass_failures.append({"index": index, "key": key, "mass": mass, "expected": expected_mass})
            if atomic_number > 0 and atomic_number != expected_number:
                wrong_positive.append({"index": index, "key": key, "atomic_number": atomic_number})
            if atomic_number != expected_number:
                changes.append({"index": index, "key": key, "from": atomic_number, "to": expected_number})
                repaired[index] = expected_number
        elif atomic_number <= 0:
            unknown_nonpositive.append(
                {
                    "index": index,
                    "residue": residue,
                    "atom_name": atom_name,
                    "atom_type": atom_type,
                    "mass": mass,
                    "atomic_number": atomic_number,
                }
            )
    count_failures = [
        {"key": key, "observed": len(indices), "expected": int(TARGETS[key]["count"])}
        for key, indices in target_indices.items()
        if len(indices) != int(TARGETS[key]["count"])
    ]
    failures = []
    if count_failures:
        failures.append({"target_count_failures": count_failures})
    if mass_failures:
        failures.append({"mass_failures": mass_failures})
    if wrong_positive:
        failures.append({"wrong_prior_positive_atomic_numbers": wrong_positive})
    if unknown_nonpositive:
        failures.append({"unexpected_nonpositive_atomic_numbers": unknown_nonpositive[:20]})
    all_pairs = bond_pairs(parse_i8(text, sections["BONDS_INC_HYDROGEN"])) | bond_pairs(
        parse_i8(text, sections["BONDS_WITHOUT_HYDROGEN"])
    )
    zn_indices = target_indices[("ZN1", "ZN", "ZN")]
    sg_indices = target_indices[("CY1", "SG", "S1")]
    zn_sg_pairs = sorted(pair for pair in all_pairs if zn_indices and zn_indices[0] in pair and (pair[0] in sg_indices or pair[1] in sg_indices))
    if len(zn_sg_pairs) != 4 or sorted(i for pair in zn_sg_pairs for i in pair if i in sg_indices) != sorted(sg_indices):
        failures.append({"zn_sg_bond_failure": {"observed_pairs": zn_sg_pairs, "zn_indices": zn_indices, "sg_indices": sg_indices}})
    if failures:
        raise ValueError(json.dumps({"status": "fail", "failures": failures}, indent=2))

    atomic_section = sections["ATOMIC_NUMBER"]
    old_payload = text[atomic_section.data_start : atomic_section.data_end]
    new_payload = format_i8_like(old_payload, repaired)
    repaired_text = text[: atomic_section.data_start] + new_payload + text[atomic_section.data_end :]
    outside_unchanged = text[: atomic_section.data_start] + text[atomic_section.data_end :] == repaired_text[: atomic_section.data_start] + repaired_text[atomic_section.data_end :]
    if not outside_unchanged or len(text) != len(repaired_text):
        raise AssertionError("repair changed bytes outside ATOMIC_NUMBER or changed file length")
    return repaired_text, {
        "status": "complete",
        "atom_count": atom_count,
        "changes": changes,
        "change_count": len(changes),
        "target_counts": {"ZN1/ZN/typeZN": len(zn_indices), "CY1/SG/typeS1": len(sg_indices)},
        "zn_sg_bond_pairs": zn_sg_pairs,
        "preservation": {"changed_section": "ATOMIC_NUMBER", "file_length_preserved": len(text) == len(repaired_text), "outside_atomic_number_unchanged": True},
    }


def openmm_system_xml_sha256(prmtop: Path) -> str:
    from openmm import XmlSerializer, app

    amber_top = app.AmberPrmtopFile(str(prmtop))
    system = amber_top.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)
    return hashlib.sha256(XmlSerializer.serialize(system).encode("utf-8")).hexdigest()


def repair_file(input_prmtop: Path, output_prmtop: Path, *, xml_check: bool = False) -> dict[str, object]:
    if input_prmtop.resolve() == output_prmtop.resolve() or output_prmtop.exists():
        raise ValueError("Metadata repair requires a new output copy; preserve the input and earlier attempts")
    text = input_prmtop.read_text(encoding="ascii")
    repaired_text, report = repair_text(text)
    xml_before = openmm_system_xml_sha256(input_prmtop) if xml_check else None
    output_prmtop.parent.mkdir(parents=True, exist_ok=True)
    output_prmtop.write_text(repaired_text, encoding="ascii")
    xml_after = openmm_system_xml_sha256(output_prmtop) if xml_check else None
    if xml_check and xml_before != xml_after:
        raise ValueError("OpenMM System XML changed after ATOMIC_NUMBER metadata repair")
    if xml_check:
        report["openmm_system_xml"] = {
            "unchanged": True,
            "sha256_before": xml_before,
            "sha256_after": xml_after,
            "create_system_options": {
                "nonbondedMethod": "NoCutoff",
                "constraints": None,
                "removeCMMotion": False,
            },
        }
    report.update(
        {
            "input_prmtop": str(input_prmtop),
            "output_prmtop": str(output_prmtop),
            "input_sha256": sha256_file(input_prmtop),
            "output_sha256": sha256_file(output_prmtop),
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-prmtop", required=True, type=Path)
    parser.add_argument("--output-prmtop", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--xml-check", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = repair_file(args.input_prmtop, args.output_prmtop, xml_check=args.xml_check)
    except Exception as exc:
        payload = {"status": "error", "reason": str(exc)}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
