#!/usr/bin/env python3
"""Handoff utilities between Amber preparation, minimization, and technical pilot.

Subcommands
-----------
restraints
    Map observed-input heavy atoms from the PDBFixer repair inventory onto the
    Amber topology order and emit observed_heavy_restraints.json.

qualify
    After a preparation minimization has completed, validate hashes, terminal
    chemistry, peptide geometry, Zn coordination, backbone chirality, and finite
    minimized positions; export a minimized Amber restart through ParmEd and
    write the runner qualification JSON plus a mapping JSON containing
    initial_core_nm.

This is chemistry-technical preparation only. It makes no production MD,
equilibration, temperature, or response-convergence claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results" / "atomistic"
DEFAULT_HEAVY_MAPPING = DEFAULT_ROOT / "heavy_model" / "atom_residue_mapping.csv"
DEFAULT_AMBER_DIR = DEFAULT_ROOT / "amber_joint"
DEFAULT_RESTRAINTS = DEFAULT_AMBER_DIR / "observed_heavy_restraints.json"
EXPECTED_OBSERVED_HEAVY = 10112
PROTEIN_CAP_NAMES = {"ACE", "NME"}
HIS_NAMES = {"HIS", "HID", "HIE", "HIP"}
ZN_NAMES = {"ZN", "ZN1"}
CYS_ZAFF_NAMES = {"CYS", "CY1"}
SOLVENT_OR_ION_NAMES = {"HOH", "WAT", "Na+", "Cl-", "NA", "CL", "Na", "Cl"}
PEPTIDE_BOND_BROAD_A = (1.1, 1.7)
PEPTIDE_BOND_IDEAL_A = (1.2, 1.5)
ZN_SG_BOND_BROAD_NM = (0.19, 0.30)
CHIRALITY_MIN_VOLUME_NM3 = 1e-4
NEUTRAL_TOL = 1e-5
AMBER_CHARGE_SCALE = 18.2223
CRBN_ZN_CYS_RESIDUES = {323, 326, 391, 394}


@dataclass(frozen=True)
class PdbAtom:
    index: int
    name: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float
    element: str

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.resname)

    @property
    def xyz_a(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def is_heavy(self) -> bool:
        element = self.element.strip().upper()
        return element != "H" and not self.name.upper().startswith("H")


@dataclass(frozen=True)
class TopAtom:
    index: int
    name: str
    res_index: int
    resname: str


@dataclass(frozen=True)
class TopResidue:
    index: int
    name: str
    atom_indices: list[int]
    source_key: tuple[str, int, str] | None = None


@dataclass(frozen=True)
class ParsedPrmtop:
    atoms: list[TopAtom]
    residues: list[TopResidue]
    bonds: set[tuple[int, int]]
    charges_e: list[float]

    @property
    def total_charge(self) -> float:
        return float(sum(self.charges_e))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def parse_pdb_atoms(path: Path) -> list[PdbAtom]:
    atoms: list[PdbAtom] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append(
            PdbAtom(
                index=len(atoms),
                name=line[12:16].strip(),
                resname=line[17:20].strip(),
                chain=line[21:22].strip(),
                resseq=int(line[22:26]),
                x=float(line[30:38]),
                y=float(line[38:46]),
                z=float(line[46:54]),
                element=(line[76:78].strip() if len(line) >= 78 else line[12:16].strip()[0]).upper(),
            )
        )
    if not atoms:
        raise ValueError(f"No PDB atoms found: {path}")
    return atoms


def residue_order(atoms: Sequence[PdbAtom]) -> list[tuple[str, int, str]]:
    order: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for atom in atoms:
        key = atom.residue_key
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def _parse_prmtop_sections(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("%FLAG "):
            current = line.split(maxsplit=1)[1].strip()
            sections[current] = []
        elif line.startswith("%FORMAT"):
            continue
        elif current:
            sections[current].append(line.rstrip("\n"))
    return sections


def _fixed_width_strings(lines: Sequence[str], width: int = 4) -> list[str]:
    values: list[str] = []
    for line in lines:
        values.extend(line[i : i + width].strip() for i in range(0, len(line), width) if line[i : i + width].strip())
    return values


def _numbers(lines: Sequence[str], cast: Any) -> list[Any]:
    out: list[Any] = []
    for line in lines:
        for token in line.split():
            out.append(cast(token.replace("D", "E")))
    return out


def parse_prmtop(path: Path) -> ParsedPrmtop:
    sections = _parse_prmtop_sections(path)
    for required in ("ATOM_NAME", "CHARGE", "RESIDUE_LABEL", "RESIDUE_POINTER"):
        if required not in sections:
            raise ValueError(f"PRMTOP missing %FLAG {required}: {path}")
    atom_names = _fixed_width_strings(sections["ATOM_NAME"])
    charges = [float(x) / AMBER_CHARGE_SCALE for x in _numbers(sections["CHARGE"], float)]
    residue_labels = _fixed_width_strings(sections["RESIDUE_LABEL"])
    pointers = [int(x) for x in _numbers(sections["RESIDUE_POINTER"], int)]
    if len(charges) != len(atom_names):
        raise ValueError("PRMTOP atom-name and charge counts differ")
    if len(pointers) != len(residue_labels):
        raise ValueError("PRMTOP residue-label and pointer counts differ")
    residues: list[TopResidue] = []
    atom_to_res: dict[int, int] = {}
    for r_index, start_1 in enumerate(pointers):
        start = start_1 - 1
        end = (pointers[r_index + 1] - 1) if r_index + 1 < len(pointers) else len(atom_names)
        if not (0 <= start < end <= len(atom_names)):
            raise ValueError("Invalid PRMTOP RESIDUE_POINTER ordering")
        atom_indices = list(range(start, end))
        residues.append(TopResidue(r_index, residue_labels[r_index], atom_indices))
        for atom_index in atom_indices:
            atom_to_res[atom_index] = r_index
    atoms = [TopAtom(i, name, atom_to_res[i], residues[atom_to_res[i]].name) for i, name in enumerate(atom_names)]
    bonds: set[tuple[int, int]] = set()
    for flag in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        values = [int(x) for x in _numbers(sections.get(flag, []), int)]
        if len(values) % 3 != 0:
            raise ValueError(f"PRMTOP {flag} does not contain triples")
        for i in range(0, len(values), 3):
            a, b = values[i] // 3, values[i + 1] // 3
            if a != b:
                bonds.add(tuple(sorted((a, b))))
    return ParsedPrmtop(atoms, residues, bonds, charges)


def residue_name_compatible(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    if expected in HIS_NAMES and actual in HIS_NAMES:
        return True
    if expected in CYS_ZAFF_NAMES and actual in CYS_ZAFF_NAMES:
        return True
    if expected in ZN_NAMES and actual in ZN_NAMES:
        return True
    return False


def attach_residue_keys(topology: ParsedPrmtop, prepared_order: Sequence[tuple[str, int, str]]) -> ParsedPrmtop:
    if len(topology.residues) < len(prepared_order):
        raise ValueError("Amber topology has fewer residues than amber_input_renamed.pdb")
    residues: list[TopResidue] = []
    for residue in topology.residues:
        source_key = prepared_order[residue.index] if residue.index < len(prepared_order) else None
        if source_key is not None and not residue_name_compatible(source_key[2], residue.name):
            raise ValueError(f"Residue identity mismatch at {source_key}: PRMTOP has {residue.name}")
        residues.append(TopResidue(residue.index, residue.name, residue.atom_indices, source_key))
    atoms = [TopAtom(atom.index, atom.name, atom.res_index, residues[atom.res_index].name) for atom in topology.atoms]
    return ParsedPrmtop(atoms, residues, topology.bonds, topology.charges_e)


def residue_by_key(topology: ParsedPrmtop, key: tuple[str, int, str], *, allow_compatible: bool = True) -> TopResidue:
    matches = []
    for residue in topology.residues:
        if residue.source_key is None:
            continue
        if residue.source_key[:2] == key[:2] and (residue.source_key[2] == key[2] or (allow_compatible and residue_name_compatible(key[2], residue.source_key[2]))):
            matches.append(residue)
    if len(matches) != 1:
        raise ValueError(f"Expected one topology residue for {key}, found {len(matches)}")
    return matches[0]


def atom_index_in_residue(topology: ParsedPrmtop, residue: TopResidue, atom_name: str) -> int:
    matches = [idx for idx in residue.atom_indices if topology.atoms[idx].name == atom_name]
    if len(matches) != 1:
        key = residue.source_key or ("?", residue.index + 1, residue.name)
        raise ValueError(f"Expected one atom {atom_name} in residue {key}, found {len(matches)}")
    return matches[0]


def bond_exists(topology: ParsedPrmtop, a: int, b: int) -> bool:
    return tuple(sorted((int(a), int(b)))) in topology.bonds


def read_heavy_mapping(path: Path, expected_observed_heavy: int | None) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"chain", "residue_number", "residue_name", "atom", "source_status"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Heavy mapping lacks required columns: {sorted(required)}")
    observed = [r for r in rows if r["source_status"] == "observed_input"]
    if expected_observed_heavy is not None and len(observed) != expected_observed_heavy:
        raise ValueError(f"Expected {expected_observed_heavy} observed_input heavy atoms, found {len(observed)}")
    return observed


def select_observed_rows_for_topology(observed_rows: Sequence[dict[str, str]], topology: ParsedPrmtop, assembly: str) -> list[dict[str, str]]:
    if assembly != "isolated":
        return list(observed_rows)
    present = {(res.source_key[0], res.source_key[1]) for res in topology.residues if res.source_key}
    selected = []
    for row in observed_rows:
        key = (row["chain"], int(row["residue_number"]))
        if key in present:
            selected.append(row)
    if not selected:
        raise ValueError("isolated assembly selected no observed heavy atoms from the full audit CSV")
    unsupported = [row for row in selected if row["chain"] not in {"B", "C"}]
    if unsupported:
        raise ValueError(f"isolated assembly selected unsupported non-CRBN/Zn rows: {unsupported[:5]}")
    return selected


def terminal_checks(topology: ParsedPrmtop, assembly: str) -> dict[str, Any]:
    checks: dict[str, Any] = {"assembly": assembly, "status": "pass", "failures": []}

    def fail(message: str) -> None:
        checks["status"] = "fail"
        checks["failures"].append(message)

    b63 = residue_by_key(topology, ("B", 63, "ACE"))
    b64 = residue_by_key(topology, ("B", 64, "MET"))
    b428 = residue_by_key(topology, ("B", 428, "ASP"))
    b429 = residue_by_key(topology, ("B", 429, "NME"))
    ace_c, met_n = atom_index_in_residue(topology, b63, "C"), atom_index_in_residue(topology, b64, "N")
    asp_c, nme_n = atom_index_in_residue(topology, b428, "C"), atom_index_in_residue(topology, b429, "N")
    checks["crbn_cap_bonds"] = {
        "B63_ACE_C_to_B64_MET_N": bond_exists(topology, ace_c, met_n),
        "B428_ASP_C_to_B429_NME_N": bond_exists(topology, asp_c, nme_n),
    }
    for label, ok in checks["crbn_cap_bonds"].items():
        if not ok:
            fail(f"missing terminal peptide bond {label}")
    if assembly != "isolated":
        a1 = residue_by_key(topology, ("A", 1, "MET"))
        a1140_matches = [res for res in topology.residues if res.source_key and res.source_key[0] == "A" and res.source_key[1] == 1140]
        if len(a1140_matches) != 1:
            fail(f"Expected one DDB1 A1140 residue, found {len(a1140_matches)}")
            a1140 = None
        else:
            a1140 = a1140_matches[0]
        h_names = {topology.atoms[i].name for i in a1.atom_indices}
        checks["ddb1_termini"] = {"A1_has_H1_H2_H3": {"H1", "H2", "H3"} <= h_names, "A1140_residue": list(a1140.source_key) if a1140 else None, "A1140_has_OXT": bool(a1140 and any(topology.atoms[i].name == "OXT" for i in a1140.atom_indices))}
        if not checks["ddb1_termini"]["A1_has_H1_H2_H3"]:
            fail("DDB1 A1 lacks terminal H1/H2/H3")
        if not checks["ddb1_termini"]["A1140_has_OXT"]:
            fail("DDB1 A1140 lacks OXT")
    else:
        if any(res.source_key and res.source_key[0] == "A" for res in topology.residues):
            fail("isolated assembly contains DDB1 chain A residues")
        checks["ddb1_termini"] = "not_applicable_isolated"
    checks["total_charge_e"] = topology.total_charge
    checks["total_charge_neutral_tol"] = NEUTRAL_TOL
    if abs(topology.total_charge) > NEUTRAL_TOL:
        fail(f"total charge is not neutral within {NEUTRAL_TOL}: {topology.total_charge}")
    return checks


def map_observed_heavy_to_topology(heavy_rows: Sequence[dict[str, str]], topology: ParsedPrmtop) -> list[int]:
    indices: list[int] = []
    for row in heavy_rows:
        key = (row["chain"], int(row["residue_number"]), row["residue_name"])
        residue = residue_by_key(topology, key)
        indices.append(atom_index_in_residue(topology, residue, row["atom"]))
    if len(set(indices)) != len(indices):
        raise ValueError("Observed-heavy mapping produced duplicate Amber atom indices")
    return indices


def build_restraints(
    *,
    heavy_mapping: Path,
    amber_pdb: Path,
    prmtop: Path,
    output: Path,
    assembly: str = "joint",
    expected_observed_heavy: int | None = EXPECTED_OBSERVED_HEAVY,
) -> dict[str, Any]:
    prepared_atoms = parse_pdb_atoms(amber_pdb)
    order = residue_order(prepared_atoms)
    topology = attach_residue_keys(parse_prmtop(prmtop), order)
    observed_rows = read_heavy_mapping(heavy_mapping, expected_observed_heavy)
    selected_rows = select_observed_rows_for_topology(observed_rows, topology, assembly)
    observed_indices = map_observed_heavy_to_topology(selected_rows, topology)
    tchecks = terminal_checks(topology, assembly)
    if tchecks["status"] != "pass":
        raise ValueError(f"Terminal/topology checks failed: {tchecks['failures']}")
    payload = {
        "schema_version": "1.0",
        "status": "pass",
        "scope": "observed-input heavy-atom restraint index map; chemistry technical only; no MD production claim",
        "production_ready": False,
        "assembly": assembly,
        "observed_heavy_indices": observed_indices,
        "restrain_indices": observed_indices,
        "observed_heavy_count": len(observed_indices),
        "selected_observed_heavy_count": len(observed_indices),
        "full_observed_heavy_count": len(observed_rows),
        "expected_observed_heavy_count": expected_observed_heavy,
        "atom_count": len(topology.atoms),
        "residue_count_mapped_from_amber_pdb": len(order),
        "terminal_checks": tchecks,
        "zaff_scope": {
            "status": "metadata_only",
            "allowed_residue_identity_mappings": {"HIS": sorted(HIS_NAMES), "CYS_CY1": sorted(CYS_ZAFF_NAMES), "ZN_ZN1": sorted(ZN_NAMES)},
            "cy1_residues": [list(res.source_key) for res in topology.residues if res.source_key and (res.name == "CY1" or res.source_key[2] == "CY1")],
            "zn_residues": [list(res.source_key) for res in topology.residues if res.source_key and residue_name_compatible(res.source_key[2], "ZN")],
        },
        "sources": {
            "heavy_mapping": {"path": str(heavy_mapping), "sha256": sha256_file(heavy_mapping)},
            "amber_pdb": {"path": str(amber_pdb), "sha256": sha256_file(amber_pdb)},
            "prmtop": {"path": str(prmtop), "sha256": sha256_file(prmtop)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
    }
    write_json(output, payload)
    return payload


def load_positions_npy(path: Path, label: str) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
        raise ValueError(f"{label} must be a finite Nx3 array")
    return arr


def load_box_npy(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=float)
    if arr.shape != (3, 3) or not np.isfinite(arr).all() or abs(float(np.linalg.det(arr))) <= 0:
        raise ValueError("box vectors must be a finite non-singular 3x3 array")
    return arr


def peptide_bond_checks(topology: ParsedPrmtop, positions_nm: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    residues = [res for res in topology.residues if res.source_key and res.source_key[2] not in (SOLVENT_OR_ION_NAMES | ZN_NAMES)]
    by_chain: dict[str, list[TopResidue]] = {}
    for res in residues:
        assert res.source_key is not None
        by_chain.setdefault(res.source_key[0], []).append(res)
    for chain, chain_residues in by_chain.items():
        chain_residues.sort(key=lambda r: r.source_key[1])  # type: ignore[index]
        for left, right in zip(chain_residues, chain_residues[1:]):
            lk, rk = left.source_key, right.source_key
            assert lk is not None and rk is not None
            if rk[1] != lk[1] + 1:
                continue
            try:
                c = atom_index_in_residue(topology, left, "C")
                n = atom_index_in_residue(topology, right, "N")
            except ValueError as exc:
                failures.append(f"missing C/N atom for consecutive peptide check {lk}->{rk}: {exc}")
                continue
            if not bond_exists(topology, c, n):
                failures.append(f"missing peptide C-N bond for consecutive residues {lk}->{rk}")
                continue
            distance_a = 10.0 * float(np.linalg.norm(positions_nm[c] - positions_nm[n]))
            row = {"chain": chain, "left": list(lk), "right": list(rk), "C_index": c, "N_index": n, "distance_A": distance_a, "broad_bounds_A": list(PEPTIDE_BOND_BROAD_A), "ideal_reference_A": list(PEPTIDE_BOND_IDEAL_A)}
            rows.append(row)
            if not (PEPTIDE_BOND_BROAD_A[0] <= distance_a <= PEPTIDE_BOND_BROAD_A[1]):
                failures.append(f"peptide C-N distance outside broad technical bounds for {lk}->{rk}: {distance_a:.3f} A")
    for a, b in topology.bonds:
        ra, rb = topology.residues[topology.atoms[a].res_index], topology.residues[topology.atoms[b].res_index]
        if not (ra.source_key and rb.source_key) or ra.source_key[0] == rb.source_key[0]:
            continue
        names = {topology.atoms[a].name, topology.atoms[b].name}
        if names == {"C", "N"}:
            failures.append(f"cross-chain peptide-like C-N bond: {ra.source_key}->{rb.source_key}")
    return {"status": "pass" if not failures else "fail", "checked_bond_count": len(rows), "bonds": rows, "failures": failures}


def raw_zn_sg_checks(topology: ParsedPrmtop, positions_nm: np.ndarray) -> dict[str, Any]:
    failures: list[str] = []
    zn_atoms: list[tuple[TopResidue, int]] = []
    sg_atoms: list[tuple[TopResidue, int]] = []
    for residue in topology.residues:
        if residue.source_key is None:
            continue
        chain, resseq, resname = residue.source_key
        if residue_name_compatible(resname, "ZN") or residue.name in ZN_NAMES:
            try:
                zn_atoms.append((residue, atom_index_in_residue(topology, residue, "ZN")))
            except ValueError as exc:
                failures.append(str(exc))
        if chain == "B" and resseq in CRBN_ZN_CYS_RESIDUES and (residue.name in CYS_ZAFF_NAMES or resname in CYS_ZAFF_NAMES):
            try:
                sg_atoms.append((residue, atom_index_in_residue(topology, residue, "SG")))
            except ValueError as exc:
                failures.append(str(exc))
    if len(zn_atoms) != 1:
        failures.append(f"expected exactly one Zn atom for CRBN Zn-site check, found {len(zn_atoms)}")
    if len(sg_atoms) != 4:
        failures.append(f"expected four CRBN Zn-site cysteine SG atoms {sorted(CRBN_ZN_CYS_RESIDUES)}, found {len(sg_atoms)}")
    distances = []
    if len(zn_atoms) == 1:
        zn_res, zn_idx = zn_atoms[0]
        for sg_res, sg_idx in sg_atoms:
            distance_nm = float(np.linalg.norm(positions_nm[sg_idx] - positions_nm[zn_idx]))
            row = {"zn_residue": list(zn_res.source_key), "zn_index": zn_idx, "sg_residue": list(sg_res.source_key), "sg_index": sg_idx, "raw_distance_nm": distance_nm, "bounds_nm": list(ZN_SG_BOND_BROAD_NM)}
            distances.append(row)
            if not (ZN_SG_BOND_BROAD_NM[0] <= distance_nm <= ZN_SG_BOND_BROAD_NM[1]):
                failures.append(f"raw Zn-SG distance outside broad technical bounds for {sg_res.source_key}: {distance_nm:.4f} nm")
    return {"status": "pass" if not failures else "fail", "scope": "raw unwrapped bonded Zn-SG distances; no periodic minimum-image correction", "distances": distances, "failures": failures}


def signed_chirality(n: np.ndarray, ca: np.ndarray, c: np.ndarray, cb: np.ndarray) -> float:
    return float(np.dot(np.cross(n - ca, c - ca), cb - ca))


def chirality_checks(topology: ParsedPrmtop, pre_nm: np.ndarray, post_nm: np.ndarray) -> dict[str, Any]:
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    initial_invalid: list[list[Any]] = []
    post_invalid: list[list[Any]] = []
    for residue in topology.residues:
        if residue.source_key is None or residue.name == "GLY" or residue.source_key[2] == "GLY":
            continue
        names = {topology.atoms[i].name: i for i in residue.atom_indices}
        if not {"N", "CA", "C", "CB"} <= set(names):
            continue
        pre = signed_chirality(pre_nm[names["N"]], pre_nm[names["CA"]], pre_nm[names["C"]], pre_nm[names["CB"]])
        post = signed_chirality(post_nm[names["N"]], post_nm[names["CA"]], post_nm[names["C"]], post_nm[names["CB"]])
        pre_ok = pre > CHIRALITY_MIN_VOLUME_NM3
        post_ok = post > CHIRALITY_MIN_VOLUME_NM3
        row = {
            "residue": list(residue.source_key),
            "pre_signed_volume_nm3": pre,
            "post_signed_volume_nm3": post,
            "expected_sign": "positive_L",
            "minimum_abs_volume_nm3": CHIRALITY_MIN_VOLUME_NM3,
            "pre_valid_L": pre_ok,
            "post_valid_L": post_ok,
        }
        rows.append(row)
        if not pre_ok:
            initial_invalid.append(list(residue.source_key))
        if not post_ok:
            post_invalid.append(list(residue.source_key))
            failures.append(f"post-minimized backbone chirality is not positive L/nonplanar at {residue.source_key}: {post:.6g} nm^3")
    return {
        "status": "pass" if not failures else "fail",
        "checked_residue_count": len(rows),
        "expected_sign": "positive_L_for_N_CA_C_CB_signed_volume",
        "minimum_abs_volume_nm3": CHIRALITY_MIN_VOLUME_NM3,
        "initial_invalid_count": len(initial_invalid),
        "post_invalid_count": len(post_invalid),
        "initial_invalid_examples": initial_invalid[:20],
        "post_invalid_examples": post_invalid[:20],
        "failures": failures,
        "examples": rows[:20],
        "residues": rows,
    }


def beta_chirality_checks(topology: ParsedPrmtop, pre_nm: np.ndarray, post_nm: np.ndarray) -> dict[str, Any]:
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    initial_invalid: list[list[Any]] = []
    post_invalid: list[list[Any]] = []
    specs = {"THR": ("OG1", "CG2"), "ILE": ("CG1", "CG2")}
    for residue in topology.residues:
        if residue.source_key is None:
            continue
        canonical = residue.source_key[2]
        if canonical not in specs and residue.name not in specs:
            continue
        x_name, cg2_name = specs.get(canonical, specs.get(residue.name, ("", "")))
        names = {topology.atoms[i].name: i for i in residue.atom_indices}
        if not {"CA", "CB", x_name, cg2_name} <= set(names):
            failures.append(f"missing THR/ILE Cbeta stereochemistry atoms at {residue.source_key}")
            continue
        indices = [names["CA"], names["CB"], names[x_name], names[cg2_name]]
        pre = signed_chirality(*pre_nm[indices])
        post = signed_chirality(*post_nm[indices])
        pre_ok = pre > CHIRALITY_MIN_VOLUME_NM3
        post_ok = post > CHIRALITY_MIN_VOLUME_NM3
        row = {
            "residue": list(residue.source_key),
            "atom_order": ["CA", "CB", x_name, cg2_name],
            "pre_signed_volume_nm3": pre,
            "post_signed_volume_nm3": post,
            "expected_sign": "positive_source_calibrated",
            "minimum_abs_volume_nm3": CHIRALITY_MIN_VOLUME_NM3,
            "pre_valid": pre_ok,
            "post_valid": post_ok,
        }
        rows.append(row)
        if not pre_ok:
            initial_invalid.append(list(residue.source_key))
        if not post_ok:
            post_invalid.append(list(residue.source_key))
            failures.append(f"post-minimized THR/ILE Cbeta stereochemistry is not positive/nonplanar at {residue.source_key}: {post:.6g} nm^3")
    return {
        "status": "pass" if not failures else "fail",
        "checked_residue_count": len(rows),
        "expected_sign": "positive_for_source_calibrated_THR_ILE_Cbeta_signed_volume",
        "minimum_abs_volume_nm3": CHIRALITY_MIN_VOLUME_NM3,
        "initial_invalid_count": len(initial_invalid),
        "post_invalid_count": len(post_invalid),
        "initial_invalid_examples": initial_invalid[:20],
        "post_invalid_examples": post_invalid[:20],
        "failures": failures,
        "examples": rows[:20],
        "residues": rows,
    }


def export_restart_with_parmed(prmtop: Path, inpcrd: Path, positions_nm: np.ndarray, box_nm: np.ndarray, output: Path) -> None:
    try:
        import parmed as pmd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ParmEd is required to export the minimized Amber restart") from exc
    structure = pmd.load_file(str(prmtop), xyz=str(inpcrd))
    if len(structure.atoms) != len(positions_nm):
        raise ValueError("ParmEd atom count does not match minimized positions")
    structure.coordinates = np.asarray(positions_nm, dtype=float) * 10.0
    a = float(np.linalg.norm(box_nm[0])) * 10.0
    b = float(np.linalg.norm(box_nm[1])) * 10.0
    c = float(np.linalg.norm(box_nm[2])) * 10.0
    alpha = math.degrees(math.acos(np.dot(box_nm[1], box_nm[2]) / (np.linalg.norm(box_nm[1]) * np.linalg.norm(box_nm[2]))))
    beta = math.degrees(math.acos(np.dot(box_nm[0], box_nm[2]) / (np.linalg.norm(box_nm[0]) * np.linalg.norm(box_nm[2]))))
    gamma = math.degrees(math.acos(np.dot(box_nm[0], box_nm[1]) / (np.linalg.norm(box_nm[0]) * np.linalg.norm(box_nm[1]))))
    structure.box = [a, b, c, alpha, beta, gamma]
    output.parent.mkdir(parents=True, exist_ok=True)
    structure.save(str(output), overwrite=True)



def minimization_provenance(minimizer_report: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "minimizer_report": {
            "path": str(minimizer_report),
            "sha256": sha256_file(minimizer_report),
        },
        "status": report.get("status"),
        "platform": report.get("platform"),
        "precision": report.get("precision"),
        "system_creation": report.get("system_creation"),
    }

def validate_restraints_sources(restraints_payload: dict[str, Any], *, prmtop: Path, amber_pdb_override: Path | None = None) -> dict[str, str]:
    sources = restraints_payload.get("sources", {})
    if not isinstance(sources, dict):
        raise ValueError("restraints JSON lacks sources object")
    actual: dict[str, str] = {}
    for key in ("prmtop", "amber_pdb"):
        entry = sources.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"restraints JSON lacks source path/sha256 for {key}")
        path = prmtop if key == "prmtop" else (amber_pdb_override or Path(entry["path"]))
        digest = sha256_file(path)
        actual[key] = digest
        if digest != entry["sha256"]:
            raise ValueError(f"restraints JSON source hash mismatch for {key}")
    return actual


def verify_restart_roundtrip(prmtop: Path, restart: Path, positions_nm: np.ndarray, box_nm: np.ndarray) -> dict[str, Any]:
    try:
        from openmm import app, unit
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenMM is required to verify minimized restart round-trip") from exc
    loaded = app.AmberInpcrdFile(str(restart))
    loaded_positions = np.asarray(loaded.positions.value_in_unit(unit.nanometer), dtype=float)
    if loaded_positions.shape != positions_nm.shape:
        raise ValueError("restart round-trip atom count mismatch")
    max_position_delta_nm = float(np.max(np.abs(loaded_positions - positions_nm))) if positions_nm.size else 0.0
    box_vectors = loaded.boxVectors
    if box_vectors is None:
        raise ValueError("restart round-trip lacks periodic box vectors")
    loaded_box = np.asarray([v.value_in_unit(unit.nanometer) for v in box_vectors], dtype=float)
    max_box_delta_nm = float(np.max(np.abs(loaded_box - box_nm)))
    result = {"status": "pass", "max_position_delta_nm": max_position_delta_nm, "max_box_delta_nm": max_box_delta_nm, "tolerance_nm": 5e-6}
    if max_position_delta_nm > 5e-6 or max_box_delta_nm > 5e-6:
        result["status"] = "fail"
        raise ValueError(f"restart round-trip exceeded tolerance: {result}")
    return result


def _declared_hash(report: dict[str, Any], key: str) -> str | None:
    candidates = [key]
    if key == "restraints":
        candidates.append("restrain_indices")
    for container_name in ("sources", "outputs"):
        container = report.get(container_name, {})
        if not isinstance(container, dict):
            continue
        for candidate in candidates:
            entry = container.get(candidate)
            if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
                return entry["sha256"]
    return None


def validate_hashes(report: dict[str, Any], paths: dict[str, Path], *, required_keys: Sequence[str] = ("prmtop", "inpcrd", "mapping", "restraints", "minpositions", "box")) -> dict[str, str]:
    actual = {key: sha256_file(path) for key, path in paths.items()}
    for key in required_keys:
        if key not in actual:
            raise ValueError(f"missing required hash validation path: {key}")
        declared = _declared_hash(report, key)
        if declared is None:
            raise ValueError(f"minimizer report lacks required sha256 for {key}")
        if declared != actual[key]:
            raise ValueError(f"stale input hash mismatch for {key}")
    for key, digest in actual.items():
        declared = _declared_hash(report, key)
        if declared is not None and declared != digest:
            raise ValueError(f"stale input hash mismatch for {key}")
    return actual


def pre_positions_from_report_or_inpcrd(report: dict[str, Any], inpcrd: Path) -> np.ndarray:
    pre_positions = np.asarray(report.get("pre_positions_nm", []), dtype=float)
    if pre_positions.size:
        return pre_positions
    try:
        from openmm import app, unit

        return np.asarray(app.AmberInpcrdFile(str(inpcrd)).positions.value_in_unit(unit.nanometer), dtype=float)
    except Exception as exc:  # pragma: no cover - exercised only in OpenMM environment
        raise RuntimeError("pre-min coordinates are required for chirality checks") from exc


def verify_pilot_mapping(prmtop: Path, restart: Path, mapping_payload: dict[str, Any], assembly: str, *, expected_core_count: int = 269) -> dict[str, Any]:
    try:
        from openmm import app, unit
        try:
            from . import run_atomistic_technical_pilot as pilot
        except ImportError:
            import run_atomistic_technical_pilot as pilot
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenMM and run_atomistic_technical_pilot are required for mapping verification") from exc
    topology = app.AmberPrmtopFile(str(prmtop)).topology
    inpcrd = app.AmberInpcrdFile(str(restart))
    positions_nm = np.asarray(inpcrd.positions.value_in_unit(unit.nanometer), dtype=float)
    model = "isolated" if assembly == "isolated" else "flexible"
    validated = pilot.validate_mapping(mapping_payload, topology, positions_nm, model, expected_core_count=expected_core_count)
    return {
        "status": "pass",
        "model": model,
        "expected_core_count": expected_core_count,
        "core_count": len(validated["core_indices"]),
        "q_norm": float(np.linalg.norm(np.asarray(validated["q"], dtype=float))),
        "has_initial_core_nm": "initial_core_nm" in validated,
    }


def qualify_minimized(
    *,
    prmtop: Path,
    inpcrd: Path,
    mapping: Path,
    restraints: Path,
    minimizer_report: Path,
    minpositions: Path,
    box: Path,
    output_dir: Path,
    restart_output: Path | None = None,
    amber_pdb: Path | None = None,
    assembly: str = "joint",
    exporter=export_restart_with_parmed,
    restart_verifier=verify_restart_roundtrip,
    mapping_verifier=verify_pilot_mapping,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Qualification output directory must be new or empty; preserve earlier attempts")
    report = read_json(minimizer_report)
    if report.get("status") != "minimization_complete":
        raise ValueError("minimizer report status must be minimization_complete")
    mapping_payload = read_json(mapping)
    restraints_payload = read_json(restraints)
    if restraints_payload.get("status") != "pass":
        raise ValueError("restraints JSON must have status pass")
    positions = load_positions_npy(minpositions, "minpositions")
    box_nm = load_box_npy(box)
    topology = parse_prmtop(prmtop)
    if len(topology.atoms) != len(positions):
        raise ValueError("minpositions atom count does not match PRMTOP")
    prepared_pdb = amber_pdb or Path(restraints_payload.get("sources", {}).get("amber_pdb", {}).get("path", ""))
    if not str(prepared_pdb):
        raise ValueError("restraints JSON lacks amber_pdb source path and no --amber-pdb override was supplied")
    topology = attach_residue_keys(topology, residue_order(parse_pdb_atoms(Path(prepared_pdb))))
    restraints_source_hashes = validate_restraints_sources(restraints_payload, prmtop=prmtop, amber_pdb_override=amber_pdb)
    paths = {"prmtop": prmtop, "inpcrd": inpcrd, "mapping": mapping, "restraints": restraints, "minpositions": minpositions, "box": box}
    input_hashes = validate_hashes(report, paths)
    terminal = terminal_checks(topology, assembly)
    if terminal["status"] != "pass":
        raise ValueError(f"terminal checks failed: {terminal['failures']}")
    pre_positions = pre_positions_from_report_or_inpcrd(report, inpcrd)
    if pre_positions.shape != positions.shape or not np.isfinite(pre_positions).all():
        raise ValueError("pre-min coordinates must be finite and match minimized positions")
    peptide = peptide_bond_checks(topology, positions)
    chirality = chirality_checks(topology, pre_positions, positions)
    beta_chirality = beta_chirality_checks(topology, pre_positions, positions)
    metal = raw_zn_sg_checks(topology, positions)
    gates = {"metal": "pass" if metal["status"] == "pass" else "fail", "mapping": "pass", "geometry": "pass"}
    failures = []
    for name, check in (("terminal", terminal), ("peptide", peptide), ("chirality", chirality), ("beta_chirality", beta_chirality)):
        if check.get("status") != "pass":
            gates["geometry"] = "fail"
            failures.extend(f"{name}: {failure}" for failure in check.get("failures", []))
    if metal["status"] != "pass":
        failures.extend(f"metal: {failure}" for failure in metal.get("failures", []))
    if failures:
        write_json(output_dir / "qualification_failure.json", {
            "status": "fail", "production_ready": False,
            "gates": {**gates, "mapping": "not_evaluated"},
            "input_sha256": input_hashes, "failures": failures,
            "geometry_checks": {"peptide_bonds": peptide,
                "backbone_chirality": chirality, "thr_ile_cbeta_chirality": beta_chirality,
                "raw_zn_sg": metal},
        })
        raise ValueError(f"qualification failed: {failures}")
    core_indices = mapping_payload.get("core_indices")
    if not isinstance(core_indices, list) or not core_indices:
        raise ValueError("mapping lacks core_indices")
    initial_core_nm = positions[np.asarray(core_indices, dtype=int)]
    if not np.isfinite(initial_core_nm).all():
        raise ValueError("initial_core_nm would contain nonfinite values")
    updated_mapping = dict(mapping_payload)
    updated_mapping["initial_core_nm"] = initial_core_nm.tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_out = output_dir / "atomistic_mapping_with_initial.json"
    write_json(mapping_out, updated_mapping)
    restart = restart_output or (output_dir / "minimized.inpcrd")
    exporter(prmtop, inpcrd, positions, box_nm, restart)
    restart_roundtrip = restart_verifier(prmtop, restart, positions, box_nm)
    mapping_check = mapping_verifier(prmtop, restart, updated_mapping, assembly)
    if mapping_check.get("status") != "pass":
        raise ValueError(f"pilot mapping verification failed: {mapping_check}")
    qualification = {
        "schema_version": "1.0",
        "status": "pass",
        "gates": gates,
        "minimization_provenance": minimization_provenance(minimizer_report, report),
        "chemical_review": {"status": "technical_chemistry_preparation_pass", "scope": "technical preparation only; production false; no temperature/equilibration/MD-response claim"},
        "production_ready": False,
        "input_sha256": {"prmtop": sha256_file(prmtop), "inpcrd": sha256_file(restart), "mapping": sha256_file(mapping_out)},
        "source_sha256": input_hashes,
        "restraints_source_sha256": restraints_source_hashes,
        "terminal_checks": terminal,
        "mapping_checks": {"pilot_validate_mapping": mapping_check},
        "geometry_checks": {"peptide_bonds": peptide, "backbone_chirality": chirality, "thr_ile_cbeta_chirality": beta_chirality, "raw_zn_sg": metal, "restart_roundtrip": restart_roundtrip},
        "minimized_restart": str(restart),
        "mapping_with_initial_core_nm": str(mapping_out),
    }
    q_out = output_dir / "technical_qualification.json"
    write_json(q_out, qualification)
    return {"qualification": qualification, "qualification_path": str(q_out), "mapping_path": str(mapping_out), "restart_path": str(restart)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("restraints")
    r.add_argument("--heavy-mapping", type=Path, default=DEFAULT_HEAVY_MAPPING)
    r.add_argument("--amber-pdb", type=Path, required=True)
    r.add_argument("--prmtop", type=Path, required=True)
    r.add_argument("--output", type=Path, default=DEFAULT_RESTRAINTS)
    r.add_argument("--assembly", choices=("joint", "isolated"), default="joint")
    r.add_argument("--expected-observed-heavy", type=int, default=EXPECTED_OBSERVED_HEAVY)
    q = sub.add_parser("qualify")
    q.add_argument("--prmtop", type=Path, required=True)
    q.add_argument("--inpcrd", type=Path, required=True)
    q.add_argument("--mapping", type=Path, required=True)
    q.add_argument("--restraints", type=Path, required=True)
    q.add_argument("--minimizer-report", type=Path, required=True)
    q.add_argument("--minpositions", type=Path, required=True)
    q.add_argument("--box", type=Path, required=True)
    q.add_argument("--output-dir", type=Path, required=True)
    q.add_argument("--restart-output", type=Path)
    q.add_argument("--amber-pdb", type=Path, help="Relocated amber_input_renamed.pdb; bytes must match restraints JSON SHA")
    q.add_argument("--assembly", choices=("joint", "isolated"), default="joint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "restraints":
        expected = args.expected_observed_heavy if args.expected_observed_heavy >= 0 else None
        payload = build_restraints(heavy_mapping=args.heavy_mapping, amber_pdb=args.amber_pdb, prmtop=args.prmtop, output=args.output, assembly=args.assembly, expected_observed_heavy=expected)
        print(json.dumps({"status": payload["status"], "output": str(args.output), "observed_heavy_count": payload["observed_heavy_count"]}, sort_keys=True))
        return 0
    if args.command == "qualify":
        result = qualify_minimized(prmtop=args.prmtop, inpcrd=args.inpcrd, mapping=args.mapping, restraints=args.restraints, minimizer_report=args.minimizer_report, minpositions=args.minpositions, box=args.box, output_dir=args.output_dir, restart_output=args.restart_output, amber_pdb=args.amber_pdb, assembly=args.assembly)
        print(json.dumps({"status": result["qualification"]["status"], "qualification": result["qualification_path"], "mapping": result["mapping_path"], "restart": result["restart_path"]}, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
