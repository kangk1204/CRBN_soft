#!/usr/bin/env python3
"""Add neutral ACE/NME cap coordinates to a repaired heavy-atom CRBN-DDB1 PDB.

This implements the conservative Route B terminal-capping step for the 8CVP
atomistic pilot. Cap coordinates are transferred from explicit LEaP-generated
Amber template PDB files by fitting the template middle-residue backbone onto the
observed terminal residue backbone. Only cap heavy atoms are transferred.
Hydrogens, topology, OXT, solvent, ions, and force-field parameters remain the
responsibility of the downstream LEaP topology stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAIN = "B"
DEFAULT_N_TERMINUS = 64
DEFAULT_C_TERMINUS = 428
DEFAULT_N_RESNAME = "MET"
DEFAULT_C_RESNAME = "ASP"
DEFAULT_ACE_RESSEQ = 63
DEFAULT_NME_RESSEQ = 429
DEFAULT_RMSD_THRESHOLD = 0.15
DEFAULT_BOND_MIN = 1.20
DEFAULT_BOND_MAX = 1.50
CLASH_WARNING_DISTANCE = 1.20
N_CAP_FIT_ATOMS = ("N", "CA", "C")
C_CAP_FIT_ATOMS = ("CA", "C", "O")


@dataclass(frozen=True)
class AtomRecord:
    record_name: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    temp_factor: float = 0.0
    element: str = ""
    charge: str = ""
    source: str = "input"

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.chain, self.resseq, self.resname, self.name)

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def is_hydrogen(self) -> bool:
        elem = self.element.strip().upper()
        if elem:
            return elem == "H" or elem.startswith("D")
        return self.name.strip().upper().startswith(("H", "D"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pdb_atom_line(line: str) -> AtomRecord | None:
    if not line.startswith(("ATOM  ", "HETATM")):
        return None
    try:
        serial = int(line[6:11])
    except ValueError:
        serial = 0
    name = line[12:16].strip()
    altloc = line[16:17].strip()
    resname = line[17:20].strip()
    chain = line[21:22].strip()
    try:
        resseq = int(line[22:26])
    except ValueError as exc:
        raise ValueError(f"PDB atom line has non-integer residue number: {line.rstrip()}") from exc
    icode = line[26:27].strip()
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError as exc:
        raise ValueError(f"PDB atom line has invalid coordinates: {line.rstrip()}") from exc
    try:
        occ = float(line[54:60]) if line[54:60].strip() else 1.0
    except ValueError:
        occ = 1.0
    try:
        b = float(line[60:66]) if line[60:66].strip() else 0.0
    except ValueError:
        b = 0.0
    element = line[76:78].strip() if len(line) >= 78 else ""
    charge = line[78:80].strip() if len(line) >= 80 else ""
    return AtomRecord(line[:6].strip(), serial, name, altloc, resname, chain, resseq, icode, x, y, z, occ, b, element, charge)


def read_pdb(path: Path) -> tuple[list[str], list[AtomRecord]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms: list[AtomRecord] = []
    for line in lines:
        record = parse_pdb_atom_line(line)
        if record is not None:
            atoms.append(record)
    if not atoms:
        raise ValueError(f"no ATOM/HETATM records found: {path}")
    return lines, atoms


def format_atom(record: AtomRecord, serial: int) -> str:
    name = record.name.strip()
    if len(name) < 4 and not (record.element.strip().upper() in {"FE", "ZN", "MG", "MN", "CL", "NA", "CA"}):
        atom_field = f" {name:<3}"
    else:
        atom_field = f"{name:>4}"[:4]
    element = (record.element or name[:1]).strip().upper()[:2]
    return (
        f"{record.record_name:<6}{serial:5d} {atom_field}{record.altloc[:1]:1s}"
        f"{record.resname:>3s} {record.chain[:1]:1s}{record.resseq:4d}{record.icode[:1]:1s}   "
        f"{record.x:8.3f}{record.y:8.3f}{record.z:8.3f}"
        f"{record.occupancy:6.2f}{record.temp_factor:6.2f}          {element:>2s}{record.charge:>2s}"
    )


def select_residue_atoms(atoms: Sequence[AtomRecord], chain: str, resseq: int, resname: str | None = None) -> list[AtomRecord]:
    out = [a for a in atoms if a.chain == chain and a.resseq == resseq and (resname is None or a.resname == resname)]
    if not out:
        label = f"{chain}:{resseq}" + (f" {resname}" if resname else "")
        raise ValueError(f"missing residue atoms for {label}")
    return out


def atom_by_name(atoms: Sequence[AtomRecord], atom_name: str) -> AtomRecord:
    matches = [a for a in atoms if a.name == atom_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one atom named {atom_name}, found {len(matches)}")
    return matches[0]


def residue_backbone(atoms: Sequence[AtomRecord], label: str, atom_names: Sequence[str]) -> np.ndarray:
    coords = []
    for atom_name in atom_names:
        try:
            coords.append(atom_by_name(atoms, atom_name).xyz)
        except ValueError as exc:
            raise ValueError(f"{label} lacks required backbone atom {atom_name}") from exc
    return np.vstack(coords)


def kabsch_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if source.shape != target.shape or source.shape != (3, 3):
        raise ValueError("source and target backbone arrays must both be 3x3")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    src = source - source_center
    tgt = target - target_center
    cov = src.T @ tgt
    u, _s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, d])
    rotation = u @ correction @ vt
    transformed = src @ rotation + target_center
    rmsd = float(np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1))))
    translation = target_center - source_center @ rotation
    return rotation, translation, rmsd


def transform_atom(atom: AtomRecord, rotation: np.ndarray, translation: np.ndarray, *, chain: str, resseq: int, resname: str, source: str) -> AtomRecord:
    xyz = atom.xyz @ rotation + translation
    return replace(atom, record_name="ATOM", serial=0, resname=resname, chain=chain, resseq=resseq, icode="", x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]), occupancy=1.0, temp_factor=0.0, source=source)


def template_residue_groups(template_atoms: Sequence[AtomRecord], middle_resname: str) -> tuple[list[AtomRecord], list[AtomRecord], list[AtomRecord]]:
    ace = [a for a in template_atoms if a.resname == "ACE"]
    mid = [a for a in template_atoms if a.resname == middle_resname]
    nme = [a for a in template_atoms if a.resname == "NME"]
    if not ace or not mid or not nme:
        raise ValueError(f"template must contain ACE, {middle_resname}, and NME residues")
    return ace, mid, nme


def heavy_cap_atoms(atoms: Sequence[AtomRecord]) -> list[AtomRecord]:
    heavy = [a for a in atoms if not a.is_hydrogen]
    if not heavy:
        raise ValueError("cap residue has no heavy atoms")
    return heavy


def min_distance(atoms_a: Sequence[AtomRecord], atoms_b: Sequence[AtomRecord]) -> float | None:
    if not atoms_a or not atoms_b:
        return None
    best = math.inf
    for a in atoms_a:
        for b in atoms_b:
            best = min(best, float(np.linalg.norm(a.xyz - b.xyz)))
    return best


def nonbonded_clash_warnings(cap_atoms: Sequence[AtomRecord], input_atoms: Sequence[AtomRecord], bonded_allow: set[tuple[str, int, str]], threshold: float = CLASH_WARNING_DISTANCE) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    for cap in cap_atoms:
        for atom in input_atoms:
            if atom.is_hydrogen:
                continue
            if (atom.chain, atom.resseq, atom.name) in bonded_allow:
                continue
            d = float(np.linalg.norm(cap.xyz - atom.xyz))
            if d < threshold:
                warnings.append({"cap": [cap.chain, cap.resseq, cap.resname, cap.name], "atom": [atom.chain, atom.resseq, atom.resname, atom.name], "distance_angstrom": d})
    return warnings


def add_caps(
    input_heavy_pdb: Path,
    n_template_pdb: Path,
    c_template_pdb: Path,
    output_dir: Path,
    *,
    chain: str = DEFAULT_CHAIN,
    n_residue: int = DEFAULT_N_TERMINUS,
    c_residue: int = DEFAULT_C_TERMINUS,
    n_resname: str = DEFAULT_N_RESNAME,
    c_resname: str = DEFAULT_C_RESNAME,
    ace_residue: int = DEFAULT_ACE_RESSEQ,
    nme_residue: int = DEFAULT_NME_RESSEQ,
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    bond_min: float = DEFAULT_BOND_MIN,
    bond_max: float = DEFAULT_BOND_MAX,
) -> dict[str, object]:
    _, input_atoms = read_pdb(input_heavy_pdb)
    _, n_template_atoms = read_pdb(n_template_pdb)
    _, c_template_atoms = read_pdb(c_template_pdb)

    n_target = select_residue_atoms(input_atoms, chain, n_residue, n_resname)
    c_target = select_residue_atoms(input_atoms, chain, c_residue, c_resname)
    n_ace, n_mid, _n_nme = template_residue_groups(n_template_atoms, n_resname)
    _c_ace, c_mid, c_nme = template_residue_groups(c_template_atoms, c_resname)

    n_rot, n_trans, n_rmsd = kabsch_fit(
        residue_backbone(n_mid, f"N-template {n_resname}", N_CAP_FIT_ATOMS),
        residue_backbone(n_target, f"target {chain}:{n_residue} {n_resname}", N_CAP_FIT_ATOMS),
    )
    c_rot, c_trans, c_rmsd = kabsch_fit(
        residue_backbone(c_mid, f"C-template {c_resname}", C_CAP_FIT_ATOMS),
        residue_backbone(c_target, f"target {chain}:{c_residue} {c_resname}", C_CAP_FIT_ATOMS),
    )
    if n_rmsd > rmsd_threshold:
        raise ValueError(f"N-terminal template backbone RMSD {n_rmsd:.4f} A exceeds threshold {rmsd_threshold:.4f} A")
    if c_rmsd > rmsd_threshold:
        raise ValueError(f"C-terminal template backbone RMSD {c_rmsd:.4f} A exceeds threshold {rmsd_threshold:.4f} A")

    ace_atoms = [transform_atom(a, n_rot, n_trans, chain=chain, resseq=ace_residue, resname="ACE", source="n_template_ACE") for a in heavy_cap_atoms(n_ace)]
    nme_atoms = [transform_atom(a, c_rot, c_trans, chain=chain, resseq=nme_residue, resname="NME", source="c_template_NME") for a in heavy_cap_atoms(c_nme)]

    ace_c = atom_by_name(ace_atoms, "C")
    met_n = atom_by_name(n_target, "N")
    asp_c = atom_by_name(c_target, "C")
    nme_n = atom_by_name(nme_atoms, "N")
    n_bond = float(np.linalg.norm(ace_c.xyz - met_n.xyz))
    c_bond = float(np.linalg.norm(asp_c.xyz - nme_n.xyz))
    if not (bond_min <= n_bond <= bond_max):
        raise ValueError(f"ACE C to residue {n_residue} N distance {n_bond:.4f} A outside {bond_min}-{bond_max} A")
    if not (bond_min <= c_bond <= bond_max):
        raise ValueError(f"residue {c_residue} C to NME N distance {c_bond:.4f} A outside {bond_min}-{bond_max} A")

    cap_atoms = ace_atoms + nme_atoms
    warnings = nonbonded_clash_warnings(cap_atoms, input_atoms, {(chain, n_residue, "N"), (chain, c_residue, "C")})
    critical_clashes = [w for w in warnings if float(w["distance_angstrom"]) < 1.0]
    if critical_clashes:
        raise ValueError(f"critical cap/input heavy-atom clash below 1.0 A: {critical_clashes[0]}")
    cap_to_protein_ca_distances = [
        float(np.linalg.norm(cap.xyz - atom.xyz))
        for cap in cap_atoms
        for atom in input_atoms
        if atom.name == "CA" and not atom.is_hydrogen
    ]
    min_cap_to_protein_ca = min(cap_to_protein_ca_distances) if cap_to_protein_ca_distances else None

    output_dir.mkdir(parents=True, exist_ok=True)
    capped_path = output_dir / "capped_heavy.pdb"
    mapping_path = output_dir / "cap_atom_mapping.csv"
    provenance_path = output_dir / "cap_provenance.json"

    with capped_path.open("w", encoding="utf-8") as handle:
        serial = 1
        inserted_ace = False
        inserted_nme = False
        for idx, atom in enumerate(input_atoms):
            if not inserted_ace and atom.chain == chain and atom.resseq == n_residue:
                for cap in ace_atoms:
                    handle.write(format_atom(cap, serial) + "\n")
                    serial += 1
                inserted_ace = True
            handle.write(format_atom(atom, serial) + "\n")
            serial += 1
            next_atom = input_atoms[idx + 1] if idx + 1 < len(input_atoms) else None
            if not inserted_nme and atom.chain == chain and atom.resseq == c_residue and (next_atom is None or next_atom.chain != chain or next_atom.resseq != c_residue):
                for cap in nme_atoms:
                    handle.write(format_atom(cap, serial) + "\n")
                    serial += 1
                inserted_nme = True
        if not inserted_ace:
            raise ValueError(f"failed to insert ACE before {chain}:{n_residue}")
        if not inserted_nme:
            raise ValueError(f"failed to insert NME after {chain}:{c_residue}")
        handle.write("END\n")

    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "chain", "resseq", "resname", "atom", "element", "x", "y", "z"])
        writer.writeheader()
        for atom in cap_atoms:
            writer.writerow({"source": atom.source, "chain": atom.chain, "resseq": atom.resseq, "resname": atom.resname, "atom": atom.name, "element": atom.element, "x": f"{atom.x:.6f}", "y": f"{atom.y:.6f}", "z": f"{atom.z:.6f}"})

    provenance = {
        "status": "complete_with_warnings" if warnings else "complete",
        "production_ready": False,
        "reason_not_production_ready": "caps are coordinate-only heavy atoms; downstream LEaP topology, hydrogens, OXT/charged DDB1 termini, zinc parameters, solvent, ions, and minimization remain required",
        "input_heavy_pdb": str(input_heavy_pdb.resolve()),
        "n_template_pdb": str(n_template_pdb.resolve()),
        "c_template_pdb": str(c_template_pdb.resolve()),
        "sha256": {
            "input_heavy_pdb": sha256_file(input_heavy_pdb),
            "n_template_pdb": sha256_file(n_template_pdb),
            "c_template_pdb": sha256_file(c_template_pdb),
            "capped_heavy_pdb": sha256_file(capped_path),
        },
        "terminal_model": {
            "capped_chain": chain,
            "ace_residue": ace_residue,
            "n_terminal_residue": n_residue,
            "n_terminal_resname": n_resname,
            "c_terminal_residue": c_residue,
            "c_terminal_resname": c_resname,
            "nme_residue": nme_residue,
            "ddb1_caps_added": False,
            "crbn_missing_terminal_residues_rebuilt": False,
            "hydrogens_transferred_from_templates": False,
        },
        "alignment": {
            "n_template_backbone_rmsd_angstrom": n_rmsd,
            "c_template_carbonyl_plane_rmsd_angstrom": c_rmsd,
            "c_template_fit_atoms": list(C_CAP_FIT_ATOMS),
            "n_template_fit_atoms": list(N_CAP_FIT_ATOMS),
            "rmsd_threshold_angstrom": rmsd_threshold,
        },
        "peptide_bond_distances_angstrom": {
            "ACE_C_to_residue_64_N": n_bond,
            "residue_428_C_to_NME_N": c_bond,
            "accepted_range": [bond_min, bond_max],
        },
        "cap_heavy_atom_count": len(cap_atoms),
        "warnings": {"nonbonded_clashes_lt_1_2A": warnings, "critical_clash_threshold_angstrom": 1.0, "min_cap_to_protein_CA_angstrom": min_cap_to_protein_ca, "final_md_minimization_needed": True},
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"capped_pdb": str(capped_path), "mapping_csv": str(mapping_path), "provenance_json": str(provenance_path), **provenance}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-heavy-pdb", required=True, type=Path)
    parser.add_argument("--n-template-pdb", required=True, type=Path, help="LEaP-generated ACE-MET-NME PDB template")
    parser.add_argument("--c-template-pdb", required=True, type=Path, help="LEaP-generated ACE-ASP-NME PDB template")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chain", default=DEFAULT_CHAIN)
    parser.add_argument("--n-residue", default=DEFAULT_N_TERMINUS, type=int)
    parser.add_argument("--c-residue", default=DEFAULT_C_TERMINUS, type=int)
    parser.add_argument("--n-resname", default=DEFAULT_N_RESNAME)
    parser.add_argument("--c-resname", default=DEFAULT_C_RESNAME)
    parser.add_argument("--rmsd-threshold", default=DEFAULT_RMSD_THRESHOLD, type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = add_caps(
        args.input_heavy_pdb,
        args.n_template_pdb,
        args.c_template_pdb,
        args.output_dir,
        chain=args.chain,
        n_residue=args.n_residue,
        c_residue=args.c_residue,
        n_resname=args.n_resname,
        c_resname=args.c_resname,
        rmsd_threshold=args.rmsd_threshold,
    )
    print(json.dumps({"capped_pdb": result["capped_pdb"], "provenance_json": result["provenance_json"], "status": result["status"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
