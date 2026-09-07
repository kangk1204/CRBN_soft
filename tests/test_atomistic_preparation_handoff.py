from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import atomistic_preparation_handoff as h


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdb_line(serial: int, name: str, resname: str, chain: str, resseq: int, xyz: tuple[float, float, float], element: str | None = None, record: str = "ATOM") -> str:
    elem = (element or ("ZN" if name.upper().startswith("ZN") else name[0])).upper()
    return f"{record:<6}{serial:5d} {name:^4s} {resname:>3s} {chain:1s}{resseq:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {elem:>2s}\n"


def write_pdb(path: Path, residue_atoms: list[tuple[str, int, str, list[tuple[str, tuple[float, float, float], str | None]]]]) -> list[dict[str, object]]:
    rows = []
    serial = 1
    with path.open("w", encoding="utf-8") as fh:
        for chain, resseq, resname, atoms in residue_atoms:
            record = "HETATM" if resname in {"ACE", "NME", "ZN", "ZN1"} else "ATOM"
            for atom_name, xyz, element in atoms:
                fh.write(pdb_line(serial, atom_name, resname, chain, resseq, xyz, element, record=record))
                rows.append({"index": serial - 1, "chain": chain, "resseq": resseq, "resname": resname, "atom": atom_name, "xyz_a": xyz})
                serial += 1
        fh.write("END\n")
    return rows


def chunks(values, width):
    return [values[i : i + width] for i in range(0, len(values), width)]


def write_prmtop(path: Path, pdb_rows: list[dict[str, object]], bonds: list[tuple[int, int]], charges: list[float] | None = None) -> None:
    charges = charges if charges is not None else [0.0] * len(pdb_rows)
    residue_labels = []
    pointers = []
    last_key = None
    for i, row in enumerate(pdb_rows, start=1):
        key = (row["chain"], row["resseq"], row["resname"])
        if key != last_key:
            residue_labels.append(str(row["resname"]))
            pointers.append(i)
            last_key = key
    atom_names = [str(row["atom"]) for row in pdb_rows]
    scaled = [q * h.AMBER_CHARGE_SCALE for q in charges]
    bond_values = []
    for a, b in bonds:
        bond_values.extend([a * 3, b * 3, 1])
    with path.open("w", encoding="utf-8") as fh:
        fh.write("%FLAG ATOM_NAME\n%FORMAT(20a4)\n")
        for part in chunks([name[:4].ljust(4) for name in atom_names], 20):
            fh.write("".join(part) + "\n")
        fh.write("%FLAG CHARGE\n%FORMAT(5E16.8)\n")
        for part in chunks(scaled, 5):
            fh.write("".join(f"{v:16.8E}" for v in part) + "\n")
        fh.write("%FLAG RESIDUE_LABEL\n%FORMAT(20a4)\n")
        for part in chunks([name[:4].ljust(4) for name in residue_labels], 20):
            fh.write("".join(part) + "\n")
        fh.write("%FLAG RESIDUE_POINTER\n%FORMAT(10I8)\n")
        for part in chunks(pointers, 10):
            fh.write("".join(f"{v:8d}" for v in part) + "\n")
        fh.write("%FLAG BONDS_INC_HYDROGEN\n%FORMAT(10I8)\n\n")
        fh.write("%FLAG BONDS_WITHOUT_HYDROGEN\n%FORMAT(10I8)\n")
        for part in chunks(bond_values, 10):
            fh.write("".join(f"{v:8d}" for v in part) + "\n")


def fixture_residues(include_a=True):
    residues = []
    if include_a:
        residues += [
            ("A", 1, "MET", [("N", (0.0, 0.0, 0.0), "N"), ("H1", (0.0, 0.1, 0.0), "H"), ("H2", (0.1, 0.0, 0.0), "H"), ("H3", (0.0, 0.0, 0.1), "H"), ("CA", (1.0, 0.0, 0.0), "C"), ("C", (2.0, 0.0, 0.0), "C"), ("O", (2.5, 0.5, 0.0), "O")]),
            ("A", 1140, "HIE", [("N", (3.0, 0.0, 0.0), "N"), ("CA", (4.0, 0.0, 0.0), "C"), ("C", (4.0, 1.0, 0.0), "C"), ("O", (4.5, 1.5, 0.0), "O"), ("OXT", (3.5, 1.5, 0.0), "O"), ("CB", (4.0, 0.0, -1.0), "C")]),
        ]
    residues += [
        ("B", 63, "ACE", [("C", (0.000, 10.0, 0.0), "C"), ("O", (-0.5, 10.5, 0.0), "O")]),
        ("B", 64, "MET", [("N", (1.33, 10.0, 0.0), "N"), ("CA", (2.33, 10.0, 0.0), "C"), ("C", (2.33, 11.0, 0.0), "C"), ("CB", (2.33, 10.0, -1.0), "C")]),
        ("B", 323, "CY1", [("N", (0.0, 20.0, 0.0), "N"), ("CA", (1.0, 20.0, 0.0), "C"), ("C", (1.0, 21.0, 0.0), "C"), ("CB", (1.0, 20.0, -1.0), "C"), ("SG", (0.0, 30.0, 0.0), "S")]),
        ("B", 326, "CY1", [("N", (0.0, 22.0, 0.0), "N"), ("CA", (1.0, 22.0, 0.0), "C"), ("C", (1.0, 23.0, 0.0), "C"), ("CB", (1.0, 22.0, -1.0), "C"), ("SG", (2.3, 30.0, 0.0), "S")]),
        ("B", 391, "CY1", [("N", (0.0, 24.0, 0.0), "N"), ("CA", (1.0, 24.0, 0.0), "C"), ("C", (1.0, 25.0, 0.0), "C"), ("CB", (1.0, 24.0, -1.0), "C"), ("SG", (0.0, 32.3, 0.0), "S")]),
        ("B", 394, "CY1", [("N", (0.0, 26.0, 0.0), "N"), ("CA", (1.0, 26.0, 0.0), "C"), ("C", (1.0, 27.0, 0.0), "C"), ("CB", (1.0, 26.0, -1.0), "C"), ("SG", (0.0, 30.0, 2.3), "S")]),
        ("B", 428, "ASP", [("N", (0.0, 40.0, 0.0), "N"), ("CA", (1.0, 40.0, 0.0), "C"), ("C", (1.0, 41.0, 0.0), "C"), ("O", (0.5, 41.5, 0.0), "O"), ("CB", (1.0, 40.0, -1.0), "C")]),
        ("B", 429, "NME", [("N", (1.0, 42.33, 0.0), "N"), ("C", (1.0, 43.0, 0.0), "C")]),
        ("B", 500, "THR", [("N", (0.0, 50.0, 0.0), "N"), ("CA", (1.0, 50.0, 0.0), "C"), ("C", (1.0, 51.0, 0.0), "C"), ("CB", (1.0, 50.0, -1.0), "C"), ("OG1", (2.0, 50.0, -1.0), "O"), ("CG2", (1.0, 51.0, -1.0), "C")]),
        ("C", 501, "ZN", [("ZN", (1.15, 31.15, 1.15), "ZN")]),
    ]
    return residues


def index_by_key(rows):
    return {(r["chain"], r["resseq"], r["resname"], r["atom"]): int(r["index"]) for r in rows}


def make_fixture(tmp_path: Path, include_a=True, omit_cap_bond=False, cross_chain_bond=False):
    pdb_path = tmp_path / "amber_input_renamed.pdb"
    rows = write_pdb(pdb_path, fixture_residues(include_a=include_a))
    idx = index_by_key(rows)
    bonds = [
        (idx[("B", 63, "ACE", "C")], idx[("B", 64, "MET", "N")]),
        (idx[("B", 428, "ASP", "C")], idx[("B", 429, "NME", "N")]),
    ]
    if omit_cap_bond:
        bonds = bonds[1:]
    if cross_chain_bond and include_a:
        bonds.append((idx[("A", 1, "MET", "C")], idx[("B", 64, "MET", "N")]))
    prmtop = tmp_path / "solvated.prmtop"
    write_prmtop(prmtop, rows, bonds)
    mapping = tmp_path / "atom_residue_mapping.csv"
    with mapping.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["chain", "residue_number", "residue_name", "atom", "source_status"])
        writer.writeheader()
        writer.writerow({"chain": "B", "residue_number": 64, "residue_name": "MET", "atom": "CA", "source_status": "observed_input"})
        writer.writerow({"chain": "B", "residue_number": 323, "residue_name": "CYS", "atom": "SG", "source_status": "observed_input"})
        writer.writerow({"chain": "C", "residue_number": 501, "residue_name": "ZN", "atom": "ZN", "source_status": "observed_input"})
        if include_a:
            writer.writerow({"chain": "A", "residue_number": 1, "residue_name": "MET", "atom": "CA", "source_status": "observed_input"})
    positions_nm = np.array([r["xyz_a"] for r in rows], dtype=float) / 10.0
    return pdb_path, prmtop, mapping, positions_nm, rows


def test_restraints_joint_accepts_hie_a1140_and_cy1_mapping(tmp_path):
    pdb, prmtop, heavy_mapping, _positions, _rows = make_fixture(tmp_path, include_a=True)
    out = tmp_path / "observed_heavy_restraints.json"
    payload = h.build_restraints(heavy_mapping=heavy_mapping, amber_pdb=pdb, prmtop=prmtop, output=out, assembly="joint", expected_observed_heavy=4)
    assert payload["status"] == "pass"
    assert payload["observed_heavy_count"] == 4
    assert payload["terminal_checks"]["ddb1_termini"]["A1140_residue"] == ["A", 1140, "HIE"]
    assert payload["terminal_checks"]["total_charge_e"] == pytest.approx(0.0)
    assert ["B", 323, "CY1"] in payload["zaff_scope"]["cy1_residues"]


def test_restraints_fail_missing_cap_bond(tmp_path):
    pdb, prmtop, heavy_mapping, _positions, _rows = make_fixture(tmp_path, include_a=True, omit_cap_bond=True)
    with pytest.raises(ValueError, match="missing terminal peptide bond"):
        h.build_restraints(heavy_mapping=heavy_mapping, amber_pdb=pdb, prmtop=prmtop, output=tmp_path / "out.json", assembly="joint", expected_observed_heavy=4)


def test_restraints_isolated_filters_full_csv_to_present_crbn_and_zn_rows(tmp_path):
    pdb, prmtop, heavy_mapping, _positions, _rows = make_fixture(tmp_path, include_a=False)
    with heavy_mapping.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["chain", "residue_number", "residue_name", "atom", "source_status"])
        writer.writerow({"chain": "A", "residue_number": 1, "residue_name": "MET", "atom": "CA", "source_status": "observed_input"})
    payload = h.build_restraints(heavy_mapping=heavy_mapping, amber_pdb=pdb, prmtop=prmtop, output=tmp_path / "out.json", assembly="isolated", expected_observed_heavy=4)
    assert payload["terminal_checks"]["ddb1_termini"] == "not_applicable_isolated"
    assert payload["full_observed_heavy_count"] == 4
    assert payload["selected_observed_heavy_count"] == 3
    assert payload["observed_heavy_count"] == 3


def make_qualification_inputs(tmp_path: Path, *, omit_report_hash: str | None = None, post_mutator=None, cross_chain_bond=False):
    pdb, prmtop, heavy_mapping, positions, rows = make_fixture(tmp_path, include_a=True, cross_chain_bond=cross_chain_bond)
    restraints = tmp_path / "observed_heavy_restraints.json"
    h.build_restraints(heavy_mapping=heavy_mapping, amber_pdb=pdb, prmtop=prmtop, output=restraints, assembly="joint", expected_observed_heavy=4)
    inpcrd = tmp_path / "start.inpcrd"
    inpcrd.write_text("placeholder amber restart\n", encoding="utf-8")
    mapping = tmp_path / "atomistic_mapping.json"
    idx = index_by_key(rows)
    mapping_payload = {"core_indices": [idx[("B", 64, "MET", "CA")]], "reference_nm": [[1.0, 2.0, 3.0]], "q": [[0.0, 0.0, 1.0]]}
    h.write_json(mapping, mapping_payload)
    post = positions.copy()
    if post_mutator:
        post_mutator(post, idx)
    minpositions = tmp_path / "minimized_positions.npy"
    box = tmp_path / "box_vectors_nm.npy"
    np.save(minpositions, post)
    np.save(box, np.array([[17.1, 0.0, 0.0], [-5.7, 16.1, 0.0], [-5.7, -8.0, 14.0]], dtype=float))
    report = {
        "status": "minimization_complete",
        "pre_positions_nm": positions.tolist(),
        "sources": {
            "prmtop": {"path": str(prmtop), "sha256": sha(prmtop)},
            "inpcrd": {"path": str(inpcrd), "sha256": sha(inpcrd)},
            "mapping": {"path": str(mapping), "sha256": sha(mapping)},
            "restrain_indices": {"path": str(restraints), "sha256": sha(restraints)},
        },
        "outputs": {
            "minpositions": {"path": str(minpositions), "sha256": sha(minpositions)},
            "box": {"path": str(box), "sha256": sha(box)},
        },
        "platform": {
            "requested": "OpenCL",
            "properties": {"Precision": "double", "DisablePmeStream": "true"},
            "effective_properties": {"Precision": "double", "DisablePmeStream": "true"},
        },
        "precision": {
            "requested_override": "double",
            "effective": "double",
            "pme_stream": {"effective_disabled": True},
        },
        "system_creation": {
            "nonbonded_method_selected": "PME",
            "removeCMMotion": False,
        },
    }
    if omit_report_hash:
        report["sources"].pop(omit_report_hash, None)
        report["outputs"].pop(omit_report_hash, None)
    report_path = tmp_path / "minimizer_report.json"
    h.write_json(report_path, report)
    return prmtop, inpcrd, mapping, restraints, report_path, minpositions, box, idx


def fake_exporter(prmtop, inpcrd, positions_nm, box_nm, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"natoms": len(positions_nm), "box": box_nm.tolist()}), encoding="utf-8")


def fake_restart_verifier(prmtop, restart, positions_nm, box_nm):
    return {"status": "pass", "max_position_delta_nm": 0.0, "max_box_delta_nm": 0.0, "tolerance_nm": 5e-6}


def fake_mapping_verifier(prmtop, restart, mapping_payload, assembly):
    return {"status": "pass", "core_count": len(mapping_payload["core_indices"]), "model": "flexible" if assembly != "isolated" else "isolated"}


def test_qualify_writes_mapping_restart_and_preserves_frozen_fields(tmp_path):
    args = make_qualification_inputs(tmp_path)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, idx = args
    result = h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=tmp_path / "qualified", exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)
    q = result["qualification"]
    assert q["status"] == "pass"
    assert q["gates"] == {"metal": "pass", "mapping": "pass", "geometry": "pass"}
    provenance = q["minimization_provenance"]
    assert provenance["minimizer_report"] == {"path": str(report), "sha256": sha(report)}
    assert provenance["platform"]["effective_properties"]["DisablePmeStream"] == "true"
    assert provenance["precision"]["effective"] == "double"
    assert provenance["precision"]["pme_stream"]["effective_disabled"] is True
    assert provenance["system_creation"] == {
        "nonbonded_method_selected": "PME",
        "removeCMMotion": False,
    }
    updated = h.read_json(Path(result["mapping_path"]))
    original = h.read_json(mapping)
    assert updated["reference_nm"] == original["reference_nm"]
    assert updated["q"] == original["q"]
    assert updated["initial_core_nm"] == [np.load(minpositions)[idx[("B", 64, "MET", "CA")]].tolist()]
    assert Path(result["restart_path"]).exists()


def test_qualify_missing_minimizer_metadata_is_explicit_none(tmp_path):
    args = make_qualification_inputs(tmp_path)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    payload = h.read_json(report)
    payload.pop("platform")
    payload.pop("precision")
    payload.pop("system_creation")
    h.write_json(report, payload)

    result = h.qualify_minimized(
        prmtop=prmtop,
        inpcrd=inpcrd,
        mapping=mapping,
        restraints=restraints,
        minimizer_report=report,
        minpositions=minpositions,
        box=box,
        output_dir=tmp_path / "qualified",
        exporter=fake_exporter,
        restart_verifier=fake_restart_verifier,
        mapping_verifier=fake_mapping_verifier,
    )

    provenance = result["qualification"]["minimization_provenance"]
    assert provenance["minimizer_report"] == {"path": str(report), "sha256": sha(report)}
    assert provenance["platform"] is None
    assert provenance["precision"] is None
    assert provenance["system_creation"] is None


def test_qualify_mapping_verifier_failure_blocks_outputs(tmp_path):
    args = make_qualification_inputs(tmp_path)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    outdir = tmp_path / "qualified"
    def rejecting_mapping_verifier(prmtop, restart, mapping_payload, assembly):
        raise ValueError("runner mapping rejected")
    with pytest.raises(ValueError, match="runner mapping rejected"):
        h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=outdir, exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=rejecting_mapping_verifier)
    assert not (outdir / "technical_qualification.json").exists()


def test_qualify_requires_minimizer_status_pass(tmp_path):
    args = make_qualification_inputs(tmp_path)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    payload = h.read_json(report)
    payload["status"] = "failed"
    h.write_json(report, payload)
    with pytest.raises(ValueError, match="minimization_complete"):
        h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=tmp_path / "qualified", exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)


def test_qualify_allows_relocated_amber_pdb_with_matching_sha(tmp_path):
    args = make_qualification_inputs(tmp_path)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    payload = h.read_json(restraints)
    original_pdb = Path(payload["sources"]["amber_pdb"]["path"])
    payload["sources"]["amber_pdb"]["path"] = str(tmp_path / "missing_after_relocation.pdb")
    h.write_json(restraints, payload)
    report_payload = h.read_json(report)
    report_payload["sources"]["restrain_indices"]["sha256"] = sha(restraints)
    h.write_json(report, report_payload)
    result = h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=tmp_path / "qualified", amber_pdb=original_pdb, exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)
    assert result["qualification"]["status"] == "pass"


def test_qualify_rejects_missing_required_hash(tmp_path):
    args = make_qualification_inputs(tmp_path, omit_report_hash="box")
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    with pytest.raises(ValueError, match="lacks required sha256 for box"):
        h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=tmp_path / "qualified", exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)


def test_qualify_rejects_nonfinite_positions(tmp_path):
    def mutate(post, idx):
        post[0, 0] = np.nan
    args = make_qualification_inputs(tmp_path, post_mutator=mutate)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    with pytest.raises(ValueError, match="finite Nx3"):
        h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=tmp_path / "qualified", exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)


def test_qualify_rejects_post_chirality_not_positive_l(tmp_path):
    def mutate(post, idx):
        post[idx[("B", 64, "MET", "CB")]] = post[idx[("B", 64, "MET", "CA")]] + np.array([0.0, 0.0, 0.2])
    args = make_qualification_inputs(tmp_path, post_mutator=mutate)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    with pytest.raises(ValueError, match="chirality"):
        h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=tmp_path / "qualified", exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)
    failed = h.read_json(tmp_path / "qualified" / "qualification_failure.json")
    assert failed["status"] == "fail"
    assert failed["gates"]["mapping"] == "not_evaluated"
    assert failed["geometry_checks"]["backbone_chirality"]["post_invalid_count"] == 1
    assert not (tmp_path / "qualified" / "technical_qualification.json").exists()


def test_qualify_refuses_to_reuse_previous_attempt_directory(tmp_path):
    args = make_qualification_inputs(tmp_path)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    out = tmp_path / "qualified"
    out.mkdir()
    prior = out / "technical_qualification.json"
    prior.write_text('{"status":"pass"}\n')
    original = prior.read_bytes()
    with pytest.raises(ValueError, match="new or empty"):
        h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping,
            restraints=restraints, minimizer_report=report, minpositions=minpositions,
            box=box, output_dir=out, exporter=fake_exporter,
            restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)
    assert prior.read_bytes() == original


def test_qualify_rejects_bad_peptide_distance(tmp_path):
    def mutate(post, idx):
        post[idx[("B", 429, "NME", "N")]] += np.array([1.0, 0.0, 0.0])
    args = make_qualification_inputs(tmp_path, post_mutator=mutate)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    with pytest.raises(ValueError, match="peptide C-N distance"):
        h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=tmp_path / "qualified", exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)


def test_qualify_rejects_cross_chain_peptide_like_bond(tmp_path):
    args = make_qualification_inputs(tmp_path, cross_chain_bond=True)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    with pytest.raises(ValueError, match="cross-chain peptide"):
        h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=tmp_path / "qualified", exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)


def test_qualify_rejects_thr_ile_cbeta_negative(tmp_path):
    def mutate(post, idx):
        post[idx[("B", 500, "THR", "CG2")]] = post[idx[("B", 500, "THR", "CB")]] + np.array([0.0, -0.2, 0.0])
    args = make_qualification_inputs(tmp_path, post_mutator=mutate)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    with pytest.raises(ValueError, match="Cbeta"):
        h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=tmp_path / "qualified", exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)


def test_qualify_rejects_raw_zn_sg_distance_out_of_bounds(tmp_path):
    def mutate(post, idx):
        post[idx[("B", 394, "CY1", "SG")]] += np.array([1.0, 0.0, 0.0])
    args = make_qualification_inputs(tmp_path, post_mutator=mutate)
    prmtop, inpcrd, mapping, restraints, report, minpositions, box, _idx = args
    with pytest.raises(ValueError, match="raw Zn-SG"):
        h.qualify_minimized(prmtop=prmtop, inpcrd=inpcrd, mapping=mapping, restraints=restraints, minimizer_report=report, minpositions=minpositions, box=box, output_dir=tmp_path / "qualified", exporter=fake_exporter, restart_verifier=fake_restart_verifier, mapping_verifier=fake_mapping_verifier)
