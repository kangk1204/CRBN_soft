from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "strengthen_maps.py"


def load_module():
    spec = importlib.util.spec_from_file_location("strengthen_maps_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def minimal_metadata() -> dict:
    return {
        "admin": {
            "title": "DDB1-CRBN intermediate NU",
            "keywords": "CRBN",
            "current_status": {"code": {"valueOf_": "REL"}, "date": "2026-03-25T00:00:00"},
            "key_dates": {
                "deposition": "2025-05-22T00:00:00",
                "map_release": "2025-11-19T00:00:00",
            },
            "revision_history": {
                "revision": [{"version": "1.0", "date": "2025-11-19T00:00:00"}]
            },
        },
        "map": {
            "file": "emd_70781.map.gz",
            "annotation_details": "sharpened map",
            "dimensions": {"col": 320, "row": 320, "sec": 320},
            "pixel_spacing": {
                "x": {"valueOf_": "0.74"},
                "y": {"valueOf_": "0.74"},
                "z": {"valueOf_": "0.74"},
            },
        },
        "interpretation": {
            "additional_map_list": {
                "additional_map": [
                    {"file": "emd_70781_additional_1.map.gz", "annotation_details": "unsharpened"}
                ]
            },
            "half_map_list": {
                "half_map": [
                    {"file": "emd_70781_half_map_1.map.gz", "annotation_details": "half A"},
                    {"file": "emd_70781_half_map_2.map.gz", "annotation_details": "half B"},
                ]
            },
            "segmentation_list": {"segmentation": [{"file": "emd_70781_msk_1.map"}]},
        },
        "structure_determination_list": {
            "structure_determination": [
                {
                    "image_processing": [
                        {
                            "final_reconstruction": {
                                "resolution": {"valueOf_": "2.71"},
                                "resolution_method": "FSC 0.143 CUT-OFF",
                            }
                        }
                    ]
                }
            ]
        },
    }


def write_mrc(path: Path, values: np.ndarray, *, gzip_output: bool = False) -> None:
    values = np.asarray(values, dtype="<f4")
    nz, ny, nx = values.shape
    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, nx, ny, nz, 2)
    struct.pack_into("<3i", header, 28, nx, ny, nz)
    struct.pack_into("<3f", header, 40, float(nx), float(ny), float(nz))
    struct.pack_into("<3f", header, 52, 90.0, 90.0, 90.0)
    struct.pack_into("<3f", header, 76, float(values.min()), float(values.max()), float(values.mean()))
    struct.pack_into("<i", header, 92, 0)
    struct.pack_into("<3f", header, 196, 0.0, 0.0, 0.0)
    header[208:212] = b"MAP "
    struct.pack_into("<f", header, 216, float(values.std()))
    struct.pack_into("<i", header, 220, 1)
    header[224:304] = b"test map".ljust(80, b" ")
    payload = bytes(header) + values.tobytes(order="C")
    if gzip_output:
        with gzip.open(path, "wb") as handle:
            handle.write(payload)
    else:
        path.write_bytes(payload)


def test_entry_summary_discovers_api_artifacts_and_source_boundaries():
    module = load_module()
    summary = module.entry_summary("70781", minimal_metadata())

    assert summary["emdb_id"] == "EMD-70781"
    assert summary["state"] == "intermediate"
    assert summary["resolution_angstrom"] == pytest.approx(2.71)
    artifacts = {(item["kind"], item["file_name"], item["url"]) for item in summary["artifacts"]}
    assert (
        "primary_map",
        "emd_70781.map.gz",
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-70781/map/emd_70781.map.gz",
    ) in artifacts
    assert (
        "half_map",
        "emd_70781_half_map_1.map.gz",
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-70781/other/emd_70781_half_map_1.map.gz",
    ) in artifacts
    assert (
        "mask",
        "emd_70781_msk_1.map",
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-70781/masks/emd_70781_msk_1.map",
    ) in artifacts


def test_mrc_density_summary_reads_plain_and_gzipped_float32_maps(tmp_path):
    module = load_module()
    values = np.arange(24, dtype="<f4").reshape(2, 3, 4)
    plain = tmp_path / "plain.map"
    zipped = tmp_path / "plain.map.gz"
    write_mrc(plain, values)
    write_mrc(zipped, values, gzip_output=True)

    for path in (plain, zipped):
        summary = module.density_summary(path)
        assert summary["header"]["dimensions"] == [4, 3, 2]
        assert summary["header"]["mode_name"] == "float32"
        assert summary["computed_voxels"] == 24
        assert summary["computed_mean"] == pytest.approx(float(values.mean()))
        assert summary["computed_std"] == pytest.approx(float(values.std()))
        assert summary["computed_nonzero_fraction"] == pytest.approx(23 / 24)


def test_half_map_correlation_is_deterministic_for_matching_maps(tmp_path):
    module = load_module()
    base = np.linspace(-1.0, 1.0, 1000, dtype="<f4").reshape(10, 10, 10)
    shifted = base * 2.0 + 0.5
    first = tmp_path / "half1.map"
    second = tmp_path / "half2.map"
    write_mrc(first, base)
    write_mrc(second, shifted)

    result = module.sampled_half_correlation(first, second, max_points=1000)
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["sampled_points"] == 1000


def test_fit_gate_requires_orientation_coordinate_and_density_support():
    module = load_module()
    assert module.fit_gate(
        {
            "relative_orientation_deg": 9.5,
            "normalized_structural_coordinate": 0.05,
            "density_support": "supported",
        }
    )["quantitative_fit_allowed"]

    rejected = module.fit_gate(
        {
            "relative_orientation_deg": 11.0,
            "normalized_structural_coordinate": 0.05,
            "density_support": "supported",
        }
    )
    assert rejected["decision"] == "qualitative-only"
    assert any("10 degrees" in reason for reason in rejected["reasons"])

    pending = module.fit_gate(
        {
            "relative_orientation_deg": None,
            "normalized_structural_coordinate": None,
            "density_support": "supported",
        }
    )
    assert pending["decision"] == "pending"
    assert "completed fit" in pending["reasons"][0]


def test_offline_metadata_loader_reads_cached_json(tmp_path):
    module = load_module()
    cached = tmp_path / "EMD-70781" / "metadata_api.json"
    cached.parent.mkdir()
    cached.write_text(json.dumps(minimal_metadata()), encoding="utf-8")

    loaded = module.load_or_fetch_metadata("70781", output_dir=tmp_path, offline=True)
    assert loaded["admin"]["title"] == "DDB1-CRBN intermediate NU"


def test_cli_defaults_to_neutral_public_output_root():
    module = load_module()
    args = module.parse_args(["--download-ids"])
    assert args.output_dir == Path("results/strengthening")


def test_quality_decision_keeps_global_qa_separate_from_local_crbn_density():
    module = load_module()
    decision = module.quality_decision(
        chimerax={"available": False, "install_status": "blocked_by_license_acceptance"},
        density_checks={
            "EMD-70781": {
                "primary_map": {
                    "computed_std": 1.0,
                    "computed_nonzero_fraction": 1.0,
                }
            }
        },
        half_map_correlations={"EMD-70781": {"pearson_r": 0.9}},
    )

    assert decision["global_map_qa"] == "supported"
    assert decision["density_support"] == "not-assessed"
    assert decision["crbn_local_density_support"] == "not-assessed-pending-domain-fit"
    assert decision["overall_use"] == "fit-pending"
    assert any("do not establish local CRBN density" in reason for reason in decision["reasons"])




def write_synthetic_plan_d_refs(tmp_path):
    refs = tmp_path / "synthetic_refs"
    refs.mkdir()
    window = refs / "crbn_residue_window.csv"
    residues = [77, 78, 79, 318, 319, 320]
    window.write_text(
        "index,author_resnum\n" + "".join(f"{i},{resnum}\n" for i, resnum in enumerate(residues)),
        encoding="utf-8",
    )
    mean = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    pcs = np.zeros((18, 3), dtype=float)
    pc1 = np.zeros(18, dtype=float)
    pc1[0::3] = 1.0
    pcs[:, 0] = pc1 / np.linalg.norm(pc1)
    np.savez(
        refs / "crbn_pca.npz",
        mean=mean,
        pcs=pcs,
        pc1_scores=np.array([-1.0, 1.0]),
        open_mask=np.array([False, True]),
    )
    modes = np.zeros((18, 10), dtype=float)
    modes[:, 0] = pcs[:, 0]
    modes[:, 1] = np.roll(pcs[:, 0], 1)
    np.savez(refs / "crbn_anm_modes.npz", anm_eigvecs=modes)

    def pdb(path, shift):
        lines = []
        for serial, (resnum, xyz) in enumerate(zip(residues, mean + shift), start=1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA B{resnum:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C"
            )
        path.write_text("\n".join([*lines, "TER", "END"]) + "\n", encoding="utf-8")

    def cif_gz(path, shift):
        headers = [
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.pdbx_formal_charge",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.pdbx_PDB_model_num",
        ]
        rows = []
        serial = 1
        offsets = {"N": np.array([-0.25, 0.0, 0.0]), "CA": np.zeros(3), "C": np.array([0.25, 0.0, 0.0])}
        for label_seq, (resnum, xyz) in enumerate(zip(residues, mean + shift), start=1):
            for atom_name, atom_offset in offsets.items():
                pos = xyz + atom_offset
                rows.append(
                    f"ATOM {serial} C {atom_name} . ALA B 1 {label_seq} ? "
                    f"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} 1.00 0.00 ? "
                    f"{resnum} ALA B {atom_name} 1"
                )
                serial += 1
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("data_synthetic\n#\nloop_\n" + "\n".join(headers) + "\n" + "\n".join(rows) + "\n#\n")

    pdb(refs / "open_8cvp_assembly.pdb", np.array([0.0, 0.0, 0.0]))
    pdb(refs / "closed_5fqd.pdb", np.array([0.2, 0.1, 0.0]))
    cif_gz(refs / "open_8cvp.cif.gz", np.array([0.0, 0.0, 0.0]))
    cif_gz(refs / "closed_5fqd.cif.gz", np.array([0.2, 0.1, 0.0]))
    return {
        "window": window,
        "pca": refs / "crbn_pca.npz",
        "modes": refs / "crbn_anm_modes.npz",
        "open": refs / "open_8cvp_assembly.pdb",
        "closed": refs / "closed_5fqd.pdb",
        "fit_open": refs / "open_8cvp.cif.gz",
        "fit_closed": refs / "closed_5fqd.cif.gz",
    }


def point_module_at_refs(module, refs):
    module.PLAN_D_WINDOW_CSV = refs["window"]
    module.PLAN_D_PCA_NPZ = refs["pca"]
    module.PLAN_D_MODES_NPZ = refs["modes"]
    module.PLAN_D_TEMPLATES = {
        "open_8cvp_crbn": refs["open"],
        "closed_5fqd_crbn": refs["closed"],
    }
    module.PLAN_D_FIT_TEMPLATES = {
        "open_8cvp_crbn": refs["fit_open"],
        "closed_5fqd_crbn": refs["fit_closed"],
    }


def test_plan_d_config_covers_domain_halfmap_fit_and_stability_contract(tmp_path):
    module = load_module()
    config = module.plan_d_config(tmp_path / "out", tmp_path / "out" / "analysis" / "maps")

    assert [entry["emdb_id"] for entry in config["fit_entries"]] == [
        "EMD-70776",
        "EMD-70781",
        "EMD-70782",
    ]
    assert config["search_placements"] == 100
    assert config["seed"] == 20260905
    assert config["domains"] == {"NTD+HB": [77, 317], "TBD": [318, 426]}
    assert config["train_heldout_pairs"] == [
        {"train": "A", "heldout": "B"},
        {"train": "B", "heldout": "A"},
    ]
    assert config["stability_gate"] == {
        "relative_orientation_max_deg": 10.0,
        "normalized_structural_coordinate_max_abs_delta": 0.1,
        "expected_completed_fits_per_state": 4,
    }
    assert config["fit_templates"]["open_8cvp_crbn"].endswith("open_8cvp.cif.gz")
    assert config["fit_templates"]["closed_5fqd_crbn"].endswith("closed_5fqd.cif.gz")
    for entry in config["fit_entries"]:
        assert entry["half_maps"]["A"].endswith("half_map_1.map.gz")
        assert entry["half_maps"]["B"].endswith("half_map_2.map.gz")


def test_chimerax_plan_d_assets_are_ready_without_running_license_gated_binary(tmp_path):
    module = load_module()
    point_module_at_refs(module, write_synthetic_plan_d_refs(tmp_path))
    output_root = tmp_path / "results" / "strengthening"
    analysis_root = output_root / "analysis" / "maps"
    analysis_root.mkdir(parents=True)
    stale = analysis_root / "run_chimerax_emd70781.cxc"
    stale.write_text("old single-entry pilot", encoding="utf-8")

    module.write_chimerax_assets(output_root, analysis_root)

    assert not stale.exists()
    config = json.loads((analysis_root / "chimerax_plan_d_config.json").read_text())
    runner = (analysis_root / "chimerax_plan_d_runner.py").read_text(encoding="utf-8")
    cxc = (analysis_root / "run_chimerax_plan_d.cxc").read_text(encoding="utf-8")
    launcher = (analysis_root / "run_chimerax_plan_d.py").read_text(encoding="utf-8")

    assert "crbnpland" in cxc
    assert "chimerax_plan_d_config.json" in cxc
    assert str(tmp_path) not in cxc
    assert "run_chimerax_plan_d.py" in cxc
    assert "_runtime_cxc(asset_dir)" in launcher
    assert len(config["fit_entries"]) == 3
    assert "train_map," in runner
    assert "heldout_map," in runner
    assert "search=int(config['search_placements'])" in runner
    assert "seed=int(config['seed'])" in runner
    assert "_score_heldout(" in runner
    assert "shift=False" in runner
    assert "rotate=False" in runner
    assert "metric='correlation'" in runner
    assert "_relative_orientation_in_common_frame(" in runner
    assert "relative_orientation_matrix" in runner
    assert "normalized_structural_coordinate" in runner
    assert "_269ca.json" in runner
    assert "ANM" not in runner
    assert "alignment_reference" in runner


def test_chimerax_launcher_writes_runtime_cxc_from_its_own_directory(tmp_path):
    module = load_module()
    point_module_at_refs(module, write_synthetic_plan_d_refs(tmp_path))
    output_root = tmp_path / "portable output with spaces"
    analysis_root = output_root / "analysis" / "maps"
    analysis_root.mkdir(parents=True)
    module.write_chimerax_assets(output_root, analysis_root)

    spec = importlib.util.spec_from_file_location(
        "run_chimerax_plan_d_test",
        analysis_root / "run_chimerax_plan_d.py",
    )
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)

    runtime_cxc = launcher._runtime_cxc(analysis_root)
    try:
        payload = runtime_cxc.read_text(encoding="utf-8")
    finally:
        runtime_cxc.unlink(missing_ok=True)

    assert f'open "{analysis_root / "chimerax_plan_d_runner.py"}"' in payload
    assert f'crbnpland "{analysis_root / "chimerax_plan_d_config.json"}"' in payload
    assert f'log save "{analysis_root / "chimerax_plan_d_log.html"}"' in payload



def test_chimerax_assets_record_missing_reference_inputs_without_runtime_ready_claim(tmp_path):
    module = load_module()
    missing = tmp_path / "missing"
    module.PLAN_D_WINDOW_CSV = missing / "crbn_residue_window.csv"
    module.PLAN_D_PCA_NPZ = missing / "crbn_pca.npz"
    module.PLAN_D_MODES_NPZ = missing / "crbn_anm_modes.npz"
    module.PLAN_D_TEMPLATES = {
        "open_8cvp_crbn": missing / "open_8cvp_assembly.pdb",
        "closed_5fqd_crbn": missing / "closed_5fqd.pdb",
    }
    module.PLAN_D_FIT_TEMPLATES = {
        "open_8cvp_crbn": missing / "open_8cvp.cif.gz",
        "closed_5fqd_crbn": missing / "closed_5fqd.cif.gz",
    }
    output_root = tmp_path / "public_checkout"
    analysis_root = output_root / "analysis" / "maps"
    analysis_root.mkdir(parents=True)

    module.write_chimerax_assets(output_root, analysis_root)

    config = json.loads((analysis_root / "chimerax_plan_d_config.json").read_text())
    install_note = (analysis_root / "chimerax_install_ready.md").read_text(encoding="utf-8")
    assert config["reference_inputs_available"] is False
    assert {item["name"] for item in config["missing_reference_inputs"]} == {
        "crbn_residue_window.csv",
        "crbn_pca.npz",
        "crbn_anm_modes.npz",
        "open_8cvp_assembly.pdb",
        "closed_5fqd.pdb",
        "open_8cvp.cif.gz",
        "closed_5fqd.cif.gz",
    }
    assert "reference_inputs_available is true" in install_note


def load_generated_runner(tmp_path):
    module = load_module()
    point_module_at_refs(module, write_synthetic_plan_d_refs(tmp_path))
    output_root = tmp_path / "portable_output"
    analysis_root = output_root / "analysis" / "maps"
    analysis_root.mkdir(parents=True)
    module.write_chimerax_assets(output_root, analysis_root)
    spec = importlib.util.spec_from_file_location(
        "chimerax_plan_d_runner_test",
        analysis_root / "chimerax_plan_d_runner.py",
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    config = json.loads((analysis_root / "chimerax_plan_d_config.json").read_text())
    config["_test_config_base"] = str(analysis_root)
    return runner, config


def rotation_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_x(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_plan_d_state_stability_is_computed_within_state_not_across_states(tmp_path):
    runner, _config = load_generated_runner(tmp_path)
    rows = []
    for state, base_orientation, base_coord in [("open", 2.0, 0.0), ("closed", 60.0, 1.0)]:
        for delta in [0.0, 1.0, 2.0, 3.0]:
            rows.append(
                {
                    "state": state,
                    "relative_orientation_deg": base_orientation + delta,
                    "normalized_structural_coordinate": base_coord + delta * 0.01,
                }
            )

    stability = runner._state_stability(
        rows,
        {
            "relative_orientation_max_deg": 10.0,
            "normalized_structural_coordinate_max_abs_delta": 0.1,
            "expected_completed_fits_per_state": 4,
        },
    )

    assert stability["open"]["state_gate_pass"]
    assert stability["closed"]["state_gate_pass"]
    assert stability["open"]["relative_orientation_range_deg"] == pytest.approx(3.0)
    assert stability["closed"]["relative_orientation_range_deg"] == pytest.approx(3.0)


def test_plan_d_state_stability_uses_so3_pairwise_orientation_not_scalar_angle(tmp_path):
    runner, _config = load_generated_runner(tmp_path)
    rows = []
    for matrix in [np.eye(3), rotation_x(12.0), rotation_z(12.0), rotation_x(-12.0)]:
        rows.append(
            {
                "state": "open",
                "fit_status": "completed",
                "relative_orientation_deg": 12.0,
                "relative_orientation_matrix": matrix.tolist(),
                "normalized_structural_coordinate": 0.01,
            }
        )

    stability = runner._state_stability(
        rows,
        {
            "relative_orientation_max_deg": 10.0,
            "normalized_structural_coordinate_max_abs_delta": 0.1,
            "expected_completed_fits_per_state": 4,
        },
    )

    assert not stability["open"]["state_gate_pass"]
    assert not stability["open"]["relative_orientation_pass"]
    assert stability["open"]["orientation_stability_metric"] == "max_pairwise_so3_geodesic_deg"
    assert stability["open"]["relative_orientation_range_deg"] > 10.0


def test_plan_d_state_stability_rejects_incomplete_state_fits(tmp_path):
    runner, _config = load_generated_runner(tmp_path)
    rows = [
        {
            "state": "open",
            "fit_status": "completed",
            "relative_orientation_deg": 1.0,
            "normalized_structural_coordinate": 0.01,
        },
        {
            "state": "open",
            "fit_status": "no_retained_fit",
        },
    ]

    stability = runner._state_stability(
        rows,
        {
            "relative_orientation_max_deg": 10.0,
            "normalized_structural_coordinate_max_abs_delta": 0.1,
            "expected_completed_fits_per_state": 4,
        },
    )

    assert not stability["open"]["state_gate_pass"]
    assert stability["open"]["incomplete_fit_gate"]
    assert stability["open"]["n_completed_fits"] == 1


def test_coordinate_and_domain_orientation_are_invariant_to_global_template_rotation(tmp_path):
    runner, config = load_generated_runner(tmp_path)
    root = Path(config["repository_root"])
    with tempfile.TemporaryDirectory() as tmp:
        domains, _domain_indices, _template_paths, coord_residues, refs = runner._prepare_templates(
            root,
            config,
            Path(tmp),
            base=Path(config["_test_config_base"]),
        )
    score_ref = runner._load_score_reference(root, config, coord_residues, base=Path(config["_test_config_base"]))
    coords = np.asarray(score_ref["mean"], dtype=float)
    global_rot = rotation_z(37.0)
    global_shift = np.array([10.0, -5.0, 3.0])
    moved = coords @ global_rot + global_shift

    _raw_a, coord_a = runner._coordinate(
        coords,
        score_ref["mean"],
        score_ref["pc1"],
        score_ref["closed_mean"],
        score_ref["open_mean"],
    )
    _raw_b, coord_b = runner._coordinate(
        moved,
        score_ref["mean"],
        score_ref["pc1"],
        score_ref["closed_mean"],
        score_ref["open_mean"],
    )
    assert coord_b == pytest.approx(coord_a, abs=1e-10)

    residues = coord_residues
    ntd_idx = [i for i, resnum in enumerate(residues) if domains["NTD+HB"][0] <= resnum <= domains["NTD+HB"][-1]]
    tbd_idx = [i for i, resnum in enumerate(residues) if domains["TBD"][0] <= resnum <= domains["TBD"][-1]]
    ntd_ref = coords[ntd_idx]
    tbd_ref = coords[tbd_idx]
    ntd_fit = ntd_ref @ global_rot + global_shift
    tbd_fit = tbd_ref @ global_rot + global_shift
    orientation = runner._rotation_angle_deg(
        runner._kabsch_rotation(ntd_ref, ntd_fit),
        runner._kabsch_rotation(tbd_ref, tbd_fit),
    )
    assert orientation == pytest.approx(0.0, abs=1e-8)


def test_postfit_projection_reports_mode1_and_low_frequency_basis(tmp_path):
    runner, _config = load_generated_runner(tmp_path)
    displacement = np.array([1.0, 1.0, 0.0])
    mode1 = np.array([1.0, 0.0, 0.0])
    low_basis = np.eye(3)[:, :2]
    coords = np.array([[0.0, 0.0, 0.0]])
    domain_indices = [np.array([0])]

    result = runner._postfit_projection(displacement, mode1, low_basis, coords, domain_indices)

    assert result["mode1_abs_cosine"] == pytest.approx(2 ** -0.5)
    assert result["low_frequency_subspace_projection_norm"] == pytest.approx(1.0)
    assert result["rigid_two_domain_subspace_projection_norm"] == pytest.approx(1.0)


def test_plan_d_existing_output_is_not_overwritten(tmp_path):
    module = load_module()
    namespace = {'__name__': 'test_generated_runner'}
    exec(compile(module.CHIMERAX_PLAN_D_RUNNER, '<runner>', 'exec'), namespace)
    (tmp_path / 'chimerax_plan_d').mkdir()
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'repository_root': str(tmp_path)}))
    with pytest.raises(FileExistsError):
        namespace['run_plan_d'](None, str(config))


def test_plan_d_summary_rejects_duplicate_and_partial_runs():
    spec = importlib.util.spec_from_file_location('plan_d_export', ROOT/'scripts/summarize_chimerax_plan_d.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {'fit_entries': [{'emdb_id': 'EMD-1', 'state': 'open'}],
              'templates': {'t': 'pdb'}, 'train_heldout_pairs': [{'train': 'A', 'heldout': 'B'}],
              'domains': {'body': [1, 2]}, 'seed': 1, 'search_placements': 100}
    row = {'emdb_id': 'EMD-1', 'state': 'open', 'template': 't', 'train_half': 'A',
           'heldout_half': 'B', 'domain_metrics': {'body': {}}}
    summary = {'results': [row], 'seed': 1, 'search_placements': 100, 'state_stability': {'open': {}}}
    module.validate_execution(summary, config)
    for rows in ([], [row, row]):
        with pytest.raises(ValueError, match='identities'):
            module.validate_execution({**summary, 'results': rows}, config)


def test_nonzero_relative_orientation_is_global_frame_invariant(tmp_path):
    runner, _config = load_generated_runner(tmp_path)
    rng = np.random.default_rng(37)
    mean = rng.normal(size=(20, 3))
    mean[10:] += np.array([4, 1, 0])
    indices = {'NTD+HB': np.arange(10), 'TBD': np.arange(10, 20)}
    fitted = mean.copy()
    fitted[10:] = fitted[10:] @ rotation_z(48.0) + [2, -3, 1]
    angle, matrix, aligned = runner._relative_orientation_in_common_frame(fitted, mean, indices)
    moved = fitted @ rotation_z(73.0) + [17, -8, 4]
    angle2, matrix2, aligned2 = runner._relative_orientation_in_common_frame(moved, mean, indices)
    assert angle == pytest.approx(48.0, abs=1e-10)
    assert angle2 == pytest.approx(angle, abs=1e-10)
    np.testing.assert_allclose(matrix2, matrix, atol=1e-12)
    np.testing.assert_allclose(aligned2, aligned, atol=1e-12)
