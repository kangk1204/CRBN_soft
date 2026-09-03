from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.axes
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(f"figure_sync_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def write_fig1_inputs(data: Path) -> tuple[np.ndarray, np.ndarray]:
    data.mkdir()
    closed = np.linspace(-0.2, 0.2, 65)
    opened = np.linspace(9.0, 9.4, 5)
    pc1 = np.concatenate([closed, opened])
    open_mask = np.array([False] * 65 + [True] * 5)
    labels = np.array([f"P{index:03d}" for index in range(70)])
    variance = np.array([0.883, 0.04, 0.02, 0.015, 0.01, 0.009, 0.008, 0.006, 0.005, 0.004])

    np.savez(
        data / "crbn_pca.npz",
        pc1_scores=pc1,
        pc2_scores=np.linspace(-1.0, 1.0, 70),
        open_mask=open_mask,
        variance_ratio=variance,
        mean=np.zeros((269, 3)),
    )
    np.savez(
        data / "crbn_anm_modes.npz",
        cum_overlap=np.linspace(0.744, 0.881, 10),
        rmsip=np.array(0.641),
    )
    np.savez(data / "pca_diffvec.npz", labels=labels)

    with (data / "crbn_curation_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb", "global_state"])
        writer.writeheader()
        for index, label in enumerate(labels):
            if index < 64:
                state = "drug-conditioned"
            elif index < 67:
                state = "genuine-apo"
            else:
                state = "native-substrate"
            writer.writerow({"pdb": label, "global_state": state})

    (data / "window_sensitivity.json").write_text(
        json.dumps(
            {
                "empty_middle": {
                    "a_paper_rule": {
                        "band_15_85_pct": [1.5, 7.5],
                        "n_occupants": 0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return pc1, open_mask


def test_fig1_converts_empty_band_to_the_displayed_coordinate(tmp_path, monkeypatch):
    module = load_script("build_fig1")
    pc1, open_mask = write_fig1_inputs(tmp_path / "data")
    monkeypatch.setattr(module, "DATA", tmp_path / "data")

    values = module.load_inputs()
    closed_mean = float(pc1[~open_mask].mean())
    open_mean = float(pc1[open_mask].mean())
    expected_coordinate = (pc1 - closed_mean) / (open_mean - closed_mean)
    expected_band = (np.array([1.5, 7.5]) - closed_mean) / (open_mean - closed_mean)
    np.testing.assert_allclose(values["normalized_coordinate"], expected_coordinate)
    np.testing.assert_allclose(values["normalized_band"], expected_band)

    captured: dict[str, object] = {}
    spans: list[tuple[float, float]] = []
    original_axvspan = matplotlib.axes.Axes.axvspan

    def record_span(axis, xmin, xmax, *args, **kwargs):
        spans.append((float(xmin), float(xmax)))
        return original_axvspan(axis, xmin, xmax, *args, **kwargs)

    monkeypatch.setattr(module, "apply_publication_style", lambda _figure_id: None)
    monkeypatch.setattr(
        module,
        "save_figure_set",
        lambda figure, _root, _stem: captured.setdefault("figure", figure),
    )
    monkeypatch.setattr(matplotlib.axes.Axes, "axvspan", record_span)
    module.build_figure(values)

    assert spans == [pytest.approx(tuple(expected_band))]
    figure = captured["figure"]
    assert len(figure.axes) == 4
    x_positions = {round(axis.get_position().x0, 3) for axis in figure.axes}
    y_positions = {round(axis.get_position().y0, 3) for axis in figure.axes}
    assert len(x_positions) == 2
    assert len(y_positions) == 2


def test_frozen_structural_render_hashes_fail_closed(tmp_path, monkeypatch):
    fig2 = load_script("build_fig2")
    panel_a = tmp_path / "panel_a.png"
    panel_b = tmp_path / "panel_b.png"
    panel_a.write_bytes(b"panel-a")
    panel_b.write_bytes(b"panel-b")
    hashes = {
        panel_a.name: hashlib.sha256(panel_a.read_bytes()).hexdigest(),
        panel_b.name: hashlib.sha256(panel_b.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(fig2, "PANELS", tmp_path)
    monkeypatch.setattr(fig2, "FROZEN_PANEL_SHA256", hashes)
    fig2.verify_structural_rasters()
    panel_b.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen structural panel changed"):
        fig2.verify_structural_rasters()

    fig4 = load_script("build_fig4")
    pocket = tmp_path / "pocket.png"
    pocket.write_bytes(b"pocket")
    monkeypatch.setattr(fig4, "STRUCTURE_INPUT", pocket)
    monkeypatch.setattr(
        fig4,
        "FROZEN_STRUCTURE_SHA256",
        hashlib.sha256(pocket.read_bytes()).hexdigest(),
    )
    fig4._verify_structure_input()
    pocket.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen structural panel changed"):
        fig4._verify_structure_input()


def test_release_render_hash_contracts_and_table_labels_are_current():
    fig2 = load_script("build_fig2")
    fig4 = load_script("build_fig4")
    assert fig2.FROZEN_PANEL_SHA256 == {
        "fig2_anm3d.png": "17c8b267c8a06e2b7da5a1665a96979176992dd66ad67998973be58fd47a0de9",
        "fig2_pc13d.png": "e59077e6b8361b1aa70852de304c3f265da9f52d01ddf51b1d0dafcb2d2f2213",
    }
    assert (
        fig4.FROZEN_STRUCTURE_SHA256
        == "e72b571169fe71ce6b6dc4c50cde67adce3fff9ae696ed3ef3482353f1e4f072"
    )

    table_source = (SCRIPTS / "build_tables.py").read_text(encoding="utf-8")
    assert "First-order bond length preservation at the boundary" in table_source
    assert "Equal-displacement boundary rigid null" in table_source
    assert "descriptive entry-level Fisher's exact" in table_source
    assert "Random rigid direction, equal boundary displacement" not in table_source


def test_endpoint_axis_band_and_pocket_definitions_are_frozen(tmp_path, monkeypatch):
    fig3 = load_script("build_fig3")
    geometry = tmp_path / "hinge_geometry.json"
    geometry.write_text(
        json.dumps(
            {
                "rotation_angle_deg": 82.45706028167992,
                "axis_proximal_boundary_residues": [316, 317, 318, 319, 320],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fig3, "HINGE_INPUT", geometry)
    low, high, residues = fig3._load_axis_band()
    assert (low, high) == (315.5, 320.5)
    assert residues == (316, 317, 318, 319, 320)

    fig4 = load_script("build_fig4")
    assert fig4.ANNOTATED_RESIDUES == (378, 380, 386)
    assert fig4.CONTACT_RESIDUES == (377, 378, 379, 380, 386, 400, 402)
    assert fig4.ZINC_RESIDUES == (323, 326, 391, 394)

    exporter = (SCRIPTS / "export_figure_source_data.py").read_text(encoding="utf-8")
    assert "5FQD LVY contacts <=4.5 A" in exporter
    assert "screw-axis-proximal boundary residue" in exporter


def test_table_notes_do_not_present_domain_partitions_as_hinge_calls():
    table_source = (SCRIPTS / "build_tables.py").read_text(encoding="utf-8")
    compact = " ".join(table_source.split())
    # The notes were rewritten to lead with the connectivity-preserving nulls and to mark
    # the unconstrained ones as reference models. That states the same caution more directly
    # than the phrases it replaced, so the required strings move with it; the two
    # prohibitions below are what actually keep a domain partition from reading as a
    # hinge call, and they are unchanged.
    assert "connectivity-preserving" in compact
    assert "unconstrained reference: does not constrain the boundary" in compact
    assert "three-dimensional boundary-rotation" in compact
    assert "inferred hinge" not in table_source
    assert "hinge-geometry-specific" not in table_source
