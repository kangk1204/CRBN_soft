from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
class _Atom:
    def __init__(self, coord: tuple[float, float, float]) -> None:
        self.coord = coord


class _FakeCmd:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.util = SimpleNamespace(cnc=lambda selection: self.calls.append(("cnc", selection)))

    def reinitialize(self) -> None:
        self.calls.append(("reinitialize",))

    def bg_color(self, color: str) -> None:
        self.calls.append(("bg_color", color))

    def set(self, name: str, value: object, selection: str | None = None) -> None:
        self.calls.append(("set", name, value, selection))

    def load(self, path: str, name: str) -> None:
        self.calls.append(("load", path, name))

    def remove(self, selection: str) -> None:
        self.calls.append(("remove", selection))

    def create(self, name: str, selection: str) -> None:
        self.calls.append(("create", name, selection))

    def delete(self, name: str) -> None:
        self.calls.append(("delete", name))

    def hide(self, representation: str) -> None:
        self.calls.append(("hide", representation))

    def show(self, representation: str, selection: str) -> None:
        self.calls.append(("show", representation, selection))

    def alter(self, selection: str, expression: str) -> None:
        self.calls.append(("alter", selection, expression))

    def sort(self) -> None:
        self.calls.append(("sort",))

    def color(self, color: str, selection: str) -> None:
        self.calls.append(("color", color, selection))

    def spectrum(
        self,
        expression: str,
        palette: str,
        selection: str,
        minimum: float,
        maximum: float,
    ) -> None:
        self.calls.append(("spectrum", expression, palette, selection, minimum, maximum))

    def get_model(self, selection: str) -> SimpleNamespace:
        self.calls.append(("get_model", selection))
        if "378+380+386" in selection:
            coords = [(2.0, 0.0, 0.0), (2.0, 1.0, 0.0)]
        elif "323+326+391+394" in selection:
            coords = [(-2.0, 0.0, 0.0), (-2.0, -1.0, 0.0)]
        else:
            coords = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (-0.5, -0.5, 0.0)]
        return SimpleNamespace(atom=[_Atom(coord) for coord in coords])

    def set_view(self, view: list[float]) -> None:
        self.calls.append(("set_view", tuple(view)))

    def zoom(self, selection: str, buffer: float, complete: int) -> None:
        self.calls.append(("zoom", selection, buffer, complete))

    def ray(self, width: int, height: int) -> None:
        self.calls.append(("ray", width, height))

    def png(self, path: str, dpi: int) -> None:
        self.calls.append(("png", path, dpi))
        Image.new("RGBA", (2, 2), (255, 255, 255, 255)).save(path)


def _write_minimal_render_inputs(root: Path) -> None:
    (root / "data").mkdir()
    (root / "render").mkdir()
    (root / "render" / "closed_5fqd_lig.pdb").write_text("MODEL\nEND\n", encoding="utf-8")
    resnums = np.array([222, 318, 319, 330, 424, 430])
    eigenvectors = np.ones((3 * len(resnums), 10), dtype=float)
    eigenvalues = np.ones(10, dtype=float)
    np.savez(
        root / "data" / "crbn_anm_modes.npz",
        resnums=resnums,
        anm_eigvecs=eigenvectors,
        anm_eigvals=eigenvalues,
    )


def test_fig4_pocket_renderer_colours_only_measured_tbd_residues(tmp_path, monkeypatch):
    _write_minimal_render_inputs(tmp_path)
    fake_cmd = _FakeCmd()
    monkeypatch.setenv("FIG4_ROOT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "pymol", SimpleNamespace(cmd=fake_cmd))

    runpy.run_path(str(ROOT / "scripts" / "render_fig4_pocket.py"), run_name="__main__")

    calls = fake_cmd.calls
    sentinel_call = ("alter", "pocket", "b=-1.0")
    grey_call = ("color", "grey70", "pocket")
    assert sentinel_call in calls
    assert ("alter", "pocket", "b=0.0") not in calls
    assert grey_call in calls

    spectrum_calls = [call for call in calls if call[0] == "spectrum"]
    assert spectrum_calls == [
        (
            "spectrum",
            "b",
            "blue_white_red",
            "pocket and resi 318+319+330+424 and name CA",
            0.0,
            1.0,
        )
    ]
    assert all(call[3] != "pocket and name CA" for call in spectrum_calls)

    residue_alters = [call for call in calls if call[0] == "alter" and call[1] != "pocket"]
    assert calls.index(sentinel_call) < min(calls.index(call) for call in residue_alters)
    assert calls.index(grey_call) < min(calls.index(call) for call in spectrum_calls)
    assert [call[1] for call in residue_alters] == [
        "pocket and resi 318",
        "pocket and resi 319",
        "pocket and resi 330",
        "pocket and resi 424",
    ]
