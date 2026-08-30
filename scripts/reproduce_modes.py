#!/usr/bin/env python3
"""Reproduce the CRBN dynamics analysis: PCA of the curated ensemble, an
anisotropic network model (ANM) built on the open reference structure, a
Gaussian network model (GNM), and the overlap/RMSIP cross-checks.

Numpy-only (no ProDy) so it is portable and matches the committed arrays.
The 269-residue window is read from data/crbn_residue_window.csv.

Inputs
  data/crbn_ensemble.ens.npz     70 x 269 x 3 curated Cα (superposed)
Outputs (regenerated; --verify cross-checks them against the matching saved
reference)
  crbn_pca.npz                   pc1..3, variance ratios, PC1 scores, diff vector
  crbn_pc_projections.csv        per-structure PC1/PC2 with open/closed label
  crbn_residue_fluctuations.csv  per-residue ANM & PCA square fluctuation

Headline numbers reproduced: PC1 coordinate-variance fraction 88.3%, PC1-axis directional
overlap 0.9996, ANM(open) mode-1 directional overlap 0.744, and ANM-PCA RMSIP 0.641.

Verification can read either the local ``data/`` directory or an external
data directory/ZIP without copying or extracting it.

Usage:  python scripts/reproduce_modes.py [--verify] [--data-source PATH]
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

CUTOFF_ANM = 15.0   # Å
CUTOFF_GNM = 10.0   # Å  (GNM contact cutoff; Fig 3a cross-correlation map)
N_MODES = 20

ENSEMBLE_NAME = "crbn_ensemble.ens.npz"
WINDOW_NAME = "crbn_residue_window.csv"
MODE_NAME = "crbn_anm_modes.npz"
VERIFY_INPUTS = (ENSEMBLE_NAME, WINDOW_NAME, MODE_NAME)
REFERENCE_KEYS = (
    "anm_diff_overlap",
    "cum_overlap",
    "anm_eigvals",
    "anm_eigvecs",
    "rmsip",
    "resnums",
    "overlap_anm_pca",
)


class AnalysisDataSource:
    """Read exact analysis inputs from a directory or a ZIP without extraction."""

    def __init__(self, path: Path, archive: bool) -> None:
        self.path = path
        self.archive = archive

    @classmethod
    def open(cls, raw_path: str | os.PathLike[str]) -> "AnalysisDataSource":
        path = Path(raw_path)
        if path.is_dir():
            return cls(path, archive=False)
        if not path.exists():
            raise ValueError(f"data source does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"data source is neither a directory nor a regular ZIP: {path}")
        try:
            with zipfile.ZipFile(path, "r") as archive:
                archive.infolist()
        except (OSError, zipfile.BadZipFile) as exc:
            raise ValueError(f"data source is not a readable ZIP: {path}") from exc
        return cls(path, archive=True)

    @property
    def description(self) -> str:
        kind = "ZIP data source" if self.archive else "data directory"
        return f"{kind} ({self.path})"

    @staticmethod
    def _member(name: str) -> str:
        return f"data/{name}"

    def _zip_info(self, archive: zipfile.ZipFile, name: str) -> zipfile.ZipInfo:
        member = self._member(name)
        matches = [info for info in archive.infolist() if info.filename == member]
        if len(matches) > 1:
            raise ValueError(f"{self.path}: duplicate required ZIP member {member}")
        if not matches or matches[0].is_dir():
            raise FileNotFoundError(member)
        if matches[0].flag_bits & 0x1:
            raise ValueError(f"{self.path}: encrypted ZIP member is not supported: {member}")
        return matches[0]

    def preflight(self, names: Sequence[str]) -> None:
        missing: list[str] = []
        if not self.archive:
            missing = [name for name in names if not (self.path / name).is_file()]
        else:
            with zipfile.ZipFile(self.path, "r") as archive:
                for name in names:
                    try:
                        self._zip_info(archive, name)
                    except FileNotFoundError:
                        missing.append(self._member(name))
        if missing:
            raise ValueError(
                f"{self.description}: missing required input(s): {', '.join(missing)}"
            )

    def read_bytes(self, name: str) -> bytes:
        if not self.archive:
            path = self.path / name
            try:
                return path.read_bytes()
            except FileNotFoundError as exc:
                raise ValueError(f"{self.description}: missing required input: {name}") from exc
        with zipfile.ZipFile(self.path, "r") as archive:
            try:
                info = self._zip_info(archive, name)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"{self.description}: missing required input: {self._member(name)}"
                ) from exc
            try:
                return archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise ValueError(
                    f"{self.description}: could not read ZIP member {info.filename}"
                ) from exc

    def read_text(self, name: str) -> str:
        try:
            return self.read_bytes(name).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.description}: {name} is not valid UTF-8") from exc

    def load_npz(self, name: str) -> np.lib.npyio.NpzFile:
        try:
            if self.archive:
                return np.load(io.BytesIO(self.read_bytes(name)), allow_pickle=False)
            return np.load(self.path / name, allow_pickle=False)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError(f"{self.description}: {name} is not a readable NPZ") from exc


def require(condition: object, message: object) -> None:
    """Keep verification checks active under ``python -O``."""

    if not bool(condition):
        raise AssertionError(message)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the primary PCA/ANM measurements and optionally verify them."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="cross-check results without writing analysis outputs",
    )
    parser.add_argument(
        "--data-source",
        type=Path,
        metavar="PATH",
        help=(
            "external directory containing the required files directly, or a ZIP containing "
            "them under data/ (verification only)"
        ),
    )
    args = parser.parse_args(argv)
    if args.data_source is not None and not args.verify:
        parser.error("--data-source requires --verify; external sources are read-only")
    return args

def _copy_reference(npz: np.lib.npyio.NpzFile, description: str) -> dict[str, np.ndarray]:
    missing = [key for key in REFERENCE_KEYS if key not in npz.files]
    if missing:
        raise ValueError(f"{description}: mode reference lacks required key(s): {', '.join(missing)}")
    return {key: np.asarray(npz[key]).copy() for key in REFERENCE_KEYS}


def load_source_reference(source: AnalysisDataSource) -> tuple[dict[str, np.ndarray], str]:
    with source.load_npz(MODE_NAME) as npz:
        return _copy_reference(npz, source.description), source.description


def load_verify_reference(
    source: AnalysisDataSource, *, use_git_head: bool
) -> tuple[dict[str, np.ndarray], str]:
    """Load the immutable default reference from git or the authoritative source."""

    if use_git_head:
        path = f"data/{MODE_NAME}"
        blob = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            capture_output=True,
            check=False,
        )
        if blob.returncode == 0 and blob.stdout:
            try:
                with np.load(io.BytesIO(blob.stdout), allow_pickle=False) as npz:
                    return _copy_reference(npz, "committed snapshot (git HEAD)"), (
                        "committed snapshot (git HEAD)"
                    )
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                raise ValueError(f"committed snapshot is not a readable NPZ: {path}") from exc
    return load_source_reference(source)


def load_analysis_inputs(
    source: AnalysisDataSource,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and validate the coordinate tensor, residue window, and labels."""

    with source.load_npz(ENSEMBLE_NAME) as ensemble:
        missing = [key for key in ("_confs", "_labels") if key not in ensemble.files]
        if missing:
            raise ValueError(
                f"{source.description}: {ENSEMBLE_NAME} lacks key(s): {', '.join(missing)}"
            )
        try:
            confs = np.asarray(ensemble["_confs"], dtype=float).copy()
            raw_labels = np.asarray(ensemble["_labels"]).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source.description}: {ENSEMBLE_NAME} has invalid coordinate or label arrays"
            ) from exc

    if confs.ndim != 3 or confs.shape[0] < 2 or confs.shape[2] != 3:
        raise ValueError(
            f"{source.description}: _confs must have shape (n>=2, residues, 3), got {confs.shape}"
        )
    if not np.isfinite(confs).all():
        raise ValueError(f"{source.description}: _confs contains non-finite coordinates")
    if raw_labels.ndim != 1 or len(raw_labels) != confs.shape[0]:
        raise ValueError(
            f"{source.description}: _labels must be one-dimensional with {confs.shape[0]} entries"
        )
    full_labels = [str(label).strip() for label in raw_labels]
    if any(not label for label in full_labels) or len(set(full_labels)) != len(full_labels):
        raise ValueError(f"{source.description}: _labels must be non-empty and unique")
    labels = [label.split("_")[0].split()[0][:4] for label in full_labels]

    try:
        reader = csv.DictReader(io.StringIO(source.read_text(WINDOW_NAME)))
        if reader.fieldnames is None or "author_resnum" not in reader.fieldnames:
            raise ValueError(f"{WINDOW_NAME}: missing author_resnum column")
        values: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            raw = row.get("author_resnum")
            if raw is None or not raw.strip():
                raise ValueError(f"{WINDOW_NAME}:{line_number}: missing author_resnum")
            try:
                values.append(int(raw))
            except ValueError as exc:
                raise ValueError(
                    f"{WINDOW_NAME}:{line_number}: invalid author_resnum {raw!r}"
                ) from exc
    except csv.Error as exc:
        raise ValueError(f"{source.description}: {WINDOW_NAME} is malformed CSV") from exc

    if len(values) != confs.shape[1]:
        raise ValueError(
            f"{source.description}: residue window has {len(values)} rows but _confs has "
            f"{confs.shape[1]} residues"
        )
    if len(values) != 269:
        raise ValueError(f"{source.description}: expected 269 ordered residues, found {len(values)}")
    if len(set(values)) != len(values) or any(
        right <= left for left, right in zip(values, values[1:])
    ):
        raise ValueError(
            f"{source.description}: author_resnum values must be unique and strictly increasing"
        )
    return confs, np.asarray(values, dtype=int), labels


def validate_reference(
    reference: dict[str, np.ndarray], resnums: np.ndarray, description: str
) -> dict[str, np.ndarray]:
    """Validate the complete scientific contract of a saved mode reference."""

    expected_shapes = {
        "anm_diff_overlap": (N_MODES,),
        "cum_overlap": (N_MODES,),
        "anm_eigvals": (N_MODES,),
        "anm_eigvecs": (3 * len(resnums), N_MODES),
        "rmsip": (),
        "resnums": (len(resnums),),
        "overlap_anm_pca": (10, 10),
    }
    checked: dict[str, np.ndarray] = {}
    for key, shape in expected_shapes.items():
        try:
            value = np.asarray(reference[key], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{description}: invalid numeric reference key {key}") from exc
        if value.shape != shape:
            raise ValueError(f"{description}: {key} shape {value.shape}, expected {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{description}: {key} contains non-finite values")
        checked[key] = value
    if not np.array_equal(checked["resnums"], np.asarray(resnums, dtype=float)):
        raise ValueError(f"{description}: resnums do not exactly match the analysis window")
    if np.any(checked["anm_eigvals"] <= 0):
        raise ValueError(f"{description}: anm_eigvals must be strictly positive")
    gram = checked["anm_eigvecs"].T @ checked["anm_eigvecs"]
    if not np.allclose(gram, np.eye(N_MODES), rtol=0.0, atol=1e-8):
        raise ValueError(f"{description}: anm_eigvecs are not an orthonormal basis")
    return checked

def pca(confs):
    n, m, _ = confs.shape
    mean = confs.mean(0)
    X = (confs - mean).reshape(n, -1)
    C = np.cov(X.T)
    w, v = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    vr = w / w.sum()
    scores = X @ v
    return mean, v, w, vr, scores

def gnm_kirchhoff(coords, cutoff):
    n = len(coords); K = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[j] - coords[i]) <= cutoff:
                K[i, j] = K[j, i] = -1; K[i, i] += 1; K[j, j] += 1
    return K

def anm_hessian(coords, cutoff):
    n = len(coords); H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            d = coords[j] - coords[i]; r = np.linalg.norm(d)
            if r <= cutoff:
                k = np.outer(d, d) / r**2
                H[3*i:3*i+3, 3*j:3*j+3] = -k
                H[3*j:3*j+3, 3*i:3*i+3] = -k
                H[3*i:3*i+3, 3*i:3*i+3] += k
                H[3*j:3*j+3, 3*j:3*j+3] += k
    return H

def modes_from(H, k):
    w, v = np.linalg.eigh(H)
    nz = w > 1e-9
    return w[nz][:k], v[:, nz][:, :k]

def run_analysis(args: argparse.Namespace) -> int:
    verify = bool(args.verify)
    source = AnalysisDataSource.open(args.data_source or Path("data"))
    source.preflight(VERIFY_INPUTS if verify else (ENSEMBLE_NAME, WINDOW_NAME))
    confs, resnums, labels = load_analysis_inputs(source)

    mean, pcv, pcw, vr, scores = pca(confs)
    pc1 = pcv[:, 0]
    # normalise PC1 scores to the RMSD-like convention used in the committed data
    # (score = projection / sqrt(n_atoms)) so open cluster lands near 8-9, closed near 0
    s1 = scores[:, 0] / np.sqrt(confs.shape[1])
    if abs(s1.min()) > abs(s1.max()):   # sign so the open cluster is positive
        s1 = -s1; pc1 = -pc1; pcv[:, 0] = -pcv[:, 0]; scores[:, 0] = -scores[:, 0]
    # open/closed split by the natural gap: the 5 fully-open structures are cleanly
    # separated from the rest (a large jump in sorted PC1). Cut at the widest gap in
    # the top of the distribution.
    srt = np.sort(s1)[::-1]
    gaps = srt[:-1] - srt[1:]
    ncut = int(np.argmax(gaps[:15])) + 1     # index of the widest gap among the leaders
    thresh = (srt[ncut-1] + srt[ncut]) / 2
    open_mask = s1 >= thresh
    require(0 < int(open_mask.sum()) < len(open_mask), "PC1 split produced an empty state")
    scores[:, 0] = s1                         # store normalised PC1
    diff = confs[open_mask].mean(0) - confs[~open_mask].mean(0)
    dvec = diff.reshape(-1)
    dnorm = float(np.linalg.norm(dvec))
    require(np.isfinite(dnorm) and dnorm > 0.0, "open/closed difference vector is degenerate")
    dvec /= dnorm
    ov_pc1_diff = abs(pc1 @ dvec)

    # ANM on the open 8CVP conformer only. Its coordinates are already stored in the
    # curated tensor in the same rigidly superposed frame as the difference vector.
    # Superposition changes only rotation and translation, not the internal geometry
    # used to build the ANM Hessian.
    matches = [i for i, label in enumerate(labels) if label == "8CVP"]
    require(len(matches) == 1, f"expected one 8CVP conformer, found {len(matches)}")
    open_ca = confs[matches[0]]
    Ha = anm_hessian(open_ca, CUTOFF_ANM)
    aw, av = modes_from(Ha, N_MODES)
    require(
        aw.shape == (N_MODES,) and av.shape == (3 * len(resnums), N_MODES),
        f"ANM produced incomplete modes: eigvals {aw.shape}, eigvecs {av.shape}",
    )
    anm_diff_overlap = np.array([abs(av[:, m] @ dvec) for m in range(N_MODES)])
    # secondary axis using only the three apo open structures (8CVP/8D7X/8D7Y) as the
    # open end-state; reported in the text as a sensitivity check (0.77) alongside the
    # canonical five-open/65-closed axis (0.744).
    apo_labels = {"8CVP", "8D7X", "8D7Y"}
    apo_mask = np.array([lab in apo_labels for lab in labels])
    if apo_mask.sum() == 3:
        diff_apo = confs[apo_mask].mean(0) - confs[~open_mask].mean(0)
        dvec_apo = diff_apo.reshape(-1)
        apo_norm = float(np.linalg.norm(dvec_apo))
        require(np.isfinite(apo_norm) and apo_norm > 0.0, "three-apo axis is degenerate")
        dvec_apo /= apo_norm
        anm_apo_overlap = abs(av[:, 0] @ dvec_apo)
    else:
        anm_apo_overlap = float("nan")
    # ANM-PCA RMSIP over 10x10
    pca_modes = pcv[:, :10]
    ov = np.array([[abs(av[:, i] @ pca_modes[:, j]) for j in range(10)] for i in range(10)])
    rmsip = np.sqrt((ov[:10, :10]**2).sum() / 10)

    print(f"PC1 variance {vr[0]*100:.1f}%  PC2 {vr[1]*100:.1f}%  PC3 {vr[2]*100:.1f}%")
    print(f"PC1-difference overlap {ov_pc1_diff:.3f}")
    print(f"ANM(open) mode-1 overlap {anm_diff_overlap[0]:.3f}  top-10 cum "
          f"{np.sqrt((anm_diff_overlap[:10]**2).sum()):.3f}")
    print(f"ANM mode-1 overlap, three-apo-only axis {anm_apo_overlap:.3f}")
    print(f"ANM-PCA RMSIP {rmsip:.3f}")

    # GNM modes (Fig 3a) and the combined mode artifact consumed by the figure builders
    Kg = gnm_kirchhoff(open_ca, CUTOFF_GNM)
    gw_all, gv_all = np.linalg.eigh(Kg)
    keep_g = gw_all > 1e-8
    gw, gv = gw_all[keep_g][:20], gv_all[:, keep_g][:, :20]
    require(
        gw.shape == (N_MODES,) and gv.shape == (len(resnums), N_MODES),
        f"GNM produced incomplete modes: eigvals {gw.shape}, eigvecs {gv.shape}",
    )
    if not verify:
        np.savez("data/crbn_anm_modes.npz",
                 anm_eigvals=aw[:20], anm_eigvecs=av[:, :20],
                 gnm_eigvals=gw, gnm_eigvecs=gv,
                 # keep all ten PC columns: the RMSIP is a 10x10 quantity, so a
                 # truncated matrix cannot be recomputed from the saved artifact
                 overlap_anm_pca=ov[:, :10], anm_diff_overlap=anm_diff_overlap,
                 cum_overlap=np.sqrt(np.cumsum(anm_diff_overlap**2)),
                 rmsip=rmsip, resnums=resnums)
        np.savez("data/crbn_pca.npz", mean=mean, pcs=pcv[:, :3], eigvals=pcw[:3],
                 variance_ratio=vr[:10], pc1_scores=scores[:, 0], pc2_scores=scores[:, 1],
                 diff_vec=dvec, open_mask=open_mask)
        # pca_diffvec.npz: the canonical open–closed axis (sign arbitrary) plus the
        # data-derived open set. This is
        # the source consumed by build_fig1.py, pca_robustness.py, anm_robustness.py,
        # anm_null_significance.py and hinge_intermediate.py (the latter three assert their
        # fallback list against it). The open set here is NOT hardcoded -- it is the
        # widest-gap cut of the PC1 distribution computed above.
        np.savez("data/pca_diffvec.npz", diff_vec=dvec, labels=np.array(labels),
                 pc1_scores=scores[:, 0], open_mask=open_mask)
        with open("data/crbn_pc_projections.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n"); w.writerow(["pdb", "PC1", "PC2", "state"])
            for i, lab in enumerate(labels):
                w.writerow([lab, f"{scores[i,0]:.3f}", f"{scores[i,1]:.3f}",
                            "open" if open_mask[i] else "closed"])
    # per-residue square fluctuations (ANM slow modes, PCA)
    anm_sqf = np.zeros(len(resnums)); pca_sqf = np.zeros(len(resnums))
    for m in range(10):
        anm_sqf += (av[:, m].reshape(-1, 3)**2).sum(1) / aw[m]
    for m in range(10):
        pca_sqf += (pcv[:, m].reshape(-1, 3)**2).sum(1) * pcw[m]
    if not verify:
        with open("data/crbn_residue_fluctuations.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n"); w.writerow(["resnum", "anm_sqfluct", "pca_sqfluct"])
            for i, r in enumerate(resnums):
                w.writerow([int(r), f"{anm_sqf[i]:.6f}", f"{pca_sqf[i]:.6f}"])
        print("wrote data/crbn_pca.npz, pca_diffvec.npz, crbn_pc_projections.csv, "
              "crbn_residue_fluctuations.csv")
    else:
        print("verify mode: analysis output files left untouched")

    if verify:
        # Cross-check against the matching saved reference, not against a file written
        # during this run (that would be a self-comparison that cannot fail).
        # Tolerances allow small numerical differences between this numpy
        # implementation and the committed arrays (CA extraction order, eigensolver
        # conventions, and the ProDy origin of the first committed version); the
        # science reproduces to <0.03 on every metric.
        current_raw, current_src = load_source_reference(source)
        current = validate_reference(current_raw, resnums, current_src)
        reference_raw, src = load_verify_reference(
            source,
            use_git_head=args.data_source is None,
        )
        reference = validate_reference(reference_raw, resnums, src)
        require(abs(vr[0] * 100 - 88.3) < 0.5, vr[0] * 100)
        require(ov_pc1_diff > 0.98, ov_pc1_diff)
        require(abs(anm_diff_overlap[0] - 0.744) < 0.01, anm_diff_overlap[0])
        # Verify the complete matrix in the current artifact. In verify mode this script does
        # not write outputs, so this is an independent recomputation rather than a self-check.
        saved_ov = current["overlap_anm_pca"]
        matrix_dmax = float(np.max(np.abs(saved_ov - ov)))
        require(matrix_dmax < 2e-3, matrix_dmax)

        # Compare the primary ANM arrays against the immutable reference, not just the
        # RMSIP scalar. Eigenvectors are sign-arbitrary and are checked by absolute cosine.
        drift = []
        for key, got in (
            ("anm_diff_overlap", anm_diff_overlap),
            ("cum_overlap", np.sqrt(np.cumsum(anm_diff_overlap**2))),
            ("anm_eigvals", aw[:20]),
            ("resnums", resnums),
        ):
            want = reference[key]
            got_array = np.asarray(got, float)
            if want.shape != got_array.shape:
                drift.append(f"{key}: shape {got_array.shape} vs {want.shape}")
            else:
                dmax = float(np.max(np.abs(got_array - want)))
                if dmax > 2e-3:
                    drift.append(f"{key}: max |diff| {dmax:.2e}")
        # Eigenvectors are sign-arbitrary, so compare them up to a per-column sign.
        ev = reference["anm_eigvecs"]
        if ev.shape != av[:, :20].shape:
            drift.append(f"anm_eigvecs: shape {av[:, :20].shape} vs {ev.shape}")
        else:
            cos = np.abs((av[:, :20] * ev).sum(0))
            if float(cos.min()) < 0.999:
                drift.append(f"anm_eigvecs: worst |cos| {float(cos.min()):.4f}")
        require(
            abs(rmsip - float(reference["rmsip"])) < 0.02,
            (rmsip, float(reference["rmsip"])),
        )
        require(not drift, f"recomputed arrays disagree with {src}:\n  " + "\n  ".join(drift))
        print(f"cross-checked primary ANM arrays against {src}; complete 10x10 overlap matrix verified")
        require(int(open_mask.sum()) == 5, int(open_mask.sum()))
        require(abs(anm_apo_overlap - 0.77) < 0.02, anm_apo_overlap)
        print(f"verify OK: PC1 {vr[0]*100:.1f}%, PC1-diff overlap {ov_pc1_diff:.3f}, "
              f"ANM mode-1 {anm_diff_overlap[0]:.3f}, RMSIP {rmsip:.3f}, 5 open "
              f"(88.3% / 0.9996 / 0.744 / 0.641)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_analysis(args)
    except (AssertionError, KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        mode = "verify" if args.verify else "analysis"
        raise SystemExit(f"{mode} aborted: {exc}") from None

if __name__ == "__main__":
    raise SystemExit(main())
