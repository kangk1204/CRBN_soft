import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage_review_bundle import stage


def make_bundle(path, *, extra=None, bad_hash=False):
    h = lambda raw: hashlib.sha256(raw).hexdigest()
    base_raw = b"old input"
    new_raw = b"new result"
    base = json.dumps({"files": [{"path": "data/input.csv", "bytes": len(base_raw), "sha256": h(base_raw)}]}).encode()
    new = {"baseline_manifest_sha256": h(base), "baseline_readme_sha256": h(b"baseline"),
           "readme_sha256": h(b"review"), "public_commit": "a" * 40,
           "external_raw_data": "Pinned archive needed for trajectory recomputation",
           "files": [{"path": "review_response/analysis/result.csv", "bytes": len(new_raw),
                      "sha256": "bad" if bad_hash else h(new_raw)}]}
    with zipfile.ZipFile(path, "w") as z:
        for name, raw in {"BUNDLE_MANIFEST.json": base, "README.md": b"baseline",
                          "REVIEW_BUNDLE_MANIFEST.json": json.dumps(new).encode(), "REVIEW_README.md": b"review",
                          "data/input.csv": base_raw, "review_response/analysis/result.csv": new_raw}.items():
            z.writestr(name, raw)
        if extra:
            z.writestr(extra, b"x")


def test_verified_and_repeatable_staging(tmp_path):
    bundle = tmp_path / "bundle.zip"
    make_bundle(bundle)
    repo, output = tmp_path / "repo", tmp_path / "out"
    result = stage(bundle, repo, output)
    assert result["staged_files"] == 2
    assert (repo / "data/input.csv").read_bytes() == b"old input"
    assert (output / "analysis/result.csv").read_bytes() == b"new result"
    assert stage(bundle, repo, output)["verified"]


@pytest.mark.parametrize("extra", ["../escape", "unexpected.csv"])
def test_rejects_unexpected_or_unsafe_members_before_write(tmp_path, extra):
    bundle = tmp_path / "bundle.zip"
    make_bundle(bundle, extra=extra)
    with pytest.raises(ValueError):
        stage(bundle, tmp_path / "repo", tmp_path / "out")
    assert not (tmp_path / "repo").exists()


def test_corrupt_source_rejected_before_write(tmp_path):
    bundle = tmp_path / "bundle.zip"
    make_bundle(bundle, bad_hash=True)
    with pytest.raises(ValueError, match="hash"):
        stage(bundle, tmp_path / "repo", tmp_path / "out")
    assert not (tmp_path / "repo").exists()


def test_late_destination_conflict_causes_no_early_writes(tmp_path):
    bundle = tmp_path / "bundle.zip"
    make_bundle(bundle)
    conflict = tmp_path / "out/analysis/result.csv"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("different")
    with pytest.raises(ValueError, match="conflicts"):
        stage(bundle, tmp_path / "repo", tmp_path / "out")
    assert not (tmp_path / "repo").exists()


def test_destination_symlink_rejected(tmp_path):
    bundle = tmp_path / "bundle.zip"
    make_bundle(bundle)
    repo = tmp_path / "repo"
    repo.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (repo / "data").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        stage(bundle, repo, tmp_path / "out")
    assert not list(elsewhere.iterdir())
