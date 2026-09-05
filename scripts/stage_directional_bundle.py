#!/usr/bin/env python3
"""Verify and stage a CRBN directional-mechanics bundle.

The bundle restores public-code-independent data into two locations:
root-level ``data/``/``render/`` inputs required by the public code checkout and
directional outputs under ``results/directional_mechanics``. Public code
snapshots are verified by hash but are not staged over the checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))
try:
    from stage_strengthening_bundle import safe_name, write_verified  # type: ignore
finally:
    sys.path.pop(0)


def destination(name: str, repo: Path, output: Path) -> Path | None:
    safe_name(name)
    if name.startswith(("data/", "render/")):
        return repo / name
    if name.startswith("directional/"):
        return output / name.removeprefix("directional/")
    return None


def _read_manifest(archive: zipfile.ZipFile) -> dict:
    try:
        return json.loads(archive.read("BUNDLE_MANIFEST.json"))
    except KeyError as exc:
        raise ValueError("Bundle lacks BUNDLE_MANIFEST.json") from exc


def _verify_members(archive: zipfile.ZipFile, manifest: dict) -> None:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP entry")
    for info in archive.infolist():
        safe_name(info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("Symlink in bundle")
    expected = {row["path"] for row in manifest["files"]} | {"BUNDLE_MANIFEST.json", "README.md"}
    if set(names) != expected:
        raise ValueError("Manifest/ZIP member mismatch")


def _verify_hashes(archive: zipfile.ZipFile, manifest: dict) -> None:
    for row in manifest["files"]:
        raw = archive.read(row["path"])
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Bundle hash mismatch: {row['path']}")


def _verify_aliases(archive: zipfile.ZipFile, manifest: dict, repo: Path, output: Path) -> None:
    file_names = {row["path"] for row in manifest["files"]}
    seen: set[str] = set()
    for alias in manifest.get("aliases", []):
        safe_name(alias["source"])
        safe_name(alias["path"])
        if alias["source"] not in file_names:
            raise ValueError("Invalid alias source")
        if alias["path"] in file_names or alias["path"] in seen:
            raise ValueError("Invalid alias role or duplicate alias")
        if destination(alias["source"], repo, output) is None or destination(alias["path"], repo, output) is None:
            raise ValueError("Invalid alias role or duplicate alias")
        if hashlib.sha256(archive.read(alias["source"])).hexdigest() != alias["sha256"]:
            raise ValueError("Alias hash mismatch")
        seen.add(alias["path"])


def stage(bundle: Path, repo: Path, output: Path) -> dict[str, object]:
    with zipfile.ZipFile(bundle) as archive:
        manifest = _read_manifest(archive)
        _verify_members(archive, manifest)
        _verify_hashes(archive, manifest)
        _verify_aliases(archive, manifest, repo, output)
        for row in manifest["files"]:
            dest = destination(row["path"], repo, output)
            if dest is not None:
                write_verified(dest, archive.read(row["path"]))
        for alias in manifest.get("aliases", []):
            src = destination(alias["source"], repo, output)
            dst = destination(alias["path"], repo, output)
            if src is None or dst is None:
                raise ValueError("Invalid alias role")
            raw = src.read_bytes()
            if hashlib.sha256(raw).hexdigest() != alias["sha256"]:
                raise ValueError("Alias hash mismatch")
            write_verified(dst, raw)

    controls = output / "data/controls"
    if controls.is_dir():
        for source in controls.glob("*.cif.gz"):
            write_verified(repo / "data/_controls_cif_cache" / source.name, source.read_bytes())
    return {
        "manifest_files": len(manifest["files"]),
        "aliases": len(manifest.get("aliases", [])),
        "public_commit": manifest["public_commit"],
        "output": str(output),
        "excluded_mode_vectors": len(manifest.get("excluded_generated_mode_vectors", [])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/directional_mechanics")
    args = parser.parse_args(argv)
    print(json.dumps(stage(args.bundle, args.repo_root.resolve(), args.output_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
