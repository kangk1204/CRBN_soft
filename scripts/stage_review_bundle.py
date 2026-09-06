#!/usr/bin/env python3
"""Verify and stage the combined CRBN mechanics and observation-window bundle.

Historical inputs retain their original manifest. New inputs and source tables
have a second manifest. Code snapshots are verified, never installed over code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile

from stage_strengthening_bundle import safe_name, write_verified

ROOT = Path(__file__).resolve().parents[1]


def destination(name: str, repo: Path, output: Path) -> Path | None:
    safe_name(name)
    if name.startswith(("data/", "render/")):
        return repo / name
    if name.startswith("directional/"):
        return repo / "results/directional_mechanics" / name.removeprefix("directional/")
    if name.startswith("review_response/"):
        return output / name.removeprefix("review_response/")
    return None


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verified_plan(z: zipfile.ZipFile, repo: Path, output: Path) -> tuple[list, dict]:
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP entry")
    for item in z.infolist():
        safe_name(item.filename)
        if stat.S_ISLNK(item.external_attr >> 16):
            raise ValueError("Symlink in bundle")
    base_bytes = z.read("BUNDLE_MANIFEST.json")
    base = json.loads(base_bytes)
    new = json.loads(z.read("REVIEW_BUNDLE_MANIFEST.json"))
    if digest(base_bytes) != new["baseline_manifest_sha256"]:
        raise ValueError("Baseline manifest hash mismatch")
    if digest(z.read("README.md")) != new["baseline_readme_sha256"]:
        raise ValueError("Baseline README hash mismatch")
    if digest(z.read("REVIEW_README.md")) != new["readme_sha256"]:
        raise ValueError("Review README hash mismatch")
    rows = base["files"] + new["files"]
    by_name = {row["path"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("Duplicate manifest entry")
    reserved = {"BUNDLE_MANIFEST.json", "README.md", "REVIEW_BUNDLE_MANIFEST.json", "REVIEW_README.md"}
    if set(names) != set(by_name) | reserved:
        raise ValueError("Manifest/ZIP member mismatch")
    plan = []
    for row in rows:
        raw = z.read(row["path"])
        if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
            raise ValueError(f"Bundle hash mismatch: {row['path']}")
        path = destination(row["path"], repo, output)
        if path is not None:
            plan.append((path, row["path"], row["sha256"]))
    aliases = base.get("aliases", []) + new.get("aliases", [])
    seen = set(by_name) | reserved
    for alias in aliases:
        safe_name(alias["source"])
        safe_name(alias["path"])
        if alias["source"] not in by_name or alias["path"] in seen:
            raise ValueError("Invalid alias source or duplicate alias")
        seen.add(alias["path"])
        src = destination(alias["source"], repo, output)
        dst = destination(alias["path"], repo, output)
        if src is None or dst is None or alias["sha256"] != by_name[alias["source"]]["sha256"]:
            raise ValueError("Invalid alias role or hash")
        plan.append((dst, alias["source"], alias["sha256"]))
    paths = [str(row[0].absolute()) for row in plan]
    if len(paths) != len(set(paths)):
        raise ValueError("Multiple members resolve to the same destination")
    # Reject every conflict before writing the first member.
    for path, _, expected in plan:
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"Symlink in staging destination: {path}")
        if path.exists() and (not path.is_file() or digest(path.read_bytes()) != expected):
            raise ValueError(f"Existing file conflicts with verified bundle: {path}")
    return plan, new


def stage(bundle: Path, repo: Path, output: Path, *, verify_only: bool = False) -> dict:
    with zipfile.ZipFile(bundle) as z:
        plan, manifest = verified_plan(z, repo, output)
        if not verify_only:
            for path, member, _ in plan:
                write_verified(path, z.read(member))
    return {"verified": True, "staged": not verify_only, "staged_files": len(plan),
            "public_commit": manifest["public_commit"], "output": str(output),
            "raw_trajectory_requirement": manifest["external_raw_data"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/review_response")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(stage(args.bundle, args.repo_root.resolve(), args.output_dir.resolve(),
                           verify_only=args.verify_only), indent=2))


if __name__ == "__main__":
    main()
