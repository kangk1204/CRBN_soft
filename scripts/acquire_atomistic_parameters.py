#!/usr/bin/env python3
"""Acquire the version-pinned, complete official ZAFF parameter sources."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.request


SOURCES = {
    "ZAFF.prep": ("https://ambermd.org/tutorials/advanced/tutorial20/files/zaff/ZAFF.prep",
                  "eefd39e5a4db122813accf40df27669379a91f5ed89b7f35aba5528aa84225a5"),
    "ZAFF.frcmod": ("https://ambermd.org/tutorials/advanced/tutorial20/files/zaff/ZAFF.frcmod",
                    "3492d30e3cb8bfa389fd332f595a5faa30fb1d20b1ad431e0a0c9614a67da275"),
}


def acquire(output_dir: Path, offline: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (url, expected) in SOURCES.items():
        path = output_dir / name
        cached = path.exists()
        if cached:
            body = path.read_bytes()
        elif offline:
            raise FileNotFoundError(f"Offline parameter source is missing: {path}")
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise ValueError(f"Parameter response exceeds size limit: {url}")
        actual = hashlib.sha256(body).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned parameter SHA256 mismatch: {path}")
        if not cached:
            path.write_bytes(body)
        records.append({"name": name, "url": url, "sha256": actual, "bytes": len(body),
                        "acquired_this_run": not cached})
    report = {"verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "offline": offline,
              "files": records, "production_ready": False,
              "scope": "Source acquisition only; CRBN model construction and validation are separate."}
    (output_dir / "acquisition.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/atomistic_parameters"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(acquire(args.output_dir, args.offline), indent=2))


if __name__ == "__main__":
    main()
