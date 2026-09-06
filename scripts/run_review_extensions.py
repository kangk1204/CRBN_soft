#!/usr/bin/env python3
"""Run fixed-core residue-window and public-data comparisons for CRBN.

The existing directional-mechanics workflow is unchanged. Each selected stage
receives the same configuration and writes below the requested output folder.
Offline mode requires previously acquired source files and never authorizes a
network fallback. Scientific exclusions are recorded by the relevant analysis;
an execution error cannot be reported as a completed stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts/review_extensions_config.json"
STAGES = ("window", "external", "figures")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def validate_config(config: dict) -> None:
    window = config["window_extension"]
    if config["crbn_position_count"] != 269 or window["core_position_count"] != 269:
        raise ValueError("The primary observable must retain the frozen 269 CRBN positions")
    if window["core_internal_dimension"] != 801:
        raise ValueError("Mean core-internal compliance must use the same 801 dimensions")
    if set(config["models"]) != {"isolated", "fixed", "rigid", "flexible"}:
        raise ValueError("All four DDB1 treatments are required")
    if config["contact"]["candidate_count"] != 142:
        raise ValueError("The primary contact universe must remain the original 142 groups")
    if config["contact"]["spring_factors"] != [0.8, 0.9, 1.1, 1.2]:
        raise ValueError("The approved spring perturbation factors have changed")
    if set(window["perturbation_policies"]) != {"original_edges", "expanded_incident_edges"}:
        raise ValueError("Both prespecified perturbation policies are required")


def load_stage(stage: str):
    module_name = {"window": "review_window", "external": "review_external",
                   "figures": "build_review_figures"}[stage]
    try:
        return importlib.import_module(f"scripts.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name not in {"scripts", f"scripts.{module_name}"}:
            raise
        return importlib.import_module(module_name)


def run(config_path: Path, output_dir: Path, offline: bool = False,
        stages: tuple[str, ...] = STAGES) -> dict:
    config_path, output_dir = config_path.resolve(), output_dir.resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    if not stages or len(stages) != len(set(stages)) or set(stages) - set(STAGES):
        raise ValueError("Select distinct stages from window, external, figures")
    protocol = output_dir / "protocol"
    frozen = protocol / "frozen_config.json"
    if frozen.exists() and json.loads(frozen.read_text()) != config:
        raise ValueError("Output folder contains a different frozen configuration")
    write_json(frozen, config)
    report = {"config_sha256": sha256(config_path), "offline": offline,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "requested_stages": list(stages), "status": "running", "stages": {}}
    manifest = output_dir / "verification/analysis_run.json"
    write_json(manifest, report)
    try:
        for stage in stages:
            destination = output_dir / "analysis" / ("figure_sources" if stage == "figures" else stage)
            module = load_stage(stage)
            print(f"Starting {stage}", flush=True)
            if stage == "figures":
                result = module.build(config_path, destination)
            else:
                result = module.run(config_path, destination, offline=offline)
            report["stages"][stage] = {"status": "complete", "result": result}
            write_json(manifest, report)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/review_response")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args(argv)
    report = run(args.config, args.output_dir, args.offline, tuple(args.stages))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
