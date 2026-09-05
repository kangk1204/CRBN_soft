#!/usr/bin/env python3
"""Run the frozen CRBN strengthening analyses through one configurable interface.

Raw data must be staged as documented in the accompanying data bundle. Offline
execution uses cached files and never turns a failed acquisition into evidence.
Figure rendering writes analysis figures under the selected output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("ensemble", "ddb1", "contacts", "controls", "external", "maps", "figures")


def commands(config, output, config_path, offline):
    common = ["--offline"] if offline else []
    scripts = ROOT / "scripts"
    def command(name, *args):
        return [sys.executable, str(scripts / name), *map(str, args)]
    return {
        "ensemble": command("strengthen_ensemble.py", "--output-dir", output / "analysis/ensemble",
                            "--structure-dir", output / "data/structures", "--config", config_path, *common),
        "ddb1": command("strengthen_ddb1.py", "--output-dir", output / "analysis/ddb1",
                        "--primary-cutoff", config["primary_cutoff_A"], "--sensitivity-cutoffs",
                        *config["sensitivity_cutoffs_A"], "--alphas", *config["interface_strengths"],
                        "--primary-modes", config["primary_mode_count"], "--sensitivity-modes",
                        config["sensitivity_mode_count"], "--verify", *common),
        "contacts": command("strengthen_contacts.py", "--config", config_path,
                            "--output-dir", output / "analysis/contacts", *common),
        "controls": command("strengthen_controls.py", "--output-dir", output / "analysis/controls", *common),
        "external": command("strengthen_external.py", "--output-dir", output,
                            "--max-qrg", config["saxs"]["primary_qRg_max_when_no_published_range"], *common),
        "maps": command("strengthen_maps.py", "--output-dir", output,
                        "--download-ids", "70776", "70777", "70778", "70781", "70782", "70783", "70784",
                        "--artifact-kinds", "primary_map", "half_map", "additional_map", "mask", *common),
        "figures": command("build_strengthening_figures.py", "--input-root", output,
                           "--output-dir", output / "manuscript/figures",
                           "--source-dir", output / "analysis/figure_sources", "--require-all"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/strengthening_config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/strengthening")
    parser.add_argument("--stages", choices=STAGES, nargs="+", default=list(STAGES))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--show-commands", action="store_true", help="show the exact commands without executing")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    config = json.loads(args.config.read_text())
    cmds = commands(config, output, args.config.resolve(), args.offline)
    if args.show_commands:
        print(json.dumps({s: cmds[s] for s in args.stages}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    report = {"config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "offline": args.offline, "stages": []}
    for stage in args.stages:
        started = time.time()
        print(f"Running {stage}", flush=True)
        log = logs / f"workflow_{stage}.log"
        with log.open("w") as handle:
            result = subprocess.run(
                cmds[stage], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        report["stages"].append({"stage": stage, "command": cmds[stage], "returncode": result.returncode,
                                 "seconds": time.time()-started, "log": str(log.relative_to(output))})
        (logs / "workflow_run.json").write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode:
            print(f"{stage} failed; see {log}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
