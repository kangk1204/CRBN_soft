#!/usr/bin/env python3
"""Verify that data/enhanced_sampling_zn/PROVENANCE.json still describes the tree.

`recorded_hashes_at_run` is history: it pins each driver hash recorded inside a
committed run output and must never be edited except to add another as-run
record from another committed output. `current_file_sha256` tracks the files as
they stand now, so it goes stale whenever a driver is touched. This check fails
loudly on either kind of drift instead of leaving a misleading hash in the
record.

Usage:  python scripts/check_provenance.py [--update]
"""
import hashlib, json, pathlib, sys

PROV = pathlib.Path("data/enhanced_sampling_zn/PROVENANCE.json")
SEARCH = [pathlib.Path("scripts/enhanced_sampling_zn"), pathlib.Path("scripts")]
OUTPUTS = {
    pathlib.Path("data/enhanced_sampling_zn/umbrella_forward.json"): "es_umbrella.py",
    pathlib.Path("data/enhanced_sampling_zn/umbrella_reverse.json"): "es_umbrella.py",
    pathlib.Path("data/enhanced_sampling_zn/metad2d_trace.json"): "es_metad_2d.py",
    pathlib.Path("data/enhanced_sampling_zn/pathcv_trace.json"): "es_pathcv.py",
}

def sha256(path):
    # Normalise line endings before hashing: the repository stores text with LF
    # (.gitattributes), but a Windows working tree checks out CRLF, which would
    # otherwise make identical content hash differently across platforms.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def binary_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    current = prov.get("current_file_sha256", {})
    recorded = prov.get("recorded_hashes_at_run", {})
    recovered = prov.get("recovered_artifact_sha256", {})
    stale, missing = [], []
    for name, recorded_hash in sorted(current.items()):
        hit = next((d / name for d in SEARCH if (d / name).exists()), None)
        if hit is None:
            missing.append(name); continue
        actual = sha256(hit)
        if actual != recorded_hash:
            stale.append((name, recorded_hash, actual))
            if "--update" in sys.argv:
                current[name] = actual
    for name, rec, act in stale:
        print(f"STALE  {name}\n  recorded {rec}\n  actual   {act}")
    for name in missing:
        print(f"MISSING {name} (recorded in PROVENANCE but not found in the tree)")
    contract_errors = []
    for output, script in OUTPUTS.items():
        if not output.exists():
            contract_errors.append(f"{output}: missing committed output")
            continue
        meta = json.loads(output.read_text(encoding="utf-8")).get("meta", {})
        actual_script = meta.get("script_sha256")
        if not actual_script:
            contract_errors.append(f"{output}: missing meta.script_sha256")
        elif recorded.get(script) != actual_script:
            contract_errors.append(f"{output}: meta.script_sha256 {actual_script} is not "
                                   f"recorded_hashes_at_run[{script!r}]")
        actual_params = meta.get("zn_params_sha256")
        if actual_params and recorded.get("zn_bonded_params.json") != actual_params:
            contract_errors.append(f"{output}: meta.zn_params_sha256 {actual_params} is not "
                                   "recorded_hashes_at_run['zn_bonded_params.json']")
        start_name = pathlib.Path(str(meta.get("start_pdb", ""))).name
        if not start_name:
            start_name = "5FQD.pdb" if "forward" in output.name or "metad2d" in output.name or "pathcv" in output.name else "8CVP.pdb"
        start_hash = meta.get("start_pdb_sha256")
        if start_hash and prov.get("start_structure_sha256", {}).get(start_name) != start_hash:
            contract_errors.append(f"{output}: meta.start_pdb_sha256 {start_hash} is not "
                                   f"start_structure_sha256[{start_name!r}]")
    for err in contract_errors:
        print(f"CONTRACT {err}")
    artifact_errors = []
    for path_text, expected_hash in sorted(recovered.items()):
        path = pathlib.Path(path_text)
        if not path.exists():
            artifact_errors.append(f"{path}: missing recovered artifact")
            continue
        actual_hash = binary_sha256(path)
        if actual_hash != expected_hash:
            artifact_errors.append(
                f"{path}: SHA-256 {actual_hash} != recorded {expected_hash}"
            )
    for err in artifact_errors:
        print(f"ARTIFACT {err}")
    if "--update" in sys.argv and stale:
        prov["current_file_sha256"] = current
        PROV.write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
        print(f"updated {len(stale)} hash(es)")
        return 1 if contract_errors or missing or artifact_errors else 0
    if stale or missing or contract_errors or artifact_errors:
        print(f"FAIL: {len(stale)} stale, {len(missing)} missing, "
              f"{len(contract_errors)} contract, {len(artifact_errors)} artifact")
        return 1
    print(f"provenance OK: {len(current)} current files and {len(OUTPUTS)} "
          f"as-run output contracts match their recorded hashes; {len(recovered)} "
          "recovered artifact(s) match")
    return 0

if __name__ == "__main__":
    sys.exit(main())
