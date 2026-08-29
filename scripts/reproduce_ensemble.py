#!/usr/bin/env python3
"""Reproduce the RCSB->curation stage of the CRBN ensemble study.

The RCSB is a live archive, so the ensemble is defined against an explicit
SEARCH FREEZE DATE (see FREEZE_DATE). Re-running after that date will find
additional depositions; the study ensemble is the archive as of the freeze.

Pipeline (all steps computed from RCSB, none hardcoded):
  1. Query the RCSB Search API for every polymer entity whose reference
     sequence is UniProt Q96SW2 (human cereblon). The query JSON and the
     returned entity list are written to data/rcsb_query_Q96SW2.json.
  2. Fetch each entity's sequence length from the RCSB Data API and drop
     entities shorter than MIN_LEN (isolated peptide fragments).
  3. Inventory: write every entity with length, fragment-filter status, and
     whether it is in the curated analysis ensemble.

The 70-conformer x 269-Ca coordinate tensor (data/crbn_ensemble.ens.npz) is
rebuilt end-to-end from raw mmCIF by reproduce_tensor.py; this script covers
the query/inventory stage and cross-checks that every curated conformer is a
real Q96SW2 deposition present in the current archive.

Outputs
  crbn_structure_inventory.csv   inventory: entity, pdb, length,
                                 passes_fragment_filter, in_curated_ensemble
  rcsb_query_Q96SW2.json         the exact query + freeze date + returned ids

Usage
  python scripts/reproduce_ensemble.py --verify
      Query the live archive and CHECK it against the committed freeze snapshot.
      Writes nothing to the freeze files. Optionally records the live result under
      data/live_audit/<date>.json with --record-audit.
  python scripts/reproduce_ensemble.py --write-freeze
      Explicitly (re)write the freeze snapshot. Only for defining a NEW freeze date;
      it overwrites data/rcsb_query_Q96SW2.json and data/crbn_structure_inventory.csv.

The freeze files are study evidence: they record the archive as of FREEZE_DATE and
must not change when someone merely verifies the pipeline.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

RCSB_QUERY = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
UNIPROT = "Q96SW2"
POST_FREEZE_AUDIT = "data/post_freeze_rcsb_audit.json"
MIN_LEN = 50          # entities shorter than this are isolated peptide fragments
FREEZE_DATE = "2026-07-20"   # RCSB search freeze; ensemble = archive as of this date
# Archive size at freeze (recorded, not asserted against a live query, since the
# live archive grows over time). The curated ensemble is frozen; re-derive the
# tensor with reproduce_tensor.py to re-evaluate new depositions.
FROZEN_TOTAL_ENTITIES = 99
FROZEN_STRUCTURES_AFTER_FRAGMENT = 98
ENSEMBLE_PATH = Path("data/crbn_ensemble.ens.npz")
TRUSTED_ENSEMBLE_SHA256 = "3ae8c142f2b171002da532b40361d9e652ab1f9f20b3ccb5b293a6ffd9a6ed9f"


def load_trusted_residue_window(path=ENSEMBLE_PATH):
    """Load the sole required ProDy object only after archive authentication."""
    import numpy as np

    path = Path(path)
    with path.open("rb") as archive_handle:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: archive_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != TRUSTED_ENSEMBLE_SHA256:
            raise RuntimeError(
                f"refusing to unpickle unauthenticated ensemble {path}: {digest}"
            )
        archive_handle.seek(0)
        with np.load(archive_handle, allow_pickle=True) as ensemble:
            atoms = ensemble["_atoms"]
            window = [int(residue) for residue in atoms[0].getResnums()]
    if len(window) != 269 or len(set(window)) != len(window):
        raise RuntimeError("trusted ensemble contains an invalid residue window")
    return window

def load_post_freeze_audit():
    with open(POST_FREEZE_AUDIT, encoding="utf-8") as fh:
        audit = json.load(fh)
    assert audit["freeze_date"] == FREEZE_DATE, audit["freeze_date"]
    assert audit["uniprot"] == UNIPROT, audit["uniprot"]
    entries = audit["post_freeze_entities"]
    assert len(entries) == 8, len(entries)
    assert len({e["pdb_entity"] for e in entries}) == len(entries)
    assert all(e["initial_release_date"] > FREEZE_DATE for e in entries)
    included = [e for e in entries if e["paper_rule_call"] == "eligible_post_freeze"]
    excluded = [e for e in entries if e["paper_rule_call"] == "excluded_post_freeze"]
    sensitivity = audit["sensitivity_if_eligible_entries_are_added"]
    assert len(included) == 4 and len(excluded) == 4, (len(included), len(excluded))
    assert sensitivity["n_conformers"] == 74, sensitivity
    assert sensitivity["n_residues"] == 269, sensitivity
    assert sensitivity["anm_best_rank"] == 1, sensitivity
    return audit

def rcsb_polymer_entities(accession=UNIPROT):
    q = {"query": {"type": "terminal", "service": "text",
                   "parameters": {"attribute":
                       "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                       "operator": "exact_match", "value": accession}},
         "return_type": "polymer_entity",
         "request_options": {"return_all_hits": True}}
    req = urllib.request.Request(RCSB_QUERY, data=json.dumps(q).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return d["total_count"], [x["identifier"] for x in d["result_set"]], q

def entity_lengths(entity_ids):
    q = ("query($ids:[String!]!){polymer_entities(entity_ids:$ids){"
         "rcsb_id entity_poly{rcsb_sample_sequence_length}}}")
    out = {}
    for k in range(0, len(entity_ids), 50):
        body = json.dumps({"query": q, "variables": {"ids": entity_ids[k:k+50]}}).encode()
        req = urllib.request.Request(RCSB_GRAPHQL, data=body,
                                     headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        for pe in d["data"]["polymer_entities"]:
            if pe:
                out[pe["rcsb_id"]] = pe["entity_poly"]["rcsb_sample_sequence_length"]
    return out

def main():
    verify = "--verify" in sys.argv
    write_flags = {"--write-window", "--write-freeze", "--record-audit"}
    requested_writes = sorted(write_flags.intersection(sys.argv))
    if verify and requested_writes:
        sys.exit("verify is read-only; do not combine --verify with write flags: "
                 + ", ".join(requested_writes))

    import numpy as np

    total, entities, query = rcsb_polymer_entities()
    print(f"RCSB Q96SW2 polymer entities (live): {total}  [freeze {FREEZE_DATE}: {FROZEN_TOTAL_ENTITIES}]")

    lengths = entity_lengths(entities)
    fragments = sorted(e for e, L in lengths.items() if L < MIN_LEN)
    passed = [e for e in entities if lengths.get(e, 0) >= MIN_LEN]
    n_after_fragment = len({e.split("_")[0] for e in passed})
    print(f"fragment entities (<{MIN_LEN} res): {fragments} "
          f"(lengths {[lengths[e] for e in fragments]})")
    print(f"after fragment exclusion: {n_after_fragment} structures "
          f"[freeze {FREEZE_DATE}: {FROZEN_STRUCTURES_AFTER_FRAGMENT}]")

    with np.load(ENSEMBLE_PATH, allow_pickle=False) as ens:
        kept = sorted({str(label)[:4] for label in ens["_labels"]})
        n_conf, n_ca, _ = ens["_confs"].shape
    print(f"analysis window {n_ca} Cα; curated conformers {n_conf}")

    # The 269-residue analysis window is emitted as a plain-text committed input
    # (data/crbn_residue_window.csv) so that the downstream numpy-only scripts need
    # neither ProDy nor any artifact they themselves produce.
    if "--write-window" in sys.argv:
        win = load_trusted_residue_window()
        with open("data/crbn_residue_window.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["index", "author_resnum"])
            for i, r in enumerate(win):
                w.writerow([i, r])
        print(f"REWROTE data/crbn_residue_window.csv ({len(win)} residues)")
    else:
        committed = [int(row["author_resnum"]) for row in
                     csv.DictReader(open("data/crbn_residue_window.csv"))]
        assert len(committed) == n_ca, "committed residue window disagrees with the ensemble"
        print(f"residue window matches the ensemble ({len(committed)} residues)")

    if "--write-freeze" in sys.argv:
        with open("data/rcsb_query_Q96SW2.json", "w") as fh:
            json.dump({"freeze_date": FREEZE_DATE, "endpoint": RCSB_QUERY,
                       "query": query, "total_count": total,
                       "entities": sorted(entities)}, fh, indent=1)
        with open("data/crbn_structure_inventory.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["pdb_entity", "pdb_id", "seq_length",
                        "passes_fragment_filter", "in_curated_ensemble"])
            for e in sorted(entities):
                pdb = e.split("_")[0]
                w.writerow([e, pdb, lengths.get(e, ""),
                            int(lengths.get(e, 0) >= MIN_LEN), int(pdb in kept)])
        print(f"REWROTE freeze snapshot: data/crbn_structure_inventory.csv "
              f"({len(entities)} entities) and data/rcsb_query_Q96SW2.json")
    elif "--record-audit" in sys.argv:
        import datetime, os
        os.makedirs("data/live_audit", exist_ok=True)
        stamp = datetime.date.today().isoformat()
        out = f"data/live_audit/{stamp}.json"
        with open(out, "w") as fh:
            json.dump({"audited_on": stamp, "freeze_date": FREEZE_DATE,
                       "live_total_count": total, "endpoint": RCSB_QUERY,
                       "entities": sorted(entities)}, fh, indent=1)
        print(f"wrote live audit {out} (freeze snapshot untouched)")
    else:
        print("freeze snapshot left untouched "
              "(use --write-freeze to redefine it, --record-audit to log the live archive)")

    if verify:
        # the committed freeze snapshot is the study evidence: check it is intact and
        # that the live archive is a superset of it
        frozen = json.load(open("data/rcsb_query_Q96SW2.json"))
        assert frozen["freeze_date"] == FREEZE_DATE, (frozen["freeze_date"], FREEZE_DATE)
        assert frozen["total_count"] == FROZEN_TOTAL_ENTITIES, frozen["total_count"]
        frozen_entities = set(frozen["entities"])
        assert len(frozen_entities) == FROZEN_TOTAL_ENTITIES, len(frozen_entities)
        lost = sorted(frozen_entities - set(entities))
        assert not lost, f"entities present at freeze but absent from the live archive: {lost}"
        new_since = sorted(set(entities) - frozen_entities)
        print(f"freeze snapshot intact: {len(frozen_entities)} entities, none withdrawn; "
              f"{len(new_since)} deposited since the freeze {new_since if new_since else ''}")
        audit = load_post_freeze_audit()
        audited_entities = sorted(e["pdb_entity"] for e in audit["post_freeze_entities"])
        missing_from_live = sorted(set(audited_entities) - set(entities))
        assert not missing_from_live, (
            f"post-freeze audit entities absent from live archive: {missing_from_live}")
        unexpected = sorted(set(audited_entities) - set(new_since))
        assert not unexpected, (
            f"post-freeze audit contains entities not absent from the freeze snapshot: {unexpected}")
        included = [e for e in audit["post_freeze_entities"]
                    if e["paper_rule_call"] == "eligible_post_freeze"]
        excluded = [e for e in audit["post_freeze_entities"]
                    if e["paper_rule_call"] == "excluded_post_freeze"]
        sens = audit["sensitivity_if_eligible_entries_are_added"]
        print("post-freeze audit snapshot: %d entities released after %s; "
              "%d eligible under paper rules, %d excluded by fixed-window coverage; "
              "eligible-addition sensitivity %dx%d, open=%d, ANM m1 %.3f rank %d"
              % (len(audited_entities), FREEZE_DATE, len(included), len(excluded),
                 sens["n_conformers"], sens["n_residues"], sens["n_open"],
                 sens["anm_mode1_overlap"], sens["anm_best_rank"]))
        subprocess.run(
            [sys.executable, "scripts/post_freeze_sensitivity.py", "--verify"],
            check=True,
        )
        entity_pdbs = {e.split("_")[0] for e in entities}
        missing = [p for p in kept if p not in entity_pdbs]
        assert not missing, f"curated conformers absent from RCSB set: {missing}"
        assert len(fragments) >= 1, fragments
        # Live archive may exceed the freeze; require it never SHRANK below freeze.
        assert total >= FROZEN_TOTAL_ENTITIES, (total, FROZEN_TOTAL_ENTITIES)
        assert n_after_fragment >= FROZEN_STRUCTURES_AFTER_FRAGMENT
        print(f"verify OK: live {total} entities >= freeze {FROZEN_TOTAL_ENTITIES}; "
              f"fragment {fragments} dropped; {n_after_fragment} structures; "
              f"curated {n_conf} conformers all present in current Q96SW2 archive")

if __name__ == "__main__":
    main()
