#!/usr/bin/env python3
"""Classify the 65 closed CRBN structures as monovalent (IMiD/molecular glue),
bivalent PROTAC, or apo, from the ligands RCSB reports bound in each entry.

For each closed PDB the non-polymer entities are read from the RCSB data API; common
crystallization/buffer/ion components are excluded; the largest-molecular-weight
remaining MODELLED ligand defines the class:
    max modelled ligand MW  > 700 Da  -> PROTAC     (bivalent: two warheads + linker)
    0 < max modelled MW    <= 700 Da  -> monovalent  (single IMiD / glue warhead)
    no modelled drug-like ligand      -> unmodelled  (see note; NOT necessarily apo)

The 700 Da cut is an operational separator for THIS dataset (all deposited bivalent
degraders are > 720 Da, all monovalent warheads < 700 Da), not a scientific definition
of ligand valency; it is used only as a convenience label.

The two entries with no modelled drug-like ligand are NOT apo: their RCSB titles are
recorded in `note`. 6BN8 is a dBET55 PROTAC ternary complex whose PROTAC atoms were not
built into the model; 9NR3 is a drug-free / native-substrate (GLUL-cN) complex. Both are
excluded from the modelled-ligand primary analysis (n = 55 monovalent / 8 PROTAC).

Writes data/closed_ligand_classification.csv, sorted by PDB id
(pdb, max_ligand_mw_Da, ligand_comp_ids, class, note). Requires network access to data.rcsb.org.
"""
import csv, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BUFFER = {"ZN","SO4","GOL","EDO","CL","NA","MG","CA","PO4","ACT","DMS","PEG","FMT",
          "IOD","BR","K","MN","NI","CD","HOH","WAT","TRS","MES","EPE","IMD","BME",
          "PG4","1PE","P6G","PGE","DTT","CIT","FLC","MPD","ACY","NO3","SCN","CO"}

def j(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

pdbs = [r["pdb"] for r in csv.DictReader(open(DATA / "closed_subpca.csv"))]
mw_cache = {}
rows = []
for p in pdbs:
    e = j(f"https://data.rcsb.org/rest/v1/core/entry/{p}")
    title = e.get("struct", {}).get("title", "").strip()
    nps = e.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids", [])
    ligs = []
    for nid in nps:
        ne = j(f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{p}/{nid}")
        comp = ne.get("pdbx_entity_nonpoly", {}).get("comp_id")
        if not comp or comp in BUFFER:
            continue
        if comp not in mw_cache:
            mw_cache[comp] = j(f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp}"
                               )["chem_comp"].get("formula_weight")
        if mw_cache[comp]:
            ligs.append((comp, round(mw_cache[comp], 1)))
    mx = max([m for _, m in ligs], default=0.0)
    cl = "unmodelled" if mx == 0 else "PROTAC" if mx > 700 else "monovalent"
    # for entries with no modelled drug ligand, record the RCSB title so the status is explicit
    note = "" if mx > 0 else f"no modelled drug-like ligand; RCSB title: {title}"
    rows.append({"pdb": p, "max_ligand_mw_Da": mx,
                 "ligand_comp_ids": ";".join(c for c, _ in ligs) or "-",
                 "class": cl, "note": note})

rows.sort(key=lambda r: r["pdb"])          # deterministic PDB-sorted order (stable diffs)
with open(DATA / "closed_ligand_classification.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["pdb", "max_ligand_mw_Da", "ligand_comp_ids", "class", "note"], lineterminator="\n")
    w.writeheader(); w.writerows(rows)
from collections import Counter
print("wrote closed_ligand_classification.csv:", dict(Counter(r["class"] for r in rows)))
