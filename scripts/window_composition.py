#!/usr/bin/env python3
"""Which conformers force each gap in the 269-residue analysis window?

The window is the intersection of what every conformer resolves, so it is not a neutral
choice: any element that is disordered in a subset of the ensemble is removed for all of it.
If the subset that forces a gap is systematically open or systematically closed, the window
is biased with respect to the very coordinate being measured, and the paper has to say so.

This script computes, per gap, how many conformers fail to resolve it and which they are. It
uses the pipeline's own parser and chain rule (scripts/reproduce_tensor.py) so the counts are
the ones the ensemble was actually built from -- an earlier hand-run of this check used a
different fallback chain rule and produced numbers that were wrong in the paper's favour.

Also reported: the N-terminal belt (below the window start), which is excluded on the same
principle and shows the same asymmetry.

Usage
  python scripts/window_composition.py [--verify]
Output  data/window_composition.json
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce_tensor as R          # noqa: E402  (parser + chain rule of record)

BELT = list(range(40, 77))            # N-terminal belt, below the window
SENSOR_LOOP = (341, 361)              # contains His353


def chain_of(pdb, ca):
    if pdb in R.CHAIN_MAP and R.CHAIN_MAP[pdb] in ca:
        return R.CHAIN_MAP[pdb]
    return R.best_chain(ca, pdb)[0]


def main():
    verify = "--verify" in sys.argv
    R.CACHE_WRITES_ENABLED = not verify
    curated = [r["pdb"] for r in csv.DictReader(
        open(ROOT / "data" / "crbn_curation_log.csv", encoding="utf-8"))]
    state = {r["pdb"]: r["conformation"] for r in csv.DictReader(
        open(ROOT / "data" / "ens_classified.csv", encoding="utf-8"))}
    apo = {"8CVP", "8D7X", "8D7Y"}

    win = [int(x) for x in R.WIN]
    gaps = [(win[i] + 1, win[i + 1] - 1) for i in range(len(win) - 1)
            if win[i + 1] - win[i] > 1]

    resolved = {}
    for pdb in curated:
        ca = R.parse_ca(R.fetch_cif(pdb))
        ch = chain_of(pdb, ca)
        got = ca[ch]
        resolved[pdb] = {
            "chain": ch,
            "gaps": {f"{a}-{b}": int(sum(r in got for r in range(a, b + 1))) for a, b in gaps},
            "belt_40_76": int(sum(r in got for r in BELT)),
        }

    per_gap = {}
    for a, b in gaps:
        key = f"{a}-{b}"
        n = b - a + 1
        missing = {p: resolved[p]["gaps"][key] for p in curated
                   if resolved[p]["gaps"][key] < n}
        per_gap[key] = {
            "length": n,
            "n_conformers_incomplete": len(missing),
            "incomplete": {p: {"resolved": v, "state": state.get(p),
                               "genuine_apo": p in apo} for p, v in sorted(missing.items())},
            "overlaps_sensor_loop": not (b < SENSOR_LOOP[0] or a > SENSOR_LOOP[1]),
        }

    belts = {"genuine_apo": sorted(resolved[p]["belt_40_76"] for p in curated if p in apo),
             "open_ternary": sorted(resolved[p]["belt_40_76"] for p in curated
                                    if state.get(p) == "open" and p not in apo),
             "closed": sorted(resolved[p]["belt_40_76"] for p in curated
                              if state.get(p) == "closed")}
    belt_summary = {k: {"min": min(v), "max": max(v),
                        "median": sorted(v)[len(v) // 2], "n": len(v)}
                    for k, v in belts.items() if v}

    out = {"n_curated": len(curated), "window_size": len(win), "n_gaps": len(gaps),
           "gaps": per_gap, "belt_40_76": belt_summary,
           "note": ("Counts come from scripts/reproduce_tensor.py's parser and chain rule, "
                    "so they describe the ensemble as built.")}

    if not verify:
        with open(ROOT / "data" / "window_composition.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
        print("wrote data/window_composition.json")

    for key, g in per_gap.items():
        if g["n_conformers_incomplete"]:
            who = ", ".join(f"{p} {d['resolved']}/{g['length']}"
                            + ("*" if d["genuine_apo"] else "")
                            for p, d in g["incomplete"].items())
            flag = "  [sensor loop]" if g["overlaps_sensor_loop"] else ""
            print(f"  gap {key:>9} ({g['length']:2d} res): "
                  f"{g['n_conformers_incomplete']} incomplete -> {who}{flag}")
    print("  * = genuine-apo open structure")
    print("  belt 40-76: " + "; ".join(
        f"{k} {v['min']}-{v['max']} (median {v['median']}, n={v['n']})"
        for k, v in belt_summary.items()))

    if verify:
        sl = [g for k, g in per_gap.items() if g["overlaps_sensor_loop"]]
        assert sl, "no gap overlaps the sensor loop"
        for g in sl:
            who = g["incomplete"]
            assert who, "the sensor-loop gap must be forced by someone"
            assert all(d["genuine_apo"] for d in who.values()), who
            assert all(d["resolved"] == 0 for d in who.values()), who
        assert belt_summary["closed"]["median"] > belt_summary["genuine_apo"]["max"]
        print("verify OK: the sensor-loop gap is forced only by genuine-apo open structures, "
              "which resolve none of it, and the belt is better ordered in closed structures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
