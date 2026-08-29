#!/usr/bin/env python3
"""Categorical statistics linking global ligand/substrate state to CRBN conformation.

The primary association uses a three-way global-state classification (drug-conditioned /
native-substrate / genuine-apo; column global_state in data/crbn_curation_log.csv):

  drug-conditioned : an IMiD/glue/PROTAC was present in the crystallisation/complex
                     (includes 6BN8, a dBET55 PROTAC ternary complex whose PROTAC is
                     unmodelled in the coordinates)
  native-substrate : a native substrate/cofactor complex (9NR3, GLUL-cN)
  genuine-apo      : no ligand or substrate (8CVP, 8D7X, 8D7Y; all open)

Fisher's exact on drug-conditioned vs genuine-apo (closed vs open): the 66 drug-conditioned
structures are 64 closed / 2 open, the 3 genuine-apo are 0 closed / 3 open, p = 0.000191.
This is a property of the PDB sample, not a thermodynamic population estimate.

Inputs
  data/crbn_curation_log.csv       global_state per structure
  data/crbn_pc_projections.csv     open/closed per structure (from reproduce_modes.py)

Usage:  python scripts/functional_residue_stats.py [--verify]
"""
import sys, csv
from scipy.stats import fisher_exact

try:
    from study_groups import load_study_groups
except ModuleNotFoundError:
    from scripts.study_groups import load_study_groups


def read_unique(path, key):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    out = {}
    for row in rows:
        value = row[key].strip().upper()
        if not value or value in out:
            raise ValueError(f"{path}: blank or duplicate {key}={value!r}")
        out[value] = row
    return out


def main():
    curation = read_unique("data/crbn_curation_log.csv", "pdb")
    projections = read_unique("data/crbn_pc_projections.csv", "pdb")
    if set(curation) != set(projections):
        raise ValueError(
            "curation/projection label mismatch: "
            f"missing projections={sorted(set(curation) - set(projections))}, "
            f"unexpected projections={sorted(set(projections) - set(curation))}"
        )
    gs = {pdb: row["global_state"] for pdb, row in curation.items()}
    conf = {pdb: row["state"] for pdb, row in projections.items()}
    bad_states = sorted({state for state in gs.values()
                         if state not in {"drug-conditioned", "native-substrate", "genuine-apo"}})
    bad_calls = sorted({state for state in conf.values() if state not in {"open", "closed"}})
    if bad_states or bad_calls:
        raise ValueError(f"invalid categorical labels: global_state={bad_states}, state={bad_calls}")

    def counts(states):
        cl = op = 0
        for pdb, s in gs.items():
            if s in states:
                if conf[pdb] == "closed": cl += 1
                else: op += 1
        return cl, op

    dc_cl, dc_op = counts({"drug-conditioned"})
    ga_cl, ga_op = counts({"genuine-apo"})
    ns_cl, ns_op = counts({"native-substrate"})
    odds, p = fisher_exact([[dc_cl, dc_op], [ga_cl, ga_op]])

    print(f"drug-conditioned: {dc_cl} closed / {dc_op} open  (n={dc_cl+dc_op})")
    print(f"genuine-apo:      {ga_cl} closed / {ga_op} open  (n={ga_cl+ga_op})")
    print(f"native-substrate: {ns_cl} closed / {ns_op} open  (n={ns_cl+ns_op}; 9NR3)")
    print(f"Fisher exact (drug-conditioned vs genuine-apo) p = {p:.6f}")

    # The entry-level table treats each deposition as an independent observation, which it
    # is not: all three genuine-apo entries (8CVP, 8D7X, 8D7Y) come from one publication,
    # and that same study also deposited closed drug-conditioned entries. A study cannot be
    # duplicated into both exposure arms of a Fisher table. With only one apo-contributing
    # publication, an independent study-level association is not estimable from this archive.
    groups = load_study_groups(sorted(gs))
    study_states = {}
    for pdb, state in gs.items():
        if state not in ("drug-conditioned", "genuine-apo"):
            continue
        study = groups[pdb]
        study_states.setdefault(study, set()).add(state)
    apo_studies = sorted(study for study, states in study_states.items()
                         if "genuine-apo" in states)
    cross_arm_studies = sorted(study for study, states in study_states.items()
                               if len(states) > 1)
    study_test_estimable = len(apo_studies) >= 2 and not cross_arm_studies
    n_studies = len(set(groups.values()))
    print(f"study-grouped ({n_studies} curated groups): Fisher not estimable; "
          f"genuine-apo is contributed by {len(apo_studies)} publication and "
          f"{len(cross_arm_studies)} publication contributes to both exposure states")

    # open sub-ensemble composition (all five open structures)
    open_states = [s for pdb, s in gs.items() if conf[pdb] == "open"]
    from collections import Counter
    print(f"open sub-ensemble: {dict(Counter(open_states))} = {len(open_states)} open")

    if "--verify" in sys.argv:
        assert (dc_cl, dc_op) == (64, 2), (dc_cl, dc_op)
        assert (ga_cl, ga_op) == (0, 3), (ga_cl, ga_op)
        assert abs(p - 0.000191) < 1e-5, p
        assert len(open_states) == 5, len(open_states)
        assert not study_test_estimable
        assert n_studies == 38, n_studies
        assert len(apo_studies) == 1, apo_studies
        assert cross_arm_studies, cross_arm_studies
        print(f"verify OK: drug-conditioned 64/2, genuine-apo 0/3, p={p:.6f}; 5 open "
              "(3 genuine-apo + 2 drug-conditioned); study-level association not estimable")

if __name__ == "__main__":
    main()
