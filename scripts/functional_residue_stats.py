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

def main():
    gs   = {r["pdb"]: r["global_state"] for r in csv.DictReader(open("data/crbn_curation_log.csv"))}
    conf = {r["pdb"]: r["state"] for r in csv.DictReader(open("data/crbn_pc_projections.csv"))}

    def counts(states):
        cl = op = 0
        for pdb, s in gs.items():
            if s in states:
                if conf.get(pdb, "closed") == "closed": cl += 1
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
    # and that same study also deposited six of the closed drug-conditioned entries. The
    # clustered test collapses each primary-citation DOI to one observation per arm. With a
    # single apo study the smallest p this design can reach is about 1/43 = 0.023, so the
    # entry-level p is reported as a tabulation and this one as the inferential statement.
    doi = {r["pdb"]: (r["primary_citation_doi"] or f"no_doi:{r['pdb']}")
           for r in csv.DictReader(open("data/curation_study_groups.csv"))}
    arms = {}
    for pdb, state in gs.items():
        if state not in ("drug-conditioned", "genuine-apo"):
            continue
        key = (state, doi.get(pdb, f"no_doi:{pdb}"))
        arms.setdefault(key, []).append(conf.get(pdb))
    tab = {"drug-conditioned": [0, 0], "genuine-apo": [0, 0]}
    for (state, _), calls in arms.items():
        # A study counts once per arm. A study that deposited any open conformer counts as
        # open: this is the assignment least favourable to the association being tested,
        # since the hypothesis is that drug-conditioned entries are closed. The opposite
        # convention (closed if any closed) gives p = 0.047, so the choice matters and we
        # take the conservative one.
        tab[state][1 if "open" in calls else 0] += 1
    p_study = fisher_exact([tab["drug-conditioned"], tab["genuine-apo"]])[1]
    n_studies = len({d for pdb, d in doi.items() if pdb in gs})
    print(f"study-clustered ({n_studies} publication groups): "
          f"drug-conditioned {tab['drug-conditioned'][0]} closed / {tab['drug-conditioned'][1]} open, "
          f"genuine-apo {tab['genuine-apo'][0]} closed / {tab['genuine-apo'][1]} open, "
          f"p_study = {p_study:.4f}")

    # open sub-ensemble composition (all five open structures)
    open_states = [s for pdb, s in gs.items() if conf.get(pdb) == "open"]
    from collections import Counter
    print(f"open sub-ensemble: {dict(Counter(open_states))} = {len(open_states)} open")

    if "--verify" in sys.argv:
        assert (dc_cl, dc_op) == (64, 2), (dc_cl, dc_op)
        assert (ga_cl, ga_op) == (0, 3), (ga_cl, ga_op)
        assert abs(p - 0.000191) < 1e-5, p
        assert len(open_states) == 5, len(open_states)
        # the clustered test must not be significant at 0.05; if this ever flips, the
        # reported sentence describing it has to change with it
        assert p_study > 0.05, p_study
        print(f"verify OK: drug-conditioned 64/2, genuine-apo 0/3, p={p:.6f}; 5 open "
              f"(3 genuine-apo + 2 drug-conditioned); p_study={p_study:.4f}")

if __name__ == "__main__":
    main()
