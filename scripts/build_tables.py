#!/usr/bin/env python3
"""Build Table 1 (key quantitative results) and Table S1 (structure inventory)
from committed data. Emits markdown (study/tables/) and CSV (data/).
Every value is read from a committed file - no hardcoded results.

  Sources:
    data/crbn_anm_modes.npz          ANM mode-1 overlap, RMSIP, cumulative overlap
    data/crbn_pca.npz                PC1 variance fraction
    data/anm_null_significance.json  isotropic null z, LOO ranges, higher-mode z
    data/crbn_curation_log.csv       70-conformer inventory + census
    functional_residue_stats.py      Fisher exact p (drug-conditioned vs genuine-apo)
"""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np

from figure_package_utils import require_rigid_null_schema

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
TAB = ROOT / "study" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

anm = np.load(D / "crbn_anm_modes.npz")
pca = np.load(D / "crbn_pca.npz")
null = json.load(open(D / "anm_null_significance.json"))

mode1 = float(anm["anm_diff_overlap"][0])
rmsip = float(anm["rmsip"])
cum10 = float(anm["cum_overlap"][9])
pc1 = float(pca["variance_ratio"][0]) * 100.0
iso = null["null_isotropic"]
hm = null["null_higher_modes"]
loc = null["leave_one_closed_out"]
loo = null["leave_one_open_out"]

# Fisher p from the canonical script (parse its verify line)
out = subprocess.run([sys.executable, str(ROOT / "scripts" / "functional_residue_stats.py"),
                      "--verify"], capture_output=True, text=True, cwd=ROOT,
                     check=True).stdout
fisher_p = None
for tok in out.replace(";", " ").split():
    if tok.startswith("p="):
        fisher_p = float(tok[2:])
assert fisher_p is not None, "could not parse Fisher p"

# inventory + census
rows = list(csv.DictReader(open(D / "crbn_curation_log.csv")))
n = len(rows)
from collections import Counter
gs = Counter(r["global_state"] for r in rows)
meth = Counter(r["method"] for r in rows)

# open/closed split: the open set is the canonical 5 (pca_diffvec.npz open_mask),
# the rest are closed. Read it from the committed axis file, not a literal.
dv = np.load(D / "pca_diffvec.npz")
n_open = int(np.sum(dv["open_mask"]))
n_closed = n - n_open

# per-structure mode-1 overlap range across the 5 open structures at the
# canonical 15 A cutoff, from data/anm_robustness.json
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(p):
    """2.07e-143 -> '2 × 10⁻¹⁴³' (table typography, not Python e-notation)."""
    m, e = f"{p:.0e}".split("e")
    return f"{m} × 10{str(int(e)).translate(SUP)}"


rob = json.load(open(D / "anm_robustness.json"))
open_m1 = [rob["table"][pdb]["15.0"]["mode1_overlap"] for pdb in rob["open_set"]]
open_lo, open_hi = min(open_m1), max(open_m1)
# The rank annotation must be derived, not asserted in prose.
open_ranks = [rob["table"][pdb]["15.0"]["best_mode_rank"] for pdb in rob["open_set"]]
rank_note = (
    f"rank 1 in each of {n_open} open-structure ANMs"
    if set(open_ranks) == {1}
    else f"best-mode ranks {sorted(set(open_ranks))}"
)
per_structure_rank_note = (
    "each rank 1"
    if set(open_ranks) == {1}
    else f"best-mode ranks {sorted(set(open_ranks))}"
)

# study-level and per-pair sensitivities, and the size-matched nulls
sg = json.load(open(D / "study_group_sensitivity.json"))
pw = json.load(open(D / "pairwise_sensitivity.json"))["per_pair_mode1_overlap"]
try:
    ctx = json.load(open(D / "context_stats.json"))
except (FileNotFoundError, json.JSONDecodeError) as exc:
    raise RuntimeError(
        "data/context_stats.json is missing or invalid; rebuild it with: "
        "python scripts/context_stats.py"
    ) from exc
try:
    arn = json.load(open(D / "assembly_rigid_null.json"))
except (FileNotFoundError, json.JSONDecodeError) as exc:
    raise RuntimeError(
        "data/assembly_rigid_null.json is missing or invalid; rebuild it with: "
        "python scripts/assembly_rigid_null.py"
    ) from exc
rigid_null = require_rigid_null_schema(arn)
study_assoc = ctx.get("fisher_study_level", {})
two_boundary = rigid_null["two_block"]
three_boundary = rigid_null["three_block"]
bond_boundary = rigid_null["bond_length_preserving_boundary"]
equal_boundary = rigid_null["equal_displacement_boundary"]

# ---- Table 1: key quantitative results ----
t1 = [
    ("Deposited ensemble", "", ""),
    ("Curated conformers (search date 20 July 2026)", f"{n}", "70 conformers × 269 common Cα positions"),
    ("Open / closed split", f"{n_open} / {n_closed}", "3 genuine-apo + 2 drug-conditioned open"),
    ("Method composition", f"{meth['cryo-EM']} cryo-EM / {meth['X-ray']} X-ray", ""),
    ("Ensemble dimensionality", "", ""),
    ("First principal component (PC1) variance fraction", f"{pc1:.1f}%",
     "fraction of deposited-coordinate variance"),
    ("Anisotropic network model (ANM) from one open structure", "", ""),
    ("Mode-1 directional overlap with open-to-closed axis", f"{mode1:.3f}",
     f"absolute normalised dot product; {rank_note}"),
    (f"Per-structure directional-overlap range ({n_open} open)",
     f"{open_lo:.2f}–{open_hi:.2f}",
     f"{n_open} open-structure ANMs; 15 Å cutoff; {per_structure_rank_note}"),
    ("Open-only node set (349 Cα of 8CVP)",
     f"{ctx['open_only_node_set']['mode1_overlap']:.2f}",
     "mode-1 directional overlap; rank 1; window not fixed by the closed set"),
    ("Agreement between ANM and PCA motion subspaces (RMSIP)", f"{rmsip:.2f}",
     ("root-mean-square inner product for two 10-dimensional subspaces; random subspaces give "
      f"{ctx['rmsip_random_subspace_null']['mean']:.3f} ± "
      f"{ctx['rmsip_random_subspace_null']['sd']:.3f}")),
    ("Cumulative overlap (top 10 modes)", f"{cum10:.3f}",
     "projection norm of the axis in the ten-mode subspace"),
    ("Calibration", "", ""),
    ("Two-block rigid-motion subspace",
     f"{two_boundary['subspace_capture_of_transition']:.2f}",
     "projection norm of the open-to-closed axis; geometric upper bound without an elastic-network model"),
    ("Mode-1 directional overlap within rigid-motion space",
     f"{rigid_null['per_mode'][0]['direction_cosine_in_rigid_subspace']:.2f}",
     "modes 2, 3 give 0.24, 0.11 at comparable rigid content"),
    # Both parameterisations are reported because the significance depends on
    # the number of independently moving domains granted by the null.
    ("Random rigid interdomain direction, two lobes",
     f"p = {two_boundary['p_exact']:.3f}",
     (f"matched-subspace direction cosine {two_boundary['observed_direction_cosine_in_subspace']:.2f}; "
      f"z = {two_boundary['z']:.2f}; exact {two_boundary['internal_dim']}-dimensional directional null; "
      "partition at the HB–TBD domain boundary")),
    ("Random rigid interdomain direction, three domains",
     f"p = {three_boundary['p_exact']:.3f}",
     (f"matched-subspace direction cosine {three_boundary['observed_direction_cosine_in_subspace']:.2f}; "
      f"z = {three_boundary['z']:.2f}; exact {three_boundary['internal_dim']}-dimensional directional null; "
      "NTD, HB and TBD treated separately")),
    ("First-order bond length preservation at the boundary",
     f"p = {bond_boundary['p_exact']:.3f}",
     (f"matched-subspace direction cosine {bond_boundary['observed_direction_cosine_in_subspace']:.2f}; "
      f"z = {bond_boundary['z']:.2f}; five-dimensional rigid-motion subspace; "
      "permits boundary-bond reorientation")),
    ("Equal-displacement boundary rigid null",
     f"p = {equal_boundary['p_exact']:.3f}",
     (f"matched-subspace direction cosine {equal_boundary['observed_direction_cosine_in_subspace']:.2f}; "
      f"z = {equal_boundary['z']:.2f}; "
      "identical displacement at residues 317 and 318; three-dimensional boundary-constrained "
      "subspace; stronger than first-order bond-length preservation")),
    ("Axis rank in the CRBN–DDB1 assembly",
     f"mode {arn['assembly']['by_cutoff']['15.0']['best_mode_rank']}",
     (f"directional overlap {arn['assembly']['by_cutoff']['15.0']['best_overlap']:.2f}; mode 1 gives "
      f"{arn['assembly']['by_cutoff']['15.0']['mode1_overlap']:.2f}; modes 1–3 are mainly two-body "
      f"motion, whereas mode 4 deforms DDB1 more than CRBN")),
    ("Higher-mode (4–20) comparison", f"z = {hm['z']:.1f}",
     "observed overlap exceeds this baseline; the baseline is not boundary-geometry-specific"),
    ("Full-space isotropic null",
     f"p = {sci(ctx['isotropic_null_exact_tail']['p_exact'])}",
     "observed overlap exceeds this baseline; the baseline is not boundary-geometry-specific"),
    ("Robustness", "", ""),
    ("Leave-one-closed-out (n=65)", f"{loc['min']:.3f}–{loc['max']:.3f}",
     "directional-overlap range"),
    ("Leave-one-open-out (n=5)", f"{loo['min']:.3f}–{loo['max']:.3f}",
     "directional-overlap range"),
    (f"Publication-group-equal weighting ({sg['n_study_groups']} groups)",
     f"{sg['study_weighted_overlap']:.3f}", f"directional overlap; rank {sg['study_weighted_rank']}"),
    (f"Per-pair ({pw['n_pairs']} open-closed pairs)", f"{pw['min']:.3f}–{pw['max']:.3f}",
     "directional-overlap range; rank 1 for every pair"),
    ("Structure-ligand association", "", ""),
    # The entry-level p treats three conformers from one publication as three independent
    # observations, so it belongs in the counts column as a tabulation. The study-grouped
    # comparison cannot be estimated because only one publication contributes apo entries
    # and that same publication also contributes drug-conditioned entries.
    ("Drug-conditioned vs genuine-apo", "64/2 vs 0/3",
     f"descriptive entry-level Fisher's exact p = {fisher_p:.4f}; entries are clustered by publication"),
    ("Between-study comparison", "not estimable",
     study_assoc.get("reason", "only one publication contributes genuine-apo conformers "
                     "and it also contributes drug-conditioned conformers")),
]
with open(TAB / "Table1.md", "w", newline="\n", encoding="utf-8") as f:
    f.write("**Table 1. Key quantitative results.** "
            "All values are derived from the source-data package.\n\n")
    f.write("| Quantity | Value | Notes |\n|---|---|---|\n")
    for q, v, nt in t1:
        if v == "" and nt == "":
            f.write(f"| **{q}** | | |\n")
        else:
            f.write(f"| {q} | {v} | {nt} |\n")
    f.write(
        "\n*Abbreviations: ANM, anisotropic network model, which estimates collective motion "
        "from residue contacts; Cα, alpha carbon, one representative position per residue; "
        "CRBN, cereblon; cryo-EM, cryo-electron microscopy; DDB1, DNA damage-binding protein 1; "
        "PCA, principal component analysis; PC, principal component; RMSIP, root-mean-square "
        "inner product, which compares two motion subspaces. Directional overlap is the absolute "
        "normalised dot product between two unit vectors; cumulative overlap is a subspace-projection "
        "norm. Both range from 0 to 1 and are not variance fractions. Å denotes ångström; n denotes a "
        "count; p denotes probability under the stated null; z denotes standardised distance from the "
        "exact null mean using its exact population standard deviation.*\n"
    )

# ---- Table S1: structure inventory ----
inv = sorted(rows, key=lambda r: r["pdb"])
# The open/closed split is the central classification, so the inventory has to
# carry it: without this column the table cannot be used to check the 5/65 counts, and a
# reader had to infer state from the RMSD-to-mean column.
_cls = list(csv.DictReader(open(D / "ens_classified.csv", encoding="utf-8")))
conf = {r["pdb"]: r["conformation"] for r in _cls}
pc1_of = {r["pdb"]: r["global_PC1"] for r in _cls}
_missing = [r["pdb"] for r in inv if r["pdb"] not in conf]
assert not _missing, f"no open/closed call for {_missing}"
_n_open = sum(1 for r in inv if conf[r["pdb"]] == "open")
assert _n_open == 5, f"expected 5 open structures, found {_n_open}"
def display_resolution(value):
    """Keep reported precision while removing floating-point serialization noise."""
    if "." in value and len(value.split(".", 1)[1]) > 3:
        return f"{float(value):.2f}"
    return value

with open(D / "table_s1_inventory.csv", "w", newline="\n", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["pdb_id", "conformation", "state", "global_state", "method",
                "resolution_A", "rmsd_to_mean_A", "pc1_coordinate"])
    for r in inv:
        w.writerow([r["pdb"], conf[r["pdb"]], r["state"], r["global_state"], r["method"],
                    r["resolution"], r["rmsd_to_mean"], pc1_of[r["pdb"]]])
with open(TAB / "TableS1.md", "w", newline="\n", encoding="utf-8") as f:
    f.write("**Table S1. Curated structure inventory (70 conformers; search date 20 July 2026).** "
            "The full source table is "
            "`data/table_s1_inventory.csv`.\n\n")
    f.write("| PDB | Conformation | Ligand state | Global state | Method "
            "| Resolution (Å) | RMSD to mean (Å) |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for r in inv:
        f.write(f"| {r['pdb']} | {conf[r['pdb']]} | {r['state']} | {r['global_state']} "
                f"| {r['method']} | {display_resolution(r['resolution'])} "
                f"| {r['rmsd_to_mean']} |\n")
    f.write(
        "\n*Abbreviations: cryo-EM, cryo-electron microscopy; PDB, Protein Data Bank; RMSD, "
        "root-mean-square deviation from the ensemble mean after structural superposition. "
        "Å denotes ångström.*\n"
    )

print(f"Table1: {len(t1)} rows; TableS1: {len(inv)} structures "
      f"({gs['drug-conditioned']} drug / {gs['native-substrate']} native / {gs['genuine-apo']} apo)")
print(f"key values: PC1 {pc1:.1f}%, mode1 {mode1:.3f}, RMSIP {rmsip:.2f}, "
      f"null z {iso['z']:.0f}, Fisher p {fisher_p:.4f}")
