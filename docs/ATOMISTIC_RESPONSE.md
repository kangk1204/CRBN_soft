# Atomistic test of DDB1 mobility

This optional workflow tests the effect of DDB1 mobility on the response of
CRBN along a fixed structural direction. It is under development. Its current
simulation runner performs a short, zero-force technical test; it does not
produce an equilibrated finite-force comparison or a manuscript result.

The existing network analyses and their frozen inputs remain unchanged. This
workflow measures the same 269 CRBN C-alpha atoms, but simulates a chemically
completed protein construct, water and ions. It compares isolated CRBN with
fixed, rigid and flexible DDB1. The rigid condition preserves the entire DDB1
atomic shape, including side chains, using four mass-carrying anchors and
virtual interaction sites. It is a defined computational boundary condition.

## Environment and input data

Create a separate environment; OpenMM is not required by the network workflow:

```bash
conda env create -f environment-atomistic.yml
conda activate crbn-atomistic125
```

Run commands from the repository root. Stage the matching frozen data bundle
as described in the main README. The preparation needs
`data/crbn_residue_window.csv`, `data/crbn_ensemble.ens.npz`,
`data/pca_diffvec.npz` and `data/_cif_cache/8CVP.cif.gz`. The Git repository alone
does not contain these complete inputs. Core reference coordinates and the
closure vector are transported into the prepared coordinate frame without
refitting the ensemble or changing the measurement window.

The first construct is DDB1 1–1140 and CRBN 64–428 with ACE/NME caps. The
unobserved internal segments DDB1 546–550 and CRBN 342–357, and missing side-chain
atoms, are explicitly restored. A single restored model supports technical
feasibility only. Alternative loop models, protonation choices and salt
conditions must be fixed before the scientific comparison.

## Preparation

```bash
python scripts/atomistic_input_audit.py --config scripts/atomistic_config.json \
  --output-dir results/atomistic/input_audit --offline
python scripts/acquire_atomistic_parameters.py
python scripts/prepare_atomistic_heavy_model.py \
  --config scripts/atomistic_config.json \
  --output-dir results/atomistic/heavy_model --offline
mkdir -p results/atomistic/cap_templates
tleap -f scripts/atomistic_cap_templates.in
python scripts/atomistic_caps.py \
  --input-heavy-pdb results/atomistic/heavy_model/repaired_heavy.pdb \
  --n-template-pdb results/atomistic/cap_templates/ace_met_nme.pdb \
  --c-template-pdb results/atomistic/cap_templates/ace_asp_nme.pdb \
  --output-dir results/atomistic/capped_heavy
python scripts/atomistic_modeled_stereochemistry.py \
  --input results/atomistic/capped_heavy/capped_heavy.pdb \
  --repair-csv results/atomistic/heavy_model/atom_residue_mapping.csv \
  --l-template results/atomistic/cap_templates/ace_met_nme.pdb \
  --output-dir results/atomistic/stereochemistry --offline
python scripts/prepare_atomistic_amber.py \
  --input-pdb results/atomistic/stereochemistry/capped_heavy.pdb \
  --prep data/atomistic_parameters/ZAFF.prep \
  --frcmod data/atomistic_parameters/ZAFF.frcmod \
  --config scripts/atomistic_config.json \
  --output-dir results/atomistic/amber --tleap tleap --offline
```

The parameter acquisition command retrieves two SHA-256-pinned files from the
[official Amber ZAFF distribution](https://ambermd.org/tutorials/advanced/tutorial20/ZAFF.php).
Subsequent invocations can use `--offline`. The selected model is center 1:
four complete CY1 residues and ZN1, including all charges, bonds, angles and
nonbonded exceptions. Changing only the sulfur or zinc charge does not reproduce
this parameterization. The small files under `tests/fixtures/zaff_contract/`
are deliberately incomplete unit-test fixtures and cannot parameterize MD.

The Amber stage uses ff14SB/TIP3P, a 12-A solvent buffer and neutralizing ions.
Neutralization is not a 150-mM salt condition. It retains LEaP warnings and
refuses to construct a mapping after a failed LEaP invocation. A topology
success flag alone does not qualify the geometry for dynamics.

The stereochemistry step only corrects specified newly modeled atoms. It
preserves observed coordinates and the backbone. It refuses to reflect a
side chain containing observed atoms or an unsupported additional stereocenter.
A sign correction can leave a nearly planar center or introduce a clash;
post-minimization stereochemistry and geometry checks remain required.

### Optional template loop closure

When a restored internal loop has invalid geometry, an observed loop template
can be fitted before rebuilding the Amber system. The closure prototype accepts
a separately verified donor-coordinate JSON, the target heavy-atom PDB and its
observed/modeled atom inventory:

```bash
python scripts/atomistic_loop_closure.py --config scripts/atomistic_config.json \
  --input donor_loop_coordinates.json --pdb capped_heavy.pdb \
  --source-csv atom_residue_mapping.csv \
  --output-dir results/atomistic/loop_closure --offline
```

The JSON schema is `schema_version: 1`, `coordinate_unit: "nm"`. Its `donor`
contains `pdb_id`, `label_asym_id`, `source_sha256`, an explicit `canonical_mapping`,
`additional_bonds` and `atoms` with canonical residue, label sequence ID, residue
name, atom name and `xyz_nm`. Its `target` declares the chain, internal loop
residues and two flanking anchor residues. This self-contained coordinate JSON
is hashed; extraction from the stated CIF must be verified separately. A source
identifier alone does not certify coordinate provenance.

The algorithm varies backbone phi/psi and global pose while retaining Pro phi,
peptide omega, sidechain geometry and covalent bond geometry. Both anchors use
N/CA/C/O. Only modeled loop coordinate columns may change. It tests the actual
target junction lengths, angles and peptide planes after reinserting the fixed
target flanks and after PDB rounding. Solver convergence and the independent
geometry gates are reported separately; a bounded solver may produce an
acceptable preparation candidate without establishing a converged optimum.

A `technical_loop_geometry_pass` does not qualify MD. It does not certify all
steric contacts, Ramachandran preferences or force-field relaxation. Rebuild the
complete Amber topology and coordinates from the candidate and rerun the
following preparation stages. Ligand/construct differences in the template and
alternative loop models remain necessary considerations for scientific use.

The optional `--steric-aware` flag adds a coarse heavy-atom clash objective and
screen to the same bounded closure. It includes loop–environment and internal
loop contacts, excluding directly bonded and 1–3 pairs. Every non-loop heavy
atom stays fixed during closure, whether observed or modeled. A fixed 0.20 nm
minimum separation is tested after fitting and PDB rounding; the objective
uses a 0.22 nm margin. All unresolved pairs and the policy are reported. This
screen addresses gross overlap, not atom-specific van der Waals energetics or
full force-field qualification. The original default closure remains available
for reproducing its earlier output.

LEaP-generated hydrogens also need geometric checks. Reconstruct all non-Gly
C-alpha hydrogens from the three heavy-bond directions and the topology's
equilibrium C–H length. This changes only newly added HA coordinates. Next,
minimize modeled atoms, water and ions while keeping the observed solute heavy
atoms fixed. This allows modeled heavy-atom clashes to relax without moving
experimental coordinates. Neither step changes the force field.

```bash
python scripts/atomistic_alpha_hydrogens.py \
  --prmtop results/atomistic/amber/solvated.prmtop \
  --inpcrd results/atomistic/amber/solvated.inpcrd \
  --output-dir results/atomistic/alpha_hydrogens
python scripts/atomistic_preparation_handoff.py restraints \
  --heavy-mapping results/atomistic/heavy_model/atom_residue_mapping.csv \
  --amber-pdb results/atomistic/amber/amber_input_joint_renamed.pdb \
  --prmtop results/atomistic/amber/solvated.prmtop \
  --output results/atomistic/amber/observed_heavy_restraints.json
python scripts/relax_atomistic_hydrogens.py \
  --prmtop results/atomistic/amber/solvated.prmtop \
  --inpcrd results/atomistic/alpha_hydrogens/alpha_hydrogen_repaired.inpcrd \
  --prep data/atomistic_parameters/ZAFF.prep \
  --observed-heavy-json results/atomistic/amber/observed_heavy_restraints.json \
  --output-dir results/atomistic/hydrogen_minimized \
  --max-iterations 1200 --platform OpenCL --offline
```

Proceed only when `relax_atomistic_hydrogens.json` reports
`hydrogen_minimization_complete`. This stage uses no bond constraints and no
dynamics. It requires unchanged selected observed coordinates and correct
post-minimization C-alpha, C-beta and HA stereochemistry. The inventory must be
bound to the same topology hash. Omitting `--observed-heavy-json` fixes all
solute heavy atoms; that diagnostic mode cannot resolve modeled heavy clashes.
Initial near-planar C-beta geometry is recorded as a preparation defect and
must meet the complete nonplanarity criterion after relaxation. A failed
output remains unqualified; these commands do not guarantee a usable model.

Build isolated CRBN with `--assembly isolated` and a separate output directory.
Use the same corrected input and frozen measurement basis. Solvent and ions are
rebuilt after DDB1 removal; this is not a mass-zero approximation to isolation.

Preparation minimization uses temporary restraints on explicitly mapped
observed heavy atoms. Restored atoms can relax. Its output retains the frozen
reference and closure vector; the relaxed starting coordinates are recorded
separately. The subsequent simulation replaces those preparation restraints
with the six collective CRBN translation/rotation restraints.

```bash
python scripts/relax_atomistic_preparation.py \
  --prmtop results/atomistic/amber/solvated.prmtop \
  --inpcrd results/atomistic/hydrogen_minimized/hydrogen_minimized.rst7 \
  --mapping results/atomistic/amber/atomistic_mapping.json \
  --restrain-indices results/atomistic/amber/observed_heavy_restraints.json \
  --prep data/atomistic_parameters/ZAFF.prep \
  --output-dir results/atomistic/minimized --restraint-k 100000 \
  --max-iterations 1000 --platform OpenCL --offline
python scripts/atomistic_preparation_handoff.py qualify \
  --prmtop results/atomistic/amber/solvated.prmtop \
  --inpcrd results/atomistic/hydrogen_minimized/hydrogen_minimized.rst7 \
  --mapping results/atomistic/amber/atomistic_mapping.json \
  --restraints results/atomistic/amber/observed_heavy_restraints.json \
  --amber-pdb results/atomistic/amber/amber_input_joint_renamed.pdb \
  --minimizer-report results/atomistic/minimized/relax_atomistic_preparation.json \
  --minpositions results/atomistic/minimized/minimized_positions.npy \
  --box results/atomistic/minimized/box_vectors_nm.npy \
  --output-dir results/atomistic/qualified
```

The last step rejects stale inputs, missing peptide bonds, wrong or nearly
planar C-alpha/C-beta stereochemistry, and invalid raw Zn–S bonded distances.
It verifies an Amber coordinate/box round trip and binds the input hashes in
`technical_qualification.json`. Qualification is for a short technical test;
it does not establish equilibration or production readiness. Use an external
job deadline for minimization: an iteration limit is not a wall-time limit.
Use a new output directory for every attempt. Failed geometries remain recorded
and do not qualify for the simulation runner. Preparation restraints are removed
when constructing the technical simulation; all stereochemistry and peptide
screens are checked again during its minimization and dynamics.

## Measurement and scientific acceptance

With fixed unit direction `q`, measure `Q = q.T @ (x_core - x_reference)`.
The scientific extension will apply the conjugate energy `-h*Q` at zero and
both signs of two force magnitudes. One common force magnitude is fixed from
the largest zero-force fluctuation among the three complex conditions before
comparing their response. The present technical runner uses zero force only.

`atomistic_response_analysis.py` consumes a CSV with the columns `model`,
`replicate`, `force_kj_mol_nm`, `time_ps` and `closure_nm`. It requires an explicit
equilibration discard, a complete shared force grid and independently justified
replicates. It reports finite-force slopes, sampling precision, linearity and
temporal diagnostics. Replicate labels alone do not establish independence.

`atomistic_covariance.py` consumes an NPZ with `core_displacement_nm` of shape
`(frames,269,3)`, `time_ps`, `reference_nm` and `q_ambient`. Coordinates must be
unwrapped and expressed in the fixed sampling gauge. It does no frame alignment.
It reports the equilibrium covariance estimates `Var(Q)/(RT)` and
`S = 801*Var(Q)/trace(Cov_internal)`, with convergence explicitly unverified.
Zero-force provenance and equilibration require the simulation records.

The protocol requires three independent initializations, adequate effective
sampling, stable estimates over time, finite-force linearity, agreement with
the equilibrium fluctuation estimate and gauge/timestep sensitivity checks.
Only a converged covariance can support normalization by total internal
fluctuation. The code never requires the atomistic result to reproduce the
network-model ordering or its numerical recovery fractions.

## Software checks

```bash
python -m pytest tests/test_atomistic*.py tests/test_prepare_atomistic*.py \
  tests/test_relax_atomistic*.py tests/test_verify_zaff*.py \
  tests/test_zaff*.py -q
OPENMM_TEST_PLATFORM=OpenCL python -m pytest \
  tests/test_atomistic_boundary.py tests/test_atomistic_technical_pilot.py -q
```

Engine tests use double precision. Tests that require the frozen data bundle
skip explicitly when it is absent; this is a code-only check, not reproduction
of the prepared CRBN system. Software tests, preparation checks, short technical
MD and converged scientific response estimates are separate validation stages.
