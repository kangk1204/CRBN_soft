# CRBN Soft

CRBN Soft provides Python workflows for analysing deposited human cereblon
(CRBN) structures. The code performs coordinate curation, principal component
analysis (PCA), anisotropic network model (ANM) calculations, sensitivity
tests, and reproducible figure and table generation.

This is a code-only repository. It does not include coordinate arrays, cached
Protein Data Bank files, generated images, or word-processing documents.

## Quick start

You need Git and either Conda or Mamba. The commands below work on Linux and
macOS. Windows users can run them in Windows Subsystem for Linux.

Clone the repository:

```bash
git clone https://github.com/kangk1204/CRBN_soft.git
cd CRBN_soft
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate crbn-soft
```

Mamba can be used in the same way:

```bash
mamba env create -f environment.yml
mamba activate crbn-soft
```

Run the checks that do not need analysis inputs:

```bash
python -m pytest -q
python -m ruff check .
```

Both commands should finish without errors.

## Add the analysis inputs

The numerical workflows need a separate CRBN input bundle. Copy its `data/`
directory into the repository root. Do not place it inside another `data/`
directory.

This Git repository does not currently contain or link to a downloadable copy
of that bundle. The repository alone supports the code-only checks in the quick
start, but it is not a clean-clone numerical reproduction package. To run the
numerical workflows, use the exact `data/` and `render/` directories distributed
with the corresponding archived software/data record. If those directories are
not supplied with the release you received, stop here rather than substituting
files from a different run.

The layout should begin like this:

```text
CRBN_soft/
├── data/
│   ├── curation_study_groups.csv
│   ├── curation_study_overrides.csv
│   ├── crbn_ensemble.ens.npz
│   ├── crbn_residue_window.csv
│   ├── crbn_curation_log.csv
│   ├── crbn_anm_modes.npz
│   ├── pca_diffvec.npz
│   └── ...
├── scripts/
└── environment.yml
```

Check that the main coordinate file is in the correct place:

```bash
ls data/crbn_ensemble.ens.npz
```

If this command prints the file name, the path is correct. Large input and
generated data remain untracked under `data/`.

The small `data/curation_study_groups.csv` and
`data/curation_study_overrides.csv` files are the exceptions: both are tracked.
The first freezes the RCSB primary-citation DOI snapshot used by this release;
the second records the manual grouping of missing-DOI deposition series.
Grouped analyses fail rather than treating a new missing DOI as an independent
study. The current 70-entry curation resolves to 38 study groups.

## Run the main analysis

Always run commands from the repository root.

First, reproduce the main PCA and ANM measurements from the prepared inputs:

```bash
python scripts/reproduce_modes.py --verify
```

A successful run reports approximately 88.3% for PC1, 0.744 for the ANM
mode-1 directional overlap, and 0.641 for the ten-mode subspace comparison.
Directional overlap is an absolute normalized dot product; it is not an
R-squared value.

Run the main sensitivity checks:

```bash
python scripts/pairwise_sensitivity.py --verify
python scripts/study_group_sensitivity.py --verify
python scripts/pca_robustness.py --verify
python scripts/anm_null_significance.py --verify
```

## Rebuild from Protein Data Bank coordinates

The following checks connect to the RCSB Protein Data Bank. They download raw
mmCIF coordinates and can take longer on the first run:

```bash
python scripts/reproduce_tensor.py --verify
python scripts/reproduce_ensemble.py --verify
python scripts/sensor_loop_sensitivity.py --verify
python scripts/assembly_rigid_null.py --verify
```

For each matched rigid-motion subspace, the directional-null tail probability,
mean, standard deviation, and 95th percentile are evaluated exactly from
`|cos(theta)|^2 ~ Beta(1/2, (d-1)/2)`. They are not Monte Carlo estimates and
do not depend on a random seed or number of draws. In the JSON output,
`p_exact` is canonical; `p_empirical` remains only as a deprecated compatibility
alias with the same value. The top-level directional-null `n_draws` is `0` and
its `seed` is `null`. The separate junction-continuity diagnostic remains a
2,000-draw calculation and records its own sampling method and seed.

Downloaded coordinate files are stored under `data/_cif_cache/` during normal
generation runs. Adding `--write-assembly` to the assembly command creates
`render/open_8cvp_assembly.pdb` from the RCSB record before running its check.
Verification mode avoids changing tracked reference files.

## Create tables and plots

After the input bundle is in place, create the required Fig. 4 pocket panel.
This panel is a mandatory input to `build_fig4.py`, not an optional decoration:

```bash
conda install -c conda-forge pymol-open-source
pymol -cq scripts/render_fig4_pocket.py
```

The renderer requires `render/closed_5fqd_lig.pdb` from the same input bundle
and writes `figures/panels/render_closed_pocket.png`. If it is missing,
`build_fig4.py` stops with the generation command instead of producing an
incomplete figure.

Then rebuild the tables and two-dimensional plots with:

```bash
python scripts/build_tables.py
python scripts/build_fig1.py
python scripts/build_fig2.py
python scripts/build_fig3.py
python scripts/build_fig4.py
python scripts/build_fig5_robustness.py
python scripts/build_figS1.py
python scripts/build_figS2.py
python scripts/build_figS3.py
```

Generated files are written below `figures/` and `study/`. Both directories are
excluded from Git.

The Fig. 2 three-dimensional panels are optional placeholders in the composite
builder. To render them, use PyMOL and the prepared files in the top-level
`render/` directory:

```bash
pymol -cq scripts/render_fig2_3d.py
```

## Troubleshooting

- `No such file or directory: data/...` means the input bundle is missing or
  nested at the wrong level.
- `ModuleNotFoundError` usually means that `crbn-soft` is not active. Run
  `conda activate crbn-soft` and try again.
- A network error during a coordinate rebuild usually means that RCSB is
  temporarily unavailable. The local PCA and ANM checks do not need network
  access once the input bundle is present.
- Run every command from the directory that contains this README.

## Citation and license

Software citation metadata are provided in `CITATION.cff`. The code is released
under the MIT License.
