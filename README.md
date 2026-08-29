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

The layout should begin like this:

```text
CRBN_soft/
├── data/
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

If this command prints the file name, the path is correct. Input and generated
data remain untracked because `data/` is excluded from Git.

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
python scripts/anm_null_significance.py --verify
```

## Rebuild from Protein Data Bank coordinates

The following checks connect to the RCSB Protein Data Bank. They download raw
mmCIF coordinates and can take longer on the first run:

```bash
python scripts/reproduce_tensor.py --verify
python scripts/reproduce_ensemble.py --verify
python scripts/sensor_loop_sensitivity.py --verify
python scripts/assembly_rigid_null.py --write-assembly --verify
```

Downloaded coordinate files are stored under `data/_cif_cache/` during normal
generation runs. The assembly command creates `render/open_8cvp_assembly.pdb`
from the RCSB record before running its check. Verification mode avoids changing
tracked reference files.

## Create tables and plots

After the input bundle is in place, rebuild the tables and two-dimensional
plots with:

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

The optional three-dimensional renderers require PyMOL and prepared files in a
top-level `render/` directory. Install PyMOL separately, then run a renderer, for
example:

```bash
conda install -c conda-forge pymol-open-source
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
