# CRBN Soft

CRBN Soft contains Python workflows for analysing human cereblon (CRBN) structures from the Protein Data Bank. The code rebuilds curated coordinate matrices, performs principal component and elastic-network analyses, runs sensitivity checks, and creates reproducible tables and plots from prepared source data.

This repository contains software only. Large coordinate caches, derived arrays, rendered images and article files are not stored here.

## What You Need

- Linux or macOS terminal
- Conda or Mamba
- Git
- Basic familiarity with running Python commands

## Install

Clone the repository:

```bash
git clone https://github.com/kangk1204/CRBN_soft.git
cd CRBN_soft
```

Create the analysis environment:

```bash
conda env create -f environment.yml
conda activate crbn
```

If you use Mamba, the same step is usually faster:

```bash
mamba env create -f environment.yml
mamba activate crbn
```

## Data Layout

Most scripts expect a `data/` directory at the repository root. The software release does not include large data files. Place the prepared source-data package in this layout before running the full workflow:

```text
CRBN_soft/
  data/
    crbn_ensemble.ens.npz
    crbn_residue_window.csv
    crbn_curation_log.csv
    ...
  render/
    open_8cvp.pdb
    closed_5fqd.pdb
    ...
```

Raw structures can also be downloaded from the RCSB Protein Data Bank by the reproduction scripts when network access is available.

## Common Commands

Run the main structure workflow:

```bash
python scripts/reproduce_ensemble.py --verify
python scripts/reproduce_tensor.py --verify
python scripts/reproduce_modes.py --verify
```

Run sensitivity analyses:

```bash
python scripts/pairwise_sensitivity.py --verify
python scripts/study_group_sensitivity.py --verify
python scripts/assembly_rigid_null.py --verify
python scripts/sensor_loop_sensitivity.py --verify
```

Build tables and plots after the required data files are present:

```bash
python scripts/build_tables.py
python scripts/build_fig1.py
python scripts/build_fig2.py
python scripts/build_fig3.py
python scripts/build_fig4.py
python scripts/build_fig5_robustness.py
```

Some structural renderers require PyMOL:

```bash
pymol -cq scripts/render_fig2_3d.py
pymol -cq scripts/render_fig4_pocket.py
```

## Checks

Run the lightweight repository checks:

```bash
python -m pytest -q
python -m ruff check .
```

The full numerical checks require the prepared `data/` and `render/` directories.

## License

The code is released under the MIT License.
