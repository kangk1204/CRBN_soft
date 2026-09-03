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

That same distributed bundle also contains a self-contained snapshot for
validating figure provenance. Extract it into a separate empty directory
and follow its `README_DATA_BUNDLE.md`; its exact figure builders, validator,
manifests and referenced files should remain together. Copy only the analysis
inputs needed by this checkout, rather than replacing this checkout's scripts
with the provenance snapshot.

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
│   ├── hinge_geometry.json
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

For the primary PCA/ANM verifier only, the matching inputs can instead be read
directly from an external location without copying them into this checkout:

```bash
python scripts/reproduce_modes.py --verify --data-source /absolute/path/to/data
python scripts/reproduce_modes.py --verify --data-source /absolute/path/to/bundle.zip
```

The directory form must contain `crbn_ensemble.ens.npz`,
`crbn_residue_window.csv`, and `crbn_anm_modes.npz` directly. The ZIP form must
contain each of those files once under `data/`. ZIP contents are read in memory
and are never extracted. Verification modifies neither the checkout nor the
selected source. The other workflows below still require the local `data/` and,
where noted, `render/` layout.

The small `data/curation_study_groups.csv` and
`data/curation_study_overrides.csv` files are the exceptions: both are tracked.
The first freezes the RCSB primary-citation DOI snapshot used by this release;
the second records publisher-verified DOI assignments for entries whose RCSB
records lack a primary DOI. It also provides an explicit series identifier when
no external DOI can be verified. Grouped analyses fail rather than treating an
unresolved entry as an independent study. The current 70-entry curation resolves
to 38 study groups.

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

Figure generation uses SciencePlots with deterministic local overrides. The
environment file includes the tested SciencePlots version. Data-bearing panels
are generated from the supplied arrays and tables; prepared molecular renders
are composed without pixel modification.

The released external bundle contains three frozen reference renders below
`figures/panels/`: two for Fig. 2 and one for Fig. 4. Their SHA256 values are
checked before composition so a changed raster cannot silently produce a
different figure. To rebuild the Fig. 4 pocket render from coordinates, run:

```bash
conda install -c conda-forge pymol-open-source
pymol -cq scripts/render_fig4_pocket.py
```

The renderer requires `render/closed_5fqd_lig.pdb` from the same input bundle
and writes `figures/panels/render_closed_pocket.png`. Re-rendering can change
pixels across PyMOL versions. Validate any regenerated panel scientifically and
visually; a new accepted raster requires an intentional hash update in a new
software release.

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
python scripts/export_figure_source_data.py
```

Each main builder writes matching PNG, PDF, and SVG outputs from one code path.
The exporter writes exact per-panel CSV records below `figures/source_data/`.
Generated files are written below `figures/` and `study/`; both directories are
excluded from Git.

Fig. 3 reads the endpoint-derived screw-axis record and shades only residues
316–320 near the HB–TBD boundary. Fig. 4 keeps the three UniProt ligand
annotations separate from the seven 5FQD S-lenalidomide (LVY) heavy-atom contacts in the
common residue window; neither definition is silently substituted for the
other.

To regenerate the Fig. 2 three-dimensional panels from the prepared files in
the top-level `render/` directory, use PyMOL:

```bash
pymol -cq scripts/render_fig2_3d.py
```

As with Fig. 4, regenerated Fig. 2 rasters require scientific and visual
validation and an intentional frozen-hash update in a new software release.

## Troubleshooting

- `missing required input(s)` lists every file that is absent or nested at the
  wrong level in the selected directory or ZIP.
- `data source is not a readable ZIP` means that the selected non-directory
  path is corrupt or is not a ZIP file.
- `duplicate required ZIP member` means that the archive is ambiguous and must
  be rebuilt with one exact `data/...` member per required file.
- `ModuleNotFoundError` usually means that `crbn-soft` is not active. Run
  `conda activate crbn-soft` and try again.
- A network error during a coordinate rebuild usually means that RCSB is
  temporarily unavailable. The local PCA and ANM checks do not need network
  access once the input bundle is present.
- Run every command from the directory that contains this README.

## Citation and license

Software citation metadata are provided in `CITATION.cff`. The code is released
under the MIT License.
