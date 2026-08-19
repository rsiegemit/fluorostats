# Benchmarks

The benchmark harness behind the FluoroStats methods paper: the scripts that
regenerate the paper's figures and tables, the precomputed metric tables they
plot, the dataset manifest, and the baseline-validation record. Each script also
doubles as a worked example of the library's analysis modules.

- **[`DATA_MANIFEST.md`](DATA_MANIFEST.md)** — every dataset used, with licence and source URL.
- **[`VALIDATION.md`](VALIDATION.md)** — published-vs-observed baselines (StarDist, Cellpose, phantoms).
- `figstyle.py` — shared publication styling (Okabe–Ito palette, panel labels, scale bars).
- `results/` — small precomputed metric tables the figures plot (+ `ve_crops.npz` image crops).
- `previews/` — expected output for the offline figures.

## Figures that reproduce offline (from bundled `results/` — no download)

| script | figure | library modules exercised |
|---|---|---|
| `fig3_vascular.py` | vessel-network accuracy (REAVER 6-tool, VesselExpress, 3D phantom, Bland–Altman) | `metrics_3d`, `skeleton`, `agreement` |
| `fig5_homogeneity_stats.py` | spatial homogeneity + end-to-end statistics | `objects.centroid_homogeneity`, `stats`, `power.power_curve` |
| `make_ed1_correctness.py` | topology / skeleton / density-normalisation correctness | `objects`, `skeleton` |
| `fig_ed3_robustness.py` | segmentation robustness to noise + denoising | `segment`, `preprocess` |
| `fig_ed4_generalization.py` | cross-dataset generalisation (DSB2018, CTC 3D) | `segment`, `validate` |

```bash
pip install -r requirements.lock      # from the repo root (pinned env)
pip install -e ".[all]"
cd benchmarks
python fig5_homogeneity_stats.py      # writes figures/ (git-ignored)
```

## Figures that require a dataset download (scripts provided)

These read raw microscopy rather than a bundled table. Fetch the dataset from
`DATA_MANIFEST.md` into `$FLUOROSTATS_DATA` (default `./data/downloads`), then run:

| script | figure | dataset |
|---|---|---|
| `fig2_nuclei.py` | 2D nuclei: fluorostats vs threshold panel + DL baselines | BBBC039 |
| `fig4_viability.py` | depth-resolved Live/Dead viability | S-BIAD2130 |
| `fig_ed2_runtime.py` | runtime + determinism | BBBC024 |

The deep-learning baselines (StarDist, Cellpose) were evaluated on a GPU/ROCm
cluster; their per-image outputs are shipped in `results/`, so the comparison
figures regenerate without re-running the networks (see `VALIDATION.md`).

## Not scriptable here

The combined applications panel (Fig 6) is built from the authors' own confocal
constructs (GelMA/hybrid `.oib` stacks), which are not publicly deposited; the
closest public analogues are listed in `DATA_MANIFEST.md`. The analysis runs
through the same `fluorostats` modules as everything above.
