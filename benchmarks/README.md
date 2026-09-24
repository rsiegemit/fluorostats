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
| `fig2_nuclei.py` | Figure 2 — nucleus segmentation & the deep-learning boundary (a–f) | BBBC039 + BBBC024 |
| `fig4_viability.py` | depth-resolved Live/Dead viability | S-BIAD2130 |
| `fig_ed2_runtime.py` | runtime + determinism | BBBC024 |

For Figure 2, the summary panels (b forest, d crowding curve) reproduce from the
bundled `results/` tables; only the image panels (a/c/e) need the raw BBBC039/BBBC024
images. `b2_nuclei_methods.py` is the per-image compute step that produces
`b2_nuclei_methods_perimage.csv` (the threshold-panel F1 table the figure plots).

The deep-learning baselines (StarDist, Cellpose, Omnipose) were evaluated on a
GPU/ROCm cluster; their per-image outputs are shipped in `results/`
(`stardist_eval.csv` etc.), so the comparison panels regenerate without re-running
the networks (see `VALIDATION.md`).

## Collaborator application figure (Fig 6)

Fig 6 is built from the authors' own confocal data — a FITC-dextran penetration
time-course and a HUVEC/GelMA Live/Dead series. The raw microscopy is available
from the authors under a data-use agreement and is not deposited, but the derived
per-stack tables are bundled here so the quantitative panels can be recomputed:

| table | content |
|---|---|
| `results/fig6_depth_profiles_long.csv` | per-stack normalized intensity vs depth |
| `results/fig6_depth_auc_per_stack.csv` | per-stack retention AUC and λ fit with range-guard flag |
| `results/fig6_depth_lambda_timecourse.csv` | per-timepoint λ summary (guarded vs naive) |
| `results/fig6_depth_group_summary.csv` | group mean ± SEM depth profiles |
| `results/fig6_viability_per_stack.csv` | per-stack Live/Dead coverage, counts, viability, regime flags |
| `results/fig6_viability_group_summary.csv` | day-1 vs day-5 group summaries |

The representative micrograph panel cannot be regenerated from these tables — it
renders the raw stacks directly. The analysis runs through the same `fluorostats`
modules as everything above.
