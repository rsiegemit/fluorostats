# Benchmarks

The benchmark harness behind the FluoroStats methods paper: the scripts that
regenerate the paper's figures and tables, the precomputed metric tables they
plot, the dataset manifest, and the baseline-validation record. Each script also
doubles as a worked example of the library's analysis modules.

- **[`DATA_MANIFEST.md`](DATA_MANIFEST.md)** — every dataset used, with licence and source URL.
- **[`VALIDATION.md`](VALIDATION.md)** — published-vs-observed baselines (StarDist, Cellpose, phantoms).
- `figstyle.py` — shared publication styling (Okabe–Ito palette, panel labels, scale bars).
- `results/` — small precomputed metric tables the figures plot (+ `ve_crops.npz` image crops).
- `producers/` — one script per `results/` table that regenerates it from the public data.
- `previews/` — expected output for the offline figures.

## Figures that reproduce offline (from bundled `results/` — no download)

| script | figure | tables read (`results/`) |
|---|---|---|
| `fig3_vascular.py` | vessel-network accuracy (REAVER 6-tool, VesselExpress, 3D phantom, Bland–Altman) | `b4_reaver_ranking`, `ve_crops.npz`, `b_vascular_phantom_3d`, `b_ve_metrics`, `b_ve_agreement` |
| `fig5_homogeneity_stats.py` | spatial homogeneity + end-to-end statistics | `b_homogeneity_examples`, `b_homogeneity_consensus(_corr)`, `b_homogeneity_multi_corr`, `b_vascular_sproutangio_multi`, `b_vascular_sproutangio_stats` |
| `figS1_correctness.py` | topology / skeleton / density-normalisation correctness | `b1_topology_phantoms`, `b_skeleton_tree`, `b_density_normalization(_cv)` |
| `figS3_robustness.py` | segmentation robustness to noise + denoising | `b_noise_robustness`, `b_denoising`, `b_nuclei_size` |
| `figS4_generalization.py` | cross-dataset generalisation (DSB2018, CTC 3D) | `b_dsb2018`, `b2_ctc_multi` |

These scripts import from `fluorostats` only the figure kit (`pubfig`, through
`figstyle.py`); the measurements behind the tables (`metrics_3d`, `skeleton`,
`objects`, `segment`, `preprocess`, `validate`, `stats`, `agreement`, ...) run in
the producers.

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
| `fig2_nuclei.py` | Figure 2 — nucleus segmentation & the deep-learning boundary (a–f) | BBBC039 + BBBC024 + BBBC024_c75 |
| `fig4_viability.py` | depth-resolved Live/Dead viability | S-BIAD2130 |
| `figS2_runtime.py` | runtime + determinism | BBBC024 |

For Figure 2, the summary panels (b forest, d crowding curve) reproduce from the
bundled `results/` tables; only the image panels (a/c/e) need the raw BBBC039/BBBC024
images. The per-image table the figure plots, `b2_nuclei_methods_perimage.csv`, is
written by `producers/b2_nuclei_methods_perimage.py` (see below).

The deep-learning baselines (StarDist, Cellpose, Omnipose) were evaluated on a
GPU/ROCm cluster; their per-image outputs are shipped in `results/`
(`stardist_eval.csv` etc.), so the comparison panels regenerate without re-running
the networks (see `VALIDATION.md`).

## Regenerating the results tables (`producers/`)

Every table in `results/` that the public data can regenerate has a producer in
`producers/` named after it; it reads `$FLUOROSTATS_DATA`, measures with the
`fluorostats` library, uses explicit seeds, and overwrites `results/<same name>`.
Next to each table it writes `results/<same name>.run.json`: the producer, its
parameters (including `small_object_rule` wherever a `min_size` applies), the
fluorostats git commit (`-dirty` when tracked files other than `results/` were
modified) and the python / numpy / scipy / scikit-image / fluorostats versions
(timing tables also record the CPU).

```bash
# from the repo root
FLUOROSTATS_DATA=/path/to/downloads PYTHONPATH=src python benchmarks/producers/b_dsb2018.py
```

"Verified" is the check made when the producers were added: the producer was run on
the full dataset and its output compared with the committed table (EXACT =
byte-identical; NUMERIC = every value equal within 1e-9 relative). STALE marks a
table that a re-run with the current library no longer reproduces: since
`segment.binarize` stopped removing objects of exactly `min_size` pixels, its
fluorostats values differ in the third or fourth decimal. Stale tables are not
regenerated here.

| table(s) | producer | data | verified |
|---|---|---|---|
| `b1_topology_phantoms.csv` | `b1_topology_phantoms.py` | none (synthetic) | EXACT |
| `b2_ctc_multi.csv` | `b2_ctc_multi.py` | CTC A549 + CHO | EXACT |
| `b2_nuclei_methods_perimage.csv` | `b2_nuclei_methods_perimage.py` | BBBC039 | EXACT |
| `b3_viability_depth_profile.csv` | `b3_viability_depth_profile.py` | S-BIAD2130 | EXACT |
| `b3_viability_multi.csv` | `b3_viability_multi.py` | S-BIAD2130 | EXACT when added; STALE |
| `b4_reaver_ranking_perimage.csv` | `b4_reaver_ranking_perimage.py` | REAVER | NUMERIC when added (the committed `fluorostats` column was written after a CSV round-trip, so its last digit differs); STALE |
| `b_clustering_curve.csv` | `b_clustering_curve.py` | BBBC024 (+ c25/c50/c75) | EXACT |
| `b_denoising.csv` | `b_denoising.py` | BBBC024 | EXACT |
| `b_density_normalization.csv`, `b_density_normalization_cv.csv` | `b_density_normalization.py` | BBBC024 | EXACT |
| `b_dl_ci.csv` | `b_dl_ci.py` (also reads `stardist_eval.csv`, `cellpose_eval.csv`) | BBBC039 | EXACT when added; STALE |
| `b_dsb2018.csv` | `b_dsb2018.py` | DSB2018 | EXACT when added; STALE |
| `b_homogeneity_multi.csv`, `b_homogeneity_multi_corr.csv` | `b_homogeneity_multi.py` | none (synthetic) | EXACT |
| `b_noise_robustness.csv` | `b_noise_robustness.py` | BBBC024 | EXACT |
| `b_nuclei_size.csv` | `b_nuclei_size.py` | BBBC024 | EXACT |
| `b_skeleton_tree.csv` | `b_skeleton_tree.py` | none (synthetic) | EXACT |
| `b_timing.csv` | `b_timing.py` | BBBC039 | re-run differs (wall-clock) |
| `b_timing_all_metrics.csv` | `b_timing_all_metrics.py` | BBBC024, BBBC039, Kerkhoff | re-run differs (wall-clock) |
| `b_vascular_phantom_3d.csv` | `b_vascular_phantom_3d.py` | none (synthetic) | EXACT |
| `b_vascular_sproutangio_multi.csv` | `b_vascular_sproutangio_multi.py` | SproutAngio | EXACT when added; STALE |
| `b_ve_metrics.csv` | `b_ve_metrics.py` | VesselExpress Figure 2C–E volumes | not run (data not downloaded locally; produced on the cluster) |
| `b_viability_external.csv` | `b_viability_external.py` | Kerkhoff synthetic | EXACT when added; STALE |

Derived tables summarise committed tables for the figures, so a figure script only
reads and plots (no metric is computed in a figure script). Their producers need no
raw data and are re-run offline:

| table(s) | producer | reads | verified |
|---|---|---|---|
| `b2_nuclei_ci.csv` | `b2_nuclei_ci.py` | `b2_nuclei_methods_perimage.csv`, `stardist_eval.csv`, `cellpose_eval.csv`, `omnipose_eval.csv` | EXACT |
| `b4_reaver_ranking.csv` | `b4_reaver_ranking.py` | `b4_reaver_ranking_perimage.csv` | EXACT |
| `b_homogeneity_consensus.csv`, `b_homogeneity_consensus_corr.csv` | `b_homogeneity_consensus.py` | `b_homogeneity_multi.csv` | EXACT |
| `b_homogeneity_examples.csv` | `b_homogeneity_examples.py` | none (synthetic, seed 1) | EXACT |
| `b_vascular_sproutangio_stats.csv` | `b_vascular_sproutangio_stats.py` | `b_vascular_sproutangio_multi.csv` | EXACT |
| `b_ve_agreement.csv` | `b_ve_agreement.py` | `b_ve_metrics.csv` | EXACT |
| `b_viability_external_agreement.csv` | `b_viability_external_agreement.py` | `b_viability_external.csv` | EXACT |

Timing tables are wall-clock measurements: a re-run reproduces the relative cost of
each operation, not the committed milliseconds. Their two DL rows (StarDist 215.1 ms,
Cellpose 5547.0 ms per image) are constants measured on the cluster CPU.

### Tables without a producer here

| table(s) | why | provenance |
|---|---|---|
| `stardist_eval.csv` | deep-learning inference run on the GPU/ROCm cluster | StarDist `2D_versatile_fluo` (pretrained) on all 200 BBBC039 images; per-image count and F1@IoU 0.5 against the same watershed-reconstructed GT as `producers/common.py` |
| `cellpose_eval.csv` | as above | Cellpose `nuclei` model, `diameter=None`, `channels=[0,0]`; same GT and matcher |
| `omnipose_eval.csv` | as above | Omnipose `cyto2_omni` (`omni=True`), same images, GT and matcher (columns named `thirddl_*`) |
| `ve_crops.npz` | image crops, not a measurement; no producer script exists | two 400×400 crops of VesselExpress volumes: raw plane, VesselExpress binary, fluorostats `binarize(method="auto")` mask |
| `fig6_*.csv` (6 tables) | collaborator data under a data-use arrangement; the producing scripts are excluded with the data | see the next section |

The cluster scripts that produced the DL tables are not part of this repository.

## Collaborator application figure (Fig 6)

Fig 6 is built from the originating laboratory's own confocal data — a FITC-dextran
penetration time-course and a HUVEC/GelMA Live/Dead series. The raw microscopy remains
with that laboratory and is available on request; it is not deposited. The derived
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
