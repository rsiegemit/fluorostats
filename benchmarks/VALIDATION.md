# Baseline-validation record

Before comparing fluorostats against other tools, we confirmed that **our re-runs of
those tools reproduce their published numbers** — otherwise a "fluorostats vs X"
comparison would be meaningless (our X might be misconfigured). This file records the
published reference values and the values we observed on the same metric.

## Deep-learning nucleus segmentation — published vs. observed

Metric: **F1 at IoU ≥ 0.5** (predicted instances matched to ground-truth instances),
the standard DSB2018 / Caicedo-2019 nuclei metric. Our runs were on **BBBC039**
(U2OS Hoechst, instance GT, n = 200), with inference on the GPU/ROCm cluster
described in `README.md` (models and settings listed there under "Tables without a
producer here"). Observed values: `results/b2_nuclei_ci.csv`, subset `all200`.

| Tool | Published | Source | Observed (BBBC039) | In published band? |
|---|---|---|---|---|
| StarDist 2D (`2D_versatile_fluo`) | AP@0.5 = 0.864 | Schmidt et al. 2018 (arXiv:1806.03535) | **F1@0.5 = 0.871** | ✅ yes |
| Cellpose (`nuclei`) | AP@0.5 ≈ 0.8 | Stringer et al. 2021 Nat. Methods | **F1@0.5 = 0.862** | ✅ yes (0.8–0.9) |
| CellProfiler | F1@0.5 ≈ 0.82 | Caicedo et al. 2019 Nat. Methods | — (not re-run) | — |

The Cellpose package version used on the cluster is not recorded in this repository.
Both DL baselines reproduce their published instance-F1, so the fluorostats comparison
below is trustworthy.

## fluorostats vs. the validated baselines

fluorostats is not a trained instance segmenter: its objects are the connected
components of a thresholded mask, optionally split by the distance-transform
watershed (`objects.watershed_split`). The benchmark scripts score those objects as
instances, against the same ground truth and matcher as the DL tools:

- **instance F1 @ IoU 0.5** — `results/b_dl_ci.csv` (all 200 images, paired bootstrap
  differences) and `results/b2_nuclei_ci.csv` (every threshold configuration, first
  100 images);
- **nucleus count** — `results/b_nuclei_count_agreement.csv`
  (`producers/b_nuclei_count_agreement.py`): Lin's CCC, Spearman ρ, count MAE and mean
  bias against the ground-truth count, for fluorostats and the three DL tools.

fluorostats never reports instance AP or CTC SEG/DET/TRA.

## Exact-match / topology validation

- **Topology & volume fraction on analytic phantoms** — Euler characteristic, connectivity,
  and volume fraction match the closed-form ground truth to zero error (reproduces BoneJ's own
  validation approach). See `results/b1_topology_phantoms.csv`.
- **Skeleton metrics** — branch and junction counts match the known counts of synthetic
  bifurcating trees (`results/b_skeleton_tree.csv`). No side-by-side comparison with Fiji
  AnalyzeSkeleton is recorded here.

## Viability against synthetic ground truth

On the Kerkhoff synthetic Live/Dead set (exact ground-truth viability),
`live_dead_by_count(method="maxima")` and the other viability modes are scored against
ground truth (Lin's CCC, MAE: `results/b_viability_external_agreement.csv`). The
`skimage_peakcount` column is a direct `skimage.feature.peak_local_max` call with the
library's parameters, so it equals the maxima mode by construction. The Fiji Live/Dead
macro published with the set (Find Maxima, default prominence 20, after a mean filter) was not
run, so no agreement with it is claimed.

## Reproducing these numbers

The DL per-image outputs are shipped in `results/` (so the tables/figures regenerate without
a GPU). The published references are cited above. Dataset sources are in `DATA_MANIFEST.md`.
