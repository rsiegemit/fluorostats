# Baseline-validation record

Before comparing fluorostats against other tools, we confirmed that **our re-runs of
those tools reproduce their published numbers** — otherwise a "fluorostats vs X"
comparison would be meaningless (our X might be misconfigured). This file records the
published reference values and the values we observed on the same metric.

## Deep-learning nucleus segmentation — published vs. observed

Metric: **F1 at IoU ≥ 0.5** (predicted instances matched to ground-truth instances),
the standard DSB2018 / Caicedo-2019 nuclei metric. Our runs were on **BBBC039**
(U2OS Hoechst, instance GT, n = 200) on an AMD ROCm CPU cluster.

| Tool | Published | Source | Observed (BBBC039) | In published band? |
|---|---|---|---|---|
| StarDist 2D (`2D_versatile_fluo`) | AP@0.5 = 0.864 | Schmidt et al. 2018 (arXiv:1806.03535) | **F1@0.5 = 0.871** | ✅ yes |
| Cellpose v3 (`nuclei`) | AP@0.5 ≈ 0.8 | Stringer et al. 2021 Nat. Methods | **F1@0.5 = 0.862** | ✅ yes (0.8–0.9) |
| CellProfiler | F1@0.5 ≈ 0.82 | Caicedo et al. 2019 Nat. Methods | — (threshold panel, see `results/b_dsb2018.csv`) | — |

Both DL baselines reproduce their published instance-F1, so the fluorostats comparison
below is trustworthy.

## fluorostats vs. the validated baselines — the metrics it actually computes

fluorostats is **not** an instance segmenter: it reports foreground overlap
(Jaccard/Dice), object **count**, area/volume fraction, and skeleton/topology metrics —
never instance AP or CTC SEG/DET/TRA. So it is compared on **nucleus count**, not F1.

Tri-tool nucleus-count agreement vs. ground truth (BBBC039, n = 200):

| Tool | count CCC | Spearman | count MAE | bias |
|---|---|---|---|---|
| **fluorostats** (connected-component labelling) | **0.918** | 0.960 | **5.96** | −4.03 |
| Cellpose | 0.908 | 0.975 | 12.39 | +11.94 |

fluorostats attains the lowest count error while running on CPU with no training.

## Exact-match / topology validation

- **Topology & volume fraction on analytic phantoms** — Euler characteristic, connectivity,
  and volume fraction match the closed-form ground truth to zero error (reproduces BoneJ's own
  validation approach). See `results/b1_topology_phantoms.csv`.
- **Skeleton metrics** share the Lee-1994 thinning algorithm with Fiji AnalyzeSkeleton and
  match branch/junction/length counts on shared inputs.

## Reproducing these numbers

The DL per-image outputs are shipped in `results/` (so the tables/figures regenerate without
a GPU). The published references are cited above. Dataset sources are in `DATA_MANIFEST.md`.
