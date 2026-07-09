# fluorostats methods paper — handoff for Claude Cowork

Everything needed to draft the methods paper. This document is self-contained:
numbers, figures, structure, honesty ledger, and reproducibility pointers. Where a
claim needs the raw table, it points to a CSV in `benchmarks/results/` and a figure
in `benchmarks/figures/handoff/`.

---

## 0. How to use this handoff
- **Write from Sections 3–7 below.** They map one-to-one onto paper sections.
- Every number here is traceable: `benchmarks/BENCHMARK_INDEX.md` (registry) →
  `benchmarks/results/*.csv` (raw) → `benchmarks/00_BENCHMARK_RESULTS.md` (narrative).
- Comparison table: `COMPARISON_MATRIX.md`. Literature: `research/00_SYNTHESIS.md`
  (+ 13 dossiers, ~145 refs). Datasets: `data/00_DATA_MASTER.md`. DL validation:
  `data/PUBLISHED_BASELINES.md`.
- **Framing rule (important):** thresholding algorithms (Otsu, Li, …) are
  fluorostats' own *configs*, NOT rival software. Comparisons are against distinct
  **software**: StarDist, Cellpose, Omnipose, REAVER, AngioTool, VesselExpress,
  the Kerkhoff Fiji macro, CellProfiler; and against reference *implementations*
  (scipy, skan, hand-coded) for correctness validation.

---

## 1. What fluorostats is
An open-source Python library + CLI that turns fluorescence microscopy (confocal
z-stacks, widefield, any major format) into publication-ready quantities: 3D
volume fraction, connectivity/topology, skeleton/vascular metrics, object
morphometry, spatial homogeneity, Live/Dead viability, and the statistics + figures
to report them. Training-free, CPU-only, 19 modules, v0.7.0, 105 tests.

## 2. Thesis / positioning
fluorostats gives **comparable-or-better accuracy than established tools across a
broad range of fluorescence quantification tasks, training-free and on CPU**, with
every metric validated to reference-implementation exactness — while being honest
about the one regime it cedes to deep learning (heavily overlapping instances).
It is the reproducible, general, auditable quantifier for the 80% of analyses that
do not need a trained instance segmenter.

## 3. Suggested paper structure
1. **Introduction / gap** — fragmented tooling (ImageJ macros, per-domain tools,
   DL requiring GPUs/training); need for one general, validated, training-free
   quantifier. Source: `research/00_SYNTHESIS.md`.
2. **Design** — modular architecture (io→preprocess→segment→metrics→stats→report),
   19 modules; every metric a pure function; opt-in additions.
3. **Validation (correctness)** — Section 6 here: exact vs reference implementations.
4. **Benchmarks (comparison)** — Sections 4–5 here: vs DL and domain software on
   public data.
5. **Scope & limitations** — Section 7 (honesty ledger): the DL crossover, threshold
   choice, consensus caveat.
6. **Performance** — Section 5.4 (runtime).
7. **Conclusion / availability** — GitHub, CC data, reproducible scripts.

## 4. Headline comparison results (by claim)

### 4.1 Nuclei segmentation — fluorostats ≥ validated DL on well-separated targets
- BBBC039 (n=200): fluorostats F1 **0.896 [0.873, 0.916]** vs StarDist 0.871,
  Cellpose 0.862, Omnipose 0.802. Paired differences **statistically significant**
  (fs−StarDist +0.025 [0.004,0.042]; fs−Cellpose +0.034 [0.008,0.057]).
  → `b_dl_ci.csv`, `omnipose_eval.csv`; **fig2_dl_ci.png**, **fig12_vs_dl_all.png**.
- 12-method comparison (fluorostats' own threshold family + DL): Li config tops at
  0.934. → `b2_nuclei_methods.csv`; **fig1_nuclei_ranking.png**.
- DSB2018 (StarDist's own data): fluorostats **0.789 ≈ 91% of DL's published 0.864**,
  training-free. → `b_dsb2018.csv`.
- **DL baselines validated first**: StarDist reproduced 0.871 ≈ published 0.864;
  Cellpose 0.862. → `data/PUBLISHED_BASELINES.md`. (Credibility linchpin — state this.)

### 4.2 Scope boundary — DL wins the crowded regime (state honestly)
- BBBC024 clustering curve: every non-DL method (incl. fluorostats) collapses
  0.92→0.13 (c00→c75); DL holds ~0.96. → `b_clustering_curve.csv`;
  **fig3_clustering_curve.png**, **fig7_scope_boundary.png**.

### 4.3 Vascular (2D) — ties specialists on their own benchmark
- REAVER dataset (n=36), 6-tool ranking: fluorostats #4, ties AngioTool, beats RAVE
  + AngioQuant. → `b4_reaver_ranking.csv`; **fig8_vascular_ranking.png**.
- SproutAngio VEGF dose-response (real .czi): all methods detect the response
  (ρ 0.59–0.74). → `b_vascular_sproutangio_multi.csv`.

### 4.4 Vascular (3D)
- **Synthetic phantom (exact GT):** length ≤2.4%, branches + VF exact.
  → `b_vascular_phantom_3d.csv`; **fig10_vascular_phantom.png**.
- **VesselExpress (real light-sheet, Zenodo 6025935):** fluorostats(li)=(auto→li)
  Dice **0.598** vs VesselExpress software's segmentation; Otsu default 0.089.
  → `b_vesselexpress.csv`; **fig11_vesselexpress_seg.png**.
- **fluorostats vs VesselExpress metric agreement (vessel VF, n=9):** the two
  software **rank volumes consistently (Spearman 0.75)** but fluorostats(auto→li)
  reports ~1.7× higher absolute VF (VE 0.029 vs fs 0.049; CCC 0.11) — a systematic
  offset (fluorostats(li) more inclusive than VE's pipeline), like tool-to-tool
  differences in REAVER. Skeleton-length comparison omitted: skeletonizing full
  250 MB dense light-sheet volumes is intractable (honest scope note).
  → `b_ve_metrics.csv`; **fig13_vesselexpress_metric.png**.

### 4.5 Viability (Live/Dead) — ties the published Fiji tool exactly
- Kerkhoff Fiji macro (Zenodo 10395753, synthetic GT): fluorostats
  `live_dead_by_count(maxima)` MAE **0.016**, CCC **0.987** = the macro exactly.
  → `b_viability_external.csv`; **fig5_viability_external.png**.
- Depth-aware: 2D shortcuts bias live fraction +5% (MIP) to +25% (mean-of-slice)
  vs true 3D (S-BIAD2130). → `b3_viability_multi.csv`.
- **Discovered + built:** external comparison exposed the missing peak-counting
  mode; we added it and now tie the tool. Honest: maxima is NOT universal
  (over-counts flat/noisy cells) → `b_maxima_regimes.csv`.

### 4.6 Spatial homogeneity — tracks all 5 established statistics
- fluorostats Gini vs Clark-Evans / Ripley's K / Morisita / quadrat variance /
  lacunarity: |ρ| 0.96–0.997, AUC 1.0. → `b_homogeneity_multi_corr.csv`;
  **fig9_homogeneity.png**.

## 5. Validation results (correctness — exact vs reference)
| Capability | Result | CSV |
|---|---|---|
| Statistics (7 fns incl. stratified MWU) | exact vs scipy + hand-coded | `validate_stats.csv`, `b_stratified_stats.csv` |
| Agreement (BA/CCC/ICC) | 11/11 machine-precision | `b_agreement_validation.csv` |
| Instance metrics (F1/AP) | 23/23 | `b_validate_ap.csv` |
| Volume fraction / density | 7/7; zoom-invariant CV=0 | `b_volfrac_validation.csv`, `b_density_normalization_cv.csv` |
| Connectivity (Euler best tracker ρ=1.0) | 6/6 exact | `connectivity_discrimination_correlations.csv` |
| Skeleton (length ≤1%, branches exact) | phantoms + 4 algorithms | `b1_skeleton_phantoms.csv`, `b_skeleton_methods.csv` |
| Nucleus size / depth / prune / bg-subtract | 3.5% / 0.07 / best-of-5 / best-of-6 | `b_nuclei_size.csv`, `b_depth_metrics.csv`, `b_prune_skeleton.csv`, `b_background_subtract.csv` |

### 5.4 Runtime
fluorostats 2D segmentation ~14.5 ms/image (CPU) — **15× faster than StarDist,
380× than Cellpose**; on par with classical thresholds. Full per-metric table
(every fs metric vs every comparator): `b_timing_all_metrics.csv`;
**fig4_timing_all.png**.

## 6. Figure inventory (`benchmarks/figures/handoff/`)
1. fig1_nuclei_ranking — 12-method nuclei F1 (fluorostats highlighted)
2. fig2_dl_ci — fluorostats vs StarDist/Cellpose, bootstrap CIs (significant)
3. fig3_clustering_curve — F1 vs overlap, all methods collapse
4. fig4_timing_all — per-metric runtime, fluorostats vs comparators (log)
5. fig5_viability_external — vs Kerkhoff Fiji macro (MAE + CCC)
6. fig6_noise — noise robustness
7. fig7_scope_boundary — separated vs crowded (fs vs DL)
8. fig8_vascular_ranking — REAVER 6-tool ranking
9. fig9_homogeneity — vs 5 spatial statistics
10. fig10_vascular_phantom — 3D phantom exact-GT accuracy
11. fig11_vesselexpress_seg — real 3D vessels vs VesselExpress software
12. fig12_vs_dl_all — fluorostats vs 3 DL segmenters
13. fig13_vesselexpress_metric — fluorostats vs VesselExpress vessel-VF (rank-consistent, systematic offset)

## 7. Honesty ledger (state these plainly in Limitations)
- **Crowded/overlapping instances:** fluorostats (and all non-DL methods) collapse;
  use DL there. This is the paper's scope statement, not a weakness to hide.
- **Threshold choice matters:** Otsu default under-segments dim/sparse signal
  (light-sheet vessels: 0.089 → 0.598 with Li). `auto` mitigates via a heuristic;
  `consensus` fails when most algorithms share a failure mode. Not an oracle.
- **Counting method (viability):** maxima ties the Fiji macro on crowded cells but
  over-counts flat/noisy cells; `cc` is more noise-robust. Regime-dependent; no
  auto-selector is reliable (crowding ≈ noise statistically).
- **Hard 3D cytoplasmic (CTC A549/CHO):** untuned Dice 0.52–0.69.
- **Power:** bootstrap-from-small-pilot is optimistic (documented property).
- **VesselExpress GT is pipeline-generated**, not manual expert tracing — a
  software-agreement comparison, not a gold-standard accuracy test.
- Several benchmark *scripts* had bugs caught + fixed (Stouffer convention,
  zoom-invariance, skimage compat); competitor DL baselines were validated against
  published numbers before any comparison.

## 8. Reproducibility
- Library: `github.com/rsiegemit/fluorostats` v0.7.0 (main).
- All benchmark scripts in `benchmarks/`; run with `python3.13`.
- Datasets public + CC; download URLs in `data/00_DATA_MASTER.md` (raw data
  gitignored, ~21 GB local + VesselExpress 41 GB on cluster).
- DL baselines: AMD ROCm cluster (Cellpose v3, StarDist, Omnipose — CPU;
  scripts in `benchmarks/cluster/`).

## 9. Open decisions for the writer
- Which figures are main vs supplementary (suggest 2,3,5,8,11,12 main).
- Whether to add CIs to more comparisons (only BBBC039 has them now).
- Final Live/Dead rerun of the driving GelMA application (deferred; see
  `CHANGELOG_AND_RERUNS.md`) — decide if count-based/auto viability is adopted.
- Proof-stage citation verification (flag lists in each `research/` dossier).
