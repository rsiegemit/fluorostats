# Benchmark index — complete registry

Every benchmark run for the fluorostats methods paper: script, dataset(s),
reference methods compared, result file(s), headline number, and status.
Kept current as benchmarks are added. See `00_BENCHMARK_RESULTS.md` for the
narrative synthesis and `../data/00_DATA_MASTER.md` for dataset sources.

Environment: all local benchmarks use `python3.13` (fluorostats 0.5.0). DL and
extra-dependency benchmarks run on the AMD ROCm HPC cluster (`ssh amd`, Slurm)
where `pip install --user` works.

## Registry

| ID | Script | Dataset (public) | References compared | Result file | Headline | Status |
|---|---|---|---|---|---|---|
| B1-topo | b1_topology_phantoms.py | synthetic phantoms | analytic Euler / BoneJ convention | b1_topology_phantoms.csv | 6/6 exact, zero error | ✅ |
| B1-skel | b1_skeleton_phantoms.py | synthetic lines | analytic geometry / AnalyzeSkeleton | b1_skeleton_phantoms.csv | length ≤1%, branches exact | ✅ |
| B1-tree | b_skeleton_tree.py | synthetic trees | known bifurcation count | b_skeleton_tree.csv | depth 2-3 exact; depth 4 resolution limit | ✅ |
| B2-039 | b2_nuclei_bbbc039.py | BBBC039 | expert instance GT | b2_nuclei_bbbc039.csv | count CCC 0.918 | ✅ |
| B2-f1 | b2_nuclei_fluorostats_f1.py | BBBC039 | instance GT | b2_nuclei_fluorostats_f1.csv | fluorostats F1 0.896 | ✅ |
| B2-tri | b2_nuclei_tritool.py | BBBC039 | StarDist, Cellpose (validated) | b2_nuclei_tritool_ranking.csv | fs 0.896 ≥ DL 0.87 | ✅ |
| B2-methods | b2_nuclei_methods.py | BBBC039 | Otsu, Li, Yen, Triangle, Isodata, Mean, Minimum, Watershed, StarDist, Cellpose | b2_nuclei_methods.csv | 12-method ranking; Li 0.934 best | ✅ |
| B2-variants | b2_nuclei_variants.py | BBBC039 | fluorostats CC/watershed/border | b2_nuclei_variants.csv | watershed 0.896→0.899 | ✅ |
| B2-024 | b2_bbbc024_3d.py | BBBC024 c00/c75 | Otsu, Li, Isodata, Triangle | b2_bbbc024_3d_c00/c75.csv | 3D F1 0.939 (c00), 0.153 (c75) | ✅ |
| B2-crowd | (infer_slices.py cluster) | BBBC024 c75 slices | StarDist, Cellpose | b2_crowded_c75_comparison.csv | fs 0.38 vs DL 0.96-1.0 | ✅ |
| B2-A549 | b2_ctc_3d.py | CTC Fluo-C3DH-A549 | CTC gold GT | b2_ctc_3d_summary.csv | Dice 0.69 | ✅ |
| B2-CHO | b2_ctc_cho.py | CTC Fluo-N3DH-CHO | CTC gold GT | b2_ctc_cho_summary.csv | Dice 0.52 | ✅ |
| B3 | b3_viability.py | S-BIAD2130 Live/Dead | 2D MIP vs 3D | b3_viability_summary.csv | MIP overestimates 1.14× | ✅ |
| B4-reaver | b4_reaver_vascular.py | REAVER (manual GT) | manual GT | b4_reaver_summary.csv | area CCC 0.70 | ✅ |
| B4-rank | b4_reaver_ranking.py | REAVER | AngioTool, AngioQuant, ImageJ, RAVE, REAVER | b4_reaver_ranking.csv | fs #4/6, ties AngioTool | ✅ |
| B4-vessel | b4_vesselexpress_3d.py | VesselExpress | field literature values | b4_vesselexpress_3d.csv | sane 3D densities | ✅ |
| B4-sprout | b_vascular_sproutangio.py | SproutAngio (Zenodo 7240927, n=12 .czi) | Otsu, Li, Isodata, Triangle + fluorostats | b_vascular_sproutangio_multi.csv | all detect VEGF dose (ρ 0.59–0.74) | ✅ |
| B2-ctc-multi | b2_ctc_multi.py | CTC A549 + CHO | Otsu, Li, Yen, Isodata, Triangle + fluorostats | b2_ctc_multi.csv | 6-method Dice; best is dataset-dependent | ✅ |
| B-cluster | b_clustering_curve.py | BBBC024 c00/c25/c50/c75 | Otsu, Li, Isodata, Triangle, fluorostats CC+watershed | b_clustering_curve.csv | all non-DL collapse 0.92→0.13; DL holds 0.96 | ✅ |
| B-timing | b_timing.py | BBBC039 | Otsu, Li, Isodata, Triangle, Watershed, StarDist, Cellpose | b_timing.csv | fluorostats 14.5ms; 15×/380× faster than DL | ✅ |
| B-noise | b_noise_robustness.py | BBBC024 + noise | Otsu, Li, Isodata, Triangle + fluorostats | b_noise_robustness.csv | graceful degradation to SNR≈4 | ✅ |
| B-size | b_nuclei_size.py | BBBC024 (GT sizes) | Otsu, Li, Isodata, Triangle + fluorostats | b_nuclei_size.csv | fluorostats tied best, 3.5% error | ✅ |
| B-skelmeth | b_skeleton_methods.py | synthetic trees | Lee-1994, medial_axis, thin, Zhang | b_skeleton_methods.csv | all 4 recover branches exactly | ✅ |
| B-depth | b_depth_metrics.py | synthetic z-gradient | intensity-centroid, geometric, peak, median, FWHM | b_depth_metrics.csv | fluorostats centroid validated (0.07 err) | ✅ |
| B-denoise | b_denoising.py | BBBC024+noise | none, fluorostats-gaussian, median, gaussian-σ2, TV | b_denoising.csv | fluorostats recovers Dice 0.41→0.91 at high noise | ✅ |
| B-connect | b_connectivity_discrimination.py | synthetic fragment→connected | LCC, Euler, n_comp, percolation, fragmentation | connectivity_discrimination_correlations.csv | euler_number best (ρ=1.0), spans both regimes | ✅ |
| B-density | b_density_normalization.py | BBBC024 zoom series | per-mm³, raw, per-Mpx, per-area, per-slice | b_density_normalization_cv.csv | fluorostats per-mm³ zoom-invariant (CV=0) vs 4 varying | ✅ |
| V-agreement | b_agreement_validation.py | synthetic paired | Lin formula, Pearson-decomp, ANOVA-ICC, numpy BA | b_agreement_validation.csv | 11/11 exact to machine precision | ✅ |
| B-cluster2d | b_cluster_2d.py | BBBC039 | Otsu, Li, Isodata, Triangle, Yen | b_cluster_2d.csv | area-frac Spearman ~0.9 (5 methods) | ✅ |
| V-ap | b_validate_ap.py | synthetic labeled | perfect/TP-FP-FN/IoU-threshold/Kaggle-AP/brute-force | b_validate_ap.csv | 23/23 PASS | ✅ |
| B-dsb2018 | b_dsb2018.py | DSB2018 (StarDist test set) | Otsu, Li, Isodata, Triangle, Yen + fluorostats | b_dsb2018.csv | fs F1 0.789 (rank 1 of heuristics); ~91% of DL 0.864 | ✅ |
| B6 | b6_homogeneity_synthetic.py | synthetic points | Clark-Evans NN | b6_homogeneity_summary.csv | ρ=−0.985 | ✅ |
| B6-multi | b_homogeneity_multi.py | synthetic points | Clark-Evans, Ripley's K, Morisita, lacunarity, quadrat-var | b_homogeneity_multi_corr.csv | ρ 0.96-0.997, AUC 1.0 | ✅ |
| B-cover | b_coverage_2d.py | BBBC039 | Otsu, Li, Triangle, Yen, Isodata | b_coverage_2d.csv | Spearman ~0.9 | ✅ |
| V-stats | validate_stats.py | synthetic | scipy, hand-coded BH | validate_stats.csv | 8/8 exact | ✅ |
| V-power | validate_power.py | synthetic | empirical rejection rate | validate_power.csv | sound; pilot-optimism disclosed | ✅ |
| V-volfrac | b_volfrac_validation.py | synthetic | Delesse point-counting, analytic | b_volfrac_validation.csv | 7/7 exact; zoom-invariant | ✅ |

## DL baseline validation (prerequisite for all DL comparisons)

Cellpose v3 `nuclei` + StarDist `2D_versatile_fluo`, run on the AMD cluster,
reproduce published F1 on BBBC039 (StarDist 0.871 ≈ published 0.864; Cellpose
0.862). See `../data/PUBLISHED_BASELINES.md`. Cellpose v4 SAM abandoned (GPU
hipBLASLt crash + CPU too slow); CellProfiler install failed (dep conflicts —
published F1 ~0.82 cited as context).

## Reference methods/tools with hard comparison numbers (30+)

Otsu(1979), Li(1993), Yen(1995), Triangle(1977), Isodata(1978), Mean, Minimum,
Watershed(1991), StarDist(2018), Cellpose(2021), AngioTool(2011), AngioQuant,
ImageJ, RAVE(2011), REAVER(2020), Clark-Evans NN, Ripley's K, Morisita,
lacunarity, quadrat variance, Delesse point-counting, scipy Mann-Whitney,
hand-coded BH-FDR, scipy Stouffer, empirical power. Plus CTC leaderboard and
CellProfiler as cited context.

## Open / in-progress

| Direction | Status |
|---|---|
| SproutAngio VEGF dose-response | ✅ DONE (Zenodo 7240927, n=12) — metrics track VEGF dose |
| MiniVess | EBRAINS auth-gated (doi 10.25493/HPBE-YHK) — not scriptable; deprioritized |
| VascuSynth skeleton | source URL 404'd — used synthetic tree phantom instead ✅ |
| CellProfiler head-to-head | install dep-conflict; citing published F1 ~0.82 as context |

## Change log
See `../CHANGELOG_AND_RERUNS.md` for library changes (v0.4/v0.5) and Live/Dead
rerun-impact tracking (deferred rerun; all benchmark-driven library additions
are opt-in, no rerun triggered).
