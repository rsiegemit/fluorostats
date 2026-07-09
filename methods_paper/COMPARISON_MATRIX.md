# fluorostats — capability × software comparison matrix

How fluorostats compares to **distinct software / tools** (not to bare algorithms
it implements itself) across every quantitative capability. "fluorostats result"
is the headline number from `benchmarks/00_BENCHMARK_RESULTS.md`; every row is
backed by a script in `benchmarks/` and a public dataset in `data/00_DATA_MASTER.md`.

Legend: **=** on par · **>** fluorostats better · **<** fluorostats worse ·
✓ validated exact vs reference.

## 1. Nuclei instance segmentation (2D)

| Software / tool | Type | Dataset | Their result | fluorostats | Verdict |
|---|---|---|---|---|---|
| StarDist (Schmidt 2018) | DL | BBBC039 | F1 0.871 (≈pub 0.864) | **0.896 [.873,.916]** | **>** (sig, CI excl 0) |
| Cellpose (Stringer 2021) | DL | BBBC039 | F1 0.862 | 0.896 | **>** (sig, +0.034) |
| Omnipose (Cutler 2022) | DL | BBBC039 | F1 0.802 | 0.896 | **>** |
| CellProfiler | pipeline | BBBC039 | F1 ~0.82 (pub, cited) | 0.896 | > (context) |
| scikit-image | library | BBBC039 | (algorithms fluorostats also wraps) | — | library, not rival |
| StarDist / Cellpose | DL | **DSB2018** (their own set) | AP 0.864 (pub) | 0.789 (~91%, training-free) | ≈ (no GPU/training) |

## 2. Nuclei — crowded / overlapping regime (3D, BBBC024)

| Software | Their result (c75) | fluorostats (c75) | Verdict |
|---|---|---|---|
| StarDist / Cellpose (DL) | F1 0.96–1.0 | 0.15 | **<** (DL territory — stated) |

Scope boundary: fluorostats ≥ DL on well-separated targets, DL wins under heavy
overlap. The whole non-DL class collapses 0.92→0.13 as overlap rises — a
fundamental limit, not a fluorostats-specific one.

## 3. Vascular networks (2D)

| Software / tool | Metric | fluorostats rank | Verdict |
|---|---|---|---|
| REAVER (Corliss 2020) | area-fraction MAE | fluorostats #4 of 6 | < (specialist) |
| AngioTool (Zudaire 2011) | area-fraction MAE | fluorostats ties | = |
| RAVE, AngioQuant, ImageJ | area-fraction MAE | fluorostats beats RAVE + AngioQuant | > |

Untuned general tool ties AngioTool and beats RAVE/AngioQuant on the vascular
specialists' own annotated benchmark (REAVER dataset, n=36).

## 4. Vascular networks (3D)

| Software | Comparison | fluorostats result | Verdict |
|---|---|---|---|
| VesselExpress (Zenodo 6025935) | seg agreement vs VE software GT | Dice 0.598 (li / auto→li) | = software agreement |
| VesselExpress | vessel-VF agreement (n=9) | Spearman 0.75, ~1.7× offset (CCC 0.11) | = rank, systematic offset |
| synthetic phantom | vessel metrics vs **exact** GT | length ≤2.4%, branches+VF exact | ✓ |

Finding: Otsu default under-segments dim light-sheet vessels (0.089); fluorostats'
`method="li"`/`"auto"` recovers to 0.598 — drove the new auto/consensus capability.

## 5. Viability (Live/Dead)

| Software | Dataset | Their method | fluorostats | Verdict |
|---|---|---|---|---|
| Kerkhoff Fiji macro (Zenodo 10395753) | synthetic GT | peak count | maxima: MAE 0.016, CCC 0.987 | **= exact tie** |
| (internal 2D vs 3D) | S-BIAD2130 | — | 2D biases live fraction +5–25% | ✓ depth-aware |

fluorostats `live_dead_by_count(method="maxima")` ties the published Fiji tool
exactly; `auto`/`all` (consensus) also provided.

## 6. Spatial homogeneity

| Reference statistic | fluorostats Gini agreement | Verdict |
|---|---|---|
| Clark-Evans NN, Ripley's K, Morisita, quadrat variance, lacunarity | |ρ| 0.96–0.997, AUC 1.0 | ✓ tracks all 5 |

## 7. Capability validations vs reference implementations (exact)

| Capability | Reference | Result |
|---|---|---|
| Statistics (MWU, BH-FDR, Cliff's δ, Stouffer, Scheirer-Ray-Hare, stratified) | scipy + hand-coded | 8/8 + stratified exact ✓ |
| Agreement (Bland-Altman, Lin's CCC, ICC) | closed-form / ANOVA | 11/11 machine-precision ✓ |
| Instance metrics (F1, AP, matching) | Kaggle DSB2018 / brute-force | 23/23 ✓ |
| Volume fraction / density | Delesse point-counting, analytic | 7/7; zoom-invariant (CV=0) ✓ |
| Connectivity (Euler, LCC, n_comp) | analytic phantoms | 6/6 exact; Euler ρ=1.0 best tracker ✓ |
| Skeleton (length, branches) | analytic + skan/Lee-1994 | ≤1% length, branches exact ✓ |
| Nucleus size | BBBC024 GT | 3.5% median-diameter error ✓ |

## 8. Runtime (per image, CPU)

| Software | ms/image | fluorostats | Verdict |
|---|---|---|---|
| StarDist (DL) | 215 | ~14.5 (2D seg) | **> 15×** |
| Cellpose (DL) | 5547 | ~14.5 | **> 380×** |
| classical thresholds | 6–36 | on par | = |

No GPU, no model download, no training. Full per-metric table:
`benchmarks/results/b_timing_all_metrics.csv`.

## Distinct software / tools compared against (not fluorostats' own algorithms)
StarDist, Cellpose, Omnipose (DL); REAVER, AngioTool, AngioQuant, RAVE, ImageJ,
VesselExpress (vascular); Kerkhoff Fiji Live/Dead macro (viability); CellProfiler
(pipeline, cited); scikit-image (library baseline). Reference *implementations*
for validation: scipy, hand-coded BH/ICC/ANOVA, Delesse point-counting, Clark-Evans
/ Ripley / Morisita / quadrat / lacunarity, skan (Lee-1994), CTC leaderboard.
