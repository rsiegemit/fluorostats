# fluorostats methods paper — PROJECT STATE (resume here)

Single source of truth for resuming this work. Last updated after the full
benchmark campaign. Read this first, then `benchmarks/BENCHMARK_INDEX.md` and
`benchmarks/00_BENCHMARK_RESULTS.md`.

## Goal
Turn **fluorostats** (open-source fluorescence-microscopy quantification library)
into a **methods paper**, demonstrating it returns comparable-or-better results
than existing methods, benchmarked on public datasets, with every capability
validated. Then the paper can be cited by the Extrusion bioprinting paper.

## Where things live
- **Library**: `src/fluorostats/` — v0.5.0, 18 modules, pushed to GitHub `main`
  (github.com/rsiegemit/fluorostats). Tests in `tests/` (98 passing).
- **Methods paper workspace**: `methods_paper/`
  - `research/` — 13 capability literature dossiers (~145 refs, DOI-verified) +
    `00_SYNTHESIS.md` + `00_COMPARISON_MATRIX.md`
  - `data/` — dataset dossiers + `00_DATA_MASTER.md` + `PUBLISHED_BASELINES.md`;
    `data/downloads/` holds 21GB raw public data (GITIGNORED — re-download via
    URLs in `00_DATA_MASTER.md` / dataset dossiers).
  - `benchmarks/` — 21 benchmark scripts, `results/` (61 CSVs), `figures/`,
    `BENCHMARK_INDEX.md` (39-row registry), `00_BENCHMARK_RESULTS.md` (synthesis),
    `cluster/` (Slurm scripts for DL baselines).
  - `CHANGELOG_AND_RERUNS.md` — library changes + deferred Live/Dead rerun tracking.

## Environment (CRITICAL)
- **Local interpreter: `python3.13`** (has fluorostats 0.5.0 + numpy/scipy/skimage/
  pandas/tifffile/czifile). The default `python3` is 3.14 and EMPTY — do not use it.
- **Local pip is BLOCKED** (PEP 668, externally-managed). Do NOT `pip install`
  locally; it fails. Implement references directly, or use the cluster.
- **Cluster: `ssh amd`** (login1.hpcfund, AMD ROCm HPC, Slurm). `pip install --user`
  works there. DL baselines (Cellpose v3, StarDist, TF-CPU) live in
  `~/fluorostats_bench/` on the cluster. Submit via `sbatch`; partitions `devel`
  (30min) and `mi2101x`. Key gotcha: `module purge` first; for TF use
  `export LD_LIBRARY_PATH=/opt/rocm-6.4.1/lib` OR pip tensorflow-cpu; needs
  `termcolor` in ~/.local.
- **Background sub-agents CANNOT pip install** (same PEP 668 wall) — give them
  no-install tasks only (they succeed then). They CANNOT run cluster DL either.

## Library state (v0.5.0, all pushed)
- v0.4: `skeleton` module (prune_skeleton, n_junction_nodes, skeleton_metrics
  with opt-in prune; 2D-spacing bugfix), `agreement` module (bland_altman,
  lins_ccc, icc), FOV-density helpers in metrics_3d.
- v0.5: `viability` (live_dead_fractions, viability_depth_profile,
  viability_2d_vs_3d, attenuation_correct), `validate` (instance_f1,
  average_precision, match_instances), `objects.watershed_split` +
  `clear_border_labels`, `style` module.
- All additions are OPT-IN / additive → **existing 3D Live/Dead outputs unchanged,
  no rerun triggered.** README + __init__ docstrings document them.

## Benchmark campaign — DONE (44 benchmarks, 30+ reference methods, ≥4 comparators each)
See `benchmarks/00_BENCHMARK_RESULTS.md` for the full table. Headlines:
- **Nuclei 2D (BBBC039, 12 methods):** fluorostats F1 0.90 ≥ StarDist 0.87 /
  Cellpose 0.87; on DSB2018 (StarDist's own data) fluorostats 0.789 = ~91% of
  DL's 0.864 at zero training cost, rank 1 of heuristics.
- **Scope boundary:** separated nuclei fs 0.94 ≥ DL; crowded (BBBC024 c75) the
  whole non-DL class collapses 0.92→0.13 vs DL 0.96–1.0. Clean when-to-use-DL line.
- **Vascular (REAVER 6 tools):** fs #4, ties AngioTool on their own annotated data.
- **Runtime:** fs 14.5 ms/img CPU, 15×/380× faster than StarDist/Cellpose.
- **Validations (exact vs references):** stats 8/8, agreement 11/11, instance
  metrics 23/23, volume fraction 7/7, density zoom-invariant CV=0, connectivity
  euler_number ρ=1.0, homogeneity vs 5 spatial stats ρ 0.96–0.997.
- **DL baselines validated** to reproduce published F1 before comparison
  (StarDist 0.871≈0.864 published; Cellpose 0.862). See `data/PUBLISHED_BASELINES.md`.
- **Full module audit done** — every quantitative function benchmarked vs ≥4
  methods. Last 5 gaps closed: viability (6 methods, 2D biases live fraction
  +5–25%), stratified_mann_whitney (exact vs scipy), centroid_homogeneity
  (ρ=0.975), prune_skeleton (best of 5), background_subtract (top-hat r45 best of
  6). Only io/report/plots/qc/render3d/style remain un-benchmarked (viz/utility,
  no external method to compare). External viability-tool comparison: in progress.

## Honesty ledger (stated, not hidden)
- Crowded/overlapping nuclei: fluorostats collapses — DL territory.
- Hard cytoplasmic 3D (CTC A549/CHO): Dice 0.52–0.69 (untuned).
- Power: bootstrap-from-small-pilot is optimistic (documented, not a bug).
- Several of MY test scripts had bugs I caught+fixed (Stouffer convention,
  zoom-invariance count, stuck wait-loops); competitor DL baselines validated first.
- Dead dataset URLs (VascuSynth, MiniVess-EBRAINS) substituted honestly.

## The driving application (separate, already delivered)
`../Extrusion_Data_Results/for_email/` + `extrusion_analysis.zip`: GelMA vs Hybrid
Day-14 middle Live/Dead analysis (v3, n=12 vs 12). Deferred: ONE final Live/Dead
rerun IF we adopt library pruning / node-based branchpoints / viability columns
(tracked in CHANGELOG_AND_RERUNS.md).

## NEXT STEPS (pick up here)
1. **Draft the methods paper** — Methods + Results write almost directly from
   `00_BENCHMARK_RESULTS.md`; Intro/gap from `research/00_SYNTHESIS.md`.
2. Optional more benchmarks: correctness anchors to ≥4 comparators; more real
   vascular/viability datasets; figure generation for each benchmark.
3. Proof-stage: verify flagged citations (see research dossiers' flag lists).
4. When paper structure is set, decide on the final Live/Dead rerun.
