# fluorostats paper — Claude Code figure-generation prompts

Copy-paste prompts for building the manuscript figures locally (where the repo, the CSVs in
`benchmarks/results/`, and the raw images in `data/downloads/` live). Run each from the repo root.
**Prepend the SHARED STYLE BLOCK to every prompt** (or tell Claude Code to read this file's top
section first). Fig 1 is already built (`benchmarks/figures/handoff/fig1_schematic.svg`).

Environment note (from PROJECT_STATE.md): use `python3.13` (has fluorostats + numpy/scipy/skimage/
pandas/tifffile/czifile). Local `pip install` is blocked — no new deps. Raw images are gitignored;
re-download via URLs in `data/00_DATA_MASTER.md` if missing. Existing plotting code lives in
`benchmarks/b_make_figures.py` and the `fluorostats.style` module — reuse/restyle rather than
starting from scratch.

---

## SHARED STYLE BLOCK (prepend to every prompt)

> **Style — apply to every panel.** Publication-grade, colourblind-safe, one visual system.
> - Palette: Okabe–Ito — black `#000000`, orange `#E69F00`, sky `#56B4E9`, green `#009E73`,
>   yellow `#F0E442`, blue `#0072B2`, vermillion `#D55E00`, purple `#CC79A7`.
> - **fluorostats is always blue `#0072B2`.** DL tools: StarDist orange `#E69F00`, Cellpose
>   vermillion `#D55E00`, Omnipose purple `#CC79A7`. Classical thresholds in greys.
> - Distinguish series by marker shape / line style as well as colour (survive greyscale).
> - Ranked bars sorted best→worst with 95% bootstrap-CI whiskers; timing on log scale; agreement
>   panels get an identity line + reported bias ± limits. White background (not cream).
> - Sans-serif (Helvetica/Arial); bold lowercase panel labels (a, b, c) top-left; scale bar +
>   on-figure colour key on every image panel; magenta/green (not red/green) for 2-channel overlays.
> - Export **vector PDF + 300-dpi PNG** to `benchmarks/figures/main/` (main) or
>   `benchmarks/figures/extended/` (Extended Data). Self-contained caption in a sibling `.txt`.
> - Multi-panel: size so the whole figure reduces to a single column/page with details legible.

---

## PROMPT 1 — Figure 2: Nucleus segmentation and the deep-learning boundary

```
Build Figure 2 for the fluorostats methods paper: "Nucleus segmentation and the deep-learning
boundary." A multi-panel figure (a–f). Follow the SHARED STYLE BLOCK. Output to
benchmarks/figures/main/fig2_nuclei_boundary.{pdf,png} plus a caption .txt.

Panels:
(a) 12-method F1 ranking on BBBC039. Data: benchmarks/results/b2_nuclei_methods.csv. Horizontal bars
    sorted best→worst, fluorostats (Otsu+CC) highlighted in blue. ADD 95% bootstrap CIs as whiskers
    to EVERY method (resample the per-image F1 scores, 10,000 draws); if per-image scores aren't in
    the CSV, recompute them from the benchmark and cache a per-image CSV.
(b) Bootstrap-CI forest plot, fluorostats vs StarDist, Cellpose, Omnipose on BBBC039 (n=200). Data:
    benchmarks/results/b_dl_ci.csv + omnipose_eval.csv. One row per method: mean F1 with 95% CI;
    annotate the paired differences (fs−StarDist +0.025 [0.004,0.042]; fs−Cellpose +0.034
    [0.008,0.057]). fluorostats blue, DL in their assigned colours.
(c) Qualitative overlay: 3–4 BBBC039 crops, columns = raw / expert GT / fluorostats / StarDist,
    same crops across columns. Instance outlines coloured; scale bar + colour key on-figure. Needs
    BBBC039 images + masks (data/downloads/; URL in data/00_DATA_MASTER.md) and a StarDist mask set
    (already produced for b_dl_ci — reuse; do NOT retrain).
(d) Crossover curve: instance F1 vs BBBC024 clustering (c00→c75) for fluorostats/threshold methods,
    with the DL line (~0.96) overlaid. Data: benchmarks/results/b_clustering_curve.csv +
    b2_crowded_c75_comparison.csv (DL points). Shade the region where DL overtakes.
(e) Separated-vs-crowded fields: one well-separated BBBC024 c00 field and one crowded c75 field,
    fluorostats overlay on each, showing why CC labelling merges touching objects. Needs BBBC024
    volumes (data/downloads/).
(f) Scope decision map: a 2D "when to use fluorostats vs a trained segmenter" map. X = instance
    overlap/crowding (use the clustering fraction as the measured axis); shade green where fluorostats
    is at parity (F1 ≥ DL, up to ~c25) and red where DL wins (≥c50), with the measured crossover
    marked. Synthesize from b_clustering_curve.csv + the crowded head-to-head; keep it schematic but
    data-anchored. This panel stands in for the paper's Scope section, so make it self-explanatory.

All accuracy = instance F1 / AP matched over IoU thresholds (see fluorostats.validate). Reuse
existing figure code in b_make_figures.py where it exists.
```

---

## PROMPT 2 — Figure 3: Vascular networks

```
Build Figure 3 for the fluorostats methods paper: "Vascular networks." Panels a–d. Follow the
SHARED STYLE BLOCK. Output benchmarks/figures/main/fig3_vascular.{pdf,png} + caption .txt.

(a) REAVER six-tool ranking on the REAVER benchmark (n=36). Data:
    benchmarks/results/b4_reaver_ranking.csv (and per-image residuals from b4_reaver_vascular.py /
    b4_reaver_summary.csv). Show TWO things REAVER's paper shows: ACCURACY (mean absolute error vs
    manual GT, ranked bars) and PRECISION (residual variance / Brown–Forsythe spread). Mark each tool
    unbiased/biased with a two-tailed test that mean error = 0, Bonferroni-corrected (flag unbiased
    tools, e.g. with "#"). fluorostats blue; it should read as statistically level with AngioTool and
    unbiased. If per-image residuals aren't cached, recompute them from the REAVER benchmark script.
(b) Vessel qualitative overlay: columns = raw / segmentation / skeleton / branchpoints, comparing
    fluorostats and VesselExpress on 2–3 light-sheet crops. Extend the existing
    fig11_vesselexpress_seg. Needs VesselExpress volumes (Zenodo 6025935) + fluorostats(li/auto)
    masks (already produced for b_vesselexpress). Scale bar + colour key.
(c) 3D synthetic phantom accuracy vs exact GT: centreline length error (0.6–2.4%), branch count
    (exact), volume fraction (exact). Data: benchmarks/results/b_vascular_phantom_3d.csv. Small
    expected-vs-measured panel (bars or dot-with-error).
(d) VesselExpress metric agreement: Bland–Altman of vessel volume fraction, fluorostats vs
    VesselExpress (n=9), with the ~1.7× systematic offset drawn and labelled (bias ± limits); inset a
    rank scatter (Spearman 0.75). Data: benchmarks/results/b_ve_metrics.csv.
```

---

## PROMPT 3 — Figure 4: Depth-resolved viability

```
Build Figure 4 for the fluorostats methods paper: "Depth-resolved viability." Panels a–c. Follow the
SHARED STYLE BLOCK. Output benchmarks/figures/main/fig4_viability.{pdf,png} + caption .txt. This
figure is the paper's "catch more" headline — make panel (a) the visual anchor.

(a) 2D-vs-3D bias. Take the true voxelwise 3D live fraction as reference (S-BIAD2130). Show, for
    each 2D/heuristic reduction (mid-plane, brightest-focus, MIP, mean-of-per-slice,
    attenuation-corrected), the PAIRED per-sample delta vs 3D (points + connecting lines or a slope
    plot), and report the DIRECTION and the count of samples biased (Theart-style, e.g. "MIP high in
    N/…"). Alongside, a per-z live-fraction profile (live fraction vs depth) showing the depth trend
    and how attenuation-correction flattens it. Data: benchmarks/results/b3_viability_multi.csv; get
    the per-z profile from fluorostats.viability (viability_depth_profile) on the S-BIAD2130 stack.
(b) Tie to the published Fiji macro (Kerkhoff, synthetic GT). Bland–Altman + identity-line scatter,
    fluorostats maxima vs macro, annotate CCC 0.987 / MAE 0.016; include the other modes (CC, area,
    Otsu-CC) as fainter series. Data: benchmarks/results/b_viability_external.csv.
(c) Live/Dead qualitative overlay: raw two-channel (magenta/green), fluorostats live/dead
    classification, and a 2D-projection view beside the 3D view to visualise the bias. Needs
    S-BIAD2130 stack (data/downloads/; URL in data/00_DATA_MASTER.md). Scale bar + key.
```

---

## PROMPT 4 — Figure 5: Spatial homogeneity and the integrated statistics layer

```
Build Figure 5 for the fluorostats methods paper: "Spatial homogeneity and the integrated statistics
layer." Panels a–c. Follow the SHARED STYLE BLOCK. Output
benchmarks/figures/main/fig5_homogeneity_stats.{pdf,png} + caption .txt.

(a) Point-pattern panels: three synthetic fields — regular/lattice, Poisson (CSR), clustered — each
    rendered with its tile-based Gini value beneath (the spatstat/Amgad visual grammar). Regenerate
    the patterns from benchmarks/b_homogeneity_multi.py (same generator used for the correlations).
(b) Five-statistic correlation: fluorostats tile Gini vs Morisita, quadrat variance, gliding-box
    lacunarity, Clark–Evans NN, Ripley's K/L across the regular→clustered sweep. Data:
    benchmarks/results/b_homogeneity_multi_corr.csv. Small-multiple scatter or a correlation strip
    with |ρ| annotated (0.96–0.997) and AUC = 1.0. Restyle the existing fig9_homogeneity.
(c) End-to-end statistics worked example (NEW — exhibits the "image → statistic, no export" claim).
    Use the SproutAngio VEGF dose groups (real .czi; benchmarks/results/b_vascular_sproutangio_multi.csv
    has per-sample metrics by VEGF group 1/3/5). Show the fluorostats stats pipeline output on these:
    (i) per-group metric distributions (e.g. volume fraction, length density, junction density) as a
    SuperPlot-style dot plot; (ii) Mann–Whitney U + Cliff's δ effect size for a group contrast;
    (iii) a BH-FDR-adjusted q across the several metrics; (iv) a bootstrap fold-change CI; (v) a
    small power curve. Call the actual fluorostats.stats / power functions so the panel is literally
    the library's output, not a mock-up. Keep it compact — one clean multi-part panel.
```

---

## PROMPT 5 — Extended Data Fig. 1: Correctness

```
Build Extended Data Figure 1 for the fluorostats methods paper: "Correctness against analytic ground
truth." Follow the SHARED STYLE BLOCK. Output benchmarks/figures/extended/ed1_correctness.{pdf,png}
+ caption .txt. This figure carries the correctness detail trimmed from the main text, so make the
expected-vs-measured agreement unmistakable.

Panels:
(a) Topology phantom battery: small renders of the phantoms (ball χ=1, torus χ=0, N disjoint balls
    χ=N, and a genus-g surface) each labelled expected vs measured Euler number / component count —
    zero error. Data: benchmarks/results/b1_topology_phantoms.csv.
(b) Skeleton trees: synthetic trees at depth 2/3 (branch/junction exact: 7/3, 15/7) and the depth-4
    raster-resolution undercount (27 vs 31), rendered with expected/measured counts. Data:
    b1_skeleton_phantoms.csv, b_skeleton_tree.csv.
(c) Zoom-invariance: density vs digital-zoom for five normalisation schemes; fluorostats per-mm³ flat
    (CV=0) while raw count / per-Mpx / per-area / per-slice drift. Data:
    b_density_normalization_cv.csv.
Optionally (d) a compact expected|measured|error table strip as a graphic (topology + skeleton +
volume fraction), mirroring Extended Data Table 1.
```

---

## PROMPT 6 — Extended Data Fig. 2: Runtime & determinism

```
Build Extended Data Figure 2 for the fluorostats methods paper: "Runtime and determinism." Follow the
SHARED STYLE BLOCK. Output benchmarks/figures/extended/ed2_runtime.{pdf,png} + caption .txt.

(a) Per-image / per-metric runtime on a LOG scale vs comparators. Data:
    benchmarks/results/b_timing_all_metrics.csv (+ b_timing.csv for the 2D-seg headline). Highlight
    fluorostats blue; annotate the headline (fluorostats 14.5 ms/img; StarDist 215; Cellpose 5,547;
    ~15× / ~380×). Show the shared-operation parity with SciPy/scikit-image, and separate the few
    validation-time ops (average precision ~20 s, instance F1 ~2 s) so they read as once-per-volume.
(b) Determinism: run a representative fluorostats metric N times on the same input and show
    bit-identical outputs (zero variance), contrasted with the seed/hardware variance a trained
    pipeline would show. A simple "spread = 0" panel is enough; compute it live on one dataset.
Name the CPU (cores/RAM) in the caption.
```

---

## PROMPT 7 — Extended Data Fig. 3: Robustness (optional, if you want it)

```
Build Extended Data Figure 3: "Robustness." Follow the SHARED STYLE BLOCK. Output
benchmarks/figures/extended/ed3_robustness.{pdf,png} + caption .txt. Three panels from existing CSVs:
(a) noise robustness — foreground Dice vs SNR (benchmarks/results/b_noise_robustness.csv);
(b) denoising recovery — Dice with none/median/gaussian/TV/fluorostats (b_denoising.csv);
(c) per-nucleus size recovery — median-diameter error across methods (b_nuclei_size.csv).
fluorostats highlighted in each.
```

---

## PROMPT 8 — Extended Data Fig. 4: Generalization / per-dataset (optional)

```
Build Extended Data Figure 4: "Generalization across datasets." Follow the SHARED STYLE BLOCK. Output
benchmarks/figures/extended/ed4_generalization.{pdf,png} + caption .txt.
(a) DSB2018 ranking — fluorostats F1 0.789 vs classical baselines, with the published DL AP 0.864
    marked (benchmarks/results/b_dsb2018.csv);
(b) CTC 3D foreground Dice — A549/CHO across threshold methods (b2_ctc_multi.csv);
Frame the message: because fluorostats is training-free, performance does not collapse on unseen
modalities the way a trained model off its training distribution can. fluorostats highlighted.
```

---

## Notes for whoever runs these

- **Image-dependent panels** (2c, 2e, 3b, 4c) need the raw datasets locally: BBBC039, BBBC024,
  VesselExpress (Zenodo 6025935), S-BIAD2130. Reuse the masks already produced by the benchmark
  scripts — do NOT retrain any model. If a dataset is missing, the URL is in `data/00_DATA_MASTER.md`.
- **Panels needing a recompute** (not just replotting): 2a (per-method bootstrap CIs), 3a (per-image
  residuals + zero-bias test), 4a (per-z profile via `viability_depth_profile`), 5c (live stats-layer
  output). Everything else is a replot/restyle of an existing CSV.
- Keep the figure numbering as in `DRAFT_v0.5.md` / `FIGURE_BRIEFS.md`. Save captions next to each
  figure so they can drop straight into the manuscript.
```
