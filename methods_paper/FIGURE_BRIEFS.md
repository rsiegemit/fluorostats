# fluorostats paper — figure plan, assessment & build briefs

> **v0.4 consolidation (READ FIRST).** Display items were reduced to the reference-set norm:
> **6 main figures + 2 main tables**, per-experiment numbers → Supplementary Tables S1–S9, and the
> correctness phantom battery → **Extended Data Fig. 1**. The old 8-figure numbering in §1 and §5
> below maps to the final scheme as: old **Fig 3 (nuclei) + Fig 4 (scope) → merged Fig 2** (panels
> a ranking, b CI forest, c overlay, d crossover curve, e separated/crowded); old **Fig 5 → Fig 3**
> (vascular); old **Fig 6 → Fig 4** (viability); old **Fig 7 → Fig 5** (homogeneity); old **Fig 8 →
> Fig 6** (runtime); old **Fig 2 (correctness) → Extended Data Fig 1**. Fig 1 (schematic) unchanged.
> The panel-level build specs in §5 are otherwise current — apply the mapping above. The 8 small
> per-experiment tables are now Supplementary Tables S1–S9 (numbers unchanged; see the manuscript).

Two audiences: (1) an assessment of the existing 13 benchmark plots against the
conventions of the reference papers — what to keep, merge, move, or supplement, and where
they need more data; (2) per-figure build briefs Claude Code can execute against the CSVs
in `benchmarks/results/`. Figure 1 is already built (`fig1_schematic.svg`).

The guiding conventions, from the exemplars: **Figure 1 is a schematic** (Cellpose, StarDist,
QuPath, CellProfiler); **group figures by task, not by plot type** (all consolidate this way);
**every methods paper about image quantification carries qualitative image overlays** —
raw / ground-truth / prediction (Cellpose Fig. 3, REAVER Fig. 2, VesselExpress, VesSAP) — and
your current 13 have **none**, which is the single biggest gap; **rankings get error bars / CIs**
and the modern bar is bootstrap CIs + paired significance (Cell Tracking Challenge, DSB, Weber
2019); **agreement is shown with Bland–Altman + identity-line scatter** (REAVER, VesselExpress),
not bars; **timing on log scale**; **one colour per method, colourblind-safe, used identically
everywhere** (Okabe-Ito; Crameri 2020).

---

## 1. Final figure plan (8 main figures)

| Fig | Title | Panels | Built from | Status |
|---|---|---|---|---|
| 1 | Overview schematic | single | — | **BUILT** (`fig1_schematic.svg`) |
| 2 | Correctness | a phantom battery, b zoom-invariance | b1_topology_phantoms, b1_skeleton_phantoms, b_skeleton_tree, b_density_normalization_cv | **NEW** composition |
| 3 | Nuclei vs deep learning | a ranking, b CI forest, c qualitative overlay | b2_nuclei_methods (a), b_dl_ci/omnipose (b), **images (c)** | keep + restyle; **c new** |
| 4 | Scope boundary | a crossover curve, b separated/crowded bars, c qualitative fields | b_clustering_curve (a), b2_crowded/scope (b), **images (c)** | merge existing; **c new** |
| 5 | Vascular networks | a REAVER ranking, b vessel overlay, c 3D phantom, d VE agreement | b4_reaver_ranking (a), **images (b)**, b_vascular_phantom_3d (c), b_ve_metrics (d) | keep + merge; **a needs data, b new** |
| 6 | Depth-resolved viability | a 2D-vs-3D bias + z-profile, b macro tie, c Live/Dead overlay | b3_viability_multi (a), b_viability_external (b), **images (c)** | **a & c new**, b recast |
| 7 | Spatial homogeneity | a point-pattern panels, b five-stat correlation | synthetic patterns (a), b_homogeneity_multi_corr (b) | **a new**, b keep |
| 8 | Runtime & determinism | single (log) | b_timing_all_metrics / b_timing | keep + restyle |

Supplementary / Extended Data: DSB2018 ranking (b_dsb2018), CTC 3D Dice (b2_ctc_multi),
noise robustness (b_noise_robustness), denoising (b_denoising), size recovery (b_nuclei_size),
full per-metric timing (b_timing_all_metrics), VesselExpress segmentation Dice (b_vesselexpress).

---

## 2. Assessment of your existing 13 plots (grounded in the references)

**Keep, becomes a main panel:**
- `fig1_nuclei_ranking` → **Fig 3a.** Good and expected (ranked bars, best→worst). *Needs more data:* add bootstrap 95% CIs as whiskers — the references (DSB, CTC, Weber 2019) treat CI-free rankings as dated, and you already compute CIs elsewhere, so extend them to every bar. Highlight fluorostats in its fixed hue.
- `fig2_dl_ci` → **Fig 3b.** This is your strongest single plot and exactly what the classic challenge papers lack (a bootstrap-CI forest with paired significance). Keep almost as-is; restyle to the palette; add Omnipose as a fourth row so all three DL baselines bracket fluorostats in one panel.
- `fig3_clustering_curve` → **Fig 4a.** Keep, but **merge in the DL line** (~0.96 across clustering) so the crossover reads in a single panel — the Schulz/Cellpose convention of drawing the boundary as one curve with a mechanism, rather than splitting fluorostats and DL across two figures.
- `fig8_vascular_ranking` → **Fig 5a.** Keep the ranking. *Needs more data:* REAVER's own convention is to split **accuracy (error vs GT)** from **precision (residual variance)** and to mark unbiased tools via a Bonferroni-corrected zero-bias test. Add the precision axis and the zero-bias flag; that is what turns "ties AngioTool" into "statistically indistinguishable error, and unbiased where a specialist is biased."
- `fig9_homogeneity` → **Fig 7b.** Keep; restyle. Consider adding Spearman CIs.
- `fig4_timing_all` → **Fig 8.** Keep; log scale is already right; restyle and mark fluorostats.
- `fig10_vascular_phantom` → **Fig 5c.** Keep, merged into the vascular figure rather than standing alone.
- `fig11_vesselexpress_seg` → **Fig 5b.** Keep — this is your one genuine qualitative overlay and it belongs in the vascular figure. Extend it to the raw/segmentation/skeleton/branchpoints column format the vascular papers use.

**Recast (same data, better form per the references):**
- `fig5_viability_external` → **Fig 6b.** Currently MAE+CCC bars. Recast as **Bland–Altman + identity-line scatter** (fluorostats vs macro), the agreement grammar REAVER/VesselExpress use; keep the CCC/MAE as annotations. Add per-image overlays (Kerkhoff's signature) if images are available.
- `fig13_vesselexpress_metric` → **Fig 5d.** Recast the rank-consistent-but-offset story as a **Bland–Altman** with the ~1.7× systematic offset drawn and labelled, plus the Spearman 0.75 as a rank-scatter inset.

**Merge / retire:**
- `fig7_scope_boundary` → fold into **Fig 4b** (companion to the crossover curve). Strong as a panel, redundant as a standalone figure.
- `fig12_vs_dl_all` → **retire**; it duplicates Fig 3a (ranking) and Fig 3b (CI forest). Its content is fully covered once Omnipose is added to 3b.

**Move to supplementary (robustness sweeps — supplementary by convention):**
- `fig6_noise` (noise robustness) → Supplementary.

**Net:** of the 13, ten become main-figure panels, one is retired, one moves to supplementary, and two are recast into agreement plots. The gaps the 13 do not cover are (i) a schematic — now Fig 1, built; (ii) a correctness figure — now Fig 2; (iii) qualitative image overlays — now Figs 3c, 4c, 5b, 6c; (iv) the 2D-vs-3D viability-bias figure, the paper's "catch more" headline — now Fig 6a; (v) point-pattern panels — now Fig 7a.

---

## 3. The biggest missing thing: qualitative overlays

None of the 13 shows an image. Every segmentation/vascular methods paper in the reference set
leads its results with qualitative panels, because reviewers want to *see* the tool work. Four
are needed, each requiring the raw images + masks (which live with the benchmark data, not the
CSVs):

- **Fig 3c** — nuclei: raw / expert GT / fluorostats / StarDist, 3–4 example crops, same crops
  across columns, scale bar, colour key on-figure.
- **Fig 4c** — a well-separated field beside a crowded (c75) field, fluorostats overlay on each,
  to *show* why the crossover happens.
- **Fig 5b** — vessels: raw / segmentation / skeleton / branchpoints, fluorostats vs VesselExpress
  (extend `fig11`).
- **Fig 6c** — Live/Dead: raw two-channel, fluorostats live/dead classification, and a 2D-projection
  vs 3D view that visualizes the bias.

Convention (Schmied 2021): scale bar on every panel, colour/overlay key printed on the figure,
magenta/green rather than red/green for two-channel, marked inset zooms.

---

## 4. Global style spec (apply to every rebuilt figure)

- **Palette — Okabe-Ito, colourblind-safe:** black `#000000`, orange `#E69F00`, sky `#56B4E9`,
  green `#009E73`, yellow `#F0E442`, blue `#0072B2`, vermillion `#D55E00`, purple `#CC79A7`.
- **fluorostats is always blue `#0072B2`**; deep-learning tools in a consistent second family
  (StarDist orange, Cellpose vermillion, Omnipose purple); classical thresholds in greys.
- **Redundant encoding:** distinguish series by marker shape / line style as well as colour, so
  figures survive greyscale.
- **Rankings** sorted best→worst with 95% bootstrap-CI whiskers; **timing** on log scale;
  **agreement** panels get an identity line and reported bias ± limits; consistent error-bar
  definition stated in every caption.
- **Captions** self-contained: one bold title sentence, then per-panel text with n, dataset,
  statistic and error definition; scale-bar value and colour key for image panels.
- **Panel labels** bold lowercase (a, b, c), top-left. Sans-serif (Helvetica/Arial). Export
  vector (PDF/SVG) at 300+ dpi raster fallback.
- Drop the cream background of the current figures for clean white.

---

## 5. Per-figure build briefs (for Claude Code)

### Fig 2 — Correctness *(new)*
Panels: **(a)** small multiples of the analytic phantoms with expected-vs-measured labels —
ball (χ=1), torus (χ=0), N disjoint balls (χ=N), a genus-g surface; and skeleton trees (depth 2/3
exact 7/3, 15/7; depth 4 → 27 vs 31). **(b)** zoom-invariance: density vs digital-zoom for the five
schemes, showing fluorostats per-mm³ flat (CV=0) while raw/per-Mpx/per-area/per-slice drift.
Data: `b1_topology_phantoms.csv`, `b1_skeleton_phantoms.csv`, `b_skeleton_tree.csv`,
`b_density_normalization_cv.csv`. Note: much of the correctness story is Table 1; keep this figure
lean (it exists to make "exact" visual).

### Fig 3 — Nuclei vs deep learning
**(a)** 12-method F1 ranking bars **with bootstrap 95% CI whiskers** (extend CIs to all methods),
fluorostats highlighted. Data: `b2_nuclei_methods.csv` (+ recompute CIs). **(b)** bootstrap-CI
forest of fluorostats vs StarDist, Cellpose, Omnipose on BBBC039 n=200, with the paired
differences annotated. Data: `b_dl_ci.csv`, `omnipose_eval.csv`. **(c)** qualitative overlay
*(new; needs BBBC039 images + masks)*: raw / GT / fluorostats / StarDist, 3–4 crops.

### Fig 4 — Scope boundary
**(a)** instance-F1-vs-clustering crossover curve (c00→c75) for fluorostats/threshold methods
**with the DL line (~0.96) overlaid**. Data: `b_clustering_curve.csv` + DL points from
`b2_crowded_c75_comparison.csv`. **(b)** separated-vs-crowded bars, fluorostats vs StarDist/Cellpose.
Data: the crowded head-to-head (Table 3). **(c)** qualitative separated vs crowded field with
fluorostats overlay *(new; needs BBBC024 c00/c75 images)*.

### Fig 5 — Vascular networks
**(a)** REAVER six-tool ranking, **accuracy + precision split with the zero-bias flag** *(needs
per-image residuals/variance + a Bonferroni zero-bias test — recompute from the REAVER per-image
outputs, not just the summary MAE)*. Data: `b4_reaver_ranking.csv` (+ per-image residuals).
**(b)** vessel qualitative overlay (raw / segmentation / skeleton / branchpoints), fluorostats vs
VesselExpress; extend `fig11_vesselexpress_seg` *(needs the light-sheet crops)*. **(c)** 3D phantom
exact-GT (length ≤2.4%, branches/VF exact). Data: `b_vascular_phantom_3d.csv`. **(d)** VesselExpress
vessel-VF agreement as Bland–Altman with the ~1.7× offset drawn + Spearman 0.75 rank inset.
Data: `b_ve_metrics.csv`.

### Fig 6 — Depth-resolved viability
**(a)** *(new, the headline "catch more" figure)*: 2D-vs-3D bias — paired per-sample deltas of live
fraction for MIP / mid-plane / brightest / mean-of-slice / attenuation-corrected vs the 3D reference
(report direction + count of samples affected, Theart-style), plus a per-z live-fraction profile.
Data: `b3_viability_multi.csv` (+ per-z profile from the viability module). **(b)** tie to the Fiji
macro as Bland–Altman + identity-line scatter (CCC 0.987, MAE 0.016 annotated). Data:
`b_viability_external.csv`. **(c)** Live/Dead qualitative overlay, 2D projection vs 3D *(new; needs
S-BIAD2130 crops)*.

### Fig 7 — Spatial homogeneity
**(a)** *(new)*: point-pattern panels — regular / Poisson-CSR / clustered — each with its Gini value,
the spatstat/Amgad visual grammar. Data: regenerate synthetic patterns from `b_homogeneity_multi.py`.
**(b)** five-statistic correlation (Gini vs Morisita / quadrat / lacunarity / Clark–Evans / Ripley),
|ρ| 0.96–0.997, AUC 1.0. Data: `b_homogeneity_multi_corr.csv`.

### Fig 8 — Runtime & determinism
Per-metric / per-image runtime on log scale vs comparators (fluorostats 14.5 ms; StarDist 215;
Cellpose 5,547), fluorostats highlighted; annotate 15× / 380×. Data: `b_timing_all_metrics.csv`,
`b_timing.csv`. Optionally a small determinism panel (identical outputs across repeated runs).

---

## 6. Data that must be computed (does not exist yet)

1. **Bootstrap CIs for the 12-method ranking** (Fig 3a) — extend the CI computation to every method.
2. **REAVER accuracy/precision split + Bonferroni zero-bias test** (Fig 5a) — needs per-image
   residuals/variance, not the summary MAE.
3. **Per-sample paired deltas + per-z live-fraction profile** (Fig 6a) — from the per-image / per-z
   viability outputs.
4. **Qualitative overlays** (Figs 3c, 4c, 5b, 6c) — the raw images + masks (BBBC039, BBBC024,
   VesselExpress, S-BIAD2130), not in the CSVs.
5. **Omnipose row added to the CI forest** (Fig 3b) — already have `omnipose_eval.csv`.

Hand each figure (or panel) to Claude Code with the brief above; it has the images, masks, and
CSVs locally. I built Fig 1 here because it is a diagram, not a data plot.
