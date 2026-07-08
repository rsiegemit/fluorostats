# Competitor mapping: AngioTool

**Paper:** Zudaire E, Gambardella L, Kurcz C, Vermeren S. "A Computational Tool
for Quantitative Analysis of Vascular Networks." *PLoS ONE* 6(11): e27385 (2011).
DOI: [10.1371/journal.pone.0027385](https://doi.org/10.1371/journal.pone.0027385)
· PMC: [PMC3217985](https://pmc.ncbi.nlm.nih.gov/articles/PMC3217985/)
· PubMed: [22110636](https://pubmed.ncbi.nlm.nih.gov/22110636/)

Status: all URLs below were fetched/verified 2026-07-08 unless flagged.

---

## 1. Validation images (what the paper ran on) — and they are downloadable

AngioTool was validated on three 2D fluorescence-microscopy systems, all mouse:

- **Embryonic hindbrain** — E11.5 mouse, endomucin-stained, imaging the
  developing subventricular vascular plexus. Three hindbrains analyzed; the paper
  reports "consistent results across all analysed parameters."
- **Postnatal retina** — P6 wild-type pups, isolectin-B4 stained retinal
  vasculature (the classic flat-mount retina network). Four P6 retinas; "consistent
  read-outs" for total vessel length, branching index, and lacunarity.
- **Allantois explants** — E8.5 embryo-derived cultures treated with PI3K inhibitor
  (LY294002), ROCK inhibitor (Y-27632), or vehicle (DMSO/water) controls.

**These exact images are provided as downloadable test data** on the AngioTool site
(paper: "The images reported in this study are provided as test images..."), plus
40 supplementary figures (S1–S40) of raw microscopy at the PLoS article.

Download bundles (verified on the CCR Downloads page):
- `Hindbrain.zip`
- `Retina.zip`
- `Allantois Water-Y27632.zip`
- `Allantois DMSO-LY294002.zip`

Source: <https://ccrod.cancer.gov/wiki-html/ROB2/Downloads_62196327.html>

> These four zips are the single best head-to-head benchmark: they are AngioTool's
> *own* published validation images, so running fluorostats on them is a fair,
> like-for-like comparison on the competitor's home turf. No tumor or Matrigel/
> tube-formation images are in the original validation set.

---

## 2. Software availability (still downloadable)

Hosted at NIH/NCI Center for Cancer Research, Radiation Oncology Branch (ROB2 wiki).
Developed by Dr. Enrique Zudaire (NCI Angiogenesis Core Facility). GNU GPL.

- Home: <https://ccrod.cancer.gov/confluence/display/ROB2/Home>
- Downloads: <https://ccrod.cancer.gov/confluence/display/ROB2/Downloads>
  (HTML mirror: <https://ccrod.cancer.gov/wiki-html/ROB2/Downloads_62196327.html>)

Files available:
- `AngioToolSetup.exe` — v0.5, 25 May 2011 (32-bit Windows)
- `AngioToolSetup64.exe` — v0.6a, October 2014 (64-bit Windows)
- `AngioTool v0.5 src.zip` — Java source, v0.5

Community fork (unofficial, useful for cross-platform / source reference):
- `imagejan/angiotool` on GitHub — <https://github.com/imagejan/angiotool>
  (flagged: third-party fork, not the official NIH release.)

---

## 3. Metrics AngioTool computes (Table 1)

Per-image outputs written to CSV:

- **Explant area** (analysis region area)
- **Vessels area** (area covered by segmented vessels)
- **Vessel density** = % of area occupied by vessels
- **Total number of junctions** (branch points)
- **Branching index** = junctions per unit area (sprouting-activity proxy)
- **Total vessel length**
- **Average vessel length**
- **Total number of endpoints**
- **Lacunarity** (mean over box sizes; spatial-gap/heterogeneity measure)

**Segmentation method:** multiscale Hessian-based enhancement filter — image
convolved with a fast recursive Gaussian kernel, tube-like structures detected from
Hessian-matrix eigenvalues across scales, then thresholded and skeletonized before
morphometry. This is a 2D pipeline on single-channel fluorescence images.

Quantitative validation is reported as **reproducibility across replicate samples**
(3 hindbrains, 4 retinas giving consistent parameters) rather than accuracy against a
pixel-level ground truth — the paper provides no gold-standard error numbers. This is
a weakness fluorostats can exploit: there is no published accuracy figure to beat, only
consistency, so a ground-truth-anchored accuracy claim would be novel.

---

## 4. Independent public datasets for a fair AngioTool-vs-fluorostats head-to-head

AngioTool's own zips (Section 1) have **no ground-truth masks** (validation was
reproducibility-only). For accuracy comparison you need annotated 2D network images:

- **Edinburgh OCTA parafoveal dataset** — 55 en-face retinal OCTA ROIs (30 train /
  25 test) from 11 participants, **with pixelwise manual vessel segmentations**
  (inter/intra-rater κ = 0.77–0.80). 2D vessel networks; standard network metrics
  (vessel density, fractal dimension, FAZ) apply.
  - Data DOI: <https://doi.org/10.7488/ds/2729>
  - Code: <https://github.com/giaylenia/OCTA_segm_study>
  - Paper: Giarratano et al., PMC [PMC7718823](https://pmc.ncbi.nlm.nih.gov/articles/PMC7718823/)
  - *Best pick for a ground-truth-anchored accuracy head-to-head.*
- **SproutAngio fibrin-bead assay dataset** — in-vitro HUVEC sprouting, confocal,
  graded VEGF-A (0–50 ng/ml), phalloidin/DAPI; manual measurements + ImageJ-plugin
  comparisons included. Mostly 3D/lumen-focused (less directly AngioTool-shaped).
  - Data DOI: <https://doi.org/10.5281/zenodo.7240927>
  - Tool paper: Sci Rep, PMC [PMC10160097](https://pmc.ncbi.nlm.nih.gov/articles/PMC10160097/)
- **Endothelial-spheroid sprouting quantification dataset** — Zenodo
  <https://doi.org/10.5281/zenodo.6444392> (PMC [PMC9093605](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9093605/));
  data + software. (Flagged: DOI cited from search result, not independently fetched.)

Classic fundus vessel-segmentation sets (DRIVE, FIVES) exist but are fundus
photographs, not fluorescence networks — a modality mismatch for AngioTool's intended
use; use only if arguing generality.

---

## 5. How fluorostats runs on the same images + metric alignment

Both tools operate on 2D single-channel network images -> segment -> skeletonize ->
graph. Direct metric correspondences for a table:

| AngioTool metric | fluorostats equivalent |
|---|---|
| Total number of junctions | junction / node count |
| Branching index (junctions/area) | junction density |
| Total vessel length | total branch length (skeleton) |
| Average vessel length | mean branch length |
| Total number of endpoints | endpoint / leaf count |
| Vessels area / vessel density | segmented area fraction |
| Lacunarity | (check fluorostats support; else compute post-hoc from mask) |

**Recommended protocol:**
1. Run AngioTool (v0.6a 64-bit) on its four official zips -> CSV of the metrics above.
2. Run fluorostats on the identical images -> map to the table.
3. Report per-image agreement (junctions, total length, branch density) as the
   primary like-for-like comparison on the competitor's own data.
4. For an *accuracy* claim (not just agreement), add the Edinburgh OCTA set
   (10.7488/ds/2729): score both tools' vessel masks against the manual ground truth.

Verify fluorostats emits total/average branch length, junction count, endpoint count,
and area fraction from a 2D mask; confirm whether lacunarity is built in before
promising that row.
