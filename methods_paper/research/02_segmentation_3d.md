# Segmentation and Thresholding Methods for Fluorescence Microscopy (2D and 3D Confocal)

Research for positioning **fluorostats** in a methods paper. fluorostats uses a
deliberately classical pipeline: Gaussian denoising → rolling-ball background
subtraction → Otsu/Li global thresholding (tunable threshold-scale) →
connected-component size filtering. No training, deterministic, fast, interpretable.

All references below were located and verified via web search of primary venues
(IEEE, Nature Methods, eLife, MICCAI/Springer, PeerJ, Broad Institute BBBC).
Citation counts, where given, are approximate and drawn from the highly-cited
status noted by indexers; treat them as order-of-magnitude, not exact.

---

## A. Classical thresholding (the fluorostats family)

### 1. Otsu (1979) — global histogram thresholding
- **Authors / Year:** N. Otsu, 1979
- **Title:** A Threshold Selection Method from Gray-Level Histograms
- **Venue:** IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66
- **DOI:** 10.1109/TSMC.1979.4310076
- **Citations:** ~50,000+ (one of the most-cited papers in image processing)
- **Description:** Chooses the global threshold that minimizes intra-class
  (within-class) intensity variance, equivalently maximizing between-class
  variance. The default method behind fluorostats' thresholding. Fast, parameter-free,
  optimal for bimodal histograms with well-separated foreground/background.

### 2. Li & Lee / Li & Tam (1993, 1998) — minimum cross-entropy thresholding
- **Authors / Year:** C.H. Li & C.K. Lee, 1993 (iterative form: Li & Tam, 1998)
- **Title:** Minimum Cross Entropy Thresholding
- **Venue:** Pattern Recognition, 26(4), 617–625
- **DOI:** 10.1016/0031-3203(93)90115-D
- **Description:** Selects the threshold minimizing cross-entropy (Kullback
  information distance) between the original and thresholded image. The alternative
  threshold in fluorostats. Often outperforms Otsu on skewed/sparse-signal histograms
  (e.g., sparse puncta on dark background), which is common in fluorescence.

### 3. Sauvola & Pietikäinen (2000) — local/adaptive thresholding
- **Authors / Year:** J. Sauvola & M. Pietikäinen, 2000
- **Title:** Adaptive Document Image Binarization
- **Venue:** Pattern Recognition, 33(2), 225–236
- **DOI:** 10.1016/S0031-3203(99)00055-2
- **Description:** Per-pixel threshold from local window mean and standard deviation
  (extends Niblack). The reference method for *local* thresholding, robust to uneven
  illumination. Relevant as an alternative fluorostats could offer where global Otsu
  fails on shading/vignetting gradients not fully removed by rolling-ball.

### 4. Sternberg (1983) — rolling-ball background subtraction
- **Authors / Year:** S.R. Sternberg, 1983
- **Title:** Biomedical Image Processing
- **Venue:** IEEE Computer, 16(1), 22–34
- **DOI:** 10.1109/MC.1983.1654163
- **Description:** The morphological "rolling ball" grayscale opening that estimates
  and subtracts smooth background. This is exactly the fluorostats background step
  (also the basis of ImageJ's Subtract Background and skimage's `rolling_ball`).
  Establishes fluorostats' preprocessing as standard, decades-validated practice.

### 5. van der Walt et al. (2014) — scikit-image (implementation substrate)
- **Authors / Year:** S. van der Walt, J.L. Schönberger, J. Nunez-Iglesias,
  F. Boulogne, J.D. Warner, N. Yager, E. Gouillart, T. Yu, and the scikit-image
  contributors, 2014
- **Title:** scikit-image: image processing in Python
- **Venue:** PeerJ, 2, e453
- **DOI:** 10.7717/peerj.453
- **Citations:** ~7,000+
- **Description:** Open-source BSD-licensed Python image library providing Otsu, Li,
  Sauvola, rolling-ball, Gaussian filtering, connected components, and watershed —
  the same well-tested primitives fluorostats builds on. Cite to ground
  reproducibility and to explain why fluorostats is trustworthy without bespoke code.

---

## B. Watershed / marker-based instance splitting (the gap fluorostats leaves open)

### 6. Watershed for touching nuclei in 3D confocal (Lin et al., 2003)
- **Authors / Year:** G. Lin et al., 2003
- **Title:** A hybrid 3D watershed algorithm incorporating gradient cues and object
  models for automatic segmentation of nuclei in confocal image stacks
- **Venue:** Cytometry Part A, 56A(1), 23–36
- **DOI:** 10.1002/cyto.a.10079
- **Description:** Representative classical 3D marker-controlled watershed for
  separating touching nuclei in confocal stacks. Directly relevant: fluorostats'
  connected-component step merges touching objects into one label; watershed is the
  standard classical remedy. Honest limitation to state — fluorostats measures signal
  *volume/intensity*, not per-cell instances, so it does not split touching cells.

---

## C. Modern deep-learning segmentation (where DL wins)

### 7. Ronneberger et al. (2015) — U-Net
- **Authors / Year:** O. Ronneberger, P. Fischer, T. Brox, 2015
- **Title:** U-Net: Convolutional Networks for Biomedical Image Segmentation
- **Venue:** MICCAI 2015, LNCS 9351, Springer, 234–241
- **DOI:** 10.1007/978-3-319-24574-4_28
- **Citations:** ~90,000+
- **Description:** Encoder–decoder CNN with skip connections; the foundational
  architecture for essentially all subsequent DL microscopy segmentation (including
  Cellpose, StarDist, PlantSeg). Trainable from few annotated images with heavy
  augmentation. The baseline "why not just use a U-Net" question a reviewer will ask.

### 8. Schmidt et al. (2018) — StarDist
- **Authors / Year:** U. Schmidt, M. Weigert, C. Broaddus, G. Myers, 2018
- **Title:** Cell Detection with Star-Convex Polygons
- **Venue:** MICCAI 2018, LNCS 11071, Springer, 265–273
- **DOI:** 10.1007/978-3-030-00934-2_30
- **Citations:** ~2,500+
- **Description:** Predicts a star-convex polygon (2D) / polyhedron (3D via Weigert
  et al. 2020) per object, excelling at crowded, roundish, touching nuclei without
  merge errors. Clear win over Otsu+connected-components precisely on the dense-nuclei
  case fluorostats cannot split.

### 9. Stringer et al. (2021) — Cellpose
- **Authors / Year:** C. Stringer, T. Wang, M. Michaelos, M. Pachitariu, 2021
- **Title:** Cellpose: a generalist algorithm for cellular segmentation
- **Venue:** Nature Methods, 18(1), 100–106
- **DOI:** 10.1038/s41592-020-01018-x
- **Citations:** ~4,000+
- **Description:** Generalist model predicting spatial flow fields to a cell center;
  trained on 70,000+ objects; generalizes across modalities without retraining, with a
  3D extension reusing the 2D model. The strongest general-purpose baseline for
  instance segmentation; wins on arbitrary cell shapes, membranes, and low-contrast.

### 10. Pachitariu & Stringer (2022) — Cellpose 2.0
- **Authors / Year:** M. Pachitariu, C. Stringer, 2022
- **Title:** Cellpose 2.0: how to train your own model
- **Venue:** Nature Methods, 19, 1634–1641
- **DOI:** 10.1038/s41592-022-01663-4
- **Description:** Model zoo + human-in-the-loop fine-tuning; 100–200 annotated ROIs
  suffice to specialize. Relevant because it quantifies the annotation *cost* of DL —
  the cost fluorostats avoids entirely (zero training).

### 11. Stringer & Pachitariu (2025) — Cellpose3
- **Authors / Year:** C. Stringer, M. Pachitariu, 2025
- **Title:** Cellpose3: one-click image restoration for improved cellular segmentation
- **Venue:** Nature Methods (2025)
- **DOI:** 10.1038/s41592-025-02595-5
- **Description:** Adds learned denoising/deblurring/upsampling restoration before
  segmentation. Shows the trajectory of the DL frontier; contrast with fluorostats'
  fixed, interpretable Gaussian + rolling-ball preprocessing.

### 12. Wolny et al. (2020) — PlantSeg (3D dense tissue)
- **Authors / Year:** A. Wolny et al., 2020
- **Title:** Accurate and versatile 3D segmentation of plant tissues at cellular
  resolution
- **Venue:** eLife, 9, e57613
- **DOI:** 10.7554/eLife.57613
- **Citations:** ~500+
- **Description:** 3D CNN boundary prediction + graph-partitioning/watershed for
  densely packed volumetric confocal & light-sheet data. The state of the art for
  *dense 3D* tissue where every voxel belongs to a cell — the hardest case for a
  global-threshold pipeline, and a fair "upper bound" comparator for 3D confocal.

---

## D. Benchmark / validation datasets and challenges

### 13. Ulman et al. (2017) — Cell Tracking Challenge (3D fluorescence datasets)
- **Authors / Year:** V. Ulman et al., 2017
- **Title:** An objective comparison of cell-tracking algorithms
- **Venue:** Nature Methods, 14, 1141–1152
- **DOI:** 10.1038/nmeth.4473
- **Description:** Provides annotated 3D fluorescence stacks (e.g., Fluo-C3DL-MDA231,
  Fluo-N3DL-TRIC, Fluo-C3DH-A549) with segmentation ground truth (SEG measure = mean
  Jaccard/IoU). Directly usable to validate fluorostats' 3D segmentation.
  See also Maška et al. (2023), Nature Methods, DOI 10.1038/s41592-023-01879-y — the
  10-year retrospective, useful for framing.

### 14. Caicedo et al. (2019) — 2018 Data Science Bowl / BBBC038
- **Authors / Year:** J.C. Caicedo et al., 2019
- **Title:** Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl
- **Venue:** Nature Methods, 16, 1247–1253
- **DOI:** 10.1038/s41592-019-0612-7
- **Description:** 2D nucleus dataset (fluorescence + histopathology + brightfield),
  the canonical instance-segmentation benchmark; data at Broad BBBC038. Good for a 2D
  agreement study. (Underlying collection: Ljosa, Sokolnicki & Carpenter, 2012,
  *Annotated high-throughput microscopy image sets for validation*, Nature Methods
  9, 637, DOI 10.1038/nmeth.2083 — the BBBC resource paper.)

---

## Honest comparison: fluorostats vs. the field

| Scenario | Winner | Why |
|---|---|---|
| Well-separated signal, good SNR, bimodal histogram | **Tie** — Otsu ≈ Cellpose/StarDist for *volume/coverage* metrics | Global threshold recovers the same foreground mask; DL adds no accuracy for area/volume-fraction endpoints. |
| Touching / overlapping nuclei needing per-cell counts | **DL (StarDist, Cellpose)** | Connected components merge contacts; star-convex/flow models separate them. |
| Dense 3D tissue (every voxel is a cell) | **DL (PlantSeg, Cellpose 3D)** | Boundary-learning + watershed handles packing a global threshold cannot. |
| Low SNR / faint puncta / uneven background | **Li threshold or DL**, context-dependent | Li's cross-entropy handles skewed histograms; Cellpose3 restoration helps most. fluorostats' rolling-ball + Li is competitive if background is smooth. |
| No GPU, no training data, need determinism & audit trail | **fluorostats** | Zero training, reproducible bit-for-bit, seconds-per-stack, every step interpretable and cite-able. |
| Volume fraction / total-intensity / coverage quantification | **fluorostats (fit for purpose)** | These endpoints don't require instance separation; the added DL machinery is unnecessary complexity. |

**Framing for the paper:** fluorostats is not competing to be the best *instance*
segmenter. Its claim is that for *quantitative signal endpoints* (volume fraction,
integrated intensity, object counts of well-separated objects) a classical,
deterministic, training-free pipeline gives results equivalent to DL while being
faster, GPU-free, and fully reproducible/interpretable. Be explicit that DL wins on
touching/overlapping cells and dense tissue instance counts.

---

## Proposed benchmarks

**Strongest benchmark design (lead with this):**

*Agreement study on the Cell Tracking Challenge 3D fluorescence datasets.* Take the
public annotated 3D confocal stacks (Fluo-C3DL-MDA231, Fluo-C3DH-A549, and one dense
set such as Fluo-N3DL-TRIC). Run four pipelines on the identical voxels:
(1) fluorostats (Gaussian + rolling-ball + Otsu/Li + size filter),
(2) Otsu-in-Fiji (3D, as the classical-tool control),
(3) Cellpose 3D (pretrained `cyto`/`nuclei`),
(4) StarDist-3D.
Report, against the challenge SEG ground truth:
- **Foreground agreement:** voxel-wise Dice/IoU of each method vs. ground-truth
  foreground (tests whether fluorostats "catches the same signal").
- **Volume fraction:** signal-positive voxels / tissue voxels per method; Bland–Altman
  vs. ground truth (tests the actual fluorostats endpoint).
- **Instance metric (SEG / mean Jaccard):** where fluorostats will predictably lag on
  dense sets — report it honestly to delimit scope.
- **Runtime & hardware:** wall-clock per stack, CPU-only vs. GPU, to quantify the
  efficiency/reproducibility advantage.

*Hypothesis to state up front:* fluorostats matches DL on **foreground Dice and
volume fraction** for well-separated / moderate-density stacks (within CI), and lags
only on **per-instance SEG** for dense tissue — cleanly separating "signal
quantification" (fluorostats' job) from "instance counting" (DL's job).

**Secondary 2D benchmark:** DSB2018 / BBBC038 fluorescence subset — fluorostats vs.
StarDist vs. Cellpose vs. Otsu, reporting IoU and count agreement. Cheap to run,
widely recognized, lets reviewers place fluorostats on a familiar leaderboard axis.

**Public datasets to use:** Cell Tracking Challenge (celltrackingchallenge.net, 3D
fluorescence with SEG ground truth); BBBC038 / DSB2018 (2D nuclei); other BBBC sets
(e.g., BBBC006 U2OS, BBBC008) for 2D synthetic/real controls.

---

## Flags / caveats on citations

- **Citation counts are approximate** (from indexer "highly cited" signals), not exact
  live counts — verify before final submission if exact numbers are needed.
- **Cellpose3 (2025)** volume/page numbers were not fully confirmed from the search
  snippet (DOI 10.1038/s41592-025-02595-5 is confirmed); confirm pagination at proof.
- **StarDist-3D** specifically is Weigert, Schmidt et al. (2020), *Star-convex
  Polyhedra for 3D Object Detection and Segmentation in Microscopy*, WACV 2020 — cite
  this rather than the 2018 2D paper when referring to the 3D variant. (Located via
  StarDist project references; verify DOI 10.1109/WACV45572.2020.9093435 at proof.)
- All 14 primary DOIs above were surfaced from their canonical publisher pages and are
  considered verified; none are invented.
