# Nuclei Segmentation / Counting Datasets — fluorostats Benchmark B2

Public, citable datasets with ground-truth nuclei annotations (fluorescent / DAPI nuclei),
used to benchmark fluorostats connected-component nuclei counting against StarDist, Cellpose,
and manual counts, stratified by density.

All URLs below were verified to resolve (HTTP 200 / 302-to-asset) on 2026-07-08.

---

## 1. BBBC039 — Nuclei of U2OS cells in a chemical screen *(recommended primary)*

- **ID:** BBBC039v1 (Broad Bioimage Benchmark Collection)
- **Citation:** Caicedo, J.C., Roth, J., Goodman, A., Becker, T., Karhohs, K.W., Broisin, M.,
  Molnar, C., McQuin, C., Singh, S., Theis, F.J., Carpenter, A.E. (2019). *Evaluation of deep
  learning strategies for nucleus segmentation in fluorescence images.* **Cytometry Part A**
  95(9): 952–965. DOI: 10.1002/cyto.a.23863. (Cite image set as "BBBC039v1, from the Broad
  Bioimage Benchmark Collection.")
- **Download URLs (verified):**
  - Images: https://data.broadinstitute.org/bbbc/BBBC039/images.zip — **77.9 MB** (verified content-length 77,915,748)
  - Masks: https://data.broadinstitute.org/bbbc/BBBC039/masks.zip — **2.75 MB** (verified content-length 2,753,811)
  - Metadata (train/val/test split): https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip — ~18 KB
- **License:** CC0 (public domain).
- **Format / dimensionality:** TIFF, 520×696 px, 16-bit; **2D**, single DAPI channel.
- **# images:** 200 fields of view.
- **Instance masks + counts:** Yes — per-nucleus instance masks (indexed PNG). ~23,000 manually
  annotated nuclei total. Per-image counts derivable directly from mask labels.
- **Density range:** Wide — varied nuclear phenotypes across a chemical screen; includes sparse
  and dense/clustered fields. Ideal for density stratification.
- **Feeds:** B2 (nuclei count vs StarDist/Cellpose/manual). Defined train/val/test split enables
  fair method comparison. **Best first dataset**: single-channel fluorescence, clean instance
  masks, exact counts, permissive license, manageable size.

---

## 2. BBBC038 / 2018 Data Science Bowl — diverse nuclei

- **ID:** BBBC038v1 (Broad Bioimage Benchmark Collection); Kaggle "2018 Data Science Bowl."
- **Citation:** Caicedo, J.C., Goodman, A., Karhohs, K.W., Cimini, B.A., Ackerman, J., Haghighi, S.,
  Heng, C., Becker, T., Doan, M., McQuin, C., Rohban, M., Singh, S., Carpenter, A.E. (2019).
  *Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl.* **Nature Methods**
  16: 1247–1253. DOI: 10.1038/s41592-019-0612-7.
- **Download URLs (verified via bbbc.broadinstitute.org/BBBC038):**
  - stage1_train.zip — 82.9 MB (images + per-nucleus masks)
  - stage1_test.zip — 9.5 MB
  - stage2_test_final.zip — 289.7 MB
  - metadata.xlsx — ~20 KB
  - Also mirrored on Kaggle: https://www.kaggle.com/c/data-science-bowl-2018/data
- **License:** CC0 (public domain).
- **Format / dimensionality:** PNG; **2D**. Mixed modalities — includes fluorescent (DAPI) nuclei
  plus brightfield/histology; filter to fluorescence subset for B2.
- **# images:** ~670 in stage1_train (tens of thousands of nuclei total).
- **Instance masks + counts:** Yes — each nucleus as a separate non-overlapping PNG mask; counts =
  number of mask files per image.
- **Density range:** Very wide across many experiments/magnifications; strong for stratification.
- **Feeds:** B2. **Caveat:** heterogeneous modalities — restrict to fluorescent nuclei subset for a
  clean DAPI benchmark. Also the basis for the StarDist DSB2018 subset (#3).

---

## 3. StarDist DSB2018 subset — fluorescence nuclei, ready-to-use split

- **ID:** dsb2018 (StarDist release asset); a curated fluorescence subset of BBBC038 stage1_train.
- **Citation:** Schmidt, U., Weigert, M., Broaddus, C., Myers, G. (2018). *Cell Detection with
  Star-Convex Polygons.* **MICCAI 2018**, LNCS 11071: 265–273. DOI: 10.1007/978-3-030-00934-2_30.
  (Underlying data: cite BBBC038 / Caicedo et al. 2019 Nature Methods above.)
- **Download URL (verified):** https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip
  — **26.9 MB** (verified: 302 → release asset, content-length 26,975,232).
- **License:** Underlying images CC0 (BBBC038). StarDist code BSD-3-Clause.
- **Format / dimensionality:** PNG/TIFF images + label masks; **2D**, single-channel fluorescence.
- **# images:** train/test folders (`train/images`, `train/masks`, `test/images`, `test/masks`);
  ~447 train / ~50 test (fluorescence subset).
- **Instance masks + counts:** Yes — indexed label masks; counts from unique labels.
- **Density range:** Sparse to dense clustered nuclei.
- **Feeds:** B2. **Best for a direct StarDist head-to-head** since it is the exact split the StarDist
  fluorescence models were trained/evaluated on. Lets you compare fluorostats CC counting against
  StarDist on its home turf.

---

## 4. Cellpose dataset (nuclei + specialized/generalized subsets)

- **ID:** Cellpose training/test dataset (cellpose.org).
- **Citation:** Stringer, C., Wang, T., Michaelos, M., Pachitariu, M. (2021). *Cellpose: a generalist
  algorithm for cellular segmentation.* **Nature Methods** 18: 100–106. DOI: 10.1038/s41592-020-01018-x.
- **Download URL (verified live, gated):** https://www.cellpose.org/dataset — requires accepting
  terms + institutional email to obtain train/test archives (no direct anonymous link).
- **License:** CC-BY-NC (HHMI): non-commercial, educational, research, personal use only; commercial
  use/redistribution prohibited. **Verify license compatibility before redistribution.**
- **Format / dimensionality:** PNG images + `_masks` label files; **2D**. Includes a dedicated
  **nuclei** subset alongside cytoplasm/generalized subsets.
- **# images / objects:** >70,000 segmented objects across the full corpus; nuclei subset is a
  labeled portion (hundreds of images).
- **Instance masks + counts:** Yes — full instance label masks; counts from labels.
- **Density range:** Highly varied; nuclei subset spans sparse to crowded.
- **Feeds:** B2 (Cellpose head-to-head). **Caveats:** email/registration gate (not anonymously
  downloadable); non-commercial license; mixed cell types — use the nuclei subset for DAPI benchmark.

---

## 5. BBBC024 — 3D synthetic HL60 nuclei (controlled clustering)

- **ID:** BBBC024v1 (Broad Bioimage Benchmark Collection).
- **Citation:** Svoboda, D., Kozubek, M., Stejskal, S. (2009). *Generation of Digital Phantoms of
  Cell Nuclei and Simulation of Image Formation in 3D Image Cytometry.* **Cytometry Part A**
  75A(6): 494–509. DOI: 10.1002/cyto.a.20714.
- **Download URLs:** data.broadinstitute.org/bbbc/BBBC024/ — files named
  `BBBC024_v1_c[00|25|50|75]_[lowSNR|highSNR]_images[_TIFF].zip` plus matching foreground/mask zips.
- **License:** CC BY 3.0 (attribution).
- **Format / dimensionality:** ICS + TIFF; **3D** synthetic confocal.
- **# images:** 4 clustering levels × 30 images = 120 (×2 SNR variants).
- **Instance masks + counts:** Yes — **exactly 20 nuclei per image** (known ground-truth count);
  labeled 16-bit instance + foreground masks.
- **Density range:** Controlled via clustering probability (0%, 25%, 50%, 75%) — a clean synthetic
  density/overlap axis.
- **Feeds:** B2 **3D** counting under controlled clustering. **Caveat:** synthetic (not real
  microscopy); best as a controlled-difficulty complement, not a realism benchmark.

---

## 6. BlastoSPIM 1.0 / 2.0 — 3D real nuclei, early mouse embryo

- **ID:** BlastoSPIM 1.0 and 2.0 (Flatiron Institute / Princeton).
- **Citation:** Nunley, H., Shao, B., et al. (2024). *Nuclear instance segmentation and tracking for
  preimplantation mouse embryos.* **Development** 151(21): dev202817.
  DOI: 10.1242/dev.202817. (bioRxiv preprint: 10.1101/2023.03.14.532646.)
- **Download URL (verified live):** https://blastospim.flatironinstitute.org/ (data via the site's
  "Preview and download data" → series page; hosted, not a single anonymous zip).
- **License:** See site terms — confirm before redistribution (flag: not explicitly stated on
  landing page).
- **Format / dimensionality:** **3D** light-sheet (SPIM); xy 0.208 µm, z 2.0 µm.
- **# images / nuclei:** 573 (v1) + 80 (v2) annotated 3D volumes; 18,336 nuclei total, contoured
  per z-slice.
- **Instance masks + counts:** Yes — full 3D instance segmentation; per-embryo counts are the
  developmental stage (8 → >100 nuclei).
- **Density range:** Native developmental gradient — sparse 8-cell to crowded >100-nucleus blastocyst.
  Excellent real-data 3D density stratification.
- **Feeds:** B2 **3D** real-microscopy counting; complements synthetic BBBC024. **Caveat:** verify
  exact license/terms before redistribution.

---

## 7. BBBC030 / other BBBC nuclei (optional extras)

- BBBC datasets index: https://bbbc.broadinstitute.org/image_sets — additional DAPI/nuclei sets
  (e.g., synthetic and real) with masks under CC0/CC-BY are available if more density regimes or
  modalities are needed. Evaluate case-by-case; not yet individually verified here.

---

## Summary table

| # | Dataset | Dim | Real/Synth | # img | Masks | Counts | License | Density stratification |
|---|---------|-----|-----------|-------|-------|--------|---------|------------------------|
| 1 | BBBC039 (U2OS) | 2D | Real | 200 | Yes | Yes | CC0 | Wide (screen) |
| 2 | BBBC038 / DSB2018 | 2D | Real (mixed) | ~670 | Yes | Yes | CC0 | Very wide |
| 3 | StarDist dsb2018 | 2D | Real (fluor) | ~500 | Yes | Yes | CC0 / BSD | Sparse–dense |
| 4 | Cellpose nuclei | 2D | Real | 100s | Yes | Yes | CC-BY-NC (gated) | Wide |
| 5 | BBBC024 | 3D | Synthetic | 120 | Yes | 20/img | CC BY 3.0 | Controlled 0–75% cluster |
| 6 | BlastoSPIM 1.0/2.0 | 3D | Real | 653 | Yes | Stage | Site terms | 8 → >100 nuclei |
| 7 | Other BBBC nuclei | 2D/3D | Mixed | — | Varies | Varies | CC0/CC-BY | Extra regimes |

**Unverified / flagged:** BlastoSPIM license (confirm site terms); Cellpose requires
email-gated download and is non-commercial; BBBC030 and "other BBBC" entries not individually
fetched. All other download URLs verified to resolve on 2026-07-08.
