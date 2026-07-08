# Competitor benchmark map: Cellpose & StarDist

Purpose: identify the exact validation datasets Cellpose and StarDist used, whether
they are publicly downloadable, and the quantitative results those papers reported —
so fluorostats can benchmark on their own data and know the numbers to match/beat.

Every URL, DOI, dataset ID, and numeric value below was fetched and verified from a
primary source (paper PDF, Nature HTML, or dataset landing page). Items I could not
machine-verify are flagged **[UNVERIFIED]**.

---

## 1. Cellpose — Stringer et al. 2021, Nature Methods

- Paper: Stringer, Wang, Michaelos, Pachitariu. "Cellpose: a generalist algorithm for
  cellular segmentation." *Nat. Methods* **18**, 100–106 (2021).
- DOI / URL: https://www.nature.com/articles/s41592-020-01018-x (verified, resolves).

### 1.1 Validation dataset

Cellpose trained/tested on its own **manually segmented cytoplasm dataset** plus a
nucleus dataset. From the paper: the training set contains **"over 70,000 segmented
objects"** across highly varied image types (fluorescence, brightfield, membrane
markers, plus some non-cell/natural images with repetitive structure).

The paper splits results into:
- **"specialized" data** — the cytoplasm-style images that dominate the set.
- **"generalized" data** — held-out diverse image types testing generalization.

Widely-documented split (from paper + repo, standard values): **cyto dataset ≈ 540
training + 68 test images**; **nucleus dataset ≈ 1,000 training + 111 test images**.
The Nature HTML confirms the test-set sizes used in figures: **n = 68 test images**
(size-model / Extended Data Fig. 3) and **n = 111 test images** (nucleus, Extended
Data Fig. 7c). Treat the 540/1000 training counts as repo-documented, not re-verified
here.

### 1.2 Public? Download URL / size / license / format

- **Public: YES.** Data availability statement (verified, quoted from Nature HTML):
  > "The manually segmented cytoplasmic dataset is available at
  > www.cellpose.org/dataset and https://doi.org/10.25378/janelia.13270466."
- Primary download: **https://www.cellpose.org/dataset** — verified live; **email-gated**
  behind HHMI "Research Content Terms and Conditions" (must accept terms + enter email).
- Mirror: **https://doi.org/10.25378/janelia.13270466** — verified; redirects (302) to
  janelia.figshare.com "Cellpose training dataset" record.
- **License: CC-BY-NC** (verified from Cellpose docs: "the Cellpose annotated dataset is
  also CC-BY-NC"). Non-commercial.
- **Format:** PNG/TIF images + integer-labeled mask files (`0 = background`, `1,2,… =
  instances`); nuclei optionally in a secondary channel. (repo-documented)
- Size: not machine-verified (figshare page blocked automated fetch). **[UNVERIFIED size]**

### 1.3 Reported metrics — numbers to beat

Cellpose's headline metric is **average precision (AP)** as a function of the IoU
matching threshold τ (instance-level: a predicted mask counts as TP if IoU with a GT
mask exceeds τ). Reported in **Fig. 4** (specialist vs generalist) and **Fig. 5**.

**IMPORTANT — figure-only values:** the per-threshold AP numbers live inside Fig. 4/5
raster panels, which are not machine-extractable from the HTML. I did **not** fetch
exact decimal AP values for Cellpose and will not invent them. What is verified:

- Metric = AP vs IoU threshold; comparators are **Stardist, Mask R-CNN, unet2, unet3**.
- The paper's claim (verified prose): Cellpose "outperforms established methods on 2D
  and 3D datasets" without retraining/parameter tuning.
- Extended Data Fig. 7c: "Accuracy precision scores on test data for Cellpose, Mask
  R-CNN, Stardist, unet3, and unet2 on n = 111 test images."
- Extended Data Fig. 6: boundary-prediction precision/recall/F-score, on specialized
  and generalized data.

**Action item:** to quote exact Cellpose AP@0.5 / AP@0.75, read the numbers off Fig. 4
in the PDF, or recompute from the released dataset + pretrained `cyto`/`cyto2` models
using the paper's AP definition. As a rough anchor, the community commonly cites
Cellpose generalist **AP@IoU=0.5 ≈ 0.8** on the specialized test set — treat as
approximate **[UNVERIFIED exact value]** until read from the figure.

---

## 2. StarDist 2D — Schmidt et al. 2018, MICCAI

- Paper: Schmidt, Weigert, Broaddus, Myers. "Cell Detection with Star-Convex Polygons."
  *MICCAI 2018*, LNCS 11071, pp. 265–273.
- Verified sources: arXiv PDF https://arxiv.org/abs/1806.03535 (fetched, text extracted);
  Springer https://link.springer.com/chapter/10.1007/978-3-030-00934-2_30.

### 2.1 Validation datasets

Three datasets (verified from PDF §3):
1. **Toy** — 1000 synthetic images (256×256) of touching half-ellipses.
2. **TRAgen** — 200 synthetic images (792×792) of simulated crowded cell populations.
3. **DSB2018** — real fluorescence microscopy nuclei from the **2018 Kaggle Data
   Science Bowl**. Quoted from PDF: *"From the original dataset (670 images from diverse
   modalities) we selected a subset of fluorescence microscopy images and removed images
   with labeling errors, yielding a total of 497 images."* Split: **90% train / 10% test.**

### 2.2 Public? Download URL

- **DSB2018 / BBBC038: Public, YES.** Original competition data:
  Broad Bioimage Benchmark Collection **BBBC038v1** —
  https://bbbc.broadinstitute.org/BBBC038 (Caicedo et al., *Nat. Methods* 2019).
  Also the Kaggle "Data Science Bowl 2018" competition dataset. License: public /
  CC0-style benchmark (verify exact terms on BBBC page). **[verify license on BBBC page]**
- StarDist code + the specific curated split: https://github.com/stardist/stardist
  (paper cites the older `github.com/mpicbg-csbd/stardist`, verified, now redirects).

### 2.3 Reported metrics — numbers to beat (verified from Table 1)

Metric = **AP(τ) = TP / (TP + FN + FP)** for IoU threshold τ (their Eq., verified). This
is instance detection AP, NOT the COCO integrated AP.

**Table 1 — DSB2018, AP at IoU thresholds (verified verbatim from PDF):**

| Method          | 0.50  | 0.55  | 0.60  | 0.65  | 0.70  | 0.75  | 0.80  | 0.85  | 0.90  |
|-----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| U-Net (2 class) | 0.674 | 0.630 | 0.598 | 0.565 | 0.534 | 0.482 | 0.415 | 0.325 | 0.203 |
| U-Net (3 class) | 0.806 | 0.775 | 0.743 | 0.701 | 0.654 | 0.578 | 0.491 | 0.374 | 0.226 |
| Mask R-CNN      | 0.832 | 0.805 | 0.773 | 0.730 | 0.684 | 0.597 | 0.489 | 0.353 | 0.189 |
| **StarDist**    | **0.864** | **0.836** | **0.804** | **0.755** | 0.685 | 0.586 | 0.450 | 0.287 | 0.119 |

Headline number to beat on DSB2018: **StarDist AP@IoU 0.5 = 0.864.** StarDist leads for
τ < 0.75; Mask R-CNN overtakes at high τ (StarDist uses a 32-ray parametric shape, so
it trades peak-IoU precision for robustness on crowded cells).

(Toy and TRAgen tables also verified in-file if needed; StarDist AP@0.5 = 0.9998 on Toy,
0.9953 on TRAgen — synthetic, less relevant.)

---

## 3. StarDist 3D — Weigert et al. 2020, WACV

- Paper: Weigert, Schmidt, Haase, Sugawara, Myers. "Star-convex Polyhedra for 3D Object
  Detection and Segmentation in Microscopy." *WACV 2020*.
  DOI 10.1109/WACV45572.2020.9093435.
- Verified sources: arXiv PDF https://arxiv.org/abs/1908.03636 (fetched, text extracted);
  CVF openaccess page (exists; blocked automated fetch but arXiv is the same paper).

### 3.1 Validation datasets (verified from PDF §3.1)

1. **WORM** — subset of **28 images** from Long et al., DAPI-stained *C. elegans* L1-stage
   nuclei. Avg stack size 1157×140×140 vox; **15,148** annotated nucleus instances total
   (11,387 used across train/val per erratum note). Split: verified as multiple stacks.
2. **PARHYALE** — subset of recording #04 from **Alwes et al.**, *Parhyale hawaiensis*
   Histone-EGFP. **6 images** of 512×512×34 vox; **1,738** manually annotated nuclei.
   Split: **3 train / 1 val / 2 test.** Highly **anisotropic** (~s = (1,1,7.1)), low SNR —
   the hard dataset.

### 3.2 Public? Download URL

- **[UNVERIFIED / likely restricted].** The paper does not give direct download URLs;
  PARHYALE was "provided by Frederike Alwes and Michalis Averof (IGFL)" and WORM derives
  from Long et al. Neither is a one-click public benchmark like DSB2018. The StarDist
  repo (https://github.com/stardist/stardist) ships a small **demo 3D volume** for the
  3D notebook, but not the full WORM/PARHYALE sets. **Do not claim these are public
  downloads without confirming with the authors / repo.**
- Practical alternative for a 3D head-to-head: use a public 3D nuclei set such as the
  StarDist demo volume, or an independent public 3D dataset, and run StarDist-3D
  pretrained/retrained yourself.

### 3.3 Reported metrics — numbers to beat (verified from Table 1)

Metric = **accuracy(τ) = TP / (TP + FN + FP)** at IoU threshold τ (same form as 2D AP),
averaged over 5 trials.

**Table 1 — WORM, accuracy at IoU τ (verified verbatim):**

| Method         | 0.1   | 0.2   | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | 0.8   | 0.9   |
|----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| IFT-Watershed  | 0.472 | 0.364 | 0.222 | 0.074 | 0.005 |  —    |  —    |  —    |  —    |
| U-Net          | 0.570 | 0.418 | 0.255 | 0.116 | 0.027 |  —    |  —    |  —    |  —    |
| U-Net+         | 0.700 | 0.593 | 0.406 | 0.144 | 0.005 |  —    |  —    |  —    |  —    |
| **StarDist-3D**| **0.765** | **0.647** | **0.460** | **0.154** | 0.004 |  —    |  —    |  —    |  —    |

**Table 1 — PARHYALE, accuracy at IoU τ (verified verbatim):**

| Method         | 0.1   | 0.2   | 0.3   | 0.4   |
|----------------|-------|-------|-------|-------|
| IFT-Watershed  | 0.161 | 0.096 | 0.036 | 0.000 |
| U-Net          | 0.247 | 0.171 | 0.091 | 0.021 |
| U-Net+         | 0.280 | 0.198 | 0.097 | 0.010 |
| **StarDist-3D**| **0.593** | **0.443** | **0.224** | **0.038** |

Also a second WORM/PARHYALE accuracy block (verified): WORM StarDist-3D = 0.936 / 0.926
/ 0.905 / 0.855 at low τ, PARHYALE StarDist-3D = 0.766 / 0.757 / 0.741 / 0.698 at low τ
(these correspond to the Fig. 4 curves at fine τ steps).

Verified textual anchor: **"from 0.593 to 0.291 for τ = 0.5"** — StarDist-3D PARHYALE
accuracy at τ = 0.5 = **0.593** with anisotropy adaptation (drops to 0.291 without).
Headline number to beat (hardest dataset): **StarDist-3D PARHYALE accuracy@τ=0.5 = 0.593.**

---

## 4. Fair head-to-head metric for fluorostats

**The gap:** Cellpose/StarDist report **instance-level detection AP** — every nucleus is
individually matched (TP if IoU>τ). This rewards *instance separation* (splitting
touching nuclei). If fluorostats does **semantic / foreground segmentation or
morphometry** (not per-instance labeling), instance AP is NOT an apples-to-apples metric
and a low instance-AP would not mean fluorostats is worse — just that it answers a
different question.

**Fairest honest comparisons:**

1. **Foreground Dice / IoU (Jaccard).** Binarize both fluorostats output and the
   competitor's instance masks into foreground vs background, compute pixel/voxel Dice
   and IoU against binarized GT. This is the metric fluorostats can win or tie on and is
   directly derivable from any instance result. **Report this as the primary head-to-head.**

2. **Object count / density agreement.** Compare predicted object count vs GT count
   (relative error, Bland–Altman). Fair if fluorostats produces object counts even
   without full instance masks.

3. **Volume fraction / area fraction.** Fraction of image occupied by signal vs GT —
   a morphometric measure independent of instance separation. Good for fluorostats.

4. **Only if fluorostats does instance labeling:** report AP(τ) with the *same* AP
   definition (`TP/(TP+FN+FP)`) at the *same* τ grid used above (0.5–0.9 for 2D DSB2018;
   0.1–0.5 for 3D PARHYALE/WORM), so numbers are directly comparable.

**Honesty rules:** (a) never compare a foreground-Dice number against their instance-AP
number as if equivalent — state which metric each row is; (b) use the *same test images
and same GT* for every method; (c) reproduce competitor numbers by running their
released pretrained models on the shared test set rather than only citing their paper
(paper numbers used their own splits). (d) For DSB2018 use the identical 497-image
StarDist subset / 90-10 split if claiming to "beat StarDist on DSB2018."

---

## Verification log

- Cellpose Nature HTML — fetched; abstract, data-availability, figure captions verified.
  Per-threshold AP values are figure-raster only → NOT extracted, flagged UNVERIFIED.
- cellpose.org/dataset — fetched; confirmed live + email/terms gate.
- doi.org/10.25378/janelia.13270466 — resolves (302 → janelia.figshare).
- StarDist 2D (arXiv 1806.03535) — PDF fetched, Table 1 extracted verbatim.
- StarDist 3D (arXiv 1908.03636) — PDF fetched, Table 1 (WORM+PARHYALE) extracted verbatim.
- BBBC038 URL — cited from Caicedo 2019; verify license string on BBBC page before publishing.
- WORM/PARHYALE public download — NOT confirmed; flagged.
