# Community Benchmark Challenges for fluorostats

Purpose: identify established, citable benchmarks with public leaderboards where a
foreground/volume-fraction or object-count tool can report a comparable, honest number.

All leaderboard scores below were extracted **directly from the official Cell Tracking
Challenge (CTC) results spreadsheets** downloaded on 2026-07-08 (verified, not scraped
from prose):
- CSB (segmentation): http://public.celltrackingchallenge.net/documents/CellSegmentationBenchmark.xlsx
- CTB (tracking): http://public.celltrackingchallenge.net/documents/CellTrackingBenchmark.xlsx

Leaderboard snapshot images referenced by the site are dated **2025-08-15**.

---

## 1. Cell Tracking Challenge (CTC)

- Site: https://celltrackingchallenge.net/
- Founding paper: Ulman et al., *An objective comparison of cell-tracking algorithms*,
  **Nature Methods 14, 1141–1152 (2017)**. DOI: 10.1038/nmeth.4473
- 10-year update: Maška et al., *The Cell Tracking Challenge: 10 years of objective
  benchmarking*, **Nature Methods 20, 1010–1020 (2023)**. DOI: 10.1038/s41592-023-01879-y
  (open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC10333123/)
- Two benchmarks: **CSB** (Cell Segmentation Benchmark) and **CTB** (Cell Tracking Benchmark).

### 1a. 3D fluorescence datasets (download URLs + sizes)

All URLs verified live on 2026-07-08. Pattern:
`https://data.celltrackingchallenge.net/{training-datasets|test-datasets}/{NAME}.zip`
Training sets ship with gold+silver truth; test sets are unlabeled (submit to organizers).

| Dataset | Content | Voxel size (µm) | Δt | Train zip | Test zip |
|---|---|---|---|---|---|
| **Fluo-C3DH-A549** | GFP-actin A549 lung cancer | 0.126×0.126×1.0 | 2 min | 244 MB | 294 MB |
| **Fluo-C3DH-H157** | GFP H157 lung cancer | 0.126×0.126×0.5 | 2 min | 7.0 GB | 7.1 GB |
| **Fluo-C3DL-MDA231** | GFP MDA231 breast carcinoma | 1.242×1.242×6.0 | 80 min | 182 MB | 179 MB |
| **Fluo-N3DH-CE** | *C. elegans* embryo nuclei | 0.09×0.09×1.0 | 1 min | 3.1 GB | 1.7 GB |
| **Fluo-N3DH-CHO** | GFP-PCNA CHO nuclei | 0.202×0.202×1.0 | 9.5 min | 98 MB | 105 MB |
| **Fluo-N3DL-DRO** | *Drosophila* embryo | 0.406×0.406×2.03 | 30 s | 5.8 GB | 5.9 GB |
| **Fluo-N3DL-TRIC** | *Tribolium* embryo (projected) | NA (cartographic) | 1.5 min | 20.6 GB | 19.9 GB |
| **Fluo-N3DL-TRIF** | *Tribolium* embryo (full) | 0.38×0.38×0.38 | 1.5 min | **320 GB** | **467 GB** |
| **Fluo-C3DH-A549-SIM** | *simulated* GFP-actin A549 | 0.126×0.126×1.0 | 20 s | 314 MB | 327 MB |
| **Fluo-N3DH-SIM+** | *simulated* HL60 nuclei | 0.125×0.125×0.200 | 29 min | 3.1 GB | 5.9 GB |

**Downloadable now, small enough for a laptop (recommended entry points):**
Fluo-N3DH-CHO (98 MB), Fluo-C3DL-MDA231 (182 MB), Fluo-C3DH-A549 (244 MB),
Fluo-C3DH-A549-SIM (314 MB). TRIF (320/467 GB) is effectively out of reach.

### 1b. Metric definitions (verified from Maška et al. 2023 + SEG.pdf)

- **SEG** — "evaluates the average intersection over union overlap (IoU) ... between the
  reference cell instance masks and the segmentation masks." Formally the Jaccard index
  `J(R,S) = |R∩S| / |R∪S|`, where a reference object `R` is matched to computed object `S`
  iff `|R∩S| > 0.5·|R|` (at most one match per reference); unmatched references score 0.
  SEG is the mean `J` over all reference objects.
- **DET** — detection accuracy `DET = 1 − min(AOGM-D, AOGM-D0)/AOGM-D0`, a normalized
  node-editing cost (AOGM-D0 = cost of building the reference node set from scratch).
- **TRA** — tracking accuracy `TRA = 1 − min(AOGM, AOGM0)/AOGM0` over the full acyclic
  oriented graph (nodes + edges), weighted by human-curation effort.
- **Overall:** `OP_CSB = 0.5·(SEG + DET)`, `OP_CTB = 0.5·(SEG + TRA)`.
- All measures ∈ [0,1], higher is better. Scores are the mean over videos 01 and 02.

### 1c. Top leaderboard scores — 3D fluorescence (numbers to beat)

**Top SEG (Cell Segmentation Benchmark)** — mean of video 01/02, from CSB xlsx:

| Dataset | Best SEG | Top method |
|---|---|---|
| Fluo-C3DH-A549 | **0.9083** | DKFZ-GE |
| Fluo-C3DH-H157 | **0.8919** | KTH-SE (1*) |
| Fluo-C3DL-MDA231 | **0.7096** | KIT-GE (3) |
| Fluo-N3DH-CE | **0.7590** | MU-CZ (2*) |
| Fluo-N3DH-CHO | **0.9248** | CALT-US (*) |
| Fluo-N3DL-DRO | **0.7604** | CZB-US |
| Fluo-N3DL-TRIC | **0.8208** | KIT-GE (2) |
| Fluo-N3DL-TRIF | **0.7935** | MPI-GE (CBG) (2) |
| Fluo-C3DH-A549-SIM | **0.9549** | DKFZ-GE |
| Fluo-N3DH-SIM+ | **0.9062** | DKFZ-GE |

**Top TRA (Cell Tracking Benchmark)** — mean of video 01/02, from CTB xlsx:

| Dataset | Best TRA | Top method |
|---|---|---|
| Fluo-C3DH-A549 | **1.0000** | DREX-US |
| Fluo-C3DH-H157 | **0.9872** | KTH-SE (1) |
| Fluo-C3DL-MDA231 | **0.8845** | KIT-GE (3) |
| Fluo-N3DH-CE | **0.9937** | AMOLF-NL |
| Fluo-N3DH-CHO | **0.9532** | KTH-SE (1) |
| Fluo-N3DL-DRO | **0.8019** | CZB-US |
| Fluo-N3DL-TRIC | **0.9517** | MPI-GE (CBG) (3) |
| Fluo-N3DL-TRIF | **0.9545** | MPI-GE (CBG) (3) |
| Fluo-N3DH-SIM+ | **0.9740** | BGU-IL (5) |

---

## 2. Broad Bioimage Benchmark Collection (BBBC)

- Site: https://bbbc.broadinstitute.org/
- Reference: Ljosa, Sokolnicki & Carpenter, *Annotated high-throughput microscopy image
  sets for validation*, **Nature Methods 9, 637 (2012)**. DOI: 10.1038/nmeth.2083

| Set | Title | Type | Ground truth | Notes |
|---|---|---|---|---|
| **BBBC024** | 3D HL60 Cell Line (synthetic) | 3D, synthetic | FG/BG + counts | 4 clustering levels (0/25/50/75%) × high/low SNR × 30 imgs; **20 nuclei/image** exactly |
| **BBBC027** | 3D Colon Tissue (synthetic) | 3D, synthetic | FG/BG + counts | high & low SNR, 30 imgs each; clustered nuclei |
| **BBBC032** | Mouse embryo blastocyst | 3D, real fluorescence | FG/BG + counts | only 1 field (4 files) |
| **BBBC034** | Induced pluripotent stem cells | 3D, real (BF + fluor) | FG/BG | 3 fields |
| **BBBC038** | Kaggle 2018 Data Science Bowl | 2D nuclei | per-nucleus masks | mixed modalities; competition metric = mean AP over IoU thresholds |
| **BBBC039** | Nuclei of U2OS cells | 2D fluorescence | FG/BG, outlines, counts, labels | 200 fields |

**No official published accuracy baselines are posted on the BBBC set pages themselves**
(BBBC024/027/038/039 all state "no baseline" — flagged as unverified for any specific score).
BBBC038 (2018 DSB) does have a Kaggle leaderboard (private-LB **mean Average Precision**
averaged over IoU thresholds 0.5→0.95 step 0.05); top ~0.63 mAP — cite the Kaggle
competition page directly if used, not the BBBC page.

### Most relevant to fluorostats
**BBBC024 and BBBC027** are 3D synthetic with **FG/BG masks and exact counts** — directly
usable for a **volume-fraction / foreground-fraction** metric and an **object-count** metric
without needing instance segmentation. BBBC024's fixed "20 nuclei/image" makes counting
error trivially reportable.

---

## 3. How fluorostats can report a comparable, honest number

**Key honesty caveat:** fluorostats is **not an instance segmenter** — it does not assign
per-object identities, so the CTC SEG/DET/TRA and BBBC038 mean-AP metrics (which require
per-instance matching) **cannot be reported directly**. State this explicitly.

Honest, comparable numbers fluorostats *can* report:

1. **Foreground / volume fraction (Jaccard on the binary FG mask).** On BBBC024 / BBBC027
   (which ship FG/BG truth) compute `J = |FG_pred ∩ FG_gt| / |FG_pred ∪ FG_gt|` and Dice.
   This is the *semantic* (not instance) analogue of SEG — label it as such and never
   present it as the CTC SEG leaderboard number.
2. **Object count error.** On BBBC024 (20/image), BBBC027, BBBC039, and CTC training
   volumes (which ship instance masks → derive counts), report count MAE / % error and
   correlation vs. truth. This is the metric most native to fluorostats.
3. **Total-volume / volume-fraction accuracy.** Sum of GT instance-mask voxels gives a
   reference total foreground volume; report % error of fluorostats' volume estimate.
   No competitor leaderboard exists for this, so present as a self-contained accuracy claim,
   not a ranking.

Recommendation: use CTC **training** volumes (labels are public) to compute semantic-Jaccard
and count metrics locally; do **not** claim a CTC leaderboard rank, since the ranked metrics
are instance-based and the test truth is withheld.

---

## Recommendation: which to enter / report first

1. **BBBC024** (3D HL60 synthetic, small, FG/BG + fixed 20-count truth) — cleanest first
   target for foreground-Jaccard + count-error, fully local, no submission needed.
2. **BBBC027** (3D colon synthetic) — second, same metric family, harder clustering.
3. **CTC Fluo-N3DH-CHO** (98 MB) and **Fluo-C3DH-A549** (244 MB) training sets — compute
   semantic-Jaccard + count error against their public instance truth; cite the top SEG/TRA
   in §1c as context for where instance methods land, while clearly stating fluorostats
   measures a different (semantic/volumetric) quantity.

Flagged unverified: no BBBC page publishes a numeric accuracy baseline; the BBBC038 ~0.63
mAP figure is from the Kaggle competition, not confirmed against the source in this pass.
