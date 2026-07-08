# 3D Fluorescence Microscopy Segmentation Datasets

Public, citable datasets with ground-truth masks to benchmark **fluorostats** (Otsu-based
volume-fraction / foreground-Dice) against Cellpose-3D, StarDist-3D, and Otsu-in-Fiji.
Feeds **benchmark B2: segmentation agreement**.

All download URLs below were verified (HTTP 200) on 2026-07-08. Sizes are from HTTP
`Content-Length` (CTC) or dataset pages (BBBC).

---

## Recommendation — download this first

**BBBC032 — Mouse embryo blastocyst cells** (real spinning-disk confocal, manually
annotated 3D masks, CC0 public domain, 1.14 GB). Real fluorescence data, permissive
license, human-annotated ground truth, and a manageable size. Direct dataset:
`https://data.broadinstitute.org/bbbc/BBBC032/BBBC032_v1_dataset.zip`
Ground truth: `https://data.broadinstitute.org/bbbc/BBBC032/BBBC032_v1_DatasetGroundTruth.tif`

If a *simulated* set with perfect masks is preferred as a first sanity check, use
**Fluo-N3DH-SIM+** (CTC) or **BBBC024** (small, foreground+count masks).

---

## Group A — Cell Tracking Challenge (CTC), 3D fluorescence

**Shared citations (cite BOTH):**
- Maška M, Ulman V, Svoboda D, et al. *A benchmark for comparison of cell tracking
  algorithms.* Bioinformatics 30(11):1609–1617, 2014. DOI: 10.1093/bioinformatics/btu080
- Ulman V, Maška M, Magnusson KEG, et al. *An objective comparison of cell-tracking
  algorithms.* Nature Methods 14(12):1141–1152, 2017. DOI: 10.1038/nmeth.4473

**License / usage:** Data are provided under the Cell Tracking Challenge terms
(CC BY 4.0 per the challenge's data policy); free for research use with the citations
above. Confirm the current terms on the download page before redistribution.
**Format:** 3D TIFF stacks (`t*.tif`), one file per time point. Ground truth ships as
gold (GT) and silver (ST) label TIFFs in `SEG` folders where available.
**Caveat for B2:** CTC ground truth is oriented to *tracking/instance* evaluation. For a
foreground volume-fraction / foreground-Dice benchmark, derive a binary foreground mask
by thresholding the label maps (any label > 0 = foreground). Segmentation GT does not
cover every frame in real (non-SIM) sets.

Download base: `https://data.celltrackingchallenge.net/training-datasets/<NAME>.zip`
(test sets under `.../test-datasets/`). Test-set GT is withheld — use **training** sets.

| Dataset | Cells / modality | Voxel (µm) | Seg. masks | Train size (verified) |
|---|---|---|---|---|
| **Fluo-C3DL-MDA231** | GFP MDA-MB-231 breast carcinoma, confocal | 1.242×1.242×6.0 | Silver (ST) | 192 MB ✓ |
| **Fluo-C3DH-A549** | GFP-actin A549 lung cancer, spinning disk | 0.126×0.126×1.0 | Silver (ST) | 257 MB ✓ |
| **Fluo-C3DH-H157** | GFP H157 lung carcinoma in matrigel, spinning disk | 0.126×0.126×0.5 | Silver (ST) | 7.0 GB |
| **Fluo-N3DL-TRIC** | Tribolium embryo nuclei, light-sheet | anisotropic | Gold (sparse) | 20.6 GB ✓ |
| **Fluo-C3DH-A549-SIM** | Simulated A549 | 0.126×0.126×1.0 | Perfect (GT) | 314 MB |
| **Fluo-N3DH-SIM+** | Simulated HL60 nuclei | 0.125×0.125×0.200 | Perfect (GT) | 3.1 GB |

Verified URLs:
- `https://data.celltrackingchallenge.net/training-datasets/Fluo-C3DL-MDA231.zip` (192,582,984 B)
- `https://data.celltrackingchallenge.net/training-datasets/Fluo-C3DH-A549.zip` (256,705,648 B)
- `https://data.celltrackingchallenge.net/training-datasets/Fluo-N3DL-TRIC.zip` (22,088,481,712 B)

Notes / caveats:
- **MDA231** and **H157** are cell-body (cytoplasm) sets — good match for fluorostats'
  foreground/volume-fraction use case. **TRIC** is nuclei and huge (20.6 GB), with only
  *sparse* gold GT (blastoderm lineages) — best skipped for B2 or used as a stress test.
- SIM sets (A549-SIM, N3DH-SIM+) have **perfect per-voxel masks** — ideal for a clean,
  bias-free Dice comparison since there's no annotator ambiguity.

---

## Group B — Broad Bioimage Benchmark Collection (BBBC), 3D

**Standard BBBC attribution (required, in addition to each set's own citation):**
> "We used image set [BBBCxxx] from the Broad Bioimage Benchmark Collection
> (Ljosa V, Sokolnicki KL, Carpenter AE. *Annotated high-throughput microscopy image
> sets for validation.* Nature Methods 9(7):637, 2012. DOI: 10.1038/nmeth.2083)."

Download base: `https://data.broadinstitute.org/bbbc/<ACCESSION>/`

### BBBC032 — Mouse embryo blastocyst cells  ★ real, best first pick
- **Modality:** spinning-disk confocal (PerkinElmer UltraVIEW VoX + Leica SP8), 63×,
  4-channel (405/488/568/647 nm). **Real** data.
- **Dimensions:** 1024×1344×172 (0.5 µm z-spacing); ~103×136×86 µm.
- **Format:** multi-channel 3D TIFF. **Masks:** manually annotated nuclei (real GT).
- **Size:** dataset 1.14 GB; ground truth 473.5 MB.
- **URLs:** `https://data.broadinstitute.org/bbbc/BBBC032/BBBC032_v1_dataset.zip` ·
  `https://data.broadinstitute.org/bbbc/BBBC032/BBBC032_v1_DatasetGroundTruth.tif`
- **Citation:** Rivron N, et al. *Blastocyst-like structures generated solely from stem
  cells.* Nature 557:106–111, 2018. DOI: 10.1038/s41586-018-0051-0 (via BBBC).
- **License:** CC0 (public domain), copyright waived by N. Rivron.
- **B2 fit:** strong — real fluorescence nuclei with human masks; derive foreground mask
  from labels for volume-fraction / Dice.

### BBBC050 — Nuclei of mouse embryonic cells  ★ real, small
- **Modality:** fluorescence (H2B-mRFP1 / mCherry), 3D time-series. **Real** data.
- **Voxel:** 0.8×0.8×1.75–2.0 µm; 11 train + 4 test embryos.
- **Masks:** three GT variants (NSN uniform, NDN central, QCANet individual nuclei).
- **Size:** Images 128 MB; GroundTruth 2 MB.
- **URLs:** `https://data.broadinstitute.org/bbbc/BBBC050/Images.zip` ·
  `https://data.broadinstitute.org/bbbc/BBBC050/GroundTruth.zip`
- **Citation:** Tokuoka Y, et al. *3D convolutional neural networks-based segmentation to
  acquire quantitative criteria of the nucleus during mouse embryogenesis.* npj Syst Biol
  Appl 6:32, 2020. DOI: 10.1038/s41540-020-00152-8.
- **License:** CC BY 3.0.
- **B2 fit:** strong and lightweight — real nuclei, tiny GT. Use NSN (uniform) masks for
  a foreground/volume comparison.

### BBBC024 — 3D HL60 cell line (synthetic)  ★ small sanity check
- **Modality:** simulated confocal (virtual Zeiss S100). 4 clustering levels × 2 SNR,
  30 images each, 20 nuclei/image.
- **Format:** ICS + TIFF. **Masks:** foreground/background + counts (16-bit labels).
- **Size:** small (multiple sub-archives; not individually listed on page).
- **URLs:** paired downloads at `https://data.broadinstitute.org/bbbc/BBBC024/`
  (per clustering probability 0/25/50/75% × high/low SNR).
- **Citation:** Svoboda D, Kozubek M, Stejskal S. *Generation of digital phantoms of cell
  nuclei and simulation of image formation in 3D image cytometry.* Cytometry A 75A(6):
  494–509, 2009. DOI: 10.1002/cyto.a.20714.
- **License:** CC BY 3.0 (David Svoboda).
- **B2 fit:** good bias-free check (perfect synthetic masks); SNR/clustering sweep lets
  you probe where Otsu breaks down.

### BBBC027 — 3D colon tissue (synthetic)
- **Modality:** simulated; clustered nuclei, high/low SNR (30 images).
- **Masks:** binary foreground + counts. **Format:** microscopy stacks (per BBBC).
- **Size:** 6 archives (3 per SNR variant); larger than BBBC024.
- **URLs:** parts 1–3 per SNR at `https://data.broadinstitute.org/bbbc/BBBC027/`.
- **Citation:** Svoboda D, Homola O, Stejskal S. *Generation of 3D digital phantoms of
  colon tissue.* ICIAR 2011, LNCS 6754:31–39. DOI: 10.1007/978-3-642-21596-4_4.
- **License:** CC BY 3.0 (David Svoboda).
- **B2 fit:** dense clustered synthetic tissue — stress test for foreground thresholding.

### BBBC034 — Induced pluripotent human stem cells (hiPSC)  ★ real
- **Modality:** confocal/widefield (Zeiss AxioObserver, 100×/1.25 water). 4 channels
  (CellMask Deep Red, GFP, Hoechst DNA, brightfield). **Real** data.
- **Dimensions:** 1024×1024×52; ~66.6×66.6×15 µm.
- **Masks:** manually annotated/segmented 3D nuclei (colony center/edge have no GT).
- **Size:** dataset 572.2 MB; GT provided (CSV, 872 KB — check whether label images are
  included before relying on it for voxel-wise Dice).
- **URLs:** dataset + GT at `https://data.broadinstitute.org/bbbc/BBBC034/`.
- **Citation:** Allen Institute for Cell Science (AICS), via BBBC. (No standalone paper;
  cite AICS + the Ljosa 2012 BBBC reference.)
- **License:** CC BY 4.0.
- **B2 fit:** real hiPSC nuclei; verify the GT is a voxel label image (not only a
  count/centroid CSV) before using for Dice.

---

## Summary table — B2 suitability

| Dataset | Real/Sim | Masks | License | Size | B2 priority |
|---|---|---|---|---|---|
| BBBC032 | Real | Manual nuclei | CC0 | 1.6 GB | ★★★ first pick |
| BBBC050 | Real | 3× nuclei GT | CC BY 3.0 | 130 MB | ★★★ lightweight |
| Fluo-C3DL-MDA231 (CTC) | Real | Silver | CC BY 4.0* | 192 MB | ★★ cytoplasm |
| Fluo-C3DH-A549 (CTC) | Real | Silver | CC BY 4.0* | 257 MB | ★★ cytoplasm |
| Fluo-N3DH-SIM+ (CTC) | Sim | Perfect | CC BY 4.0* | 3.1 GB | ★★ clean Dice |
| BBBC024 | Sim | Perfect fg+count | CC BY 3.0 | small | ★★ sanity/SNR sweep |
| BBBC034 | Real | Manual (verify) | CC BY 4.0 | 573 MB | ★ verify GT type |
| BBBC027 | Sim | Binary fg | CC BY 3.0 | medium | ★ stress test |

\* CTC license per challenge data policy — confirm on the download page before redistribution.

## Open items to verify before use
- CTC exact license string (page states research-use terms; treat as CC BY 4.0 pending
  confirmation on the download page).
- BBBC034 GT granularity: confirm voxel label images exist, not just a centroid CSV.
- CTC silver-truth (ST) masks are algorithm-derived consensus, not manual — acceptable
  for cross-tool *agreement* but not an absolute human gold standard.
