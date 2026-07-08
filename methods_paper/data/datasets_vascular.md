# Vascular / Endothelial / Tubular Network Datasets — Benchmark B4 (fluorostats)

Public, citable datasets for benchmarking **fluorostats** network metrics (junctions,
branches, total length, connectivity) against **AngioTool**, **Angiogenesis Analyzer**,
and **REAVER**. All entries verified by fetching the source page (fetch date: 2026-07-08).

**Modality legend:** FL = fluorescence microscopy; FUNDUS = color fundus photography
(NOT fluorescence — see caveats). GT = ground-truth network/segmentation annotations.

---

## Tier 1 — Fluorescence vascular networks (direct fit for B4)

### 1. REAVER Vascular Networks Fluorescent Image Dataset  ⭐ BEST FIRST
- **Citation:** Corliss BA, Doty R, Yates PA, Peirce SM. *REAVER: A program for
  improved analysis of high-resolution vascular network images.* Microcirculation
  27(5):e12618, 2020. DOI: 10.1111/micc.12618. Dataset: Corliss et al., Zenodo, 2019.
- **Dataset DOI:** 10.5281/zenodo.3340165
- **Download:** https://zenodo.org/records/3340165/files/REAVER_Vascular_Networks_Image_Dataset.zip?download=1
- **Size:** 62.9 MB (zip); ~15.2 GB uncompressed
- **Format / dims / modality:** 8-bit greyscale TIFF (converted from Nikon IDS) · **2D**
  confocal · **FL**
- **Ground truth:** YES — "ImageJ_Manual" folder holds manual segmentation used as
  ground truth; 36 images per analysis folder
- **License:** CC-BY-4.0
- **Feeds B4 · caveat:** THE reference validation set — this is the exact dataset REAVER
  used to rank REAVER/AngioTool/AngioQuant/RAVE. Ideal head-to-head for fluorostats vs
  those tools. Code: https://github.com/bacorliss/REAVER_public (MATLAB 2019a).

### 2. MiniVess — rodent cerebrovasculature (in vivo two-photon)
- **Citation:** Poon C, Teikari P, Rachmadi MF, et al. *A dataset of rodent
  cerebrovasculature from in vivo multiphoton fluorescence microscopy imaging.*
  Scientific Data 10:141, 2023. DOI: 10.1038/s41597-023-02048-8.
  (Preprint: bioRxiv 2022.07.19.500542.)
- **Dataset DOI (EBRAINS):** 10.25493/HPBE-YH  ·  page:
  https://www.nature.com/articles/s41597-023-02048-8
- **Contents:** 70 **3D** image volumes with segmented ground truths
- **Modality:** **FL** — two-photon (multiphoton) fluorescence, in vivo
- **Ground truth:** YES — segmentations via classical image processing + U-Net +
  manual proofreading
- **License:** open (EBRAINS; verify exact CC terms at repository)
- **Feeds B4 · caveat:** First annotated 3D fluorescence cerebrovascular set — excellent
  for 3D network-length/junction validation with GT. Download requires EBRAINS access
  (DOI 10.25493/HPBE-YH); confirm final license on the EBRAINS landing page before use.

### 3. VesselExpress — 3D light-sheet vasculature (multi-organ)
- **Citation:** Spangenberg P, Hagemann N, Squire A, Förster N, et al. *VesselExpress:
  Rapid and fully automated blood vasculature analysis in 3D light-sheet image volumes
  of different organs.* Cell Reports Methods 3(3):100436, 2023.
  DOI: 10.1016/j.crmeth.2023.100436.
- **Dataset DOI:** 10.5281/zenodo.6025935 (v2, 2022)
- **Download:** https://zenodo.org/records/6025935/files/VesselExpress_Data.zip?download=1
  (MD5 3934d42de8f6621182afa5d2e489124f)
- **Size:** 41.3 GB
- **Format / dims / modality:** ZIP of 3D volumes · **3D** · **FL** light-sheet
- **Ground truth:** YES — raw + segmented + skeletonized volumes provided
- **License:** CC-BY-4.0
- **Feeds B4 · caveat:** Large. Skeletonized volumes give branch/length reference for 3D
  network metrics. Software: https://github.com/RUB-Bioinf/VesselExpress.

### 4. SproutAngio — fibrin bead assay, VEGF-A dose series
- **Citation:** Beter M, Laakkonen J, Ylä-Herttuala S, et al. *SproutAngio: an
  open-source bioimage informatics tool for quantitative analysis of sprouting
  angiogenesis and lumen space.* Scientific Reports 13:6902, 2023.
  DOI: 10.1038/s41598-023-33090-6. Dataset: Beter M, Zenodo, 2022.
- **Dataset DOI:** 10.5281/zenodo.7240927
- **Download:** per-file, e.g.
  https://zenodo.org/records/7240927/files/group1-01.czi?download=1
- **Size:** 4.8 GB · 50 files (5 VEGF-A dose groups: 0/1/10/20/50 ng/ml)
- **Format / dims / modality:** Zeiss CZI · **3D** confocal z-stacks · **FL**
  (DAPI nuclei + phalloidin F-actin)
- **Ground truth:** NONE included — image-only (use for cross-tool metric comparison,
  not accuracy-vs-GT)
- **License:** CC-BY-4.0
- **Feeds B4 · caveat:** In vitro sprouting/tube-like network assay closest to
  tube-formation use case; dose gradient gives a graded network-density series. No GT,
  so benchmark = agreement between fluorostats and AngioTool/Angiogenesis Analyzer.

---

## Tier 2 — Retinal vessel-segmentation standards (network topology reference; NOT fluorescence)

> ⚠️ **Caveat for all Tier 2:** DRIVE/STARE/HRF are **color fundus photographs**, not
> fluorescence. They are the field-standard vessel-segmentation benchmarks with expert
> GT, so they validate fluorostats' segmentation/skeletonization/junction logic on a
> known network topology — but contrast/illumination differ from fluorescence. Report
> them as topology/segmentation sanity checks, not as fluorescence performance.

### 5. DRIVE (Digital Retinal Images for Vessel Extraction)
- **Citation:** Staal J, Abràmoff MD, Niemeijer M, Viergever MA, van Ginneken B.
  *Ridge-based vessel segmentation in color images of the retina.* IEEE TMI
  23(4):501–509, 2004. DOI: 10.1109/TMI.2004.825627.
- **Download / host:** https://drive.grand-challenge.org/ (registration required)
- **Contents:** 40 images (20 train / 20 test), 768×584 px, JPEG, 45° FOV + ROI masks
- **Ground truth:** YES — manual vessel segmentation (train public; test held out).
  Second-observer set also provided.
- **License / terms:** Grand Challenge Terms of Service (research use)
- **Feeds B4 (topology check) · caveat:** FUNDUS, not FL. Test GT withheld — evaluate on
  the 20 training images with public GT.

### 6. STARE (STructured Analysis of the Retina)
- **Citation:** Hoover A, Kouznetsova V, Goldbaum M. *Locating blood vessels in retinal
  images by piecewise threshold probing of a matched filter response.* IEEE TMI
  19(3):203–210, 2000. DOI: 10.1109/42.845178.
- **Download / host:** https://cecas.clemson.edu/~ahoover/stare/
- **Contents:** 20 fundus images, 700×605 px · GT from two independent expert annotators
- **Ground truth:** YES (two experts; first used for analysis)
- **License / terms:** free for research (see site)
- **Feeds B4 (topology check) · caveat:** FUNDUS, not FL; small set.

### 7. HRF (High-Resolution Fundus)
- **Citation:** Budai A, Bock R, Maier A, Hornegger J, Michelson G. *Robust vessel
  segmentation in fundus images.* Int J Biomed Imaging 2013:154860, 2013.
  DOI: 10.1155/2013/154860.
- **Download / host:** https://www5.cs.fau.de/research/data/fundus-images/
- **Contents:** 45 images (15 healthy / 15 diabetic retinopathy / 15 glaucoma),
  3504×2336 px · binary vessel GT + FOV masks + optic-disk annotations. ~73 MB total.
- **Ground truth:** YES — gold-standard binary vessel masks for all 45
- **License:** CC-BY-4.0 (research use); cite Budai et al. 2013
- **Feeds B4 (topology check) · caveat:** FUNDUS, not FL; highest resolution of the
  fundus standards — good stress test for fine-branch detection.

---

## Summary table

| # | Dataset | Modality | Dim | GT | Size | License | Direct-fit for B4? |
|---|---------|----------|-----|----|----|---------|-----|
| 1 | REAVER | FL confocal | 2D | Yes | 63 MB | CC-BY-4.0 | ⭐ Primary (tool-vs-tool + GT) |
| 2 | MiniVess | FL 2-photon | 3D | Yes | (EBRAINS) | open | Strong 3D + GT |
| 3 | VesselExpress | FL light-sheet | 3D | Yes | 41 GB | CC-BY-4.0 | 3D skeleton reference |
| 4 | SproutAngio | FL confocal | 3D | No | 4.8 GB | CC-BY-4.0 | Tube/sprout assay, no GT |
| 5 | DRIVE | FUNDUS | 2D | Yes | small | GC ToS | Topology check only |
| 6 | STARE | FUNDUS | 2D | Yes | small | research | Topology check only |
| 7 | HRF | FUNDUS | 2D | Yes | 73 MB | CC-BY-4.0 | Topology check only |

**Recommended B4 core:** #1 REAVER (2D FL, GT, exact tool-comparison provenance) as the
primary benchmark; #2 MiniVess and #3 VesselExpress for 3D-with-GT; #4 SproutAngio for
the in-vitro tube/sprout use case; #5–7 as segmentation/topology sanity checks (flag as
fundus, not fluorescence).
