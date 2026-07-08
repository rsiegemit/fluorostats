# Competitor Benchmark: REAVER (Corliss et al. 2020)

**Status:** VERIFIED by fetching primary sources (Zenodo record, PMC full text, GitHub repo). Numbers below are quoted from the paper's PMC full text. Where a figure could not be independently re-derived, it is flagged.

## 1. Paper

- **Title:** "REAVER: A program for improved analysis of high-resolution vascular network images"
- **Authors:** Bruce A. Corliss, Richard W. Doty, Corbin Mathews, Paul A. Yates, Tingting Zhang, Shayn M. Peirce (University of Virginia)
- **Journal:** *Microcirculation*, 2020
- **DOI:** [10.1111/micc.12618](https://onlinelibrary.wiley.com/doi/10.1111/micc.12618)
- **Open-access full text:** [PMC7507177](https://pmc.ncbi.nlm.nih.gov/articles/PMC7507177/)
- **Preprint:** bioRxiv [10.1101/707570](https://www.biorxiv.org/content/10.1101/707570v1.full)
- REAVER = "Rapid Editable Analysis of Vessel Elements Routine." MATLAB (2018a/2019a), Image Processing Toolbox required.

## 2. Dataset (the benchmark)

- **Name:** REAVER Vascular Networks Fluorescent Image Dataset
- **Zenodo DOI:** [10.5281/zenodo.3340165](https://doi.org/10.5281/zenodo.3340165)
- **Direct download URL (wget-able):**
  `https://zenodo.org/records/3340165/files/REAVER_Vascular_Networks_Image_Dataset.zip`
- **Size:** 62.9 MB (zip)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Format:** 8-bit greyscale TIFs (converted from Nikon IDS). Manual ground-truth TIFs use 3 channels:
  - **Red** = binary segmentation mask (0 / 255)
  - **Green** = binary skeleton (0 / 255)
  - **Blue** = original raw image
- **Content:** 36 2D maximum-intensity-projection images from z-stacks, from **six murine tissues**, single C57Bl6/J mouse. Acquired on Nikon TE-2000E confocal: 20x objective (530 µm FOV) and 60x objective (212 µm FOV).
- **Ground truth:** "mixed-manual" — ImageJ macros gave an initial threshold/segmentation guess, then a human manually corrected with the paintbrush. This gives pixel-level truth labels (enables sensitivity/specificity/accuracy).

### Availability of the dataset
- **NOT in the git repo.** The GitHub README links out to the Zenodo DOI above. So: `git clone` the code, `wget` the dataset zip separately.
- Code repo: `git clone https://github.com/uva-peirce-cottler-lab/public_REAVER` (BSD-3-Clause). Older mirror: github.com/bacorliss/REAVER_public.

## 3. Reported accuracy numbers (the numbers to beat)

REAVER benchmarked itself against **other automated programs** — the paper's comparison anchor is **AngioTool** (also RAVE, AngioQuant appear in specificity ranking). All reductions are in **mean absolute error vs. manual ground truth**, REAVER relative to AngioTool:

| Metric | REAVER vs AngioTool | P-value | Note |
|---|---|---|---|
| **Vessel length density** | **76.5% reduction** in MAE | 6.57e-3 | REAVER had lowest MAE, sig. diff from all other programs |
| **Vessel area fraction** | **75.8% reduction** in error | 6.16e-8 | REAVER highest accuracy |
| **Mean vessel diameter** | **83.9% reduction** in error | 8.29e-7 | |
| **Branchpoint count** | **94.6% reduction** in error | 4.43e-5 | REAVER lowest MAE, sig. diff from all others |

**Pixel-level segmentation (vs manual):**

| Metric | REAVER vs AngioTool | P-value |
|---|---|---|
| **Accuracy** | +6.4% (highest mean accuracy) | 1.73e-7 |
| **Sensitivity** | +34.1% | 1.00e-15 |
| **Specificity** | RAVE/AngioQuant best; REAVER +0.4% over AngioTool | 4.39e-2 |
| **Execution time** | −36.4% vs AngioTool | 1.8e-16 |

- Manual analysis cost: **3089 ± 1355 seconds per image**; all automated tools ran in <1% of that.
- **CAUTION:** the headline numbers are *relative improvements over AngioTool*, not absolute error values. The paper reports absolute MAE per program in figures/tables that are not fully machine-readable here — pull the exact per-program absolute MAE from Figures 3–5 / supplementary tables of PMC7507177 before quoting absolute values in our paper.

## 4. Metrics REAVER computes (alignment with fluorostats)

REAVER outputs 10 metrics; those that map to fluorostats:

| REAVER metric | fluorostats equivalent |
|---|---|
| `vessel_length_density_mmpmm2` (µm length / image area) | total length / area |
| Vessel length (µm) | total length |
| Branchpoint count | junctions |
| Segment count | branches |
| Vessel area fraction | area fraction (from mask) |
| Mean segment length, tortuosity, valency, diameter | secondary |

Direct overlap for a head-to-head: **total length, junctions (branchpoints), branches (segments), area fraction.**

## 5. How to run the head-to-head

1. **Get data:**
   `wget https://zenodo.org/records/3340165/files/REAVER_Vascular_Networks_Image_Dataset.zip`
2. **Ground truth:** for each image, the manual-analysis TIF's **red channel = segmentation mask** and **green channel = skeleton**. Compute reference length density / branchpoints / area fraction from these channels (matches how REAVER scored itself).
3. **Run fluorostats** on the raw greyscale TIFs (same 36 images). Produce: total length, junctions, branches, area fraction, in the same physical units (µm; FOV 530 µm @20x, 212 µm @60x — derive µm/px per magnification).
4. **Score:** compute mean absolute error of fluorostats vs. the manual ground truth, per metric, over all 36 images. Report fluorostats MAE alongside REAVER's and AngioTool's published MAE (from the paper's tables).
5. **Win condition:** fluorostats MAE ≤ REAVER's on length density and branchpoint count (their strongest claims), on the *same* images with the *same* ground truth. Report per-image scatter + correlation vs. manual, mirroring their Figures.

**To pin down before publishing:** exact per-program absolute MAE values from PMC7507177 Figures 3–5 / supplement (only relative % reductions were extractable here).
