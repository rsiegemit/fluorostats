# Competitor 3D Light-Sheet Vascular Tools & Datasets

Mapping **VesSAP**, **VesselExpress**, and **ClearMap/TubeMap** to their released 3D
fluorescence datasets, for a fluorostats methods-paper benchmark on competitors' own data.

**Scope honesty:** All three target whole-organ / whole-brain light-sheet volumes (terabyte
scale), not in-vitro constructs. Benchmarking fluorostats here tests its **3D generality**
(skeleton + connectivity + density metrics on real light-sheet vessel volumes), *not* its
in-vitro construct application niche. The fair, tractable comparison is on a **downloadable
subvolume**, not a whole organ.

All facts below were verified by fetching the cited URLs / APIs on 2026-07-08. Items I could
not verify are flagged **[UNVERIFIED]**.

---

## TL;DR — the tractable target

**VesselExpress ships a ready-to-run 3D light-sheet vessel volume in its GitHub repo:**
`VesselExpress/data/test.tiff` — **verified: 100 × 500 × 500 voxels, uint16, ~48 MB**,
voxel size **2.0 × 1.015625 × 1.015625 µm** (z,y,x) per the repo `config.json`
(`graphAnalysis.pixel_dimensions`), i.e. a **0.0516 mm³** subvolume
(200 × 508 × 508 µm extent). This is the fairest, lowest-friction fluorostats comparison:
small, calibrated, downloadable in one `curl`, and it exercises the exact
segment → skeleton → graph → density path fluorostats implements.

Direct download (verified):
`https://raw.githubusercontent.com/RUB-Bioinf/VesselExpress/master/VesselExpress/data/test.tiff`

---

## 1. VesselExpress (Spangenberg et al., 2023, Cell Reports Methods)

- **Paper:** "Rapid and fully automated blood vasculature analysis in 3D light-sheet image
  volumes of different organs." Cell Reports Methods, 27 Mar 2023.
  DOI: `10.1016/j.crmeth.2023.100436`. Open-access full text:
  `https://pmc.ncbi.nlm.nih.gov/articles/PMC10088239/`
- **Code:** `https://github.com/RUB-Bioinf/VesselExpress` (Snakemake pipeline; Docker
  `phispa1812/vesselexpress` and `..._cli`; Napari plugin `vessel-express-napari`).
- **Pipeline:** segmentation → skeletonization → graph construction/analysis → optional
  Blender rendering. Input: 2D/3D TIFF. Output: **nine phenotypic features as text files**,
  plus 3D TIFs of segmented + skeletonized vasculature and graph/branch/terminal-point images.

### Datasets (verified via Zenodo API)
- **Full processed dataset:** `VesselExpress_Data.zip` — **41.31 GB**, single file,
  MD5 `3934d42de8f6621182afa5d2e489124f`. License **CC-BY-4.0**.
  Contains raw, segmented, and skeletonized 3D light-sheet volumes of **different organs**.
  - Concept/record DOI `10.5281/zenodo.5733150` and version DOI `10.5281/zenodo.6025935`
    **both resolve to the same 41.3 GB zip** (verified: identical filename + MD5). The README
    calls 5733150 the "example data … different organs with preset parameters per organ."
  - **This is the only Zenodo asset — there is no small subset zip there.**
- **In-repo tractable subvolume:** `test.tiff` (see TL;DR) + `config.json` /
  `config_standard.json`. This is the practical benchmark input.

### Metrics reported (nine features; from PMC full text)
1. Vessel length density  2. Total vessel length  3. Branch length  4. Branch (segment)
diameter  5. Tortuosity  6. Branch volume density  7. Branching-point density
8. Terminal-point density  9. Branching angle.
- Quantitative anchor: mean microvessel diameter in cerebral cortex **4.8 ± 0.2 µm**;
  whole-brain vascular fractional volume **1–2%**. Obesity vs. WT compared vessel length
  density split by caliber (<4 µm, 4–6 µm, >6 µm).
- Default calibration in repo config: **2.0 × 1.0156 × 1.0156 µm** voxels.

### fluorostats overlap (strong)
Length density, branch/junction (branching-point) density, and segment diameter/radius map
**directly** onto fluorostats' 3D skeleton + connectivity outputs. Same segment→skeleton→graph
model. **Best head-to-head target.**

---

## 2. VesSAP (Todorov et al., 2020, Nature Methods)

- **Paper:** "Machine learning analysis of whole mouse brain vasculature." Nature Methods,
  2020. DOI: `10.1038/s41592-020-0792-1`. Open access (PMC):
  `https://pmc.ncbi.nlm.nih.gov/articles/PMC7591801/`. Preprint: bioRxiv `10.1101/613257`.
- **Code / data portal:** `https://github.com/vessap/vessap`;
  Code Ocean capsule `https://doi.org/10.24433/CO.1402016.v1`;
  data portal `http://DISCOtechnologies.org/VesSAP`.
- **Data access:** portal serves a **"Guide to access the VesSAP data and algorithms
  repository" (PDF, `VesSAP_data_repository_access.pdf`, updated 2021-10-01)** — access is
  **gated through a request/guide document, not a single direct download URL** [UNVERIFIED:
  exact file sizes/formats behind the guide]. Repository includes imaging protocol, original
  scans, registered atlas data (1,238 structures), trained segmentation models, training data,
  and segmented whole-brain scans.
- **License:** **CC BY-NC 4.0** (non-commercial) — note for reuse/redistribution.

### Metrics reported
Centerlines, **bifurcation points**, and **segment radii**; aggregated per **71 Allen-atlas
anatomical clusters** as local vessel length, bifurcation density, and average radius.
- Quantitative anchors (verified): whole-brain vessel length density **545.74 ± 94 mm/mm³**;
  cortical **913 ± 110 mm/mm³** (C57BL/6J). Vessel length vs. bifurcation density strongly
  correlated (Pearson r = 0.9657). Reported in SI units at **3 µm** voxel spacing.
- **Imaging voxel size:** **2.83 × 2.83 × 4.99 µm** (X,Y,Z).

### Tractability
Whole-brain scans are **terabyte-scale**; the pipeline itself tiles into
**50 × 100 × 100-voxel** sub-volumes. Usable for fluorostats **only on a subset/tile**, and
the CC-BY-NC license + gated access add friction. Metrics (length density, bifurcation
density, radius) map well, but download friction is higher than VesselExpress.

---

## 3. ClearMap 2 / TubeMap (Kirst et al., 2020, Cell)

- **Correct citation:** Renier 2016 Cell (`10.1016/j.cell.2016.05.007`) is **c-Fos / activity
  mapping**, *not* vasculature. The **vasculature** pipeline is **TubeMap**, Kirst et al.,
  "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature," Cell 2020,
  `https://www.cell.com/cell/fulltext/S0092-8674(20)30109-4` (fetch blocked 403; citation
  confirmed via multiple secondary sources).
- **Code:** `https://github.com/ClearAnatomics/ClearMap` (ClearMap 2 with WobblyStitcher,
  TubeMap, CellMap). License **GPL-3.0**. Zenodo software archive `10.5281/zenodo.3924619`.
  Docs: `https://christophkirst.github.io/ClearMap2Documentation`.
- **Tutorial data:** OSF repository
  `https://osf.io/sa3x8/?view_only=4427a838cbd0468c9fbad9cab465d866` (linked from
  `https://idisco.info/clearmap-2/`); TubeMap handbook PDF
  `https://idisco.info/wp-content/uploads/2020/02/handbook.pdf`.
  **[UNVERIFIED: exact OSF file list, sizes, and formats — the view-only OSF link and the
  Cell paper's data-availability statement were not machine-fetchable here.]**
- **Metrics:** TubeMap binarizes vessels, CNN-fills, skeletonizes, builds a graph; classifies
  arteries/veins; investigates **branching** and vessel properties. Same graph model as
  fluorostats. **[UNVERIFIED: exact published length-density / branch-density numbers.]**

### Tractability
Designed for O(TB) whole-brain data. Tutorial subset on OSF is the only plausibly tractable
entry, but its size/format is unverified and access is view-only-link gated. Lower priority
than VesselExpress for a clean head-to-head.

---

## Recommended fluorostats comparison

**Primary:** VesselExpress `test.tiff` (100×500×500, 0.0516 mm³, 2.0×1.016×1.016 µm).
Run fluorostats' 3D skeleton + connectivity and report, in VesselExpress's own units:
- **Vessel length density (mm/mm³)** — direct competitor metric.
- **Branch / junction density (per mm³)** — fluorostats connectivity vs. VesselExpress
  branching-point density.
- **Mean segment radius/diameter (µm)** — anchor to cortex 4.8 µm caliber.

**Cross-check anchors** (whole-organ context, not head-to-head): VesSAP whole-brain
545.74 ± 94 mm/mm³ / cortical 913 ± 110 mm/mm³ length density; VesSAP length↔bifurcation
r = 0.9657.

**Framing:** report as a **3D-generality** benchmark on a downloadable competitor subvolume,
explicitly noting these tools target whole-organ scale while fluorostats' niche is in-vitro
constructs.

---

## Verified download quick-reference

| Asset | URL | Size | License | Verified |
|---|---|---|---|---|
| VesselExpress `test.tiff` (subvolume) | `raw.githubusercontent.com/RUB-Bioinf/VesselExpress/master/VesselExpress/data/test.tiff` | ~48 MB (100×500×500 uint16) | repo (MIT-style) | ✅ downloaded + measured |
| VesselExpress full data (Zenodo) | `zenodo.org/records/6025935` (== 5733150) | 41.31 GB | CC-BY-4.0 | ✅ Zenodo API |
| VesSAP data portal | `DISCOtechnologies.org/VesSAP` (gated guide PDF) | TB-scale | CC-BY-NC-4.0 | ⚠️ gated / partial |
| ClearMap/TubeMap tutorial data | `osf.io/sa3x8` (view-only) | unknown | GPL-3.0 (code) | ⚠️ unverified |
