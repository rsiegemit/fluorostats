# Competitor Mapping: CellProfiler & ilastik — Benchmark Data for fluorostats Head-to-Head

Purpose: map CellProfiler (Carpenter 2006; McQuin 2018) and ilastik (Berg 2019) onto the
public example/validation datasets they use, capture any published accuracy figures, and
identify the best shared public fluorescence dataset for a fluorostats-vs-CellProfiler
head-to-head (Bland-Altman / ICC on volume, area fraction, object counts).

All URLs below were fetched/verified July 2026. Items I could not fully verify are flagged
**[UNVERIFIED]**.

---

## 1. CellProfiler example pipelines + image sets

CellProfiler ships downloadable example pipelines bundled with images and expected outputs.

- Examples landing page: https://cellprofiler.org/examples (verified)
- Examples GitHub repo (pipelines + images + per-example READMEs):
  https://github.com/CellProfiler/examples (verified — folders: ExampleHuman,
  ExampleFly, ExampleTumor, ExampleColocalization, ExampleYeastColonies, ~13 more)
- Tutorials repo: https://github.com/CellProfiler/tutorials (verified)
- Published pipelines index: https://cellprofiler.org/published-pipelines (verified)

Example pipelines relevant to fluorostats (intensity / area / object counting). Download
URLs verified to resolve on the `cellprofiler-examples.s3.amazonaws.com` bucket:

| Example | What it measures | Download (zip) | fluorostats replicable? |
|---|---|---|---|
| **ExampleHuman** (human HT29 cells, Carpenter 2006) | Identifies nuclei + cells; morphology, **intensity, area**, texture, counts | https://cellprofiler-examples.s3.amazonaws.com/ExampleHuman.zip | Yes — nuclei/cell **area + intensity + count** |
| ExampleFly (Drosophila Kc167, clumpy) | Morphology, intensity, texture; clump splitting | https://cellprofiler-examples.s3.amazonaws.com/ExampleFly.zip | Yes — count + area |
| ExampleTumor (mouse lung) | **Object counting + size/area** | https://cellprofiler-examples.s3.amazonaws.com/ExampleTumor.zip | Yes — count + area |
| ExamplePercentPositive ("Counting & scoring") | Object counting, intensity, % positive | https://cellprofiler-examples.s3.amazonaws.com/ExamplePercentPositive.zip | Yes — count + intensity |
| ExampleSpeckles (speckle/foci counting) | Per-nucleus foci **counting** | https://cellprofiler-examples.s3.amazonaws.com/ExampleSpeckles.zip | Partial — count |
| ExampleColocalization | Two-channel **intensity** overlap | https://cellprofiler-examples.s3.amazonaws.com/ExampleColocalization.zip | Yes — intensity |
| ExampleVitraImages (C→N translocation) | Nuclear/cytoplasmic **intensity ratio** | https://cellprofiler-examples.s3.amazonaws.com/ExampleVitraImages.zip | Yes — intensity by region |
| ExampleIlluminationCorrection | Intensity (background correction) | https://cellprofiler-examples.s3.amazonaws.com/ExampleIlluminationCorrection.zip | Preprocessing only |

**ExampleHuman** is the canonical demonstrator (fluorescence DNA channel; Carpenter 2006
Genome Biology reference confirmed in its README). It produces per-object **area, integrated/
mean intensity, and object counts** — exactly the quantities fluorostats computes — but note
these example sets are illustrative and do **not ship independent ground-truth masks**, so
they support pipeline-parity comparison, not accuracy scoring. For accuracy scoring use a BBBC
set with ground truth (Section 3–4).

---

## 2. ilastik example / benchmark datasets and reported accuracy

- ilastik paper: Berg et al., "ilastik: interactive machine learning for (bio)image
  analysis," *Nature Methods* 16, 1226–1232 (2019).
  https://www.nature.com/articles/s41592-019-0582-9 (verified — landing page)
- Publications / examples hub: https://www.ilastik.org/publications (verified)

ilastik provides pre-defined workflows for pixel classification / segmentation, object
classification, counting, and tracking. The Berg 2019 paper is primarily a tool/usability
paper; **it does not report a single headline segmentation-accuracy number on a public
benchmark** in a form directly comparable to F1 tables. **[UNVERIFIED: no specific
per-dataset accuracy figure extracted]** — ilastik's accuracy is user/annotation-dependent
by design (interactive Random Forest on user-drawn labels), which weakens it as a fixed
comparator. Practical consequence: ilastik is best treated as a *segmentation front-end*
whose masks feed measurement, not as a fixed-accuracy baseline. If included in the paper,
report it as "ilastik pixel-classification workflow, default/trained-on-N-strokes," not as a
canonical score.

---

## 3. CellProfiler in independent comparison studies (public data)

Yes — CellProfiler has a **published, quantitative baseline** on public BBBC data, which is
the strongest hook for a head-to-head.

**Caicedo et al., "Nucleus segmentation across imaging experiments: the 2018 Data Science
Bowl," *Nature Methods* 16, 1247–1253 (2019).**
- https://www.nature.com/articles/s41592-019-0612-7 (verified — paywalled full text)
- PubMed: https://pubmed.ncbi.nlm.nih.gov/31636459/ (verified)
- Dataset released as **BBBC038v1** (see Section 4).

Reported figures (from indexed abstract/summary; **[UNVERIFIED against full-text PDF —
confirm exact values before citing]**):
- At IoU threshold 0.5: top deep-learning model **F1 = 0.889** vs **CellProfiler reference
  F1 = 0.819**.
- On the hardest slice (small fluorescent nuclei): top model **F1 = 0.932** vs
  **CellProfiler F1 = 0.844**.
- Top-3 DL models beat the CellProfiler reference on small-fluorescent, "purple," and
  "pink-and-purple tissue" image types.

Related independent evaluation (context, verify before citing):
- Caicedo et al. 2018 bioRxiv, "Evaluation of Deep Learning Strategies for Nucleus
  Segmentation in fluorescence images":
  https://www.biorxiv.org/content/10.1101/335216v2.full.pdf (verified URL) — this is the
  **BBBC039** evaluation paper (Section 4).

Takeaway for the methods paper: CellProfiler is a **respected but beatable baseline** on
public fluorescence nuclei data — an F1 ≈ 0.82–0.84 reference the literature already
publishes. fluorostats can position against that same public data.

---

## 4. Best shared public dataset for a fluorostats-vs-CellProfiler head-to-head

### RECOMMENDED: BBBC039 — Nuclei of U2OS cells (Hoechst fluorescence)
- Page: https://bbbc.broadinstitute.org/BBBC039 (verified)
- Modality: **fluorescence microscopy, Hoechst DNA stain** (single, clean channel)
- Size: **200 fields of view**, TIFF, 520×696 px, 16-bit; **~23,000 manually annotated
  nuclei** as instance ground truth
- Ground truth: PNG instance masks (touching nuclei get distinct labels)
- Downloads (verified):
  - Images: https://data.broadinstitute.org/bbbc/BBBC039/images.zip (77.9 MB)
  - Masks: https://data.broadinstitute.org/bbbc/BBBC039/masks.zip (2.8 MB)
  - Metadata: https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip (18 KB)
- Citation: BBBC039v1, Caicedo et al. 2018; via BBBC (Ljosa et al., *Nature Methods*, 2012)

Why BBBC039 is the best choice:
- Pure fluorescence, single channel — squarely in fluorostats' wheelhouse (intensity, area
  fraction, object counts).
- Ships **instance ground truth**, enabling an absolute reference (not just CP-vs-fluorostats
  agreement, but both-vs-truth).
- Manageable size (200 FOV) for a full both-tools run.
- Has a published CellProfiler-relevant evaluation lineage (Caicedo 2018 bioRxiv).

### ALTERNATIVE / broader: BBBC038v1 (2018 Data Science Bowl)
- Page: https://bbbc.broadinstitute.org/BBBC038 (verified)
- Downloads: stage1_train.zip (82.9 MB), stage1_test.zip (9.5 MB),
  stage2_test_final.zip (289.7 MB), via https://data.broadinstitute.org/bbbc/BBBC038/
- Mixed fluorescence + histology; instance PNG masks (one nucleus per mask, non-overlapping).
- Use only its **fluorescent subset** for a like-for-like fluorostats comparison. This set
  carries the **published CellProfiler F1 baseline** from Caicedo 2019 (Section 3) — good if
  you want to cite an existing CP number rather than rerun CP.

### Other candidate BBBC fluorescence sets (verified pages, secondary):
- BBBC006 — Human U2OS, Hoechst 33342, z-stack focus series, GT provided:
  https://bbbc.broadinstitute.org/BBBC006
- BBBC005 — Synthetic cells (controlled counts/focus):
  https://bbbc.broadinstitute.org/BBBC005
- BBBC benchmarking methodology: https://bbbc.broadinstitute.org/benchmarking (verified)

### How to obtain CellProfiler's output as the comparator
1. Install CellProfiler 4.2.x (Section 5); load the standard nuclei-ID pipeline
   (IdentifyPrimaryObjects on the DNA channel — the ExampleHuman/ExampleSpeckles nuclei
   pattern) or an established BBBC039 pipeline.
2. Run **headless** over the BBBC039 image set:
   `cellprofiler -c -r -p pipeline.cppipe -o output/ -i images/`
3. Export per-object measurements (ExportToSpreadsheet): object count, `AreaShape_Area`,
   intensity — plus an ObjectsImage to derive area fraction / "volume" (area × count proxy).
4. Compute the same quantities with fluorostats on the identical TIFFs.
5. Compare CP vs fluorostats via **Bland-Altman + ICC** on area fraction and object counts;
   optionally score **both vs the BBBC039 ground-truth masks** (F1 / IoU) for an absolute
   anchor. fluorostats already has stats/ICC helpers (v0.2 `stats` module).

---

## 5. Install / runnability (confirmed available)

**CellProfiler** (verified):
- PyPI: https://pypi.org/project/CellProfiler/ — latest **4.2.8**; `pip install CellProfiler==4.2.8` (Python **3.8**; pip is the recommended path)
- Bioconda: `conda install bioconda::cellprofiler` (4.2.8.1 and prior 4.2.x); wiki:
  https://github.com/CellProfiler/CellProfiler/wiki/Conda-Installation
- Standalone GUI builds also distributed from cellprofiler.org.
- **Headless CLI** supported (`cellprofiler -c -r ...`) — required for batch benchmarking.
- Note: Python 3.8 pin means an isolated env (conda/venv) is advisable alongside fluorostats.

**ilastik** (verified):
- Binaries (Win/macOS/Linux; regular + GPU builds from 1.4.0): https://www.ilastik.org/download
- Conda: `ilastik-forge` channel, https://anaconda.org/ilastik-forge/ilastik
- **Headless mode** supported for all workflows except Carving:
  https://www.ilastik.org/documentation/basics/headless
  (e.g. `LAZYFLOW_THREADS=4 LAZYFLOW_TOTAL_RAM_MB=4000 ./run_ilastik.sh --headless ...`)

---

## Open items to verify before publication
- **[UNVERIFIED]** Exact Caicedo 2019 F1 values (0.819 / 0.844 / 0.889 / 0.932) — confirm
  against the full-text PDF (paywalled; try institutional access / the OpenReview PDF:
  https://openreview.net/pdf/04ffd1fb2b6ff6d4841dac0aff149e3a1155e935.pdf).
- **[UNVERIFIED]** Whether an official CellProfiler pipeline for BBBC039 exists, or whether we
  build the IdentifyPrimaryObjects pipeline ourselves.
- **[UNVERIFIED]** ilastik has no single published benchmark F1 to cite; treat as
  configurable front-end, not fixed baseline.
- Confirm current CellProfiler latest (4.2.8 verified; check for any 4.2.9+ at pip time).
