# Volume Fraction, Area Coverage, Porosity, and Stereological Quantification

Research category for positioning **fluorostats** in a methods paper. fluorostats computes
volume fraction (fraction of voxels containing signal) for 3D confocal, area fraction / coverage
for 2D fluorescence, and FOV-normalized densities (per mm³) to make metrics voxel-size / digital-zoom
invariant across acquisitions.

All citations below were verified by fetching the source (PMC / publisher / PubMed) unless flagged.

---

## 1. Foundations of stereology: point counting and the Delesse principle

### 1.1 Delesse (1848) / Delesse–Glagolev principle — area fraction ≈ volume fraction
- **Concept, not a single fetched DOI.** Delesse (1848) established that on a random 2D section the
  areal fraction of a phase is an unbiased estimator of its volume fraction (A_A = V_V). Glagolev (1933)
  reduced this to *point counting*: the fraction of grid points falling on a phase estimates its area
  (and hence volume) fraction (P_P = A_A = V_V). This is the theoretical bedrock for **every**
  voxel-counting or pixel-counting volume/area-fraction measurement, including fluorostats.
- **Why it matters for fluorostats:** fluorostats' "fraction of voxels containing signal" is the
  digital, exhaustive-sampling limit of Delesse–Glagolev point counting — instead of a sparse test grid,
  *every voxel is a test point*. This makes voxel counting a valid, unbiased estimator of volume fraction
  **provided segmentation is unbiased** (the caveat carried through the rest of this document).
- **Flag:** The 1848 Delesse and 1933 Glagolev primary sources were not individually fetched
  (pre-digital originals). The principle is uncontroversial and restated in every reference below;
  cite via a modern review (Marcos et al. 2012; Kubínová & Janáček 2015) rather than the originals if a
  verified DOI is required.

### 1.2 Gundersen & Jensen (1987) — efficiency of systematic sampling; the Cavalieri estimator
- **Authors/Year/Title:** Gundersen HJG, Jensen EB. *The efficiency of systematic sampling in stereology
  and its prediction.* **Journal of Microscopy**, 1987; 147(3):229–263.
- **DOI:** 10.1111/j.1365-2818.1987.tb02837.x
- **Method:** Establishes systematic uniform random (SUR) sampling and the Cavalieri volume estimator
  (V = T · ΣA), plus the Gundersen–Jensen coefficient-of-error (CE) prediction for how many sections/points
  are needed. Seminal, thousands of citations.
- **Relation to fluorostats:** This is the *gold-standard efficiency framework* fluorostats does **not**
  implement. fluorostats measures A_A / V_V exhaustively per FOV but does not do SUR sampling across the
  organ/construct, nor CE prediction. Cite as the reference against which "unbiased design-based" is defined.
- **Verification:** DOI/volume/pages confirmed via web search of Journal of Microscopy 1987;147:229–263
  (widely reproduced); abstract text matched. Treat volume/page as high-confidence.

### 1.3 Marcos, Monteiro & Rocha (2012) — design-based stereology review with practical guidelines
- **Authors/Year/Title:** Marcos R, Monteiro RAF, Rocha E. *The use of design-based stereology to evaluate
  volumes and numbers in the liver: a review with practical guidelines.* **Journal of Anatomy**, 2012;
  220(4):303–317.
- **DOI:** 10.1111/j.1469-7580.2012.01475.x  (PMC3375768) — **fetched & verified**
- **Method:** Practical Cavalieri point-counting for total volume (V = T · (a/p) · ΣP); notes ~200 points
  over 10–15 sections gives 5–10% CE. Emphasizes design-based methods "guarantee no bias" via systematic
  sampling with equal selection probability, contrasting with model-based assumptions.
- **Relation to fluorostats:** Best single citation for "design-based stereology is the unbiased gold
  standard" and for the point-counting formula fluorostats' voxel counting approximates.

---

## 2. Stereology meets confocal / 3D images (the direct methodological neighbors)

### 2.1 Peterson (2001) — virtual test probes on 3D confocal images
- **Authors/Year/Title:** Peterson DA. *Confocal microscopy and stereology: estimating volume, number,
  surface area and length by virtual test probes applied to three-dimensional images.*
  (in Microscopy Research and Technique / related), 2001. **PubMed 11525261.**
- **Method:** Computer-generated virtual probes (point grids, disector, fakir, slicer) applied to confocal
  optical-section stacks to estimate V, N, S, L without physical sectioning.
- **Relation to fluorostats:** The closest classical analog to fluorostats' domain — stereological probes
  on confocal stacks. fluorostats replaces the sparse virtual point grid with exhaustive voxel counting:
  simpler and fully automated, but forfeits the design-based sampling that guarantees unbiasedness for
  *number*, *surface*, and *length* (fluorostats does not estimate these). For *volume fraction* the two
  converge in the limit.
- **Flag:** Exact journal/volume/pages not fetched (PubMed stub only); DOI not captured. Confirm venue
  (Microsc Res Tech 2001) before citing.

### 2.2 Kubínová & Janáček (2015) — confocal stereology review
- **Authors/Year/Title:** Kubínová L, Janáček J. *Confocal stereology: an efficient tool for measurement of
  microscopic structures.* **Cell and Tissue Research**, 2015; 360(1):13–28.
- **DOI:** 10.1007/s00441-015-2138-3 — **fetched & verified**
- **Method:** Reviews ~30 years of combining stereological probes with confocal optical sectioning to
  estimate volume, number, surface area, length from 3D images.
- **Relation to fluorostats:** Modern review to cite for the state of the art fluorostats simplifies;
  also a clean secondary citation for the Delesse/Cavalieri principles.

---

## 3. Voxel/pixel-based volume fraction & porosity in scaffolds and hydrogels (fluorostats' applied niche)

### 3.1 Riley, Wei, Bao, Cheng, Wilson, Liu, Gong & Segura (2023) — void volume fraction of granular scaffolds ⭐ STRONGEST
- **Authors/Year/Title:** Riley L, Wei G, Bao Y, Cheng P, Wilson KL, Liu Y, Gong Y, Segura T.
  *Void volume fraction of granular scaffolds.* **Small**, 2023; 19(40):e2303466.
- **DOI:** 10.1002/smll.202303466  (PMC10592564) — **fetched & verified**
- **Method:** Fluorescent labeling of particles / dextran-filled void space, confocal z-stacks. VVF computed
  two ways: **(2D) average void area fraction across z-slices** (most common) and **(3D) triangulated mesh
  volume (Imaris)**. **Crucially documents that VVF is highly sensitive to microscope magnification, z-gap,
  number of z-slices, software (Fiji/MATLAB/Imaris), and threshold.**
- **Relation to fluorostats:** *This is the keystone citation.* (a) It validates fluorostats' exact 2D
  method — area fraction per slice averaged over the stack — as the field-standard VVF approach. (b) It
  independently documents the *magnification/zoom sensitivity problem* that fluorostats' FOV-normalization
  (per-mm³ densities) is designed to solve. Frame fluorostats as the tool that directly addresses the
  reproducibility gap this paper identifies.

### 3.2 Jamshidi & Falamaki (2021) — image analysis for hydrogel porosity/heterogeneity
- **Authors/Year/Title:** Jamshidi M, Falamaki C. *Image analysis method for heterogeneity and porosity
  characterization of biomimetic hydrogels.* **F1000Research**, 2021; 9:1461.
- **DOI:** 10.12688/f1000research.27372.2  (PMC8256190) — **fetched & verified**
- **Method:** Python pipeline (contrast normalization, Gaussian/Sobel filtering, adaptive Gaussian local
  thresholding in 25×25 windows, watershed) on cryo-SEM images; quantifies void fraction, pore-size
  distribution, and spatial heterogeneity via KDE density maps. Mean pore Ø ≈ 12.36 µm.
- **Relation to fluorostats:** Parallel open-source, threshold-based porosity workflow — but on **cryo-SEM,
  not confocal**, and 2D only. Good for showing fluorostats fits an established methodological family;
  contrast: fluorostats is 3D-confocal-native and adds FOV normalization for cross-acquisition comparability.
- **Note:** Modality is cryo-SEM (image-analysis porosity), not confocal — cite as porosity-quantification
  precedent, not as a confocal method.

### 3.3 Micro-CT bone volume fraction (BV/TV) — the accepted 3D volume-fraction standard in an adjacent field
- **Representative citations (verified via search, not individually fetched):**
  - Bouxsein ML, Boyd SK, Christiansen BA, Guldberg RE, Jepsen KJ, Müller R. *Guidelines for assessment of
    bone microstructure in rodents using micro-computed tomography.* **J Bone Miner Res**, 2010; 25(7):1468–1486.
    DOI: 10.1002/jbmr.141. (The canonical BV/TV / micro-CT nomenclature standard — **recommend fetching to
    confirm before citing**; widely known.)
- **Concept:** BV/TV (bone volume / total volume) is a threshold-then-count 3D volume fraction from µCT
  voxels — the most standardized voxel-based volume-fraction metric in biomedicine, with community consensus
  on thresholding and reporting.
- **Relation to fluorostats:** Precedent that *voxel-counting volume fraction is a legitimate, standardized,
  reference-grade metric* when acquisition and threshold are controlled. fluorostats does the analogous
  operation for fluorescence-confocal signal. The µCT field's insistence on fixed voxel size/threshold
  reporting is exactly the reproducibility discipline fluorostats' FOV-normalization automates.
- **Flag:** Bouxsein 2010 details from general knowledge + search; **verify DOI/pages by fetching** before
  final inclusion.

---

## 4. 2D fluorescence coverage / confluence / area fraction

### 4.1 Busschots, O'Toole, O'Leary & Stordal (2014) — confluence as Area Fraction ⭐ (2D benchmark ref)
- **Authors/Year/Title:** Busschots S, O'Toole SA, O'Leary JJ, Stordal B. *Non-invasive and non-destructive
  measurements of confluence in cultured adherent cell lines.* **MethodsX**, 2014; 2:8–13.
- **DOI:** 10.1016/j.mex.2014.11.002  (PMC4487325) — **fetched & verified**
- **Method:** Phase-contrast images → ImageJ (16-bit, background subtraction, threshold, watershed,
  analyze-particles) → **Area Fraction (AF)** = surface area covered by cells. Validated against
  hemocytometer counts (Spearman r ≈ 0.99).
- **Relation to fluorostats:** Direct precedent for fluorostats' 2D "area fraction / coverage." Same
  operation (threshold → fraction of covered pixels), validated against an orthogonal ground truth.
  fluorostats generalizes this to fluorescence and adds FOV-normalized density.

### 4.2 Ljosa & Carpenter (2009) — quantitative analysis of 2D fluorescence for cell screening
- **Authors/Year/Title:** Ljosa V, Carpenter AE. *Introduction to the Quantitative Analysis of
  Two-Dimensional Fluorescence Microscopy Images for Cell-Based Screening.* **PLOS Computational Biology**,
  2009; 5(12):e1000603.
- **DOI:** 10.1371/journal.pcbi.1000603 — **fetched & verified**
- **Method:** Tutorial: illumination correction, global/local thresholding, watershed segmentation; measures
  object area ("area of the image occupied by a cell/nucleus/focus") and intensity (mean/integrated).
- **Relation to fluorostats:** Authoritative (CellProfiler team) reference framing area-occupied and
  intensity fraction as standard fluorescence readouts — legitimizes fluorostats' coverage metric and
  situates it relative to CellProfiler.

---

## 5. Comparison: where fluorostats sits, its contribution, and its limits

**Voxel-based volume fraction is standard and valid.**
- Delesse–Glagolev (A_A = V_V = P_P) makes exhaustive voxel counting an unbiased estimator of volume
  fraction *given unbiased segmentation*. µCT BV/TV and confocal VVF (Riley 2023) show the voxel-count
  approach is the accepted practice in real applications. fluorostats is squarely in this tradition.

**Design-based stereology remains the gold standard for unbiased estimation.**
- Gundersen & Jensen (1987), Marcos (2012), Peterson (2001), Kubínová & Janáček (2015): SUR sampling +
  virtual probes give provably unbiased estimates with predictable CE, and uniquely handle number, surface,
  and length. fluorostats does **not** do design-based sampling or CE prediction, and estimates only
  volume/area fraction — not N, S, L. State plainly: fluorostats trades design-based rigor for full
  automation and cross-acquisition comparability, and is a screening/comparative tool, not a replacement
  for gold-standard stereology when absolute unbiased estimates are required.

**Genuine contribution: FOV-normalized density (per mm³) for cross-zoom / cross-voxel-size comparability.**
- Riley 2023 explicitly documents that VVF/void metrics drift with magnification, z-gap, voxel size, and
  software — a known reproducibility hole. fluorostats' per-mm³ FOV normalization directly targets this:
  it makes density metrics invariant to digital zoom and voxel size, so the *same physical sample* yields
  the *same metric* across acquisition settings. This is a real, publishable methodological contribution
  that neither classical stereology tools nor confluence tools package as a default.

**Limits to state honestly.**
- (1) Raw *volume fraction itself* is intensive (dimensionless) and already zoom-invariant in principle;
  the normalization contribution is strongest for *count/number densities* (objects per mm³), where raw
  counts scale with FOV. Be precise about which metrics normalization fixes.
- (2) All voxel counting is only as unbiased as the threshold/segmentation — the shared Achilles' heel
  flagged by Riley 2023, Jamshidi 2021, and the µCT literature. fluorostats inherits threshold sensitivity.
- (3) No design-based sampling → not a substitute for Cavalieri/point-counting when unbiasedness must be
  guaranteed and reported with CE.

---

## 6. Proposed benchmarks

**Benchmark A (validity) — voxel volume fraction vs. stereological point counting on the SAME stacks.** ⭐ STRONGEST
- Take N confocal z-stacks of a labeled 3D construct. On each: (i) fluorostats voxel volume fraction;
  (ii) independent, blinded Cavalieri/Delesse point counting with a SUR test grid (per Marcos 2012;
  Gundersen & Jensen 1987 CE targeting ~5–10%). Report Bland–Altman agreement + Lin's concordance between
  the two. **Success:** no systematic bias and concordance within the stereological CE. This directly
  answers "is fluorostats' fast voxel count a valid stand-in for gold-standard point counting?" — the single
  most convincing result for the paper. Point-counting can be done in the Fiji Stereology plugin for an
  independent, citable comparator.

**Benchmark B (the headline contribution) — FOV-normalization robustness across digital zoom.** ⭐
- Image the *same* physical sample at ≥2 digital zooms (e.g., 1× and 2×) / two voxel sizes, matched region.
  Show **raw** counts / raw per-FOV numbers diverge across zoom (reproducing the Riley 2023 sensitivity),
  while fluorostats' **FOV-normalized per-mm³ densities remain stable** (overlapping CIs across zooms).
  Quantify with %CV of the metric across zoom levels: raw high-CV vs. normalized low-CV. This is the clean,
  visual demonstration of the tool's raison d'être and maps 1:1 onto the reproducibility gap Riley documents.

**Benchmark C (threshold robustness, supporting).**
- Sweep threshold across a reasonable range and report volume-fraction / density stability vs. the point-
  counting ground truth — honestly characterizing the shared segmentation limitation (Riley 2023,
  Jamshidi 2021) rather than hiding it.

---

## 7. Citation quick-list (verification status)

| # | Citation | DOI | Status |
|---|----------|-----|--------|
| 1 | Delesse (1848) / Glagolev (1933) principle | — | Concept; cite via review. Originals NOT fetched. |
| 2 | Gundersen & Jensen 1987, J Microsc 147:229–263 | 10.1111/j.1365-2818.1987.tb02837.x | Search-verified; high confidence |
| 3 | Marcos et al. 2012, J Anat 220:303–317 | 10.1111/j.1469-7580.2012.01475.x | ✅ Fetched |
| 4 | Peterson 2001 (confocal virtual probes) | — | PubMed 11525261; venue/DOI NOT confirmed — flag |
| 5 | Kubínová & Janáček 2015, Cell Tissue Res 360:13–28 | 10.1007/s00441-015-2138-3 | ✅ Fetched |
| 6 | Riley et al. 2023, Small 19:e2303466 | 10.1002/smll.202303466 | ✅ Fetched ⭐ |
| 7 | Jamshidi & Falamaki 2021, F1000Res 9:1461 | 10.12688/f1000research.27372.2 | ✅ Fetched (cryo-SEM) |
| 8 | Bouxsein et al. 2010, JBMR 25:1468–1486 (µCT BV/TV) | 10.1002/jbmr.141 | Search/knowledge — VERIFY before citing |
| 9 | Busschots et al. 2014, MethodsX 2:8–13 | 10.1016/j.mex.2014.11.002 | ✅ Fetched |
| 10 | Ljosa & Carpenter 2009, PLoS Comput Biol 5:e1000603 | 10.1371/journal.pcbi.1000603 | ✅ Fetched |

**Unverifiable / needs confirmation before final citation:** #1 (Delesse/Glagolev primaries — use a modern
review instead), #4 (Peterson 2001 exact venue/DOI), #8 (Bouxsein 2010 DOI/pages).
