# 08 — Spatial Homogeneity / Distribution Uniformity / Dispersion Metrics

Research for positioning **fluorostats** in a methods paper.

**fluorostats recap:** quantifies spatial uniformity of cell/signal distribution *without segmentation*, by tiling the field into an 8×8 grid and computing the **Gini coefficient** and **coefficient of variation (CV)** of per-tile signal; also centroid-based Gini/CV for objects. Tests claims like "cells are more homogeneously distributed in material A."

All references below were verified against Crossref / publisher metadata (DOIs confirmed). None are invented.

---

## Verified references

### A. Gini / CV / dispersion indices applied to imaging (closest prior art)

**1. Martin, Zhang, Williamson, Tingley, Pickus, Zurakowski, Nia, Shirihai, Han, Grinstaff (2026). "Generalizable, high-throughput image analysis of subcellular structures using dispersion indices." *iScience*. DOI: 10.1016/j.isci.2026.115371**
- Metric: **Gini coefficient, coefficient of variation, Theil's L (GE(0)), Theil's T (GE(1))** computed directly on pixel intensities — "substituting pixel intensity for income and number of pixels for population."
- **This is the single closest published concept to fluorostats:** economic-inequality indices applied to raw image intensity, explicitly *segmentation-free*, used to quantify diffuseness vs aggregation (autophagic puncta, mitochondrial clustering, microtubule dynamics). Fluorostats differs by working at the **tile** level (8×8 spatial grid) rather than whole-cell pixel pools, giving it a spatial/lateral-uniformity reading rather than a per-cell concentration reading. Cite as the direct methodological ancestor and contrast the spatial-tiling contribution.

**2. Cai, Chatelet, Howlin, Wang, Webb (2019). "A novel application of Gini coefficient for the quantitative measurement of bacterial aggregation." *Scientific Reports* 9. DOI: 10.1038/s41598-019-55567-z**
- Metric: **discrete Gini coefficient** ("aggregation coefficient") describing how compact vs scattered bacterial aggregates are; designed for high-throughput comparison across samples.
- Establishes precedent for Gini as an aggregation/dispersion readout in cell biology. Same 0 (even) → 1 (concentrated) interpretation fluorostats uses. Cite to justify Gini's biological validity.

**3. Lechthaler, Pauly, Mücklich (2020). "Objective homogeneity quantification of a periodic surface using the Gini coefficient." *Scientific Reports* 10. DOI: 10.1038/s41598-020-70758-9**
- Metric: **Gini coefficient** to objectively score surface homogeneity of a periodic (laser-patterned) material surface.
- Materials-science precedent for "homogeneity via Gini," directly parallel to fluorostats' "cells more homogeneous in material A" use case. Cite for cross-domain generality of Gini-as-homogeneity.

### B. Cell-seeding / bioprinting uniformity (the target application domain)

**4. Liu, Xu (2023). "Improving Uniformity of Cell Distribution in Post-Inkjet-Based Bioprinting." *J. Manufacturing Science and Engineering* 146(1):014501. DOI: 10.1115/1.4063134**
- Metric: post-printing cell distribution uniformity within microspheres/sheets; shows uniform distribution improves viability and proliferation.
- Motivates *why* uniformity matters (biological outcome depends on it). Good "stakes" citation; fluorostats offers a standardized metric this kind of study currently lacks.

**5. Reynolds, Rasmussen, Hansson, Dufva, Riehle, Gadegaard (2018). "Controlling fluid flow to improve cell seeding uniformity." *PLOS ONE* 13. DOI: 10.1371/journal.pone.0207211**
- Metric: **Mean Absolute Error (MAE)** of observed vs target cell density across the culture area (DAPI + CellProfiler segmentation). MAE=0 is ideal.
- A representative *segmentation-dependent* uniformity metric. Contrast: fluorostats needs no per-cell detection (no CellProfiler pipeline), avoiding segmentation error propagation.

**6. Thevenot, Nair, Dey, Yang, Tang (2008). "Method to analyze three-dimensional cell distribution and infiltration in degradable scaffolds." *Tissue Engineering Part C: Methods* 14. DOI: 10.1089/ten.tec.2008.0221**
- Metric: fluorescence staining + cryosectioning; **horizontal (x–y) distribution and vertical (z) penetration depth**.
- Classic scaffold cell-distribution workflow. Fluorostats' lateral Gini/CV is a lightweight, quantitative alternative to qualitative "horizontal distribution" assessment.

**7. Di Stolfo, Lee, Vanhecke, Balog, Taladriz-Blanco, Petri-Fink, Rothen-Rutishauser (2025). "The impact of cell density variations on nanoparticle uptake across bioprinted A549 gradients." *Frontiers in Bioengineering and Biotechnology* 13. DOI: 10.3389/fbioe.2025.1584635**
- Metric: nuclei counts per field of view (Imaris "Spots"); gradient stability judged by **visual inspection of surface plots** — *no explicit uniformity index*.
- Illustrates the gap: modern bioprinting papers still assess uniformity qualitatively. Direct opening for fluorostats to supply an objective scalar (Gini/CV) here.

### C. Point-pattern / spatial statistics (rigorous comparators)

**8. Jafari-Mamaghani, Andersson, Krieger (2010). "Spatial Point Pattern Analysis of Neurons Using Ripley's K-Function in 3D." *Frontiers in Neuroinformatics* 4. DOI: 10.3389/fninf.2010.00009**
- Metric: **Ripley's K-function** (2D→3D) with edge correction; detects clustering vs dispersion across a range of radii.
- The canonical rigorous alternative. Ripley's K is multiscale and statistically principled but **requires point coordinates (segmentation)** and a defined null model. Fluorostats trades that rigor for segmentation-free simplicity on raw intensity. Best head-to-head benchmark target (see below).

**9. Jiao, Berman, Kiehl, Torquato (2011). "Spatial Organization and Correlations of Cell Nuclei in Brain Tumors." *PLOS ONE* 6. DOI: 10.1371/journal.pone.0027323**
- Metric: **pair correlation function g(r), structure factor S(k), nearest-neighbor functions F(r)/G(r)** on segmented nuclei.
- State-of-the-art point-pattern characterization of cell nuclei. Again segmentation-dependent and coordinate-based; contrast with fluorostats' pixel/tile approach.

### D. Lacunarity / quadrat multiscale heterogeneity (foundational)

**10. Plotnick, Gardner, Hargrove, Prestegaard, Perlmutter (1996). "Lacunarity analysis: A general technique for the analysis of spatial patterns." *Physical Review E* 53:5461. DOI: 10.1103/PhysRevE.53.5461**
- Metric: **lacunarity via the gliding-box algorithm** — multiscale dispersion; explicitly framed as an improvement over the single-scale **variance:mean ratio of quadrat counts** because it examines dispersion across a range of scales.
- Key theoretical anchor for the **tile-size / scale-dependence limitation** of fluorostats (an 8×8 grid is a single quadrat scale). Cite to acknowledge fluorostats' fixed-scale limitation and to point at lacunarity as the principled multiscale extension.

**(Foundational companion, verified):** Allain & Cloitre (1991). "Characterizing the lacunarity of random and deterministic fractal sets." *Physical Review A* 44:3552. DOI: 10.1103/PhysRevA.44.3552 — origin of the gliding-box lacunarity method; cite alongside Plotnick for the method's provenance.

---

## Positioning: fluorostats vs the landscape

| Approach | Input needed | Scale | Rigor | Interpretability |
|---|---|---|---|---|
| **fluorostats (Gini/CV over 8×8 tiles)** | raw intensity, **no segmentation** | single (tile) | descriptive scalar | high (0–1 Gini, familiar CV) |
| Dispersion indices on pixels (Martin 2026) | raw intensity, no segmentation | whole-cell/whole-image | descriptive | high |
| MAE vs target (Reynolds 2018) | **segmented** cell density | single | descriptive | needs target density |
| Ripley's K (Jafari-Mamaghani 2010) | **point coordinates** | multiscale (radius) | statistical (null model) | moderate |
| g(r)/S(k)/NN (Jiao 2011) | **point coordinates** | multiscale | statistical | moderate/low |
| Lacunarity (Plotnick 1996) | binary or intensity | **multiscale** | descriptive | moderate |

**fluorostats' advantages:**
- Works on **raw fluorescence intensity** — no cell segmentation, no detection-error propagation, no per-object pipeline (unlike Ripley's K, g(r), NN, MAE approaches which all need coordinates or counts).
- **Interpretable scalars** (Gini 0–1, CV %) already familiar to biologists; directly supports "material A vs B" hypothesis tests.
- Fast, high-throughput, minimal parameters.

**fluorostats' limitations (state honestly):**
- **Single fixed scale** (8×8 tiling) — cannot separate fine- from coarse-scale heterogeneity the way lacunarity (Plotnick 1996) or Ripley's K (multiscale radii) can. Tile-size dependence is the main threat to validity.
- **Not a formal point-pattern statistic** — no null model / significance test for complete spatial randomness; it's a descriptive dispersion score, not an inference against CSR.
- Intensity-based, so it conflates "more cells" with "brighter cells" unless normalized; centroid-mode mitigates this but reintroduces a detection step.

---

## Proposed benchmarks

**Strongest benchmark — synthetic control gradient + concordance with Ripley's K / NN index:**
1. Generate synthetic fields with **known ground-truth uniformity**: a parametric series from perfectly uniform (jittered lattice / Poisson-disk) → complete spatial randomness (Poisson) → clustered (Thomas/Matérn cluster process), sweeping a clustering parameter. Render each as a fluorescence image (Gaussian blobs at each point) *and* keep the exact coordinates.
2. On the **same images**, compute (a) fluorostats lateral Gini and CV over the 8×8 grid, (b) Ripley's K (or its L-function deviation) and the average **nearest-neighbor index (NNI, R = observed/expected NN distance)** from the true coordinates.
3. **Show monotonic concordance:** fluorostats Gini/CV should rise monotonically with the clustering parameter and correlate strongly (report Spearman ρ) with Ripley's K peak deviation and with (1−NNI). Demonstrate fluorostats **separates known-uniform from known-clustered** controls with high effect size (Cliff's δ / AUC).

**Supporting benchmarks:**
- **Tile-size sensitivity sweep** (4×4, 8×8, 16×16, 32×32) to characterize the scale-dependence limitation quantitatively — turns the weakness into a documented operating range, and connects to lacunarity's multiscale argument (Plotnick 1996).
- **Segmentation-robustness:** add Poisson/shot noise and blur; show fluorostats' intensity Gini/CV degrades more gracefully than segmentation-dependent MAE / Ripley's K (whose detection step fails first).
- **Cross-method agreement on real data:** on a real bioprinted/seeded-scaffold dataset (cf. Liu & Xu 2023; Di Stolfo 2025), rank samples by fluorostats and by an Imaris/CellProfiler-derived NN index; report rank concordance to argue fluorostats reproduces the expensive pipeline's conclusions at a fraction of the effort.

---

## 200-word summary

The spatial-homogeneity literature splits into two camps. **Descriptive intensity/dispersion metrics** — Gini, CV, and related inequality indices — have a solid, growing precedent in biological imaging: Martin et al. (2026, *iScience*) apply Gini/CV/Theil directly to pixel intensities *without segmentation* (the closest prior art to fluorostats); Cai et al. (2019) and Lechthaler et al. (2020) use Gini for bacterial aggregation and surface homogeneity respectively. **Rigorous point-pattern statistics** — Ripley's K (Jafari-Mamaghani 2010), pair-correlation/nearest-neighbor functions (Jiao 2011), and multiscale lacunarity (Plotnick 1996; Allain & Cloitre 1991) — are more principled but require segmented coordinates and a null model. Cell-seeding/bioprinting studies (Reynolds 2018; Liu & Xu 2023; Di Stolfo 2025) still assess uniformity via segmentation-dependent MAE or, tellingly, *visual inspection with no index at all* — the gap fluorostats fills. Fluorostats' edge: interpretable Gini/CV on **raw intensity, no segmentation**, high-throughput. Its honest limits: **single fixed tile scale** (vs multiscale lacunarity/Ripley's K) and no CSR significance test.

**Strongest benchmark:** on synthetic uniform→clustered control images with known coordinates, show fluorostats' tile Gini/CV rises monotonically with clustering and correlates (Spearman ρ) with Ripley's K deviation and (1−nearest-neighbor index), cleanly separating known-uniform from known-clustered fields.

---

## Citation-verification notes
- All 10 primary references (+Allain & Cloitre companion) verified via Crossref/publisher metadata; **DOIs confirmed**. No fabricated citations.
- Martin et al. iScience carries a **2026** date (DOI 10.1016/j.isci.2026.115371) per publisher metadata — flag for double-check of final page/volume, but DOI resolves and authorship is confirmed.
- ASME paper (Liu & Xu) landing page returns HTTP 403 to automated fetch; metadata (authors, title, DOI 10.1115/1.4063134) confirmed independently via Crossref. Not unverifiable — just paywalled to scraping.
