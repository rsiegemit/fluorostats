# Nuclei Segmentation, Counting, and Morphometry in Fluorescence Microscopy

Research dossier for positioning **fluorostats** in a methods paper.

**fluorostats scope (this category):** per-object measurement of segmented nuclei from DAPI/405 channels via **3D connected-component (CC) labeling** with size filtering; per-object volume, equivalent spherical diameter (ESD, µm), object centroids, object density per mm³, and centroid-based spatial homogeneity. Classical, training-free, deterministic, fast.

---

## Reference table

| # | Ref | Year | 2D/3D | Method class | DOI / URL |
|---|-----|------|-------|--------------|-----------|
| 1 | Schmidt, Weigert, Broaddus, Myers — *Cell Detection with Star-Convex Polygons* (StarDist) | 2018 | 2D | DL (star-convex polygon regression) | [10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30) |
| 2 | Weigert, Schmidt, Haase, Sugawara, Myers — *Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy* (StarDist-3D) | 2020 | 3D | DL (star-convex polyhedra) | [10.1109/WACV45572.2020.9093435](https://doi.org/10.1109/WACV45572.2020.9093435) |
| 3 | Stringer, Wang, Michaelos, Pachitariu — *Cellpose: a generalist algorithm for cellular segmentation* | 2021 (online 2020) | 2D + pseudo-3D | DL (flow-field / gradient tracking) | [10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x) |
| 4 | Pachitariu & Stringer — *Cellpose 2.0: how to train your own model* | 2022 | 2D | DL (human-in-the-loop finetuning) | [10.1038/s41592-022-01663-4](https://doi.org/10.1038/s41592-022-01663-4) |
| 5 | Carpenter et al. — *CellProfiler: image analysis software for identifying and quantifying cell phenotypes* | 2006 | 2D (3D later) | Classical pipeline (threshold + distance-transform watershed) | [10.1186/gb-2006-7-10-r100](https://doi.org/10.1186/gb-2006-7-10-r100) |
| 6 | McQuin et al. — *CellProfiler 3.0: Next-generation image processing for biology* | 2018 | 2D + 3D | Classical pipeline + optional DL | [10.1371/journal.pbio.2005970](https://doi.org/10.1371/journal.pbio.2005970) |
| 7 | Caicedo et al. — *Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl* | 2019 | 2D | Benchmark / challenge (DL winners) | [10.1038/s41592-019-0612-7](https://doi.org/10.1038/s41592-019-0612-7) |
| 8 | Malpica et al. — *Applying watershed algorithms to the segmentation of clustered nuclei* | 1997 | 2D | Classical (morphological watershed) | [10.1002/(SICI)1097-0320(19970801)28:4<289::AID-CYTO3>3.0.CO;2-7](https://doi.org/10.1002/(SICI)1097-0320(19970801)28:4%3C289::AID-CYTO3%3E3.0.CO;2-7) |
| 9 | Lin et al. — *A hybrid 3D watershed algorithm incorporating gradient cues and object models for automatic segmentation of nuclei in confocal image stacks* | 2003 | 3D | Classical (3D gradient-weighted watershed) | [10.1002/cyto.a.10079](https://doi.org/10.1002/cyto.a.10079) |
| 10 | Ronneberger, Fischer, Brox — *U-Net: Convolutional Networks for Biomedical Image Segmentation* | 2015 | 2D | DL (encoder-decoder; foundation of StarDist/Cellpose backbones) | [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28) |
| 11 | Ljosa, Sokolnicki, Carpenter — *Annotated high-throughput microscopy image sets for validation* (BBBC) | 2012 | 2D/3D | Benchmark dataset resource | [10.1038/nmeth.2083](https://doi.org/10.1038/nmeth.2083) |
| 12 | Bolte & Cordelières — *A guided tour into subcellular colocalization analysis in light microscopy* (ImageJ **3D Objects Counter**) | 2006 | 3D | Classical (threshold + 3D connected components + per-object stats) | [10.1111/j.1365-2818.2006.01706.x](https://doi.org/10.1111/j.1365-2818.2006.01706.x) |
| 13 | Krupa et al. — *NuMorph: Tools for cortical cellular phenotyping in tissue-cleared whole-brain images* | 2021 | 3D | Hybrid (watershed + 3D DL) | [10.1016/j.celrep.2021.109802](https://doi.org/10.1016/j.celrep.2021.109802) |

Refs 1–12 have DOIs verified via Crossref/dblp. Ref 8 (Malpica) and Ref 13 (NuMorph) DOIs assembled from indexed metadata — see "Citations to double-check" below.

---

## Per-reference notes

**1. StarDist (Schmidt et al., MICCAI 2018).** Predicts, for each pixel, radial distances to the object boundary along a fixed set of rays, yielding a star-convex polygon per nucleus; NMS resolves overlaps. Outperformed Mask R-CNN and U-Net + watershed baselines on crowded DAPI/H&E nuclei. **The reference standard for 2D touching-nuclei separation.** Widely cited (thousands).

**2. StarDist-3D (Weigert et al., WACV 2020).** Extends star-convex polyhedra to volumetric stains. Directly relevant: the DL comparator for fluorostats' 3D DAPI use case. Assumes roughly star-convex (blob-like) nuclei — a good fit for DAPI, a limitation for elongated/lobed nuclei.

**3. Cellpose (Stringer et al., Nature Methods 2021).** Generalist model predicting spatial gradient ("flow") fields that are integrated to recover instance masks; trained on 70,000+ objects. 3D handled by combining 2D flows across orthogonal planes (no 3D-labeled data needed). The other dominant generalist baseline alongside StarDist.

**4. Cellpose 2.0 (2022).** Human-in-the-loop finetuning — a few user-corrected images specialize the generalist model. Relevant framing: DL needs curation/finetuning to hit peak accuracy on a new microscope/stain; fluorostats needs none.

**5. CellProfiler (Carpenter et al., 2006).** Canonical modular bioimage pipeline. `IdentifyPrimaryObjects` = threshold → distance transform → seeded watershed to split touching nuclei, then per-object morphometry. The classical workhorse fluorostats most resembles conceptually (though CP adds watershed splitting that fluorostats' pure CC labeling does not).

**6. CellProfiler 3.0 (McQuin et al., 2018).** Adds native 3D volumetric processing and optional DL plugins. A fair "classical-with-3D" comparator.

**7. 2018 Data Science Bowl (Caicedo et al., 2019).** 3,891 teams; 37,333 hand-annotated nuclei across 841 2D images / 30+ experiments. Established that DL generalizes across microscopes without per-image parameter tuning — the empirical case that DL beats hand-tuned classical pipelines on heterogeneous data. **Key public benchmark dataset (DSB2018).**

**8. Malpica et al. (1997).** Foundational watershed-for-clustered-nuclei paper. Cite to represent the classical touching-nuclei-splitting lineage that fluorostats deliberately does *not* implement.

**9. Lin et al. (2003).** 3D gradient-weighted watershed on confocal nuclei stacks — the 3D classical analogue and the historically standard alternative to plain 3D CC labeling.

**10. U-Net (Ronneberger et al., 2015).** Architectural foundation underpinning StarDist and Cellpose. Cite for lineage/context, not as a direct nuclei-counter.

**11. BBBC (Ljosa et al., 2012).** The Broad Bioimage Benchmark Collection: ground-truth nucleus counts, outlines, and foreground masks (e.g. BBBC039 fluorescent nuclei, BBBC024 synthetic 3D nuclei). **Primary source of public ground truth for a fluorostats benchmark.**

**12. ImageJ 3D Objects Counter (Bolte & Cordelières, 2006).** *The closest published analogue to fluorostats:* threshold a 3D stack, run 3D connected-component labeling, and report per-object volume, centroid, and count. fluorostats is essentially a modernized, statistics-integrated 3D Objects Counter. **Cite as the direct methodological antecedent and a natural head-to-head comparator.**

**13. NuMorph (Krupa et al., 2021).** 3D cell counting/phenotyping in tissue-cleared whole mouse brain (cf. ClearMap, Renier 2016). Represents the cleared-tissue 3D counting application domain; combines watershed with 3D DL. Good to cite for the "cleared tissue / large-volume 3D" use case fluorostats targets.

---

## How fluorostats compares (honest assessment)

**Where fluorostats is adequate or advantageous:**
- **Discrete, well-separated DAPI/405 nuclei:** 3D CC labeling gives correct counts, and does so deterministically, with no training, no GPU, and near-instant runtime.
- **Reproducibility:** deterministic output (no stochastic weights, no model-version drift) — an underrated advantage for a methods pipeline.
- **Integrated downstream analytics:** unlike StarDist/Cellpose (which output masks and stop), fluorostats bundles per-object **volume + ESD size distributions**, **density per mm³**, **centroid-based spatial homogeneity**, and **statistics** in one pass. Its peer is the ImageJ 3D Objects Counter (Ref 12), which it extends with distributions and stats.
- **No annotation burden:** DL needs labeled training data or at least finetuning (Ref 4) to reach peak accuracy on a new stain/microscope.

**Where DL clearly wins:**
- **Touching / overlapping nuclei.** Pure CC labeling **merges** contacting nuclei into one component → **under-counting** in dense tissue. This is exactly the failure StarDist (1,2) and Cellpose (3) were built to solve, and the DSB2018 result (7) is the empirical proof they generalize.
- **Heterogeneous imaging** (varying SNR, uneven illumination, different microscopes) where a single global threshold + size filter is brittle; DL is far more robust (7).
- **Non-blob morphologies** and low-contrast boundaries.

**Fair positioning statement:** fluorostats is not a competitor to StarDist/Cellpose on the segmentation step for crowded nuclei; it is a **fast, training-free, deterministic, statistics-integrated per-object quantifier** for well-separated 3D nuclei — with the honest caveat that it under-counts touching nuclei, and can/should accept masks from StarDist/Cellpose as an alternative front-end for dense samples.

---

## Proposed benchmarks

**B1 — Counting agreement vs manual ground truth (headline benchmark).**
On the **same DAPI z-stacks**, compare nucleus counts from (a) fluorostats CC labeling, (b) StarDist-3D, (c) Cellpose (3D mode), and (d) expert manual counts. Report per-image agreement (Bland–Altman bias + limits, ICC, % error vs manual). **Stratify by local nuclear density** (sparse / medium / crowded) to quantify precisely where fluorostats begins to under-count from touching-nucleus merges. *Expected and honest result: near-parity with DL and manual in sparse fields; growing negative bias (under-count) for fluorostats as density rises, with DL tracking manual better.* This single stratified plot is the paper's most credible and honest figure.

**B2 — Public 2D nuclei benchmark for cross-method calibration.**
Run all methods on **DSB2018** (Ref 7) and **BBBC039** (fluorescent nuclei, Ref 11), reporting the standard mean average precision over IoU thresholds (AP@[0.5:0.95]) and F1. fluorostats' 2D-slice CC labeling will trail StarDist/Cellpose here — report it honestly to bound the touching-nuclei gap on labeled data.

**B3 — Synthetic 3D ground truth for morphometry validation.**
Use **BBBC024** (synthetic 3D HL60 nuclei with known counts/volumes, Ref 11) to validate fluorostats' **volume and ESD** measurements against ground truth (not just counts) — the metrics StarDist/Cellpose don't natively provide, isolating fluorostats' distinctive contribution.

**B4 — Direct antecedent comparison.**
Head-to-head vs **ImageJ 3D Objects Counter** (Ref 12) on identical stacks: confirm count/volume equivalence (validating fluorostats' core), then show the added value (size distributions, density/mm³, spatial homogeneity, statistics) that fluorostats layers on top.

**Public datasets to use:** DSB2018 (`github.com/carpenterlab/2019_caicedo_dsb`); BBBC (`bbbc.broadinstitute.org`) — BBBC039 (2D fluorescent nuclei w/ masks), BBBC024 (synthetic 3D nuclei w/ ground-truth counts & volumes), BBBC034/BBBC050 (3D nuclei).

---

## Citations to double-check before submission

- **Ref 8 (Malpica et al., 1997)** — *Cytometry* 28(4):289–297. The compound legacy Wiley DOI (`10.1002/(SICI)...`) resolves but should be confirmed on Wiley; author list and title verified via secondary sources, not the publisher page directly.
- **Ref 13 (NuMorph, Krupa et al., 2021)** — *Cell Reports* 37(2):109802; DOI from Cell Press metadata — confirm exact author order and article number on the publisher page.
- Refs 1–7, 10–12 DOIs verified via Crossref / dblp during this search. StarDist (1), StarDist-3D (2), Cellpose (3), DSB2018 (7), U-Net (10) confirmed exact.
- **ClearMap (Renier et al., 2016, *Cell* 165:1789–1802, DOI 10.1016/j.cell.2016.05.007)** appeared in searches as the canonical cleared-tissue counting tool and is a strong optional 14th reference — verify before adding.
