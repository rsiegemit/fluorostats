# Vascular / Endothelial / Tubular Network Quantification Tools

Research for positioning **fluorostats** in a methods paper. Category: vascular, endothelial, and tubular network quantification — 2D tube-formation-assay tools through 3D vascular morphometry.

**fluorostats network metrics (reference):** skeleton total length; number of branches; number of junctions; mean branch length; connectivity (connected components, largest-connected-component fraction, Euler number); FOV-normalized densities (length per mm³, junctions per mm³) that are voxel-size invariant. Applied to endothelial cells in 3D bioprinted constructs.

All DOIs below were confirmed against PubMed/PMC/publisher pages during this search. Citation counts are omitted where not reliably visible; where noted they are approximate and should be re-checked at submission.

---

## Reference table

| # | Tool | Citation | DOI / URL | Dim | Open? | Key network metrics |
|---|------|----------|-----------|-----|-------|---------------------|
| 1 | **AngioTool** | Zudaire E, Gambardella L, Kurcz C, Vermeren S. *A Computational Tool for Quantitative Analysis of Vascular Networks.* PLoS ONE 2011;6(11):e27385. | [10.1371/journal.pone.0027385](https://doi.org/10.1371/journal.pone.0027385) | 2D | Yes (GUI, free binary) | Vessel area, vessel % area (density), total vessel length, total junctions, junction/branching index (per area), avg vessel length, total endpoints, lacunarity |
| 2 | **Angiogenesis Analyzer for ImageJ** | Carpentier G, Berndt S, Ferratge S, Rasband W, Cuendet M, Uzan G, Albanese P. *Angiogenesis Analyzer for ImageJ — comparative morphometric analysis of Endothelial Tube Formation Assay and Fibrin Bead Assay.* Sci Rep 2020;10:11568. | [10.1038/s41598-020-67289-8](https://doi.org/10.1038/s41598-020-67289-8) | 2D | Yes (ImageJ macro) | Nodes, junctions, master/segments/branches, meshes, total & mean segment length, total tube length, branching intervals |
| 3 | **REAVER** | Corliss BA, Doty RW, Mathews C, Yates PA, Zhang T, Peirce SM. *REAVER: A program for improved analysis of high-resolution vascular network images.* Microcirculation 2020;27(5):e12618. | [10.1111/micc.12618](https://doi.org/10.1111/micc.12618) | 2D | Yes (MATLAB, BSD-3) | Vessel length density, vessel area fraction, mean vessel diameter, branchpoint count, tortuosity |
| 4 | **RAVE** | Seaman ME, Peirce SM, Kelly K. *Rapid Analysis of Vessel Elements (RAVE): A Tool for Studying Physiologic, Pathologic and Tumor Angiogenesis.* PLoS ONE 2011;6(6):e20807. | [10.1371/journal.pone.0020807](https://doi.org/10.1371/journal.pone.0020807) | 2D | Yes (MATLAB GUI) | Vessel volume fraction, vessel length density, fractal dimension (tortuosity), vessel radii |
| 5 | **Q-VAT** | Callewaert B, Gsell W, et al. *Q-VAT: Quantitative Vascular Analysis Tool.* Front Cardiovasc Med 2023;10:1147462. | [10.3389/fcvm.2023.1147462](https://doi.org/10.3389/fcvm.2023.1147462) | 2D (whole-slide, tiled) | Yes (Fiji macro) | Vessel density, length, diameter-stratified macro/micro vessels, junctions, multi-stain overlap %, tile-wise batch |
| 6 | **VesselExpress** | Spangenberg P, Hagemann N, et al. *Rapid and fully automated blood vasculature analysis in 3D light-sheet image volumes of different organs.* Cell Rep Methods 2023;3(3):100436. | [10.1016/j.crmeth.2023.100436](https://doi.org/10.1016/j.crmeth.2023.100436) | **3D** | Yes (Snakemake pipeline, GitHub) | Vessel diameter, length, branchpoints, segment count + 2 more (6 params total); segmentation→skeleton→graph→render, batch/parallel, ~100× faster than prior tools |
| 7 | **VesSAP** | Todorov MI, Paetzold JC, ... Ertürk A. *Machine learning analysis of whole mouse brain vasculature.* Nat Methods 2020;17:442–449. | [10.1038/s41592-020-0792-1](https://doi.org/10.1038/s41592-020-0792-1) | **3D** | Yes (CNN pipeline) | Deep-learning vessel segmentation; length, radius, branching density mapped to Allen brain atlas |
| 8 | **VESNA** | Schüttler et al. *VESNA: an open-source tool for automated 3D vessel segmentation and network analysis.* BMC Bioinformatics 2025;26:254. | [10.1186/s12859-025-06270-6](https://doi.org/10.1186/s12859-025-06270-6) | **3D** | Yes (Fiji macro, GitHub) | 3D segmentation + skeletonization, batch; vessel length, branches, junctions on 3D fluorescence stacks |
| 9 | **3D morphometric descriptors (cleared organs)** | Vittori et al. *3D imaging and morphometric descriptors of vascular networks on optically cleared organs.* iScience 2023;26(10):107873. | [10.1016/j.isci.2023.107873](https://doi.org/10.1016/j.isci.2023.107873) | **3D** | Method paper | Network-level: density, **connectivity**, fractal dimension; segment-level: length, diameter, tortuosity |

**Bonus / secondary:** AngioQuant (older 2D tube-assay), WimTube/WIMASIS (commercial 2D web), and the review *A Comprehensive Look at In Vitro Angiogenesis Image Analysis Software* (Int J Mol Sci 2023;24:17625, [10.3390/ijms242417625](https://doi.org/10.3390/ijms242417625)) — a useful landscape citation comparing 2D tube-formation packages.

---

## Where fluorostats overlaps and where it adds value

**Overlap (fluorostats reproduces the established 2D metric set):** branches, junctions, total skeleton length, mean branch length. These map directly onto AngioTool (junctions, total/avg length), Angiogenesis Analyzer (nodes/junctions/segments/branches, tube length), and REAVER (branchpoint count, length density). fluorostats can therefore be validated head-to-head against these on 2D projections.

**Where fluorostats adds value:**
1. **True 3D skeletonization.** AngioTool, Angiogenesis Analyzer, REAVER, RAVE, and Q-VAT are all **2D** (Q-VAT is 2D whole-slide even though it scales in-plane). fluorostats operates natively in 3D, like VesselExpress / VesSAP / VESNA, but is aimed at in-vitro 3D bioprinted constructs rather than cleared whole organs or brain atlases.
2. **Explicit connectivity/topology.** fluorostats reports connected components, largest-connected-component fraction, and Euler number. Most established tools stop at counts of junctions/branches; only the cleared-organ descriptor work and graph-based 3D pipelines report connectivity explicitly. This distinguishes a fragmented network from a well-anastomosed one — critical for perfusability in bioprinted tissue.
3. **FOV-normalized, voxel-size-invariant densities.** length per mm³ and junctions per mm³ make measurements comparable across digital zoom / voxel size / FOV heterogeneity. Established 2D tools normalize per unit *area* (AngioTool branching index, REAVER length density) and are sensitive to acquisition settings; explicit voxel-invariant volumetric normalization is uncommon and directly targets a real reproducibility problem.

**Where established tools are more specialized (be honest about this):**
- **REAVER** — best-in-class *accuracy* on 2D high-res images (ground-truth benchmarked; large error reductions vs AngioTool/RAVE) and reports **vessel diameter**, which fluorostats does not emphasize.
- **VesSAP / VesselExpress** — mature 3D, atlas registration (VesSAP) and extreme throughput (~100×; VesselExpress), plus vessel diameter. fluorostats is not a whole-organ/whole-brain tool.
- **Q-VAT** — whole-slide tiling and multi-stain colocalization.
- **Angiogenesis Analyzer** — the de-facto standard for tube-formation-assay morphology, with the richest 2D segment taxonomy (meshes, master segments).

**One-line positioning:** fluorostats is not trying to beat REAVER on 2D diameter accuracy or VesSAP on whole-brain scale; it fills the gap for **reproducible 3D network topology (connectivity + Euler number) with voxel-invariant density normalization in in-vitro bioprinted endothelial constructs** — a regime where 2D tools cannot operate and heavyweight cleared-organ pipelines are overkill.

---

## Proposed benchmarks (head-to-head)

**B1 — 2D concordance (strongest, do this first).** Take the same set of endothelial network images (tube-formation assay or max-intensity Z-projections of the bioprinted constructs). Run **AngioTool** and **Angiogenesis Analyzer** as the reference 2D tools, and run fluorostats on the same 2D images. Compare, per image:
- total junctions (fluorostats vs AngioTool "total junctions" vs Angiogenesis Analyzer "junctions"),
- number of branches / segments,
- total skeleton length.

Report Pearson/ICC correlation and Bland–Altman agreement. **Expected result / claim to test:** fluorostats matches the established tools on 2D (high correlation, small bias), establishing measurement validity before extending the same skeleton/graph pipeline to 3D. Flag explicitly whether fluorostats "catches more" junctions — its skeletonization may detect junctions that 2D thresholding merges or misses; quantify this as a systematic offset rather than asserting it.

**B2 — 2D→3D extension.** On the same constructs, compare the 2D-projection metrics against fluorostats' native 3D metrics to show what 3D recovers: junctions and branches hidden by Z-projection overlap, and connectivity/Euler-number information that has no 2D analog. This demonstrates the added value directly on shared samples.

**B3 — Cross-tool 3D sanity check (optional).** If a 3D fluorescence stack is available, run **VESNA** or **VesselExpress** alongside fluorostats and compare total length, branchpoint count, and segment count. Purpose is concordance on 3D primitives, not to claim superiority — the differentiator is connectivity metrics and voxel-invariant normalization, which those pipelines report differently or not at all.

**B4 — Voxel-invariance demonstration.** Re-image or downsample the same construct at different voxel sizes / digital zooms and show that fluorostats' per-mm³ densities stay stable while per-area or per-voxel counts drift. This is a self-contained figure that no 2D competitor can produce.

---

## Citation-verification notes

- DOIs for **AngioTool (10.1371/journal.pone.0027385)** and **REAVER (10.1111/micc.12618)** were directly confirmed by fetching the PMC full-text pages, including author lists and metric lists.
- AngioTool: RA Robert-Moreno / S. Vermeren author line confirmed as Zudaire, Gambardella, Kurcz, Vermeren.
- **VesselExpress** DOI (10.1016/j.crmeth.2023.100436), **VesSAP** (10.1038/s41592-020-0792-1), **VESNA** (10.1186/s12859-025-06270-6), **Q-VAT** (10.3389/fcvm.2023.1147462), **RAVE** (10.1371/journal.pone.0020807), **Angiogenesis Analyzer** (10.1038/s41598-020-67289-8), and **iScience cleared-organ descriptors** (10.1016/j.isci.2023.107873) come from search-result metadata and matching PubMed/PMC/publisher landing pages, but were not each individually fetched — **verify these seven DOIs and full author lists before submission.**
- The **iScience 2023** cleared-organ paper and **Q-VAT** author lists were only partially captured in search; complete the author strings from the DOI before citing.
- VesSAP full author list is long (Todorov, Paetzold, ... Ertürk) — pull the complete list from Nature Methods before submission.
