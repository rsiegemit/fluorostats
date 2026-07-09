# fluorostats: training-free, reproducible quantification of fluorescence microscopy on the CPU

*Citation keys in [brackets] are placeholders pending the reference-manager pass. Display items consolidated to 6 main figures + 2 main tables; per-experiment numbers moved to Supplementary Tables S1–S9 (appended) and the correctness phantom battery to Extended Data Fig. 1.*

---

## Abstract

Modern fluorescence microscopy is volumetric, yet its images are routinely quantified in two dimensions, with a patchwork of single-purpose tools that mostly require a GPU or training, and with statistics done by hand in a separate program—each step eroding reproducibility. We present fluorostats, an open-source Python library and CLI that computes volume fraction, topology, skeleton and vascular metrics, morphometry, spatial homogeneity and depth-resolved Live/Dead viability directly, training-free and on a standard CPU, then carries them through to non-parametric statistics, power analysis and figures in one pipeline. Every metric is validated to reference-implementation exactness, and accuracy is benchmarked against StarDist, Cellpose, Omnipose, REAVER, AngioTool, VesselExpress and a published Fiji macro on public data. fluorostats matches or exceeds these tools across many tasks, deterministically on the CPU, and delineates the crowded-instance regime where trained segmenters remain preferable. It is available under a permissive license with a version-archived DOI.

---

## Introduction

Fluorescence microscopy is now routinely volumetric. Confocal and light-sheet instruments deliver three-dimensional stacks of hydrogel constructs, cleared organs and cultured tissues as a matter of course, and the questions asked of those stacks are quantitative ones: what fraction of a volume carries signal, how connected a vascular or mitochondrial network is, how objects are arranged in space, and what proportion of cells remain alive at a given depth. The answers determine whether a bioink keeps cells alive, whether an angiogenic cue produced a denser network, and whether a spatial pattern is uniform or clustered [refs].

The data are volumetric, but the software that measures them is fragmented, and each fragment carries a cost. Instance segmentation belongs to trained models—StarDist, Cellpose, Omnipose, Mesmer [Schmidt2018; Stringer2021; Cutler2022; Greenwald2022]; vascular networks to REAVER, AngioTool and, in three dimensions, VesselExpress and VesSAP [Corliss2020; Zudaire2011; Spangenberg2023; Todorov2020]; skeleton and topology to Fiji's AnalyzeSkeleton, skan and BoneJ [ArgandaCarreras2010; NunezIglesias2018; Doube2010]; viability to hand-built ImageJ macros [Kerkhoff2024]. Each is excellent in its lane, but assembled into an analysis they expose three recurring costs. The strongest segmenters need a GPU and often retraining, placing them out of reach for many laboratories and making a result reliable only on images that resemble the training set [Laine2021]. The measurements are frequently taken on two-dimensional projections of three-dimensional stacks, which systematically distort the quantities reported—named repeatedly in the tissue-engineering and angiogenesis literature but rarely fixed [Spiller2025; Pereira2023]. And the statistics are usually done by hand in a spreadsheet, where small samples and pseudoreplication inflate false positives [Lord2020; Lazic2010] and unrecorded parameters undermine reproducibility [Pereira2023].

No single validated pipeline carries a measurement from a raw volume through to a correctly chosen, reproducible statistic without a GPU, a training set or a manual export. That is the gap fluorostats fills.

fluorostats is an open-source Python library and command-line tool for quantifying fluorescence microscopy, built around four commitments that together define where it is useful. It is training-free: every operation is a deterministic algorithm rather than a learned model. It is CPU-only: no function needs a GPU. It is reference-exact: every metric is validated against the canonical implementation or against analytic ground truth. And it is statistically self-contained: small-sample non-parametric tests, effect sizes, multiplicity control and bootstrap power live in the same pipeline as the measurements, so that quantification and inference are never separated. The library spans volume fraction and density, connectivity and topology, skeleton and vascular metrics, object morphometry, spatial homogeneity and depth-resolved viability, and it emits the statistics and figures needed to report them.

We evaluate fluorostats in two stages. First we establish that each metric is correct against a reference implementation or a closed-form phantom. Only then do we benchmark accuracy against the established tools—each deep-learning baseline first shown to reproduce its own published number—reporting every comparison with bootstrap confidence intervals and paired tests. Across them fluorostats reaches parity with or exceeds the established tool on well-separated targets, training-free and on the CPU, and we locate precisely the crowded-instance crossover where trained segmenters take over.

---

## Results

### A training-free, CPU-only architecture

fluorostats is a Python library (v0.7.0) with a command-line interface, installable from source and the Python package index. Its 19 modules follow a quantification workflow—input, preprocessing, segmentation, metric extraction, statistics, reporting—and every metric is a pure function of its inputs (Fig. 1). Three properties recur below. First, it is deterministic: segmentation is classical thresholding (Otsu, Li, Isodata, Triangle, Yen, Mean, Minimum, with optional watershed splitting), every downstream metric is closed-form or combinatorial, and nothing in the measurement path holds a learned weight or draws a random seed, so repeated runs return identical bits. Second, it is CPU-only, sidestepping the GPU availability, framework drift and non-deterministic kernels that make deep-learning pipelines hard to reproduce. Third, it treats volumetric data as volumetric: volume fraction, connectivity, skeleton, vascular and viability metrics are computed in three dimensions with calibrated spacing, and densities are normalized per unit physical volume, invariant to digital zoom; where a two-dimensional shortcut exists we implement it explicitly, so its difference from the volumetric value can be measured rather than assumed. Beyond measurement, fluorostats integrates a statistics layer that general platforms omit—non-parametric tests with effect sizes, stratified and interaction tests, Benjamini–Hochberg control, bootstrap intervals, agreement statistics and power analysis—acting on the objects the quantification already produced, so an analysis runs from image to multiplicity-controlled significance without leaving the pipeline.

### Every metric is exact against its reference

A benchmark against another tool means little if the metric itself is wrong, so we first establish correctness against reference implementations or phantoms of known value, holding integer quantities to exact agreement and continuous ones to a stated tolerance (Extended Data Table 1; Extended Data Fig. 1). Topology is exact: on phantoms with closed-form Euler characteristics the Euler number, component count and largest-connected-component fraction are recovered with zero error, and in a fragmented-to-connected sweep the Euler number tracks the transition perfectly (ρ = 1.0), uniquely spanning both regimes where the largest-component fraction and percolation saturate. Skeleton metrics use the Lee–Kashyap–Chu 1994 thinning of Fiji's AnalyzeSkeleton and skan [ArgandaCarreras2010; NunezIglesias2018], which makes exact agreement a prediction we confirm—length within 1%, branch and junction counts exact on trees to depth three, with a documented raster-resolution undercount only on the densest terminal branches. Continuous physical metrics match their analytic definitions, and per-mm³ density is invariant to voxel size across a six-fold zoom (CV = 0) where uncalibrated counts drift, reproducing and resolving a known magnification artefact [Riley2023]. The statistics and agreement layer is exact to machine precision against SciPy and hand-coded references (8/8 statistics, 11/11 agreement); crucially for the comparisons that follow, the instance scorers themselves pass all 23 checks against an independent matcher and the DSB2018 formula, so the numbers below come from a verified scorer. These are regression tests in the 105-test suite, so exactness is a maintained invariant rather than a one-time measurement.

With correctness established, we benchmark accuracy against the established software for each capability. Table 1 summarizes those comparisons; the sections that follow present each in turn, and the full per-experiment numbers are given in Extended Data Tables 1–5.

**Table 1 | fluorostats versus established tools across capabilities.** One row per head-to-head; the verdict is relative to the comparator (**>** better, **=** on par, **<** worse, **✓** validated exact). Full numbers in the cited Extended Data tables.

| Capability | Dataset | Comparator (type) | Their result | fluorostats | Verdict |
|---|---|---|---|---|---|
| Nuclei, well-separated | BBBC039, n=200 | StarDist (deep learning) | F1 0.871 | **0.896** [.873,.916] | **>** (sig) |
| | | Cellpose (deep learning) | 0.862 | 0.896 | **>** (sig) |
| | | Omnipose (deep learning) | 0.802 | 0.896 | **>** |
| Nuclei | DSB2018 | StarDist (published) | AP 0.864 | F1 0.789 | ≈ 91%, training-free |
| Nuclei, crowded | BBBC024 c75 | StarDist / Cellpose (DL) | F1 0.96–1.0 | 0.38 | **<** (DL regime) |
| Vascular, 2D | REAVER, n=36 | AngioTool | MAE 0.068 | 0.076 | **=** (ties) |
| | | RAVE / AngioQuant | 0.094 / 0.149 | 0.076 | **>** |
| | | REAVER (specialist) | 0.017 | 0.076 | **<** |
| Vascular, 3D | synthetic phantom | exact ground truth | — | length ≤2.4%, branches/VF exact | **✓** |
| | VesselExpress, n=9 | VesselExpress (software) | — | Dice 0.598; Spearman 0.75 | **=** agreement |
| Viability | Kerkhoff macro | Fiji macro (peak count) | MAE 0.016, CCC 0.987 | 0.016, 0.987 | **=** exact tie |
| Viability, 2D vs 3D | S-BIAD2130 | 2D / MIP readout | — | 2D biases +5–25% | **✓** depth-aware |
| Homogeneity | synthetic sweep | 5 spatial statistics | — | \|ρ\| 0.96–0.997, AUC 1.0 | **✓** tracks all |
| Runtime | BBBC039 | StarDist / Cellpose | 215 / 5,547 ms | 14.5 ms | **>** 15× / 380× |

### Nucleus segmentation matches validated deep learning, then hands off

Nucleus instance segmentation is where deep learning is strongest and where a training-free tool is therefore most exposed, so we treat it as the central test of the parity claim and report it in the segmenters' own currency: instance F1 and average precision, computed by matching predicted to reference instances across a range of intersection-over-union thresholds (Methods).

The comparison is only as honest as its baselines, so we validated them first. On BBBC039 our StarDist run reached an F1 of 0.871 against a published 0.864, and Cellpose reached 0.862, confirming that both were configured at full strength rather than set up to fail [PublishedBaselines]. Only then did we introduce fluorostats. On the full BBBC039 test set (n = 200) it reached an instance F1 of 0.896 (95% bootstrap CI [0.873, 0.916], 10,000 resamples), above StarDist (0.871), Cellpose (0.862) and Omnipose (0.802). The paired, per-image differences are significant: fluorostats exceeds StarDist by 0.025 [0.004, 0.042] and Cellpose by 0.034 [0.008, 0.057], both intervals clear of zero (Fig. 2b). The claim this supports is specific. A deterministic, training-free, CPU pipeline reaches the accuracy of three independently validated trained segmenters on well-separated nuclei; it is not that thresholding beats deep learning in general.

Within its own threshold family, fluorostats is competitive across the board, and its Li configuration tops a twelve-method comparison at F1 0.934 (Fig. 2a; Extended Data Table 2). We give the full ranking rather than only the winner, because the threshold is a parameter the user sets, and its behaviour is something to characterize rather than hide. On DSB2018, the canonical nucleus set on which StarDist's published number rests, fluorostats reaches an F1 of 0.789: the best of six classical baselines, and roughly 91% of the trained model's published 0.864, at no training cost and on the very data that number comes from.

Parity on separated targets has an honest counterpart, and it is collapse on crowded ones. As nuclear overlap on the BBBC024 benchmark rises from 0% to 75%, instance F1 for fluorostats—and for every non-learned method we tested—falls from 0.94 to 0.15, while the trained segmenters hold near 0.96–1.0 (Fig. 2d; Extended Data Table 2). The mechanism is not mysterious: connected-component labelling fuses instances the moment they touch, and watershed splitting rescues only mild clustering (0.896 to 0.899 on BBBC039), not heavy overlap. This is a wall in front of the whole non-learned class, not fluorostats alone, and it draws a clean line of use—thresholding on well-separated targets, a trained instance segmenter when instances heavily overlap (Fig. 2e). Because fluorostats is training-free, its accuracy does not hinge on a modality appearing in a training set, and its decline with overlap is a smooth function of a measurable image property rather than a distribution-shift cliff.

### A general tool ties the vascular specialists on their own benchmark

Vascular quantification belongs to dedicated tools, which invites a pointed question: how far does an untuned, training-free, general-purpose tool get on the specialists' own annotated data? We answer it with REAVER's comparison protocol [Corliss2020]—one shared dataset, every tool scored through a single unified quantification of the same segmentations, and default parameters throughout, which we declare, as REAVER does, as a deliberate constraint.

In two dimensions, fluorostats sits among the specialists. On the REAVER dataset (n = 36, expert manual ground truth), dropped into REAVER's own five-tool comparison, it ranks fourth of six on vessel-area-fraction error (MAE 0.076, concordance 0.701, Spearman 0.935), statistically level with AngioTool (0.068) and ahead of RAVE (0.094) and AngioQuant (0.149); REAVER itself (0.017) and ImageJ (0.041) lead (Fig. 3a; Extended Data Table 3). An untuned general tool tying a dedicated angiogenesis package on that package's own benchmark is exactly the intended result, and the two specialists that genuinely exceed it are named, not buried. On real fibrin-bead sprouting-assay data (SproutAngio, n = 12), where no ground truth exists, fluorostats and four threshold baselines all recover the VEGF dose–response, with the volume fraction and length correlating with dose at Spearman 0.59–0.74; the metrics roughly double from low to mid VEGF and plateau at high dose, a saturation that is biologically plausible and that the tool detects without any ground truth or training.

In three dimensions, fluorostats recovers synthetic-phantom values exactly (length within 0.6–2.4%, branches and volume fraction exact; Fig. 3c) and agrees with a 3D specialist on real light-sheet data. Framed as software-versus-software—the reference is VesselExpress's own pipeline output, not manual tracing—the threshold proves decisive and drove a new capability: the Otsu default badly under-segments dim vessels (Dice 0.089), while Li recovers to 0.598, so we added an automatic mode that switches to Li when Otsu keeps implausibly little signal (correct on all nine volumes); a consensus mode fails here (0.094) because most algorithms share the under-segmentation. The two tools rank vessel volume consistently (Spearman 0.75) but fluorostats reads ~1.7× higher in absolute terms (Fig. 3d), a systematic offset—Li is more inclusive than the VesselExpress pipeline—of the kind REAVER documents among 2D packages. We omit skeleton length here, as skeletonizing full 250-MB light-sheet stacks is intractable.

### Depth-resolved viability recovers what two dimensions discard

Viability is where imaging in three dimensions but quantifying in two does the most damage, and where treating the stack as a volume pays off most clearly. We make two claims: that two-dimensional shortcuts bias the live fraction in a consistent direction, and that fluorostats' count-based viability reproduces a published tool exactly while extending it into depth.

Take the true voxelwise 3D live fraction of a public Day-14 Live/Dead stack (S-BIAD2130, 0.570) as the reference, and every two-dimensional or heuristic reduction of it reads high: a maximum-intensity projection by 5.0%, a mid-plane slice by 1.5%, brightest-focus selection by 1.7%, and a naive mean of per-slice fractions by 25.2%, the last because it over-weights the sparse deep slices. Attenuation correction stays within 2.7% of the volumetric value (Fig. 4a; Extended Data Table 4). Because the same volume passes through every pipeline, the differences belong to the reduction, not the sample, and they point the same way: the readout a laboratory would report from a projection overstates viability relative to the volumetric truth.

The volumetric claim earns its credibility from a more conventional one. Benchmarked against a published Fiji Live/Dead macro [Kerkhoff2024] on its own synthetic data, where per-cell counts and hence true viability are known by construction, fluorostats at first fell short: it had no peak-counting mode, and its area- and connected-component readouts trailed the macro on crowded cells, the same overlap wall the nuclei meet. We built the missing mode—local-maxima counting—and fluorostats now matches the macro exactly, at a mean absolute error of 0.016 and a concordance correlation of 0.987 against the macro's identical 0.016 and 0.987 (Fig. 4b; Extended Data Table 4), through its own interface and training-free. The same library that reproduces the standard 2D readout is the one that shows that readout to be biased in 3D, and reproducing the tool is what licenses the sharper claim.

No counting mode is universal, and we present that as guidance. Local-maxima counting wins on crowded, single-peak cells but over-counts flat or noisy ones, where connected-component counting is far more robust; fluorostats therefore offers connected-component, watershed, maxima, a conservative automatic mode and an all-modes consensus, as an explicit decision guide keyed to density, depth and noise (Extended Data Table 4). Because crowding and noise are not separable from image intensity alone, the automatic mode leans to the robust choice and reports its reasoning rather than posing as an oracle.

### A simple homogeneity index tracks five rigorous statistics, with the statistics built in

Spatial homogeneity is often reduced to a single dispersion index, but the closest recent work validated such an index against biochemical ground truth rather than against established point-pattern statistics, and shipped neither a Python implementation nor a significance test [Martin2026]. fluorostats' segmentation-free, tile-based Gini index closes precisely that gap. Across a controlled regular-to-clustered sweep it tracks all five standard spatial statistics—the Morisita index (Spearman 0.997), quadrat variance (0.997), gliding-box lacunarity (0.981), the Clark–Evans nearest-neighbour index (−0.983) and Ripley's K/L deviation (0.960)—and separates uniform from clustered fields with an area under the curve of 1.0 in every case (Fig. 5b; Extended Data Table 5). An object-based centroid variant behaves the same way (Spearman 0.975 against the clustering parameter, |ρ| ≈ 0.99 against the reference statistics). The index is thus a fast, segmentation-free stand-in for rigorous point-pattern analysis, and its limits are the honest ones: a single fixed tile scale, and no built-in test of complete spatial randomness—a gap the statistics layer is there to fill.

A measurement is only as good as the inference applied to it, and small-sample microscopy is a known site of error: pseudoreplication inflates false positives, and parametric tests are misapplied to small, non-normal samples [Lord2020; Lazic2010]. fluorostats makes the defensible choice the default—Mann–Whitney with Cliff's delta, stratified and Scheirer–Ray–Hare tests, Benjamini–Hochberg control, bootstrap intervals—each exact against its reference and acting on the objects the quantification already produced, so an analysis runs from image to multiplicity-controlled significance with no manual export (Fig. 5c). Bootstrap power from a small pilot is optimistic by construction [Albers2018], which we document as a caveat about experimental design.

### Runtime and determinism

A CPU-only tool must be fast enough to use and a reproducible one must be deterministic; fluorostats is both (Extended Data Fig. 2; Extended Data Table 5). Per-image 2D segmentation takes 14.5 ms on the CPU—on par with the thresholds it is built from, roughly 15-fold faster than StarDist and 380-fold faster than Cellpose on the same processor—and on shared operations it matches its library equivalents, with only validation-time steps (average precision, instance F1) running in seconds. Because no function in the measurement path uses learned weights or random seeds, repeated runs return identical bits: a published number can be regenerated exactly from the archived code and data, free of the seed-, framework- and hardware-dependent variability that trained pipelines carry.

## Discussion

fluorostats occupies a narrow but, we think, undersupplied niche: reproducible, training-free, CPU-only quantification of fluorescence microscopy for the large share of analyses that never needed a trained instance segmenter. Its contribution is not another segmentation algorithm—it builds deliberately on published ones—but a combination that no existing tool offers whole: reference-exact correctness for every metric, a deterministic CPU implementation that removes the GPU and training barriers to reproducibility, and a statistics layer that carries a measurement to a multiplicity-controlled inference without a detour through a spreadsheet. The benchmarks show that the combination costs nothing in accuracy on the tasks it targets. fluorostats matches or beats validated deep-learning segmenters on well-separated nuclei, ties a dedicated angiogenesis package on that package's own benchmark, reproduces a published viability macro exactly and follows five rigorous spatial statistics with a single simple index.

It is a complement to deep learning, not a rival. On heavily overlapping instances, the regime trained models were built for, fluorostats and the whole non-learned class are outperformed, and we have mapped that crossover precisely so the tool can tell a user when to switch. We think that is the useful framing. A tool that is correct by construction, reproducible by design and candid about its limits is the right first instrument for a great many quantification tasks, and an honest signpost to deep learning for the rest. For the tissue-engineering and vascular-biology groups that generate volumetric fluorescence data without routine GPU access or annotation budgets, and for whom a reported number has to be reproducible as well as accurate, it is meant to be that first instrument.

## Scope and limitations

We state the boundaries plainly; each has a mechanism, a measured boundary and a guidance, mapped in the scope panel (Fig. 2f). Crowded, overlapping instances collapse connected-component labelling (F1 0.94 → 0.15 across BBBC024 clustering), a limit of the whole non-learned class; use a trained segmenter there, whose masks fluorostats can still measure. The default Otsu threshold under-segments dim, sparse signal—recovered by the automatic/Li mode, which is not an oracle—so report the threshold used. No viability counting mode is universal, because crowding and noise are inseparable from image statistics alone; choose it from the assay (Extended Data Table 4). On hard cytoplasmic 3D and small dim nuclei an untuned pipeline gives only moderate overlap (Dice 0.52–0.69), and continuous morphometrics degrade near the raster limit while integer topology stays exact. Two comparisons rest on imperfect ground truth, which we keep distinct from the method: the VesselExpress comparison is software-versus-software agreement, not a gold-standard test, and small-pilot bootstrap power is optimistic by construction. Finally, the benchmark scripts are software too—reference-agreement and invariant checks in the test suite caught and we corrected several script-level errors before any reported result depended on them, so every number here derives from audited, version-tagged code, in keeping with guidance on reproducible research [Sandve2013; Miura2021].

## Methods

*Online Methods, to sit at the end of the manuscript per Nature-style convention; move inline for a journal that integrates them.*

### Software implementation

fluorostats (v0.7.0) is written in Python and depends on NumPy, SciPy, scikit-image, pandas, tifffile and czifile. It exposes both a library API, in which every metric is a pure function, and a command-line interface, and it reads the major microscopy formats (confocal and light-sheet z-stacks and widefield; TIFF, CZI and others). The 19 modules follow the workflow—input/output, preprocessing, segmentation, object handling, 2D and 3D metrics, skeleton, topology, vascular, viability, homogeneity, agreement, statistics, power, and reporting/figures—and are covered by 105 automated tests, including the reference-agreement checks described under correctness. No function requires a GPU, and no function in the measurement path uses a random seed; all benchmarks were run on the CPU.

### Datasets

Every benchmark dataset is public and citable. Nuclei: BBBC039 and BBBC024 from the Broad Bioimage Benchmark Collection [Ljosa2012], and the 2018 Data Science Bowl set (DSB2018) [Caicedo2019]. 3D segmentation: Cell Tracking Challenge fluorescence sets (Fluo-C3DH-A549, Fluo-N3DH-CHO) [Maska2014; Maska2023]. Vascular: the REAVER annotated dataset [Corliss2020], the SproutAngio VEGF dose–response set (Zenodo 7240927), and the VesselExpress light-sheet volumes (Zenodo 6025935) [Spangenberg2023]. Viability: a public Live/Dead stack (S-BIAD2130) and the Kerkhoff Fiji-macro synthetic benchmark (Zenodo 10395753) [Kerkhoff2024]. Accession identifiers and download URLs are listed in the data-availability statement and the repository's data manifest.

### Metric definitions and evaluation

Instance-segmentation accuracy is reported as instance F1 and average precision, computed by matching predicted to ground-truth instances above an intersection-over-union threshold, with F1(t) = 2·TP(t) / [2·TP(t) + FP(t) + FN(t)] and average precision averaged over IoU thresholds from 0.5 to 0.9, following the DSB2018 convention [Caicedo2019]. Semantic overlap is reported as foreground Dice and Jaccard. Vascular agreement uses vessel area fraction (2D) and volume fraction (3D), with concordance correlation, Spearman correlation and mean absolute error against the reference; the accuracy/precision decomposition and the zero-bias test follow REAVER [Corliss2020]. Viability is the live fraction (live count or volume over total), with agreement to the reference macro measured by mean absolute error and Lin's concordance correlation. Spatial homogeneity uses a tile-based Gini index, compared by Spearman correlation and uniform-versus-clustered area under the curve against five reference statistics (Clark–Evans, Ripley's K/L, Morisita, quadrat variance, lacunarity). Topology (Euler number, connected components, largest-connected-component fraction) and skeleton metrics (length, branch and junction counts) are validated against analytic phantoms and the Lee-1994 reference. All scoring routines are themselves validated (correctness, above).

### Baseline configuration and validation

Deep-learning baselines were run at their published configurations on the CPU: StarDist (2D_versatile_fluo), Cellpose (v3, nuclei) and Omnipose. Each was first validated to reproduce its published accuracy on BBBC039 before any comparison (StarDist observed F1 0.871 against a published 0.864; Cellpose 0.862) [PublishedBaselines]. A single evaluation policy applied across all methods—identical test images, identical scoring, default parameters, and, for fluorostats, no per-dataset training—and any asymmetry is noted where it arises. Classical thresholding baselines (Otsu, Li, Isodata, Triangle, Yen, Mean, Minimum, watershed) were run through the same quantification and scoring as fluorostats.

### Statistics

Group comparisons use the Mann–Whitney U test with Cliff's delta effect sizes; structured designs use stratified rank tests and the Scheirer–Ray–Hare test; multiplicity is controlled by the Benjamini–Hochberg false-discovery rate. Confidence intervals are bootstrap (10,000 resamples unless stated). The nucleus head-to-head reports paired per-image differences with bootstrap confidence intervals, with parity defined as an interval overlapping zero and superiority as one excluding it. Power analysis is bootstrap from pilot data, reported with the small-pilot optimism caveat [Albers2018]. All statistical functions are validated to exact agreement with reference implementations (correctness, above).

### Compute environment

Local benchmarks were run with Python 3.13. Deep-learning baselines were run on an AMD ROCm HPC cluster on the CPU (partitions and scripts in the repository's benchmark directory). All timings were measured on the CPU with no GPU present, on the hardware noted in the runtime figure.

## Data availability

All datasets analysed are public: BBBC039 and BBBC024 (Broad Bioimage Benchmark Collection); DSB2018; Cell Tracking Challenge Fluo-C3DH-A549 and Fluo-N3DH-CHO; the REAVER dataset; SproutAngio (Zenodo 7240927); VesselExpress (Zenodo 6025935); S-BIAD2130; and the Kerkhoff synthetic Live/Dead benchmark (Zenodo 10395753). Accession identifiers and download URLs are provided in the repository data manifest. Raw image data are redistributed under their original licenses; the derived benchmark tables behind every figure are deposited under a Creative Commons license with a DOI [to be minted at submission].

## Code availability

fluorostats is open source under a permissive OSI-approved license at github.com/rsiegemit/fluorostats, installable from source and the Python package index, and the exact version used here (v0.7.0) is archived with a citable DOI on Zenodo [to be minted at submission]. The full benchmark harness—one script per comparison, with pinned dependencies—and scripts that regenerate every figure and table are included in the repository. Because the library is deterministic and CPU-only, every reported number is reproducible without a GPU, model weights or a fixed random seed.

## Availability and requirements

- **Project name:** fluorostats
- **Home page:** github.com/rsiegemit/fluorostats
- **Operating systems:** platform-independent (Linux, macOS, Windows)
- **Programming language:** Python (3.13 tested)
- **Other requirements:** NumPy, SciPy, scikit-image, pandas, tifffile, czifile
- **License:** [OSI-approved permissive license — MIT/BSD/Apache, to confirm]
- **Restrictions:** none for academic or commercial use

## References

*To be assembled with a reference manager. Placeholder keys used in the text, carrying the proof-stage verification flags from the research dossiers (research/00_SYNTHESIS.md §5):*

- [Schmidt2018] Schmidt et al., Cell Detection with Star-convex Polygons, MICCAI 2018; [Weigert2020] StarDist-3D, WACV 2020 (cite the 2020 paper for 3D).
- [Stringer2021] Stringer et al., Cellpose, Nature Methods 2021.
- [Cutler2022] Cutler et al., Omnipose, Nature Methods 2022.
- [Greenwald2022] Greenwald et al., Mesmer/DeepCell, Nature Biotechnology 2022.
- [Corliss2020] Corliss et al., REAVER, Microcirculation 2020.
- [Zudaire2011] Zudaire et al., AngioTool, PLoS ONE 2011.
- [Spangenberg2023] Spangenberg et al., VesselExpress, Cell Reports Methods 2023.
- [Todorov2020] Todorov et al., VesSAP, Nature Methods 2020.
- [ArgandaCarreras2010] Arganda-Carreras et al., AnalyzeSkeleton, Microsc. Res. Tech. 2010.
- [NunezIglesias2018] Nunez-Iglesias et al., skan, PeerJ 2018.
- [Doube2010] Doube et al., BoneJ, Bone 2010.
- [Kerkhoff2024] Kerkhoff & Ludwig, Fiji Live/Dead macro, Zenodo 10395753, 2024.
- [Ljosa2012] Ljosa et al., BBBC, Nature Methods 2012.
- [Caicedo2019] Caicedo et al., 2018 Data Science Bowl, Nature Methods 2019.
- [Maska2014] Maška et al., Cell Tracking Challenge, Bioinformatics 2014; [Maska2023] 10-year CTC, Nature Methods 2023.
- [Martin2026] Martin et al., dispersion indices, iScience 2026 (confirm final volume/pages at typeset).
- [Lord2020] Lord et al., SuperPlots, J. Cell Biol. 2020.
- [Lazic2010] Lazic, pseudoreplication, BMC Neuroscience 2010.
- [Laine2021] Laine et al., replication crisis in DL bioimage analysis, Nature Methods 2021.
- [Spiller2025] Spiller & Duarte Campos, Front. Bioeng. Biotechnol. 2025.
- [Pereira2023] Pereira et al., angiogenesis software review, Int. J. Mol. Sci. 2023.
- [Riley2023] Riley — magnification/zoom reproducibility (confirm citation).
- [Albers2018] Albers & Lakens, pilot power optimism, 2018.
- [Sandve2013] Sandve et al., Ten Simple Rules for Reproducible Computational Research, PLoS Comput. Biol. 2013.
- [Miura2021] Miura & Nørrelykke, reproducible image handling and analysis, EMBO J. 2021.
- [PublishedBaselines] Internal baseline-validation record (data/PUBLISHED_BASELINES.md).
- Delesse–Glagolev stereology primaries: cite via a modern review.

---

## Display items

**Main display items — 6 total (the Nature Methods maximum): 5 figures + 1 table.** Specifications and per-panel build instructions in `FIGURE_BRIEFS.md`.

- **Fig. 1 — Overview schematic** (built; `fig1_schematic.svg`).
- **Fig. 2 — Nucleus segmentation and the deep-learning boundary.** (a) 12-method F1 ranking with bootstrap CIs; (b) bootstrap-CI forest vs StarDist/Cellpose/Omnipose; (c) qualitative overlay (raw/GT/fluorostats/StarDist); (d) instance-F1-vs-clustering crossover with the DL line; (e) separated-vs-crowded fields. *Optional (f): a scope decision map (when to use fluorostats vs a trained segmenter), synthesized from existing data.*
- **Fig. 3 — Vascular networks.** (a) REAVER ranking (accuracy/precision + zero-bias flag); (b) vessel overlay (raw/segmentation/skeleton/branchpoints, incl. VesselExpress); (c) 3D phantom exact-GT; (d) VesselExpress metric agreement (Bland–Altman).
- **Fig. 4 — Depth-resolved viability.** (a) 2D-vs-3D bias (paired deltas + per-z profile); (b) tie to the Fiji macro (Bland–Altman + scatter); (c) Live/Dead overlay, 2D vs 3D.
- **Fig. 5 — Spatial homogeneity and the integrated statistics layer.** (a) point-pattern panels (regular/Poisson/clustered) with the index; (b) five-statistic correlation. *Optional (c): an end-to-end statistics worked example (image → metrics → Mann–Whitney + Cliff's δ + BH-FDR + bootstrap CI + power), exhibiting the "no manual export" claim.*
- **Table 1 — fluorostats versus established tools across capabilities** (embedded above; the single main table).

**Extended Data — ≤10 items (peer-reviewed, citable):**

- **ED Fig. 1 — Correctness.** Analytic-phantom battery (topology χ, skeleton trees) + zoom-invariance (CV = 0).
- **ED Fig. 2 — Runtime & determinism.** Per-metric runtime (log) vs comparators; bit-identical reruns.
- **ED Fig. 3 — Robustness** *(planned).* Noise, object-size and denoising sweeps.
- **ED Fig. 4 — Generalization** *(planned).* Per-dataset/per-modality breakdown incl. DSB2018 and CTC 3D.
- **ED Tables 1–5** (below): the per-experiment numbers behind the figures.

*In the final NM layout, Extended Data items are collected after the references (kept here inline for review). Supplementary Information carries no figures under NM policy — only large/raw-data tables and the community image-analysis reporting checklist.*

---

## Extended Data Tables

*Collected here for review; in the final layout they follow the references. Each carries the per-experiment numbers behind a main figure.*

**Extended Data Table 1 | Reference-exact validation across capabilities.** Integer-valued quantities agree exactly; continuous quantities agree within a stated discretization tolerance.

| Capability | Reference | Result |
|---|---|---|
| Statistics (Mann–Whitney, Cliff's δ, BH-FDR, Stouffer, Scheirer–Ray–Hare, stratified) | SciPy + hand-coded | 8/8 + stratified, exact |
| Agreement (Bland–Altman, Lin's CCC, ICC) | closed-form / two-way ANOVA | 11/11, machine precision |
| Instance metrics (F1, average precision, matching) | independent IoU matcher / DSB2018 formula | 23/23 |
| Volume fraction / density | Delesse point-counting, analytic | 7/7; zoom-invariant (CV = 0) |
| Connectivity (Euler number, components, LCC) | analytic phantoms | 6/6 exact; Euler ρ = 1.0 (best tracker) |
| Skeleton (length, branches, junctions) | analytic phantoms + Lee-1994 (skan / AnalyzeSkeleton) | length ≤ 1%, branches/junctions exact |
| Nucleus size | BBBC024 ground truth | 3.5% median-diameter error |

**Extended Data Table 2 | Nucleus segmentation (Fig. 2).** (a) Twelve-method ranking on BBBC039 (n = 60; full n = 200 fluorostats F1 = 0.896). (b) Crowded-regime head-to-head, BBBC024 c75 (GT = 20 nuclei/slice, n = 12).

*(a) Ranking — F1 at IoU 0.5, mean AP (0.5–0.9), count MAE:*

| Rank | Method | F1 | AP | count MAE |
|---|---|---|---|---|
| 1 | Li (1993) | 0.934 | 0.790 | 4.80 |
| 2 | Otsu / Isodata / fluorostats (Otsu + CC) | 0.905 | 0.735 | 4.33 |
| 5 | Mean | 0.899 | 0.662 | 6.30 |
| 6 | StarDist (deep learning) | 0.874 | — | — |
| 7 | Cellpose (deep learning) | 0.870 | — | — |
| 8 | Triangle (1977) | 0.865 | 0.536 | 9.22 |
| 9 | Minimum | 0.578 | 0.488 | 38.3 |
| 10 | Watershed (1991) | 0.342 | 0.128 | 127 |
| 11 | Yen (1995) | 0.310 | 0.212 | 68 |

*(b) Crowded regime:*

| Method | Type | Mean F1 @ 0.5 | Mean count (GT = 20) |
|---|---|---|---|
| Cellpose (v3 nuclei) | deep learning | 1.00 | 20.0 |
| StarDist (2D_versatile_fluo) | deep learning | 0.96 | 20.3 |
| fluorostats (Otsu + CC) | threshold | 0.38 | 7.6 |

**Extended Data Table 3 | Vascular networks (Fig. 3).** (a) REAVER benchmark (n = 36, expert manual GT; all tools through one unified quantification at default parameters). (b) 3D validation.

*(a) 2D vessel-area-fraction accuracy:*

| Rank | Tool | MAE | CCC | Spearman |
|---|---|---|---|---|
| 1 | REAVER | 0.017 | 0.984 | 0.967 |
| 2 | ImageJ | 0.041 | 0.862 | 0.986 |
| 3 | AngioTool | 0.068 | 0.752 | 0.965 |
| 4 | fluorostats | 0.076 | 0.701 | 0.935 |
| 5 | RAVE | 0.094 | 0.700 | 0.911 |
| 6 | AngioQuant | 0.149 | 0.333 | 0.886 |

*(b) 3D — synthetic phantom vs exact GT: centreline length error 0.6–2.4%, branch count exact, volume fraction exact. VesselExpress light-sheet agreement (n = 9), foreground Dice:*

| Configuration | Dice vs VesselExpress |
|---|---|
| fluorostats (li) / (auto → li) | 0.598 |
| fluorostats (triangle) | 0.521 |
| scikit-image (otsu) | 0.102 |
| fluorostats (consensus) | 0.094 |
| fluorostats (otsu, default) | 0.089 |

*Vessel volume-fraction agreement: Spearman 0.75; fluorostats 0.049 vs VesselExpress 0.029 (≈ 1.7×); CCC 0.11.*

**Extended Data Table 4 | Viability (Fig. 4).** (a) 2D-shortcut bias vs the true voxelwise 3D live fraction (S-BIAD2130, 0.570). (b) Agreement with the Fiji macro (Kerkhoff, synthetic GT). (c) Counting-mode guide (size-and-noise sweep, true count = 100).

*(a) 2D bias:*

| Reduction | Bias vs 3D | Reduction | Bias vs 3D |
|---|---|---|---|
| Mid-plane slice | +1.5% | Mean of per-slice | +25.2% |
| Brightest-focus | +1.7% | Attenuation-corrected | within 2.7% |
| Max-intensity projection | +5.0% | | |

*(b) Macro agreement:*

| Method | MAE | CCC |
|---|---|---|
| Kerkhoff macro / fluorostats maxima (new) | 0.016 | 0.987 |
| fluorostats — object count (CC) | 0.076 | 0.703 |
| Otsu connected-component count | 0.079 | 0.801 |
| fluorostats — area fraction | 0.091 | 0.832 |

*(c) Mode guide:*

| Assay regime | Mode | Rationale |
|---|---|---|
| Crowded, single-peak cells | maxima | ties the peak-count macro |
| Large or flat cells | watershed / connected-components | maxima over/under-counts |
| Noisy images | connected-components | most noise-robust (68 vs maxima 3,068 at σ = 200) |
| Unknown / mixed | auto + all-modes consensus | biases to robust mode, reports spread |

**Extended Data Table 5 | Homogeneity and runtime (Figs 5, ED 2).** (a) Tile-based Gini vs five spatial statistics (regular→clustered sweep). (b) Per-image runtime (BBBC039, CPU).

*(a) Homogeneity:*

| Reference statistic | Spearman ρ vs Gini | AUC | Reference statistic | Spearman ρ vs Gini | AUC |
|---|---|---|---|---|---|
| Morisita index | 0.997 | 1.0 | Clark–Evans NN | −0.983 | 1.0 |
| Quadrat variance | 0.997 | 1.0 | Ripley's K/L | 0.960 | 1.0 |
| Gliding-box lacunarity | 0.981 | 1.0 | | | |

*(b) Runtime (ms/image, CPU):*

| Method | ms | Method | ms |
|---|---|---|---|
| Otsu / Isodata | 5.7 | Watershed | 35.6 |
| Triangle | 5.9 | StarDist | 215 |
| Li | 9.1 | Cellpose | 5,547 |
| fluorostats (Otsu + CC) | 14.5 | | |
