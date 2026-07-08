# Skeletonization Algorithms and Skeleton/Branch Analysis Tools for Bioimaging

Research category for positioning **fluorostats** — a tool that quantifies skeleton
morphometry of segmented fluorescence structures via 3D skeletonization + graph
analysis (built on scikit-image `skeletonize` and the `skan` library), reporting
total length (µm), branch count, junction count, and mean branch length on
physically calibrated (micrometre) volumes.

---

## 1. References

All citations below were verified by fetching PubMed / publisher / arXiv records.
Where a field could not be independently confirmed it is flagged **[UNVERIFIED]**.

### Foundational thinning / skeletonization algorithms

**[R1] Lee, Kashyap & Chu (1994) — 3D medial surface/axis thinning (the algorithm fluorostats uses)**
- Ta-Chih Lee, Rangasami L. Kashyap, Chong-Nam Chu. "Building skeleton models via 3-D medial surface/axis thinning algorithms." *CVGIP: Graphical Models and Image Processing* 56(6):462–478, Nov 1994.
- DOI: 10.1006/cgip.1994.1042
- Citations: ~120+ (Semantic Scholar / ACM DL).
- **Method:** Parallel 3D thinning that iteratively removes simple points while preserving the Euler characteristic (via a precomputed Euler table) and 26-connectivity (octree lookup on 3×3×3 neighborhoods), yielding a one-voxel-thick medial surface/axis.
- **Relevance:** This is the exact algorithm behind `skimage.morphology.skeletonize(..., method='lee')` for 3D volumes — the core skeletonizer fluorostats depends on. Direct methodological citation.

**[R2] Zhang & Suen (1984) — fast parallel 2D thinning**
- T. Y. Zhang, C. Y. Suen. "A fast parallel algorithm for thinning digital patterns." *Communications of the ACM* 27(3):236–239, 1984.
- DOI: 10.1145/357994.358023
- Citations: several thousand (a canonical thinning reference).
- **Method:** Two-subiteration parallel boundary-peeling that deletes SE/NW boundary and corner points while preserving endpoints and connectivity, producing a unit-width 2D skeleton.
- **Relevance:** Historical/foundational; the default 2D `skeletonize` in scikit-image derives from this family (Zhang–Suen). Cite to root fluorostats' 2D fallback in classical morphology.

**[R3] Blum (1967) / grassfire — medial axis transform**
- Harry Blum. "A transformation for extracting new descriptors of shape." In *Models for the Perception of Speech and Visual Form* (Wathen-Dunn, ed.), MIT Press, 1967, pp. 362–380.
- **Method:** The medial axis (MAT) as the locus of centers of maximal inscribed balls, computed via the "grassfire" propagation from the boundary; equivalent in the discrete setting to a distance-transform ridge. Basis for `skimage.morphology.medial_axis`.
- **Relevance:** Theoretical grounding for distance-transform skeletonization — the alternative to thinning; explains why fluorostats can optionally recover branch *width/radius* from the distance map. **[UNVERIFIED]** exact page range (chapter in an edited MIT Press volume; commonly cited as pp. 362–380).

### Skeleton/branch graph-analysis tools (direct comparators)

**[R4] Nunez-Iglesias et al. (2018) — skan (fluorostats' analysis backend)**
- Juan Nunez-Iglesias, Adam J. Blanch, Oliver Looker, Matthew W. Dixon, Leann Tilley. "A new Python library to analyse skeleton images confirms malaria parasite remodelling of the red blood cell membrane skeleton." *PeerJ* 6:e4312, 2018.
- DOI: 10.7717/peerj.4312 (verified via PubMed 29472997)
- **Method:** Converts a binary skeleton to a graph (SciPy sparse adjacency), classifies voxels as endpoint/junction/path by neighbor count, and emits a tidy pandas DataFrame of per-branch statistics (branch type, branch distance / Euclidean distance, mean pixel intensity). Numba-JIT for performance.
- **Relevance:** **fluorostats is built directly on skan.** skan provides the branch table; fluorostats adds physical calibration, aggregate metrics, and the statistics/normalization pipeline. Primary "we-build-on" citation.

**[R5] Arganda-Carreras et al. (2010) — AnalyzeSkeleton / Skeletonize3D (Fiji)**
- Ignacio Arganda-Carreras, Rodrigo Fernández-González, Arrate Muñoz-Barrutia, Carlos Ortiz-De-Solorzano. "3D reconstruction of histological sections: application to mammary gland tissue." *Microscopy Research and Technique* 73(11):1019–1029, Oct 2010.
- DOI: 10.1002/jemt.20829
- **Method:** Tags skeleton voxels as endpoint (<2 neighbors), slab (=2), or junction (>2) over the 26-neighborhood; merges adjacent junction voxels into single junctions; counts branches, triple/quadruple points, and measures average and maximum branch length. Skeletonize3D implements the Lee 1994 thinning [R1] inside Fiji.
- **Relevance:** The de facto reference implementation and **the primary benchmark comparator**. skan (and thus fluorostats) was explicitly inspired by this plugin, so numerical agreement here is the key validation target.

**[R6] Arshadi, Günther, Eddison, Harrington & Ferreira (2021) — SNT**
- Cameron Arshadi, Ulrik Günther, Mark Eddison, Kyle I. S. Harrington, Tiago A. Ferreira. "SNT: a unifying toolbox for quantification of neuronal anatomy." *Nature Methods* 18:374–377, 2021.
- DOI: 10.1038/s41592-021-01105-7 (verified via PubMed 33795878)
- **Method:** End-to-end neuronal-anatomy framework (tracing, proof-editing, graph/skeleton quantification, Sholl/Strahler analysis, whole-brain connectomics) on the ImageJ2/Fiji stack.
- **Relevance:** State-of-the-art for *neurite* graphs; heavier and interaction-driven vs. fluorostats' lightweight batch pipeline. Contrast: SNT targets curated single-neuron morphology; fluorostats targets high-throughput population statistics over segmented structures.

**[R7] Peng, Ruan, Long, Simpson & Myers (2010) — Vaa3D / V3D**
- Hanchuan Peng, Zongcai Ruan, Fuhui Long, Julie H. Simpson, Eugene W. Myers. "V3D enables real-time 3D visualization and quantitative analysis of large-scale biological image data sets." *Nature Biotechnology* 28(4):348–353, 2010.
- DOI: 10.1038/nbt.1612
- **Method:** 3D+ visualization and automated/semi-automated neuron tracing producing SWC skeleton graphs for morphometry.
- **Relevance:** Large-scale reconstruction platform; relevant to skeleton-graph lineage but oriented to tracing rather than segmentation-derived morphometry like fluorostats.

### Applications (neurons / vessels / mitochondria / cytoskeleton)

**[R8] Viana et al. (2015) — MitoGraph (mitochondrial 3D skeleton graphs)**
- Matheus P. Viana, Swee Lim, Susanne M. Rafelski. "Quantifying mitochondrial content in living cells." *Methods in Cell Biology* 125:77–93, 2015.
- DOI: 10.1016/bs.mcb.2014.10.003
- **Method:** Segments 3D mitochondrial networks and reduces them to node-and-edge skeletons (nodes = ends/branch points, edges = segments) with per-edge length and local width; C++ open source.
- **Relevance:** Closest *application analogue* — same "segment → 3D skeleton graph → length/branch/junction stats on calibrated volumes" recipe fluorostats implements, but hard-specialized to mitochondria. fluorostats generalizes it to any segmented fluorescence structure with a statistics layer. **[UNVERIFIED]** exact page range 77–93.

**[R9] 3DVascNet — Prakash et al. (2024) — 3D vascular network skeleton quantification**
- (Author list per AHA journal record) "3DVascNet: An Automated Software for Segmentation and Quantification of Mouse Vascular Networks in 3D." *Arteriosclerosis, Thrombosis, and Vascular Biology (ATVB)*, 2024. Preprint: bioRxiv 2023.10.19.563201.
- DOI: 10.1161/ATVBAHA.124.320672
- **Method:** Deep-learning segmentation followed by medial-axis skeletonization → graph with terminal/branch points, total vessel length, branch length, radius, and volume-normalized densities.
- **Relevance:** Same skeleton-graph morphometry family in vasculature; demonstrates the branch/junction/length/density metric set fluorostats reports and the value of the volume-normalization step. **[UNVERIFIED]** full author list (confirm on the AHA/DOI record before citing).

**[R10] Henty-Ridilla / actin-quantification tools (2024)**
- Dennis Zimmermann et al. (see record). "Computational tools for quantifying actin filament numbers, lengths, and bundling." *Biology Open* 13(3):bio060267, 2024.
- DOI: 10.1242/bio.060267 (PubMed 38372564)
- **Method:** MATLAB pipelines: noise filtering → background subtraction → thresholding → skeletonization to count filaments and measure length/bundling from fluorescence micrographs.
- **Relevance:** Cytoskeleton application of skeleton morphometry; shows the demand fluorostats meets in a Python/reproducible form. **[UNVERIFIED]** exact author list (confirm lead/senior authors on DOI record).

**[R11] van der Walt et al. (2014) — scikit-image (the skeletonizer host library)**
- Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart, Tony Yu, and the scikit-image contributors. "scikit-image: image processing in Python." *PeerJ* 2:e453, 2014.
- DOI: 10.7717/peerj.453
- **Method:** Open-source Python image-processing library; its `morphology` module supplies `skeletonize` (Zhang–Suen 2D / Lee 3D) and `medial_axis`.
- **Relevance:** The library providing fluorostats' skeletonization primitives. Mandatory dependency citation.

---

## 2. Comparison to fluorostats

**Parity (must demonstrate):** fluorostats' core metrics — total length, branch count,
junction count, mean branch length — are exactly the quantities defined by
AnalyzeSkeleton [R5] and computed by skan [R4]. Because fluorostats *uses* skan for
graph construction and the *same* Lee 1994 thinning [R1] that Fiji's Skeletonize3D
uses, its per-branch/junction outputs should be numerically identical (up to
voxel-scaling) to AnalyzeSkeleton on the same segmented volume. This makes a clean
equivalence benchmark possible (Section 3).

**Where fluorostats adds value:**
1. **Physical calibration by default** — lengths are reported in µm on
   micrometre-calibrated 3D volumes, not voxels (AnalyzeSkeleton reports voxel/scaled
   length; skan reports raw graph distance requiring manual spacing handling).
2. **Integrated statistics/normalization layer** — fluorostats does not stop at a
   per-branch table; it aggregates to per-object/per-condition summaries and feeds a
   statistics pipeline (group comparisons, power, volume-normalized densities), which
   neither skan nor AnalyzeSkeleton provide. This mirrors the volume-normalization
   step shown to matter in vascular [R9] and mitochondrial [R8] work, but as a
   general, reusable layer.
3. **Reproducible, scriptable, batch-first** — pure-Python, no GUI dependence
   (vs. Fiji [R5], SNT [R6], Vaa3D [R7]); one pipeline over many volumes.

**Limitations to state honestly:**
- fluorostats inherits skeletonization artifacts common to all thinning methods
  (spurious short branches from boundary noise, junction over-merging) — it does not
  yet claim a novel pruning/validation step beyond what skan offers.
- It targets *segmentation-derived* skeletons; it does not do interactive tracing/
  proof-editing (SNT [R6], Vaa3D [R7]) or provide width/radius from the distance map
  by default the way MitoGraph [R8] / 3DVascNet [R9] do (a candidate extension via
  `medial_axis` [R3, R11]).
- No Strahler/Sholl ordering yet (available in AnalyzeSkeleton's Strahler add-on and SNT).

---

## 3. Proposed benchmarks

**B1 — Numerical equivalence vs. AnalyzeSkeleton (the strongest, do this first).**
Take a panel of segmented binary fluorescence volumes (e.g., 20–50 objects spanning
simple filaments → dense branched networks; include synthetic phantoms with *known*
ground-truth branch/junction counts and total length). Run **Fiji AnalyzeSkeleton
[R5]** (Skeletonize3D → AnalyzeSkeleton) and **fluorostats** on the *identical* binary
inputs with matched voxel spacing. Compare, per object:
- branch count, junction count, endpoint count (exact integer agreement expected),
- total length and mean branch length (agreement within sub-voxel rounding after unit
  conversion).
Report Bland–Altman + concordance (CCC) and per-object scatter with y=x. Because both
use Lee 1994 thinning [R1], **demonstrating ≥99% agreement on counts and <1–2% length
deviation establishes fluorostats' measurement validity** and lets reviewers trust
downstream stats. Discrepancies should be traceable to junction-merging conventions or
diagonal-length weighting — document the convention fluorostats uses.

**B2 — Ground-truth accuracy on phantoms.** Using the synthetic phantoms from B1
(digitally generated trees/lattices with analytic total length and known
branch/junction counts), report absolute error for fluorostats and AnalyzeSkeleton
side by side — separates *tool* error from *thinning-algorithm* error shared by both.

**B3 — Added-value demonstration (the differentiator).** On a real biological dataset
with two conditions (e.g., control vs. treatment mitochondrial or vascular networks),
show that once branch/junction/length parity is established (B1), fluorostats' extra
layer — µm calibration, volume-normalized branch/junction density, group statistics
with effect sizes and power — surfaces a significant, reproducible difference in a
single scripted run, whereas AnalyzeSkeleton/skan stop at raw per-branch tables
requiring bespoke downstream code. This shows fluorostats = "AnalyzeSkeleton parity +
statistics pipeline."

**B4 — Cross-tool sanity (optional).** Where domain tools exist (MitoGraph [R8] for
mitochondria, 3DVascNet [R9] for vessels), compare total length / branch count on their
demo data to show fluorostats reproduces domain-specialist numbers with a general tool.

---

## Summary (≈200 words)

fluorostats occupies a well-established niche: skeleton-graph morphometry of segmented
fluorescence structures. Its foundations are solid and citable — 3D skeletons come from
the Lee, Kashyap & Chu (1994) thinning algorithm [R1] via scikit-image [R11], and the
branch/junction graph statistics come from skan (Nunez-Iglesias et al., 2018) [R4],
which was itself modeled on Fiji's AnalyzeSkeleton (Arganda-Carreras et al., 2010) [R5].
The metric set fluorostats reports — total length, branch count, junction count, mean
branch length — is exactly the AnalyzeSkeleton/skan standard, so numerical equivalence
is provable rather than merely asserted. fluorostats' genuine contribution is the layer
*above* the branch table: default micrometre calibration, volume-normalized densities,
and an integrated statistics/power pipeline over many objects — a generalization of the
per-domain normalization seen in MitoGraph [R8] and 3DVascNet [R9], packaged as a
reproducible, GUI-free, batch-first Python tool (contrast SNT [R6], Vaa3D [R7]).

**Strongest benchmark:** B1 — run Fiji AnalyzeSkeleton vs. fluorostats on identical
segmented volumes (plus synthetic phantoms with known counts/length). Because both use
Lee-1994 thinning, ≥99% agreement on branch/junction counts and <2% length deviation
establishes measurement validity; then B3 layers on fluorostats' statistics to show
added value.

### Citations flagged for verification before final submission
- [R3] Blum 1967 — confirm exact page range (edited MIT Press volume).
- [R8] MitoGraph — confirm exact page range 77–93.
- [R9] 3DVascNet — confirm full author list on the AHA/DOI record.
- [R10] actin tools (Biology Open 2024) — confirm lead/senior author names on DOI record.
