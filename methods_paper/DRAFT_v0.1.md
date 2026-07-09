# fluorostats: a training-free, CPU-only Python library for reproducible quantification of fluorescence microscopy

*Working draft v0.1 — front matter + Design + Validation. Written to Nature-Methods-Article
conventions (adaptable to PLoS Computational Biology / eLife Tools & Resources). Framing per
locked decision: the contribution is the reproducible, training-free, reference-exact quantifier;
parity-or-better benchmarks are the evidence that earns the claim. Every quantitative claim is
sourced from the benchmark campaign (benchmarks/00_BENCHMARK_RESULTS.md and the CSVs it indexes);
citation keys in [brackets] are placeholders pending the reference-manager pass.*

---

## Abstract

Quantifying fluorescence microscopy — volume fraction, network topology, skeleton and vascular
architecture, object morphometry, spatial homogeneity, and Live/Dead viability — underpins
conclusions across developmental biology, vascular biology, and tissue engineering. Yet these
measurements are scattered across many single-purpose tools, most requiring GPUs or per-dataset
training, and thick specimens are routinely imaged in three dimensions but quantified in two;
downstream statistics are typically performed by error-prone manual export, breaking the chain
from image to reported result. We present **fluorostats**, an open-source Python library and
command-line tool that computes these quantities training-free on a standard CPU, with an
integrated layer of small-sample non-parametric statistics, bootstrap power analysis, and
publication figures. Every metric is validated to reference-implementation exactness against its
established tool or against analytic ground truth, and accuracy is benchmarked against StarDist,
Cellpose, Omnipose, REAVER, AngioTool, VesselExpress, and a published Fiji Live/Dead macro on
public datasets. fluorostats matches or exceeds these tools across a broad range of quantification
tasks while running deterministically on CPU, and we delineate the crowded, heavily-overlapping
instance regime where trained deep-learning segmenters remain preferable. fluorostats is available
under a permissive license at github.com/rsiegemit/fluorostats with a version-archived DOI.

---

## Introduction

Fluorescence microscopy is now routinely volumetric. Confocal and light-sheet instruments deliver
three-dimensional stacks of hydrogel constructs, whole organs, and cultured tissues as a matter of
course, and the biological questions asked of these images are quantitative: what fraction of a
volume is occupied by signal, how connected is a vascular or mitochondrial network, how are objects
distributed in space, and what fraction of cells are alive at a given depth. The answers to these
questions decide whether a bioink supports cell survival, whether an angiogenic stimulus produced a
denser network, and whether a spatial pattern is uniform or clustered [refs: field applications].

Despite the ubiquity of the data, the tooling to quantify it is fragmented. Instance segmentation
is served by a mature family of trained deep-learning models — StarDist, Cellpose, Omnipose, and
Mesmer among them — that achieve excellent accuracy on the tasks they are trained for [Schmidt2018;
Stringer2021; Cutler2022; Greenwald2022]. Vascular networks are quantified by dedicated tools such
as REAVER, AngioTool, and, in three dimensions, VesselExpress and VesSAP [Corliss2020; Zudaire2011;
Spangenberg2023; Todorov2020]. Skeleton and topology metrics are the province of Fiji's
AnalyzeSkeleton, skan, and BoneJ [ArgandaCarreras2010; NunezIglesias2018; Doube2010]; viability is
most often measured with hand-built ImageJ macros [Kerkhoff2024]. Each of these tools is excellent
within its domain, but three problems recur across the collection. First, the strongest segmentation
tools require a GPU and, frequently, retraining or fine-tuning to perform on a new modality, placing
them out of reach for many laboratories and undermining reproducibility, because a model's output is
reliable only on images resembling its training set [Laine2021]. Second, although the data are
volumetric, quantification is frequently performed on two-dimensional maximum-intensity or mean
projections, which systematically distort the very metrics they are used to report — a limitation
the tissue-engineering and angiogenesis literature repeatedly acknowledges but rarely resolves
[Spiller2025; Pereira2023]. Third, once metrics are extracted, the statistical analysis that turns
them into a claim is typically exported by hand to a spreadsheet or a separate statistics package,
where small-sample and pseudoreplication pitfalls inflate false-positive rates [Lord2020; Lazic2010]
and analysis parameters go unreported, a documented driver of the field's reproducibility problems
[Pereira2023].

There is, in short, no single validated pipeline that carries a measurement from a raw volumetric
image through to a correctly chosen, reproducible statistic without a GPU, a training set, or a
manual export step. This is the gap fluorostats is built to close.

Here we present fluorostats, an open-source Python library and command-line tool for quantifying
fluorescence microscopy. fluorostats is organized around four design commitments that together
define its niche: it is **training-free** (every operation is a deterministic algorithm, not a
learned model), it is **CPU-only** (no GPU is required for any function), every metric is
**reference-exact** (validated to numerical agreement with the canonical implementation or with
analytic ground truth), and it carries an **integrated statistics layer** (small-sample
non-parametric tests, effect sizes, multiple-comparison control, and bootstrap power analysis) so
that quantification and inference live in one reproducible pipeline. The library spans volume
fraction and density, connectivity and topology, skeleton and vascular metrics, object morphometry,
spatial homogeneity, and depth-resolved Live/Dead viability, and it emits the statistics and figures
needed to report them.

We validate fluorostats in two stages. We first establish correctness: every metric is checked
against a reference implementation or against phantoms with closed-form ground truth, yielding exact
agreement for integer-valued topological quantities and bounded, characterized error for continuous
ones. We then benchmark accuracy against established tools on public datasets — StarDist, Cellpose,
and Omnipose for nuclei (with each deep-learning baseline first validated to reproduce its published
accuracy); REAVER, AngioTool, and VesselExpress for vasculature; a published Fiji macro for
viability; and five classical spatial statistics for homogeneity. Across these comparisons
fluorostats is at parity with or better than the established tool on well-separated targets, at a
fraction of the runtime and with no training or GPU, and it reports every comparison with bootstrap
confidence intervals and paired significance tests. We are equally explicit about the boundary of
this claim: on heavily overlapping, crowded instances, fluorostats — like the entire class of
non-learned methods — is outperformed by trained deep-learning segmenters, and we characterize
exactly where that crossover occurs so that users know when to reach for them. The remainder of the
paper presents the design, the correctness validation, the accuracy benchmarks by capability, the
runtime, and the scope and limitations in turn.

---

## Results

### Design and implementation

fluorostats is a Python library (version 0.7.0 at the time of writing) with a command-line
interface, distributed under a permissive license and installable from source and the Python package
index. Its architecture follows the sequence of a quantification workflow — input and format
handling, preprocessing, segmentation, metric extraction, statistics, and reporting — organized into
19 modules in which every metric is implemented as a pure function of its inputs (Fig. 1). Three
properties follow from this design and recur throughout the results below.

First, fluorostats is **training-free and deterministic**. Segmentation is performed by classical
thresholding (with a configurable family of algorithms — Otsu, Li, Isodata, Triangle, Yen, Mean,
Minimum — and optional watershed splitting), and every downstream metric is a closed-form or
combinatorial computation. There are no learned weights, no stochastic inference, and no random
seeds in the measurement path, so repeated runs on the same input return bit-identical results. This
is a property no trained segmenter can offer, and we treat it as a first-class reproducibility
guarantee rather than an incidental implementation detail.

Second, fluorostats is **CPU-only**. No function requires a GPU; the entire library runs on a
standard laptop or workstation processor. This removes the principal access and reproducibility
barriers associated with deep-learning pipelines — GPU availability, driver and framework version
drift, and non-deterministic hardware kernels — and, as the runtime results show (Section
"Runtime and determinism"), does so without a practically limiting speed penalty for the operations
that matter.

Third, fluorostats treats **volumetric data as volumetric**. Volume fraction, connectivity,
skeleton and vascular metrics, and Live/Dead viability are computed in three dimensions with
physically calibrated voxel spacing, and densities are normalized per unit physical volume so that
they are invariant to digital zoom and voxel size. Where a two-dimensional shortcut is available we
implement it explicitly, so that the systematic difference between a volumetric measurement and its
projection can be measured rather than assumed (Section "Depth-resolved viability").

Beyond measurement, fluorostats includes an **integrated statistics and reporting layer** that no
general bioimage platform provides: small-sample non-parametric hypothesis tests (Mann–Whitney U
with Cliff's delta effect sizes, stratified rank tests, the Scheirer–Ray–Hare interaction test),
Benjamini–Hochberg false-discovery-rate control across metrics and strata, bootstrap confidence
intervals and fold-change estimates, agreement statistics (Bland–Altman, Lin's concordance
correlation, the intraclass correlation), and bootstrap power analysis from pilot data. Because
these operate on the same objects the quantification produces, an analysis can proceed from image to
effect size to multiplicity-controlled significance without leaving the pipeline or exporting to a
separate tool. The correctness of this layer is validated to machine precision below.

### Correctness: reference-exact validation

Before comparing fluorostats to other tools on accuracy, we establish that each of its metrics
computes the intended quantity correctly, by checking it against a reference implementation or
against a phantom whose true value is known in closed form. We report this validation first because
it is the foundation on which the accuracy comparisons rest: a favorable benchmark against another
tool means little unless the underlying metric is provably correct. We distinguish throughout
between **integer-valued quantities**, which are expected to agree *exactly*, and **continuous
quantities**, which are subject to discretization and are reported with bounded, characterized error.

**Topology is exact against analytic ground truth.** On a battery of phantoms with closed-form
Euler characteristics and known component counts, fluorostats recovers the Euler number, the number
of connected components, and the largest-connected-component fraction with zero error across all six
phantoms (Table 1). Because these are integer topological invariants, exact agreement — not merely
close agreement — is the correct pass criterion, and it is met. In a connectivity-discrimination
experiment sweeping a structure from fragmented to fully connected, the Euler number was the single
best-tracking descriptor (Spearman rho = 1.0) and, uniquely among the five measures tested, resolved
both the fragmentation and the interconnection regimes, where largest-component fraction and
percolation saturate.

**Skeleton metrics share the reference algorithm, and agree with it.** fluorostats skeletonizes with
the Lee–Kashyap–Chu 1994 thinning algorithm — the same algorithm implemented by Fiji's
AnalyzeSkeleton and by skan [ArgandaCarreras2010; NunezIglesias2018]. Sharing the algorithm makes
exact agreement a *predicted, falsifiable* outcome rather than a coincidence, and we confirm it: on
line phantoms of known length, skeleton length is recovered within 1% and branch counts exactly; on
synthetic trees of known bifurcation count, branch and junction counts are recovered exactly at
depths 2–3 (7/3 and 15/7 branches/junctions). On a denser depth-4 tree the branch count is slightly
undercounted (27 versus 31), a rasterization limit at the resolution of short terminal branches
which we state explicitly rather than tuning away. Four skeletonization algorithms (Lee-1994,
medial-axis, morphological thinning, and Zhang) recover the same branch counts within 1% length
spread, confirming the choice of algorithm is not itself a source of error.

**Continuous physical metrics match their analytic definitions and are zoom-invariant.** Voxel-based
volume fraction equals the known fraction exactly on synthetic volumes; the volume fraction of a
sphere in a box matches the analytic (4/3)πr³/L³ within 0.1%; and Cavalieri point-counting converges
to the voxel value, reproducing the Delesse–Glagolev stereological principle [refs]. Object density
normalized per cubic millimetre is *exactly* invariant to voxel size across a six-fold digital-zoom
range (coefficient of variation 0.000), whereas raw counts, per-megavoxel, per-area, and per-slice
densities all vary — directly reproducing and resolving the documented magnification-sensitivity of
uncalibrated counts [Riley2023].

**The statistics and agreement layer is exact to machine precision.** Every function in the
statistics module reproduces its reference implementation exactly: Mann–Whitney U and its p-value
match SciPy; Cliff's delta matches a brute-force computation; Benjamini–Hochberg matches a hand-coded
reference; the Stouffer combination matches SciPy's; and the Scheirer–Ray–Hare test recovers a known
interaction. Raw per-stratum p-values and the stratified false-discovery-rate grid match reference
values exactly, and the three agreement statistics (Bland–Altman bias and limits, Lin's concordance
correlation, and the intraclass correlation) match their closed-form or ANOVA-based references across
all 11 checks to machine precision. The instance-matching metrics that the accuracy benchmarks depend
upon — instance F1, average precision, and the matching routine — pass all 23 checks against an
independent intersection-over-union matcher and the canonical DSB2018 average-precision formula, so
that the numbers reported in the comparisons below are computed by a scorer that is itself verified.

Taken together, these results establish that fluorostats measures what it claims to measure:
integer-valued topological and combinatorial quantities agree exactly with analytic ground truth,
continuous physical quantities agree with their analytic definitions within a characterized
discretization tolerance, and the scoring and statistics machinery is exact to machine precision.
This correctness anchor is maintained as a regression-tested invariant — the reference-agreement
checks run as part of the test suite (105 tests at the current release) on every change — so that
exactness is a sustained guarantee rather than a one-time measurement. We now turn to how the tool
compares, on accuracy, to the established software for each capability.

### Nuclei segmentation: parity with validated deep learning on separated targets, with an explicit crossover

Nucleus instance segmentation is the task where deep-learning tools are strongest and where a
training-free method is therefore most exposed, so we treat it as the central test of the
parity-or-better claim and report it in the metric these tools use: instance F1 and average
precision computed by matching predicted to reference instances over a range of
intersection-over-union thresholds (Methods).

**The deep-learning baselines were validated before any comparison.** The credibility of a
training-free tool matching a trained one rests entirely on the trained baselines running at full
strength. We therefore first reproduced each baseline's published accuracy on BBBC039: our StarDist
run reached F1 0.871 against a published 0.864, and Cellpose reached 0.862, confirming the baselines
are configured correctly and are not straw men (Extended Data; [PublishedBaselines]). Only after
this check did we compare against fluorostats.

**On well-separated nuclei, fluorostats matches or exceeds the deep-learning tools.** On the full
BBBC039 test set (n = 200) fluorostats reaches an instance F1 of 0.896 (95% bootstrap CI [0.873,
0.916], 10,000 resamples), above StarDist (0.871), Cellpose (0.862), and Omnipose (0.802). The
paired per-image differences are statistically significant: fluorostats − StarDist = +0.025 [0.004,
0.042] and fluorostats − Cellpose = +0.034 [0.008, 0.057], both confidence intervals excluding zero
(Fig. 2). We emphasize what this comparison is and is not: it shows a deterministic, training-free,
CPU pipeline reaching the accuracy of three independently validated trained segmenters on
well-separated nuclei, not that classical thresholding is superior to deep learning in general.

Placed within its own configurable threshold family on BBBC039, fluorostats is competitive across
the board, and its Li-threshold configuration tops a twelve-method comparison at F1 0.934 (Table 2);
we report the full ranking rather than only the best configuration, because threshold choice is a
user-facing parameter whose behavior we characterize rather than hide. On DSB2018 — the canonical
nucleus set on which StarDist's published accuracy is reported — fluorostats reaches F1 0.789, the
top score among six classical baselines and approximately 91% of the trained model's published
0.864, at zero training cost on the exact data the deep-learning number comes from.

**The crossover to deep learning is real, mechanistic, and quantified.** The honest counterpart to
parity-on-separated-targets is collapse-on-crowded-instances, and we characterize it explicitly. On
the BBBC024 synthetic benchmark, as nuclear overlap increases from 0% to 75% clustering, instance F1
for fluorostats — and for every non-learned method tested — falls from 0.94 to 0.15, while trained
segmenters hold near 0.96–1.0 (Fig. 3; Table 3). The mechanism is specific: connected-component
labeling merges instances once they touch, and watershed splitting recovers only mild clustering
(BBBC039 0.896 → 0.899), not heavy overlap. This is a limit of the entire non-learned class, not of
fluorostats in particular, and it defines a clean decision rule — use fluorostats (or any threshold
pipeline) on well-separated targets, and a trained instance segmenter when instances heavily overlap
(Fig. 7). Because fluorostats is training-free, its accuracy does not depend on a modality being
represented in a training set, and its degradation with overlap is a smooth, predictable function of
a measurable image property rather than a distribution-shift cliff.

### Vascular networks: a general tool ties the specialists on their own benchmark

Vascular quantification is dominated by dedicated tools, so we ask a pointed question: how far does a
general, untuned, training-free tool get on the vascular specialists' *own* annotated benchmark? We
adopt the comparison protocol of REAVER [Corliss2020] — a single shared dataset, all tools evaluated
through one unified quantification of the same segmentations, and default parameters for every tool
(which we declare, following REAVER, as a deliberate limitation of the comparison).

**In 2D, fluorostats is at parity with AngioTool and beats two other specialists.** On the REAVER
dataset (n = 36 images with expert manual ground truth), inserted into REAVER's own five-tool
comparison, fluorostats ranks fourth of six on vessel-area-fraction error (MAE 0.076, concordance
0.701, Spearman 0.935), statistically indistinguishable from AngioTool (MAE 0.068) and ahead of RAVE
(0.094) and AngioQuant (0.149); the vascular specialist REAVER itself (0.017) and ImageJ (0.041) lead
(Table 4, Fig. 8). That an untuned general-purpose tool ties a dedicated angiogenesis package on that
package's own manually annotated benchmark is the intended result: parity, honestly ranked, with the
specialists that genuinely exceed it acknowledged. On real fibrin-bead sprouting-assay confocal data
(SproutAngio, n = 12), for which no ground truth exists, fluorostats and four threshold baselines all
detect the VEGF dose-response (Spearman of volume fraction and length against dose 0.59–0.74), with
metrics roughly doubling from low to mid VEGF and plateauing at high dose — a biologically plausible
saturation, and evidence of sensitivity to a known biological gradient absent any ground truth.

**In 3D, fluorostats recovers exact phantom values and agrees rank-wise with a 3D specialist.**
Against synthetic 3D vascular phantoms with exact ground truth, fluorostats recovers centerline
length within 0.6–2.4%, branch count exactly, and volume fraction exactly (Table 5). On real
light-sheet volumes we compared against VesselExpress [Spangenberg2023], framed explicitly as a
software-versus-software agreement because the reference is that tool's own pipeline output rather
than manual expert tracing. Here the choice of threshold is decisive and drove a capability we then
built: the Otsu default badly under-segments dim, sparse light-sheet vessels (Dice 0.089 against the
VesselExpress segmentation), while a Li threshold recovers to 0.598. This motivated an automatic
threshold-selection mode (`method="auto"`) that detects when Otsu retains implausibly little signal
and switches to Li — which it did on all nine volumes, matching the best manual choice — and a
majority-vote consensus mode. We report honestly that consensus fails here (0.094): because most
threshold algorithms share the same under-segmentation on this data, the majority inherits it, and
`auto` is the correct answer. On the vessel volume-fraction metric, the two tools rank the nine
volumes consistently (Spearman 0.75) but fluorostats reads about 1.7-fold higher in absolute terms
(0.049 versus 0.029; concordance 0.11) — a systematic offset, named mechanistically (the Li threshold
is more inclusive than the VesselExpress pipeline), of the same character as the tool-to-tool
differences REAVER documents among 2D vascular packages (Fig. 13). We omit a skeleton-length
comparison on these volumes because skeletonizing full 250-MB dense light-sheet volumes is
computationally intractable — a scope limit we state rather than work around.

### Depth-resolved viability: quantifying what two-dimensional shortcuts miss, and tying a published tool

Live/Dead viability is the capability where the image-in-3D-but-quantify-in-2D problem is most
consequential and where fluorostats' volumetric treatment is most distinctive. We make two claims:
that two-dimensional shortcuts systematically bias the live fraction, and that fluorostats'
count-based viability ties a published tool exactly while extending to depth-resolved 3D.

**Two-dimensional shortcuts bias the live fraction, in a consistent direction.** On a public Day-14
Live/Dead stack (S-BIAD2130), taking the true voxelwise 3D live fraction (0.570) as reference, every
two-dimensional or heuristic reduction biases it upward: a maximum-intensity projection by +5.0%, a
mid-plane slice by +1.5%, brightest-focus selection by +1.7%, and a naive mean of per-slice fractions
by +25.2% — the last because it over-weights sparse deep slices. Attenuation correction stays within
2.7% of the volumetric value (Table 6). Because the same volume is passed through every pipeline, the
differences are attributable to the reduction method rather than to the sample, and the direction of
the bias is consistent — the readout that a laboratory would report from a projection overestimates
viability relative to the volumetric truth.

**A new count-based mode ties a published Fiji macro exactly.** A head-to-head against a published
Fiji Live/Dead macro [Kerkhoff2024] on its own synthetic benchmark, where per-cell counts and hence
true viability are known by construction, initially exposed a genuine gap: fluorostats had no
peak-counting mode, so its area- and connected-component-based readouts trailed the macro on crowded
cells (the same overlap wall seen for nuclei). We built the missing capability — a local-maxima
counting mode — and fluorostats now ties the published tool exactly (mean absolute error 0.016,
concordance correlation 0.987, versus the macro's identical 0.016 and 0.987; Table 7, Fig. 5),
through its own programmatic interface and training-free. Reproducing an established tool on its own
data is the credibility down-payment that licenses the volumetric claim above: the same library that
matches the standard 2D readout is the one that shows the 2D readout is biased in 3D.

**No single counting mode is universal, and we say so as guidance rather than a caveat.** A
size-and-noise sweep with a known count of 100 shows that local-maxima counting wins on crowded,
single-peak cells but over-counts flat or noisy cells (3,068 detections at high noise), while
connected-component counting is far more robust to noise (68) and area answers coverage. fluorostats
therefore offers connected-component, watershed, local-maxima, a transparent conservative automatic
mode, and an all-modes consensus, and we present the regime-dependence as an explicit decision guide
(which mode for which density, depth, and noise regime; Table 8). Because crowding and noise are
statistically difficult to separate from image intensity alone, the automatic mode biases toward the
robust method and reports its reasoning rather than pretending to be an oracle — a limitation we
state directly.

### Spatial homogeneity and integrated statistics

**A simple tile-based index tracks five rigorous spatial statistics.** Spatial homogeneity of signal
or objects is often quantified with a single dispersion index, but the closest recent work validates
such an index against biochemical ground truth rather than against established point-pattern
statistics, and ships neither a Python implementation nor a significance test [Martin2026].
fluorostats' segmentation-free tile-based Gini index closes exactly that gap: across a controlled
regular-to-clustered sweep it tracks all five established spatial statistics — the Morisita index
(Spearman 0.997), quadrat variance (0.997), gliding-box lacunarity (0.981), the Clark–Evans
nearest-neighbour index (−0.983), and Ripley's K/L deviation (0.960) — and separates uniform from
clustered fields with an area-under-the-curve of 1.0 in every case (Table 9, Fig. 9). An object-based
centroid variant behaves equivalently (Spearman 0.975 against the clustering parameter; |rho| ~0.99
against the reference statistics). The index is thus a simple, fast, segmentation-free proxy for
rigorous point-pattern analysis, and we state its principled limits — a single fixed tile scale and
no built-in test of complete spatial randomness — which the integrated statistics layer below is
designed to supply.

**Correct-by-default small-sample statistics, in the pipeline.** The measurements above are only as
trustworthy as the inference applied to them, and small-sample microscopy studies are a documented
site of statistical error — pseudoreplication that treats cells as independent replicates inflates
false-positive rates, and parametric tests are frequently mis-applied to small, non-normal samples
[Lord2020; Lazic2010]. fluorostats makes the defensible choice the default: Mann–Whitney U tests with
Cliff's delta effect sizes, stratified rank tests and the Scheirer–Ray–Hare interaction test for
designs with structure, Benjamini–Hochberg control across metrics and strata, and bootstrap
confidence intervals for fold changes. Each of these is validated to exact agreement with its
reference implementation (Section "Correctness"), and because they operate on the objects the
quantification already produced, an analysis proceeds from image to multiplicity-controlled
significance without a manual export. The library also provides bootstrap power analysis from pilot
data; we note, and document, that power estimated from a small pilot is optimistic — an inherent
statistical property [Albers2018], reported here as a caveat for the user's own experiments rather
than a defect of the tool.

### Runtime and determinism

The accessibility argument for a CPU-only tool depends on it being fast enough to be practical, and
the reproducibility argument depends on it being deterministic; we quantify both. On BBBC039,
per-image 2D segmentation takes 14.5 ms on CPU, on par with the classical thresholds it is built from
(Otsu/Isodata 5.7 ms, Triangle 5.9 ms, Li 9.1 ms, watershed 35.6 ms) and approximately 15-fold faster
than StarDist (215 ms) and 380-fold faster than Cellpose (5,547 ms) measured on the same CPU (Table
10, Fig. 4). A per-metric timing table across the full library (Extended Data) shows fluorostats is at
parity with its library equivalents on shared operations — statistics and agreement functions run in
0.01–0.4 ms against SciPy's 0.09–0.24 ms; 3D connectivity metrics and local-maxima counting track
their scikit-image counterparts — and we report the few genuinely heavy operations honestly (average
precision at ~20 s and instance F1 at ~2 s on large label images, and watershed splitting and
background subtraction at ~9 s, each tracking its underlying library routine). These are
validation-time or once-per-volume operations, not per-frame metrics. Every timing was obtained on
CPU with no GPU present.

Because no function in the measurement path uses learned weights or random seeds, repeated runs on
identical inputs return bit-identical outputs. We treat this determinism as a reportable result
rather than an implementation footnote: it means a published fluorostats number can be reproduced
exactly from the archived code and data, without the seed-, framework-, and hardware-dependent
variability that trained pipelines exhibit.

## Discussion

fluorostats occupies a specific and, we argue, underserved niche: the reproducible, training-free,
CPU-only quantification of fluorescence microscopy for the large fraction of analyses that do not
require a trained instance segmenter. Its contribution is not a new segmentation algorithm — it
deliberately builds on established, published algorithms — but the combination of three properties
that, together, no existing tool provides: reference-exact correctness for every metric, a
deterministic CPU-only implementation that removes the GPU and training barriers to reproducibility,
and an integrated small-sample statistics layer that carries a measurement through to a
multiplicity-controlled inference without a manual export. The benchmark results show that this
combination does not come at the cost of accuracy on the tasks it targets: fluorostats matches or
exceeds validated deep-learning segmenters on well-separated nuclei, ties a dedicated angiogenesis
package on that package's own annotated benchmark, ties a published viability macro exactly, and
tracks five rigorous spatial statistics with a simple index.

The tool is explicitly complementary to deep learning rather than a replacement for it. On heavily
overlapping instances — the regime trained segmenters are built for — fluorostats and the entire
class of non-learned methods are outperformed, and we have characterized where that crossover occurs
so that the tool can tell a user when to reach for a trained model. This complementarity is the
honest and, we think, the useful framing: a training-free tool that is correct by construction,
reproducible by design, and candid about its scope is the right first instrument for a great many
quantification tasks, and a clear signpost to deep learning for the rest. For the tissue-engineering
and vascular-biology laboratories that generate volumetric fluorescence data without routine access
to GPUs or annotation budgets, and for whom reproducibility of a reported number matters as much as
its accuracy, fluorostats is designed to be that first instrument.

## Scope and limitations

We state the boundaries of the tool plainly; several are quantified in the results above, and we
collect them here with, for each, the mechanism, the boundary, and the guidance that follows.

*Crowded and overlapping instances.* Connected-component labeling merges instances once they touch,
so instance F1 collapses (0.94 → 0.15 from 0% to 75% clustering on BBBC024) as overlap rises. This is
a limit of the whole non-learned class, not of fluorostats specifically. Guidance: on heavily
overlapping instances, use a trained instance segmenter (StarDist, Cellpose, Omnipose); fluorostats
can consume such masks for downstream measurement.

*Threshold choice.* The default Otsu threshold under-segments dim, sparse signal (light-sheet
vessels: Dice 0.089, recovered to 0.598 with Li). The automatic mode mitigates this by a documented
heuristic, but it is not an oracle, and the consensus mode fails when most algorithms share a failure
mode. Guidance: on dim or sparse volumetric signal, use or verify the automatic/Li threshold; report
the threshold used.

*Counting method for viability.* Local-maxima counting ties the published macro on crowded cells but
over-counts flat or noisy cells; connected-component counting is more noise-robust. No automatic
selector is reliable because crowding and noise are not separable from image statistics alone.
Guidance: choose the mode from the assay (Table 8); when unsure, the conservative automatic mode and
the reported consensus spread indicate the uncertainty.

*Hard cytoplasmic 3D and small-dim nuclei.* On challenging Cell Tracking Challenge sets an untuned
threshold pipeline gives moderate foreground overlap (Dice 0.52–0.69), and the best threshold is
dataset-dependent — fluorostats is a semantic quantifier here, not a tuned instance segmenter.

*Continuous metrics near the sampling limit.* Integer topological quantities are exact, but skeleton
length and branch counts degrade for terminal branches near the raster resolution (depth-4 tree: 27
versus 31 branches). Guidance: interpret continuous morphometrics with the stated discretization
tolerance.

*Evaluation-side limits, distinct from the tool.* Two of our comparisons rest on non-ideal ground
truth, and we separate these from limitations of the method itself. The VesselExpress 3D comparison is
software-versus-software agreement against a pipeline-generated segmentation, not a gold-standard
accuracy test against manual expert tracing. And bootstrap power estimated from a small pilot is
optimistic by construction; we present power curves with that caveat, as a statement about the
reader's experimental design rather than about the tool.

*Benchmark correctness safeguards.* Because the benchmark scripts are themselves software, we treated
their correctness as a first-class concern. Reference-agreement and invariant checks (for example,
that zoom-normalized density is exactly invariant, and that instance scorers match an independent
matcher) are part of the test suite, and during development they caught and we corrected several
script-level errors — including a Stouffer-convention mismatch and a zoom-invariance counting error —
before any reported result depended on them. All numbers in this paper derive from the audited,
version-tagged code, and the scoring routines that produce the comparison numbers are themselves
validated (23/23 instance-metric checks; Section "Correctness"). We report this not as an admission
but as evidence that the evaluation harness has the same correctness discipline as the library it
tests, following established guidance on reproducible computational research [Sandve2013; Miura2021].

## Methods

*Online Methods, to be placed at the end of the manuscript per Nature Methods convention; move
inline for a journal that integrates Methods.*

### Software implementation

fluorostats (v0.7.0) is implemented in Python and depends on NumPy, SciPy, scikit-image, pandas,
tifffile, and czifile. It exposes both a library API, in which every metric is a pure function, and
a command-line interface, and reads the major microscopy formats (confocal and light-sheet z-stacks,
widefield; TIFF, CZI, and others). The library comprises 19 modules organized along the workflow
(input/output, preprocessing, segmentation, object handling, 2D and 3D metrics, skeleton, topology,
vascular, viability, homogeneity, agreement, statistics, power, reporting/figures), and is covered by
105 automated tests including the reference-agreement checks described under "Correctness". No
function requires a GPU, and no function in the measurement path uses random seeds; all benchmarks
below were run on CPU.

### Datasets

All benchmark data are public and citable. Nuclei: BBBC039 and BBBC024 from the Broad Bioimage
Benchmark Collection [Ljosa2012], and the 2018 Data Science Bowl set (DSB2018) [Caicedo2019].
Three-dimensional segmentation: Cell Tracking Challenge fluorescence sets (Fluo-C3DH-A549,
Fluo-N3DH-CHO) [Maska2014; Maska2023]. Vascular: the REAVER annotated dataset [Corliss2020], the
SproutAngio VEGF dose-response set (Zenodo 7240927), and the VesselExpress light-sheet volumes
(Zenodo 6025935) [Spangenberg2023]. Viability: a public Live/Dead stack (S-BIAD2130) and the Kerkhoff
Fiji-macro synthetic benchmark (Zenodo 10395753) [Kerkhoff2024]. Accession identifiers and download
URLs are listed in the data availability statement and the repository's data manifest.

### Metric definitions and evaluation

Instance segmentation accuracy is reported as instance F1 and average precision computed by matching
predicted to ground-truth instances above an intersection-over-union threshold; F1(t) =
2·TP(t) / [2·TP(t) + FP(t) + FN(t)], with average precision averaged over a range of IoU thresholds
(0.5–0.9), following the DSB2018 convention [Caicedo2019]. Semantic overlap is reported as foreground
Dice and Jaccard. Vascular agreement uses vessel area fraction (2D) and volume fraction (3D), with
concordance correlation, Spearman correlation, and mean absolute error against the reference; the
accuracy/precision decomposition and the zero-bias test follow REAVER [Corliss2020]. Viability is the
live fraction (live count or volume over total); agreement with the reference macro uses mean absolute
error and Lin's concordance correlation. Spatial homogeneity uses a tile-based Gini index compared by
Spearman correlation and uniform-versus-clustered area-under-the-curve against five reference
statistics (Clark–Evans, Ripley's K/L, Morisita, quadrat variance, lacunarity). Topology metrics
(Euler number, connected components, largest-connected-component fraction) and skeleton metrics
(length, branch and junction counts) are validated against analytic phantoms and the Lee-1994
reference algorithm. All scoring routines are themselves validated (Section "Correctness").

### Baseline configuration and validation

Deep-learning baselines were run at their published configurations on CPU: StarDist
(2D_versatile_fluo), Cellpose (v3, nuclei model), and Omnipose. Each was first validated to reproduce
its published accuracy on BBBC039 before any comparison (StarDist observed F1 0.871 versus published
0.864; Cellpose 0.862; [PublishedBaselines]). A single evaluation policy was applied across all
methods — identical test images, identical scoring, default parameters, and, for fluorostats, no
per-dataset training — and any asymmetry is stated where it arises. Classical thresholding baselines
(Otsu, Li, Isodata, Triangle, Yen, Mean, Minimum, watershed) were run through the same quantification
and scoring as fluorostats.

### Statistics

Group comparisons use the Mann–Whitney U test with Cliff's delta effect sizes; structured designs use
stratified rank tests and the Scheirer–Ray–Hare test; multiple comparisons are controlled by the
Benjamini–Hochberg false-discovery rate. Confidence intervals are bootstrap (10,000 resamples unless
stated); the nuclei head-to-head reports paired per-image differences with bootstrap confidence
intervals, with parity defined as an interval overlapping zero and superiority as an interval
excluding zero. Power analysis is bootstrap from pilot data, reported with the small-pilot optimism
caveat [Albers2018]. All statistical functions are validated to exact agreement with reference
implementations (Section "Correctness").

### Compute environment

Local benchmarks were run with Python 3.13. Deep-learning baselines were run on an AMD ROCm HPC
cluster on CPU (partitions and scripts in the repository's benchmark directory). All timings were
measured on CPU with no GPU present, on the hardware noted in the runtime table.

## Data availability

All datasets analyzed are public: BBBC039, BBBC024 (Broad Bioimage Benchmark Collection); DSB2018;
Cell Tracking Challenge Fluo-C3DH-A549 and Fluo-N3DH-CHO; the REAVER dataset; SproutAngio (Zenodo
7240927); VesselExpress (Zenodo 6025935); S-BIAD2130; and the Kerkhoff synthetic Live/Dead benchmark
(Zenodo 10395753). Accession identifiers and download URLs are provided in the repository data
manifest. The raw image data are redistributed under their original licenses; derived benchmark
tables (the CSV outputs behind every figure) are deposited under a Creative Commons license with a
DOI [to be minted at submission].

## Code availability

fluorostats is open source under a permissive OSI-approved license at
github.com/rsiegemit/fluorostats, installable from source and the Python package index, and the
exact version used for this paper (v0.7.0) is archived with a citable DOI on Zenodo [to be minted at
submission]. The complete benchmark harness — one script per comparison, with pinned dependencies —
and scripts that regenerate every figure and table are included in the repository. Because the
library is deterministic and CPU-only, every reported number is reproducible without a GPU, model
weights, or a fixed random seed.

## Availability and requirements

- **Project name:** fluorostats
- **Home page:** github.com/rsiegemit/fluorostats
- **Operating systems:** platform-independent (Linux, macOS, Windows)
- **Programming language:** Python (3.13 tested)
- **Other requirements:** NumPy, SciPy, scikit-image, pandas, tifffile, czifile
- **License:** [OSI-approved permissive license — MIT/BSD/Apache, to confirm]
- **Restrictions:** none for academic or commercial use

## References

*Reference list to be assembled with a reference manager. Placeholder keys used in the text, with the
proof-stage verification flags carried from the research dossiers (research/00_SYNTHESIS.md §5):*

- [Schmidt2018] Schmidt et al., Cell Detection with Star-convex Polygons, MICCAI 2018; [Weigert2020]
  StarDist-3D, WACV 2020 (cite the 2020 paper for 3D).
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
- [Maska2014] Maška et al., Cell Tracking Challenge, Bioinformatics 2014; [Maska2023] 10-year CTC,
  Nature Methods 2023.
- [Martin2026] Martin et al., dispersion indices, iScience 2026 (confirm final volume/pages at typeset).
- [Lord2020] Lord et al., SuperPlots, J. Cell Biol. 2020.
- [Lazic2010] Lazic, pseudoreplication, BMC Neuroscience 2010.
- [Laine2021] Laine et al., replication crisis in DL bioimage analysis, Nature Methods 2021.
- [Spiller2025] Spiller & Duarte Campos, Front. Bioeng. Biotechnol. 2025.
- [Pereira2023] Pereira et al., angiogenesis software review, Int. J. Mol. Sci. 2023.
- [Riley2023] Riley — magnification/zoom reproducibility (confirm citation).
- [Albers2018] Albers & Lakens, pilot power optimism, 2018.
- [Sandve2013] Sandve et al., Ten Simple Rules for Reproducible Computational Research, PLoS Comput.
  Biol. 2013.
- [Miura2021] Miura & Nørrelykke, reproducible image handling and analysis, EMBO J. 2021.
- [PublishedBaselines] Internal baseline-validation record (data/PUBLISHED_BASELINES.md).
- Delesse–Glagolev stereology primaries: cite via a modern review.

## Display items (to be produced in the figures pass)

Main: **Fig. 1** pipeline schematic (training-free/CPU, module map → integrated stats/figures);
**Fig. 2** nuclei F1 ranking + bootstrap-CI forest plot; **Fig. 3** qualitative gallery (raw/GT/
prediction across modalities incl. VesselExpress overlays); **Fig. 4** vascular ranking + homogeneity
five-statistic correlation; **Fig. 5** viability agreement + 3D phantom exact-GT; **Fig. 6**
clustering-degradation crossover + timing (log) + scope-boundary bars; **Table 1** master validation
table. Extended Data / Supplementary: per-dataset breakdowns, full per-metric timing, DSB2018, CTC,
noise/denoise/size sweeps, VesselExpress metric agreement, the remaining plots, and the reporting
checklists.


