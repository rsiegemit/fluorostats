# Skeleton / Filament Datasets for fluorostats Benchmark B1

Benchmark **B1 (skeleton validity)** validates fluorostats skeleton morphometry
(total length, branch count, junction/bifurcation count, endpoints) against:

1. **Exact ground truth** — synthetic phantoms with known topology (primary strategy).
2. **Gold-standard manual reconstructions** (SWC graphs) from public neuron/vessel data.
3. **Cross-tool agreement** with Fiji `AnalyzeSkeleton` (both use the Lee-1994
   3-D medial-surface thinning algorithm and should agree *exactly* on skeleton
   topology given the same binary mask).

All entries below were verified by fetching the source pages (July 2026).
Sizes marked "~" are approximate/estimated and flagged as such.

---

## RECOMMENDED PRIMARY: Synthetic tubular phantoms (exact ground truth)

Because fluorostats and Fiji AnalyzeSkeleton **share the Lee-1994 thinning
algorithm** (Lee, Kashyap & Chu 1994, *CVGIP* 56(6):462–478), the two tools
should produce topologically identical skeletons from the *same binary mask*.
Synthetic phantoms give exact, closed-form ground truth (no external data, no
license, fully reproducible in CI) and are the cleanest way to validate B1.

**Two complementary approaches:**

### A. In-house parametric phantoms (RECOMMENDED — build this first)
Draw tubular structures with analytically known counts into a 3-D (or 2-D)
binary volume, then skeletonize:

- **Straight rod**: 1 branch, 0 junctions, 2 endpoints, length = L (in voxels,
  scalable by voxel size). Validates length calibration directly.
- **Y-junction**: 3 branches, 1 junction, 3 endpoints. Validates bifurcation
  detection. Known branch lengths from segment geometry.
- **Grid/lattice**: N horizontal + M vertical bars → closed-form junction and
  branch counts. Stresses degree-4 nodes.
- **Binary tree (n generations)**: 2^n − 1 branches, 2^(n−1) − 1 junctions,
  2^(n−1) endpoints. Validates recursive topology and total length = sum of
  segment lengths.

Caveats to document: (i) thinning of thick/curved tubes can shift junction
voxels and create short spurs — count *topological* branches, not raw voxel
runs, and set a spur-pruning threshold; (ii) diagonal vs. orthogonal length
weighting (√2, √3 for 26-connectivity) must match between fluorostats and the
ground-truth length formula — this is itself a thing B1 should test; (iii) two
junctions closer than the tube radius may merge — keep phantom features well
separated. These caveats are the *point* of B1: they surface real algorithmic
edge cases.

**Ground-truth-length trick**: for a piecewise-linear centerline the true
length is the sum of Euclidean segment lengths; render the tube by dilating the
centerline, so the analytic length is known before skeletonization.

### B. VascuSynth — procedural 3-D vascular trees with exported ground truth
- **Citation**: Jassi P, Hamarneh G. "VascuSynth: Vascular Tree Synthesis
  Software." *Insight Journal*, Jan–Jun 2011. Also: Hamarneh G, Jassi P.
  "VascuSynth: Simulating vascular trees for generating volumetric image data
  with ground-truth segmentation and tree analysis." *Computerized Medical
  Imaging and Graphics* 34(8):605–616, 2010 (DOI: 10.1016/j.compmedimag.2010.06.002).
- **What it exports**: a volumetric image (default 100×100×100 voxels) **plus**
  ground-truth segmentation, **bifurcation locations, per-branch properties,
  and full tree hierarchy** — i.e., exactly the topology B1 needs (branch count,
  junction count, per-branch length).
- **Software (source)**: https://github.com/erudianart/VascuSynth (requires
  CMake + ITK). Insight Journal record: https://insight-journal.org/browse/publication/794/
- **Precomputed sample datasets** (no need to build ITK):
  https://vascusynth.cs.sfu.ca/Data.html — the **March 2013 VascuSynth Dataset**
  = **120 datasets (10 groups × 12 volumes)**. Downloads:
  `March_2013_VascuSynth_Dataset.zip` (full bundle) or `Group1.zip`…`Group10.zip`
  + `README.txt`. Formats/total size not stated on page (verify via README).
- **License**: not stated on the data page — confirm terms in README before
  redistribution; software is academic/open (ITK-based).
- **Feeds B1**: branch length + bifurcation-count validation against known tree
  hierarchy. Widely used precisely this way (prior work measured "total length
  of correct branches" and "number of correct bifurcations" vs. true values).
- **Caveat**: ground truth is the *synthesis graph*, not a re-derived skeleton;
  small differences vs. thinning are expected at bifurcations — treat as
  tolerance-band agreement, not exact match (unlike approach A).

---

## GOLD-STANDARD MANUAL RECONSTRUCTIONS (real images + SWC graphs)

### 1. DIADEM Challenge datasets  ← best real-data first choice
- **Citation**: Brown KM, Barrionuevo G, Canty AJ, De Paola V, Hirsch JA,
  Jefferis GSXE, Lu J, Snippe M, Sugihara I, Ascoli GA. "The DIADEM data sets:
  representative light microscopy images of neuronal morphology to advance
  automation of digital reconstructions." *Neuroinformatics* 9(2–3):143–157,
  2011. **DOI: 10.1007/s12021-010-9095-5**. PMID: 21249531.
  Metric paper: Gillette TA, Brown KM, Ascoli GA, "The DIADEM Metric,"
  *Neuroinformatics* 9(2–3):233–245, 2011 (PMC4339018).
- **Access**: diademchallenge.org now redirects to **https://diadem.janelia.org/**
  ("image stacks, gold standard reconstructions, and an objective metric").
  Per-dataset README pages (historically at diademchallenge.org, e.g.
  `olfactory_projection_fibers_readme.html`).
- **Six dataset collections** (species/region/modality vary); five commonly
  redistributed via Fiji: **Olfactory Projection Fibers (OP)**, **Neocortical
  Layer 1 Axons (NC)**, **Cerebellar Climbing Fibers (CF)**, **Hippocampal CA3
  Interneuron**, **Neuromuscular Projection Fibers**. Fiji visualization guide:
  https://imagej.net/events/diadem-challenge-data
- **Format**: 8-bit/RGB TIFF image stacks (per-slice sequences; CF ~3.4 GB,
  others smaller) **+ gold-standard SWC** reconstructions (e.g. `OP_1.swc`).
  All **3-D**.
- **Ground truth**: yes — SWC graphs give exact bifurcation, termination, and
  topology; SWC → per-branch length and junction/endpoint counts directly.
- **License/registration**: challenge historically required
  registration/agreement to terms; verify current terms on the Janelia mirror
  before use. Cite the data paper.
- **Feeds B1**: junction/branch/endpoint counts + total length vs. manual SWC.
  **Best real-data starting point** (small, well-curated, SWC is the reference
  standard for skeleton topology).
- **Caveat**: gold-standard SWC is a *manual centerline trace*, not a thinning
  of a binary mask — expect systematic length differences (tracing smooths;
  thinning follows voxels). Use as tolerance-band comparison; segment images to
  a mask first, then compare fluorostats-vs-AnalyzeSkeleton *exactly* and both
  vs. SWC approximately.

### 2. BigNeuron  ← largest real 3-D corpus with gold reconstructions
- **Citation**: Peng H, Hawrylycz M, Roskams J, Hill S, Spruston N, Meijering E,
  Ascoli GA. "BigNeuron: Large-Scale 3D Neuron Reconstruction from Optical
  Microscopy Images." *Neuron* 87(2):252–256, 2015. (Follow-up: Manubens-Gil
  et al., "BigNeuron: a resource to benchmark…," *Nature Methods* 2023,
  DOI: 10.1038/s41592-023-01848-5.)
- **Downloads** (GitHub, verified): https://github.com/BigNeuron/Data/releases
  - `Gold166_v1` — **166 neuron datasets: raw image stacks + gold-standard
    reconstructions** (gold166.zip; ~1.5 GB est., flagged approximate). ← use this
  - `data_v1.0_first2000` — 2,000 fruitfly image stacks (8-bit Vaa3D raw,
    voxel 0.32×0.32×1 µm), no gold SWC (dev set only).
  - `gold166_bt_v1.0` — 7,978 algorithm reconstructions of the gold166 set
    (for comparing tracers; two tarballs 2,407 + 5,571 reconstructions).
- **Format**: Vaa3D raw / TIFF image stacks + SWC reconstructions. **3-D**.
- **Ground truth**: yes — consensus gold-standard SWC per neuron.
- **License**: requires citing BigNeuron + Peng et al. 2015; fruitfly stacks
  also require agreement to Allen Institute terms.
- **Feeds B1**: same as DIADEM but at scale (multi-species). Same manual-SWC-vs-
  thinning caveat.

### 3. NeuroMorpho.Org  ← massive SWC library (topology only, images separate)
- **Citation**: Ascoli GA, Donohue DE, Halavi M. "NeuroMorpho.Org: a central
  resource for neuronal morphologies." *J Neurosci* 27(35):9247–9251, 2007.
  Current: Tecuatl C, Ljungquist B, Ascoli GA (2024), "Accelerating the
  continuous community sharing of digital neuromorphology data." RRID:SCR_002145.
- **Access**: https://neuromorpho.org/ — largest public inventory of digitally
  reconstructed neurons/glia; **all reconstructions downloadable as SWC**,
  centrally curated to the SWC standard.
- **Format**: SWC only (no raw images bundled). **3-D** graphs.
- **Ground truth**: the SWC *is* the reference topology (branch/junction/
  endpoint counts, per-branch length computable directly).
- **License/terms**: https://www.neuromorpho.org/useterm.jsp — free, cite
  required references.
- **Feeds B1**: use as a large source of **realistic tree topologies** to
  render into synthetic volumes (rasterize SWC → mask → skeletonize → compare
  recovered counts to the SWC's known counts). Bridges "synthetic exactness"
  with "realistic morphology."
- **Caveat**: no images → cannot test segmentation; only the skeleton-from-mask
  step. Rasterization thickness affects recovered junctions (document radius).

### 4. MitoGraph validation data (mitochondrial networks)  ← domain-matched
- **Citation**: Viana MP, Lim S, Rafelski SM. "Quantifying mitochondrial
  content in living cells." *Methods in Cell Biology* 125:77–93, 2015
  (DOI: 10.1016/bs.mcb.2014.10.003; PMID 25640425). Software paper context:
  Rafelski lab MitoGraph.
- **Access**: https://github.com/vianamp/MitoGraph (software + example data);
  R scripts / example datasets in `MitoGraph-Contrib-RScripts`;
  mirror: https://rafelski.com/susanne/MitoGraph
- **Format**: 3-D fluorescence z-stacks of tubular mito networks; MitoGraph
  outputs graph/topology (nodes, edges, lengths).
- **Ground truth**: MitoGraph v2.0 **validated against manual counts** — 96%
  correct node counts, 91% correct edge counts in WT budding yeast. Manual
  node/edge counts serve as the reference for tubular-network topology.
- **License**: open-source (GitHub); cite Viana et al. 2015.
- **Feeds B1**: directly analogous use case (tubular fluorescence networks,
  total length + node/edge counts). Strong external cross-check because
  fluorostats targets the *same* quantities on the *same* kind of data.
- **Caveat**: MitoGraph uses its own graph extraction (not Lee-1994 thinning),
  so agreement is method-level, not voxel-exact — a good independent sanity
  check rather than a topology oracle.

---

## VESSEL / CURVILINEAR (2-D, secondary)

### 5. DRIVE & STARE retinal vessels  ← 2-D masks, skeletonizable
- **DRIVE**: 40 fundus images (565×584), expert vessel segmentations (2 obs. on
  test set). https://drive.grand-challenge.org/ — Staal et al., *IEEE TMI*
  23(4):501–509, 2004 (DOI: 10.1109/TMI.2004.825627).
- **STARE**: 20 images (700×605), 2 manual segmentations each.
  https://cecas.clemson.edu/~ahoover/stare/ — Hoover et al., *IEEE TMI*
  19(3):203–210, 2000 (DOI: 10.1109/42.845178).
- **Format**: 2-D images + binary vessel masks (no SWC skeleton graph provided).
- **Ground truth**: pixel-level vessel masks, **not** skeleton graphs — you
  derive the skeleton yourself, so there is *no independent* branch/junction
  ground truth. Value = a real 2-D binary mask on which fluorostats and Fiji
  AnalyzeSkeleton must agree **exactly** (both Lee-1994).
- **License**: DRIVE via grand-challenge (registration); STARE academic use.
- **Feeds B1 (cross-tool only)**: exact fluorostats-vs-AnalyzeSkeleton parity
  on a nontrivial real mask. Does *not* give topological ground truth.

---

## Summary: recommended B1 plan

1. **Build in-house parametric phantoms** (rod, Y, grid, binary tree) → exact
   counts + exact lengths. Primary, reproducible, no license. Assert
   fluorostats == analytic ground truth **and** fluorostats == AnalyzeSkeleton
   (byte-exact topology, shared Lee-1994).
2. **Add VascuSynth** precomputed volumes (SFU sample zips) for realistic 3-D
   trees with exported bifurcation/branch ground truth.
3. **Rasterize NeuroMorpho / DIADEM SWC** into masks for realistic-morphology
   round-trip tests (recovered counts vs. known SWC counts).
4. **Cross-check MitoGraph** validation data as a domain-matched external
   benchmark (tubular fluorescence networks, node/edge counts).
5. Keep **DRIVE/STARE** only as 2-D exact-parity checks vs. AnalyzeSkeleton.

**Key algorithmic note for the paper**: fluorostats and Fiji AnalyzeSkeleton
both implement Lee-Kashyap-Chu 1994 thinning + Arganda-Carreras's Skeletonize3D/
AnalyzeSkeleton graph analysis (imagej.net/plugins/analyze-skeleton). Given an
identical binary mask and matched connectivity/length-weighting conventions,
they should agree **exactly** — so B1's cross-tool test is a correctness proof,
while the phantom/SWC tests validate against real-world topology.
