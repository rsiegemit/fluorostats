# Competitor skeleton & topology benchmarks — reference values, phantoms, and sample data

Purpose: validate **fluorostats** against the tools it competes with, using **their own** benchmark data
and reported values. This document maps each tool to (a) the exact metric fluorostats matches, (b) the
number/phantom to reproduce, and (c) any downloadable competitor sample data.

Verification status is flagged per item: **[VERIFIED]** = confirmed by fetching a primary source in this
session; **[PARTIAL]** = confirmed indirectly / from secondary source; **[UNVERIFIED]** = claimed but not
confirmed against the primary paper (paywalled) — do not cite as fact without checking the PDF.

---

## 1. Fiji / ImageJ — AnalyzeSkeleton + Skeletonize3D

**Citation [VERIFIED]:** Arganda-Carreras I., Fernández-González R., Muñoz-Barrutia A., Ortiz-De-Solorzano C.
"3D reconstruction of histological sections: Application to mammary gland tissue." *Microscopy Research and
Technique* 73(11):1019–1029, 2010. **DOI: 10.1002/jemt.20829**
(The imagej.net AnalyzeSkeleton page also credits it inside BoneJ — Doube et al. 2010, Bone 47(6):1076–1079.)

**Thinning algorithm [VERIFIED]:** Skeletonize3D implements the 3D medial-surface/axis thinning of
**Lee, Kashyap & Chu (1994)**, "Building skeleton models via 3-D medial surface/axis thinning algorithms,"
*CVGIP: Graphical Models and Image Processing* 56(6):462–478 (DOI: 10.1006/cgip.1994.1042).
**This is the same algorithm fluorostats uses** → exact voxel-for-voxel agreement is expected on identical
binary input, so this is fluorostats' tightest benchmark.

**Metrics AnalyzeSkeleton reports [VERIFIED]** (per skeleton / per tree):
- # branches (slab segments: endpoint–endpoint, endpoint–junction, junction–junction)
- # junction voxels, # actual junctions, # triple points, # quadruple points
- # end-point voxels, # slab voxels
- average branch length, maximum branch length (in calibrated units)

**Metric fluorostats matches:** branch count, junction count, triple-point count, endpoint count, and total/
average branch length — all derived from the Lee-1994 skeleton graph.

**Number to reproduce:** there is **no single canonical "gold number"** published for a named image. Instead,
reproduce **exact equality** on shared input:
- The Fiji docs use the **"bat cochlea volume"** sample (Fiji ▸ File ▸ Open Samples ▸ Bat Cochlea Volume) for
  Skeletonize3D/AnalyzeSkeleton demos. Run both tools on it and confirm fluorostats returns identical
  branch/junction/endpoint counts. **[VERIFIED that this is the demo image; exact counts NOT published — must
  be generated locally by running Fiji.]**
- Strategy: run Fiji AnalyzeSkeleton on any phantom, record its output, and require fluorostats to match to 0.

**Downloadable sample data:**
- Bat Cochlea Volume ships inside Fiji/ImageJ (Open Samples menu) — no separate URL. **[VERIFIED as bundled sample]**
- Plugin docs: https://imagej.net/plugins/analyze-skeleton/ and https://imagej.net/plugins/skeletonize3d
- Author software page: https://sites.google.com/site/iargandacarreras/software

---

## 2. BoneJ — Connectivity (Euler characteristic → connectivity density)

**Citation [VERIFIED]:** Doube M., Kłosowski M.M., Arganda-Carreras I., Cordelières F.P., Dougherty R.P.,
Jackson J.S., Schmid B., Hutchinson J.R., Shefelbine S.J. "BoneJ: Free and extensible bone image analysis in
ImageJ." *Bone* 47(6):1076–1079, 2010. **DOI: 10.1016/j.bone.2010.08.023** (PMC3193171).

**Method [VERIFIED]:** Connectivity computes the **Euler characteristic (χ)** of the foreground from voxel
neighbourhoods (Odgaard & Gundersen 1993; Toriwaki & Yonekura 2002), then:
- Connectivity  **β₁ = 1 − Δχ**  (Δχ = the sample's contribution to χ of the whole connected structure)
- **Connectivity density  Conn.D = β₁ / stack volume** (interpretable as trabecular number per mm³).

**Validation approach [VERIFIED]:** "Algorithms in BoneJ were validated by running them on test data and
comparing computed results to expected results, including **synthetic images, images of real objects, and
mathematically defined clouds of points with known geometry**." Specifically, **"Connectivity was validated by
creating simple connected structures and measuring their Euler characteristics"** and was reported to compute χ
**"without fault"** and **"insensitive to hole position and feature size."**

**Metric fluorostats matches:** the **Euler characteristic / connectivity (β₁)**, i.e. the topological invariant.
This maps directly to fluorostats' χ and to its LCC / PHI-style connectivity measures.

**Number to reproduce [VERIFIED formulas; exact test-object numbers NOT published in the note]:**
Use analytic Euler numbers of simple solids — the same "known geometry" strategy BoneJ used:
- Solid ball / any simply-connected solid: **χ = 1**, β₁ = 0
- Solid torus (1 tunnel/handle): **χ = 0**, β₁ = 1
- Object with *k* tunnels: **χ = 1 − k**, β₁ = k
- *N* disjoint balls: **χ = N**
These are exact; require fluorostats to return them with **zero error**.

**Downloadable sample data:**
- BoneJ does not ship a single "connectivity gold dataset" file; validation used self-generated synthetic
  solids. Legacy site & guide: https://bonej.org/connectivity , https://bonej.org/legacy
- Source (Connectivity.java, algorithm reference): https://github.com/bonej-org/BoneJ2
- ImageJ page: https://imagej.net/plugins/bonej
- **[UNVERIFIED]** whether any downloadable trabecular sample stack is hosted with published χ values —
  the Bone note directs readers to bonej.org rather than a fixed dataset DOI.

---

## 3. MitoGraph — tubular network length & volume in budding yeast

**Citation [VERIFIED]:** Viana M.P., Lim S., Rafelski S.M. "Quantifying mitochondrial content in living cells."
*Methods in Cell Biology* (Biophysical Methods in Cell Biology) 125:77–93, 2015.
**DOI: 10.1016/bs.mcb.2014.10.003** [PARTIAL — DOI from secondary sources; PubMed 25640425].
Follow-up: Viana et al. 2020, *Cell Systems* (topology of budding-yeast mito networks).

**What MitoGraph measures [VERIFIED]:** skeleton **total length** and **volume** of tubular mitochondrial
networks; outputs a graph (`.gnet` edge list, `.coo` node coords), per-skeleton width, and a `.mitograph` file
with **"volume from voxels, average width, std width, total length and volume from length."**

**Metric fluorostats matches:** **total skeleton length** and **network volume** (and node/edge graph counts).

**Reported validation numbers [PARTIAL — from secondary literature, confirm against PDF before citing]:**
- **Reproducibility ~95.9% (range 87.5–99.8%)** for surface volume and skeleton volume (MitoGraph v2.0).
- **~96% of nodes and ~91% of edges** correct vs. manual annotation in budding-yeast cells.
- Later mammalian prospective (PMC6322684) reports **~7% total node-detection error** (3.6–7% from
  overlapping tubules) — this is a *later* paper, not the 2015 ground-truth phantom, so keep separate.
- **[UNVERIFIED]** the exact known-volume tube/sphere phantom % error from the 2015 paper itself
  (paywalled; the specific simulated-tube error figure was not confirmable this session).

**Downloadable sample data [VERIFIED]:**
- **Example dataset:** https://github.com/vianamp/MitoGraphTools ("an example dataset to test MitoGraph on").
- **Software repo:** https://github.com/vianamp/MitoGraph
- V2.0 page: https://rafelski.com/susanne/MitoGraph

**Number to reproduce:** run MitoGraph on the MitoGraphTools example set, capture its total-length/volume, and
require fluorostats to match. For a *guaranteed-exact* check, use analytic phantoms (below), where MitoGraph and
fluorostats can both be measured against a closed-form ground truth.

---

## 4. Synthetic phantoms with analytically known topology and skeleton length (zero-error ground truth)

These give **exact** ground truth — the same class of "known-geometry" test BoneJ and MitoGraph used — so
fluorostats can be validated with provably zero error, independent of any competitor's reported numbers.

### 4a. Topology phantoms (exact Euler characteristic χ)
Build binary voxel volumes and compare fluorostats' χ / connectivity to the closed form.

| Phantom | Construction | χ (Euler) | β₁ (loops) | # components |
|---|---|---|---|---|
| Solid ball | filled sphere radius R | **1** | 0 | 1 |
| Solid torus | swept disk around a circle (1 tunnel) | **0** | 1 | 1 |
| k-tunnel solid | block with k non-intersecting drilled tunnels | **1 − k** | k | 1 |
| N disjoint balls | N well-separated filled spheres | **N** | 0 | N |
| Genus-g surface solid | connect-sum of g tori | **1 − g** | g | 1 |

Rule of thumb (solids): **χ = (#components) − (#independent tunnels) + (#enclosed cavities)**.
Keep features ≥ a few voxels wide and separated to avoid discretization ambiguity.

### 4b. Skeleton-length phantoms (exact total length)
Rasterize analytic curves at fine voxel resolution; ground-truth length is the closed-form arc length.

| Phantom | Ground-truth length | Notes |
|---|---|---|
| Straight tube, axis-aligned, length L | **L** | tests calibration/units |
| Straight tube, diagonal (dx,dy,dz) | **√(dx²+dy²+dz²)** | tests anisotropic-voxel handling |
| Circular ring, radius R | **2πR** | one loop → also χ test (torus, χ=0) |
| Y-junction, three arms Lₐ,L_b,L_c | **Lₐ+L_b+L_c**, 1 junction, 3 endpoints | tests branch/junction counting vs AnalyzeSkeleton |
| Grid/lattice of straight segments | **Σ segment lengths** | tests junction enumeration at scale |

### 4c. Combined phantom (matches all three tools at once)
A **thickened torus** (tube of radius r swept on a circle of radius R):
- Topology: **χ = 0, β₁ = 1** → matches BoneJ connectivity.
- Skeleton: single closed loop, **length = 2πR** → matches AnalyzeSkeleton branch length & MitoGraph total length.
- Volume: **≈ 2π²Rr²** (analytic torus volume) → matches MitoGraph volume-from-voxels.

Because the Lee-1994 skeleton is deterministic, fluorostats vs. Fiji AnalyzeSkeleton should agree exactly on
the *same binarized phantom*; against BoneJ/MitoGraph the analytic χ, length, and volume are the zero-error targets.

---

## Sources (verified this session)
- AnalyzeSkeleton: https://imagej.net/plugins/analyze-skeleton/
- Skeletonize3D (Lee-1994): https://imagej.net/plugins/skeletonize3d
- BoneJ paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC3193171/
- BoneJ connectivity: https://bonej.org/connectivity ; source: https://github.com/bonej-org/BoneJ2
- MitoGraph repo: https://github.com/vianamp/MitoGraph ; data: https://github.com/vianamp/MitoGraphTools
- MitoGraph prospective (node-error figures): https://pmc.ncbi.nlm.nih.gov/articles/PMC6322684/
- MitoGraph PubMed: https://pubmed.ncbi.nlm.nih.gov/25640425/
