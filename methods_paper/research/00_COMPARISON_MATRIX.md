# fluorostats benchmark comparison matrix

Actionable companion to `00_SYNTHESIS.md`. This is the **tooling logistics
layer**: every comparator we'd benchmark against, how to obtain it, the exact
overlapping metric, the public dataset, and which benchmark (B1–B6) it feeds.
Built to drive the download/setup phase.

Legend — **Access:** `pip` = Python package · `Fiji` = ImageJ/Fiji update site or
plugin · `standalone` = separate binary/repo · `commercial` = paid/closed ·
`builtin` = already a fluorostats dependency. **Setup effort:** Low / Med / High.

---

## 1. Segmentation & signal capture → **B2**

| Comparator | Access | Install path | Overlapping metric vs fluorostats | Setup |
|---|---|---|---|---|
| Otsu / Li thresholding | builtin | scikit-image (already used) | volume fraction, foreground mask | Low |
| Otsu-in-Fiji (3D) | Fiji | Fiji default + 3D Suite | foreground Dice/IoU, volume fraction | Low |
| **Cellpose 3** | pip | `pip install cellpose`; models auto-download; GPU optional | per-instance masks, foreground Dice | Med (GPU better) |
| **StarDist-3D** | pip | `pip install stardist csbdeep`; pretrained 3D model | per-instance masks, nuclei counts | Med |
| U-Net (ref lineage) | pip | not benchmarked directly (backbone) | — | — |

**Public ground truth:** Cell Tracking Challenge 3D fluorescence
(Fluo-C3DL-MDA231, Fluo-C3DH-A549, Fluo-N3DL-TRIC) — download from
celltrackingchallenge.net; DSB2018 / BBBC038 (broadinstitute BBBC).
**Reports:** voxel Dice/IoU, volume-fraction Bland-Altman vs truth, per-instance
SEG (honest lag on dense), CPU-vs-GPU runtime.

---

## 2. Vascular / endothelial networks → **B4**

| Comparator | Access | Install path | Overlapping metric | Setup |
|---|---|---|---|---|
| **AngioTool** | standalone | NIH/CCR download (Windows/Java binary) | junctions, branch count, total length, lacunarity | Low |
| **Angiogenesis Analyzer** | Fiji | ImageJ macro toolset (Gilles Carpentier) | nodes, junctions, segments, total length | Low |
| REAVER | Fiji/MATLAB | GitHub (Bagley/Peirce lab) | length, diameter, branch density | Med |
| VesSAP / VesselExpress | pip/standalone | GitHub (3D, DL) — optional, organ-scale | 3D skeleton graph | High |

**Data:** our own 2D network images + Z-projections; tube-formation assay images
if available. **Reports:** ICC + Bland-Altman on junctions/branches/length (2D
parity), then 3D recovery of Z-projection-occluded junctions as a systematic
offset.

---

## 3. Skeleton morphometry → **B1** (validity anchor)

| Comparator | Access | Install path | Overlapping metric | Setup |
|---|---|---|---|---|
| **Fiji AnalyzeSkeleton + Skeletonize3D** | Fiji | Fiji builtin (Arganda-Carreras) | branch/junction/endpoint count, total length | Low |
| skan | builtin | already a fluorostats dependency | branch graph statistics | — |

**Key fact:** fluorostats and AnalyzeSkeleton both use **Lee-1994** thinning →
expect **exact integer agreement** on counts, <1–2% length deviation. Add
synthetic phantoms with known ground-truth counts. **Reports:** y=x scatter,
Bland-Altman, CCC.

---

## 4. Topology / connectivity → **B1**

| Comparator | Access | Install path | Overlapping metric | Setup |
|---|---|---|---|---|
| **BoneJ (Connectivity)** | Fiji | BoneJ update site | Euler characteristic, connectivity density | Low |
| scikit-image `euler_number` | builtin | already used | Euler number | — |
| MitoGraph (PHI) | standalone | GitHub (Rafelski lab) — optional | largest-connected-component fraction | Med |

**Data:** synthetic phantoms with analytically known χ (ball χ=1; k tunnels
χ=1−k; N balls χ=N; ball+cavity χ=2) + real segmented stacks. **Reports:**
zero-error pass on phantoms; pin 6- vs 26-connectivity; LCC-fraction vs MitoGraph
PHI tolerance.

---

## 5. Viability (Live/Dead) → **B3** (headline)

| Comparator | Access | Install path | Overlapping metric | Setup |
|---|---|---|---|---|
| Mid-plane 2D area ratio | builtin | fluorostats itself (single-plane mode) | live area fraction | Low |
| MIP area ratio | builtin | fluorostats itself (project-then-threshold) | live area fraction | Low |
| Manual ImageJ counting | Fiji | Cell Counter plugin | live/dead counts | Med (manual) |
| AutoCount (Sharara 2025) | standalone | check repo availability | Live/Dead counts | Med |

**Data:** our Live/Dead confocal stacks spanning the death gradient (already have
GelMA vs Hybrid × depth × day). **Reports:** 3D live-fraction vs 2D/MIP ratio;
overestimate-grows-with-thickness curve; per-z live-fraction profile; per-z
attenuation-normalization control. **No download needed — runnable now.**

---

## 6. Nuclei counting → **B2**

| Comparator | Access | Install path | Overlapping metric | Setup |
|---|---|---|---|---|
| **StarDist-3D** | pip | `pip install stardist` | nuclei count, centroid | Med |
| **Cellpose** | pip | `pip install cellpose` (nuclei model) | nuclei count | Med |
| Fiji 3D Objects Counter | Fiji | Fiji builtin (Bolte-Cordelieres) | CC-labeled object count | Low |
| Expert manual count | — | human annotator | ground-truth count | High (manual) |

**Data:** our DAPI immuno stacks + DSB2018/BBBC038. **Reports:** Bland-Altman
bias, ICC, %error **stratified by local density** (sparse/medium/crowded) —
expected honest negative bias for fluorostats as density rises.

---

## 7. Spatial homogeneity → **B6**

| Comparator | Access | Install path | Overlapping metric | Setup |
|---|---|---|---|---|
| Ripley's K | pip | `astropy`/`pointpats` | clustering deviation | Low |
| Nearest-neighbor index | pip | `scipy.spatial` | dispersion | Low |
| Lacunarity (FracLac) | Fiji | Fiji plugin | multiscale gap structure | Med |

**Data:** synthetic clustering sweep (jittered lattice → Poisson CSR →
Thomas/Matérn) rendered as fluorescence — **we generate this**, no download.
**Reports:** Gini/CV vs clustering parameter (monotonic), Spearman vs Ripley's K
& NN index, AUC uniform-vs-clustered, tile-size sensitivity (4×4→32×32).

---

## 8. Volume fraction / coverage → **B5**

| Comparator | Access | Install path | Overlapping metric | Setup |
|---|---|---|---|---|
| Cavalieri/Delesse point counting | Fiji | Fiji Stereology / Grid plugin | volume/area fraction | Low (blinded manual) |
| µCT BV/TV convention | — | conceptual reference | volume fraction | — |

**Data:** same stacks imaged at **two digital zooms** (need to acquire or find
existing dual-zoom pairs) + blinded point counting on shared stacks. **Reports:**
voxel VF vs point-count CCC within stereological CE; raw counts diverge across
zoom while per-mm³ densities stay stable (reproduces Riley 2023).

---

## 9. Statistics & power → validation-only (no external comparator)

| Item | Access | Purpose |
|---|---|---|
| R `coin`/`effectsize`/`rcompanion` | standalone (R) | cross-check Mann-Whitney/Cliff's δ/SRH numeric agreement | 
| Bootstrap power calibration | builtin | simulate known effect sizes, confirm predicted vs empirical rejection rate |

No head-to-head; frame as correctness cross-check + calibration figure.

---

## 10. Visualization → optional equivalence

| Comparator | Access | Note |
|---|---|---|
| Imaris "Surfaces" | commercial | render same stack at matched iso/camera — equivalence only if license available |
| ClearVolume / napari | pip | interactive rendering (out of scope for fluorostats) |

Low priority; only if an Imaris license is on hand.

---

## Setup priority ladder (what to download first)

**Tier 0 — runnable now, no downloads (start here):**
- **B3 viability** (2D-vs-3D) — pure fluorostats on existing Live/Dead stacks.
- **B6 homogeneity** — synthetic controls we generate + pip `pointpats`/`scipy`.
- **B1 topology phantoms** — synthetic χ phantoms + builtin scikit-image.

**Tier 1 — Fiji + one download (highest validity payoff):**
- Install **Fiji** once → unlocks AnalyzeSkeleton (B1), BoneJ (B1), Angiogenesis
  Analyzer (B4), 3D Objects Counter (B2 nuclei), Stereology (B5).
- Download **AngioTool** binary (B4).

**Tier 2 — Python DL stack:**
- `pip install cellpose stardist csbdeep` (B2 segmentation, B2 nuclei). GPU
  optional but faster.

**Tier 3 — public datasets:**
- Cell Tracking Challenge 3D fluorescence (B2), DSB2018/BBBC038 (B2 nuclei).

**Tier 4 — optional / conditional:**
- MitoGraph (B1 LCC cross-check), REAVER (B4), Imaris (viz), dual-zoom
  acquisition for B5.

---

## Immediate next actions (proposed)

1. **Tier 0 now:** implement B3 (viability 2D-vs-3D) and B1-phantom (topology
   correctness) as fluorostats benchmark scripts — zero external dependencies,
   both produce headline-quality figures.
2. **Install Fiji** (Tier 1) — single biggest unlock; enables 5 of 6 benchmarks'
   reference tools.
3. **pip DL stack** (Tier 2) for the segmentation agreement study.
4. **Fetch public datasets** (Tier 3) for ground-truth-anchored B2.

Each benchmark writes results as CSVs + figures into `methods_paper/benchmarks/`,
reusing fluorostats' own stats layer (Bland-Altman, ICC/CCC, Spearman) for the
agreement analysis — dogfooding the tool on its own validation.
