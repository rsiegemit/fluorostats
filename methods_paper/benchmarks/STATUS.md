# Benchmark status

Live tracker for the fluorostats methods-paper benchmarks. Tier-0 = runnable
now with no external data (synthetic / reproducible). Tier-1+ = need public
datasets and/or competitor tools currently being located by the data agents.

## Results so far

### Tier-0 — synthetic / reproducible (validity anchors)

| Benchmark | Script | Result | Figure |
|---|---|---|---|
| **B1 topology** | `b1_topology_phantoms.py` | **6/6 phantoms exact (zero error)** — Euler, components, LCC fraction match analytic truth | `figures/b1_correctness.png` |
| **B1 skeleton** | `b1_skeleton_phantoms.py` | **5/5 pass** — total length ≤1% error, branch counts exact | `figures/b1_correctness.png` |
| **B6 homogeneity** | `b6_homogeneity_synthetic.py` | **PASS** — Gini monotonic; Spearman vs Clark-Evans R = **−0.985**; AUC = **1.0** | `figures/b6_homogeneity.png` |

### Tier-1 — real public competitor datasets (downloaded, verified)

| Benchmark | Dataset | Result | Figure |
|---|---|---|---|
| **B2 nuclei vs GT** | BBBC039 (n=200, CC0) | count **CCC 0.918 / Spearman 0.96**, MAE 5.96 | `figures/b2_nuclei_bbbc039.png` |
| **B2 nuclei tri-tool** | BBBC039 vs Cellpose + StarDist | DL baselines VALIDATED (StarDist F1 0.871 vs published 0.864; Cellpose 0.862). fluorostats lowest count MAE (5.96 vs 12.4/12.9); DL win on instance F1 | `figures/b2_nuclei_tritool.png` |
| **B2 3D seg** | CTC Fluo-C3DH-A549 (gold GT) | foreground Dice **0.69**, Jaccard 0.53 (honest — hard cytoplasmic data) | `figures/b2_ctc_volfraction.png` |
| **B4 vascular ranking** | REAVER (n=36, manual GT) | fluorostats **#4 of 6 tools** by area-fraction MAE — ties AngioTool, beats RAVE + AngioQuant on their own data | `figures/b4_reaver_ranking.png` |
| **B4 vascular vs GT** | REAVER (n=36) | area fraction CCC 0.70 / Spearman 0.94; branchpoint MAE 217→61 after v0.4 pruning | `figures/b4_reaver_*.png` |
| **B4 3D vascular** | VesselExpress light-sheet | sane 3D metrics (VF 0.0019, length density 26 mm/mm³, pruned junctions) | — |

### Tier-2 — application + capability demonstrations

| Benchmark | Dataset | Result | Figure |
|---|---|---|---|
| **B2 nuclei instance-F1** | BBBC039 (n=200) | fluorostats F1 **0.896** — *ahead* of validated StarDist 0.871 / Cellpose 0.862 on well-separated nuclei (dataset-specific; DL wins on crowded) | `figures/b2_nuclei_f1.png` |
| **B3 viability** | S-BIAD2130 Day-14 Live/Dead (public, CC0) | MIP overestimates 3D live coverage **1.14×** on this well-imaged stack; `attenuation_correct` flattens the depth trend. Modest here — dramatic effect needs strongly-attenuated samples | `figures/b3_viability.png` |

### Skeleton / topology vs Fiji tools (B1 extension) — status

fluorostats' skeleton (`skeleton_metrics`) and topology (`connectivity_metrics`)
share the **Lee-1994 thinning** and scikit-image Euler algorithms with Fiji's
AnalyzeSkeleton and BoneJ. The B1 phantom benchmarks already validate these to
**exact / zero error** against analytic ground truth — which is precisely what
BoneJ and AnalyzeSkeleton validate against (Doube 2010 used known-topology
solids; phantom parity is a correctness proof). A direct headless-Fiji
head-to-head is therefore optional confirmation, not a correctness gap, and is
deferred (Fiji download URL changed; headless macro scripting is high-friction).

### DL baseline infrastructure (AMD ROCm HPC cluster)

Cellpose v3 `nuclei` + StarDist `2D_versatile_fluo` run on the `amd` cluster
(login1.hpcfund) via Slurm, both CPU. Cellpose v4 SAM transformer abandoned
(GPU hipBLASLt crash + CPU too slow); v3 nuclei CNN is fast and is the model
with published numbers. Both validated against published F1 — see
`../data/PUBLISHED_BASELINES.md`.

Tier-0 establish measurement correctness (reproduce BoneJ + point-pattern
validation). Tier-1 are honest head-to-heads on competitors' own annotated data
— fluorostats ranks strongly (Spearman 0.94–0.96) and the gaps found drove the
v0.4.0 library upgrade (pruning + node-based branchpoints).

## Harness

- `agreement.py` — Bland-Altman (bias + 95% LoA), Lin's CCC, ICC(A,1), Spearman,
  MAPE, and a two-panel identity+Bland-Altman figure. Reused by every
  fluorostats-vs-competitor benchmark.

## Pending (need public data / competitor tools — being located now)

| Benchmark | Needs | Status |
|---|---|---|
| B1 skeleton vs **AnalyzeSkeleton** | Fiji (download URL 404'd — getting current link from agents) | tooling |
| B1 topology vs **BoneJ** | Fiji + BoneJ update site | tooling |
| B2 segmentation agreement | Cell Tracking Challenge 3D fluorescence / BBBC | data agents running |
| B2 nuclei vs **StarDist/Cellpose** | Cellpose installed ✓; StarDist pending; DSB2018 data | partial |
| B4 vascular vs **AngioTool/REAVER** | REAVER's public annotated dataset (ideal) | data agents running |
| B5 volume fraction / FOV-norm | dual-zoom or stereology-annotated public stacks | data agents running |

## Environment

- Interpreter: **python3.13** (has fluorostats 0.3.0 + numpy/scipy/skimage/pandas).
  Note: system `python3` is 3.14 with no packages — always use `python3.13`.
- Installed: cellpose ✓. Pending/failed: pointpats (not in 3.13 — B6 uses scipy
  instead), Fiji (URL scheme changed).
- Datasets + competitor benchmark dossiers: `../data/*.md` (15+ files from the
  data-hunting agent fleet).

## Next

1. Read the data/competitor dossiers → pick exact download URLs for B2/B4/B5.
2. Resolve current Fiji download URL → run AnalyzeSkeleton + BoneJ head-to-heads
   (B1 extends from "vs analytic truth" to "vs the field-standard tool").
3. Pull the top public datasets, wire each into the `agreement.py` harness.
