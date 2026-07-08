# Change log & rerun-impact tracker

Tracks every change made during the methods-paper effort and whether it forces
a rerun of the **GelMA vs Hybrid Live/Dead analysis** (the `for_email/` v3
deliverable) or anything else.

**Rule of thumb:** only changes to a fluorostats **library metric function**
that the Live/Dead analysis actually calls can invalidate those numbers.
Visualization, benchmark scripts, and research docs cannot.

Live/Dead analysis depends on these library functions:
`metrics_3d.volume_fraction`, `metrics_3d.connectivity_metrics`
(n_components, euler, largest_component_fraction),
`metrics_3d.skeleton_metrics` (total_length, n_branches, n_junctions),
`metrics_3d.normalise_skeleton_metrics` (length/junction density),
`morphometry.lateral_homogeneity` / `depth_*`, `stats.*`.

---

## Changes so far

| # | Change | Where | Touches a Live/Dead metric? | Rerun needed? |
|---|---|---|---|---|
| 1 | v0.3 `style.py` (publication matplotlib defaults) | library | No — plotting only | **No** |
| 2 | v0.3 `render3d` (dark isosurface, smoothing, live_dead_mip, mip_grid, layer_split, depth_coded) | library | No — rendering only | **No** (figures already regenerated) |
| 3 | Live/Dead figures restyled + PPTX-style 3D | `for_email/plots/` | No — same underlying CSVs | **No** — numbers unchanged |
| 4 | Research dossiers (13) + synthesis + data master | `methods_paper/research`, `/data` | No | **No** |
| 5 | Benchmark harness `agreement.py` (Bland-Altman/CCC/ICC) | `methods_paper/benchmarks` | No — new, isolated | **No** |
| 6 | B1 topology + skeleton phantom benchmarks | benchmarks | No — read-only use of library | **No** |
| 7 | B6 homogeneity synthetic benchmark | benchmarks | No | **No** |
| 8 | B4 REAVER benchmark + **skeleton spur-pruning + node-based branchpoint counter** | benchmarks ONLY | **No** — pruning lives in the benchmark script, NOT the library | **No** |

**Net so far: zero reruns required.** No library metric function has been
modified. The Live/Dead v3 results (`for_email/data/*_v3.csv`) remain valid.

---

## Important clarification about the branchpoint finding

The REAVER benchmark's branchpoint over-count was in **benchmark code**
(`b4_reaver_vascular.py::count_branchpoints`, a raw skeleton neighbour-count),
**not** in fluorostats' library. The library's `skeleton_metrics` reports
`n_junctions` as junction-to-junction branches (skan `branch_type==2`) — a
different, more conservative quantity that was used in the Live/Dead analysis
and is untouched.

So the Live/Dead `junction_density_per_mm3` numbers are NOT affected by anything
discovered here.

---

## Forward-looking decisions that WOULD trigger a rerun

These are NOT yet done — flagged so we decide deliberately:

| Potential change | If we do it | Rerun impact |
|---|---|---|
| **A. Add skeleton spur-pruning to the library** (`skeleton_metrics(prune=True)`) | Pruned skeletons slightly reduce `total_length` and change branch/junction counts | **Rerun Live/Dead skeleton metrics** (length density, junction density, mean branch length, n_branches). VF, largest-component, homogeneity, depth: unaffected. Direction (Hybrid > GelMA) very likely unchanged; absolute values shift. |
| **B. Add a field-standard node-based branchpoint metric** (`n_junction_nodes`) to align with AngioTool/REAVER/AnalyzeSkeleton | New metric; if reported in the paper's Live/Dead comparison | **Rerun** to add the new column. Existing `n_junctions` (JtoJ) can stay as-is; this is additive. |
| **C. Change segmentation defaults** (threshold_scale/bg_radius) based on benchmark tuning | Alters the mask for every stack | **Full rerun** of ALL Live/Dead metrics. Only do this if a benchmark proves a default is wrong; keep current defaults for the published v3 unless so. |
| **D. Promote `agreement.py` (Bland-Altman/CCC/ICC) into `fluorostats.stats`** | Pure addition | **No rerun** — new functions, nothing recomputed. |

**Recommendation:** decisions A and B are worth making (they make fluorostats
match the field-standard skeleton conventions and would strengthen the paper),
but do them as an explicit, batched library update — then rerun ONLY the
Live/Dead skeleton metrics once, regenerate the affected figures, and note it
here. Do NOT change segmentation defaults (C) without a benchmark mandate.

---

## DECISION (2026-07-08): defer the Live/Dead rerun until the end

Per the user: **do not rerun the Live/Dead analysis piecemeal.** Keep running
baselines and benchmarks; accumulate all warranted library changes (pruning,
node-based branchpoints, any others the benchmarks justify) as a single batched
upgrade. **Rerun the Live/Dead analysis exactly once, at the end, after every
library change is finalized.** Log each accumulated library change below as it
lands, with its expected Live/Dead impact, so the final rerun is a known,
auditable set.

### LANDED in v0.5.0 (new capabilities — additive/opt-in, no rerun triggered)

| Change | Module | Rerun impact |
|---|---|---|
| `viability` module (live/dead fractions, depth profile, 2D-vs-3D, attenuation correct) | viability | None now; **candidate for the final Live/Dead rerun** to add depth-resolved viability + 2D-vs-3D columns |
| `validate` module (instance_f1, match_instances, average_precision) | validate | None — used for benchmarking, not the Live/Dead metrics |
| `objects.watershed_split` (split touching objects) | objects | None until used; candidate for the rerun IF we want crowded-region splitting (only matters for dense stacks) |
| `objects.clear_border_labels` (drop edge objects) | objects | None until used; candidate for the rerun IF we adopt border-excluded counts |
| +11 tests; suite 98 green | tests | — |

**Benchmark validation backing v0.5:** fluorostats instance F1 on BBBC039 = 0.896
(ahead of validated StarDist 0.871 / Cellpose 0.862 on well-separated nuclei).
Viability module demonstrated on public S-BIAD2130 (MIP overestimates 3D live
coverage 1.14× on that well-imaged stack; attenuation_correct flattens the
depth trend). Added to the deferred-rerun candidate list — decide at final rerun
whether to report depth-resolved viability / border-excluded counts for the
GelMA vs Hybrid data.

---

### LANDED in v0.4.0 (safe additions — existing 3D outputs unchanged, no rerun triggered)

| Change | Module | Rerun impact |
|---|---|---|
| New `skeleton.py` module (extracted from metrics_3d, re-exported for compat) | skeleton | None — pure refactor, same objects |
| `skeleton_metrics(prune=False default)` + `prune_skeleton()` | skeleton | None until `prune=True` is used in Live/Dead |
| `n_junction_nodes()` + additive `n_junction_nodes` key | skeleton | None — new key; existing keys byte-identical |
| New `agreement.py` module (bland_altman, lins_ccc, icc, agreement_report) | agreement | None — pure addition |
| **Bugfix: `skeleton_metrics` now matches `spacing` to mask ndim** (worked only on 3D before) | skeleton | **None for Live/Dead** — 3D path identical (`[-3:]` of a 3-tuple is unchanged); only enables 2D use |
| +12 tests (test_skeleton, test_agreement); suite 87 green | tests | — |

Verified: 3D `skeleton_metrics(prune=False)` returns identical legacy keys
(total_length_um, n_branches, n_junctions, mean_branch_length_um) — confirmed
against a fixed 3D mask. **Live/Dead v3 results remain valid; rerun still deferred.**

Still deferred to the batched final rerun: making `prune=True` the default,
reporting `n_junction_nodes` in the Live/Dead pipeline, and any segmentation
default change from the B2 debris finding.

---

### Original discoveries from the benchmarks (B4 REAVER, B2 BBBC039) that justified the above:

**Safe additions — implemented as opt-in, do NOT change existing outputs → no rerun by themselves:**
| Addition | Module | Why (benchmark evidence) | Rerun trigger |
|---|---|---|---|
| `prune_skeleton()` + `skeleton_metrics(prune=False default)` | metrics_3d/skeleton | B4: raw skeleton over-counts branchpoints ~3.6× on real vascular masks; pruning cut MAE 217→61 | Only if `prune=True` becomes default |
| `n_junction_nodes()` (degree≥3 node count) | metrics_3d/skeleton | fluorostats `n_junctions` (skan JtoJ, type 2) ≠ field-standard branchpoint node count used by AngioTool/REAVER/AnalyzeSkeleton | Only if reported in Live/Dead |
| `bland_altman`, `lins_ccc`, `icc`, `agreement_report` | stats | needed for every method-comparison benchmark; broadly useful | none (pure addition) |

**Behavior-changing — deferred to the batched final rerun:**
| Change | Live/Dead impact |
|---|---|
| Make skeleton `prune=True` default | rerun length/junction/branch metrics (VF, largest-comp, homogeneity, depth unaffected) |
| Report `n_junction_nodes` in Live/Dead | additive column; rerun to populate |
| B2 sparse over-count fix (debris filter / adaptive min_size) | investigate first; if defaults change → full rerun |

**Benchmark evidence backing these (real public data):**
- B4 REAVER (n=36, manual GT): area fraction CCC 0.70 / Spearman 0.94; branchpoint MAE 61 (needs field-standard pruning + node count).
- B2 BBBC039 (n=200, GT): count CCC 0.92 / Spearman 0.96; crowded undercount 6.3% (expected CC-merge); sparse over-count outliers (debris).
- B1 topology/skeleton phantoms: exact (validity anchor, no change needed).
- B6 homogeneity: ρ=−0.985 vs Clark-Evans (no change needed).

---

## Rerun procedure (when triggered)

1. Bump fluorostats version; run full test suite.
2. Re-run `Extrusion_Data_Results/update_v3.py` + `refresh_all_v3.py` +
   `refresh_styled.py` (these regenerate per-file CSVs, stats, and figures).
3. Diff new vs old `*_v3.csv`; confirm direction/significance unchanged.
4. Update `for_email/` + rebuild `extrusion_analysis.zip`.
5. Log the rerun here with before/after headline numbers.
