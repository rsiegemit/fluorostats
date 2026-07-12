# For Claude Cowork — NEW v0.8.0 capability: depth-penetration profiling

Scope: this handoff is **only** about the new depth-penetration module and its
benchmark. The figures/prose for the existing capabilities are covered in
`COWORK_NEXT.md`. Your job here is the **write-up** (a Methods subsection + a short
Results paragraph + a Limitations line + a display-item decision). Everything is
already built, benchmarked, committed on `main`, and traceable to files — pull every
number from the files below, not from memory.

## 1. What it is (for the Methods subsection)

New in v0.8.0: **depth-resolved fluorescent-probe penetration / permeability
profiling** from confocal z-stacks.

- `src/fluorostats/depth.py` — **pure per-stack functions** (arrays in, dataclasses
  out; trivially unit-testable):
  - `intensity_depth_profile` — collapse each z-slice to a per-slice **spatial mean
    (or median)** intensity vs physical depth.
  - `subtract_background` — subtract a matched **blank / no-fluo control** (scalar or
    a depth-resolved blank profile, resampled depth-for-depth; negatives clipped to 0).
  - `normalize_to_surface` — divide by the **near-surface reference** (mean of the
    first `n_surface` slices, default 3) so decay *shape* compares across acquisitions
    with different gain/laser.
  - `auc_depth` — **trapezoidal area-under-curve** over one or more physical depth
    windows, with endpoints linearly interpolated so the window is exactly `[z0, z1]`
    regardless of slice spacing.
- `src/fluorostats/depth_batch.py` — **manifest-driven batch driver**. One JSON
  (groups, stacks, channel, reducer, background blanks, `n_surface`, `auc_windows_um`)
  → tidy CSVs (`depth_profiles_long.csv`, `auc_per_stack.csv`, `group_depth_summary.csv`)
  + figures + group **mean ± SEM** curves + a built-in **Welch t-test** for a
  two-condition contrast. CLI: `fluorostats depth <manifest.json>`. `role: "aux"`
  stacks are drawn faintly and excluded from the group stats.
- **Purpose:** quantify how far a probe penetrates a material — e.g. FITC-dextran into
  GelMA vs hybrid hydrogels (example manifest: `methods_paper/tools/permeability_fd2000.json`).

## 2. What's validated (for the Results paragraph — numbers are final)

Benchmark script: `benchmarks/b_depth_penetration.py`. Full write-up with the table:
**`benchmarks/DEPTH_BENCHMARK_RESULTS.md`**. Raw numbers:
`benchmarks/results/b_depth_penetration_correctness.csv` and
`…_parity_timing.csv`. Unit tests: `tests/test_depth.py` — **13/13 pass**.

- **A. Correctness on synthetic Beer–Lambert ground truth** (`I(z)=I0·e^(−z/λ)+bg`,
  known closed-form AUC): recovers analytic AUC to **0.001–0.083 %** on noiseless
  stacks (exact), **0.07 %** on noisy stacks (surface SNR ≈ 10); a two-condition
  contrast (short λ=30 vs long λ=70) recovers the true retained fraction **exactly**
  (0.309 / 0.548) with the built-in Welch test **p = 4.2 × 10⁻²⁴**.
- **B. Faithful reimplementation parity:** reproduces **Fiji "Plot Z-axis Profile"**
  (per-slice mean) **bit-for-bit**; `auc_depth` matches `scipy.integrate.trapezoid`
  exactly; its trapezoidal AUC is **0.016 %** off analytic vs **2.8 %** for the naive
  rectangular sum an Excel/Prism workflow uses (**~175× more accurate**).
- **D. Determinism + timing:** bit-identical across runs; 11–117 ms for 30×512²–60×1024²
  stacks (**speed is not the differentiator** — say so; the point is reproducibility).

## 3. Competitor framing (for Results / Discussion)

Verified dossier: **`data/competitor_depth_penetration.md`** — 6 tools (Fiji/ImageJ,
Imaris, ZEISS ZEN, Nikon NIS-Elements, CellProfiler, MATLAB) + a 10-column capability
matrix, primary-source-cited with unverified items flagged.

**The defensible claim (do not overstate):** every tool can *plot* a z-profile, and
several *can* script/batch (ImageJ macros, CellProfiler headless, ZEN OAD, NIS
GA3/JOBS, MATLAB) — so "only we can script it" is **false and not claimed**. fluorostats
is **not** "more accurate" per step; it is **bit-identical** on the shared per-slice
mean. What no GUI/commercial tool provides is the **whole penetration pipeline as one
config-driven, reproducible unit** (blank-subtract → surface-normalize → multi-window
AUC → group mean±SEM → significance → tidy CSV), turning a ~8–12-step Fiji+Excel
workflow (or a one-off MATLAB script) into one manifest.

## 4. Honesty points to state plainly (Limitations)

From §4 of the dossier — keep these in the text:
- **Standard pipeline, not a novel algorithm.** Contribution = reproducibility +
  integration + a tested reference implementation, not a better number on any step.
- **Correctness rests on synthetic ground truth** because **no public benchmark
  dataset exists** for confocal depth penetration (a genuine evidentiary limitation
  vs the nuclei/BBBC039 comparison — state it, don't hide it).
- **Whole-image mean assumes a roughly uniform field**; heterogeneous fields need the
  **user to pick the ROI upstream** (fluorostats profiles what it's given).
- **Small-n stats:** the Welch test is labelled "underpowered, descriptive only" for
  the typical handful of stacks/condition.
- **AUC, not a decay fit:** it integrates the observed curve (no distributional
  assumption); it does **not** fit a penetration constant λ.

## 5. Display-item decision (for the writer)

The validation figure is ready: `benchmarks/figures/main/b_depth_penetration.{pdf,png,txt}`
(panel a = ground-truth recovery, panel b = two-condition discrimination; caption in
the `.txt`). It is authored at the same 130.7 mm text width as the other figures.

**Suggestion:** this is a *capability demonstration*, not a headline result, so it fits
best as an **Extended Data / Supplementary figure** with a one-paragraph Results
mention and a Methods subsection — not a main figure. Your call on final numbering.

## 6. Open decisions / what Claude Code can still do

- **Real data:** the benchmark is on synthetic GT. If the real `permeability_fd2000`
  `.oib` stacks are available locally, Claude Code can run the actual pipeline for a
  real-data figure alongside the synthetic validation — say the word.
- **Wire-in:** if you want the depth figure in the LaTeX, Claude Code can add it to
  the (untracked) manuscript zip and rebuild, same as the other figures.

State/library: v0.8.0 on GitHub `main` (pushed); `origin/main == local`. Resume pointer:
`PROJECT_STATE.md`.
