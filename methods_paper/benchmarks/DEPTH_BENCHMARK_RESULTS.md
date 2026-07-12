# Depth-penetration module — benchmark results

Validation of the v0.8.0 depth-penetration capability (`src/fluorostats/depth.py`
pure per-stack functions + `src/fluorostats/depth_batch.py` manifest driver).

**Why not a "5-tool timed race":** the tools people actually use for depth-intensity
profiling (Fiji Plot Z-axis Profile, Imaris, ZEISS ZEN, Nikon NIS-Elements,
CellProfiler, MATLAB) are GUI/commercial and not scriptable head-to-head; there is
**no public ground-truth dataset** for confocal depth penetration (unlike BBBC039 for
nuclei); and the computation is sub-second numpy, so timing is not a differentiator.
So correctness is shown on **constructed Beer–Lambert ground truth** and the field
comparison is a **faithful-reimplementation parity check + a verified capability
matrix** — mirroring the paper's existing correctness-phantom + competitor-dossier
methodology.

Reproduce: `python3.13 methods_paper/benchmarks/b_depth_penetration.py`
Outputs: `results/b_depth_penetration_correctness.csv`,
`results/b_depth_penetration_parity_timing.csv`,
figure `figures/main/b_depth_penetration.{pdf,png}`.
Unit tests: `tests/test_depth.py` — 13/13 pass.

## A. Correctness on synthetic Beer–Lambert ground truth
Stacks with `I(z) = I0·e^(−z/λ) + bg` have known closed-form AUC
(`I0·λ·(1−e^(−Z/λ))`; normalised `λ·(1−e^(−Z/λ))`).

| Test | Result |
|---|---|
| **A1 exactness** (noiseless, λ=20–160) | normalised-AUC error **0.001–0.083 %** vs closed form (pure trapezoid discretisation) |
| **A2 noise robustness** (SNR≈10, 8 reps, matched blank) | recovered norm-AUC **39.84 ± 0.03** vs exact-pipeline GT 39.83 → **0.07 %** |
| **A3 discrimination** (short λ=30 vs long λ=70) | retained fraction **0.309 / 0.548** = GT 0.309 / 0.548 exactly; ordering correct; built-in Welch **p = 4.2 × 10⁻²⁴** |

## B. Faithful reimplementation parity
| Check | Result |
|---|---|
| Fiji **Plot Z-axis Profile** = per-slice spatial mean | **bit-exact** (max|Δ| = 0.0, float64) |
| `auc_depth` vs `scipy.integrate.trapezoid` | identical (|Δ| = 0.0) |
| Trapezoidal AUC vs analytic | **0.016 %** error |
| Naive rectangular sum (Excel/Prism manual) vs analytic | **2.807 %** error → fluorostats **~175× more accurate** |

fluorostats reproduces the standard profile the field uses, and folds in the
blank-subtraction / surface-normalisation / AUC / group-stats those tools leave to
manual Excel/Prism.

## C. Competitor capability matrix
See `methods_paper/data/competitor_depth_penetration.md` — 6 tools
(Fiji/ImageJ, Imaris, ZEISS ZEN, Nikon NIS-Elements, CellProfiler, MATLAB) scored on
per-slice profile / blank subtraction / surface normalisation / AUC-over-window /
group mean±SEM / significance test / tidy CSV / batch-manifest / scriptable-
deterministic / manual-step count. Summary: every tool can *plot* a z-profile; only
fluorostats runs the whole penetration pipeline reproducibly from one config with no
manual export.

## D. Determinism + timing
- **Deterministic:** two identical runs → bit-identical outputs (spread 0).
- **Timing** (full profile→AUC pipeline): 30×512² **11.5 ms**, 60×512² **23.2 ms**,
  60×1024² **117 ms** — sub-second; documents that speed is not the differentiator.

## Honest scope
This is a *reproducible pipeline over standard primitives*, not a novel algorithm.
The differentiator is correctness of the full chain + one-config reproducibility +
integrated stats, not the accuracy of any single step. Whole-image mean assumes a
roughly uniform field (the user selects the ROI/crop upstream).
