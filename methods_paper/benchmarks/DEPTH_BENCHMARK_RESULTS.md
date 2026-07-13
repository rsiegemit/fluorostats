# Depth-penetration module — benchmark results

Validation of the v0.8.0 depth-penetration capability (`src/fluorostats/depth.py`
pure per-stack functions + `src/fluorostats/depth_batch.py` manifest driver).

**Why not a "5-tool timed race":** the tools people actually use for depth-intensity
profiling (Fiji Plot Z-axis Profile, Imaris, ZEISS ZEN, Nikon NIS-Elements,
CellProfiler, MATLAB) are GUI/commercial and not scriptable head-to-head; there is
**no public ground-truth dataset** for confocal depth penetration (unlike BBBC039 for
nuclei — backed by a documented repository search in `data/competitor_depth_penetration.md`
§5); and the computation is sub-second numpy, so timing is not a differentiator.
So correctness is shown on **constructed Beer–Lambert ground truth** and the field
comparison is a **faithful-reimplementation parity check + a verified capability
matrix** — mirroring the paper's existing correctness-phantom + competitor-dossier
methodology.

Reproduce: `python3.13 methods_paper/benchmarks/b_depth_penetration.py`
Outputs: `results/b_depth_penetration_correctness.csv`,
`results/b_depth_penetration_parity_timing.csv`,
figure `figures/main/b_depth_penetration.{pdf,png}`.
Unit tests: `tests/test_depth.py` — 18/18 pass (incl. λ recovery / offset / bad-fit flag).

## A. Correctness on synthetic Beer–Lambert ground truth
Stacks with `I(z) = I0·e^(−z/λ) + bg` have known closed-form AUC
(`I0·λ·(1−e^(−Z/λ))`; normalised `λ·(1−e^(−Z/λ))`) and a known penetration
constant **λ**.

| Test | Result |
|---|---|
| **A1 exactness** (noiseless, λ=20–160) | normalised-AUC error **0.001–0.083 %** vs closed form (pure trapezoid discretisation) |
| **A2 noise robustness** (SNR≈10, 8 reps, matched blank) | recovered norm-AUC **39.84 ± 0.03** vs exact-pipeline GT 39.83 → **0.07 %** |
| **A3 discrimination** (short λ=30 vs long λ=70) | retained fraction **0.309 / 0.548** = GT 0.309 / 0.548 exactly; ordering correct; built-in Welch **p = 4.2 × 10⁻²⁴** |
| **Aλ recovery** (noiseless, known λ=20/40/80/160) | recovered λ **exact to 0.000 %**, R² = 1.00000 (`fit_penetration_depth`, scipy `curve_fit`) |
| **Aλ recovery** (SNR≈10 noisy, 8 reps, λ=40) | recovered λ **40.01 ± 0.03 µm → 0.03 %**, mean R² = 1.0000 |

**λ (penetration constant).** `depth.fit_penetration_depth` fits the field-standard
single-exponential `I(z)=I0·e^(−z/λ)` (or `+c` with `offset=True`) to the **absolute,
background-subtracted** profile, so λ is gain/laser-independent (I0 absorbs gain). It
carries `r_squared`/`rmse`/`fit_ok` (converged **and** R² ≥ 0.85, tuned on real data) so
a poor fit is flagged, never silently reported as a λ. **λ is additive — AUC stays the
model-agnostic default.** Parity note: the fit is scipy nonlinear least squares (the
field-standard estimator); it recovers the constructed λ to numerical exactness on
noiseless stacks and within noise at SNR≈10.

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

## E. Real data — GelMA vs GelMA-CMCMA hybrid (FITC-dextran 2000 kDa)
Run on the real confocal `.oib` stacks (`fluorostats depth
tools/permeability_fd2000.json`; 100 slices × 4.8 µm = 480 µm depth, blank-subtracted
against matched no-fluo controls). Primary stacks only for group stats; short stacks are
`role:"aux"`. Outputs: `methods_paper/permeability/{auc_per_stack,group_depth_summary,
depth_profiles_long}.csv` + figures.

| Gel | primary n | λ (µm), fit_ok fits | R² | AUC 0–100 (norm) | AUC 0–200 (norm) | AUC full (norm) |
|---|---|---|---|---|---|---|
| GelMA  | 2 | **81.9 ± 1.6** (2/2 in range) | 0.90 | 86.7 ± 6.3 | 97.5 ± 5.7 | 100.8 ± 4.5 |
| Hybrid | 3 | **not determinable** — 2/3 stacks give λ > 475 µm depth (degenerate → NaN); 1 in-range fit = 263 µm | 0.87 | 98.1 ± 1.6 | 188.6 ± 7.6 | 357.4 ± 88 |

- **Δ (Hybrid − GelMA):** ΔAUC(0–200) ≈ **+91** (Welch **p = 0.0009**); ΔAUC(full) ≈ **+257**
  (p = 0.037). AUC(0–100) does not separate (p = 0.23): both gels pass the probe freely in the
  first 100 µm; the gels diverge deeper. **A λ contrast is not computed** — only 1 hybrid stack
  yields an in-range λ (see below).
- **Read:** a **clean, biologically interpretable difference** — the GelMA-CMCMA hybrid is
  **more permeable**, letting FITC-dextran penetrate far deeper (near-flat profile out to
  480 µm), whereas GelMA decays with **λ ≈ 82 µm**. **AUC is the robust discriminator here.**
  The single-exponential **λ holds cleanly for GelMA (R² ≈ 0.90) but not for the hybrid**: its
  profile barely decays over the 480 µm window, so the fit returns a decay length *longer than
  the acquired stack* (2596 / 568 µm on two stacks). Those are unmeasurable extrapolations, not
  measurements — the **λ-in-range guard** in `fit_penetration_depth` flags them `fit_ok=False`
  and NaNs the λ (a high R² alone does **not** rescue a near-flat fit). This is exactly the
  "single-exponential does not hold → use AUC" case, and why λ is reported **only with its R²
  and range check, never alone**.

## Honest scope
This is a *reproducible pipeline over standard primitives*, not a novel algorithm.
The differentiator is correctness of the full chain + one-config reproducibility +
integrated stats, not the accuracy of any single step. Whole-image mean assumes a
roughly uniform field (the user selects the ROI/crop upstream). The **λ fit is
single-exponential** (Beer–Lambert / Amira Correct-Z-Drop / Bonda et al. 2020 STAR
Protocols 1(3):100180) — the field standard, not a new model; it is reported **only with
its R²/`fit_ok`**, and where the single-exponential does not hold (e.g. a near-flat,
deeply-penetrating profile) **AUC is the model-agnostic fallback**. Real-condition n is
small (2–3 stacks/gel), so the Welch tests are **descriptive only**.
