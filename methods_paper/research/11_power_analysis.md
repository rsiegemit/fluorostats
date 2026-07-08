# Statistical Power Analysis & Sample-Size Determination (Bootstrap/Simulation, Imaging)

Research supporting fluorostats' bootstrap-based power analysis feature: resampling from
observed pilot samples to estimate power vs. sample size (including joint power under BH-FDR
across multiple metrics), for planning microscopy replicate counts without parametric
normality assumptions.

## The reproducibility problem (why this matters)

1. **Button, K.S., Ioannidis, J.P.A., Mokrysz, C., Nosek, B.A., Flint, J., Robinson, E.S.J.,
   Munafò, M.R. (2013). "Power failure: why small sample size undermines the reliability of
   neuroscience."** *Nature Reviews Neuroscience* 14(5): 365–376. DOI: 10.1038/nrn3475.
   Highly cited (>8,000). Median statistical power across neuroscience studies ≈ 20%. Low power
   not only misses true effects but inflates the false-positive rate among "significant" results
   and produces exaggerated effect-size estimates (winner's curse). *The canonical motivation
   for taking sample-size planning seriously in biology.*

2. **Szucs, D., Ioannidis, J.P.A. (2017). "Empirical assessment of published effect sizes and
   power in the recent cognitive neuroscience and psychology literature."** *PLOS Biology*
   15(3): e2000797. DOI: 10.1371/journal.pbio.2000797. Analyzed 26,841 tests from 3,801 papers;
   median power ≈ 0.12 (small effects), 0.44 (medium), 0.73 (large). Corroborates Button et al.
   at scale. *Underpowering is systemic, not anecdotal.*

## Classical / parametric baseline (what fluorostats avoids)

3. **Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).**
   Lawrence Erlbaum Associates. ISBN 0-8058-0283-5. Foundational text; defines effect size
   (Cohen's d) and small/medium/large benchmarks (0.2/0.5/0.8). *The parametric-formula tradition
   fluorostats departs from — closed-form power requires assuming a distribution and a single
   scalar effect size, which imaging metrics (skewed, bounded, heavy-tailed) often violate.*

## Simulation / bootstrap-based power (fluorostats' direct lineage)

4. **Green, P., MacLeod, C.J. (2016). "SIMR: an R package for power analysis of generalized
   linear mixed models by simulation."** *Methods in Ecology and Evolution* 7(4): 493–498.
   DOI: 10.1111/2041-210X.12504. Widely cited. Monte Carlo simulation of GLMMs (lme4) to compute
   power and power curves across sample sizes. *Closest established analogue in spirit: simulate
   the analysis, count rejections. fluorostats differs by resampling observed data
   (nonparametric bootstrap) rather than simulating from a fitted parametric model.*

5. **Kumle, L., Võ, M.L.-H., Draschkow, D. (2021). "Estimating power in (generalized) linear
   mixed models: An open introduction and tutorial in R."** *Behavior Research Methods* 53:
   2528–2543. DOI: 10.3758/s13428-021-01546-0. (PMC8613146). Accessible tutorial on
   simulation-based power for multilevel/nested designs — directly relevant to microscopy's
   cells-within-images-within-animals hierarchy. *Establishes simulation as the accepted route
   when analytic power formulas don't exist.*

6. **Beasley, T.M., et al. — "Simulating Statistical Power Curves with the Bootstrap and Robust
   Estimation."** UNT Digital Library (ark:/67531/metadc2846). Demonstrates bootstrap resampling
   to build power curves with robust estimators, without distributional assumptions. *Direct
   methodological precedent for fluorostats' resample-from-pilot approach; verify exact
   authors/venue/year before citing (record incomplete from search).* ⚠️ FLAGGED — verify.

## Power under multiple testing / FDR (the joint-power differentiator)

7. **Sheng, Y., Jung, S.-H., et al. (2024). "Computing Power and Sample Size for the False
   Discovery Rate in Multiple Applications."** *Genes* 15(3): 344. DOI: 10.3390/genes15030344.
   (PMC10970028). Introduces the R package **FDRsamplesize2**; computes sample size for a target
   *average power* at a desired FDR across many tests. *fluorostats' joint power under BH-FDR is
   the imaging-workflow counterpart — but computed by bootstrap over the actual multi-metric
   pilot data rather than by formula, so it captures correlation between metrics that formula-based
   average-power methods typically assume away.*

8. **Jung, S.-H. (2005). "Sample size for FDR-control in microarray data analysis."**
   *Bioinformatics* 21(14): 3097–3104. DOI: 10.1093/bioinformatics/bti456. Origin of the
   average-power-at-fixed-FDR framework. *The theoretical root of point 7; establishes that
   "power" must be redefined (average/expected proportion of true effects detected) once FDR
   replaces per-test alpha — exactly the quantity fluorostats reports across metrics.*

## The key limitation (must acknowledge in the paper)

9. **Albers, C., Lakens, D. (2018). "When power analyses based on pilot data are biased:
   Inaccurate effect size estimators and follow-up bias."** *Journal of Experimental Social
   Psychology* 74: 187–195. DOI: 10.1016/j.jesp.2017.09.004. Effect sizes from small pilots are
   so noisy that pilot-based power analyses systematically yield underpowered main studies; a
   pilot d that is an overestimate leads to a too-small n. *The central caveat for any
   resample-from-pilot method, including fluorostats: bootstrap inherits the pilot's sampling
   error, so small-n power curves are optimistic and uncertain.*

10. **Teare, M.D., et al. (2014). "Sample size requirements to estimate key design parameters
    from external pilot randomised controlled trials: a simulation study."** *Trials* 15: 264.
    DOI: 10.1186/1745-6215-15-264. (PMC4227298). Quantifies how pilot size drives instability of
    downstream sample-size estimates. *Supports fluorostats reporting uncertainty bands on power
    curves and recommending a minimum pilot size.*

## Imaging-specific rigor context

11. **Lee, J.-Y., Kitaoka, M. (2018). "A beginner's guide to rigor and reproducibility in
    fluorescence imaging experiments."** *Molecular Biology of the Cell* 29(13): 1519–1525.
    DOI: 10.1091/mbc.E17-05-0276. Practical guidance on replicates, controls, and quantification
    in fluorescence microscopy. *Establishes the audience/venue: imaging biologists need
    sample-size guidance but have almost no imaging-native tooling for it — the gap fluorostats
    fills.*

## Comparison to fluorostats

Bootstrap/simulation-based power estimation is **well-established in statistics** (points 4–8)
but is **essentially absent from the bioimage-analysis toolchain**. Imaging tools (CellProfiler,
QuPath, Fiji, scikit-image) quantify morphometry and intensity but stop short of experimental
planning; power analysis, when done at all, is done afterward in G*Power or R with parametric
formulas that assume normality — an assumption imaging metrics (areas, counts, intensities;
skewed, bounded, zero-inflated) routinely break.

**fluorostats' differentiators:**
- **Nonparametric, pilot-driven:** resamples the observed pilot distribution — no normality,
  no pre-specified scalar effect size (contrast points 3–4).
- **In the imaging workflow:** power estimation lives beside the morphometry/stats it consumes,
  so the pilot data *are* the effect-size input — no manual d-guessing.
- **Joint power under BH-FDR across metrics:** bootstrap over the multi-metric pilot naturally
  respects inter-metric correlation, which formula-based average-power methods (points 7–8)
  approximate crudely.

**Honest limits (frame proactively, cite 9–10):** bootstrap power inherits pilot sampling error;
tiny pilots give optimistic, wide-CI curves. fluorostats should (a) report uncertainty on power
curves, (b) warn below a minimum pilot n, (c) frame output as planning guidance, not a guarantee.

## How the methods paper should present this + validation

**Positioning:** "A nonparametric, pilot-driven power estimator embedded in the imaging analysis
workflow, extending bootstrap power curves to joint detection under FDR across multiple
morphometric readouts — bringing an established statistical method into a domain that currently
lacks it."

**Validation plan (calibration study):**
1. **Ground-truth simulation.** Draw pilots from distributions with *known* effect sizes
   (including non-normal: log-normal intensities, Poisson counts, bounded areas). Run the
   bootstrap power estimator; compare its predicted power at each n against the **empirical
   rejection rate** from many independent full-size datasets simulated from the same truth.
   Well-calibrated ⇒ predicted ≈ empirical.
2. **Pilot-size sensitivity.** Repeat across pilot n (e.g., 3, 5, 10, 20) to quantify and
   display the optimism/instability documented in Albers & Lakens (point 9) — turning the
   limitation into an honest, plotted uncertainty band.
3. **FDR joint-power check.** With several metrics at known effects and known correlation,
   confirm the bootstrap joint-power estimate under BH-FDR matches the empirical average power
   and realized FDR — and that it beats a formula (FDRsamplesize2) when metrics are correlated.
4. **Cross-check vs. parametric.** Where normality *does* hold, show fluorostats' bootstrap
   power agrees with G*Power/pwr, establishing correctness before claiming the nonparametric
   advantage where normality fails.

---
### 200-word summary
Bootstrap/simulation-based power analysis is well-established in statistics — SIMR (Green &
MacLeod 2016) simulates GLMMs to build power curves; FDRsamplesize2 (Sheng/Jung 2024) and Jung
(2005) compute average power at a target FDR; bootstrap power curves with robust estimation are
documented — yet this machinery is absent from bioimage-analysis tools, which quantify images but
leave sample-size planning to post-hoc parametric formulas that assume normality imaging metrics
violate. The reproducibility literature (Button et al. 2013, *Nat Rev Neurosci*; Szucs &
Ioannidis 2017) shows median power near 20%, making credible planning urgent. fluorostats'
differentiator is a **nonparametric, pilot-driven** estimator living **inside the imaging
workflow**, resampling observed pilot data (no normality, no guessed effect size) and uniquely
reporting **joint power under BH-FDR across correlated metrics**. The honest limit — bootstrap
inherits pilot sampling error, so small pilots give optimistic, wide-CI curves (Albers & Lakens
2018; Teare 2014) — should be framed proactively with plotted uncertainty bands and a minimum-n
warning.

**Validation idea:** simulate pilots from *known* (non-normal) effect sizes; confirm the
estimator's predicted power matches the empirical rejection rate from many independent full-size
datasets, and that joint FDR power matches realized average power/FDR — a direct calibration
check.

### Unverifiable citations flagged
- **#6 (Beasley et al., bootstrap power curves, UNT Digital Library):** ark handle confirmed but
  exact author list/venue/year not fully verified — confirm before citing.
- All others (#1–5, 7–11) have DOIs consistent with well-known, verifiable publications; #3
  (Cohen 1988) is a book (ISBN, no DOI).
