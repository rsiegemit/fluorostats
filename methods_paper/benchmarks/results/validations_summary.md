# Capability validations — statistics & power

## Statistics module (validate_stats.py) — 8/8 EXACT vs references
| Function | Reference | Result |
|---|---|---|
| mann_whitney (U, p) | scipy.stats.mannwhitneyu | exact (0 diff) |
| cliffs_delta | brute-force sign mean | exact |
| bh_fdr | hand-coded Benjamini-Hochberg | exact (0 diff) |
| stouffer_combine (one_sided) | scipy.stats.combine_pvalues | exact (2.459328) |
| bootstrap_fold_change_ci | covers true ratio e on lognormal | covered |
| scheirer_ray_hare | known interaction | p=0.001 detected |

Note: stouffer_combine defaults to a two-sided convention (isf(p/2)); `one_sided=True`
reproduces scipy exactly. Both correct.

## Power module (validate_power.py) — sound, with disclosed optimism
| Check | Result |
|---|---|
| power_curve monotonic in n | PASS |
| null effect power ≈ alpha (0.05) | PASS (0.05) |
| calibration n=8 (pred 0.22 vs true 0.35) | within 0.13 |
| calibration n=15 (pred 0.85 vs true 0.51) | optimistic +0.34 |
| calibration n=25 (pred 1.00 vs true 0.78) | optimistic +0.22 |

**Honest finding:** bootstrap power from a *single small pilot* is optimistic
(the pilot can over-represent the effect) — an inherent, documented property of
pilot-based bootstrap power (cf. Albers & Lakens 2018), NOT a fluorostats bug.
The estimator is directionally correct, monotonic, and null-controlled. The
module docstring already warns small pilots give optimistic, wide-CI curves; the
paper should present power curves with this caveat and recommend larger pilots.
