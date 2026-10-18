"""Produce results/b_depth_penetration_*.csv — the depth module on synthetic Beer–Lambert stacks (Figure S5).

There is no public ground-truth set for confocal depth penetration, so the
stacks are constructed: every plane of a noiseless stack is the constant
``I0 * exp(-z / lambda) + bg``, so its per-slice mean, AUC and λ are known in
closed form; the noisy stacks add seeded Gaussian noise (sigma 100, surface
SNR ~10). Tables:

- ``b_depth_penetration_correctness``: normalized-AUC recovery on noiseless
  stacks (A1), on noisy replicates (A2) and for two conditions (A3); λ recovery
  by ``depth.fit_penetration_depth`` on noiseless stacks and noisy replicates (Alam);
- ``b_depth_penetration_profile``: Figure S5(a), the mean ± s.d. surface-normalized
  profile of 8 noisy replicates beside the analytic ``exp(-z / lambda)``;
- ``b_depth_penetration_conditions``: Figure S5(c), the retained fraction over
  0–100 µm of each of 6 replicates at λ = 30 and 70 µm, with the noiseless truth;
- ``b_depth_penetration_parity``: the profile against a float64 per-slice mean
  computed here (not Fiji; ``b_zprofile_fiji`` is the Fiji comparison), the AUC
  against ``scipy.integrate.trapezoid``, the trapezoid and a rectangular sum against
  the analytic integral, and two identical runs.

Pure numpy; no data needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate

from common import write_table
from fluorostats import depth

I0, BG, DZ, YX = 1000.0, 50.0, 2.0, 128
SIGMA = 100.0                       # noise s.d.: surface SNR ~10
N_Z_AUC, N_Z_LAMBDA = 60, 100       # slices: AUC tests, λ tests
LAMBDAS = (20.0, 40.0, 80.0, 160.0)
NOISY_LAMBDA, N_REP = 40.0, 8
CONDITIONS = {"short": 30.0, "long": 70.0}
N_COND_REP, Z_WINDOW = 6, 100.0


def beer_lambert_stack(n_z, yx, i0, lam, bg, sigma=0.0, dz=1.0, seed=0):
    """(Z, Y, X) float32 stack whose per-slice mean is I0*exp(-z/lam)+bg (+ seeded noise)."""
    z = np.arange(n_z, dtype=np.float64) * dz
    per_slice = i0 * np.exp(-z / lam) + bg
    vol = np.repeat(per_slice[:, None, None], yx * yx, axis=1).reshape(n_z, yx, yx)
    if sigma > 0:
        vol = vol + np.random.default_rng(seed).normal(0.0, sigma, size=vol.shape)
    return vol.astype(np.float32)


def analytic_auc(i0, lam, z0, z1):
    """Integral of I0*exp(-z/lam) over [z0, z1]."""
    return i0 * lam * (np.exp(-z0 / lam) - np.exp(-z1 / lam))


def run_pipeline(vol, dz, bg, z0, z1, n_surface=1):
    """Profile → background → surface normalization → (absolute AUC, normalized AUC)."""
    prof = depth.intensity_depth_profile(vol, dz)
    sub = depth.subtract_background(prof, float(bg))
    nrm = depth.normalize_to_surface(sub, n_surface=n_surface)
    return (depth.auc_depth(prof.depth_um, sub.mean, z_min=z0, z_max=z1),
            depth.auc_depth(prof.depth_um, nrm.normalized, z_min=z0, z_max=z1))


def noisy_pair(n_z, lam, seed, blank_seed):
    """A noisy stack and its matched noisy no-fluorophore blank, as profiles."""
    vol = beer_lambert_stack(n_z, YX, I0, lam, BG, SIGMA, DZ, seed)
    blank = beer_lambert_stack(n_z, YX, 0.0, lam, BG, SIGMA, DZ, blank_seed)
    return depth.intensity_depth_profile(vol, DZ), depth.intensity_depth_profile(blank, DZ)


def noiseless_norm_auc(lam, z0, z1, n_surface):
    """The pipeline's exact normalized AUC on a noiseless stack and blank (truth for noisy tests)."""
    p = depth.intensity_depth_profile(beer_lambert_stack(N_Z_AUC, YX, I0, lam, BG, dz=DZ), DZ)
    b = depth.intensity_depth_profile(beer_lambert_stack(N_Z_AUC, YX, 0.0, lam, BG, dz=DZ), DZ)
    nn = depth.normalize_to_surface(depth.subtract_background(p, b), n_surface=n_surface)
    return depth.auc_depth(p.depth_um, nn.normalized, z_min=z0, z_max=z1)


def auc_rows():
    z1 = (N_Z_AUC - 1) * DZ
    rows = []
    for lam in LAMBDAS:
        abs_auc, norm_auc = run_pipeline(beer_lambert_stack(N_Z_AUC, YX, I0, lam, BG, dz=DZ), DZ, BG, 0.0, z1)
        gt_abs, gt_norm = analytic_auc(I0, lam, 0.0, z1), analytic_auc(1.0, lam, 0.0, z1)
        rows.append(dict(test="A1_exact_noiseless", lam=lam, gt_norm_auc=gt_norm, fs_norm_auc=norm_auc,
                         err_norm_pct=abs(norm_auc - gt_norm) / gt_norm * 100,
                         err_abs_pct=abs(abs_auc - gt_abs) / gt_abs * 100))
    gt = noiseless_norm_auc(NOISY_LAMBDA, 0.0, z1, n_surface=3)
    noisy = []
    for r in range(N_REP):
        prof, blank = noisy_pair(N_Z_AUC, NOISY_LAMBDA, r, 100 + r)
        nrm = depth.normalize_to_surface(depth.subtract_background(prof, blank), n_surface=3)
        noisy.append(depth.auc_depth(prof.depth_um, nrm.normalized, z_min=0.0, z_max=z1))
    rows.append(dict(test="A2_noisy_recovery", lam=NOISY_LAMBDA, gt_norm_auc=gt,
                     fs_norm_auc=float(np.mean(noisy)),
                     err_norm_pct=float(np.abs(np.array(noisy) - gt).mean()) / gt * 100, err_abs_pct=np.nan))
    return rows


def conditions():
    """Per-replicate retained fraction over 0–Z_WINDOW µm for the two λ conditions."""
    rows = []
    for name, lam in CONDITIONS.items():
        truth = noiseless_norm_auc(lam, 0.0, Z_WINDOW, 3) / Z_WINDOW
        for r in range(N_COND_REP):
            vol = beer_lambert_stack(N_Z_AUC, YX, I0, lam, BG, SIGMA, DZ, 200 + r)
            rows.append(dict(condition=name, lambda_um=lam, replicate=r, truth=truth,
                             retained_fraction=run_pipeline(vol, DZ, BG, 0.0, Z_WINDOW, n_surface=3)[1] / Z_WINDOW))
    return pd.DataFrame(rows)


def condition_rows(cond):
    rows = []
    for name, lam in CONDITIONS.items():
        c = cond[cond.condition == name]
        truth = float(c.truth.iloc[0])
        rows.append(dict(test="A3_discrimination", lam=lam, gt_norm_auc=truth,
                         fs_norm_auc=float(c.retained_fraction.mean()),
                         err_norm_pct=abs(c.retained_fraction.mean() - truth) / truth * 100, err_abs_pct=np.nan))
    return rows


def lambda_rows():
    rows = []
    for lam in LAMBDAS:
        prof = depth.intensity_depth_profile(beer_lambert_stack(N_Z_LAMBDA, YX, I0, lam, BG, dz=DZ), DZ)
        fit = depth.fit_penetration_depth(prof.depth_um, depth.subtract_background(prof, BG).mean)
        rows.append(dict(test="Alam_exact_noiseless", gt_lambda=lam, fs_lambda=fit.lambda_um,
                         lambda_err_pct=abs(fit.lambda_um - lam) / lam * 100, r_squared=fit.r_squared))
    lams, r2 = [], []
    for r in range(N_REP):
        prof, blank = noisy_pair(N_Z_LAMBDA, NOISY_LAMBDA, r, 100 + r)
        fit = depth.fit_penetration_depth(prof.depth_um, depth.subtract_background(prof, blank).mean)
        lams.append(fit.lambda_um)
        r2.append(fit.r_squared)
    rows.append(dict(test="Alam_noisy_recovery", gt_lambda=NOISY_LAMBDA, fs_lambda=float(np.mean(lams)),
                     lambda_err_pct=abs(np.mean(lams) - NOISY_LAMBDA) / NOISY_LAMBDA * 100,
                     r_squared=float(np.mean(r2))))
    return rows


def profile():
    """Surface-normalized profiles of the noisy replicates (reference = surface slice)."""
    curves = []
    for r in range(N_REP):
        prof, blank = noisy_pair(N_Z_AUC, NOISY_LAMBDA, r, 100 + r)
        curves.append(depth.normalize_to_surface(depth.subtract_background(prof, blank), n_surface=1).normalized)
    curves = np.vstack(curves)
    z = np.arange(N_Z_AUC) * DZ
    return pd.DataFrame(dict(depth_um=z, analytic=np.exp(-z / NOISY_LAMBDA),
                             mean=curves.mean(0), sd=curves.std(0, ddof=1)))


def parity():
    n_z, lam = 50, 45.0
    z1 = (n_z - 1) * DZ
    vol = beer_lambert_stack(n_z, 96, I0, lam, 30.0, 250.0, DZ, 7)
    fs = depth.intensity_depth_profile(vol, DZ)
    ref = vol.astype(np.float64).reshape(n_z, -1).mean(axis=1)
    fs_auc = depth.auc_depth(fs.depth_um, fs.mean, z_min=0.0, z_max=z1)
    scipy_auc = float(integrate.trapezoid(fs.mean, fs.depth_um))
    clean = depth.intensity_depth_profile(beer_lambert_stack(n_z, 96, I0, lam, 0.0, dz=DZ), DZ)
    gt = analytic_auc(I0, lam, 0.0, z1)
    trap = depth.auc_depth(clean.depth_um, clean.mean, z_min=0.0, z_max=z1)
    rect = float(np.sum(clean.mean * DZ))                       # left-Riemann sum
    vol2 = beer_lambert_stack(60, 256, I0, 40.0, BG, 300.0, DZ, 1)
    same = run_pipeline(vol2, DZ, BG, 0.0, 100.0) == run_pipeline(vol2, DZ, BG, 0.0, 100.0)
    return pd.DataFrame([
        dict(check="perslice_mean_vs_float64_mean", value=float(np.max(np.abs(fs.mean - ref))),
             ok=bool(np.array_equal(fs.mean, ref))),
        dict(check="scipy_trapezoid_parity", value=abs(fs_auc - scipy_auc), ok=abs(fs_auc - scipy_auc) < 1e-6),
        dict(check="trapezoid_err_pct_vs_analytic", value=abs(trap - gt) / gt * 100, ok=True),
        dict(check="rectangular_err_pct_vs_analytic", value=abs(rect - gt) / gt * 100, ok=True),
        dict(check="determinism_bit_identical", value=0.0, ok=bool(same)),
    ])


def main():
    params = dict(i0=I0, bg=BG, dz_um=DZ, yx=YX, sigma=SIGMA, lambdas_um=list(LAMBDAS),
                  noisy_lambda_um=NOISY_LAMBDA, n_replicates=N_REP, conditions_um=CONDITIONS,
                  n_condition_replicates=N_COND_REP, window_um=Z_WINDOW,
                  seeds="noisy stack r, blank 100 + r, conditions 200 + r")
    cond = conditions()
    table = pd.DataFrame(auc_rows() + condition_rows(cond) + lambda_rows())
    write_table(table, "b_depth_penetration_correctness", params)
    write_table(profile(), "b_depth_penetration_profile", {**params, "n_surface": 1})
    write_table(cond, "b_depth_penetration_conditions", {**params, "n_surface": 3})
    write_table(parity(), "b_depth_penetration_parity",
                {"reference": "float64 per-slice mean computed here, not Fiji (see b_zprofile_fiji)"})
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
