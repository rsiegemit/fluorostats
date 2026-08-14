"""Live/Dead viability quantification from two-channel fluorescence.

The classic Calcein-AM (live) / ethidium-or-PI (dead) assay reports two
signals. In thick 3D samples, confocal intensity attenuates with depth, so a
single plane or a maximum-intensity projection *overestimates* viability and
misses depth-dependent core death. This module quantifies viability in a
depth-aware way:

    - ``live_dead_fractions``     — whole-volume live / dead area or volume
                                    fractions and a viability ratio.
    - ``viability_depth_profile`` — per-z live / dead fraction (reveals a
                                    depth gradient a 2D method cannot see).
    - ``viability_2d_vs_3d``      — mid-plane vs MIP vs full-3D live fraction,
                                    quantifying the 2D overestimation.
    - ``attenuation_correct``     — per-z intensity normalisation to separate a
                                    biological death gradient from optical decay.

Segmentation is delegated to :mod:`fluorostats.segment`; pass already-segmented
boolean masks, or raw channels plus a ``threshold``/``method`` to segment here.
"""

from __future__ import annotations

import numpy as np

from .segment import binarize


def _mask(channel, mask=None, method="otsu", threshold_scale=1.0, min_size=32):
    if mask is not None:
        return mask > 0
    return binarize(channel, method=method, threshold_scale=threshold_scale,
                    min_size=min_size)


def live_dead_fractions(
    live: np.ndarray,
    dead: np.ndarray | None = None,
    *,
    live_mask=None,
    dead_mask=None,
    **seg,
) -> dict:
    """Whole-volume live / dead fractions and viability ratio.

    Returns dict: live_fraction, dead_fraction (of all voxels), and
    viability = live / (live + dead) area (NaN if no cells detected).
    """
    lm = _mask(live, live_mask, **seg)
    lf = float(lm.mean())
    out = {"live_fraction": lf}
    if dead is not None or dead_mask is not None:
        dm = _mask(dead, dead_mask, **seg)
        df = float(dm.mean())
        denom = lm.sum() + dm.sum()
        out["dead_fraction"] = df
        out["viability"] = float(lm.sum() / denom) if denom > 0 else float("nan")
    return out


def _estimate_regime(channel, **seg) -> dict:
    """Estimate cell size, SNR, and crowding to pick a counting method.

    Crowding = (smoothed peak count) / (connected-component count): when cells
    overlap, CC merges them into few blobs while one peak still marks each cell,
    so the ratio rises. Gated behind SNR so noise (which inflates peaks) does not
    masquerade as crowding.
    """
    import numpy as _np
    from .objects import label_3d, count_local_maxima

    seg = {**seg, "min_size": 5}          # never let a large min_size hide cells
    mask = _mask(channel, **seg)
    if not mask.any():
        return {"radius": 0.0, "snr": 0.0, "crowding": 1.0, "n_cc": 0}
    ch = _np.asarray(channel, float)
    fg, bg = ch[mask], ch[~mask]
    snr = float((fg.mean() - bg.mean()) / (bg.std() + 1e-9))
    labels, n_cc = label_3d(mask, min_size=5)
    sizes = _np.bincount(labels.ravel())[1:]
    med = float(_np.median(sizes)) if sizes.size else 0.0
    radius = (med / _np.pi) ** 0.5 if ch.ndim == 2 else (med * 0.75 / _np.pi) ** (1 / 3)
    n_pk = count_local_maxima(ch, min_distance=max(2, int(radius / 1.5)),
                              threshold_rel=0.2,
                              smooth_sigma=max(1.0, radius / 1.5))["count"]
    crowding = n_pk / max(n_cc, 1)
    return {"radius": radius, "snr": snr, "crowding": float(crowding), "n_cc": n_cc}


def choose_count_method(channel, **seg) -> dict:
    """Pick a counting method for an intensity channel from its image regime.

    Transparent heuristic (returns its reasoning; always overridable):

    - low SNR (< 3): ``"cc"`` — connected-component counting is the most
      noise-robust; peak finding would count noise.
    - crowded touching cells (peak count exceeds connected-components by >50%,
      crowding > 1.5) with adequate SNR: ``"maxima"`` (with auto ``smooth_sigma`` ≈ radius/1.5) — peaks
      separate overlapping cells that CC merges.
    - otherwise (well-separated): ``"cc"`` — exact and robust.

    Returns dict: method, smooth_sigma, and the estimated regime metrics.
    """
    r = _estimate_regime(channel, **seg)
    # Conservative: default to the noise-robust method (CC), escalate to maxima
    # ONLY when the image is clearly clean AND clearly crowded. Crowding and noise
    # both inflate peak counts and are not cleanly separable from image statistics,
    # so we bias toward the safe method and avoid firing maxima on noisy data.
    if r["n_cc"] == 0:
        method, reason = "cc", "no cells detected"
    elif r["snr"] >= 8.0 and r["crowding"] > 1.5:
        method, reason = "maxima", (
            f"clean (SNR {r['snr']:.1f}) and crowded (peaks/CC {r['crowding']:.2f})")
    elif r["snr"] < 3.0:
        method, reason = "cc", f"low SNR ({r['snr']:.1f}) — CC is noise-robust"
    else:
        method, reason = "cc", (
            f"defaulting to robust CC (SNR {r['snr']:.1f}, crowding {r['crowding']:.2f})")
    smooth = r["radius"] / 1.5 if method == "maxima" else 0.0
    return {"method": method, "smooth_sigma": smooth, "reason": reason, **r}


def live_dead_by_count(
    live: np.ndarray,
    dead: np.ndarray,
    *,
    method: str = "auto",
    min_distance: int = 3,
    threshold_rel: float = 0.15,
    smooth_sigma: float = 0.0,
    **seg,
) -> dict:
    """Viability from per-object COUNTS rather than area — robust to crowding.

    ``live_dead_fractions`` reports area-based viability, which under-reports in
    dense fields where cells overlap (merged blobs). Counting objects tracks the
    true live/(live+dead) cell ratio even when cells touch. Counting modes:

    - ``method="auto"`` (default) — picks per-channel via :func:`choose_count_method`
      (transparent regime heuristic); the chosen method + reasoning are returned.
    - ``method="maxima"`` — prominence peak counting (Fiji "Find Maxima"); best for
      crowded single-peak cells. Over-counts flat/noisy cells unless ``smooth_sigma``.
    - ``method="watershed"`` — distance-transform watershed split of the mask.
    - ``method="cc"`` — connected components; exact for separated cells, noise-robust.
    - ``method="all"`` — run every mode (they are cheap) and return each viability
      plus a ``consensus`` (median) and ``spread`` (max−min). Large spread flags a
      hard regime where the counting choice matters. NOTE: without ground truth you
      cannot know which single method is "best" for a given image, so ``all`` reports
      a consensus and a disagreement flag rather than silently picking one.

    No single mode is universally best. Returns dict: n_live, n_dead, viability,
    plus (auto) the chosen method + reasoning, or (all) per-method viabilities,
    consensus, and spread.
    """
    from .objects import count_local_maxima, watershed_split, label_3d

    def _count_one(ch, m, sig):
        cseg = {"min_size": 5, **seg}     # counting keeps small real cells
        if m == "maxima":
            return count_local_maxima(ch, min_distance, threshold_rel,
                                      smooth_sigma=sig)["count"]
        if m == "watershed":
            return watershed_split(_mask(ch, **cseg), min_distance=min_distance)[1]
        if m == "cc":
            return label_3d(_mask(ch, **cseg), min_size=5)[1]
        raise ValueError(f"unknown method {method!r} (use auto/all/maxima/watershed/cc)")

    def _viab(nl, nd):
        t = nl + nd
        return nl / t if t > 0 else float("nan")

    if method == "all":
        modes = ["cc", "watershed", "maxima"]
        # auto-smooth the maxima channel from each channel's estimated radius
        sig_l = choose_count_method(live, **seg)["radius"] / 1.5
        sig_d = choose_count_method(dead, **seg)["radius"] / 1.5
        per = {}
        for m in modes:
            nl = _count_one(live, m, sig_l if m == "maxima" else 0.0)
            nd = _count_one(dead, m, sig_d if m == "maxima" else 0.0)
            per[m] = {"n_live": nl, "n_dead": nd, "viability": _viab(nl, nd)}
        vals = [p["viability"] for p in per.values() if p["viability"] == p["viability"]]
        consensus = float(np.median(vals)) if vals else float("nan")
        spread = float(max(vals) - min(vals)) if vals else float("nan")
        return {"by_method": per, "consensus": consensus, "spread": spread,
                "viability": consensus}

    def _count(ch):
        m, sig = method, smooth_sigma
        info = None
        if method == "auto":
            info = choose_count_method(ch, **seg)
            m, sig = info["method"], info["smooth_sigma"]
        n = _count_one(ch, m, sig)
        return n, (info["method"] if info else m), (info["reason"] if info else "")

    n_live, ml, rl = _count(live)
    n_dead, md, rd = _count(dead)
    out = {"n_live": n_live, "n_dead": n_dead, "viability": _viab(n_live, n_dead)}
    if method == "auto":
        out.update({"method_live": ml, "method_dead": md,
                    "reason_live": rl, "reason_dead": rd})
    return out


def viability_depth_profile(
    live: np.ndarray,
    dead: np.ndarray | None = None,
    z_axis: int = 0,
    *,
    live_mask=None,
    dead_mask=None,
    **seg,
) -> dict:
    """Per-z live (and dead) area fraction — the depth-resolved viability.

    Returns dict of 1D arrays (length = n z slices): live_by_z, and if a dead
    channel is given, dead_by_z and viability_by_z.
    """
    lm = _mask(live, live_mask, **seg)
    axes = tuple(a for a in range(lm.ndim) if a != z_axis)
    live_by_z = lm.mean(axis=axes)
    out = {"live_by_z": live_by_z}
    if dead is not None or dead_mask is not None:
        dm = _mask(dead, dead_mask, **seg)
        dead_by_z = dm.mean(axis=axes)
        la = lm.sum(axis=axes).astype(float)
        da = dm.sum(axis=axes).astype(float)
        denom = la + da
        out["dead_by_z"] = dead_by_z
        out["viability_by_z"] = np.divide(la, denom, out=np.full_like(la, np.nan),
                                          where=denom > 0)
    return out


def viability_2d_vs_3d(
    live: np.ndarray,
    z_axis: int = 0,
    *,
    live_mask=None,
    **seg,
) -> dict:
    """Compare live fraction from a single mid plane, a MIP, and the full 3D
    volume — quantifying how much a 2D readout overestimates viability.

    Returns dict: live_fraction_midplane, live_fraction_mip, live_fraction_3d,
    overestimate_mip = mip / 3d (>1 means the 2D MIP overstates coverage), and
    overestimate_midplane = midplane / 3d.
    """
    lm = _mask(live, live_mask, **seg)
    nz = lm.shape[z_axis]
    mid = np.take(lm, nz // 2, axis=z_axis)
    mip = lm.max(axis=z_axis)
    f3d = float(lm.mean())
    f_mid = float(mid.mean())
    f_mip = float(mip.mean())
    return {
        "live_fraction_midplane": f_mid,
        "live_fraction_mip": f_mip,
        "live_fraction_3d": f3d,
        "overestimate_mip": float(f_mip / f3d) if f3d > 0 else float("nan"),
        "overestimate_midplane": float(f_mid / f3d) if f3d > 0 else float("nan"),
    }


def attenuation_correct(
    volume: np.ndarray,
    z_axis: int = 0,
    *,
    reference: str = "max",
) -> np.ndarray:
    """Per-z intensity normalisation to compensate depth attenuation.

    Each z slice is scaled so its mean matches a reference slice (the brightest
    slice by default, i.e. shallow, least-attenuated). Signal that then *stays*
    low with depth reflects genuine biology, not optical decay — use before
    :func:`viability_depth_profile` to check a death gradient is real.

    Returns a new float array; the input is not modified.
    """
    vol = volume.astype(np.float32)
    axes = tuple(a for a in range(vol.ndim) if a != z_axis)
    slice_means = vol.mean(axis=axes)
    if reference == "max":
        ref = slice_means.max()
    elif reference == "first":
        ref = slice_means.flat[0]
    else:
        ref = slice_means.mean()
    scale = np.divide(ref, slice_means, out=np.ones_like(slice_means),
                      where=slice_means > 0)
    shape = [1] * vol.ndim
    shape[z_axis] = -1
    return vol * scale.reshape(shape)


__all__ = [
    "live_dead_fractions",
    "viability_depth_profile",
    "viability_2d_vs_3d",
    "attenuation_correct",
]
