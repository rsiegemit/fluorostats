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
boolean masks, or raw channels plus a ``threshold``/``method`` to segment here. The
segmentation keywords are ``method``, ``min_size``, ``threshold_scale`` and
``opening`` (`segment.binarize`'s; its default 3D ``"ball"`` opening erases cells
thinner than 3 z-slices, so use ``opening="disk"`` for a monolayer).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from .objects import count_local_maxima, label_3d, watershed_split
from .segment import binarize

# Thresholding method used to segment a channel when none is given.
DEFAULT_SEG_METHOD = "otsu"
# Smallest object (voxels) kept when segmenting a channel for area/coverage.
AREA_MIN_SIZE = 32
# Smallest object (voxels) kept when counting cells: counting keeps small real cells.
COUNT_MIN_SIZE = 5
# `choose_count_method`'s regime thresholds: SNR at or above CLEAN_SNR is clean, below
# LOW_SNR is noisy; peaks per connected component above CROWDED is crowded. Hand-set
# heuristics (v0.6), checked only on the synthetic regimes in tests/test_viability.py;
# no benchmark table chose them, hence "auto" escalates to maxima only when an image
# is both clearly clean and clearly crowded.
CLEAN_SNR = 8.0
LOW_SNR = 3.0
CROWDED = 1.5
# The regime estimate's peak count: peaks above REGIME_THRESHOLD_REL of the image
# maximum, smoothed with sigma = radius / RADIUS_PER_SIGMA (also the smoothing the
# maxima count gets from the estimated radius). Hand-set with the constants above;
# no benchmark table chose them.
REGIME_THRESHOLD_REL = 0.2
RADIUS_PER_SIGMA = 1.5


def _mask(
    channel: np.ndarray | None,
    mask: np.ndarray | None = None,
    method: str = DEFAULT_SEG_METHOD,
    threshold_scale: float = 1.0,
    min_size: int = AREA_MIN_SIZE,
    opening: str = "ball",
) -> np.ndarray:
    if mask is not None:
        return mask > 0
    assert channel is not None  # callers pass a channel whenever they pass no mask
    return binarize(
        channel, method=method, threshold_scale=threshold_scale, min_size=min_size, opening=opening
    )


def _count_cc(ch, sig, cseg, min_distance, threshold_rel) -> int:
    return label_3d(_mask(ch, **cseg), min_size=COUNT_MIN_SIZE)[1]


def _count_watershed(ch, sig, cseg, min_distance, threshold_rel) -> int:
    return watershed_split(_mask(ch, **cseg), min_distance=min_distance)[1]


def _count_maxima(ch, sig, cseg, min_distance, threshold_rel) -> int:
    return count_local_maxima(ch, min_distance, threshold_rel, smooth_sigma=sig)["count"]


class _Counter(NamedTuple):
    """A count method: ``count(ch, sigma, seg, min_distance, threshold_rel)`` and
    whether it takes a smoothing sigma estimated from the cell radius (``smoothed``;
    the others are given 0.0)."""

    count: Callable[..., int]
    smoothed: bool


# The object-counting modes of `live_dead_by_count`, in the order ``method="all"``
# reports them (and `viability_batch` writes their columns).
_COUNTERS = {
    "cc": _Counter(_count_cc, smoothed=False),
    "watershed": _Counter(_count_watershed, smoothed=False),
    "maxima": _Counter(_count_maxima, smoothed=True),
}
COUNT_METHODS = tuple(_COUNTERS)


# Well-known Live/Dead dyes, matched in channel names when a run does not give its
# channel indices. Live: calcein, FITC, GFP, Alexa Fluor 488. Dead: propidium iodide,
# ethidium homodimer. Other dyes (e.g. DAPI, far-red labels) are never guessed.
LIVE_DYES = re.compile(r"calcein|fitc|gfp|alexa.*488|af\s*488", re.IGNORECASE)
DEAD_DYES = re.compile(r"propidium|\bpi\b|ethd|ethidium", re.IGNORECASE)


def live_dead_channels(
    channel_names: list[str], live: int | None = None, dead: int | None = None
) -> tuple[int, int]:
    """Indices ``(live, dead)`` of the live and dead channels among ``channel_names``.

    An index given explicitly is used as is. Otherwise the first channel whose name
    matches `LIVE_DYES` / `DEAD_DYES` is chosen; with no match the channel is a guess
    by position (live 0, dead 1; the other of the two when that one is already the
    other role's channel), announced with a RuntimeWarning. Raises ValueError if both resolve to one channel or an
    index is negative or past the channel count.
    """

    def by_name(given, pattern):
        if given is not None:
            return given
        return next((i for i, name in enumerate(channel_names) if pattern.search(name)), None)

    def guess(role, preferred, taken):
        default = preferred if preferred != taken else 1 - preferred
        warnings.warn(
            f"no channel name identifies the {role} channel among "
            f"{channel_names}; using channel {default} by position. Set "
            f"{role}_channel in the manifest if that is wrong.",
            RuntimeWarning,
            stacklevel=3,
        )
        return default

    li = by_name(live, LIVE_DYES)
    di = by_name(dead, DEAD_DYES)
    if li is None:
        li = guess("live", 0, di)
    if di is None:
        di = guess("dead", 1, li)
    if min(li, di) < 0:
        raise ValueError(f"channel indices must not be negative, got live {li}, dead {di}")
    if li == di:
        raise ValueError(
            f"live and dead are both channel {li} of {channel_names}; set "
            "live_channel and dead_channel explicitly"
        )
    if max(li, di) >= len(channel_names):
        raise ValueError(
            f"channel {max(li, di)} requested but the file has "
            f"{len(channel_names)} channels {channel_names}"
        )
    return li, di


def live_ratio(n_live: float | np.integer, n_dead: float | np.integer) -> float:
    """Viability ``live / (live + dead)``; NaN when there is nothing to count."""
    total = n_live + n_dead
    return float(n_live / total) if total > 0 else float("nan")


def live_dead_fractions(
    live: np.ndarray,
    dead: np.ndarray | None = None,
    *,
    live_mask: np.ndarray | None = None,
    dead_mask: np.ndarray | None = None,
    **seg,
) -> dict:
    """Whole-volume live / dead fractions and viability ratio.

    Returns dict: live_fraction, dead_fraction (of all voxels), and
    viability = live / (live + dead) area (NaN if no cells detected). Without a
    dead channel or mask, dead_fraction and viability are NaN.
    """
    lm = _mask(live, live_mask, **seg)
    lf = float(lm.mean())
    out = {"live_fraction": lf, "dead_fraction": float("nan"), "viability": float("nan")}
    if dead is not None or dead_mask is not None:
        dm = _mask(dead, dead_mask, **seg)
        df = float(dm.mean())
        out["dead_fraction"] = df
        out["viability"] = live_ratio(lm.sum(), dm.sum())
    return out


def _estimate_regime(channel: np.ndarray, **seg) -> dict:
    """Estimate cell size, SNR, and crowding to pick a counting method.

    Crowding = (smoothed peak count) / (connected-component count): when cells
    overlap, CC merges them into few blobs while one peak still marks each cell,
    so the ratio rises. Gated behind SNR so noise (which inflates peaks) does not
    masquerade as crowding.

    The radius is in voxels, from the median object voxel count, so it assumes
    isotropic voxels (on anisotropic stacks it is a voxel-count equivalent).
    """
    seg = {**seg, "min_size": COUNT_MIN_SIZE}  # never let a large min_size hide cells
    mask = _mask(channel, **seg)
    if not mask.any():  # no cells: nothing was measured
        return {"radius": float("nan"), "snr": float("nan"), "crowding": float("nan"), "n_cc": 0}
    ch = np.asarray(channel, float)
    fg, bg = ch[mask], ch[~mask]
    snr = float((fg.mean() - bg.mean()) / (bg.std() + 1e-9))
    labels, n_cc = label_3d(mask, min_size=COUNT_MIN_SIZE)
    sizes = np.bincount(labels.ravel())[1:]
    med = float(np.median(sizes)) if sizes.size else 0.0
    radius = (med / np.pi) ** 0.5 if ch.ndim == 2 else (med * 0.75 / np.pi) ** (1 / 3)
    n_pk = count_local_maxima(
        ch,
        min_distance=max(2, int(radius / RADIUS_PER_SIGMA)),
        threshold_rel=REGIME_THRESHOLD_REL,
        smooth_sigma=max(1.0, radius / RADIUS_PER_SIGMA),
    )["count"]
    crowding = n_pk / max(n_cc, 1)
    return {"radius": radius, "snr": snr, "crowding": float(crowding), "n_cc": n_cc}


def choose_count_method(channel: np.ndarray, **seg) -> dict:
    """Pick a counting method for an intensity channel from its image regime.

    Transparent heuristic (returns its reasoning; always overridable):

    - low SNR (< `LOW_SNR`): ``"cc"`` — connected-component counting is the most
      noise-robust; peak finding would count noise.
    - crowded touching cells (crowding = peaks / connected components > `CROWDED`)
      with a clean image (SNR >= `CLEAN_SNR`): ``"maxima"`` (with auto
      ``smooth_sigma`` ≈ radius/1.5) — peaks separate overlapping cells that CC merges.
    - otherwise (well-separated): ``"cc"`` — exact and robust.

    NOTE: ``"maxima"`` re-implements the Find Maxima step of the published Kerkhoff
    Fiji Live/Dead macro with that macro's parameters (``min_distance`` 3,
    ``threshold_rel`` 0.15); it is not validated against the macro itself. Its
    agreement with ground truth is in
    benchmarks/results/b_viability_external_agreement.csv. ``auto`` only escalates
    to it in the clean + crowded regime above, so to reproduce those counts pass
    ``method="maxima"`` explicitly to ``live_dead_by_count``.

    Returns dict: method, smooth_sigma, and the estimated regime metrics (radius,
    snr, crowding: NaN when no cell is segmented; n_cc). The radius and
    smooth_sigma are in voxels (isotropic voxels assumed).
    """
    r = _estimate_regime(channel, **seg)
    # Conservative: default to the noise-robust method (CC), escalate to maxima
    # ONLY when the image is clearly clean AND clearly crowded. Crowding and noise
    # both inflate peak counts and are not cleanly separable from image statistics,
    # so we bias toward the safe method and avoid firing maxima on noisy data.
    if r["n_cc"] == 0:
        method, reason = "cc", "no cells detected"
    elif r["snr"] >= CLEAN_SNR and r["crowding"] > CROWDED:
        method, reason = (
            "maxima",
            (f"clean (SNR {r['snr']:.1f}) and crowded (peaks/CC {r['crowding']:.2f})"),
        )
    elif r["snr"] < LOW_SNR:
        method, reason = "cc", f"low SNR ({r['snr']:.1f}) — CC is noise-robust"
    else:
        method, reason = (
            "cc",
            (f"defaulting to robust CC (SNR {r['snr']:.1f}, crowding {r['crowding']:.2f})"),
        )
    smooth = r["radius"] / RADIUS_PER_SIGMA if _COUNTERS[method].smoothed else 0.0
    return {"method": method, "smooth_sigma": smooth, "reason": reason, **r}


def _count_record() -> dict:
    """The flat `live_dead_by_count` result with nothing computed yet (NaN, or "" for text)."""
    nan = float("nan")
    out: dict = {
        "n_live": nan,
        "n_dead": nan,
        "viability": nan,
        "consensus": nan,
        "spread": nan,
        "method_live": "",
        "method_dead": "",
        "reason_live": "",
        "reason_dead": "",
    }
    for m in COUNT_METHODS:
        out.update({f"n_live_{m}": nan, f"n_dead_{m}": nan, f"viability_{m}": nan})
    return out


def _count_every_method(live: np.ndarray, dead: np.ndarray, count_one, rseg: dict) -> dict:
    """``method="all"`` of `live_dead_by_count`: every count method on both channels,
    their median consensus and spread, and the deprecated nested ``by_method``."""
    out = _count_record()
    # auto-smooth the smoothed methods from each channel's estimated radius
    # (no cells -> radius NaN -> no smoothing)
    sig_l = float(np.nan_to_num(choose_count_method(live, **rseg)["radius"])) / RADIUS_PER_SIGMA
    sig_d = float(np.nan_to_num(choose_count_method(dead, **rseg)["radius"])) / RADIUS_PER_SIGMA
    for m in COUNT_METHODS:
        smoothed = _COUNTERS[m].smoothed
        nl = count_one(live, m, sig_l if smoothed else 0.0)
        nd = count_one(dead, m, sig_d if smoothed else 0.0)
        out.update({f"n_live_{m}": nl, f"n_dead_{m}": nd, f"viability_{m}": live_ratio(nl, nd)})
    vals = [
        out[f"viability_{m}"]
        for m in COUNT_METHODS
        if out[f"viability_{m}"] == out[f"viability_{m}"]
    ]
    if vals:
        out["consensus"] = out["viability"] = float(np.median(vals))
    if len(vals) > 1:  # one viability has no spread
        out["spread"] = float(max(vals) - min(vals))
    # deprecated v0.8.2 nested form of the per-method keys
    out["by_method"] = {
        m: {k: out[f"{k}_{m}"] for k in ("n_live", "n_dead", "viability")} for m in COUNT_METHODS
    }
    return out


def live_dead_by_count(
    live: np.ndarray,
    dead: np.ndarray,
    *,
    method: str = "auto",
    min_distance: int = 3,
    threshold_rel: float = 0.15,
    smooth_sigma: float = 0.0,
    threshold: str = DEFAULT_SEG_METHOD,
    **seg,
) -> dict:
    """Viability from per-object COUNTS rather than area — robust to crowding.

    ``live_dead_fractions`` reports area-based viability, which under-reports in
    dense fields where cells overlap (merged blobs). Counting objects tracks the
    true live/(live+dead) cell ratio even when cells touch. Counting modes:

    - ``method="auto"`` (default) — picks per-channel via :func:`choose_count_method`
      (transparent regime heuristic); the chosen method + reasoning are returned.
    - ``method="maxima"`` — intensity peak counting (`objects.count_local_maxima`,
      a height floor at a fraction of the image maximum, not a prominence); best for
      crowded single-peak cells. Over-counts flat/noisy cells unless ``smooth_sigma``.
    - ``method="watershed"`` — distance-transform watershed split of the mask.
    - ``method="cc"`` — connected components; exact for separated cells, noise-robust.
    - ``method="all"`` — run every mode (they are cheap) and return each viability
      plus a ``consensus`` (median) and ``spread`` (max−min). Large spread flags a
      hard regime where the counting choice matters. NOTE: without ground truth you
      cannot know which single method is "best" for a given image, so ``all`` reports
      a consensus and a disagreement flag rather than silently picking one.

    ``min_distance`` and ``smooth_sigma`` are in voxels (pixels in 2D) on every
    axis, so they assume isotropic voxels.

    ``threshold`` is the `segment.binarize` method the counts segment with (``method``
    is the counting mode); other ``seg`` keys (``min_size``, ``threshold_scale``)
    pass through to it.

    No single mode is universally best. Every mode returns the same flat keys
    (what a mode does not compute is NaN, or "" for text):

    - ``n_live``, ``n_dead``, ``viability`` — the counts and their ratio (NaN
      counts for ``all``: there is no single count; its viability is the consensus);
    - ``consensus`` — median viability over the methods run (equals ``viability``);
      ``spread`` — max − min of those viabilities, NaN when fewer than two methods
      gave one (a single method ran, or ``all`` found cells with only one);
    - ``method_live``, ``method_dead``, ``reason_live``, ``reason_dead`` — the
      method used on each channel and (``auto``) why ("" for ``all``);
    - ``n_live_<m>``, ``n_dead_<m>``, ``viability_<m>`` for every ``m`` in
      `COUNT_METHODS` — the per-method counts that were computed.

    Deprecated: ``method="all"`` also returns ``by_method`` (``{m: {n_live, n_dead,
    viability}}``), the v0.8.2 nested form of the per-method keys; it will be
    removed in a later release, so read ``n_live_<m>`` etc. instead. Reading it
    gives no DeprecationWarning (a plain dict key cannot warn when read).
    """
    methods = ("auto", "all", *COUNT_METHODS)
    if method not in methods:
        raise ValueError(f"unknown method {method!r}; use one of {methods}")
    cseg = {"min_size": COUNT_MIN_SIZE, **seg, "method": threshold}
    rseg = {**seg, "method": threshold}  # regime estimate, same segmentation

    def _count_one(ch, m, sig):
        return _COUNTERS[m].count(ch, sig, cseg, min_distance, threshold_rel)

    if method == "all":
        return _count_every_method(live, dead, _count_one, rseg)

    out = _count_record()

    def _count(ch):
        m, sig, reason = method, smooth_sigma, ""
        if method == "auto":
            info = choose_count_method(ch, **rseg)
            m, sig, reason = info["method"], info["smooth_sigma"], info["reason"]
        return _count_one(ch, m, sig), m, reason

    n_live, ml, rl = _count(live)
    n_dead, md, rd = _count(dead)
    v = live_ratio(n_live, n_dead)
    out.update(
        {
            "n_live": n_live,
            "n_dead": n_dead,
            "viability": v,
            "consensus": v,
            "method_live": ml,
            "method_dead": md,
            "reason_live": rl,
            "reason_dead": rd,
            f"n_live_{ml}": n_live,
            f"n_dead_{md}": n_dead,
        }
    )
    if ml == md:
        out[f"viability_{ml}"] = v
    return out


def viability_depth_profile(
    live: np.ndarray,
    dead: np.ndarray | None = None,
    z_axis: int = 0,
    *,
    live_mask: np.ndarray | None = None,
    dead_mask: np.ndarray | None = None,
    **seg,
) -> dict:
    """Per-z live (and dead) area fraction — the depth-resolved viability.

    Returns dict of 1D arrays (length = n z slices): live_by_z, dead_by_z and
    viability_by_z (the last two all NaN without a dead channel or mask).
    """
    lm = _mask(live, live_mask, **seg)
    axes = tuple(a for a in range(lm.ndim) if a != z_axis)
    live_by_z = lm.mean(axis=axes)
    no_dead = np.full(live_by_z.shape, np.nan)
    out = {"live_by_z": live_by_z, "dead_by_z": no_dead, "viability_by_z": no_dead.copy()}
    if dead is not None or dead_mask is not None:
        dm = _mask(dead, dead_mask, **seg)
        dead_by_z = dm.mean(axis=axes)
        la = lm.sum(axis=axes).astype(float)
        da = dm.sum(axis=axes).astype(float)
        denom = la + da
        out["dead_by_z"] = dead_by_z
        out["viability_by_z"] = np.divide(la, denom, out=np.full_like(la, np.nan), where=denom > 0)
    return out


def viability_2d_vs_3d(
    live: np.ndarray,
    z_axis: int = 0,
    *,
    live_mask: np.ndarray | None = None,
    **seg,
) -> dict:
    """Compare live fraction from a single mid plane, a MIP, and the full 3D
    volume — quantifying how much a 2D readout overestimates viability.

    Returns dict: live_fraction_midplane, live_fraction_mip, live_fraction_3d,
    overestimate_mip = mip / 3d (>1 means the 2D MIP overstates coverage), and
    overestimate_midplane = midplane / 3d.

    The 3D mask uses `segment.binarize`'s ``opening`` (``seg``): the default
    ``"ball"`` erases cells thinner than 3 z-slices, so a monolayer reads a live
    fraction of 0 (overestimate NaN; with a RuntimeWarning when ``"disk"`` would
    keep cells) or one biased to its thickest cells; pass ``opening="disk"`` for
    thin samples.

    Projection: the MIP here is the maximum projection of the 3D live MASK (the
    volume is segmented once in 3D, then projected), so a voxel counts as covered
    if any slice above it is foreground. Segmenting the intensity MIP instead
    (``live_dead_fractions(live.max(z_axis))``, as `viability_batch`'s
    ``live_cov_frac`` does) thresholds the projected intensities and generally
    gives a different fraction.
    """
    lm = _mask(live, live_mask, **seg)
    if not lm.any() and live_mask is None and seg.get("opening", "ball") == "ball":
        if _mask(live, **{**seg, "opening": "disk"}).any():
            warnings.warn(
                "viability_2d_vs_3d: the 'ball' opening erased every live cell from the "
                '3D mask (cells thinner than 3 z-slices); pass opening="disk" for a '
                "monolayer. The 3D fractions are 0 and the overestimates NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
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

    ``reference`` is ``"max"``, ``"first"`` (the first slice) or ``"mean"`` (the
    mean of the slice means). Returns a new float array; the input is not modified.
    """
    references = ("max", "first", "mean")
    if reference not in references:
        raise ValueError(f"reference must be one of {references}, got {reference!r}")
    vol = volume.astype(np.float32)
    axes = tuple(a for a in range(vol.ndim) if a != z_axis)
    slice_means: np.ndarray = vol.mean(axis=axes)
    if reference == "max":
        ref = slice_means.max()
    elif reference == "first":
        ref = slice_means.flat[0]
    else:  # "mean"
        ref = slice_means.mean()
    scale = np.divide(ref, slice_means, out=np.ones_like(slice_means), where=slice_means > 0)
    shape = [1] * vol.ndim
    shape[z_axis] = -1
    return vol * scale.reshape(shape)


__all__ = [
    "live_dead_channels",
    "live_ratio",
    "live_dead_fractions",
    "live_dead_by_count",
    "choose_count_method",
    "viability_depth_profile",
    "viability_2d_vs_3d",
    "attenuation_correct",
]
