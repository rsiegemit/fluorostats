"""Tests for fluorostats.viability."""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats import viability
from fluorostats.viability import (
    live_dead_fractions,
    viability_depth_profile,
    viability_2d_vs_3d,
    attenuation_correct,
    live_dead_by_count,
)


def _two_channel_dots(n_live, n_dead, size=200, sep=9):
    """Synthetic 2-channel image: n_live green + n_dead red gaussian dots."""
    live = np.zeros((size, size), np.float32)
    dead = np.zeros((size, size), np.float32)

    def place(img, n):
        g = int(np.ceil(np.sqrt(n)))
        xs = np.linspace(sep, size - sep, g)
        pts = [(y, x) for y in xs for x in xs][:n]
        for y, x in pts:
            yy, xx = int(y), int(x)
            img[yy - 1 : yy + 2, xx - 1 : xx + 2] = 255.0

    place(live, n_live)
    place(dead, n_dead)
    return live, dead


def test_live_dead_by_count_maxima_recovers_true_viability():
    # 90 live / 10 dead -> true viability 0.9; peak counting should recover it
    live, dead = _two_channel_dots(90, 10)
    r = live_dead_by_count(live, dead, method="maxima", min_distance=3)
    assert r["n_live"] > r["n_dead"]
    assert r["viability"] == pytest.approx(0.9, abs=0.05)


def test_live_dead_by_count_bad_method_raises():
    live, dead = _two_channel_dots(5, 5)
    with pytest.raises(ValueError):
        live_dead_by_count(live, dead, method="nonsense")


def _stack_with_depth_gradient(nz=20, size=32):
    """Live signal present only in the top half — a depth-death gradient."""
    live = np.zeros((nz, size, size), np.float32)
    live[: nz // 2] = 100.0  # shallow slices bright, deep slices empty
    return live


def test_live_dead_fractions_basic():
    live = np.zeros((4, 10, 10), np.float32)
    live[:, :5, :] = 100
    dead = np.zeros((4, 10, 10), np.float32)
    dead[:, 5:, :] = 100
    r = live_dead_fractions(live, dead)
    assert r["live_fraction"] == pytest.approx(0.5, abs=0.01)
    assert r["viability"] == pytest.approx(0.5, abs=0.05)


def test_depth_profile_reveals_gradient():
    live = _stack_with_depth_gradient()
    prof = viability_depth_profile(live)["live_by_z"]
    assert prof[:10].mean() > 0.5  # shallow slices have signal
    assert prof[10:].mean() == pytest.approx(0.0, abs=1e-6)  # deep empty


def test_depth_profile_with_dead_channel_returns_viability_by_z():
    live = np.zeros((4, 10, 10), np.float32)
    live[:, :5, :] = 100
    dead = np.zeros((4, 10, 10), np.float32)
    dead[:, 5:, :] = 100
    out = viability_depth_profile(live, dead)
    assert "dead_by_z" in out and "viability_by_z" in out
    # Equal live/dead area per slice -> viability ~0.5 on every slice
    assert np.allclose(out["viability_by_z"], 0.5, atol=0.05)


def test_2d_overestimates_vs_3d():
    live = _stack_with_depth_gradient()
    r = viability_2d_vs_3d(live)
    # MIP collapses depth -> looks fully covered; 3D fraction is ~half
    assert r["live_fraction_mip"] > r["live_fraction_3d"]
    assert r["overestimate_mip"] > 1.5


def test_attenuation_correct_equalises_slice_means():
    vol = np.ones((5, 8, 8), np.float32)
    for z in range(5):
        vol[z] *= 1.0 - 0.15 * z  # simulated depth decay
    corr = attenuation_correct(vol)
    means = corr.reshape(5, -1).mean(axis=1)
    assert np.allclose(means, means[0], rtol=1e-4)  # all slices equalised
    assert not np.shares_memory(corr, vol)


def test_accepts_premasked_input():
    mask = np.zeros((4, 10, 10), bool)
    mask[:, :5, :] = True
    r = live_dead_fractions(None, live_mask=mask)
    assert r["live_fraction"] == pytest.approx(0.5, abs=0.01)


def _two_channel_stack():
    live = np.zeros((4, 40, 40), np.float32)
    dead = np.zeros((4, 40, 40), np.float32)
    # a handful of bright blobs per channel
    for y, x in [(10, 10), (10, 30), (30, 10)]:
        live[:, y - 3 : y + 3, x - 3 : x + 3] = 200.0
    dead[:, 30 - 3 : 30 + 3, 30 - 3 : 30 + 3] = 200.0
    return live, dead


def test_choose_count_method_returns_regime():
    live, _ = _two_channel_stack()
    out = viability.choose_count_method(live)
    assert out["method"] in {"cc", "maxima"}
    assert {"snr", "crowding", "n_cc", "smooth_sigma", "reason"} <= set(out)


def test_choose_count_method_no_cells():
    out = viability.choose_count_method(np.zeros((3, 20, 20), np.float32))
    assert out["method"] == "cc"
    assert out["n_cc"] == 0


def test_live_dead_by_count_method_all_reports_consensus():
    live, dead = _two_channel_stack()
    out = viability.live_dead_by_count(live, dead, method="all")
    per = [out[f"viability_{m}"] for m in viability.COUNT_METHODS]
    per = [v for v in per if v == v]  # methods that found cells
    assert out["consensus"] == np.median(per)
    assert out["spread"] == max(per) - min(per)
    assert out["viability"] == out["consensus"]
    assert out["method_live"] == out["method_dead"] == ""


def test_method_all_spread_is_nan_when_one_method_gives_a_viability(monkeypatch):
    # Regression: with one finite per-method viability the spread was 0.0 ("all
    # methods agree"), not NaN like the other modes' "one method ran".
    live, dead = _two_channel_stack()
    for m in viability.COUNT_METHODS[1:]:
        monkeypatch.setitem(viability._COUNTERS, m, lambda *args: 0)  # finds nothing
    out = viability.live_dead_by_count(live, dead, method="all")
    first = viability.COUNT_METHODS[0]
    assert out["consensus"] == out[f"viability_{first}"] and np.isnan(out["spread"])


def test_live_dead_by_count_auto_reports_reasoning():
    live, dead = _two_channel_stack()
    out = viability.live_dead_by_count(live, dead, method="auto")
    assert "method_live" in out and "reason_live" in out


def test_attenuation_correct_first_reference():
    vol = np.ones((4, 8, 8), np.float32)
    vol[0] *= 4.0  # bright top slice
    out = viability.attenuation_correct(vol, reference="first")
    # every slice scaled so its mean matches slice 0's mean
    means = out.mean(axis=(1, 2))
    assert np.allclose(means, means[0])


def test_attenuation_correct_mean_reference():
    vol = np.ones((3, 8, 8), np.float32)
    vol[1] *= 2.0
    out = viability.attenuation_correct(vol, reference="mean")
    assert out.shape == vol.shape


def test_viability_choose_count_method_regimes():
    size, z = 96, 3
    yy, xx = np.mgrid[0:size, 0:size]
    # clean + crowded -> maxima
    clean = np.full((z, size, size), 5.0, np.float32)
    for cy in range(12, size - 12, 8):
        for cx in range(12, size - 12, 8):
            clean[z // 2] += 5000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0**2))
    assert viability.choose_count_method(clean, method="otsu", min_size=3)["method"] in {
        "maxima",
        "cc",
    }
    # low SNR -> cc
    noisy = np.random.default_rng(0).normal(100, 90, (z, size, size)).astype(np.float32)
    noisy[z // 2, 40:56, 40:56] += 120
    assert viability.choose_count_method(noisy, method="otsu", min_size=3)["method"] == "cc"
    # no cells -> cc
    assert (
        viability.choose_count_method(
            np.zeros((z, size, size), np.float32), method="otsu", min_size=3
        )["method"]
        == "cc"
    )


# ---------------------------------------------------------------------------
# viability.py:112 — choose_count_method "maxima" branch (clean + crowded)
# ---------------------------------------------------------------------------
def test_choose_count_method_maxima_branch():
    # Bright continuous bars on a near-zero, low-noise background: each bar is a
    # single connected component (CC merges the ridge) but carries periodic
    # bright bumps that the peak finder resolves separately -> high SNR and
    # peaks/CC > 1.5, which selects "maxima".
    h = w = 160
    img = np.zeros((h, w), dtype=float)
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(2)
    for row in range(20, h - 20, 30):
        img += 3000.0 * np.exp(-((yy - row) ** 2) / (2 * 2.0**2))  # continuous bar
        for cx in range(15, w - 15, 12):
            img += 5000.0 * np.exp(-(((yy - row) ** 2 + (xx - cx) ** 2) / (2 * 1.2**2)))
    img += rng.normal(0, 0.5, size=img.shape)  # tiny noise -> high SNR
    img = np.clip(img, 0, None)
    out = viability.choose_count_method(img)
    assert out["method"] == "maxima"
    assert out["snr"] >= 8.0
    assert out["crowding"] > 1.5


@pytest.mark.parametrize("n_live,n_dead,expected", [(3, 1, 0.75), (0, 4, 0.0), (5, 0, 1.0)])
def test_live_ratio(n_live, n_dead, expected):
    assert viability.live_ratio(n_live, n_dead) == expected


def test_live_ratio_is_nan_with_nothing_to_count():
    assert np.isnan(viability.live_ratio(0, 0))


def test_attenuation_correct_rejects_unknown_reference():
    with pytest.raises(ValueError, match="'max', 'first', 'mean'"):
        attenuation_correct(np.ones((3, 4, 4)), reference="median")


def test_count_method_error_lists_the_registry():
    live = np.zeros((20, 20))
    with pytest.raises(ValueError, match="'cc', 'watershed', 'maxima'"):
        live_dead_by_count(live, live, method="bogus")
    assert viability.COUNT_METHODS == ("cc", "watershed", "maxima")


def test_count_threshold_method_is_passed_to_segmentation():
    """`threshold` picks the binarize method for counting; `method` stays the count mode."""
    from fluorostats.objects import label_3d
    from fluorostats.segment import binarize

    live, dead = _two_channel_dots(12, 5)
    out = live_dead_by_count(live, dead, method="cc", threshold="li")
    assert out["n_live"] == label_3d(binarize(live, method="li", min_size=5), min_size=5)[1]
    assert out["n_dead"] == label_3d(binarize(dead, method="li", min_size=5), min_size=5)[1]


def test_count_all_has_the_same_top_level_keys():
    """method="all" returns n_live/n_dead/viability like every other mode (NaN counts:
    there is no single count), plus its per-method detail."""
    live, dead = _two_channel_dots(12, 5)
    out = live_dead_by_count(live, dead, method="all")
    assert {"n_live", "n_dead", "viability"} <= set(out)
    assert np.isnan(out["n_live"]) and np.isnan(out["n_dead"])
    assert out["viability"] == out["consensus"]


def test_regime_without_cells_is_nan_not_zero():
    # Nothing segmented: SNR, crowding and radius were never measured -> NaN.
    out = viability.choose_count_method(np.zeros((16, 16), np.float32))
    assert out["n_cc"] == 0 and out["method"] == "cc"
    assert np.isnan(out["snr"]) and np.isnan(out["crowding"]) and np.isnan(out["radius"])


_FLAT_KEYS = {
    "n_live",
    "n_dead",
    "viability",
    "consensus",
    "spread",
    "method_live",
    "method_dead",
    "reason_live",
    "reason_dead",
    *(f"{k}_{m}" for m in viability.COUNT_METHODS for k in ("n_live", "n_dead", "viability")),
}


@pytest.mark.parametrize("method", ["all", "auto", *viability.COUNT_METHODS])
def test_live_dead_by_count_has_one_flat_contract(method):
    # Every mode returns the same flat keys of scalars/strings (MOD-5); what a mode
    # did not compute is NaN or "".
    live, dead = _two_channel_dots(12, 5)
    out = live_dead_by_count(live, dead, method=method)
    by_method = out.pop("by_method", None)  # deprecated, "all" only
    assert (by_method is not None) == (method == "all")
    assert set(out) == _FLAT_KEYS
    assert all(isinstance(v, (int, float, str, np.integer, np.floating)) for v in out.values())
    if method in viability.COUNT_METHODS:
        assert (out["method_live"], out["method_dead"]) == (method, method)
        assert out[f"n_live_{method}"] == out["n_live"]
        assert out[f"viability_{method}"] == out["viability"] == out["consensus"]
        assert np.isnan(out["spread"])  # one method: no spread
        others = [m for m in viability.COUNT_METHODS if m != method]
        assert all(np.isnan(out[f"n_live_{m}"]) for m in others)


def test_live_dead_by_count_auto_fills_the_chosen_methods_columns():
    live, dead = _two_channel_dots(12, 5)
    out = live_dead_by_count(live, dead, method="auto")
    ml, md = out["method_live"], out["method_dead"]
    assert out[f"n_live_{ml}"] == out["n_live"] and out[f"n_dead_{md}"] == out["n_dead"]
    assert out["reason_live"] and out["reason_dead"]
    assert out["consensus"] == out["viability"] and np.isnan(out["spread"])


@pytest.mark.parametrize(
    "names,expected",
    [
        (["Calcein AM", "PI"], (0, 1)),
        (["Propidium iodide", "Alexa Fluor 488"], (1, 0)),
        (["DAPI", "FITC", "EthD-1"], (1, 2)),
        (["GFP", "ethidium homodimer"], (0, 1)),
    ],
)
def test_live_dead_channels_from_dye_names(names, expected):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert viability.live_dead_channels(names) == expected


def test_live_dead_channels_explicit_and_fallback():
    assert viability.live_dead_channels(["PI", "Calcein"], live=0, dead=1) == (0, 1)
    with pytest.warns(RuntimeWarning, match="no channel name identifies the live"):
        assert viability.live_dead_channels(["DAPI", "AF647"], dead=1) == (0, 1)
    with pytest.warns(RuntimeWarning, match="dead channel"):
        assert viability.live_dead_channels(["Ch1", "Ch2"], live=0) == (0, 1)
    with pytest.raises(ValueError, match="both channel 1"):
        viability.live_dead_channels(["Ch1", "Calcein"], live=None, dead=None)
    with pytest.raises(ValueError, match="has 2 channels"):
        viability.live_dead_channels(["a", "b"], live=0, dead=2)


def test_count_all_keeps_the_deprecated_by_method():
    # v0.8.2 callers read out["by_method"][m]: kept (deprecated) with the same values
    # as the flat per-method keys.
    live, dead = _two_channel_dots(12, 5)
    out = live_dead_by_count(live, dead, method="all")
    assert set(out["by_method"]) == set(viability.COUNT_METHODS)
    for m, d in out["by_method"].items():
        assert set(d) == {"n_live", "n_dead", "viability"}
        assert all(d[k] == out[f"{k}_{m}"] for k in d)
