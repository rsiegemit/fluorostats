"""Regression tests for correctness bugs fixed in the release-hardening pass,
plus direct coverage of a few pure peripheral functions."""
import numpy as np
import pandas as pd
import pytest

from fluorostats import stats, objects, preprocess, metrics_2d, plots, qc


def test_stouffer_weights_align_after_nan_drop():
    """A NaN p-value must drop its weight too, not shift it onto a surviving p."""
    full = stats.stouffer_combine([0.01, 0.5], weights=[1.0, 1.0])
    with_nan = stats.stouffer_combine([0.01, np.nan, 0.5], weights=[1.0, 100.0, 1.0])
    assert with_nan["n"] == 2
    assert with_nan["z"] == pytest.approx(full["z"], rel=1e-9)


def test_object_volumes_ignore_label_gaps():
    """Non-contiguous label images must not yield spurious zero-volume objects."""
    labels = np.array([[0, 1, 1], [0, 3, 3]])           # label 2 missing
    vols = objects.object_volumes_voxels(labels)
    assert sorted(vols.tolist()) == [2, 2]
    assert (vols > 0).all()


def test_equivalent_diameters_no_zero_from_gaps():
    labels = np.array([[[0, 1, 1], [0, 3, 3]]])
    d = objects.equivalent_diameters_um(labels, (1.0, 1.0, 1.0))
    assert len(d) == 2 and (d > 0).all()


def test_auto_crop_leaves_borderless_image_untouched():
    """A full-frame (borderless) image must not be trimmed by `margin`."""
    img = np.random.default_rng(0).integers(0, 255, (50, 60)).astype(np.uint8)
    out, coords = preprocess.auto_crop(img, margin=5)
    assert out.shape == img.shape
    assert coords == (0, 50, 0, 60)


def test_auto_crop_trims_a_real_border():
    img = np.zeros((50, 60), np.uint8)
    img[10:40, 12:48] = 200                             # active interior, flat border
    out, _ = preprocess.auto_crop(img, margin=0)
    assert out.shape[0] < 50 and out.shape[1] < 60


def test_plots_many_conditions_no_color_collision(tmp_path):
    """>8 conditions previously reused Set2 colors; just assert all renderers run."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame([{"condition": f"C{c:02d}", "volume_fraction": float(rng.random())}
                       for c in range(10) for _ in range(3)])
    plots.bar_mean_sem(df, "volume_fraction", tmp_path / "bar.png")
    plots.boxplot_by_condition(df, "volume_fraction", tmp_path / "box.png")
    plots.summary_panel(df, ["volume_fraction"], tmp_path / "panel.png")
    assert (tmp_path / "bar.png").exists() and (tmp_path / "box.png").exists()


def test_coverage_metrics():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    m = metrics_2d.coverage_metrics(mask)
    assert m["area_fraction"] == pytest.approx(100 / 400)
    assert m["n_components"] == 1
    assert metrics_2d.coverage_metrics(np.zeros((5, 5), bool))["area_fraction"] == 0


def test_select_green_channel():
    arr = np.zeros((2, 8, 8), np.float32)
    arr[1] = 5.0
    g = preprocess.select_green_channel(arr, ["Red", "GFP"])
    assert g.shape == (8, 8) and g.mean() == pytest.approx(5.0)


def test_qc_overlay_writes_png(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (32, 32)).astype(np.uint8)
    mask = np.zeros((32, 32), bool)
    mask[8:24, 8:24] = True
    qc.overlay_2d(img, mask, tmp_path / "ov.png")
    assert (tmp_path / "ov.png").exists()
