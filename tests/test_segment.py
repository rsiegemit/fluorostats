"""Tests for segmentation: known-geometry synthetic volumes."""

import numpy as np
import pytest

from fluorostats import segment
from fluorostats.segment import binarize, choose_threshold_method, THRESHOLD_METHODS
from fluorostats.metrics_3d import volume_fraction


def test_threshold_family_all_run():
    vol = np.zeros((8, 32, 32), np.float64)
    vol[2:6, 8:24, 8:24] = 500.0
    for m in THRESHOLD_METHODS:
        assert binarize(vol, method=m, min_size=5).sum() > 0


def test_auto_selects_and_consensus_runs():
    vol = np.zeros((8, 32, 32), np.float64)
    vol[2:6, 8:24, 8:24] = 500.0
    info = choose_threshold_method(vol)
    assert info["method"] in THRESHOLD_METHODS
    assert "skew" in info and "fractions" in info
    assert binarize(vol, method="auto", min_size=5).sum() > 0
    assert binarize(vol, method="consensus", min_size=5).sum() > 0


def test_auto_rule_is_self_consistent():
    # auto's returned method must match its own documented rule, whatever the image
    rng = np.random.default_rng(0)
    for vol in [
        rng.normal(50, 10, (4, 64, 64)),
        np.where(rng.random((4, 64, 64)) < 0.01, 200.0, 3.0),
    ]:
        info = choose_threshold_method(vol)
        of, lf = info["fractions"]["otsu"], info["fractions"]["li"]
        expected = "li" if (of < 0.01 and lf > 3 * of) else "otsu"
        assert info["method"] == expected


class TestBinarize3D:
    def test_bright_block_detected(self):
        """A bright block in a dark volume should be segmented."""
        vol = np.zeros((10, 64, 64), dtype=np.float64)
        vol[3:7, 20:44, 20:44] = 1000.0
        # Add background noise
        rng = np.random.default_rng(42)
        vol += rng.normal(10, 5, vol.shape)

        mask = binarize(vol, method="otsu", min_size=16)

        # The block should be mostly segmented
        block_mask = mask[3:7, 20:44, 20:44]
        assert block_mask.mean() > 0.9, f"Block coverage too low: {block_mask.mean()}"

        # Background should be mostly empty
        bg_mask = mask.copy()
        bg_mask[3:7, 20:44, 20:44] = False
        assert bg_mask.mean() < 0.05, f"Background noise too high: {bg_mask.mean()}"

    def test_volume_fraction_accuracy(self):
        """Volume fraction of a known-volume object should be accurate."""
        vol = np.zeros((20, 64, 64), dtype=np.float64)
        # Sphere-ish block: 4*24*24 = 2304 voxels out of 20*64*64 = 81920
        vol[8:12, 20:44, 20:44] = 2000.0
        rng = np.random.default_rng(42)
        vol += rng.normal(10, 5, vol.shape)

        mask = binarize(vol, method="otsu", min_size=16)
        vf = volume_fraction(mask)

        expected = (4 * 24 * 24) / (20 * 64 * 64)
        assert abs(vf - expected) < 0.01, f"VF {vf:.4f} too far from expected {expected:.4f}"


class TestBinarize2D:
    def test_bright_region_2d(self):
        """A bright square in a 2D image should be segmented."""
        img = np.zeros((128, 128), dtype=np.float64)
        img[30:90, 30:90] = 500.0
        rng = np.random.default_rng(42)
        img += rng.normal(10, 5, img.shape)

        mask = binarize(img, method="otsu", min_size=16)

        region_mask = mask[30:90, 30:90]
        assert region_mask.mean() > 0.95

    def test_li_threshold(self):
        """Li threshold should also work."""
        img = np.zeros((64, 64), dtype=np.float64)
        img[10:50, 10:50] = 300.0
        rng = np.random.default_rng(42)
        img += rng.normal(10, 5, img.shape)

        mask = binarize(img, method="li", min_size=16)
        assert mask.sum() > 0


def test_choose_threshold_selects_li_for_sparse_signal():
    """A few saturating pixels drive Otsu's threshold sky-high so it keeps
    <1% of the image; the dim diffuse block is only recovered by Li."""
    img = np.zeros((128, 128), dtype=np.float64)
    img[:2, :2] = 4000.0  # 4 extremely bright px -> Otsu split point
    img[50:70, 50:70] = 20.0  # dim diffuse foreground Li catches
    out = segment.choose_threshold_method(img)
    assert out["method"] == "li"
    assert "fractions" in out and "otsu" in out["fractions"]


def test_binarize_consensus_path():
    img = np.zeros((32, 32), dtype=np.float64)
    img[8:24, 8:24] = 200.0
    mask = segment.binarize(img, method="consensus", min_size=0)
    assert mask.dtype == bool
    assert mask[16, 16]


def test_binarize_auto_path_runs():
    img = np.zeros((32, 32), dtype=np.float64)
    img[8:24, 8:24] = 200.0
    mask = segment.binarize(img, method="auto", min_size=0)
    assert mask.any()


def test_binarize_flat_image_is_all_false():
    """A perfectly flat image has no signal above threshold."""
    flat = np.full((16, 16), 7.0, dtype=np.float64)
    mask = segment.binarize(flat, method="otsu", min_size=0)
    assert not mask.any()


def test_binarize_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown threshold method"):
        segment.binarize(np.zeros((8, 8)), method="nope")


def test_segment_constant_image_paths():
    flat = np.zeros((16, 16), np.float32)
    # isodata raises on a constant image: both meta-methods announce it and go on
    with pytest.warns(RuntimeWarning, match="isodata failed"):
        choice = segment.choose_threshold_method(flat)
    assert choice["method"] == "otsu" and np.isnan(choice["fractions"]["isodata"])
    with pytest.warns(RuntimeWarning, match="isodata failed"):
        out = segment.binarize(flat, method="consensus")
    assert out.shape == flat.shape and not out.any()


def test_consensus_announces_a_failed_method(monkeypatch):
    from fluorostats import segment

    def boom(a):
        raise RuntimeError("no histogram")

    monkeypatch.setitem(segment.THRESHOLD_METHODS, "yen", boom)
    img = np.zeros((32, 32))
    img[8:24, 8:24] = 100.0
    with pytest.warns(RuntimeWarning, match="yen failed"):
        mask = segment.binarize(img, method="consensus", min_size=1)
    assert mask[16, 16] and not mask[0, 0]  # the other five still decide


def test_min_size_keeps_objects_of_exactly_that_size():
    # "Smaller objects are removed": an object of exactly min_size pixels stays, on
    # every scikit-image version, matching objects.label_3d's min_size. Sizes are
    # measured after binarize's salt-noise opening, which reshapes small objects.
    from fluorostats.objects import label_3d

    img = np.zeros((40, 40))
    img[5:12, 5:12] = 100.0
    img[20:32, 20:32] = 100.0
    size = int(binarize(img, method="otsu", min_size=0)[:20, :20].sum())  # small object
    kept = binarize(img, method="otsu", min_size=size)
    assert kept[8, 8] and kept[25, 25]
    assert not binarize(img, method="otsu", min_size=size + 1)[8, 8]
    assert label_3d(kept[None], min_size=size)[1] == 2


@pytest.mark.parametrize("bg_radius", [0, 3])
def test_segment_channel_is_denoise_background_binarize(bg_radius):
    from fluorostats import preprocess

    rng = np.random.default_rng(4)
    img = rng.normal(10, 2, (4, 24, 24))
    img[:, 6:14, 6:14] += 25
    out, mask = segment.segment_channel(
        img, method="li", min_size=8, sigma=1.2, bg_radius=bg_radius, threshold_scale=0.9
    )
    ref = preprocess.denoise(img, sigma=1.2)
    if bg_radius:
        ref = preprocess.background_subtract(ref, radius=bg_radius)
    np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(mask, binarize(ref, method="li", min_size=8, threshold_scale=0.9))


@pytest.mark.parametrize("n,kept", [(4, True), (3, False)])
def test_remove_small_keeps_objects_of_exactly_min_size(n, kept):
    m = np.zeros((10, 10), bool)
    m[0, :n] = True
    assert segment.remove_small(m, 4).any() == kept
    assert segment.remove_small(m, 0) is m


def _sheet(thickness):
    rng = np.random.default_rng(0)
    vol = rng.normal(10, 1, (10, 48, 48))
    vol[4 : 4 + thickness] += 50
    return vol


def test_ball_opening_erases_a_two_slice_sheet_disk_keeps_it():
    vol = _sheet(2)
    n_true = 2 * 48 * 48
    assert binarize(vol, min_size=0).sum() < 0.2 * n_true  # default "ball"
    for opening in ("disk", "none"):
        assert binarize(vol, min_size=0, opening=opening).sum() >= 0.99 * n_true
    # a 3-slice sheet survives the default
    assert binarize(_sheet(3), min_size=0).sum() >= 0.9 * 3 * 48 * 48


def test_opening_options_and_2d():
    img = np.zeros((20, 20))
    img[5:15, 5:15] = 100.0
    img[0, 0] = 100.0  # salt pixel
    assert not binarize(img, min_size=0)[0, 0]
    assert not binarize(img, min_size=0, opening="disk")[0, 0]
    assert binarize(img, min_size=0, opening="none")[0, 0]
    with pytest.raises(ValueError, match="Unknown opening"):
        binarize(img, opening="cube")
    _, mask = segment.segment_channel(_sheet(2), sigma=0.0, min_size=0, opening="disk")
    assert mask.sum() >= 0.99 * 2 * 48 * 48


def test_choose_threshold_method_warns_when_a_method_fails():
    img = np.zeros((8, 8))
    img[0, 0] = np.nan
    with pytest.warns(RuntimeWarning, match="otsu failed"):
        info = segment.choose_threshold_method(img)
    assert np.isnan(info["fractions"]["otsu"])


def test_remove_small_labels_keeps_touching_objects_apart():
    # Two touching labels: a mask-level filter would merge them into one object of
    # 6 and keep it; per label, the 2-pixel object is dropped and the other is 1.
    labels = np.zeros((3, 6), np.int32)
    labels[1, :2] = 5
    labels[1, 2:6] = 9
    out, n = segment.remove_small_labels(labels, 3)
    assert n == 1 and out.dtype == labels.dtype
    np.testing.assert_array_equal(out[1], [0, 0, 1, 1, 1, 1])
    assert segment.remove_small_labels(labels, 10)[1] == 0
