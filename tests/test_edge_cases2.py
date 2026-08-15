"""Final coverage-closing tests for the last uncovered guard branches."""
import numpy as np
import pytest

from fluorostats import io, keyence, morphometry, objects, preprocess, skeleton, viability

tifffile = pytest.importorskip("tifffile")


# ---------------------------------------------------------------------------
# io.py:435 — OME channel-name append (valid Channel with a Name)
# ---------------------------------------------------------------------------
def test_ome_metadata_channel_name_append():
    ome_xml = (
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image><Pixels PhysicalSizeX="0.5" PhysicalSizeY="0.5" PhysicalSizeZ="2.0">'
        '<Channel Name="EGFP"/>'
        '<Channel Fluor="mCherry"/>'
        '</Pixels></Image></OME>'
    )
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    out = io._parse_ome_metadata(ome_xml, meta)
    assert out["channel_names"] == ["EGFP", "mCherry"]
    assert out["voxel_size_um"] == (2.0, 0.5, 0.5)


# ---------------------------------------------------------------------------
# keyence.py:139 — 2D single-channel CHF slice promoted to (1, Y, X)
# ---------------------------------------------------------------------------
def test_keyence_2d_single_channel_promotion(tmp_path):
    rng = np.random.default_rng(0)
    for zi in range(1, 4):
        plane = rng.integers(100, 3000, size=(24, 32)).astype(np.uint16)  # 2D
        tifffile.imwrite(tmp_path / f"Image _Z{zi:03d}_CHF.tif", plane)
    arr, meta = keyence.load_keyence_stack(tmp_path)
    # channel is None -> stacked to (C=1, Z, Y, X)
    assert arr.shape == (1, 3, 24, 32)
    assert meta["n_slices_loaded"] == 3


# ---------------------------------------------------------------------------
# morphometry.py:102 — depth_span early return when no slice qualifies
# ---------------------------------------------------------------------------
def test_depth_span_no_slice_above_cutoff():
    # A single non-zero slice: profile.max() > 0, but with the default
    # relative_threshold a lone spike still qualifies. Force the "above.size==0"
    # branch by making every slice's mean fall below the cutoff except via a
    # threshold that excludes all: use a volume whose profile is all equal but
    # nonzero and a relative_threshold > 1.0 so nothing meets the cutoff.
    vol = np.ones((4, 5, 5), dtype=float)
    out = morphometry.depth_span(vol, relative_threshold=1.5)
    assert out["span_slices"] == 0
    assert out["z_lo"] == 0 and out["z_hi"] == 0


# ---------------------------------------------------------------------------
# morphometry.py:165 — _gini all-zero (sum==0) guard
# ---------------------------------------------------------------------------
def test_morphometry_gini_all_zero():
    assert morphometry._gini(np.zeros(5)) == 0.0


# ---------------------------------------------------------------------------
# objects.py:90 — watershed_split size-filter drops every label
# ---------------------------------------------------------------------------
def test_watershed_split_all_labels_below_min_size():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True  # single-voxel object -> non-empty mask
    labels, n = objects.watershed_split(mask, min_size=1000)
    assert n == 0
    assert labels.max() == 0


# ---------------------------------------------------------------------------
# objects.py:248, 251, 261 — local _gini/_cv guards (empty / all-zero)
# ---------------------------------------------------------------------------
def test_objects_gini_cv_guards():
    assert np.isnan(objects._gini(np.array([])))       # 248: size == 0
    assert objects._gini(np.zeros(4)) == 0.0           # 251: sum == 0
    assert np.isnan(objects._cv(np.zeros(4)))          # 261: mean == 0


# ---------------------------------------------------------------------------
# preprocess.py:169 — auto_crop y1 <= y0 (active band thinner than 2*margin)
# ---------------------------------------------------------------------------
def test_auto_crop_active_band_thinner_than_margin():
    # Mostly-uniform image with a thin active band of two adjacent rows.
    # margin large enough that active[0]+margin >= active[-1]-margin+1.
    img = np.zeros((20, 20), dtype=float)
    img[10, :] = np.arange(20)          # one high-variance row
    img[11, :] = np.arange(20)[::-1]    # a second adjacent high-variance row
    out, box = preprocess.auto_crop(img, margin=5)
    assert box == (0, 20, 0, 20)
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# skeleton.py:65 — prune loop break when skeleton has < 3 voxels
# ---------------------------------------------------------------------------
def test_prune_skeleton_too_few_voxels():
    s = np.zeros((10, 10), dtype=bool)
    s[5, 5] = True
    s[5, 6] = True  # 2 voxels < 3 -> break immediately
    out = skeleton.prune_skeleton(s, min_branch_length_px=10.0)
    assert out.sum() == 2


# ---------------------------------------------------------------------------
# skeleton.py:78 — prune loop break when a skeleton has no short spurs
# ---------------------------------------------------------------------------
def test_prune_skeleton_no_spurs():
    line = np.zeros((3, 20), dtype=bool)
    line[1, :] = True  # a straight line: no type-1 spur branches
    out = skeleton.prune_skeleton(line, min_branch_length_px=100.0)
    assert out.sum() == 20


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
        img += 3000.0 * np.exp(-((yy - row) ** 2) / (2 * 2.0 ** 2))  # continuous bar
        for cx in range(15, w - 15, 12):
            img += 5000.0 * np.exp(-(((yy - row) ** 2 + (xx - cx) ** 2) / (2 * 1.2 ** 2)))
    img += rng.normal(0, 0.5, size=img.shape)  # tiny noise -> high SNR
    img = np.clip(img, 0, None)
    out = viability.choose_count_method(img)
    assert out["method"] == "maxima"
    assert out["snr"] >= 8.0
    assert out["crowding"] > 1.5
