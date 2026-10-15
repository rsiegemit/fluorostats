"""Tests for io_canonical: (C, Z, Y, X) / (C, Y, X) layout from reader arrays.

Axis-name reordering is exercised end to end through the loaders in
test_io.py (TIFF) and test_io_vendors.py (ND2, CZI fakes).
"""

import warnings

import numpy as np
import pytest
import tifffile

from fluorostats import io_canonical
from fluorostats.io import load_volume


def test_canonicalize_volume_bad_ndim_raises():
    with pytest.raises(ValueError, match="Unexpected array shape"):
        io_canonical.canonicalize_volume(np.zeros((5,)), {"channel_names": []})


def test_canonicalize_volume_high_ndim_recurses():
    # 6D with leading singletons collapses to 4D via recursion
    arr = np.zeros((1, 1, 2, 3, 8, 8), dtype=np.uint16)
    out = io_canonical.canonicalize_volume(arr, {"channel_names": []})
    assert out.shape == (2, 3, 8, 8)


def test_canonicalize_image_bad_ndim_raises():
    with pytest.raises(ValueError, match="Unexpected 2D image shape"):
        io_canonical.canonicalize_image(np.zeros((2, 2, 2, 2)), {"channel_names": []})


def test_squeeze_trailing_singleton():
    arr = np.zeros((2, 3, 8, 8, 1), dtype=np.uint16)
    out = io_canonical.squeeze_singleton_dims(arr)
    assert out.shape == (2, 3, 8, 8)


def test_squeeze_non_singleton_high_ndim():
    # ndim>4 with no leading/trailing singleton -> final while arr=arr[0]
    arr = np.zeros((2, 3, 4, 8, 8), dtype=np.uint16)
    out = io_canonical.squeeze_singleton_dims(arr)
    assert out.ndim == 4


def test_canonicalize_volume_high_ndim_squeezes():
    # 6D non-singleton -> squeeze collapses to 4D
    arr = np.zeros((2, 2, 3, 4, 8, 8), dtype=np.uint16)
    out = io_canonical.canonicalize_volume(arr, {"channel_names": []})
    assert out.ndim == 4


# ---------------------------------------------------------------------------
# Partially labelled TIFF axes and the shape fallback's refusals
# ---------------------------------------------------------------------------


def _rgb_stack(n_z):
    """(Z, Y, X, S) uint8 stack whose sample s holds the value 10 * (s + 1)."""
    return np.stack([np.full((n_z, 16, 16), 10 * (s + 1), np.uint8) for s in range(3)], -1)


def test_rgb_z_stack_unlabelled_axis_is_taken_as_z(tmp_path):
    # Regression: tifffile labels this 'QYXS'; the Q axis sent the whole array to the
    # shape rule, which read it as C=6, Z=16, Y=16, X=3.
    p = tmp_path / "rgb_stack.tif"
    tifffile.imwrite(p, _rgb_stack(6), photometric="rgb")
    with pytest.warns(RuntimeWarning, match="assuming the unlabelled axis of length 6 is z"):
        arr, meta = load_volume(p)
    assert arr.shape == (3, 6, 16, 16)
    assert [arr[c].mean() for c in range(3)] == [10, 20, 30]
    assert len(meta["channel_names"]) == 3


def test_planar_rgb_z_stack_unlabelled_axis_is_taken_as_z(tmp_path):
    # 'QSYX' (12, 3, Y, X) used to become 64 channels (channel-last guess).
    p = tmp_path / "planar.tif"
    tifffile.imwrite(
        p, _rgb_stack(12).transpose(0, 3, 1, 2), photometric="rgb", planarconfig="separate"
    )
    with pytest.warns(RuntimeWarning, match="unlabelled axis of length 12 is z"):
        arr, _ = load_volume(p)
    assert arr.shape == (3, 12, 16, 16)
    assert [arr[c].mean() for c in range(3)] == [10, 20, 30]


def test_plain_z_stack_says_z_was_assumed(tmp_path):
    p = tmp_path / "plain.tif"
    tifffile.imwrite(p, np.zeros((5, 16, 16), np.uint16), photometric="minisblack")
    with pytest.warns(RuntimeWarning, match="unlabelled axis of length 5 is z"):
        arr, _ = load_volume(p)
    assert arr.shape == (1, 5, 16, 16)


@pytest.mark.parametrize(
    "shape, said", [((3, 8, 32, 4), "X = 4"), ((12, 3, 16, 16), "16 channels")]
)
def test_shape_fallback_refuses_implausible_layouts(shape, said):
    with pytest.raises(ValueError, match=said):
        io_canonical.canonicalize_volume(np.zeros(shape), {"channel_names": []})


def test_two_unlabelled_axes_keep_the_shape_rule_and_its_refusal(tmp_path):
    p = tmp_path / "qqyx.tif"
    tifffile.imwrite(p, np.zeros((3, 8, 32, 4), np.uint8), photometric="minisblack")
    with pytest.raises(ValueError, match="X = 4.*axes named"):
        load_volume(p)


@pytest.mark.parametrize("n_z", [6, 8])
def test_unlabelled_channel_last_stack_with_few_slices_is_read(tmp_path, n_z):
    # Regression: (Z <= 10, Y, X, C <= 4) was read as (C, Z, Y, X) with X = C and
    # refused, with advice (name the axes) a .npy file cannot follow.
    p = tmp_path / "zyxc.npy"
    zyxc = np.zeros((n_z, 32, 40, 2), np.uint16)
    zyxc[..., 1] = 7
    np.save(p, zyxc)
    with pytest.warns(RuntimeWarning, match="assuming channel-last"):
        arr, _ = load_volume(p)
    assert arr.shape == (2, n_z, 32, 40) and arr[1].min() == 7 and arr[0].max() == 0


def test_refused_guess_is_not_announced_first_and_advice_fits_npy(tmp_path):
    # Regression: (12, 3, 16, 16) warned "transposing" and then refused; the advice
    # named OME-TIFF for a .npy file.
    p = tmp_path / "zcyx.npy"
    np.save(p, np.zeros((12, 3, 16, 16), np.uint16))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=r"16 channels.*\(C, Z, Y, X\) order") as e:
            load_volume(p)
    assert "OME-TIFF" not in str(e.value)
    assert not [w for w in caught if "transposing" in str(w.message)]


def test_single_plane_unlabelled_axis_needs_no_assumption():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = io_canonical.canonicalize_volume(np.zeros((1, 16, 16)), {"channel_names": []}, "QYX")
    assert out.shape == (1, 1, 16, 16)


def test_unlabelled_4d_guess_is_announced(tmp_path):
    # Regression: a 4D .npy with a leading axis <= 10 was taken as (C, Z, Y, X) silently.
    p = tmp_path / "czyx.npy"
    np.save(p, np.zeros((5, 8, 32, 32), np.uint16))
    with pytest.warns(RuntimeWarning, match=r"read as \(C, Z, Y, X\) with 5 channels") as w:
        arr, _ = load_volume(p)
    assert all("OME-TIFF" not in str(x.message) for x in w)  # a .npy cannot name axes
    assert arr.shape == (5, 8, 32, 32)


def test_channel_names_are_padded_into_a_new_list():
    names = ["DAPI"]
    meta = {"channel_names": names}
    io_canonical.canonicalize_volume(np.zeros((1, 2, 3, 16, 16)), meta, "CZSYX")
    assert meta["channel_names"] == ["DAPI", "Ch2", "Ch3"]
    assert names == ["DAPI"]
    image_meta = {"channel_names": names}
    io_canonical.canonicalize_image(np.zeros((16, 16, 3)), image_meta)
    assert image_meta["channel_names"] == ["DAPI", "Ch2", "Ch3"]
    assert names == ["DAPI"]
