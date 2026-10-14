"""Tests for io_canonical: (C, Z, Y, X) / (C, Y, X) layout from reader arrays.

Axis-name reordering is exercised end to end through the loaders in
test_io.py (TIFF) and test_io_vendors.py (ND2, CZI fakes).
"""

import numpy as np
import pytest

from fluorostats import io_canonical


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
