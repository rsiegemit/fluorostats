"""Tests for fluorostats.qc."""

import imageio.v3 as iio
import numpy as np

from fluorostats import qc


def test_qc_overlay_writes_png(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (32, 32)).astype(np.uint8)
    mask = np.zeros((32, 32), bool)
    mask[8:24, 8:24] = True
    qc.overlay_2d(img, mask, tmp_path / "ov.png")
    rgb = iio.imread(tmp_path / "ov.png")[..., :3].astype(int)
    assert rgb.shape == (32, 32, 3)
    outside, inside = rgb[~mask], rgb[mask]
    assert (outside[:, 0] == outside[:, 1]).all() and (outside[:, 1] == outside[:, 2]).all()
    assert (inside[:, 0] > inside[:, 1]).all() and (inside[:, 2] > inside[:, 1]).all()  # magenta
