"""Tests for fluorostats.qc."""

import numpy as np

from fluorostats import qc


def test_qc_overlay_writes_png(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (32, 32)).astype(np.uint8)
    mask = np.zeros((32, 32), bool)
    mask[8:24, 8:24] = True
    qc.overlay_2d(img, mask, tmp_path / "ov.png")
    assert (tmp_path / "ov.png").exists()
