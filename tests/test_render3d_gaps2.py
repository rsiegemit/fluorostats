"""Gap-filling tests for fluorostats.render3d — lines 285, 325, 452."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fluorostats.render3d import mip_overlay, live_dead_mip, depth_coded_mip


# ---------------------------------------------------------------------------
# mip_overlay — line 285: ax=None creates its own figure
# ---------------------------------------------------------------------------

def test_mip_overlay_creates_own_axes_when_none():
    """ax=None → fig, ax = plt.subplots() fires (line 285)."""
    intensity = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32) * 100
    ax = mip_overlay(intensity)   # no ax passed
    assert ax is not None
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# live_dead_mip — line 325: ax=None creates its own figure
# ---------------------------------------------------------------------------

def test_live_dead_mip_creates_own_axes_when_none():
    """ax=None → fig, ax = plt.subplots() fires (line 325)."""
    live = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32)
    ax = live_dead_mip(live)   # no ax passed
    assert ax is not None
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# depth_coded_mip — line 452: ax=None creates its own figure
# ---------------------------------------------------------------------------

def test_depth_coded_mip_creates_own_axes_when_none():
    """ax=None → fig, ax = plt.subplots() fires (line 452)."""
    vol = np.random.RandomState(0).rand(8, 16, 16).astype(np.float32)
    ax = depth_coded_mip(vol)   # no ax passed
    assert ax is not None
    plt.close(ax.figure)
