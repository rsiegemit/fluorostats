"""The names deprecated in 0.9.0.dev0 are gone: each old spelling fails loudly
(ImportError, AttributeError, TypeError) instead of resolving to its replacement."""

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fluorostats import (
    depth,
    depth_batch,
    metrics_2d,
    metrics_3d,
    objects,
    plots,
    pubfig,
    render3d,
    ring,
    skeleton,
    texture,
)


_MIP_PANELS = ("mip_overlay", "live_dead_mip", "mip_grid", "depth_coded_mip", "layer_split_mip")
_WINDOW_HELPERS = ("parse_window", "window_label", "window_title", "window_bounds")
_DEPTH_FIGURES = ("plot_depth_curves", "plot_auc", "plot_retention_multiwindow")


@pytest.mark.parametrize("module", ["fluorostats.style", "fluorostats.deprecation"])
def test_removed_modules_do_not_import(module):
    with pytest.raises(ImportError):
        importlib.import_module(module)


@pytest.mark.parametrize(
    "module, name",
    [
        *((render3d, n) for n in _MIP_PANELS),
        *((depth_batch, n) for n in _WINDOW_HELPERS + _DEPTH_FIGURES),
        *((metrics_3d, n) for n in ("skeleton_metrics", "prune_skeleton", "n_junction_nodes")),
        (plots, "compute_pvalues"),
    ],
)
def test_moved_names_no_longer_resolve_at_their_old_home(module, name):
    with pytest.raises(AttributeError, match=name):
        getattr(module, name)


@pytest.mark.parametrize(
    "func, args, old",
    [
        (ring.ring_morphometry, (np.ones((5, 5)),), {"spacing": (1.0, 1.0)}),
        (texture.mesh_size, (np.eye(5),), {"spacing": 1.0}),
        (metrics_2d.coverage_metrics, (np.eye(5, dtype=bool),), {"pixel_size_um": 1.0}),
        (depth.intensity_depth_profile, (np.ones((3, 4, 4)),), {"voxel_size_z_um": 1.0}),
        (skeleton.prune_skeleton, (np.eye(5, dtype=bool),), {"min_branch_length_px": 1.0}),
        (objects.watershed_split, (np.eye(5, dtype=bool),), {"footprint_size": 7}),
    ],
)
def test_removed_keywords_are_a_type_error(func, args, old):
    with pytest.raises(TypeError, match=next(iter(old))):
        func(*args, **old)


def test_watershed_split_takes_no_fourth_positional_argument():
    with pytest.raises(TypeError):
        objects.watershed_split(np.eye(5, dtype=bool), 0, 3, 7)  # v0.8.2 footprint_size


@pytest.mark.parametrize("old", [{"dx": 3.0}, {"dy": 3.0}])
def test_panel_dx_dy_are_a_type_error(old):
    _, ax = plt.subplots()
    with pytest.raises(TypeError, match=next(iter(old))):
        pubfig.panel(ax, "a", **old)
    plt.close("all")
