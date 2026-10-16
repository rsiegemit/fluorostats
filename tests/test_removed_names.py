"""The names deprecated in 0.9.0.dev0 are gone: each old spelling fails loudly
(ImportError, AttributeError) instead of resolving to its replacement."""

import importlib

import pytest

from fluorostats import (
    depth_batch,
    metrics_3d,
    plots,
    render3d,
)


_MIP_PANELS = ("mip_overlay", "live_dead_mip", "mip_grid", "depth_coded_mip", "layer_split_mip")
_WINDOW_HELPERS = ("parse_window", "window_label", "window_title", "window_bounds")
_DEPTH_FIGURES = ("plot_depth_curves", "plot_auc", "plot_retention_multiwindow")


@pytest.mark.parametrize("module", ["fluorostats.style"])
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
