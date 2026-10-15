import numpy as np
import pytest

from fluorostats import depth, depth_batch, depth_figures, metrics_2d, ring, skeleton, texture
from fluorostats.deprecation import renamed_keyword


@renamed_keyword("spacing", "voxel_size_um")
def _f(x, voxel_size_um=1.0):
    return x * voxel_size_um


def test_old_keyword_still_works_and_warns():
    with pytest.warns(DeprecationWarning, match="'spacing' is deprecated; use 'voxel_size_um'"):
        assert _f(2, spacing=3.0) == 6.0
    assert _f(2, voxel_size_um=3.0) == 6.0 and _f(2, 3.0) == 6.0


def test_both_names_is_an_error():
    with pytest.raises(TypeError, match="both 'spacing'"):
        _f(2, spacing=1.0, voxel_size_um=2.0)


@pytest.mark.parametrize(
    "func, args, old",
    [
        (ring.ring_morphometry, (np.ones((5, 5)),), {"spacing": (1.0, 1.0)}),
        (texture.mesh_size, (np.eye(5),), {"spacing": 1.0}),
        (metrics_2d.coverage_metrics, (np.eye(5, dtype=bool),), {"pixel_size_um": 1.0}),
        (depth.intensity_depth_profile, (np.ones((3, 4, 4)),), {"voxel_size_z_um": 1.0}),
        (skeleton.prune_skeleton, (np.eye(5, dtype=bool),), {"min_branch_length_px": 1.0}),
    ],
)
def test_renamed_spacing_arguments_still_work(func, args, old):
    """MOD-5: the old names keep working, with a DeprecationWarning, for one cycle."""
    with pytest.warns(DeprecationWarning, match="is deprecated; use"):
        func(*args, **old)


def test_watershed_footprint_size_is_accepted_and_ignored():
    """The seeding no longer uses a footprint; old calls warn instead of failing."""
    from fluorostats.objects import watershed_split

    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    with pytest.warns(DeprecationWarning, match="'footprint_size' no longer has an effect"):
        old = watershed_split(m, min_distance=3, footprint_size=7)
    with pytest.warns(DeprecationWarning, match="'footprint_size' no longer has an effect"):
        positional = watershed_split(m, 0, 3, 7)  # v0.8.2 positional form
    new = watershed_split(m, min_distance=3)
    for res in (old, positional):
        assert res[1] == new[1] and (res[0] == new[0]).all()


@pytest.mark.parametrize(
    "module, name, target",
    [
        (depth_batch, "parse_window", depth),
        (depth_batch, "window_label", depth),
        (depth_batch, "plot_retention_multiwindow", depth_figures),
    ],
)
def test_re_exported_names_resolve_with_a_warning(module, name, target):
    # STY-4: these re-exports were kept without a DeprecationWarning.
    short = module.__name__.rsplit(".", 1)[-1]
    with pytest.warns(DeprecationWarning, match=f"{short}.{name} is deprecated; use"):
        assert getattr(module, name) is getattr(target, name)
    assert name not in module.__all__
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        module.nope
