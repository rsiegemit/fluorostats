import numpy as np
import pytest

from fluorostats import depth, metrics_2d, ring, skeleton, texture
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



@pytest.mark.parametrize("func, args, old", [
    (ring.ring_morphometry, (np.ones((5, 5)),), {"spacing": (1.0, 1.0)}),
    (texture.mesh_size, (np.eye(5),), {"spacing": 1.0}),
    (metrics_2d.coverage_metrics, (np.eye(5, dtype=bool),), {"pixel_size_um": 1.0}),
    (depth.intensity_depth_profile, (np.ones((3, 4, 4)),), {"voxel_size_z_um": 1.0}),
    (skeleton.prune_skeleton, (np.eye(5, dtype=bool),), {"min_branch_length_px": 1.0}),
    (skeleton.prune_skeleton, (np.eye(5, dtype=bool),), {"spacing": (1.0, 1.0)}),
])
def test_renamed_spacing_arguments_still_work(func, args, old):
    """MOD-5: the old names keep working, with a DeprecationWarning, for one cycle."""
    with pytest.warns(DeprecationWarning, match="is deprecated; use"):
        func(*args, **old)
