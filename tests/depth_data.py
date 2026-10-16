"""Synthetic depth stacks shared by the depth-driver test files."""

import numpy as np
import tifffile


def decay_stack(path, lam=20.0, n_z=16, yx=8, bg=5.0, i0=150.0):
    """Write a synthetic z-stack whose per-slice mean is i0*exp(-z/lam)+bg."""
    z = np.arange(n_z)[:, None, None]
    vol = np.broadcast_to((i0 * np.exp(-z / lam) + bg).astype(np.float32), (n_z, yx, yx))
    tifffile.imwrite(path, np.ascontiguousarray(vol))


def flat_stack(path, value=5.0, n_z=16, yx=8):
    """Write a spatially-flat stack (used as blank)."""
    vol = np.full((n_z, yx, yx), value, dtype=np.float32)
    tifffile.imwrite(path, vol)
