"""Quality-control overlay images for visual verification."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np


def overlay_3d(
    vol: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    alpha: float = 0.5,
) -> None:
    """Save a MIP overlay: grayscale intensity + magenta mask.

    Parameters
    ----------
    vol : ndarray (Z, Y, X)
        Single-channel intensity volume.
    mask : ndarray (Z, Y, X)
        Binary mask, same shape as vol.
    out_path : Path
        Output PNG path.
    alpha : float
        Mask overlay opacity (0–1).
    """
    # Maximum intensity projection along Z
    mip_intensity = vol.max(axis=0).astype(np.float64)
    mip_mask = mask.max(axis=0)

    _save_overlay(mip_intensity, mip_mask, out_path, alpha)


def overlay_2d(
    img: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    alpha: float = 0.5,
) -> None:
    """Save a 2D overlay: grayscale intensity + magenta mask.

    Parameters
    ----------
    img : ndarray (Y, X)
        Single-channel intensity image.
    mask : ndarray (Y, X)
        Binary mask.
    out_path : Path
        Output PNG path.
    alpha : float
        Mask overlay opacity.
    """
    _save_overlay(img.astype(np.float64), mask, out_path, alpha)


def _save_overlay(
    intensity: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    alpha: float,
) -> None:
    """Composite a grayscale image with a magenta binary mask overlay."""
    # Normalize intensity to 0–255
    vmin, vmax = intensity.min(), intensity.max()
    if vmax > vmin:
        gray = ((intensity - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        gray = np.zeros_like(intensity, dtype=np.uint8)

    # Build RGB from grayscale
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float64)

    # Magenta overlay where mask is True
    magenta = np.array([255.0, 0.0, 255.0])
    mask_3d = mask.astype(bool)
    rgb[mask_3d] = rgb[mask_3d] * (1 - alpha) + magenta * alpha

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out_path), rgb.astype(np.uint8))
