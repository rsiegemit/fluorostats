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
        Mask overlay opacity (0–1).
    """
    _save_overlay(img.astype(np.float64), mask, out_path, alpha)


def overlay_rgb(
    intensity: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 255),
    alpha: float = 0.5,
) -> np.ndarray:
    """A 2D image stretched min-max to grey, with ``mask`` blended in ``color``.

    Returns a ``uint8`` (Y, X, 3) image; a flat image is black. The one overlay
    rule shared by the QC PNGs and `render3d.mip_overlay`.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    vmin, vmax = intensity.min(), intensity.max()
    if vmax > vmin:
        gray = ((intensity - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        gray = np.zeros_like(intensity, dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float64)
    sel = np.asarray(mask, dtype=bool)
    rgb[sel] = rgb[sel] * (1 - alpha) + np.asarray(color, dtype=float) * alpha
    return rgb.astype(np.uint8)


def _save_overlay(
    intensity: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    alpha: float,
) -> None:
    """Write `overlay_rgb` (magenta mask) to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out_path), overlay_rgb(intensity, mask, (255, 0, 255), alpha))
