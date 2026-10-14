"""Shared paths and loaders for the results-table producers.

Each producer in this folder regenerates one committed table in
``benchmarks/results/`` (same file name) from raw public data under
``$FLUOROSTATS_DATA`` (default ``<repo>/data/downloads``; folder layout in
``benchmarks/DATA_MANIFEST.md``). Run from the repo root::

    PYTHONPATH=src python benchmarks/producers/<table_stem>.py

Producers are deterministic: no randomness in the measurement path, and every
synthetic generator or noise model takes an explicit seed.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed

DATA = Path(os.environ.get(
    "FLUOROSTATS_DATA", str(Path(__file__).resolve().parents[2] / "data" / "downloads")))
RESULTS = Path(__file__).resolve().parents[1] / "results"


def require(path: Path) -> Path:
    """Return ``path`` or exit with the dataset the producer needs (ERR-1)."""
    if not path.exists():
        raise SystemExit(f"missing {path}: download it per benchmarks/DATA_MANIFEST.md "
                         f"into $FLUOROSTATS_DATA (currently {DATA})")
    return path


def bbbc039_gt_instances(mask_path: str | Path) -> np.ndarray:
    """BBBC039 instance GT from its 3-class mask (R: 1 = interior, >=2 = boundary).

    Interior components seed a watershed over the boundary class inside the
    foreground; this is the same reconstruction the DL baselines were scored on.
    """
    m = np.array(Image.open(mask_path))
    r = m[..., 0] if m.ndim == 3 else m
    seeds, n = ndi.label(r == 1)
    if n == 0:
        return np.zeros(r.shape, int)
    return watershed((r >= 2).astype(np.uint8), seeds, mask=(r > 0))


def bbbc039_pairs() -> list[tuple[str, str, str]]:
    """Sorted ``(stem, image path, mask path)`` for every BBBC039 image with a mask."""
    root = require(DATA / "BBBC039")
    imgs = {Path(p).stem: p for p in glob.glob(str(root / "images" / "images" / "*.tif"))}
    masks = {Path(p).stem: p for p in glob.glob(str(root / "masks" / "masks" / "*.png"))}
    return [(s, imgs[s], masks[s]) for s in sorted(set(imgs) & set(masks))]


def bbbc024_volumes(subdir: str = "BBBC024", n: int | None = None) -> list[tuple[str, str]]:
    """First ``n`` sorted ``(image-final, image-labels)`` path pairs of a BBBC024 set."""
    root = require(DATA / subdir)
    finals = sorted(glob.glob(str(root / "**" / "image-final_*.tif"), recursive=True))[:n]
    return [(f, f.replace("image-final_", "image-labels_")) for f in finals]


def foreground_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Foreground (semantic) Dice of two masks or label images; 1.0 if both empty."""
    # exception REU-3: fluorostats.validate scores instances (F1/AP) but has no
    # foreground Dice yet; move this into validate (with a test) when it gains one.
    p = pred > 0
    g = gt > 0
    s = p.sum() + g.sum()
    return 2 * (p & g).sum() / s if s else 1.0


def load_sbiad2130_channel(ch: int, n_ch: int = 3, down: int = 8) -> np.ndarray:
    """One channel of the S-BIAD2130 Day-14 Live/Dead ZCYX stack, XY-downsampled on read."""
    stack = require(DATA / "viability" / "zccs1035_Day14_LiveDead.tif")
    with tifffile.TiffFile(stack) as t:
        nz = len(t.pages) // n_ch
        return np.stack([t.pages[z * n_ch + ch].asarray()[::down, ::down].astype(np.float32)
                         for z in range(nz)])
