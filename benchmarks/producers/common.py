"""Shared paths, loaders and method definitions for the results-table producers.

Each producer in this folder regenerates one committed table in
``benchmarks/results/`` (same file name) from raw public data under
``$FLUOROSTATS_DATA`` (default ``<repo>/data/downloads``; folder layout in
``benchmarks/DATA_MANIFEST.md``). Run from the repo root::

    PYTHONPATH=src python benchmarks/producers/<table_stem>.py

Producers are deterministic: no randomness in the measurement path, and every
synthetic generator or noise model takes an explicit seed. Each table is written
by ``write_table`` together with ``<table_stem>.run.json``, the record of the
producer's parameters and package versions.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import skimage
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import threshold_minimum
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

import fluorostats
from fluorostats.objects import label_3d
from fluorostats.segment import THRESHOLD_METHODS, segment_channel

DATA = Path(os.environ.get(
    "FLUOROSTATS_DATA", str(Path(__file__).resolve().parents[2] / "data" / "downloads")))
RESULTS = Path(__file__).resolve().parents[1] / "results"

# Global thresholds the comparison tables score: the library's registry plus
# Minimum (Prewitt 1966), which the library does not offer as a binarize method.
THRESHOLD_FNS = {**THRESHOLD_METHODS, "minimum": threshold_minimum}
# Column labels "<Name>_<year of the method's paper>" used by most tables.
YEAR_LABELS = {"otsu": "Otsu_1979", "li": "Li_1993", "isodata": "Isodata_1978",
               "triangle": "Triangle_1977", "yen": "Yen_1995"}

# The two "fluorostats Otsu + connected components" pipelines the tables score,
# named by what they run (helpers below); used as column / row labels.
RAW_OTSU_CC = "fluorostats(raw Otsu+CC)"
DENOISED_OTSU_CC = "fluorostats(denoised Otsu+CC)"

# Side (px) of the square field of the example point patterns.
EXAMPLE_FIELD = 500

# CPU inference ms per BBBC039 image of the two DL baselines, measured on the
# cluster (StarDist 2D_versatile_fluo, Cellpose nuclei; 20 images); carried as constants.
DL_CPU_MS = {"StarDist": 215.1, "Cellpose": 5547.0}


def write_table(df, stem: str, params: dict, *, timing: bool = False, **csv_kw) -> None:
    """Write ``results/<stem>.csv`` and its provenance record ``results/<stem>.run.json``.

    The record names the producer, its ``params`` and the python / numpy /
    scipy / scikit-image / fluorostats versions; a ``timing`` table also records
    the CPU (``platform.machine()``, ``platform.processor()``). No host name,
    user or absolute path is written. ``csv_kw`` goes to ``DataFrame.to_csv``
    (``index`` defaults to False).
    """
    df.to_csv(RESULTS / f"{stem}.csv", **{"index": False, **csv_kw})
    record = {"table": f"{stem}.csv", "producer": f"benchmarks/producers/{Path(sys.argv[0]).name}",
              "params": params,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "scikit-image": skimage.__version__,
                           "fluorostats": fluorostats.__version__}}
    if timing:
        record["platform"] = {"machine": platform.machine(), "processor": platform.processor()}
    (RESULTS / f"{stem}.run.json").write_text(json.dumps(record, indent=2) + "\n")


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


def load_sbiad2130_channel(ch: int, n_ch: int = 3, down: int = 8) -> np.ndarray:
    """One channel of the S-BIAD2130 Day-14 Live/Dead ZCYX stack, XY-downsampled on read."""
    stack = require(DATA / "viability" / "zccs1035_Day14_LiveDead.tif")
    with tifffile.TiffFile(stack) as t:
        nz = len(t.pages) // n_ch
        return np.stack([t.pages[z * n_ch + ch].asarray()[::down, ::down].astype(np.float32)
                         for z in range(nz)])


def thresholds(names, labels: dict[str, str] | None = None) -> dict:
    """``{column label: threshold function}`` for the registry ``names``, in order.

    ``labels`` maps a registry name to the column label its table uses; names
    without an entry keep the registry name.
    """
    labels = labels or {}
    return {labels.get(n, n): THRESHOLD_FNS[n] for n in names}


def threshold_cc(img: np.ndarray, threshold_fn, max_size: int) -> np.ndarray:
    """Global threshold -> drop objects of <= ``max_size`` px -> connected-component labels."""
    return label_3d(remove_small_objects(img > threshold_fn(img), max_size=max_size))[0]


def raw_otsu_cc(img: np.ndarray, min_size: int) -> np.ndarray:
    """RAW_OTSU_CC: Otsu on the raw image -> drop objects of <= ``min_size`` px -> label."""
    return label_3d(remove_small_objects(img > THRESHOLD_FNS["otsu"](img), max_size=min_size),
                    min_size=min_size)[0]


def denoised_otsu_cc(img: np.ndarray, min_size: int) -> np.ndarray:
    """DENOISED_OTSU_CC: Gaussian denoise (sigma 1) -> ``segment.binarize`` Otsu (objects
    smaller than ``min_size`` dropped) -> label, keeping components of >= ``min_size``."""
    return label_3d(segment_channel(img, method="otsu", min_size=min_size, sigma=1.0)[1],
                    min_size=min_size)[0]


def kerkhoff_peak_count(img: np.ndarray) -> int:
    """Cell count of the published Kerkhoff Fiji Live/Dead macro (its Find Maxima step).

    A comparator, not library code.
    """
    return len(peak_local_max(img, min_distance=3, threshold_abs=img.max() * 0.15))


def example_patterns(rng: np.random.Generator, n: int = 225) -> dict[str, np.ndarray]:
    """Regular, random (CSR) and clustered sets of ``n`` (y, x) points in the
    EXAMPLE_FIELD square, drawn from ``rng`` in that order (the regular grid draws nothing)."""
    g = int(np.sqrt(n))
    xs = np.linspace(30, EXAMPLE_FIELD - 30, g)
    regular = np.array([(y, x) for y in xs for x in xs])
    random = rng.uniform(20, EXAMPLE_FIELD - 20, (n, 2))
    k = 7
    cen = rng.uniform(60, EXAMPLE_FIELD - 60, (k, 2))
    clustered = np.clip(np.array([cen[i % k] + rng.normal(0, 22, 2) for i in range(n)]), 5, EXAMPLE_FIELD - 5)
    return {"regular": regular, "random": random, "clustered": clustered}


# Reference point-pattern statistics: the independent, textbook definitions the
# tile Gini is validated against (field-area normalised), not library code.
# Points are (x, y) in a square field of side ``field`` px. Clark-Evans here uses
# the known field area, unlike objects.nearest_neighbor_stats (point bounding box).
def ripley_L_dev(pts, area, r=40):
    n = len(pts)
    pairs = cKDTree(pts).query_pairs(r)
    K = area * 2 * len(pairs) / (n * (n - 1)) if n > 1 else 0
    return np.sqrt(K / np.pi) - r


def clark_evans(pts, area, n):
    d, _ = cKDTree(pts).query(pts, k=2)
    return d[:, 1].mean() / (0.5 / np.sqrt(n / area))


def quadrat_counts(pts, field, q=8):
    return np.histogram2d(pts[:, 1], pts[:, 0], bins=[np.linspace(0, field, q + 1)] * 2)[0]


def morisita(pts, field, q=8):
    H = quadrat_counts(pts, field, q)
    n = H.sum()
    return q * q * (np.sum(H * (H - 1))) / (n * (n - 1)) if n > 1 else 1.0


def lacunarity(pts, field, box=64):
    img = np.zeros((field, field))
    idx = np.clip(pts.astype(int), 0, field - 1)
    img[idx[:, 1], idx[:, 0]] = 1
    s = ndi.uniform_filter(img, box) * box * box
    m = s.mean()
    return 1 + s.var() / m ** 2 if m > 0 else 1.0


def quadrat_var(pts, field, q=8):
    H = quadrat_counts(pts, field, q)
    return H.var() / H.mean() if H.mean() > 0 else 0
