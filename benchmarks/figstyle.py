"""Methods-paper figure settings, layered on the library's figure kit.

The general publication tooling (Okabe-Ito palette, type scale, panel letters,
image helpers, scale bars, export, lollipop and Bland-Altman panels) lives in
`fluorostats.pubfig` and is re-exported here. This module adds only what is
specific to this paper: the tool colour identities, the manuscript text width,
the figure output folders, the raw-data root, and two dataset loaders.

    from figstyle import (OKABE, TOOL, apply_style, save, panel, scalebar,
                          caption, boot_ci, SUPP)
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

from fluorostats import stats  # noqa: E402
from fluorostats import pubfig  # noqa: E402
from fluorostats.pubfig import (  # noqa: E402,F401
    FS, OKABE, apply_style, bland_altman, composite2ch, fill_mask, image_axes, imnorm, lollipop,
    outline, panel, round_sig, scalebar,
)

# Fixed tool identities (colour + marker + linestyle) so series survive greyscale.
# fluorostats is ALWAYS blue and only fluorostats; DL tools: StarDist orange,
# Cellpose vermillion, Omnipose purple; classical baselines neutral grey.
TOOL = {
    "fluorostats": dict(c=OKABE["blue"],       m="o", ls="-"),
    "StarDist":    dict(c=OKABE["orange"],     m="s", ls="--"),
    "Cellpose":    dict(c=OKABE["vermillion"], m="^", ls="-."),
    "Omnipose":    dict(c=OKABE["purple"],     m="D", ls=":"),
    "DL":          dict(c=OKABE["black"],      m="*", ls="--"),
}

MAIN = Path(__file__).resolve().parent / "figures" / "main"
SUPP = Path(__file__).resolve().parent / "figures" / "supplementary"

# Raw-image data root. Portable: override with $FLUOROSTATS_DATA (see
# DATA_MANIFEST.md), else default to ./data/downloads relative to the repo root.
DATA = Path(os.environ.get(
    "FLUOROSTATS_DATA", str(Path(__file__).resolve().parents[1] / "data" / "downloads")))

# Manuscript text width for this submission: \textwidth = 372.0 pt = 130.74 mm =
# 5.147 in. Every figure is authored at this width so \includegraphics[width=
# \textwidth] is 1:1 (no LaTeX rescale) and 6-7 pt type stays 6-7 pt on the page.
TW = 5.147


def save(fig, name, folder=None, tight=True, pad=0.4):
    """`pubfig.save` into this paper's main-figure folder unless told otherwise."""
    pubfig.save(fig, name, folder or MAIN, tight=tight, pad=pad)


def caption(name, text, folder=None):
    """`pubfig.caption` into this paper's main-figure folder unless told otherwise."""
    pubfig.caption(name, text, folder or MAIN)


def boot_ci(x, reps=10000, seed=0, ci=95):
    """Bootstrap mean + CI of a 1D array as ``(mean, lo, hi)`` (``ci`` in percent)."""
    r = stats.bootstrap_mean_ci(x, n_boot=reps, ci=ci / 100, seed=seed)
    return r["mean"], r["ci_low"], r["ci_high"]


# ---- dataset loaders used by the figure scripts ----

def gt_instances_bbbc039(mask_path):
    """Reconstruct BBBC039 instance ground truth from its 3-level mask PNG
    (1 = interior seed, >= 2 = boundary): seeded watershed over the object region."""
    from PIL import Image
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    m = np.array(Image.open(mask_path))
    r = m[..., 0] if m.ndim == 3 else m
    seeds, _ = ndi.label(r == 1)
    return watershed((r >= 2).astype(np.uint8), seeds, mask=(r > 0))


def mid_slice(path):
    """Middle z-slice of a TIFF volume as float32 (for a quick 2D preview)."""
    import tifffile
    vol = tifffile.imread(path)
    return vol[vol.shape[0] // 2].astype(np.float32)


def load_channels(path, ch, n_ch=3, down=1):
    """Load one interleaved channel (``ch`` of ``n_ch``) from a multi-page TIFF as a
    z-stack, optionally downsampled by ``down``. Returns (z, y, x) float32."""
    import tifffile
    sl = []
    with tifffile.TiffFile(path) as t:
        nz = len(t.pages) // n_ch
        for z in range(nz):
            sl.append(t.pages[z * n_ch + ch].asarray()[::down, ::down].astype(np.float32))
    return np.stack(sl)
