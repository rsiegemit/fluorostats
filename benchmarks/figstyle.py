"""Methods-paper figure settings, layered on the library's figure kit.

The general publication tooling (Okabe-Ito palette, type scale, panel letters,
image helpers, scale bars, export, lollipop and Bland-Altman panels) lives in
`fluorostats.pubfig` and is re-exported here. This module adds only what is
specific to this paper: the tool colour identities, the manuscript text width
and the figure output folders. The raw-data root, the missing-data check and the
dataset loaders are the producers' (``producers/common.py``), re-exported here.

    from figstyle import (OKABE, TOOL, apply_style, save, panel, scalebar,
                          caption, SUPP)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

from fluorostats import pubfig  # noqa: E402
from fluorostats.pubfig import (  # noqa: E402,F401
    FS, OKABE, apply_style, bland_altman, composite2ch, fill_mask, image_axes, imnorm, lollipop,
    outline, panel, round_sig, scalebar,
)
from producers.common import (  # noqa: E402,F401
    BBBC039_UM_PER_PX, DATA, DENOISED_OTSU_CC, RAW_OTSU_CC, bbbc039_gt_instances, example_patterns, load_sbiad2130_channel,
    raw_otsu_cc, require,
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


# ---- dataset loaders used by the figure scripts ----

def mid_slice(path):
    """Middle z-slice of a TIFF volume as float32 (for a quick 2D preview)."""
    import tifffile
    vol = tifffile.imread(path)
    return vol[vol.shape[0] // 2].astype(np.float32)
