"""B3 (viability) — 2D overestimates viability; 3D reveals depth-dependent death.

Public data: BioImage Archive S-BIAD2130 (Jung et al. 2025, CC0), a Day-14
3D Live/Dead confocal z-stack (113 slices, 3 channels, z-spacing 5.5 um).

Uses the new fluorostats.viability module to show the headline result the
methods paper argues: a single mid-plane or a maximum-intensity projection
OVERESTIMATES live coverage relative to the true 3D fraction, and the per-z
profile reveals a depth gradient (signal falling with depth) that a 2D readout
structurally cannot see. Attenuation correction probes how much of the gradient
is optical vs biological.

Loads memory-safely: each z-slice downsampled on read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

from fluorostats.viability import (
    viability_2d_vs_3d, viability_depth_profile, attenuation_correct,
)
from fluorostats.segment import binarize

STACK = Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/"
             "downloads/viability/zccs1035_Day14_LiveDead.tif")
OUT = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent / "figures"
Z_UM = 5.5
DOWN = 8   # XY downsample factor


def load_channel(ch: int, n_ch: int = 3) -> np.ndarray:
    """Load one channel of the ZCYX stack, downsampled in XY on read."""
    slices = []
    with tifffile.TiffFile(STACK) as t:
        pages = t.pages
        nz = len(pages) // n_ch
        for z in range(nz):
            p = pages[z * n_ch + ch].asarray()
            slices.append(p[::DOWN, ::DOWN].astype(np.float32))
    return np.stack(slices)


def main():
    # channel means from metadata inspection: ch0~267, ch1~1533, ch2~4516.
    # Use the brightest cell-signal channel as the viability channel.
    print("Loading viability channel (downsampled)...", flush=True)
    live = load_channel(2)
    print(f"  volume {live.shape}, z-spacing {Z_UM} um", flush=True)

    seg = dict(method="otsu", threshold_scale=1.0, min_size=8)
    cmp = viability_2d_vs_3d(live, **seg)
    prof = viability_depth_profile(live, **seg)["live_by_z"]

    # attenuation-corrected profile (does the gradient survive optical correction?)
    corr = attenuation_correct(live)
    prof_corr = viability_depth_profile(corr, **seg)["live_by_z"]

    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame([{
        "n_slices": live.shape[0], "z_um": Z_UM,
        **{k: round(v, 4) for k, v in cmp.items()},
    }]).to_csv(OUT / "b3_viability_summary.csv", index=False)
    pd.DataFrame({"z_slice": np.arange(len(prof)),
                  "depth_um": np.arange(len(prof)) * Z_UM,
                  "live_fraction_raw": prof,
                  "live_fraction_attn_corrected": prof_corr}).to_csv(
        OUT / "b3_viability_depth_profile.csv", index=False)

    print("\n=== 2D vs 3D live coverage (S-BIAD2130 Day-14 Live/Dead) ===", flush=True)
    for k, v in cmp.items():
        print(f"  {k:28s} {v:.4f}", flush=True)
    print(f"\n  MIP overestimates 3D live coverage by {cmp['overestimate_mip']:.2f}x", flush=True)
    print(f"  mid-plane overestimates by {cmp['overestimate_midplane']:.2f}x", flush=True)

    # Figure: depth profile (raw vs attenuation-corrected) + 2D/3D bars
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fluorostats.style import apply_style, PALETTE
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    depth = np.arange(len(prof)) * Z_UM
    ax1.plot(depth, prof, color=PALETTE["accent"], lw=2.4, label="raw live fraction")
    ax1.plot(depth, prof_corr, color=PALETTE["primary"], lw=2.0, ls="--",
             label="attenuation-corrected")
    ax1.set_xlabel("depth (µm)"); ax1.set_ylabel("live area fraction per z-slice")
    ax1.set_title("Depth-resolved viability — a gradient 2D cannot see")
    ax1.legend(frameon=False)
    bars = ["mid-plane", "MIP", "full 3D"]
    vals = [cmp["live_fraction_midplane"], cmp["live_fraction_mip"], cmp["live_fraction_3d"]]
    cols = [PALETTE["muted"], PALETTE["muted"], PALETTE["accent"]]
    ax2.bar(bars, vals, color=cols, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("live area fraction")
    ax2.set_title(f"2D overestimates viability\nMIP = {cmp['overestimate_mip']:.1f}× the true 3D fraction")
    fig.suptitle("B3 viability — public Day-14 Live/Dead z-stack (S-BIAD2130)",
                 fontweight="semibold")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "b3_viability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved results + figures/b3_viability.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
