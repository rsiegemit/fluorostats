"""Produce results/b3_viability_multi.csv — viability via six reductions vs full 3D (S-BIAD2130).

Live (ch 2) and dead (ch 1) channels of the S-BIAD2130 Day-14 stack (XY
downsampled 8x): ``viability.live_dead_fractions`` on the full volume is the
reference; mid-plane, MIP, mean of per-slice profile, attenuation-corrected 3D
and brightest-slice reductions are reported with their bias against it.
Data: ``$FLUOROSTATS_DATA/viability/zccs1035_Day14_LiveDead.tif``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, load_sbiad2130_channel
from fluorostats.viability import attenuation_correct, live_dead_fractions, viability_depth_profile

SEG = dict(method="otsu", threshold_scale=1.0, min_size=8)


def viability(live, dead):
    return live_dead_fractions(live, dead, **SEG)["viability"]


def main():
    live = load_sbiad2130_channel(2)
    dead = load_sbiad2130_channel(1)
    ref = viability(live, dead)
    mid = live.shape[0] // 2
    focus = int(np.argmax(live.sum((1, 2))))
    methods = {
        "full_3D_voxelwise(REF)": ref,
        "midplane_slice": viability(live[mid], dead[mid]),
        "MIP": viability(live.max(0), dead.max(0)),
        "mean_of_per_slice": float(np.nanmean(viability_depth_profile(live, dead, **SEG)["live_by_z"])),
        "attenuation_corrected_3D": viability(attenuation_correct(live), attenuation_correct(dead)),
        "brightest_focus_slice": viability(live[focus], dead[focus]),
    }
    rows = [{"method": name, "viability": round(float(v), 4),
             "abs_bias_vs_3D": round(float(v) - ref, 4),
             "rel_bias_pct": round((float(v) - ref) / ref * 100, 1)} for name, v in methods.items()]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b3_viability_multi.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
