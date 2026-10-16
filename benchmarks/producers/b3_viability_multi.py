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

from common import (SBIAD2130_DEAD_CH, SBIAD2130_DOWNSAMPLE, SBIAD2130_LIVE_CH, SBIAD2130_SEG, SMALL_OBJECT_RULE,
                    load_sbiad2130_channel, write_table)
from fluorostats.viability import attenuation_correct, live_dead_fractions, viability_depth_profile

def viability(live, dead):
    return live_dead_fractions(live, dead, **SBIAD2130_SEG)["viability"]


def main():
    live = load_sbiad2130_channel(SBIAD2130_LIVE_CH)
    dead = load_sbiad2130_channel(SBIAD2130_DEAD_CH)
    ref = viability(live, dead)
    mid = live.shape[0] // 2
    focus = int(np.argmax(live.sum((1, 2))))
    methods = {
        "full_3D_voxelwise(REF)": ref,
        "midplane_slice": viability(live[mid], dead[mid]),
        "MIP": viability(live.max(0), dead.max(0)),
        "mean_of_per_slice": float(np.nanmean(viability_depth_profile(live, dead, **SBIAD2130_SEG)["live_by_z"])),
        "attenuation_corrected_3D": viability(attenuation_correct(live), attenuation_correct(dead)),
        "brightest_focus_slice": viability(live[focus], dead[focus]),
    }
    rows = [{"method": name, "viability": round(float(v), 4),
             "bias_vs_3D": round(float(v) - ref, 4),
             "rel_bias_pct": round((float(v) - ref) / ref * 100, 1)} for name, v in methods.items()]
    df = pd.DataFrame(rows)
    write_table(df, "b3_viability_multi", {"live_channel": SBIAD2130_LIVE_CH, "dead_channel": SBIAD2130_DEAD_CH,
                                                 "xy_downsample": SBIAD2130_DOWNSAMPLE, "seg": SBIAD2130_SEG,
                                                 "small_object_rule": SMALL_OBJECT_RULE})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
