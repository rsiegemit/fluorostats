"""Produce results/b3_viability_depth_profile.csv — per-z live coverage, raw vs attenuation-corrected.

S-BIAD2130 Day-14 Live/Dead z-stack (113 slices, z = 5.5 µm, XY downsampled 8x
on read), live channel 2: ``viability.viability_depth_profile`` before and after
``viability.attenuation_correct``.
Data: ``$FLUOROSTATS_DATA/viability/zccs1035_Day14_LiveDead.tif``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (SBIAD2130_DOWNSAMPLE, SBIAD2130_LIVE_CH, SBIAD2130_SEG, SBIAD2130_Z_UM, SMALL_OBJECT_RULE,
                    load_sbiad2130_channel, write_table)
from fluorostats.viability import attenuation_correct, viability_depth_profile

def main():
    live = load_sbiad2130_channel(SBIAD2130_LIVE_CH)
    prof = viability_depth_profile(live, **SBIAD2130_SEG)["live_by_z"]
    prof_corr = viability_depth_profile(attenuation_correct(live), **SBIAD2130_SEG)["live_by_z"]
    z = np.arange(len(prof))
    df = pd.DataFrame({"z_slice": z, "depth_um": z * SBIAD2130_Z_UM,
                       "live_fraction_raw": prof, "live_fraction_attn_corrected": prof_corr})
    write_table(df, "b3_viability_depth_profile", {"live_channel": SBIAD2130_LIVE_CH, "z_um": SBIAD2130_Z_UM,
                                                         "xy_downsample": SBIAD2130_DOWNSAMPLE, "seg": SBIAD2130_SEG,
                                                         "small_object_rule": SMALL_OBJECT_RULE})
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
