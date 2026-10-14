"""Produce results/b3_viability_depth_profile.csv — per-z live coverage, raw vs attenuation-corrected.

S-BIAD2130 Day-14 Live/Dead z-stack (113 slices, z = 5.5 µm, XY downsampled 8x
on read), live channel 2: ``viability.viability_depth_profile`` before and after
``viability.attenuation_correct``.
Data: ``$FLUOROSTATS_DATA/viability/zccs1035_Day14_LiveDead.tif``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, load_sbiad2130_channel
from fluorostats.viability import attenuation_correct, viability_depth_profile

Z_UM = 5.5
LIVE_CH = 2
SEG = dict(method="otsu", threshold_scale=1.0, min_size=8)


def main():
    live = load_sbiad2130_channel(LIVE_CH)
    prof = viability_depth_profile(live, **SEG)["live_by_z"]
    prof_corr = viability_depth_profile(attenuation_correct(live), **SEG)["live_by_z"]
    z = np.arange(len(prof))
    df = pd.DataFrame({"z_slice": z, "depth_um": z * Z_UM,
                       "live_fraction_raw": prof, "live_fraction_attn_corrected": prof_corr})
    df.to_csv(RESULTS / "b3_viability_depth_profile.csv", index=False)
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
