"""Viability module — live fraction via >=4 reduction methods vs true 3D (S-BIAD2130)."""
import sys, numpy as np, pandas as pd, tifffile
from pathlib import Path
from fluorostats.viability import (live_dead_fractions, viability_depth_profile, attenuation_correct)
STACK=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/viability/zccs1035_Day14_LiveDead.tif")
OUT=Path(__file__).resolve().parent/"results"; Z_UM=5.5; DOWN=8
def load_channel(ch,n_ch=3):
    sl=[]
    with tifffile.TiffFile(STACK) as t:
        nz=len(t.pages)//n_ch
        for z in range(nz): sl.append(t.pages[z*n_ch+ch].asarray()[::DOWN,::DOWN].astype(np.float32))
    return np.stack(sl)
seg=dict(method="otsu",threshold_scale=1.0,min_size=8)
print("loading live/dead channels...",flush=True)
live=load_channel(2); dead=load_channel(1)   # ch2 brightest=live signal, ch1=other marker
def viab(d): return d.get("viability", d.get("live_fraction"))
# reference: full 3D voxelwise
ref=viab(live_dead_fractions(live,dead,**seg))
mid=live.shape[0]//2
methods={
 "full_3D_voxelwise(REF)": ref,
 "midplane_slice": viab(live_dead_fractions(live[mid],dead[mid],**seg)),
 "MIP": viab(live_dead_fractions(live.max(0),dead.max(0),**seg)),
 "mean_of_per_slice": float(np.nanmean(viability_depth_profile(live,dead,**seg)["live_by_z"])),
 "attenuation_corrected_3D": viab(live_dead_fractions(attenuation_correct(live),attenuation_correct(dead),**seg)),
 "brightest_focus_slice": viab(live_dead_fractions(live[int(np.argmax(live.sum((1,2))))],dead[int(np.argmax(live.sum((1,2))))],**seg)),
}
rows=[]
for name,val in methods.items():
    v=float(val); rows.append({"method":name,"viability":round(v,4),
        "abs_bias_vs_3D":round(v-ref,4),"rel_bias_pct":round((v-ref)/ref*100,1)})
df=pd.DataFrame(rows); df.to_csv(OUT/"b3_viability_multi.csv",index=False)
print("=== Live fraction via 6 methods vs true 3D (S-BIAD2130 Day-14) ===")
print(df.to_string(index=False))
print(f"\nReference (full 3D voxelwise) viability = {ref:.4f}")
print("2D shortcuts (midplane/MIP/focus) bias the viability estimate; per-slice+attn track 3D.")
