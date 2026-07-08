"""Depth/infiltration metrics — fluorostats depth_centroid vs 4 alt estimators."""
import numpy as np, pandas as pd
from pathlib import Path
from fluorostats.morphometry import depth_centroid, depth_profile, depth_span
RES=Path(__file__).resolve().parent/"results"
def make(nz=60, center=30, width=6, size=48):
    z=np.arange(nz); prof=np.exp(-0.5*((z-center)/width)**2)  # gaussian in z, known center
    vol=np.zeros((nz,size,size),np.float32)
    for i in range(nz): vol[i]=prof[i]*100
    return vol, center
rows=[]
for true_c in [15,30,45]:
    vol,c=make(center=true_c)
    prof=depth_profile(vol)
    fs=depth_centroid(vol)["z_centroid"]                       # fluorostats: intensity-weighted
    geo=float(np.mean(np.where(prof>prof.max()*0.1)[0]))       # geometric center of suprathresh
    peak=float(np.argmax(prof))                                # peak/mode
    med=float(np.searchsorted(np.cumsum(prof)/prof.sum(),0.5)) # median of profile
    # FWHM midpoint
    half=prof>prof.max()/2; idx=np.where(half)[0]; fwhm=float((idx[0]+idx[-1])/2) if len(idx) else np.nan
    for name,val in [("fluorostats(intensity_centroid)",fs),("geometric_center",geo),
                     ("peak_mode",peak),("profile_median",med),("FWHM_midpoint",fwhm)]:
        rows.append({"true_center":true_c,"method":name,"recovered":round(val,2),"abs_err":round(abs(val-true_c),2)})
df=pd.DataFrame(rows); df.to_csv(RES/"b_depth_metrics.csv",index=False)
piv=df.pivot(index="method",columns="true_center",values="abs_err")
print("=== Depth-center recovery error (5 estimators, known gaussian center) ===")
print(piv.to_string())
print("\nmean abs error per method:")
print(df.groupby("method")["abs_err"].mean().round(3).sort_values().to_string())
