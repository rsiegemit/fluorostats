"""Bootstrap 95% CIs on fluorostats-vs-DL F1 gaps (paired, BBBC039 n=200)."""
import numpy as np, pandas as pd
from pathlib import Path
RES=Path(__file__).resolve().parent/"results"
fs=pd.read_csv(RES/"b2_nuclei_fluorostats_f1.csv")[["image","fs_f1"]]
sd=pd.read_csv(RES/"stardist_eval.csv")[["image","stardist_f1"]]
cp=pd.read_csv(RES/"cellpose_eval.csv")[["image","cellpose_f1"]]
df=fs.merge(sd,on="image").merge(cp,on="image")
print(f"matched {len(df)} images")
rng=np.random.default_rng(0)
def boot_ci(x, reps=10000):
    n=len(x); idx=rng.integers(0,n,(reps,n)); means=x[idx].mean(1)
    return float(x.mean()), float(np.percentile(means,2.5)), float(np.percentile(means,97.5))
rows=[]
for name,col in [("fluorostats","fs_f1"),("StarDist","stardist_f1"),("Cellpose","cellpose_f1")]:
    m,lo,hi=boot_ci(df[col].values); rows.append({"method":name,"mean_F1":round(m,4),
        "CI_low":round(lo,4),"CI_high":round(hi,4)})
# paired differences
for name,col in [("fluorostats - StarDist","stardist_f1"),("fluorostats - Cellpose","cellpose_f1")]:
    d=(df["fs_f1"]-df[col]).values; m,lo,hi=boot_ci(d)
    sig = "YES (excludes 0)" if (lo>0 or hi<0) else "no (CI spans 0)"
    rows.append({"method":name,"mean_F1":round(m,4),"CI_low":round(lo,4),
                 "CI_high":round(hi,4),"diff_significant":sig})
out=pd.DataFrame(rows); out.to_csv(RES/"b_dl_ci.csv",index=False)
print(f"=== BBBC039 F1 with bootstrap 95pct CIs (n={len(df)}, 10k resamples) ===")
print(out.to_string(index=False))
