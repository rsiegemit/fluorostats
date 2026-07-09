"""fluorostats vs VesselExpress SOFTWARE — vessel volume-fraction agreement
(the vascular density readout both produce; skeleton length omitted — skeletonizing
full 250MB dense light-sheet volumes is intractable, an honest scope note)."""
import os, glob, sys, numpy as np, tifffile, pandas as pd
sys.path.insert(0, os.environ["WORK"]+"/fluorostats_src")
from fluorostats.segment import binarize
from fluorostats.metrics_3d import volume_fraction
from fluorostats.agreement import lins_ccc
from scipy.stats import spearmanr
BASE=os.environ["WORK"]+"/vesselexpress/data"; OUT=os.environ["WORK"]+"/vesselexpress"
def vf(m):
    r=volume_fraction(m); return float(r["volume_fraction"]) if isinstance(r,dict) else float(r)
rows=[]
for rp in sorted(glob.glob(BASE+"/**/Raw/*.tif",recursive=True)):
    name=os.path.basename(rp)
    binc=glob.glob(os.path.dirname(rp).replace("/Raw","/Binary")+"/Binary_"+name.replace(".tif","*"))
    if not binc: continue
    ve=tifffile.imread(binc[0])>0
    fs=binarize(tifffile.imread(rp).astype(np.float32),method="auto",min_size=50)
    rows.append({"volume":name,"VesselExpress_VF":round(vf(ve),5),"fluorostats_VF":round(vf(fs),5)})
    print("done",name,flush=True)
df=pd.DataFrame(rows); df.to_csv(OUT+"/ve_metrics.csv",index=False)
print("\n=== fluorostats vs VesselExpress SOFTWARE — vessel volume fraction (n=%d) ==="%len(df))
print(df.to_string(index=False))
a,b=df.VesselExpress_VF.values,df.fluorostats_VF.values
print(f"\nVesselExpress mean VF={a.mean():.4f}  fluorostats mean VF={b.mean():.4f}")
print(f"agreement: CCC={lins_ccc(a,b):.3f}  Spearman={spearmanr(a,b).statistic:.3f}  mean|diff|={np.abs(a-b).mean():.4f}")
