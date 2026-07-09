"""3D vascular: fluorostats (threshold configs incl. auto/consensus) vs the
VesselExpress SOFTWARE's own segmentation (Zenodo 6025935 Binary GT).
Also scikit-image (raw) as a named distinct-library baseline."""
import os, glob, sys, numpy as np, tifffile, pandas as pd
sys.path.insert(0, os.environ["WORK"]+"/fluorostats_src")
from skimage import filters
from skimage.morphology import remove_small_objects
from fluorostats.segment import binarize, choose_threshold_method
from fluorostats.metrics_3d import volume_fraction
BASE=os.environ["WORK"]+"/vesselexpress/data"; OUT=os.environ["WORK"]+"/vesselexpress"
def dice(p,g): p=p>0;g=g>0;s=p.sum()+g.sum();return float(2*(p&g).sum()/s) if s else 1.0
def vf(m):
    r=volume_fraction(m); return float(r["volume_fraction"]) if isinstance(r,dict) else float(r)
# fluorostats configs (all in-library thresholding options)
FS={"fluorostats(otsu)":"otsu","fluorostats(li)":"li","fluorostats(triangle)":"triangle",
    "fluorostats(auto)":"auto","fluorostats(consensus)":"consensus"}
raws=sorted(glob.glob(BASE+"/**/Raw/*.tif",recursive=True))
print("found",len(raws),"raw volumes vs VesselExpress-software GT",flush=True)
rows=[]
for rp in raws:
    name=os.path.basename(rp)
    cand=glob.glob(os.path.dirname(rp).replace("/Raw","/Binary")+"/Binary_"+name.replace(".tif","*"))
    if not cand: print("no VE-binary for",name,flush=True); continue
    raw=tifffile.imread(rp).astype(np.float32); gtb=tifffile.imread(cand[0])>0
    row={"volume":name,"VE_gt_vf":round(float(gtb.mean()),5)}
    for label,m in FS.items():
        msk=binarize(raw,method=m,min_size=50); row["dice_"+label]=round(dice(msk,gtb),4)
    # scikit-image as a distinct-library software baseline (raw skimage, not fluorostats)
    row["dice_scikit-image(otsu)"]=round(dice(remove_small_objects(raw>filters.threshold_otsu(raw),50),gtb),4)
    row["auto_chose"]=choose_threshold_method(raw)["method"]
    rows.append(row); print("done",name,"auto->",row["auto_chose"],flush=True)
df=pd.DataFrame(rows); df.to_csv(OUT+"/vesselexpress_bench.csv",index=False)
print("\n=== 3D vascular Dice vs VesselExpress-software segmentation (n=%d) ==="%len(df))
mcols=[c for c in df.columns if c.startswith("dice_")]
print(df[["volume","auto_chose"]+mcols].to_string(index=False))
print("\nMean Dice vs VesselExpress GT:")
print(df[mcols].mean().round(3).sort_values(ascending=False).to_string())
print("\nauto picked:", df.auto_chose.value_counts().to_dict())
