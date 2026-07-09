"""Recompute per-image instance F1@0.5 for threshold methods + fluorostats on BBBC039 (local)."""
import glob, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from skimage import filters
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from fluorostats.validate import instance_f1
from fluorostats.objects import label_3d, watershed_split
D=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/BBBC039")
IMG=D/"images"/"images"; MSK=D/"masks"/"masks"
RES=Path("results")
def gt_instances(stem):
    m=np.array(Image.open(MSK/f"{stem}.png")); r=m[...,0] if m.ndim==3 else m
    interior=(r==1); boundary=(r>=2); seeds,n=ndi.label(interior)
    if n==0: return np.zeros(r.shape,int)
    return watershed(boundary.astype(np.uint8),seeds,mask=(r>0))
THR={"Otsu (1979)":filters.threshold_otsu,"Li (1993)":filters.threshold_li,
     "Isodata (1978)":filters.threshold_isodata,"Triangle (1977)":filters.threshold_triangle,
     "Yen (1995)":filters.threshold_yen,"Mean":filters.threshold_mean,"Minimum":filters.threshold_minimum}
def cc(mask): return ndi.label(remove_small_objects(mask,20))[0]
files=sorted(glob.glob(str(IMG/"*.tif")))[:100]
rows=[]
for f in files:
    stem=Path(f).stem; 
    try: gt=gt_instances(stem)
    except Exception: continue
    im=np.asarray(Image.open(f)).astype(np.float32)
    rec={"image":stem}
    for name,fn in THR.items():
        try: rec[name]=instance_f1(cc(im>fn(im)),gt)["f1"]
        except Exception: rec[name]=0.0
    otsu=remove_small_objects(im>filters.threshold_otsu(im),20)
    rec["fluorostats (Otsu+CC)"]=instance_f1(label_3d(otsu,min_size=20)[0],gt)["f1"]
    rec["Watershed (1991)"]=instance_f1(watershed_split(otsu,min_size=20,min_distance=4)[0],gt)["f1"]
    rows.append(rec)
df=pd.DataFrame(rows); df.to_csv(RES/"b2_nuclei_methods_perimage.csv",index=False)
print("cached per-image F1 for",len(df),"images x",len(df.columns)-1,"methods")
print(df.drop(columns=["image"]).mean().round(3).sort_values(ascending=False).to_string())
