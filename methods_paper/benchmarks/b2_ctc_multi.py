"""CTC 3D foreground segmentation — fluorostats vs 4+ threshold methods."""
import glob, re, numpy as np, pandas as pd, tifffile
from pathlib import Path
from skimage import filters
from fluorostats.preprocess import denoise, background_subtract
from fluorostats.segment import binarize
CTC=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/CTC")
RES=Path(__file__).resolve().parent/"results"
def dice(p,g): 
    p=p>0;g=g>0; s=p.sum()+g.sum(); return 2*(p&g).sum()/s if s else 1.0
THR={"Otsu_1979":filters.threshold_otsu,"Li_1993":filters.threshold_li,
     "Isodata_1978":filters.threshold_isodata,"Triangle_1977":filters.threshold_triangle,
     "Yen_1995":filters.threshold_yen}
def fs_seg(v):
    s=denoise(v.astype(np.float32),sigma=1.0);s=background_subtract(s,radius=25)
    return binarize(s,method="otsu",threshold_scale=0.9,min_size=50)
def pairs(ds):
    out=[]
    for sq in ("01","02"):
        for gp in sorted(glob.glob(str(CTC/ds/f"{sq}_GT"/"SEG"/"*.tif"))):
            nm=Path(gp).name
            m3=re.search(r"man_seg_(\d+)_(\d+)\.tif",nm); m2=re.search(r"man_seg0*(\d+)\.tif",nm)
            if m3:
                t,z=int(m3.group(1)),int(m3.group(2)); v=CTC/ds/sq/f"t{t:03d}.tif"
                if v.exists():
                    vol=tifffile.imread(v)
                    if z<vol.shape[0]: out.append((vol[z],tifffile.imread(gp)))
            elif m2:
                t=int(m2.group(1)); v=CTC/ds/sq/f"t{t:03d}.tif"
                if v.exists(): out.append((tifffile.imread(v),tifffile.imread(gp)))
    return out
rows=[]
for ds,top in [("Fluo-C3DH-A549",0.908),("Fluo-N3DH-CHO",0.925)]:
    P=pairs(ds)
    scores={m:[] for m in list(THR)+["fluorostats"]}
    for img,gt in P:
        for name,fn in THR.items():
            try: scores[name].append(dice(img>fn(img.astype(np.float32)),gt))
            except Exception: scores[name].append(0.0)
        scores["fluorostats"].append(dice(fs_seg(img),gt))
    for m,v in scores.items():
        rows.append({"dataset":ds,"method":m,"mean_dice":round(float(np.mean(v)),3),"ctc_top_SEG":top})
df=pd.DataFrame(rows); df.to_csv(RES/"b2_ctc_multi.csv",index=False)
print("=== CTC 3D foreground Dice — fluorostats vs threshold methods ===")
for ds in df.dataset.unique():
    print(f"\n{ds} (CTC top instance SEG={df[df.dataset==ds].ctc_top_SEG.iloc[0]}, context):")
    print(df[df.dataset==ds][["method","mean_dice"]].sort_values("mean_dice",ascending=False).to_string(index=False))
