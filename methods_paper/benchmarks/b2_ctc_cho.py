"""B2 3D — fluorostats foreground accuracy on CTC Fluo-N3DH-CHO (per-slice GT)."""
import glob, re, numpy as np, pandas as pd, tifffile
from pathlib import Path
from fluorostats.preprocess import denoise, background_subtract
from fluorostats.segment import binarize
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import agreement_report
CTC=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/CTC/Fluo-N3DH-CHO")
RES=Path(__file__).resolve().parent/"results"
def seg(v):
    s=denoise(v.astype(np.float32),sigma=1.0); s=background_subtract(s,radius=25)
    return binarize(s,method="otsu",threshold_scale=0.9,min_size=50)
rows=[]
for sq in ("01","02"):
    for gp in sorted(glob.glob(str(CTC/f"{sq}_GT"/"SEG"/"man_seg_*.tif"))):
        m=re.search(r"man_seg_(\d+)_(\d+)\.tif",Path(gp).name)
        if not m: continue
        t,s=int(m.group(1)),int(m.group(2))
        ip=CTC/sq/f"t{t:03d}.tif"
        if not ip.exists(): continue
        vol=tifffile.imread(ip); 
        if s>=vol.shape[0]: continue
        img=vol[s]; gt=tifffile.imread(gp)
        mask=seg(img); p=mask>0; g=gt>0
        inter=(p&g).sum(); union=(p|g).sum()
        dice=2*inter/(p.sum()+g.sum()) if (p.sum()+g.sum()) else 1.0
        jac=inter/union if union else 1.0
        rows.append({"seq":sq,"t":t,"z":s,"dice":dice,"jaccard":jac,
                     "fs_af":float(p.mean()),"gt_af":float(g.mean())})
df=pd.DataFrame(rows); df.to_csv(RES/"b2_ctc_cho.csv",index=False)
vf=agreement_report(df["fs_af"],df["gt_af"],"fluorostats","GT")
print("=== fluorostats vs CTC gold GT (Fluo-N3DH-CHO, %d annotated slices) ==="%len(df))
print("mean Dice=%.3f  mean Jaccard=%.3f  area-frac CCC=%.3f  Spearman=%.3f"%(
    df.dice.mean(),df.jaccard.mean(),vf["ccc"],vf["spearman"]))
print("Context: CTC top instance SEG for CHO = 0.925 (different metric; fluorostats is semantic).")
pd.DataFrame([{"dataset":"Fluo-N3DH-CHO","n_slices":len(df),"mean_dice":round(df.dice.mean(),3),
  "mean_jaccard":round(df.jaccard.mean(),3),"af_CCC":round(vf["ccc"],3),"ctc_top_SEG":0.925}]).to_csv(
  RES/"b2_ctc_cho_summary.csv",index=False)
