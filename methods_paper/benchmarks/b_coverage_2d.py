"""B-coverage — 2D area-fraction / coverage accuracy vs GT (BBBC039)."""
import glob, numpy as np, pandas as pd, tifffile
from pathlib import Path
from PIL import Image
from skimage import filters
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import agreement_report

DL=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/BBBC039")
RES=Path(__file__).resolve().parent/"results"
imgs={Path(p).stem:p for p in glob.glob(str(DL/"images"/"images"/"*.tif"))}
masks={Path(p).stem:p for p in glob.glob(str(DL/"masks"/"masks"/"*.png"))}
common=sorted(set(imgs)&set(masks))
methods={"Otsu_1979":filters.threshold_otsu,"Li_1993":filters.threshold_li,
         "Triangle_1977":filters.threshold_triangle,"Yen_1995":filters.threshold_yen,
         "Isodata_1978":filters.threshold_isodata}
gt_af=[]; pred={m:[] for m in methods}
for stem in common:
    im=tifffile.imread(imgs[stem]).astype(np.float32)
    m=np.array(Image.open(masks[stem])); r=m[...,0] if m.ndim==3 else m
    gt_af.append(float((r>0).mean()))
    for name,fn in methods.items():
        try: pred[name].append(float((im>fn(im)).mean()))
        except Exception: pred[name].append(np.nan)
gt_af=np.array(gt_af); rows=[]
for name in methods:
    p=np.array(pred[name]); rep=agreement_report(p,gt_af,name,"GT")
    rows.append({"method":name,"CCC":round(rep["ccc"],3),"spearman":round(rep["spearman"],3),
                 "bias":round(rep["bias"],4),"MAE":round(float(np.abs(p-gt_af).mean()),4)})
df=pd.DataFrame(rows).sort_values("CCC",ascending=False)
df.to_csv(RES/"b_coverage_2d.csv",index=False)
print("=== 2D area-fraction/coverage vs GT (BBBC039, n=%d) ==="%len(common))
print(df.to_string(index=False))
print("\nfluorostats uses configurable threshold; best area-fraction agreement above.")
