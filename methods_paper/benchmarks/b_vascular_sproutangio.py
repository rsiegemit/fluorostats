"""SproutAngio VEGF dose-response — MANY methods (volume fraction, fast)."""
import glob, re, numpy as np, pandas as pd, czifile
from pathlib import Path
from skimage import filters
from skimage.morphology import remove_small_objects
from scipy import stats as sps
from fluorostats.preprocess import denoise, background_subtract
from fluorostats.segment import binarize
SA=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/sproutangio")
RES=Path(__file__).resolve().parent/"results"
THR={"Otsu_1979":filters.threshold_otsu,"Li_1993":filters.threshold_li,
     "Isodata_1978":filters.threshold_isodata,"Triangle_1977":filters.threshold_triangle}
recs=[]
for fp in sorted(glob.glob(str(SA/"*.czi"))):
    grp=int(re.search(r"group(\d)",Path(fp).name).group(1))
    fa=np.squeeze(czifile.imread(fp)).astype(np.float32)[0]
    row={"group":grp}
    for name,fn in THR.items():
        try: row[name]=float(remove_small_objects(fa>fn(fa),100).mean())
        except Exception: row[name]=np.nan
    sm=background_subtract(denoise(fa,sigma=1.0),radius=15)
    row["fluorostats"]=float(binarize(sm,method="otsu",threshold_scale=0.9,min_size=100).mean())
    recs.append(row)
df=pd.DataFrame(recs); df.to_csv(RES/"b_vascular_sproutangio_multi.csv",index=False)
methods=[c for c in df.columns if c!="group"]
print("=== SproutAngio VEGF dose-response: volume fraction by method ===")
agg=df.groupby("group")[methods].mean().round(4); print(agg.to_string())
print("\nSpearman(VF vs VEGF dose) per method — does each detect the sprouting response?")
for m in methods:
    print(f"  {m:22s} rho={sps.spearmanr(df.group,df[m]).statistic:+.3f}")
