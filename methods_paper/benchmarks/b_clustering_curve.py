"""Clustering degradation curve — MANY methods vs nuclear overlap (BBBC024)."""
import glob, numpy as np, pandas as pd, tifffile
from pathlib import Path
from skimage import filters
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
from fluorostats.objects import label_3d, watershed_split
from fluorostats.validate import instance_f1
D=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads")
RES=Path(__file__).resolve().parent/"results"
LEVELS={"c00":"BBBC024","c25":"BBBC024_c25","c50":"BBBC024_c50","c75":"BBBC024_c75"}
MS=30; NVOL=5
THR={"Otsu_1979":filters.threshold_otsu,"Li_1993":filters.threshold_li,
     "Isodata_1978":filters.threshold_isodata,"Triangle_1977":filters.threshold_triangle}
def lab_cc(vol,t): return ndi.label(remove_small_objects(vol>t,MS))[0]
rows=[]
for lvl,sub in LEVELS.items():
    base=D/sub
    finals=sorted(glob.glob(str(base/"**/image-final_*.tif"),recursive=True))[:NVOL]
    acc={m:[] for m in list(THR)+["fluorostats_CC","fluorostats_watershed"]}
    for fp in finals:
        gp=fp.replace("image-final_","image-labels_")
        if not Path(gp).exists(): continue
        vol=tifffile.imread(fp).astype(np.float32); gt=tifffile.imread(gp).astype(np.int32)
        for name,fn in THR.items():
            try: acc[name].append(instance_f1(lab_cc(vol,fn(vol)),gt)["f1"])
            except Exception: acc[name].append(0.0)
        otsu=vol>filters.threshold_otsu(vol)
        cc,_=label_3d(remove_small_objects(otsu,MS),min_size=MS)
        ws,_=watershed_split(remove_small_objects(otsu,MS),min_size=MS,min_distance=6)
        acc["fluorostats_CC"].append(instance_f1(cc,gt)["f1"])
        acc["fluorostats_watershed"].append(instance_f1(ws,gt)["f1"])
    for m,v in acc.items():
        rows.append({"clustering":lvl,"method":m,"mean_F1":round(float(np.mean(v)),3) if v else None})
    print(f"done {lvl}",flush=True)
df=pd.DataFrame(rows); piv=df.pivot(index="method",columns="clustering",values="mean_F1")
piv.to_csv(RES/"b_clustering_curve.csv")
print("=== Instance F1 vs nuclear clustering (BBBC024, many methods) ===")
print(piv.to_string())
