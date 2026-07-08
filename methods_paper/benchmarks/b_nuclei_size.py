"""Per-nucleus size recovery — fluorostats vs 4+ threshold methods (BBBC024 GT)."""
import glob, numpy as np, pandas as pd, tifffile
from pathlib import Path
from skimage import filters
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
from fluorostats.objects import equivalent_diameters_um, label_3d
BASE=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/BBBC024")
RES=Path(__file__).resolve().parent/"results"; VOX=(1.0,1.0,1.0)
THR={"Otsu_1979":filters.threshold_otsu,"Li_1993":filters.threshold_li,
     "Isodata_1978":filters.threshold_isodata,"Triangle_1977":filters.threshold_triangle}
finals=sorted(glob.glob(str(BASE/"image-final_*.tif")))[:6]
gt_all=[]; per_method={m:[] for m in list(THR)+["fluorostats(Otsu+CC)"]}
for fp in finals:
    gp=fp.replace("image-final_","image-labels_")
    if not Path(gp).exists(): continue
    vol=tifffile.imread(fp).astype(np.float32); gt=tifffile.imread(gp).astype(np.int32)
    gt_all.extend(equivalent_diameters_um(gt,VOX).tolist())
    for name,fn in THR.items():
        try:
            lab,_=ndi.label(remove_small_objects(vol>fn(vol),30))
            per_method[name].extend(equivalent_diameters_um(lab,VOX).tolist())
        except Exception: pass
    lab,_=label_3d(remove_small_objects(vol>filters.threshold_otsu(vol),30),min_size=30)
    per_method["fluorostats(Otsu+CC)"].extend(equivalent_diameters_um(lab,VOX).tolist())
gt_med=float(np.median(gt_all)); rows=[]
for m,v in per_method.items():
    med=float(np.median(v)) if v else float('nan')
    rows.append({"method":m,"median_diam_um":round(med,2),"GT_median":round(gt_med,2),
                 "pct_error":round(abs(med-gt_med)/gt_med*100,1)})
df=pd.DataFrame(rows).sort_values("pct_error"); df.to_csv(RES/"b_nuclei_size.csv",index=False)
print("=== Per-nucleus diameter recovery vs BBBC024 GT (%.1f um), many methods ==="%gt_med)
print(df.to_string(index=False))
