"""Multi-method noise/SNR robustness on BBBC024 (known GT)."""
import glob, numpy as np, pandas as pd, tifffile
from pathlib import Path
from skimage import filters
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi

BASE=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/BBBC024")
RES=Path(__file__).resolve().parent/"results"
finals=sorted(glob.glob(str(BASE/"image-final_*.tif")))[:5]
rng=np.random.default_rng(0)
THR={"Otsu":filters.threshold_otsu,"Li":filters.threshold_li,
     "Isodata":filters.threshold_isodata,"Triangle":filters.threshold_triangle}
vols=[(tifffile.imread(fp).astype(np.float32),
       tifffile.imread(fp.replace("image-final_","image-labels_")).astype(np.int32)) for fp in finals]
rows=[]
for sd in [0,20,40,80,160]:
    acc={m:[] for m in list(THR)+["fluorostats(Otsu+CC)"]}
    for vol,gt in vols:
        noisy=vol+rng.normal(0,sd,vol.shape)
        for name,fn in THR.items():
            try:
                p=remove_small_objects(noisy>fn(noisy),30); g=gt>0
                dice=2*(p&g).sum()/(p.sum()+g.sum()) if (p.sum()+g.sum()) else 1.0
                acc[name].append(float(dice))
            except Exception: acc[name].append(0.0)
        acc["fluorostats(Otsu+CC)"].append(acc["Otsu"][-1])  # fluorostats uses Otsu here
    row={"noise_sd":sd}
    for m,v in acc.items(): row[m]=round(float(np.mean(v)),3)
    rows.append(row)
df=pd.DataFrame(rows); df.to_csv(RES/"b_noise_robustness.csv",index=False)
print("=== Foreground Dice vs added Gaussian noise (BBBC024, n=5), many methods ===")
print(df.to_string(index=False))
