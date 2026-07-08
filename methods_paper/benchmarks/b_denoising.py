"""Denoising/preprocessing comparison — fluorostats (gaussian) vs 4 alternatives."""
import glob, numpy as np, pandas as pd, tifffile
from pathlib import Path
from skimage import filters, restoration
from skimage.filters import median as med_filt
from scipy import ndimage as ndi
from fluorostats.preprocess import denoise as fs_denoise
BASE=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/BBBC024")
RES=Path(__file__).resolve().parent/"results"
def dice(p,g): p=p>0;g=g>0;s=p.sum()+g.sum();return 2*(p&g).sum()/s if s else 1.0
finals=sorted(glob.glob(str(BASE/"image-final_*.tif")))[:5]
rng=np.random.default_rng(0)
denoisers={
 "none":lambda v:v,
 "fluorostats(gaussian σ1)":lambda v:fs_denoise(v,sigma=1.0),
 "median_3":lambda v:med_filt(v,footprint=np.ones((3,3,3))),
 "gaussian_σ2":lambda v:ndi.gaussian_filter(v,2.0),
 "TV_chambolle":lambda v:restoration.denoise_tv_chambolle(v,weight=0.1),
}
rows=[]
for sd in [40,80,160]:
    acc={m:[] for m in denoisers}
    for fp in finals:
        gt=tifffile.imread(fp.replace("image-final_","image-labels_"))
        vol=tifffile.imread(fp).astype(np.float32)+rng.normal(0,sd,tifffile.imread(fp).shape)
        for name,fn in denoisers.items():
            try:
                d=fn(vol.astype(np.float32)); acc[name].append(dice(d>filters.threshold_otsu(d),gt))
            except Exception: acc[name].append(0.0)
    row={"noise_sd":sd}
    for m,v in acc.items(): row[m]=round(float(np.mean(v)),3)
    rows.append(row)
df=pd.DataFrame(rows); df.to_csv(RES/"b_denoising.csv",index=False)
print("=== Denoising method effect on segmentation Dice (BBBC024+noise, n=5) ===")
print(df.to_string(index=False))
