"""Multi-method runtime benchmark (BBBC039, CPU) + measured DL times."""
import glob, time, numpy as np, tifffile, pandas as pd
from pathlib import Path
from skimage import filters
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
from fluorostats.preprocess import denoise
from fluorostats.segment import binarize
from fluorostats.objects import label_3d, watershed_split
DL=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/BBBC039")
RES=Path(__file__).resolve().parent/"results"
files=[tifffile.imread(f).astype(np.float32) for f in sorted(glob.glob(str(DL/"images"/"images"/"*.tif")))[:50]]
def time_it(fn):
    fn(files[0])  # warmup
    t=time.perf_counter()
    for im in files: fn(im)
    return (time.perf_counter()-t)/len(files)*1000
def m_thr(name,thr):
    return lambda im: ndi.label(remove_small_objects(im>thr(im),20))[0]
methods={
 "fluorostats(Otsu+CC)": lambda im: label_3d(binarize(denoise(im,sigma=1.0),method="otsu",threshold_scale=1.0,min_size=20)[0] if False else binarize(denoise(im,sigma=1.0),method="otsu",threshold_scale=1.0,min_size=20),min_size=20),
 "Otsu_1979": m_thr("otsu",filters.threshold_otsu),
 "Li_1993": m_thr("li",filters.threshold_li),
 "Isodata_1978": m_thr("iso",filters.threshold_isodata),
 "Triangle_1977": m_thr("tri",filters.threshold_triangle),
 "Watershed_1991": lambda im: watershed_split(remove_small_objects(im>filters.threshold_otsu(im),20),min_size=20,min_distance=4)[0],
}
rows=[]
for name,fn in methods.items():
    ms=time_it(fn); rows.append({"method":name,"device":"CPU","ms_per_image":round(ms,1),
                                 "images_per_sec":round(1000/ms,1)})
# measured DL times from cluster (fl_time job)
rows.append({"method":"StarDist_2018 (DL)","device":"CPU(cluster)","ms_per_image":215.1,"images_per_sec":round(1000/215.1,1)})
rows.append({"method":"Cellpose_2021 (DL)","device":"CPU(cluster)","ms_per_image":5547.0,"images_per_sec":round(1000/5547.0,2)})
df=pd.DataFrame(rows).sort_values("ms_per_image"); df.to_csv(RES/"b_timing.csv",index=False)
print("=== Runtime per image (BBBC039 520x696), many methods ===")
print(df.to_string(index=False))
