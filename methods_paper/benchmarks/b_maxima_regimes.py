"""When is maxima counting better? Sweep cell SIZE and NOISE (known true count)."""
import numpy as np, pandas as pd
from pathlib import Path
from skimage.morphology import remove_small_objects
from skimage.draw import disk
from scipy import ndimage as ndi
from fluorostats.objects import count_local_maxima, watershed_split, label_3d
RES=Path(__file__).resolve().parent/"results"
rng=np.random.default_rng(0)
def field(n_cells, radius, noise, size=400, sep=None):
    img=np.zeros((size,size),np.float32)
    g=int(np.ceil(np.sqrt(n_cells))); xs=np.linspace(radius+2,size-radius-2,g)
    pts=[(y,x) for y in xs for x in xs][:n_cells]
    for (y,x) in pts:
        rr,cc=disk((y,x),radius,shape=img.shape); img[rr,cc]=255.0
    if noise>0: img=np.clip(img+rng.normal(0,noise,img.shape),0,None)
    return img
def cnt_maxima(img,md): return count_local_maxima(img,min_distance=md,threshold_rel=0.3)["count"]
def cnt_ws(img):
    m=remove_small_objects(img>img.max()*0.3,5); return watershed_split(m,min_size=5,min_distance=4)[1]
def cnt_cc(img):
    m=remove_small_objects(img>img.max()*0.3,5); return label_3d(m,min_size=5)[1]
rows=[]
N=100
# regime 1: vary cell radius (well-separated, no noise)
for r in [2,4,8,14,22]:
    img=field(N,r,0)
    md=max(3,int(r*1.2))
    rows.append({"regime":"cell_radius","param":r,"true":N,
                 "maxima":cnt_maxima(img,md),"watershed":cnt_ws(img),"CC":cnt_cc(img)})
# regime 2: vary noise (small cells r=3)
for nz in [0,20,60,120,200]:
    img=field(N,3,nz)
    rows.append({"regime":"noise_sd","param":nz,"true":N,
                 "maxima":cnt_maxima(img,3),"watershed":cnt_ws(img),"CC":cnt_cc(img)})
df=pd.DataFrame(rows)
for m in ["maxima","watershed","CC"]: df[m+"_err"]=(df[m]-df["true"]).abs()
df.to_csv(RES/"b_maxima_regimes.csv",index=False)
print("=== Count error by regime (true=100 cells); lower=better ===")
for reg in ["cell_radius","noise_sd"]:
    d=df[df.regime==reg]; print(f"\n-- {reg} --")
    print(d[["param","maxima","watershed","CC"]].to_string(index=False))
print("\nMean abs count error:")
print(df.groupby("regime")[["maxima_err","watershed_err","CC_err"]].mean().round(1).to_string())
