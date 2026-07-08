"""preprocess.background_subtract vs 4 background-correction methods (uneven illumination)."""
import glob, numpy as np, pandas as pd, tifffile
from pathlib import Path
from skimage import filters, restoration
from scipy import ndimage as ndi
from fluorostats.preprocess import background_subtract
BASE=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/BBBC024")
RES=Path(__file__).resolve().parent/"results"
def dice(p,g): p=p>0;g=g>0;s=p.sum()+g.sum();return 2*(p&g).sum()/s if s else 1.0
finals=sorted(glob.glob(str(BASE/"image-final_*.tif")))[:3]
def add_gradient(v):  # simulate uneven illumination: smooth additive background ramp
    z,y,x=v.shape
    gy=np.linspace(0,1,y)[None,:,None]; gx=np.linspace(0,1,x)[None,None,:]
    return v + (gy*gx)*v.max()*0.6
methods={
 "none": lambda v:v,
 "fluorostats(white_tophat_r15)": lambda v:background_subtract(v,radius=15),
 "fluorostats(white_tophat_r45)": lambda v:background_subtract(v,radius=45),
 "gaussian_highpass": lambda v:np.clip(v-ndi.gaussian_filter(v,25),0,None),
 "morphological_opening": lambda v:np.clip(v-ndi.grey_opening(v,size=(1,15,15)),0,None),
 "rolling_ball_r15": lambda v:np.clip(v-np.stack([restoration.rolling_ball(v[i],radius=15) for i in range(v.shape[0])]),0,None),
}
rows=[]
for fp in finals:
    gt=tifffile.imread(fp.replace("image-final_","image-labels_"))
    _v=tifffile.imread(fp).astype(np.float32); z0=_v.shape[0]//2-8
    v0=add_gradient(_v[z0:z0+16]); gt=gt[z0:z0+16]
    for name,fn in methods.items():
        try:
            d=fn(v0); rows.append({"method":name,"dice":dice(d>filters.threshold_otsu(d),gt)})
        except Exception as e: rows.append({"method":name,"dice":0.0})
df=pd.DataFrame(rows).groupby("method")["dice"].mean().round(3).sort_values(ascending=False)
df.to_csv(RES/"b_background_subtract.csv")
print("=== background correction: seg Dice under uneven illumination (BBBC024 mid-slab, n=3) ===")
print(df.to_string())
