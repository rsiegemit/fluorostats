"""MASTER timing: every fluorostats metric AND every benchmark comparator,
timed on the same data per operation-class. group = fluorostats | comparator."""
import glob, time, numpy as np, pandas as pd, tifffile
from pathlib import Path
from skimage import filters, restoration
from skimage.morphology import skeletonize, medial_axis, thin, remove_small_objects
from skimage.filters import median as med_filt
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy import stats as sps
from fluorostats import (metrics_3d, metrics_2d, morphometry, objects, skeleton,
                         viability, stats, agreement, validate, preprocess, segment)
D=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads")
RES=Path(__file__).resolve().parent/"results"
def tm(fn,reps=5):
    fn()
    t=time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter()-t)/reps*1000
# data
vol=tifffile.imread(sorted(glob.glob(str(D/"BBBC024/image-final_*.tif")))[0]).astype(np.float32)
gt=tifffile.imread(sorted(glob.glob(str(D/"BBBC024/image-labels_*.tif")))[0]).astype(np.int32)
img2=tifffile.imread(sorted(glob.glob(str(D/"BBBC039/images/images/*.tif")))[0]).astype(np.float32)
slab=vol[vol.shape[0]//2-8:vol.shape[0]//2+8]
mask=remove_small_objects(vol>filters.threshold_otsu(vol),20); lab=ndi.label(mask)[0]
sk=skeletonize(vol[vol.shape[0]//2]>filters.threshold_otsu(vol[vol.shape[0]//2]))
kf=tifffile.imread([x for x in glob.glob(str(D/"kerkhoff_lda/synthetic/**/*.tif"),recursive=True) if "700 Live; 700 Dead-1" in x][0]).astype(np.float32)
live,dead=kf[1],kf[0]
a=np.random.default_rng(0).normal(0,1,200); b=np.random.default_rng(1).normal(0.5,1,200)
cent=objects.object_centroids(lab); pts=cent[:,1:] if cent.shape[1]==3 else cent
H=W=vol.shape[1]
sdf=pd.DataFrame({"grp":["A"]*24+["B"]*24,"st":(["s1"]*12+["s2"]*12)*2,"v":np.random.default_rng(3).normal(0,1,48)})
def clark_evans(p):
    d,_=cKDTree(p).query(p,k=2); return d[:,1].mean()/(0.5/np.sqrt(len(p)/(H*W)))
def morisita(p,g=8):
    h=np.histogram2d(p[:,0],p[:,1],bins=g)[0].ravel(); n=h.sum(); return len(h)*(h*(h-1)).sum()/(n*(n-1))
def quadrat(p,g=8):
    h=np.histogram2d(p[:,0],p[:,1],bins=g)[0].ravel(); return h.var()/h.mean()
def ripley(p,r=40):
    d=cKDTree(p); return d.query_pairs(r).__len__()
def lacunarity(m,box=16):
    s=np.add.reduceat(np.add.reduceat(m.astype(float),np.arange(0,m.shape[0],box),0),np.arange(0,m.shape[1],box),1)
    return 1+s.var()/(s.mean()**2+1e-9)
def perc(l):  # spanning component check
    for i in range(1,l.max()+1):
        ys,xs=np.where(l[l.shape[0]//2]==i) if l.ndim==3 else np.where(l==i)
    return int(l.max()>0)
T=[
 # ---- segmentation / preprocess class (3D vol) ----
 ("fluorostats","segment.binarize(otsu) 3D", lambda: segment.binarize(vol,method="otsu",min_size=20)),
 ("comparator","threshold_otsu 3D", lambda: (vol>filters.threshold_otsu(vol))),
 ("comparator","threshold_li 3D", lambda: (vol>filters.threshold_li(vol))),
 ("comparator","threshold_isodata 3D", lambda: (vol>filters.threshold_isodata(vol))),
 ("comparator","threshold_triangle 3D", lambda: (vol>filters.threshold_triangle(vol))),
 ("comparator","threshold_yen 3D", lambda: (vol>filters.threshold_yen(vol))),
 ("comparator","threshold_mean 3D", lambda: (vol>filters.threshold_mean(vol))),
 ("fluorostats","preprocess.denoise(gauss) 3D", lambda: preprocess.denoise(vol,sigma=1.0)),
 ("comparator","median_filter 3D", lambda: med_filt(vol,footprint=np.ones((3,3,3)))),
 ("comparator","gaussian_sigma2 3D", lambda: ndi.gaussian_filter(vol,2.0)),
 ("fluorostats","preprocess.background_subtract 3D-slab", lambda: preprocess.background_subtract(slab,radius=15)),
 ("comparator","rolling_ball 3D-slab", lambda: np.stack([restoration.rolling_ball(slab[i],radius=15) for i in range(slab.shape[0])])),
 ("comparator","morph_opening 3D-slab", lambda: ndi.grey_opening(slab,size=(1,15,15))),
 ("fluorostats","objects.label_3d", lambda: objects.label_3d(mask,min_size=20)),
 ("fluorostats","objects.watershed_split", lambda: objects.watershed_split(mask,min_size=20,min_distance=4)),
 ("comparator","skimage watershed 3D", lambda: __import__("skimage.segmentation",fromlist=["watershed"]).watershed(-ndi.distance_transform_edt(mask),mask=mask)),
 # ---- 3D metrics ----
 ("fluorostats","metrics_3d.volume_fraction", lambda: metrics_3d.volume_fraction(mask)),
 ("fluorostats","metrics_3d.connectivity_metrics", lambda: metrics_3d.connectivity_metrics(lab)),
 ("comparator","euler_number(skimage) 3D", lambda: __import__("skimage.measure",fromlist=["euler_number"]).euler_number(mask)),
 ("fluorostats","metrics_3d.density_per_mm3", lambda: metrics_3d.density_per_mm3(lab.max(),vol.shape,(1.,1.,1.))),
 ("fluorostats","objects.equivalent_diameters_um", lambda: objects.equivalent_diameters_um(lab,(1.,1.,1.))),
 ("fluorostats","objects.count_local_maxima 3D", lambda: objects.count_local_maxima(vol,min_distance=3)),
 ("comparator","peak_local_max 3D", lambda: peak_local_max(vol,min_distance=3,threshold_abs=vol.max()*0.15)),
 # ---- 2D metrics ----
 ("fluorostats","metrics_2d.area_fraction", lambda: metrics_2d.area_fraction(img2>filters.threshold_otsu(img2))),
 ("fluorostats","metrics_2d.coverage_metrics", lambda: metrics_2d.coverage_metrics(img2>filters.threshold_otsu(img2))),
 # ---- morphometry ----
 ("fluorostats","morphometry.lateral_homogeneity", lambda: morphometry.lateral_homogeneity(mask)),
 ("fluorostats","morphometry.depth_profile", lambda: morphometry.depth_profile(vol)),
 ("fluorostats","morphometry.depth_centroid", lambda: morphometry.depth_centroid(vol)),
 ("fluorostats","morphometry.depth_span", lambda: morphometry.depth_span(vol)),
 ("fluorostats","objects.centroid_homogeneity", lambda: objects.centroid_homogeneity(cent,vol.shape)),
 ("comparator","clark_evans NN", lambda: clark_evans(pts)),
 ("comparator","morisita index", lambda: morisita(pts)),
 ("comparator","quadrat_variance", lambda: quadrat(pts)),
 ("comparator","ripleys_K", lambda: ripley(pts)),
 ("comparator","lacunarity", lambda: lacunarity(mask[mask.shape[0]//2])),
 # ---- skeleton (2D) ----
 ("fluorostats","skeleton.skeleton_metrics 2D", lambda: skeleton.skeleton_metrics(sk,voxel_size_um=(1.,1.))),
 ("fluorostats","skeleton.prune_skeleton 2D", lambda: skeleton.prune_skeleton(sk,min_branch_length_px=10)),
 ("fluorostats","skeleton.n_junction_nodes 2D", lambda: skeleton.n_junction_nodes(sk)),
 ("comparator","skeletonize Lee 2D", lambda: skeletonize(sk)),
 ("comparator","medial_axis 2D", lambda: medial_axis(sk)),
 ("comparator","thin 2D", lambda: thin(sk)),
 ("comparator","skeletonize zhang 2D", lambda: skeletonize(sk,method="zhang")),
 # ---- viability ----
 ("fluorostats","viability.live_dead_fractions", lambda: viability.live_dead_fractions(live,dead,min_size=8)),
 ("fluorostats","viability.live_dead_by_count cc", lambda: viability.live_dead_by_count(live,dead,method="cc")),
 ("fluorostats","viability.live_dead_by_count maxima", lambda: viability.live_dead_by_count(live,dead,method="maxima")),
 ("fluorostats","viability.live_dead_by_count watershed", lambda: viability.live_dead_by_count(live,dead,method="watershed")),
 ("fluorostats","viability.live_dead_by_count auto", lambda: viability.live_dead_by_count(live,dead,method="auto")),
 ("fluorostats","viability.live_dead_by_count all", lambda: viability.live_dead_by_count(live,dead,method="all")),
 ("comparator","Kerkhoff macro (peak count)", lambda: (len(peak_local_max(live,min_distance=3,threshold_abs=live.max()*0.15)),len(peak_local_max(dead,min_distance=3,threshold_abs=dead.max()*0.15)))),
 ("fluorostats","viability.attenuation_correct", lambda: viability.attenuation_correct(vol)),
 # ---- stats ----
 ("fluorostats","stats.mann_whitney", lambda: stats.mann_whitney(a,b)),
 ("comparator","scipy.mannwhitneyu", lambda: sps.mannwhitneyu(a,b,alternative="two-sided")),
 ("fluorostats","stats.cliffs_delta", lambda: stats.cliffs_delta(a,b)),
 ("fluorostats","stats.bh_fdr", lambda: stats.bh_fdr([0.01,0.02,0.2,0.5])),
 ("fluorostats","stats.bootstrap_fold_change_ci", lambda: stats.bootstrap_fold_change_ci(np.abs(a)+1,np.abs(b)+1,n_boot=1000)),
 ("fluorostats","stats.stouffer_combine", lambda: stats.stouffer_combine([0.01,0.04,0.2])),
 ("comparator","scipy.combine_pvalues", lambda: sps.combine_pvalues([0.01,0.04,0.2],method="stouffer")),
 ("fluorostats","stats.stratified_mann_whitney", lambda: stats.stratified_mann_whitney(sdf,["v"],"grp","A","B",strata=["st"])),
 # ---- agreement ----
 ("fluorostats","agreement.bland_altman", lambda: agreement.bland_altman(a,b)),
 ("fluorostats","agreement.lins_ccc", lambda: agreement.lins_ccc(a,b)),
 ("fluorostats","agreement.icc", lambda: agreement.icc(a,b)),
 ("comparator","scipy.pearsonr", lambda: sps.pearsonr(a,b)),
 # ---- validate ----
 ("fluorostats","validate.instance_f1", lambda: validate.instance_f1(lab,gt)),
 ("fluorostats","validate.average_precision", lambda: validate.average_precision(lab,gt)),
]
rows=[]
for grp,name,fn in T:
    reps=3 if any(k in name for k in ["rolling","average_precision","median_filter","watershed","instance_f1"]) else 5
    try: rows.append({"group":grp,"metric":name,"ms":round(tm(fn,reps),3)})
    except Exception as e: rows.append({"group":grp,"metric":name,"ms":None,"err":str(e)[:70]})
rows.append({"group":"comparator","metric":"StarDist (per 2D img, CPU cluster)","ms":215.1})
rows.append({"group":"comparator","metric":"Cellpose (per 2D img, CPU cluster)","ms":5547.0})
df=pd.DataFrame(rows); df.to_csv(RES/"b_timing_all_metrics.csv",index=False)
print("=== MASTER per-metric + per-comparator timing (3D vol %s) ==="%str(vol.shape))
print(df[["group","metric","ms"]].to_string(index=False))
if df.ms.isna().any(): print("\nERRORS:", df[df.ms.isna()][["metric","err"]].to_string(index=False))
