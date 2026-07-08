"""External viability head-to-head: fluorostats vs Kerkhoff Fiji macro method vs
exact ground-truth viability (Zenodo 10395753 synthetic Live/Dead set)."""
import re, glob, numpy as np, pandas as pd, tifffile
from pathlib import Path
from skimage import filters
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
from fluorostats.viability import live_dead_fractions, live_dead_by_count
from fluorostats.objects import watershed_split
import sys; sys.path.insert(0,str(Path(__file__).resolve().parent))
from agreement import agreement_report
D=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/kerkhoff_lda/synthetic")
RES=Path(__file__).resolve().parent/"results"
def count_peaks(img):  # Kerkhoff macro = Fiji Find Maxima (prominence peak counting)
    return len(peak_local_max(img, min_distance=3, threshold_abs=img.max()*0.15))
def count_cc(mask):
    return ndi.label(remove_small_objects(mask,8))[0].max()
rows=[]
for f in sorted(glob.glob(str(D/"**/*.tif"),recursive=True)):
    m=re.search(r"(\d+) Live; (\d+) Dead",Path(f).name)
    if not m: continue
    L,De=int(m.group(1)),int(m.group(2)); true_v=L/(L+De)
    im=tifffile.imread(f).astype(np.float32); live,dead=im[1],im[0]
    # Kerkhoff-macro-equivalent: peak counting
    pl,pd_=count_peaks(live),count_peaks(dead); mac=pl/(pl+pd_) if pl+pd_ else np.nan
    # fluorostats object-count (watershed split touching cells)
    lm=remove_small_objects(live>filters.threshold_otsu(live),8)
    dm=remove_small_objects(dead>filters.threshold_otsu(dead),8)
    nl=watershed_split(lm,min_size=8,min_distance=3)[0].max()
    nd=watershed_split(dm,min_size=8,min_distance=3)[0].max()
    fs_cnt=nl/(nl+nd) if nl+nd else np.nan
    # fluorostats area-fraction viability
    ld=live_dead_fractions(live,dead,method="otsu",min_size=8)
    fs_area=ld.get("viability",ld.get("live_fraction"))
    # fluorostats NEW native prominence-peak counting (live_dead_by_count)
    fs_max=live_dead_by_count(live,dead,method="maxima",min_distance=3,threshold_rel=0.15)["viability"]
    # Otsu-CC count
    oc_l,oc_d=count_cc(lm),count_cc(dm); otsu=oc_l/(oc_l+oc_d) if oc_l+oc_d else np.nan
    rows.append({"file":Path(f).name,"true_viability":true_v,
                 "Kerkhoff_macro_peakcount":mac,"fluorostats_objcount":fs_cnt,
                 "fluorostats_areafrac":fs_area,"fluorostats_maxima(NEW)":fs_max,"Otsu_CC_count":otsu})
df=pd.DataFrame(rows); df.to_csv(RES/"b_viability_external.csv",index=False)
print(f"=== External viability vs exact ground truth (n={len(df)} synthetic images) ===")
methods=["Kerkhoff_macro_peakcount","fluorostats_maxima(NEW)","fluorostats_objcount","fluorostats_areafrac","Otsu_CC_count"]
summ=[]
for mth in methods:
    d=df.dropna(subset=[mth]); mae=float(np.abs(d[mth]-d.true_viability).mean())
    rep=agreement_report(d.true_viability.values,d[mth].values)
    ccc=rep.get("ccc",rep.get("lins_ccc")); 
    summ.append({"method":mth,"MAE":round(mae,4),"CCC":round(float(ccc),3)})
sd=pd.DataFrame(summ).sort_values("MAE"); print(sd.to_string(index=False))
sd.to_csv(RES/"b_viability_external_summary.csv",index=False)
