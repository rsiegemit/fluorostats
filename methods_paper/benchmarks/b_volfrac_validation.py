"""Validate volume_fraction exactness + FOV-normalized density reproducibility."""
import numpy as np, pandas as pd
from pathlib import Path
from fluorostats.metrics_3d import volume_fraction, fov_volume_mm3, density_per_mm3
RES=Path(__file__).resolve().parent/"results"; rng=np.random.default_rng(0); rows=[]

# 1. VF == known fraction exactly (voxel counting = exhaustive point counting)
for p in [0.05,0.1,0.2,0.35]:
    vol=rng.random((60,60,60))<p
    rows.append({"test":f"VF_random_p={p}","fluorostats":round(volume_fraction(vol),4),
                 "reference":round(float(vol.mean()),4),"PASS":abs(volume_fraction(vol)-vol.mean())<1e-12})

# 2. Sphere-in-box analytic VF = (4/3)pi r^3 / L^3
L=120; r=40; zz,yy,xx=np.ogrid[:L,:L,:L]
sph=((zz-L/2)**2+(yy-L/2)**2+(xx-L/2)**2)<=r**2
vf=volume_fraction(sph); analytic=(4/3)*np.pi*r**3/L**3
rows.append({"test":"VF_sphere_vs_analytic","fluorostats":round(vf,4),
             "reference":round(analytic,4),"PASS":abs(vf-analytic)/analytic<0.02})

# 3. Cavalieri/Delesse point-counting converges to voxel VF as grid densifies
errs=[]
for step in [8,4,2,1]:
    pts=sph[::step,::step,::step]; errs.append(abs(pts.mean()-vf))
rows.append({"test":"point_count_converges(step8->1 err)","fluorostats":round(errs[-1],5),
             "reference":round(errs[0],5),"PASS":errs[-1]<errs[0]})

# 4. FOV-normalized density is voxel-size invariant (the reproducibility claim)
# same object imaged at 2 digital zooms: voxel 0.5um (fine) vs 1.0um (coarse)
# Same physical sample -> same object count (1000), same physical FOV, two voxel grids.
count=1000
d_fine=density_per_mm3(count,(50,512,512),(2.0,0.5,0.5))     # fine zoom
d_coarse=density_per_mm3(count,(25,256,256),(4.0,1.0,1.0))   # coarse zoom, SAME physical FOV
rows.append({"test":"density_per_mm3_zoom_invariant","fluorostats":round(d_fine,1),
             "reference":round(d_coarse,1),"PASS":abs(d_fine-d_coarse)/d_fine<0.02})

df=pd.DataFrame(rows); df.to_csv(RES/"b_volfrac_validation.csv",index=False)
print(df.to_string(index=False)); print(f"\n{int(df.PASS.sum())}/{len(df)} PASS")
