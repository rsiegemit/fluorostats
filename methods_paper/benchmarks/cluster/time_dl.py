"""Measure precise per-image inference time for a DL tool on BBBC039 (20 imgs)."""
import sys, glob, os, time, numpy as np, tifffile
TOOL=sys.argv[1]; IMG=os.path.expanduser('~/fluorostats_bench/data/BBBC039/images')
files=sorted(glob.glob(IMG+'/*.tif'))[:20]
if TOOL=='cellpose':
    from cellpose import models; m=models.Cellpose(gpu=False, model_type='nuclei')
    run=lambda im: m.eval(im, channels=[0,0], diameter=None)[0]
else:
    from stardist.models import StarDist2D; from csbdeep.utils import normalize
    m=StarDist2D.from_pretrained('2D_versatile_fluo'); run=lambda im: m.predict_instances(normalize(im))[0]
run(tifffile.imread(files[0]).astype('float32'))  # warmup
t0=time.perf_counter()
for f in files: run(tifffile.imread(f).astype('float32'))
dt=(time.perf_counter()-t0)/len(files)*1000
print(f'{TOOL}_MS_PER_IMAGE={dt:.1f}', flush=True)
