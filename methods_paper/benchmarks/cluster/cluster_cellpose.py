"""Run Cellpose on BBBC039, write per-image nucleus counts."""
import glob, os, csv, sys
import numpy as np, tifffile
from cellpose import models
model = models.CellposeModel(gpu=True)
files = sorted(glob.glob(os.path.expanduser('~/fluorostats_bench/data/BBBC039/images/*.tif')))
rows = []
for i, f in enumerate(files):
    img = tifffile.imread(f).astype(np.float32)
    try:
        out = model.eval(img)
        masks = out[0]
        n = int(masks.max())
    except Exception as e:
        n = -1
        print('ERR', f, e, flush=True)
    rows.append((os.path.basename(f).replace('.tif',''), n))
    if i % 40 == 0:
        print(f'[{i}/{len(files)}] {os.path.basename(f)} -> {n}', flush=True)
with open(os.path.expanduser('~/fluorostats_bench/cellpose_counts.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['image','cellpose_count']); w.writerows(rows)
print('DONE cellpose', len(rows), flush=True)
