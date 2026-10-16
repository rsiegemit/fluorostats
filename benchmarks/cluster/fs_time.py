"""Time fluorostats 2D segmentation per image, for ``producers/b_seg_timing.py``.

Times ``common.denoised_otsu_cc`` (the method Fig S2 and ``b_timing`` time) and
``common.raw_otsu_cc``, each image once per method.

Runs on the same node, in the same job, as the StarDist and Cellpose timings of
``bbbc024_dl_infer.py`` so the three are compared on one CPU. Imports
them from the fluorostats checkout given as SRC, so the timed code is the code the
benchmarks use, and writes ``versions.json`` in the same shape as the
deep-learning script (``seconds`` per method, then per image file).

    python fs_time.py SRC IN_DIR OUT_DIR [MIN_SIZE]
"""
from __future__ import annotations

import glob
import json
import os
import platform
import sys
import time


def main() -> None:
    src, in_dir, out_dir = sys.argv[1:4]
    min_size = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    sys.path.insert(0, os.path.join(src, "benchmarks", "producers"))
    import numpy as np
    import skimage
    import tifffile
    from common import denoised_otsu_cc, raw_otsu_cc

    import fluorostats

    os.makedirs(out_dir, exist_ok=True)
    methods = {"denoised Otsu+CC": denoised_otsu_cc, "raw Otsu+CC": raw_otsu_cc}
    seconds: dict = {m: {} for m in methods}
    for f in sorted(glob.glob(os.path.join(in_dir, "*.tif"))):
        img = tifffile.imread(f).astype(np.float32)
        for m, fn in methods.items():
            t0 = time.perf_counter()
            fn(img, min_size)
            seconds[m][os.path.basename(f)] = round(time.perf_counter() - t0, 5)
    meta = {"tool": "fluorostats", "methods": list(methods), "min_size": min_size,
            "fluorostats": fluorostats.__version__, "numpy": np.__version__,
            "scikit-image": skimage.__version__, "python": platform.python_version(),
            "n_files": len(seconds["raw Otsu+CC"]), "seconds": seconds}
    with open(os.path.join(out_dir, "versions.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    for m, s in seconds.items():
        print(f"fluorostats {m}: {len(s)} images, median {np.median(list(s.values())):.4f}s")


if __name__ == "__main__":
    main()
