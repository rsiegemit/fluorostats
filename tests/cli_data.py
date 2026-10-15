"""Synthetic inputs shared by the CLI test files (tests/test_cli*.py)."""

import numpy as np
import tifffile


def blob_2d(seed):
    img = np.random.default_rng(seed).integers(0, 40, (64, 64)).astype(np.uint8)
    side = 26 + 2 * (seed % 5)  # replicates differ, so tests have data
    img[16 : 16 + side, 16 : 16 + side] = 220  # bright foreground patch
    return img


def blob_3d(seed):
    vol = np.random.default_rng(seed).integers(0, 40, (8, 48, 48)).astype(np.uint16)
    side = 20 + 2 * (seed % 5)
    vol[2:6, 12 : 12 + side, 12 : 12 + side] = 900
    return vol


def make_dataset(root, maker):
    for ci, cond in enumerate(("GelMA", "Hybrid")):
        d = root / cond
        d.mkdir(parents=True)
        for i in range(3):  # 3 replicates -> plots + p-values run
            tifffile.imwrite(d / f"s{i}.tif", maker(10 * ci + i))
