"""Produce results/b_homogeneity_examples.csv — tile Gini of three example point patterns.

The regular / random (CSR) / clustered patterns of ``common.example_patterns``
(generator seed 1) are scored by ``objects.centroid_homogeneity`` (8x8 tiles over
the square field). No data needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import EXAMPLE_FIELD, RESULTS, example_patterns
from fluorostats.objects import centroid_homogeneity

SEED = 1


def main():
    rows = []
    for kind, p in example_patterns(np.random.default_rng(SEED)).items():
        p3 = np.column_stack([np.zeros(len(p)), p[:, 0], p[:, 1]])
        rows.append({"pattern": kind, "n_points": len(p), "field_px": EXAMPLE_FIELD,
                     "tile_gini": centroid_homogeneity(p3, (1, EXAMPLE_FIELD, EXAMPLE_FIELD))["centroid_gini"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b_homogeneity_examples.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
