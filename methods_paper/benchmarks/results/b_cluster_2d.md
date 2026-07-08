# B-cluster-2D — fluorostats 2D cluster/coverage metrics vs GT (BBBC039)

**Dataset:** BBBC039 2D nuclei, n = 200 images.
**Library:** fluorostats 0.5.0 — `metrics_2d.area_fraction`, `metrics_2d.coverage_metrics`.
**Threshold methods:** Otsu, Li, Isodata, Triangle, Yen (skimage.filters).

Ground truth from BBBC039 RGBA instance masks (channel 0):
`foreground = ch0>0`; `nucleus_count = CC(ch0==1)`; `cluster_count = CC(ch0>0)`.

GT means: area_fraction = 0.1956, nuclei/img = 97.0, clusters/img = 97.9.

fluorostats `area_fraction` == independent `(img>thr).mean()`: **True** (Otsu spot-check, atol 1e-9).

## Area fraction agreement (fluorostats area_fraction vs GT)
| method | CCC | spearman | pearson | bias | MAE | n |
| --- | --- | --- | --- | --- | --- | --- |
| Triangle_1977 | 0.575 | 0.898 | 0.699 | 0.048 | 0.048 | 200 |
| Li_1993 | 0.505 | 0.907 | 0.538 | 0.022 | 0.022 | 200 |
| Otsu_1979 | 0.441 | 0.882 | 0.45 | -0.0041 | 0.0233 | 200 |
| Isodata_1978 | 0.419 | 0.892 | 0.429 | -0.0023 | 0.0226 | 200 |
| Yen_1995 | -0.138 | -0.16 | -0.269 | -0.1167 | 0.1619 | 200 |

## Cluster count agreement (fluorostats n_components vs GT foreground clusters)
| method | CCC | spearman | pearson | bias | MAE | n |
| --- | --- | --- | --- | --- | --- | --- |
| Triangle_1977 | -0.014 | 0.545 | -0.361 | 209.605 | 215.795 | 200 |
| Otsu_1979 | -0.018 | 0.809 | -0.324 | 164.9 | 166.22 | 200 |
| Li_1993 | -0.019 | 0.879 | -0.329 | 144.645 | 145.095 | 200 |
| Isodata_1978 | -0.021 | 0.833 | -0.323 | 148.75 | 148.75 | 200 |
| Yen_1995 | -0.067 | -0.098 | -0.361 | -17.92 | 105.9 | 200 |

## Cluster count vs GT nucleus count (expected weaker — clusters merge touching nuclei)
| method | CCC | spearman | pearson | bias | MAE | n |
| --- | --- | --- | --- | --- | --- | --- |
| Triangle_1977 | -0.014 | 0.537 | -0.356 | 210.56 | 216.55 | 200 |
| Otsu_1979 | -0.018 | 0.826 | -0.318 | 165.855 | 166.795 | 200 |
| Li_1993 | -0.018 | 0.867 | -0.324 | 145.6 | 146.14 | 200 |
| Isodata_1978 | -0.021 | 0.819 | -0.318 | 149.705 | 149.705 | 200 |
| Yen_1995 | -0.067 | -0.117 | -0.358 | -16.965 | 105.765 | 200 |

## Descriptive cluster metrics (no direct GT counterpart)
| method | mean_largest_frac | mean_cluster_area_px | mean_n_components |
| --- | --- | --- | --- |
| Otsu_1979 | 0.057 | 528.4 | 262.8 |
| Li_1993 | 0.054 | 708.2 | 242.6 |
| Isodata_1978 | 0.053 | 523.4 | 246.7 |
| Triangle_1977 | 0.06 | 800.6 | 307.5 |
| Yen_1995 | 0.353 | 304.1 | 80.0 |

## Notes
- fluorostats `n_components` counts connected foreground clusters, so it is
  compared against GT *cluster* count (touching-nuclei-merged), not the raw
  nucleus count; the nucleus-count row is included to quantify that gap.
- `largest_component_fraction`, `mean_cluster_area_px`, and `median_cluster_area_px`
  have no per-image GT counterpart in BBBC039 and are reported descriptively.
