# FOV-normalized object density is digital-zoom invariant

**Physical sample:** source=`BBBC024`, true object count = **20**, native grid = `(129, 565, 807)` voxels, native voxel = `(1.0, 0.5, 0.5)` µm (z,y,x).

**Fixed physical field of view** = `(0.129, 0.282, 0.404)` mm (z,y,x).

The same physical sample is re-imaged at several digital zooms. A scheme that measures a real physical density should be **constant** (CV ≈ 0) across zoom levels.

## Coefficient of variation across zoom (lower = more invariant)

| scheme | A_fixed_FOV | B_fixed_voxel |
| --- | --- | --- |
| fluorostats_per_mm3 | 0.000 | 0.160 |
| raw_count | 0.000 | 44.721 |
| per_megavoxel | 143.065 | 0.160 |
| per_mm2_area | 0.000 | 16.577 |
| per_z_slice | 58.427 | 31.378 |

Regime A = fixed physical FOV, varying voxel size (pure digital zoom / rebinning). Regime B = fixed voxel size, varying crop extent (different physical FOV). True per-mm³ density = **1360.12 objects/mm³**.

## Regime A — pure digital zoom, fixed physical FOV

| zoom | shape | fov_vol_mm3 | fluorostats_per_mm3 | raw_count | per_megavoxel | per_mm2_area | per_z_slice |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.50 | (64, 282, 404) | 0.01 | 1360.12 | 20.00 | 2.74 | 175.46 | 0.31 |
| 0.75 | (97, 424, 605) | 0.01 | 1360.12 | 20.00 | 0.80 | 175.46 | 0.21 |
| 1.00 | (129, 565, 807) | 0.01 | 1360.12 | 20.00 | 0.34 | 175.46 | 0.16 |
| 1.50 | (194, 848, 1210) | 0.01 | 1360.12 | 20.00 | 0.10 | 175.46 | 0.10 |
| 2.00 | (258, 1130, 1614) | 0.01 | 1360.12 | 20.00 | 0.04 | 175.46 | 0.08 |
| 3.00 | (387, 1695, 2421) | 0.01 | 1360.12 | 20.00 | 0.01 | 175.46 | 0.05 |

## Regime B — fixed voxel size, varying physical FOV

| zoom | shape | fov_vol_mm3 | fluorostats_per_mm3 | raw_count | per_megavoxel | per_mm2_area | per_z_slice |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25 | (81, 356, 508) | 0.00 | 1365.31 | 5.00 | 0.34 | 110.59 | 0.06 |
| 0.50 | (102, 448, 641) | 0.01 | 1365.60 | 10.00 | 0.34 | 139.29 | 0.10 |
| 0.75 | (117, 513, 733) | 0.01 | 1363.78 | 15.00 | 0.34 | 159.56 | 0.13 |
| 1.00 | (129, 565, 807) | 0.01 | 1360.12 | 20.00 | 0.34 | 175.46 | 0.16 |

## Interpretation

- **fluorostats `density_per_mm3`** is invariant in BOTH regimes: it divides count by the physical volume (`voxel_count × voxel_volume`), so digital zoom and FOV cancel out.

- **raw count** is invariant only under pure digital zoom (Regime A) but collapses when the physical FOV changes (Regime B).

- **per-megavoxel**, **per-mm² area**, and **per-z-slice** all vary with acquisition settings — they conflate voxel-grid choices with biology and are not comparable across zoom or FOV.
