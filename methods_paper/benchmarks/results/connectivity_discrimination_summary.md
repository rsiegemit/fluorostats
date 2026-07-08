# Connectivity-metric discrimination benchmark

Synthetic 3D lattice of 27 spherical blobs (3x3x3). A known fraction of adjacent blob pairs is bridged with cylinders, sweeping from fully **fragmented** (0 bridges) to a fully **connected** spanning network (all adjacent pairs bridged). Gradient = fragmentation level (1 - fraction bridged). 11 levels x 5 seeds = 55 volumes.

## Spearman correlation vs. known fragmentation level

Ranked by |rho|. Sign shows direction (positive = rises with fragmentation).

| measure | source | spearman_rho_vs_fragmentation | abs_rho | p_value |
| --- | --- | --- | --- | --- |
| euler_number | fluorostats | 1.0 | 1.0 | 0.0 |
| mean_component_size | implemented (label_3d + component sizes) | -0.9992 | 0.9992 | 1.5384870420933972e-75 |
| n_components | fluorostats | 0.9702 | 0.9702 | 2.6435689462606177e-34 |
| largest_component_fraction | fluorostats | -0.967 | 0.967 | 3.739483194730525e-33 |
| spanning_indicator | implemented (label_3d + bbox span) | -0.8727 | 0.8727 | 3.8949234657374814e-18 |


## Verdict

Best discriminator: **euler_number** (fluorostats), |rho| = 1.000, rho = +1.000.

Measures that track the gradient essentially monotonically (|rho| >= 0.99): euler_number, mean_component_size.
