"""CSV report generation: per-file rows and per-condition summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Column order for the per-file CSV
_COLUMNS_3D = [
    "file",
    "condition",
    "replicate",
    "shape",
    "voxel_size_um",
    "volume_fraction",
    "n_components",
    "euler_number",
    "largest_component_fraction",
    "total_length_um",
    "n_branches",
    "n_junctions",
    "mean_branch_length_um",
]

_COLUMNS_2D = [
    "file",
    "condition",
    "replicate",
    "shape",
    "area_fraction",
    "n_components",
    "largest_component_fraction",
    "mean_cluster_area_px",
    "median_cluster_area_px",
]

_NUMERIC_3D = [
    "volume_fraction",
    "n_components",
    "euler_number",
    "largest_component_fraction",
    "total_length_um",
    "n_branches",
    "n_junctions",
    "mean_branch_length_um",
]

_NUMERIC_2D = [
    "area_fraction",
    "n_components",
    "largest_component_fraction",
    "mean_cluster_area_px",
    "median_cluster_area_px",
]


def write_per_file_csv(
    rows: list[dict],
    out_path: Path,
    mode: str = "3d",
) -> pd.DataFrame:
    """Write per-file results to CSV. Returns the DataFrame."""
    cols = _COLUMNS_3D if mode == "3d" else _COLUMNS_2D
    df = pd.DataFrame(rows, columns=cols)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_path), index=False)
    return df


def write_per_condition_csv(
    df: pd.DataFrame,
    out_path: Path,
    mode: str = "3d",
) -> pd.DataFrame:
    """Write per-condition summary statistics to CSV."""
    numeric = _NUMERIC_3D if mode == "3d" else _NUMERIC_2D
    available = [c for c in numeric if c in df.columns]

    summary = (
        df.groupby("condition")[available]
        .agg(["count", "mean", "std", "median"])
    )

    # Flatten multi-level column names
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(str(out_path), index=False)
    return summary
