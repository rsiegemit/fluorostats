"""CLI entry point: image-quant quant3d / quant2d."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from . import io as iq_io
from . import metrics_2d, metrics_3d, plots, preprocess, qc, report, segment


@click.group()
def cli():
    """FluoroStats — Universal fluorescence microscopy image quantification."""


@cli.command()
def formats():
    """List all supported file formats and their availability."""
    from . import io as iq_io
    click.echo("Supported formats:\n")
    for ext, desc in sorted(iq_io.supported_formats().items()):
        marker = "x" if "requires" not in desc else " "
        click.echo(f"  [{marker}] {ext:<8} {desc}")
    click.echo(f"\nInstall all: pip install fluorostats[all]")


# ---------------------------------------------------------------------------
# quant3d
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True), help="Folder with .oib/.oif/.tif volumes")
@click.option("--output", "output_dir", required=True, type=click.Path(), help="Output folder for results")
@click.option("--condition-from", "condition_from", type=click.Choice(["parent", "grandparent", "filename"]), default="parent", help="How to assign condition labels")
@click.option("--channel", default=None, help="Force channel by index (int) or name substring")
@click.option("--threshold", type=click.Choice(["otsu", "li"]), default="otsu")
@click.option("--threshold-scale", type=float, default=0.9, help="Scale threshold by this factor (lower=more sensitive, default 0.9)")
@click.option("--min-size", type=int, default=64, help="Min object size in voxels")
@click.option("--sigma", type=float, default=1.0, help="Gaussian blur sigma")
@click.option("--bg-radius", type=int, default=0, help="Background subtraction disk radius (0=off, recommended for confocal)")
@click.option("--no-overlays", is_flag=True, help="Skip QC overlay PNGs")
@click.option("--no-plots", is_flag=True, help="Skip comparison plots")
@click.option("--no-skeleton", is_flag=True, help="Skip skeleton analysis (faster)")
def quant3d(
    input_dir,
    output_dir,
    condition_from,
    channel,
    threshold,
    threshold_scale,
    min_size,
    sigma,
    bg_radius,
    no_overlays,
    no_plots,
    no_skeleton,
):
    """Quantify 3D confocal z-stacks: volume fraction, connectivity, skeleton."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve channel override
    ch_override = _parse_channel(channel)

    # Find volume files
    files = _find_files(input_path, iq_io.VOLUME_SUFFIXES)
    if not files:
        click.echo(f"No volume files found in {input_path}")
        return

    click.echo(f"Found {len(files)} volume file(s)")

    # Save run config
    _save_config(output_path, locals())

    rows: list[dict] = []
    rep_counter = _ReplicateCounter()

    with click.progressbar(files, label="Processing volumes") as bar:
        for fpath in bar:
            condition = _get_condition(fpath, condition_from)
            replicate = rep_counter.next(condition)

            # Load
            arr, meta = iq_io.load_volume(fpath)
            voxel_size = meta["voxel_size_um"]

            # Channel selection
            vol = preprocess.select_green_channel(
                arr, meta["channel_names"], override=ch_override
            )

            # Preprocess
            vol = preprocess.denoise(vol, sigma=sigma)
            if bg_radius > 0:
                vol = preprocess.background_subtract(vol, radius=bg_radius)

            # Segment
            mask = segment.binarize(vol, method=threshold, min_size=min_size, threshold_scale=threshold_scale)

            # Metrics
            vf = metrics_3d.volume_fraction(mask)
            conn = metrics_3d.connectivity_metrics(mask)

            skel = {"total_length_um": 0.0, "n_branches": 0, "n_junctions": 0}
            if not no_skeleton:
                skel = metrics_3d.skeleton_metrics(mask, voxel_size_um=voxel_size)

            row = {
                "file": fpath.name,
                "condition": condition,
                "replicate": replicate,
                "shape": str(arr.shape),
                "voxel_size_um": str(voxel_size),
                "volume_fraction": vf,
                **conn,
                **skel,
            }
            rows.append(row)

            # QC overlay
            if not no_overlays:
                overlay_path = output_path / "overlays" / f"{fpath.stem}_overlay.png"
                qc.overlay_3d(vol, mask, overlay_path)

    # Reports
    df = report.write_per_file_csv(rows, output_path / "per_file.csv", mode="3d")
    summary = report.write_per_condition_csv(df, output_path / "per_condition.csv", mode="3d")

    click.echo(f"\nPer-file results: {output_path / 'per_file.csv'}")
    click.echo(f"Per-condition summary: {output_path / 'per_condition.csv'}")

    click.echo("\n--- Per-condition summary ---")
    click.echo(summary.to_string(index=False))

    # Plots
    _3d_metrics = ["volume_fraction", "n_components", "euler_number",
                   "largest_component_fraction", "total_length_um",
                   "n_junctions", "mean_branch_length_um"]

    if not no_plots and len(df["condition"].unique()) > 1:
        plot_dir = output_path / "plots"
        has_replicates = any(len(df[df["condition"] == c]) > 1 for c in df["condition"].unique())

        for metric in _3d_metrics:
            if metric not in df.columns or not df[metric].notna().any():
                continue
            if has_replicates:
                plots.bar_mean_sem(df, metric, plot_dir / f"{metric}_bar.png")
            plots.boxplot_by_condition(df, metric, plot_dir / f"{metric}_box.png")

        # Summary panel with all metrics
        plots.summary_panel(
            df, _3d_metrics, plot_dir / "summary_panel.png",
            title="3D Confocal Quantification Summary",
            show_pvalues=has_replicates,
        )

        # P-value table
        if has_replicates:
            pvals = plots.compute_pvalues(df, _3d_metrics)
            if len(pvals) > 0:
                pvals.to_csv(str(plot_dir / "pvalues.csv"), index=False)
                click.echo(f"\nStatistical comparisons: {plot_dir / 'pvalues.csv'}")

        click.echo(f"Plots saved to {plot_dir}")


# ---------------------------------------------------------------------------
# quant2d
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True), help="Folder with .png/.tif images")
@click.option("--output", "output_dir", required=True, type=click.Path(), help="Output folder for results")
@click.option("--condition-from", "condition_from", type=click.Choice(["parent", "grandparent", "filename"]), default="parent", help="How to assign condition labels")
@click.option("--channel", default=None, help="Force channel by index (int) or name substring")
@click.option("--threshold", type=click.Choice(["otsu", "li"]), default="li")
@click.option("--threshold-scale", type=float, default=1.0, help="Scale threshold by this factor (lower=more sensitive)")
@click.option("--min-size", type=int, default=64, help="Min object size in pixels")
@click.option("--sigma", type=float, default=1.0, help="Gaussian blur sigma")
@click.option("--bg-radius", type=int, default=15, help="Background subtraction disk radius")
@click.option("--auto-crop/--no-auto-crop", default=True, help="Auto-crop microscope borders (default: on)")
@click.option("--no-overlays", is_flag=True, help="Skip QC overlay PNGs")
@click.option("--no-plots", is_flag=True, help="Skip comparison plots")
def quant2d(
    input_dir,
    output_dir,
    condition_from,
    channel,
    threshold,
    threshold_scale,
    min_size,
    sigma,
    bg_radius,
    auto_crop,
    no_overlays,
    no_plots,
):
    """Quantify 2D fluorescence images: area fraction (cell coverage)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ch_override = _parse_channel(channel)

    files = _find_files(input_path, iq_io.IMAGE_SUFFIXES)
    if not files:
        click.echo(f"No image files found in {input_path}")
        return

    click.echo(f"Found {len(files)} image file(s)")
    _save_config(output_path, locals())

    rows: list[dict] = []
    rep_counter = _ReplicateCounter()

    with click.progressbar(files, label="Processing images") as bar:
        for fpath in bar:
            condition = _get_condition(fpath, condition_from)
            replicate = rep_counter.next(condition)

            arr, meta = iq_io.load_image(fpath)

            # Auto-crop microscope borders before channel selection
            if auto_crop:
                arr, crop_coords = preprocess.auto_crop(arr)

            img = preprocess.select_green_channel(
                arr, meta["channel_names"], override=ch_override
            )

            img = preprocess.denoise(img, sigma=sigma)
            if bg_radius > 0:
                img = preprocess.background_subtract(img, radius=bg_radius)

            mask = segment.binarize(img, method=threshold, min_size=min_size, threshold_scale=threshold_scale)

            cov = metrics_2d.coverage_metrics(mask)

            row = {
                "file": fpath.name,
                "condition": condition,
                "replicate": replicate,
                "shape": str(arr.shape),
                **cov,
            }
            rows.append(row)

            if not no_overlays:
                overlay_path = output_path / "overlays" / f"{fpath.stem}_overlay.png"
                qc.overlay_2d(img, mask, overlay_path)

    df = report.write_per_file_csv(rows, output_path / "per_file.csv", mode="2d")
    summary = report.write_per_condition_csv(df, output_path / "per_condition.csv", mode="2d")

    click.echo(f"\nPer-file results: {output_path / 'per_file.csv'}")
    click.echo(f"Per-condition summary: {output_path / 'per_condition.csv'}")

    click.echo("\n--- Per-condition summary ---")
    click.echo(summary.to_string(index=False))

    _2d_metrics = ["area_fraction", "n_components", "largest_component_fraction",
                   "mean_cluster_area_px", "median_cluster_area_px"]

    if not no_plots and len(df["condition"].unique()) > 1:
        plot_dir = output_path / "plots"
        has_replicates = any(len(df[df["condition"] == c]) > 1 for c in df["condition"].unique())

        for metric in _2d_metrics:
            if metric not in df.columns or not df[metric].notna().any():
                continue
            if has_replicates:
                plots.bar_mean_sem(df, metric, plot_dir / f"{metric}_bar.png")
            plots.boxplot_by_condition(df, metric, plot_dir / f"{metric}_box.png")

        plots.summary_panel(
            df, _2d_metrics, plot_dir / "summary_panel.png",
            title="2D Endothelial Coverage Summary",
            show_pvalues=has_replicates,
        )

        if has_replicates:
            pvals = plots.compute_pvalues(df, _2d_metrics)
            if len(pvals) > 0:
                pvals.to_csv(str(plot_dir / "pvalues.csv"), index=False)
                click.echo(f"\nStatistical comparisons: {plot_dir / 'pvalues.csv'}")

        click.echo(f"Plots saved to {plot_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_files(root: Path, suffixes: set[str]) -> list[Path]:
    """Recursively find files with matching suffixes."""
    files = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in suffixes:
            files.append(f)
    return files


def _get_condition(fpath: Path, mode: str) -> str:
    """Derive the experimental condition label from the file path."""
    if mode == "parent":
        return fpath.parent.name
    if mode == "grandparent":
        return fpath.parent.parent.name
    # filename: strip extension, use as condition
    return fpath.stem


class _ReplicateCounter:
    """Auto-assign replicate IDs per condition."""

    def __init__(self):
        self._counts: dict[str, int] = {}

    def next(self, condition: str) -> int:
        self._counts[condition] = self._counts.get(condition, 0) + 1
        return self._counts[condition]


def _parse_channel(channel: str | None) -> int | str | None:
    if channel is None:
        return None
    try:
        return int(channel)
    except ValueError:
        return channel


def _save_config(output_dir: Path, params: dict) -> None:
    """Save run parameters for reproducibility."""
    config = {}
    for k, v in params.items():
        if isinstance(v, Path):
            config[k] = str(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            config[k] = v

    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
