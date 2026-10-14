"""CLI entry point: fluorostats quant3d / quant2d / depth / viability / formats."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import click

from . import io as iq_io
from . import metrics_2d, metrics_3d, plots, preprocess, qc, report, segment, skeleton, spatial


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
    click.echo("\nInstall all readers: pip install " + " ".join(iq_io.OPTIONAL_READERS))
    click.echo('(from a fluorostats checkout: pip install -e ".[all]")')


# ---------------------------------------------------------------------------
# depth — probe-penetration / permeability analysis (manifest driven)
# ---------------------------------------------------------------------------

@cli.command("depth")
@click.argument("manifest", type=click.Path(exists=True))
@click.option("--output", "output_dir", default=None, type=click.Path(),
              help="Override the manifest's output_dir")
def depth_cmd(manifest, output_dir):
    """Depth-penetration analysis from a JSON manifest.

    Compares how far a fluorescent probe penetrates a material across
    conditions: blank-subtracted and surface-normalised mean-intensity-vs-
    depth profiles plus AUC over one or more depth windows. Writes tidy CSVs
    (for Prism/replotting) and publication figures. See
    ``fluorostats.depth_batch`` for the manifest schema.
    """
    from . import depth_batch
    out = depth_batch.run(Path(manifest), output_dir)
    click.echo(f"Done. Outputs in {out}/")


# ---------------------------------------------------------------------------
# viability — Live/Dead viability analysis (manifest driven)
# ---------------------------------------------------------------------------

@cli.command("viability")
@click.argument("manifest", type=click.Path(exists=True))
@click.option("--output", "output_dir", default=None, type=click.Path(),
              help="Override the manifest's output_dir")
def viability_cmd(manifest, output_dir):
    """Live/Dead viability analysis from a JSON manifest.

    Quantifies two-channel Live/Dead z-stacks grouped by condition: area /
    coverage and count-based viability (cc/watershed/maxima consensus and
    spread), transparent regime detection, per-FOV cell density, and 2D-MIP-
    vs-3D coverage bias. Writes tidy CSVs. See ``fluorostats.viability_batch``
    for the manifest schema.
    """
    from . import viability_batch
    out = viability_batch.run(Path(manifest), output_dir)
    click.echo(f"Done. Outputs in {out}/")


# ---------------------------------------------------------------------------
# quant3d
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True), help="Folder with .oib/.oif/.tif volumes")
@click.option("--output", "output_dir", required=True, type=click.Path(), help="Output folder for results")
@click.option("--condition-from", "condition_from", type=click.Choice(["parent", "grandparent", "filename"]), default="parent", help="How to assign condition labels")
@click.option("--channel", default=None, help="Force channel by index (int) or name substring")
@click.option("--threshold", type=click.Choice(segment.METHODS), default="otsu",
              help="Threshold algorithm (CAN AFFECT RESULTS); see segment.binarize")
@click.option("--threshold-scale", type=float, default=0.9, help="Scale threshold by this factor (lower=more sensitive, default 0.9)")
@click.option("--min-size", type=int, default=64, help="Min object size in voxels")
@click.option("--sigma", type=float, default=1.0, help="Gaussian blur sigma")
@click.option("--bg-radius", type=int, default=0, help="Background subtraction disk radius (0=off; enabling is recommended for confocal)")
@click.option("--no-overlays", is_flag=True, help="Skip QC overlay PNGs")
@click.option("--no-plots", is_flag=True, help="Skip comparison plots")
@click.option("--no-skeleton", is_flag=True, help="Skip skeleton analysis (faster)")
@click.option("--prune/--no-prune", default=False,
              help="Prune short skeleton spurs before counting (field-standard branch/junction counts, "
                   "matching Fiji AnalyzeSkeleton / AngioTool / REAVER). Default off = raw skeleton.")
@click.option("--min-branch-length", type=float, default=5.0,
              help="Minimum branch length in µm for spur pruning when --prune is set (default 5).")
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
    prune,
    min_branch_length,
):
    """Quantify 3D confocal z-stacks: volume fraction, connectivity, skeleton,
    and spatial-coverage uniformity (tile CV + Moran's I)."""
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

    measure = functools.partial(
        _measure_volume, ch_override=ch_override, sigma=sigma, bg_radius=bg_radius,
        threshold=threshold, threshold_scale=threshold_scale, min_size=min_size,
        no_skeleton=no_skeleton, prune=prune, min_branch_length=min_branch_length,
        overlay_dir=None if no_overlays else output_path / "overlays")
    rows = _quantify_files(files, condition_from, "Processing volumes", measure)

    # Every reported metric is plotted except n_branches, as in earlier releases.
    plotted = [m for m in report.METRICS_3D if m != "n_branches"]
    _report_and_plot(rows, output_path, "3d", plotted, no_plots,
                     "3D Confocal Quantification Summary")


# ---------------------------------------------------------------------------
# quant2d
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True), help="Folder with .png/.tif images")
@click.option("--output", "output_dir", required=True, type=click.Path(), help="Output folder for results")
@click.option("--condition-from", "condition_from", type=click.Choice(["parent", "grandparent", "filename"]), default="parent", help="How to assign condition labels")
@click.option("--channel", default=None, help="Force channel by index (int) or name substring")
@click.option("--threshold", type=click.Choice(segment.METHODS), default="li",
              help="Threshold algorithm (CAN AFFECT RESULTS); see segment.binarize")
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
    """Quantify 2D fluorescence images: area fraction (cell coverage) and
    spatial-coverage uniformity (tile CV + Moran's I)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ch_override = _parse_channel(channel)

    files = _find_files(input_path, iq_io.IMAGE_SUFFIXES)
    if not files:
        click.echo(f"No image files found in {input_path} "
                   f"(quant2d matches {sorted(iq_io.IMAGE_SUFFIXES)}). "
                   "Vendor volume formats (.oib/.czi/.nd2/.lif) are handled by `quant3d`.")
        return

    click.echo(f"Found {len(files)} image file(s)")
    _save_config(output_path, locals())

    measure = functools.partial(
        _measure_image, ch_override=ch_override, sigma=sigma, bg_radius=bg_radius,
        threshold=threshold, threshold_scale=threshold_scale, min_size=min_size,
        auto_crop=auto_crop, overlay_dir=None if no_overlays else output_path / "overlays")
    rows = _quantify_files(files, condition_from, "Processing images", measure)
    _report_and_plot(rows, output_path, "2d", list(report.METRICS_2D), no_plots,
                     "2D Coverage Summary")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quantify_files(files, condition_from, label, measure) -> list[dict]:
    """Run ``measure(path) -> dict`` over files, tagging condition + replicate."""
    rows: list[dict] = []
    rep_counter = _ReplicateCounter()
    with click.progressbar(files, label=label) as bar:
        for fpath in bar:
            condition = _get_condition(fpath, condition_from)
            replicate = rep_counter.next(condition)
            rows.append({"file": fpath.name, "condition": condition,
                         "replicate": replicate, **measure(fpath)})
    return rows


def _segment(img, *, sigma, bg_radius, threshold, threshold_scale, min_size):
    """Denoise, optionally background-subtract, and threshold one channel."""
    img = preprocess.denoise(img, sigma=sigma)
    if bg_radius > 0:
        img = preprocess.background_subtract(img, radius=bg_radius)
    mask = segment.binarize(img, method=threshold, min_size=min_size,
                            threshold_scale=threshold_scale)
    return img, mask


def _measure_volume(fpath, *, ch_override, no_skeleton, prune, min_branch_length,
                    overlay_dir, **seg) -> dict:
    """quant3d's per-file metrics (volume fraction, connectivity, skeleton, coverage)."""
    arr, meta = iq_io.load_volume(fpath)
    voxel_size = meta["voxel_size_um"]
    ch = preprocess.green_channel_index(meta["channel_names"], arr.shape[0], ch_override)
    vol, mask = _segment(arr[ch], **seg)

    # same keys as skeleton_metrics() so the CSV row shape is identical
    # whether or not --no-skeleton is set.
    skel = dict(skeleton.EMPTY_METRICS)
    if not no_skeleton:
        skel = metrics_3d.skeleton_metrics(
            mask, voxel_size_um=voxel_size,
            prune=prune, min_branch_length_um=min_branch_length)
    if overlay_dir is not None:
        qc.overlay_3d(vol, mask, overlay_dir / f"{fpath.stem}_overlay.png")
    return {
        "shape": str(arr.shape),
        "voxel_size_um": str(voxel_size),
        "voxel_size_source": meta.get("voxel_size_source", ""),
        "channel_used": _channel_label(ch, meta["channel_names"]),
        "volume_fraction": metrics_3d.volume_fraction(mask),
        **metrics_3d.connectivity_metrics(mask),
        **skel,
        **spatial.coverage_heterogeneity(mask),
    }


def _measure_image(fpath, *, ch_override, auto_crop, overlay_dir, **seg) -> dict:
    """quant2d's per-file metrics (area fraction, clusters, coverage uniformity)."""
    arr, meta = iq_io.load_image(fpath)
    if auto_crop:   # crop microscope borders before channel selection
        arr, _ = preprocess.auto_crop(arr)
    ch = preprocess.green_channel_index(meta["channel_names"], arr.shape[0], ch_override)
    img, mask = _segment(arr[ch], **seg)
    if overlay_dir is not None:
        qc.overlay_2d(img, mask, overlay_dir / f"{fpath.stem}_overlay.png")
    return {
        "shape": str(arr.shape),
        "channel_used": _channel_label(ch, meta["channel_names"]),
        **metrics_2d.coverage_metrics(mask),
        **spatial.coverage_heterogeneity(mask),
    }


def _channel_label(index: int, names: list[str]) -> str:
    """The measured channel as ``"<index> (<name>)"`` for the per-file record."""
    return f"{index} ({names[index]})" if index < len(names) else str(index)


def _report_and_plot(rows, output_path, mode, metrics, no_plots, summary_title) -> None:
    """Write per-file + per-condition CSVs, echo the summary, then plots/p-values."""
    df = report.write_per_file_csv(rows, output_path / "per_file.csv", mode=mode)
    summary = report.write_per_condition_csv(df, output_path / "per_condition.csv", mode=mode)

    click.echo(f"\nPer-file results: {output_path / 'per_file.csv'}")
    click.echo(f"Per-condition summary: {output_path / 'per_condition.csv'}")

    click.echo("\n--- Per-condition summary ---")
    click.echo(summary.to_string(index=False))

    _write_plots_and_pvalues(df, metrics, output_path, no_plots, summary_title=summary_title)

def _write_plots_and_pvalues(df, metrics, output_path, no_plots, summary_title):
    """Shared quant2d/quant3d tail: per-metric plots, summary panel, p-values.

    No-op unless plots are enabled and there is more than one condition.
    """
    if no_plots or len(df["condition"].unique()) <= 1:
        return

    plot_dir = output_path / "plots"
    has_replicates = any(len(df[df["condition"] == c]) > 1 for c in df["condition"].unique())

    for metric in metrics:
        if metric not in df.columns or not df[metric].notna().any():
            continue
        if has_replicates:
            plots.bar_mean_sem(df, metric, plot_dir / f"{metric}_bar.png")
        plots.boxplot_by_condition(df, metric, plot_dir / f"{metric}_box.png")

    plots.summary_panel(
        df, metrics, plot_dir / "summary_panel.png",
        title=summary_title,
        show_pvalues=has_replicates,
    )

    if has_replicates:
        pvals = plots.compute_pvalues(df, metrics)
        if len(pvals) > 0:
            pvals.to_csv(str(plot_dir / "pvalues.csv"), index=False)
            click.echo(f"\nStatistical comparisons: {plot_dir / 'pvalues.csv'}")

    click.echo(f"Plots saved to {plot_dir}")


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


def _environment() -> dict:
    """Package versions + optional-reader availability, for the reproducibility record.

    (The resolved per-file ``voxel_size_um`` is recorded per row in ``per_file.csv``.)
    """
    import importlib.metadata as _im
    import importlib.util as _iu
    import platform

    pkgs: dict[str, str | None] = {}
    for p in ("fluorostats", "numpy", "scipy", "scikit-image", "tifffile",
              "pandas", "matplotlib", "skan"):
        try:
            pkgs[p] = _im.version(p)
        except Exception:  # pragma: no cover - core deps are always installed
            pkgs[p] = None
    readers = {r: _iu.find_spec(r) is not None
               for r in iq_io.OPTIONAL_READERS}
    return {"python": platform.python_version(), "packages": pkgs,
            "optional_readers": readers}


def _save_config(output_dir: Path, params: dict) -> None:
    """Save run parameters + environment for reproducibility."""
    config: dict = {}
    for k, v in params.items():
        if isinstance(v, Path):
            config[k] = str(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            config[k] = v
    config["_environment"] = _environment()

    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
