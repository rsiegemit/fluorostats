"""CLI entry point: fluorostats quant3d / quant2d / depth / viability / formats."""

from __future__ import annotations

import functools
from pathlib import Path

import click
import matplotlib

from . import io as iq_io
from . import (
    metrics_2d,
    metrics_3d,
    plots,
    preprocess,
    qc,
    report,
    segment,
    skeleton,
    spatial,
    stats,
)

# How a file's condition label is derived from its path (--condition-from), by name.
CONDITION_FROM = {
    "parent": lambda path: path.parent.name,
    "grandparent": lambda path: path.parent.parent.name,
    "filename": lambda path: path.stem,  # the name without its extension
}


@click.group()
def cli():
    """FluoroStats — Universal fluorescence microscopy image quantification."""
    matplotlib.use("Agg")  # every command writes figures to files; none opens a window


@cli.command()
def formats():
    """List all supported file formats and their availability."""
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
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(),
    help="Output folder; default: the manifest's output_dir",
)
def depth_cmd(manifest, output_dir):
    """Depth-penetration analysis from a JSON manifest.

    Compares how far a fluorescent probe penetrates a material across
    conditions: blank-subtracted and surface-normalised mean-intensity-vs-
    depth profiles plus AUC over one or more depth windows. Writes tidy CSVs
    (for Prism/replotting) and publication figures. See
    ``fluorostats.depth_batch`` for the manifest schema.
    """
    from . import depth_batch  # on use: the other commands never load the batch driver

    out = depth_batch.run(Path(manifest), output_dir)
    click.echo(f"Done. Outputs in {out}/")


# ---------------------------------------------------------------------------
# viability — Live/Dead viability analysis (manifest driven)
# ---------------------------------------------------------------------------


@cli.command("viability")
@click.argument("manifest", type=click.Path(exists=True))
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(),
    help="Output folder; default: the manifest's output_dir",
)
def viability_cmd(manifest, output_dir):
    """Live/Dead viability analysis from a JSON manifest.

    Quantifies two-channel Live/Dead z-stacks grouped by condition: area /
    coverage and count-based viability (cc/watershed/maxima consensus and
    spread), transparent regime detection, per-FOV cell density, and 2D-MIP-
    vs-3D coverage bias. Writes tidy CSVs. See ``fluorostats.viability_batch``
    for the manifest schema.
    """
    from . import viability_batch  # on use: the other commands never load the batch driver

    out = viability_batch.run(Path(manifest), output_dir)
    click.echo(f"Done. Outputs in {out}/")


# ---------------------------------------------------------------------------
# quant3d
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "input_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder with .oib/.oif/.tif volumes",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Output folder for results (not inside --input)",
)
@click.option(
    "--condition-from",
    "condition_from",
    type=click.Choice(list(CONDITION_FROM)),
    default="parent",
    show_default=True,
    help="How to assign condition labels",
)
@click.option(
    "--channel",
    default=None,
    help="Channel to measure, by index (int) or name substring; default: the channel named as a green dye, else a positional guess (warned). CAN AFFECT RESULTS",
)
@click.option(
    "--threshold",
    type=click.Choice(segment.METHODS),
    default="otsu",
    show_default=True,
    help="Threshold algorithm (CAN AFFECT RESULTS); see segment.binarize",
)
@click.option(
    "--threshold-scale",
    type=click.FloatRange(min=0, min_open=True),
    default=0.9,
    show_default=True,
    help="Scale threshold by this factor (lower=more sensitive; CAN AFFECT RESULTS)",
)
@click.option(
    "--min-size",
    type=click.IntRange(min=0),
    default=64,
    show_default=True,
    help="Remove objects smaller than this, in voxels (CAN AFFECT RESULTS)",
)
@click.option(
    "--sigma",
    type=click.FloatRange(min=0),
    default=1.0,
    show_default=True,
    help="Gaussian blur sigma in pixels, in-plane per z-slice (CAN AFFECT RESULTS)",
)
@click.option(
    "--bg-radius",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Background subtraction disk radius in pixels (0=off; enabling is recommended for confocal; CAN AFFECT RESULTS)",
)
@click.option(
    "--opening",
    type=click.Choice(list(segment.OPENINGS)),
    default="ball",
    show_default=True,
    help="Salt-noise opening after thresholding: ball (3x3x3; erases structures thinner "
    "than 3 z-slices), disk (per z-slice), none (CAN AFFECT RESULTS)",
)
@click.option("--no-overlays", is_flag=True, help="Skip QC overlay PNGs")
@click.option("--no-plots", is_flag=True, help="Skip comparison plots")
@click.option(
    "--no-skeleton",
    is_flag=True,
    help="Skip skeleton analysis (faster); skeleton columns are left blank",
)
@click.option(
    "--prune/--no-prune",
    default=False,
    show_default=True,
    help="Prune short skeleton spurs before counting (field-standard branch/junction counts, "
    "matching Fiji AnalyzeSkeleton / AngioTool / REAVER); off = raw skeleton (CAN AFFECT RESULTS)",
)
@click.option(
    "--min-branch-length",
    type=click.FloatRange(min=0),
    default=5.0,
    show_default=True,
    help="Minimum branch length in µm for spur pruning when --prune is set (CAN AFFECT RESULTS)",
)
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
    opening,
    no_overlays,
    no_plots,
    no_skeleton,
    prune,
    min_branch_length,
):
    """Quantify 3D confocal z-stacks: volume fraction, connectivity, skeleton,
    and spatial-coverage uniformity (tile CV + Moran's I)."""
    input_path = Path(input_dir)
    output_path = _output_outside_input(input_path, Path(output_dir))

    # Resolve channel override
    ch_override = _parse_channel(channel)

    # Find volume files
    files = _find_files(input_path, iq_io.VOLUME_SUFFIXES, iq_io.DIRECTORY_FORMATS)
    if not files:
        raise click.UsageError(f"No volume files found in {input_path}")
    config = dict(locals())  # recorded in run_config.json once the run has succeeded
    output_path.mkdir(parents=True, exist_ok=True)
    previous = _remove_owned_outputs(output_path)

    click.echo(f"Found {len(files)} volume file(s)")

    measure = functools.partial(
        _measure_volume,
        ch_override=ch_override,
        sigma=sigma,
        bg_radius=bg_radius,
        opening=opening,
        threshold=threshold,
        threshold_scale=threshold_scale,
        min_size=min_size,
        no_skeleton=no_skeleton,
        prune=prune,
        min_branch_length=min_branch_length,
        overlay_dir=None if no_overlays else output_path / "overlays",
        input_root=input_path,
    )
    rows = _quantify_files(files, condition_from, "Processing volumes", measure)

    # Every reported metric is plotted except n_branches, as in earlier releases.
    plotted = [m for m in report.METRICS_3D if m != "n_branches"]
    _report_and_plot(
        rows, output_path, "3d", plotted, no_plots, "3D Confocal Quantification Summary"
    )
    _finish(output_path, config, previous)


# ---------------------------------------------------------------------------
# quant2d
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "input_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder with .png/.tif images",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Output folder for results (not inside --input)",
)
@click.option(
    "--condition-from",
    "condition_from",
    type=click.Choice(list(CONDITION_FROM)),
    default="parent",
    show_default=True,
    help="How to assign condition labels",
)
@click.option(
    "--channel",
    default=None,
    help="Channel to measure, by index (int) or name substring; default: the channel named as a green dye, else a positional guess (warned). CAN AFFECT RESULTS",
)
@click.option(
    "--threshold",
    type=click.Choice(segment.METHODS),
    default="li",
    show_default=True,
    help="Threshold algorithm (CAN AFFECT RESULTS); see segment.binarize",
)
@click.option(
    "--threshold-scale",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
    help="Scale threshold by this factor (lower=more sensitive; CAN AFFECT RESULTS)",
)
@click.option(
    "--min-size",
    type=click.IntRange(min=0),
    default=64,
    show_default=True,
    help="Remove objects smaller than this, in pixels (CAN AFFECT RESULTS)",
)
@click.option(
    "--sigma",
    type=click.FloatRange(min=0),
    default=1.0,
    show_default=True,
    help="Gaussian blur sigma in pixels (CAN AFFECT RESULTS)",
)
@click.option(
    "--bg-radius",
    type=click.IntRange(min=0),
    default=15,
    show_default=True,
    help="Background subtraction disk radius in pixels (0=off; CAN AFFECT RESULTS)",
)
@click.option(
    "--auto-crop/--no-auto-crop",
    default=True,
    show_default=True,
    help="Auto-crop microscope borders before measuring (CAN AFFECT RESULTS)",
)
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
    output_path = _output_outside_input(input_path, Path(output_dir))

    ch_override = _parse_channel(channel)

    files = _find_files(input_path, iq_io.IMAGE_SUFFIXES)
    if not files:
        raise click.UsageError(
            f"No image files found in {input_path} "
            f"(quant2d matches {sorted(iq_io.IMAGE_SUFFIXES)}). "
            "Vendor volume formats (.oib/.czi/.nd2/.lif) are handled by "
            "`quant3d`."
        )
    config = dict(locals())  # recorded in run_config.json once the run has succeeded
    output_path.mkdir(parents=True, exist_ok=True)
    previous = _remove_owned_outputs(output_path)

    click.echo(f"Found {len(files)} image file(s)")

    measure = functools.partial(
        _measure_image,
        ch_override=ch_override,
        sigma=sigma,
        bg_radius=bg_radius,
        threshold=threshold,
        threshold_scale=threshold_scale,
        min_size=min_size,
        auto_crop=auto_crop,
        overlay_dir=None if no_overlays else output_path / "overlays",
        input_root=input_path,
    )
    rows = _quantify_files(files, condition_from, "Processing images", measure)
    _report_and_plot(
        rows, output_path, "2d", list(report.METRICS_2D), no_plots, "2D Coverage Summary"
    )
    _finish(output_path, config, previous)


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
            rows.append(
                {
                    "file": fpath.name,
                    "condition": condition,
                    "replicate": replicate,
                    **measure(fpath),
                }
            )
    return rows


def _segment(img, *, sigma, bg_radius, threshold, threshold_scale, min_size, opening="ball"):
    """The CLI's options mapped onto `segment.segment_channel` (quant2d has no
    --opening: its 2D opening is disk(1) for every choice but "none")."""
    return segment.segment_channel(
        img,
        method=threshold,
        min_size=min_size,
        threshold_scale=threshold_scale,
        sigma=sigma,
        bg_radius=bg_radius,
        opening=opening,
    )


def _measure_volume(
    fpath, *, ch_override, no_skeleton, prune, min_branch_length, overlay_dir, input_root, **seg
) -> dict:
    """quant3d's per-file metrics (volume fraction, connectivity, skeleton, coverage)."""
    arr, meta = iq_io.load_volume(fpath)
    voxel_size = meta["voxel_size_um"]
    ch = preprocess.green_channel_index(meta["channel_names"], arr.shape[0], ch_override)
    vol, mask = _segment(arr[ch], **seg)

    # --no-skeleton: same keys as skeleton_metrics() so the CSV keeps its shape, but
    # NaN (not measured) rather than 0 (measured, nothing there), so skipped
    # metrics are left out of plots and statistics instead of entering as zeros.
    skel = dict(skeleton.UNMEASURED_METRICS)
    if not no_skeleton:
        skel = skeleton.skeleton_metrics(
            mask, voxel_size_um=voxel_size, prune=prune, min_branch_length_um=min_branch_length
        )
    if overlay_dir is not None:
        qc.overlay_3d(vol, mask, _overlay_path(fpath, input_root, overlay_dir))
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


def _measure_image(fpath, *, ch_override, auto_crop, overlay_dir, input_root, **seg) -> dict:
    """quant2d's per-file metrics (area fraction, clusters, coverage uniformity)."""
    arr, meta = iq_io.load_image(fpath)
    crop_box = ""
    if auto_crop:  # crop microscope borders before channel selection
        arr, (y0, y1, x0, x1) = preprocess.auto_crop(arr)
        crop_box = f"[{y0}:{y1}, {x0}:{x1}]"
    ch = preprocess.green_channel_index(meta["channel_names"], arr.shape[0], ch_override)
    img, mask = _segment(arr[ch], **seg)
    if overlay_dir is not None:
        qc.overlay_2d(img, mask, _overlay_path(fpath, input_root, overlay_dir))
    return {
        "shape": str(arr.shape),
        "voxel_size_um": str(meta["voxel_size_um"]),
        "voxel_size_source": meta.get("voxel_size_source", ""),
        "auto_crop_box": crop_box,
        "channel_used": _channel_label(ch, meta["channel_names"]),
        **metrics_2d.coverage_metrics(mask, voxel_size_um=meta["voxel_size_um"]),
        **spatial.coverage_heterogeneity(mask),
    }


def _overlay_path(fpath: Path, input_root: Path, overlay_dir: Path) -> Path:
    """The QC overlay for ``fpath``, mirroring its subfolder under the input root so
    same-named files in different condition folders do not overwrite each other."""
    rel = fpath.relative_to(input_root).parent
    return overlay_dir / rel / f"{fpath.stem}_overlay.png"


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

    Draws only when plots are enabled and there is more than one condition.
    """
    if not no_plots and len(df["condition"].unique()) > 1:
        _draw_plots_and_pvalues(df, metrics, output_path / "plots", summary_title)


def _finish(output_path: Path, config: dict, previous: list[Path]) -> None:
    """Record the run (last, so only a completed run claims the folder) and name the
    earlier outputs it removed and did not write again."""
    report.write_run_config(output_path, config)
    for line in report.stale_notices(previous):
        click.echo(line)


def _draw_plots_and_pvalues(df, metrics, plot_dir, summary_title) -> None:
    """Draw the per-metric plots, the summary panel and pvalues.csv into ``plot_dir``."""
    has_replicates = any(len(df[df["condition"] == c]) > 1 for c in df["condition"].unique())
    # unmeasured metrics (e.g. skeleton columns under --no-skeleton) are neither
    # plotted nor listed as untestable pairs in pvalues.csv
    metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]

    for metric in metrics:
        if has_replicates:
            plots.bar_mean_sem(df, metric, plot_dir / f"{metric}_bar.png")
        plots.boxplot_by_condition(df, metric, plot_dir / f"{metric}_box.png")

    # p-values first, so the summary panel's stars use the same BH q-values as pvalues.csv
    pvals = stats.compute_pvalues(df, metrics) if has_replicates else None
    plots.summary_panel(
        df,
        metrics,
        plot_dir / "summary_panel.png",
        title=summary_title,
        show_pvalues=has_replicates,
        pvalues=pvals,
    )

    if pvals is not None:
        if len(pvals) > 0:
            pvals.to_csv(str(plot_dir / "pvalues.csv"), index=False)
            click.echo(f"\nStatistical comparisons: {plot_dir / 'pvalues.csv'}")

    click.echo(f"Plots saved to {plot_dir}")


def _output_outside_input(input_path: Path, output_path: Path) -> Path:
    """``output_path``, refused when it is (inside) the input folder.

    Inputs are found recursively, so a rerun would read this tool's own overlay and
    plot images back in as samples.
    """
    if output_path.resolve().is_relative_to(input_path.resolve()):
        raise click.UsageError(
            f"--output {output_path} is inside the input folder {input_path}; a rerun "
            "would read its overlay/plot images as samples. Choose an output folder "
            "outside --input."
        )
    return output_path


def _remove_owned_outputs(output_path: Path) -> list[Path]:
    """Delete the files an earlier run wrote; return the paths deleted.

    Only names this tool writes are touched (any reported metric's ``_bar``/``_box``
    PNG, the summary panel, ``pvalues.csv``, ``*_overlay.png`` under ``overlays/``,
    ``run_config.json``), so an earlier run's results never sit next to this run's
    and nothing else in the folder is removed.
    """
    plot_dir = output_path / "plots"
    owned = [
        plot_dir / f"{m}_{kind}.png"
        for m in {*report.METRICS_2D, *report.METRICS_3D}
        for kind in ("bar", "box")
    ]
    owned += [
        plot_dir / "summary_panel.png",
        plot_dir / "pvalues.csv",
        output_path / "run_config.json",
    ]
    owned += (output_path / "overlays").rglob("*_overlay.png")
    return report.remove_outputs(owned)


def _find_files(root: Path, suffixes: set[str], directory_formats=()) -> list[Path]:
    """Recursively find files with matching suffixes, in sorted order.

    A folder that one of ``directory_formats`` accepts (e.g. a Keyence BZ-X export,
    one volume stored as per-slice files) is one input: it is returned in place of
    the files inside it.
    """
    entries = [root, *sorted(root.rglob("*"))]
    folders = [d for d in entries if d.is_dir() and any(f.probe(d) for f in directory_formats)]
    files = [
        f
        for f in entries
        if f.is_file()
        and f.suffix.lower() in suffixes
        and not any(f.is_relative_to(d) for d in folders)
    ]
    return sorted(folders + files)


def _get_condition(fpath: Path, mode: str) -> str:
    """Derive the experimental condition label from the file path (:data:`CONDITION_FROM`)."""
    return CONDITION_FROM[mode](fpath)


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
