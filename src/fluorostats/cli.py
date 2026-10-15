"""CLI entry point: fluorostats quant3d / quant2d / depth / viability / formats."""

from __future__ import annotations

import functools
from pathlib import Path

import click
import matplotlib
import pandas as pd

from . import cli_measure, cli_paths
from . import io as iq_io
from . import (
    plots,
    pubfig,
    report,
    segment,
    stats,
)
from .cli_paths import CONDITION_FROM  # the --condition-from registry, importable from here

# REU-5: no committed benchmark table chose these result-changing defaults yet (an
# open item): quant3d --threshold-scale 0.9 and --min-branch-length 5 µm, quant2d
# --threshold li and --bg-radius 15.

# Every command that writes an output folder takes the same --overwrite (CFG-1).
_overwrite_option = click.option(
    "--overwrite",
    is_flag=True,
    help="Replace files in the output folder that another tool or the user wrote "
    "(default: refuse; an earlier run of the same command is always replaced)",
)


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
@_overwrite_option
def depth_cmd(manifest, output_dir, overwrite):
    """Depth-penetration analysis from a JSON manifest.

    Compares how far a fluorescent probe penetrates a material across
    conditions: blank-subtracted and surface-normalised mean-intensity-vs-
    depth profiles plus AUC over one or more depth windows. Writes tidy CSVs
    (for Prism/replotting) and publication figures. See
    ``fluorostats.depth_batch`` for the manifest schema.
    """
    from . import depth_batch  # on use: the other commands never load the batch driver

    out = _run_driver(depth_batch.run, manifest, output_dir, overwrite)
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

    out = _run_driver(viability_batch.run, manifest, output_dir)
    click.echo(f"Done. Outputs in {out}/")


def _run_driver(run, manifest: str, *args) -> Path:
    """A batch driver's ``run``. A problem with the manifest or the data it names (a
    ValueError, which JSON syntax errors and file-format errors are too) is printed as
    ``Error: <message>`` with exit status 1, not as a traceback."""
    try:
        return run(Path(manifest), *args)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


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
@_overwrite_option
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
    overwrite,
):
    """Quantify 3D confocal z-stacks: volume fraction, connectivity, skeleton,
    and spatial-coverage uniformity (tile CV + Moran's I)."""
    input_path = Path(input_dir)
    output_path = cli_paths.output_outside_input(input_path, Path(output_dir))

    # Resolve channel override
    ch_override = _parse_channel(channel)

    # Find volume files
    files = cli_paths.find_files(input_path, iq_io.VOLUME_SUFFIXES, iq_io.DIRECTORY_FORMATS)
    if not files:
        raise click.UsageError(f"No volume files found in {input_path}")
    config = dict(locals())  # recorded in run_config.json once the run has succeeded
    previous = cli_paths.claim_output(output_path, "quant3d", overwrite)

    click.echo(f"Found {len(files)} volume file(s)")

    measure = functools.partial(
        cli_measure.measure_volume,
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
    rows, failures = cli_measure.quantify_files(
        files, condition_from, "Processing volumes", measure
    )

    # Every reported metric is plotted except n_branches, as in earlier releases.
    plotted = [m for m in report.METRICS_3D if m != "n_branches"]
    _report_and_plot(
        rows, output_path, "3d", plotted, no_plots, "3D Confocal Quantification Summary"
    )
    _finish(output_path, config, previous, "quant3d", failures, len(files))


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
@_overwrite_option
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
    overwrite,
):
    """Quantify 2D fluorescence images: area fraction (cell coverage) and
    spatial-coverage uniformity (tile CV + Moran's I)."""
    input_path = Path(input_dir)
    output_path = cli_paths.output_outside_input(input_path, Path(output_dir))

    ch_override = _parse_channel(channel)

    files = cli_paths.find_files(input_path, iq_io.IMAGE_SUFFIXES)
    if not files:
        raise click.UsageError(
            f"No image files found in {input_path} "
            f"(quant2d matches {sorted(iq_io.IMAGE_SUFFIXES)}). "
            "Vendor volume formats (.oib/.czi/.nd2/.lif) are handled by "
            "`quant3d`."
        )
    config = dict(locals())  # recorded in run_config.json once the run has succeeded
    previous = cli_paths.claim_output(output_path, "quant2d", overwrite)

    click.echo(f"Found {len(files)} image file(s)")

    measure = functools.partial(
        cli_measure.measure_image,
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
    rows, failures = cli_measure.quantify_files(files, condition_from, "Processing images", measure)
    _report_and_plot(
        rows, output_path, "2d", list(report.METRICS_2D), no_plots, "2D Coverage Summary"
    )
    _finish(output_path, config, previous, "quant2d", failures, len(files))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _finish(output_path: Path, config: dict, previous: list[Path], tool: str, failures, n) -> None:
    """Write ``failures.csv`` (file, condition, error) when files of the ``n`` failed,
    record the run (last, so only a completed run claims the folder), name the earlier
    outputs it removed and did not write again, then exit 1 naming the failed files."""
    if failures:
        pd.DataFrame(failures, columns=["file", "condition", "error"]).to_csv(
            output_path / "failures.csv", index=False
        )
    report.write_run_config(output_path, config, tool=tool)
    for line in report.stale_notices(previous):
        click.echo(line)
    if failures:
        listed = "".join(f"\n  {f['condition']}/{f['file']}: {f['error']}" for f in failures)
        raise click.ClickException(
            f"{len(failures)} of {n} file(s) could not be measured (failures.csv):{listed}"
        )


def _draw_plots_and_pvalues(df, metrics, plot_dir, summary_title) -> None:
    """Draw the per-metric plots, the summary panel and pvalues.csv into ``plot_dir``,
    in the house figure theme."""
    pubfig.apply_style("house")
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


def _parse_channel(channel: str | None) -> int | str | None:
    if channel is None:
        return None
    try:
        return int(channel)
    except ValueError:
        return channel
