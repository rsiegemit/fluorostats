"""Per-file measurement for the quant commands: one file in, one row of metrics out.

`measure_volume` (quant3d) and `measure_image` (quant2d) load a file, segment the
measured channel with the command's options and return its row; `quantify_files`
runs one of them over the input files, tagging each row with its condition and
replicate. The commands themselves (options, output folder, reports) are in `cli`.
"""

from __future__ import annotations

import click

from . import cli_paths
from . import io as iq_io
from . import metrics_2d, metrics_3d, preprocess, qc, segment, skeleton, spatial


def quantify_files(files, condition_from, label, measure) -> list[dict]:
    """Run ``measure(path) -> dict`` over files, tagging condition + replicate."""
    rows: list[dict] = []
    rep_counter = _ReplicateCounter()
    with click.progressbar(files, label=label) as bar:
        for fpath in bar:
            condition = cli_paths.get_condition(fpath, condition_from)
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


def measure_volume(
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
        qc.overlay_3d(vol, mask, cli_paths.overlay_path(fpath, input_root, overlay_dir))
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


def measure_image(fpath, *, ch_override, auto_crop, overlay_dir, input_root, **seg) -> dict:
    """quant2d's per-file metrics (area fraction, clusters, coverage uniformity)."""
    arr, meta = iq_io.load_image(fpath)
    crop_box = ""
    if auto_crop:  # crop microscope borders before channel selection
        arr, (y0, y1, x0, x1) = preprocess.auto_crop(arr)
        crop_box = f"[{y0}:{y1}, {x0}:{x1}]"
    ch = preprocess.green_channel_index(meta["channel_names"], arr.shape[0], ch_override)
    img, mask = _segment(arr[ch], **seg)
    if overlay_dir is not None:
        qc.overlay_2d(img, mask, cli_paths.overlay_path(fpath, input_root, overlay_dir))
    return {
        "shape": str(arr.shape),
        "voxel_size_um": str(meta["voxel_size_um"]),
        "voxel_size_source": meta.get("voxel_size_source", ""),
        "auto_crop_box": crop_box,
        "channel_used": _channel_label(ch, meta["channel_names"]),
        **metrics_2d.coverage_metrics(mask, voxel_size_um=meta["voxel_size_um"]),
        **spatial.coverage_heterogeneity(mask),
    }


def _channel_label(index: int, names: list[str]) -> str:
    """The measured channel as ``"<index> (<name>)"`` for the per-file record."""
    return f"{index} ({names[index]})" if index < len(names) else str(index)


class _ReplicateCounter:
    """Auto-assign replicate IDs per condition."""

    def __init__(self):
        self._counts: dict[str, int] = {}

    def next(self, condition: str) -> int:
        self._counts[condition] = self._counts.get(condition, 0) + 1
        return self._counts[condition]
