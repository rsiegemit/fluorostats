"""Where the CLI's inputs come from and where its outputs go.

Input discovery (the files and folder volumes under ``--input``, and each one's
condition label) and output-folder hygiene (refusing an output inside the input,
mirroring overlay paths, removing an earlier run's outputs before a rerun).
"""

from __future__ import annotations

from pathlib import Path

import click

from . import report

# How a file's condition label is derived from its path (--condition-from), by name.
CONDITION_FROM = {
    "parent": lambda path: path.parent.name,
    "grandparent": lambda path: path.parent.parent.name,
    "filename": lambda path: path.stem,  # the name without its extension
}


def get_condition(fpath: Path, mode: str) -> str:
    """Derive the experimental condition label from the file path (:data:`CONDITION_FROM`)."""
    return CONDITION_FROM[mode](fpath)


def find_files(root: Path, suffixes: set[str], directory_formats=()) -> list[Path]:
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


def overlay_path(fpath: Path, input_root: Path, overlay_dir: Path) -> Path:
    """The QC overlay for ``fpath``, mirroring its subfolder under the input root so
    same-named files in different condition folders do not overwrite each other."""
    rel = fpath.relative_to(input_root).parent
    return overlay_dir / rel / f"{fpath.stem}_overlay.png"


def output_outside_input(input_path: Path, output_path: Path) -> Path:
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


def remove_owned_outputs(output_path: Path) -> list[Path]:
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
