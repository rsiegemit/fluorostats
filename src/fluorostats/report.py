"""CSV report generation (per-file rows, per-condition summaries) and the
run_config.json record of how an output folder was produced."""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .io import OPTIONAL_READERS


# The reported metrics, in CSV column order: the one list every consumer (the
# per-file and per-condition CSVs, the CLI's plots and p-value tables) derives from.
METRICS_3D = [
    "volume_fraction",
    "n_components",
    "euler_number",
    "largest_component_fraction",
    "total_length_um",
    "n_branches",
    "n_junctions",
    "n_junction_nodes",
    "mean_branch_length_um",
    "coverage_tile_cv",
    "coverage_morans_i",
]

METRICS_2D = [
    "area_fraction",
    "n_components",
    "largest_component_fraction",
    "mean_cluster_area_px",
    "median_cluster_area_px",
    "coverage_tile_cv",
    "coverage_morans_i",
]

# Per report mode: (identifier columns that precede the metrics in the per-file
# CSV, the reported metrics, record-only columns that follow them). Record-only
# columns are written per file but not summarised, plotted or tested; the 2D ones
# were added after the others, so they come last to keep earlier columns in place.
_MODES = {
    "3d": (
        [
            "file",
            "condition",
            "replicate",
            "shape",
            "voxel_size_um",
            "voxel_size_source",
            "channel_used",
        ],
        METRICS_3D,
        [],
    ),
    "2d": (
        ["file", "condition", "replicate", "shape", "channel_used"],
        METRICS_2D,
        [
            "voxel_size_um",
            "voxel_size_source",
            "auto_crop_box",
            "mean_cluster_area_um2",
            "median_cluster_area_um2",
        ],
    ),
}


def _mode(mode: str) -> tuple[list[str], list[str], list[str]]:
    """``(identifier, metric, record-only columns)`` of a report mode; unknown modes are refused."""
    if mode not in _MODES:
        raise ValueError(f"Unknown report mode {mode!r}; expected one of {sorted(_MODES)}")
    return _MODES[mode]


def write_per_file_csv(
    rows: list[dict],
    out_path: Path,
    mode: str = "3d",
) -> pd.DataFrame:
    """Write per-file results to CSV. Returns the DataFrame."""
    ids, metrics, records = _mode(mode)
    df = pd.DataFrame(rows, columns=ids + metrics + records)

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
    _, metrics, _ = _mode(mode)
    available = [c for c in metrics if c in df.columns]

    summary = df.groupby("condition")[available].agg(["count", "mean", "std", "median"])

    # Flatten multi-level column names
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(str(out_path), index=False)
    return summary


def environment() -> dict:
    """Package versions + optional-reader availability, for the reproducibility record.

    ``packages["fluorostats"]`` is the version of the code that ran
    (``fluorostats.__version__``, not an installed distribution's), and
    ``fluorostats_commit`` its git commit when it runs from a checkout (see
    `_git_commit`), else None. (The resolved per-file ``voxel_size_um`` is
    recorded per row in ``per_file.csv``.)
    """
    import importlib.metadata as _im
    import importlib.util as _iu
    import platform

    from . import __version__

    pkgs: dict[str, str | None] = {"fluorostats": __version__}
    for p in ("numpy", "scipy", "scikit-image", "tifffile", "pandas", "matplotlib", "skan"):
        try:
            pkgs[p] = _im.version(p)
        except _im.PackageNotFoundError:
            pkgs[p] = None
    readers = {r: _iu.find_spec(r) is not None for r in OPTIONAL_READERS}
    return {
        "python": platform.python_version(),
        "packages": pkgs,
        "fluorostats_commit": _git_commit(Path(__file__).parent),
        "optional_readers": readers,
    }


def _git_commit(package_dir: Path) -> str | None:
    """Short SHA of the git checkout ``package_dir`` is tracked in, with ``-dirty``
    when tracked files under it have changes; None when it is not tracked by git
    (an installed package) or git is unavailable."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=package_dir, capture_output=True, encoding="utf-8", check=True
        ).stdout.strip()

    try:
        git("ls-files", "--error-unmatch", "__init__.py")
        sha = git("rev-parse", "--short", "HEAD")
        changed = git("status", "--porcelain", "--untracked-files=no", "--", ".")
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if changed else sha


def write_run_config(
    output_dir: Path, params: dict, nested: dict | None = None, tool: str | None = None
) -> Path:
    """Write ``run_config.json``: the run's parameters plus `environment`, so an
    output folder says how it was produced.

    ``tool`` (the command that ran, e.g. ``"depth"``) is the first key, then
    ``"status": "complete"`` (replacing `mark_output_dir`'s marker); the tool is how
    `claim_output_dir` knows which tool's outputs a folder holds. From ``params``
    only paths (as strings) and plain scalars are kept. ``nested`` adds structured
    values verbatim (e.g. a batch driver's resolved manifest); anything not
    JSON-native in it is written as its string form.
    """
    config: dict = {} if tool is None else {"tool": tool, "status": "complete"}
    for k, v in params.items():
        if isinstance(v, Path):
            config[k] = str(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            config[k] = v
    if nested:
        config.update(json.loads(json.dumps(nested, default=str)))
    config["_environment"] = environment()

    config_path = output_dir / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_path


def remove_outputs(paths: Iterable[Path]) -> list[Path]:
    """Delete those of ``paths`` that exist (an earlier run's outputs); return them sorted.

    A tool calls this at the start of a run with the names it writes, so no earlier
    result sits next to this run's. Only the given paths are touched: a symlink is
    removed itself, never followed.
    """
    removed = sorted(p for p in paths if os.path.lexists(p))
    for path in removed:
        path.unlink()
    return removed


def mark_output_dir(out_dir: Path, tool: str) -> None:
    """Create ``out_dir`` and write the ``run_config.json`` marker
    ``{"tool": tool, "status": "incomplete"}``.

    A run marks its folder before its first output, so a folder that an interrupted
    or failed run left is still recognised as ``tool``'s on rerun; `write_run_config`
    replaces the marker with the full record (``"status": "complete"``) at the end.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = {"tool": tool, "status": "incomplete"}
    (out_dir / "run_config.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")


def claim_output_dir(
    out_dir: Path, tool: str, owned: Iterable[Path], overwrite: bool = False
) -> list[Path]:
    """Make ``out_dir`` ready for a run of ``tool``; return the earlier outputs removed.

    ``owned`` are the paths the tool writes (``run_config.json`` among them). The
    folder's ``run_config.json`` must name ``tool``: a folder another tool, or a run
    that recorded no tool (one from before 0.9), wrote is refused, and so is one that
    holds any owned name but no record (files the user put there). Then
    `remove_outputs` deletes the owned files and, when the folder exists,
    `mark_output_dir` marks it as ``tool``'s. ``overwrite`` replaces the files
    anyway, with a warning naming those that were not this tool's. Files are named
    relative to ``out_dir``. ValueError otherwise, and for a folder at an owned name
    (``overwrite`` or not: a run deletes files only).
    """
    out_dir = Path(out_dir)
    owned = list(owned)
    folders = [p for p in owned if p.is_dir() and not p.is_symlink()]
    if folders:
        raise ValueError(
            f"output folder {out_dir}: {_names(folders, out_dir)} is a folder, where "
            f"fluorostats {tool} writes a file. Move or rename it, or choose another folder."
        )
    record = out_dir / "run_config.json"
    present = [p for p in owned if os.path.lexists(p) and p != record]
    found = None
    if record.exists():
        recorded = _read_record(record)
        if recorded is None or recorded.get("tool") != tool:
            found = f"holds {_whose(recorded)}"
    elif present:
        found = f"holds {_names(present, out_dir)} but no run_config.json"
    if found is None:
        removed = remove_outputs(owned)
    else:
        removed = _foreign(out_dir, tool, owned, overwrite, found)
    if out_dir.is_dir():
        mark_output_dir(out_dir, tool)
    return removed


def _read_record(record: Path) -> dict | None:
    """A run_config.json's contents; None when it is unreadable or not a JSON object
    (the caller then treats the folder as not this tool's)."""
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _whose(record: dict | None) -> str:
    """Who wrote a folder whose run_config.json holds ``record`` (None: unreadable)."""
    if record is not None and "tool" not in record:
        return (
            "a run from an earlier fluorostats version (its run_config.json names no "
            "tool; pass --overwrite once to rerun there)"
        )
    recorded = None if record is None else record["tool"]
    if isinstance(recorded, str):
        return f"written by fluorostats {recorded} (run_config.json)"
    return "an unrecorded run (run_config.json)"


def _foreign(out_dir: Path, tool: str, owned: list[Path], overwrite: bool, found: str):
    """Refuse a folder whose files this tool did not write, or (``overwrite``) say
    which are replaced and remove them."""
    if not overwrite:
        raise ValueError(
            f"output folder {out_dir} {found}; fluorostats {tool} would replace files it "
            "did not write. Choose another folder, or pass --overwrite (overwrite=True)."
        )
    names = _names([p for p in owned if os.path.lexists(p)], out_dir)
    warnings.warn(
        f"{out_dir}: replacing {names}, not written by an earlier fluorostats {tool} run.",
        RuntimeWarning,
        stacklevel=3,
    )
    return remove_outputs(owned)


def _names(paths: list[Path], out_dir: Path) -> str:
    """``paths`` relative to ``out_dir``, sorted and comma-separated, for a message."""
    return ", ".join(sorted(p.relative_to(out_dir).as_posix() for p in paths))


def stale_notices(removed: list[Path]) -> list[str]:
    """One line per removed earlier output that this run did not write again."""
    return [f"Removed stale {p} (not produced by this run)" for p in removed if not p.exists()]
