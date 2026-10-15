"""CSV report generation (per-file rows, per-condition summaries) and the
run_config.json record of how an output folder was produced."""

from __future__ import annotations

import json
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

    ``tool`` (the command that ran, e.g. ``"depth"``) is the first key; it is how
    `claim_output_dir` knows which tool's outputs a folder holds. From ``params``
    only paths (as strings) and plain scalars are kept. ``nested`` adds structured
    values verbatim (e.g. a batch driver's resolved manifest); anything not
    JSON-native in it is written as its string form.
    """
    config: dict = {} if tool is None else {"tool": tool}
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
    result sits next to this run's, and writes ``run_config.json`` last, so a run that
    fails part-way leaves no record claiming the folder. Only the given paths are touched.
    """
    removed = sorted(p for p in paths if p.exists())
    for path in removed:
        path.unlink()
    return removed


def claim_output_dir(
    out_dir: Path, tool: str, owned: Iterable[Path], overwrite: bool = False
) -> list[Path]:
    """Make ``out_dir`` ready for a run of ``tool``; return the earlier outputs removed.

    ``owned`` are the paths the tool writes (``run_config.json`` among them). The
    folder's ``run_config.json`` must name ``tool``: a folder another tool, or a run
    that recorded no tool, wrote is refused, and so is one that holds any owned name
    but no record (files the user put there, or a run that failed part-way). Then
    `remove_outputs` deletes the owned files. ``overwrite`` replaces them anyway,
    with a warning naming the files that were not this tool's. ValueError otherwise.
    """
    owned = list(owned)
    record = Path(out_dir) / "run_config.json"
    present = sorted(p.name for p in owned if p.exists() and p != record)
    if record.exists():
        recorded = _recorded_tool(record)
        if recorded != tool:
            who = f"written by fluorostats {recorded}" if recorded else "an unrecorded run"
            return _foreign(out_dir, tool, owned, overwrite, f"holds {who} (run_config.json)")
    elif present:
        found = f"holds {', '.join(present)} but no run_config.json"
        return _foreign(out_dir, tool, owned, overwrite, found)
    return remove_outputs(owned)


def _recorded_tool(record: Path) -> str | None:
    """The ``tool`` a run_config.json names; None when it names none or is unreadable
    (the caller then treats the folder as not this tool's)."""
    try:
        tool = json.loads(record.read_text(encoding="utf-8")).get("tool")
    except (OSError, ValueError, AttributeError):
        return None
    return tool if isinstance(tool, str) else None


def _foreign(out_dir: Path, tool: str, owned: list[Path], overwrite: bool, found: str):
    """Refuse a folder whose files this tool did not write, or (``overwrite``) say
    which are replaced and remove them."""
    if not overwrite:
        raise ValueError(
            f"output folder {out_dir} {found}; fluorostats {tool} would replace files it "
            "did not write. Choose another folder, or pass --overwrite (overwrite=True)."
        )
    names = ", ".join(sorted(p.name for p in owned if p.exists()))
    warnings.warn(
        f"{out_dir}: replacing {names}, not written by an earlier fluorostats {tool} run.",
        RuntimeWarning,
        stacklevel=3,
    )
    return remove_outputs(owned)


def stale_notices(removed: list[Path]) -> list[str]:
    """One line per removed earlier output that this run did not write again."""
    return [f"Removed stale {p} (not produced by this run)" for p in removed if not p.exists()]
