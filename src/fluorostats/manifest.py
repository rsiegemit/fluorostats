"""Shared checks for the JSON manifests the batch drivers read.

`depth_batch` and `viability_batch` both describe their input as groups of
stacks; `check_groups` validates that shared part before any image is loaded.
`require_groups` checks a manifest's top-level shape first; `unknown_keys`,
`not_strings`, `is_int` and `is_number` are the field checks both use;
`check_depth` checks the rest of a depth manifest. `resolve_path` owns where a
manifest's relative paths point: the folder of the manifest file (`resolve_paths`
applies it to every path field when a driver loads a file). The checks take a dict
and use its paths as given, so a manifest built in Python, with no file, keeps
relative paths relative to the working directory.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .depth import parse_window, window_label
from .stats import REDUCERS


def is_int(v: object) -> bool:
    """An integer, not a bool (JSON true/false are ints to Python)."""
    return isinstance(v, int) and not isinstance(v, bool)


def is_number(v: object) -> bool:
    """An int or float, not a bool."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def not_strings(m: dict, keys: Iterable[str]) -> list[str]:
    """One problem per key of ``keys`` present in ``m`` whose value is not a string
    (a number given as ``output_dir`` would otherwise fail only when the run writes)."""
    return [
        f"{k} must be a string, got {m[k]!r}" for k in keys if k in m and not isinstance(m[k], str)
    ]


def unknown_keys(where: str, d: dict, allowed: Iterable[str]) -> list[str]:
    """One problem per key of ``d`` not in ``allowed`` (a misspelt field would otherwise
    be ignored and its default used), naming ``where`` and the allowed fields."""
    allowed = tuple(allowed)
    return [
        f"{where}: unknown field {k!r} (allowed: {', '.join(allowed)})"
        for k in d
        if k not in allowed
    ]


def _repeated(values: list) -> list:
    """The values that occur more than once, each once, in first-seen order. A missing
    value (None or "") is reported as missing elsewhere, so it is skipped here; a
    numeric 0 is a value like any other."""
    present = [v for v in values if v is not None and v != ""]
    return [v for i, v in enumerate(present) if v in present[i + 1 :] and v not in present[:i]]


def resolve_path(manifest_path: str | Path | None, value: str) -> str:
    """Where a manifest's path ``value`` points: unchanged when absolute or when there
    is no manifest file (``manifest_path`` None: a dict manifest's paths are used as
    given, relative to the working directory), else joined to the manifest's folder.

    Raises ValueError when the manifest-relative path does not exist but ``value``
    does relative to the working directory (the rule before 0.9), naming both, so
    a manifest written for the old rule fails instead of reading another file."""
    if manifest_path is None or Path(value).is_absolute():
        return value
    resolved = Path(manifest_path).parent / value
    here = Path(value)
    if not resolved.exists() and here.exists() and resolved.resolve() != here.resolve():
        raise ValueError(
            f"path {value!r} is resolved against the manifest's folder, where "
            f"{resolved} does not exist; it exists relative to the current directory "
            f"as {here.resolve()}. Give it relative to {resolved.parent} or as an "
            "absolute path."
        )
    return str(resolved)


def resolve_paths(m: dict, manifest_path: str | Path) -> dict:
    """``m`` with its path fields (``output_dir``, each group's ``background`` and
    each stack's ``path``) resolved by `resolve_path`; a field of the wrong type is
    kept as given for the checks to report. Raises ValueError listing every path
    that exists only relative to the working directory."""
    problems: list[str] = []

    def one(v):
        if not isinstance(v, str):
            return v
        try:
            return resolve_path(manifest_path, v)
        except ValueError as e:
            problems.append(str(e))
            return v

    groups = [_resolve_group(grp, one) if isinstance(grp, dict) else grp for grp in m["groups"]]
    out = {**m, "groups": groups}
    if "output_dir" in m:
        out["output_dir"] = one(m["output_dir"])
    if problems:
        raise ValueError("manifest paths:\n  - " + "\n  - ".join(problems))
    return out


def _resolve_group(grp: dict, one) -> dict:
    """A group with its background(s) and stack paths passed through ``one``."""
    out = dict(grp)
    bg = grp.get("background")
    if bg is not None:
        out["background"] = [one(b) for b in bg] if isinstance(bg, list) else one(bg)
    stacks = grp.get("stacks")
    if isinstance(stacks, list):
        out["stacks"] = [
            {**st, "path": one(st["path"])} if isinstance(st, dict) and "path" in st else st
            for st in stacks
        ]
    return out


def require_groups(m: object) -> None:
    """Raise ValueError unless ``m`` (a parsed manifest) is a JSON object with a
    non-empty ``groups`` list; the batch loaders call it before reading any field."""
    if not isinstance(m, dict):
        raise ValueError(
            f"a manifest must be a JSON object with a 'groups' list, got a {type(m).__name__}"
        )
    groups = m.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(
            "manifest needs a non-empty 'groups' list (at least one group with stacks), "
            f"got {groups!r}"
        )


def check_groups(
    groups: list[dict],
    roles: tuple[str, ...] | None = None,
    group_keys: Iterable[str] | None = None,
    stack_keys: Iterable[str] | None = None,
) -> list[str]:
    """Problems with a manifest's groups: each an object, names (non-empty strings,
    unique), stacks (a non-empty list of objects; labels unique within the group,
    paths present), background paths, and (when given) each stack's role and
    unknown group / stack fields. Empty when all is well. Paths are checked as given
    (relative ones against the working directory); the batch loaders resolve a
    manifest file's paths first (`resolve_paths`)."""
    objects = [grp for grp in groups if isinstance(grp, dict)]
    problems = [
        f"group name {name!r} is used by more than one group; names must be unique"
        for name in _repeated([grp.get("name") for grp in objects])
    ]
    for gi, grp in enumerate(groups):
        if not isinstance(grp, dict):
            problems.append(f"group {gi} must be an object with 'name' and 'stacks', got {grp!r}")
            continue
        problems += _group_problems(gi, grp, roles, group_keys, stack_keys)
    return problems


def _group_problems(gi: int, grp: dict, roles, group_keys, stack_keys) -> list[str]:
    """Problems with group number ``gi``, an object."""
    name = grp.get("name")
    problems = []
    if not name:
        problems.append(f"group {gi} has no 'name'")
    elif not isinstance(name, str):
        problems.append(f"group {gi}: name must be a non-empty string, got {name!r}")
    if group_keys is not None:
        problems += unknown_keys(f"group {name!r}", grp, group_keys)
    stacks = grp.get("stacks")
    if not stacks:
        problems.append(f"group {name or gi!r} has no stacks")
    elif not isinstance(stacks, list):
        problems.append(f"group {name!r}: stacks must be a list of stack objects, got {stacks!r}")
        stacks = []
    problems += _background_problems(name, grp.get("background"))
    objects = [st for st in stacks or [] if isinstance(st, dict)]
    problems += [
        f"group {name!r}: stack label {label!r} is used by more than one stack; "
        "labels must be unique within a group"
        for label in _repeated([st.get("label") for st in objects])
    ]
    for st in stacks or []:
        problems += _stack_problems(name, st, roles, stack_keys)
    return problems


def _background_problems(group: str | None, bg) -> list[str]:
    """Problems with a group's ``background``: absent, a path, or a list of paths."""
    paths = [bg] if isinstance(bg, str) else bg if isinstance(bg, list) else None
    if bg is not None and (paths is None or not all(isinstance(bp, str) for bp in paths)):
        return [f"group {group!r}: background must be a path or a list of paths, got {bg!r}"]
    return [
        f"group {group!r}: background {bp} does not exist"
        for bp in paths or []
        if not Path(bp).exists()
    ]


def _stack_problems(group: str | None, st, roles, stack_keys) -> list[str]:
    """Problems with one stack entry of group ``group``."""
    if not isinstance(st, dict):
        return [
            f"group {group!r}: every stack must be an object with 'path' and 'label', got {st!r}"
        ]
    if "path" not in st or "label" not in st:
        return [f"group {group!r}: every stack needs 'path' and 'label': {st}"]
    if not isinstance(st["path"], str):
        return [f"group {group!r}: stack path must be a string, got {st['path']!r}"]
    if not (isinstance(st["label"], str) or is_number(st["label"])):
        return [f"group {group!r}: stack label must be a string or a number, got {st['label']!r}"]
    problems = []
    if stack_keys is not None:
        problems += unknown_keys(f"group {group!r}: stack {st['label']!r}", st, stack_keys)
    if not Path(st["path"]).exists():
        problems.append(f"group {group!r}: stack {st['path']} does not exist")
    if roles is not None and st.get("role", "primary") not in roles:
        problems.append(
            f"group {group!r}: stack {st['label']!r} role must be one of "
            f"{roles}, got {st['role']!r}"
        )
    return problems


# Depth manifests (`depth_batch`): the stack roles, and every field a manifest, group
# and stack may have (anything else is refused).
_DEPTH_ROLES = ("primary", "aux")
_DEPTH_FIELDS = (
    "title",
    "signal_label",
    "channel",
    "reducer",
    "n_surface",
    "fit_offset",
    "fit_r2_threshold",
    "parametric",
    "auc_windows_um",
    "output_dir",
    "groups",
)
_DEPTH_GROUP_FIELDS = ("name", "color", "background", "stacks")
_DEPTH_STACK_FIELDS = ("label", "path", "role")


def check_depth(m: dict) -> list[str]:
    """Problems with a depth manifest (`depth_batch`; defaults already filled), apart
    from group colours, which `depth_batch` checks with matplotlib."""
    # the removed singular window key is named with its replacement, not as unknown
    known = {k: v for k, v in m.items() if k != "auc_window_um"}
    problems = unknown_keys("manifest", known, _DEPTH_FIELDS)
    if "auc_window_um" in m:
        problems.append("'auc_window_um' was removed; use 'auc_windows_um': [[z0, z1]]")
    if not is_int(m["channel"]) or m["channel"] < 0:
        problems.append(f"channel must be a non-negative integer, got {m['channel']!r}")
    reducers = tuple(REDUCERS)
    if m["reducer"] not in reducers:
        problems.append(f"reducer must be one of {reducers}, got {m['reducer']!r}")
    if not is_int(m["n_surface"]) or m["n_surface"] < 1:
        problems.append(f"n_surface must be a positive integer, got {m['n_surface']!r}")
    problems += [
        f"{k} must be true or false, got {m[k]!r}"
        for k in ("fit_offset", "parametric")
        if not isinstance(m[k], bool)
    ]
    problems += not_strings(m, ("title", "signal_label", "output_dir"))
    r2 = m["fit_r2_threshold"]
    if not (is_number(r2) and 0 <= r2 <= 1):
        problems.append(f"fit_r2_threshold must be a number in [0, 1], got {r2!r}")
    problems += _window_problems(m["auc_windows_um"])
    return problems + check_groups(
        m["groups"],
        roles=_DEPTH_ROLES,
        group_keys=_DEPTH_GROUP_FIELDS,
        stack_keys=_DEPTH_STACK_FIELDS,
    )


def _window_problems(windows) -> list[str]:
    """Problems with ``auc_windows_um``: a list of distinct [z0, z1] pairs (µm) and/or
    "full" (windows with one label, e.g. [0, 100] and [0, 100.0], are the same)."""
    form = "a list of [z0, z1] pairs (µm) and/or 'full', e.g. [[0, 100], 'full']"
    if not isinstance(windows, list) or not windows:
        return [f"auc_windows_um must be {form}; got {windows!r}"]
    if all(is_number(w) for w in windows):
        return [f"auc_windows_um must be {form}; a single window is [{windows!r}], not {windows!r}"]
    problems = []
    for w in windows:
        if isinstance(w, str) and w.lower() == "full":
            continue
        if not (isinstance(w, list) and len(w) == 2 and all(is_number(v) for v in w)):
            problems.append(
                f"auc_windows_um entry {w!r} must be 'full' or a [z0, z1] pair of "
                f"numbers; auc_windows_um is {form}"
            )
        elif not 0 <= w[0] < w[1]:
            problems.append(f"AUC window {w!r} must be [z0, z1] with 0 <= z0 < z1 (µm)")
    if problems:
        return problems
    return [
        f"AUC window {label} is given more than once; each window must be distinct"
        for label in _repeated([window_label(parse_window(w)) for w in windows])
    ]
