"""Shared checks for the JSON manifests the batch drivers read.

`depth_batch` and `viability_batch` both describe their input as groups of
stacks; `check_groups` validates that shared part before any image is loaded.
`unknown_keys`, `is_int` and `is_number` are the field checks both use;
`check_depth` checks the rest of a depth manifest.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .stats import REDUCERS


def is_int(v) -> bool:
    """An integer, not a bool (JSON true/false are ints to Python)."""
    return isinstance(v, int) and not isinstance(v, bool)


def is_number(v) -> bool:
    """An int or float, not a bool."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def unknown_keys(where: str, d: dict, allowed: Iterable[str]) -> list[str]:
    """One problem per key of ``d`` not in ``allowed`` (a misspelt field would otherwise
    be ignored and its default used), naming ``where`` and the allowed fields."""
    allowed = tuple(allowed)
    return [f"{where}: unknown field {k!r} (allowed: {', '.join(allowed)})"
            for k in d if k not in allowed]


def _repeated(values: list) -> list:
    """The values that occur more than once, each once, in first-seen order."""
    return [v for i, v in enumerate(values) if v and v in values[i + 1:] and v not in values[:i]]


def check_groups(groups: list[dict], roles: tuple[str, ...] | None = None,
                 group_keys: Iterable[str] | None = None,
                 stack_keys: Iterable[str] | None = None) -> list[str]:
    """Problems with a manifest's groups: names (present and unique), stacks (at least
    one; labels unique within the group, paths present), background paths, and (when
    given) each stack's role and unknown group / stack fields. Empty when all is well."""
    problems = [f"group name {name!r} is used by more than one group; names must be unique"
                for name in _repeated([grp.get("name") for grp in groups])]
    for gi, grp in enumerate(groups):
        name = grp.get("name")
        if not name:
            problems.append(f"group {gi} has no 'name'")
        if group_keys is not None:
            problems += unknown_keys(f"group {name!r}", grp, group_keys)
        if not grp.get("stacks"):
            problems.append(f"group {name or gi!r} has no stacks")
        bg = grp.get("background")
        for bp in ([bg] if isinstance(bg, str) else list(bg or [])):
            if not Path(bp).exists():
                problems.append(f"group {name!r}: background {bp} does not exist")
        problems += [f"group {name!r}: stack label {label!r} is used by more than one stack; "
                     "labels must be unique within a group"
                     for label in _repeated([st.get("label") for st in grp.get("stacks", [])])]
        for st in grp.get("stacks", []):
            problems += _stack_problems(name, st, roles, stack_keys)
    return problems


def _stack_problems(group: str, st: dict, roles, stack_keys) -> list[str]:
    """Problems with one stack entry of group ``group``."""
    if "path" not in st or "label" not in st:
        return [f"group {group!r}: every stack needs 'path' and 'label': {st}"]
    problems = []
    if stack_keys is not None:
        problems += unknown_keys(f"group {group!r}: stack {st['label']!r}", st, stack_keys)
    if not Path(st["path"]).exists():
        problems.append(f"group {group!r}: stack {st['path']} does not exist")
    if roles is not None and st.get("role", "primary") not in roles:
        problems.append(f"group {group!r}: stack {st['label']!r} role must be one of "
                        f"{roles}, got {st['role']!r}")
    return problems


# Depth manifests (`depth_batch`): the stack roles, and every field a manifest, group
# and stack may have (anything else is refused).
_DEPTH_ROLES = ("primary", "aux")
_DEPTH_FIELDS = ("title", "signal_label", "channel", "reducer", "n_surface", "fit_offset",
                 "fit_r2_threshold", "parametric", "auc_windows_um", "auc_window_um",
                 "output_dir", "groups")
_DEPTH_GROUP_FIELDS = ("name", "color", "background", "stacks")
_DEPTH_STACK_FIELDS = ("label", "path", "role")


def check_depth(m: dict) -> list[str]:
    """Problems with a depth manifest (`depth_batch`; defaults already filled), apart
    from group colours, which `depth_batch` checks with matplotlib."""
    problems = unknown_keys("manifest", m, _DEPTH_FIELDS)
    if not is_int(m["channel"]) or m["channel"] < 0:
        problems.append(f"channel must be a non-negative integer, got {m['channel']!r}")
    reducers = tuple(REDUCERS)
    if m["reducer"] not in reducers:
        problems.append(f"reducer must be one of {reducers}, got {m['reducer']!r}")
    if not is_int(m["n_surface"]) or m["n_surface"] < 1:
        problems.append(f"n_surface must be a positive integer, got {m['n_surface']!r}")
    problems += [f"{k} must be true or false, got {m[k]!r}" for k in ("fit_offset", "parametric")
                 if not isinstance(m[k], bool)]
    r2 = m["fit_r2_threshold"]
    if not (is_number(r2) and 0 <= r2 <= 1):
        problems.append(f"fit_r2_threshold must be a number in [0, 1], got {r2!r}")
    problems += _window_problems(m["auc_windows_um"])
    return problems + check_groups(m["groups"], roles=_DEPTH_ROLES,
                                   group_keys=_DEPTH_GROUP_FIELDS, stack_keys=_DEPTH_STACK_FIELDS)


def _window_problems(windows) -> list[str]:
    """Problems with ``auc_windows_um``: a list of [z0, z1] pairs (µm) and/or "full"."""
    form = "a list of [z0, z1] pairs (µm) and/or 'full', e.g. [[0, 100], 'full']"
    if not isinstance(windows, list) or not windows:
        return [f"auc_windows_um must be {form}; got {windows!r}"]
    if all(is_number(w) for w in windows):
        return [f"auc_windows_um must be {form}; a single window is [{windows!r}], "
                f"not {windows!r}"]
    problems = []
    for w in windows:
        if isinstance(w, str) and w.lower() == "full":
            continue
        if not (isinstance(w, list) and len(w) == 2 and all(is_number(v) for v in w)):
            problems.append(f"auc_windows_um entry {w!r} must be 'full' or a [z0, z1] pair of "
                            f"numbers; auc_windows_um is {form}")
        elif not 0 <= w[0] < w[1]:
            problems.append(f"AUC window {w!r} must be [z0, z1] with 0 <= z0 < z1 (µm)")
    return problems
