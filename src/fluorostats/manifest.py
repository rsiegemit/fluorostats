"""Shared checks for the JSON manifests the batch drivers read.

`depth_batch` and `viability_batch` both describe their input as groups of
stacks; `check_groups` validates that shared part before any image is loaded.
`is_int` and `is_number` are the field checks both use; `check_depth` checks
the rest of a depth manifest.
"""

from __future__ import annotations

from pathlib import Path

from .stats import REDUCERS


def is_int(v) -> bool:
    """An integer, not a bool (JSON true/false are ints to Python)."""
    return isinstance(v, int) and not isinstance(v, bool)


def is_number(v) -> bool:
    """An int or float, not a bool."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_groups(groups: list[dict], roles: tuple[str, ...] | None = None) -> list[str]:
    """Problems with a manifest's groups: names (present and unique), stacks (at least
    one; labels and paths), background paths, and (when ``roles`` is given) each
    stack's role. Empty when all is well."""
    problems = []
    names = [grp.get("name") for grp in groups]
    repeated = [n for i, n in enumerate(names) if n and n in names[i + 1:] and n not in names[:i]]
    for name in repeated:
        problems.append(f"group name {name!r} is used by more than one group; names must be unique")
    for gi, grp in enumerate(groups):
        name = grp.get("name")
        if not name:
            problems.append(f"group {gi} has no 'name'")
        if not grp.get("stacks"):
            problems.append(f"group {name or gi!r} has no stacks")
        bg = grp.get("background")
        for bp in ([bg] if isinstance(bg, str) else list(bg or [])):
            if not Path(bp).exists():
                problems.append(f"group {name!r}: background {bp} does not exist")
        for st in grp.get("stacks", []):
            if "path" not in st or "label" not in st:
                problems.append(f"group {name!r}: every stack needs 'path' and 'label': {st}")
                continue
            if not Path(st["path"]).exists():
                problems.append(f"group {name!r}: stack {st['path']} does not exist")
            if roles is not None and st.get("role", "primary") not in roles:
                problems.append(f"group {name!r}: stack {st['label']!r} role must be one of "
                                f"{roles}, got {st['role']!r}")
    return problems


# Depth manifests (`depth_batch`): the stack roles.
_DEPTH_ROLES = ("primary", "aux")


def check_depth(m: dict) -> list[str]:
    """Problems with a depth manifest (`depth_batch`; defaults already filled), apart
    from group colours, which `depth_batch` checks with matplotlib."""
    problems = []
    if not is_int(m["channel"]) or m["channel"] < 0:
        problems.append(f"channel must be a non-negative integer, got {m['channel']!r}")
    reducers = tuple(REDUCERS)
    if m["reducer"] not in reducers:
        problems.append(f"reducer must be one of {reducers}, got {m['reducer']!r}")
    if not is_int(m["n_surface"]) or m["n_surface"] < 1:
        problems.append(f"n_surface must be a positive integer, got {m['n_surface']!r}")
    if not isinstance(m["parametric"], bool):
        problems.append(f"parametric must be true or false, got {m['parametric']!r}")
    problems += _window_problems(m["auc_windows_um"])
    return problems + check_groups(m["groups"], roles=_DEPTH_ROLES)


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
