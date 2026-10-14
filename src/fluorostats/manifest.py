"""Shared checks for the JSON manifests the batch drivers read.

`depth_batch` and `viability_batch` both describe their input as groups of
stacks; `check_groups` validates that shared part before any image is loaded.
"""

from __future__ import annotations

from pathlib import Path


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
