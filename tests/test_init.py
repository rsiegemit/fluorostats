"""Tests for the fluorostats package namespace (__init__)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import fluorostats

_SRC = str(Path(fluorostats.__file__).resolve().parents[1])


@pytest.mark.parametrize("stmt", ["import fluorostats", "import fluorostats.io"])
def test_import_does_not_load_pyplot(stmt):
    # Regression: the package imported its figure modules eagerly, which called
    # matplotlib.use("Agg") and loaded pyplot, so `import fluorostats.io` switched an
    # interactive session to the non-interactive Agg backend.
    code = f"import sys; {stmt}; print('matplotlib.pyplot' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": _SRC}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "False"


def test_figure_and_driver_modules_load_on_first_use():
    from fluorostats import pubfig
    from fluorostats.plots import summary_panel

    assert fluorostats.plots.summary_panel is summary_panel
    assert fluorostats.pubfig is pubfig
    for name in (
        "render3d",
        "projections",
        "depth_figures",
        "depth_contrasts",
        "depth_batch",
        "viability_batch",
        "report",
    ):
        assert getattr(fluorostats, name).__name__ == f"fluorostats.{name}"
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        fluorostats.nope


def test_package_version_matches_pyproject():
    """One version: the package and pyproject.toml agree."""
    import pathlib
    import re

    import fluorostats

    pyproject = pathlib.Path(__file__).parents[1] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip(
            "needs a source tree with pyproject.toml (tests run against an installed wheel)"
        )
    declared = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    assert declared and declared.group(1) == fluorostats.__version__
