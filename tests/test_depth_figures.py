"""Tests for fluorostats.depth_figures (depth-batch figures)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fluorostats.depth import parse_window, window_label  # noqa: E402
from fluorostats.depth_figures import (group_color, plot_auc, plot_depth_curves,  # noqa: E402
                                       plot_retention_multiwindow)
from fluorostats.style import material_color  # noqa: E402


@pytest.fixture
def last_axes(monkeypatch):
    """Keep the figure a plot function saves open, and return its axes afterwards."""
    monkeypatch.setattr(plt, "close", lambda fig=None: None)
    yield lambda: plt.gcf().axes[0]
    monkeypatch.undo()
    plt.close("all")


def _auc_row(group, i, windows, norm, role="primary"):
    """A depth_batch row whose normalised AUC is ``norm * width`` for every window."""
    auc = {}
    for w in windows:
        width = 10.0 if w == ("full",) else w[1] - w[0]
        auc[window_label(w)] = {"normalized": norm * width, "absolute": 2 * norm * width,
                                "width_um": width}
    return {"group": group, "stack": f"{group}{i}", "role": role,
            "result": {"depth_um": np.linspace(0, 10, 8), "auc": auc}}


def _saved(out_base):
    return [(out_base.parent / f"{out_base.name}{ext}").exists() for ext in (".png", ".pdf")]


def test_group_color_uses_override_else_palette():
    assert group_color("GelMA", "#123456") == "#123456"
    assert group_color("GelMA", None) == material_color("GelMA")


def test_plot_depth_curves_with_none_agg(tmp_path, last_axes):
    """A group whose aggregate is None (no primary stacks) still draws its stacks."""
    res = {"depth_um": np.linspace(0, 10, 8), "bg_subtracted": np.ones(8),
           "normalized": np.ones(8), "raw_mean": np.ones(8)}
    rows = [{"group": "A", "stack": "s0", "role": "aux", "result": res}]
    out_base = tmp_path / "fig_test"
    plot_depth_curves(rows, {"A": None}, {"A": "#3F6AB3"}, "bg_subtracted",
                      "intensity", "title", out_base)
    assert _saved(out_base) == [True, True]
    lines = last_axes().get_lines()
    assert len(lines) == 1 and lines[0].get_linestyle() == "--"   # one faint aux stack


def test_plot_auc_group_no_primary_vals(tmp_path, last_axes):
    """A group whose stacks are all aux draws no bar but keeps its tick."""
    window = parse_window([0, 10])
    out_base = tmp_path / "auc_test"
    plot_auc([_auc_row("A", 0, [window], 0.5, role="aux")], {"A": "#3F6AB3"}, window,
             out_base, use_normalized=True)
    ax = last_axes()
    assert _saved(out_base) == [True, True]
    assert len(ax.patches) == 0
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A"]


def test_plot_auc_sem_errorbar(tmp_path, last_axes):
    """Three primary stacks: the bar is their mean and an SEM error bar is drawn."""
    window = parse_window([0, 10])
    rows = [_auc_row("A", i, [window], v) for i, v in enumerate((1.0, 1.2, 1.1))]
    out_base = tmp_path / "auc_sem"
    plot_auc(rows, {"A": "#3F6AB3"}, window, out_base, use_normalized=True)
    ax = last_axes()
    assert _saved(out_base) == [True, True]
    assert ax.patches[0].get_height() == pytest.approx(11.0)
    assert len(ax.containers) == 2                        # the bar and its error bar


def test_plot_retention_multiwindow_bars_are_mean_retained_fraction(tmp_path, last_axes):
    windows = [parse_window([0, 10]), parse_window([0, 20]), parse_window("full")]
    rows = ([_auc_row("A", i, windows, v) for i, v in enumerate((0.4, 0.6))]
            + [_auc_row("B", 0, windows, 0.9)]
            + [_auc_row("B", 1, windows, 0.1, role="aux")])   # aux is excluded
    out_base = tmp_path / "retention"
    plot_retention_multiwindow(rows, {"A": "#3F6AB3", "B": "#B33F3F"}, windows, out_base,
                               title="t")
    ax = last_axes()
    assert [round(p.get_height(), 6) for p in ax.patches] == [0.5, 0.5, 0.5, 0.9, 0.9, 0.9]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0–10 µm", "0–20 µm", "full depth"]
    assert ax.get_title() == "t"
    assert _saved(out_base) == [True, True]
