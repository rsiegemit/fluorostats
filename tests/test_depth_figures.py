"""Tests for fluorostats.depth_figures (depth-batch figures)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fluorostats.depth import PenetrationFit, parse_window, window_label  # noqa: E402
from fluorostats.depth_figures import (
    group_color,
    plot_auc,
    plot_depth_curves,  # noqa: E402
    plot_retention_multiwindow,
)
from fluorostats.pubfig import material_color  # noqa: E402


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
        auc[window_label(w)] = {
            "normalized": norm * width,
            "absolute": 2 * norm * width,
            "width_um": width,
            "covered_um": width,
        }
    return {
        "group": group,
        "stack": f"{group}{i}",
        "role": role,
        "result": {"depth_um": np.linspace(0, 10, 8), "auc": auc},
    }


def _saved(out_base):
    return [(out_base.parent / f"{out_base.name}{ext}").exists() for ext in (".png", ".pdf")]


def test_group_color_uses_override_else_palette():
    assert group_color("GelMA", "#123456") == "#123456"
    assert group_color("GelMA", None) == material_color("GelMA")


def test_plot_depth_curves_with_none_agg(tmp_path, last_axes):
    """A group whose aggregate is None (no primary stacks) still draws its stacks."""
    res = {
        "depth_um": np.linspace(0, 10, 8),
        "bg_subtracted": np.ones(8),
        "normalized": np.ones(8),
        "raw_mean": np.ones(8),
    }
    rows = [{"group": "A", "stack": "s0", "role": "aux", "result": res}]
    out_base = tmp_path / "fig_test"
    plot_depth_curves(
        rows, {"A": None}, {"A": "#3F6AB3"}, "bg_subtracted", "intensity", "title", out_base
    )
    assert _saved(out_base) == [True, True]
    lines = last_axes().get_lines()
    assert len(lines) == 1 and lines[0].get_linestyle() == "--"  # one faint aux stack


def test_plot_auc_group_no_primary_vals(tmp_path, last_axes):
    """A group whose stacks are all aux draws no bar but keeps its tick."""
    window = parse_window([0, 10])
    out_base = tmp_path / "auc_test"
    plot_auc(
        [_auc_row("A", 0, [window], 0.5, role="aux")],
        {"A": "#3F6AB3"},
        window,
        out_base,
        use_normalized=True,
    )
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
    assert len(ax.containers) == 2  # the bar and its error bar


def test_plot_retention_multiwindow_bars_are_mean_retained_fraction(tmp_path, last_axes):
    windows = [parse_window([0, 10]), parse_window([0, 20]), parse_window("full")]
    rows = (
        [_auc_row("A", i, windows, v) for i, v in enumerate((0.4, 0.6))]
        + [_auc_row("B", 0, windows, 0.9)]
        + [_auc_row("B", 1, windows, 0.1, role="aux")]
    )  # aux is excluded
    out_base = tmp_path / "retention"
    plot_retention_multiwindow(rows, {"A": "#3F6AB3", "B": "#B33F3F"}, windows, out_base, title="t")
    ax = last_axes()
    assert [round(p.get_height(), 6) for p in ax.patches] == [0.5, 0.5, 0.5, 0.9, 0.9, 0.9]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0–10 µm", "0–20 µm", "full depth"]
    assert ax.get_title() == "t"
    assert _saved(out_base) == [True, True]


def test_plot_auc_drops_a_nan_auc_instead_of_drawing_a_nan_bar(tmp_path, last_axes):
    # Regression: a stack whose window lies past its depth has AUC NaN, and the inline
    # np.mean made the whole group's bar and error bar NaN (the printed summary, which
    # uses stats.describe, said n=2, mean 5).
    window = parse_window([0, 10])
    rows = [_auc_row("A", i, [window], v) for i, v in enumerate((0.4, 0.6, np.nan))]
    plot_auc(rows, {"A": "#3F6AB3"}, window, tmp_path / "auc_nan", use_normalized=True)
    ax = last_axes()
    assert ax.patches[0].get_height() == pytest.approx(5.0)
    (seg,) = ax.containers[1].lines[2][0].get_segments()  # the SEM whisker
    np.testing.assert_allclose(seg[:, 1], [4.0, 6.0])  # SEM of (4, 6) = 1


def test_retention_draws_no_whisker_for_a_single_stack(tmp_path, last_axes):
    # Regression: a group with one primary stack got an SEM of 0.0 (a drawn zero-length
    # whisker with caps), claiming a precision nobody measured; it is undefined (NaN).
    windows = [parse_window([0, 10]), parse_window([0, 20])]
    rows = [_auc_row("A", i, windows, v) for i, v in enumerate((0.4, 0.6))]
    rows += [_auc_row("B", 0, windows, 0.9)]
    plot_retention_multiwindow(
        rows, {"A": "#3F6AB3", "B": "#B33F3F"}, windows, tmp_path / "ret", title="t"
    )
    ax = last_axes()
    a_err, b_err = (c for c in ax.containers if hasattr(c, "has_yerr"))
    assert np.isfinite(a_err.lines[2][0].get_segments()).all()
    assert np.isnan(b_err.lines[2][0].get_segments()).all()


def test_fractional_window_names_keep_their_suffix(tmp_path):
    """'0.5_10um' and '0.25_10um' used to both save as 'fig_auc_0.png' and overwrite."""
    windows = [parse_window([0.5, 10]), parse_window([0.25, 10])]
    rows = [_auc_row("A", 0, windows, 0.5)]
    for w in windows:
        plot_auc(rows, {"A": "#3F6AB3"}, w, tmp_path / f"fig_auc_{window_label(w)}")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "fig_auc_0.25_10um.pdf",
        "fig_auc_0.25_10um.png",
        "fig_auc_0.5_10um.pdf",
        "fig_auc_0.5_10um.png",
    ]


def test_import_keeps_the_callers_backend():
    """(Re)loading the module must not switch the user's matplotlib backend."""
    import subprocess
    import sys

    code = (
        "import importlib, matplotlib, fluorostats.depth_figures as m; matplotlib.use('svg'); "
        "importlib.reload(m); print(matplotlib.get_backend())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "svg"


def _overlay_case(floor, fit_offset):
    """One group whose curve is exactly 100·exp(-z/10) + floor, with matching fits."""
    z = np.arange(0.0, 41.0, 2.0)
    curve = 100 * np.exp(-z / 10) + floor
    fit = PenetrationFit(
        lambda_um=10.0,
        I0=100.0,
        offset=floor if fit_offset else None,
        r_squared=1.0,
        rmse=0.0,
        n_points=z.size,
        fit_ok=True,
    )
    rows = [
        {
            "group": "A",
            "stack": f"s{i}",
            "role": "primary",
            "result": {"depth_um": z, "bg_subtracted": curve, "fit": fit},
        }
        for i in range(2)
    ]
    agg = {
        "depth_um": z,
        "sub_mean": curve,
        "sub_sem": np.zeros_like(z),
        "n": 2,
        "lambda_mean_um": 10.0,
        "lambda_sem_um": 0.0,
        "n_lambda_ok": 2,
        "r_squared_mean": 1.0,
    }
    return rows, {"A": agg}, curve


@pytest.mark.parametrize(
    "floor, fit_offset, n_surface", [(0.0, False, 1), (50.0, True, 1), (50.0, True, 3)]
)
def test_lambda_overlay_follows_n_surface_and_fit_offset(
    tmp_path, last_axes, floor, fit_offset, n_surface
):
    """The λ overlay is anchored on the manifest's n_surface slices and, when the floor
    was fitted, draws floor + (surface - floor)·exp(-z/λ)."""
    rows, aggs, curve = _overlay_case(floor, fit_offset)
    plot_depth_curves(
        rows,
        aggs,
        {"A": "#3F6AB3"},
        "bg_subtracted",
        "y",
        "t",
        tmp_path / "curves",
        n_surface=n_surface,
        fit_offset=fit_offset,
    )
    overlay = [ln for ln in last_axes().get_lines() if ln.get_zorder() == 4]
    assert len(overlay) == 1
    z = overlay[0].get_xdata()
    surface = curve[:n_surface].mean()
    expected = floor + (surface - floor) * np.exp(-z / 10.0)
    np.testing.assert_allclose(overlay[0].get_ydata(), expected)
    if n_surface == 1:
        np.testing.assert_allclose(overlay[0].get_ydata(), curve)  # exact model, exact anchor


@pytest.mark.parametrize("sem, expected", [(1.6, "λ=10±2 µm"), (float("nan"), "λ=10 µm")])
def test_lambda_label_omits_the_sem_when_it_is_undefined(sem, expected):
    from fluorostats.depth_figures import _plot_lambda_model

    z = np.arange(0.0, 41.0, 2.0)
    agg = {
        "depth_um": z,
        "lambda_mean_um": 10.0,
        "lambda_sem_um": sem,
        "n_lambda_ok": 1,
        "n": 1,
        "r_squared_mean": 0.9,
    }
    fig, ax = plt.subplots()
    assert expected in _plot_lambda_model(ax, agg, 100 * np.exp(-z / 10), "#000", 1, 0.0)


def test_retention_divides_by_the_covered_width_not_the_window(tmp_path, last_axes):
    # Regression (M2): a stack covering 50 µm of a 0-100 µm window and keeping all its
    # signal has a normalised AUC of 50; it was drawn as 0.5 retained, not 1.
    windows = [parse_window([0, 100]), parse_window([0, 10])]
    row = _auc_row("A", 0, windows, 1.0)
    row["result"]["auc"]["0_100um"].update(normalized=50.0, covered_um=50.0)
    plot_retention_multiwindow([row], {"A": "#3F6AB3"}, windows, tmp_path / "r", title="t")
    assert [p.get_height() for p in last_axes().patches] == [1.0, 1.0]
