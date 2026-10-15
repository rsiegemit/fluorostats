import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fluorostats import agreement, pubfig  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_apply_style_sets_compact_rcparams():
    pubfig.apply_style()
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["axes.labelsize"] == pubfig.FS["label"]
    assert plt.rcParams["axes.spines.top"] is False
    pubfig.apply_style(font_scale=2.0)
    assert plt.rcParams["axes.labelsize"] == 2 * pubfig.FS["label"]
    plt.rcdefaults()


def test_unknown_theme_is_refused():
    with pytest.raises(ValueError, match="theme must be one of \\('compact', 'house'\\)"):
        pubfig.apply_style("poster")


def test_save_figure_writes_each_path_at_its_dpi_and_closes(tmp_path):
    from PIL import Image

    fig = plt.figure(figsize=(1, 1))
    out = tmp_path / "new" / "f"
    pubfig.save_figure(
        fig, {out.with_suffix(".png"): 50, tmp_path / "f2.png": 100, out.with_suffix(".pdf"): None}
    )
    assert Image.open(out.with_suffix(".png")).size == (50, 50)
    assert Image.open(tmp_path / "f2.png").size == (100, 100)
    assert out.with_suffix(".pdf").exists()
    assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize("xanchor", [None, 0.03])
def test_panel_draws_letter_and_title(xanchor):
    fig, ax = plt.subplots()
    pubfig.panel(ax, "a", "recovery", xanchor=xanchor)
    texts = [t.get_text() for t in ax.texts]
    assert texts == ["a", "recovery"]
    pubfig.panel(ax, "b", xanchor=xanchor)  # no title -> letter only
    assert [t.get_text() for t in ax.texts][-1] == "b"


def test_imnorm_maps_percentiles_to_unit_range():
    a = np.arange(1000.0)
    n = pubfig.imnorm(a)
    assert n.min() == 0.0 and n.max() == 1.0


def test_image_axes_strips_ticks_and_frames():
    fig, ax = plt.subplots()
    pubfig.image_axes(ax)
    assert list(ax.get_xticks()) == [] and list(ax.get_yticks()) == []
    assert all(sp.get_visible() for sp in ax.spines.values())


@pytest.mark.parametrize("width", [1, 3])
def test_outline_and_fill_overlay_rgba(width):
    fig, ax = plt.subplots()
    lab = np.zeros((20, 20), int)
    lab[5:12, 5:12] = 1
    pubfig.outline(ax, lab, (1, 0, 0), width=width)
    pubfig.fill_mask(ax, lab, (0, 1, 0))
    assert len(ax.images) == 2
    assert ax.images[0].get_array().shape == (20, 20, 4)


def test_composite2ch_is_rgb_in_unit_range():
    rng = np.random.default_rng(0)
    rgb = pubfig.composite2ch(rng.random((8, 8)), rng.random((8, 8)))
    assert rgb.shape == (8, 8, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


@pytest.mark.parametrize(
    "loc,center", [("lower right", False), ("upper left", False), ("lower left", True)]
)
def test_scalebar_label_stays_on_bar(loc, center):
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((100, 200)))
    pubfig.scalebar(ax, 50, "20 µm", loc=loc, center_label=center)
    assert ax.texts[0].get_text() == "20 µm"
    assert ax.texts[0].get_ha() == ("center" if center else ("right" if "right" in loc else "left"))


@pytest.mark.parametrize("layout,tight", [(None, True), ("constrained", True), (None, False)])
def test_save_and_caption_write_into_new_folder(tmp_path, layout, tight):
    out = tmp_path / "figs" / "main"
    fig = plt.figure(figsize=(2, 1.5), layout=layout)
    fig.add_subplot().plot([0, 1])
    pubfig.save(fig, "f1", out, tight=tight)
    pubfig.caption("f1", "  Caption text.  ", out)
    assert (out / "f1.pdf").exists() and (out / "f1.png").exists()
    assert (out / "f1.txt").read_text() == "Caption text.\n"


def test_lollipop_orders_and_highlights():
    fig, ax = plt.subplots()
    order = pubfig.lollipop(
        ax,
        ["otsu", "fluorostats", "li"],
        [0.91, 0.95, 0.93],
        los=[0.9, 0.94, 0.92],
        his=[0.92, 0.96, 0.94],
    )
    assert order == ["fluorostats", "li", "otsu"]
    order2 = pubfig.lollipop(
        ax, ["a", "b"], [0.5, 0.7], colors=["red", "blue"], floor=0.4, ascending=True
    )
    assert order2 == ["a", "b"]


@pytest.mark.parametrize("annotate", [True, False])
def test_bland_altman_panel_matches_agreement(annotate):
    rng = np.random.default_rng(1)
    a, b = rng.normal(5, 1, 30), rng.normal(5.3, 1, 30)
    fig, ax = plt.subplots()
    bias, loa = pubfig.bland_altman(ax, a, b, annotate=annotate)
    ref = agreement.bland_altman(a, b)
    assert bias == ref["bias"]
    assert loa == pytest.approx(1.96 * ref["sd_diff"])
    assert len(ax.texts) == (1 if annotate else 0)


@pytest.mark.parametrize("x,expected", [(382.6, 380), (14.8, 15), (0, 0.0), (-7.77, -7.8)])
def test_round_sig(x, expected):
    assert pubfig.round_sig(x) == expected


def test_palette_keys():
    for k in ["primary", "accent", "ink", "paper", "grid"]:
        assert pubfig.PALETTE[k].startswith("#")
    assert {"mesh", "background"} <= set(pubfig.DARK_PALETTE)


def test_material_color_exact_substring_and_fallback():
    assert pubfig.material_color("GelMA") == pubfig.MATERIAL_COLORS["GelMA"]
    assert pubfig.material_color("my GelMA sample") == pubfig.MATERIAL_COLORS["GelMA"]
    assert pubfig.material_color("totally unknown") == pubfig.PALETTE["muted"]


def test_darken_and_lighten_move_away_from_the_colour():
    assert pubfig.darken("#888888", 0.2) != "#888888" != pubfig.lighten("#888888", 0.2)


def test_panel_dx_dy_are_ignored_with_a_warning():
    # STY-4: retired keywords were accepted and ignored silently.
    fig, ax = plt.subplots()
    with pytest.warns(DeprecationWarning, match="panel\\(\\): 'dx' and 'dy' are deprecated"):
        pubfig.panel(ax, "a", dx=3.0)
    assert [t.get_text() for t in ax.texts] == ["a"]
