import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fluorostats import agreement, pubfig, stats  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_apply_style_sets_journal_rcparams():
    pubfig.apply_style()
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["axes.labelsize"] == pubfig.FS["label"]
    assert plt.rcParams["axes.spines.top"] is False
    plt.rcdefaults()


@pytest.mark.parametrize("xanchor", [None, 0.03])
def test_panel_draws_letter_and_title(xanchor):
    fig, ax = plt.subplots()
    pubfig.panel(ax, "a", "recovery", xanchor=xanchor)
    texts = [t.get_text() for t in ax.texts]
    assert texts == ["a", "recovery"]
    pubfig.panel(ax, "b", xanchor=xanchor)          # no title -> letter only
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


@pytest.mark.parametrize("loc,center", [("lower right", False), ("upper left", False),
                                        ("lower left", True)])
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
    order = pubfig.lollipop(ax, ["otsu", "fluorostats", "li"], [0.91, 0.95, 0.93],
                            los=[0.9, 0.94, 0.92], his=[0.92, 0.96, 0.94])
    assert order == ["fluorostats", "li", "otsu"]
    order2 = pubfig.lollipop(ax, ["a", "b"], [0.5, 0.7], colors=["red", "blue"],
                             floor=0.4, ascending=True)
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


def test_bootstrap_mean_ci_is_seeded_and_brackets_mean():
    x = np.random.default_rng(2).normal(0.8, 0.1, 50)
    r = stats.bootstrap_mean_ci(x, n_boot=2000, seed=0)
    assert r == stats.bootstrap_mean_ci(x, n_boot=2000, seed=0)
    assert r["ci_low"] < r["mean"] < r["ci_high"]
    assert r["mean"] == float(x.mean())
    wide = stats.bootstrap_mean_ci(x, n_boot=2000, ci=0.99, seed=0)
    assert wide["ci_low"] <= r["ci_low"] and wide["ci_high"] >= r["ci_high"]
