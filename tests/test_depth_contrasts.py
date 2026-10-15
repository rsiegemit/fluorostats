"""Tests for fluorostats.depth_contrasts (the depth driver's two-group statistics)."""
import pytest

from fluorostats import depth_contrasts, stats
from fluorostats.depth import PenetrationFit


def _row(group, i, auc, lam, role="primary", fit_ok=True):
    fit = PenetrationFit(lambda_um=lam, I0=1.0, offset=None, r_squared=0.99, rmse=0.0,
                         n_points=10, fit_ok=fit_ok)
    return {"group": group, "stack": f"{group}{i}", "role": role,
            "result": {"auc": {"0_10um": {"normalized": auc}, "full": {"normalized": 2 * auc}},
                       "fit": fit}}


WINDOWS = [(0.0, 10.0), ("full",)]
ROWS = ([_row("a", i, v, 5.0 + i) for i, v in enumerate((1.0, 2.0, 3.0))]
        + [_row("b", i, v, 9.0 + i) for i, v in enumerate((4.0, 5.0, 6.0))]
        + [_row("b", 9, 100.0, 1.0, role="aux")]                   # aux: never tested
        + [_row("a", 9, 0.5, 50.0, fit_ok=False)])                 # rejected fit: no λ


def test_contrasts_are_the_library_tests_with_bh_across_windows():
    tests = depth_contrasts.contrasts(ROWS, WINDOWS)
    assert [(c["metric"], c["window"]) for c in tests] == [
        ("auc_normalized", "0_10um"), ("auc_normalized", "full"), ("lambda_um", "")]
    a0 = [0.5, 1.0, 2.0, 3.0]                                      # group a's primary AUCs
    ref = stats.mann_whitney(a0, [4.0, 5.0, 6.0])
    c = tests[0]
    assert (c["group_a"], c["group_b"], c["n_a"], c["n_b"]) == ("a", "b", 4, 3)
    assert (c["U"], c["p"], c["cliffs_delta"]) == (ref["u"], ref["p"], ref["cliffs_delta"])
    qs = stats.bh_fdr([tests[0]["p"], tests[1]["p"]])
    assert [tests[0]["q"], tests[1]["q"]] == list(qs)
    assert tests[0]["correction"] == "Benjamini-Hochberg FDR across 2 windows"
    lam = tests[2]
    assert (lam["n_a"], lam["n_b"]) == (3, 3) and lam["q"] == lam["p"]
    assert lam["correction"] == "none (single test; q = p)"
    assert all(c["test"] == "Mann-Whitney U, two-sided" for c in tests)


def test_no_contrasts_without_exactly_two_groups():
    three = ROWS + [_row("c", 0, 7.0, 3.0)]
    assert depth_contrasts.contrasts(three, WINDOWS) == []


def test_write_contrasts_csv_formats_floats_and_keeps_text(tmp_path):
    out = tmp_path / "c.csv"
    depth_contrasts.write_contrasts_csv(out, depth_contrasts.contrasts(ROWS, WINDOWS))
    lines = out.read_text().splitlines()
    assert lines[0] == "metric,window,group_a,group_b,n_a,n_b,U,p,q,cliffs_delta,test,correction"
    assert lines[1].startswith("auc_normalized,0_10um,a,b,4,3,0,")
    assert lines[3].startswith("lambda_um,,a,b,3,3,")


def test_printed_summary_is_the_contrasts_rows(capsys):
    tests = depth_contrasts.contrasts(ROWS, WINDOWS)
    depth_contrasts.print_auc_summary(ROWS, WINDOWS, False, tests)
    depth_contrasts.print_lambda_summary(ROWS, True, tests)
    text = capsys.readouterr().out
    c = tests[0]
    assert (f"Mann-Whitney a vs b: U={c['U']:.1f}, p={c['p']:.4f}, Cliff's δ="
            f"{c['cliffs_delta']:+.2f}, BH q={c['q']:.4f} across 2 windows (n=4,3)") in text
    assert text.count("Welch") == 1 and "Welch t-test (parametric, requested) on λ," in text
    assert "[a/primary] a9: λ=  50.00 µm  R²=0.9900  [fit_ok=False]" in text


@pytest.mark.parametrize("n_groups,which", [(1, ""), (3, "s")])
def test_summary_says_why_contrasts_are_skipped(capsys, n_groups, which):
    rows = [_row(g, 0, 1.0, 5.0) for g in "abc"[:n_groups]]
    depth_contrasts.print_auc_summary(rows, WINDOWS[:1], False, [])
    depth_contrasts.print_lambda_summary(rows, False, [])
    text = capsys.readouterr().out
    assert f"contrasts skipped: {n_groups} group{which}; a two-group" in text
    assert f"contrasts skipped: {n_groups} group{which} with an accepted λ fit" in text
