"""Tests for fluorostats.cli_measure: the quant commands' per-file loop (the
measurements themselves are tested end to end in test_cli.py)."""

from pathlib import Path

from fluorostats import cli_measure


def test_quantify_files_records_a_failed_file_and_measures_the_rest():
    # Regression (vendor report): one unreadable file aborted the whole batch.
    files = [Path("A/s1.tif"), Path("A/bad.lif"), Path("A/s2.tif"), Path("B/s1.tif")]

    def measure(path):
        if path.suffix == ".lif":
            raise ValueError("cannot read this LIF")
        return {"m": len(path.name)}

    rows, failures = cli_measure.quantify_files(files, "parent", "x", measure)
    assert [(r["file"], r["condition"], r["replicate"]) for r in rows] == [
        ("s1.tif", "A", 1),
        ("s2.tif", "A", 2),  # a failed file takes no replicate number
        ("s1.tif", "B", 1),
    ]
    assert failures == [
        {"file": "bad.lif", "condition": "A", "error": "ValueError: cannot read this LIF"}
    ]
