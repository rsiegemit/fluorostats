"""Nikon ND2 loader tests with a fake nd2 module (see test_io_vendors.py)."""

import sys
import types

import numpy as np
import pytest

from fluorostats import io, io_vendors


# ---------------------------------------------------------------------------
# Nikon .nd2
# ---------------------------------------------------------------------------

def _make_fake_nd2(arr, voxel, channels):
    fake = types.ModuleType("nd2")

    class VS:
        x, y, z = voxel

    class ChObj:
        def __init__(self, name):
            self.name = name

    class ChMeta:
        def __init__(self, name):
            self.channel = ChObj(name)

    class Metadata:
        def __init__(self):
            self.channels = [ChMeta(n) for n in channels]

    class ND2File:
        def __init__(self, path):
            self.metadata = Metadata()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def asarray(self):
            return arr

        def voxel_size(self):
            return VS()

    fake.ND2File = ND2File
    return fake


def test_load_nd2_volume(monkeypatch, tmp_path):
    arr = np.zeros((2, 5, 8, 8), dtype=np.uint16)
    fake = _make_fake_nd2(arr, (0.3, 0.3, 1.5), ["405", "488"])
    monkeypatch.setitem(sys.modules, "nd2", fake)
    path = tmp_path / "s.nd2"
    path.write_bytes(b"")
    out, meta = io.load_volume(path)
    assert out.shape == (2, 5, 8, 8)
    assert meta["format"] == "nikon_nd2"
    assert meta["voxel_size_um"] == (1.5, 0.3, 0.3)
    assert meta["channel_names"] == ["405", "488"]


def test_nd2_metadata_exception_paths():
    # voxel_size raising and metadata missing channels -> defaults
    class F:
        def voxel_size(self):
            raise RuntimeError("nope")

    meta = io_vendors._parse_nd2_metadata(F())
    assert meta["voxel_size_um"] == (1.0, 1.0, 1.0)
    assert meta["channel_names"] == []


def test_nd2_channel_name_fallback_to_ch():
    # ch.channel.name empty -> fall back to ch.name
    class ChObj:
        name = ""

    class ChMeta:
        channel = ChObj()
        name = "backup"

    class Meta:
        channels = [ChMeta()]

    class F:
        metadata = Meta()

    meta = io_vendors._parse_nd2_metadata(F())
    assert meta["channel_names"] == ["backup"]


def test_nd2_channels_iteration_raises():
    # metadata.channels present but iterating/attr access raises -> except: pass
    class BadChannels:
        def __iter__(self):
            raise RuntimeError("bad")

    class Meta:
        channels = BadChannels()

    class F:
        metadata = Meta()

    meta = io_vendors._parse_nd2_metadata(F())
    assert meta["channel_names"] == []


# ---------------------------------------------------------------------------
# Readers that label their axes are reordered by name, not by shape
# ---------------------------------------------------------------------------

def _zc_stack(n_z=5, n_c=2):
    """(Z, C, Y, X) stack whose channel c holds the value c + 1."""
    return np.stack([np.full((n_z, 8, 8), c + 1, np.uint16) for c in range(n_c)], axis=1)


def test_nd2_zcyx_is_reordered_by_axis_name(monkeypatch, tmp_path):
    fake = _make_fake_nd2(_zc_stack(), (0.3, 0.3, 1.5), ["405", "488"])
    fake.ND2File.sizes = {"Z": 5, "C": 2, "Y": 8, "X": 8}
    monkeypatch.setitem(sys.modules, "nd2", fake)
    path = tmp_path / "s.nd2"
    path.write_bytes(b"")
    out, _ = io.load_volume(path)
    assert out.shape == (2, 5, 8, 8)
    assert (out[0].mean(), out[1].mean()) == (1, 2)


def test_nd2_time_and_position_keep_first_index(monkeypatch, tmp_path):
    arr = np.stack([np.stack([_zc_stack(), _zc_stack() + 10])] * 3, axis=1)   # (T, P, Z, C, Y, X)
    fake = _make_fake_nd2(arr, (0.3, 0.3, 1.5), [])
    fake.ND2File.sizes = {"T": 2, "P": 3, "Z": 5, "C": 2, "Y": 8, "X": 8}
    monkeypatch.setitem(sys.modules, "nd2", fake)
    path = tmp_path / "tp.nd2"
    path.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="time point 0 of 2") as rec:
        out, _ = io.load_volume(path)
    assert any("position 0 of 3" in str(w.message) for w in rec)
    assert out.shape == (2, 5, 8, 8)
    assert (out[0].mean(), out[1].mean()) == (1, 2)


def test_nd2_rgb_samples_become_channels(monkeypatch, tmp_path):
    arr = np.stack([np.full((4, 8, 8), v, np.uint8) for v in (10, 20, 30)], -1)   # (Z, Y, X, S)
    fake = _make_fake_nd2(arr, (0.3, 0.3, 1.5), [])
    fake.ND2File.sizes = {"Z": 4, "Y": 8, "X": 8, "S": 3}
    monkeypatch.setitem(sys.modules, "nd2", fake)
    path = tmp_path / "rgb.nd2"
    path.write_bytes(b"")
    out, _ = io.load_volume(path)
    assert out.shape == (3, 4, 8, 8)
    assert [out[c].mean() for c in range(3)] == [10, 20, 30]


def test_labelled_axes_with_unknown_name_fall_back_to_shape(monkeypatch, tmp_path):
    arr = np.zeros((3, 2, 5, 8, 8), np.uint16)
    fake = _make_fake_nd2(arr, (0.3, 0.3, 1.5), [])
    fake.ND2File.sizes = {"U": 3, "C": 2, "Z": 5, "Y": 8, "X": 8}
    monkeypatch.setitem(sys.modules, "nd2", fake)
    path = tmp_path / "u.nd2"
    path.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="first of 3 entries"):
        out, _ = io.load_volume(path)
    assert out.shape == (2, 5, 8, 8)
