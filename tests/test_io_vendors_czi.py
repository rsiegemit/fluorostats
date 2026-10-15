"""Zeiss CZI loader tests with a fake czifile module (see test_io_vendors.py)."""

import sys
import types

import numpy as np
import pytest

from fluorostats import io, io_vendors


# ---------------------------------------------------------------------------
# Zeiss .czi
# ---------------------------------------------------------------------------

def _make_fake_czifile(xml, arr):
    fake = types.ModuleType("czifile")

    class CziFile:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def asarray(self):
            return arr

        def metadata(self):
            return xml

    fake.CziFile = CziFile
    return fake


def test_load_czi_volume(monkeypatch, tmp_path):
    xml = """<root>
      <Distance Id="X"><Value>2.5e-7</Value></Distance>
      <Distance Id="Y"><Value>2.5e-7</Value></Distance>
      <Distance Id="Z"><Value>1e-6</Value></Distance>
      <Channel Name="EGFP"/>
      <Channel Name="mCherry"/>
    </root>"""
    arr = np.zeros((1, 1, 2, 3, 8, 8), dtype=np.uint16)  # extra singleton dims
    monkeypatch.setitem(sys.modules, "czifile", _make_fake_czifile(xml, arr))
    path = tmp_path / "s.czi"
    path.write_bytes(b"")
    out, meta = io.load_volume(path)
    assert out.shape == (2, 3, 8, 8)
    assert meta["format"] == "zeiss_czi"
    assert meta["channel_names"] == ["EGFP", "mCherry"]
    assert meta["voxel_size_um"][0] == pytest.approx(1.0)
    assert meta["voxel_size_um"][2] == pytest.approx(0.25)


def test_czi_dyename_fallback(monkeypatch, tmp_path):
    xml = """<root>
      <DyeName>DAPI</DyeName>
    </root>"""
    arr = np.zeros((1, 4, 8, 8), dtype=np.uint16)
    monkeypatch.setitem(sys.modules, "czifile", _make_fake_czifile(xml, arr))
    path = tmp_path / "d.czi"
    path.write_bytes(b"")
    _, meta = io.load_volume(path)
    assert meta["channel_names"] == ["DAPI"]


def test_czi_metadata_exception_path():
    # metadata() raising -> defaults returned (except: pass)
    class Bad:
        def metadata(self):
            raise RuntimeError("x")

    meta = io_vendors._parse_czi_metadata(Bad())
    assert meta["voxel_size_um"] == (1.0, 1.0, 1.0)


def test_czi_metadata_empty_string():
    class Empty:
        def metadata(self):
            return ""

    meta = io_vendors._parse_czi_metadata(Empty())
    assert meta["channel_names"] == []


# ---------------------------------------------------------------------------
# Readers that label their axes are reordered by name, not by shape
# ---------------------------------------------------------------------------

def _zc_stack(n_z=5, n_c=2):
    """(Z, C, Y, X) stack whose channel c holds the value c + 1."""
    return np.stack([np.full((n_z, 8, 8), c + 1, np.uint16) for c in range(n_c)], axis=1)


def test_czi_scene_axes_reorder_by_name(monkeypatch, tmp_path):
    # czifile >= 2026 reads the first scene and labels it ('S' = colour samples).
    fake = _make_fake_czifile("", _zc_stack())
    scene = types.SimpleNamespace(axes="ZCYX")
    fake.CziFile.scenes = {0: scene}
    monkeypatch.setitem(sys.modules, "czifile", fake)
    path = tmp_path / "s.czi"
    path.write_bytes(b"")
    out, _ = io.load_volume(path)
    assert out.shape == (2, 5, 8, 8)
    assert (out[0].mean(), out[1].mean()) == (1, 2)


def test_czi_legacy_axes_with_time_and_scene(monkeypatch, tmp_path):
    # Older czifile labels the whole file: 'S' = scene, '0' = colour samples.
    t0 = _zc_stack()[None, :, :, :, :, None]                     # (S, Z, C, Y, X, 0)
    arr = np.stack([np.concatenate([t0, t0 + 10]), np.concatenate([t0, t0 + 10]) + 20], axis=1)
    fake = _make_fake_czifile("", arr)                            # (S, T, Z, C, Y, X, 0)
    fake.CziFile.axes = "STZCYX0"
    monkeypatch.setitem(sys.modules, "czifile", fake)
    path = tmp_path / "old.czi"
    path.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="scene 0 of 2") as rec:
        out, _ = io.load_volume(path)
    assert any("time point 0 of 2" in str(w.message) for w in rec)
    assert out.shape == (2, 5, 8, 8)
    assert (out[0].mean(), out[1].mean()) == (1, 2)
