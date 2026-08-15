"""Vendor-format loader tests using mocked third-party libraries.

The real vendor libs (oiffile/czifile/nd2/readlif) may be installed, but we
inject minimal fakes via monkeypatch.setitem(sys.modules, ...) so the loaders
run on synthetic arrays + metadata and return the canonical (C,Z,Y,X) form
without needing real vendor files.
"""

import sys
import types

import numpy as np
import pytest

from fluorostats import io


# ---------------------------------------------------------------------------
# _require ImportError path
# ---------------------------------------------------------------------------

def test_require_missing_dependency_raises(monkeypatch):
    # Force importlib to fail even though the package may be installed.
    import importlib

    def _fail(name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", _fail)
    with pytest.raises(ImportError, match="requires the 'oiffile' package"):
        io._require("oiffile", "olympus")


# ---------------------------------------------------------------------------
# Olympus .oib / .oif
# ---------------------------------------------------------------------------

def _make_fake_oiffile(mainfile):
    fake = types.ModuleType("oiffile")

    class OifFile:
        def __init__(self, path):
            self.path = path
            self.mainfile = mainfile

        def asarray(self):
            return np.zeros((2, 4, 8, 8), dtype=np.uint16)

        def close(self):
            pass

    fake.OifFile = OifFile
    return fake


def test_load_olympus_volume(monkeypatch, tmp_path):
    mainfile = {
        "Channel 1 Parameters": {"DyeName": "FITC"},
        "Channel 2 Parameters": {"DyeName": "(null)"},
        "Reference Image Parameter": {
            "WidthConvertValue": "0.25",
            "HeightConvertValue": "0.25",
        },
        "Axis 3 Parameters Common": {
            "AxisCode": "Z",
            "Interval": "2000",
            "PixUnit": "nm",
        },
    }
    monkeypatch.setitem(sys.modules, "oiffile", _make_fake_oiffile(mainfile))
    path = tmp_path / "sample.oib"
    path.write_bytes(b"")
    arr, meta = io.load_volume(path)
    assert arr.shape == (2, 4, 8, 8)
    assert meta["format"] == "olympus"
    assert meta["channel_names"][0] == "FITC"
    # z interval 2000 nm -> 2.0 um
    assert meta["voxel_size_um"] == (2.0, 0.25, 0.25)


def test_olympus_z_units_um_and_default(monkeypatch, tmp_path):
    mainfile = {
        "Axis 3 Parameters Common": {
            "AxisCode": "Z",
            "Interval": "3",
            "PixUnit": "um",
        },
    }
    monkeypatch.setitem(sys.modules, "oiffile", _make_fake_oiffile(mainfile))
    path = tmp_path / "sample.oif"
    path.write_bytes(b"")
    _, meta = io.load_volume(path)
    assert meta["voxel_size_um"][0] == 3.0

    # unknown unit -> divide by 1000
    mainfile2 = {
        "Axis 3 Parameters Common": {
            "AxisCode": "Z",
            "Interval": "4000",
            "PixUnit": "pm",
        },
    }
    monkeypatch.setitem(sys.modules, "oiffile", _make_fake_oiffile(mainfile2))
    _, meta2 = io.load_volume(path)
    assert meta2["voxel_size_um"][0] == 4.0


def test_olympus_metadata_mainfile_raises(monkeypatch):
    # If mainfile access raises, defaults are returned.
    class BadOif:
        @property
        def mainfile(self):
            raise RuntimeError("no mainfile")

    meta = io._parse_olympus_metadata(BadOif())
    assert meta["voxel_size_um"] == (1.0, 1.0, 1.0)
    assert meta["channel_names"] == []


def test_section_get_attr_and_error():
    # object with attribute
    class S:
        DyeName = "Cy5"

    assert io._section_get(S(), "DyeName") == "Cy5"
    # object where subscript+attr both fail -> None
    assert io._section_get(object(), "Missing") is None


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

    meta = io._parse_czi_metadata(Bad())
    assert meta["voxel_size_um"] == (1.0, 1.0, 1.0)


def test_czi_metadata_empty_string():
    class Empty:
        def metadata(self):
            return ""

    meta = io._parse_czi_metadata(Empty())
    assert meta["channel_names"] == []


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

    meta = io._parse_nd2_metadata(F())
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

    meta = io._parse_nd2_metadata(F())
    assert meta["channel_names"] == ["backup"]


# ---------------------------------------------------------------------------
# Leica .lif
# ---------------------------------------------------------------------------

def _make_fake_readlif(n_channels, n_z, scale, frame_ndim=2, empty=False):
    fake = types.ModuleType("readlif")

    class Dims:
        z = n_z

    class Image:
        def __init__(self):
            self.channels = n_channels
            self.dims = Dims()
            self.scale = scale

        def get_frame(self, z, t, c):
            if frame_ndim == 3:
                return np.ones((8, 8, 3), dtype=np.uint16)
            return np.ones((8, 8), dtype=np.uint16)

    class LifFile:
        def __init__(self, path):
            self.image_list = [] if empty else [{"name": "img0"}]

        def get_image(self, i):
            return Image()

    fake.LifFile = LifFile
    return fake


def test_load_lif_volume(monkeypatch, tmp_path):
    fake = _make_fake_readlif(2, 3, [1e6, 2e6, 1e6])
    monkeypatch.setitem(sys.modules, "readlif", fake)
    path = tmp_path / "s.lif"
    path.write_bytes(b"")
    arr, meta = io.load_volume(path)
    assert arr.shape == (2, 3, 8, 8)
    assert meta["format"] == "leica_lif"
    # scale 1e6 -> 1/1e6*1e6 = 1.0 um; y scale 2e6 -> 0.5
    assert meta["voxel_size_um"] == (1.0, 0.5, 1.0)
    assert meta["channel_names"] == ["Ch1", "Ch2"]


def test_load_lif_volume_rgb_frames(monkeypatch, tmp_path):
    fake = _make_fake_readlif(1, 2, [1e6], frame_ndim=3)
    monkeypatch.setitem(sys.modules, "readlif", fake)
    path = tmp_path / "rgb.lif"
    path.write_bytes(b"")
    arr, _ = io.load_volume(path)
    assert arr.shape == (1, 2, 8, 8)


def test_load_lif_empty_raises(monkeypatch, tmp_path):
    fake = _make_fake_readlif(1, 1, [1e6], empty=True)
    monkeypatch.setitem(sys.modules, "readlif", fake)
    path = tmp_path / "empty.lif"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="No images"):
        io.load_volume(path)


def test_lif_metadata_exception_paths():
    # scale access raising and channels raising -> defaults
    class BadImg:
        @property
        def scale(self):
            raise RuntimeError("no scale")

        @property
        def channels(self):
            raise RuntimeError("no channels")

    meta = io._parse_lif_metadata(BadImg())
    assert meta["voxel_size_um"] == (1.0, 1.0, 1.0)
    assert meta["channel_names"] == []


def test_lif_metadata_scale_none():
    class Img:
        scale = None
        channels = 0

    meta = io._parse_lif_metadata(Img())
    assert meta["voxel_size_um"] == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# supported_formats optional-present branch (already installed) + load_auto
# ---------------------------------------------------------------------------

def test_load_auto_falls_back_to_image_on_volume_error(monkeypatch, tmp_path):
    # A .tif that fails volume loading falls back to load_image.
    import tifffile

    arr = np.random.default_rng(0).integers(0, 255, (16, 16), dtype=np.uint8)
    path = tmp_path / "img.tif"
    tifffile.imwrite(str(path), arr)

    def boom(p):
        raise RuntimeError("forced volume failure")

    monkeypatch.setattr(io, "load_volume", boom)
    out, meta = io.load_auto(path)
    assert out.shape == (1, 16, 16)


# ---------------------------------------------------------------------------
# canonicalization edge cases
# ---------------------------------------------------------------------------

def test_canonicalize_volume_bad_ndim_raises():
    with pytest.raises(ValueError, match="Unexpected array shape"):
        io._canonicalize_volume(np.zeros((5,)), {"channel_names": []})


def test_canonicalize_volume_high_ndim_recurses():
    # 6D with leading singletons collapses to 4D via recursion
    arr = np.zeros((1, 1, 2, 3, 8, 8), dtype=np.uint16)
    out = io._canonicalize_volume(arr, {"channel_names": []})
    assert out.shape == (2, 3, 8, 8)


def test_canonicalize_image_bad_ndim_raises():
    with pytest.raises(ValueError, match="Unexpected 2D image shape"):
        io._canonicalize_image(np.zeros((2, 2, 2, 2)), {"channel_names": []})


def test_squeeze_trailing_singleton():
    arr = np.zeros((2, 3, 8, 8, 1), dtype=np.uint16)
    out = io._squeeze_singleton_dims(arr)
    assert out.shape == (2, 3, 8, 8)


def test_supported_formats_missing_optional_branch(monkeypatch):
    # Force _require to raise so the "requires: pip install" note is emitted.
    def _boom(pkg, extra):
        raise ImportError("missing")

    monkeypatch.setattr(io, "_require", _boom)
    fmts = io.supported_formats()
    assert "requires" in fmts[".czi"]
    assert "requires" in fmts[".oib"]


def test_nd2_channels_iteration_raises():
    # metadata.channels present but iterating/attr access raises -> except: pass
    class BadChannels:
        def __iter__(self):
            raise RuntimeError("bad")

    class Meta:
        channels = BadChannels()

    class F:
        metadata = Meta()

    meta = io._parse_nd2_metadata(F())
    assert meta["channel_names"] == []


def test_ome_metadata_parse_error():
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    out = io._parse_ome_metadata("<not valid xml", meta)
    assert out["voxel_size_um"] == (1.0, 1.0, 1.0)


def test_imagej_metadata_exception():
    # spacing present but non-float -> except: pass leaves defaults
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    out = io._parse_imagej_metadata({"spacing": "notanumber"}, meta)
    assert out["voxel_size_um"] == (1.0, 1.0, 1.0)


def test_squeeze_non_singleton_high_ndim():
    # ndim>4 with no leading/trailing singleton -> final while arr=arr[0]
    arr = np.zeros((2, 3, 4, 8, 8), dtype=np.uint16)
    out = io._squeeze_singleton_dims(arr)
    assert out.ndim == 4


def test_canonicalize_volume_high_ndim_squeezes():
    # 6D non-singleton -> squeeze collapses to 4D
    arr = np.zeros((2, 2, 3, 4, 8, 8), dtype=np.uint16)
    out = io._canonicalize_volume(arr, {"channel_names": []})
    assert out.ndim == 4


def test_safe_float_paths():
    assert io._safe_float("2.5", 1.0) == 2.5
    assert io._safe_float("-1", 9.0) == 9.0  # non-positive -> default
    assert io._safe_float("abc", 7.0) == 7.0  # unparseable -> default
