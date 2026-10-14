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


def test_olympus_z_ignores_non_z_axis_and_zero_interval():
    # A non-Z axis section, or a Z section with a non-positive interval, leaves z
    # at its default rather than being misread as the z step.
    class Oif:
        mainfile = {
            "Axis 0 Parameters Common": {"AxisCode": "X", "Interval": "7", "PixUnit": "um"},
            "Axis 3 Parameters Common": {"AxisCode": "Z", "Interval": "0", "PixUnit": "um"},
        }

    assert io._parse_olympus_metadata(Oif())["voxel_size_um"][0] == 1.0


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


def test_load_lif_deinterleaves_multichannel(monkeypatch, tmp_path):
    """Multi-channel LIF: frames are appended Z-major/C-minor, so channel c, slice z
    must come back de-interleaved (regression for the (Z,C)-vs-(C,Z) reshape bug)."""
    fake = types.ModuleType("readlif")

    class Dims:
        z = 3

    class Image:
        channels = 2
        dims = Dims()
        scale = [2.0, 2.0, 0.5]

        def get_frame(self, z, t, c):
            return np.full((4, 4), z * 10 + c, dtype=np.uint16)   # encodes (z, c)

    class LifFile:
        def __init__(self, path):
            self.image_list = [{"name": "img0"}]

        def get_image(self, i):
            return Image()

    fake.LifFile = LifFile
    monkeypatch.setitem(sys.modules, "readlif", fake)
    path = tmp_path / "m.lif"
    path.write_bytes(b"")
    arr, _ = io.load_volume(path)                 # (C, Z, Y, X)
    assert arr.shape == (2, 3, 4, 4)
    for c in range(2):
        for z in range(3):
            assert arr[c, z, 0, 0] == z * 10 + c   # correct de-interleave


def test_load_lif_volume(monkeypatch, tmp_path):
    # readlif's scale is pixels per µm (x, y, z): 2 px/µm -> 0.5 µm voxels.
    fake = _make_fake_readlif(2, 3, [2.0, 4.0, 0.5])
    monkeypatch.setitem(sys.modules, "readlif", fake)
    path = tmp_path / "s.lif"
    path.write_bytes(b"")
    arr, meta = io.load_volume(path)
    assert arr.shape == (2, 3, 8, 8)
    assert meta["format"] == "leica_lif"
    assert meta["voxel_size_um"] == (2.0, 0.25, 0.5)
    assert meta["voxel_size_source"] == "z:file,y:file,x:file"
    assert meta["channel_names"] == ["Ch1", "Ch2"]


def test_load_lif_volume_rgb_frames(monkeypatch, tmp_path):
    fake = _make_fake_readlif(1, 2, [2.0], frame_ndim=3)
    monkeypatch.setitem(sys.modules, "readlif", fake)
    path = tmp_path / "rgb.lif"
    path.write_bytes(b"")
    arr, _ = io.load_volume(path)
    assert arr.shape == (1, 2, 8, 8)


def test_load_lif_empty_raises(monkeypatch, tmp_path):
    fake = _make_fake_readlif(1, 1, [2.0], empty=True)
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


def test_imagej_metadata_unparseable_spacing_warns():
    # spacing present but non-float -> announced, defaults kept
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    with pytest.warns(RuntimeWarning, match="'notanumber' is not a number"):
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


def test_lif_warns_when_extra_images_are_ignored(monkeypatch, tmp_path):
    fake = _make_fake_readlif(1, 2, [2.0, 2.0, 0.5])
    base = fake.LifFile

    class TwoImages(base):
        def __init__(self, path):
            super().__init__(path)
            self.image_list = [{"name": "img0"}, {"name": "img1"}]

    fake.LifFile = TwoImages
    monkeypatch.setitem(sys.modules, "readlif", fake)
    p = tmp_path / "s.lif"
    p.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="reading image 0 at time point 0 only"):
        io.load_volume(p)


def test_lif_missing_z_length_keeps_xy_calibration(monkeypatch, tmp_path):
    # Regression: a None z scale used to raise inside the parser, and the broad
    # except then discarded the valid x/y calibration as well.
    fake = _make_fake_readlif(1, 2, [2.0, 2.0, None, None])
    monkeypatch.setitem(sys.modules, "readlif", fake)
    path = tmp_path / "noz.lif"
    path.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="no z calibration"):
        _, meta = io.load_volume(path)
    assert meta["voxel_size_um"] == (1.0, 0.5, 0.5)


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
