"""Vendor-format loader tests using mocked third-party libraries: Olympus and
Leica LIF, plus the shared helpers. CZI, ND2 and TIFF have their own files
(test_io_vendors_<format>.py).

The real vendor libs (oiffile/readlif) may be installed, but we inject minimal
fakes via monkeypatch.setitem(sys.modules, ...) so the loaders run on synthetic
arrays + metadata and return the canonical (C,Z,Y,X) form without needing real
vendor files."""

import sys
import types

import numpy as np
import pytest

from fluorostats import io, io_vendors


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
        io_vendors.require("oiffile", "olympus")


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


def test_olympus_unnamed_channel_keeps_its_position(monkeypatch, tmp_path):
    mainfile = {"Channel 1 Parameters": {"DyeName": "(null)"},
                "Channel 2 Parameters": {"DyeName": "FITC"}}
    monkeypatch.setitem(sys.modules, "oiffile", _make_fake_oiffile(mainfile))
    path = tmp_path / "sample.oib"
    path.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="calibration"):
        _, meta = io.load_volume(path)
    assert meta["channel_names"] == ["Ch1", "FITC"]


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

    assert io_vendors._parse_olympus_metadata(Oif())["voxel_size_um"][0] == 1.0


def test_olympus_metadata_mainfile_raises(monkeypatch):
    # If mainfile access raises, defaults are returned.
    class BadOif:
        @property
        def mainfile(self):
            raise RuntimeError("no mainfile")

    meta = io_vendors._parse_olympus_metadata(BadOif())
    assert meta["voxel_size_um"] == (1.0, 1.0, 1.0)
    assert meta["channel_names"] == []


def test_section_get_attr_and_error():
    # object with attribute
    class S:
        DyeName = "Cy5"

    assert io_vendors._section_get(S(), "DyeName") == "Cy5"
    # object where subscript+attr both fail -> None
    assert io_vendors._section_get(object(), "Missing") is None


def test_safe_float_paths():
    assert io_vendors._safe_float("2.5", 1.0) == 2.5
    assert io_vendors._safe_float("-1", 9.0) == 9.0  # non-positive -> default
    assert io_vendors._safe_float("abc", 7.0) == 7.0  # unparseable -> default


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

    meta = io_vendors._parse_lif_metadata(BadImg())
    assert meta["voxel_size_um"] == (1.0, 1.0, 1.0)
    assert meta["channel_names"] == []


def test_lif_metadata_scale_none():
    class Img:
        scale = None
        channels = 0

    meta = io_vendors._parse_lif_metadata(Img())
    assert meta["voxel_size_um"] == (1.0, 1.0, 1.0)


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


def test_lif_y_copied_from_x_is_said_and_recorded(monkeypatch, tmp_path):
    fake = _make_fake_readlif(1, 2, [2.0, None, 0.5])
    monkeypatch.setitem(sys.modules, "readlif", fake)
    path = tmp_path / "noy.lif"
    path.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="y voxel size copied from x") as rec:
        _, meta = io.load_volume(path)
    assert not [w for w in rec if "no y calibration" in str(w.message)]
    assert meta["voxel_size_um"] == (2.0, 0.5, 0.5)
    assert meta["voxel_size_source"] == "z:file,y:copied-from-x,x:file"
