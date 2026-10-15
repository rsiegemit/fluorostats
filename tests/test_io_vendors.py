"""Vendor-format loader tests using mocked third-party libraries: Olympus and
Leica LIF, plus the shared helpers. CZI, ND2 and TIFF have their own files
(test_io_vendors_<format>.py).

The real vendor libs (oiffile/readlif) may be installed, but we inject minimal
fakes via monkeypatch.setitem(sys.modules, ...) so the loaders run on synthetic
arrays + metadata and return the canonical (C,Z,Y,X) form without needing real
vendor files."""

import re
import sys
import types
import warnings

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


def _plane_files(prefix="s_", **sizes):
    """Olympus plane file names in natural order, e.g. ``s_C001Z002.tif``."""
    names = [prefix]
    for axis, n in sizes.items():
        names = [f"{name}{axis}{i:03d}" for name in names for i in range(1, n + 1)]
    return [f"Storage00001/{name}.tif" for name in names]


def _axis_sections(z_interval="2000", **sizes):
    """The mainfile's per-axis sections, as an Olympus file writes them: every axis
    code is listed, with MaxSize 0 for an axis the image does not have."""
    codes = {"X": 8, "Y": 8, "C": 1, "Z": 0, "T": 0, **sizes}
    return {
        f"Axis {i} Parameters Common": {
            "AxisCode": code,
            "MaxSize": str(n),
            "Interval": z_interval if code == "Z" else "0",
            "PixUnit": "nm" if code == "Z" else "um",
        }
        for i, (code, n) in enumerate(codes.items())
    }


def _make_fake_oiffile(mainfile, files=None):
    """Fake oiffile whose plane ``asarray(name)`` holds the last digit of each of
    the name's axis indices (``s_C002Z003.tif`` -> 23), so plane order can be checked."""
    fake = types.ModuleType("oiffile")
    files = _plane_files(C=2, Z=4) if files is None else files

    class OifFile:
        series = [files]

        def __init__(self, path):
            self.path = path
            self.mainfile = mainfile

        def asarray(self, name):
            digits = re.findall(r"[CZT]\d*(\d)", name.rsplit("_", 1)[-1])
            return np.full((8, 8), int("".join(digits)), np.uint16)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    fake.OifFile = OifFile
    return fake


def _load_olympus(monkeypatch, tmp_path, mainfile, files=None):
    monkeypatch.setitem(sys.modules, "oiffile", _make_fake_oiffile(mainfile, files))
    path = tmp_path / "sample.oib"
    path.write_bytes(b"")
    return io.load_volume(path)


def test_load_olympus_volume(monkeypatch, tmp_path):
    mainfile = {
        "Channel 1 Parameters": {"DyeName": "FITC"},
        "Channel 2 Parameters": {"DyeName": "(null)"},
        "Reference Image Parameter": {
            "WidthConvertValue": "0.25",
            "HeightConvertValue": "0.25",
        },
        **_axis_sections(C=2, Z=4),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # axes by name: no layout guess to announce
        arr, meta = _load_olympus(monkeypatch, tmp_path, mainfile)
    assert arr.shape == (2, 4, 8, 8)
    assert [arr[c, z, 0, 0] for c in range(2) for z in range(4)] == [
        10 * c + z for c in range(1, 3) for z in range(1, 5)
    ]
    assert meta["format"] == "olympus"
    assert meta["channel_names"] == ["FITC", "Ch2"]
    # z interval 2000 nm -> 2.0 um
    assert meta["voxel_size_um"] == (2.0, 0.25, 0.25)


def test_olympus_single_plane_channels_are_not_z(monkeypatch, tmp_path):
    # Regression: a 2-channel single-plane file came back as 1 channel x 2 slices,
    # and the unused Z axis section's interval was read as the z step.
    mainfile = {"Channel 1 Parameters": {"DyeName": "DAPI"}, **_axis_sections(C=2)}
    with pytest.warns(RuntimeWarning, match="calibration") as rec:
        arr, meta = _load_olympus(monkeypatch, tmp_path, mainfile, _plane_files(C=2))
    assert arr.shape == (2, 1, 8, 8)
    assert [arr[c, 0, 0, 0] for c in range(2)] == [1, 2]
    assert meta["channel_names"] == ["DAPI", "Ch2"]
    assert meta["voxel_size_source"] == "z:default,y:default,x:default"
    assert not [w for w in rec if "4D array" in str(w.message)]


def test_olympus_time_series_keeps_time_point_0(monkeypatch, tmp_path):
    # Regression: C x T frames were read as a z-stack with the unused Z interval.
    mainfile = _axis_sections(z_interval="5000", C=1, T=5)
    with pytest.warns(RuntimeWarning, match="keeping only time point 0 of 5"):
        arr, meta = _load_olympus(monkeypatch, tmp_path, mainfile, _plane_files(C=1, T=5))
    assert arr.shape == (1, 1, 8, 8)
    assert meta["voxel_size_um"][0] == 1.0


def test_olympus_czt_keeps_every_channel_and_slice(monkeypatch, tmp_path):
    mainfile = _axis_sections(C=2, Z=3, T=2)
    with pytest.warns(RuntimeWarning, match="keeping only time point 0 of 2"):
        arr, _ = _load_olympus(monkeypatch, tmp_path, mainfile, _plane_files(C=2, Z=3, T=2))
    assert arr.shape == (2, 3, 8, 8)
    assert arr[1, 2, 0, 0] == 2 * 100 + 3 * 10 + 1  # C002 Z003 T001


def test_olympus_axis_letters_in_the_file_prefix_are_not_axes(monkeypatch, tmp_path):
    # Regression: 'y2_40X_2_C001Z001.tif' was parsed from its 'y2' prefix, and 81
    # of 82 planes were dropped without a word.
    files = _plane_files(prefix="y2_40X_2_", C=2, Z=3)
    arr, _ = _load_olympus(monkeypatch, tmp_path, _axis_sections(C=2, Z=3), files)
    assert arr.shape == (2, 3, 8, 8)


def test_olympus_planes_that_disagree_with_the_mainfile_raise(monkeypatch, tmp_path):
    files = _plane_files(C=2, Z=3)
    with pytest.raises(ValueError, match="Z: 3 planes, mainfile 4"):
        _load_olympus(monkeypatch, tmp_path, _axis_sections(C=2, Z=4), files)
    with pytest.raises(ValueError, match="T: 0 planes, mainfile 2"):
        _load_olympus(monkeypatch, tmp_path, _axis_sections(C=2, Z=3, T=2), files)
    with pytest.raises(ValueError, match="5 plane files"):  # one plane file missing
        _load_olympus(monkeypatch, tmp_path, _axis_sections(C=2, Z=3), files[:2] + files[3:])


def test_olympus_comma_decimal_pixel_size_is_read(monkeypatch, tmp_path):
    # Regression: '0,207.0' (0.207 written under a comma-decimal locale) failed to
    # parse and was reported as "no y/x calibration found in the file".
    mainfile = {
        "Reference Image Parameter": {
            "WidthConvertValue": "0,207.0",
            "HeightConvertValue": "0,207.0",
        },
        **_axis_sections(C=2, Z=4),
    }
    with pytest.warns(RuntimeWarning, match=r"'0,207.0' read as 0.207 µm") as rec:
        _, meta = _load_olympus(monkeypatch, tmp_path, mainfile)
    assert meta["voxel_size_um"] == (2.0, 0.207, 0.207)
    assert meta["voxel_size_source"] == "z:file,y:file,x:file"
    assert not [w for w in rec if "calibration found" in str(w.message)]


def test_olympus_unnamed_channel_keeps_its_position(monkeypatch, tmp_path):
    mainfile = {
        "Channel 1 Parameters": {"DyeName": "(null)"},
        "Channel 2 Parameters": {"DyeName": "FITC"},
        **_axis_sections(C=2, Z=4),
    }
    _, meta = _load_olympus(monkeypatch, tmp_path, mainfile)
    assert meta["channel_names"] == ["Ch1", "FITC"]


def test_olympus_z_units_um_and_default(monkeypatch, tmp_path):
    _, meta = _load_olympus(monkeypatch, tmp_path, _axis_sections(z_interval="3", C=2, Z=4))
    assert meta["voxel_size_um"][0] == 0.003
    mainfile = _axis_sections(C=2, Z=4)
    mainfile["Axis 3 Parameters Common"] = {
        **mainfile["Axis 3 Parameters Common"],
        "PixUnit": "um",
        "Interval": "3",
    }
    _, meta = _load_olympus(monkeypatch, tmp_path, mainfile)
    assert meta["voxel_size_um"][0] == 3.0

    # unknown unit -> divide by 1000
    mainfile["Axis 3 Parameters Common"] = {
        **mainfile["Axis 3 Parameters Common"],
        "PixUnit": "pm",
        "Interval": "4000",
    }
    with pytest.warns(RuntimeWarning, match="PixUnit 'pm' not recognised"):
        _, meta2 = _load_olympus(monkeypatch, tmp_path, mainfile)
    assert meta2["voxel_size_um"][0] == 4.0


def test_olympus_z_ignores_non_z_axis_and_zero_interval():
    # A non-Z axis section, or a Z section with a non-positive interval, leaves z
    # at its default rather than being misread as the z step.
    class Oif:
        mainfile = {
            "Axis 0 Parameters Common": {"AxisCode": "X", "Interval": "7", "PixUnit": "um"},
            "Axis 3 Parameters Common": {
                "AxisCode": "Z",
                "MaxSize": "4",
                "Interval": "0",
                "PixUnit": "um",
            },
        }

    assert io_vendors._parse_olympus_metadata(Oif())["voxel_size_um"][0] == 1.0


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


def _install_readlif(monkeypatch, reader):
    """Install a fake readlif laid out like the real one: ``LifFile`` lives in
    ``readlif.reader``; the top-level package holds only ``__version__``."""
    package = types.ModuleType("readlif")
    package.__version__ = "0.6.6"
    package.reader = reader
    monkeypatch.setitem(sys.modules, "readlif", package)
    monkeypatch.setitem(sys.modules, "readlif.reader", reader)


def _make_fake_readlif(n_channels, n_z, scale, frame_ndim=2, empty=False, n_tiles=1):
    fake = types.ModuleType("readlif.reader")

    class Dims:
        z = n_z
        m = n_tiles

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
    fake = types.ModuleType("readlif.reader")

    class Dims:
        z = 3

    class Image:
        channels = 2
        dims = Dims()
        scale = [2.0, 2.0, 0.5]

        def get_frame(self, z, t, c):
            return np.full((4, 4), z * 10 + c, dtype=np.uint16)  # encodes (z, c)

    class LifFile:
        def __init__(self, path):
            self.image_list = [{"name": "img0"}]

        def get_image(self, i):
            return Image()

    fake.LifFile = LifFile
    _install_readlif(monkeypatch, fake)
    path = tmp_path / "m.lif"
    path.write_bytes(b"")
    arr, _ = io.load_volume(path)  # (C, Z, Y, X)
    assert arr.shape == (2, 3, 4, 4)
    for c in range(2):
        for z in range(3):
            assert arr[c, z, 0, 0] == z * 10 + c  # correct de-interleave


def test_load_lif_volume(monkeypatch, tmp_path):
    # readlif's scale is pixels per µm (x, y, z): 2 px/µm -> 0.5 µm voxels.
    fake = _make_fake_readlif(2, 3, [2.0, 4.0, 0.5])
    _install_readlif(monkeypatch, fake)
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
    _install_readlif(monkeypatch, fake)
    path = tmp_path / "rgb.lif"
    path.write_bytes(b"")
    arr, _ = io.load_volume(path)
    assert arr.shape == (1, 2, 8, 8)


def test_load_lif_empty_raises(monkeypatch, tmp_path):
    fake = _make_fake_readlif(1, 1, [2.0], empty=True)
    _install_readlif(monkeypatch, fake)
    path = tmp_path / "empty.lif"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="No images"):
        io.load_volume(path)


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
    _install_readlif(monkeypatch, fake)
    p = tmp_path / "s.lif"
    p.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="reading image 0, time point 0, mosaic tile 0 only"):
        io.load_volume(p)


def test_lif_missing_z_length_keeps_xy_calibration(monkeypatch, tmp_path):
    # Regression: a None z scale used to raise inside the parser, and the broad
    # except then discarded the valid x/y calibration as well.
    fake = _make_fake_readlif(1, 2, [2.0, 2.0, None, None])
    _install_readlif(monkeypatch, fake)
    path = tmp_path / "noz.lif"
    path.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="no z calibration"):
        _, meta = io.load_volume(path)
    assert meta["voxel_size_um"] == (1.0, 0.5, 0.5)


def test_lif_y_copied_from_x_is_said_and_recorded(monkeypatch, tmp_path):
    fake = _make_fake_readlif(1, 2, [2.0, None, 0.5])
    _install_readlif(monkeypatch, fake)
    path = tmp_path / "noy.lif"
    path.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="y voxel size copied from x") as rec:
        _, meta = io.load_volume(path)
    assert not [w for w in rec if "no y calibration" in str(w.message)]
    assert meta["voxel_size_um"] == (2.0, 0.5, 0.5)
    assert meta["voxel_size_source"] == "z:file,y:copied-from-x,x:file"


def test_lif_negative_length_is_a_calibration(monkeypatch, tmp_path):
    # Regression: a z stack acquired top-down has a negative z Length in the LIF
    # XML (readlif's scale is then negative) and was reported as uncalibrated.
    _install_readlif(monkeypatch, _make_fake_readlif(1, 2, [2.0, 2.0, -0.5]))
    path = tmp_path / "neg.lif"
    path.write_bytes(b"")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, meta = io.load_volume(path)
    assert meta["voxel_size_um"] == (2.0, 0.5, 0.5)
    assert meta["voxel_size_source"] == "z:file,y:file,x:file"


def test_lif_reading_the_first_mosaic_tile_is_announced(monkeypatch, tmp_path):
    # Regression: only tile 0 of a mosaic was read; the warning named images and
    # time points but not tiles.
    _install_readlif(monkeypatch, _make_fake_readlif(1, 2, [2.0, 2.0, 0.5], n_tiles=4))
    path = tmp_path / "tiles.lif"
    path.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match=r"\(1 images, 1 time points, 4 mosaic tiles"):
        io.load_volume(path)
