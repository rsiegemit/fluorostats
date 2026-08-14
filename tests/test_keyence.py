"""Tests for the Keyence BZ-X reader (fluorostats.keyence)."""
import struct
import zipfile

import numpy as np
import pytest

from fluorostats import keyence

tifffile = pytest.importorskip("tifffile")


def _double_bits(value: float) -> int:
    """Encode a float as the little-endian Int64 bit pattern Keyence stores."""
    return struct.unpack("<q", struct.pack("<d", value))[0]


def _write_gci(path, *, calibration_nm=400.0, pitch=25, total=3):
    """Write a minimal Keyence .gci archive matching the real XML layout."""
    P = "GroupFileProperty"
    entries = {
        f"{P}/Image/properties.xml":
            f'<Store><Calibration Type="System.Double">{_double_bits(calibration_nm)}</Calibration></Store>',
        f"{P}/Lens/properties.xml":
            '<Store><LensName Type="System.String">PlanApo 10x 0.45/4.00mm :Default</LensName></Store>',
        f"{P}/Stack/properties.xml":
            f'<Store><TotalNumber Type="System.Int32">{total}</TotalNumber>'
            f'<Pitch Type="System.Int32">{pitch}</Pitch></Store>',
        f"{P}/Channel0/properties.xml": '<Store><Comment Type="System.String">DAPI</Comment></Store>',
        f"{P}/Channel1/properties.xml": '<Store><Comment Type="System.String">GFP</Comment></Store>',
        f"{P}/Channel0/Shooting/Parameter/ExposureTime/properties.xml":
            '<Store><Numerator Type="System.Int32">35000</Numerator>'
            '<Denominator Type="System.Int32">1000000</Denominator></Store>',
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, body in entries.items():
            z.writestr(name, '<?xml version="1.0"?>' + body)


@pytest.fixture
def keyence_folder(tmp_path):
    rng = np.random.default_rng(0)
    for zi in range(1, 4):
        vol = (rng.integers(100, 3000, size=(2, 24, 32))).astype(np.uint16)
        tifffile.imwrite(tmp_path / f"Image _Z{zi:03d}_CHF.bz.ome.tif", vol)
    _write_gci(tmp_path / "Image .gci")
    return tmp_path


def test_parse_gci(keyence_folder):
    meta = keyence.parse_gci(keyence_folder / "Image .gci")
    assert meta["px_um"] == pytest.approx(0.4)          # 400 nm -> 0.4 µm
    assert meta["z_step_um"] == pytest.approx(2.5)       # Pitch 25 in 0.1-µm units
    assert meta["n_slices"] == 3
    assert meta["channels"]["0"]["name"] == "DAPI"
    assert meta["channels"]["1"]["name"] == "GFP"
    assert meta["channels"]["0"]["exposure_ms"] == pytest.approx(35.0)
    assert "10x" in meta["objective"]


def test_load_stack_all_channels(keyence_folder):
    arr, meta = keyence.load_keyence_stack(keyence_folder)
    assert arr.shape == (2, 3, 24, 32)                   # (C, Z, Y, X)
    assert arr.dtype == np.uint16
    assert meta["vox_xy_um"] == pytest.approx(0.4)
    assert meta["vox_z_um"] == pytest.approx(2.5)
    assert meta["n_slices_loaded"] == 3


def test_load_stack_single_channel_and_downsample(keyence_folder):
    arr, meta = keyence.load_keyence_stack(keyence_folder, downsample=2, channel=0)
    assert arr.shape == (3, 12, 16)                      # (Z, Y, X), XY halved
    assert meta["vox_xy_um"] == pytest.approx(0.8)


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        keyence.load_keyence_stack(tmp_path)


def test_is_keyence_folder(keyence_folder, tmp_path):
    assert keyence.is_keyence_folder(keyence_folder) is True
    assert keyence.is_keyence_folder(tmp_path / "does_not_exist") is False
    (tmp_path / "empty").mkdir()
    assert keyence.is_keyence_folder(tmp_path / "empty") is False


def test_load_volume_autodetects_folder(keyence_folder):
    from fluorostats.io import load_volume
    arr, meta = load_volume(keyence_folder)                  # a folder, not a file
    assert arr.shape == (2, 3, 24, 32)
    assert meta["voxel_size_um"] == pytest.approx((2.5, 0.4, 0.4))
    assert meta["channel_names"] == ["DAPI", "GFP"]
