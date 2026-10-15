"""Tests for the Keyence BZ-X reader (fluorostats.keyence)."""

import struct
import zipfile

import numpy as np
import pytest
import tifffile

from fluorostats import keyence


def _double_bits(value: float) -> int:
    """Encode a float as the little-endian Int64 bit pattern Keyence stores."""
    return struct.unpack("<q", struct.pack("<d", value))[0]


def _write_gci(path, *, calibration_nm=400.0, pitch=25, total=3):
    """Write a minimal Keyence .gci archive matching the real XML layout."""
    P = "GroupFileProperty"
    entries = {
        f"{P}/Image/properties.xml": f'<Store><Calibration Type="System.Double">{_double_bits(calibration_nm)}</Calibration></Store>',
        f"{P}/Lens/properties.xml": '<Store><LensName Type="System.String">PlanApo 10x 0.45/4.00mm :Default</LensName></Store>',
        f"{P}/Stack/properties.xml": f'<Store><TotalNumber Type="System.Int32">{total}</TotalNumber>'
        f'<Pitch Type="System.Int32">{pitch}</Pitch></Store>',
        f"{P}/Channel0/properties.xml": '<Store><Comment Type="System.String">DAPI</Comment></Store>',
        f"{P}/Channel1/properties.xml": '<Store><Comment Type="System.String">GFP</Comment></Store>',
        f"{P}/Channel0/Shooting/Parameter/ExposureTime/properties.xml": '<Store><Numerator Type="System.Int32">35000</Numerator>'
        '<Denominator Type="System.Int32">1000000</Denominator></Store>',
        # entries parse_gci does not read must be skipped, not misparsed
        f"{P}/Unrelated/properties.xml": '<Store><Pitch Type="System.Int32">99</Pitch></Store>',
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
    assert meta["px_um"] == pytest.approx(0.4)  # 400 nm -> 0.4 µm
    assert meta["z_step_um"] == pytest.approx(2.5)  # Pitch 25 in 0.1-µm units
    assert meta["n_slices"] == 3
    assert meta["channels"]["0"]["name"] == "DAPI"
    assert meta["channels"]["1"]["name"] == "GFP"
    assert meta["channels"]["0"]["exposure_ms"] == pytest.approx(35.0)
    assert "10x" in meta["objective"]


def test_load_stack_all_channels(keyence_folder):
    arr, meta = keyence.load_keyence_stack(keyence_folder)
    assert arr.shape == (2, 3, 24, 32)  # (C, Z, Y, X)
    assert arr.dtype == np.uint16
    assert meta["vox_xy_um"] == pytest.approx(0.4)
    assert meta["vox_z_um"] == pytest.approx(2.5)
    assert meta["n_slices_loaded"] == 3


def test_load_stack_single_channel_and_downsample(keyence_folder):
    arr, meta = keyence.load_keyence_stack(keyence_folder, downsample=2, channel=0)
    assert arr.shape == (3, 12, 16)  # (Z, Y, X), XY halved
    assert meta["vox_xy_um"] == pytest.approx(0.8)
    assert meta["voxel_size_um"] == pytest.approx((2.5, 0.8, 0.8))


def test_voxel_size_without_gci_is_the_placeholder(keyence_folder):
    (keyence_folder / "Image .gci").unlink()
    _, meta = keyence.load_keyence_stack(keyence_folder, downsample=2)
    assert meta["voxel_size_um"] == (1.0, 1.0, 1.0)


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

    arr, meta = load_volume(keyence_folder)  # a folder, not a file
    assert arr.shape == (2, 3, 24, 32)
    assert meta["voxel_size_um"] == pytest.approx((2.5, 0.4, 0.4))
    assert meta["channel_names"] == ["DAPI", "GFP"]


def test_keyence_int64_bad_value():
    assert keyence._int64_to_double("not-an-int") is None


# ---------------------------------------------------------------------------
# keyence.py:139 — 2D single-channel CHF slice promoted to (1, Y, X)
# ---------------------------------------------------------------------------
def test_keyence_2d_single_channel_promotion(tmp_path):
    rng = np.random.default_rng(0)
    for zi in range(1, 4):
        plane = rng.integers(100, 3000, size=(24, 32)).astype(np.uint16)  # 2D
        tifffile.imwrite(tmp_path / f"Image _Z{zi:03d}_CHF.tif", plane)
    arr, meta = keyence.load_keyence_stack(tmp_path)
    # channel is None -> stacked to (C=1, Z, Y, X)
    assert arr.shape == (1, 3, 24, 32)
    assert meta["n_slices_loaded"] == 3


def test_z_step_unit_is_recorded_as_an_assumption_and_warned_once(keyence_folder):
    import warnings

    from fluorostats.io import load_volume

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _, meta = load_volume(keyence_folder)
    assumed = [w for w in rec if "0.1-µm units" in str(w.message)]
    assert len(assumed) == 1
    assert not [w for w in rec if "no z calibration" in str(w.message)]
    assert meta["voxel_size_source"] == "z:file-assumed-0.1um,y:file,x:file"


@pytest.mark.parametrize(
    "indices, said",
    [((1, 2, 4), r"missing z slice\(s\): \[3\]"), ((1, 2, 2), r"duplicated z slice\(s\): \[2\]")],
)
def test_missing_or_duplicate_z_slices_warn(tmp_path, indices, said):
    for i, zi in enumerate(indices):
        tifffile.imwrite(
            tmp_path / f"Image{i} _Z{zi:03d}_CHF.bz.ome.tif", np.zeros((2, 4, 4), np.uint16)
        )
    with pytest.warns(RuntimeWarning, match=said):
        arr, _ = keyence.load_keyence_stack(tmp_path)
    assert arr.shape == (2, 3, 4, 4)


def test_gci_handlers_return_new_dicts():
    meta = {"z_step_um": None, "n_slices": None, "channels": {"0": {"name": "DAPI"}}}
    stack = keyence._gci_stack(
        "Stack/properties.xml", "<TotalNumber>4</TotalNumber><Pitch>20</Pitch>", meta
    )
    exposure = keyence._gci_exposure(
        "Channel0/Shooting/Parameter/ExposureTime/properties.xml",
        "<Numerator>1</Numerator><Denominator>100</Denominator>",
        meta,
    )
    assert (stack["n_slices"], stack["z_step_um"]) == (4, 2.0)
    assert exposure["channels"]["0"] == {"name": "DAPI", "exposure_ms": 10.0}
    assert meta == {"z_step_um": None, "n_slices": None, "channels": {"0": {"name": "DAPI"}}}
    zero = "<Numerator>1</Numerator><Denominator>0</Denominator>"
    assert keyence._gci_exposure("Channel0/x", zero, meta) is meta  # nothing to record


@pytest.mark.parametrize("total", [4, 2])
def test_slice_count_other_than_the_gci_count_warns(tmp_path, total):
    # Regression: trailing missing slices were not detected (no gap inside the stack).
    for zi in range(1, 4):
        tifffile.imwrite(
            tmp_path / f"Image _Z{zi:03d}_CHF.bz.ome.tif", np.zeros((2, 4, 4), np.uint16)
        )
    _write_gci(tmp_path / "Image .gci", total=total)
    with pytest.warns(RuntimeWarning, match=f"lists {total} z slices but 3 were found"):
        _, meta = keyence.load_keyence_stack(tmp_path)
    assert (meta["n_slices"], meta["n_slices_loaded"]) == (total, 3)


def test_meta_has_the_same_keys_with_and_without_gci(keyence_folder):
    _, with_gci = keyence.load_keyence_stack(keyence_folder)
    (keyence_folder / "Image .gci").unlink()
    _, without = keyence.load_keyence_stack(keyence_folder)
    assert set(without) == set(with_gci)
    assert {
        k: without[k] for k in ("px_um", "vox_xy_um", "n_slices", "z_step_note")
    } == dict.fromkeys(("px_um", "vox_xy_um", "n_slices", "z_step_note"))
