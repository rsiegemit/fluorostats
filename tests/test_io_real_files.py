"""Regression tests on real vendor sample files (Olympus, Leica, Nikon, Zeiss,
OME-TIFF, ImageJ), checked against each format's native reader.

They run only when ``FLUOROSTATS_VENDOR_DATA`` names a folder holding the
sample files (any layout; each is found by name). The files are public test
images and are not part of the repository; the documentation lists where to
download them. The vendor readers (oiffile, readlif, nd2, czifile) must be
installed. The synthetic fakes in test_io_vendors*.py cover the code paths;
these tests catch what fakes cannot: a reader API or file layout that differs
from what the fakes assume.
"""

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pytest
import tifffile

from fluorostats import io

DATA = os.environ.get("FLUOROSTATS_VENDOR_DATA")
pytestmark = pytest.mark.skipif(
    not DATA,
    reason="needs FLUOROSTATS_VENDOR_DATA: a folder of the real vendor sample files",
)


def _file(name: str) -> Path:
    found = sorted(Path(str(DATA)).rglob(name))
    assert found, f"{name} is not in FLUOROSTATS_VENDOR_DATA={DATA}"
    return found[0]


def _canonical(arr: np.ndarray, axes: str) -> np.ndarray:
    """(C, Z, Y, X) of a native array labelled ``axes``: every other axis at index
    0, colour samples (S) as extra channels after their channel."""
    arr = arr[tuple(slice(None) if a in "CZYXS" else 0 for a in axes)]
    kept = [a for a in axes if a in "CZYXS"]
    missing = [a for a in "CSZ" if a not in kept]
    arr = arr.reshape((1,) * len(missing) + arr.shape)
    order = missing + kept
    arr = arr.transpose([order.index(a) for a in "CSZYX"])
    return arr.reshape(-1, *arr.shape[2:])


def _native_olympus(path: Path) -> np.ndarray:
    """Planes picked by their file names' C/Z indices, time point 1 only."""
    import oiffile

    planes = {}
    with oiffile.OifFile(str(path)) as oif:
        for name in oif.glob("*.tif"):
            m = re.search(r"C(\d+)(?:Z(\d+))?(?:T(\d+))?\.tif$", name)
            if m and int(m.group(3) or 1) == 1:
                planes[int(m.group(1)) - 1, int(m.group(2) or 1) - 1] = oif.asarray(name)
    n_c, n_z = (max(k[i] for k in planes) + 1 for i in (0, 1))
    return np.stack([np.stack([planes[c, z] for z in range(n_z)]) for c in range(n_c)])


def _native_lif(path: Path) -> np.ndarray:
    from readlif.reader import LifFile

    img = LifFile(str(path)).get_image(0)
    return np.stack(
        [
            np.stack([np.asarray(img.get_frame(z=z, t=0, c=c)) for z in range(img.dims.z)])
            for c in range(img.channels)
        ]
    )


def _native_nd2(path: Path) -> np.ndarray:
    import nd2

    with nd2.ND2File(str(path)) as f:
        return _canonical(f.asarray(), "".join(f.sizes))


def _native_czi(path: Path) -> np.ndarray:
    import czifile

    with czifile.CziFile(str(path)) as czi:
        return _canonical(czi.asarray(), next(iter(czi.scenes.values())).axes)


def _native_tiff(path: Path) -> np.ndarray:
    with tifffile.TiffFile(str(path)) as tif:
        return _canonical(tif.series[0].asarray(), tif.series[0].axes)


def _checksums(arr: np.ndarray) -> list[str]:
    return [hashlib.sha1(np.ascontiguousarray(ch).tobytes()).hexdigest() for ch in arr]


SCALE_ND2 = 0.34285714285714286
CASES = [
    # name, native reader, shape, voxel_size_um, voxel_size_source, channel_names
    (
        "Spleenx20.oib",  # 2 channels, one plane (was read as 1 channel x 2 slices)
        _native_olympus,
        (2, 1, 640, 640),
        (1.0, 0.994, 0.994),
        "z:default,y:file,x:file",
        ["DAPI", "Alexa Fluor 594"],
    ),
    (
        "partB_OSN_9_Apple.oib",  # plane names 'y2_40X_2_C001Z001.tif' (was 1 of 82 planes)
        _native_olympus,
        (2, 41, 512, 512),
        (1.0, 0.621, 0.621),
        "z:file,y:file,x:file",
        ["Alexa Fluor 488", "Alexa Fluor 647"],
    ),
    (
        "h1-rc-1.oib",  # C x T time series (was read as a z-stack, z 5.0 µm)
        _native_olympus,
        (1, 1, 64, 64),
        (1.0, 0.066, 0.066),
        "z:default,y:file,x:file",
        ["EGFP"],
    ),
    (
        "PR2729_frameOrderCombinedScanTypes.lif",  # negative z length, 4 mosaic tiles
        _native_lif,
        (2, 3, 64, 64),
        (6.227615, 3.9682539682539684, 3.9682539682539684),
        "z:file,y:file,x:file",
        ["Ch1", "Ch2"],
    ),
    (
        "ND2_dims_rgb_t3p2c2z3x64y64.nd2",  # T, P, and RGB samples per channel
        _native_nd2,
        (6, 3, 32, 32),
        (0.01, SCALE_ND2, SCALE_ND2),
        "z:file,y:file,x:file",
        [f"Brightfield[{s}]" for _ in range(2) for s in range(3)],
    ),
    (
        "s_3_t_1_c_3_z_5.czi",  # 3 scenes
        _native_czi,
        (3, 5, 325, 475),
        (1.0, 1.0833333333333333, 1.0833333333333333),
        "z:file,y:file,x:file",
        ["EGFP", "TaRFP", "Bright"],
    ),
    (
        "s_3_t_1_c_3_z_5.ome.tiff",  # the same data converted to OME-TIFF, 3 series
        _native_tiff,
        (3, 5, 325, 475),
        (1.0, 1.0833333333333333, 1.0833333333333333),
        "z:file,y:file,x:file",
        ["EGFP", "TaRFP", "Bright"],
    ),
    (
        "confocal-series.tif",  # ImageJ ZCYX hyperstack
        _native_tiff,
        (2, 25, 400, 400),
        (0.05445500181716341, 0.05445500181716341, 0.05445500181716341),
        "z:file,y:file,x:file",
        ["Ch1", "Ch2"],
    ),
]


@pytest.mark.parametrize("name, native, shape, voxel, source, names", CASES)
def test_real_file_matches_its_native_reader(name, native, shape, voxel, source, names):
    path = _file(name)
    arr, meta = io.load_volume(path)
    assert arr.shape == shape
    assert meta["voxel_size_um"] == pytest.approx(voxel, rel=1e-6)
    assert meta["voxel_size_source"] == source
    assert meta["channel_names"] == names
    assert _checksums(arr) == _checksums(native(path))


def test_czi_and_its_ome_tiff_conversion_load_the_same():
    czi_arr, czi_meta = io.load_volume(_file("s_3_t_1_c_3_z_5.czi"))
    ome_arr, ome_meta = io.load_volume(_file("s_3_t_1_c_3_z_5.ome.tiff"))
    assert _checksums(czi_arr) == _checksums(ome_arr)
    keys = ("voxel_size_um", "voxel_size_source", "channel_names")
    assert {k: czi_meta[k] for k in keys} == {k: ome_meta[k] for k in keys}
