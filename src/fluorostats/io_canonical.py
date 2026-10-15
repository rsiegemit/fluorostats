"""Canonical array layout for loaded microscopy files.

Every loader hands its reader's array to ``canonicalize_volume`` -> ``(C, Z, Y, X)``
or ``canonicalize_image`` -> ``(C, Y, X)``. When the reader names its axes
(tifffile, nd2, czifile letters) the array is reordered by name; otherwise the
order is guessed from the shape, with a warning wherever the guess is ambiguous.
Both fill ``meta["channel_names"]`` up to the channel count.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["canonicalize_volume", "canonicalize_image", "squeeze_singleton_dims"]


def squeeze_singleton_dims(arr: np.ndarray) -> np.ndarray:
    """Remove singleton dimensions (common in CZI: B, S, T axes)."""
    while arr.ndim > 4 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim > 4:
        warnings.warn(f"array of shape {arr.shape} has more than 4 non-singleton axes; keeping "
                      "only the first index of each extra leading axis (e.g. time point, scene).",
                      RuntimeWarning, stacklevel=4)
    while arr.ndim > 4:
        arr = arr[0]
    return arr


# Axis letters kept in a canonical array (tifffile convention); S = colour samples,
# which become extra channels.
_CANONICAL_AXES = "CSZYX"


def _by_axis_names(arr: np.ndarray, axes: str, index_axes: dict[str, str]) -> np.ndarray | None:
    """(C, Z, Y, X) from the reader's axis letters (C, S, Z, Y, X), or None when
    the letters do not fit the array or a non-singleton axis is neither one of
    them nor an ``index_axes`` letter (then only the shape can be used).

    Colour samples become extra channels (channel-major). Each ``index_axes``
    axis (time point, position, ...) is reduced to its first index, announced.
    """
    sizes = dict(zip(axes, arr.shape))
    if len(axes) != arr.ndim or len(sizes) != arr.ndim or not {"Y", "X"} <= set(axes):
        return None
    if any(n > 1 and a not in _CANONICAL_AXES and a not in index_axes for a, n in sizes.items()):
        return None
    for a, n in sizes.items():
        if a in index_axes and n > 1:
            warnings.warn(f"keeping only {index_axes[a]} 0 of {n} (axis {a!r} of the file).",
                          RuntimeWarning, stacklevel=4)
    arr = arr[tuple(slice(None) if a in _CANONICAL_AXES else 0 for a in axes)]
    order = [a for a in "CSZ" if a not in sizes] + [a for a in axes if a in _CANONICAL_AXES]
    arr = arr[(np.newaxis,) * (len(order) - arr.ndim)]
    arr = arr.transpose([order.index(a) for a in _CANONICAL_AXES])
    return arr.reshape(-1, *arr.shape[2:])


# tifffile's letters for an axis it cannot name (sequence, image index).
_UNLABELLED_AXES = "QI"


def _unlabelled_as_z(axes: str, shape: tuple[int, ...], index_axes: dict[str, str]) -> str:
    """``axes`` with its one unlabelled axis renamed Z (announced), when that axis
    is the only one, is longer than 1, and Y and X are labelled but Z is not;
    otherwise ``axes`` unchanged (two unlabelled axes are left to the shape rule)."""
    unlabelled = [i for i, a in enumerate(axes) if a in _UNLABELLED_AXES and a not in index_axes]
    if (len(unlabelled) != 1 or len(axes) != len(shape) or shape[unlabelled[0]] == 1
            or "Z" in axes or not {"Y", "X"} <= set(axes)):
        return axes
    i = unlabelled[0]
    warnings.warn(f"the file labels axes {axes!r}; assuming the unlabelled axis of length "
                  f"{shape[i]} is z. If it is not, save the file with all axes named.",
                  RuntimeWarning, stacklevel=4)
    return axes[:i] + "Z" + axes[i + 1:]


# More channels than any realistic fluorescence acquisition: a labelled file with
# this many channels and one z-slice is most likely a mislabelled z-stack (e.g.
# tifffile's imagej=True writes a 3D array without axes as channels=N).
_MAX_PLAUSIBLE_CHANNELS = 4


def canonicalize_volume(arr: np.ndarray, meta: dict, axes: str | None = None,
                        index_axes: dict[str, str] | None = None) -> np.ndarray:
    """Ensure array is (C, Z, Y, X): by axis name when the reader labels its axes
    (``axes``), otherwise by shape."""
    index_axes = index_axes or {}
    axes = _unlabelled_as_z(axes, arr.shape, index_axes) if axes else None
    named = _by_axis_names(arr, axes, index_axes) if axes else None
    arr = named if named is not None else _checked(_volume_by_shape(arr))
    if named is not None and arr.shape[1] == 1 and arr.shape[0] > _MAX_PLAUSIBLE_CHANNELS:
        warnings.warn(f"the file declares {arr.shape[0]} channels and one z-slice; if it is a "
                      f"z-stack, re-save it with axes 'ZYX' (or slices={arr.shape[0]}).",
                      RuntimeWarning, stacklevel=4)
    n_ch = arr.shape[0]
    if len(meta["channel_names"]) < n_ch:
        for i in range(len(meta["channel_names"]), n_ch):
            meta["channel_names"].append(f"Ch{i + 1}")
    return arr


# The shape rule's guess is refused beyond these: no fluorescence acquisition has
# more channels, and an image this narrow is a misread colour axis.
_MAX_GUESSED_CHANNELS = 10
_MIN_GUESSED_WIDTH = 5


def _checked(arr: np.ndarray) -> np.ndarray:
    """The shape rule's (C, Z, Y, X) ``arr``, or a ValueError when it is implausible."""
    n_c, n_x = arr.shape[0], arr.shape[-1]
    if n_c > _MAX_GUESSED_CHANNELS or n_x < _MIN_GUESSED_WIDTH:
        raise ValueError(
            f"the file names no usable axes, and its shape read as (C, Z, Y, X) = {arr.shape} "
            f"gives {n_c} channels and X = {n_x}, which is not a plausible layout. Save it "
            "with its axes named (OME-TIFF, or an ImageJ hyperstack).")
    return arr


def _volume_by_shape(arr: np.ndarray) -> np.ndarray:
    """(C, Z, Y, X) guessed from the array shape alone (no axis labels)."""
    ndim = arr.ndim
    if ndim == 3:
        arr = arr[np.newaxis]
    elif ndim == 4:
        # Axis-order heuristic: a leading axis longer than any channel count is far more
        # likely to be Z than C, so the array is taken as channel-LAST. Either way the
        # layout is a guess, and it is announced.
        if arr.shape[0] > _MAX_GUESSED_CHANNELS:
            warnings.warn(f"4D array leading axis = {arr.shape[0]} (> {_MAX_GUESSED_CHANNELS}); "
                          "assuming channel-last and transposing to (C, Z, Y, X).",
                          RuntimeWarning, stacklevel=3)
            arr = np.moveaxis(arr, -1, 0)
        elif arr.shape[0] > 1:
            warnings.warn(f"4D array of shape {arr.shape} has no axis labels; read as "
                          f"(C, Z, Y, X) with {arr.shape[0]} channels. If it is (Z, Y, X, C) "
                          "or (Z, C, Y, X), save it with its axes named (OME-TIFF, or an "
                          "ImageJ hyperstack) to avoid the guess.", RuntimeWarning, stacklevel=3)
    elif ndim == 5:
        if arr.shape[0] > 1:
            warnings.warn(f"5D array of shape {arr.shape}: keeping only the first of "
                          f"{arr.shape[0]} entries along the leading axis (e.g. time point).",
                          RuntimeWarning, stacklevel=3)
        arr = arr[0]
    elif ndim == 2:
        arr = arr[np.newaxis, np.newaxis]
    elif ndim > 5:
        arr = squeeze_singleton_dims(arr)
        if arr.ndim != 4:  # pragma: no cover - squeeze always yields 4D
            return _volume_by_shape(arr)
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} with {ndim} dims")
    return arr


def canonicalize_image(arr: np.ndarray, meta: dict, axes: str | None = None,
                       index_axes: dict[str, str] | None = None) -> np.ndarray:
    """Ensure 2D image is (C, Y, X): by axis name when the reader labels a single
    plane (``axes``), otherwise by shape (a labelled z-stack keeps the shape rule)."""
    named = _by_axis_names(arr, axes, index_axes or {}) if axes else None
    if named is not None and named.shape[1] == 1:
        arr = named[:, 0]
    elif arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3:
        if arr.shape[2] <= 4:
            arr = np.moveaxis(arr, -1, 0)
        elif arr.shape[0] > 4:
            warnings.warn(f"3D array of shape {arr.shape} read as a 2D image: its {arr.shape[0]} "
                          "planes are treated as channels. If it is a z-stack, load it with "
                          "load_volume / quant3d instead.", RuntimeWarning, stacklevel=3)
    else:
        raise ValueError(f"Unexpected 2D image shape: {arr.shape}")

    n_ch = arr.shape[0]
    if len(meta["channel_names"]) < n_ch:
        for i in range(len(meta["channel_names"]), n_ch):
            meta["channel_names"].append(f"Ch{i + 1}")

    return arr
