"""Deep-learning label predictions for the benchmark producers (BBBC024 crowding,
BBBC039 baselines, repeat runs and timing; see DATA_MANIFEST.md).

Runs on a GPU/CPU node with Cellpose 3, Omnipose and StarDist installed (not
fluorostats dependencies), writes one label TIFF per input plus ``versions.json``
(library versions, model name, the weights file and its SHA-256, the command).
A model that fails to load or run stops the script: nothing is substituted. Scoring
happens locally in the producer with ``fluorostats.validate.instance_f1``, so the
deep-learning lines use the same scorer as the fluorostats lines.

    python bbbc024_dl_infer.py {cellpose,stardist,omnipose} {3d,2d} IN_DIR OUT_DIR [PROB] [NORM]

PROB (StarDist only): detection probability threshold, ``default`` for the
model's own (0.479). NORM (StarDist 3d only): ``volume`` (percentiles over the
whole stack, the default) or ``plane`` (each plane on its own).

3d: each ``*.tif`` in IN_DIR is a (z, y, x) volume.
  - Cellpose: the pretrained ``nuclei`` model in its own 3D mode (``do_3D``), with
    the diameter estimated once on the volume's mid plane by Cellpose's size model
    (the model's default diameter if that fails; each value is recorded).
  - StarDist: ``2D_versatile_fluo`` on every z plane; the output is the stack of
    per-plane labels, unstitched. The producer links planes into 3D objects
    (``producers/zstitch.py``), so stitching settings can be tuned without
    re-running the model.
2d: each ``*.tif`` is one plane; the models run as published (Cellpose ``nuclei``
  with the diameter estimated per image, StarDist default thresholds, Omnipose
  ``cyto2_omni`` with ``omni=True``). Omnipose runs in 2d only.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import tifffile

NORM_PCT = (1.0, 99.8)  # StarDist's documented normalisation percentiles


def weights_record(path) -> dict:
    """The weights file a model loaded, with its SHA-256 (fails if it is missing).

    ``path`` may be a list of paths (Omnipose) or one path (Cellpose 3)."""
    path = str(path if isinstance(path, (str, os.PathLike)) else path[0])
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return {"weights": path, "weights_sha256": digest}


def cellpose_runner(mode: str):
    import torch
    from cellpose import models, version

    # BBBC024_DL_CPU=1 forces CPU (some ROCm builds fail in hipBLASLt; setting
    # TORCH_BLAS_PREFER_HIPBLASLT=0 avoids that on the GPU)
    gpu = torch.cuda.is_available() and os.environ.get("BBBC024_DL_CPU") != "1"
    model = models.Cellpose(gpu=gpu, model_type="nuclei")
    diameters: dict = {}   # 3d: the diameter used for each volume, in file order

    def run(img: np.ndarray) -> np.ndarray:
        if mode == "2d":
            return model.eval(img, channels=[0, 0], diameter=None)[0]
        diam = float(model.sz.eval(img[img.shape[0] // 2], channels=[0, 0])[0])
        if not np.isfinite(diam) or diam <= 0:   # size model found nothing: model default
            diam = float(model.diam_mean)
        diameters[len(diameters)] = diam
        return model.eval(img, channels=[0, 0], diameter=diam, do_3D=True)[0]

    return run, {"tool": "cellpose", "cellpose": str(version), "torch": torch.__version__,
                 "model": "nuclei", "gpu": gpu, **weights_record(model.cp.pretrained_model),
                 "diameters_3d": diameters}


def omnipose_runner(mode: str):
    import importlib.metadata as md

    import torch
    from cellpose_omni import models

    if mode != "2d":
        raise SystemExit("omnipose runs in 2d mode only")
    gpu = torch.cuda.is_available() and os.environ.get("BBBC024_DL_CPU") != "1"
    model = models.CellposeModel(gpu=gpu, model_type="cyto2_omni")

    def run(img: np.ndarray) -> np.ndarray:
        return model.eval(img, channels=[0, 0], omni=True)[0]

    return run, {"tool": "omnipose", "omnipose": md.version("omnipose"),
                 "cellpose-omni": md.version("cellpose-omni"), "torch": torch.__version__,
                 "model": "cyto2_omni", "omni": True, "gpu": gpu,
                 **weights_record(model.pretrained_model)}


def stardist_runner(mode: str, prob: str = "default", norm: str = "volume"):
    import csbdeep
    import stardist
    import tensorflow as tf
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    weights = weights_record(os.path.join(str(model.logdir), "weights_best.h5"))
    kw = {} if prob == "default" else {"prob_thresh": float(prob)}

    def run(img: np.ndarray) -> np.ndarray:
        if mode == "2d":
            return model.predict_instances(normalize(img, *NORM_PCT), **kw)[0]
        vol = normalize(img, *NORM_PCT) if norm == "volume" else img
        return np.stack([
            model.predict_instances(vol[z] if norm == "volume" else normalize(vol[z], *NORM_PCT),
                                    **kw)[0].astype(np.int32)
            for z in range(vol.shape[0])])

    return run, {"tool": "stardist", "stardist": stardist.__version__,
                 "csbdeep": csbdeep.__version__, "tensorflow": tf.__version__,
                 "model": "2D_versatile_fluo", "normalise_percentiles": list(NORM_PCT),
                 **weights, "prob_thresh": prob, "normalise": norm if mode == "3d" else "image",
                 "stitched": False}


def main() -> None:
    tool, mode, src, dst = sys.argv[1:5]
    extra = sys.argv[5:]
    if tool not in ("cellpose", "stardist", "omnipose") or mode not in ("3d", "2d") or (
            extra[1:] and extra[1] not in ("volume", "plane")):
        raise SystemExit(__doc__)
    os.makedirs(dst, exist_ok=True)
    runners = {"cellpose": cellpose_runner, "omnipose": omnipose_runner}
    run, meta = runners[tool](mode) if tool in runners else stardist_runner(mode, *extra)
    meta["command"] = " ".join(["python", *sys.argv])
    files = sorted(glob.glob(os.path.join(src, "*.tif")))
    seconds = {}
    for f in files:
        img = tifffile.imread(f).astype(np.float32)
        t0 = time.perf_counter()
        lab = np.asarray(run(img)).astype(np.int32)
        seconds[os.path.basename(f)] = round(time.perf_counter() - t0, 2)
        tifffile.imwrite(os.path.join(dst, os.path.basename(f)), lab, compression="zlib")
        print(f"{tool} {mode} {os.path.basename(f)} objects={len(np.unique(lab)) - 1} "
              f"{seconds[os.path.basename(f)]}s", flush=True)
    meta.update({"mode": mode, "n_files": len(files), "python": platform.python_version(),
                 "host": platform.node(), "seconds": seconds})
    with open(os.path.join(dst, "versions.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main()
