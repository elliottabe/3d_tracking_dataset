"""Task 10: backend x fusion-variant benchmark harness (promotion gate).

Runs a fixed set of "bouts" through every requested ``(backend, fusion)``
combination -- dispatched through Task-9's ``load_inference_model(front_end=,
fusion_mode=)`` + ``predict_batch`` -- and reports, per combination:

  * accuracy -- DLT reprojection error (px) of the backend's predicted 3-D
    keypoints against each camera's own 2-D detections. Reuses the existing
    ``ReprojectionTool``/``Camera`` DLT primitive
    (``jarvis_jax/geometry/reprojection_tool.py``, exercised through
    ``rt.reproject_point`` exactly as ``jarvis_jax.tracking.qc.
    per_camera_reproj_error``/``loo_reproj`` already do) rather than inventing
    a new projection convention. That primitive is homogeneous-coordinate
    3x4-matrix-times-point + perspective divide -- the SAME convention as
    ``jarvis_jax/hybridnet/reproject.py``'s DLT matmul (see that module's
    ``camera_matrices`` shape/orientation docstring). We do not call
    ``per_camera_reproj_error`` verbatim because it returns a single
    per-camera median; the benchmark needs the raw per-keypoint errors to
    compute ``p95_reproj_px`` across the whole combo, so
    ``_reproj_errors_for_frame`` below re-walks the same
    reproject_point-per-keypoint loop and keeps every value.
  * throughput -- wall-clock frames/sec of ``predict_batch`` (model loading
    is excluded from the timed region).

No silent caps: any bout that fails to prepare or run for a given
``(backend, fusion)`` combo is logged (``logging.warning`` + printed) and
recorded verbatim in the report's top-level ``"skipped"`` list -- it is never
dropped without a trace. Other bouts continue to run.

Backend dispatch is a small, explicit name -> ``load_inference_model`` kwargs
table (``_BACKEND_LOADER_KWARGS``) so adding a new backend/front-end is a
one-line addition, not a new branch in ``run_benchmark`` itself.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from jarvis_jax.predict.infer_3d import load_inference_model, predict_batch

log = logging.getLogger(__name__)

# backend name -> extra kwargs for load_inference_model (besides fusion_mode,
# which comes from `fusion_variants`, and any ckpt paths / shared kwargs
# supplied via `model_kwargs`). Add new backends here.
_BACKEND_LOADER_KWARGS = {
    "hybridnet_jax": {"front_end": "vitpose"},
    "hybridnet_jax_efficienttrack": {"front_end": "efficienttrack"},
}


def _reproj_errors_for_frame(rt, kp3d_j3, kp2d_by_cam, vis_by_cam):
    """Raw per-(camera, visible keypoint) DLT reprojection errors (px) for one
    frame's predicted 3-D keypoints.

    Reuses ``rt.reproject_point`` -- the same ReprojectionTool DLT primitive
    used by ``jarvis_jax.tracking.qc.per_camera_reproj_error``/``loo_reproj``
    -- to reproject the backend's full-camera 3-D solution into each camera
    and compare against that camera's own 2-D detection (the
    "reprojection of the full-camera solution" variant of the LOO/reproj
    metric described in task-10-brief.md, chosen because the backend's
    kp3d is produced by joint multi-view volumetric fusion, not per-camera
    triangulation, so there is no cheaper way to hold a camera fully out of
    the forward pass itself).
    """
    kp3d_j3 = np.asarray(kp3d_j3, dtype=float)
    n_kp = len(kp3d_j3)
    errs = []
    for c, kp2d in kp2d_by_cam.items():
        vis = np.asarray(vis_by_cam.get(c, np.ones(n_kp, bool)), bool)
        for j in range(n_kp):
            if not vis[j]:
                continue
            uv = rt.reproject_point(kp3d_j3[j])[c]
            errs.append(float(np.linalg.norm(uv - np.asarray(kp2d[j], float))))
    return errs


def _stack_frame_field(frames, key):
    return np.stack([f[key] for f in frames], axis=0)


def _build_batch_inputs(frames, *, fusion):
    """Stack a bout's per-frame fields into predict_batch's batched inputs.

    Raises (rather than silently degrading) if a frame is missing a field
    required for the requested combination -- e.g. `masks` when
    `fusion != 'none'` -- so the caller's skip-and-log path fires instead of
    quietly running mask-fusion without masks.
    """
    crops4 = _stack_frame_field(frames, "crops4")
    centerHM = _stack_frame_field(frames, "centerHM")
    cameraMatrices = _stack_frame_field(frames, "cameraMatrices")
    has_masks = all("masks" in f for f in frames)
    if fusion != "none" and not has_masks:
        raise ValueError(
            f"fusion_mode={fusion!r} requires per-frame 'masks' but at least "
            "one frame is missing it"
        )
    masks = _stack_frame_field(frames, "masks") if has_masks else None
    return crops4, centerHM, cameraMatrices, masks


def run_benchmark(bouts, backends, fusion_variants, *, out_json,
                   backend_loader_kwargs=None, model_kwargs=None):
    """Benchmark every ``(backend, fusion)`` pair in ``backends`` x
    ``fusion_variants`` on the fixed ``bouts``, write a JSON report to
    ``out_json``, and return the same dict.

    Args:
        bouts: list of bout dicts, each with:
            ``bout_id`` (str, optional -- used only for logging/reports),
            ``rt`` (a ``ReprojectionTool``-like object exposing
                ``reproject_point(X3d) -> (num_cam, 2)``),
            ``frames`` (list of per-frame dicts with ``crops4``, ``centerHM``,
                ``cameraMatrices`` -- the ``predict_batch`` per-frame inputs
                -- plus ``kp2d_by_cam``/``vis_by_cam`` ground-truth 2-D
                detections/visibility used for the accuracy metric, and an
                optional ``masks`` field required when a fusion variant
                other than ``'none'`` is requested).
        backends: list of backend names (keys of ``_BACKEND_LOADER_KWARGS``,
            or ``backend_loader_kwargs`` if supplied).
        fusion_variants: list of ``fusion_mode`` strings forwarded to
            ``load_inference_model`` (e.g. ``'none'``, ``'carve'``).
        out_json: path to write the JSON report to.
        backend_loader_kwargs: optional ``{backend_name: kwargs}`` overrides/
            additions to ``_BACKEND_LOADER_KWARGS`` (e.g. for a new backend
            not baked into this module).
        model_kwargs: optional kwargs forwarded to every
            ``load_inference_model`` call (e.g. ``vitpose_ckpt``,
            ``v2v_final_dir``, ``efficienttrack_ckpt``, ``num_keypoints``) --
            shared across all backends/fusion variants in this run.

    Returns:
        dict keyed ``f"{backend}:{fusion}"`` -> ``{"mean_reproj_px",
        "p95_reproj_px", "fps", "n_frames"}``, plus a top-level ``"skipped"``
        list of ``{"bout_id", "backend", "fusion", "error"}`` records for any
        bout that could not be run for that combination.
    """
    loader_kwargs_by_backend = dict(_BACKEND_LOADER_KWARGS)
    if backend_loader_kwargs:
        loader_kwargs_by_backend.update(backend_loader_kwargs)
    model_kwargs = dict(model_kwargs or {})

    report: dict = {"skipped": []}

    for backend in backends:
        if backend not in loader_kwargs_by_backend:
            raise ValueError(
                f"Unknown backend {backend!r}; known backends: "
                f"{sorted(loader_kwargs_by_backend)}"
            )
        base_kwargs = dict(loader_kwargs_by_backend[backend])

        for fusion in fusion_variants:
            key = f"{backend}:{fusion}"

            loader_kwargs = dict(model_kwargs)
            loader_kwargs.update(base_kwargs)
            loader_kwargs["fusion_mode"] = fusion
            vitpose_ckpt = loader_kwargs.pop("vitpose_ckpt", None)
            v2v_final_dir = loader_kwargs.pop("v2v_final_dir", None)
            model = load_inference_model(vitpose_ckpt, v2v_final_dir, **loader_kwargs)

            all_errs = []
            n_frames = 0
            total_time = 0.0

            for bout in bouts:
                bout_id = bout.get("bout_id", "<unknown>")
                try:
                    rt = bout["rt"]
                    frames = bout["frames"]
                    if not frames:
                        raise ValueError("bout has no frames")

                    crops4, centerHM, cameraMatrices, masks = _build_batch_inputs(
                        frames, fusion=fusion)

                    t0 = time.perf_counter()
                    kp3d, _conf, _center3D = predict_batch(
                        model, crops4, centerHM, cameraMatrices, masks=masks)
                    dt = time.perf_counter() - t0

                    for i, f in enumerate(frames):
                        all_errs.extend(_reproj_errors_for_frame(
                            rt, kp3d[i], f["kp2d_by_cam"], f["vis_by_cam"]))
                    n_frames += len(frames)
                    total_time += dt
                except Exception as e:  # noqa: BLE001 -- deliberately broad: any
                    # per-bout failure must be logged + recorded, never silently
                    # swallowed or allowed to abort the whole benchmark run.
                    msg = f"benchmark_backends: skipping bout {bout_id!r} for {key}: {e!r}"
                    log.warning(msg)
                    print(msg)
                    report["skipped"].append({
                        "bout_id": bout_id, "backend": backend, "fusion": fusion,
                        "error": repr(e),
                    })

            if all_errs:
                arr = np.asarray(all_errs, dtype=float)
                mean_px = float(np.mean(arr))
                p95_px = float(np.percentile(arr, 95))
            else:
                mean_px = float("nan")
                p95_px = float("nan")
            fps = (n_frames / total_time) if total_time > 0 else float("nan")

            report[key] = {
                "mean_reproj_px": mean_px,
                "p95_reproj_px": p95_px,
                "fps": fps,
                "n_frames": n_frames,
            }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)
    return report
