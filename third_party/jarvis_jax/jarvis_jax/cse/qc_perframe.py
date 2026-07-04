# jarvis_jax/cse/qc_perframe.py
"""Per-frame QC metrics feeding the pseudo-label Gate A (soft/hard IoU, marker reproj).

Unlike jarvis_jax.cse.qc.qc_report (which bundles median metrics across all
frames into one summary dict), this returns one value PER FRAME so a
downstream gate can threshold/filter frame-by-frame.

NOTE on real return types (verified against jarvis_jax/cse/qc.py):
  * silhouette_iou_report(rt, mesh_mm, masks_by_cam) -> dict with
    {"hard": {cam: iou}, "soft": {cam: iou}, "hard_mean": float, "soft_mean": float}
    i.e. per-camera dicts under "hard"/"soft" -- matches the aggregation below.
  * per_camera_reproj_error(rt, kp3d_mm, kp2d_by_cam, vis_by_cam) -> dict
    {cam: median_px_error}, NOT a flat per-keypoint list/array. `np.median`
    of the dict itself would raise; we take the median of `.values()`.
"""
from __future__ import annotations
import numpy as np
from jarvis_jax.cse.qc import silhouette_iou_report, per_camera_reproj_error


def per_frame_qc(rt, *, mesh_by_frame, kp3d_by_frame, kp2d_by_frame,
                 vis_by_frame, masks_by_frame):
    """Per-frame soft/hard silhouette IoU, marker reprojection error, n_cams.

    Returns dict of (T,) float arrays: "soft_iou", "hard_iou", "reproj_px",
    "n_cams". NaN where a frame has no usable camera for that metric.
    """
    T = len(mesh_by_frame)
    soft = np.full(T, np.nan); hard = np.full(T, np.nan)
    reproj = np.full(T, np.nan); ncam = np.zeros(T, int)
    for t in range(T):
        masks_c = {c: m for c, m in masks_by_frame[t].items() if m is not None}
        ncam[t] = len(masks_c)
        if masks_c:
            rep = silhouette_iou_report(rt, mesh_by_frame[t], masks_c)  # {"hard":{c:},"soft":{c:}}
            hv = [v for v in rep["hard"].values() if v is not None]
            sv = [v for v in rep["soft"].values() if v is not None]
            if hv: hard[t] = float(np.median(hv))
            if sv: soft[t] = float(np.median(sv))
        # per_camera_reproj_error returns {cam: median_px_error}, not a flat
        # per-keypoint list -- median over cameras' per-cam medians.
        errs = per_camera_reproj_error(rt, kp3d_by_frame[t],
                                       kp2d_by_frame[t], vis_by_frame[t])
        if errs:
            reproj[t] = float(np.median(list(errs.values())))
    return {"soft_iou": soft, "hard_iou": hard, "reproj_px": reproj,
            "n_cams": ncam.astype(np.float32)}
