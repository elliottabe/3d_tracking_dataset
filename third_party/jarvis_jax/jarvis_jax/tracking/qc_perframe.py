# jarvis_jax/cse/qc_perframe.py
"""Per-frame QC metrics feeding the pseudo-label Gate A (soft/hard IoU, marker reproj).

Unlike jarvis_jax.tracking.qc.qc_report (which bundles median metrics across all
frames into one summary dict), this returns one value PER FRAME so a
downstream gate can threshold/filter frame-by-frame.

NOTE on real return types (verified against jarvis_jax/cse/qc.py):
  * mesh_mask_iou_report(rt, mesh_mm, masks_by_cam) -> dict with
    {"hard": {cam: iou}, "soft": {cam: iou}, "hard_mean": float, "soft_mean": float}
    i.e. per-camera dicts under "hard"/"soft" -- matches the aggregation below.
  * per_camera_reproj_error(rt, kp3d_mm, kp2d_by_cam, vis_by_cam) -> dict
    {cam: median_px_error}, NOT a flat per-keypoint list/array. `np.median`
    of the dict itself would raise; we take the median of `.values()`.

Both of those are exactly what qc.qc_report also computes per frame, so they
live in ``qc.frame_metrics`` and a caller writing BOTH artifacts (Stage E of
scripts/run_bout.py) computes them once and passes them to both.
"""
from __future__ import annotations
import numpy as np
from jarvis_jax.tracking.qc import frame_metrics as _frame_metrics_fn
from jarvis_jax.tracking.qc import _row_median


def per_frame_qc(rt, *, mesh_by_frame, kp3d_by_frame, kp2d_by_frame,
                 vis_by_frame, masks_by_frame, frame_metrics=None):
    """Per-frame soft/hard mesh-vs-mask IoU, marker reprojection error, n_cams.

    Returns dict of (T,) float arrays: "soft_iou", "hard_iou", "reproj_px",
    "n_cams". NaN where a frame has no usable camera for that metric.

    ``frame_metrics``: an already-computed ``qc.frame_metrics(...)`` for these
    same inputs (see that function). None (default) computes it here, so
    standalone callers are unchanged.
    """
    T = len(mesh_by_frame)
    fm = frame_metrics
    if fm is None:
        fm = _frame_metrics_fn(rt, kp3d_by_frame=kp3d_by_frame,
                               mesh_by_frame=mesh_by_frame,
                               kp2d_by_frame=kp2d_by_frame,
                               vis_by_frame=vis_by_frame,
                               masks_by_frame=masks_by_frame)
    # median over the cameras that HAVE the metric this frame -- identical to
    # the old np.median over the per-camera dict's values.
    hard = _row_median(fm["hard_iou"])
    soft = _row_median(fm["soft_iou"])
    reproj = _row_median(fm["reproj_px"])
    ncam = fm["mask_present"].sum(axis=1)
    return {"soft_iou": soft, "hard_iou": hard, "reproj_px": reproj,
            "n_cams": ncam.astype(np.float32)}
