"""DLT triangulation of per-camera 2-D keypoints into 3-D mm keypoints + conf.

Geometric (no learned lift): per keypoint, use the cameras where the 2-D peak
confidence >= conf_thresh (>=2 required) and solve the DLT via
center3d.triangulate_dlt_batched. Keypoints with <2 confident views -> NaN
(down-weighted downstream by conf3d=0).
"""
from __future__ import annotations
import numpy as np
from jarvis_jax.geometry.center3d import triangulate_dlt_batched


def view_median_conf(conf):
    """conf (T,C,K) -> (T,C) median peak confidence across keypoints per view.

    The per-view summary the gate in `triangulate_keypoints` thresholds on.
    Median rather than mean so a handful of genuinely occluded keypoints cannot
    drag an otherwise-good view under the threshold.
    """
    conf = np.asarray(conf, np.float32)
    if conf.ndim != 3:
        raise ValueError(f"view_median_conf expects (T,C,K), got {conf.shape}")
    return np.median(conf, axis=2)


def triangulate_keypoints(kp2d, conf, cam_mats, *, conf_thresh: float = 0.3,
                          view_conf_thresh: float | None = None):
    """kp2d (T,C,K,2), conf (T,C,K), cam_mats (C,4,3) -> (kp3d (T,K,3), conf3d (T,K)).

    NOTE: kp2d must be FINITE even where conf < conf_thresh (invalid views are zeroed by
    validity masking inside triangulate_dlt_batched; a NaN/Inf pixel would poison that
    point's SVD via 0*NaN).

    `conf_thresh` gates individual keypoints. `view_conf_thresh` (None = off)
    additionally gates whole (frame, camera) views on their median keypoint
    confidence, dropping every keypoint of a view that falls below it.

    The per-view gate exists because the detector emits all K channels for
    whatever crop it is given, including one containing no fly, and reports
    high PER-KEYPOINT confidence while doing so -- so `conf_thresh` alone
    cannot stop a misplaced crop from entering the DLT and producing a
    confident, wrong 3D point. That is worse than dropping the view: a NaN is
    visibly missing downstream, a plausible wrong point is not. Measured on 273
    real views per class (scripts/calibrate_view_gate.py), per-view medians are
    0.963 with a fly present against 0.379 without, so a threshold of 0.6 keeps
    100% of genuine views while rejecting ~92% of empty ones. It is a strong
    mitigation, not a cure -- ~5% of empty crops still score above 0.9.
    """
    kp2d = np.asarray(kp2d, np.float32); conf = np.asarray(conf, np.float32)
    cam_mats = np.asarray(cam_mats, np.float32)
    T, C, K, _ = kp2d.shape
    valid = conf >= conf_thresh                                   # (T,C,K)
    if view_conf_thresh is not None:
        view_ok = view_median_conf(conf) >= float(view_conf_thresh)   # (T,C)
        valid = valid & view_ok[:, :, None]
    kp3d = np.full((T, K, 3), np.nan, np.float32)
    conf3d = np.zeros((T, K), np.float32)
    # batch over (T*K) points; each has C views
    pts = kp2d.transpose(0, 2, 1, 3).reshape(T * K, C, 2)         # (TK,C,2)
    val = valid.transpose(0, 2, 1).reshape(T * K, C)             # (TK,C)
    cams = np.broadcast_to(cam_mats[None], (T * K, C, 4, 3))
    nvalid = val.sum(1)                                          # (TK,)
    ok = nvalid >= 2
    if ok.any():
        X = np.asarray(triangulate_dlt_batched(pts[ok], cams[ok], val[ok]))   # (n,3)
        finite = np.isfinite(X).all(1)
        idx = np.where(ok)[0][finite]
        flat3d = kp3d.reshape(T * K, 3); flatc = conf3d.reshape(T * K)
        flat3d[idx] = X[finite]
        cmean = (conf.transpose(0, 2, 1).reshape(T * K, C) * val).sum(1) / np.maximum(nvalid, 1)
        flatc[idx] = cmean[idx]
        kp3d = flat3d.reshape(T, K, 3); conf3d = flatc.reshape(T, K)
    return kp3d, conf3d
