"""DLT triangulation of per-camera 2-D keypoints into 3-D mm keypoints + conf.

Geometric (no learned lift): per keypoint, use the cameras where the 2-D peak
confidence >= conf_thresh (>=2 required) and solve the DLT via
center3d.triangulate_dlt_batched. Keypoints with <2 confident views -> NaN
(down-weighted downstream by conf3d=0).
"""
from __future__ import annotations
import numpy as np
from jarvis_jax.geometry.center3d import triangulate_dlt_batched


def triangulate_keypoints(kp2d, conf, cam_mats, *, conf_thresh: float = 0.3):
    """kp2d (T,C,K,2), conf (T,C,K), cam_mats (C,4,3) -> (kp3d (T,K,3), conf3d (T,K))."""
    kp2d = np.asarray(kp2d, np.float32); conf = np.asarray(conf, np.float32)
    cam_mats = np.asarray(cam_mats, np.float32)
    T, C, K, _ = kp2d.shape
    valid = conf >= conf_thresh                                   # (T,C,K)
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
