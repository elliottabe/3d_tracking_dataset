"""Per-recording affine-camera bundle adjustment (jaxls LM).

Factors each camera into (K2, R, t) (affine_camera), then jointly optimizes
per-camera (R, t) [optionally K2] and one 3-D point per (frame, keypoint) to
minimize robust reprojection error with a soft prior to the factory cameras.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool  # noqa: F401  (used by callers)


@dataclass
class Observations:
    cam_idx: np.ndarray       # (n_obs,) int32
    point_idx: np.ndarray     # (n_obs,) int32
    uv: np.ndarray            # (n_obs, 2) float64
    n_cams: int
    n_points: int


def assemble_observations(kp2d: np.ndarray, min_cams: int = 2) -> Observations:
    """kp2d[f,c,k] = (x, y, visible). One point per (f,k) seen by >= min_cams."""
    kp2d = np.asarray(kp2d)
    F, C, K, _ = kp2d.shape
    cam_idx, point_idx, uv = [], [], []
    pid = 0
    for f in range(F):
        for k in range(K):
            seen = [c for c in range(C) if kp2d[f, c, k, 2] > 0]
            if len(seen) < min_cams:
                continue
            for c in seen:
                cam_idx.append(c); point_idx.append(pid); uv.append(kp2d[f, c, k, :2])
            pid += 1
    return Observations(
        cam_idx=np.asarray(cam_idx, np.int32),
        point_idx=np.asarray(point_idx, np.int32),
        uv=np.asarray(uv, np.float64).reshape(-1, 2),
        n_cams=C, n_points=pid,
    )


def initial_points(obs: Observations, cam_mats) -> np.ndarray:
    """DLT-triangulate each point from the cameras that observe it."""
    X = np.zeros((obs.n_points, 3), np.float64)
    for pid in range(obs.n_points):
        m = obs.point_idx == pid
        cams = obs.cam_idx[m]; uvs = obs.uv[m]
        # Build (2n x 4) DLT system: uv * P[2] - P[0:2].
        A = np.zeros((2 * len(cams), 4))
        for i, (c, uvp) in enumerate(zip(cams, uvs)):
            P = np.asarray(cam_mats[c])
            A[2 * i:2 * i + 2] = uvp.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]
        _, _, Vh = np.linalg.svd(A)
        Xh = Vh[-1]
        X[pid] = (Xh / Xh[3])[:3]
    return X
