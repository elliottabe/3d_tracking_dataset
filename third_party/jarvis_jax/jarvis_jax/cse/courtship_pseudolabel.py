"""Silhouette-mesh pseudo-labels: reproject FK'd sites -> 2D, gate per-frame + per-kp."""
from dataclasses import dataclass
import numpy as np


@dataclass
class GateCfg:
    tau_iou: float = 0.05        # per-frame soft-IoU floor (sparse-splat metric)
    tau_cont: float = 8.0        # per-frame containment residual ceiling (px)
    tau_reproj: float = 30.0     # per-frame marker reproj ceiling (px)
    consensus_px: float = 25.0   # Gate B: max mesh<->detector distance (px)
    tau_conf: float = 0.5        # Gate B: min detector confidence


def reproject_sites(rt, kp3d_mm):
    """kp3d_mm (T,K,3) world-mm -> mesh2d (T,C,K,2) full-px (NaN where site NaN)."""
    T, K, _ = kp3d_mm.shape
    C = rt.num_cameras
    out = np.full((T, K, C, 2), np.nan, np.float32)
    for t in range(T):
        for k in range(K):
            X = kp3d_mm[t, k]
            if np.isfinite(X).all():
                out[t, k] = np.asarray(rt.reproject_point(X))   # (C,2)
    return np.transpose(out, (0, 2, 1, 3))                       # (T,C,K,2)


def gate_pseudolabels(mesh2d, det_kp2d, det_conf, qc_pf, masks_valid, *, cfg):
    """mesh2d (T,C,K,2), det_kp2d (T,C,K,2), det_conf (T,C,K), qc_pf dict of (T,),
    masks_valid (T,C). Returns labels (T,C,K,3) [x,y,vis] and frame_keep (T,)."""
    T, C, K, _ = mesh2d.shape
    soft = qc_pf["soft_iou"]; reproj = qc_pf["reproj_px"]
    frame_keep = (np.nan_to_num(soft, nan=-1.0) >= cfg.tau_iou) \
        & (np.nan_to_num(reproj, nan=1e9) <= cfg.tau_reproj)
    if "cont_px" in qc_pf:                        # containment residual is optional
        frame_keep &= (np.nan_to_num(qc_pf["cont_px"], nan=1e9) <= cfg.tau_cont)
    labels = np.zeros((T, C, K, 3), np.float32)
    labels[..., :2] = mesh2d
    finite = np.isfinite(mesh2d).all(-1)                                # (T,C,K)
    dist = np.linalg.norm(mesh2d - det_kp2d, axis=-1)                   # (T,C,K)
    consensus = np.isfinite(dist) & (dist <= cfg.consensus_px) & (det_conf >= cfg.tau_conf)
    vis = (finite & consensus & masks_valid[:, :, None]
           & frame_keep[:, None, None])
    labels[..., 2] = vis.astype(np.float32)
    xy = labels[..., :2]
    xy[~np.isfinite(xy)] = 0.0                                         # scrub NaN coords (vis already 0)
    return labels, frame_keep
