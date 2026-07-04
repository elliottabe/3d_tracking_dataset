"""Silhouette-mesh pseudo-labels: reproject FK'd sites -> 2D, gate per-frame + per-kp."""
from dataclasses import dataclass
import numpy as np

from jarvis_jax.cse.build_pseudolabel_dataset import bbox_from_mask


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


def records_for_bout(labels, masks, frames_iter, cam_names, rec_tag, start_frame):
    """labels (T,C,K,3) [x,y,vis], masks (T,C,H,W) bool, frames_iter yields
    (t, imgs (C,H,W,3) u8). Returns one record dict per (t,c) with >=1 visible
    keypoint, in the format `write_pseudolabel_coco` consumes."""
    T, C, K, _ = labels.shape
    recs = []
    for t, imgs in frames_iter:
        for c in range(C):
            if not (labels[t, c, :, 2] > 0).any():
                continue
            m = np.asarray(masks[t, c], bool)
            rgb = np.asarray(imgs[c])
            recs.append({
                "file_name": f"{rec_tag}/{cam_names[c]}/Frame_{start_frame + t}.jpg",
                "img_w": rgb.shape[1], "img_h": rgb.shape[0],
                "rgb": rgb, "mask": m,
                "keypoints": labels[t, c].astype(np.float32),
                "bbox": bbox_from_mask(m)})
    return recs
