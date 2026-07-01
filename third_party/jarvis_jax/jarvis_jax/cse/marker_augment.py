"""Inject silhouette wing-tips as high-confidence 3-D targets for the existing
distal wing markers (WingL/R_V12/V13), raising their weight in kps_to_opt.
Reuses the STAC marker_cost unchanged: it fits site positions to kp_data with
per-coordinate weight kps_to_opt.
"""
from __future__ import annotations
import numpy as np

# coco keypoint indices of the distal wing markers (WingL_V12=7, WingL_V13=8, WingR_V12=29, WingR_V13=30)
WING_MARKER_IDS = {"left": (7, 8), "right": (29, 30)}


def augment_wing_markers(kp_data, kps_to_opt, tips, *, wing_weight=5.0, min_cams=2):
    kp2 = np.array(kp_data, dtype=np.float64, copy=True)          # (T, n_kp, 3)
    w2 = np.array(kps_to_opt, dtype=np.float64, copy=True)        # (n_kp*3,)
    used_ids = set()
    for t, frame in enumerate(tips):
        for side, ids in WING_MARKER_IDS.items():
            entry = frame.get(side) if frame else None
            if entry is None:
                continue
            tip3d, ncam = entry
            if ncam < min_cams:
                continue
            for idx in ids:
                kp2[t, idx, :] = tip3d
                used_ids.add(idx)
    for idx in used_ids:
        w2[idx * 3:idx * 3 + 3] = wing_weight
    return kp2, w2
