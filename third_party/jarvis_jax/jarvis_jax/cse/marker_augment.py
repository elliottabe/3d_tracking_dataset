"""Inject silhouette wing-tips as high-confidence 3-D targets for the existing
distal wing markers (WingL/R_V12/V13), raising their weight in kps_to_opt.
Reuses the STAC marker_cost unchanged: it fits site positions to kp_data with
per-coordinate weight kps_to_opt.

Review finding (Task 4): unconditionally overriding a PRESENT wing marker
(i.e. one with real GT/tracked kp_data) with the single mask-edge SAM tip at
a high weight degrades the whole-body solve (reproj 2.13->8.42px on GT data)
-- the tip is a single, noisier target standing in for two independent
markers (V12+V13), and dragging both onto it at high weight distorts the fit
even when the original markers were already good. ``only_missing`` (default
True) gates the injection so it only fills in markers that are actually
missing (all-NaN) for that frame, leaving present markers/weights untouched
-- i.e. augmentation is then a no-op on full GT data and only helps in the
ablation where wing keypoints are withheld (set to NaN).
"""
from __future__ import annotations
import numpy as np

# coco keypoint indices of the distal wing markers (WingL_V12=7, WingL_V13=8, WingR_V12=29, WingR_V13=30)
WING_MARKER_IDS = {"left": (7, 8), "right": (29, 30)}


def augment_wing_markers(kp_data, kps_to_opt, tips, *, wing_weight=0.5, min_cams=2, only_missing=True):
    """Fill/override distal wing markers with the silhouette-triangulated tip.

    Args:
        kp_data: (T, n_kp, 3) keypoint array (coco order).
        kps_to_opt: (n_kp*3,) per-coordinate optimization weight.
        tips: length-T list of per-frame ``{"left": (xyz, ncam) or None,
            "right": ...}`` silhouette wing-tip triangulations.
        wing_weight: weight assigned to a wing marker's 3 coords once it is
            filled by a tip. Default lowered to 0.5 (from 2.0, Task 5
            review): at 2.0 the tip -- a single point standing in for two
            distinct markers (V12+V13) -- still over-drags the whole-body
            solve, degrading non-wing reprojection ~2x (2.06->4.28px); 0.5
            avoids that drag while keeping a positive wing recovery signal.
        min_cams: minimum camera count for a tip to be used.
        only_missing: if True (default), a wing marker is only overwritten
            when its current ``kp_data`` row for that (frame, marker) is
            all-NaN (i.e. missing/withheld); present (finite) markers and
            their weights are left untouched, so augmentation is a no-op
            when GT/tracked wing keypoints are already present. If False,
            keeps the old unconditional-override behavior.
    """
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
                if only_missing and np.isfinite(kp2[t, idx, :]).all():
                    continue
                kp2[t, idx, :] = tip3d
                used_ids.add(idx)
    for idx in used_ids:
        w2[idx * 3:idx * 3 + 3] = wing_weight
    return kp2, w2
