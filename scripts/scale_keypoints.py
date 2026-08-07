"""Resolve which keypoints the shared body-size scale is fit to.

scaling.scale_keypoints: 'trunk' -> the configured trunk list;
'all' -> every keypoint (pair with estimator=norm_ratio -- all+umeyama
provably collapses, see configs/preprocessing/v2_3.yaml).
"""
from __future__ import annotations


def resolve_scale_keypoints(cfg, kp_names):
    mode = str(cfg.scaling.get("scale_keypoints", "trunk"))
    if mode == "all":
        return list(kp_names)
    if mode == "trunk":
        return list(cfg.scaling.trunk_keypoints)
    raise ValueError(
        f"scaling.scale_keypoints must be 'trunk' or 'all', got {mode!r}")
