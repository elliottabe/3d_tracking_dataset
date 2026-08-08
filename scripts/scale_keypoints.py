"""Resolve which keypoints the shared body-size scale is fit to.

scaling.scale_keypoints: 'trunk' -> the configured trunk list;
'all' -> every keypoint (pair with estimator=norm_ratio -- all+umeyama
provably collapses, see configs/preprocessing/v2_3.yaml);
'rigid_segment' -> the pipeline default (see
scripts/estimate_recording_scale.py's rigid-segment estimator, and
configs/pipeline.yaml's scaling block for why). That estimator is a direct
per-pair length measurement, not a fit over a single keypoint LIST -- but
this helper's one job (its callers, e.g. scripts/run_bout.py's scale.json
bookkeeping, only want a flat keypoint-name list to record) is unchanged: for
'rigid_segment' it returns every keypoint name that participates in at least
one rigid-segment pair (leg chain only, include_thorax=False), in
rigid_segment_pairs' order, deduplicated.
"""
from __future__ import annotations


def resolve_scale_keypoints(cfg, kp_names):
    mode = str(cfg.scaling.get("scale_keypoints", "trunk"))
    if mode == "all":
        return list(kp_names)
    if mode == "trunk":
        return list(cfg.scaling.trunk_keypoints)
    if mode == "rigid_segment":
        try:
            from scripts.estimate_recording_scale import rigid_segment_pairs
        except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/
            from estimate_recording_scale import rigid_segment_pairs
        pairs = rigid_segment_pairs(list(kp_names), include_thorax=False)
        names = []
        for a, b in pairs:
            for n in (a, b):
                if n not in names:
                    names.append(n)
        return names
    raise ValueError(
        f"scaling.scale_keypoints must be 'trunk', 'all', or 'rigid_segment', "
        f"got {mode!r}")
