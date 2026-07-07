"""Temporal smoothing / outlier rejection of triangulated bout keypoints before
STAC.

The courtship pipeline triangulates raw per-frame 2D into kp3d and (until now)
fed that straight to STAC. Distal leg keypoints (esp. `*_TaTip`) occasionally
mistriangulate and jump many mm frame-to-frame; STAC then bends the leg to chase
the outlier, producing the jittery/curling legs seen in the IK. This module
reuses the free-walking preprocessing filter (`utils.keypoint_filter.
filter_keypoints`: confidence mask -> MAD bone-length outlier reject -> isolated
-spike removal -> PCHIP gap-fill -> Savitzky-Golay) on the per-bout kp3d.
Validated on Session0 bout1 fly1 (male): foot-tip max accel 12.5 -> 0.9 mm and
IK leg/foot-tip jitter ~7x lower, with reprojection accuracy unchanged.

`utils` is a repo-root package (NOT part of the jarvis_jax distribution), so the
import is resolved lazily with a path fallback computed from this file's known
location under `<repo>/third_party/jarvis_jax/jarvis_jax/cse/`.
"""
import os
import sys

import numpy as np

# per-leg proximal->distal joint order (matches viz.core.colors / the anatomy)
_LEG_SEGS = ["ThxCx", "Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip"]
_LEGS = ("T1L", "T2L", "T3L", "T1R", "T2R", "T3R")


def courtship_skeleton_edges(kp_names):
    """(E, 2) int index pairs used by the bone-length outlier filter:
    head/thorax/abdomen + wing chains + per-leg proximal->distal chains (each
    leg also linked to Scutellum). Derived from names so it works for any KP
    ordering; missing keypoints are skipped."""
    idx = {n: i for i, n in enumerate(kp_names)}

    def e(a, b):
        return (idx[a], idx[b]) if a in idx and b in idx else None

    edges = [
        e("EyeL", "Antenna_Base"), e("EyeR", "Antenna_Base"),
        e("Antenna_Base", "Scutellum"),
        e("Scutellum", "WingL_base"), e("WingL_base", "WingL_V12"), e("WingL_V12", "WingL_V13"),
        e("Scutellum", "WingR_base"), e("WingR_base", "WingR_V12"), e("WingR_V12", "WingR_V13"),
        e("Scutellum", "Abd_A4"), e("Abd_A4", "Abd_tip"),
    ]
    for leg in _LEGS:
        chain = [idx[f"{leg}_{s}"] for s in _LEG_SEGS if f"{leg}_{s}" in idx]
        for k in range(len(chain) - 1):
            edges.append((chain[k], chain[k + 1]))
        if chain and "Scutellum" in idx:
            edges.append((idx["Scutellum"], chain[0]))
    return np.array([x for x in edges if x is not None], dtype=int)


def _filter_keypoints():
    """Import utils.keypoint_filter.filter_keypoints, adding the repo root to
    sys.path if the pipeline's cwd/PYTHONPATH didn't already expose it."""
    try:
        from utils.keypoint_filter import filter_keypoints
    except ImportError:
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from utils.keypoint_filter import filter_keypoints
    return filter_keypoints


def filter_bout_kp3d(kp3d, conf3d, kp_names, filter_cfg):
    """Return a cleaned copy of ``kp3d`` (T, K, 3).

    No-op copy when ``filter_cfg`` is falsy or ``filter_cfg.enabled`` is False,
    so callers can gate on config without branching. ``conf3d`` (T, K) may be
    None (confidence masking is then skipped by the underlying filter).
    """
    kp3d = np.asarray(kp3d, np.float64)
    enabled = bool(filter_cfg.get("enabled", False)) if filter_cfg is not None else False
    if not enabled:
        return kp3d.copy()
    filter_keypoints = _filter_keypoints()
    edges = courtship_skeleton_edges(kp_names)
    conf = np.asarray(conf3d) if conf3d is not None else None
    filtered, _report, _edge_nan = filter_keypoints(
        kp3d.copy(), conf, edges, filter_cfg, kp_names=list(kp_names))
    filtered = np.asarray(filtered, np.float64)

    # Coverage invariant: smoothing must never DELETE a keypoint the raw
    # triangulation had. Outliers flagged mid-sequence are replaced by the
    # interpolant (finite), but an outlier flagged inside a LEADING/TRAILING run
    # that interpolation can't extrapolate (see filter_cfg.interpolation.
    # max_edge_extrap_frames) is left NaN -- which would starve STAC and NaN the
    # whole frame's qpos (then the model->world bridge FK has NaN sites). Fall
    # back to the raw value wherever the filter dropped a keypoint the input had,
    # so filtering only ever cleans/smooths and never reduces coverage below the
    # input triangulation.
    bad = (~np.isfinite(filtered).all(-1)) & np.isfinite(kp3d).all(-1)   # (T, K)
    filtered[bad] = kp3d[bad]

    # Preserve raw kinematics for named keypoints (default: wings). The savgol /
    # isolated-spike smoothing stages are NOT wing-aware (only confidence &
    # bone-length exclude wings), so they flatten the rapid wing extension/song
    # of courting males (~2x reduction in wing speed observed). Restore those
    # keypoints to the raw triangulation wherever it is finite, so legs/body get
    # the full smoothing (which fixes foot-tip curl) while wing motion is kept
    # intact. Gaps the filter interpolated (raw NaN) are left as-is.
    patterns = list(filter_cfg.get("preserve_raw_patterns", ["Wing"]) or [])
    if patterns:
        keep = [i for i, n in enumerate(kp_names) if any(p in n for p in patterns)]
        if keep:
            keep = np.array(keep)
            rawk = kp3d[:, keep]
            fin = np.isfinite(rawk).all(-1)                 # (T, n_keep)
            sub = filtered[:, keep]
            sub[fin] = rawk[fin]
            filtered[:, keep] = sub
    return filtered
