"""Full-array vertex index selection for mask-driven FK factors.

`mask_fk_indices` maps an fps subset of the canonical mesh down to the
full-vertex-array indices FK needs, optionally dropping vertices whose
segment id is in `exclude_seg_ids`.
"""
from __future__ import annotations
import numpy as np


def mask_fk_indices(mesh_npz, subset: str = "fps_300", exclude_seg_ids=None):
    """FULL-vertex-array indices for the mask factor's projected subset.

    INDEX-SPACE HAZARD (Phase-4 carry-forward, mirrors
    silhouette_ik_solve._wing_fk_indices): the mesh npz's ``fps_*`` arrays hold
    indices INTO the full ``vertices``/``vertex_geom`` arrays (values
    0..len(vertices)-1, e.g. 0..139352 for fly_v1_visual_canonical_wings.npz),
    so they are ALREADY full-array space and can be passed straight to
    ``fk.make_fk_repose(indices=...)``. In contrast,
    ``wing_landmarks.wing_side_vertices`` /
    ``active_parts.excluded_fps_indices`` return indices INTO the fps subset
    (0..299) -- those MUST be bridged via ``fps[idx]`` before FK, or they
    silently select the wrong vertices (verified thorax-vertex bug in
    _wing_fk_indices' docstring). This helper only ever returns full-array
    indices, and applies ``exclude_seg_ids`` in full-array space.

    Args:
        mesh_npz: canonical mesh npz path.
        subset: which fps subset to use ("fps_300" default).
        exclude_seg_ids: optional list of segment ids to drop (full-array
            filtering via ``vertex_segment``), for Phase-4 active-parts.

    Returns:
        np.ndarray (M,) int32 full-vertex-array indices.
    """
    z = np.load(mesh_npz, allow_pickle=True)
    fps = np.asarray(z[subset], dtype=np.int64)   # full-array indices already
    if exclude_seg_ids:
        seg = np.asarray(z["vertex_segment"])     # (n_vertices,) full-array
        excl = set(int(s) for s in exclude_seg_ids)
        keep = np.array([int(seg[i]) not in excl for i in fps])
        fps = fps[keep]
    return fps.astype(np.int32)
