"""3D training losses for HybridNet 3D (JAX).

Functions
---------
build_skeleton_edges  -- map skeleton name-pairs to index pairs (numpy, not JIT'd)
graph_laplacian       -- bone-vector shape prior (jit-friendly)
heatmap3d_mse         -- 3D Gaussian target vs predicted volume (jit-friendly)

All JAX functions use masks/jnp.where throughout — no data-dependent Python
branching on traced values.

Grid-index mapping (inverse of soft_argmax_3d in hybridnet/model.py)
---------------------------------------------------------------------
soft_argmax_3d world coordinate:
    world = idx * grid_spacing * 2 - roi_cube / 2.0

Inverse (GT world → grid index):
    idx = (world - center3D + roi_cube / 2.0) / (grid_spacing * 2)

where `world` is the absolute world-space GT keypoint and `center3D` is the
per-sample 3D bounding-box centre (added back in model.__call__ after
soft_argmax_3d, so it must be subtracted here to get cube-local coords).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# build_skeleton_edges (pure numpy / Python — called once at setup time)
# ---------------------------------------------------------------------------

def build_skeleton_edges(
    keypoint_names: list[str],
    skeleton: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Map skeleton bones (name-pairs) to index pairs.

    Mirrors ``JARVIS-HybridNet/jarvis/hybridnet/loss.py::build_skeleton_edges``.
    Bones referencing unknown keypoint names are silently skipped.

    Args:
        keypoint_names: Ordered list of keypoint name strings.
        skeleton:       Iterable of ``(name_a, name_b)`` bone pairs.

    Returns:
        ei: ``(E,)`` int32 numpy array of source joint indices.
        ej: ``(E,)`` int32 numpy array of destination joint indices.
    """
    name_to_idx = {n: i for i, n in enumerate(keypoint_names)}
    ei_list, ej_list = [], []
    for bone in skeleton:
        a, b = bone[0], bone[1]
        if a in name_to_idx and b in name_to_idx:
            ei_list.append(name_to_idx[a])
            ej_list.append(name_to_idx[b])
    if not ei_list:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    return np.array(ei_list, dtype=np.int32), np.array(ej_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# graph_laplacian
# ---------------------------------------------------------------------------

def graph_laplacian(
    pred: jnp.ndarray,   # (B, K, 3)
    gt: jnp.ndarray,     # (B, K, 3)
    valid: jnp.ndarray,  # (B, K) bool
    ei: jnp.ndarray,     # (E,) int
    ej: jnp.ndarray,     # (E,) int
) -> jnp.ndarray:
    """Bone-vector (graph-gradient) shape prior.

    For each skeleton bone ``(i, j)`` penalise the squared difference between
    the predicted and GT bone vectors::

        ||(pred_i - pred_j) - (gt_i - gt_j)||^2

    Bones with at least one invalid endpoint are masked out.
    Returns the mean squared bone-vector error over valid bones (scalar).

    Translation-invariant (uses differences), so the absolute position of the
    root does not affect the loss.

    Args:
        pred:  ``(B, K, 3)`` predicted 3D keypoints.
        gt:    ``(B, K, 3)`` ground-truth 3D keypoints.
        valid: ``(B, K)`` bool mask of present keypoints.
        ei:    ``(E,)`` source bone-endpoint indices.
        ej:    ``(E,)`` destination bone-endpoint indices.

    Returns:
        Scalar mean squared bone-vector error over valid bones.
    """
    # Handle empty skeleton gracefully
    if ei.shape[0] == 0:
        return jnp.zeros((), dtype=pred.dtype)

    dpred = pred[:, ei, :] - pred[:, ej, :]   # (B, E, 3)
    dgt = gt[:, ei, :] - gt[:, ej, :]         # (B, E, 3)

    # Mask: both endpoints must be valid — shape (B, E), one entry per bone
    emask = (valid[:, ei] & valid[:, ej]).astype(pred.dtype)  # (B, E)

    # Per-bone squared error: sum over 3 spatial components, then mask
    sq_err = ((dpred - dgt) ** 2).sum(axis=-1)  # (B, E)
    total = (sq_err * emask).sum()
    count = emask.sum()                          # number of valid (sample, bone) pairs
    return total / jnp.maximum(count, 1.0)


# ---------------------------------------------------------------------------
# heatmap3d_mse
# ---------------------------------------------------------------------------

def heatmap3d_mse(
    pred_vol: jnp.ndarray,   # (B, J, G, G, G)
    gt_kp: jnp.ndarray,      # (B, J, 3)  world-space GT keypoints
    valid: jnp.ndarray,      # (B, J) bool
    *,
    grid_spacing: int = 1,
    roi_cube: int = 48,
    center3D: jnp.ndarray,   # (B, 3)  3D bounding-box centre
    sigma: float = 2.0,
    bg_weight: float = 0.1,
    fg_thresh: float = 0.01,
) -> jnp.ndarray:
    """3-D Gaussian heatmap MSE loss for v2vNet output.

    Renders a 3-D isotropic Gaussian target volume for each GT joint (at its
    grid-index location) and computes the masked MSE vs the predicted volume.

    Grid-index mapping (exact inverse of ``soft_argmax_3d`` in model.py)::

        idx = (gt_kp - center3D + roi_cube / 2.0) / (grid_spacing * 2)

    Joints whose grid index falls outside ``[0, G-1]`` on ANY axis are treated
    as invalid (combined with the ``valid`` mask) so out-of-cube GT points do
    not produce a degenerate/clipped Gaussian target.

    Foreground-weighted MSE (matching 2D ``heatmap_mse``): per joint, the
    squared error is averaged separately over foreground voxels (target >
    ``fg_thresh``) and background voxels, then combined as
    ``mse_fg + bg_weight * mse_bg``. This prevents the sparse-target dilution
    (~13 000× for G=24) that would otherwise collapse the model to predict
    all-zeros.

    Args:
        pred_vol:     ``(B, J, G, G, G)`` softplus-activated v2vNet output.
        gt_kp:        ``(B, J, 3)`` ground-truth world-space 3D keypoints.
        valid:        ``(B, J)`` bool — provided validity mask.
        grid_spacing: World units per grid step (default 1).
        roi_cube:     Full cube side-length in world units (default 48).
        center3D:     ``(B, 3)`` per-sample 3D bounding-box centre.
        sigma:        Gaussian standard deviation in grid units (default 2.0).
        bg_weight:    Weight for background voxel MSE term (default 0.1).
        fg_thresh:    Threshold above which a voxel is considered foreground
                      in the GT Gaussian (default 0.01).

    Returns:
        Scalar foreground-weighted masked MSE loss.
    """
    B, J, G, _, _ = pred_vol.shape

    # 1. Convert GT world coords to grid indices
    #    gt_kp is absolute world; center3D is the per-sample cube centre.
    #    cube-local world: gt_kp - center3D[:, None, :]
    #    inverse of soft_argmax_3d: idx = (local_world + roi_cube/2) / (grid_spacing*2)
    gt_local = gt_kp - center3D[:, None, :]                          # (B, J, 3)
    idx = (gt_local + roi_cube / 2.0) / (grid_spacing * 2.0)         # (B, J, 3)

    # 2. Out-of-grid mask: joint is valid only if ALL three indices are in [0, G-1]
    in_grid = jnp.all((idx >= 0.0) & (idx <= float(G - 1)), axis=-1)  # (B, J) bool
    joint_valid = valid & in_grid                                       # (B, J) bool

    # 3. Build 3D coordinate grid
    #    grid_coords[d] has shape (G,) with values 0..G-1
    coords = jnp.arange(G, dtype=jnp.float32)
    # Expand for broadcasting with (B, J, G, G, G):
    #   cx: varies along dim 2 (depth)
    #   cy: varies along dim 3 (height)
    #   cz: varies along dim 4 (width)
    cx = coords[None, None, :, None, None]   # (1, 1, G, 1, 1)
    cy = coords[None, None, None, :, None]   # (1, 1, 1, G, 1)
    cz = coords[None, None, None, None, :]   # (1, 1, 1, 1, G)

    # GT grid indices per joint: (B, J) for each axis
    ix = idx[:, :, 0][:, :, None, None, None]  # (B, J, 1, 1, 1)
    iy = idx[:, :, 1][:, :, None, None, None]
    iz = idx[:, :, 2][:, :, None, None, None]

    # 4. Render 3D Gaussian target: exp(-((cx-ix)^2 + (cy-iy)^2 + (cz-iz)^2) / (2*sigma^2))
    two_sigma_sq = 2.0 * sigma ** 2
    gauss = jnp.exp(
        -(
            (cx - ix) ** 2 + (cy - iy) ** 2 + (cz - iz) ** 2
        ) / two_sigma_sq
    )  # (B, J, G, G, G)

    # 5. For invalid joints, zero out the Gaussian target (so they contribute 0 to loss)
    jv = joint_valid[:, :, None, None, None].astype(pred_vol.dtype)  # (B, J, 1, 1, 1)
    gauss = gauss * jv   # (B, J, G, G, G) — zeroed for invalid joints

    # 6. Foreground-weighted MSE per joint
    #    Mirrors heatmap_mse in losses.py: average fg/bg separately, then combine.
    eps = 1e-6
    sq_err = (pred_vol - gauss) ** 2   # (B, J, G, G, G)

    fg = (gauss > fg_thresh).astype(pred_vol.dtype)   # (B, J, G, G, G)
    bg = 1.0 - fg

    # Sum squared error over voxels per joint
    nfg = fg.sum(axis=(2, 3, 4)) + eps   # (B, J)
    nbg = bg.sum(axis=(2, 3, 4)) + eps   # (B, J)
    mse_fg = (sq_err * fg).sum(axis=(2, 3, 4)) / nfg   # (B, J)
    mse_bg = (sq_err * bg).sum(axis=(2, 3, 4)) / nbg   # (B, J)

    per_joint_mse = mse_fg + bg_weight * mse_bg   # (B, J)

    w = joint_valid.astype(pred_vol.dtype)         # (B, J)
    return (per_joint_mse * w).sum() / jnp.maximum(w.sum(), 1.0)
