"""SAM3 mask-fusion core for HybridNet3D.

Turns per-camera SAM3 silhouette masks into:

1. A 3-D "consistency volume" -- how much the cameras agree that a given
   voxel lies inside the animal's silhouette -- by reprojecting the masks
   through the SAME machinery as `reproject.reproject_heatmaps` (a mask is
   just a single-"joint" heatmap).
2. A soft gate that attenuates a keypoint/probability volume outside the
   silhouette, without ever hard-zeroing it (a single bad/blank camera mask
   must not carve a genuine voxel out of the volume).
3. A crop-level masking helper (Option B: zero background pixels before they
   reach the 2-D backbone).

This module is pure-JAX and parameter-free; it does not reimplement the DLT /
trilinear-upsample reprojection math -- see `reproject.py` for that.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jarvis_jax.hybridnet.reproject import reproject_heatmaps

_VALID_AGGREGATES = ("mean", "topk")


def mask_consistency_volume(
    masks: jnp.ndarray,            # (B, num_cam, hm, hm)
    center3D: jnp.ndarray,          # (B, 3)
    centerHM: jnp.ndarray,          # (B, num_cam, 2)
    cameraMatrices: jnp.ndarray,    # (B, num_cam, 4, 3)
    *,
    grid_size: int = 48,
    heatmap_size: int = 226,
    aggregate: str = "mean",
    topk_k: int | None = None,
) -> jnp.ndarray:
    """Reproject per-camera masks into a `(B, G, G, G)` cross-camera consistency volume.

    Each camera's mask is treated as a single-"joint" heatmap (J=1) and run
    through the exact same `reproject_heatmaps` machinery used for keypoint
    heatmaps, so the reprojection geometry (DLT projection, clamping,
    trilinear upsample, index gather) is identical and not reimplemented
    here.

    Aggregation across cameras is deliberately never a strict product: a
    single bad/blank camera mask (e.g. SAM3 dropout, occlusion) must not
    force a genuine voxel to zero just because one camera disagrees.

    - ``aggregate='mean'``: the mean across cameras. `reproject_heatmaps`
      already computes exactly this internally (see
      `reproject._reproject_single`'s ``gathered.mean(axis=1)``), so this
      case reuses it directly with J=1 -- no extra per-camera work needed.
    - ``aggregate='topk'``: to be robust to WORSE cases than a single blank
      camera (e.g. one or two noisy/occluded cameras among many), average
      only the top-`k` per-camera values at each voxel instead of all
      `num_cam` of them. This requires the individual per-camera reprojected
      volumes, which `reproject_heatmaps`'s internal mean does not expose.
      Rather than re-deriving the projection/gather math, each camera is
      reprojected on its own by slicing the camera axis down to size 1 and
      calling `reproject_heatmaps` again per camera; with `num_cam == 1` the
      internal ``mean(axis=1)`` is a no-op, so this yields exactly that
      camera's own reprojected volume using the identical, already-tested
      code path. Default ``topk_k = num_cam - 1`` (drop only the single
      worst camera per voxel -- the minimal choice that still tolerates one
      bad camera without discarding real disagreement).

    Args:
        masks: ``(B, num_cam, hm, hm)`` per-camera soft/binary silhouette
            masks, aligned to the padded heatmap size.
        center3D, centerHM, cameraMatrices: see `reproject.reproject_heatmaps`.
        grid_size: cube side length (default 48, matches HybridNet3D).
        heatmap_size: spatial size of each input mask (default 226).
        aggregate: ``'mean'`` or ``'topk'``.
        topk_k: number of cameras to average per voxel when
            ``aggregate='topk'``. Defaults to ``num_cam - 1``.

    Returns:
        ``(B, grid_size, grid_size, grid_size)`` float32 consistency volume.
    """
    if aggregate not in _VALID_AGGREGATES:
        raise ValueError(f"aggregate must be one of {_VALID_AGGREGATES}, got {aggregate!r}")

    num_cam = masks.shape[1]
    masks_j1 = masks[:, :, None, :, :]  # (B, num_cam, 1, hm, hm)

    if aggregate == "mean":
        vol = reproject_heatmaps(
            masks_j1, center3D, centerHM, cameraMatrices,
            grid_size=grid_size, grid_spacing=1, heatmap_size=heatmap_size,
        )  # (B, 1, G, G, G)
        return vol[:, 0]

    # aggregate == "topk": reproject each camera independently (see docstring)
    # and average the top-k values per voxel.
    per_cam = []
    for c in range(num_cam):
        vol_c = reproject_heatmaps(
            masks_j1[:, c : c + 1],
            center3D,
            centerHM[:, c : c + 1],
            cameraMatrices[:, c : c + 1],
            grid_size=grid_size, grid_spacing=1, heatmap_size=heatmap_size,
        )  # (B, 1, G, G, G)
        per_cam.append(vol_c[:, 0])
    stacked = jnp.stack(per_cam, axis=1)  # (B, num_cam, G, G, G)

    k = topk_k if topk_k is not None else max(1, num_cam - 1)
    k = min(k, num_cam)
    # top_k operates on the last axis -> move the camera axis there.
    moved = jnp.moveaxis(stacked, 1, -1)               # (B, G, G, G, num_cam)
    top_vals, _ = jax.lax.top_k(moved, k)               # (B, G, G, G, k)
    return top_vals.mean(axis=-1)


def soft_gate(
    volume: jnp.ndarray,       # (B, J, G, G, G)
    consistency: jnp.ndarray,  # (B, G, G, G)
    *,
    temperature: float = 1.0,
    floor: float = 0.0,
) -> jnp.ndarray:
    """Gate a keypoint/probability volume by cross-camera mask consistency.

    ``gate = floor + (1 - floor) * sigmoid((consistency - 0.5) / temperature)``

    - ``floor=1.0`` is an exact identity: the gate is exactly 1 everywhere
      regardless of `consistency`, so masking never fully overrides the
      caller's choice to disable it.
    - ``temperature`` controls the sharpness of the sigmoid transition around
      the 0.5 consistency threshold; low temperature approaches a hard gate,
      high temperature approaches a uniform 0.5 (near-identity up to scale).
    - The gate is broadcast over the joint/channel axis ``J``.

    Args:
        volume: ``(B, J, G, G, G)`` volume to gate.
        consistency: ``(B, G, G, G)`` cross-camera mask consistency, e.g.
            from `mask_consistency_volume`.
        temperature: sigmoid temperature (default 1.0).
        floor: minimum gate value (default 0.0); ``1.0`` disables gating.

    Returns:
        ``(B, J, G, G, G)`` gated volume.
    """
    gate = floor + (1.0 - floor) * jax.nn.sigmoid((consistency - 0.5) / temperature)
    return volume * gate[:, None]


def apply_carve_gate(
    vol: jnp.ndarray,               # (B, J, G, G, G)
    masks: jnp.ndarray,             # (B, num_cam, heatmap_size, heatmap_size)
    center3D: jnp.ndarray,          # (B, 3)
    centerHM: jnp.ndarray,          # (B, num_cam, 2)
    cameraMatrices: jnp.ndarray,    # (B, num_cam, 4, 3)
    *,
    temperature: float,
    floor: float,
    grid_size: int = 48,
    heatmap_size: int = 226,
) -> jnp.ndarray:
    """Apply the SAM3 'carve' fusion gate to a pre-V2VNet volume.

    This is the single source of truth for the ``fusion_mode == 'carve'``
    gate: builds the cross-camera mask consistency volume via
    `mask_consistency_volume` and attenuates ``vol`` with `soft_gate`. It is
    called identically by all three carve call sites in the codebase --
    inference (``HybridNet3D.__call__`` and ``predict.infer_3d.step_vol``)
    and training (``train.train_3d.loss_fn``) -- so the gate math lives in
    exactly one place. Callers remain responsible for their own
    ``fusion_mode == 'carve' and masks is not None`` guard; this function
    unconditionally applies the gate.

    Args:
        vol: ``(B, J, grid_size, grid_size, grid_size)`` pre-V2VNet volume
            (e.g. the ``/255``-normalized reprojected heatmap volume).
        masks: ``(B, num_cam, heatmap_size, heatmap_size)`` per-camera
            silhouette masks, aligned to the padded heatmap size.
        center3D, centerHM, cameraMatrices: see `mask_consistency_volume`.
        temperature, floor: `soft_gate` parameters (typically
            ``model.gate_temperature`` / ``model.gate_floor``).
        grid_size: cube side length (default 48, matches HybridNet3D).
        heatmap_size: spatial size of each input mask (default 226, the
            padded heatmap size ``reproject_volume`` reprojects at).

    Returns:
        ``(B, J, grid_size, grid_size, grid_size)`` gated volume.
    """
    consistency = mask_consistency_volume(
        masks, center3D, centerHM, cameraMatrices,
        grid_size=grid_size, heatmap_size=heatmap_size,
    )
    return soft_gate(vol, consistency, temperature=temperature, floor=floor)


def mask_input_crops(
    crops_nhwc: jnp.ndarray,  # (B, num_cam, H, W, C)
    masks: jnp.ndarray,       # (B, num_cam, H, W)
) -> jnp.ndarray:
    """Zero background pixels in per-camera crops using the SAM3 mask (Option B).

    Args:
        crops_nhwc: ``(B, num_cam, H, W, C)`` per-camera image crops.
        masks: ``(B, num_cam, H, W)`` per-camera silhouette masks, broadcast
            over the channel axis.

    Returns:
        ``(B, num_cam, H, W, C)`` crops with background zeroed.
    """
    return crops_nhwc * masks[..., None]
