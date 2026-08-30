"""JAX port of JARVIS ReprojectionLayer.

Port of:
  third_party/JARVIS-HybridNet/jarvis/hybridnet/repro_layer.py

Pure-JAX, parameter-free.  ``reproject_heatmaps`` is vmapped over the batch
dimension so the per-sample helper operates on un-batched tensors.

Parity notes
------------
The DLT matmul uses ``lax.Precision.HIGHEST`` to minimise GPU tensor-core
approximation vs. the PyTorch CPU/GPU reference.  The trilinear upsample uses
``jax.image.resize(method='linear')`` which is align_corners=False, matching
``torch.nn.functional.interpolate(mode='trilinear')`` default.  Integer
truncation ``.astype(jnp.int32)`` matches PyTorch ``.int()``.

Known residual: at ~5 out of 5.5 M voxels the upsampled coordinate is within
one float32 ULP of an integer boundary; XLA and CUDA resolve the FMA rounding
to opposite sides, causing a 1-pixel index error in one camera whose
contribution (1/num_cameras of the heatmap difference) may reach ~0.13 on
random synthetic heatmaps.  On real heatmaps, the probability of a coordinate
landing exactly at a float32 integer boundary is negligible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def _build_half_grid(grid_size: int, grid_spacing: float,
                     rotation: jnp.ndarray | None = None) -> jnp.ndarray:
    """Build the base (G/2)^3 grid in 3-D world coordinates (no center offset).

    Matches the PyTorch constructor::

        half_gridsize = int(grid_size/2/2)   # G/4
        grid[i,j,k] = [i - half_gridsize, j - half_gridsize, k - half_gridsize]
        grid *= grid_spacing * 2

    `grid_spacing` is now a FLOAT. At the shipped 1.0 the grid spans 48 world
    units with 1 voxel = 1 unit ~ 0.17 mm; the distal tarsal segment is 1.59
    voxels, which is why stage 2 uses 0.25.

    `rotation` (3,3) rotates the grid BASIS. The V2VNet is otherwise
    world-frame-locked (it learns "dorsal is roughly +Z" for whichever
    calibration dominates training), so rotating the basis during training --
    with the labels rotated by the same R -- both removes that dependence and
    multiplies an otherwise small (3,800-sample) dataset.

    Returns shape ``(G/2, G/2, G/2, 3)``, float32.
    """
    half_g = grid_size // 2  # 24 when grid_size=48
    half_half_g = half_g // 2  # 12

    idx = jnp.arange(half_g, dtype=jnp.float32)  # [0..23]
    # Build the 3-D index grid; each axis runs from -half_half_g to +half_half_g-1
    ig, jg, kg = jnp.meshgrid(idx, idx, idx, indexing="ij")
    # Stack into (G/2, G/2, G/2, 3)
    grid = jnp.stack([ig - half_half_g, jg - half_half_g, kg - half_half_g], axis=-1)
    grid = grid * (grid_spacing * 2)
    if rotation is not None:
        grid = jnp.einsum("...i,ij->...j", grid, rotation.T,
                          precision=lax.Precision.HIGHEST)
    return grid


def _reproject_single(
    heatmaps: jnp.ndarray,      # (num_cam, J, hm, hm)
    center3D: jnp.ndarray,      # (3,)
    centerHM: jnp.ndarray,      # (num_cam, 2)
    camera_matrices: jnp.ndarray,  # (num_cam, 4, 3)
    *,
    grid_size: int,
    grid_spacing: float,
    heatmap_size: int,
    rotation: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Reproject heatmaps for a *single* frameset (no batch dimension).

    Returns ``(J, G, G, G)`` float32.
    """
    num_cam = heatmaps.shape[0]
    J = heatmaps.shape[1]
    hm = heatmap_size  # shorthand; equals heatmaps.shape[2]
    G = grid_size
    half_g = G // 2

    # ------------------------------------------------------------------
    # 1. Build the (G/2)^3 grid offset by center3D
    # ------------------------------------------------------------------
    base_grid = _build_half_grid(grid_size, grid_spacing, rotation)   # (G/2, G/2, G/2, 3)
    grid = base_grid + center3D                              # broadcast (G/2,G/2,G/2,3)

    # ------------------------------------------------------------------
    # 2. DLT projection for all cameras simultaneously
    #    grid: (G/2, G/2, G/2, 3) → append 1 → (G/2, G/2, G/2, 4)
    #    camera_matrices: (num_cam, 4, 3)
    #
    #    Use HIGHEST precision to minimise GPU tensor-core approximation
    #    vs. the PyTorch float32 reference.
    # ------------------------------------------------------------------
    ones = jnp.ones((*grid.shape[:3], 1), dtype=jnp.float32)
    x_hom = jnp.concatenate([grid, ones], axis=-1)          # (G/2,G/2,G/2,4)

    # Flatten grid points: (N4, 4) where N4 = (G/2)^3
    x_flat = x_hom.reshape(-1, 4)                           # (N4, 4)

    # Matmul: (num_cam, N4, 4) @ (num_cam, 4, 3) → (num_cam, N4, 3)
    # Broadcast x_flat over cameras
    proj = jnp.einsum(
        "pi,cij->cpj", x_flat, camera_matrices,
        precision=lax.Precision.HIGHEST,
    )  # (num_cam, N4, 3)
    proj = proj.reshape(num_cam, half_g, half_g, half_g, 3)    # (num_cam, G/2,G/2,G/2, 3)

    # PyTorch does: partial_all.permute(1,2,3,4,0) so shape becomes (G/2,G/2,G/2,3,num_cam)
    # then val1 = partial_all[:,:,:,0] / partial_all[:,:,:,2]  → (G/2,G/2,G/2,num_cam)
    # We keep (num_cam, G/2,G/2,G/2, 3) and index last dim:
    val1 = proj[..., 0] / proj[..., 2]   # (num_cam, G/2, G/2, G/2)
    val2 = proj[..., 1] / proj[..., 2]   # (num_cam, G/2, G/2, G/2)

    # ------------------------------------------------------------------
    # 3. Clamp and shift (per-camera using centerHM)
    #    centerHM: (num_cam, 2)  →  centerHM[:,0] = x-center, [:,1] = y-center
    #    PyTorch: centerHM = centerHM.permute(1,0)  → (2, num_cam)
    #             then centerHM[0] → x-center, centerHM[1] → y-center
    # ------------------------------------------------------------------
    cx = centerHM[:, 0].reshape(num_cam, 1, 1, 1)   # (num_cam,1,1,1)
    cy = centerHM[:, 1].reshape(num_cam, 1, 1, 1)   # (num_cam,1,1,1)

    val1 = (jnp.clip(val1, cx - (hm - 1), cx + hm - 2) - cx + hm - 1)
    val2 = (jnp.clip(val2, cy - (hm - 1), cy + hm - 2) - cy + hm - 1)

    # ------------------------------------------------------------------
    # 4. Trilinear upsample from (G/2)^3 to G^3
    #    PyTorch: F.interpolate(val.permute(3,0,1,2).view(1,-1,G/2,G/2,G/2), ...)
    #    val starts as (G/2,G/2,G/2,num_cam) there; here we have (num_cam,G/2,G/2,G/2).
    #    Either way: upsample each camera's coordinate map from (G/2)^3 to G^3.
    #
    #    jax.image.resize with method="linear" on a 5-D (batch, D, H, W, chan) tensor
    #    does trilinear on the spatial dims; align_corners=False is the XLA default,
    #    matching PyTorch F.interpolate(mode='trilinear') default.
    # ------------------------------------------------------------------
    # val1/val2 shape: (num_cam, G/2, G/2, G/2)
    # Reshape to (num_cam, G/2, G/2, G/2, 1) then resize spatial dims
    def _resize_coord(v):
        # v: (num_cam, G/2, G/2, G/2)
        v5 = v[..., jnp.newaxis]                        # (num_cam, G/2, G/2, G/2, 1)
        v5r = jax.image.resize(v5, (num_cam, G, G, G, 1), method="linear")
        return v5r[..., 0]                              # (num_cam, G, G, G)

    val1 = _resize_coord(val1)   # (num_cam, G, G, G)
    val2 = _resize_coord(val2)   # (num_cam, G, G, G)

    # ------------------------------------------------------------------
    # 5. Integer-truncated flat heatmap index
    #    res = (val2/2).int() * hm + (val1/2).int()
    #    .astype(jnp.int32) truncates toward zero, matching PyTorch .int()
    #    shape: (num_cam, G, G, G)
    # ------------------------------------------------------------------
    res = ((val2 / 2).astype(jnp.int32) * hm
           + (val1 / 2).astype(jnp.int32))              # (num_cam, G, G, G)

    # ------------------------------------------------------------------
    # 6. Add per-camera offset and gather from flattened heatmaps
    #    cam_offset[c] = c * hm * hm
    #    PyTorch: reproPoints.flatten(1) → (num_cam, G^3)
    #             .transpose(1,0) + cam_offset → add broadcast offset
    #             .transpose(1,0).flatten() → (num_cam * G^3,)
    # ------------------------------------------------------------------
    cam_offset = jnp.arange(num_cam, dtype=jnp.int32) * hm * hm  # (num_cam,)

    res_flat = res.reshape(num_cam, -1)                  # (num_cam, G^3)
    # Add per-camera offset: (num_cam, G^3) + (num_cam, 1)
    res_flat = res_flat + cam_offset[:, jnp.newaxis]     # (num_cam, G^3)
    # Flatten to 1-D index: (num_cam * G^3,)
    res_1d = res_flat.flatten()

    # Clamp to valid range (mirrors PyTorch .clamp(0, ...))
    max_idx = num_cam * hm * hm - 1
    res_1d = jnp.clip(res_1d, 0, max_idx)

    # heatmaps: (num_cam, J, hm, hm) → transpose to (J, num_cam, hm, hm)
    # (mirrors torch.transpose(heatmaps[b], 0, 1) in PyTorch forward)
    heatmaps_t = jnp.transpose(heatmaps, (1, 0, 2, 3))  # (J, num_cam, hm, hm)
    # Flatten cam/hm dimensions: (J, num_cam*hm*hm)
    heatmaps_flat = heatmaps_t.reshape(J, -1)

    # Gather: for each joint, gather the flattened indices
    # heatmaps_flat: (J, num_cam*hm*hm); res_1d: (num_cam*G^3,)
    gathered = heatmaps_flat[:, res_1d]                  # (J, num_cam*G^3)

    # Reshape and mean over cameras (mirrors torch.mean(..., dim=1))
    gathered = gathered.reshape(J, num_cam, G, G, G)     # (J, num_cam, G, G, G)
    out = gathered.mean(axis=1)                          # (J, G, G, G)

    return out


def reproject_heatmaps(
    heatmaps: jnp.ndarray,           # (B, num_cam, J, hm, hm)
    center3D: jnp.ndarray,           # (B, 3)
    centerHM: jnp.ndarray,           # (B, num_cam, 2)
    camera_matrices: jnp.ndarray,    # (B, num_cam, 4, 3)
    *,
    grid_size: int = 48,
    grid_spacing: float = 1.0,
    heatmap_size: int = 226,
    rotation: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Reproject 2-D per-camera heatmaps into a 3-D heatmap volume.

    Batched version of JARVIS ``ReprojectionLayer.forward``.

    Args:
        heatmaps:        ``(B, num_cam, J, hm, hm)``  per-camera keypoint heatmaps.
        center3D:        ``(B, 3)``  3-D bounding-box centre for each frameset.
        centerHM:        ``(B, num_cam, 2)``  2-D crop centre (pixels) per camera.
        camera_matrices: ``(B, num_cam, 4, 3)``  DLT projection matrices.
        grid_size:       Side length of the output 3-D volume (default 48).
        grid_spacing:    World-space units per grid cell (default 1.0, float).
        heatmap_size:    Spatial size of each input heatmap (default 226).
        rotation:        Optional ``(3,3)`` (shared across the batch) or
                         ``(B,3,3)`` (per-sample) rotation applied to the
                         grid basis. Must be orthogonal. ``None`` (default)
                         reproduces the shipped world-locked grid exactly.

    Returns:
        ``(B, J, grid_size, grid_size, grid_size)`` float32.
    """
    if rotation is not None:
        rotation = jnp.asarray(rotation, jnp.float32)
        R2 = rotation if rotation.ndim == 2 else rotation[0]
        if not bool(jnp.allclose(R2 @ R2.T, jnp.eye(3, dtype=jnp.float32), atol=1e-4)):
            raise ValueError("rotation must be orthogonal (R @ R.T == I)")
        rot_in_axis = None if rotation.ndim == 2 else 0
    else:
        rot_in_axis = None

    _single = lambda hm, c3, cHM, cM, rot: _reproject_single(
        hm, c3, cHM, cM,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        heatmap_size=heatmap_size,
        rotation=rot,
    )
    return jax.vmap(_single, in_axes=(0, 0, 0, 0, rot_in_axis))(
        heatmaps, center3D, centerHM, camera_matrices, rotation)
