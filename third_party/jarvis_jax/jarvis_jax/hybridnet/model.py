"""3D soft-argmax readout + HybridNet3D wiring.

JAX/NNX port of the JARVIS HybridNetBackbone forward pass (PyTorch reference:
third_party/JARVIS-HybridNet/jarvis/hybridnet/model.py).

Key design decisions
--------------------
* ``soft_argmax_3d`` mirrors the PyTorch reference exactly:
    - relu applied inside to ensure non-negative volumes (zeros remain zero,
      which allows sparse test volumes to yield exact peak recovery)
    - per-joint normalization over the G^3 volume
    - expectation computed against ``arange(G)`` with ``indexing='ij'``
      → x→dim-2, y→dim-3, z→dim-4
    - world scaling: ``idx * grid_spacing * 2 - roi_cube``
    - confidence: ``clamp(max_over_volume, max=255) / 255``
* ``HybridNet3D.__call__`` transposes between channel conventions:
    - ViTPose out: (B, num_cam, H, W, J)  [channels-last]
    - reproject expects: (B, num_cam, J, H, W)  [channels-first per joint]
    - V2VNet expects: (B, D, H, W, J)  [channels-last]
    - soft_argmax_3d expects: (B, J, G, G, G)  [joint-first]
* Heatmaps are padded from 224 → 226 (1 pixel each side) before reprojection,
  matching ``F.pad(heatmaps_batch, [1,1,1,1], 'constant', 0.)`` in the PyTorch
  forward (pad last two dims: W +1 each side, H +1 each side → 224+2=226).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jarvis_jax.data.device import normalize_image
from jarvis_jax.hybridnet.reproject import reproject_heatmaps


# ---------------------------------------------------------------------------
# soft_argmax_3d
# ---------------------------------------------------------------------------

def soft_argmax_3d(
    vol: jnp.ndarray,
    *,
    grid_spacing: int = 1,
    roi_cube: int = 48,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable 3D soft-argmax over a volumetric heatmap.

    Mirrors the PyTorch HybridNetBackbone forward (lines 71-87).

    Args:
        vol:          ``(B, J, G, G, G)`` raw v2vNet logits (any sign).
        grid_spacing: World-space units per grid step (default 1).
        roi_cube:     Full cube side length in world units (default 48).
                      World coordinate: ``idx * grid_spacing * 2 - roi_cube / 2.0``.

    Returns:
        points: ``(B, J, 3)`` world-space 3-D keypoints (before center3D offset).
        conf:   ``(B, J)``  confidence in [0, 1], clamped max over the volume.
    """
    # Apply relu to ensure non-negative volumes.
    # Using relu (not softplus) so that zero-valued voxels remain zero, which
    # allows sparse test volumes to yield exact peak recovery.
    # In HybridNet3D, softplus is applied to the v2vNet output before calling
    # this function, matching the PyTorch reference exactly; relu is then a
    # no-op on the already-positive softplus output.
    hm = jax.nn.relu(vol)   # (B, J, G, G, G)

    G = hm.shape[2]

    # Per-joint normalization: sum over D,H,W dims (2,3,4)
    norm = jnp.sum(hm, axis=(2, 3, 4), keepdims=True)  # (B, J, 1, 1, 1)
    # Avoid division by zero (shouldn't happen after softplus, but safe)
    norm = jnp.where(norm == 0, jnp.ones_like(norm), norm)

    # Build coordinate grids with indexing='ij' to match PyTorch exactly:
    #   xx[i,j,k] = i  (varies along dim-2 = D)
    #   yy[i,j,k] = j  (varies along dim-3 = H)
    #   zz[i,j,k] = k  (varies along dim-4 = W)
    coords = jnp.arange(G, dtype=jnp.float32)
    xx, yy, zz = jnp.meshgrid(coords, coords, coords, indexing='ij')
    # Shapes: (G, G, G) → broadcast over (B, J, G, G, G)
    xx = xx[None, None]   # (1, 1, G, G, G)
    yy = yy[None, None]
    zz = zz[None, None]

    # Soft-argmax expectation
    x = jnp.sum(hm * xx, axis=(2, 3, 4)) / norm[..., 0, 0, 0]  # (B, J)
    y = jnp.sum(hm * yy, axis=(2, 3, 4)) / norm[..., 0, 0, 0]
    z = jnp.sum(hm * zz, axis=(2, 3, 4)) / norm[..., 0, 0, 0]

    # World-space scaling: idx * grid_spacing * 2 - roi_cube / 2.0
    x = x * grid_spacing * 2 - roi_cube / 2.0   # (B, J)
    y = y * grid_spacing * 2 - roi_cube / 2.0
    z = z * grid_spacing * 2 - roi_cube / 2.0

    points = jnp.stack([x, y, z], axis=-1)  # (B, J, 3)

    # Confidence: clamp(max over the G^3 volume, max=255) / 255
    hm_flat = hm.reshape(*hm.shape[:2], -1)          # (B, J, G^3)
    vol_max = jnp.max(hm_flat, axis=-1)               # (B, J)
    conf = jnp.clip(vol_max, 0.0, 255.0) / 255.0     # (B, J)

    return points, conf


# ---------------------------------------------------------------------------
# HybridNet3D
# ---------------------------------------------------------------------------

class HybridNet3D(nnx.Module):
    """End-to-end 3-D pose model: frozen ViTPose front-end + trainable V2VNet.

    Architecture (forward data flow)::

        crops (B, num_cam, 448, 448, 4) uint8
            → normalize_image (vmapped over cameras)
            → ViTPose (frozen, use_running_average=True)
            → heatmaps (B, num_cam, 224, 224, J)
            → pad to 226×226
            → transpose to (B, num_cam, J, 226, 226)
            → reproject_heatmaps → (B, J, 48, 48, 48)
            → transpose to (B, 48, 48, 48, J)  [channels-last for V2VNet]
            → V2VNet → (B, 24, 24, 24, J)
            → transpose to (B, J, 24, 24, 24)
            → soft_argmax_3d → points_local (B, J, 3), conf (B, J)
            → add center3D → points3D world (B, J, 3)

    Args:
        vitpose: A ``ViTPose`` module (frozen; run in eval mode).
        v2vnet:  A ``V2VNet`` module (trainable).
        cfg:     A ``ViTPoseConfig`` (provides ``num_keypoints``).
    """

    def __init__(self, vitpose, v2vnet, cfg):
        self.vitpose = vitpose
        self.v2vnet = v2vnet
        self.cfg = cfg

    def predict_heatmaps(
        self,
        crops4_u8: jnp.ndarray,          # (B, num_cam, 448, 448, 4) uint8
    ) -> jnp.ndarray:                     # (B, num_cam, 224, 224, J)
        """Run ViTPose (frozen/eval) over all cameras to produce 2-D heatmaps.

        The front-end is run with ``use_running_average=True`` (eval mode) since
        ViTPose is frozen.  vmap is applied over the camera axis (dim 1) inside
        the batch dimension.

        Args:
            crops4_u8: ``(B, num_cam, 448, 448, 4)`` uint8 RGBA crops.

        Returns:
            ``(B, num_cam, 224, 224, J)`` float32 heatmap logits.
        """
        B = crops4_u8.shape[0]
        num_cam = crops4_u8.shape[1]

        # Flatten (B, num_cam) → (B*num_cam,), process, unflatten
        crops_flat = crops4_u8.reshape(B * num_cam, 448, 448, 4)
        # Normalize each image
        imgs_float = jax.vmap(normalize_image)(crops_flat)   # (B*num_cam, 448, 448, 4)
        # Run ViTPose in eval/frozen mode
        hm_flat = self.vitpose(imgs_float, use_running_average=True)  # (B*num_cam, 224, 224, J)
        # Unflatten back to (B, num_cam, 224, 224, J)
        J = hm_flat.shape[-1]
        return hm_flat.reshape(B, num_cam, 224, 224, J)

    def __call__(
        self,
        crops4_u8: jnp.ndarray,          # (B, num_cam, 448, 448, 4) uint8
        center3D: jnp.ndarray,           # (B, 3)
        centerHM: jnp.ndarray,           # (B, num_cam, 2)
        cameraMatrices: jnp.ndarray,     # (B, num_cam, 4, 3)
        *,
        use_running_average: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Full 3-D pose prediction forward pass.

        Args:
            crops4_u8:       ``(B, num_cam, 448, 448, 4)`` uint8 RGBA crops.
            center3D:        ``(B, 3)``  3-D bounding-box centre (world coords).
            centerHM:        ``(B, num_cam, 2)``  2-D crop centre per camera.
            cameraMatrices:  ``(B, num_cam, 4, 3)``  DLT projection matrices.
            use_running_average: Passed to V2VNet (controls dropout).

        Returns:
            vol:     ``(B, J, 24, 24, 24)``  softplus-activated 3-D heatmap volume.
            points3D: ``(B, J, 3)``  world-space 3-D keypoints.
            conf:    ``(B, J)``  per-joint confidence in [0, 1].
        """
        # NOTE: this forward is paralleled by loss_fn in train/train_3d.py, which
        # inserts stop_gradient after ViTPose. Keep grid/pad/transpose constants
        # (grid_size=48, grid_spacing=1, heatmap_size=226, pad=(1,1,1,1)) in sync.

        # 1. 2-D heatmaps: (B, num_cam, 224, 224, J) — ViTPose is always frozen/eval
        hm = self.predict_heatmaps(crops4_u8)      # (B, num_cam, 224, 224, J)

        # 2. Pad 224 → 226 (1 pixel each side on H and W).
        #    PyTorch: F.pad(heatmaps_batch, [1,1,1,1], 'constant', 0.)
        #    PyTorch pad format: (last_dim_left, last_dim_right,
        #                         second_last_left, second_last_right)
        #    hm layout: (B, num_cam, H=224, W=224, J) — H is dim 2, W is dim 3
        #    We need to pad H and W (dims 2 and 3 in this 5-D tensor).
        #    jnp.pad format: tuple of (before, after) per axis.
        hm = jnp.pad(hm, [(0, 0), (0, 0), (1, 1), (1, 1), (0, 0)])  # (B, num_cam, 226, 226, J)

        # 3. Transpose to (B, num_cam, J, H, W) for reproject_heatmaps
        #    From (B, num_cam, 226, 226, J) → (B, num_cam, J, 226, 226)
        hm = jnp.transpose(hm, (0, 1, 4, 2, 3))   # (B, num_cam, J, 226, 226)

        # 4. Reproject to 3D: (B, J, 48, 48, 48)
        vol3d = reproject_heatmaps(
            hm, center3D, centerHM, cameraMatrices,
            grid_size=48,
            grid_spacing=1,
            heatmap_size=226,
        )  # (B, J, 48, 48, 48)

        # 5. Divide by 255 (matches PyTorch: v2vNet(heatmaps3D / 255.))
        vol3d = vol3d / 255.0

        # 6. Transpose to (B, D, H, W, J) = (B, 48, 48, 48, J) for V2VNet
        vol3d = jnp.transpose(vol3d, (0, 2, 3, 4, 1))   # (B, 48, 48, 48, J)

        # 7. V2VNet: (B, 48, 48, 48, J) → (B, 24, 24, 24, J)
        vol3d = self.v2vnet(vol3d, use_running_average=use_running_average)

        # 8. Transpose back to (B, J, 24, 24, 24)
        vol3d = jnp.transpose(vol3d, (0, 4, 1, 2, 3))   # (B, J, 24, 24, 24)

        # 9. Apply softplus (matches PyTorch: heatmap_final = self.softplus(v2vNet_out))
        #    then call soft_argmax_3d. soft_argmax_3d applies relu internally,
        #    which is a no-op on softplus output (all values already positive).
        #    Note: returned vol3d is single-softplus'd (the reference double-softpluses
        #    the returned heatmap; single-softplus is deliberate and cleaner here).
        vol3d = jax.nn.softplus(vol3d)   # (B, J, 24, 24, 24)

        # 10. soft_argmax_3d: returns points_local (B,J,3) in cube-centred coords, conf (B,J)
        points_local, conf = soft_argmax_3d(vol3d, grid_spacing=1, roi_cube=48)

        # 11. Add center3D offset to get world coordinates
        #     center3D: (B, 3) → broadcast to (B, 1, 3)
        points3D = points_local + center3D[:, None, :]   # (B, J, 3)

        return vol3d, points3D, conf
