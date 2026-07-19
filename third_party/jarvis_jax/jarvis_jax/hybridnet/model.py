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
    - world scaling: ``idx * grid_spacing * 2 - roi_cube / 2.0``
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
from jarvis_jax.hybridnet.mask_fuse import (
    apply_carve_gate,
    mask_input_crops,
)
from jarvis_jax.hybridnet.reproject import reproject_heatmaps


# ---------------------------------------------------------------------------
# soft_argmax_3d
# ---------------------------------------------------------------------------

def soft_argmax_3d(
    vol: jnp.ndarray,
    *,
    grid_spacing: int = 1,
    roi_cube: int = 48,
    sharpen: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable 3D soft-argmax over a volumetric heatmap.

    Mirrors the PyTorch HybridNetBackbone forward (lines 71-87).

    Args:
        vol:          ``(B, J, G, G, G)`` raw v2vNet logits (any sign).
        grid_spacing: World-space units per grid step (default 1).
        roi_cube:     Full cube side length in world units (default 48).
                      World coordinate: ``idx * grid_spacing * 2 - roi_cube / 2.0``.
        sharpen:      Exponent applied to the non-negative heatmap before taking
                      the expectation (default 1.0 = original behavior, exact
                      PyTorch-reference parity). Values >1 suppress the low
                      background "floor" spread across all ``G^3`` voxels — that
                      floor biases the expectation toward the grid centroid
                      (``center3D``), shrinking the predicted skeleton inward
                      (see scripts/diag_shrinkage.py). A one-hot volume recovers
                      its exact peak for any ``sharpen``. ``conf`` is always
                      computed from the un-sharpened heatmap so its scale is
                      unchanged.

    Returns:
        points: ``(B, J, 3)`` world-space 3-D keypoints (before center3D offset).
        conf:   ``(B, J)``  confidence in [0, 1], clamped max over the volume.
    """
    assert grid_spacing == 1, "soft_argmax_3d world offset assumes grid_spacing==1 (offset=roi_cube/2); generalize to roi_cube/grid_spacing/2 + matching grid if this changes"

    # Apply relu to ensure non-negative volumes.
    # Using relu (not softplus) so that zero-valued voxels remain zero, which
    # allows sparse test volumes to yield exact peak recovery.
    # In HybridNet3D, softplus is applied to the v2vNet output before calling
    # this function, matching the PyTorch reference exactly; relu is then a
    # no-op on the already-positive softplus output.
    hm0 = jax.nn.relu(vol)   # (B, J, G, G, G) — un-sharpened, used for conf
    # Sharpen the distribution before the expectation (center-bias fix).
    # sharpen=1.0 leaves hm0 untouched (byte-identical to the original path).
    hm = hm0 ** sharpen if sharpen != 1.0 else hm0

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
    # Computed on the UN-sharpened heatmap so its scale is independent of sharpen.
    hm_flat = hm0.reshape(*hm0.shape[:2], -1)         # (B, J, G^3)
    vol_max = jnp.max(hm_flat, axis=-1)               # (B, J)
    conf = jnp.clip(vol_max, 0.0, 255.0) / 255.0     # (B, J)

    return points, conf


# ---------------------------------------------------------------------------
# HybridNet3D
# ---------------------------------------------------------------------------

class HybridNet3D(nnx.Module):
    """End-to-end 3-D pose model: frozen 2-D front-end + trainable V2VNet.

    The 2-D front-end is pluggable: either ``ViTPose`` (the original,
    4-channel RGB+mask front-end) or ``EfficientTrack`` (the PyTorch
    ``HybridNetBackbone``'s faithful 3-channel RGB front-end). Both expose a
    ``(crops) -> (N, 224, 224, J)`` contract once flattened over cameras.

    Architecture (forward data flow)::

        crops (B, num_cam, 448, 448, C) [uint8 for ViTPose, float for EfficientTrack]
            → [ViTPose only] normalize_image (vmapped over cameras)
            → front_end (frozen, use_running_average=True where applicable)
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
        front_end: A 2-D keypoint module -- ``ViTPose`` (frozen; normalized
            RGB+mask input) or ``EfficientTrack`` (frozen; raw RGB input, no
            ImageNet normalization -- matches PyTorch ``HybridNetBackbone``,
            which feeds ``effTrack`` un-normalized images). Run in eval mode.
        v2vnet:  A ``V2VNet`` module (trainable).
        cfg:     A config object (provides ``num_keypoints``, optional
            ``sharpen``, and optional SAM3 mask-fusion fields ``fusion_mode``
            ('none'/'carve'/'input_mask'), ``gate_temperature``, ``gate_floor``
            -- all read via ``getattr`` with backward-compatible defaults, so
            any bare cfg object works unchanged; see ``__call__``'s ``masks``
            arg and ``jarvis_jax.hybridnet.mask_fuse``).
    """

    def __init__(self, front_end, v2vnet, cfg):
        self.front_end = front_end
        self.v2vnet = v2vnet
        self.cfg = cfg
        # soft-argmax sharpening exponent (center-bias fix); read from cfg if
        # present, else 1.0 (original behavior). See soft_argmax_3d / sweep diag.
        self.sharpen = float(getattr(cfg, "sharpen", 1.0))
        # SAM3 mask-fusion hook (Task 7). 'none' (default) never touches
        # masks -- the forward is byte-identical to the pre-fusion path.
        # 'carve': gate the pre-V2VNet volume by cross-camera mask consistency.
        # 'input_mask': zero background pixels in the crops before the 2-D
        # front-end. See mask_fuse.py for the underlying ops.
        self.fusion_mode = str(getattr(cfg, "fusion_mode", "none"))
        self.gate_temperature = float(getattr(cfg, "gate_temperature", 1.0))
        self.gate_floor = float(getattr(cfg, "gate_floor", 0.0))
        # ViTPose is the only front-end that expects ImageNet-normalized
        # input (RGB channels normalized + mask channel passed through raw);
        # every other front-end (e.g. EfficientTrack) matches the PyTorch
        # HybridNetBackbone reference, which feeds effTrack RAW images (no
        # normalization at all -- see export_hybridnet_fixture.py, whose
        # `imgs_nchw` are raw `torch.rand` in [0, 1)).
        from jarvis_jax.models.vitpose import ViTPose  # local: no cycle risk, avoids import at module load if unused
        self._is_vitpose = isinstance(front_end, ViTPose)

    @property
    def vitpose(self):
        """Backward-compat alias for the pre-refactor ``vitpose`` attribute
        name (e.g. ``model.vitpose.decoder...`` in existing tests/scripts).
        Returns whatever front-end was actually passed in -- callers that use
        this alias are expected to have constructed the model with a ViTPose
        front-end, as before."""
        return self.front_end

    def predict_heatmaps(
        self,
        crops_u8: jnp.ndarray,            # (B, num_cam, 448, 448, C)
        masks: jnp.ndarray | None = None,  # (B, num_cam, H, W) crop-resolution, 'input_mask' only
    ) -> jnp.ndarray:                     # (B, num_cam, 224, 224, J)
        """Run the frozen 2-D front-end over all cameras to produce heatmaps.

        The front-end is run in eval mode (frozen). For ViTPose, images are
        first ImageNet-normalized via ``normalize_image`` (byte-identical to
        the pre-refactor behavior); for any other front-end (e.g.
        EfficientTrack), crops are passed through as float32 with NO
        normalization, matching the PyTorch ``HybridNetBackbone`` reference.

        Args:
            crops_u8: ``(B, num_cam, 448, 448, C)`` crops -- uint8 RGBA for
                ViTPose, float RGB for EfficientTrack.
            masks: ``(B, num_cam, H, W)`` per-camera silhouette masks at
                *crop* resolution (matching ``crops_u8``'s H, W -- NOT the
                226 padded heatmap size used by ``fusion_mode='carve'``).
                Only consulted when ``self.fusion_mode == 'input_mask'``;
                ignored (not even read) otherwise, including when ``None``.

        Returns:
            ``(B, num_cam, 224, 224, J)`` float32 heatmap logits.
        """
        # Option B (Task 6/7): zero background pixels before the front-end.
        # Gated on fusion_mode so 'none'/'carve' never touch crops_u8 here --
        # this is the byte-identical guard for fusion_mode='none'.
        if self.fusion_mode == "input_mask" and masks is not None:
            crops_u8 = mask_input_crops(crops_u8, masks)

        if self._is_vitpose:
            B = crops_u8.shape[0]
            num_cam = crops_u8.shape[1]
            H, W, C = crops_u8.shape[2], crops_u8.shape[3], crops_u8.shape[4]
            # Flatten (B, num_cam) → (B*num_cam,), process, unflatten.
            crops_flat = crops_u8.reshape(B * num_cam, H, W, C)
            # Normalize each image, then run ViTPose in eval/frozen mode.
            imgs_float = jax.vmap(normalize_image)(crops_flat)   # (B*num_cam, H, W, C)
            hm_flat = self.front_end(imgs_float, use_running_average=True)  # (B*num_cam, 224, 224, J)
            J = hm_flat.shape[-1]
            return hm_flat.reshape(B, num_cam, 224, 224, J)

        # Non-ViTPose front-ends (e.g. EfficientTrack) consume raw crops
        # directly -- no normalization (PyTorch HybridNetBackbone parity).
        # Pass the un-flattened (B, num_cam, H, W, C) tensor straight through:
        # EfficientTrack.predict_heatmaps flattens/unflattens the leading
        # batch dims itself (see its docstring), so this is exactly the same
        # math as the old model.py-side manual flatten -- just relocated.
        imgs_float = crops_u8.astype(jnp.float32)
        return self.front_end.predict_heatmaps(imgs_float)  # (B, num_cam, 224, 224, J)

    def reproject_volume(
        self,
        crops_u8: jnp.ndarray,           # (B, num_cam, 448, 448, C)
        center3D: jnp.ndarray,           # (B, 3)
        centerHM: jnp.ndarray,           # (B, num_cam, 2)
        cameraMatrices: jnp.ndarray,     # (B, num_cam, 4, 3)
        masks: jnp.ndarray | None = None,  # (B, num_cam, H, W) crop-res, 'input_mask' only
    ) -> jnp.ndarray:                    # (B, J, 48, 48, 48)
        """Compute the reprojected 3-D volume — the exact intermediate cached by the trainer.

        Runs the frozen 2-D front-end, pads heatmaps 224→226, and reprojects
        into a ``(B, J, 48, 48, 48)`` volume (the pre-v2vNet, channels-first layout).
        This is the value that the cached trainer precomputes and stores; the
        remaining ``transpose → v2vNet → softplus → soft_argmax_3d`` steps are
        applied by ``__call__``.

        Args:
            crops_u8:        ``(B, num_cam, 448, 448, C)`` crops (uint8 RGBA for ViTPose,
                             float RGB for EfficientTrack).
            center3D:        ``(B, 3)``  3-D bounding-box centre (world coords).
            centerHM:        ``(B, num_cam, 2)``  2-D crop centre per camera.
            cameraMatrices:  ``(B, num_cam, 4, 3)``  DLT projection matrices.
            masks:           ``(B, num_cam, H, W)`` crop-resolution masks, only
                             consulted (via ``predict_heatmaps``) when
                             ``self.fusion_mode == 'input_mask'``. This is a
                             DIFFERENT resolution than the 226-sized masks used
                             by ``fusion_mode='carve'`` (applied in ``__call__``,
                             not here) -- see class/``predict_heatmaps`` docs.

        Returns:
            ``(B, J, 48, 48, 48)``  reprojected volume in ``(B, J, D, H, W)``
            layout, normalised by 255 — the direct input to the channels-last
            transpose that precedes V2VNet.
        """
        # NOTE: Keep grid/pad/transpose constants in sync with __call__ and
        # loss_fn in train/train_3d.py:
        # (grid_size=48, grid_spacing=1, heatmap_size=226, pad=(1,1,1,1)).

        # 1. 2-D heatmaps: (B, num_cam, 224, 224, J) — ViTPose is always frozen/eval
        hm = self.predict_heatmaps(crops_u8, masks=masks)      # (B, num_cam, 224, 224, J)

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

        return vol3d  # (B, J, 48, 48, 48)

    def __call__(
        self,
        crops_u8: jnp.ndarray,           # (B, num_cam, 448, 448, C)
        center3D: jnp.ndarray,           # (B, 3)
        centerHM: jnp.ndarray,           # (B, num_cam, 2)
        cameraMatrices: jnp.ndarray,     # (B, num_cam, 4, 3)
        *,
        masks: jnp.ndarray | None = None,
        use_running_average: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Full 3-D pose prediction forward pass.

        Args:
            crops_u8:        ``(B, num_cam, 448, 448, C)`` crops (uint8 RGBA for ViTPose,
                             float RGB for EfficientTrack).
            center3D:        ``(B, 3)``  3-D bounding-box centre (world coords).
            centerHM:        ``(B, num_cam, 2)``  2-D crop centre per camera.
            cameraMatrices:  ``(B, num_cam, 4, 3)``  DLT projection matrices.
            masks:           Optional SAM3 per-camera silhouette masks; ignored
                             entirely unless ``self.fusion_mode`` consults them
                             (never even read when ``fusion_mode == 'none'`` --
                             the byte-identical regression guard). Expected
                             shape depends on ``self.fusion_mode``:
                               - ``'carve'``:      ``(B, num_cam, 226, 226)``
                                 (the padded heatmap size ``reproject_volume``
                                 reprojects at).
                               - ``'input_mask'``: ``(B, num_cam, H, W)``
                                 matching ``crops_u8``'s crop resolution
                                 (e.g. 448×448) -- applied before the 2-D
                                 front-end. See ``predict_heatmaps``.
                             The two modes use different resolutions and are
                             mutually exclusive per call (``fusion_mode``
                             selects which one, if either, ``masks`` is read
                             as).
            use_running_average: Passed to V2VNet (controls dropout).

        Returns:
            vol:     ``(B, J, 24, 24, 24)``  softplus-activated 3-D heatmap volume.
            points3D: ``(B, J, 3)``  world-space 3-D keypoints.
            conf:    ``(B, J)``  per-joint confidence in [0, 1].
        """
        # NOTE: this forward is paralleled by loss_fn in train/train_3d.py, which
        # inserts stop_gradient after ViTPose. Keep grid/pad/transpose constants
        # (grid_size=48, grid_spacing=1, heatmap_size=226, pad=(1,1,1,1)) in sync.

        # 1–5. ViTPose → pad 224→226 → reproject → /255: (B, J, 48, 48, 48)
        # `masks` is threaded through unconditionally; predict_heatmaps only
        # actually reads it when fusion_mode=='input_mask', so this is a no-op
        # for 'none'/'carve' regardless of whether masks is provided.
        vol3d = self.reproject_volume(crops_u8, center3D, centerHM, cameraMatrices,
                                       masks=masks)

        # 5b. 'carve': gate the pre-V2VNet volume by cross-camera SAM3 mask
        # consistency (Task 6). Only constructed when fusion_mode=='carve' AND
        # masks were actually provided -- fusion_mode=='none' never reaches
        # this branch, so no gate/consistency volume is ever built for it
        # (byte-identical regression guard).
        if self.fusion_mode == "carve" and masks is not None:
            vol3d = apply_carve_gate(vol3d, masks, center3D, centerHM, cameraMatrices,
                                      temperature=self.gate_temperature,
                                      floor=self.gate_floor)

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
        points_local, conf = soft_argmax_3d(vol3d, grid_spacing=1, roi_cube=48,
                                            sharpen=self.sharpen)

        # 11. Add center3D offset to get world coordinates
        #     center3D: (B, 3) → broadcast to (B, 1, 3)
        points3D = points_local + center3D[:, None, :]   # (B, J, 3)

        return vol3d, points3D, conf
