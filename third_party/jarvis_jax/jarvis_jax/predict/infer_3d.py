"""Batched, sharded JAX 3D inference.

Default front-end/fusion is the production path: frozen ViTPose + run4
v2vNet, fusion_mode='none'. A second, opt-in front-end (EfficientTrack, the
faithful JAX port of the PyTorch HybridNetBackbone's 2-D backbone) can be
selected via ``load_inference_model(..., front_end='efficienttrack', ...)``
for parallel benchmarking -- see task-9-brief.md. Both front-ends feed the
SAME ``HybridNet3D`` reproject -> V2VNet -> soft-argmax 3-D fusion; there is
no separate DLT triangulation path in this codebase (the "vitpose_dlt" name
used in early planning docs was inaccurate).
"""
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.convert.build_checkpoint import load_vitpose
from jarvis_jax.hybridnet.mask_fuse import mask_consistency_volume, soft_gate
from jarvis_jax.hybridnet.model import HybridNet3D
from jarvis_jax.hybridnet.v2vnet import V2VNet
from jarvis_jax.sharding import data_parallel_mesh, replicate, shard_batch
from jarvis_jax.geometry.center3d import estimate_center3d_from_masks

_FRONT_ENDS = ("vitpose", "efficienttrack")


class _Cfg:
    """Minimal cfg object HybridNet3D reads (num_keypoints/sharpen/fusion).

    ``fusion_mode``/``gate_temperature``/``gate_floor`` are read by
    ``HybridNet3D`` via ``getattr`` with the same defaults ('none'/1.0/0.0),
    so passing them here is purely additive: any existing caller that built
    this object with just ``(num_keypoints, sharpen)`` still gets the
    byte-identical 'none' fusion behavior.
    """
    def __init__(self, num_keypoints, sharpen, *, fusion_mode="none",
                 gate_temperature=1.0, gate_floor=0.0):
        self.num_keypoints = num_keypoints
        self.sharpen = sharpen
        self.fusion_mode = fusion_mode
        self.gate_temperature = gate_temperature
        self.gate_floor = gate_floor


def _maybe_convert_pth(ckpt, *, kind, num_keypoints):
    """If *ckpt* is a ``.pth`` file, convert it to a sibling Orbax dir and
    return that dir; otherwise return *ckpt* unchanged (assumed to already be
    an Orbax checkpoint directory, e.g. one produced by
    ``convert_efficienttrack_pth``/``convert_v2vnet_pth`` ahead of time)."""
    if not str(ckpt).endswith(".pth"):
        return ckpt
    out_dir = str(ckpt)[: -len(".pth")] + f"_{kind}_orbax"
    if kind == "efficienttrack":
        from jarvis_jax.convert.load_efficienttrack import convert_efficienttrack_pth
        convert_efficienttrack_pth(str(ckpt), num_joints=num_keypoints,
                                    out_dir=out_dir, strip_prefix="effTrack.")
    else:  # kind == "v2vnet"
        from jarvis_jax.convert.load_v2vnet_torch import convert_v2vnet_pth
        convert_v2vnet_pth(str(ckpt), in_ch=num_keypoints, out_ch=num_keypoints,
                            out_dir=out_dir)
    return out_dir


def load_inference_model(vitpose_ckpt, v2v_final_dir, *, sharpen=3.0, num_keypoints=50,
                          front_end="vitpose", fusion_mode="none",
                          gate_temperature=1.0, gate_floor=0.0,
                          efficienttrack_ckpt=None):
    """Load a frozen 2-D front-end + V2VNet inference model, replicated across the device mesh.

    Args:
        vitpose_ckpt: Path to the Orbax ViTPose checkpoint directory. Only
            used when ``front_end='vitpose'`` (the default); ignored (may be
            ``None``) when ``front_end='efficienttrack'``.
        v2v_final_dir: Path to the Orbax V2VNet checkpoint directory (or a
            ``.pth`` HybridNet checkpoint, auto-converted). Same directory
            works for either front-end -- the V2VNet weights are independent
            of which 2-D front-end feeds them.
        sharpen: Soft-argmax sharpening exponent (default 3.0, center-bias fix).
        num_keypoints: Number of keypoints (default 50).
        front_end: ``'vitpose'`` (default, byte-identical to the pre-existing
            behavior) or ``'efficienttrack'`` (opt-in: the faithful JAX port
            of the PyTorch HybridNetBackbone's 2-D backbone, loaded from
            ``efficienttrack_ckpt``). See task-9-brief.md.
        fusion_mode: SAM3 mask-fusion mode forwarded to ``HybridNet3D``
            ('none'/'carve'/'input_mask'); default 'none' never touches masks
            (byte-identical guard). See :func:`predict_batch`'s ``masks`` arg.
        gate_temperature, gate_floor: ``fusion_mode='carve'`` gate params.
        efficienttrack_ckpt: Path to the Orbax EfficientTrack checkpoint
            directory (or a ``.pth`` HybridNet checkpoint, auto-converted via
            ``convert_efficienttrack_pth`` with ``strip_prefix="effTrack."``).
            Required when ``front_end='efficienttrack'``.

    Returns:
        A :class:`~jarvis_jax.hybridnet.model.HybridNet3D` with params replicated
        across the data-parallel mesh.  ``model._mesh`` holds the
        :class:`jax.sharding.Mesh` for use in :func:`predict_batch`.
    """
    if front_end not in _FRONT_ENDS:
        raise ValueError(f"front_end must be one of {_FRONT_ENDS}, got {front_end!r}")

    frontend_channels = None  # None => predict_batch passes crops through unsliced

    if front_end == "vitpose":
        # --- Exactly the pre-existing construction path (byte-identical). ---
        vit_cfg = ViTPoseConfig(num_keypoints=num_keypoints)
        front = load_vitpose(vitpose_ckpt, vit_cfg)

        v2v = V2VNet(num_keypoints, num_keypoints, rngs=nnx.Rngs(0))
        gdef, state = nnx.split(v2v)
        ckptr = ocp.StandardCheckpointer()
        try:
            state = ckptr.restore(v2v_final_dir, state)
        except TypeError:
            state = ckptr.restore(v2v_final_dir, args=ocp.args.StandardRestore(state))
        v2v = nnx.merge(gdef, state)
    else:
        # --- Opt-in EfficientTrack front-end (Task 9). ---
        if efficienttrack_ckpt is None:
            raise ValueError(
                "front_end='efficienttrack' requires efficienttrack_ckpt "
                "(Orbax dir, or a .pth HybridNet checkpoint to auto-convert)")
        from jarvis_jax.convert.load_efficienttrack import load_efficienttrack_ckpt
        from jarvis_jax.convert.load_v2vnet_torch import load_v2vnet_ckpt

        et_dir = _maybe_convert_pth(efficienttrack_ckpt, kind="efficienttrack",
                                     num_keypoints=num_keypoints)
        v2v_dir = _maybe_convert_pth(v2v_final_dir, kind="v2vnet",
                                      num_keypoints=num_keypoints)
        # HybridNet's effTrack submodule is 3-channel (RGB only, no SAM3 mask
        # channel) -- see convert/load_efficienttrack.py module docstring and
        # test_hybridnet_efficienttrack_parity.py (faithful-parity note).
        front = load_efficienttrack_ckpt(et_dir, num_joints=num_keypoints, in_channels=3)
        v2v = load_v2vnet_ckpt(v2v_dir, num_keypoints, num_keypoints)
        frontend_channels = 3

    model = HybridNet3D(front, v2v, _Cfg(num_keypoints, sharpen,
                                          fusion_mode=fusion_mode,
                                          gate_temperature=gate_temperature,
                                          gate_floor=gate_floor))

    # Replicate params across the mesh so sharded-batch jit steps work.
    mesh = data_parallel_mesh()
    gd, st = nnx.split(model)
    model = nnx.merge(gd, replicate(st, mesh))
    model._mesh = mesh
    # Consulted by predict_batch to slice crops4's channel axis down to what
    # this front-end actually accepts (None => vitpose, unsliced/unchanged).
    model._frontend_channels = frontend_channels
    return model


def _make_steps():
    # Split the forward into two jitted stages with a host barrier between them.
    # A single fused jit of ViTPose+reproject+V2VNet needs ~33 GiB at J=250 and
    # OOMs a 48 GB GPU; separate jits each peak at their own (much smaller) max
    # and free between, while the (B,J,48^3) volume handoff is <1 GB.
    from jarvis_jax.hybridnet.model import soft_argmax_3d

    @nnx.jit
    def step_vol(model, crops4, center3D, centerHM, cameraMatrices, masks=None):
        # Front-end -> pad -> reproject -> /255  ->  (B, J, 48, 48, 48).
        # `masks` is threaded through unconditionally; reproject_volume (via
        # predict_heatmaps) only actually reads it when
        # fusion_mode=='input_mask' -- a no-op otherwise, including masks=None
        # (byte-identical guard for the default fusion_mode='none' path).
        vol3d = model.reproject_volume(crops4, center3D, centerHM, cameraMatrices,
                                        masks=masks)
        # 'carve': gate the pre-V2VNet volume by cross-camera SAM3 mask
        # consistency (mirrors HybridNet3D.__call__'s post-reproject_volume
        # step, Task 6/7 -- predict_batch's split-jit steps call
        # reproject_volume/v2vnet directly instead of __call__, so this branch
        # is replicated here to keep parity). fusion_mode=='none' never
        # reaches this branch regardless of masks (byte-identical guard).
        if model.fusion_mode == "carve" and masks is not None:
            consistency = mask_consistency_volume(
                masks, center3D, centerHM, cameraMatrices,
                grid_size=48, heatmap_size=226,
            )
            vol3d = soft_gate(vol3d, consistency,
                               temperature=model.gate_temperature,
                               floor=model.gate_floor)
        return vol3d

    @nnx.jit
    def step_3d(model, vol3d, center3D):
        v = jnp.transpose(vol3d, (0, 2, 3, 4, 1))          # (B,48,48,48,J)
        v = model.v2vnet(v, use_running_average=True)
        v = jnp.transpose(v, (0, 4, 1, 2, 3))
        v = jax.nn.softplus(v)
        pts, conf = soft_argmax_3d(v, grid_spacing=1, roi_cube=48, sharpen=model.sharpen)
        return pts + center3D[:, None, :], conf

    return step_vol, step_3d


_STEP_VOL, _STEP_3D = _make_steps()


def predict_batch(model, crops4, centerHM, cameraMatrices, masks=None):
    """Run sharded 3D inference on a batch of crops.

    Args:
        model: A :class:`~jarvis_jax.hybridnet.model.HybridNet3D` returned by
            :func:`load_inference_model`.
        crops4: ``(B, nc, 448, 448, 4)`` uint8 numpy array.  B must be
            divisible by the device count. The mask channel (index 3) is
            always used for ``center3D`` estimation regardless of front-end;
            it is then sliced away before the front-end forward when the
            selected front-end expects fewer channels (e.g. the
            EfficientTrack front-end loaded from a HybridNet checkpoint is
            RGB-only -- see ``load_inference_model``'s ``model._frontend_channels``).
        centerHM: ``(B, nc, 2)`` float32 crop centres in full-image pixels.
        cameraMatrices: ``(B, nc, 4, 3)`` float32 DLT projection matrices.
        masks: Optional per-camera SAM3 silhouette masks for the mask-fusion
            path (see ``HybridNet3D.__call__``'s ``masks`` arg for the
            per-``fusion_mode`` shape contract). Ignored entirely (never even
            sharded/passed to the jitted steps) when ``None`` or when
            ``model.fusion_mode == 'none'`` -- the mask-free call is
            byte-identical to the pre-fusion behavior.

    Returns:
        kp3d:    ``(B, J, 3)`` float32 world-space 3-D joints (J = num_keypoints).
        conf:    ``(B, J)`` float32 per-joint confidence in [0, 1].
        center3D: ``(B, 3)`` float32 lattice-snapped 3D centres from SAM3 masks.
    """
    # center3D always uses the FULL 4-channel crops4 (mask in channel 3),
    # regardless of what the front-end itself consumes.
    center3D, _ = estimate_center3d_from_masks(
        crops4, np.asarray(centerHM), np.asarray(cameraMatrices)
    )
    mesh = model._mesh
    frontend_channels = getattr(model, "_frontend_channels", None)
    crops_for_model = np.asarray(crops4)
    if frontend_channels is not None:
        crops_for_model = crops_for_model[..., :frontend_channels]
    cr = shard_batch(jnp.asarray(crops_for_model), mesh)
    c3 = shard_batch(jnp.asarray(center3D), mesh)
    chm = shard_batch(jnp.asarray(np.asarray(centerHM), dtype=jnp.float32), mesh)
    cam = shard_batch(jnp.asarray(np.asarray(cameraMatrices), dtype=jnp.float32), mesh)
    use_masks = masks is not None and getattr(model, "fusion_mode", "none") != "none"
    mk = shard_batch(jnp.asarray(np.asarray(masks), dtype=jnp.float32), mesh) if use_masks else None
    # Stage 1: front-end + reproject -> volume; host round-trip frees stage-1 memory.
    vol = np.asarray(_STEP_VOL(model, cr, c3, chm, cam, mk))
    vol = shard_batch(jnp.asarray(vol), mesh)
    # Stage 2: V2VNet + soft-argmax -> 3-D joints.
    kp3d, conf = _STEP_3D(model, vol, c3)
    return np.asarray(kp3d), np.asarray(conf), center3D
