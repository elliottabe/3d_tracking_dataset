"""Batched, sharded JAX 3D inference (frozen ViTPose + run4 v2vNet)."""
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.convert.build_checkpoint import load_vitpose
from jarvis_jax.hybridnet.model import HybridNet3D
from jarvis_jax.hybridnet.v2vnet import V2VNet
from jarvis_jax.sharding import data_parallel_mesh, replicate, shard_batch
from jarvis_jax.geometry.center3d import estimate_center3d_from_masks


class _Cfg:
    """Minimal cfg object HybridNet3D reads (num_keypoints + sharpen)."""
    def __init__(self, num_keypoints, sharpen):
        self.num_keypoints = num_keypoints
        self.sharpen = sharpen


def load_inference_model(vitpose_ckpt, v2v_final_dir, *, sharpen=3.0, num_keypoints=50):
    """Load a frozen ViTPose + V2VNet inference model, replicated across the device mesh.

    Args:
        vitpose_ckpt: Path to the Orbax ViTPose checkpoint directory.
        v2v_final_dir: Path to the Orbax V2VNet checkpoint directory.
        sharpen: Soft-argmax sharpening exponent (default 3.0, center-bias fix).
        num_keypoints: Number of keypoints (default 50).

    Returns:
        A :class:`~jarvis_jax.hybridnet.model.HybridNet3D` with params replicated
        across the data-parallel mesh.  ``model._mesh`` holds the
        :class:`jax.sharding.Mesh` for use in :func:`predict_batch`.
    """
    vit_cfg = ViTPoseConfig()
    vitpose = load_vitpose(vitpose_ckpt, vit_cfg)

    v2v = V2VNet(num_keypoints, num_keypoints, rngs=nnx.Rngs(0))
    gdef, state = nnx.split(v2v)
    ckptr = ocp.StandardCheckpointer()
    try:
        state = ckptr.restore(v2v_final_dir, state)
    except TypeError:
        state = ckptr.restore(v2v_final_dir, args=ocp.args.StandardRestore(state))
    v2v = nnx.merge(gdef, state)

    model = HybridNet3D(vitpose, v2v, _Cfg(num_keypoints, sharpen))

    # Replicate params across the mesh so sharded-batch jit steps work.
    mesh = data_parallel_mesh()
    gd, st = nnx.split(model)
    model = nnx.merge(gd, replicate(st, mesh))
    model._mesh = mesh
    return model


def _make_step():
    @nnx.jit
    def step(model, crops4, center3D, centerHM, cameraMatrices):
        _, kp3d, conf = model(crops4, center3D, centerHM, cameraMatrices,
                              use_running_average=True)
        return kp3d, conf
    return step


_STEP = _make_step()


def predict_batch(model, crops4, centerHM, cameraMatrices):
    """Run sharded 3D inference on a batch of crops.

    Args:
        model: A :class:`~jarvis_jax.hybridnet.model.HybridNet3D` returned by
            :func:`load_inference_model`.
        crops4: ``(B, nc, 448, 448, 4)`` uint8 numpy array.  B must be
            divisible by the device count.
        centerHM: ``(B, nc, 2)`` float32 crop centres in full-image pixels.
        cameraMatrices: ``(B, nc, 4, 3)`` float32 DLT projection matrices.

    Returns:
        kp3d:    ``(B, 50, 3)`` float32 world-space 3-D keypoints.
        conf:    ``(B, 50)`` float32 per-joint confidence in [0, 1].
        center3D: ``(B, 3)`` float32 lattice-snapped 3D centres from SAM3 masks.
    """
    center3D, _ = estimate_center3d_from_masks(
        crops4, np.asarray(centerHM), np.asarray(cameraMatrices)
    )
    mesh = model._mesh
    cr = shard_batch(jnp.asarray(np.asarray(crops4)), mesh)
    c3 = shard_batch(jnp.asarray(center3D), mesh)
    chm = shard_batch(jnp.asarray(np.asarray(centerHM), dtype=jnp.float32), mesh)
    cam = shard_batch(jnp.asarray(np.asarray(cameraMatrices), dtype=jnp.float32), mesh)
    kp3d, conf = _STEP(model, cr, c3, chm, cam)
    return np.asarray(kp3d), np.asarray(conf), center3D
