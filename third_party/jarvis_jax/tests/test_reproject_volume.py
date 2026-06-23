"""Tests for HybridNet3D.reproject_volume (Task 1: cache-point extraction).

Verifies:
- reproject_volume returns (B, 50, 48, 48, 48) — the exact pre-v2vNet volume
- values are finite
- __call__ still returns the documented shapes after the refactor (refactor-safe)
"""
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.hybridnet.v2vnet import V2VNet
from jarvis_jax.hybridnet.model import HybridNet3D


def _toy(cfg):
    vit = ViTPose(cfg, rngs=nnx.Rngs(0))
    v2v = V2VNet(cfg.num_keypoints, cfg.num_keypoints, rngs=nnx.Rngs(1))
    return HybridNet3D(vit, v2v, cfg)


def test_reproject_volume_shape_and_call_consistency():
    cfg = ViTPoseConfig()
    m = _toy(cfg)
    nc = 7
    rng = np.random.RandomState(0)
    crops = jnp.asarray(rng.randint(0, 256, (1, nc, 448, 448, 4), np.uint8))
    c3d = jnp.zeros((1, 3), jnp.float32)
    cHM = jnp.full((1, nc, 2), 224.0, jnp.float32)
    camM = jnp.asarray(rng.randn(1, nc, 4, 3).astype("float32"))
    vol_in = m.reproject_volume(crops, c3d, cHM, camM)
    assert vol_in.shape == (1, 50, 48, 48, 48)
    assert bool(jnp.isfinite(vol_in).all())
    # __call__ must still run and produce the documented shapes (refactor safe)
    vol, pts, conf = m(crops, c3d, cHM, camM, use_running_average=True)
    assert vol.shape == (1, 50, 24, 24, 24) and pts.shape == (1, 50, 3)
