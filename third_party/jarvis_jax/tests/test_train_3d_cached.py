"""Tests for the cached v2vNet training loop (train_3d_cached.py).

Tests:
  (a) test_cached_step_reduces_loss — GPU; deterministic loss decrease on a
      tiny synthetic batch of random volumes + one visible joint.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

gpu = any(d.platform == "gpu" for d in jax.devices())
needs_gpu = pytest.mark.skipif(not gpu, reason="needs GPU")


# ---------------------------------------------------------------------------
# (a) Loss-decrease test — GPU required
# ---------------------------------------------------------------------------

@needs_gpu
def test_cached_step_reduces_loss():
    """Run ~30 cached steps on synthetic data; loss must decrease and be finite."""
    from flax import nnx
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.train.train_3d_cached import (
        CachedConfig, make_v2v_optimizer, make_cached_step,
    )

    nd = jax.device_count()
    rng = np.random.RandomState(0)

    # Synthetic cache: random volumes, GT at grid centre (world 0), one visible joint
    B = nd  # one sample per device
    vols = jnp.asarray(rng.rand(B, 50, 48, 48, 48).astype("float16"))
    kp3d = jnp.zeros((B, 50, 3), jnp.float32)      # GT at world origin = cube centre
    c3d = jnp.zeros((B, 3), jnp.float32)
    vis = jnp.zeros((B, 50), bool).at[:, 0].set(True)  # only joint 0 is visible

    batch = {"volumes": vols, "kp3d": kp3d, "center3D": c3d, "vis": vis}

    # Build model + optimizer
    v2v = V2VNet(50, 50, rngs=nnx.Rngs(0))
    tcfg = CachedConfig(total_steps=30, lr=1e-3, warmup_steps=2)
    opt = make_v2v_optimizer(v2v, tcfg)

    # Empty skeleton edges (laplacian off)
    ei = np.zeros((0,), np.int32)
    ej = np.zeros((0,), np.int32)
    step = make_cached_step(0.0, ei, ej, 1, 48, 2.0)

    losses = [float(step(v2v, opt, batch)) for _ in range(30)]

    print(f"\nloss trajectory (first 5): {losses[:5]}")
    print(f"loss trajectory (last 5): {losses[-5:]}")

    assert np.isfinite(losses).all(), f"Non-finite loss encountered: {losses}"
    assert losses[-1] < losses[0], (
        f"Loss did NOT decrease: first={losses[0]:.6f}  last={losses[-1]:.6f}")
