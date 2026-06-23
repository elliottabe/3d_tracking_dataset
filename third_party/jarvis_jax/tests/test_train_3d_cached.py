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


# ---------------------------------------------------------------------------
# (b) Run-config persistence — no GPU
# ---------------------------------------------------------------------------

def test_save_run_config_writes_json_and_appends_history(tmp_path):
    """run_config.json holds the latest config; history appends one line/launch."""
    import json
    from jarvis_jax.train.train_3d_cached import save_run_config

    run_dir = str(tmp_path / "myrun")

    # First launch: 20000-step config.
    cfg1 = {"total_steps": 20000, "batch_size": 64, "lr": 3e-4,
            "laplacian_weight": 0.05, "resumed_from_step": 0}
    path = save_run_config(run_dir, cfg1)
    assert path == os.path.join(run_dir, "run_config.json")
    with open(path) as f:
        latest = json.load(f)
    assert latest == cfg1

    # Resume launch: extended to 30000, resumed from 20000.
    cfg2 = {**cfg1, "total_steps": 30000, "resumed_from_step": 20000}
    save_run_config(run_dir, cfg2)

    # run_config.json reflects the LATEST config (the resume).
    with open(os.path.join(run_dir, "run_config.json")) as f:
        assert json.load(f)["total_steps"] == 30000

    # History keeps BOTH launches, in order.
    hist_path = os.path.join(run_dir, "run_config_history.jsonl")
    with open(hist_path) as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    assert len(lines) == 2
    assert lines[0]["total_steps"] == 20000 and lines[0]["resumed_from_step"] == 0
    assert lines[1]["total_steps"] == 30000 and lines[1]["resumed_from_step"] == 20000
