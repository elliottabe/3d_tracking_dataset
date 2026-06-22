"""Tests for the 3D training loop (train_3d.py).

Two tests:
(a) freeze test — CPU-only, fast: after one train step the v2vNet params
    CHANGE but the ViTPose params are UNCHANGED.
(b) smoke gate — GPU + V3 data + ViTPose ckpt: run_training_3d over 8 steps
    returns final_loss < first_loss and finite val_mpjpe_3d.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
VIT = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_8gpu_20260620/final"
gpu = any(d.platform == "gpu" for d in jax.devices())
run = pytest.mark.skipif(
    not (os.path.isdir(ROOT) and os.path.isdir(VIT) and gpu),
    reason="needs V3 data + ViTPose ckpt + GPU")


# ---------------------------------------------------------------------------
# (a) Freeze test — CPU ok, fast
# ---------------------------------------------------------------------------

def test_vitpose_frozen_v2vnet_trains():
    """After one train step: v2vNet params change, ViTPose params unchanged."""
    from flax import nnx
    from jarvis_jax.hybridnet.model import HybridNet3D
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.train.train_3d import (
        HybridNetConfig, make_v2v_optimizer, make_train_step_3d
    )

    cfg = ViTPoseConfig()
    vitpose = ViTPose(cfg, rngs=nnx.Rngs(0))
    v2vnet = V2VNet(cfg.num_keypoints, cfg.num_keypoints, rngs=nnx.Rngs(1))
    model = HybridNet3D(vitpose, v2vnet, cfg)

    tcfg = HybridNetConfig(total_steps=1, batch_size=1, warmup_steps=0)
    opt = make_v2v_optimizer(model, tcfg)

    # Snapshot a param from each submodule before the step
    vit_before = np.array(
        model.vitpose.decoder.up3.deconv.kernel.value)
    v2v_before = np.array(
        model.v2vnet.front_basic.conv.kernel.value)

    # Minimal synthetic batch (1 sample, 2 cameras)
    B, nc, J = 1, 2, cfg.num_keypoints
    rng = np.random.RandomState(42)
    batch = {
        "crops4": jnp.asarray(
            rng.randint(0, 255, (B, nc, 448, 448, 4), dtype=np.uint8)),
        "centerHM": jnp.zeros((B, nc, 2), dtype=jnp.float32),
        "center3D": jnp.zeros((B, 3), dtype=jnp.float32),
        "cameraMatrices": jnp.broadcast_to(
            jnp.eye(4, 3, dtype=jnp.float32)[None, None], (B, nc, 4, 3)),
        "kp3d": jnp.zeros((B, J, 3), dtype=jnp.float32),
        "vis": jnp.zeros((B, J), dtype=bool),
    }

    # ei/ej empty (no skeleton for this test)
    ei = np.zeros((0,), dtype=np.int32)
    ej = np.zeros((0,), dtype=np.int32)
    step = make_train_step_3d(
        laplacian_weight=0.0, ei=ei, ej=ej,
        grid_spacing=1, roi_cube=48, sigma=2.0)

    loss = step(model, opt, batch)
    assert jnp.isfinite(loss), f"loss is not finite: {loss}"

    vit_after = np.array(
        model.vitpose.decoder.up3.deconv.kernel.value)
    v2v_after = np.array(
        model.v2vnet.front_basic.conv.kernel.value)

    vit_delta = float(np.abs(vit_after - vit_before).max())
    v2v_delta = float(np.abs(v2v_after - v2v_before).max())

    print(f"\nFreeze test: vitpose_delta={vit_delta:.6e}  v2vnet_delta={v2v_delta:.6e}")
    assert vit_delta == 0.0, (
        f"ViTPose params changed after train step (delta={vit_delta:.2e}). "
        f"Freeze is broken.")
    assert v2v_delta > 0.0, (
        f"V2VNet params did NOT change after train step. "
        f"Either loss is trivially zero or optimizer is broken.")


# ---------------------------------------------------------------------------
# (b) Smoke gate — GPU + V3 data + ViTPose ckpt
# ---------------------------------------------------------------------------

@run
def test_3d_training_reduces_loss():
    from jarvis_jax.train.train_3d import HybridNetConfig, run_training_3d
    import tempfile

    out = tempfile.mkdtemp()
    res = run_training_3d(
        ROOT,
        out_dir=out + "/final",
        vitpose_ckpt=VIT,
        tcfg=HybridNetConfig(total_steps=8, batch_size=2, warmup_steps=1),
        save_every=8,
        val_recording="2026_05_27_11_56_05",
    )
    print(f"\nSmoke: first_loss={res['first_loss']:.5f}  "
          f"final_loss={res['final_loss']:.5f}  "
          f"val_mpjpe_3d={res['val_mpjpe_3d']:.3f}")
    assert res["final_loss"] < res["first_loss"], (
        f"Loss did not decrease: first={res['first_loss']:.5f} "
        f"final={res['final_loss']:.5f}")
    assert res["val_mpjpe_3d"] >= 0.0, (
        f"val_mpjpe_3d is negative: {res['val_mpjpe_3d']}")
