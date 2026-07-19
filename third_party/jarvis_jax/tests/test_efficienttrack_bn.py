"""EfficientTrack-BN: the ImageNet-pretrainable (standard BatchNorm)
EfficientNet-b3 backbone variant of EfficientTrack, for a fair
ViT-vs-EfficientNet keypoint-backbone comparison.

Covers:
  * forward shape ((B,448,448,4) -> (B,224,224,J)) for the ImageNet-loaded,
    4-channel-inflated model built by ``build_efficienttrack_bn_imagenet``.
  * the stem's 4th (SAM-mask) input channel is zero-initialized, so a forward
    pass is invariant to that channel's contents AT INIT -- proving the 3->4
    inflation mirrors the ViT MAE zero-init pattern (a zero mask channel is a
    no-op).
  * BatchNorm running-stat (``nnx.BatchStat``) propagation THROUGH the jitted
    2D train step (``make_train_step``): a few real gradient steps must (a)
    change a Param (proves training is happening) AND (b) change the
    backbone's BatchNorm running mean/var (proves BatchStat state survives
    the `@nnx.jit` step -- NOT reset/discarded each call). Also exercises
    `eval_mpjpe` (use_running_average=True) end to end.
  * the existing InstanceNorm EfficientTrack import + a shape smoke, to prove
    nothing on that path regressed.

CPU-only; b3 is heavy, so batch/step counts are kept small (batch<=2, <=3
train steps, num_joints=4 not the real 50) -- same tradeoff as
tests/test_train_efficienttrack_2d.py.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

FIX = os.path.join(os.path.dirname(__file__), "..", "jarvis_jax", "convert",
                   "fixtures", "effnet_b3_imagenet.npz")
pytestmark = pytest.mark.skipif(
    not os.path.exists(FIX),
    reason="generate fixture (export_effnet_b3_imagenet_fixture.py)")

NUM_JOINTS = 4
CROP = 448
HEATMAP_SIZE = 224


def _build(num_joints=NUM_JOINTS, in_channels=4, seed=0):
    from jarvis_jax.models.efficienttrack import build_efficienttrack_bn_imagenet
    return build_efficienttrack_bn_imagenet(
        num_joints, FIX, in_channels=in_channels, rngs=nnx.Rngs(seed))


def test_efficienttrack_bn_forward_shape():
    model = _build()
    x = jnp.zeros((1, CROP, CROP, 4), dtype=jnp.float32)
    out = model(x, use_running_average=True)
    assert out.shape == (1, HEATMAP_SIZE, HEATMAP_SIZE, NUM_JOINTS)
    assert np.all(np.isfinite(np.asarray(out)))


def test_zero_mask_channel_is_noop_at_init():
    """Proves the stem's 4th channel is zero-initialized: with identical RGB,
    swapping the mask channel from all-zero to random noise must not change
    the output AT INIT (before any training touches the stem weights)."""
    model = _build()
    rgb = jax.random.normal(jax.random.PRNGKey(1), (1, CROP, CROP, 3))
    mask_zero = jnp.zeros((1, CROP, CROP, 1))
    mask_rand = jax.random.normal(jax.random.PRNGKey(2), (1, CROP, CROP, 1)) * 5.0

    x_zero_mask = jnp.concatenate([rgb, mask_zero], axis=-1)
    x_rand_mask = jnp.concatenate([rgb, mask_rand], axis=-1)

    out_zero = model(x_zero_mask, use_running_average=True)
    out_rand = model(x_rand_mask, use_running_average=True)

    diff = np.abs(np.asarray(out_zero) - np.asarray(out_rand))
    print(f"zero-mask no-op: max_abs_diff={diff.max():.3e}")
    np.testing.assert_allclose(np.asarray(out_zero), np.asarray(out_rand),
                               atol=1e-5, rtol=1e-5)


def test_bn_running_stats_and_params_update_through_train_step():
    """The load-bearing regression guard for BN-state training support:
    BatchStat (running mean/var) must change across `make_train_step` calls,
    proving nnx.jit's default reference-graph-update semantics thread
    BatchStat (not just Param, which the optimizer explicitly touches) back
    out of the jitted step -- this is NOT something `make_train_step` has to
    do anything special for (nnx.jit updates the full object graph reachable
    from its Module/Optimizer args by default), but it is exactly the kind of
    thing that silently breaks if a future refactor donates/copies state
    incorrectly, so it is asserted here explicitly rather than assumed."""
    from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step, eval_mpjpe

    model = _build(num_joints=NUM_JOINTS)
    tcfg = TrainConfig(lr=1e-2, warmup_steps=1, total_steps=3, batch_size=1,
                       mask_weight=0.0, backbone_lr_mult=1.0, seed=0)
    opt = make_optimizer(model, tcfg)
    step = make_train_step(tcfg.mask_weight, aug_params=None, lr_swap=None,
                           heatmap_size=HEATMAP_SIZE)

    bn = model.backbone.stem_bn
    mean_before = np.asarray(bn.mean.value).copy()
    var_before = np.asarray(bn.var.value).copy()
    w_before = np.asarray(model.backbone.stem_conv.kernel.value).copy()

    rng = np.random.default_rng(0)
    img4_u8 = rng.integers(0, 256, size=(1, CROP, CROP, 4), dtype=np.uint8)
    kp_xy = rng.uniform(0, HEATMAP_SIZE, size=(1, NUM_JOINTS, 2)).astype(np.float32)
    vis = np.ones((1, NUM_JOINTS), dtype=np.float32)

    base_key = jax.random.PRNGKey(0)
    losses = []
    for i in range(tcfg.total_steps):
        loss = float(step(model, opt, jax.random.fold_in(base_key, i),
                          jnp.asarray(img4_u8), jnp.asarray(kp_xy), jnp.asarray(vis)))
        losses.append(loss)
    assert all(np.isfinite(losses)), losses

    mean_after = np.asarray(bn.mean.value)
    var_after = np.asarray(bn.var.value)
    w_after = np.asarray(model.backbone.stem_conv.kernel.value)

    mean_diff = np.abs(mean_after - mean_before).max()
    var_diff = np.abs(var_after - var_before).max()
    w_diff = np.abs(w_after - w_before).max()
    print(f"stem_bn.mean max_abs_diff={mean_diff:.3e}  "
          f"stem_bn.var max_abs_diff={var_diff:.3e}  "
          f"stem_conv.kernel (Param) max_abs_diff={w_diff:.3e}")

    assert mean_diff > 0.0, "BatchStat mean did not change -- BN state not propagating through nnx.jit step"
    assert var_diff > 0.0, "BatchStat var did not change -- BN state not propagating through nnx.jit step"
    assert w_diff > 0.0, "Param (stem_conv kernel) did not change -- optimizer not training"

    class _TinyDS:
        def __init__(self, imgs, kps, viss, heatmap_size=HEATMAP_SIZE):
            self.imgs, self.kps, self.viss = imgs, kps, viss
            self.heatmap_size = heatmap_size

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            return self.imgs[i], self.kps[i], self.viss[i]

    val_ds = _TinyDS(img4_u8, kp_xy, vis)
    val_mpjpe = eval_mpjpe(model, val_ds, 1)
    assert np.isfinite(val_mpjpe)
    assert val_mpjpe >= 0.0


def test_instancenorm_efficienttrack_import_and_shape_smoke():
    """Regression guard: the existing InstanceNorm EfficientTrack path must be
    completely unaffected by the EfficientTrackBN addition."""
    from jarvis_jax.models.efficienttrack import EfficientTrack

    model = EfficientTrack(num_joints=NUM_JOINTS, in_channels=4, rngs=nnx.Rngs(0))
    x = jnp.zeros((1, CROP, CROP, 4), dtype=jnp.float32)
    out = model(x)
    assert out.shape == (1, HEATMAP_SIZE, HEATMAP_SIZE, NUM_JOINTS)
