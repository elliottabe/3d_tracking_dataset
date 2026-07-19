"""EfficientTrack drop-in for the 2D keypoint trainer (train/train.py).

The trainer (make_optimizer/make_train_step/eval_mpjpe) calls the model
GENERICALLY -- ``model(img, use_running_average=...)`` -> (N,224,224,J)
heatmaps -- so any model satisfying that contract works, unchanged. This test
proves EfficientTrack (EfficientNet-b3 + BiFPN, see
jarvis_jax/models/efficienttrack.py) trains through the SAME loop ViTPose
uses, with no trainer changes: a handful of real gradient steps on a tiny
synthetic batch reduce the heatmap-MSE loss, and eval_mpjpe runs end to end.

CPU-only, small num_joints (4, not the real 50) and batch_size<=2 to keep a
real b3+BiFPN forward/backward tractable in a few tens of seconds on CPU.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jarvis_jax.models.efficienttrack import EfficientTrack
from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step, eval_mpjpe

NUM_JOINTS = 4
BATCH = 2
CROP = 448
HEATMAP_SIZE = 224


class _TinyDS:
    """Minimal stand-in for V3Dataset exposing exactly what eval_mpjpe/batches
    read: ``heatmap_size``, ``__len__``, and ``__getitem__`` -> (img4_u8,
    kp_xy, vis) per-sample tuples (see jarvis_jax/data/v3.py)."""

    def __init__(self, imgs, kps, viss, heatmap_size=HEATMAP_SIZE):
        self.imgs, self.kps, self.viss = imgs, kps, viss
        self.heatmap_size = heatmap_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.kps[i], self.viss[i]


def _synthetic_batch(rng, n=BATCH):
    img4_u8 = rng.integers(0, 256, size=(n, CROP, CROP, 4), dtype=np.uint8)
    # kp_xy is in heatmap-coord space (0..HEATMAP_SIZE), per transform_keypoints.
    kp_xy = rng.uniform(0, HEATMAP_SIZE, size=(n, NUM_JOINTS, 2)).astype(np.float32)
    vis = np.ones((n, NUM_JOINTS), dtype=np.float32)
    return img4_u8, kp_xy, vis


def test_efficienttrack_trains_through_shared_loop():
    rng = np.random.default_rng(0)
    model = EfficientTrack(num_joints=NUM_JOINTS, in_channels=4, rngs=nnx.Rngs(0))

    # mask_weight=0.0: keep the loss to plain heatmap MSE (no SAM-mask
    # containment term) -- simplest possible signal for a "does it learn"
    # smoke test; ViTPose's real training config also defaults mask_weight=0.0
    # (see configs/train/vit2d.yaml), so this matches the common case.
    tcfg = TrainConfig(lr=1e-2, warmup_steps=1, total_steps=5, batch_size=BATCH,
                       mask_weight=0.0, backbone_lr_mult=1.0, seed=0)
    opt = make_optimizer(model, tcfg)
    step = make_train_step(tcfg.mask_weight, aug_params=None, lr_swap=None,
                           heatmap_size=HEATMAP_SIZE)

    img4_u8, kp_xy, vis = _synthetic_batch(rng)
    base_key = jax.random.PRNGKey(0)

    losses = []
    n_steps = 5
    for i in range(n_steps):
        loss = float(step(model, opt, jax.random.fold_in(base_key, i),
                          jnp.asarray(img4_u8), jnp.asarray(kp_xy), jnp.asarray(vis)))
        losses.append(loss)

    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    val_ds = _TinyDS(img4_u8, kp_xy, vis)
    val_mpjpe = eval_mpjpe(model, val_ds, BATCH)
    assert np.isfinite(val_mpjpe)
    assert val_mpjpe >= 0.0
