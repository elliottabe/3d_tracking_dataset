"""KeypointDetect training: optimizer, jitted train step, MPJPE eval."""
import dataclasses

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from jarvis_jax.data.device import normalize_image, render_heatmaps
from jarvis_jax.train.losses import heatmap_mse, mask_containment
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints, mpjpe


@dataclasses.dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    total_steps: int = 2000
    batch_size: int = 8
    mask_weight: float = 0.0
    mask_dilate: int = 0        # dilate the mask (px) before the containment penalty (0 = off)
    # Backbone learning-rate multiplier (full fine-tune with a smaller LR on the
    # MAE-pretrained ViT than on the fresh decoder head — ViTPose-standard, more
    # stable than a single LR). 1.0 = uniform LR. Everything still trains, so the
    # zero-initialised 4th (SAM-mask) patch-embed channel keeps learning.
    backbone_lr_mult: float = 0.1
    seed: int = 0


def _param_labels(params):
    """Label each Param leaf 'backbone' (ViT) or 'head' (decoder) by its path."""
    return jax.tree_util.tree_map_with_path(
        lambda path, _: "backbone"
        if "backbone" in jax.tree_util.keystr(path) else "head",
        params)


def make_optimizer(model, cfg):
    """AdamW with warmup+cosine schedule, two LR groups: the backbone trains at
    ``cfg.lr * cfg.backbone_lr_mult`` and the decoder head at ``cfg.lr``."""
    decay_steps = max(cfg.total_steps, cfg.warmup_steps + 1)

    def sched(peak):
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=peak, warmup_steps=cfg.warmup_steps,
            decay_steps=decay_steps, end_value=0.0)

    head_tx = optax.adamw(sched(cfg.lr), weight_decay=cfg.weight_decay)
    bb_tx = optax.adamw(
        sched(cfg.lr * cfg.backbone_lr_mult), weight_decay=cfg.weight_decay)
    tx = optax.multi_transform(
        {"backbone": bb_tx, "head": head_tx}, _param_labels)
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def make_train_step(mask_weight, aug_params=None, lr_swap=None, heatmap_size=224,
                    sigma=7.0, joint_weight=None, mask_dilate=0):
    """Return an nnx.jit train step. When aug_params.enabled, the batch is
    augmented on-device (using the per-step `key`) before normalize/render.

    ``sigma`` (scalar or per-channel ``(K,)``) sets the target Gaussian width, and
    ``joint_weight`` (optional ``(K,)``) re-weights each channel's loss — used by
    the CSE wing fine-tune to sharpen + emphasise the densely-packed wing verts.
    ``mask_dilate`` (static int) grows the SAM mask before the containment
    penalty (Phase 5 edge slack); captured as a Python int at call time like
    ``mask_weight``, so the jitted step recompiles per distinct value."""
    from jarvis_jax.data.augment import augment_batch, AugParams
    mw = float(mask_weight)
    md = int(mask_dilate)
    ap = aug_params if aug_params is not None else AugParams(enabled=False)
    if ap.enabled and lr_swap is None:
        raise ValueError("augmentation enabled but lr_swap is None")
    swap = jnp.asarray(lr_swap) if lr_swap is not None else None
    sig = jnp.asarray(sigma, dtype=jnp.float32)
    jw = None if joint_weight is None else jnp.asarray(joint_weight, dtype=jnp.float32)

    def loss_fn(model, img4_u8, kp_xy, vis):
        img = normalize_image(img4_u8)
        hm = render_heatmaps(kp_xy, vis, heatmap_size=heatmap_size, sigma=sig)
        pred = model(img, use_running_average=False)
        loss = heatmap_mse(pred, hm, vis, joint_weight=jw)
        if mw > 0.0:
            mask = img[..., 3]
            mask224 = jax.image.resize(
                mask, (mask.shape[0], pred.shape[1], pred.shape[2]), method="nearest")
            loss = loss + mw * mask_containment(pred, mask224, dilate=md)
        return loss

    @nnx.jit
    def step(model, optimizer, key, img4_u8, kp_xy, vis):
        if ap.enabled:
            img4_u8, kp_xy, vis = augment_batch(
                key, img4_u8, kp_xy, vis, ap, swap, heatmap_size)
        loss, grads = nnx.value_and_grad(loss_fn)(model, img4_u8, kp_xy, vis)
        optimizer.update(model, grads)
        return loss

    return step


def eval_mpjpe(model, ds, batch_size, *, in_size=448):
    """Average MPJPE over the dataset (single device). GT keypoints are the true
    annotation coords (kp_xy scaled to in_size), not a decode of GT heatmaps."""
    from jarvis_jax.data.v3 import batches
    model.eval()
    scale = in_size / float(ds.heatmap_size)
    total, count = 0.0, 0
    for img4_u8, kp_xy, vis in batches(ds, batch_size, shuffle=False,
                                       drop_last=False):
        img = normalize_image(jnp.asarray(img4_u8))
        pred = model(img, use_running_average=True)
        pk = heatmaps_to_keypoints(pred, in_size=in_size)
        gk = jnp.asarray(kp_xy) * scale
        n = int(vis.sum())
        if n == 0:
            continue
        total += float(mpjpe(pk, gk, jnp.asarray(vis))) * n
        count += n
    return total / max(count, 1)
