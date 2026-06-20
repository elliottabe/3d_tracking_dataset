"""KeypointDetect training: optimizer, jitted train step, MPJPE eval."""
import dataclasses

import jax
import jax.numpy as jnp
import optax
from flax import nnx

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


def make_train_step(mask_weight):
    """Return an nnx.jit train step with `mask_weight` baked in as a constant."""
    mw = float(mask_weight)

    def loss_fn(model, img4, hm, vis):
        pred = model(img4, use_running_average=False)
        loss = heatmap_mse(pred, hm, vis)
        if mw > 0.0:
            mask = img4[..., 3]
            mask224 = jax.image.resize(
                mask, (mask.shape[0], pred.shape[1], pred.shape[2]),
                method="nearest")
            loss = loss + mw * mask_containment(pred, mask224)
        return loss

    @nnx.jit
    def step(model, optimizer, img4, hm, vis):
        loss, grads = nnx.value_and_grad(loss_fn)(model, img4, hm, vis)
        optimizer.update(model, grads)
        return loss

    return step


def eval_mpjpe(model, ds, batch_size, *, in_size=448):
    """Average MPJPE over the dataset (single device)."""
    from jarvis_jax.data.v3 import batches
    model.eval()
    total, count = 0.0, 0
    for img4, hm, vis in batches(ds, batch_size, shuffle=False, drop_last=False):
        pred = model(jnp.asarray(img4), use_running_average=True)
        pk = heatmaps_to_keypoints(pred, in_size=in_size)
        gk = heatmaps_to_keypoints(jnp.asarray(hm), in_size=in_size)
        n = int(vis.sum())
        if n == 0:
            continue
        total += float(mpjpe(pk, gk, jnp.asarray(vis))) * n
        count += n
    return total / max(count, 1)
