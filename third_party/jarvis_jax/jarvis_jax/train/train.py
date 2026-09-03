"""KeypointDetect training: optimizer, jitted train step, MPJPE eval."""
import dataclasses

import jax
import numpy as np
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
    # Gaussian target sigma (heatmap px) passed through to make_train_step's
    # `sigma` kwarg at the 2D training entrypoint's call site (see
    # jarvis_jax/scripts/train_keypoints.py::run_training). Default matches
    # gaussian_heatmaps'/V3Dataset's 2026-08-29 sharpened default (2.0, not
    # make_train_step's own 7.0 default -- that default is left alone because
    # several OTHER callers (densepose CSE training, tracking/finetune_detector,
    # and multiple unit tests) rely on it implicitly and are out of scope here).
    target_sigma: float = 7.0   # do NOT lower; sigma=2.0 measured a 6.5x regression (see data/transforms.py)
    # Mask-channel ablation (2026-08-31 mask-channel-ablation task): when
    # True, train_keypoints.py wraps every dataset (train/val) in
    # jarvis_jax.data.mask_zero.ZeroMaskDataset, which zeroes the 4th
    # (SAM-mask) input channel of every sample. Model stays in_ch=4 (capacity
    # held fixed) -- only the mask's INFORMATION is removed, both at train
    # and eval time, so a checkpoint trained with this on has never seen a
    # populated mask. Default False -- byte-identical to before this field
    # existed for every other caller of TrainConfig.
    mask_ablation: bool = False
    # Center-channel (instance-cue) ablation arm: when True, train_keypoints.py
    # wraps every dataset (train/val) in
    # jarvis_jax.data.center_channel.CenterChannelDataset, which REPLACES the
    # 4th input channel with a Gaussian at the TARGET fly's own bbox center
    # (in place of the SAM mask), and trains/evaluates through
    # normalize_image_center_channel instead of normalize_image. Mutually
    # exclusive with mask_ablation (train_keypoints.run_training raises if
    # both are set). Default False -- byte-identical to before this field
    # existed for every other caller of TrainConfig.
    center_channel_input: bool = False
    # ---- distractor-aware supervision (2026-09-03; data/distractor.py,
    # docs/benchmark/2026-09-03-maskoff-attention). All OFF by default =
    # byte-identical loss and data stream for every existing run.
    # Hard-negative background: add the mean squared error over the
    # `hardneg_k` worst background pixels per channel, weighted. 0 = off.
    hardneg_k: int = 0
    hardneg_weight: float = 0.0
    # Repulsion: penalise each channel at the OTHER fly's same-part keypoints
    # (footprint-weighted MSE, own foreground excluded). >0 makes
    # train_keypoints wrap the train set in DistractorKeypointDataset.
    repulsion_weight: float = 0.0
    # Probability per draw of gray-filling the other fly's body from its SAM
    # mask, as inference does (DistractorGrayFillDataset). 0 = never (legacy).
    distractor_fill_p: float = 0.0
    # Probability per single-fly draw of pasting a same-camera donor fly into
    # the frame (CopyPasteKeypointDataset); its keypoints become the
    # distractor rows. 0 = off.
    copy_paste_p: float = 0.0


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
                    sigma=7.0, joint_weight=None, mask_dilate=0, normalize_fn=None,
                    *, n_keypoints=None, part_of_k=None, hardneg_k=0,
                    hardneg_weight=0.0, repulsion_weight=0.0):
    """Return an nnx.jit train step. When aug_params.enabled, the batch is
    augmented on-device (using the per-step `key`) before normalize/render.

    ``sigma`` (scalar or per-channel ``(K,)``) sets the target Gaussian width, and
    ``joint_weight`` (optional ``(K,)``) re-weights each channel's loss — used by
    the CSE wing fine-tune to sharpen + emphasise the densely-packed wing verts.
    ``mask_dilate`` (static int) grows the SAM mask before the containment
    penalty (Phase 5 edge slack); captured as a Python int at call time like
    ``mask_weight``, so the jitted step recompiles per distinct value.
    ``normalize_fn`` (defaults to ``jarvis_jax.data.device.normalize_image`` --
    byte-identical to before this parameter existed for every caller that
    does not pass it) selects how the 4th input channel is rescaled; the
    ``center_channel`` ablation arm passes
    ``jarvis_jax.data.device.normalize_image_center_channel`` instead (see
    that function's docstring for why the two differ).

    Distractor-aware supervision (2026-09-03, ``data/distractor.py``): when the
    batch carries ``2*n_keypoints`` keypoint rows, rows ``[n_keypoints:]`` are
    the OTHER fly's keypoints. They are warped/flipped with the rest (with
    ``lr_swap`` extended to cover them and ``n_fit`` keeping them out of the
    affine fit), never rendered as targets, and -- when ``repulsion_weight >
    0`` -- rendered through ``repulsion_footprint(part_of_k)`` into the loss.
    ``hardneg_k``/``hardneg_weight`` are passed straight to ``heatmap_mse``.
    A batch with exactly ``n_keypoints`` rows behaves exactly as before."""
    from jarvis_jax.data.augment import augment_batch, AugParams
    from jarvis_jax.data.distractor import repulsion_footprint
    norm_fn = normalize_fn if normalize_fn is not None else normalize_image
    mw = float(mask_weight)
    md = int(mask_dilate)
    ap = aug_params if aug_params is not None else AugParams(enabled=False)
    if ap.enabled and lr_swap is None:
        raise ValueError("augmentation enabled but lr_swap is None")
    swap = jnp.asarray(lr_swap) if lr_swap is not None else None
    sig = jnp.asarray(sigma, dtype=jnp.float32)
    jw = None if joint_weight is None else jnp.asarray(joint_weight, dtype=jnp.float32)
    K = None if n_keypoints is None else int(n_keypoints)
    rw, hk, hw = float(repulsion_weight), int(hardneg_k), float(hardneg_weight)
    pok = None if part_of_k is None else np.asarray(part_of_k)
    if rw > 0.0 and (K is None or pok is None):
        raise ValueError("repulsion_weight > 0 needs n_keypoints and part_of_k")

    def _split(kp_xy, vis):
        """(target kp, target vis, distractor kp | None, distractor vis | None)."""
        if K is not None and kp_xy.shape[1] > K:
            return kp_xy[:, :K], vis[:, :K], kp_xy[:, K:], vis[:, K:]
        return kp_xy, vis, None, None

    def loss_fn(model, img4_u8, kp_xy, vis):
        kp_t, vis_t, kp_d, vis_d = _split(kp_xy, vis)
        img = norm_fn(img4_u8)
        hm = render_heatmaps(kp_t, vis_t, heatmap_size=heatmap_size, sigma=sig)
        pred = model(img, use_running_average=False)
        rep = None
        if rw > 0.0 and kp_d is not None:
            rep = repulsion_footprint(kp_d, vis_d, pok, heatmap_size=heatmap_size, sigma=sig)
        loss = heatmap_mse(pred, hm, vis_t, joint_weight=jw, hardneg_k=hk,
                           hardneg_weight=hw, rep_fp=rep, rep_weight=rw)
        if mw > 0.0:
            mask = img[..., 3]
            mask224 = jax.image.resize(
                mask, (mask.shape[0], pred.shape[1], pred.shape[2]), method="nearest")
            loss = loss + mw * mask_containment(pred, mask224, dilate=md)
        return loss

    @nnx.jit
    def step(model, optimizer, key, img4_u8, kp_xy, vis):
        if ap.enabled:
            extra = K is not None and kp_xy.shape[1] > K
            swap_all = jnp.concatenate([swap, swap + K]) if extra else swap
            img4_u8, kp_xy, vis = augment_batch(
                key, img4_u8, kp_xy, vis, ap, swap_all, heatmap_size,
                n_fit=K if extra else None)
        loss, grads = nnx.value_and_grad(loss_fn)(model, img4_u8, kp_xy, vis)
        optimizer.update(model, grads)
        return loss

    return step


@nnx.jit
def _eval_forward(model, img):
    return model(img, use_running_average=True)


def eval_mpjpe(model, ds, batch_size, *, in_size=448, normalize_fn=None):
    """Average MPJPE over the dataset (single device). GT keypoints are the true
    annotation coords (kp_xy scaled to in_size), not a decode of GT heatmaps.

    ``normalize_fn`` (default ``jarvis_jax.data.device.normalize_image``, i.e.
    byte-identical to before this parameter existed) mirrors
    ``make_train_step``'s kwarg of the same name -- pass
    ``normalize_image_center_channel`` to evaluate a model trained with the
    ``center_channel`` ablation arm's 4th-channel encoding."""
    from jarvis_jax.data.v3 import batches
    norm_fn = normalize_fn if normalize_fn is not None else normalize_image
    model.eval()
    scale = in_size / float(ds.heatmap_size)
    total, count = 0.0, 0
    for img4_u8, kp_xy, vis in batches(ds, batch_size, shuffle=False,
                                       drop_last=False):
        img = norm_fn(jnp.asarray(img4_u8))
        pred = _eval_forward(model, img)
        pk = heatmaps_to_keypoints(pred, in_size=in_size)
        gk = jnp.asarray(kp_xy) * scale
        n = int(vis.sum())
        if n == 0:
            continue
        total += float(mpjpe(pk, gk, jnp.asarray(vis))) * n
        count += n
    return total / max(count, 1)
