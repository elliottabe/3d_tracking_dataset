"""Train a 2D KeypointDetect model on red_data_unified_V3 (JAX, data-parallel).

Drives ViTPose (default), EfficientTrack (EfficientNet-b3 + BiFPN, InstanceNorm,
random-init), OR EfficientTrack-BN (same BiFPN+head, but a STANDARD BatchNorm
EfficientNet-b3 backbone warm-started from ImageNet) through the identical
train loop (make_optimizer/make_train_step/eval_mpjpe), so the architectures
can be compared head-to-head -- see `model_node.get("arch", ...)` in
`main_from_cfg`. ViTPose does a full fine-tune from the MAE-pretrained
backbone with a lower backbone LR (see TrainConfig.backbone_lr_mult);
EfficientTrack has no such checkpoint here and always starts from random init;
EfficientTrack-BN's backbone IS pretrained (ImageNet) so it uses the same
lower-LR backbone treatment via `.backbone` path-matching, while its BiFPN+head
still train from scratch like EfficientTrack's. Everything trains, so the 4th
(SAM-mask) input channel (patch-embed for ViTPose, stem conv for the
EfficientNet variants) learns from its zero-init no-op start. End-to-end
localization evidence comes from a real run of this script (the synthetic
overfit gate was dropped as flaky — see tests/test_train_step.py).

CLI (Hydra; see configs/):
    python -m jarvis_jax.scripts.train_keypoints \\
        run_id=myrun train=vit2d model=vitpose paths=hyak
    python -m jarvis_jax.scripts.train_keypoints \\
        run_id=myrun train=vit2d model=efficienttrack paths=hyak
    python -m jarvis_jax.scripts.train_keypoints \\
        run_id=myrun train=vit2d model=efficienttrack_bn paths=hyak
"""
import json
import os

import hydra
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass, run_dir_for

register_resolvers()

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.models.efficienttrack import EfficientTrack, build_efficienttrack_bn_imagenet
from jarvis_jax.convert.build_checkpoint import build
from jarvis_jax.data.prefetch import prefetch
from jarvis_jax.data.v3 import V3Dataset, batches
from jarvis_jax.sharding import data_parallel_mesh, replicate
from jarvis_jax.train.train import (
    TrainConfig, make_optimizer, make_train_step, eval_mpjpe,
)
from jarvis_jax.data.augment import build_lr_swap, AugParams
from jarvis_jax.train.checkpoint import make_manager, save_step, restore_latest

DEFAULT_MAE_NPZ = "/gscratch/portia/eabe/data/Johnson_lab/mae_vitb.npz"
# Committed torchvision efficientnet_b3 ImageNet fixture (see
# jarvis_jax/models/effnet_b3_std.py / convert/export_effnet_b3_imagenet_fixture.py),
# resolved relative to the package so it works from any checkout.
DEFAULT_EFFNET_B3_IMAGENET_NPZ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "convert", "fixtures", "effnet_b3_imagenet.npz")


def _epochs(ds, batch_size, base_seed, weights=None, num_workers=8):
    """Infinite stream of batches, reshuffled each epoch (or weighted-resampled
    each epoch when `weights` is given)."""
    epoch = 0
    while True:
        yield from batches(ds, batch_size, shuffle=True, seed=base_seed + epoch,
                           weights=weights, num_workers=num_workers)
        epoch += 1


def run_training(root, *, out_dir, mae_npz=DEFAULT_MAE_NPZ, tcfg=None,
                 vitpose_cfg=None, aug_params=None, arch="vitpose",
                 effnet_b3_imagenet_npz=None,
                 val_recording="2026_05_27_11_56_05",
                 log_every=50, eval_every=500, smoke=False,
                 ckpt_dir=None, save_every=500, oversample=None, num_workers=8):
    cfg = vitpose_cfg if vitpose_cfg is not None else ViTPoseConfig()
    tcfg = tcfg or TrainConfig()
    if smoke:
        # batch must be divisible by the number of devices for data-parallel sharding
        n_devices = len(jax.devices())
        tcfg.total_steps, tcfg.batch_size, eval_every, log_every = 4, n_devices, 4, 1

    n_dev = len(jax.devices())
    if tcfg.batch_size % n_dev != 0:
        raise ValueError(
            f"batch_size ({tcfg.batch_size}) must be divisible by the JAX device "
            f"count ({n_dev}) for data-parallel sharding.")

    if arch == "efficienttrack":
        # EfficientTrack (EfficientNet-b3 + BiFPN) has no MAE-style pretrained
        # checkpoint in this pipeline -- it is always randomly initialised.
        # It still exposes a `.backbone` submodule, so `make_optimizer`'s
        # backbone/head LR split (_param_labels) applies unchanged.
        model = EfficientTrack(num_joints=cfg.num_keypoints, in_channels=cfg.in_ch,
                               rngs=nnx.Rngs(tcfg.seed))
    elif arch == "efficienttrack_bn":
        # EfficientTrack-BN: same BiFPN+head as `efficienttrack`, but driven by
        # the STANDARD (BatchNorm) EfficientNet-b3 backbone warm-started from
        # ImageNet -- gives the EfficientNet side of the ViT-vs-EfficientNet
        # comparison a pretrained init too (unlike plain `efficienttrack`,
        # always random-init). `.backbone` is still the LR-split path
        # `_param_labels` matches on, unchanged.
        npz_path = effnet_b3_imagenet_npz or DEFAULT_EFFNET_B3_IMAGENET_NPZ
        model = build_efficienttrack_bn_imagenet(
            cfg.num_keypoints, npz_path, in_channels=cfg.in_ch,
            rngs=nnx.Rngs(tcfg.seed))
    elif mae_npz and os.path.exists(mae_npz):
        model = build(mae_npz, cfg)          # MAE-pretrained backbone
    else:
        print(f"WARNING: MAE npz not found ({mae_npz}); using RANDOM init "
              f"(real training should use MAE init).")
        model = ViTPose(cfg, rngs=nnx.Rngs(tcfg.seed))

    opt = make_optimizer(model, tcfg)

    mngr = make_manager(ckpt_dir) if ckpt_dir else None
    start = 0
    if mngr is not None:
        model, opt, start = restore_latest(mngr, model, opt)
        if start:
            print(f"resuming from checkpoint at step {start}")

    names = json.load(open(
        os.path.join(root, "annotations", "instances_train.json")))["keypoint_names"]
    lr_swap = build_lr_swap(names)
    aug = aug_params if aug_params is not None else AugParams()
    step = make_train_step(tcfg.mask_weight, aug, lr_swap, heatmap_size=cfg.heatmap_size,
                           mask_dilate=tcfg.mask_dilate)
    base_key = jax.random.PRNGKey(tcfg.seed)
    mesh = data_parallel_mesh()

    # Replicate params + optimizer state across the mesh so the data-sharded
    # jit step has consistent device placement. Critical on resume: Orbax
    # restore commits restored arrays to device 0, which clashes with the
    # sharded batch otherwise ("Received incompatible devices"). Harmless on a
    # fresh run (uncommitted arrays would be auto-replicated anyway).
    gdef_m, st_m = nnx.split(model)
    model = nnx.merge(gdef_m, replicate(st_m, mesh))
    gdef_o, st_o = nnx.split(opt)
    opt = nnx.merge(gdef_o, replicate(st_o, mesh))

    train_ds = V3Dataset(root, "train")
    val_ds = V3Dataset(root, "val", recordings=[val_recording])
    val_ds_all = V3Dataset(root, "val")     # full val = the truthful headline metric

    # Weighted sampling: oversample the under-represented target (sex, behavior)
    # class (e.g. female-courtship, ~2.2% of train) by `factor`. None -> uniform.
    weights = None
    if oversample is not None and float(oversample.get("factor", 1.0)) > 1.0:
        weights = train_ds.sampling_weights(
            oversample.get("sex"), oversample.get("behavior"),
            float(oversample["factor"]))
        n_tgt = int(sum(
            (oversample.get("sex") in (None, s)) and
            (oversample.get("behavior") in (None, b))
            for s, b in zip(train_ds.sex, train_ds.behavior)))
        print(f"weighted sampling: {n_tgt}/{len(train_ds)} anns match "
              f"(sex={oversample.get('sex')}, behavior={oversample.get('behavior')}) "
              f"@ {oversample['factor']}x -> ~{100*weights[weights>weights.min()].sum():.1f}% of samples")
    host_stream = _epochs(train_ds, tcfg.batch_size, tcfg.seed, weights=weights,
                         num_workers=num_workers)
    dev_stream = prefetch(host_stream, mesh, depth=2)

    final_loss = 0.0
    for i in range(start, tcfg.total_steps):
        img4_u8, kp_xy, vis = next(dev_stream)
        final_loss = float(step(model, opt, jax.random.fold_in(base_key, i),
                                img4_u8, kp_xy, vis))
        if (i + 1) % log_every == 0:
            print(f"step {i+1}/{tcfg.total_steps} loss {final_loss:.5f}")
        if (i + 1) % eval_every == 0:
            full = eval_mpjpe(model, val_ds_all, tcfg.batch_size)
            fem = eval_mpjpe(model, val_ds, tcfg.batch_size)
            print(f"  val MPJPE {full:.3f}px (female {val_recording}: {fem:.3f}px)")
        if mngr is not None and (i + 1) % save_every == 0:
            save_step(mngr, i + 1, model, opt)

    if mngr is not None:
        save_step(mngr, tcfg.total_steps, model, opt)
        mngr.wait_until_finished()

    val_mpjpe = eval_mpjpe(model, val_ds_all, tcfg.batch_size)
    fem_mpjpe = eval_mpjpe(model, val_ds, tcfg.batch_size)
    print(f"final val MPJPE {val_mpjpe:.3f}px "
          f"(female {val_recording}: {fem_mpjpe:.3f}px)")

    ckptr = ocp.StandardCheckpointer()
    ckptr.save(out_dir, nnx.split(model)[1], force=True)
    ckptr.wait_until_finished()
    return {"final_loss": final_loss, "val_mpjpe": val_mpjpe,
            "steps": tcfg.total_steps}


def main_from_cfg(cfg):
    tcfg = build_dataclass(TrainConfig, cfg.train)
    # Build ViTPoseConfig robustly: works whether model=vitpose (flat cfg.model.*)
    # OR model=hybridnet (ViT nested at cfg.model.vitpose).
    model_node = cfg.model.get("vitpose", cfg.model)
    # Model-selection: `arch` picks the detector architecture trained through
    # this SAME loop (ViT-vs-EfficientNet comparison). Absent (e.g. existing
    # model=vitpose / model=hybridnet configs never set it) -> "vitpose",
    # so the ViTPose path is byte-for-byte unchanged. ViTPoseConfig doubles as
    # the generic "model metadata" container (num_keypoints/in_ch/heatmap_size)
    # for both archs; EfficientTrack simply ignores the ViT-only fields.
    arch = model_node.get("arch", "vitpose")
    vitpose_cfg = build_dataclass(ViTPoseConfig, model_node)
    aug_params = build_dataclass(AugParams, cfg.aug)
    # Optional weighted oversampling of an under-represented (sex, behavior) class.
    oversample = None
    if "sampling" in cfg and cfg.sampling is not None:
        from omegaconf import OmegaConf
        oversample = OmegaConf.to_container(cfg.sampling, resolve=True)
    run_dir = run_dir_for(cfg)
    # EfficientTrack / EfficientTrack-BN have no MAE-pretrained checkpoint in
    # this pipeline -- never attempt to load one for either.
    mae_npz = cfg.paths.mae_npz if arch not in ("efficienttrack", "efficienttrack_bn") else None
    # EfficientTrack-BN's ImageNet fixture path is configurable via
    # model.effnet_b3_imagenet_npz (see configs/model/efficienttrack_bn.yaml);
    # None -> run_training's committed-fixture default.
    effnet_b3_imagenet_npz = (
        model_node.get("effnet_b3_imagenet_npz", None) if arch == "efficienttrack_bn" else None)
    return run_training(
        cfg.paths.data_root,
        out_dir=os.path.join(run_dir, "final"),
        mae_npz=mae_npz,
        tcfg=tcfg,
        vitpose_cfg=vitpose_cfg,
        aug_params=aug_params,
        arch=arch,
        effnet_b3_imagenet_npz=effnet_b3_imagenet_npz,
        smoke=bool(cfg.train.get("smoke", False)),
        ckpt_dir=os.path.join(run_dir, "ckpt"),
        save_every=cfg.train.save_every,
        eval_every=cfg.train.get("eval_every", 500),
        oversample=oversample,
        num_workers=cfg.train.get("num_workers", 8),
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
