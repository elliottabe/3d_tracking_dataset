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

Warm-start fine-tune (fresh optimizer/step, NEW run_id/out-dir, seeded ONLY
from another finished run's `final/` weights -- see `train.warm_start` /
``run_training``'s warm_start handling; distinct from the Orbax
`ckpt_dir`-based resume, which continues the SAME run's optimizer+step):
    python -m jarvis_jax.scripts.train_keypoints \\
        run_id=ft_run train=vit2d model=efficienttrack_bn paths=hyak \\
        train.warm_start=/path/to/source_run/final train.lr=3e-5
"""
import json
import os

import hydra
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass, run_dir_for

register_resolvers()

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.models.efficienttrack import EfficientTrack, build_efficienttrack_bn_imagenet
from jarvis_jax.convert.build_checkpoint import build
from jarvis_jax.data.prefetch import prefetch
from jarvis_jax.data.v3 import V3Dataset, batches
from jarvis_jax.data.v5_2d import V5Dataset
from jarvis_jax.data.device import normalize_image, normalize_image_center_channel
from jarvis_jax.data.center_channel import DEFAULT_CENTER_SIGMA_PX
from jarvis_jax.sharding import data_parallel_mesh, replicate
from jarvis_jax.train.train import (
    TrainConfig, make_optimizer, make_train_step, eval_mpjpe,
)
from jarvis_jax.data.augment import build_lr_swap, AugParams
from jarvis_jax.train.checkpoint import (
    make_manager, save_step, restore_latest, warm_start_restore,
)


def select_dataset_cls(root):
    """Pick the 2D keypoint dataset class matching `root`'s on-disk layout.

    `red_data_3d_v5`-shaped roots are flat -- no `train/`/`val/` directory
    anywhere, the split lives only in `annotations/instances_{split}.json`
    -- and are identifiable by having BOTH an `images/` directory and a
    `manifest.json` file directly under `root` (this is the same pair of
    markers `jarvis_jax/data/v5_2d.py`'s module docstring describes for the
    v5 tree, and the same file `jarvis_jax/data/v5_3d.py::V5FramesetDataset`
    requires to exist at its root). Anything else -- V3/V4-style roots,
    which have `train/`/`val/` and carry no `manifest.json` -- keeps using
    `V3Dataset`, unchanged.

    Constructing the wrong class against a v5 root fails loudly (missing
    `train/<file>` -> FileNotFoundError on the first sample; see
    `tests/test_v5_2d.py::test_v3dataset_cannot_read_v5_layout_by_design`)
    rather than silently, so this selection is the only place that needs to
    get the layout right.
    """
    is_v5 = (os.path.isdir(os.path.join(root, "images")) and
             os.path.isfile(os.path.join(root, "manifest.json")))
    return V5Dataset if is_v5 else V3Dataset


DEFAULT_MAE_NPZ = "/gscratch/portia/eabe/data/Johnson_lab/mae_vitb.npz"
# Committed torchvision efficientnet_b3 ImageNet fixture (see
# jarvis_jax/models/effnet_b3_std.py / convert/export_effnet_b3_imagenet_fixture.py),
# resolved relative to the package so it works from any checkout.
DEFAULT_EFFNET_B3_IMAGENET_NPZ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "convert", "fixtures", "effnet_b3_imagenet.npz")


def error_weights(error, *, alpha=4.0, cap=8.0):
    """Per-annotation error-weighted sampling probabilities (sum-normalised)
    from a ``mine_hard_frames.py`` per-annotation MPJPE ``error`` (px) array
    (NaN where an annotation had no visible joints, see ``mine_errors``).

    Formula (also documented in ``configs/sampling/hard_error.yaml``)::

        r      = error / median(error[finite])            # relative-to-typical error
        r      = clip(nan_to_num(r, nan=1.0), 0, cap)      # NaN -> neutral (median-like); cap the tail
        weight = 1.0 + alpha * r                            # every weight > 0 -> nothing is ever forgotten
        weight = weight / weight.sum()                      # sum-normalised sampling prob

    A median-error annotation (r=1) keeps weight ``1+alpha``; the hardest
    annotations (r capped at `cap`) get ``1+alpha*cap``; NaN-error
    annotations are treated as median (no signal to up- or down-weight).
    """
    error = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(error)
    if not finite.any():
        raise ValueError("error_weights: `error` has no finite values")
    median = float(np.median(error[finite]))
    if median <= 0.0:
        median = 1.0   # degenerate all-zero-error edge case; avoid div-by-0
    r = error / median
    r = np.clip(np.nan_to_num(r, nan=1.0), 0.0, float(cap))
    w = 1.0 + float(alpha) * r
    return w / w.sum()


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
                 ckpt_dir=None, save_every=500, oversample=None, num_workers=8,
                 warm_start=None):
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

    if warm_start:
        # WARM-START fine-tune: overwrite the just-built model's weights from
        # a FINISHED run's `final/` dir (e.g. et2d_bn_imagenet/final), then
        # fall through to a FRESH optimizer + step 0 below -- this is a NEW
        # run (new run_id/out_dir/ckpt_dir), not a resume of the source run.
        # For arch=efficienttrack_bn the ImageNet init just performed above is
        # therefore wasted (but harmless) work -- this restore fully
        # overwrites it (backbone conv + BN running stats + BiFPN + head,
        # i.e. every leaf `nnx.split` captures).
        _norm_before = float(sum(
            jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(nnx.split(model)[1])))
        model = warm_start_restore(model, warm_start)
        _norm_after = float(sum(
            jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(nnx.split(model)[1])))
        assert _norm_after != _norm_before, (
            f"warm_start restore from {warm_start!r} left the model's weights "
            "unchanged (shape mismatch against the source checkpoint, or a "
            "no-op restore) -- refusing to silently fine-tune from the wrong init")
        print(f"[warm-start] loaded weights from {warm_start}; fresh optimizer + step 0")

    opt = make_optimizer(model, tcfg)

    mngr = make_manager(ckpt_dir) if ckpt_dir else None
    start = 0
    if mngr is not None:
        model, opt, start = restore_latest(mngr, model, opt)
        if start:
            print(f"resuming from checkpoint at step {start}")

    if tcfg.mask_ablation and tcfg.center_channel_input:
        raise ValueError(
            "train.mask_ablation and train.center_channel_input are mutually "
            "exclusive -- both wrap the 4th input channel differently and "
            "cannot both apply to the same run.")
    # Which function rescales the 4th input channel to float — normal (SAM
    # mask, already {0,1}) vs the center_channel ablation arm (a 0..255-coded
    # Gaussian that needs /255 to land back in [0,1]; see
    # device.py::normalize_image_center_channel). Threaded through BOTH
    # make_train_step (below) and every eval_mpjpe call further down so
    # train and eval agree on the encoding.
    norm_fn = (normalize_image_center_channel if tcfg.center_channel_input
              else normalize_image)

    names = json.load(open(
        os.path.join(root, "annotations", "instances_train.json")))["keypoint_names"]
    lr_swap = build_lr_swap(names)
    aug = aug_params if aug_params is not None else AugParams()
    # sigma explicit here (not left to make_train_step's own 7.0 default) so
    # train.target_sigma (configs/train/vit2d.yaml) actually reaches the
    # rendered heatmap targets -- 2026-08-29 fix, see task-9-report.md
    # "Fix round 1": before this, a run launched with train.target_sigma=2.0
    # silently trained on sigma=7.0 targets.
    step = make_train_step(tcfg.mask_weight, aug, lr_swap, heatmap_size=cfg.heatmap_size,
                           sigma=tcfg.target_sigma, mask_dilate=tcfg.mask_dilate,
                           normalize_fn=norm_fn)
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

    # Select the 2D keypoint dataset class by root layout (V3/V4 split-rooted
    # tree vs v5's flat images/+manifest.json tree) -- see select_dataset_cls.
    dataset_cls = select_dataset_cls(root)
    train_ds = dataset_cls(root, "train")
    val_ds = dataset_cls(root, "val", recordings=[val_recording])
    val_ds_all = dataset_cls(root, "val")     # full val = the truthful headline metric
    if len(val_ds) == 0:
        # An empty cohort does not error in eval_mpjpe -- it reports 0.000px,
        # which reads as a PERFECT score. A whole 30k-step run
        # (v12_bal_maskoff) logged "female 2026_05_27_11_56_05: 0.000px" at
        # every eval because that recording no longer exists: the courtship
        # pairs were re-filed under their true capture ids
        # (2026_05_27_11_56_05 -> 2026_04_02_12_11_50). Fail at second zero
        # instead, the way finetune_detector already does.
        have = sorted({f.split("/")[0] for f in val_ds_all.file_names})
        raise ValueError(
            f"val_recording={val_recording!r} matches NO annotation in "
            f"{root}'s val split, so the per-cohort MPJPE would be reported as "
            f"a perfect 0.000px. Available val recordings: {have}")

    if tcfg.mask_ablation:
        # Mask-channel ablation: zero the 4th (SAM-mask) input channel of
        # every train/val sample, both here (dataset construction) and via
        # the identical wrap in eval_keypoints_2d.py / ad-hoc eval scripts,
        # so this run never sees a populated mask at train OR eval time.
        # Model stays in_ch=4 (capacity fixed) -- see mask_zero.py.
        from jarvis_jax.data.mask_zero import ZeroMaskDataset
        print("[mask-ablation] zeroing the SAM-mask input channel "
              "(train.mask_ablation=true) -- model keeps in_ch=4, capacity fixed")
        train_ds = ZeroMaskDataset(train_ds)
        val_ds = ZeroMaskDataset(val_ds)
        val_ds_all = ZeroMaskDataset(val_ds_all)

    if tcfg.center_channel_input:
        # Center-channel ablation: replace the 4th input channel with a
        # Gaussian at the TARGET fly's own bbox center (train-time: always
        # GT-driven), both here and via normalize_image_center_channel at
        # train/eval — see center_channel.py module docstring for the
        # eval-time GT-vs-predicted-center distinction this does NOT resolve
        # on its own (that needs a separate post-hoc eval pass with
        # predicted_centers_xy once a CenterDetect checkpoint exists).
        from jarvis_jax.data.center_channel import CenterChannelDataset
        print("[center-channel] replacing the 4th input channel with a "
              "target-fly-center Gaussian (train.center_channel_input=true, "
              f"sigma={DEFAULT_CENTER_SIGMA_PX}px) -- model keeps in_ch=4, "
              "capacity fixed")
        train_ds = CenterChannelDataset(train_ds)
        val_ds = CenterChannelDataset(val_ds)
        val_ds_all = CenterChannelDataset(val_ds_all)

    # Weighted sampling: EITHER error-weighted hard-example resampling (from a
    # jarvis_jax.scripts.mine_hard_frames.py error npz -- configs/sampling/
    # hard_error.yaml) OR the older category oversampling of an
    # under-represented (sex, behavior) class (e.g. female-courtship, ~2.2% of
    # train) by `factor`. Mutually exclusive; weights_file wins if both are
    # set. None -> uniform.
    weights = None
    if oversample is not None and oversample.get("weights_file"):
        weights_file = oversample["weights_file"]
        mined = np.load(weights_file)
        error = mined["error"]
        if len(error) != len(train_ds):
            raise ValueError(
                f"sampling.weights_file mismatch: {weights_file} has "
                f"{len(error)} per-annotation errors but the train split has "
                f"{len(train_ds)} annotations -- stale/mismatched mining file?")
        alpha = float(oversample.get("error_alpha", 4.0))
        cap = float(oversample.get("error_cap", 8.0))
        weights = error_weights(error, alpha=alpha, cap=cap)
        print(f"error-weighted sampling: weights_file={weights_file} "
              f"(alpha={alpha}, cap={cap}) -- preferred over any category "
              f"factor oversampling")
    elif oversample is not None and oversample.get("balance_key"):
        # Class-balanced sampling over every condition at once (V4). Preferred
        # over the single-class `factor` path when the dataset carries a
        # `category` per annotation -- see V3Dataset.balanced_weights.
        bkey = oversample["balance_key"]
        alpha = float(oversample.get("balance_alpha", 0.5))
        max_repeat = oversample.get("max_repeat")
        max_repeat = float(max_repeat) if max_repeat is not None else None
        counts = train_ds.class_counts(bkey)
        if set(counts) == {"unknown"}:
            # Every annotation untagged => balancing is a silent no-op. That is
            # the V4-build regression this guard exists to catch; fail instead.
            raise ValueError(
                f"sampling.balance_key={bkey!r} but every train annotation is "
                f"'unknown' on that axis -- the dataset json carries no {bkey} "
                f"field, so balanced sampling would silently degrade to "
                f"uniform. Rebuild the dataset with scripts/"
                f"build_detector_dataset.py (which emits sex/behavior/category).")
        weights = train_ds.balanced_weights(
            key=bkey, alpha=alpha, max_repeat=max_repeat)
        share = {c: 0.0 for c in counts}
        for w, lab in zip(weights, getattr(train_ds, bkey)):
            share[lab] += float(w)
        print(f"balanced sampling: key={bkey} alpha={alpha} "
              f"max_repeat={max_repeat}")
        for c in sorted(counts, key=lambda c: -counts[c]):
            n_c = counts[c]
            print(f"    {c:<20} {n_c:>6} anns ({100*n_c/len(train_ds):5.1f}% of data) "
                  f"-> {100*share[c]:5.1f}% of samples "
                  f"({share[c]*len(train_ds)/n_c:5.2f}x repeat)")
    elif oversample is not None and float(oversample.get("factor", 1.0)) > 1.0:
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
            full = eval_mpjpe(model, val_ds_all, tcfg.batch_size, normalize_fn=norm_fn)
            fem = eval_mpjpe(model, val_ds, tcfg.batch_size, normalize_fn=norm_fn)
            print(f"  val MPJPE {full:.3f}px (female {val_recording}: {fem:.3f}px)")
        if mngr is not None and (i + 1) % save_every == 0:
            save_step(mngr, i + 1, model, opt)

    if mngr is not None:
        save_step(mngr, tcfg.total_steps, model, opt)
        mngr.wait_until_finished()

    val_mpjpe = eval_mpjpe(model, val_ds_all, tcfg.batch_size, normalize_fn=norm_fn)
    fem_mpjpe = eval_mpjpe(model, val_ds, tcfg.batch_size, normalize_fn=norm_fn)
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
    # Warm-start fine-tune: load ONLY the model weights from a finished run's
    # `final/` dir (train.warm_start, configs/train/vit2d.yaml -- default ''
    # = off), then train with a FRESH optimizer/step count in THIS (new)
    # run's own run_dir/ckpt_dir -- see run_training's warm_start handling.
    # '' -> None so run_training's `if warm_start:` is unambiguously off.
    warm_start = cfg.train.get("warm_start", "") or None
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
        warm_start=warm_start,
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
