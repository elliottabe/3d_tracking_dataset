"""Train JAX CenterDetect (EfficientTrack ``model_size="medium"``, EfficientNet-b1
backbone, ``num_joints=1``, ``in_channels=3``, ``image_size=320``) to reliably
find BOTH flies -- the Phase B training path described in
``.superpowers/sdd/2026-08-29-coarse-to-fine-3d/centerdetect-jax-phaseb.md``.

Three modifications over the plain PyTorch recipe (see that file for the full
rationale; briefly):
  1. Oversample two-fly frames: ``V5CenterDetectDataset.balanced_weights``
     (reused from ``V3Dataset``) keyed on ``num_flies`` (0/1/2 flies per
     frame), not sex.
  2. Per-instance loss normalisation:
     ``jarvis_jax.train.losses.centerdetect_instance_mse`` -- each ANIMAL
     contributes equally to the loss regardless of how many co-occur in its
     frame, applied at both CenterDetect output scales (res1 @80px, res2
     @160px, matching JARVIS's own two-scale supervision).
  3. Decode-side top-2: every per-epoch evaluation always extracts exactly 2
     peaks (``jarvis_jax.eval.centerdetect_decode.extract_top_k_peaks``, no
     confidence threshold, ``suppression_radius=15`` matching production).

Per-epoch (NOT per-loss-step) two-peak-rate evaluation on the held-out
labelled two-fly val frames is the acceptance signal this script reports --
loss alone hid the PyTorch retrain's epoch-10-vs-epoch-40 degradation (see
the phase-b report), so this script evaluates and CHECKPOINTS every epoch
specifically so that curve stays visible. It ALSO evaluates, every epoch, a
single-fly-frame FALSE-POSITIVE rate (spurious second peak) -- the risk the
two-peak metric alone cannot see: training heavily on two-fly frames can
teach the model to always emit two confident peaks, hallucinating a second
fly where there is one. Both curves land in the same ``metrics.json``.

Optional 4th modification, OFF BY DEFAULT (``--copy-paste-p 0.0``, this
file's original behaviour when the flag is omitted -- see
``.superpowers/sdd/2026-08-29-coarse-to-fine-3d/centerdetect-copypaste.md``):
copy-paste synthesis of two-fly TRAIN frames
(``jarvis_jax.data.copy_paste.CopyPasteCenterDetectDataset``) -- cuts a fly
out along its own SAM mask from a single-fly image and pastes it into
ANOTHER single-fly image on the SAME camera, at a sampled separation, to
break the 6.7%-two-fly-images data bottleneck that oversampling alone
cannot fix (oversampling only reweights the SAME 1,487 real two-fly images,
adding no pose/position/separation diversity). Applied to the TRAIN dataset
ONLY -- val is always the plain, unwrapped ``V5CenterDetectDataset``, so the
acceptance metrics above are always measured on real frames.

CLI:
    python -m jarvis_jax.scripts.train_centerdetect \\
        --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5_valfix \\
        --run-dir /gscratch/portia/eabe/data/Johnson_lab/jax_centerdetect_runs/<run_id> \\
        --epochs 30 --batch-size 32 \\
        --warm-start-pth .../fly50_V6/models/CenterDetect/Run_20260810-094955/EfficientTrack-medium_final.pth \\
        [--copy-paste-p 0.3]

    # Render N composites to <dump-dir> and exit WITHOUT training (visual
    # verification path -- see the copy-paste task brief):
    python -m jarvis_jax.scripts.train_centerdetect --root ... --run-dir /tmp/unused \\
        --copy-paste-p 1.0 --dump-samples 12 --dump-dir figures/2026-08-31-copypaste/grid
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

# Deferred (torch/jax-heavy) imports happen inside functions so this module
# stays importable (e.g. for --help, or from a CPU-only unit test) without
# forcing a device choice at import time.


# JARVIS's own sigma-by-output-resolution convention
# (``dataset2D.py::HeatmapGenerator(..., sigma=-2)``): sigma = 1.0 * out/64.
# NOT size-dependent (JARVIS's own `fact` multiplier for that branch is
# computed but never applied -- see dataset2D.py's commented-out `#sigma =
# self.fact*size/64.0` inside HeatmapGenerator.__call__), so every instance's
# target footprint is the same size regardless of the fly's real size in
# frame -- this loss's "per-instance" normalisation term relies on that
# (equal footprints -> a translation-invariant per-instance error).
OUT1 = 80    # quarter-res (final_conv1 / res1)
OUT2 = 160   # half-res    (deconv1     / res2)
SIGMA1 = 1.0 * OUT1 / 64.0   # 1.25
SIGMA2 = 1.0 * OUT2 / 64.0   # 2.5

IMAGE_SIZE = 320
SUPPRESSION_RADIUS = 15     # production default; see task brief + decode module docstring
CAPTURE_RADIUS_PX = 40.0    # matches the multianimal-collapse measurement's own definition


def build_model(*, warm_start_pth=None, seed=0):
    """EfficientTrack medium/b1, num_joints=1, in_channels=3. If
    `warm_start_pth` is given, seed weights from that PyTorch CenterDetect
    checkpoint (e.g. the shipped fly50_V6 EfficientTrack-medium_final.pth) --
    matching the PyTorch retrain's own methodology of fine-tuning FROM the
    collapsing checkpoint, so the two-peak-rate curves are comparable epoch
    for epoch. Falls back to random init (with a loud warning) if the path
    is absent."""
    from flax import nnx
    from jarvis_jax.models.efficienttrack import EfficientTrack
    if warm_start_pth and os.path.exists(warm_start_pth):
        from jarvis_jax.convert.load_efficienttrack import convert_efficienttrack_pth
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            model = convert_efficienttrack_pth(
                warm_start_pth, num_joints=1, out_dir=os.path.join(tmp, "ckpt"),
                model_size="medium")
        print(f"[warm-start] loaded CenterDetect weights from {warm_start_pth}")
        return model
    if warm_start_pth:
        print(f"WARNING: warm_start_pth={warm_start_pth!r} does not exist; "
              f"using RANDOM init.")
    return EfficientTrack(num_joints=1, in_channels=3, model_size="medium",
                          rngs=nnx.Rngs(seed))


def make_train_step(*, bg_weight=0.1, lr=3e-4, weight_decay=0.05,
                    warmup_steps=100, total_steps=2000, backbone_lr_mult=1.0):
    """Returns (step_fn, make_opt_fn). `step_fn(model, opt, img_u8, centers_xy,
    valid) -> (total_loss, fg1, bg1, fg2, bg2)`. No augmentation, no aug-key
    argument -- CenterDetect's own PyTorch training path this ports uses
    color/affine augmentation too, but that is out of scope for the 3
    approved modifications (oversampling, per-instance loss, decode-side
    top-2); adding it is a candidate follow-up, not silently bundled in
    here."""
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from jarvis_jax.data.device import IMAGENET_MEAN_J, IMAGENET_STD_J
    from jarvis_jax.train.losses import centerdetect_instance_mse

    scale1 = OUT1 / float(IMAGE_SIZE)
    scale2 = OUT2 / float(IMAGE_SIZE)

    def normalize_rgb(img_u8):
        img = img_u8.astype(jnp.float32) / 255.0
        return (img - IMAGENET_MEAN_J) / IMAGENET_STD_J

    def loss_fn(model, img_u8, centers_xy, valid):
        img = normalize_rgb(img_u8)
        res1, res2 = model.forward_both(img)
        t1, fg1, bg1 = centerdetect_instance_mse(
            res1, centers_xy * scale1, valid, heatmap_size=OUT1, sigma=SIGMA1,
            bg_weight=bg_weight)
        t2, fg2, bg2 = centerdetect_instance_mse(
            res2, centers_xy * scale2, valid, heatmap_size=OUT2, sigma=SIGMA2,
            bg_weight=bg_weight)
        return t1 + t2, (fg1, bg1, fg2, bg2)

    @nnx.jit
    def step(model, optimizer, img_u8, centers_xy, valid):
        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model, img_u8, centers_xy, valid)
        optimizer.update(model, grads)
        return loss, aux

    def make_opt(model):
        decay_steps = max(total_steps, warmup_steps + 1)

        def sched(peak):
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=peak, warmup_steps=warmup_steps,
                decay_steps=decay_steps, end_value=0.0)

        head_tx = optax.adamw(sched(lr), weight_decay=weight_decay)
        bb_tx = optax.adamw(sched(lr * backbone_lr_mult), weight_decay=weight_decay)

        def _labels(params):
            import jax
            return jax.tree_util.tree_map_with_path(
                lambda path, _: "backbone"
                if "backbone" in jax.tree_util.keystr(path) else "head", params)

        tx = optax.multi_transform({"backbone": bb_tx, "head": head_tx}, _labels)
        return nnx.Optimizer(model, tx, wrt=nnx.Param)

    return step, make_opt


def eval_two_peak_rate(model, val_ds, two_fly_idx, *, batch_size=16,
                       suppression_radius=SUPPRESSION_RADIUS,
                       capture_radius_px=CAPTURE_RADIUS_PX, mesh=None):
    """Two-peak rate on the held-out labelled two-fly val frames (the
    acceptance metric, not loss -- see module docstring). Decodes from res2
    (the half-res / 160px scale -- matches production's own
    ``_forward_multi_peak_only``). Returns a dict with overall/male/female
    rates and the per-frame nearest-peak distances (px, full-image
    coordinates) for the localisation-vs-SAM3 style reporting.
    """
    import jax.numpy as jnp
    from flax import nnx
    from jarvis_jax.data.device import IMAGENET_MEAN_J, IMAGENET_STD_J
    from jarvis_jax.eval.centerdetect_decode import (
        extract_top_k_peaks, peaks_to_full_image, two_peak_hit,
    )

    model.eval()
    hits = []
    dists_px = []
    sex_pairs = []     # list[tuple(str,str)] aligned with `hits`, sorted

    for i0 in range(0, len(two_fly_idx), batch_size):
        idxs = two_fly_idx[i0:i0 + batch_size]
        imgs, centers, valid = zip(*(val_ds[i] for i in idxs))
        img_u8 = jnp.asarray(np.stack(imgs))
        img = (img_u8.astype(jnp.float32) / 255.0 - IMAGENET_MEAN_J) / IMAGENET_STD_J
        if mesh is not None and img.shape[0] % len(mesh.devices.flat) == 0:
            # The model is replicated across the mesh, but an UNSHARDED input
            # runs the forward on the default device alone -- measured: one GPU
            # at 62% while the other three sat at 0-2% for the whole eval.
            from jarvis_jax.sharding import shard_batch
            img = shard_batch(img, mesh)
        _, res2 = model.forward_both(img)
        peaks_hm, conf = extract_top_k_peaks(np.asarray(res2), k=2,
                                             suppression_radius=suppression_radius)
        for b, i in enumerate(idxs):
            img_w, img_h = val_ds.img_wh[i]
            peaks_full = peaks_to_full_image(peaks_hm[b:b + 1], heatmap_size=OUT2,
                                             img_w=img_w, img_h=img_h)[0]
            gt_full = val_ds.centers[i]                      # (2,2) full-image px
            is_hit, d, _ = two_peak_hit(peaks_full, gt_full, capture_radius_px=capture_radius_px)
            hits.append(is_hit)
            dists_px.extend(d.tolist())
            sex_pairs.append(tuple(val_ds.sexes[i]))

    hits = np.asarray(hits, dtype=bool)
    male_mask = np.asarray([("male" in sp) and ("female" not in sp) for sp in sex_pairs])
    # A frame's two flies are (male, female) in this dataset (courtship
    # pairs) -- so "male" and "female" partitions are the SAME frame set
    # viewed from each fly's own label; report the frame-level rate under
    # BOTH sex labels present (courtship two-fly frames are always mixed-sex
    # here -- see the dataset guard test) rather than inventing a
    # male-only/female-only split that doesn't exist in this data.
    return {
        "n_frames": int(len(hits)),
        "two_peak_rate": float(hits.mean()) if len(hits) else float("nan"),
        "dists_px": dists_px,
        "dists_median_px": float(np.median(dists_px)) if dists_px else float("nan"),
        "dists_mean_px": float(np.mean(dists_px)) if dists_px else float("nan"),
        "dists_p90_px": float(np.percentile(dists_px, 90)) if dists_px else float("nan"),
    }


def eval_single_fly_false_positive(model, val_ds, single_fly_idx, *, batch_size=16,
                                   suppression_radius=SUPPRESSION_RADIUS,
                                   conf_ratio_thresholds=(0.25, 0.5, 0.75),
                                   max_frames=256, seed=0, mesh=None):
    """False-positive (spurious second peak) rate on the held-out labelled
    SINGLE-fly val frames -- the risk ``eval_two_peak_rate`` cannot see
    (see module docstring / the copy-paste-synthesis task brief's
    acceptance criterion 2): training heavily on two-fly frames can teach
    the model to always emit two confident peaks, hallucinating a second
    fly where there is only one. A two-peak-rate gain bought with
    single-fly false positives is not a real gain -- the production
    pipeline would triangulate a phantom animal.

    Decodes top-2 peaks exactly like ``eval_two_peak_rate`` (same scale,
    same ``suppression_radius``). ``conf1`` is the peak nearest the single
    GT fly (should be a hit almost always); ``conf2`` is the OTHER peak --
    on a genuinely single-fly frame conf2 should be low. Reports, for each
    threshold `t` in `conf_ratio_thresholds`, the fraction of frames where
    ``conf2 / max(conf1, eps) >= t`` (a peak nearly as confident as the
    real fly's own peak), plus the raw conf2/conf1 distribution -- "their
    confidence relative to the true peak" per the task brief.
    """
    import jax.numpy as jnp
    from jarvis_jax.data.device import IMAGENET_MEAN_J, IMAGENET_STD_J
    from jarvis_jax.eval.centerdetect_decode import extract_top_k_peaks, peaks_to_full_image

    model.eval()
    conf1s, conf2s = [], []
    # This rate is estimated, not enumerated: 1,509 single-fly frames were being
    # pushed through an unsharded forward + host-side NMS EVERY epoch, which
    # measured ~40% of a 352s epoch while the training half took ~200s. A fixed
    # random subsample of `max_frames` gives the rate to within a few percent
    # (binomial s.e. at n=256 is ~3%), which is far finer than the effect sizes
    # this metric exists to catch. Deterministic seed so the subset is the SAME
    # frames every epoch -- an epoch-to-epoch curve over a MOVING subset would
    # confound sampling noise with real change.
    if max_frames is not None and len(single_fly_idx) > max_frames:
        rng = np.random.default_rng(seed)
        single_fly_idx = list(np.asarray(single_fly_idx)[
            np.sort(rng.choice(len(single_fly_idx), size=max_frames, replace=False))])

    for i0 in range(0, len(single_fly_idx), batch_size):
        idxs = single_fly_idx[i0:i0 + batch_size]
        imgs, centers, valid = zip(*(val_ds[i] for i in idxs))
        img_u8 = jnp.asarray(np.stack(imgs))
        img = (img_u8.astype(jnp.float32) / 255.0 - IMAGENET_MEAN_J) / IMAGENET_STD_J
        if mesh is not None and img.shape[0] % len(mesh.devices.flat) == 0:
            # The model is replicated across the mesh, but an UNSHARDED input
            # runs the forward on the default device alone -- measured: one GPU
            # at 62% while the other three sat at 0-2% for the whole eval.
            from jarvis_jax.sharding import shard_batch
            img = shard_batch(img, mesh)
        _, res2 = model.forward_both(img)
        peaks_hm, conf = extract_top_k_peaks(np.asarray(res2), k=2,
                                             suppression_radius=suppression_radius)
        for b, i in enumerate(idxs):
            img_w, img_h = val_ds.img_wh[i]
            peaks_full = peaks_to_full_image(peaks_hm[b:b + 1], heatmap_size=OUT2,
                                             img_w=img_w, img_h=img_h)[0]
            gt = val_ds.centers[i][0]                        # (2,) the ONE real fly
            d = np.linalg.norm(peaks_full - gt[None, :], axis=-1)
            nearest = int(np.argmin(d))
            other = 1 - nearest
            conf1s.append(float(conf[b, nearest]))
            conf2s.append(float(conf[b, other]))

    conf1s = np.asarray(conf1s, dtype=np.float64)
    conf2s = np.asarray(conf2s, dtype=np.float64)
    ratio = conf2s / np.maximum(conf1s, 1e-6)
    out = {
        "n_frames": int(len(conf1s)),
        "conf1_mean": float(conf1s.mean()) if len(conf1s) else float("nan"),
        "conf2_mean": float(conf2s.mean()) if len(conf2s) else float("nan"),
        "conf2_median": float(np.median(conf2s)) if len(conf2s) else float("nan"),
        "conf_ratio_mean": float(ratio.mean()) if len(ratio) else float("nan"),
        "conf_ratio_median": float(np.median(ratio)) if len(ratio) else float("nan"),
    }
    for t in conf_ratio_thresholds:
        key = f"fp_rate_ratio_ge_{t:g}"
        out[key] = float((ratio >= t).mean()) if len(ratio) else float("nan")
    return out


def run_training(root, run_dir, *, epochs=30, batch_size=32, lr=3e-4,
                 bg_weight=0.1, seed=0, num_workers=8, warm_start_pth=None,
                 balance_alpha=0.5, max_repeat=20.0, image_size=IMAGE_SIZE,
                 log_every=50, copy_paste_p=0.0, copy_paste_sep_low=None,
                 copy_paste_sep_high=None, copy_paste_near_boundary=None,
                 copy_paste_near_frac=None, copy_paste_feather_sigma=None):
    """Full Phase B training loop. Writes ``run_dir/metrics.json`` (per-epoch
    two-peak rate + loss) and one Orbax checkpoint per epoch under
    ``run_dir/ckpt/epoch_<N>`` so the best (NOT necessarily last -- see task
    brief) checkpoint can be picked after the fact."""
    import jax
    from flax import nnx
    import orbax.checkpoint as ocp
    from jarvis_jax.data.v5_centerdetect import V5CenterDetectDataset, batches
    from jarvis_jax.sharding import data_parallel_mesh, replicate, shard_batch
    from jarvis_jax.data.prefetch import prefetch

    os.makedirs(run_dir, exist_ok=True)
    ckpt_root = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_root, exist_ok=True)

    train_ds = V5CenterDetectDataset(root, "train", image_size=image_size)
    val_ds = V5CenterDetectDataset(root, "val", image_size=image_size)
    two_fly_idx = val_ds.two_fly_indices()
    single_fly_idx = val_ds.single_fly_indices()
    print(f"train: {len(train_ds)} imgs ({train_ds.class_counts('num_flies')}); "
          f"val: {len(val_ds)} imgs, {len(two_fly_idx)} two-fly, "
          f"{len(single_fly_idx)} single-fly")

    weights = train_ds.balanced_weights(key="num_flies", alpha=balance_alpha,
                                        max_repeat=max_repeat)
    counts = train_ds.class_counts(key="num_flies")
    share = {c: 0.0 for c in counts}
    for w, lab in zip(weights, train_ds.num_flies):
        share[lab] += float(w)
    print(f"balanced sampling: key=num_flies alpha={balance_alpha} max_repeat={max_repeat}")
    for c in sorted(counts, key=lambda c: -counts[c]):
        n_c = counts[c]
        print(f"    {c}fly {n_c:>6} imgs ({100 * n_c / len(train_ds):5.1f}% of data) "
              f"-> {100 * share[c]:5.1f}% of samples "
              f"({share[c] * len(train_ds) / n_c:5.2f}x repeat)")

    # Optional 4th modification: copy-paste synthesis of two-fly TRAIN
    # frames -- OFF (copy_paste_p=0.0) reproduces this script's original
    # behaviour exactly (the control run, job 39394959, used no such flag).
    # val_ds is NEVER wrapped -- see module docstring.
    if copy_paste_p > 0.0:
        from jarvis_jax.data.copy_paste import CopyPasteCenterDetectDataset
        cp_kwargs = {"p": copy_paste_p, "seed": seed}
        if copy_paste_sep_low is not None:
            cp_kwargs["sep_low"] = copy_paste_sep_low
        if copy_paste_sep_high is not None:
            cp_kwargs["sep_high"] = copy_paste_sep_high
        if copy_paste_near_boundary is not None:
            cp_kwargs["sep_near_boundary"] = copy_paste_near_boundary
        if copy_paste_near_frac is not None:
            cp_kwargs["sep_near_frac"] = copy_paste_near_frac
        if copy_paste_feather_sigma is not None:
            cp_kwargs["feather_sigma"] = copy_paste_feather_sigma
        train_ds = CopyPasteCenterDetectDataset(train_ds, **cp_kwargs)
        est_two_fly_share = share.get("2", 0.0) + share.get("1", 0.0) * copy_paste_p
        print(f"copy-paste synthesis: p={copy_paste_p} sep=[{cp_kwargs.get('sep_low', 'default')},"
              f"{cp_kwargs.get('sep_high', 'default')}]px -- estimated two-fly-labelled "
              f"sample share (real + synthetic) ~= {100 * est_two_fly_share:.1f}%")

    steps_per_epoch = len(train_ds) // batch_size
    total_steps = steps_per_epoch * epochs
    model = build_model(warm_start_pth=warm_start_pth, seed=seed)
    step, make_opt = make_train_step(bg_weight=bg_weight, lr=lr,
                                     total_steps=total_steps)
    opt = make_opt(model)

    mesh = data_parallel_mesh()
    gdef_m, st_m = nnx.split(model)
    model = nnx.merge(gdef_m, replicate(st_m, mesh))
    gdef_o, st_o = nnx.split(opt)
    opt = nnx.merge(gdef_o, replicate(st_o, mesh))

    metrics = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        # Two things here are deliberate and both were measured:
        #
        # 1. `prefetch` (same helper train_keypoints.py uses) moves host->device
        #    transfer and sharding onto a background thread, so batch assembly
        #    overlaps device compute instead of serialising with it.
        # 2. losses are accumulated as DEVICE arrays -- never `float(loss)`
        #    per step. A per-step float() is a blocking device->host sync that
        #    defeats JAX's async dispatch entirely: the loop degenerates to
        #    assemble -> compute -> WAIT -> assemble, and the GPU idles through
        #    every host turn. Materialise once per epoch (and at the log
        #    cadence, which is rare enough not to matter).
        host_stream = batches(train_ds, batch_size, shuffle=True,
                              seed=seed + epoch, weights=weights,
                              num_workers=num_workers)
        for img_d, c_d, v_d in prefetch(host_stream, mesh, depth=2):
            loss, aux = step(model, opt, img_d, c_d, v_d)
            losses.append(loss)                      # device array, NOT float()
            global_step += 1
            if global_step % log_every == 0:
                fg1, bg1, fg2, bg2 = (float(x) for x in aux)
                print(f"  step {global_step}/{total_steps} loss={float(loss):.5f} "
                      f"(fg1={fg1:.4f} bg1={bg1:.4f} fg2={fg2:.4f} bg2={bg2:.4f})")
        train_loss = float(np.mean([float(x) for x in losses])) if losses else float("nan")

        two_peak = eval_two_peak_rate(model, val_ds, two_fly_idx,
                                      batch_size=batch_size, mesh=mesh)
        false_pos = eval_single_fly_false_positive(model, val_ds, single_fly_idx,
                                                    batch_size=batch_size, mesh=mesh)
        dt = time.time() - t0
        print(f"epoch {epoch}/{epochs}: train_loss={train_loss:.5f} "
              f"two_peak_rate={two_peak['two_peak_rate']:.3f} "
              f"({two_peak['n_frames']} val two-fly frames) "
              f"median_dist={two_peak['dists_median_px']:.1f}px "
              f"fp_rate(ratio>=0.5)={false_pos.get('fp_rate_ratio_ge_0.5', float('nan')):.3f} "
              f"({false_pos['n_frames']} val single-fly frames) "
              f"[{dt:.1f}s]")
        metrics.append({"epoch": epoch, "train_loss": train_loss,
                        "two_peak_rate": two_peak["two_peak_rate"],
                        "n_two_fly_val": two_peak["n_frames"],
                        "dists_median_px": two_peak["dists_median_px"],
                        "dists_mean_px": two_peak["dists_mean_px"],
                        "dists_p90_px": two_peak["dists_p90_px"],
                        "n_single_fly_val": false_pos["n_frames"],
                        "fp_conf1_mean": false_pos["conf1_mean"],
                        "fp_conf2_mean": false_pos["conf2_mean"],
                        "fp_conf2_median": false_pos["conf2_median"],
                        "fp_conf_ratio_mean": false_pos["conf_ratio_mean"],
                        "fp_conf_ratio_median": false_pos["conf_ratio_median"],
                        "fp_rate_ratio_ge_0.25": false_pos.get("fp_rate_ratio_ge_0.25"),
                        "fp_rate_ratio_ge_0.5": false_pos.get("fp_rate_ratio_ge_0.5"),
                        "fp_rate_ratio_ge_0.75": false_pos.get("fp_rate_ratio_ge_0.75"),
                        "epoch_seconds": dt})
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        ckptr = ocp.StandardCheckpointer()
        ckptr.save(os.path.join(ckpt_root, f"epoch_{epoch:03d}"),
                  nnx.split(model)[1], force=True)
        ckptr.wait_until_finished()

    best = max(metrics, key=lambda m: m["two_peak_rate"])
    print(f"best epoch by two_peak_rate: {best['epoch']} "
          f"(two_peak_rate={best['two_peak_rate']:.3f}, "
          f"last epoch was {metrics[-1]['epoch']} at "
          f"{metrics[-1]['two_peak_rate']:.3f})")
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"metrics": metrics, "best_epoch": best}, f, indent=2)
    return metrics


def jax_asarray(x):
    import jax.numpy as jnp
    return jnp.asarray(x)


def dump_copy_paste_samples(root, dump_dir, n, *, image_size=IMAGE_SIZE, seed=0,
                            copy_paste_p=1.0, copy_paste_sep_low=None,
                            copy_paste_sep_high=None, copy_paste_near_boundary=None,
                            copy_paste_near_frac=None, copy_paste_feather_sigma=None):
    """Render `n` FULL-RESOLUTION (not the training-time squished 320x320)
    copy-paste composites to `dump_dir`, plus a `meta.json` recording each
    one's host/donor file names, camera, same-recording flag, z-order, and
    sampled/achieved separation -- for visual verification (per the task
    brief: a synthesis pipeline whose output nobody has looked at is exactly
    how a model learns to detect seams instead of flies). PIL-only (no
    matplotlib dependency in this package); the annotated/labelled grid PNG
    for actual review is built by the repo's own
    ``scripts/viz/render_copypaste_samples.py`` from these dumps.

    Spreads its `n` draws round-robin across every camera present in the
    train split (so the dump cannot accidentally be all one viewpoint), and
    always synthesizes (``p`` is only used to size the estimated real-run
    sample composition printed alongside; every dumped row is an actual
    composite, using ``sample_with_meta`` which ignores `p`).
    """
    from jarvis_jax.data.copy_paste import CopyPasteCenterDetectDataset
    from jarvis_jax.data.v5_centerdetect import V5CenterDetectDataset
    from collections import defaultdict
    from PIL import Image as PILImage

    os.makedirs(dump_dir, exist_ok=True)
    train_ds = V5CenterDetectDataset(root, "train", image_size=image_size)
    cp_kwargs = {"p": copy_paste_p, "seed": seed}
    for k, v in (("sep_low", copy_paste_sep_low), ("sep_high", copy_paste_sep_high),
                ("sep_near_boundary", copy_paste_near_boundary),
                ("sep_near_frac", copy_paste_near_frac),
                ("feather_sigma", copy_paste_feather_sigma)):
        if v is not None:
            cp_kwargs[k] = v
    cp = CopyPasteCenterDetectDataset(train_ds, **cp_kwargs)

    single_idx = train_ds.single_fly_indices()
    by_cam = defaultdict(list)
    for i in single_idx:
        cam = train_ds.file_names[i].split("/")[1]
        by_cam[cam].append(i)
    cams = sorted(by_cam)
    print(f"[dump-samples] {len(single_idx)} single-fly train rows across "
          f"{len(cams)} cameras: {cams}")

    records = []
    for k in range(n):
        cam = cams[k % len(cams)]
        pool = by_cam[cam]
        host_idx = pool[(k // len(cams)) % len(pool)]
        composite_full, _img_out, _centers_out, meta = cp.sample_with_meta(
            host_idx, seed=seed * 100000 + k)
        fn = f"sample_{k:03d}.jpg"
        PILImage.fromarray(composite_full).save(os.path.join(dump_dir, fn))
        meta["camera"] = cam
        meta["file"] = fn
        records.append(meta)
        print(f"  [{k}] cam={cam} host={meta['host_file_name']} "
              f"donor={meta['donor_file_name']} same_rec={meta['same_recording']} "
              f"top_is_donor={meta['top_is_donor']} "
              f"sep_sampled={meta['sep_sampled_px']:.1f}px "
              f"sep_achieved={meta['sep_achieved_px']:.1f}px")

    seps = np.asarray([r["sep_achieved_px"] for r in records])
    print(f"[dump-samples] achieved separation: mean={seps.mean():.1f}px "
          f"median={np.median(seps):.1f}px min={seps.min():.1f}px max={seps.max():.1f}px")
    with open(os.path.join(dump_dir, "meta.json"), "w") as f:
        json.dump(records, f, indent=2)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bg-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--warm-start-pth", default=None)
    ap.add_argument("--balance-alpha", type=float, default=0.5)
    ap.add_argument("--max-repeat", type=float, default=20.0)
    ap.add_argument("--copy-paste-p", type=float, default=0.0,
                    help="Probability a single-fly TRAIN draw is turned into a "
                    "synthetic two-fly composite (0.0 = off, this script's "
                    "original behaviour). See jarvis_jax.data.copy_paste.")
    ap.add_argument("--copy-paste-sep-low", type=float, default=None)
    ap.add_argument("--copy-paste-sep-high", type=float, default=None)
    ap.add_argument("--copy-paste-near-boundary", type=float, default=None)
    ap.add_argument("--copy-paste-near-frac", type=float, default=None)
    ap.add_argument("--copy-paste-feather-sigma", type=float, default=None)
    ap.add_argument("--dump-samples", type=int, default=0,
                    help="If >0, render this many copy-paste composites to "
                    "--dump-dir and EXIT without training.")
    ap.add_argument("--dump-dir", default=None)
    args = ap.parse_args()

    if args.dump_samples > 0:
        if not args.dump_dir:
            ap.error("--dump-samples requires --dump-dir")
        dump_copy_paste_samples(
            args.root, args.dump_dir, args.dump_samples, seed=args.seed,
            copy_paste_p=args.copy_paste_p if args.copy_paste_p > 0 else 1.0,
            copy_paste_sep_low=args.copy_paste_sep_low,
            copy_paste_sep_high=args.copy_paste_sep_high,
            copy_paste_near_boundary=args.copy_paste_near_boundary,
            copy_paste_near_frac=args.copy_paste_near_frac,
            copy_paste_feather_sigma=args.copy_paste_feather_sigma)
        return

    run_training(args.root, args.run_dir, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, bg_weight=args.bg_weight,
                 seed=args.seed, num_workers=args.num_workers,
                 warm_start_pth=args.warm_start_pth,
                 balance_alpha=args.balance_alpha, max_repeat=args.max_repeat,
                 copy_paste_p=args.copy_paste_p,
                 copy_paste_sep_low=args.copy_paste_sep_low,
                 copy_paste_sep_high=args.copy_paste_sep_high,
                 copy_paste_near_boundary=args.copy_paste_near_boundary,
                 copy_paste_near_frac=args.copy_paste_near_frac,
                 copy_paste_feather_sigma=args.copy_paste_feather_sigma)


if __name__ == "__main__":
    main()
