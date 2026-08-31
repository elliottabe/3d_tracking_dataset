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
specifically so that curve stays visible.

CLI:
    python -m jarvis_jax.scripts.train_centerdetect \\
        --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5_valfix \\
        --run-dir /gscratch/portia/eabe/data/Johnson_lab/jax_centerdetect_runs/<run_id> \\
        --epochs 30 --batch-size 32 \\
        --warm-start-pth .../fly50_V6/models/CenterDetect/Run_20260810-094955/EfficientTrack-medium_final.pth
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
                       capture_radius_px=CAPTURE_RADIUS_PX):
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


def run_training(root, run_dir, *, epochs=30, batch_size=32, lr=3e-4,
                 bg_weight=0.1, seed=0, num_workers=8, warm_start_pth=None,
                 balance_alpha=0.5, max_repeat=20.0, image_size=IMAGE_SIZE,
                 log_every=50):
    """Full Phase B training loop. Writes ``run_dir/metrics.json`` (per-epoch
    two-peak rate + loss) and one Orbax checkpoint per epoch under
    ``run_dir/ckpt/epoch_<N>`` so the best (NOT necessarily last -- see task
    brief) checkpoint can be picked after the fact."""
    import jax
    from flax import nnx
    import orbax.checkpoint as ocp
    from jarvis_jax.data.v5_centerdetect import V5CenterDetectDataset, batches
    from jarvis_jax.sharding import data_parallel_mesh, replicate, shard_batch

    os.makedirs(run_dir, exist_ok=True)
    ckpt_root = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_root, exist_ok=True)

    train_ds = V5CenterDetectDataset(root, "train", image_size=image_size)
    val_ds = V5CenterDetectDataset(root, "val", image_size=image_size)
    two_fly_idx = val_ds.two_fly_indices()
    print(f"train: {len(train_ds)} imgs ({train_ds.class_counts('num_flies')}); "
          f"val: {len(val_ds)} imgs, {len(two_fly_idx)} two-fly")

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
        for img_u8, centers_xy, valid in batches(
                train_ds, batch_size, shuffle=True, seed=seed + epoch,
                weights=weights, num_workers=num_workers):
            img_d = shard_batch(jax_asarray(img_u8), mesh)
            c_d = shard_batch(jax_asarray(centers_xy), mesh)
            v_d = shard_batch(jax_asarray(valid), mesh)
            loss, aux = step(model, opt, img_d, c_d, v_d)
            losses.append(float(loss))
            global_step += 1
            if global_step % log_every == 0:
                fg1, bg1, fg2, bg2 = (float(x) for x in aux)
                print(f"  step {global_step}/{total_steps} loss={float(loss):.5f} "
                      f"(fg1={fg1:.4f} bg1={bg1:.4f} fg2={fg2:.4f} bg2={bg2:.4f})")
        train_loss = float(np.mean(losses)) if losses else float("nan")

        two_peak = eval_two_peak_rate(model, val_ds, two_fly_idx, batch_size=batch_size)
        dt = time.time() - t0
        print(f"epoch {epoch}/{epochs}: train_loss={train_loss:.5f} "
              f"two_peak_rate={two_peak['two_peak_rate']:.3f} "
              f"({two_peak['n_frames']} val two-fly frames) "
              f"median_dist={two_peak['dists_median_px']:.1f}px "
              f"[{dt:.1f}s]")
        metrics.append({"epoch": epoch, "train_loss": train_loss,
                        "two_peak_rate": two_peak["two_peak_rate"],
                        "n_two_fly_val": two_peak["n_frames"],
                        "dists_median_px": two_peak["dists_median_px"],
                        "dists_mean_px": two_peak["dists_mean_px"],
                        "dists_p90_px": two_peak["dists_p90_px"],
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
    args = ap.parse_args()
    run_training(args.root, args.run_dir, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, bg_weight=args.bg_weight,
                 seed=args.seed, num_workers=args.num_workers,
                 warm_start_pth=args.warm_start_pth,
                 balance_alpha=args.balance_alpha, max_repeat=args.max_repeat)


if __name__ == "__main__":
    main()
