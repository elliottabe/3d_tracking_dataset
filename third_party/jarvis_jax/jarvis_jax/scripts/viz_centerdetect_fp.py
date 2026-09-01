"""Show WHERE the CenterDetect background term changes the prediction.

The metric this exists to illustrate is `fp_conf_ratio_median`: on a val
frame that contains ONE fly, decode the top-2 peaks, and report the second
peak's confidence as a fraction of the first's. Under the uniform background
term that ratio sat at 0.87 -- a phantom fly nearly as confident as the real
one, which no threshold can filter and which the production pipeline would
happily triangulate.

EXPECTATION, if the focal background term works:
  * LEFT column (uniform): two comparably bright blobs on a single-fly
    frame -- one on the fly (white circle = GT), one somewhere else, the
    red 'x'. Its printed ratio is near 0.9.
  * RIGHT column (focal): ONE bright blob on the fly. The second peak still
    exists (top-2 always returns two) but should be dim -- ratio well under
    0.5, and visibly darker in the heatmap.
  * BOTTOM row is a genuine TWO-fly frame and is the control: BOTH columns
    must keep two bright peaks on the two real flies. If focal dims the
    second peak here too, it did not learn "spurious peaks are expensive",
    it just learned "never emit a second peak" -- which is the ORIGINAL
    collapse coming back, and the fix must be rejected.

So: dim-second-peak on single-fly AND bright-second-peak on two-fly is the
pass. Dim in both columns of the bottom row is a failure that the two_peak
scalar would also catch; bright in both columns of the top rows is a
no-effect result.

Usage:
    python -m jarvis_jax.scripts.viz_centerdetect_fp \
        --run uniform=/path/to/cd_bg10 --run focal=/path/to/cd_focal_bg10 \
        --root /path/to/red_data_3d_v5_valfix --out figures/<topic>/fp.png
"""
import argparse
import json
import os

import numpy as np

IMAGE_SIZE = 320
OUT2 = 160
SUPPRESSION_RADIUS = 15


def _restore(ckpt_dir):
    import jax
    import orbax.checkpoint as ocp
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jarvis_jax.models.efficienttrack import EfficientTrack

    ctor = lambda: EfficientTrack(num_joints=1, in_channels=3,
                                  model_size="medium", rngs=nnx.Rngs(0))
    gdef, abstract = nnx.split(nnx.eval_shape(ctor))
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    target = jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl), abstract)
    model = nnx.merge(gdef, ocp.StandardCheckpointer().restore(ckpt_dir,
                                                               target=target))
    model.eval()
    return model


def _best_ckpt(run_dir):
    """The epoch this run itself selected, so the figure shows the weights the
    scorecard describes -- not whatever epoch happens to be last on disk."""
    with open(os.path.join(run_dir, "summary.json")) as f:
        best = json.load(f)["best_epoch"]["epoch"]
    return os.path.join(run_dir, "ckpt", f"epoch_{best:03d}"), best


def _predict(model, ds, idxs):
    import jax.numpy as jnp
    from jarvis_jax.data.device import IMAGENET_MEAN_J, IMAGENET_STD_J
    from jarvis_jax.eval.centerdetect_decode import (extract_top_k_peaks,
                                                     peaks_to_full_image)
    imgs = [ds[i][0] for i in idxs]
    u8 = jnp.asarray(np.stack(imgs))
    x = (u8.astype(jnp.float32) / 255.0 - IMAGENET_MEAN_J) / IMAGENET_STD_J
    _, hm = model.forward_both(x)
    hm = np.asarray(hm)
    peaks_hm, conf = extract_top_k_peaks(hm, k=2,
                                         suppression_radius=SUPPRESSION_RADIUS)
    full = []
    for b, i in enumerate(idxs):
        w, h = ds.img_wh[i]
        full.append(peaks_to_full_image(peaks_hm[b:b + 1], heatmap_size=OUT2,
                                        img_w=w, img_h=h)[0])
    return np.stack(imgs), hm, np.stack(full), np.asarray(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="LABEL=DIR", help="repeatable; column order is "
                    "the order given (put the baseline first)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-single", type=int, default=3,
                    help="single-fly frames, chosen as the WORST offenders "
                         "under the FIRST run (the baseline)")
    ap.add_argument("--n-two", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jarvis_jax.data.v5_centerdetect import V5CenterDetectDataset

    runs = [r.split("=", 1) for r in args.run]
    ds = V5CenterDetectDataset(args.root, args.split, image_size=IMAGE_SIZE,
                               cache_images=False)
    n_flies = np.array([int(v.sum()) for v in (ds[i][2] for i in range(len(ds)))])
    single = np.flatnonzero(n_flies == 1)
    two = np.flatnonzero(n_flies >= 2)
    print(f"{len(ds)} val frames: {len(single)} single-fly, {len(two)} two-fly")

    # Rank single-fly frames by the BASELINE's conf2/conf1 -- showing the
    # frames where the baseline is worst is the honest comparison; picking
    # them at random would mostly show frames neither model gets wrong.
    base_model = _restore(_best_ckpt(runs[0][1])[0])
    rng = np.random.default_rng(args.seed)
    pool = single if len(single) <= 256 else np.sort(
        rng.choice(single, 256, replace=False))
    ratios = []
    for i0 in range(0, len(pool), 16):
        _, _, _, c = _predict(base_model, ds, list(pool[i0:i0 + 16]))
        ratios.append(c[:, 1] / np.maximum(c[:, 0], 1e-9))
    ratios = np.concatenate(ratios)
    worst = pool[np.argsort(-ratios)[:args.n_single]]
    idxs = list(worst) + list(np.sort(rng.choice(two, min(args.n_two, len(two)),
                                                 replace=False)))
    del base_model

    nrow, ncol = len(idxs), len(runs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.4 * nrow),
                             squeeze=False)
    for col, (label, run_dir) in enumerate(runs):
        ckpt, epoch = _best_ckpt(run_dir)
        model = _restore(ckpt)
        imgs, hm, peaks, conf = _predict(model, ds, idxs)
        del model
        for row, i in enumerate(idxs):
            ax = axes[row][col]
            ax.imshow(imgs[row])
            h = hm[row, ..., 0] if hm.ndim == 4 else hm[row]
            s = IMAGE_SIZE / h.shape[0]
            ax.imshow(np.kron(h, np.ones((int(s), int(s)))),
                      cmap="inferno", alpha=0.45,
                      vmin=0.0, vmax=max(float(h.max()), 1e-6))
            w, hh = ds.img_wh[i]
            sx, sy = IMAGE_SIZE / w, IMAGE_SIZE / hh
            for c in ds.centers[i][ds[i][2].astype(bool)]:
                ax.plot(c[0] * sx, c[1] * sy, "o", mfc="none", mec="white",
                        ms=22, mew=2.0)
            r = conf[row, 1] / max(conf[row, 0], 1e-9)
            for k, (mk, cl) in enumerate((("+", "cyan"), ("x", "red"))):
                ax.plot(peaks[row, k, 0] * sx, peaks[row, k, 1] * sy, mk,
                        color=cl, ms=16, mew=3.0)
            kind = "TWO-fly (control)" if i in set(two.tolist()) else "single-fly"
            ax.set_title(f"{label}  (epoch {epoch})\n{kind}   "
                         f"conf2/conf1 = {r:.2f}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("CenterDetect 2nd-peak confidence on held-out val frames\n"
                 "white circle = labelled fly,  cyan + = top peak,  "
                 "red x = 2nd peak,  heat = predicted centre map", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
