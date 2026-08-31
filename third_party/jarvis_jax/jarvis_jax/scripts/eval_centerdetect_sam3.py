"""CenterDetect vs. SAM3-centroid localisation comparison + inference
throughput, on REAL courtship bout frames -- the acceptance bar in the task
brief is SAM3, not just a peak count.

Ground truth: Session0 ``2025_10_20_13_20_04`` bout 28 (raw frames
446306-448312, 7 cameras) -- the SAME bout the PyTorch collapse measurement
(``.superpowers/sdd/2026-08-29-coarse-to-fine-3d/centerdetect-multianimal.md``)
used, sampled at the SAME offsets (0,10,...,290 = "easy"; 1200,10,...,1490 =
"hard"; x 7 cameras = 420 (frame,camera) samples) for direct comparability.
SAM3 centroids/valid come from
``.../sam3_masks/bout_00028/sam3_masks.npz``; raw frames come from the
per-camera .mp4s in ``.../Video_recordings/courtship/Session0/<rec>/`` via
``jarvis_jax.predict.synced_reader`` (the canonical frame reader already
used by the production pipeline -- Session0 has no sync_plan.json, so this
is positional/lockstep, matching the multianimal doc's own read path).

CLI:
    python -m jarvis_jax.scripts.eval_centerdetect_sam3 \\
        --ckpt-dir /gscratch/.../jax_centerdetect_runs/<run>/ckpt/epoch_010 \\
        [--video-dir /gscratch/.../Video_recordings/courtship/Session0/2025_10_20_13_20_04] \\
        [--sam3-npz .../sam3_masks/bout_00028/sam3_masks.npz] \\
        [--bout-start-frame 446306] [--n-throughput 200]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

DEFAULT_VIDEO_DIR = ("/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/"
                     "courtship/Session0/2025_10_20_13_20_04")
DEFAULT_SAM3_NPZ = ("/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/"
                    "Session0/2025_10_20_13_20_04/sam3_masks/bout_00028/sam3_masks.npz")
BOUT_START_FRAME = 446306   # courtship_bout_summary.csv: Session0/2025_10_20_13_20_04, bout 28
EASY_OFFSETS = list(range(0, 300, 10))     # matches the multianimal-collapse measurement
HARD_OFFSETS = list(range(1200, 1500, 10))

IMAGE_SIZE = 320
OUT2 = 160
SUPPRESSION_RADIUS = 15


def _restore_model(ckpt_dir):
    import orbax.checkpoint as ocp
    import jax
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jarvis_jax.models.efficienttrack import EfficientTrack

    ctor = lambda: EfficientTrack(num_joints=1, in_channels=3, model_size="medium",
                                  rngs=nnx.Rngs(0))
    m_abstract = nnx.eval_shape(ctor)
    gdef, abstract_state = nnx.split(m_abstract)
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    target = jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl), abstract_state)
    ckptr = ocp.StandardCheckpointer()
    restored = ckptr.restore(ckpt_dir, target=target)
    model = nnx.merge(gdef, restored)
    model.eval()
    return model


def _resize_to_320(frame_rgb):
    from PIL import Image
    return np.asarray(Image.fromarray(frame_rgb).resize((IMAGE_SIZE, IMAGE_SIZE),
                                                        Image.BILINEAR), dtype=np.uint8)


def _decode_batch(model, imgs_u8):
    """imgs_u8: (N,320,320,3) uint8 -> (peaks_full-scale-in-320-space (N,2,2),
    conf (N,2)) decoded from res2 (160px)."""
    import jax.numpy as jnp
    from jarvis_jax.data.device import IMAGENET_MEAN_J, IMAGENET_STD_J
    from jarvis_jax.eval.centerdetect_decode import extract_top_k_peaks

    img = (jnp.asarray(imgs_u8).astype(jnp.float32) / 255.0 - IMAGENET_MEAN_J) / IMAGENET_STD_J
    _, res2 = model.forward_both(img)
    peaks_hm, conf = extract_top_k_peaks(np.asarray(res2), k=2,
                                         suppression_radius=SUPPRESSION_RADIUS)
    return peaks_hm, conf


def collect_samples(model, video_dir, sam3_npz, *, offsets=None, bout_start_frame=BOUT_START_FRAME,
                    batch_size=16):
    """Read real bout frames, decode CenterDetect peaks, and pair each
    (frame, camera) sample with its SAM3 centroid ground truth. Returns a
    list ready for ``centerdetect_sam3_compare.summarize``."""
    from jarvis_jax.predict.synced_reader import load_plan, read_window

    if offsets is None:
        offsets = EASY_OFFSETS + HARD_OFFSETS
    z = np.load(sam3_npz, allow_pickle=True)
    cameras = list(z["cameras"])
    centroids = z["centroids"]     # (2, C, T)
    valid = z["valid"]             # (2, C, T)
    img_h, img_w = int(z["shape"][0]), int(z["shape"][1])

    plan = load_plan(video_dir)
    samples = []
    imgs_buf, meta_buf = [], []

    def _flush():
        if not imgs_buf:
            return
        imgs = np.stack(imgs_buf)
        peaks_hm, conf = _decode_batch(model, imgs)
        sx, sy = img_w / float(OUT2), img_h / float(OUT2)
        for b, (offset, cam_i) in enumerate(meta_buf):
            peaks_full = peaks_hm[b].copy()
            peaks_full[:, 0] *= sx
            peaks_full[:, 1] *= sy
            samples.append({
                "peaks_full_xy": peaks_full,
                "sam3_centroids_xy": centroids[:, cam_i, offset],
                "sam3_valid": valid[:, cam_i, offset],
                "offset": offset, "camera": cameras[cam_i],
            })
        imgs_buf.clear(); meta_buf.clear()

    for offset in offsets:
        frames, present = next(read_window(video_dir, cameras, plan,
                                           bout_start_frame + offset, 1))
        for cam_i in range(len(cameras)):
            if not present[cam_i]:
                continue
            imgs_buf.append(_resize_to_320(frames[cam_i]))
            meta_buf.append((offset, cam_i))
            if len(imgs_buf) >= batch_size:
                _flush()
    _flush()
    return samples


def benchmark_throughput(model, *, n=200, batch_size=8):
    """Images/sec on ONE GPU (or whatever device `model` is on) -- forward
    pass only (res2), matching what production inference needs per frame.
    Excludes the first call (XLA compile). Compare to SAM3's own ~4.3-4.6
    it/s PER CAMERA (i.e. per-image, since SAM3 runs one camera at a time)."""
    import jax
    import jax.numpy as jnp
    rng = np.random.RandomState(0)
    dummy = jnp.asarray(rng.randint(0, 256, (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3),
                                    dtype=np.uint8))

    def _fwd(img_u8):
        from jarvis_jax.data.device import IMAGENET_MEAN_J, IMAGENET_STD_J
        img = (img_u8.astype(jnp.float32) / 255.0 - IMAGENET_MEAN_J) / IMAGENET_STD_J
        return model(img)

    _ = jax.block_until_ready(_fwd(dummy))       # warm-up / compile
    n_batches = max(1, n // batch_size)
    t0 = time.time()
    for _ in range(n_batches):
        out = _fwd(dummy)
    jax.block_until_ready(out)
    dt = time.time() - t0
    n_images = n_batches * batch_size
    return {"n_images": n_images, "seconds": dt, "images_per_sec": n_images / dt,
            "batch_size": batch_size}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--sam3-npz", default=DEFAULT_SAM3_NPZ)
    ap.add_argument("--bout-start-frame", type=int, default=BOUT_START_FRAME)
    ap.add_argument("--n-throughput", type=int, default=200)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from jarvis_jax.eval.centerdetect_sam3_compare import summarize

    model = _restore_model(args.ckpt_dir)
    samples = collect_samples(model, args.video_dir, args.sam3_npz,
                              bout_start_frame=args.bout_start_frame)
    result = summarize(samples)
    print(f"[sam3-compare] ckpt={args.ckpt_dir}")
    print(f"  n_samples={result['n_samples']} n_both_valid={result['n_both_valid']} "
          f"n_dist_obs={result['n_dist_obs']}")
    print(f"  two_peak_rate={result['two_peak_rate']:.3f}")
    if result["n_dist_obs"]:
        print(f"  dist px: mean={result['dist_mean_px']:.1f} median={result['dist_median_px']:.1f} "
              f"p10={result['dist_p10_px']:.1f} p90={result['dist_p90_px']:.1f} "
              f"max={result['dist_max_px']:.1f}")

    tput = benchmark_throughput(model)
    print(f"[throughput] {tput['images_per_sec']:.2f} images/sec "
          f"(batch={tput['batch_size']}, n={tput['n_images']}, {tput['seconds']:.2f}s) "
          f"-- SAM3 comparable figure: ~4.3-4.6 it/s per camera")

    out = {"sam3_compare": result, "throughput": tput, "ckpt_dir": args.ckpt_dir}
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    main()
