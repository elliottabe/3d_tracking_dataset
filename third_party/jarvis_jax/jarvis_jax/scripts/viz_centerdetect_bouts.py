"""CenterDetect on REAL courtship bout frames, against the SAM3 fly centroids.

The val-set numbers (two_peak_rate, fp_rate, the conf2/conf1 AUC) are measured
on labelled detector crops. This asks the deployment question instead: on a
real two-fly courtship frame, do the top-2 peaks land on the two flies?

REFERENCE: the SAM3 per-fly mask centroid (sam3_masks.npz `centroids`,
(2, C, T, 2) in full-frame pixels) -- an INDEPENDENT observation of where each
fly is, produced by a different model than the one under test.

EXPECTATION, if CenterDetect is working:
  * two peaks, one on each white circle, both within ~15 px.
  * the frames chosen here are deliberately the CLOSEST-together ones in the
    bout (the hard case: when the flies overlap, a center detector's failure
    is to merge them into one peak) plus a typical well-separated frame as the
    control.
Failure modes this makes visible, which the scalars cannot localise:
  * BOTH peaks on ONE fly, second circle bare -> the original collapse.
  * one peak on a fly, the other on empty arena -> the phantom.
  * peaks correct but confidence collapsing as separation shrinks -> the
    detector is about to fail on exactly the frames courtship cares about.

Per-frame numbers are printed and saved next to the PNG so the picture can be
checked rather than admired: for each fly, the distance to its nearest peak.

    python -m jarvis_jax.scripts.viz_centerdetect_bouts \
        --video-dir .../Video_recordings/courtship/Session0/2025_10_20_13_20_04 \
        --masks .../sam3_masks/bout_00028/sam3_masks.npz \
        --start-frame 446306 --cameras Cam2012630,Cam2012855 \
        --ckpt retrained=.../cd_bg10/ckpt/epoch_030 --out figures/<topic>/bout28
"""
import argparse
import json
import os

import numpy as np

IMAGE_SIZE = 320
OUT2 = 160
SUPPRESSION_RADIUS = 15


def _load_model(spec):
    """`dir` -> orbax JAX checkpoint; `pth:<path>` -> the original PyTorch
    CenterDetect, converted, so the retrain can be shown against what it
    replaced."""
    if spec.startswith("pth:"):
        import tempfile
        from jarvis_jax.convert.load_efficienttrack import convert_efficienttrack_pth
        with tempfile.TemporaryDirectory() as tmp:
            m = convert_efficienttrack_pth(spec[4:], num_joints=1,
                                           out_dir=os.path.join(tmp, "ckpt"),
                                           model_size="medium")
        m.eval()
        return m
    from jarvis_jax.scripts.viz_centerdetect_fp import _restore
    return _restore(spec)


def _grab(video_path, frame_idxs):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    out = {}
    for f in sorted(set(int(i) for i in frame_idxs)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"{video_path}: could not read frame {f}")
        out[f] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def _predict_full(model, imgs_rgb):
    """imgs_rgb: list of (H,W,3) uint8 full frames -> (heatmaps, peaks_full, conf)."""
    import cv2
    import jax.numpy as jnp
    from jarvis_jax.data.device import IMAGENET_MEAN_J, IMAGENET_STD_J
    from jarvis_jax.eval.centerdetect_decode import (extract_top_k_peaks,
                                                     peaks_to_full_image)
    small = np.stack([cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_LINEAR) for im in imgs_rgb])
    x = (jnp.asarray(small).astype(jnp.float32) / 255.0 - IMAGENET_MEAN_J) / IMAGENET_STD_J
    _, hm = model.forward_both(x)
    hm = np.asarray(hm)
    peaks_hm, conf = extract_top_k_peaks(hm, k=2, suppression_radius=SUPPRESSION_RADIUS)
    H, W = imgs_rgb[0].shape[:2]
    full = np.stack([peaks_to_full_image(peaks_hm[b:b + 1], heatmap_size=OUT2,
                                         img_w=W, img_h=H)[0]
                     for b in range(len(imgs_rgb))])
    return hm, full, np.asarray(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--masks", required=True, help="bout's sam3_masks.npz")
    ap.add_argument("--start-frame", type=int, required=True,
                    help="bout's ABSOLUTE start frame in the video")
    ap.add_argument("--cameras", default="", help="comma list; default: first 2 in the npz")
    ap.add_argument("--ckpt", action="append", required=True, metavar="LABEL=SPEC")
    ap.add_argument("--max-local-frame", type=int, default=None,
                    help="ignore local frames >= this. Bout 28's female leaves "
                         "the camera views after ~1590, so her mask centroid "
                         "there is not a fly and any 'hard case' drawn from the "
                         "tail measures the MASKS failing, not the detector.")
    ap.add_argument("--n-hard", type=int, default=2, help="closest-together frames")
    ap.add_argument("--n-easy", type=int, default=1)
    ap.add_argument("--out", required=True, help="basename for .png/.json")
    args = ap.parse_args()

    z = np.load(args.masks, allow_pickle=True)
    cams_all = [str(c) for c in z["cameras"]]
    cen = z["centroids"]                       # (2, C, T, 2) full-frame px
    valid = z["valid"]                         # (2, C, T)
    cams = ([c.strip() for c in args.cameras.split(",") if c.strip()]
            or cams_all[:2])
    for c in cams:
        if c not in cams_all:
            raise SystemExit(f"camera {c!r} not in {cams_all}")

    # Pick frames by TRUE fly separation, not at random: the hard case for a
    # center detector is two flies close enough to merge into one peak.
    ci0 = cams_all.index(cams[0])
    both = valid[0, ci0] & valid[1, ci0]
    sep = np.full(cen.shape[2], np.inf)
    sep[both] = np.linalg.norm(cen[0, ci0, both] - cen[1, ci0, both], axis=-1)
    if args.max_local_frame is not None:
        sep[args.max_local_frame:] = np.inf
    finite = np.flatnonzero(np.isfinite(sep))
    if len(finite) == 0:
        raise SystemExit("no frame has both flies masked in this camera")
    order = finite[np.argsort(sep[finite])]
    picks = list(order[:args.n_hard]) + list(order[len(order) // 2:
                                                   len(order) // 2 + args.n_easy])
    print(f"{len(finite)} frames with both flies; separation "
          f"min={sep[finite].min():.0f} med={np.median(sep[finite]):.0f} "
          f"max={sep[finite].max():.0f} px")
    print(f"picked local frames {picks} (sep = "
          f"{[round(float(sep[p])) for p in picks]} px)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [s.split("=", 1) for s in args.ckpt]
    nrow = len(picks) * len(cams)
    fig, axes = plt.subplots(nrow, len(runs),
                             figsize=(7.2 * len(runs), 2.3 * nrow), squeeze=False)
    report = {}
    for col, (label, spec) in enumerate(runs):
        model = _load_model(spec)
        row = 0
        for cam in cams:
            ci = cams_all.index(cam)
            abs_idxs = [args.start_frame + int(p) for p in picks]
            frames = _grab(os.path.join(args.video_dir, f"{cam}.mp4"), abs_idxs)
            imgs = [frames[i] for i in abs_idxs]
            hm, peaks, conf = _predict_full(model, imgs)
            for b, p in enumerate(picks):
                ax = axes[row][col]
                ax.imshow(imgs[b])
                h = hm[b, ..., 0] if hm.ndim == 4 else hm[b]
                ax.imshow(np.kron(h, np.ones((3, 3))), cmap="inferno", alpha=0.40,
                          extent=(0, imgs[b].shape[1], imgs[b].shape[0], 0),
                          vmin=0.0, vmax=max(float(h.max()), 1e-6))
                gts, dists = [], []
                for f in (0, 1):
                    if not valid[f, ci, p]:
                        continue
                    g = cen[f, ci, p]
                    gts.append(g)
                    ax.plot(g[0], g[1], "o", mfc="none", mec="white", ms=17, mew=2.0)
                    ax.annotate(f"fly{f}", (g[0], g[1]), color="white",
                                fontsize=7, xytext=(6, -10), textcoords="offset points")
                    dists.append(float(np.min(np.linalg.norm(peaks[b] - g, axis=-1))))
                for k, (mk, cl) in enumerate((("+", "cyan"), ("x", "red"))):
                    ax.plot(peaks[b, k, 0], peaks[b, k, 1], mk, color=cl,
                            ms=13, mew=2.5)
                r = conf[b, 1] / max(conf[b, 0], 1e-9)
                ax.set_title(f"{label} | {cam} | frame {args.start_frame + p} | "
                             f"sep {sep[p]:.0f}px | conf2/conf1 {r:.2f} | "
                             f"fly->peak {', '.join(f'{d:.0f}px' for d in dists)}",
                             fontsize=7.5)
                ax.set_xticks([]); ax.set_yticks([])
                report[f"{label}|{cam}|{args.start_frame + p}"] = {
                    "separation_px": float(sep[p]),
                    "conf": [float(conf[b, 0]), float(conf[b, 1])],
                    "conf_ratio": float(r),
                    "fly_to_nearest_peak_px": dists}
                row += 1
        del model
    fig.suptitle("CenterDetect top-2 peaks vs SAM3 fly centroids on real bout frames\n"
                 "white circles = the two flies (SAM3), cyan + = top peak, "
                 "red x = 2nd peak, heat = predicted centre map", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=125)
    with open(f"{args.out}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}.png / .json")
    for k, v in report.items():
        print(f"  {k}: sep={v['separation_px']:.0f}px "
              f"fly->peak={[round(d) for d in v['fly_to_nearest_peak_px']]} "
              f"ratio={v['conf_ratio']:.2f}")


if __name__ == "__main__":
    main()
