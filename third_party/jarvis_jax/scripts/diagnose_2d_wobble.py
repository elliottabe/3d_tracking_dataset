#!/usr/bin/env python3
"""2D keypoint-wobble diagnostic: decode the SAME ViTPose heatmaps several ways
and measure per-joint high-frequency wobble.

Runs ViTPose on ONE camera of ONE bout of the configured recording, keeping the
raw (N,224,224,K) heatmaps, then decodes them with: soft-centroid (current),
hard-argmax (EfficientTrack-style), sharpened soft-centroid (p=2,3,4), and
log-parabolic Gaussian subpixel. Wobble is measured in 448px CROP coords so it
isolates the decode + heatmap shape from crop/centroid jitter.

Usage (submit to ckpt via scripts/slurm_diagnose_wobble.sh; see the plan):
    python third_party/jarvis_jax/scripts/diagnose_2d_wobble.py \
        +bout_id=3 +camera=track1 +n_frames=500 +fly=0 \
        +diag_out=diagnostics/wobble_diag.npz
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import numpy as np
import hydra
from omegaconf import DictConfig

# Reuse the exact bout-loading path from the production driver.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 3)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from scripts.run_courtship_bout import (                       # noqa: E402
    bout_start_frame, open_video_captures, all_cams_frames)
from jarvis_jax.cse.courtship_bout_masks import load_bout_masks     # noqa: E402
from jarvis_jax.cse.courtship_predict_2d import load_detector      # noqa: E402
from jarvis_jax.predict.session_frameset import build_frameset     # noqa: E402
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool  # noqa: E402
from jarvis_jax.data.device import normalize_image                 # noqa: E402
from jarvis_jax.eval import decode                                 # noqa: E402
import jax.numpy as jnp                                            # noqa: E402


def _dump_heatmaps(cfg, bout_idx, fly, cam_name, n_frames):
    """Run ViTPose on `cam_name` for the first n_frames of the bout; return
    (heatmaps (N,224,224,K), kp_names)."""
    cameras = list(cfg.recording.cameras)
    if cam_name not in cameras:
        raise ValueError(f"camera {cam_name!r} not in recording.cameras {cameras}")
    cam_idx = cameras.index(cam_name)

    predictions_dir = str(cfg.recording.predictions_dir)
    bout_npz = os.path.join(predictions_dir, f"bout_{bout_idx:05d}", "sam3_masks.npz")
    rt = ReprojectionTool(cfg.recording.calib_dir)
    cam_mats = np.asarray(rt.camera_matrices, np.float32)

    masks_dict = load_bout_masks(bout_npz, fly, expected_cameras=cameras)
    T = masks_dict["T"]
    N = min(int(n_frames), T)
    start = bout_start_frame(cfg, bout_idx)

    vit = load_detector(cfg.detector.ckpt, num_keypoints=int(cfg.detector.num_keypoints))
    caps = open_video_captures(cfg.recording.session_dir, cameras)
    hms = []
    try:
        frames_iter = all_cams_frames(caps, start, N)
        for t, frame_imgs in enumerate(frames_iter):
            c4, _centerHM, _nval = build_frameset(
                np.asarray(frame_imgs), masks_dict["masks"][t], masks_dict["centroids"][t],
                masks_dict["valid"][t], cam_mats, crop=int(cfg.detector.crop))
            if c4 is None:
                continue                       # frame with <2 valid views; skip
            x = normalize_image(jnp.asarray(c4[cam_idx:cam_idx + 1]))  # (1,448,448,4)
            hm = vit(x, use_running_average=True)                      # (1,224,224,K)
            hms.append(np.asarray(hm)[0])
    finally:
        for cap in caps:
            cap.release()
    if not hms:
        raise RuntimeError("no valid frames produced a crop for this camera/bout")
    return np.stack(hms), list(cfg.detector.kp_names)


def _analyze(heatmaps, kp_names):
    """heatmaps (N,224,224,K) -> (methods dict, wobble table, concentration)."""
    methods = {
        "soft_centroid_s1": dict(method="soft_centroid", sharpen=1.0),   # current
        "hard_argmax":      dict(method="hard_argmax"),                  # baseline
        "soft_centroid_s2": dict(method="soft_centroid", sharpen=2.0),
        "soft_centroid_s3": dict(method="soft_centroid", sharpen=3.0),
        "soft_centroid_s4": dict(method="soft_centroid", sharpen=4.0),
        "gaussian":         dict(method="gaussian"),
    }
    kp_by_method, wobble_by_method = {}, {}
    for name, kw in methods.items():
        kp = decode.decode_heatmaps(heatmaps, in_size=448, **kw)        # (N,K,2)
        kp_by_method[name] = kp
        wobble_by_method[name] = decode.wobble(kp)                      # (K,)
    concentration = decode.peak_concentration(heatmaps).mean(0)         # (K,)
    return kp_by_method, wobble_by_method, concentration


def _print_table(wobble_by_method, concentration, kp_names):
    print("\n=== 2D wobble (px, mean over joints; lower = calmer) ===")
    for name, wob in wobble_by_method.items():
        print(f"  {name:20s}  mean={np.nanmean(wob):6.3f}  median={np.nanmedian(wob):6.3f}"
              f"  p95={np.nanpercentile(wob, 95):6.3f}")
    soft = np.nanmean(wobble_by_method["soft_centroid_s1"])
    hard = np.nanmean(wobble_by_method["hard_argmax"])
    print(f"\n  soft/hard wobble ratio = {soft / max(hard, 1e-9):.2f}")
    print("  -> ratio >> 1: DECODE METHOD is the cause (sharpen/gaussian will fix cheaply)")
    print("  -> ratio ~ 1 AND hard wobble high: HEATMAPS are diffuse (need filter/retrain)")
    print(f"\n  mean peak concentration (r=7) = {np.nanmean(concentration):.3f} "
          f"(near 1.0 = tight peaks)")
    order = np.argsort(-np.nan_to_num(wobble_by_method["soft_centroid_s1"]))[:8]
    print("\n  worst-wobble joints (current decode):")
    for j in order:
        print(f"    {kp_names[j]:16s} wob={wobble_by_method['soft_centroid_s1'][j]:6.3f} "
              f"conc={concentration[j]:.3f}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="courtship_pipeline")
def main(cfg: DictConfig):
    bout_idx = int(cfg.get("bout_id", 3))
    fly = int(cfg.get("fly", 0))
    cam_name = str(cfg.get("camera", cfg.recording.cameras[0]))
    n_frames = int(cfg.get("n_frames", 500))
    out = str(cfg.get("diag_out", "diagnostics/wobble_diag.npz"))

    print(f"[diag] recording={cfg.recording.name} bout={bout_idx} fly={fly} "
          f"cam={cam_name} n_frames={n_frames}")
    heatmaps, kp_names = _dump_heatmaps(cfg, bout_idx, fly, cam_name, n_frames)
    print(f"[diag] heatmaps: {heatmaps.shape}")
    kp_by_method, wobble_by_method, concentration = _analyze(heatmaps, kp_names)
    _print_table(wobble_by_method, concentration, kp_names)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out, kp_names=np.array(kp_names), concentration=concentration,
        heatmaps_sample=heatmaps[:30].astype(np.float16),   # small subset for plots
        **{f"kp__{k}": v for k, v in kp_by_method.items()},
        **{f"wob__{k}": v for k, v in wobble_by_method.items()})
    print(f"[diag] wrote {out}")


if __name__ == "__main__":
    main()
