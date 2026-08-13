#!/usr/bin/env python3
"""4-channel ViTPose 2D for the explainer clip.

Output is written in MODEL order (configs/anatomy/v1.yaml KP_NAMES). The
detector emits DETECTOR order (configs/detector/vitpose_v3.yaml kp_names);
reorder_detector_to_model bridges the two. Getting this wrong scrambles
anatomy while every numeric QC stays green -- CLAUDE.md records that bug.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io            # noqa: E402
from scripts.viz.ik_explainer import masks as masks_mod  # noqa: E402

CKPT = ("/data2/users/eabe/datasets/3d_tracking/jax_vitpose_runs/"
        "v4_8gpu_20260808/final")


def run_detect(clip: str, *, ckpt: str = CKPT, batch: int = 64,
               decode_sharpen: float = 3.0):
    from jarvis_jax.tracking.bout_masks import load_bout_masks
    from jarvis_jax.tracking.predict_2d import (load_detector, predict_bout_2d,
                                                reorder_detector_to_model)

    d = clip_io.out_dirs(clip)
    cam_mats, cam_names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    # fly=0 (single-animal clip). expected_cameras reorders the mask C axis BY
    # NAME into the DLT order -- a positional mismatch here silently pairs each
    # crop with the wrong camera matrix and collapses triangulation.
    bm = load_bout_masks(str(masks_mod.mask_npz_path(clip)), 0,
                         expected_cameras=cam_names)
    masks = np.asarray(bm["masks"], bool)              # (N,C,H,W)
    centroids = np.asarray(bm["centroids"], np.float32)  # (N,C,2)
    valid = np.asarray(bm["valid"], bool)              # (N,C)
    N = masks.shape[0]

    caps = [clip_io.video_path(clip, c) for c in cam_names]

    def frames_iter():
        import cv2
        readers = [cv2.VideoCapture(p) for p in caps]
        try:
            for _t in range(N):
                imgs = []
                for r in readers:
                    ok, img = r.read()
                    if not ok:
                        raise IOError("video ended early")
                    imgs.append(img[:, :, ::-1])          # BGR -> RGB
                yield np.stack(imgs)
        finally:
            for r in readers:
                r.release()

    vit = load_detector(ckpt, num_keypoints=50)
    kp2d, conf = predict_bout_2d(
        vit, frames_iter(), masks, centroids, valid, cam_mats,
        crop=448, batch=batch, decode_sharpen=decode_sharpen,
        distractor_masks=None)          # single-animal assay

    # DETECTOR order -> MODEL order. Single conversion point.
    det_names = clip_io.detector_kp_names()
    mod_names = clip_io.model_kp_names()
    kp2d, conf = reorder_detector_to_model(kp2d, conf, det_names, mod_names)

    out = d["predictions"] / "02_kp2d.npz"
    np.savez_compressed(out, kp2d=kp2d.astype(np.float32),
                        conf=conf.astype(np.float32),
                        cam_names=np.array(cam_names),
                        kp_names=np.array(mod_names))
    print(f"wrote {out}  kp2d={kp2d.shape} conf={conf.shape} (MODEL order)")
    return out


def qc_kp2d(clip: str, frames=(120, 450, 780)):
    """2D QC montage, one row per frame, all 7 cameras.

    EXPECTATION if the detector and the ordering are right: keypoints sit on
    the fly in every view, with Antenna_Base/EyeL/EyeR on the HEAD, Abd_tip at
    the posterior tip, and the six leg chains running proximal->distal without
    crossing the body. Confidence should DROP on views where a limb is
    occluded -- uniform high confidence everywhere means the ordering or the
    crop is wrong, not that tracking is perfect.
    FALSIFICATION: head-group markers on the abdomen, or leg chains zig-zagging
    across the body => detector->model reorder is broken.
    """
    import cv2
    from viz.core.colors import PALETTE, keypoint_groups, leg_chains

    d = clip_io.out_dirs(clip)
    z = np.load(d["predictions"] / "02_kp2d.npz", allow_pickle=True)
    kp2d, conf = z["kp2d"], z["conf"]
    cam_names = [str(c) for c in z["cam_names"]]
    kp_names = [str(n) for n in z["kp_names"]]
    groups, chains = keypoint_groups(kp_names), leg_chains(kp_names)
    gcol = {"head": PALETTE["head"], "abdomen": PALETTE["tail"],
            "thorax": (0, 255, 255), "legs": PALETTE["fly0"]}

    rows = []
    for f in frames:
        row = []
        for ci, cam in enumerate(cam_names):
            img = clip_io.read_frames(clip_io.video_path(clip, cam), [f])[0]
            p, c = kp2d[f, ci], conf[f, ci]
            cx, cy = float(np.nanmean(p[:, 0])), float(np.nanmean(p[:, 1]))
            x0 = int(np.clip(cx - 210, 0, img.shape[1] - 420))
            y0 = int(np.clip(cy - 150, 0, img.shape[0] - 300))
            crop = img[y0:y0 + 300, x0:x0 + 420].copy()
            q = p - np.array([x0, y0])
            for _leg, ch in chains.items():
                for a, b in zip(q[ch][:-1], q[ch][1:]):
                    if np.all(np.isfinite([a, b])):
                        cv2.line(crop, tuple(a.astype(int)), tuple(b.astype(int)),
                                 PALETTE["fly0"], 1, cv2.LINE_AA)
            for g, idxs in groups.items():
                for i in idxs:
                    if np.all(np.isfinite(q[i])):
                        r = 2 if c[i] >= 0.3 else 1
                        cv2.circle(crop, tuple(q[i].astype(int)), r, gcol[g], -1,
                                   cv2.LINE_AA)
            for lab in ("Antenna_Base", "Abd_tip"):
                i = kp_names.index(lab)
                if np.all(np.isfinite(q[i])):
                    cv2.putText(crop, lab, tuple((q[i] + [4, -4]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1,
                                cv2.LINE_AA)
            cv2.putText(crop, f"{cam} f{f} medconf={np.median(c):.2f}", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(crop)
        rows.append(np.hstack(row))
    out = d["qc"] / "02_kp2d_overlay.png"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--qc", action="store_true", help="write the 2D QC montage")
    a = ap.parse_args()
    if a.qc:
        qc_kp2d(a.clip)
        return 0
    run_detect(a.clip, ckpt=a.ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
