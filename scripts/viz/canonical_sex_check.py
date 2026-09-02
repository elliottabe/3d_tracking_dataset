#!/usr/bin/env python3
"""Acceptance figure for scripts/canonicalize_sam_masks.py: did the swap put the
HUMAN'S male in mask slot 1?

WHAT THE FIGURE SHOULD SHOW IF THE CANONICALIZATION IS CORRECT. Each row is one
bout, at the frame where the reviewed male's wing splay is largest (the
unmistakable courtship-song pose). Left pair = the SOURCE mask set, right pair
= the CANONICAL copy, for two cameras. Masks are outlined in the shared visual
language: cyan = slot 0, orange = slot 1. Green dots are the 2D keypoints of
the pose fly the human called male.

  * In every CANONICAL panel the ORANGE outline must sit on the same animal as
    the GREEN keypoints. That is the whole claim: fly1 == male.
  * In the SOURCE panels of a SWAPPED bout, orange must sit on the OTHER animal
    -- otherwise nothing was fixed and the "swap" was a relabel of a file that
    was already right.
  * In the SOURCE panels of a PASS-THROUGH bout, orange already sits on the
    green fly, and the two halves of the row look the same.

Anything else -- orange split across the two animals within a row, green
keypoints on neither outline, the two cameras disagreeing about which outline
is orange -- means the fly axis and the camera axis are not lined up, and the
canonical set must not be used.

Camera panels are chosen and indexed BY NAME throughout (the npz stores its own
camera order, kp2d uses the calibration-glob order, and they differ).

    python scripts/viz/canonical_sex_check.py \\
        --bouts Session1/2026_04_02_16_21_32/bout_00009 \\
        --out figures/2026-09-02-mask-canonicalization
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "jarvis_jax"))

from scripts.canonicalize_sam_masks import canonical_cameras          # noqa: E402
from viz.core.colors import PALETTE                                    # noqa: E402

DEF_SRC = "/gscratch/portia/eabe/data/Johnson_lab/processed/_courtship_backup"
DEF_DST = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship_canonical"
DEF_VID = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"
DEF_REVIEW = ("/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/"
              "id_review_reviewed_20260829.json")

SLOT_COLOR = {0: PALETTE["fly0"], 1: PALETTE["fly1"]}      # cyan / orange, BGR
KP_COLOR = PALETTE["fit"]                                  # green
LABEL_COLOR = (0, 255, 255)                                # yellow


def bout_start(session_dir, bouts_csv, bout_idx):
    from jarvis_jax.predict.sam3_driver import parse_bouts
    sys.path.insert(0, str(REPO / "scripts"))
    from run_bout import session_tag_for
    tag = session_tag_for(str(session_dir))
    bouts = parse_bouts(str(bouts_csv), tag, bout_ids=[bout_idx])
    if not bouts:
        raise KeyError(f"bout {bout_idx} not in {bouts_csv} (tag={tag})")
    return int(bouts[0]["start"])


def load_slot_masks(npz_path, cams):
    """(packed, centroids, valid) with the CAMERA axis reordered BY NAME to
    `cams`, plus the raster width."""
    with np.load(npz_path, allow_pickle=True) as z:
        names = [str(x) for x in z["cameras"]]
        order = [names.index(c) for c in cams]
        return (np.asarray(z["packed"])[:, order],
                np.asarray(z["centroids"])[:, order],
                np.asarray(z["valid"], bool)[:, order],
                int(np.asarray(z["shape"])[1]))


def wing_splay(kp3d, conf3d, kp_index, conf_min=0.2):
    """Per-frame max wing-splay angle in degrees (nan where unmeasurable) --
    the same quantity sexing.wing_song_cv takes the CV of."""
    axis = kp3d[:, kp_index["Abd_tip"]] - kp3d[:, kp_index["Scutellum"]]
    axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9)
    ok_body = ((conf3d[:, kp_index["Abd_tip"]] > conf_min)
               & (conf3d[:, kp_index["Scutellum"]] > conf_min))
    out = []
    for s in ("L", "R"):
        w = kp3d[:, kp_index[f"Wing{s}_V13"]] - kp3d[:, kp_index[f"Wing{s}_base"]]
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
        a = np.degrees(np.arccos(np.clip((w * axis).sum(1), -1, 1)))
        ok = (ok_body & (conf3d[:, kp_index[f"Wing{s}_V13"]] > conf_min)
              & (conf3d[:, kp_index[f"Wing{s}_base"]] > conf_min))
        out.append(np.where(ok, a, np.nan))
    with np.errstate(invalid="ignore"):
        return np.nanmax(np.stack(out, 1), axis=1)


def outline(img, mask, color, thickness=3):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, thickness)


def panel(frame_bgr, packed, valid, W, t, ci, kp_xy, crop, title, subtitle):
    img = frame_bgr.copy()
    for s in (0, 1):
        if not valid[s, ci, t]:
            continue
        m = np.unpackbits(packed[s, ci, t], axis=-1)[:, :W]
        outline(img, m, SLOT_COLOR[s])
    for x, y in kp_xy:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(img, (int(x), int(y)), 3, KP_COLOR, -1)
    x0, y0, x1, y1 = crop
    img = img[y0:y1, x0:x1]
    img = cv2.copyMakeBorder(img, 46, 6, 6, 6, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                LABEL_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, subtitle, (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA)
    return img


def crop_box(cents, valid, ci, t, shape, pad=170):
    xs, ys = [], []
    for s in (0, 1):
        if valid[s, ci, t]:
            xs.append(cents[s, ci, t, 0]); ys.append(cents[s, ci, t, 1])
    H, W = shape
    if not xs:
        return 0, 0, W, H
    x0 = int(max(0, min(xs) - pad)); x1 = int(min(W, max(xs) + pad))
    y0 = int(max(0, min(ys) - pad)); y1 = int(min(H, max(ys) + pad))
    return x0, y0, x1, y1


def build_row(key, args, review, kp_names):
    from viz.core.io import read_frame
    kp_index = {n: i for i, n in enumerate(kp_names)}
    sess, rec, bname = key.split("/")
    bout_idx = int(bname.split("_")[1])
    session_dir = Path(args.video_root) / sess / rec
    cams = canonical_cameras(session_dir / "calibration")
    male_fly = int(review[key]["reviewed_male_fly"])
    status = review[key]["status"]

    src_npz = Path(args.src_root) / sess / rec / "sam3_masks" / bname / "sam3_masks.npz"
    dst_npz = Path(args.dst_root) / sess / rec / "sam3_masks" / bname / "sam3_masks.npz"
    sp, sc, sv, W = load_slot_masks(src_npz, cams)
    dp, dc, dv, _ = load_slot_masks(dst_npz, cams)
    with np.load(dst_npz, allow_pickle=True) as z:
        meta = json.loads(str(z["sex_meta"]))
        H = int(np.asarray(z["shape"])[0])

    pose = Path(args.pose_root) / sess / rec / "pose" / "bouts" / bname
    with np.load(pose / f"fly{male_fly}" / "kp3d.npz") as z:
        splay = wing_splay(np.asarray(z["kp3d"], float),
                           np.asarray(z["conf3d"], float), kp_index)
    with np.load(pose / f"fly{male_fly}" / "kp2d.npz") as z:
        kp2d = np.asarray(z["kp2d"]); conf = np.asarray(z["conf"])

    # the frame to show: largest male wing splay among frames where both masks
    # are valid on every candidate camera and the two flies are well separated
    T = min(len(splay), sv.shape[2], kp2d.shape[0])
    sep = np.linalg.norm(sc[0, :, :T] - sc[1, :, :T], axis=-1)      # (C, T)
    ok = sv[:, :, :T].all(0) & dv[:, :, :T].all(0) & (sep > 250)
    n_ok = ok.sum(0)
    score = np.where(n_ok >= 2, np.nan_to_num(splay[:T], nan=-1), -1)
    t = int(np.argmax(score))
    cam_order = [ci for ci in np.argsort(-sep[:, t]) if ok[ci, t]][: args.cameras]
    if not cam_order:
        raise SystemExit(f"{key}: no camera with both masks separated at frame {t}")

    start = bout_start(session_dir, session_dir / "courtship_bouts_unified_summary.csv",
                       bout_idx)
    panels = []
    for ci in cam_order:
        cam = cams[ci]
        frame = read_frame(str(session_dir / f"{cam}.mp4"), start + t)
        kp_xy = kp2d[t, ci][conf[t, ci] > 0.5]
        crop = crop_box(sc, sv, ci, t, (H, W))
        panels.append(panel(frame, sp, sv, W, t, ci, kp_xy, crop,
                            f"SOURCE  {cam}",
                            f"male=pose fly{male_fly} -> slot "
                            f"{meta['original_male_slot']}"))
        panels.append(panel(frame, dp, dv, W, t, ci, kp_xy, crop,
                            f"CANONICAL  {cam}",
                            f"{meta['status']}: male -> slot {meta['male_slot']}"))
    h = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20))
              for p in panels]
    row = np.hstack(panels)
    hdr = np.full((34, row.shape[1], 3), 20, np.uint8)
    cv2.putText(hdr, f"{key}   review={status}  frame {t} (splay "
                     f"{splay[t]:.0f} deg)   cyan=slot0  orange=slot1  "
                     f"green=keypoints of the HUMAN'S male",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([hdr, row])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bouts", required=True,
                    help="comma-separated <Session>/<rec>/bout_XXXXX keys")
    ap.add_argument("--review", default=DEF_REVIEW)
    ap.add_argument("--src-root", default=DEF_SRC)
    ap.add_argument("--dst-root", default=DEF_DST)
    ap.add_argument("--pose-root", default=DEF_SRC)
    ap.add_argument("--video-root", default=DEF_VID)
    ap.add_argument("--cameras", type=int, default=2,
                    help="camera panels per bout (most-separated first)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from viz.config import courtship_recording
    kp_names = courtship_recording()["kp_names"]
    review = json.load(open(a.review))["bouts"]
    rows = [build_row(k.strip(), a, review, kp_names)
            for k in a.bouts.split(",") if k.strip()]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 8, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(20, 20, 20)) for r in rows]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "canonical_sex_check.png")
    cv2.imwrite(path, np.vstack(rows))
    print(f"wrote {path}  ({np.vstack(rows).shape[1]}x{np.vstack(rows).shape[0]})")
    return path


if __name__ == "__main__":
    main()
