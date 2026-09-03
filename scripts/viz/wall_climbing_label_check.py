#!/usr/bin/env python3
"""Visual acceptance for `wall_frames_15_06_46_male` (2025_10_12_15_06_46).

This recording was previously reported BLOCKED under a wrong timestamp
(2025_10_12_10_56_07): no video, no frames. The user corrected the timestamp,
renamed the export directory and extracted the frames, so it converts like any
other -- except that its VIDEO is a 921-frame excerpt (frames 1161383-1162303)
while the labels run to index 1,447,731, so the shipped jpgs are the only
possible image source and nothing is decoded here.

WHAT THIS FIGURE SHOULD SHOW IF THE CONVERSION IS CORRECT
---------------------------------------------------------
Row A  CONVERTED LABELS on four cameras of the wall-climbing fly. Cyan
       keypoints and leg chains must sit ON the animal: Antenna_Base/EyeL/EyeR
       on the head, Abd_tip at the abdomen tip, six leg chains running down
       real legs. The fly is against the arena wall at the frame edge in most
       of these -- that is the point, it is the regime the corpus is starved of
       and the one this pipeline is documented to fail at.

Row B  THE FLIP CONTROL. The same keypoints WITHOUT `y = image_height - v_raw`.
       Red must land mirrored about the 448-px midline and clearly OFF the
       animal. If red and cyan both look plausible, the flip is not determined
       by the data and row A is not evidence.

Row C  SAME CAPTURE AS general_model/wall_frames, shown rather than asserted.
       Cyan = this recording's Frame_4012, white = wall_frames' Frame_4031,
       19 frames (24 ms at 800 fps) apart, each drawn with its OWN stored
       annotation. The arena, wall and lighting must be continuous and the fly
       must be in nearly the same place. This is why the two subsets are held
       as ONE capture group by the split builder: their frame numbers never
       intersect, so content hashing cannot see the relationship -- exactly the
       leak class that put 54% of the v8 val set on both sides via
       courtship_V2/V3/V4.

Keypoints and cameras are resolved BY NAME throughout (fly50 order from
data/fly50.json); nothing is indexed by a bare integer.

    python scripts/viz/wall_climbing_label_check.py \
        --out figures/2026-09-02-climbing-recording
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from viz.core.colors import PALETTE, keypoint_groups, leg_chains  # noqa: E402

RED = Path("/gscratch/portia/eabe/data/Johnson_lab/red_data")
RAW = RED / "courtship_label_2026_09_02/2025_10_12_15_06_46_male_climbing"
STAGE = RED / "courtship_labels_2026_09_02/wall_frames_15_06_46_male"
WALL = RED / "general_model/wall_frames"
WALL_REC = "2026_07_30_13_28_99"


def mpl(c):                      # PALETTE is BGR (cv2 convention)
    return (c[2] / 255, c[1] / 255, c[0] / 255)


def load_conv(path):
    """{(camera, frame): (K,3) keypoints} resolved BY NAME from a COCO json."""
    d = json.load(open(path))
    names = d["keypoint_names"]
    by_img = {im["id"]: im["file_name"] for im in d["images"]}
    out = {}
    for a in d["annotations"]:
        _, cam, fn = by_img[a["image_id"]].split("/")
        k = np.asarray(a["keypoints"], float).reshape(-1, 3)
        out[(cam, int(fn.split("_")[1].split(".")[0]))] = k
    return names, out


def draw(ax, img, kp, names, color, flip_h=None, label=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    xy = kp[:, :2].astype(float).copy()
    vis = kp[:, 2] > 0
    if flip_h is not None:                       # the no-flip control
        xy[:, 1] = flip_h - 1 - xy[:, 1]
    for chain in leg_chains(names).values():
        pts = [i for i in chain if vis[i]]
        if len(pts) > 1:
            ax.plot(xy[pts, 0], xy[pts, 1], "-", lw=0.9, color=color, alpha=0.85)
    groups = keypoint_groups(names)
    for g, marker, size in (("head", "o", 9), ("thorax", "s", 7),
                            ("abdomen", "v", 7), ("legs", ".", 5)):
        idx = [i for i in groups[g] if vis[i]]
        if idx:
            ax.scatter(xy[idx, 0], xy[idx, 1], s=size, marker=marker,
                       color=color, linewidths=0, zorder=3)
    for n in ("Antenna_Base", "Abd_tip"):
        i = names.index(n)
        if vis[i]:
            ax.annotate(n, (xy[i, 0], xy[i, 1]), fontsize=5.5, color=color,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xticks([]); ax.set_yticks([])
    if label:
        ax.set_title(label, fontsize=7.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "figures/2026-09-02-climbing-recording"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    names, conv = load_conv(STAGE / "annotations/instances_train.json")
    cyan, red, white = mpl(PALETTE["detector"]), (1, 0.15, 0.15), (1, 1, 1)

    # Row A/B panels: four cameras BY NAME, wall-adjacent frames.
    panels = [("Cam2012862", 489102), ("Cam2012630", 4121),
              ("Cam2012855", 1447731), ("Cam2012857", 950756)]
    fig, axes = plt.subplots(3, 4, figsize=(24, 8.2))
    for c, (cam, fr) in enumerate(panels):
        img = np.asarray(Image.open(RAW / cam / f"Frame_{fr}.jpg").convert("L"))
        kp = conv[(cam, fr)]
        draw(axes[0, c], img, kp, names, cyan,
             label=f"A  CONVERTED   {cam}  Frame_{fr}   "
                   f"{int((kp[:,2]>0).sum())}/{len(names)} kp")
        draw(axes[1, c], img, kp, names, red, flip_h=img.shape[0],
             label=f"B  NO-FLIP CONTROL   {cam}  Frame_{fr}  (must be off the fly)")

    # Row C: this recording vs general_model/wall_frames, 19 frames apart.
    wall_names = None
    for sp in ("train", "val"):
        p = WALL / "annotations" / f"instances_{sp}.json"
        if p.exists():
            d = json.load(open(p))
            wall_names = d["keypoint_names"]
            by_img = {im["id"]: im["file_name"] for im in d["images"]}
            for a in d["annotations"]:
                parts = by_img[a["image_id"]].split("/")
                cam, fn = parts[-2], parts[-1]
                conv.setdefault(("WALL:" + cam, int(fn.split("_")[1].split(".")[0])),
                                np.asarray(a["keypoints"], float).reshape(-1, 3))
    assert wall_names == names, "wall_frames keypoint order differs from fly50"
    # wall_frames' 33 annotations are sparse (33 of 77 images), so the pair is
    # chosen from cameras that actually carry one, BY NAME.
    pairs = [("Cam2012853", 4012, 4031), ("Cam2012631", 6403, 6708)]
    for c, (cam, f_new, f_wall) in enumerate(pairs):
        a_img = np.asarray(Image.open(RAW / cam / f"Frame_{f_new}.jpg").convert("L"))
        draw(axes[2, 2 * c], a_img, conv[(cam, f_new)], names, cyan,
             label=f"C  THIS RECORDING  {cam}  Frame_{f_new}")
        wp = next(p for p in (WALL / sp / WALL_REC / cam / f"Frame_{f_wall}.jpg"
                              for sp in ("train", "val")) if p.exists())
        b_img = np.asarray(Image.open(wp).convert("L"))
        dt = (f_wall - f_new) / 800.0 * 1000
        draw(axes[2, 2 * c + 1], b_img, conv[("WALL:" + cam, f_wall)], names, white,
             label=f"C  general_model/wall_frames  {cam}  Frame_{f_wall}  "
                   f"(+{f_wall - f_new} frames = {dt:.0f} ms at 800 fps)")
    plt.suptitle(
        "wall_frames_15_06_46_male (2025_10_12_15_06_46), male climbing -- "
        "conversion acceptance.  A: converted labels on the fly.  "
        "B: without the bottom-origin flip they must leave the animal.  "
        "C: same capture as general_model/wall_frames -- one capture group.",
        fontsize=11)
    plt.tight_layout()
    dst = out / "wall_climbing_conversion_check.png"
    plt.savefig(dst, dpi=100)
    print("wrote", dst)


if __name__ == "__main__":
    main()
