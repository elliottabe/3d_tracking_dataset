#!/usr/bin/env python3
"""Visual acceptance for the 2026-09-02 raw->JARVIS courtship label conversion.

WHAT THIS FIGURE SHOULD SHOW IF THE CONVERSION IS CORRECT
---------------------------------------------------------
Row A  NEWLY ANNOTATED (the 18 images courtship_V2/V3/V4 left blank).
       Cyan keypoints and skeleton must sit ON the fly: Antenna_Base/EyeL/EyeR
       on the head, Abd_tip at the abdomen tip, and all six leg chains running
       down real legs, connected. These images had NO label before, so this is
       the supervision the re-export actually buys.

Row B  OLD vs NEW on an image both sources hold. White = the existing
       general_model annotation, cyan = the freshly converted one. They must
       COINCIDE exactly (verified numerically at 0/4721 differing arrays); any
       visible separation means the conversion rule drifted.

Row C  THE FLIP CONTROL, and the point of the whole figure. Same keypoints,
       same image, WITHOUT the `y = image_height - v_raw` flip. These must land
       mirrored about the 448-px midline and clearly OFF the animal. If red and
       cyan both look plausible, the flip is not actually determined by the
       data and nothing here is evidence.

Row D  THE HARD CASE: 20_04_female_climbing -- female, wall-adjacent, occluded,
       the regime this pipeline is documented to fail at, and the same physical
       capture as courtship_20_04_male. Same recipe, same expectation as row A.

Keypoints and cameras are resolved BY NAME throughout (fly50 order from
data/fly50.json); nothing is indexed by a bare integer.

    MUJOCO_GL=egl python scripts/viz/courtship_label_conversion_check.py \
        --out figures/2026-09-02-courtship-labels
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from viz.core.colors import PALETTE, keypoint_groups, leg_chains  # noqa: E402

RAW = Path("/gscratch/portia/eabe/data/Johnson_lab/red_data/courtship_label_2026_09_02")
GM = Path("/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model")
STAGE = Path("/gscratch/portia/eabe/data/Johnson_lab/red_data/courtship_labels_2026_09_02")
MISSING = 1e6


def bgr2mpl(c):
    return (c[2] / 255, c[1] / 255, c[0] / 255)


def read_raw_csv(path: Path, ndim: int):
    out = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            line = line.strip()
            if not line:
                continue
            v = line.split(",")
            frame, rest, step = int(v[0]), line.split(",")[1:], 1 + ndim
            n = len(rest) // step
            arr = np.full((n, ndim), np.nan)
            for k in range(n):
                blk = rest[k * step:(k + 1) * step]
                arr[int(blk[0])] = [float(x) for x in blk[1:]]
            out[frame] = arr
    return out


def draw(ax, kp, vis, names, color, lw=1.1, ms=7, label=None):
    chains = leg_chains(names)
    idx = {n: i for i, n in enumerate(names)}
    c = bgr2mpl(color)
    segs = list(chains.values())
    for side in ("L", "R"):
        tri = [f"Wing{side}_base", f"Wing{side}_V12", f"Wing{side}_V13", f"Wing{side}_base"]
        segs.append([idx[n] for n in tri if n in idx])
    segs.append([idx[n] for n in ("Antenna_Base", "EyeL", "Scutellum", "Abd_A4", "Abd_tip") if n in idx])
    segs.append([idx[n] for n in ("Antenna_Base", "EyeR", "Scutellum") if n in idx])
    first = True
    for ch in segs:
        pts = [(kp[i, 0], kp[i, 1]) for i in ch if vis[i]]
        if len(pts) > 1:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-", color=c, lw=lw, alpha=0.85,
                    label=label if first else None)
            first = False
    ax.plot(kp[vis, 0], kp[vis, 1], ".", color=c, ms=ms,
            label=(label if first else None))


def crop_about(kp, vis, w, h, pad=110):
    xs, ys = kp[vis, 0], kp[vis, 1]
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, w)
    return x0, x1, 0, h


def panel(ax, img, entries, names, title, pad=110):
    h, w = img.shape[:2]
    ref = entries[0]
    x0, x1, y0, y1 = crop_about(ref["kp"], ref["vis"], w, h, pad)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1])
    for e in entries:
        kp = e["kp"].copy()
        kp[:, 0] -= x0
        kp[:, 1] -= y0
        draw(ax, kp, e["vis"], names, e["color"], lw=e.get("lw", 1.1),
             ms=e.get("ms", 7), label=e["label"])
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "figures/2026-09-02-courtship-labels"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = json.loads((REPO / "data" / "fly50.json").read_text())["node_names"]
    conv = json.load(open(STAGE / "courtship_20_04_male/annotations/instances_train.json"))
    by_key = {}
    fn_of = {im["id"]: im["file_name"] for im in conv["images"]}
    for a in conv["annotations"]:
        m = re.match(r".*/(Cam\d+)/Frame_(\d+)\.jpg", fn_of[a["image_id"]])
        by_key[(m.group(1), int(m.group(2)))] = a
    img_path = {}
    for sub in ("courtship_V2", "courtship_V3", "courtship_V4"):
        for sp in ("train", "val"):
            base = GM / sub / sp
            if base.is_dir():
                for j in base.rglob("Cam*/Frame_*.jpg"):
                    img_path[(j.parent.name, int(j.stem.split("_")[1]))] = j
    # old annotations, for row B
    old = {}
    for sub in ("courtship_V2", "courtship_V3", "courtship_V4"):
        for sp in ("train", "val"):
            p = GM / sub / "annotations" / f"instances_{sp}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            b = {im["id"]: im["file_name"] for im in d["images"]}
            for a in d["annotations"]:
                m = re.match(r".*/(Cam\d+)/Frame_(\d+)\.jpg", b[a["image_id"]])
                old[(m.group(1), int(m.group(2)))] = a

    def as_kp(a):
        k = np.array(a["keypoints"], float).reshape(-1, 3)
        return k[:, :2], k[:, 2] > 0

    new_only = sorted(set(by_key) - set(old))
    rowA = [k for k in new_only if k[1] == 94198][:2] + [k for k in new_only if k[1] == 372364][:2]
    rowB = [("Cam2012853", 231521), ("Cam2012862", 240800)]

    fig, axes = plt.subplots(4, 4, figsize=(22, 11))
    for j, key in enumerate(rowA):
        a = by_key[key]
        kp, vis = as_kp(a)
        img = cv2.imread(str(img_path[key]))
        panel(axes[0, j], img, [{"kp": kp, "vis": vis, "color": PALETTE["detector"],
                                 "label": "converted (new label)"}], names,
              f"A  NEW label, previously blank\n{key[0]}  Frame_{key[1]}  "
              f"({int(vis.sum())}/50 kp)")
    for j, key in enumerate(rowB):
        a, o = by_key[key], old[key]
        kn, vn = as_kp(a)
        ko, vo = as_kp(o)
        img = cv2.imread(str(img_path[key]))
        panel(axes[1, j], img,
              [{"kp": ko, "vis": vo, "color": (255, 255, 255), "label": "existing general_model", "lw": 3.0, "ms": 12},
               {"kp": kn, "vis": vn, "color": PALETTE["detector"], "label": "converted", "lw": 1.0, "ms": 5}],
              names, f"B  OLD (white) vs NEW (cyan) -- must coincide\n{key[0]}  Frame_{key[1]}")
    # row B extra: zoom on head to show coincidence
    for j, key in enumerate(rowB):
        a, o = by_key[key], old[key]
        kn, vn = as_kp(a)
        ko, vo = as_kp(o)
        img = cv2.imread(str(img_path[key]))
        panel(axes[1, j + 2], img,
              [{"kp": ko, "vis": vo, "color": (255, 255, 255), "label": "existing", "lw": 3.0, "ms": 14},
               {"kp": kn, "vis": vn, "color": PALETTE["detector"], "label": "converted", "lw": 1.0, "ms": 5}],
              names, f"B(zoom)  {key[0]}  Frame_{key[1]}", pad=15)

    # row C: flip control
    for j, key in enumerate(rowA[:2] + rowB[:2]):
        a = by_key[key]
        kp, vis = as_kp(a)
        img = cv2.imread(str(img_path[key]))
        h = img.shape[0]
        wrong = kp.copy()
        wrong[:, 1] = h - 1 - wrong[:, 1]     # undo the flip == the WRONG convention
        panel(axes[2, j], img,
              [{"kp": kp, "vis": vis, "color": PALETTE["detector"], "label": "converted (flipped, correct)"},
               {"kp": wrong, "vis": vis, "color": PALETTE["head"], "label": "NO flip (control -- must be off the fly)"}],
              names, f"C  FLIP CONTROL\n{key[0]}  Frame_{key[1]}")

    # row D: the hard case -- female climbing, same capture
    fem_raw = RAW / "2025_10_20_13_20_04_female_climbing"
    fem_imgs = {}
    for sp in ("train", "val"):
        base = GM / "20_04_female_climbing" / sp
        if base.is_dir():
            for j2 in base.rglob("Cam*/Frame_*.jpg"):
                fem_imgs[(j2.parent.name, int(j2.stem.split("_")[1]))] = j2
    picks = sorted(fem_imgs)[:4]
    seen = set()
    picks = [k for k in sorted(fem_imgs) if not (k[0] in seen or seen.add(k[0]))][:4]
    for j, key in enumerate(picks):
        uv = read_raw_csv(fem_raw / f"{key[0]}.csv", 2)[key[1]]
        img = cv2.imread(str(fem_imgs[key]))
        h = img.shape[0]
        vis = (np.abs(uv) < MISSING).all(1)
        kp = np.stack([uv[:, 0], h - uv[:, 1]], 1)
        panel(axes[3, j], img,
              [{"kp": kp, "vis": vis, "color": PALETTE["fly0"], "label": "converted (female, climbing)"}],
              names, f"D  HARD CASE female climbing\n{key[0]}  Frame_{key[1]}  ({int(vis.sum())}/50 kp)")
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis("off")
    for r in range(4):
        axes[r, 0].legend(loc="upper left", fontsize=6, framealpha=0.75)
    fig.suptitle(
        "2026-09-02 courtship label conversion -- raw red3d CSV -> JARVIS "
        "(fly50 order, y = image_height - v_raw)\n"
        "A: 18 newly-annotated images   B: reproduces existing labels exactly   "
        "C: without the flip the skeleton leaves the animal   D: female climbing (hard case)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out / "conversion_check.png"
    fig.savefig(p, dpi=115)
    print("wrote", p)


if __name__ == "__main__":
    main()
