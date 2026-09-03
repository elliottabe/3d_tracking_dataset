"""Render the evidence for a content-keyed dataset root's merge and split.

    MUJOCO_GL=egl python scripts/viz/dataset_split_report.py \
        --root <dataset_root> --out figures/<topic>/split_report.png

WHAT EACH PANEL SHOULD LOOK LIKE IF THE BUILD IS CORRECT. Stated here so the
figure can disagree with it; a figure you cannot be wrong about proves nothing.

  A  THE CONTENT MERGE IS TWO ANIMALS, NOT ONE LABELLED TWICE.
     One image content that two per-fly subsets (courtship_<id>_female and
     _male, filed under two different recording timestamps) both point at,
     with BOTH annotations drawn -- the WORST case in the root by the metric
     that actually decides. Correct => even the worst case's keypoint sets lie
     on two different animals. Wrong => they lie on one animal, meaning the
     build attributed one animal's two labels to fly0 and fly1, every two-fly
     count is inflated, and the "two-fly" training signal is a chimera.

     RANK BY MEDIAN KEYPOINT DISPLACEMENT, NOT BY BBOX-CENTROID SEPARATION.
     This panel first ranked by centroid separation and produced a 14 px
     "hardest case" that looked exactly like one animal labelled twice -- but
     R15 gates on median keypoint displacement, and that same frame measures
     90 px. Two flies in a close courtship posture have near-concentric
     bounding boxes while their keypoints stay far apart, so the centroid is
     the wrong discriminator and picking it manufactures a false alarm.
     Panel E shows both distributions and where the threshold sits.

  B  THE SAME-ANIMAL CASE IS CAUGHT, NOT ASSIGNED.
     One of the (camera, content) slots the build recorded ABSENT because both
     subsets landed on the SAME animal. Correct => the two marker sets are
     nearly coincident (median displacement a few px, well under SAME_FLY_PX).
     If they were far apart, the build discarded a real second animal.

  C  THE HARD CASE, NOT THE FLATTERING ONE.
     A wall_frames capture -- female, against a wall, the regime this pipeline
     is documented to fail at. Correct => visibly INCOMPLETE camera coverage
     (its annotated-camera count tops out at 6/7 and four of its eleven frames
     fall below MIN_CAMS=3). That incompleteness is the mechanical reason it
     yields only 7 viable framesets and cannot support a held-out metric.

  E  THE SEPARATION DISTRIBUTION, WITH THE THRESHOLD DRAWN.
     Correct => ZERO merged two-fly pairs fall below SAME_FLY_PX on the median
     keypoint metric (they are exactly the ones R15 removed), and the surviving
     mass sits far above it. Any merged pair under the line is a chimera.

  D  WHERE THE DATA WENT.
     Images per subset, train vs val, coloured by the sex on record. Correct
     => val contains real female subsets and real two-fly courtship subsets,
     not only the easy male ones.

Colours follow viz/core/colors.py: cyan = fly0, orange = fly1. In every
courtship pair the female subset sorts first, so fly0 = female and fly1 = male;
panels label by SUBSET NAME and SEX, never by a bare index.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from PIL import Image                    # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from viz.core.colors import PALETTE      # noqa: E402

SAME_FLY_PX = 50.0      # mirrors build_generalmodel_split.SAME_FLY_PX


def _rgb(name):
    b, g, r = PALETTE[name]
    return (r / 255.0, g / 255.0, b / 255.0)


FEMALE, MALE = _rgb("fly0"), _rgb("fly1")     # cyan, orange


def _kp(ann):
    k = np.asarray(ann["keypoints"], float).reshape(-1, 3)
    return k[k[:, 2] > 0][:, :2]


def _draw(ax, root, image, anns, title):
    img = np.asarray(Image.open(os.path.join(root, "images", image["file_name"])))
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    for a in anns:
        xy = _kp(a)
        c = FEMALE if a["sex"] == "female" else MALE
        ax.scatter(xy[:, 0], xy[:, 1], s=9, color=c, edgecolor="k",
                   linewidth=0.25, zorder=3,
                   label=f"{a['subset']} ({a['sex']}, fly{a['fly_id']})")
        x, y, w, h = a["bbox"]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color=c, lw=1.2))
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inst = json.load(open(os.path.join(args.root, "annotations", "instances.json")))
    report = json.load(open(os.path.join(args.root, "build_report.json")))
    img_by_id = {i["id"]: i for i in inst["images"]}
    per_img = collections.defaultdict(list)
    for a in inst["annotations"]:
        per_img[a["image_id"]].append(a)

    def centroid_sep(anns):
        c = [np.array([a["bbox"][0] + a["bbox"][2] / 2,
                       a["bbox"][1] + a["bbox"][3] / 2]) for a in anns]
        return float(np.linalg.norm(c[0] - c[1]))

    def median_kp_sep(anns):
        """The metric R15 gates on: median displacement over keypoints visible
        in BOTH annotations. This -- not the bbox centroid -- is what decides
        whether two labels are two animals."""
        ka, kb = (np.asarray(a["keypoints"], float).reshape(-1, 3) for a in anns)
        m = (ka[:, 2] > 0) & (kb[:, 2] > 0)
        if not m.any():
            return float("inf")
        return float(np.median(np.linalg.norm(ka[m, :2] - kb[m, :2], axis=1)))

    # A: the hardest two-fly case BY THE DECISION METRIC (see docstring)
    two = [(median_kp_sep(v), centroid_sep(v), i)
           for i, v in per_img.items() if len(v) == 2]
    two.sort()
    d_a, c_a, id_a = two[0]
    # B: a slot the build recorded ABSENT for landing on the same animal
    amb = report.get("ambiguous_examples", [])

    fig = plt.figure(figsize=(19, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.30, wspace=0.14)

    ax = fig.add_subplot(gs[0, 0])
    _draw(ax, args.root, img_by_id[id_a], sorted(per_img[id_a],
                                                 key=lambda a: a["fly_id"]),
          f"A  content merge, WORST of {len(two)} two-fly images by the "
          f"R15 decision metric\n{img_by_id[id_a]['file_name']}\n"
          f"median keypoint displacement = {d_a:.0f} px  (threshold "
          f"{SAME_FLY_PX:.0f} px)   bbox-centroid = {c_a:.0f} px")

    ax = fig.add_subplot(gs[0, 1])
    if amb:
        e = amb[0]
        hit = [i for i, im in img_by_id.items()
               if im["content_md5"] == e["content_md5"]]
        if hit:
            # the ambiguous slots were recorded ABSENT, so re-read them from
            # the SOURCE subsets rather than from the merged annotations
            gm = json.load(open(os.path.join(
                args.root, "manifest.json")))["source_root"]
            shown = []
            for sub in e["subsets"]:
                for sp in ("train", "val"):
                    p = os.path.join(gm, sub, "annotations", f"instances_{sp}.json")
                    if not os.path.exists(p):
                        continue
                    b = json.load(open(p))
                    want = f"/{e['camera']}/Frame_{e['frame']}.jpg"
                    ids = {im["id"] for im in b["images"]
                           if im["file_name"].endswith(want)}
                    for a in b["annotations"]:
                        if a["image_id"] in ids:
                            shown.append(dict(a, subset=sub,
                                              sex="female" if "female" in sub
                                              else "male",
                                              fly_id=e["subsets"].index(sub)))
            _draw(ax, args.root, img_by_id[hit[0]], shown,
                  f"B  R15: two subsets on the SAME animal -> both slots ABSENT\n"
                  f"{e['recording']} {e['camera']} Frame_{e['frame']}   "
                  f"median keypoint displacement = {e['median_kp_px']:.1f} px "
                  f"(< SAME_FLY_PX)")
    else:
        ax.axis("off"); ax.set_title("B  no ambiguous slots in this root", fontsize=8)

    # C: the hard regime -- wall_frames camera coverage
    # E: the separation distribution with the threshold drawn
    ax = fig.add_subplot(gs[0, 2])
    mk = np.array([t[0] for t in two])
    bc = np.array([t[1] for t in two])
    bins = np.linspace(0, max(mk.max(), bc.max()), 45)
    ax.hist(bc, bins=bins, color="0.75", label="bbox-centroid (WRONG metric)")
    ax.hist(mk, bins=bins, histtype="step", lw=1.8, color="k",
            label="median keypoint displacement (R15)")
    ax.axvline(SAME_FLY_PX, color="crimson", lw=1.6)
    ax.text(SAME_FLY_PX + 6, ax.get_ylim()[1] * 0.92,
            f"SAME_FLY_PX = {SAME_FLY_PX:.0f}\n{(mk < SAME_FLY_PX).sum()} merged pairs "
            f"below\n(must be 0)", fontsize=6.5, color="crimson", va="top")
    ax.set_xlabel("separation between the two annotations (px)", fontsize=7)
    ax.set_ylabel("two-fly images", fontsize=7)
    ax.set_title(f"E  why the centroid is the wrong discriminator\n"
                 f"min median-kp = {mk.min():.0f} px vs "
                 f"min centroid = {bc.min():.0f} px", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[1, 0])
    wall = [i for i, im in img_by_id.items() if "wall" in
            json.load(open(os.path.join(args.root, "manifest.json")))
            ["recordings"].get(im["recording"], {}).get("behavior", "")]
    if wall:
        cov = collections.Counter()
        for i in wall:
            fr = img_by_id[i]["file_name"].split("/")[-1]
            cov[fr] += 1 if per_img[i] else 0
        best = max(wall, key=lambda i: len(per_img[i]))
        _draw(ax, args.root, img_by_id[best], per_img[best],
              f"C  HARD CASE: wall_frames (female, against a wall)\n"
              f"{img_by_id[best]['file_name']}   annotated cameras per frame: "
              f"{sorted(cov.values())} of 7")
    else:
        ax.axis("off")

    # D: composition
    ax = fig.add_subplot(gs[1, 1:])
    comp = report["composition"]["per_subset"]
    man = json.load(open(os.path.join(args.root, "manifest.json")))
    sex_of = {}
    for a in inst["annotations"]:
        sex_of[a["subset"]] = a["sex"]
    subs = sorted(comp, key=lambda s: -(comp[s].get("train", {}).get("images", 0)
                                        + comp[s].get("val", {}).get("images", 0)))
    y = np.arange(len(subs))
    tr = [comp[s].get("train", {}).get("images", 0) for s in subs]
    va = [comp[s].get("val", {}).get("images", 0) for s in subs]
    cols = [FEMALE if sex_of.get(s) == "female" else MALE for s in subs]
    ax.barh(y, tr, color=cols, alpha=0.45, label="train")
    ax.barh(y, va, left=tr, color=cols, alpha=1.0, edgecolor="k", lw=0.6,
            label="val (solid)")
    for i, s in enumerate(subs):
        if va[i]:
            ax.text(tr[i] + va[i] + 60, i, f"val {va[i]}", va="center", fontsize=5.5)
    ax.set_yticks(y); ax.set_yticklabels(subs, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("images (cyan = female on record, orange = male on record)",
                  fontsize=7)
    t = report["composition"]["totals"]
    ax.set_title(f"D  split composition   train {t['train']['images']} img / "
                 f"val {t['val']['images']} img "
                 f"({t['val_image_frac']:.1%})   "
                 f"female anns: train "
                 f"{report['composition']['female_budget']['female_anns_train']}, "
                 f"val {report['composition']['female_budget']['female_anns_val']}",
                 fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    ax.tick_params(axis="x", labelsize=6)

    fig.suptitle(f"{os.path.basename(args.root)}: content merge, R15, hard case, "
                 f"and split composition", fontsize=10)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("wrote", args.out)
    json.dump({"panelA_min_median_kp_px": d_a,
               "panelA_bbox_centroid_px": c_a,
               "panelA_image": img_by_id[id_a]["file_name"],
               "n_two_fly_images": len(two),
               "min_median_kp_px": float(mk.min()),
               "min_bbox_centroid_px": float(bc.min()),
               "merged_pairs_below_SAME_FLY_PX": int((mk < SAME_FLY_PX).sum()),
               "panelB": amb[0] if amb else None,
               "totals": report["composition"]["totals"]},
              open(os.path.splitext(args.out)[0] + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
