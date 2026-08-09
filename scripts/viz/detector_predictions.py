#!/usr/bin/env python3
"""Visualise a trained 2D detector's predictions on a multi-camera frameset,
plus the 3D keypoints triangulated from them.

Motivation: val MPJPE is a scalar over annotated crops. It says nothing about
whether the predicted skeleton is anatomically sensible, and it cannot see the
train-only categories at all (`wall`, `grooming`, `female_general` each have a
single recording, so the stratified split keeps them out of val entirely). The
wall case is the one the pipeline audit flagged as hardest, so it needs looking
at directly rather than through a number.

What it draws, per frameset:
  * one panel per camera: the 448px crop with GT (green) and predicted
    (magenta) skeletons overlaid, titled with that view's mean 2D error;
  * a 3D panel: keypoints triangulated (DLT) from the predicted 2D across all
    contributing views, and, where GT exists in >=2 views, the GT 3D too;
  * per-keypoint 3D reprojection error, as the honest check on whether the 2D
    predictions are mutually consistent in 3D.

Annotated views are cropped from their GT bbox. Views with no annotation are
still predicted on: the script triangulates from the annotated views first,
reprojects that 3D centroid into the un-annotated cameras, and crops there.
Those panels are labelled "inferred crop" -- their predictions are shown but,
having no GT, they are excluded from the error statistics.

Usage:
    python scripts/viz/detector_predictions.py \\
        --ckpt /gscratch/.../jax_vitpose_runs/v4_8gpu_20260808/final \\
        --subset wall_frames --out docs/benchmark/detector-v4/

    # a specific frameset, and the whole subset:
    python scripts/viz/detector_predictions.py --subset wall_frames \\
        --frameset Frame_6822.jpg ...
    python scripts/viz/detector_predictions.py --subset headless_22_50 --all ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
for p in (str(PROJECT_DIR), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_SOURCE_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model"
DEFAULT_CKPT = ("/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/"
                "v4_8gpu_20260808/final")

# Keypoint groups, for colouring. Matched against the fly50 names by prefix.
GROUPS = [
    ("head",  ("Antenna", "Eye"),                  "#e41a1c"),
    ("trunk", ("Scutellum", "Abd"),                "#377eb8"),
    ("wingL", ("WingL",),                          "#4daf4a"),
    ("wingR", ("WingR",),                          "#984ea3"),
    ("legL",  ("T1L", "T2L", "T3L"),               "#ff7f00"),
    ("legR",  ("T1R", "T2R", "T3R"),               "#a65628"),
]


def group_of(name):
    for gname, prefixes, colour in GROUPS:
        if name.startswith(prefixes):
            return gname, colour
    return "other", "#999999"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_subset(source_root, subset):
    """Merge a subset's train+val COCO jsons, keeping each image's orig_split.

    Framesets/calibrations live only in the SOURCE subset jsons -- the unified
    V4 build drops both (it is a 2D detector dataset), so 3D work has to read
    from here rather than from the V4 root.
    """
    merged = {"images": {}, "anns": {}, "framesets": {}, "calibrations": {},
              "keypoint_names": None}
    for split in ("train", "val"):
        path = Path(source_root) / subset / "annotations" / f"instances_{split}.json"
        if not path.is_file():
            continue
        coco = json.loads(path.read_text())
        merged["keypoint_names"] = merged["keypoint_names"] or coco.get("keypoint_names")
        merged["framesets"].update(coco.get("framesets", {}))
        merged["calibrations"].update(coco.get("calibrations", {}))
        id2img = {}
        for im in coco["images"]:
            rec = dict(im)
            rec["orig_split"] = split
            id2img[im["id"]] = rec
            merged["images"][(split, im["id"])] = rec
        for a in coco["annotations"]:
            merged["anns"][(split, a["image_id"])] = a
    return merged


def framesets_of(data):
    """(recording, frame_file) -> {camera: (image_record, annotation_or_None)}."""
    out = defaultdict(dict)
    for (split, img_id), im in data["images"].items():
        rec, cam, frame = im["file_name"].split("/")
        ann = data["anns"].get((split, img_id))
        out[(rec, frame)][cam] = (im, ann)
    return out


def crop_origin_np(bbox, img_w, img_h, crop):
    from jarvis_jax.data.transforms import crop_origin
    return crop_origin(np.asarray(bbox, np.float32), img_w, img_h, crop)


def load_crop(source_root, subset, im, x0, y0, crop):
    """(crop,crop,4) uint8: RGB + an all-zero mask channel.

    The mask channel is zero because these subsets carry no sam3_masks/ dir --
    which is also what V3Dataset feeds the trainer for them, so inference here
    matches training conditions rather than diverging from them.
    """
    from PIL import Image
    path = Path(source_root) / subset / im["orig_split"] / im["file_name"]
    with Image.open(path) as pil:
        img = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    h, w = img.shape[:2]
    patch = np.zeros((crop, crop, 3), np.uint8)
    sub = img[y0:y0 + crop, x0:x0 + crop]
    patch[:sub.shape[0], :sub.shape[1]] = sub
    return np.concatenate([patch, np.zeros((crop, crop, 1), np.uint8)], axis=-1)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def load_model(ckpt_dir, num_keypoints=50):
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.scripts.eval_keypoints_2d import restore_model
    cfg = ViTPoseConfig(num_keypoints=num_keypoints)
    return restore_model(ckpt_dir, "vitpose", cfg), cfg


def predict_crops(model, crops_u8, decode_sharpen=3.0):
    """(B,448,448,4) uint8 -> (kp2d (B,K,2) in crop px, conf (B,K))."""
    import jax.numpy as jnp
    from jarvis_jax.data.device import normalize_image
    from jarvis_jax.tracking.predict_2d import peaks_and_conf
    x = normalize_image(jnp.asarray(np.asarray(crops_u8)))
    hm = model(x, use_running_average=True)
    kp, conf = peaks_and_conf(hm, decode_sharpen=decode_sharpen)
    return np.asarray(kp), np.asarray(conf)


# ---------------------------------------------------------------------------
# per-frameset pipeline
# ---------------------------------------------------------------------------

def process_frameset(model, data, source_root, subset, rec, frame, views,
                     calib_dir, crop=448, conf_thresh=0.3, decode_sharpen=3.0):
    """Run the detector on every camera of one frameset and triangulate.

    Two passes: annotated views are cropped from their GT bbox; the resulting
    3D is reprojected to place a crop in the remaining cameras, which are then
    predicted on too (marked inferred=True, and excluded from 2D error stats
    since they have no GT).
    """
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.triangulate import triangulate_keypoints
    from scripts.build_detector_dataset import SUBSET_RULES, expand_to_canonical

    rt = ReprojectionTool(str(calib_dir))
    cam_names = list(rt.cameras.keys())          # ReprojectionTool's own order
    C = len(cam_names)
    K = 50

    panels = {}
    # --- pass 1: annotated views ------------------------------------------
    todo = [(cam, v) for cam, v in views.items() if v[1] is not None]
    if not todo:
        return None
    crops, meta = [], []
    for cam, (im, ann) in todo:
        w, h = im["width"], im["height"]
        x0, y0 = crop_origin_np(ann["bbox"], w, h, crop)
        crops.append(load_crop(source_root, subset, im, x0, y0, crop))
        meta.append((cam, im, ann, x0, y0))
    kp, conf = predict_crops(model, np.stack(crops), decode_sharpen)
    for i, (cam, im, ann, x0, y0) in enumerate(meta):
        # Source subsets are NOT all 50-node: headless stores 47 and the
        # amputated sets 44. expand_to_canonical maps them into fly50 order
        # with v=0 on the absent nodes, which is what the detector's 50
        # output channels are indexed against.
        gt, present = expand_to_canonical(ann["keypoints"], SUBSET_RULES[subset])
        panels[cam] = dict(crop=crops[i], x0=x0, y0=y0, kp=kp[i], conf=conf[i],
                           gt_xy=gt[:, :2].astype(np.float32),
                           gt_v=(gt[:, 2] > 0) & present, inferred=False)

    # --- triangulate pass-1 predictions ------------------------------------
    def triangulate(panels_):
        kp2d = np.zeros((1, C, K, 2), np.float32)
        cf = np.zeros((1, C, K), np.float32)
        for ci, cam in enumerate(cam_names):
            p = panels_.get(cam)
            if p is None:
                continue
            kp2d[0, ci] = p["kp"] + np.array([p["x0"], p["y0"]], np.float32)
            cf[0, ci] = p["conf"]
        kp3d, conf3d = triangulate_keypoints(kp2d, cf, rt.camera_matrices,
                                             conf_thresh=conf_thresh)
        return kp3d[0], conf3d[0], kp2d[0], cf[0]

    kp3d, conf3d, _, _ = triangulate(panels)

    # --- pass 2: un-annotated views, cropped from the reprojected 3D -------
    missing = [c for c in cam_names if c not in panels and c in views]
    good = np.isfinite(kp3d).all(1)
    if missing and good.sum() >= 3:
        centre3d = np.nanmedian(kp3d[good], axis=0)
        proj = rt.reproject_point(centre3d)                      # (C,2)
        crops2, meta2 = [], []
        for cam in missing:
            im = views[cam][0]
            ci = cam_names.index(cam)
            cx, cy = proj[ci]
            bbox = [cx - crop / 4, cy - crop / 4, crop / 2, crop / 2]
            x0, y0 = crop_origin_np(bbox, im["width"], im["height"], crop)
            crops2.append(load_crop(source_root, subset, im, x0, y0, crop))
            meta2.append((cam, im, x0, y0))
        if crops2:
            kp2, conf2 = predict_crops(model, np.stack(crops2), decode_sharpen)
            for i, (cam, im, x0, y0) in enumerate(meta2):
                panels[cam] = dict(crop=crops2[i], x0=x0, y0=y0, kp=kp2[i],
                                   conf=conf2[i], gt_xy=None, gt_v=None,
                                   inferred=True)
            kp3d, conf3d, _, _ = triangulate(panels)

    # --- GT 3D, where >=2 views annotated it -------------------------------
    gt2d = np.zeros((1, C, K, 2), np.float32)
    gtcf = np.zeros((1, C, K), np.float32)
    for ci, cam in enumerate(cam_names):
        p = panels.get(cam)
        if p is None or p["gt_xy"] is None:
            continue
        gt2d[0, ci] = np.nan_to_num(p["gt_xy"])
        gtcf[0, ci] = p["gt_v"].astype(np.float32)
    gt3d, gtconf3d = triangulate_keypoints(gt2d, gtcf, rt.camera_matrices,
                                           conf_thresh=0.5)

    # --- reprojection error of the predicted 3D ----------------------------
    reproj_err = np.full(K, np.nan)
    for k in range(K):
        if not np.isfinite(kp3d[k]).all():
            continue
        pr = rt.reproject_point(kp3d[k])
        errs = []
        for ci, cam in enumerate(cam_names):
            p = panels.get(cam)
            if p is None or p["conf"][k] < conf_thresh:
                continue
            obs = p["kp"][k] + np.array([p["x0"], p["y0"]])
            errs.append(np.linalg.norm(pr[ci] - obs))
        if errs:
            reproj_err[k] = float(np.mean(errs))

    return dict(panels=panels, cam_names=cam_names, kp3d=kp3d, conf3d=conf3d,
                gt3d=gt3d[0], gtconf3d=gtconf3d[0], reproj_err=reproj_err,
                rec=rec, frame=frame, rt=rt)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def draw(result, kp_names, edges, out_path, subset, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    panels, cam_names = result["panels"], result["cam_names"]
    shown = [c for c in cam_names if c in panels]
    ncam = len(shown)
    ncols = 4
    cam_rows = int(np.ceil(ncam / ncols))
    # Cameras fill the top rows; the bottom row is reserved for two 3D views
    # (one angle is not enough to tell a plausible skeleton from a folded one)
    # and a per-group reprojection summary.
    fig = plt.figure(figsize=(4.0 * ncols, 4.0 * cam_rows + 4.6))
    gs = fig.add_gridspec(cam_rows + 1, ncols,
                          height_ratios=[1.0] * cam_rows + [1.15])

    colours = [group_of(n)[1] for n in kp_names]

    # --- 2D panels ---------------------------------------------------------
    for i, cam in enumerate(shown):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        p = panels[cam]
        ax.imshow(p["crop"][:, :, :3])
        for (a, b) in edges:
            for xy, col, lw in ((p["kp"], "magenta", 1.2),):
                if p["conf"][a] > 0.3 and p["conf"][b] > 0.3:
                    ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]],
                            "-", color=col, lw=lw, alpha=0.75)
        ax.scatter(p["kp"][:, 0], p["kp"][:, 1], s=9, c=colours,
                   edgecolors="k", linewidths=0.3, zorder=3)
        sub = ""
        if p["gt_xy"] is not None:
            g = p["gt_xy"] - np.array([p["x0"], p["y0"]])
            v = p["gt_v"]
            ax.scatter(g[v, 0], g[v, 1], s=26, facecolors="none",
                       edgecolors="lime", linewidths=1.1, zorder=4)
            err = np.linalg.norm(p["kp"][v] - g[v], axis=1)
            sub = f"  err {np.mean(err):.1f}px"
        ax.set_title(f"{cam}{'  [inferred crop]' if p['inferred'] else ''}{sub}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # --- 3D panels (two angles) --------------------------------------------
    kp3d = result["kp3d"]
    ok = np.isfinite(kp3d).all(1)
    g3 = result["gt3d"]
    gok = np.isfinite(g3).all(1)
    re = result["reproj_err"]
    med = np.nanmedian(re) if np.isfinite(re).any() else float("nan")

    for j, (elev, azim) in enumerate(((18, -60), (18, 30))):
        ax3 = fig.add_subplot(gs[cam_rows, j], projection="3d")
        for (a, b) in edges:
            if ok[a] and ok[b]:
                ax3.plot(*[[kp3d[a, d], kp3d[b, d]] for d in range(3)],
                         "-", color="magenta", lw=1.1, alpha=0.75)
        ax3.scatter(kp3d[ok, 0], kp3d[ok, 1], kp3d[ok, 2],
                    c=[colours[i] for i in np.where(ok)[0]], s=26,
                    edgecolors="k", linewidths=0.3, depthshade=False)
        if gok.any():
            ax3.scatter(g3[gok, 0], g3[gok, 1], g3[gok, 2], marker="o", s=52,
                        facecolors="none", edgecolors="lime", linewidths=1.0,
                        depthshade=False)
        if ok.any():
            # Equal aspect: a fly plotted on auto-scaled axes looks folded
            # even when the 3D is fine.
            c = kp3d[ok].mean(0)
            r = max(float(np.abs(kp3d[ok] - c).max()), 1e-3) * 1.05
            ax3.set_xlim(c[0] - r, c[0] + r); ax3.set_ylim(c[1] - r, c[1] + r)
            ax3.set_zlim(c[2] - r, c[2] + r)
            ax3.set_box_aspect((1, 1, 1))
        ax3.view_init(elev=elev, azim=azim)
        ax3.tick_params(labelsize=6)
        ax3.set_title(
            f"3D (DLT)  view {j + 1}   {int(ok.sum())}/{len(ok)} kp\n"
            f"median reproj {med:.2f}px" if j == 0 else
            f"3D (DLT)  view {j + 1}", fontsize=9)

    # --- reprojection error by keypoint group -------------------------------
    axe = fig.add_subplot(gs[cam_rows, 2])
    gnames, gvals, gcols = [], [], []
    for gname, prefixes, colour in GROUPS:
        idx = [i for i, n in enumerate(kp_names) if n.startswith(prefixes)]
        vals = re[idx]
        if np.isfinite(vals).any():
            gnames.append(gname); gvals.append(float(np.nanmedian(vals)))
            gcols.append(colour)
    axe.barh(gnames, gvals, color=gcols)
    axe.set_xlabel("median 3D reprojection error (px)", fontsize=8)
    axe.set_title("consistency by body part", fontsize=9)
    axe.tick_params(labelsize=7)
    axe.grid(axis="x", alpha=0.3)

    # --- legend / provenance ------------------------------------------------
    axl = fig.add_subplot(gs[cam_rows, 3])
    axl.axis("off")
    n_inf = sum(1 for c in shown if panels[c]["inferred"])
    lines = [
        "magenta  = predicted (this checkpoint)",
        "lime O   = ground-truth annotation",
        "dot colour = body group (see bar chart)",
        "",
        f"cameras predicted:  {ncam}",
        f"  annotated (GT):   {ncam - n_inf}",
        f"  inferred crop:    {n_inf}  (no GT, excluded from err)",
        f"keypoints in 3D:    {int(ok.sum())}/{len(ok)}",
        f"median reproj err:  {med:.2f} px",
        "",
        "3D is DLT-triangulated from the PREDICTED 2D,",
        "so reprojection error measures whether the",
        "views agree -- not accuracy against truth.",
    ]
    axl.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8.5,
             family="monospace", transform=axl.transAxes)

    n_gt_views = sum(1 for c in shown if panels[c]["gt_xy"] is not None)
    fig.suptitle(
        f"{subset}  {result['rec']}/{result['frame']}   "
        f"{ncam} cameras ({n_gt_views} annotated){title_extra}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return med


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--subset", default="wall_frames")
    ap.add_argument("--frameset", default=None,
                    help="frame file name, e.g. Frame_6822.jpg (default: the "
                         "frameset with the most annotated views)")
    ap.add_argument("--all", action="store_true", help="render every frameset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="docs/benchmark/detector-v4")
    ap.add_argument("--conf-thresh", type=float, default=0.3)
    ap.add_argument("--decode-sharpen", type=float, default=3.0)
    args = ap.parse_args()

    from scripts.build_detector_dataset import FLY50, FLY50_EDGES

    data = load_subset(args.source_root, args.subset)
    fss = framesets_of(data)
    if not fss:
        raise SystemExit(f"no framesets found for subset {args.subset!r}")

    ranked = sorted(fss.items(),
                    key=lambda kv: -sum(1 for v in kv[1].values() if v[1] is not None))
    if args.frameset:
        ranked = [kv for kv in ranked if kv[0][1] == args.frameset]
        if not ranked:
            raise SystemExit(f"frameset {args.frameset!r} not in {args.subset}")
    elif not args.all:
        ranked = ranked[:1]
    if args.limit:
        ranked = ranked[:args.limit]

    model, _ = load_model(args.ckpt)
    out_dir = Path(args.out)
    print(f"checkpoint: {args.ckpt}\nsubset: {args.subset}  "
          f"framesets to render: {len(ranked)}")

    for (rec, frame), views in ranked:
        calib_dir = Path(args.source_root) / args.subset / "calib_params" / rec
        if not calib_dir.is_dir():
            print(f"  SKIP {rec}/{frame}: no calib dir at {calib_dir}")
            continue
        res = process_frameset(model, data, args.source_root, args.subset,
                               rec, frame, views, calib_dir,
                               conf_thresh=args.conf_thresh,
                               decode_sharpen=args.decode_sharpen)
        if res is None:
            print(f"  SKIP {rec}/{frame}: no annotated views")
            continue
        stem = f"{args.subset}_{rec}_{Path(frame).stem}.png"
        med = draw(res, FLY50, FLY50_EDGES, out_dir / stem, args.subset)
        n_ann = sum(1 for v in views.values() if v[1] is not None)
        n3d = int(np.isfinite(res["kp3d"]).all(1).sum())
        print(f"  {rec}/{frame}: {n_ann} annotated views, "
              f"{len(res['panels'])} predicted, {n3d}/50 kp in 3D, "
              f"median reproj {med:.2f}px -> {out_dir / stem}")


if __name__ == "__main__":
    main()
