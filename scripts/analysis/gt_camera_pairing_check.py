"""Is every GT 2D keypoint annotation paired with the image it was drawn on?

The question
------------
A reviewer noticed GT skeletons looking shifted when overlaid on the camera
frames and asked whether the annotations are being plotted on the right
cameras.  A photometric "does the skeleton land on dark pixels" proxy was
ambiguous, so this script runs the decisive geometric test instead.

The test
--------
Multi-view geometry cannot be faked.  For one frameset (one recording, one
frame, one fly, seven cameras) take the seven annotated 2-D keypoint sets and,
for EVERY assignment of keypoint-sets to cameras, triangulate each keypoint by
DLT across all seven views and reproject it.  If the seven camera rays for a
keypoint really do meet at one 3-D point, the reprojection residual is a few
pixels; if two cameras' annotations are swapped, the rays for that keypoint no
longer meet and the residual explodes by orders of magnitude.  The permutation
that minimises the residual IS the true camera assignment.

Expectation if the dataset is CORRECT
-------------------------------------
The identity permutation is the unique minimiser, its residual is single-digit
pixels, and the next-best permutation is at least an order of magnitude worse.

Expectation if the dataset is BROKEN
------------------------------------
Some non-identity permutation wins by a wide margin, and it is the SAME
permutation on every frameset of an affected recording (a systematic mapping
error), rather than a different one per frame (which would just be noise).

Order discipline
----------------
Camera identity is resolved BY NAME throughout: the camera of an annotation
comes from its image's ``file_name`` path component, and it is looked up in the
``ReprojectionTool``'s glob-sorted camera list by that name.  No integer camera
index is ever assumed to line up with anything.  Keypoints are likewise indexed
through ``keypoint_names`` from the annotation file itself.

Usage
-----
    python scripts/analysis/gt_camera_pairing_check.py --mode summary
    python scripts/analysis/gt_camera_pairing_check.py --mode permute --n-framesets 40
    python scripts/analysis/gt_camera_pairing_check.py --mode splits
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "third_party", "jarvis_jax"))
sys.path.insert(0, REPO)  # for viz.core.colors (shared visual language)

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool  # noqa: E402

DEFAULT_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5_valfix"


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
class Dataset:
    def __init__(self, root: str, ann_file: str = "instances.json"):
        self.root = root
        self.manifest = json.load(open(os.path.join(root, "manifest.json")))
        self.d = json.load(open(os.path.join(root, "annotations", ann_file)))
        self.kp_names: list[str] = self.d["keypoint_names"]
        self.images = {im["id"]: im for im in self.d["images"]}
        self.anns = {a["id"]: a for a in self.d["annotations"]}
        self.framesets: dict = self.d["framesets"]
        self._tools: dict[str, ReprojectionTool] = {}

    def tool(self, group: str) -> ReprojectionTool:
        if group not in self._tools:
            self._tools[group] = ReprojectionTool(
                os.path.join(self.root, "calibrations", group))
        return self._tools[group]

    @staticmethod
    def camera_of(image: dict) -> str:
        # file_name is "<recording>/<CamXXXXXXX>/Frame_NNNNNN.jpg"
        return image["file_name"].split("/")[1]

    def calib_group(self, recording: str) -> str:
        return self.manifest["recordings"][recording]["calib_group"]

    def frameset_obs(self, key: str):
        """Return (cam_names, obs (K, n, 3)) for a frameset, ordered by the
        ReprojectionTool's camera order, resolved BY NAME."""
        f = self.framesets[key]
        rec = f["recording"]
        rt = self.tool(self.calib_group(rec))
        cam_names = [c.name for c in rt._camera_list]
        name2idx = {n: i for i, n in enumerate(cam_names)}
        K = len(self.kp_names)
        obs = np.full((K, len(cam_names), 3), np.nan)
        seen = {}
        for iid, aid in zip(f["frames"], f["ann_ids"]):
            if iid is None or aid is None:
                continue  # frameset with a camera missing an annotation
            cam = self.camera_of(self.images[iid])
            kp = np.asarray(self.anns[aid]["keypoints"], float).reshape(-1, 3)
            ci = name2idx[cam]
            seen[cam] = aid
            obs[:, ci, :] = kp
        return rt, cam_names, obs, seen, rec


# ----------------------------------------------------------------------------
# per-frameset geometry
# ----------------------------------------------------------------------------
def identity_residuals(rt, obs, min_views=3):
    """Per-camera reprojection residual under the identity assignment.

    Returns dict cam_index -> list of (err_px, dx, dy)."""
    K, C, _ = obs.shape
    out = defaultdict(list)
    for k in range(K):
        use = np.where(obs[k, :, 2] > 0)[0]
        if len(use) < min_views:
            continue
        X = rt.reconstruct_point(obs[k, :, :2], cams_to_use=list(use))
        rp = rt.reproject_point(X)
        for ci in use:
            d = rp[ci] - obs[k, ci, :2]
            out[int(ci)].append((float(np.hypot(*d)), float(d[0]), float(d[1])))
    return out


def permutation_scan(rt, obs, max_perms=None):
    """Total reprojection residual for every assignment of keypoint-sets to
    cameras.  Only keypoints visible in ALL cameras are used, so every
    permutation is scored on exactly the same data.

    obs[:, a] is the keypoint set that the dataset assigns to camera a.
    Permutation p means: camera c is scored against obs[:, p[c]].
    Identity p == (0..C-1) is the dataset's own claim.
    """
    C = obs.shape[1]
    if not np.all(np.any(np.isfinite(obs[:, :, 0]), axis=0)):
        return None  # a camera has no annotation at all -> permutation is moot
    full = np.all(obs[:, :, 2] > 0, axis=1)
    xy = obs[full][:, :, :2]                      # (K, C, 2)
    K = xy.shape[0]
    if K < 4:
        return None
    perms = list(itertools.permutations(range(C)))
    if max_perms:
        perms = perms[:max_perms]
    P = np.asarray(perms)                          # (nP, C)
    nP = len(P)
    # xy[:, P, :] gathers, for every permutation, the keypoint set each camera
    # would be scored against; flatten (nP, K) into one batch of DLT systems.
    obs_perm = xy[:, P, :]                         # (K, nP, C, 2)
    obs_perm = np.transpose(obs_perm, (1, 0, 2, 3)).reshape(nP * K, C, 2)
    cams = np.broadcast_to(np.arange(C), (nP * K, C))
    X = rt.reconstruct_points(obs_perm, cams)      # (nP*K, 3)
    rp = rt.reproject_points(X)                    # (nP*K, C, 2)
    err = np.linalg.norm(rp - obs_perm, axis=-1)   # (nP*K, C)
    err = err.reshape(nP, K, C)
    score = np.nanmean(err, axis=(1, 2))           # (nP,)
    return P, score, K


# ----------------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------------
def mode_summary(ds: Dataset, args):
    """Identity-assignment reprojection error over many framesets, per recording
    and per camera, with the mean SIGNED residual (a constant signed offset is a
    calibration bias; a large random residual is a pairing error)."""
    keys = sorted(ds.framesets)
    rng = np.random.default_rng(0)
    by_rec = defaultdict(list)
    for k in keys:
        by_rec[ds.framesets[k]["recording"]].append(k)

    rows = []
    percam = defaultdict(list)
    for rec in sorted(by_rec):
        ks = by_rec[rec]
        if args.per_recording and len(ks) > args.per_recording:
            ks = list(rng.choice(ks, args.per_recording, replace=False))
        errs = []
        for key in ks:
            rt, cams, obs, seen, _ = ds.frameset_obs(key)
            res = identity_residuals(rt, obs)
            for ci, lst in res.items():
                a = np.array(lst)
                errs.append(a[:, 0])
                percam[(rec, cams[ci])].append(a)
        if not errs:
            continue
        a = np.concatenate(errs)
        rows.append((rec, ds.calib_group(rec), len(ks), len(a),
                     a.mean(), np.median(a), np.percentile(a, 95), a.max()))

    print(f"{'recording':<24}{'grp':>4}{'nFS':>5}{'nObs':>7}"
          f"{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}   (px)")
    for r in rows:
        print(f"{r[0]:<24}{r[1]:>4}{r[2]:>5}{r[3]:>7}"
              f"{r[4]:>9.2f}{r[5]:>9.2f}{r[6]:>9.2f}{r[7]:>9.2f}")
    allv = np.concatenate([np.concatenate([x[:, 0] for x in v])
                           for v in percam.values()])
    print(f"\nALL observations: n={len(allv)} mean={allv.mean():.2f} "
          f"median={np.median(allv):.2f} p95={np.percentile(allv,95):.2f} "
          f"max={allv.max():.2f} px")

    print("\nMean SIGNED residual (reproj - annotated), px, per recording x camera:")
    recs = sorted({r for r, _ in percam})
    cams = sorted({c for _, c in percam})
    print(f"{'recording':<24}" + "".join(f"{c[-3:]:>16}" for c in cams))
    for rec in recs:
        line = f"{rec:<24}"
        for c in cams:
            v = percam.get((rec, c))
            if v is None:
                line += f"{'-':>16}"
            else:
                a = np.concatenate(v)
                line += f"{a[:,1].mean():>7.1f},{a[:,2].mean():>7.1f}"
        print(line)
    return rows


def mode_permute(ds: Dataset, args):
    """Brute-force the camera assignment.  Identity must win, and win big."""
    keys = sorted(ds.framesets)
    rng = np.random.default_rng(1)
    by_rec = defaultdict(list)
    for k in keys:
        by_rec[ds.framesets[k]["recording"]].append(k)
    chosen = []
    for rec in sorted(by_rec):
        ks = by_rec[rec]
        n = min(args.per_recording, len(ks))
        chosen += list(rng.choice(ks, n, replace=False))
    if args.n_framesets:
        chosen = chosen[: args.n_framesets]

    print(f"{'frameset':<48}{'K':>4}{'identity':>10}{'best':>10}"
          f"{'2nd-best':>10}  best-permutation (camera order -> annotation slot)")
    winners = defaultdict(int)
    ident_scores, runner_scores = [], []
    for key in sorted(chosen):
        rt, cam_names, obs, seen, rec = ds.frameset_obs(key)
        out = permutation_scan(rt, obs)
        if out is None:
            continue
        P, score, K = out
        C = obs.shape[1]
        ident = tuple(range(C))
        ii = [i for i, p in enumerate(P) if tuple(p) == ident][0]
        order = np.argsort(score)
        best = order[0]
        second = order[1]
        winners[tuple(P[best])] += 1
        ident_scores.append(score[ii])
        runner_scores.append(score[second] if best == ii else score[best])
        tag = "IDENTITY" if tuple(P[best]) == ident else str(tuple(P[best]))
        print(f"{key:<48}{K:>4}{score[ii]:>10.2f}{score[best]:>10.2f}"
              f"{score[second]:>10.2f}  {tag}")
    print("\nwinning permutations (count):")
    for p, n in sorted(winners.items(), key=lambda kv: -kv[1]):
        tag = "IDENTITY" if p == tuple(range(len(p))) else str(p)
        print(f"  {tag:<40}{n}")
    i = np.array(ident_scores)
    r = np.array(runner_scores)
    print(f"\nidentity residual : mean {i.mean():.2f} px  median {np.median(i):.2f} px  max {i.max():.2f}")
    print(f"next-best residual: mean {r.mean():.2f} px  median {np.median(r):.2f} px  min {r.min():.2f}")
    print(f"separation factor : median {np.median(r/i):.1f}x   min {np.min(r/i):.1f}x")


def mode_splits(ds: Dataset, args):
    """Train/val overlap per recording, from the split annotation files."""
    root = ds.root
    tr = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    va = json.load(open(os.path.join(root, "annotations", "instances_val.json")))
    br = json.load(open(os.path.join(root, "build_report.json")))

    def by_rec(d):
        recs = defaultdict(lambda: {"images": set(), "frames": set(), "anns": set()})
        iid2rec = {}
        for im in d["images"]:
            rec = im["recording"]
            frame = im["file_name"].split("/")[2]
            recs[rec]["images"].add(im["id"])
            recs[rec]["frames"].add(frame)
            iid2rec[im["id"]] = rec
        for a in d["annotations"]:
            recs[iid2rec[a["image_id"]]]["anns"].add(a["id"])
        return recs

    T, V = by_rec(tr), by_rec(va)
    print("declared val_recordings in build_report.json:", br["val_recordings"])
    print()
    allrec = sorted(set(T) | set(V))
    print(f"{'recording':<24}{'trImg':>7}{'vaImg':>7}{'trFrm':>7}{'vaFrm':>7}"
          f"{'sharedFrames':>14}{'imgIdOverlap':>14}{'annIdOverlap':>14}"
          f"{'manifest':>10}{'declaredVal':>12}")
    for rec in allrec:
        t, v = T.get(rec, {"images": set(), "frames": set(), "anns": set()}), \
               V.get(rec, {"images": set(), "frames": set(), "anns": set()})
        sf = t["frames"] & v["frames"]
        si = t["images"] & v["images"]
        sa = t["anns"] & v["anns"]
        msplit = ds.manifest["recordings"].get(rec, {}).get("split", "?")
        print(f"{rec:<24}{len(t['images']):>7}{len(v['images']):>7}"
              f"{len(t['frames']):>7}{len(v['frames']):>7}{len(sf):>14}"
              f"{len(si):>14}{len(sa):>14}{msplit:>10}"
              f"{str(rec in br['val_recordings']):>12}")
    tot_t = sum(len(v["images"]) for v in T.values())
    tot_v = sum(len(v["images"]) for v in V.values())
    print(f"\ntotal train images {tot_t}, val images {tot_v}, "
          f"val_frac {tot_v/(tot_t+tot_v):.4f}")
    both = sorted(set(T) & set(V))
    print(f"recordings appearing in BOTH splits: {len(both)}")
    for r in both:
        print("   ", r)




# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
C_GT = "#ffffff"      # observed / annotator, per viz/core/colors.py
C_FIT = "#00ff00"     # green = fit (here: the 7-view DLT reprojection)
C_BAD = "#ff4d4d"


def _draw_skeleton(ax, xy, vis, kp_names, color, label, lw=1.0, ms=3.0):
    """Leg chains + wing veins + body axis, resolved BY NAME (never by a
    literal index) via viz/core/colors.py::leg_chains."""
    from viz.core.colors import leg_chains
    idx = {n: k for k, n in enumerate(kp_names)}
    first = True
    for _leg, chain in leg_chains(kp_names).items():
        pts = [(xy[k], vis[k]) for k in chain]
        for (p, pv), (q, qv) in zip(pts[:-1], pts[1:]):
            if pv and qv:
                ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=color, lw=lw,
                        label=label if first else None)
                first = False
    for a, b in [("Antenna_Base", "Scutellum"), ("Scutellum", "Abd_A4"),
                 ("Abd_A4", "Abd_tip"), ("EyeL", "EyeR"),
                 ("WingL_base", "WingL_V12"), ("WingL_V12", "WingL_V13"),
                 ("WingR_base", "WingR_V12"), ("WingR_V12", "WingR_V13")]:
        if a in idx and b in idx and vis[idx[a]] and vis[idx[b]]:
            ax.plot([xy[idx[a]][0], xy[idx[b]][0]],
                    [xy[idx[a]][1], xy[idx[b]][1]], "-", color=color, lw=lw,
                    label=label if first else None)
            first = False
    ax.plot(xy[vis, 0], xy[vis, 1], ".", color=color, ms=ms)
    if first:
        ax.plot([], [], "-", color=color, lw=lw, label=label)


def mode_figure(ds, args):
    """Two figures, each drawn next to its own failure signature.

    FIG 1 -- crop_transform_check.png.  The overlay the reviewer looked at
    (figures/2026-09-02-vitpose-maskoff-ab/overlay_worst.png) is a 448x448
    CROP, so a crop/letterbox offset is a live candidate for the apparent
    shift.  Column 1 draws GT in crop px on the crop the loader actually
    builds (jarvis_jax.data.transforms.crop_origin + transform_keypoints, the
    same two calls mask_channel_eval.py and detector_ckpt_overlay.py make).
    Column 2 is a CONTROL: the identical GT array on a crop whose origin was
    deliberately moved +40 px in x.  Column 3 is the untouched 1936x448 frame
    with GT in full-image px and the crop window outlined.

      EXPECTED IF THE TRANSFORM IS CORRECT: column 1 looks like column 3 --
      white skeleton on the fly, tarsal tips at the leg tips, head marker at
      the head -- and column 2 shows that same skeleton displaced bodily to
      the LEFT of the fly by 40 px, purely horizontally (y0 is always 0 here
      because the frame is exactly 448 tall, so a crop bug can only shift x).
      IF COLUMN 1 LOOKS LIKE COLUMN 2, the overlay script has a crop offset.

    FIG 2 -- worst_frameset_multiview.png.  All three rows of
    overlay_worst.png come from ONE frameset,
    2026_06_19_11_09_36/Frame_431686/fly0, which is the single worst frameset
    in the dataset for multi-view consistency.  Each camera panel shows the
    annotated GT (white) against the 7-view DLT triangulation reprojected into
    that camera (green), camera resolved BY NAME.

      EXPECTED IF THE ANNOTATIONS ARE PAIRED WITH THE RIGHT CAMERAS: green
      sits on top of white on every camera, both on the fly, gap of a few px.
      IF A CAMERA WERE MISPAIRED: that camera's green would be displaced by
      hundreds of px and would not lie on the fly at all, because a keypoint
      triangulated from a scrambled ray bundle lands nowhere near the animal.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from jarvis_jax.data.transforms import crop_origin, transform_keypoints

    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)
    KP = ds.kp_names
    ann_img = {a["id"]: ds.images[a["image_id"]] for a in ds.d["annotations"]}

    def load(fn):
        with Image.open(os.path.join(ds.root, "images", fn)) as p:
            return np.asarray(p.convert("RGB"), np.uint8)

    # ---- FIG 1 -------------------------------------------------------------
    ann_ids = ([int(x) for x in args.ann_ids.split(",")] if args.ann_ids
               else [25352, 25351, 25349])
    fig, axes = plt.subplots(len(ann_ids), 3, figsize=(16, 3.6 * len(ann_ids)),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 3.2]})
    SHIFT = 40
    for r, aid in enumerate(ann_ids):
        a = ds.anns[aid]
        im = ann_img[aid]
        rec, cam, frame = im["file_name"].split("/")
        img = load(im["file_name"])
        kp = np.asarray(a["keypoints"], np.float32).reshape(-1, 3)
        x0, y0 = crop_origin(np.asarray(a["bbox"], np.float32),
                             im["width"], im["height"], 448)
        hm, vis = transform_keypoints(kp, x0, y0, 448, 224)
        gt_crop = hm * 2.0                  # crop-448 px, exactly the npz's gt

        ax = axes[r][0]
        ax.imshow(img[y0:y0 + 448, x0:x0 + 448])
        _draw_skeleton(ax, gt_crop, vis, KP, C_GT, "GT (crop px)")
        ax.set_title("crop the loader built   x0=%d, y0=%d" % (x0, y0),
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("%s\n%s / %s\nann=%d  sex=%s"
                      % (rec, cam, frame[:-4], aid, a.get("sex")), fontsize=7)

        ax = axes[r][1]
        xs = min(x0 + SHIFT, im["width"] - 448)
        ax.imshow(img[y0:y0 + 448, xs:xs + 448])
        _draw_skeleton(ax, gt_crop, vis, KP, C_BAD,
                       "same GT, crop origin +%dpx" % SHIFT)
        ax.set_title("CONTROL: what a +%dpx crop offset looks like" % SHIFT,
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[r][2]
        ax.imshow(img)
        _draw_skeleton(ax, kp[:, :2], kp[:, 2] > 0, KP, C_GT,
                       "GT (full-image px)")
        ax.add_patch(plt.Rectangle((x0, y0), 448, 448, fill=False,
                                   ec="#ffe14d", lw=1.2))
        ax.set_title("untouched %dx%d frame (yellow = the 448 crop window)"
                     % (im["width"], im["height"]), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            for c in range(3):
                axes[r][c].legend(fontsize=6, loc="lower right")
    fig.suptitle(
        "Is the overlay's CROP transform introducing the apparent shift?\n"
        "left: crop_origin + transform_keypoints as the loader and the overlay "
        "use them | middle: deliberate +40px crop-origin error (the failure "
        "signature) | right: full frame, no crop", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = os.path.join(outdir, "crop_transform_check.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print("wrote", p1)

    # ---- FIG 2 -------------------------------------------------------------
    key = args.frameset
    rt, cam_names, obs, seen, rec = ds.frameset_obs(key)
    X = np.full((len(KP), 3), np.nan)
    for k in range(len(KP)):
        use = np.where(obs[k, :, 2] > 0)[0]
        if len(use) >= 3:
            X[k] = rt.reconstruct_point(obs[k, :, :2], cams_to_use=list(use))
    rp = rt.reproject_points(X)                       # (K, C, 2)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
    for ci, cname in enumerate(cam_names):
        ax = axes[ci // 4][ci % 4]
        aid = seen[cname]
        im = ann_img[aid]
        a = ds.anns[aid]
        img = load(im["file_name"])
        x0, y0 = crop_origin(np.asarray(a["bbox"], np.float32),
                             im["width"], im["height"], 448)
        ax.imshow(img[y0:y0 + 448, x0:x0 + 448])
        v = obs[:, ci, 2] > 0
        _draw_skeleton(ax, obs[:, ci, :2] - np.array([x0, y0]), v, KP, C_GT,
                       "annotated GT")
        _draw_skeleton(ax, rp[:, ci, :] - np.array([x0, y0]),
                       v & np.isfinite(rp[:, ci, 0]), KP, C_FIT,
                       "7-view DLT, reprojected")
        e = np.linalg.norm(rp[v, ci, :] - obs[v, ci, :2], axis=-1)
        ax.set_title("%s   ann=%d\nreproj %.2f px mean, %.2f px max"
                     % (cname, aid, e.mean(), e.max()), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if ci == 0:
            ax.legend(fontsize=7, loc="lower right")
    axes[1][3].axis("off")
    m = obs[:, :, 2] > 0
    tot = np.linalg.norm(rp - obs[:, :, :2], axis=-1)[m]
    axes[1][3].text(
        0.02, 0.5,
        "%s\n\ncalibration group %s\ncameras resolved BY NAME from\n"
        "each annotation's image path\n\nmean reprojection %.2f px\n"
        "max %.2f px\n\ndataset median over 3717\nframesets: 0.42 px\n"
        "this frameset ranks #1 WORST\nof 3717\n\nidentity camera assignment\n"
        "is rank 1 of 5040 permutations\nhere (next best 11.82 px)"
        % (key, ds.calib_group(rec), tot.mean(), tot.max()),
        fontsize=9, va="center", family="monospace")
    fig.suptitle(
        "%s -- annotated GT (white) vs 7-view DLT triangulation reprojected "
        "(green)\nthe frameset behind all three rows of overlay_worst.png; a "
        "mispaired camera would put green hundreds of px off the fly" % key,
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p2 = os.path.join(outdir, "worst_frameset_multiview.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print("wrote", p2)

def mode_shadow(ds, args):
    """Why does GT look like it is FLOATING on the dark arena band?

    The panel that prompted the question (overlay_worst.png row 1, ann 25352,
    Cam2012855) shows white GT lines running off the fly into the black band
    at the bottom of the frame.  Two explanations:

      (a) BENIGN -- the fly is standing on / over the band, its legs really are
          extended there, and they are simply not visible at this exposure.
      (b) REAL DEFECT -- the annotation places tarsi where no fly part is.

    These are distinguishable by lifting the shadows.  Left column: the crop as
    the overlay renders it, GT in white.  Right column: the SAME crop with a
    strong gamma lift applied to the dark end only, same GT.

      EXPECTED UNDER (a): leg and tarsus structure appears under the white
      lines once the shadows are lifted, and the tarsal tips land on real leg
      tips.  EXPECTED UNDER (b): the lifted image shows bare arena under the
      white lines and the tips sit on nothing.

    Also printed: the fraction of each annotation's keypoints whose pixel is
    darker than the frame's 10th percentile, i.e. how much of the skeleton is
    in the unreadable band -- the mechanism behind the 'floating' impression.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from jarvis_jax.data.transforms import crop_origin, transform_keypoints

    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)
    KP = ds.kp_names
    ann_img = {a["id"]: ds.images[a["image_id"]] for a in ds.d["annotations"]}
    ann_ids = ([int(x) for x in args.ann_ids.split(",")] if args.ann_ids
               else [25352, 25351, 25349])

    fig, axes = plt.subplots(len(ann_ids), 2, figsize=(11, 5.4 * len(ann_ids)),
                             squeeze=False)
    for r, aid in enumerate(ann_ids):
        a = ds.anns[aid]
        im = ann_img[aid]
        rec, cam, frame = im["file_name"].split("/")
        with Image.open(os.path.join(ds.root, "images", im["file_name"])) as p:
            img = np.asarray(p.convert("RGB"), np.uint8)
        kp = np.asarray(a["keypoints"], np.float32).reshape(-1, 3)
        x0, y0 = crop_origin(np.asarray(a["bbox"], np.float32),
                             im["width"], im["height"], 448)
        hm, vis = transform_keypoints(kp, x0, y0, 448, 224)
        gt = hm * 2.0
        crop = img[y0:y0 + 448, x0:x0 + 448]
        lifted = (255.0 * (crop / 255.0) ** 0.30).astype(np.uint8)

        # how much of the skeleton sits in the unreadable dark band
        g = img.mean(-1)
        thr = np.percentile(g, 10)
        px = kp[vis, :2].astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, im["width"] - 1)
        px[:, 1] = np.clip(px[:, 1], 0, im["height"] - 1)
        dark = float((g[px[:, 1], px[:, 0]] < thr).mean())

        for c, (arr, ttl) in enumerate(
                [(crop, "crop as the overlay renders it"),
                 (lifted, "same crop, shadows lifted (gamma 0.30)")]):
            ax = axes[r][c]
            ax.imshow(arr)
            _draw_skeleton(ax, gt, vis, KP, C_GT, "GT")
            ax.set_title(ttl, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r][0].set_ylabel(
            "%s\n%s / %s\nann=%d  %d/%d kp visible-flagged\n%.0f%% of kp in "
            "the darkest 10%% of the frame"
            % (rec, cam, frame[:-4], aid, int(vis.sum()), len(KP), 100 * dark),
            fontsize=7)
        if r == 0:
            axes[r][0].legend(fontsize=7, loc="lower right")
        print("ann %d  %s  %d/%d kp flagged visible, %.0f%% of them in the "
              "darkest 10%% of the frame" % (aid, cam, int(vis.sum()),
                                             len(KP), 100 * dark))
    fig.suptitle(
        "Is the GT floating, or are the legs just invisible against the dark "
        "arena band?\nlifting the shadows shows whether there is fly under "
        "the white lines", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(outdir, "dark_band_shadow_lift.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print("wrote", p)


def mode_dupes(ds, args):
    """Are any VAL images pixel-identical to TRAIN images?

    build_report.json's split audit reports ``cross_recording_leaks: 0``, but
    it keys on the recording NAME.  Several of this dataset's "recordings" are
    the SAME physical video exported twice under different session names --
    once per fly, from different upstream corpora (general_model/courtship_*_
    male vs _female, red_data_unified_V3 vs general_model/courtship_V2).  The
    name-keyed audit cannot see that, so this checks the pixels.

    Two annotations are compared as follows: candidate pairs are val/train
    images sharing a (camera, frame-number) key across DIFFERENT recording
    names; each candidate is then confirmed by md5 of the symlink-resolved
    file.  For confirmed pairs the labels are compared too, because "same
    pixels, other fly labelled" and "same pixels, same 50 keypoints" are very
    different problems.
    """
    import hashlib
    from collections import defaultdict

    root = ds.root
    tr = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    va = json.load(open(os.path.join(root, "annotations", "instances_val.json")))

    def by_file(d):
        im = {i["id"]: i["file_name"] for i in d["images"]}
        out = defaultdict(list)
        for a in d["annotations"]:
            out[im[a["image_id"]]].append(a)
        return out

    Tb, Vb = by_file(tr), by_file(va)
    tkey = defaultdict(list)
    for fn in Tb:
        _r, cam, fr = fn.split("/")
        tkey[(cam, fr)].append(fn)

    pairs = []
    for vfn in Vb:
        _r, cam, fr = vfn.split("/")
        for tfn in tkey.get((cam, fr), []):
            pairs.append((vfn, tfn))
    print(f"candidate val/train image pairs sharing (camera, frame): {len(pairs)}")

    cache = {}

    def h(fn):
        p = os.path.realpath(os.path.join(root, "images", fn))
        if p not in cache:
            m = hashlib.md5()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    m.update(b)
            cache[p] = m.hexdigest()
        return cache[p]

    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        i = max(0, x2 - x1) * max(0, y2 - y1)
        u = aw * ah + bw * bh - i
        return i / u if u > 0 else 0.0

    groups = defaultdict(lambda: {"img": 0, "ann": 0, "exact": 0, "near": 0})
    leaked_img, leaked_ann, exact_ann = set(), set(), set()
    for vfn, tfn in pairs:
        if h(vfn) != h(tfn):
            continue
        g = groups[(vfn.split("/")[0], tfn.split("/")[0])]
        g["img"] += 1
        leaked_img.add(vfn)
        for va_ in Vb[vfn]:
            g["ann"] += 1
            leaked_ann.add(va_["id"])
            for ta in Tb[tfn]:
                same = np.array_equal(np.asarray(va_["keypoints"]),
                                      np.asarray(ta["keypoints"]))
                if same:
                    g["exact"] += 1
                    exact_ann.add(va_["id"])
                elif iou(va_["bbox"], ta["bbox"]) > 0.9:
                    g["near"] += 1

    print(f"\n{'VAL recording':<24}{'TRAIN recording':<24}{'images':>8}"
          f"{'valAnns':>9}{'exactKP':>9}{'bboxIoU>0.9':>13}")
    for (v, t), g in sorted(groups.items()):
        print(f"{v:<24}{t:<24}{g['img']:>8}{g['ann']:>9}{g['exact']:>9}"
              f"{g['near']:>13}")
    nimg = len(set(Vb))
    nann = sum(len(x) for x in Vb.values())
    print(f"\nVAL images pixel-identical to a TRAIN image: {len(leaked_img)} / "
          f"{nimg} ({100 * len(leaked_img) / nimg:.1f}%)")
    print(f"VAL annotations on such an image:            {len(leaked_ann)} / "
          f"{nann} ({100 * len(leaked_ann) / nann:.1f}%)")
    print(f"VAL annotations whose 50 keypoints appear VERBATIM in train: "
          f"{len(exact_ann)} ({100 * len(exact_ann) / nann:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--ann", default="instances.json")
    ap.add_argument("--mode", default="summary",
                    choices=["summary", "permute", "splits", "figure",
                             "shadow", "dupes"])
    ap.add_argument("--out-dir", default="figures/2026-09-02-gt-camera-check")
    ap.add_argument("--ann-ids", default="",
                    help="comma-separated annotation ids for figure 1 "
                         "(default: the three rows of overlay_worst.png)")
    ap.add_argument("--frameset",
                    default="2026_06_19_11_09_36/Frame_431686/fly0")
    ap.add_argument("--per-recording", type=int, default=5)
    ap.add_argument("--n-framesets", type=int, default=0)
    args = ap.parse_args()
    ds = Dataset(args.root, args.ann)
    {"summary": mode_summary, "permute": mode_permute,
     "splits": mode_splits, "figure": mode_figure,
     "shadow": mode_shadow, "dupes": mode_dupes}[args.mode](ds, args)


if __name__ == "__main__":
    main()
