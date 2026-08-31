"""Is the wing V12 error made in 2D, or in the lifting to 3D?

CONTEXT. Distance from the wing MARGIN, where the veins terminate (blade chord
is 0.1151 model units):

                  XML site   fitted c0   fitted c10
  WingL_V12         0.0033      0.0336       0.0205
  WingR_V12         0.0033      0.0350       0.0190
  WingL_V13         0.0015      0.0071       0.0016
  WingR_V13         0.0015      0.0008       0.0029

V13 lands on the margin where it belongs; V12 is dragged ~30% of the blade
chord INBOARD, on both wings. This script asks where that error is introduced.

THE TWO HYPOTHESES, and what separates them:
  2D    the detector puts V12 inboard of the wing outline in the images. Then
        `dist_to_outline` is large in 2D too, and the 3D is faithfully
        reproducing a bad 2D input.
  LIFT  the detector puts V12 ON the outline in each view (dist_to_outline ~ 0)
        but triangulation lands it inboard. Then the 3D point CANNOT reproject
        onto those 2D detections -- reprojection error will be large, and
        largest in the views that disagree.

The discriminator is `reproj_err` measured against the SAME 2D points that were
triangulated. Small 2D outline distance + large reprojection error = the lifting
is at fault. Small outline distance + SMALL reprojection error would mean the 3D
faithfully explains the 2D, and the inboard reading comes from the model's wing
shape rather than from tracking at all -- a third possibility this must be able
to report rather than assume away.

`dist_to_outline` is signed: POSITIVE = inside the SAM mask (how far in from the
silhouette edge, in px), negative = outside it. A margin landmark should sit
within a few px of zero. Values are also given normalised by sqrt(mask area) so
cameras at different apparent fly sizes can be compared.

Per-camera confidence and the count of views above the DLT threshold are
reported alongside, because a landmark triangulated from few or low-confidence
views is exactly where a medial-axis bias would be expected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "third_party/jarvis_jax")):
    sys.path.insert(0, _p)

WING = ["WingL_base", "WingL_V12", "WingL_V13", "WingR_base", "WingR_V12", "WingR_V13"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="Session0")
    ap.add_argument("--recording", default="2025_10_20_13_20_04")
    ap.add_argument("--bout", type=int, default=1)
    ap.add_argument("--fly", type=int, default=0)
    ap.add_argument("--nt", type=int, default=200)
    ap.add_argument("--conf-thresh", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2
    import h5py
    from viz.core import reproject
    from jarvis_jax.tracking.bout_masks import load_bout_masks

    PROC = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"
    VIDEO = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"
    rec = os.path.join(PROC, args.session, args.recording)
    fly_dir = os.path.join(rec, "pose", "bouts", f"bout_{args.bout:05d}", f"fly{args.fly}")
    calib = os.path.join(VIDEO, args.session, args.recording, "calibration")

    cam_mats, cam_names = reproject.camera_matrices(calib)
    cam_names = list(cam_names)
    print(f"{len(cam_names)} cameras: {cam_names}")

    z = np.load(os.path.join(fly_dir, "kp2d.npz"), allow_pickle=True)
    kp2d, conf = np.asarray(z["kp2d"], float), np.asarray(z["conf"], float)
    kp3d = np.asarray(np.load(os.path.join(fly_dir, "kp3d.npz"))["kp3d"], float)
    with h5py.File(os.path.join(fly_dir, "stac_ik.h5"), "r") as f:
        kp_names = [x.decode() for x in f["kp_names"][()]]
    print(f"kp2d {kp2d.shape}, kp3d {kp3d.shape}")

    md = load_bout_masks(os.path.join(rec, "sam3_masks", f"bout_{args.bout:05d}",
                                      "sam3_masks.npz"),
                         args.fly, expected_cameras=cam_names)
    masks, mvalid = np.asarray(md["masks"]), np.asarray(md["valid"])
    print(f"masks {masks.shape}")

    T = min(args.nt, len(kp2d), len(kp3d), len(masks))
    C = len(cam_names)
    idx = {n: kp_names.index(n) for n in WING if n in kp_names}

    # reproject the triangulated 3D back into every camera
    reproj = np.stack([reproject.project(cam_mats[c], kp3d[:T].reshape(-1, 3)).reshape(T, -1, 2)
                       for c in range(C)], axis=1)          # (T,C,K,2)

    acc = {n: {"dist": [[] for _ in range(C)], "err": [[] for _ in range(C)],
               "conf": [[] for _ in range(C)], "nview": []} for n in idx}

    for t in range(T):
        # signed distance to the silhouette edge, per camera
        sdt = []
        for c in range(C):
            if not mvalid[t, c]:
                sdt.append(None)
                continue
            mk = masks[t, c].astype(np.uint8)
            din = cv2.distanceTransform(mk, cv2.DIST_L2, 3)
            dout = cv2.distanceTransform(1 - mk, cv2.DIST_L2, 3)
            sdt.append((din - dout, float(mk.sum())))
        for n, k in idx.items():
            nv = 0
            for c in range(C):
                cf = conf[t, c, k]
                acc[n]["conf"][c].append(cf)
                if cf >= args.conf_thresh:
                    nv += 1
                if sdt[c] is None:
                    continue
                sd, area = sdt[c]
                x, y = kp2d[t, c, k]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < sd.shape[0] and 0 <= xi < sd.shape[1]:
                    acc[n]["dist"][c].append(sd[yi, xi] / max(np.sqrt(area), 1.0))
                    acc[n]["err"][c].append(float(np.linalg.norm(reproj[t, c, k] - kp2d[t, c, k])))
            acc[n]["nview"].append(nv)

    print(f"\n{'landmark':12}{'views>conf':>11}{'medConf':>9}"
          f"{'dist_to_outline (frac of fly size)':>36}{'reproj err px':>16}")
    print(f"{'':12}{'':11}{'':9}{'median':>12}{'p90':>10}{'>10% in':>13}{'median':>9}{'p90':>7}")
    rep = {}
    for n in WING:
        if n not in acc:
            continue
        a = acc[n]
        dist = np.concatenate([np.asarray(d) for d in a["dist"] if len(d)]) if any(
            len(d) for d in a["dist"]) else np.array([np.nan])
        err = np.concatenate([np.asarray(e) for e in a["err"] if len(e)]) if any(
            len(e) for e in a["err"]) else np.array([np.nan])
        cf = np.nanmedian(np.concatenate([np.asarray(c) for c in a["conf"]]))
        deep = 100 * float(np.mean(dist > 0.10))
        rep[n] = dict(nview=float(np.mean(a["nview"])), conf=float(cf),
                      dist_med=float(np.nanmedian(dist)), dist_p90=float(np.nanpercentile(dist, 90)),
                      pct_deep=deep, err_med=float(np.nanmedian(err)),
                      err_p90=float(np.nanpercentile(err, 90)))
        r = rep[n]
        print(f"{n:12}{r['nview']:11.1f}{r['conf']:9.2f}"
              f"{r['dist_med']:12.3f}{r['dist_p90']:10.3f}{deep:12.0f}%"
              f"{r['err_med']:9.2f}{r['err_p90']:7.2f}")

    print(f"\nper-camera, V12 vs V13  (dist>0 = inboard of the silhouette edge)")
    print(f"{'camera':14}" + "".join(f"{s:>13}" for s in
                                     ("L_V12 dist", "L_V12 err", "L_V13 dist", "L_V13 err")))
    for c in range(C):
        vals = []
        for n in ("WingL_V12", "WingL_V13"):
            d = acc[n]["dist"][c]
            e = acc[n]["err"][c]
            vals += [np.nanmedian(d) if len(d) else np.nan,
                     np.nanmedian(e) if len(e) else np.nan]
        print(f"{cam_names[c]:14}" + "".join(f"{v:13.3f}" for v in vals))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "wing_kp_2d_check.json"), "w") as f:
            json.dump(dict(bout=args.bout, fly=args.fly, T=T, cameras=cam_names,
                           conf_thresh=args.conf_thresh, per_landmark=rep), f, indent=2)
        print(f"\nwrote {args.out}/wing_kp_2d_check.json")


if __name__ == "__main__":
    main()
