"""Turn on consensus outlier rejection in triangulation and re-measure the wing.

WHY. `triangulate_keypoints(..., reproj_resid_px=...)` already implements
consensus-based outlier-view rejection, but nothing sets it: it is absent from
every config, so `cfg.detector.get("reproj_resid_px", None)` is None and the
gate is OFF. `view_conf_thresh: 0.6` IS set, but confidence cannot see this
failure -- the bad view is confidently wrong.

Measured on bout_00001 fly0 with the gate off (scripts/analysis/wing_kp_2d_check.py):
  * the 2D detections sit ON the wing outline: WingL_V12 is 0.020 of fly size
    (~2.5 px on a ~124 px fly) from the silhouette edge, conf 0.96, 7/7 views;
  * yet the triangulated 3D cannot reproject onto them -- WingL_V12 reprojects
    10.93 px median / 46.78 px p90, against 4.64 px for WingL_V13;
  * and one camera dominates: Cam2012861 puts WingL_V12 46.5 px (38% of fly
    size) from its own 2D while its WingL_V13 is 4.2 px.
That is a lifting failure, not a detector failure, and it is what drags V12 to
~30% of the blade chord inboard of the wing margin where the vein terminates.

EXPECTATION, read the table against this:
  * WingL_V12 reprojection error falls toward WingL_V13's ~4.6 px. That is the
    whole claim. If it does not fall, the disagreement is not a droppable
    single-view outlier and the consensus search is refusing it -- check
    `n_dropped`, which says whether the gate fired at all.
  * the landmark TRIANGLE of the measured 3D (frame-independent: angle at the
    wing base, area) moves toward the model's 8.09 deg / 0.00396. Wing L starts
    at 6.56 deg, wing R at 8.07 deg -- so wing L has room to improve and wing R
    is already near the model and must NOT be degraded.
  * `n_dropped` stays small. The gate is meant to remove rare confidently-wrong
    views (~1.8% of view-points at 12 px, per the benchmark notes); dropping a
    large fraction would mean the threshold is eating good views, and the
    3-view floor would then start turning points into NaN -- watch `nan_frac`.

A threshold too tight also fails silently: MIN_CONSENSUS_VIEWS=3 means a point
that cannot find 3 mutually consistent views keeps ALL its views unchanged, so
an over-tight setting degrades to the current behaviour rather than erroring.
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


def triangle(P3):
    b, v12, v13 = P3
    a1, a2 = v12 - b, v13 - b
    n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
    if not (n1 > 0 and n2 > 0):
        return np.nan, np.nan
    return (float(np.degrees(np.arccos(np.clip(a1 @ a2 / (n1 * n2), -1, 1)))),
            float(0.5 * np.linalg.norm(np.cross(a1, a2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="Session0")
    ap.add_argument("--recording", default="2025_10_20_13_20_04")
    ap.add_argument("--bout", type=int, default=1)
    ap.add_argument("--fly", type=int, default=0)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--thresholds", default="none,20,12,8")
    ap.add_argument("--conf-thresh", type=float, default=0.3)
    ap.add_argument("--view-conf-thresh", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import h5py
    from viz.core import reproject
    from jarvis_jax.tracking.triangulate import triangulate_keypoints

    PROC = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"
    VIDEO = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"
    rec = os.path.join(PROC, args.session, args.recording)
    fly_dir = os.path.join(rec, "pose", "bouts", f"bout_{args.bout:05d}", f"fly{args.fly}")
    cam_mats, cam_names = reproject.camera_matrices(
        os.path.join(VIDEO, args.session, args.recording, "calibration"))
    cam_names = list(cam_names)

    z = np.load(os.path.join(fly_dir, "kp2d.npz"), allow_pickle=True)
    kp2d = np.asarray(z["kp2d"], np.float32)
    conf = np.asarray(z["conf"], np.float32)
    with h5py.File(os.path.join(fly_dir, "stac_ik.h5"), "r") as f:
        kp_names = [x.decode() for x in f["kp_names"][()]]
    T = min(args.nt, len(kp2d))
    kp2d, conf = kp2d[:T], conf[:T]
    # triangulate_keypoints requires FINITE pixels even where conf is low
    kp2d = np.nan_to_num(kp2d, nan=0.0, posinf=0.0, neginf=0.0)
    idx = {n: kp_names.index(n) for n in WING if n in kp_names}
    print(f"{T} frames, {len(cam_names)} cameras: {cam_names}")

    def reproj_err(X, k):
        """Median px error of point k against every CONFIDENT view (fixed set,
        so settings are compared on the same views rather than on whatever
        each one chose to keep)."""
        ph = np.concatenate([X[:, k, :], np.ones((len(X), 1))], 1)          # (T,4)
        pr = np.einsum("tj,cjk->tck", ph, np.asarray(cam_mats, np.float64))
        uv = pr[..., :2] / pr[..., 2:3]                                     # (T,C,2)
        e = np.linalg.norm(uv - kp2d[:, :, k, :], axis=-1)                  # (T,C)
        m = (conf[:, :, k] >= args.conf_thresh) & np.isfinite(e) & np.isfinite(X[:, k, 0])[:, None]
        return (float(np.median(e[m])) if m.any() else np.nan,
                float(np.percentile(e[m], 90)) if m.any() else np.nan)

    rows, store = [], {}
    for tok in args.thresholds.split(","):
        thr = None if tok.strip().lower() == "none" else float(tok)
        kp3d, conf3d = triangulate_keypoints(
            kp2d, conf, cam_mats, conf_thresh=args.conf_thresh,
            view_conf_thresh=args.view_conf_thresh, reproj_resid_px=thr)
        kp3d = np.asarray(kp3d)
        store[tok] = kp3d
        rec_row = {"thr": tok,
                   "nan_frac": float(np.mean(~np.isfinite(kp3d[:, list(idx.values()), 0])))}
        for n, k in idx.items():
            med, p90 = reproj_err(kp3d, k)
            rec_row[f"{n}_err"] = med
            rec_row[f"{n}_p90"] = p90
        for side in ("L", "R"):
            tri = [triangle(np.stack([kp3d[t, idx[f"Wing{side}_base"]],
                                      kp3d[t, idx[f"Wing{side}_V12"]],
                                      kp3d[t, idx[f"Wing{side}_V13"]]]))
                   for t in range(T) if np.isfinite(kp3d[t, idx[f"Wing{side}_V12"], 0])]
            tri = np.array(tri, float)
            rec_row[f"angle_{side}"] = float(np.nanmedian(tri[:, 0])) if len(tri) else np.nan
            rec_row[f"area_{side}"] = float(np.nanmedian(tri[:, 1])) if len(tri) else np.nan
        if thr is not None:
            base = store[args.thresholds.split(",")[0]]
            d = np.linalg.norm(kp3d[:, list(idx.values())] - base[:, list(idx.values())], axis=-1)
            rec_row["moved_median"] = float(np.nanmedian(d))
        rows.append(rec_row)
        print(f"  reproj_resid_px={tok}: done")

    print(f"\n{'reproj_resid_px':>16}{'L_V12 err':>11}{'L_V13 err':>11}{'R_V12 err':>11}"
          f"{'R_V13 err':>11}{'angleL':>9}{'angleR':>9}{'areaL':>10}{'nan%':>7}")
    print(f"{'MODEL anatomy':>16}{'-':>11}{'-':>11}{'-':>11}{'-':>11}"
          f"{8.09:>9.2f}{8.09:>9.2f}{0.00396:>10.5f}{'-':>7}")
    for r in rows:
        print(f"{r['thr']:>16}{r['WingL_V12_err']:>11.2f}{r['WingL_V13_err']:>11.2f}"
              f"{r['WingR_V12_err']:>11.2f}{r['WingR_V13_err']:>11.2f}"
              f"{r['angle_L']:>9.2f}{r['angle_R']:>9.2f}{r['area_L']:>10.5f}"
              f"{100*r['nan_frac']:>7.1f}")

    print(f"\n{'reproj_resid_px':>16}{'L_V12 p90':>11}{'L_V13 p90':>11}{'3D moved (mm)':>16}")
    for r in rows:
        print(f"{r['thr']:>16}{r['WingL_V12_p90']:>11.2f}{r['WingL_V13_p90']:>11.2f}"
              f"{r.get('moved_median', float('nan')):>16.4f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "robust_triangulation.json"), "w") as f:
            json.dump(dict(bout=args.bout, fly=args.fly, T=T, cameras=cam_names, rows=rows), f,
                      indent=2)
        print(f"\nwrote {args.out}/robust_triangulation.json")


if __name__ == "__main__":
    main()
