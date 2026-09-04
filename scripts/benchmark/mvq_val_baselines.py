#!/usr/bin/env python
"""ViTPose+DLT baseline on the mvq v12 val split, for a like-for-like MPJPE
comparison against the mvq lifter (see docs/benchmark/2026-09-mvq/p2-t1-notes.md).

EXPECTATION (write before running): this is the "current pipeline" number mvq
has to beat. It runs the SAME v5vf_maskoff ViTPose checkpoint the free-running
pipeline uses (`configs/detector/vitpose_v3.yaml`) on the SAME v12 val
framesets `V12WindowDataset` gives mvq, decodes 2D heatmap peaks, and
triangulates with the pipeline's own robust DLT gates (conf_thresh,
view_conf_thresh, reproj_resid_px). Ground truth is the dataset's own
DLT-of-human-2D-labels 3D (`kp3d_local`) -- 0 error by construction, so it is
not itself a baseline row; this script's number is what a competent
non-learned 3D lift gets INSTEAD of the human labels, from the SAME detector
2D. A `nan` cohort is a genuinely empty cohort (e.g. no group_C val
framesets), not a bug.

No 4th (SAM-mask) channel is available at this granularity, so it is passed
as an all-zero channel -- EXACT for v5vf_maskoff, whose patch_embed kernel
weights for that channel are all identically 0.0 (verified in
tracking/predict_2d.py's `zero_mask_channel` docstring), so this is not an
approximation for this checkpoint.

Geometry: V12WindowDataset's M/t_local are CROP-LOCAL (uv_crop = M@X_local +
t_local, X_local = world - center3D), so both the ViTPose 2D and the
triangulated 3D stay in the same local frame as `kp3d_local` -- no round trip
through full-frame pixels is needed. `cam_mats_local[c]` is built to the exact
(4,3) = P.T convention `triangulate_dlt_batched` expects (row 2 = [0,0,0,1]):
cam_mats_local[:, :3, :2] = M.transpose(0, 2, 1); cam_mats_local[:, 3, :2] =
t_local; cam_mats_local[:, 3, 2] = 1.
"""
import argparse, json, os, sys, time

import numpy as np
import jax, jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax")); sys.path.insert(0, ROOT)
from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.convert.build_checkpoint import load_vitpose
from jarvis_jax.data.device import normalize_image
from jarvis_jax.data.v12_windows import V12WindowDataset
from jarvis_jax.tracking.predict_2d import peaks_and_conf
from jarvis_jax.tracking.triangulate import triangulate_keypoints
from jarvis_jax.train.train_mvq import MM_PER_UNIT

DEFAULT_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902"
DEFAULT_CKPT = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v5vf_maskoff/final"


def cam_mats_local(M, t_local):
    """M (C,2,3), t_local (C,2) -> cam_mats_local (C,4,3), the (4,3)=P.T affine
    convention triangulate_dlt_batched expects (row 2 == [0,0,0,1])."""
    C = M.shape[0]
    cm = np.zeros((C, 4, 3), np.float64)
    cm[:, :3, :2] = np.transpose(M, (0, 2, 1))
    cm[:, 3, :2] = t_local
    cm[:, 3, 2] = 1.0
    return cm


def run(args):
    ds = V12WindowDataset(args.root, args.split, T=1, train=False)
    n = len(ds) if args.n is None else min(args.n, len(ds))
    vit = load_vitpose(args.ckpt, ViTPoseConfig(num_keypoints=50))

    rows = []  # (mpjpe_units, n_valid, female, two_fly, group)
    t0 = time.time()
    for i in range(n):
        s = ds[i]
        crops = s["crops"][0]                                    # (C,448,448,3) u8
        cam_valid = s["cam_valid"][0]                            # (C,)
        crops4 = np.concatenate([crops, np.zeros(crops.shape[:-1] + (1,), np.uint8)], axis=-1)
        hm = vit(normalize_image(jnp.asarray(crops4)), use_running_average=True)
        kp_crop, conf = peaks_and_conf(hm, decode_sharpen=args.decode_sharpen)   # (C,K,2),(C,K)
        kp_crop = np.asarray(kp_crop, np.float32)
        conf = np.where(cam_valid[:, None], np.asarray(conf), 0.0).astype(np.float32)

        cm = cam_mats_local(s["M"], s["t_local"][0])
        kp3d_pred, _ = triangulate_keypoints(kp_crop[None], conf[None], cm,
                                             conf_thresh=args.conf_thresh,
                                             view_conf_thresh=args.view_conf_thresh,
                                             reproj_resid_px=args.reproj_resid_px)
        gt = s["kp3d_local"][0, 0]; has = s["has3d"][0, 0]
        valid = has & np.isfinite(kp3d_pred[0]).all(-1)
        err = np.linalg.norm(np.where(valid[:, None], kp3d_pred[0], 0.0) - gt, axis=-1)
        mpjpe_i = float(err[valid].mean()) if valid.any() else float("nan")
        rows.append((mpjpe_i, int(valid.sum()), bool(ds.is_female(i)), ds.n_flies(i) > 1, ds.calib_group(i)))
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"[{i+1}/{n}] ({time.time()-t0:.0f}s)", flush=True)

    mp = np.array([r[0] for r in rows]); nv = np.array([r[1] for r in rows])
    ok = np.isfinite(mp) & (nv > 0)

    def cohort_mpjpe(mask):
        sel = mask & ok
        return float(np.average(mp[sel], weights=nv[sel])) if sel.any() else float("nan")

    female = np.array([r[2] for r in rows]); two_fly = np.array([r[3] for r in rows])
    groups = sorted({r[4] for r in rows})
    result = {
        "n_framesets": n, "ckpt": args.ckpt, "root": args.root, "split": args.split,
        "decode_sharpen": args.decode_sharpen, "conf_thresh": args.conf_thresh,
        "view_conf_thresh": args.view_conf_thresh, "reproj_resid_px": args.reproj_resid_px,
        "overall_units": cohort_mpjpe(np.ones(len(rows), bool)),
        "female_units": cohort_mpjpe(female), "two_fly_units": cohort_mpjpe(two_fly),
    }
    for g in groups:
        result[f"group_{g}_units"] = cohort_mpjpe(np.array([r[4] == g for r in rows]))
    for k in list(result):
        if k.endswith("_units"):
            result[k.replace("_units", "_mm")] = result[k] * MM_PER_UNIT if np.isfinite(result[k]) else float("nan")

    print(json.dumps(result, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print("wrote", args.out)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=None, help="limit to first N val framesets")
    ap.add_argument("--decode_sharpen", type=float, default=3.0)   # matches configs/detector/vitpose_v3.yaml
    ap.add_argument("--conf_thresh", type=float, default=0.3)
    ap.add_argument("--view_conf_thresh", type=float, default=0.6)
    ap.add_argument("--reproj_resid_px", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
