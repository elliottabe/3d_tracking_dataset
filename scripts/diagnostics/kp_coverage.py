"""Phase-1 diagnosis (decisive): do predicted keypoints land INSIDE the fly's
SAM mask? Objective accuracy proxy, independent of model confidence.

Reprojects each fly's 3D keypoints into every camera using the pipeline's own
ReprojectionTool (DLT, project calibration), then checks each 2D keypoint against
that fly/cam/frame's SAM mask (with a neighborhood slack mirroring the inference
dilation). Reports per-fly coverage and, critically, the cross-tab of
coverage vs. confidence (to catch "confidently wrong").

Run with the jarvis env:
  /gscratch/portia/eabe/miniconda3/envs/jarvis/bin/python kp_coverage.py \
      --project unified_V2_masked --pred_dir <Predictions_3D_*> [--stride 3] [--slack 10]
"""
import argparse
import glob
import os

import numpy as np
import torch

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.reprojection import get_repro_tool


def load_fly_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=2)
    if data.ndim == 1:
        data = data[None, :]
    data = data[:, 1:]
    return data.reshape(data.shape[0], -1, 4)  # (F, nkp, 4)


def unpack_mask(packed_row, W):
    return np.unpackbits(packed_row, axis=1, bitorder="big")[:, :W].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--stride", type=int, default=3, help="frame subsample")
    ap.add_argument("--slack", type=int, default=10,
                    help="px neighborhood for 'inside mask' (inference dilates ~10)")
    ap.add_argument("--conf_thresh", type=float, default=0.3)
    ap.add_argument("--female_slot", type=int, default=0,
                    help="fly slot that is the FEMALE (red=fly0 per viz/user)")
    ap.add_argument("--per_bout", action="store_true",
                    help="also print per-bout per-slot coverage (identity consistency)")
    args = ap.parse_args()

    pm = ProjectManager()
    assert pm.load(args.project), f"could not load project {args.project}"
    cfg = pm.get_cfg()
    kp_names = list(cfg.KEYPOINT_NAMES)
    nkp = len(kp_names)
    repro = get_repro_tool(cfg, None)
    dev = repro.cameraMatrices.device
    print(f"Reproj tool: {len(repro.cameras)} cams, device={dev}")

    def _ready(b):
        for fn in ("fly0.csv", "fly1.csv", "sam3_masks.npz"):
            p = os.path.join(b, fn)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                return False
        return True
    bout_dirs = [b for b in sorted(glob.glob(os.path.join(args.pred_dir, "bout_*")))
                 if _ready(b)]
    print(f"{len(bout_dirs)} bouts, stride={args.stride}, slack={args.slack}px\n")

    r = args.slack
    # per-fly accumulators: inside/total counts, and per-keypoint, and
    # confidence of keypoints that fall OUTSIDE the mask
    cov = {s: dict(inside=0, total=0,
                   kp_inside=np.zeros(nkp), kp_total=np.zeros(nkp),
                   out_conf=[], in_conf=[], blen=[]) for s in (0, 1)}
    per_bout = []  # list of (bout_name, {slot: (inside,total)})

    for bd in bout_dirs:
        bout_local = {0: [0, 0], 1: [0, 0]}
        masks = np.load(os.path.join(bd, "sam3_masks.npz"))
        packed, valid = masks["packed"], masks["valid"]
        H, W = int(masks["shape"][0]), int(masks["shape"][1])
        A, C, Fm = packed.shape[0], packed.shape[1], packed.shape[2]

        for s in range(min(2, A)):
            arr = load_fly_csv(os.path.join(bd, f"fly{s}.csv"))  # (F,nkp,4)
            F = min(arr.shape[0], Fm)
            # body length to ID female later
            blen = np.linalg.norm(arr[:, 0, :3] - arr[:, 5, :3], axis=1)
            cov[s]["blen"].append(blen[np.isfinite(blen)])

            for f in range(0, F, args.stride):
                xyz = arr[f, :, :3]
                conf = arr[f, :, 3]
                fin = np.isfinite(xyz).all(1)
                if fin.sum() == 0:
                    continue
                p3d = torch.tensor(xyz[fin], dtype=torch.float32, device=dev)
                repro2d = repro.reprojectPoint(p3d).cpu().numpy()  # (n,C,2)
                kp_ids = np.nonzero(fin)[0]

                for c in range(C):
                    if not valid[s, c, f]:
                        continue
                    m = unpack_mask(packed[s, c, f], W)
                    for li, j in enumerate(kp_ids):
                        x, y = repro2d[li, c]
                        xi, yi = int(round(x)), int(round(y))
                        if not (0 <= xi < W and 0 <= yi < H):
                            inside = False
                        else:
                            y0, y1 = max(0, yi - r), min(H, yi + r + 1)
                            x0, x1 = max(0, xi - r), min(W, xi + r + 1)
                            inside = bool(m[y0:y1, x0:x1].any())
                        cov[s]["total"] += 1
                        cov[s]["kp_total"][j] += 1
                        bout_local[s][1] += 1
                        if inside:
                            cov[s]["inside"] += 1
                            cov[s]["kp_inside"][j] += 1
                            cov[s]["in_conf"].append(conf[j])
                            bout_local[s][0] += 1
                        else:
                            cov[s]["out_conf"].append(conf[j])
        per_bout.append((os.path.basename(bd), bout_local))

    # Female identity is set explicitly (red=fly0 per viz/user), NOT by body
    # length (which barely separates the two here).
    female = args.female_slot
    male = 1 - female
    bl = {s: np.nanmedian(np.concatenate(cov[s]["blen"])) for s in (0, 1)}
    lab = {female: f"FEMALE(fly{female})", male: f"MALE(fly{male})"}
    print(f"[id] body length fly0={bl[0]:.2f} fly1={bl[1]:.2f} "
          f"(female set to fly{female})\n")

    if args.per_bout:
        print("=== per-bout coverage by slot (identity-consistency check) ===")
        print(f"{'bout':<14}{'fly0 cov':>10}{'fly1 cov':>10}")
        for name, bl_ in per_bout:
            f0 = bl_[0][0] / bl_[0][1] if bl_[0][1] else float('nan')
            f1 = bl_[1][0] / bl_[1][1] if bl_[1][1] else float('nan')
            print(f"{name:<14}{f0:>10.3f}{f1:>10.3f}")
        print()

    print("=== kp_coverage: fraction of reprojected keypoints inside SAM mask ===")
    print(f"{'fly':<18}{'coverage':>10}{'#out':>10}{'#total':>10}"
          f"{'out-conf med':>14}{'in-conf med':>13}")
    for s in (female, male):
        c = cov[s]
        frac = c["inside"] / c["total"] if c["total"] else float("nan")
        oc = np.median(c["out_conf"]) if c["out_conf"] else float("nan")
        ic = np.median(c["in_conf"]) if c["in_conf"] else float("nan")
        print(f"{lab[s]:<18}{frac:>10.3f}{len(c['out_conf']):>10}"
              f"{c['total']:>10}{oc:>14.3f}{ic:>13.3f}")

    print("\n=== Worst keypoints by coverage (female) ===")
    c = cov[female]
    with np.errstate(invalid="ignore"):
        kpcov = c["kp_inside"] / np.maximum(c["kp_total"], 1)
    order = np.argsort(kpcov)
    print(f"{'keypoint':<16}{'coverage':>10}{'n':>8}")
    for j in order[:15]:
        print(f"{kp_names[j]:<16}{kpcov[j]:>10.3f}{int(c['kp_total'][j]):>8}")


if __name__ == "__main__":
    main()
