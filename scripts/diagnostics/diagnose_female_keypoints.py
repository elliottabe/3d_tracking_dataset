"""Phase-1 diagnosis: why are female fly keypoints worse than the male's?

Computes per-fly metrics from a shard-pipeline Predictions_3D_* directory
(per-bout fly0/fly1 CSVs + sam3_masks.npz). No reprojection needed for the
metrics here; kp_coverage (keypoints-inside-mask) is handled separately in the
jarvis env.

Metrics (female vs male):
  - body length (Antenna_Base <-> Abd_tip) -> identifies which fly is female
  - keypoint confidence (per-joint, mean) -> "jumbled / diffuse heatmap" signal
  - temporal jitter (3D speed between frames) -> instability signal
  - SAM mask quality: area, valid-camera count, centroid jitter -> mask-as-cause

Usage:
  python diagnose_female_keypoints.py --pred_dir <Predictions_3D_*> \
      --config <project>/config.yaml
"""
import argparse
import glob
import os

import numpy as np
import yaml

# popcount lookup for fast mask-area from packed bits
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def load_fly_csv(path):
    """Return (frames, 50, 4) array [x,y,z,conf]; skips 2 header rows + frame col."""
    data = np.genfromtxt(path, delimiter=",", skip_header=2)
    if data.ndim == 1:
        data = data[None, :]
    data = data[:, 1:]  # drop leading frame index column
    F = data.shape[0]
    return data.reshape(F, -1, 4)


def mask_areas(packed):
    """packed (C,F,H,Wp) uint8 -> area (C,F) pixel counts via popcount."""
    return _POPCOUNT[packed].sum(axis=(2, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--conf_thresh", type=float, default=0.3,
                    help="confidence below this = 'low-confidence' keypoint")
    ap.add_argument("--female_slot", type=int, default=0,
                    help="fly slot that is the FEMALE (red=fly0 per viz/user). "
                         "-1 = auto by body length (unreliable here)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    kp_names = cfg["KEYPOINT_NAMES"]
    nkp = len(kp_names)
    idx = {n: i for i, n in enumerate(kp_names)}
    i_ant = idx.get("Antenna_Base", 0)
    i_tip = idx.get("Abd_tip", 5)

    def _ready(b):
        for fn in ("fly0.csv", "fly1.csv", "sam3_masks.npz"):
            p = os.path.join(b, fn)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                return False
        return True
    bout_dirs = [b for b in sorted(glob.glob(os.path.join(args.pred_dir, "bout_*")))
                 if _ready(b)]
    print(f"Found {len(bout_dirs)} completed bouts in {args.pred_dir}\n")

    # Accumulators per fly slot (0,1)
    acc = {s: dict(conf=[], speed=[], blen=[], marea=[], mvalid=[], mcjit=[])
           for s in (0, 1)}

    for bd in bout_dirs:
        masks = np.load(os.path.join(bd, "sam3_masks.npz"))
        packed = masks["packed"]      # (A,C,F,H,Wp)
        valid = masks["valid"]        # (A,C,F)
        centroids = masks["centroids"]  # (A,C,F,2)
        A = packed.shape[0]

        for s in range(min(2, A)):
            arr = load_fly_csv(os.path.join(bd, f"fly{s}.csv"))  # (F,nkp,4)
            xyz = arr[..., :3]
            conf = arr[..., 3]
            finite = np.isfinite(xyz).all(-1)

            # body length per frame (both endpoints finite)
            ok = finite[:, i_ant] & finite[:, i_tip]
            if ok.any():
                blen = np.linalg.norm(xyz[ok, i_ant] - xyz[ok, i_tip], axis=1)
                acc[s]["blen"].append(blen)

            # confidence per keypoint (only where finite)
            c = np.where(finite, conf, np.nan)
            acc[s]["conf"].append(c)

            # temporal speed per keypoint (consecutive finite frames)
            d = np.diff(xyz, axis=0)
            sp = np.linalg.norm(d, axis=2)
            both = finite[1:] & finite[:-1]
            sp = np.where(both, sp, np.nan)
            acc[s]["speed"].append(sp)

            # mask metrics
            area = mask_areas(packed[s])          # (C,F)
            acc[s]["marea"].append(area[valid[s]])  # only valid entries
            acc[s]["mvalid"].append(valid[s].sum(axis=0))  # (F,) cams valid
            # centroid jitter per cam (valid consecutive)
            cj = np.linalg.norm(np.diff(centroids[s], axis=1), axis=2)  # (C,F-1)
            vv = valid[s][:, 1:] & valid[s][:, :-1]
            acc[s]["mcjit"].append(cj[vv])

    # ---- identify female ----
    blen_med = {s: np.nanmedian(np.concatenate(acc[s]["blen"]))
                if acc[s]["blen"] else np.nan for s in (0, 1)}
    if args.female_slot in (0, 1):
        female = args.female_slot
    else:
        female = 0 if blen_med[0] >= blen_med[1] else 1
    male = 1 - female
    lab = {female: f"FEMALE(fly{female})", male: f"MALE(fly{male})"}
    print("=== Identity ===")
    src = ("explicit (red=fly0 per viz/user)" if args.female_slot in (0, 1)
           else "auto by body length")
    for s in (0, 1):
        print(f"  fly{s}: median body length = {blen_med[s]:.2f}  -> {lab[s]}")
    print(f"  (female assignment: {src})")
    print()

    def agg(s, key):
        return np.concatenate([a.ravel() for a in acc[s][key]]) if acc[s][key] \
            else np.array([])

    # ---- summary table ----
    print("=== Per-fly summary (female vs male) ===")
    print(f"{'metric':<32}{'FEMALE':>14}{'MALE':>14}{'F/M ratio':>12}")
    rows = []
    for name, key, reduce_fn in [
        ("mean confidence", "conf", np.nanmean),
        ("median confidence", "conf", np.nanmedian),
        ("% kp below conf thresh", "conf", None),
        ("median 3D speed (units/fr)", "speed", np.nanmedian),
        ("p90 3D speed (units/fr)", "speed", lambda x: np.nanpercentile(x, 90)),
        ("median mask area (px)", "marea", np.nanmedian),
        ("mean valid-cam count /7", "mvalid", np.nanmean),
        ("median centroid jitter (px)", "mcjit", np.nanmedian),
    ]:
        vals = {}
        for s in (female, male):
            d = agg(s, key)
            if name.startswith("% kp below"):
                vals[s] = 100.0 * np.nanmean(d < args.conf_thresh)
            else:
                vals[s] = reduce_fn(d) if d.size else np.nan
        ratio = vals[female] / vals[male] if vals[male] else np.nan
        print(f"{name:<32}{vals[female]:>14.3f}{vals[male]:>14.3f}{ratio:>12.2f}")

    # ---- per-keypoint confidence gap (worst female joints) ----
    print("\n=== Worst female keypoints (lowest mean confidence) ===")
    cf = np.concatenate(acc[female]["conf"], axis=0)  # (Ftot, nkp)
    cm = np.concatenate(acc[male]["conf"], axis=0)
    fmean = np.nanmean(cf, axis=0)
    mmean = np.nanmean(cm, axis=0)
    order = np.argsort(fmean)  # lowest first
    print(f"{'keypoint':<16}{'F conf':>9}{'M conf':>9}{'gap(M-F)':>10}")
    for j in order[:15]:
        print(f"{kp_names[j]:<16}{fmean[j]:>9.3f}{mmean[j]:>9.3f}"
              f"{mmean[j]-fmean[j]:>10.3f}")


if __name__ == "__main__":
    main()
