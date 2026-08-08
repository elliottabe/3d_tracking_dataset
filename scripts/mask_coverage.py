#!/usr/bin/env python3
"""Per-camera SAM3 mask coverage as a first-class QC signal (Task 17).

The pipeline used to silently triangulate 3-D keypoints from however many
cameras happened to have a valid SAM3 mask for a fly in a given frame, with
no record of how few that sometimes was. Inspecting real
`<recording>/sam3_masks/bout_XXXXX/sam3_masks.npz` files (keys: `packed
(2,7,T,H,W) uint8`, `valid (2,7,T) bool`, `centroids (2,7,T,2) float32`)
showed this is a real failure mode:

  - the male (fly1) is validly segmented in 7/7 cameras in every bout
    examined;
  - the FEMALE (fly0) goes missing entirely from some cameras -- e.g.
    Session0 bout22 fly0 valid in only 3/7 cameras (0% in cams 3,4,5,6),
    bout26 4/7, bout13 5/7, bout14 6/7.

SAM3 is correctly reporting "not found" rather than mis-assigning the same
animal to both fly slots (an earlier, WRONG hypothesis -- do not
re-implement it: where both flies ARE valid in a camera their centroids are
far apart, median 140-884 px, i.e. no duplication). The damage is that
triangulating from very few views is ill-conditioned and produces garbage
that downstream looks like coincident flies or bones changing length
20-50%. NOTE: correlating coverage against measured keypoint badness over 28
bout-flies gives r=-0.63 (2D confidence: r=-0.85) -- coverage is a real
contributor but not the dominant one; don't oversell it.

This module is intentionally stdlib + numpy only (no jax/mujoco/etc.) so it
can run as a cheap, dependency-light CLI over a whole recording's masks
before committing GPU time to the pipeline.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

# >=2 views is the hard minimum DLT needs to triangulate at all; 4 of a
# typical 7-camera rig is the point past which the solution stays
# well-conditioned for the hard poses (near a wall, occluded by the other
# fly) that most need triangulation to be trustworthy rather than merely
# possible. Configurable via --min-views / cfg.masks.min_views, not baked in.
MIN_VIEWS_DEFAULT = 4

# A camera counts as "usable" for a fly if it saw that fly in a strict
# majority of the bout's frames.
_USABLE_CAMERA_FRAC = 0.5
# Above this fraction of frames falling below min_views, an otherwise-ok fly
# (median_views >= min_views) is downgraded from "ok" to "degraded".
_DEGRADED_FRAME_FRAC = 0.2


def load_valid(mask_npz) -> np.ndarray:
    """(n_flies, n_cams, T) bool validity array from a sam3_masks.npz.

    Only reads the `valid` key -- deliberately ignorant of `packed`/
    `centroids`/`cameras` so this stays cheap and dependency-light.
    """
    with np.load(mask_npz) as z:
        return np.asarray(z["valid"]).astype(bool)


def per_frame_views(valid: np.ndarray) -> np.ndarray:
    """(n_flies, T) int count of cameras valid for each fly, each frame."""
    return valid.sum(axis=1).astype(int)


def coverage_report(mask_npz, *, min_views: int = MIN_VIEWS_DEFAULT) -> dict:
    """Per-fly camera-coverage summary for one bout's sam3_masks.npz.

    status is:
      - "insufficient" when the fly's median per-frame view count is below
        min_views (typical/best case is already too few views to trust);
      - "degraded" when the median is fine but more than
        `_DEGRADED_FRAME_FRAC` of frames individually fall below min_views
        (a fly that mostly tracks well but drops out episodically);
      - "ok" otherwise.
    """
    valid = load_valid(mask_npz)
    n_flies, n_cams, n_frames = valid.shape
    views = per_frame_views(valid)  # (n_flies, T)

    per_fly = {}
    for fly in range(n_flies):
        fly_views = views[fly]
        if n_frames > 0:
            per_camera_frac = valid[fly].mean(axis=1)          # (n_cams,)
            median_views = float(np.median(fly_views))
            frac_frames_below_min = float(np.mean(fly_views < min_views))
        else:
            per_camera_frac = np.zeros(n_cams, dtype=float)
            median_views = 0.0
            frac_frames_below_min = 0.0
        cams_usable = int(np.sum(per_camera_frac > _USABLE_CAMERA_FRAC))

        if median_views < min_views:
            status = "insufficient"
        elif frac_frames_below_min > _DEGRADED_FRAME_FRAC:
            status = "degraded"
        else:
            status = "ok"

        per_fly[str(fly)] = {
            "cams_usable": cams_usable,
            "per_camera_frac": [float(f) for f in per_camera_frac],
            "median_views": median_views,
            "frac_frames_below_min": frac_frames_below_min,
            "status": status,
        }

    return {
        "n_flies": int(n_flies),
        "n_cams": int(n_cams),
        "n_frames": int(n_frames),
        "per_fly": per_fly,
    }


def write_report(bout_dir, report: dict) -> Path:
    """Atomically write `report` as <bout_dir>/coverage.json (tmp + os.replace)."""
    bout_dir = Path(bout_dir)
    bout_dir.mkdir(parents=True, exist_ok=True)
    out_path = bout_dir / "coverage.json"
    tmp_path = bout_dir / "coverage.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_path, out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_bouts(masks_root: str) -> list[tuple[int, str]]:
    """Sorted [(bout_idx, mask_npz_path), ...] under masks_root/bout_*/sam3_masks.npz."""
    found = []
    for d in sorted(glob.glob(os.path.join(masks_root, "bout_*"))):
        m = re.match(r"bout_(\d+)$", os.path.basename(d))
        if not m:
            continue
        npz = os.path.join(d, "sam3_masks.npz")
        if os.path.exists(npz):
            found.append((int(m.group(1)), npz))
    return sorted(found)


def _print_table(rows: list[tuple]) -> None:
    header = f"{'bout':>6}  {'fly':>3}  {'cams_usable':>11}  {'median_views':>12}  {'frac_below_min':>14}  status"
    print(header)
    print("-" * len(header))
    for bout_idx, fly, cams_usable, median_views, frac_below, status in rows:
        print(f"{bout_idx:>6}  {fly:>3}  {cams_usable:>11}  {median_views:>12.2f}  "
              f"{frac_below:>14.2f}  {status}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--masks-root", required=True,
                    help="<recording>/sam3_masks directory (contains bout_XXXXX/sam3_masks.npz)")
    ap.add_argument("--bouts", default=None,
                    help="comma-separated bout indices to check (default: all discovered)")
    ap.add_argument("--min-views", type=int, default=MIN_VIEWS_DEFAULT)
    ap.add_argument("--out-dir", default=None,
                    help="write coverage.json here instead of alongside each bout's masks")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the report but do not write coverage.json")
    args = ap.parse_args(argv)

    bouts = _discover_bouts(args.masks_root)
    if args.bouts:
        wanted = {int(x) for x in args.bouts.split(",") if x.strip() != ""}
        bouts = [(idx, path) for idx, path in bouts if idx in wanted]

    rows = []
    status_by_fly: dict[str, dict[str, int]] = {}
    for bout_idx, npz in bouts:
        report = coverage_report(npz, min_views=args.min_views)
        for fly_key, fly_info in report["per_fly"].items():
            rows.append((bout_idx, fly_key, fly_info["cams_usable"],
                        fly_info["median_views"], fly_info["frac_frames_below_min"],
                        fly_info["status"]))
            status_by_fly.setdefault(fly_key, {"ok": 0, "degraded": 0, "insufficient": 0})
            status_by_fly[fly_key][fly_info["status"]] += 1

        if not args.dry_run:
            out_dir = args.out_dir or os.path.dirname(npz)
            write_report(out_dir, report)

    _print_table(rows)
    print()
    total_bout_flies = len(rows)
    n_insufficient = sum(1 for r in rows if r[5] == "insufficient")
    n_degraded = sum(1 for r in rows if r[5] == "degraded")
    n_ok = sum(1 for r in rows if r[5] == "ok")
    print(f"summary: {total_bout_flies} bout-flies across {len(bouts)} bout(s) -- "
          f"ok={n_ok} degraded={n_degraded} insufficient={n_insufficient}")
    for fly_key in sorted(status_by_fly):
        c = status_by_fly[fly_key]
        print(f"  fly{fly_key}: ok={c['ok']} degraded={c['degraded']} insufficient={c['insufficient']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
