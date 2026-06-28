#!/usr/bin/env python3
"""
Render predicted 3D keypoints reprojected onto the raw camera video for one bout.

Reuses the JARVIS monolith's create_multi_animal_videos3D on the D3 per-bout
per-fly CSVs (predictions/bout_<idx>/fly{0,1}.csv), which are dense over the
bout's [start,end] frame range. (The session-level data3D_fly*.csv is sparse —
only bout frames — so it is NOT suitable for the frame_start+offset indexing the
renderer uses; always render from the per-bout CSVs.)

CPU-only (reprojection on CPU + cv2 draw/encode) — fine to run on a login node
for a single bout / a few cameras.

Usage:
    python scripts/viz_predictions_reproject.py \
        --session-dir /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04 \
        --pred-dir   /gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/predictions \
        --bout 1 --cameras Cam2002091 --with-masks
"""
import argparse
import csv as _csv
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_ROOT = PROJECT_DIR / "third_party" / "JARVIS-HybridNet"
DEFAULT_JARVIS_ROOT = "/gscratch/portia/eabe/Research/Github/JARVIS-HybridNet"


def _bout_frame_range(fly_csv):
    """(frame_start, number_frames) from a dense per-bout fly CSV (2 header rows
    + one row per frame, first column = abs frame index)."""
    with open(fly_csv, newline="") as f:
        rows = list(_csv.reader(f))[2:]
    frames = [int(r[0]) for r in rows if r and r[0].strip()]
    if not frames:
        raise ValueError(f"no data rows in {fly_csv}")
    return frames[0], len(frames)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", required=True,
                    help="Raw recording dir (Cam*.mp4 + calibration/)")
    ap.add_argument("--pred-dir", required=True,
                    help="Processed predictions dir (contains bout_<idx>/fly{0,1}.csv)")
    ap.add_argument("--bout", type=int, required=True, help="bout_idx to render")
    ap.add_argument("--project", default="red_data_unified")
    ap.add_argument("--jarvis-root", default=DEFAULT_JARVIS_ROOT,
                    help="JARVIS checkout that owns projects/ (for ProjectManager)")
    ap.add_argument("--cameras", default=None,
                    help="Comma-separated camera names (default: all)")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <pred-dir>/viz/bout_<idx>)")
    ap.add_argument("--with-masks", action="store_true",
                    help="Overlay the SAM3 masks from sam3_masks/bout_<idx>/sam3_masks.npz")
    args = ap.parse_args()

    for p in (str(CODE_ROOT), str(CODE_ROOT / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)

    # ProjectManager hard-codes parent_dir to the install root (third_party copy,
    # which lacks red_data_unified). Patch __init__ so the renderer's internal
    # ProjectManager() resolves the project from the real checkout.
    import jarvis.config.project_manager as pmmod
    _orig_init = pmmod.ProjectManager.__init__

    def _patched_init(self):
        _orig_init(self)
        self.parent_dir = args.jarvis_root
    pmmod.ProjectManager.__init__ = _patched_init

    from jarvis.visualization.create_multi_animal_videos3D import (
        create_multi_animal_videos3D)

    bout_dir = os.path.join(args.pred_dir, f"bout_{args.bout:05d}")
    fly0 = os.path.join(bout_dir, "fly0.csv")
    fly1 = os.path.join(bout_dir, "fly1.csv")
    if not os.path.isfile(fly0):
        sys.exit(f"Per-bout CSV not found: {fly0} (has bout {args.bout} been predicted?)")
    data_csvs = {"fly0": fly0}
    if os.path.isfile(fly1):
        data_csvs["fly1"] = fly1

    frame_start, number_frames = _bout_frame_range(fly0)
    calib = os.path.join(args.session_dir, "calibration")
    cams = [c for c in args.cameras.split(",") if c] if args.cameras else None
    out = args.out or os.path.join(args.pred_dir, "viz", f"bout_{args.bout:05d}")
    os.makedirs(out, exist_ok=True)

    mask_file = None
    if args.with_masks:
        mf = os.path.join(os.path.dirname(args.pred_dir.rstrip("/")),
                          "sam3_masks", f"bout_{args.bout:05d}", "sam3_masks.npz")
        mask_file = mf if os.path.isfile(mf) else None
        if mask_file is None:
            print(f"[viz] --with-masks set but no mask npz at {mf}; rendering without masks")

    print(f"[viz] bout {args.bout}: frames {frame_start}..{frame_start+number_frames-1} "
          f"({number_frames} frames), flies={sorted(data_csvs)}, "
          f"cameras={cams or 'all'} -> {out}")
    result = create_multi_animal_videos3D(
        project_name=args.project,
        recording_path=args.session_dir,
        data_csvs=data_csvs,
        dataset_name=calib,
        frame_start=frame_start,
        number_frames=number_frames,
        video_cam_list=cams,
        output_dir=out,
        mask_file=mask_file,
    )
    if result is None:
        sys.exit("[viz] create_multi_animal_videos3D returned None (see errors above)")
    print(f"[viz] wrote videos to {result}")


if __name__ == "__main__":
    main()
