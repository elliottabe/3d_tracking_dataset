#!/usr/bin/env python
"""Render STAC-fit verification frames: model (posed + calibrated) with the data
keypoints, the fitted model markers, and a red tendon between each pair whose
length IS the per-keypoint residual (``show_marker_error``).

Use this to confirm keypoints land on the model marker sites even where the
*visual mesh* shows a gap (the kinematic-only calibration morph lengthens the
joint/site but leaves the base-size mesh, so a lengthened femur looks detached at
the femur-tibia joint while the site/keypoint still coincide).

Usage:
    python scripts/render_fit_check.py <ik_h5> [--start N] [--n N] [--camera track1]
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
import argparse
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stac-mjx"))
from stac_mjx.viz import viz_stac  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ik_h5", type=Path)
    ap.add_argument("--start", type=int, default=1000, help="start frame")
    ap.add_argument("--n", type=int, default=120, help="frames to render")
    ap.add_argument("--camera", default="track1")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--body-model-dir", type=Path,
                    default=Path("/home/eabe/Research/MyRepos/fruitfly_body_models"))
    ap.add_argument("--no-error", action="store_true",
                    help="hide the red residual tendons (show mesh + points only)")
    ap.add_argument("--n-stills", type=int, default=3)
    args = ap.parse_args()

    out = args.out or (args.ik_h5.parent / "fit_check")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.camera}_{args.start}{'_noerr' if args.no_error else ''}"
    mp4 = out / f"fit_check_{tag}.mp4"

    _, frames = viz_stac(
        str(args.ik_h5), None, args.n, str(mp4),
        start_frame=args.start, camera=args.camera, height=900, width=1400,
        base_path=args.body_model_dir, show_marker_error=not args.no_error,
    )
    for i in np.linspace(0, len(frames) - 1, args.n_stills).astype(int):
        png = out / f"fit_check_{tag}_f{args.start + int(i)}.png"
        imageio.imwrite(png, frames[int(i)])
        print(f"wrote {png}")
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
