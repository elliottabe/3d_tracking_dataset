#!/usr/bin/env python3
"""Re-render per-camera overlay clips for a single bout, with optional ID swap.

Given a bout directory produced by ``tools/predict3D_multianimal.py`` that
contains ``fly0.csv``, ``fly1.csv``, and (optionally) ``sam3_masks.npz``,
call JARVIS's ``create_multi_animal_videos3D`` to regenerate the per-camera
annotated clips. ``--swap-flies`` swaps which CSV feeds fly0 vs fly1 and
which packed-mask channel is used for each, so labels/colors/masks track
the true identity when the tracker mislabeled them.

The raw multi-camera recordings are assumed to sit two folders up from the
bout dir (``<session>/Predictions_3D_*/bout_NNNNN/``), matching the layout
written by the tracker.

Outputs go to ``<bout_dir>/clips_swapped/`` by default so the original
``clips/`` directory is preserved.

Usage:
    python scripts/render_bout_clips.py \\
        --bout-dir /path/to/Predictions_3D_34662595/bout_00014 \\
        --project merge_courtship_V3 \\
        --swap-flies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from jarvis.visualization.create_multi_animal_videos3D import (
    create_multi_animal_videos3D,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stack_clips import stack_clips  # noqa: E402


def _infer_start_and_len(csv_path: Path) -> tuple[int, int]:
    """Return (start_frame, n_frames) from a JARVIS per-fly CSV.

    The CSVs have two header rows then one row per frame with the source
    frame index in column 0.
    """
    data = np.genfromtxt(csv_path, delimiter=',')
    if np.isnan(data[0, 0]):
        data = data[2:]
    start = int(data[0, 0])
    return start, data.shape[0]


def _parse_bgr(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected B,G,R triple (e.g. 0,0,255); got {s!r}"
        )
    return tuple(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bout-dir", type=Path, required=True,
                    help="Predictions_3D_*/bout_NNNNN directory.")
    ap.add_argument("--project", required=True, default="red_data_unified",
                    help="JARVIS project name (e.g. merge_courtship_V3).")
    ap.add_argument("--swap-flies", action="store_true",
                    help="Swap fly0<->fly1 identities at render time "
                         "(CSVs + mask channels together).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Output directory. Defaults to "
                         "<bout_dir>/clips_swapped when --swap-flies is set, "
                         "else <bout_dir>/clips_rerender.")
    ap.add_argument("--fly0-color", type=_parse_bgr, default=None,
                    help="Override fly0 color as B,G,R (default red: 0,0,255).")
    ap.add_argument("--fly1-color", type=_parse_bgr, default=None,
                    help="Override fly1 color as B,G,R (default blue: 255,180,0).")
    ap.add_argument("--n-jobs", type=int, default=8,
                    help="Parallel jobs for per-camera frame reading.")
    ap.add_argument("--stack", action="store_true",
                    help="After rendering, vstack the per-camera clips into "
                         "<output_dir>/<output_dir_name>_vstack.mp4.")
    ap.add_argument("--stack-cameras", nargs="+",
                    default=["Cam2012630", "Cam2012631", "Cam2012861"],
                    help="Camera names in top→bottom order for --stack.")
    ap.add_argument("--stack-fps", type=int, default=60,
                    help="Playback fps for the stacked mp4.")
    args = ap.parse_args()

    bout_dir: Path = args.bout_dir.resolve()
    if not bout_dir.is_dir():
        raise SystemExit(f"--bout-dir not a directory: {bout_dir}")

    fly0_csv = bout_dir / "fly0.csv"
    fly1_csv = bout_dir / "fly1.csv"
    for p in (fly0_csv, fly1_csv):
        if not p.is_file():
            raise SystemExit(f"Missing required file: {p}")

    mask_file = bout_dir / "sam3_masks.npz"
    mask_arg = str(mask_file) if mask_file.is_file() else None

    # Session dir holds the raw Cam*.mp4 files; two levels up from the bout.
    session_dir = bout_dir.parents[1]
    if not any(session_dir.glob("Cam*.mp4")):
        raise SystemExit(
            f"No Cam*.mp4 files in inferred recording path: {session_dir}"
        )

    start_frame, n_frames = _infer_start_and_len(fly0_csv)
    _, n_frames_1 = _infer_start_and_len(fly1_csv)
    n_frames = min(n_frames, n_frames_1)

    output_dir = args.output_dir or (
        bout_dir / ("clips_swapped" if args.swap_flies else "clips_rerender")
    )

    fly_colors = None
    if args.fly0_color is not None or args.fly1_color is not None:
        fly_colors = {}
        if args.fly0_color is not None:
            fly_colors['fly0'] = args.fly0_color
        if args.fly1_color is not None:
            fly_colors['fly1'] = args.fly1_color

    print(f"Bout dir       : {bout_dir}")
    print(f"Recording path : {session_dir}")
    print(f"Mask file      : {mask_arg}")
    print(f"Frame range    : {start_frame}..{start_frame + n_frames - 1} "
          f"({n_frames} frames)")
    print(f"Swap flies     : {args.swap_flies}")
    print(f"Output dir     : {output_dir}")

    create_multi_animal_videos3D(
        project_name=args.project,
        recording_path=str(session_dir),
        data_csvs={'fly0': str(fly0_csv), 'fly1': str(fly1_csv)},
        dataset_name=None,
        frame_start=start_frame,
        number_frames=n_frames,
        fly_colors=fly_colors,
        output_dir=str(output_dir),
        n_jobs=args.n_jobs,
        mask_file=mask_arg,
        swap_flies=args.swap_flies,
    )

    if args.stack:
        stack_clips(
            clip_dir=output_dir,
            cameras=args.stack_cameras,
            playback_fps=args.stack_fps,
        )


if __name__ == "__main__":
    main()
