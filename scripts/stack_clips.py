#!/usr/bin/env python3
"""Vertically stack pre-cut per-camera clips into a single mp4.

Companion to ``cut_videos_by_frame.py``: once per-camera clips exist in a
``clips_<start>_<end>/`` directory, this script vstacks a chosen subset
(in the order given) and retimes to a watchable playback fps.

Usage:
    python scripts/stack_clips.py \
        /path/to/clips_619962_620403 \
        --cameras Cam2012630 Cam2012631 Cam2012861 \
        --playback-fps 30

If ``--cameras`` is omitted every ``Cam*_frames_*.mp4`` in the directory
is used (sorted by filename).  Camera order on the command line maps
directly to top→bottom in the output.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _discover_clips(clip_dir: Path) -> list[Path]:
    clips = sorted(clip_dir.glob("Cam*.mp4"))
    if not clips:
        raise SystemExit(f"No Cam*_frames_*.mp4 files found in {clip_dir}")
    return clips


def _cam_name(clip_path: Path) -> str:
    """Extract 'Cam2012630' from 'Cam2012630_frames_619962_620403.mp4'."""
    return clip_path.stem.split("_frames_")[0]


def _build_filter(n: int, playback_fps: int) -> str:
    parts: list[str] = []
    for i in range(n):
        parts.append(f"[{i}:v]setpts=N/{playback_fps}/TB[v{i}]")
    stack_inputs = "".join(f"[v{i}]" for i in range(n))
    parts.append(f"{stack_inputs}vstack=inputs={n}[out]")
    return ";".join(parts)


def stack_clips(
    clip_dir: Path,
    cameras: list[str] | None = None,
    output: str | Path | None = None,
    playback_fps: int = 30,
    crf: int = 23,
    preset: str = "fast",
    dry_run: bool = False,
) -> Path:
    """Vstack Cam*.mp4 clips in ``clip_dir``. Returns the output path.

    ``cameras=None`` uses every ``Cam*.mp4`` in the directory sorted by
    filename; otherwise the given list defines the top→bottom order.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH.")

    clip_dir = Path(clip_dir).resolve()
    if not clip_dir.is_dir():
        raise SystemExit(f"Not a directory: {clip_dir}")

    all_clips = _discover_clips(clip_dir)

    if cameras:
        clip_by_cam = {_cam_name(p): p for p in all_clips}
        missing = [c for c in cameras if c not in clip_by_cam]
        if missing:
            raise SystemExit(
                "Cameras not found in clip_dir: " + ", ".join(missing)
                + f"\nAvailable: {', '.join(sorted(clip_by_cam))}"
            )
        selected = [clip_by_cam[c] for c in cameras]
    else:
        selected = all_clips

    out_name = output or f"{clip_dir.name}_vstack.mp4"
    out_path = Path(out_name) if Path(out_name).is_absolute() else clip_dir / out_name

    cam_names = [_cam_name(p) for p in selected]
    print(f"Cameras (top→bottom): {', '.join(cam_names)}")
    print(f"Playback fps        : {playback_fps}")
    print(f"Output              : {out_path}")

    cmd: list[str] = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
    ]
    for p in selected:
        cmd += ["-i", str(p)]
    cmd += [
        "-filter_complex", _build_filter(len(selected), playback_fps),
        "-map", "[out]",
        "-r", str(playback_fps),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    if dry_run:
        print("\n" + " ".join(shlex.quote(x) for x in cmd))
        return out_path

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vertically stack pre-cut camera clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("clip_dir", type=Path,
                    help="Directory containing Cam*_frames_*.mp4 clips.")
    ap.add_argument("--cameras", nargs="+", default=["Cam2012630", "Cam2012631", "Cam2012861"],
                    help="Camera names in top→bottom order, e.g. "
                         "Cam2012630 Cam2012631 Cam2012861. "
                         "Defaults to all clips (sorted by filename).")
    ap.add_argument("--output", "-o", default=None,
                    help="Output filename. Bare name is placed in clip_dir. "
                         "Defaults to <clip_dir_name>_vstack.mp4.")
    ap.add_argument("--playback-fps", type=int, default=30,
                    help="Output playback frame rate.")
    ap.add_argument("--crf", type=int, default=23, help="libx264 CRF.")
    ap.add_argument("--preset", default="fast", help="libx264 preset.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the ffmpeg command without executing.")
    args = ap.parse_args()

    stack_clips(
        clip_dir=args.clip_dir,
        cameras=args.cameras,
        output=args.output,
        playback_fps=args.playback_fps,
        crf=args.crf,
        preset=args.preset,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
