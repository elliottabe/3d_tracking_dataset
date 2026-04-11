#!/usr/bin/env python3
"""Build a vertically stacked multi-camera clip for a single bout frame range.

Given a directory of per-camera ``Cam<serial>.mp4`` files recorded synchronously
on a high-speed rig, extract ``[start_frame, end_frame]`` from each selected
camera, label each pane with its camera name, and vstack them into a single
mp4. Playback fps defaults to 30 so sub-second bouts become watchable slow
motion.

Speed notes:
    The rig mp4s have a GOP of ~1 second (~800 frames), so input-side ``-ss``
    seek is fast *and* frame-accurate for H.264 — ffmpeg snaps to the preceding
    keyframe, then decodes forward only to the requested time. Decoding from
    frame 0 with the ``trim`` filter is orders of magnitude slower and can OOM
    on long recordings, so this script always uses ``-ss``/``-t``.

    On a cluster filesystem like GPFS, parsing the moov atom of a 10 GB mp4 is
    latency-bound, and a single ffmpeg invocation with 7 ``-i`` inputs opens
    them serially. To hide that latency, we run the per-camera extractions in
    parallel as 7 concurrent ffmpeg processes, then do a second, trivially fast
    pass that vstacks the 7 tiny per-camera clips.

Usage:
    python scripts/make_bout_clip.py \\
        --video-dir /path/to/Session1/2026_04_02_12_11_50 \\
        --start-frame 125664 --end-frame 126200 \\
        --cameras Cam2012630 Cam2012631 Cam2012853 Cam2012855 \\
                  Cam2012857 Cam2012861 Cam2012862 \\
        --output bout1_fly0_allcams_vstack.mp4 \\
        --hwaccel cuda

If ``--cameras`` is omitted, every ``Cam*.mp4`` in ``--video-dir`` is used
(sorted by filename). If ``--output`` is a bare filename it is written inside
``--video-dir``; absolute paths are honored as-is.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _discover_cameras(video_dir: Path) -> list[str]:
    cams = sorted(p.stem for p in video_dir.glob("Cam*.mp4"))
    if not cams:
        raise SystemExit(f"No Cam*.mp4 files found in {video_dir}")
    return cams


def _resolve_output(video_dir: Path, output: str) -> Path:
    out = Path(output)
    if not out.is_absolute():
        out = video_dir / out
    return out


def _probe_fps(video_path: Path) -> float:
    """Return the average frame rate of ``video_path`` as a float (frames/s)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "json",
            str(video_path),
        ],
        check=True, capture_output=True, text=True,
    )
    rate = json.loads(result.stdout)["streams"][0]["avg_frame_rate"]
    num, _, den = rate.partition("/")
    fps = float(num) / float(den) if den else float(num)
    if fps <= 0:
        raise SystemExit(f"Could not determine fps for {video_path}")
    return fps


def _extract_one_cmd(
    cam_path: Path,
    start_time: float,
    duration: float,
    out_path: Path,
    hwaccel: str | None,
) -> list[str]:
    """ffmpeg command for stage 1: pull a short clip from one camera.

    Uses input-side ``-ss`` for fast, frame-accurate keyframe seek, encodes
    with libx264 ultrafast + low CRF for near-lossless intermediate, and
    writes to ``out_path`` (a small file, typically a few MB).
    """
    cmd: list[str] = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
    ]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += [
        "-ss", f"{start_time:.6f}",
        "-t", f"{duration:.6f}",
        "-i", str(cam_path),
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "15",  # visually near-lossless intermediate
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    return cmd


def _build_vstack_filter(
    n_inputs: int,
    cam_names: list[str],
    playback_fps: int,
    label: bool,
) -> str:
    """Stage-2 filter graph: retime each tiny clip to ``playback_fps``, label,
    then vstack them."""
    assert n_inputs == len(cam_names)
    parts: list[str] = []
    for i, cam in enumerate(cam_names):
        chain = f"[{i}:v]setpts=N/{playback_fps}/TB"
        if label:
            chain += (
                f",drawtext=text='{cam}':x=10:y=10:fontsize=24:fontcolor=yellow"
                f":box=1:boxcolor=black@0.5"
            )
        chain += f"[v{i}]"
        parts.append(chain)
    stack_inputs = "".join(f"[v{i}]" for i in range(n_inputs))
    parts.append(f"{stack_inputs}vstack=inputs={n_inputs}[out]")
    return ";".join(parts)


def _vstack_cmd(
    intermediate_paths: list[Path],
    cam_names: list[str],
    playback_fps: int,
    output: Path,
    crf: int,
    preset: str,
    label: bool,
    overwrite: bool,
) -> list[str]:
    cmd: list[str] = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-hide_banner", "-loglevel", "warning", "-stats",
    ]
    for p in intermediate_paths:
        cmd += ["-i", str(p)]
    cmd += [
        "-filter_complex", _build_vstack_filter(
            n_inputs=len(intermediate_paths),
            cam_names=cam_names,
            playback_fps=playback_fps,
            label=label,
        ),
        "-map", "[out]",
        "-r", str(playback_fps),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    return cmd


def _run_parallel(cmds: list[list[str]]) -> None:
    """Launch every command simultaneously with subprocess.Popen and wait.

    Raises SystemExit on the first non-zero exit code (after all have finished
    or been terminated, so we don't leak children)."""
    procs = [subprocess.Popen(c) for c in cmds]
    failures: list[tuple[int, int]] = []
    for i, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            failures.append((i, rc))
    if failures:
        msgs = ", ".join(f"stage1[{i}] rc={rc}" for i, rc in failures)
        raise SystemExit(f"parallel extraction failed: {msgs}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vertically stack a single bout across multiple cameras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--video-dir", type=Path, required=True,
                    help="Directory containing Cam<serial>.mp4 files.")
    ap.add_argument("--start-frame", type=int, required=True,
                    help="First frame of the bout (inclusive, 0-based).")
    ap.add_argument("--end-frame", type=int, required=True,
                    help="Last frame of the bout (inclusive).")
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="Camera basenames without extension, e.g. "
                         "Cam2012630. Defaults to every Cam*.mp4 in --video-dir.")
    ap.add_argument("--output", default=None,
                    help="Output mp4. Bare filename is placed in --video-dir. "
                         "Defaults to bout_<start>_<end>_vstack.mp4.")
    ap.add_argument("--playback-fps", type=int, default=30,
                    help="Output playback frame rate. Source is typically "
                         "~800 fps so 30 gives ~27x slow motion.")
    ap.add_argument("--source-fps", type=float, default=None,
                    help="Override detected source fps. By default this is "
                         "probed from the first camera with ffprobe.")
    ap.add_argument("--hwaccel", default=None,
                    help="ffmpeg -hwaccel method for decode "
                         "(e.g. cuda, vaapi, qsv). None = CPU decode.")
    ap.add_argument("--crf", type=int, default=23, help="libx264 CRF.")
    ap.add_argument("--preset", default="fast", help="libx264 preset.")
    ap.add_argument("--no-label", action="store_true",
                    help="Do not overlay camera names on each pane.")
    ap.add_argument("--no-overwrite", action="store_true",
                    help="Refuse to overwrite an existing output file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the ffmpeg command without executing it.")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH.")
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe not found on PATH.")

    video_dir: Path = args.video_dir.resolve()
    if not video_dir.is_dir():
        raise SystemExit(f"--video-dir not a directory: {video_dir}")

    if args.end_frame <= args.start_frame:
        raise SystemExit(
            f"--end-frame ({args.end_frame}) must be > --start-frame "
            f"({args.start_frame})"
        )

    cam_names = args.cameras or _discover_cameras(video_dir)
    cam_paths = [video_dir / f"{c}.mp4" for c in cam_names]
    missing = [str(p) for p in cam_paths if not p.exists()]
    if missing:
        raise SystemExit("Missing camera files:\n  " + "\n  ".join(missing))

    source_fps = args.source_fps or _probe_fps(cam_paths[0])
    n_frames = args.end_frame - args.start_frame + 1
    start_time = args.start_frame / source_fps
    duration = n_frames / source_fps

    output_name = args.output or f"bout_{args.start_frame}_{args.end_frame}_vstack.mp4"
    output_path = _resolve_output(video_dir, output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cameras     : {', '.join(cam_names)}")
    print(f"Source fps  : {source_fps:.4f}")
    print(f"Frames      : {args.start_frame}..{args.end_frame} ({n_frames} frames)")
    print(f"Time window : {start_time:.4f}s .. {start_time + duration:.4f}s "
          f"({duration:.4f}s at source fps)")
    print(f"Playback    : {args.playback_fps} fps "
          f"-> clip length {n_frames / args.playback_fps:.2f}s "
          f"({source_fps / args.playback_fps:.1f}x slow-motion)")
    print(f"Output      : {output_path}")
    print(f"Decode      : {'hwaccel=' + args.hwaccel if args.hwaccel else 'CPU'}")

    with tempfile.TemporaryDirectory(prefix="make_bout_clip_") as tmp:
        tmp_dir = Path(tmp)
        intermediate_paths = [tmp_dir / f"{c}.mp4" for c in cam_names]

        stage1_cmds = [
            _extract_one_cmd(
                cam_path=cp,
                start_time=start_time,
                duration=duration,
                out_path=ip,
                hwaccel=args.hwaccel,
            )
            for cp, ip in zip(cam_paths, intermediate_paths)
        ]
        stage2_cmd = _vstack_cmd(
            intermediate_paths=intermediate_paths,
            cam_names=cam_names,
            playback_fps=args.playback_fps,
            output=output_path,
            crf=args.crf,
            preset=args.preset,
            label=not args.no_label,
            overwrite=not args.no_overwrite,
        )

        if args.dry_run:
            print()
            print(f"Stage 1 (parallel x{len(stage1_cmds)}):")
            for c in stage1_cmds:
                print("  " + " ".join(shlex.quote(x) for x in c))
            print("Stage 2:")
            print("  " + " ".join(shlex.quote(x) for x in stage2_cmd))
            return

        print(f"\nStage 1: extracting {len(stage1_cmds)} per-camera clips in parallel...")
        _run_parallel(stage1_cmds)

        print("Stage 2: vstacking intermediates...")
        try:
            subprocess.run(stage2_cmd, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
