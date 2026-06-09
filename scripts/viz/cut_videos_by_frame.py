## example usage:
##   # num-frames form, default output dir (clips_<start>_<end> inside input_dir)
##   python cut_videos_by_frame.py \
##       "/mnt/lemebel/happyhouse_102025/session6/2025_10_12_15_06_46" \
##       --start-frame 107583 --num-frames 524
##
##   # end-frame form + subset of cameras + explicit output dir
##   python cut_videos_by_frame.py \
##       "/path/to/2026_04_02_12_11_50" \
##       --start-frame 125664 --end-frame 126200 \
##       --cameras Cam2012630 Cam2012631 Cam2012853
##
## By default the script writes one cropped clip per video into
## ``<input_dir>/clips_<start>_<end>/`` (created if missing). Pass
## ``--output-dir`` to override.


import argparse
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


def crop_video_frames(
    in_path: Path,
    out_path: Path,
    start_frame: int,
    num_frames: int,
    codec: str = "avc1",
):
    """Crop a single video from start_frame for num_frames frames."""
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {in_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start_frame >= total_frames:
        print(f"[WARN] start_frame {start_frame} >= total_frames {total_frames} for {in_path.name}, skipping.")
        cap.release()
        return

    # Clamp num_frames so we don't read past the end
    max_possible = total_frames - start_frame
    num_frames = min(num_frames, max_possible)
    end_frame = start_frame + num_frames - 1

    print(
        f"[INFO] Processing {in_path.name}: total_frames={total_frames}, "
        f"start={start_frame}, end={end_frame}, num_frames={num_frames}, fps={fps:.3f}"
    )

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(
            f"[ERR] Could not open VideoWriter with codec '{codec}' for "
            f"{out_path}. Your OpenCV build may not support this fourcc; "
            f"try --codec mp4v (not VSCode-playable) or install an "
            f"opencv-python wheel with H.264/ffmpeg support."
        )

    # Position to starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0
    current_frame_idx = start_frame

    while frames_written < num_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Early end of file at frame {current_frame_idx} for {in_path.name}")
            break

        writer.write(frame)
        frames_written += 1
        current_frame_idx += 1

    cap.release()
    writer.release()

    print(f"[OK] Wrote {frames_written} frames → {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Crop each video in a directory to a frame range using OpenCV. "
                    "Writes one cropped clip per source video into a sibling "
                    "folder inside the input directory by default."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing input videos (e.g. a session folder with "
             "Cam*.mp4 files).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save cropped clips. Defaults to "
             "<input_dir>/clips_<start>_<end>.",
    )
    parser.add_argument(
        "--start-frame",
        "-s",
        type=int,
        required=True,
        help="Starting frame index (0-based, inclusive).",
    )
    frame_end = parser.add_mutually_exclusive_group(required=True)
    frame_end.add_argument(
        "--num-frames",
        "-n",
        type=int,
        default=None,
        help="Number of frames to keep (m). Output will contain frames "
             "[start_frame, start_frame + num_frames - 1].",
    )
    frame_end.add_argument(
        "--end-frame",
        "-e",
        type=int,
        default=None,
        help="Last frame to keep (inclusive). Equivalent to "
             "--num-frames (end_frame - start_frame + 1).",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Optional list of camera basenames (without extension) to include, "
             "e.g. Cam2012630 Cam2012631. Defaults to every video found in "
             "--input-dir.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="avc1",
        help="FourCC video codec (default: avc1 = H.264 in mp4, playable in "
             "the VSCode video preview and any Chromium-based viewer). "
             "Use mp4v (MPEG-4 Part 2) if your OpenCV build lacks H.264, but "
             "note that browsers and VSCode won't play mp4v. Other options: "
             "XVID, MJPG.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    if args.num_frames is not None:
        num_frames = args.num_frames
        if num_frames <= 0:
            raise SystemExit(f"--num-frames must be positive (got {num_frames})")
    else:
        if args.end_frame < args.start_frame:
            raise SystemExit(
                f"--end-frame ({args.end_frame}) must be >= --start-frame "
                f"({args.start_frame})"
            )
        num_frames = args.end_frame - args.start_frame + 1

    end_frame = args.start_frame + num_frames - 1

    if args.output_dir is None:
        output_dir = input_dir / f"clips_{args.start_frame}_{end_frame}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_videos = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS
    )
    if not all_videos:
        print(f"[WARN] No video files found in {input_dir}")
        return

    if args.cameras:
        wanted = set(args.cameras)
        video_files = [v for v in all_videos if v.stem in wanted]
        missing = wanted - {v.stem for v in video_files}
        if missing:
            raise SystemExit(
                "Requested cameras not found in input_dir: "
                + ", ".join(sorted(missing))
            )
    else:
        video_files = all_videos

    print(f"[INFO] Found {len(all_videos)} video(s) in {input_dir}; "
          f"processing {len(video_files)}")
    print(f"[INFO] Frame range : {args.start_frame}..{end_frame} "
          f"({num_frames} frames)")
    print(f"[INFO] Output dir  : {output_dir}")

    for vid in video_files:
        out_name = f"{vid.stem}_frames_{args.start_frame}_{end_frame}{vid.suffix}"
        out_path = output_dir / out_name
        crop_video_frames(
            vid,
            out_path,
            start_frame=args.start_frame,
            num_frames=num_frames,
            codec=args.codec,
        )


if __name__ == "__main__":
    main()


'''
CLIP_DIR=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1/2026_04_02_15_25_51/clips_619962_620403
ffmpeg -y -hide_banner -loglevel warning -stats \
  -i "$CLIP_DIR/Cam2012630_frames_619962_620403.mp4" \
  -i "$CLIP_DIR/Cam2012631_frames_619962_620403.mp4" \
  -i "$CLIP_DIR/Cam2012861_frames_619962_620403.mp4" \
  -filter_complex "[0:v]setpts=N/30/TB[v0];[1:v]setpts=N/30/TB[v1];[2:v]setpts=N/30/TB[v2];[v0][v1][v2]vstack=inputs=3[out]" \
  -map "[out]" -r 60 \
  -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
  "$CLIP_DIR/bout_619962_620403_vstack_3cam.mp4"
'''