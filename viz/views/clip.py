"""clip view: multi-camera clip ``cut | stack | render``, reimplemented on
viz.core (no ffmpeg subprocess, no JARVIS).

Ports three former ``scripts/viz/`` tools onto the core io/layout primitives:

- ``cut``    <- scripts/viz/cut_videos_by_frame.py: extract [start,end] from
              each camera's full session video into a per-camera clip.
- ``stack``  <- scripts/viz/stack_clips.py: vertically stack pre-cut
              per-camera clips (an existing ``clips_<start>_<end>/`` dir)
              into one mp4, frame-by-frame via ``core.layout.montage``
              instead of an ffmpeg vstack filter graph.
- ``render`` <- scripts/viz/make_bout_clip.py + render_bout_clips.py, but
              CORE-BASED and JARVIS-FREE by explicit decision: this is a RAW
              stacked multi-camera clip for a frame range, built directly
              from the full session videos via read_frames + montage +
              write_video. It does NOT call JARVIS's
              ``create_multi_animal_videos3D`` and does not draw any
              overlay -- the reprojected-overlay use case is already
              covered by ``viz reproj-video``.

All three modes stream frames (per-camera or per-timestep generators feeding
``core.io.write_video`` directly) rather than accumulating whole videos in
memory, mirroring viz/views/reproj_video.py's pattern. Output uses the
"avc1" (H.264) fourcc so clips are VSCode/browser-playable; ``write_video``
itself falls back to "mp4v" if the OpenCV build can't open an avc1 writer.

Graceful degradation follows viz/views/overlay.py and reproj_video.py:
missing camera video/clip -> warn + skip that camera; unreadable frame ->
skip that frame; nothing usable at all -> raise a clear error naming what's
missing.
"""
import os
from pathlib import Path

import cv2

from viz.core import io as vio
from viz.core import layout


def _fps_or(args, default):
    """Explicit ``--fps`` override if given, else `default`. Used by
    stack/render (playback rate, default 30 -- matches the reference
    make_bout_clip/stack_clips ``--playback-fps`` default)."""
    fps = getattr(args, "fps", None)
    return int(fps) if fps is not None else default


def _probe_fps(video_path):
    """Native fps of a camera's mp4 (cv2.CAP_PROP_FPS), or 0.0 if the probe
    fails/returns an invalid value."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def _discover_cameras(session_dir):
    """Camera basenames (no extension) for every Cam*.mp4 in session_dir."""
    return sorted(p.stem for p in Path(session_dir).glob("Cam*.mp4"))


def _discover_clips(clip_dir):
    """Pre-cut per-camera clip paths (Cam*.mp4, any suffix) in clip_dir."""
    return sorted(Path(clip_dir).glob("Cam*.mp4"))


def _cam_name(clip_path):
    """'Cam2012630' from 'Cam2012630_frames_125664_126200.mp4' (or the bare
    stem if the clip wasn't named with the ``_frames_`` convention)."""
    stem = clip_path.stem
    return stem.split("_frames_")[0] if "_frames_" in stem else stem


def _probe_frame_count(video_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _montage_stream(session_dir, cameras, start, count, tag="clip"):
    """Yield one vstacked BGR frame per timestep for `cameras` under
    session_dir[start:start+count]. read_frames yields RGB (or None on an
    unreadable frame) per camera per timestep; a None anywhere in a
    timestep skips that whole output frame (keeps all camera rows in
    lockstep rather than desyncing the stack) -- but that drop is printed
    (mirroring reproj_video.py's per-frame "warning: frame {t} unreadable
    ...; skipping" pattern) so silent data loss is visible."""
    for t, imgs in enumerate(vio.read_frames(session_dir, cameras, start, count)):
        missing = [cam for cam, img in zip(cameras, imgs) if img is None]
        if missing:
            print(f"[{tag}] warning: timestep {start + t} unreadable for camera(s) "
                  f"{', '.join(missing)}; skipping")
            continue
        tiles = [_to_bgr(img) for img in imgs]
        yield layout.montage(tiles, cols=1)


def _cut(args):
    session_dir = args.session_dir
    if not session_dir or not os.path.isdir(session_dir):
        raise FileNotFoundError(f"clip cut: --session-dir not found: {session_dir}")
    if args.start is None or args.end is None:
        raise ValueError("clip cut requires --start and --end")
    start, end = int(args.start), int(args.end)
    if end < start:
        raise ValueError(f"--end ({end}) must be >= --start ({start})")

    cameras = args.cameras or _discover_cameras(session_dir)
    if not cameras:
        raise RuntimeError(f"clip cut: no Cam*.mp4 videos found in {session_dir}")

    out_dir = args.out or os.path.join(session_dir, f"clips_{start}_{end}")
    os.makedirs(out_dir, exist_ok=True)
    # fps: explicit --fps overrides everything; otherwise each camera's clip
    # is written at ITS OWN native fps (probed below), mirroring
    # scripts/viz/cut_videos_by_frame.py (cap.get(CAP_PROP_FPS) per camera) --
    # this rig captures at ~hundreds of fps, so hardcoding 30 mislabels the
    # cut clip's duration/speed.
    explicit_fps = getattr(args, "fps", None)

    wrote_any = False
    for cam in cameras:
        video_path = os.path.join(session_dir, f"{cam}.mp4")
        if not os.path.isfile(video_path):
            print(f"[clip:cut] warning: no video for camera {cam} at {video_path}; skipping")
            continue

        total = _probe_frame_count(video_path)
        if total <= 0:
            print(f"[clip:cut] warning: could not read frame count for {cam} ({video_path}); skipping")
            continue
        if start >= total:
            print(f"[clip:cut] warning: start {start} >= total frames {total} for {cam}; skipping")
            continue
        count = min(end - start + 1, total - start)
        clip_end = start + count - 1
        out_path = os.path.join(out_dir, f"{cam}_frames_{start}_{clip_end}.mp4")

        if explicit_fps is not None:
            fps = int(explicit_fps)
        else:
            native_fps = _probe_fps(video_path)
            if native_fps and native_fps > 0:
                fps = native_fps
            else:
                print(f"[clip:cut] warning: could not probe native fps for {cam} "
                      f"({video_path}); defaulting to 30")
                fps = 30.0

        def _gen(cam=cam, count=count):
            for imgs in vio.read_frames(session_dir, [cam], start, count):
                img = imgs[0]
                if img is None:
                    continue
                yield _to_bgr(img)

        try:
            vio.write_video(out_path, _gen(), fps=fps, fourcc="avc1")
        except ValueError:
            print(f"[clip:cut] warning: no readable frames for {cam}; skipping write")
            continue
        print(f"[clip:cut] wrote {out_path}")
        wrote_any = True

    if not wrote_any:
        raise RuntimeError(f"clip cut: no clips written for cameras={cameras} in {session_dir}")
    return 0


def _stack(args):
    clip_dir = args.session_dir
    if not clip_dir or not os.path.isdir(clip_dir):
        raise FileNotFoundError(f"clip stack: --session-dir (pre-cut clip dir) not found: {clip_dir}")

    clip_dir_path = Path(clip_dir)
    all_clips = _discover_clips(clip_dir_path)
    if not all_clips:
        raise RuntimeError(f"clip stack: no Cam*.mp4 clips found in {clip_dir}")

    if args.cameras:
        by_cam = {}
        for p in all_clips:
            by_cam.setdefault(_cam_name(p), p)
        missing = [c for c in args.cameras if c not in by_cam]
        if missing:
            raise RuntimeError(
                f"clip stack: cameras not found in {clip_dir}: {', '.join(missing)}; "
                f"available: {', '.join(sorted(by_cam))}")
        selected = [by_cam[c] for c in args.cameras]
    else:
        selected = all_clips

    counts = [(p, _probe_frame_count(p)) for p in selected]
    valid = [(p, n) for p, n in counts if n > 0]
    unreadable = [str(p) for p, n in counts if n <= 0]
    if unreadable:
        print(f"[clip:stack] warning: could not read frame count for {', '.join(unreadable)}; skipping")
    if not valid:
        raise RuntimeError(f"clip stack: no readable clips among {[str(p) for p in selected]}")

    selected = [p for p, _ in valid]
    lengths = [n for _, n in valid]
    count = min(lengths)
    if len(set(lengths)) > 1:
        print(f"[clip:stack] warning: clip lengths differ ({lengths}); truncating to {count} frames")

    stems = [p.stem for p in selected]
    fps = _fps_or(args, 30)
    out = args.out or str(clip_dir_path / f"{clip_dir_path.name}_vstack.mp4")

    try:
        vio.write_video(
            out, _montage_stream(str(clip_dir_path), stems, 0, count, tag="clip:stack"),
            fps=fps, fourcc="avc1")
    except ValueError as e:
        raise RuntimeError(
            f"clip stack: no output frames for {out} -- every timestep in "
            f"0..{count - 1} had at least one unreadable camera frame among "
            f"{[str(p) for p in selected]} in {clip_dir}") from e
    print(f"[clip:stack] wrote {out}")
    return 0


def _infer_range_from_bout(bout_dir):
    """(start, end) inclusive from a bout's fly0.csv frame column, or
    (None, None) if unavailable."""
    csv_path = os.path.join(bout_dir, "fly0.csv")
    if not os.path.isfile(csv_path):
        print(f"[clip:render] warning: --bout-dir given but {csv_path} not found")
        return None, None
    _kp3d, _conf, _names, frames = vio.load_data3d_csv(csv_path)
    if len(frames) == 0:
        print(f"[clip:render] warning: no data rows in {csv_path}")
        return None, None
    return int(frames[0]), int(frames[-1])


def _render(args):
    session_dir = args.session_dir
    if not session_dir or not os.path.isdir(session_dir):
        raise FileNotFoundError(f"clip render: --session-dir not found: {session_dir}")

    start, end = args.start, args.end
    if (start is None or end is None) and getattr(args, "bout_dir", None):
        inferred_start, inferred_end = _infer_range_from_bout(args.bout_dir)
        if inferred_start is not None:
            start, end = inferred_start, inferred_end
            print(f"[clip:render] inferred frame range {start}..{end} from {args.bout_dir}/fly0.csv")

    if start is None or end is None:
        raise ValueError(
            "clip render requires --start/--end, or --bout-dir with a readable fly0.csv")
    start, end = int(start), int(end)
    if end < start:
        raise ValueError(f"--end ({end}) must be >= --start ({start})")

    cameras = args.cameras or _discover_cameras(session_dir)
    if not cameras:
        raise RuntimeError(f"clip render: no cameras given and no Cam*.mp4 found in {session_dir}")

    avail = []
    for cam in cameras:
        video_path = os.path.join(session_dir, f"{cam}.mp4")
        if not os.path.isfile(video_path):
            print(f"[clip:render] warning: no session video for camera {cam} at {video_path}; skipping")
            continue
        avail.append(cam)
    if not avail:
        raise RuntimeError(f"clip render: none of the requested cameras have videos in {session_dir}")

    counts = {c: _probe_frame_count(os.path.join(session_dir, f"{c}.mp4")) for c in avail}
    max_possible = min(counts.values()) - start
    if max_possible <= 0:
        raise RuntimeError(
            f"clip render: start frame {start} is at/past the end of one or more camera "
            f"videos in {session_dir} (frame counts: {counts})")
    count = min(end - start + 1, max_possible)
    clip_end = start + count - 1

    fps = _fps_or(args, 30)
    out = args.out or os.path.join(session_dir, f"bout_{start}_{clip_end}_vstack.mp4")

    try:
        vio.write_video(
            out, _montage_stream(session_dir, avail, start, count, tag="clip:render"),
            fps=fps, fourcc="avc1")
    except ValueError as e:
        raise RuntimeError(
            f"clip render: no output frames for {out} -- every timestep in "
            f"{start}..{clip_end} had at least one unreadable camera frame "
            f"among {avail} in {session_dir}") from e
    print(f"[clip:render] wrote {out}")
    return 0


def run(args):
    mode = args.mode
    if mode == "cut":
        return _cut(args)
    if mode == "stack":
        return _stack(args)
    if mode == "render":
        return _render(args)
    raise ValueError(f"clip: unknown mode {mode!r} (expected cut|stack|render)")
