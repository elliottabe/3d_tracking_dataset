#!/usr/bin/env python3
"""Recovery clip: a real 2D detection failure, absorbed downstream.

Left  : Camera 3 footage, detector marker vs the reprojected filtered 3D.
Right : raw / filtered / IK traces for the same keypoint, with a playhead.

See docs/specs/2026-08-14-recovery-clip-design.md. The event
(`recovery_event.EVENT`) is measured, not chosen by eye: `T1R_TaTip` on
`Cam2012853` is simultaneously the bout's worst detector-vs-consensus
disagreement (frame 441, 162.6 px, confidence 0.49 -- see
`test_the_detector_really_fails_at_the_peak_frame`) and its worst raw-3D
acceleration spike (frame 443, 1.627 mm/frame^2).

The on-screen caveat is load-bearing: the other six cameras ALSO miss the raw
3D consensus at this instant -- this is NOT "one camera failed and six rescued
it". The recovery is triangulation absorbing part of the error, then the
temporal filter and IK's anatomical constraints absorbing the rest. Its px
range is COMPUTED at render time by `caveat_lines()` from `load_tracks`'s
`cam_disagree` array, like every other number in this clip; no px literal
appears in this file.

EXPECTATION (checked by reading rendered frames, not assumed):
- output frame 0: video only, no marker separation, playhead at the window's
  first frame (420).
- the output frames covering source frame 441: the JARVIS-coloured detector
  marker sits ~163 px off the fly's foot; the white reprojected marker stays
  on the foot; the confidence readout reads ~0.49; the right panel's raw
  trace is already turning while filtered/IK stay smooth.
- output frame 359: video only at source frame 464, playhead at the window's
  last frame.
FALSIFICATION: both markers landing together => wrong camera/keypoint/frame
mapping. All three traces spiking together => wrong arrays plotted. The
playhead not tracking the left panel's source frame => two independent frame
indices instead of one.

Frames written 1920x1080 PNG (`[cv2.IMWRITE_PNG_COMPRESSION, 1]`) to
`<clip>/ik_explainer/frames/recovery_clip/f%05d.png`, then encoded to
`<clip>/ik_explainer/recovery_clip.mp4` (H.264, yuv420p, faststart, 30 fps,
silent, 360 frames = 12.0 s) via `viz.core.io.write_video`, the repo's shared
encoder (also used by `assemble.py`). Every write lands under
`<clip>/ik_explainer/`; the raw clip inputs (calibration/, Cam*.mp4,
enhanced/, data3D_*.csv) are only ever read.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from scripts.viz.ik_explainer import clip_io, draw, kp_colors     # noqa: E402
from scripts.viz.ik_explainer import recovery_event as ev         # noqa: E402
from scripts.viz.ik_explainer import trace_panel as tp            # noqa: E402
from viz.core.colors import PALETTE                                 # noqa: E402
from viz.core.io import write_video                                # noqa: E402

N_OUT = 360
SRC_FPS = 800
OUT_FPS = 30
CANVAS_W, CANVAS_H = 1920, 1080
PANEL_W, PANEL_H = CANVAS_W // 2, CANVAS_H     # 960x1080, 2-up

# Native-resolution pixels-per-mm for this rig's cropped video strip -- the
# SAME constant the four explainer acts use (act1_views.py/act2_triangulate.py/
# act4_solve.py `PX_PER_MM`), reused rather than re-derived. Applied to the
# NATIVE crop (before it is resized up to panel size), exactly like those
# acts' own `draw.scale_bar_mm` calls -- the resize then carries the drawn
# bar's real-world meaning along with it, so no separate "effective" px/mm
# needs to be computed for the enlarged panel.
PX_PER_MM_NATIVE = 80.7

# Margin added, on every side, around the tight bounding box of BOTH marker
# tracks (detector 2D + reprojection of filtered 3D) over the whole window --
# a design constant, not a measured one, chosen so the foreleg fills the
# panel with breathing room. The crop itself (its centre and size) is
# computed from the event's own data, once, and reused for all 360 output
# frames -- "fixed crop size" per the brief, so the camera view never jitters.
CROP_PAD_FRAC = 0.25

TITLE = "2D detection fails, the fit doesn't"

# Trace colours.
#   raw      -- cyan, deliberately a LOCAL constant and NOT imported from
#               PALETTE["detector"]. The left panel already spends a
#               detector-ish blue on the actual detector marker (T1R_TaTip's
#               JARVIS chain colour), and tying this trace to the shared
#               "detector" entry would make one palette role mean two
#               different things across the two panels of the same frame.
#               Raw triangulation is a 3D stage, not the detector.
#   filtered -- white, matching the white reprojected marker on the left panel:
#               same pipeline stage, same colour.
#   ik       -- the repo-wide "fit" green from viz.core.colors, so this clip
#               reads with every other figure in the project. Listed LAST so
#               trace_panel dashes it (filtered and IK nearly coincide; the
#               dashing is what keeps both visible).
_RAW_BGR = (255, 255, 0)
_FILT_BGR = (255, 255, 255)
_IK_BGR = PALETTE["fit"]
_AXIS_LETTERS = ("x", "y", "z")


def caveat_lines(tracks, event=None):
    """The on-screen caveat, with its px range COMPUTED from the loaded arrays.

    Measured at the frame the caveat is on screen over -- the displayed peak
    `EVENT["peak_2d"]`, the frame whose detector failure the left panel shows
    -- across the six cameras OTHER than the one on the left panel. Each value
    is that camera's own 2D detection vs the reprojection of the raw
    triangulation, i.e. how far each camera sits from the consensus all seven
    produced.

    The claim this supports: the other cameras are not in clean agreement that
    a single bad camera was outvoted; every one of them is some distance off
    the raw consensus, so triangulation alone does not clean this up -- the
    temporal filter and IK do the remaining work. The number is deliberately
    NOT a literal in this file: it is the only figure the clip states, and the
    design doc's own quoted range was wrong (see
    docs/specs/2026-08-14-recovery-clip-design.md).
    """
    e = ev.EVENT if event is None else event
    cam_names = list(tracks["cam_names"])
    i = int(e["peak_2d"]) - int(e["t0"])
    others = [j for j, n in enumerate(cam_names) if n != e["cam"]]
    if len(others) != len(cam_names) - 1:
        raise RuntimeError(
            f"event camera {e['cam']} not found exactly once in {cam_names}")
    d = np.asarray(tracks["cam_disagree"], np.float64)[i, others]
    if not np.all(np.isfinite(d)):
        raise RuntimeError(
            f"non-finite per-camera disagreement at frame {e['peak_2d']}: {d}")
    lo, hi = float(d.min()), float(d.max())
    print(f"[recovery_clip] per-camera detector-vs-raw disagreement at f{e['peak_2d']}: "
          + ", ".join(f"{cam_names[j]}={float(v):.1f}"
                      for j, v in zip(others, d))
          + f"  (event cam {e['cam']}="
            f"{float(tracks['cam_disagree'][i, cam_names.index(e['cam'])]):.1f})")
    return (f"the other {len(others)} cameras miss the raw 3D by "
            f"{lo:.0f}-{hi:.0f} px here too --",
            "the recovery is triangulation + filter + IK, not an outvoted camera")


def _fixed_crop_box(det2d, rep2d, frame_w, frame_h, panel_w, panel_h,
                     pad_frac=CROP_PAD_FRAC):
    """One (x0, y0, w, h) native-pixel crop box covering BOTH marker tracks
    over the whole window, with `pad_frac` margin, matched to the panel's
    aspect ratio -- fixed for every output frame (never recentred per-frame),
    so the camera view cannot jitter."""
    pts = np.concatenate([np.asarray(det2d, np.float64),
                          np.asarray(rep2d, np.float64)], axis=0)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    x_span = max((x_max - x_min) * (1.0 + 2.0 * pad_frac), 1.0)
    y_span = max((y_max - y_min) * (1.0 + 2.0 * pad_frac), 1.0)
    aspect = panel_w / float(panel_h)          # w/h
    crop_h = max(y_span, x_span / aspect)
    crop_w = crop_h * aspect
    crop_h = min(int(round(crop_h)), frame_h)
    crop_w = min(int(round(crop_w)), frame_w)
    x0 = int(np.clip(round(cx - crop_w / 2.0), 0, frame_w - crop_w))
    y0 = int(np.clip(round(cy - crop_h / 2.0), 0, frame_h - crop_h))
    return x0, y0, crop_w, crop_h


def _src_frame_for_output(f: int, t0: int, t1: int, n_out: int = N_OUT) -> int:
    """Output frame `f` (0..n_out-1) -> source frame in [t0, t1)."""
    return t0 + f * (t1 - t0) // n_out


def _speed_factor(t0: int, t1: int, n_out: int = N_OUT,
                   src_fps: int = SRC_FPS, out_fps: int = OUT_FPS) -> float:
    return src_fps / out_fps / ((t1 - t0) / n_out)


def render_recovery_clip(clip: str = clip_io.CLIP_DEFAULT) -> Path:
    e = ev.EVENT
    t0, t1 = e["t0"], e["t1"]
    tracks = ev.load_tracks(clip, e["kp"], t0, t1, e["cam"])
    frames_idx = tracks["frames"]           # t0..t1-1

    src_for_f = np.array([_src_frame_for_output(f, t0, t1) for f in range(N_OUT)],
                         np.int64)
    if src_for_f[0] != t0 or src_for_f[-1] != t1 - 1 or np.any(np.diff(src_for_f) < 0):
        raise RuntimeError(
            f"source-frame mapping is broken: first={src_for_f[0]} (want {t0}), "
            f"last={src_for_f[-1]} (want {t1 - 1}), monotonic={np.all(np.diff(src_for_f) >= 0)}")

    speed = _speed_factor(t0, t1)
    speed_label = f"800 fps -> 1/{speed:.0f} speed"
    print(f"[recovery_clip] speed factor: {speed:.4f} -> '{speed_label}'")

    # --- acceleration annotation, computed here (not hardcoded) -----------
    ar, af, ai = (ev.accel(tracks[k]) for k in ("raw", "filt", "ik"))
    peak_idx = e["peak_3d"] - t0 - 1        # accel[j] is centred on frame t0+j+1
    ar_v, af_v, ai_v = float(ar[peak_idx]), float(af[peak_idx]), float(ai[peak_idx])
    ratio_f, ratio_i = ar_v / af_v, ar_v / ai_v
    annotation = (f"raw {ar_v:.3f} -> filtered {af_v:.3f} -> IK {ai_v:.3f} "
                  f"mm/frame^2  ({ratio_f:.0f}x / {ratio_i:.0f}x)")
    annotation2 = (f"(value AT frame {e['peak_3d']}; window-max accel "
                   f"{ar.max():.3f}/{af.max():.3f}/{ai.max():.3f} would understate it)")
    print(f"[recovery_clip] at-frame accel @ f{e['peak_3d']}: "
          f"raw={ar_v:.4f} filt={af_v:.4f} ik={ai_v:.4f}  "
          f"({ratio_f:.1f}x / {ratio_i:.1f}x)")
    print(f"[recovery_clip] window-max accel (NOT annotated -- would understate "
          f"the spike): raw={ar.max():.4f} filt={af.max():.4f} ik={ai.max():.4f}")

    axis = ev.worst_axis(tracks["raw"])
    axis_letter = _AXIS_LETTERS[axis]
    ylabel = f"{e['kp']}  {axis_letter} (mm)"
    series = [("raw", tracks["raw"][:, axis], _RAW_BGR),
             ("filtered", tracks["filt"][:, axis], _FILT_BGR),
             ("ik", tracks["ik"][:, axis], _IK_BGR)]      # ik last -> dashed

    caveat = caveat_lines(tracks, e)

    # --- left panel: fixed crop from both marker tracks -------------------
    disp = clip_io.display_names(tracks["cam_names"], clip)
    cam_label = disp[e["cam"]].split("  ")[0]              # "Camera 3  60 deg" -> "Camera 3"

    video_path = clip_io.video_path(clip, e["cam"])
    frames_native = clip_io.read_frames(video_path, frames_idx)   # (n_win,H,W,3)
    frame_h, frame_w = frames_native.shape[1:3]
    x0, y0, crop_w, crop_h = _fixed_crop_box(
        tracks["det2d"], tracks["rep2d"], frame_w, frame_h, PANEL_W, PANEL_H)
    print(f"[recovery_clip] fixed native crop: x0={x0} y0={y0} w={crop_w} h={crop_h} "
          f"(native {frame_w}x{frame_h})")

    kp_map = kp_colors.jarvis_kp_colors()
    det_color = kp_map[e["kp"]]

    dirs = clip_io.out_dirs(clip)
    out_dir = dirs["frames"] / "recovery_clip"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in range(N_OUT):
        src = int(src_for_f[f])
        i = src - t0

        native = frames_native[i]
        crop = native[y0:y0 + crop_h, x0:x0 + crop_w].copy()

        det_local = np.asarray([tracks["det2d"][i] - [x0, y0]])
        rep_local = np.asarray([tracks["rep2d"][i] - [x0, y0]])
        crop = draw.draw_keypoints(crop, det_local, [e["kp"]],
                                   conf=np.asarray([tracks["conf"][i]]),
                                   radius=5, kp_colors={e["kp"]: det_color})
        crop = draw.draw_keypoints(crop, rep_local, ["_filtered_reprojection"],
                                   radius=5,
                                   kp_colors={"_filtered_reprojection": (255, 255, 255)})
        crop = draw.scale_bar_mm(crop, px_per_mm=PX_PER_MM_NATIVE, mm=1.0,
                                 origin=(10, crop.shape[0] - 14))

        left = cv2.resize(crop, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        left = draw.label(left, f"{cam_label}   detector conf {tracks['conf'][i]:.2f}",
                          (24, 225), scale=draw.CAPTION_SCALE)

        right = tp.render_trace_panel(PANEL_W, PANEL_H, series, frames_idx, src,
                                      ylabel=ylabel, annotation=annotation)
        # y = PANEL_H - 96: clear of trace_panel's own x-axis tick-label row
        # (drawn at its plot bottom + 26 px, i.e. ~PANEL_H - 34) -- placing
        # this line there overlapped "420"/"441"/"464" (seen and fixed after
        # reading a rendered frame).
        right = draw.label(right, annotation2, (110, PANEL_H - 96),
                           scale=draw.SMALL_SCALE, color=(150, 150, 150))

        canvas = np.hstack([left, right])
        # `draw.stage_title`'s own subtitle styling (light grey, thickness 1)
        # is tuned for Acts 1-4, which only ever show it over black -- here
        # it sits over this clip's light-teal video and washes out
        # completely. Fixed LOCALLY (not in draw.stage_title, which Acts 1-4
        # already rely on and have been accepted at): no subtitle from
        # stage_title; the speed label is drawn separately, white, on its own
        # dark backing strip, which survives a bright background. Verified
        # by reading a rendered frame (not assumed) -- see task-3 fix report.
        canvas = draw.stage_title(canvas, TITLE)
        (sw, sh), _base = cv2.getTextSize(speed_label, cv2.FONT_HERSHEY_SIMPLEX,
                                          draw.CAPTION_SCALE, draw.CAPTION_THICKNESS)
        cv2.rectangle(canvas, (40, 122 - sh - 10), (56 + sw, 122 + 10),
                     (0, 0, 0), -1)
        canvas = draw.label(canvas, speed_label, (48, 122), scale=draw.CAPTION_SCALE,
                           color=(255, 255, 255))
        canvas = draw.label(canvas, caveat[0], (48, 158), scale=draw.CAPTION_SCALE)
        canvas = draw.label(canvas, caveat[1], (48, 185), scale=draw.CAPTION_SCALE)
        canvas = draw.label(canvas, f"output frame {f + 1}/{N_OUT}  (src {src})",
                           (48, CANVAS_H - 16), scale=draw.SMALL_SCALE,
                           color=(150, 150, 150))

        cv2.imwrite(str(out_dir / f"f{f:05d}.png"), canvas,
                   [cv2.IMWRITE_PNG_COMPRESSION, 1])

    written = sorted(out_dir.glob("f*.png"))
    if len(written) != N_OUT:
        raise RuntimeError(
            f"expected exactly {N_OUT} frames in {out_dir}, found {len(written)} "
            "-- refusing to encode a truncated/malformed render.")
    print(f"[recovery_clip] wrote {len(written)} frames to {out_dir}")

    mp4_path = dirs["root"] / "recovery_clip.mp4"

    def _frames_iter():
        for fp in written:
            img = cv2.imread(str(fp))
            if img is None:
                raise IOError(f"failed to read {fp}")
            yield img

    write_video(str(mp4_path), _frames_iter(), fps=OUT_FPS, macro_block_size=1)
    print(f"[recovery_clip] wrote {mp4_path}")
    return mp4_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    args = ap.parse_args()
    render_recovery_clip(args.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
