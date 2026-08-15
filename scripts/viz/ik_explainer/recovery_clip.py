#!/usr/bin/env python3
"""Recovery clip: a real 2D detection failure, absorbed downstream.

Left  : a stack of three camera views of the same instant, each with the
        detector marker vs the reprojected filtered 3D. Top is the event
        camera; the two below are chosen by measurement
        (`recovery_event.select_panel_cams`) as the worst and the cleanest of
        the remaining six, so the stack brackets the on-screen caveat instead
        of illustrating one half of it. Each has its own crop and therefore
        its own 1 mm scale bar.
Right : raw / filtered / IK traces for the same keypoint, with a playhead.

See docs/specs/2026-08-14-recovery-clip-design.md. The event
(`recovery_event.EVENT`) is measured, not chosen by eye: `T1R_TaTip` on
`Cam2012853` is simultaneously the bout's worst detector-vs-consensus
disagreement -- frame 441, **122.2 px** from the RAW triangulation (the
quantity the design doc measured and ranked the bout by) and **162.6 px** from
the reprojected FILTERED 3D (the marker separation this clip's left panel
actually draws, and what
`test_the_detector_really_fails_at_the_peak_frame` asserts); detector
confidence 0.49 -- and its worst raw-3D acceleration spike (frame 443,
1.627 mm/frame^2). Two different comparisons, so two different numbers.

The on-screen caveat is load-bearing: the other six cameras ALSO miss the raw
3D consensus at this instant -- this is NOT "one camera failed and six rescued
it". The recovery is triangulation absorbing part of the error, then the
temporal filter and IK's anatomical constraints absorbing the rest. Its px
range is COMPUTED at render time by `caveat_lines()` from `load_tracks`'s
`cam_disagree` array, like every other number in this clip; no px literal
appears in this file.

EXPECTATION (checked by reading rendered frames, not assumed):
- output frame 0: all three camera panels show both markers coincident, no
  separation anywhere, playhead at the window's first frame (420).
- the output frames covering source frame 441: in the TOP panel the
  JARVIS-coloured detector marker sits ~163 px off the fly's foot while the
  white reprojected marker stays on it, and its readout says ~0.49; the middle
  (worst-of-the-rest) panel shows a visible but smaller separation; the bottom
  (cleanest) panel shows the two markers still essentially together. That
  gradient IS the caveat: the failure is shared but uneven, so no single
  camera was simply outvoted. The right panel's raw trace is already turning
  while filtered/IK stay smooth.
- output frame 359: video only at source frame 464, playhead at the window's
  last frame.
FALSIFICATION specific to the stack: all three panels showing the SAME
separation would mean one camera's tracks were drawn three times; the bottom
panel separating as badly as the top would contradict the measured selection.
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

# Extra camera views stacked under the event camera in the left panel. The
# event camera alone shows THAT the detector failed; it cannot show that the
# failure is partly shared, which is exactly what the on-screen caveat claims.
# The extras are chosen by measurement (`recovery_event.select_panel_cams`):
# the worst remaining view and the cleanest remaining view, so the stack
# brackets the claim instead of illustrating one half of it.
N_EXTRA_CAMS = 2
SUB_H = PANEL_H // (1 + N_EXTRA_CAMS)          # 360 px per camera sub-panel

# Native-resolution pixels-per-mm for this rig's cropped video strip -- the
# SAME constant the explainer acts use (`PX_PER_MM` in acts/act1_views.py and
# acts/act2_triangulate.py), reused rather than re-derived.
#
# Unlike those acts, this clip draws its scale bar AFTER the crop is resized up
# to panel size, so that the "1 mm" text lands at SMALL_SCALE in final pixels
# instead of being magnified along with the crop. That means the bar needs the
# effective px/mm of the enlarged panel, not this native value -- see
# `px_per_mm_panel` below, which scales this constant by the resize factor.
PX_PER_MM_NATIVE = 80.7

# Margin added, on every side, around the tight bounding box of BOTH marker
# tracks (detector 2D + reprojection of filtered 3D) over the whole window --
# a design constant, not a measured one, chosen so the foreleg fills the
# panel with breathing room. The crop itself (its centre and size) is
# computed from the event's own data, once, and reused for all 360 output
# frames -- "fixed crop size" per the brief, so the camera view never jitters.
CROP_PAD_FRAC = 0.25

# Radius of both marker dots, in NATIVE crop pixels (they are drawn before the
# ~3x upscale, so ~15 px on screen), and of the legend's colour samples, which
# are drawn on the already-upscaled panel and so are sized in FINAL pixels.
MARKER_RADIUS = 5
LEGEND_DOT_RADIUS = 7

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


def _build_cam_panels(clip, cam_names, tracks, frames_idx, display):
    """Everything each camera sub-panel needs, computed once (not per frame).

    Every panel is centred on its OWN markers -- the fly projects to a
    different place in each view -- but they all share ONE crop size, so all
    three are at the same zoom and their marker separations can be compared by
    eye. See `_shared_crop_size`.
    """
    idx = [tracks["cam_names"].index(n) for n in cam_names]
    dets = [np.asarray(tracks["det2d_all"][:, j], np.float64) for j in idx]
    reps = [np.asarray(tracks["rep2d_all"][:, j], np.float64) for j in idx]
    videos = [clip_io.read_frames(clip_io.video_path(clip, n), frames_idx)
              for n in cam_names]
    shapes = {v.shape[1:3] for v in videos}
    if len(shapes) != 1:
        raise RuntimeError(
            f"cameras differ in native frame size {shapes} -- a shared crop "
            "size would not mean a shared zoom")
    fh, fw = videos[0].shape[1:3]

    extents = [_crop_extent(d, r, PANEL_W, SUB_H) for d, r in zip(dets, reps)]
    cw, ch = _shared_crop_size(extents, fw, fh)
    sx, sy = PANEL_W / float(cw), SUB_H / float(ch)
    if abs(sx - sy) / max(sx, sy) > 0.01:
        raise RuntimeError(
            f"crop resize is not uniform (sx={sx:.4f}, sy={sy:.4f}) -- the "
            "1 mm scale bar would be wrong in one axis")
    px_per_mm = PX_PER_MM_NATIVE * sx
    print(f"[recovery_clip] shared crop {cw}x{ch} (native {fw}x{fh}) -> "
          f"sx={sx:.3f} sy={sy:.3f}, {px_per_mm:.1f} px/mm in every panel")

    panels = []
    for n, j, d, r, (cx, cy, _w, _h), vid in zip(cam_names, idx, dets, reps,
                                                 extents, videos):
        x0, y0 = _crop_origin(cx, cy, cw, ch, fw, fh)
        print(f"[recovery_clip]   {n} ({display[n]}): crop origin x0={x0} y0={y0}")
        panels.append({"name": n, "display": display[n], "frames": vid,
                       "det": d, "rep": r,
                       "conf": np.asarray(tracks["conf_all"][:, j]),
                       "x0": x0, "y0": y0, "cw": cw, "ch": ch,
                       "sx": sx, "sy": sy, "px_per_mm": px_per_mm})
    return panels


def _assert_text_clear_of_markers(cp, boxes, frames_idx):
    """Fail the render if any text box would cover either marker, in any frame.

    A legend that hides the marker it names is worse than no legend (learned by
    reading a rendered frame during the F3 fix). With three sub-panels at three
    different zooms the safe placements are no longer obvious by inspection, so
    every panel's boxes are checked against that panel's own tracks.
    """
    for track_name in ("det", "rep"):
        p = (cp[track_name] - [cp["x0"], cp["y0"]]) * [cp["sx"], cp["sy"]]
        r = MARKER_RADIUS * cp["sx"]
        for what, (bx0, by0, bx1, by1) in boxes:
            inside = ((p[:, 0] > bx0 - r) & (p[:, 0] < bx1 + r)
                      & (p[:, 1] > by0 - r) & (p[:, 1] < by1 + r))
            if inside.any():
                raise RuntimeError(
                    f"{cp['name']} ({cp['display']}): the {what} box "
                    f"{(bx0, by0, bx1, by1)} would cover the {track_name} marker "
                    f"at source frames {np.asarray(frames_idx)[inside].tolist()} "
                    "-- move it")


def _text_box(text, xy, *, scale=None, pad=8, dot=False):
    """The rectangle `_label_on_strip` will blacken for `text` at `xy`."""
    scale = draw.CAPTION_SCALE if scale is None else scale
    (tw, th), _b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                   draw.CAPTION_THICKNESS)
    x, y = int(xy[0]), int(xy[1])
    dot_w = 2 * LEGEND_DOT_RADIUS + 10 if dot else 0
    return (x - pad, y - th - pad, x + dot_w + tw + pad, y + pad)


def _cam_text(cp, conf):
    """Per-panel readout: real display name, arc angle, and THIS camera's own
    detector confidence -- so a low-confidence view is identifiable as such
    rather than looking like an unexplained miss."""
    return f"{cp['display']}   detector conf {conf:.2f}"


def _right_aligned_x(text, right_edge, *, scale=None, pad=8):
    scale = draw.CAPTION_SCALE if scale is None else scale
    (tw, _th), _b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                    draw.CAPTION_THICKNESS)
    return right_edge - pad - tw


def _label_on_strip(img, text, xy, *, scale=None, color=(255, 255, 255),
                    pad=8, dot_color=None):
    """`draw.label` on a solid black backing strip, optionally with a marker dot.

    Every caption in this clip is thin (CAPTION_THICKNESS 1) text over the
    left panel's light-teal video, where it washes out completely -- the
    problem commit 4a82231 fixed for the speed label by putting a black strip
    behind it. That rationale applies verbatim to the caveat, the
    camera/confidence readout and the marker legend, so they all use this one
    helper. Fixed HERE rather than in `draw.label`, which Acts 1-4 (delivered,
    accepted, only ever drawn over black) depend on the current appearance of.

    `dot_color`: draws a filled circle in the marker's own colour just left of
    the text, so a legend entry shows the marker and not only a colour of text.
    """
    scale = draw.CAPTION_SCALE if scale is None else scale
    (tw, th), _base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                      draw.CAPTION_THICKNESS)
    x, y = int(xy[0]), int(xy[1])
    dot_w = 2 * LEGEND_DOT_RADIUS + 10 if dot_color is not None else 0
    out = np.asarray(img).copy()
    cv2.rectangle(out, (x - pad, y - th - pad), (x + dot_w + tw + pad, y + pad),
                  (0, 0, 0), -1)
    if dot_color is not None:
        cv2.circle(out, (x + LEGEND_DOT_RADIUS, y - th // 2), LEGEND_DOT_RADIUS,
                   tuple(int(c) for c in dot_color), -1, cv2.LINE_AA)
    return draw.label(out, text, (x + dot_w, y), scale=scale, color=color)


def _crop_extent(det2d, rep2d, panel_w, panel_h, pad_frac=CROP_PAD_FRAC):
    """(cx, cy, crop_w, crop_h) covering BOTH marker tracks over the whole
    window, with `pad_frac` margin, matched to the panel's aspect ratio.

    Unclamped and unrounded: `_crop_origin` places it in a real frame. Split
    from the placement so several cameras can share ONE crop size (see
    `_shared_crop_size`) while each centres on its own markers.
    """
    pts = np.concatenate([np.asarray(det2d, np.float64),
                          np.asarray(rep2d, np.float64)], axis=0)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    x_span = max((x_max - x_min) * (1.0 + 2.0 * pad_frac), 1.0)
    y_span = max((y_max - y_min) * (1.0 + 2.0 * pad_frac), 1.0)
    aspect = panel_w / float(panel_h)          # w/h
    crop_h = max(y_span, x_span / aspect)
    return cx, cy, crop_h * aspect, crop_h


def _shared_crop_size(extents, frame_w, frame_h):
    """One (crop_w, crop_h) big enough for every camera's extent.

    All camera panels MUST share a crop size, because they share a native
    px/mm: that is what makes the marker separations visually comparable
    between panels. Sizing each panel to its own markers instead zoomed the
    cleanest camera to 4.0x against the event camera's 1.0x, which would have
    rendered its ~20 px miss LARGER on screen than the event camera's 122 px
    one -- a figure that inverts the very comparison it exists to make.
    """
    crop_w = min(int(round(max(w for _cx, _cy, w, _h in extents))), frame_w)
    crop_h = min(int(round(max(h for _cx, _cy, _w, h in extents))), frame_h)
    return crop_w, crop_h


def _crop_origin(cx, cy, crop_w, crop_h, frame_w, frame_h):
    """Top-left of a `crop_w x crop_h` box centred on (cx, cy), clamped into
    the frame -- fixed for every output frame (never recentred per-frame), so
    the camera view cannot jitter."""
    x0 = int(np.clip(round(cx - crop_w / 2.0), 0, frame_w - crop_w))
    y0 = int(np.clip(round(cy - crop_h / 2.0), 0, frame_h - crop_h))
    return x0, y0


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

    # --- left panel: a stack of camera views, each with its own crop --------
    disp = clip_io.display_names(tracks["cam_names"], clip)
    panel_cams = ev.select_panel_cams(tracks, e, n_extra=N_EXTRA_CAMS)
    peak_by_cam = np.nanmax(np.asarray(tracks["cam_disagree"], np.float64), axis=0)
    print("[recovery_clip] camera panels (event camera first, then worst and "
          "cleanest of the rest, by window-peak disagreement):")
    for n in panel_cams:
        print(f"    {n}  {disp[n]:22s} peak {peak_by_cam[tracks['cam_names'].index(n)]:6.1f} px")
    panels = _build_cam_panels(clip, panel_cams, tracks, frames_idx, disp)

    kp_map = kp_colors.jarvis_kp_colors()
    det_color = kp_map[e["kp"]]
    # Real names, not "blue dot"/"white dot" -- and the keypoint's real name,
    # so the legend says WHICH keypoint the whole clip is about.
    legend_entries = ((f"detector 2D ({e['kp']})", det_color),
                      ("reprojected 3D (filtered)", _FILT_BGR))
    # Text is spread across the stack rather than piled onto the top panel:
    # title + speed on panel 0, the marker legend on panel 0's bottom gutter
    # (it must be read before the failure is interpreted), and the caveat under
    # panel 2 -- it is a claim ABOUT the whole stack, so it belongs beneath it.
    # Camera labels go top-RIGHT so they never collide with the title, and the
    # scale bars bottom-RIGHT so they never collide with the legend or caveat.
    legend_xy = [(24, SUB_H - 60 + 34 * li) for li in range(len(legend_entries))]
    # The caveat is ~700 px wide, so unlike the legend it cannot dodge the
    # markers horizontally -- every panel's markers sit in x 403..557, dead
    # centre. It has to go BELOW them, and only the bottom panel has room:
    # its markers stop at y~262 (the cleanest camera moves least, which is
    # why it is the one that can carry the caveat). The frame counter was
    # moved to the bottom-right for the same reason; at - 96 and - 68 this
    # collided with the markers and the counter respectively, both caught by
    # `_assert_text_clear_of_markers` and by reading a rendered frame.
    caveat_xy = [(24, SUB_H - 60 + 34 * li) for li in range(len(caveat))]
    # Camera labels sit bottom-right, just above each scale bar, NOT top-right:
    # the title is 1.6-scale and runs to x~900, so a top-right label on the
    # first panel collided with it (again, seen in a rendered frame). Bottom
    # right is the one corner no other element claims -- the legend and caveat
    # are both bottom-LEFT.
    # ...except on the LAST panel, whose bottom edge is the caveat's. There the
    # label goes top-right instead, which is free because that panel is the
    # cleanest camera and its markers never rise above y~203.
    def cam_label_y(pi):
        return 40 if pi == len(panel_cams) - 1 else SUB_H - 56

    # Every text box, per panel, checked against that panel's own marker tracks.
    for pi, cp in enumerate(panels):
        boxes = []
        cam_text = _cam_text(cp, 0.0)
        boxes.append(("camera label",
                      _text_box(cam_text,
                                (_right_aligned_x(cam_text, PANEL_W), cam_label_y(pi)))))
        boxes.append(("scale bar", (PANEL_W - 40 - int(cp["px_per_mm"]) - 8,
                                    SUB_H - 24 - 24, PANEL_W - 32, SUB_H - 16)))
        if pi == 0:
            for (text, _c), xy in zip(legend_entries, legend_xy):
                boxes.append(("legend", _text_box(text, xy, dot=True)))
        if pi == len(panels) - 1:
            for text, xy in zip(caveat, caveat_xy):
                boxes.append(("caveat", _text_box(text, xy)))
        _assert_text_clear_of_markers(cp, boxes, frames_idx)

    dirs = clip_io.out_dirs(clip)
    out_dir = dirs["frames"] / "recovery_clip"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in range(N_OUT):
        src = int(src_for_f[f])
        i = src - t0

        subs = []
        for pi, cp in enumerate(panels):
            crop = cp["frames"][i][cp["y0"]:cp["y0"] + cp["ch"],
                                   cp["x0"]:cp["x0"] + cp["cw"]].copy()
            det_local = np.asarray([cp["det"][i] - [cp["x0"], cp["y0"]]])
            rep_local = np.asarray([cp["rep"][i] - [cp["x0"], cp["y0"]]])
            crop = draw.draw_keypoints(crop, det_local, [e["kp"]],
                                       conf=np.asarray([cp["conf"][i]]),
                                       radius=MARKER_RADIUS,
                                       kp_colors={e["kp"]: det_color})
            crop = draw.draw_keypoints(crop, rep_local, ["_filtered_reprojection"],
                                       radius=MARKER_RADIUS,
                                       kp_colors={"_filtered_reprojection": _FILT_BGR})
            sub = cv2.resize(crop, (PANEL_W, SUB_H), interpolation=cv2.INTER_LINEAR)
            # Scale bar drawn AFTER the upscale, at this panel's own px/mm.
            # Drawing it on the native crop (the Acts 1-2 convention) kept the
            # bar's real-world length correct through the resize, but carried
            # its "1 mm" text along too: SMALL_SCALE 0.5 x ~3 upscale = an
            # effective ~1.5 against TITLE_SCALE 1.6, i.e. the second-largest
            # text on screen. Here the bar length is pre-scaled instead, so the
            # label lands at SMALL_SCALE in FINAL pixels and the bar still
            # measures 1 real mm -- in EACH panel's own zoom, which differ.
            sub = draw.scale_bar_mm(
                sub, px_per_mm=cp["px_per_mm"], mm=1.0,
                origin=(PANEL_W - 40 - int(cp["px_per_mm"]), SUB_H - 24))
            cam_text = _cam_text(cp, cp["conf"][i])
            sub = _label_on_strip(
                sub, cam_text, (_right_aligned_x(cam_text, PANEL_W), cam_label_y(pi)))
            if pi == 0:
                # Marker legend (F3). Without it the panel is genuinely
                # ambiguous, and the naive reading is BACKWARDS: at the peak
                # the true tarsal tip is tucked at the fly's face while the
                # detector's error lands on a visually obvious extended leg,
                # so an uninformed viewer reads the detector marker as the
                # correct one. Drawn once, on the top panel, since all three
                # panels use the same two marker colours.
                for (text, col), xy in zip(legend_entries, legend_xy):
                    sub = _label_on_strip(sub, text, xy, color=col, dot_color=col)
            if pi == len(panels) - 1:
                for text, xy in zip(caveat, caveat_xy):
                    sub = _label_on_strip(sub, text, xy)
            # 1 px rule so three video panels don't read as one image.
            if pi:
                sub[0, :] = (60, 60, 60)
            subs.append(sub)
        left = np.vstack(subs)

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
        canvas = _label_on_strip(canvas, speed_label, (48, 122))
        # The caveat is drawn on the BOTTOM camera sub-panel, not here: it is a
        # claim about the whole camera stack ("the other cameras miss it too"),
        # so it reads correctly only underneath the stack it describes.
        # Bottom-RIGHT, under the trace panel: the bottom-left of the canvas is
        # now the caveat's, and the caveat is the load-bearing text of the two.
        counter = f"output frame {f + 1}/{N_OUT}  (src {src})"
        canvas = draw.label(canvas, counter,
                           (_right_aligned_x(counter, CANVAS_W,
                                             scale=draw.SMALL_SCALE) - 16, 28),
                           scale=draw.SMALL_SCALE, color=(150, 150, 150))

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
