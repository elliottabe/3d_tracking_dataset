#!/usr/bin/env python3
"""Act 1 -- seven camera views, 2D keypoints fading in, held on the full
raw-detector overlay to the end.

4x2 grid (7 fly-centred camera panels + 1 title/legend cell) at 1920x1080,
150 output frames (5 s @ 30 fps). Source video frame for output frame `f` is
`WINDOW_START + int(f * WINDOW_LEN / N_OUT)`, i.e. a CONTIGUOUS sub-range of
the clip's 921 frames (`WINDOW_START..WINDOW_START+WINDOW_LEN-1`), not the
whole clip -- see `WINDOW_START`/`WINDOW_LEN` below for why this window and
not another, and the speed-label derivation.

TASK-30 (user: remove Act 1's reprojection reveal): the user no longer wants
the closing raw-2D -> reprojected-3D crossfade added in TASK-20 (and kept,
unchanged in length, through TASK-29's shortening pass). It is deleted
outright -- the beat itself (frames 150-209), its caption
("reprojected 3D: one estimate, 7 views"), and the `04_kp3d_filt.npz` ->
`clip_io.project` reprojection code path that fed it (the `04_kp3d_filt.npz`
LOAD used by the reveal is gone; the file is unrelated to anything else in
this module -- the crop-centring window scan mentioned in `WINDOW_START`'s
comment below was a one-off, uncommitted script, not code that lives here).
`N_OUT` drops 210 -> 150: since the pre-reveal content already ran exactly
0-149 (unchanged by TASK-29), removing the reveal beat and truncating to
150 frames requires no rescaling of `FADE_START`/`FADE_END` -- the three
remaining phases (video-only / fade-in / raw-overlay hold) keep their
existing frame numbers, and the raw-overlay hold now runs to the act's new
last frame instead of handing off to the reveal. `WINDOW_START`/`WINDOW_LEN`
(the source-frame span) are untouched, so the same real footage is shown,
just over fewer output frames (see the speed-label derivation below for the
consequence of that).

TASK-33 (user: clean 0-180 deg arc labels, renumber cameras along the arc):
panel labels changed from "Camera N  elev <measured> deg" (numbered off
sorted-by-serial order, so both the number and the elevation value swept
non-monotonically as the arc passed its 90 deg mid-point -- e.g. Cam2012862 at
arc 120 deg reads elev -59.5 deg, the SAME elevation as Cam2012853 at arc
60 deg) to "Camera N  <arc> deg", where N and the arc angle both come from
`clip_io.display_names`/`clip_io.arc_order` (`clip_io.arc_positions_deg`
derives the arc position from the DLT optical axes -- never a hardcoded
per-camera table -- and asserts it lands within tolerance of a clean 30 deg
multiple). Panels are now placed in ARC order (0->180 deg, left-to-right,
top-to-bottom) instead of `cam_names`' sorted-by-serial order, so the grid
reads monotonically; every data lookup (kp2d/conf indexing, crop centring)
still keys off the real Cam20128xx name, never the display order.

Timeline:
  f   0- 35  video only, panels labelled by arc position ("Camera N  <arc> deg").
  f  36- 95  2D keypoints fade in, alpha = (f-36)/60; low-confidence markers
             are drawn dimmer (`draw.draw_keypoints(..., conf=...)` shrinks
             their radius itself).
  f  96-149  full raw-detector overlay, video advancing, held to the act's
             last frame (no reveal, no crossfade, no caption).

EXPECTATION if this is right: f=0 shows raw video with no markers anywhere;
f=66 (fade midpoint) shows markers at ~50% opacity on all 7 panels; f=120
shows markers ON the fly (not off to one side) in every panel, including the
wall-adjacent Cam2012857 where the leg keypoints are genuinely the messiest --
that messiness is real per-view failure, not a bug, and is the setup for why
Act 2 needs seven cameras. The act's LAST frame (f=149) should look like any
other frame in the 96-149 hold -- raw jittery overlay, no crossfade, no
on-screen caption -- since the reveal that used to occupy 150-209 no longer
exists.
FALSIFICATION: markers visible at f=0 (fade math inverted), markers sitting
off the fly in some panel (crop centre wrong for that camera / smoothing
lost track of the centroid), or any crossfade/caption/reprojected overlay
visible anywhere in the act (the reveal must be gone, not merely hidden).

Colours: per-keypoint, JARVIS per-limb-chain scheme
(`kp_colors.jarvis_kp_colors`) -- never invented here.

TASK-19 (presentation-only; read before touching `_build_title_panel`): the
on-screen title changes "ACT 1" -> "2D Keypoint tracking", and the title
card's text is stripped down to ONLY the title, the fps/speed line, and the
colour-coded keypoint legend -- the camera-count/elevation summary line and
the "dim marker = low detector confidence" note are removed from the FRAME
per the user's explicit request. Both facts remain true and are still
documented here and enforced in code (7 cameras asserted via
`assert C == len(cam_names) == 7` above; `draw.draw_keypoints`'s `conf`
argument still shrinks low-confidence markers, see `draw_keypoints`'s own
docstring in `draw.py`) -- only their on-screen captions are gone. Per-panel
camera labels (task-33: now arc position, not elevation -- see the TASK-33
section above) and the per-panel scale bar are UNCHANGED IN KIND (they are
per-panel annotations, not part of the removed title-card text block).
Typography now comes from `draw.py`'s shared `TITLE_SCALE`/`CAPTION_SCALE`/
`SMALL_SCALE` (one size per role across all four acts) instead of this
module's own ad hoc scale values.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io, draw   # noqa: E402
from scripts.viz.ik_explainer.kp_colors import (     # noqa: E402
    jarvis_kp_colors, legend_entries,
)

# --- layout ------------------------------------------------------------
N_OUT = 150   # TASK-30: was 210 -- the 60-frame reprojection reveal
              # (150-209) is deleted outright, not shortened; see module
              # docstring's TASK-30 section.

# --- Change 4 (task-14): a contiguous sub-range of the clip, not the whole
# 921 frames -- keeps the slow-motion factor (real-world seconds per output
# second) close to the original instead of speeding the fly up to cover the
# same span in fewer output frames. Chosen by scanning the Scutellum's own
# 3D trajectory (04_kp3d_filt.npz, script run by hand, not committed) for the
# window of length ~370 (~40% of 921) with the MOST walking activity: total
# path length and net displacement are both maximised (not just non-zero) at
# WINDOW_START=390 -- 10.09 mm of path length, 8.47 mm of net displacement
# over the window, i.e. genuine directed walking, not idling-in-place or a
# back-and-forth wobble that a path-length-only metric could reward. The
# window is real, walking fly footage, not the busiest-looking segment by
# chance.
WINDOW_START = 390
WINDOW_LEN = 370          # 921 * 0.40 = 368.4; 390+370=760 <= 921. ~40.2% of the clip.

# TASK-34: grid geometry (canvas/grid/cell/panel/title-cell dimensions) now
# lives in `draw.py` (`draw.panel_cell_rect` etc.) so Act 2's fly-out start
# pose can be defined as an exact inversion of THIS act's own cell rectangles
# instead of a second, separately-hardcoded copy that could silently drift.
# Values are unchanged -- this is a re-source, not a re-derivation.
CANVAS_W, CANVAS_H = draw.CANVAS_W, draw.CANVAS_H
GRID_COLS, GRID_ROWS = draw.GRID_COLS, draw.GRID_ROWS
CELL_W, CELL_H = draw.CELL_W, draw.CELL_H
MARGIN, LABEL_H = draw.GRID_MARGIN, draw.GRID_LABEL_H
PANEL_W, PANEL_H = draw.PANEL_W, draw.PANEL_H
TITLE_CELL = draw.TITLE_CELL

# --- fly-centred crop (native video pixels, before resize to panel size) -
CROP_W = 430                         # covers the observed max x-span (322 px)
                                      # with >=49 px margin on each side
FRAME_H = 448                        # full strip height; the fly's y-range
                                      # (<=280 px) never needs vertical panning
SMOOTH_SIGMA = 12.0                  # frames (~15 ms @ 800 fps): removes
                                      # per-frame keypoint jitter from the pan
                                      # without lagging real fly motion

# --- timeline ------------------------------------------------------------
# Unchanged by TASK-30 (the reveal it fed into is deleted, not this split):
# 0-35 video only / 36-95 keypoints fade in / 96-149 raw overlay, held to
# the act's new last frame.
FADE_START, FADE_END = 36, 96        # alpha ramps over frames [36, 96)
PX_PER_MM = 80.7


def _draw_raw_overlay(panel_native, uv_raw, conf_raw, x0, kp_names, kp_colors,
                       alpha):
    """Draw the raw detector 2D (leg chains + points, `conf`-shrunk) on a
    COPY of `panel_native`."""
    uv = np.asarray(uv_raw).copy()
    uv[:, 0] -= x0
    out = draw.draw_leg_chains(panel_native, uv, kp_names, alpha=alpha,
                                thickness=1, kp_colors=kp_colors)
    out = draw.draw_keypoints(out, uv, kp_names, conf=conf_raw, alpha=alpha,
                               radius=4, kp_colors=kp_colors)
    return out


def _smoothed_crop_x0(kp2d_cam: np.ndarray, frame_w: int) -> np.ndarray:
    """(N,K,2) for one camera -> (N,) smoothed left edge of the CROP_W window."""
    cx = np.nanmean(kp2d_cam[..., 0], axis=1)          # (N,)
    if not np.all(np.isfinite(cx)):
        raise ValueError("a source frame has zero finite keypoints in a "
                          "camera; crop centring needs at least one")
    cx_smooth = gaussian_filter1d(cx, sigma=SMOOTH_SIGMA, mode="nearest")
    x0 = cx_smooth - CROP_W / 2.0
    return np.clip(x0, 0, frame_w - CROP_W)


def _preload_crops(clip: str, cam_names, x0_by_cam: dict, t_for_f: np.ndarray):
    """Read each camera's video once; keep only the N_OUT wanted frames, cropped.

    Returns {cam: (N_OUT, FRAME_H, CROP_W, 3) uint8}, in output-frame order.

    DISPLAY SOURCE: `clip_io.video_path` defaults to the brightness/contrast
    -lifted `<clip>/enhanced/` copies (presentation-only -- the 2D keypoints
    drawn on top were detected against the RAW videos and are not
    re-derived here; see clip_io.py's module docstring).
    """
    t_to_f = {int(t): f for f, t in enumerate(t_for_f)}
    out = {}
    for cam in cam_names:
        path = clip_io.video_path(clip, cam)
        cap = cv2.VideoCapture(path)
        x0 = x0_by_cam[cam]
        buf = np.empty((N_OUT, FRAME_H, CROP_W, 3), np.uint8)
        t = 0
        while True:
            f = t_to_f.get(t)
            if f is None:
                if not cap.grab():
                    break
            else:
                ok, img = cap.read()
                if not ok:
                    raise IOError(f"{path}: video ended early at frame {t}")
                left = int(round(x0[t]))
                buf[f] = img[:FRAME_H, left:left + CROP_W]
            t += 1
            if t >= len(x0):
                break
        cap.release()
        out[cam] = buf
    return out


SRC_FPS = 800.0
OUT_FPS = 30.0


def _speed_factor() -> float:
    """800 fps -> 1/N slow-motion factor, recomputed for THIS window/frame
    count (task-14: never copy the old act's number). Real time elapsed by
    the WINDOW_LEN source frames at SRC_FPS, stretched over N_OUT output
    frames' worth of viewing time at OUT_FPS: factor = out_duration /
    real_duration."""
    real_duration_s = WINDOW_LEN / SRC_FPS
    out_duration_s = N_OUT / OUT_FPS
    return out_duration_s / real_duration_s


def _build_title_panel(kp_names) -> np.ndarray:
    """Task-19 (Change 2): stripped down to ONLY the title, the fps/speed
    line, and the colour-coded keypoint legend -- the camera-count/elevation
    summary and the "dim marker = low confidence" note are removed from the
    frame (the FACTS themselves -- 7 cameras, elev range, low-confidence
    markers drawn dimmer -- are unchanged in the code and still documented in
    this module's docstring; only the on-screen text is cut, per the user's
    explicit request). Task-33: no longer takes `cam_names`/`elev_deg` --
    per-panel labels now come from `clip_io.display_names`'s arc-derived
    "Camera N  <arc> deg" string, so this panel never needed raw elevation.
    """
    img = np.zeros((CELL_H, CELL_W, 3), np.uint8)
    # This title card is only CELL_W=480 px wide (one grid cell), unlike
    # Acts 2-4's full 1920-px-wide canvas -- at the shared `TITLE_SCALE`,
    # "2D Keypoint tracking" measures ~530 px on one line (checked with
    # cv2.getTextSize before picking this, not guessed) and would clip
    # against the cell edge. Wrapped across two lines, SAME `TITLE_SCALE`/
    # `TITLE_THICKNESS` as every other act's title (no size reduction), each
    # line comfortably under 480 px.
    img = draw.stage_title(img, "2D Keypoint")
    img = draw.label(img, "tracking", (48, 128), scale=draw.TITLE_SCALE,
                      color=(255, 255, 255), thickness=draw.TITLE_THICKNESS)
    speed = _speed_factor()
    img = draw.label(img, f"800 fps -> 1/{speed:.1f} speed",
                      (48, 200), scale=draw.CAPTION_SCALE)
    y = 200
    for name, color in legend_entries(kp_names):
        y += 24
        img = draw.label(img, name, (64, y), scale=draw.SMALL_SCALE, color=color)
    return img


def render_act1(clip: str = clip_io.CLIP_DEFAULT) -> Path:
    d = clip_io.out_dirs(clip)
    z = np.load(d["predictions"] / "02_kp2d.npz", allow_pickle=True)
    kp2d, conf = z["kp2d"], z["conf"]                  # (N,C,K,2), (N,C,K)
    cam_names = [str(c) for c in z["cam_names"]]
    kp_names = [str(n) for n in z["kp_names"]]
    N, C, K, _ = kp2d.shape
    assert C == len(cam_names) == 7

    kp_colors = jarvis_kp_colors(kp_names)
    # DISPLAY ONLY (task-33): "Camera N  <arc> deg" panel labels, numbered and
    # ORDERED along the rig's measured 180 deg arc (`clip_io.display_names` /
    # `clip_io.arc_order`) rather than sorted-by-serial order -- every lookup
    # below that indexes real data (kp2d/conf via `ci`, `x0_by_cam`/`crops`
    # via the real camera name) keeps using `cam_names`; `disp_name` and
    # `panel_order` are never used for indexing, only for what to draw where
    # and what text to show.
    disp_name = clip_io.display_names(cam_names, clip)
    panel_order = clip_io.arc_order(cam_names, clip)
    ci_by_name = {cam: ci for ci, cam in enumerate(cam_names)}

    if WINDOW_START + WINDOW_LEN > N:
        raise ValueError(
            f"WINDOW_START+WINDOW_LEN ({WINDOW_START + WINDOW_LEN}) exceeds "
            f"the clip's {N} frames")

    src_frame_w = 1936
    # Change 4 (task-14): a CONTIGUOUS sub-range of the clip
    # (WINDOW_START..WINDOW_START+WINDOW_LEN-1), not the whole 921 frames --
    # see WINDOW_START/WINDOW_LEN above.
    t_for_f = np.array(
        [WINDOW_START + int(f * WINDOW_LEN / N_OUT) for f in range(N_OUT)],
        np.int64)
    assert np.all(np.diff(t_for_f) > 0), "expected a strictly increasing map"

    x0_by_cam = {cam: _smoothed_crop_x0(kp2d[:, ci], src_frame_w)
                 for ci, cam in enumerate(cam_names)}
    crops = _preload_crops(clip, cam_names, x0_by_cam, t_for_f)

    out_dir = d["frames"] / "act1_views"
    out_dir.mkdir(parents=True, exist_ok=True)
    title_panel = _build_title_panel(kp_names)

    for f in range(N_OUT):
        t = int(t_for_f[f])
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)

        if f < FADE_START:
            kp_alpha = 0.0
        elif f < FADE_END:
            kp_alpha = (f - FADE_START) / float(FADE_END - FADE_START)
        else:
            kp_alpha = 1.0

        # Task-33: panels are placed in ARC order (0->180 deg left-to-right,
        # top-to-bottom), not `cam_names`' sorted-by-serial order -- `ci`
        # (the real index into kp2d/conf) is still looked up BY NAME so data
        # indexing never changes, only where each camera's panel is drawn.
        for panel_idx, cam in enumerate(panel_order):
            ci = ci_by_name[cam]
            row, col = divmod(panel_idx, GRID_COLS)
            cellx, celly = col * CELL_W, row * CELL_H
            x0 = x0_by_cam[cam][t]

            panel_native = crops[cam][f]                # (FRAME_H, CROP_W, 3)
            if kp_alpha > 0.0:
                panel_native = _draw_raw_overlay(
                    panel_native, kp2d[t, ci], conf[t, ci], x0, kp_names,
                    kp_colors, kp_alpha)
            panel_native = draw.scale_bar_mm(
                panel_native, px_per_mm=PX_PER_MM, mm=1.0,
                origin=(10, panel_native.shape[0] - 14))
            panel = cv2.resize(panel_native, (PANEL_W, PANEL_H),
                                interpolation=cv2.INTER_LINEAR)

            # TASK-34: the exact rectangle Act 2's fly-out start pose inverts
            # (`draw.panel_cell_rect`) -- was inline `celly+MARGIN+LABEL_H,
            # cellx+MARGIN`; identical arithmetic, now the single source.
            px, py, pw, ph = draw.panel_cell_rect(panel_idx)
            assert (pw, ph) == (PANEL_W, PANEL_H)
            canvas[py:py + ph, px:px + pw] = panel
            canvas = draw.label(canvas, disp_name[cam],
                                 (cellx + MARGIN, celly + MARGIN + 20),
                                 scale=draw.SMALL_SCALE)

        trow, tcol = divmod(TITLE_CELL, GRID_COLS)
        tx, ty = tcol * CELL_W, trow * CELL_H
        canvas[ty:ty + CELL_H, tx:tx + CELL_W] = title_panel
        canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}  (src {t}/{N})",
                             (tx + 48, ty + CELL_H - 24), scale=draw.SMALL_SCALE,
                             color=(150, 150, 150))

        # Task-32: PNG_COMPRESSION 1 (vs cv2's default 3) -- lossless, faster
        # zlib pass; decoded pixels are bit-identical (verified in the task-32
        # report). This is a build-speed change only.
        cv2.imwrite(str(out_dir / f"f{f:05d}.png"), canvas,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])

    print(f"wrote {N_OUT} frames to {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    a = ap.parse_args()
    render_act1(a.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
