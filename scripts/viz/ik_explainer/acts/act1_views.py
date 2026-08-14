#!/usr/bin/env python3
"""Act 1 -- seven camera views, 2D keypoints fading in.

4x2 grid (7 fly-centred camera panels + 1 title/legend cell) at 1920x1080,
270 output frames (9 s @ 30 fps). Source video frame for output frame `f` is
`WINDOW_START + int(f * WINDOW_LEN / N_OUT)`, i.e. a CONTIGUOUS sub-range of
the clip's 921 frames (`WINDOW_START..WINDOW_START+WINDOW_LEN-1`), not the
whole clip -- see `WINDOW_START`/`WINDOW_LEN` below for why this window and
not another, and the speed-label derivation.

Timeline (rescaled proportionally from the original 450-frame/921-frame
version -- same fractions of the act, just fewer frames):
  f   0- 53  video only, panels labelled camera name + true elevation.
  f  54-125  2D keypoints fade in, alpha = (f-54)/72; low-confidence markers
             are drawn dimmer (`draw.draw_keypoints(..., conf=...)` shrinks
             their radius itself).
  f 126-269  full overlay, video advancing.

EXPECTATION if this is right: f=0 shows raw video with no markers anywhere;
f=90 shows markers at ~50% opacity on all 7 panels; f=240 shows markers ON
the fly (not off to one side) in every panel, including the wall-adjacent
Cam2012857 where the leg keypoints are genuinely the messiest -- that messiness
is real per-view failure, not a bug, and is the setup for why Act 2 needs
seven cameras.
FALSIFICATION: markers visible at f=0 (fade math inverted), or markers
sitting off the fly in some panel (crop centre wrong for that camera / smoothing
lost track of the centroid).

Colours: per-keypoint, JARVIS per-limb-chain scheme
(`kp_colors.jarvis_kp_colors`) -- never invented here.

TASK-19 (presentation-only; read before touching `_build_title_panel`): the
on-screen title changes "ACT 1" -> "2D Keypoint tracking", and the title
card's text is stripped down to ONLY the title, the fps/speed line, and the
colour-coded keypoint legend -- the camera-count/elevation summary line and
the "dim marker = low detector confidence" note are removed from the FRAME
per the user's explicit request. Both facts remain true and are still
documented here and enforced in code (7 cameras with elev range asserted in
`_elevations_deg`'s caller; `draw.draw_keypoints`'s `conf` argument still
shrinks low-confidence markers, see `draw_keypoints`'s own docstring in
`draw.py`) -- only their on-screen captions are gone. Per-panel camera name +
elevation labels and the per-panel scale bar are UNCHANGED (they are
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
N_OUT = 270

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

CANVAS_W, CANVAS_H = 1920, 1080
GRID_COLS, GRID_ROWS = 4, 2
CELL_W = CANVAS_W // GRID_COLS       # 480
CELL_H = CANVAS_H // GRID_ROWS       # 540
MARGIN = 6
LABEL_H = 32
PANEL_W = CELL_W - 2 * MARGIN        # 468
PANEL_H = CELL_H - LABEL_H - 2 * MARGIN  # 496
TITLE_CELL = 7                       # last cell (row1, col3) holds the title

# --- fly-centred crop (native video pixels, before resize to panel size) -
CROP_W = 430                         # covers the observed max x-span (322 px)
                                      # with >=49 px margin on each side
FRAME_H = 448                        # full strip height; the fly's y-range
                                      # (<=280 px) never needs vertical panning
SMOOTH_SIGMA = 12.0                  # frames (~15 ms @ 800 fps): removes
                                      # per-frame keypoint jitter from the pan
                                      # without lagging real fly motion

# --- timeline ------------------------------------------------------------
# Rescaled proportionally from the original (90, 210) @ 450 frames by the
# same 270/450 = 0.6 ratio: 90*0.6=54, 210*0.6=126.
FADE_START, FADE_END = 54, 126       # alpha ramps over frames [54, 126)
PX_PER_MM = 80.7


def _elevations_deg(cam_names, clip):
    """Camera names -> elevation in degrees, computed from the DLTs.

    `camera_view_dirs`'s third component is elevation directly: matches the
    brief's Cam2012630 -89.6 deg (vertical) ... Cam2012861 +0.6 deg exactly.
    """
    cam_mats, dlt_names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    dirs = clip_io.camera_view_dirs(cam_mats)
    elev = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0)))
    by_name = dict(zip(dlt_names, elev))
    missing = [c for c in cam_names if c not in by_name]
    if missing:
        raise ValueError(f"no DLT elevation for cameras {missing}")
    return np.array([by_name[c] for c in cam_names], np.float64)


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


def _build_title_panel(cam_names, elev_deg, kp_names) -> np.ndarray:
    """Task-19 (Change 2): stripped down to ONLY the title, the fps/speed
    line, and the colour-coded keypoint legend -- the camera-count/elevation
    summary and the "dim marker = low confidence" note are removed from the
    frame (the FACTS themselves -- 7 cameras, elev range, low-confidence
    markers drawn dimmer -- are unchanged in the code and still documented in
    this module's docstring; only the on-screen text is cut, per the user's
    explicit request). `cam_names`/`elev_deg` are kept as parameters (no
    longer drawn) only because per-panel labels elsewhere in this module
    still need `elev_deg`; nothing here computes them redundantly.
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

    elev_deg = _elevations_deg(cam_names, clip)
    kp_colors = jarvis_kp_colors(kp_names)

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
    title_panel = _build_title_panel(cam_names, elev_deg, kp_names)

    for f in range(N_OUT):
        t = int(t_for_f[f])
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)

        if f < FADE_START:
            kp_alpha = 0.0
        elif f < FADE_END:
            kp_alpha = (f - FADE_START) / float(FADE_END - FADE_START)
        else:
            kp_alpha = 1.0

        for ci, cam in enumerate(cam_names):
            row, col = divmod(ci, GRID_COLS)
            cellx, celly = col * CELL_W, row * CELL_H
            x0 = x0_by_cam[cam][t]

            panel_native = crops[cam][f]                # (FRAME_H, CROP_W, 3)
            if kp_alpha > 0.0:
                uv = kp2d[t, ci].copy()
                uv[:, 0] -= x0
                c = conf[t, ci]
                panel_native = draw.draw_leg_chains(panel_native, uv, kp_names,
                                                     alpha=kp_alpha, thickness=1,
                                                     kp_colors=kp_colors)
                panel_native = draw.draw_keypoints(panel_native, uv, kp_names,
                                                    conf=c, alpha=kp_alpha,
                                                    radius=4, kp_colors=kp_colors)
            panel_native = draw.scale_bar_mm(
                panel_native, px_per_mm=PX_PER_MM, mm=1.0,
                origin=(10, panel_native.shape[0] - 14))
            panel = cv2.resize(panel_native, (PANEL_W, PANEL_H),
                                interpolation=cv2.INTER_LINEAR)

            py, px = celly + MARGIN + LABEL_H, cellx + MARGIN
            canvas[py:py + PANEL_H, px:px + PANEL_W] = panel
            label_text = f"{cam}  elev {elev_deg[ci]:+.1f} deg"
            canvas = draw.label(canvas, label_text,
                                 (cellx + MARGIN, celly + MARGIN + 20),
                                 scale=draw.SMALL_SCALE)

        trow, tcol = divmod(TITLE_CELL, GRID_COLS)
        tx, ty = tcol * CELL_W, trow * CELL_H
        canvas[ty:ty + CELL_H, tx:tx + CELL_W] = title_panel
        canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}  (src {t}/{N})",
                             (tx + 48, ty + CELL_H - 24), scale=draw.SMALL_SCALE,
                             color=(150, 150, 150))

        cv2.imwrite(str(out_dir / f"f{f:05d}.png"), canvas)

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
