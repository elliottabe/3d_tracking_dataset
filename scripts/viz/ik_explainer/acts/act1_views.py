#!/usr/bin/env python3
"""Act 1 -- seven camera views, 2D keypoints fading in.

4x2 grid (7 fly-centred camera panels + 1 title/legend cell) at 1920x1080,
450 output frames (15 s @ 30 fps). Source video frame for output frame `f` is
`int(f * N / 450)` with `N` = the clip's 921 frames, so the whole clip spans
the act.

Timeline:
  f   0- 89  video only, panels labelled camera name + true elevation.
  f  90-209  2D keypoints fade in, alpha = (f-90)/120; low-confidence markers
             are drawn dimmer (`draw.draw_keypoints(..., conf=...)` shrinks
             their radius itself).
  f 210-449  full overlay, video advancing.

EXPECTATION if this is right: f=0 shows raw video with no markers anywhere;
f=150 shows markers at ~50% opacity on all 7 panels; f=400 shows markers ON
the fly (not off to one side) in every panel, including the wall-adjacent
Cam2012857 where the leg keypoints are genuinely the messiest -- that messiness
is real per-view failure, not a bug, and is the setup for why Act 2 needs
seven cameras.
FALSIFICATION: markers visible at f=0 (fade math inverted), or markers
sitting off the fly in some panel (crop centre wrong for that camera / smoothing
lost track of the centroid).

Colours come from viz/core/colors.py via draw.py -- never invented here.
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
from viz.core.colors import PALETTE                  # noqa: E402

# --- layout ------------------------------------------------------------
N_OUT = 450
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
FADE_START, FADE_END = 90, 210       # alpha ramps over frames [90, 210)
PX_PER_MM = 80.7

_LEGEND_COLOR = {"head": PALETTE["head"], "thorax": PALETTE["thorax"],
                 "abdomen": PALETTE["tail"], "legs": PALETTE["fly0"]}


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
    """Read each camera's video once; keep only the 450 wanted frames, cropped.

    Returns {cam: (450, FRAME_H, CROP_W, 3) uint8}, in output-frame order.
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


def _build_title_panel(cam_names, elev_deg) -> np.ndarray:
    img = np.zeros((CELL_H, CELL_W, 3), np.uint8)
    img = draw.stage_title(img, "ACT 1", "Seven views, one fly")
    img = draw.label(img, "800 fps -> 1/27 speed", (48, 160), scale=0.6)
    img = draw.label(img, "keypoints:", (48, 200), scale=0.55)
    y = 200
    for name, color in _LEGEND_COLOR.items():
        y += 32
        img = draw.label(img, name, (64, y), scale=0.55, color=color)
    img = draw.label(img, "dim marker = low detector confidence",
                      (48, y + 44), scale=0.45, color=(160, 160, 160))
    lo, hi = float(np.min(elev_deg)), float(np.max(elev_deg))
    img = draw.label(
        img, f"{len(cam_names)} cameras, elev {lo:+.1f} to {hi:+.1f} deg",
        (48, y + 76), scale=0.45, color=(160, 160, 160))
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

    src_frame_w = 1936
    t_for_f = np.array([int(f * N / N_OUT) for f in range(N_OUT)], np.int64)
    assert np.all(np.diff(t_for_f) > 0), "expected a strictly increasing map"

    x0_by_cam = {cam: _smoothed_crop_x0(kp2d[:, ci], src_frame_w)
                 for ci, cam in enumerate(cam_names)}
    crops = _preload_crops(clip, cam_names, x0_by_cam, t_for_f)

    out_dir = d["frames"] / "act1_views"
    out_dir.mkdir(parents=True, exist_ok=True)
    title_panel = _build_title_panel(cam_names, elev_deg)

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
                                                     alpha=kp_alpha, thickness=1)
                panel_native = draw.draw_keypoints(panel_native, uv, kp_names,
                                                    conf=c, alpha=kp_alpha,
                                                    radius=4)
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
                                 scale=0.55)

        trow, tcol = divmod(TITLE_CELL, GRID_COLS)
        tx, ty = tcol * CELL_W, trow * CELL_H
        canvas[ty:ty + CELL_H, tx:tx + CELL_W] = title_panel
        canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}  (src {t}/{N})",
                             (tx + 48, ty + CELL_H - 24), scale=0.45,
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
