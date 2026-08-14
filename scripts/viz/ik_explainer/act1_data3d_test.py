#!/usr/bin/env python3
"""Act 1 VARIANT (diagnostic, throwaway) -- data3D.csv reprojected.

Standalone comparison video, NOT part of the shipped ik_explainer.mp4 and NOT
wired into assemble.py. Same seven-camera panel grid, crops, labels, scale
bars and JARVIS colours as `acts/act1_views.py`, but the overlaid keypoints
are the reprojection of the SHIPPED `<clip>/data3D.csv` (the vendor's own
triangulated 3D) instead of the detector 2D / this pipeline's filtered 3D.
The point is to let a human judge, by eye, how those points look on real
frames -- per CLAUDE.md, "a figure is how you notice it matches for the wrong
reason."

This file intentionally DUPLICATES a small amount of layout/plumbing code
from `acts/act1_views.py` (crop centring, title-panel scaffolding, speed-
factor math) rather than importing or refactoring it, per the task brief:
"leave act1_views.py alone entirely and duplicate the small amount you need
in the new script (duplication is acceptable here, since this is a throwaway
diagnostic and breaking the shipped act is not)." `clip_io`, `draw`, and
`kp_colors` ARE imported (shared infra, not the act itself).

UNITS/ORDER (get this exactly right -- CLAUDE.md records two prior bugs from
exactly this class of mismatch, both invisible to numeric QC alone):
  - data3D.csv is in DETECTOR order (== clip_io.detector_kp_names()) --
    ASSERTED below, not assumed.
  - data3D.csv is in 0.1 mm; `clip_io.load_shipped_kp3d_mm` applies the ONE
    x0.1 conversion used anywhere in this package -- reused here verbatim,
    not re-derived.
  - Everything downstream (02_kp2d.npz, 04_kp3d_filt.npz, the DLTs, the
    panel layout) is MODEL order; reordered via
    `clip_io.detector_to_model_index()` before reprojecting.

EXPECTATION: reprojected data3D.csv keypoints sit ON the fly in all seven
panels, at the same rough screen location as Act 1's raw/filtered overlays
(same crops, same window) -- median Antenna_Base<->Abd_tip distance after
unit conversion ~= 2.398 mm (the fly's own body length, a sanity check
independent of camera/order), and the great majority of reprojected points
land inside each camera's native 1936x448 frame.
FALSIFICATION: order assertion failing, body length far from ~2.4 mm (wrong
unit/order), or points landing off the fly / off-frame in any panel -- if
that happens, per the task brief, STOP and report BLOCKED rather than
rendering a video of wrong points.

Run (writes frames + encodes the mp4; also prints all verification numbers
used in the task-21 report so they can be reproduced on demand):
    python scripts/viz/ik_explainer/act1_data3d_test.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io, draw            # noqa: E402
from scripts.viz.ik_explainer.kp_colors import (               # noqa: E402
    jarvis_kp_colors, legend_entries,
)
from viz.core.colors import leg_chains                         # noqa: E402
from viz.core.io import write_video                             # noqa: E402

# --- layout (duplicated from acts/act1_views.py -- see module docstring) --
N_OUT = 330
WINDOW_START = 390
WINDOW_LEN = 370

CANVAS_W, CANVAS_H = 1920, 1080
GRID_COLS, GRID_ROWS = 4, 2
CELL_W = CANVAS_W // GRID_COLS
CELL_H = CANVAS_H // GRID_ROWS
MARGIN = 6
LABEL_H = 32
PANEL_W = CELL_W - 2 * MARGIN
PANEL_H = CELL_H - LABEL_H - 2 * MARGIN
TITLE_CELL = 7

CROP_W = 430
FRAME_H = 448
SMOOTH_SIGMA = 12.0

PX_PER_MM = 80.7
SRC_FPS = 800.0
OUT_FPS = 30.0

TITLE_LINE2 = "tracking"
TITLE_CAPTION = "-- data3D.csv reprojected"
EXPECTED_BODY_LENGTH_MM = 2.398
NATIVE_FRAME_W, NATIVE_FRAME_H = 1936, 448


def _elevations_deg(cam_names, clip):
    cam_mats, dlt_names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    dirs = clip_io.camera_view_dirs(cam_mats)
    elev = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0)))
    by_name = dict(zip(dlt_names, elev))
    missing = [c for c in cam_names if c not in by_name]
    if missing:
        raise ValueError(f"no DLT elevation for cameras {missing}")
    return np.array([by_name[c] for c in cam_names], np.float64)


def _smoothed_crop_x0(kp2d_cam: np.ndarray, frame_w: int) -> np.ndarray:
    cx = np.nanmean(kp2d_cam[..., 0], axis=1)
    if not np.all(np.isfinite(cx)):
        raise ValueError("a source frame has zero finite keypoints in a "
                          "camera; crop centring needs at least one")
    cx_smooth = gaussian_filter1d(cx, sigma=SMOOTH_SIGMA, mode="nearest")
    x0 = cx_smooth - CROP_W / 2.0
    return np.clip(x0, 0, frame_w - CROP_W)


def _preload_crops(clip: str, cam_names, x0_by_cam: dict, t_for_f: np.ndarray):
    """DISPLAY SOURCE: `clip_io.video_path` defaults to the brightness/
    contrast-lifted `<clip>/enhanced/` copy, matching `acts/act1_views.py`
    (presentation-only; the reprojected data3D.csv keypoints drawn on top
    are unaffected -- see clip_io.py's module docstring)."""
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


def _speed_factor() -> float:
    real_duration_s = WINDOW_LEN / SRC_FPS
    out_duration_s = N_OUT / OUT_FPS
    return out_duration_s / real_duration_s


def _draw_overlay(panel_native, uv, conf, x0, kp_names, kp_colors):
    """Reprojected data3D.csv keypoints -- leg chains + points, dimmed by the
    CSV's OWN per-keypoint confidence column (col 4 of each keypoint's 4),
    same visual treatment `draw.py` gives detector confidence."""
    pts = np.asarray(uv).copy()
    pts[:, 0] -= x0
    out = draw.draw_leg_chains(panel_native, pts, kp_names, alpha=1.0,
                                thickness=1, kp_colors=kp_colors)
    out = draw.draw_keypoints(out, pts, kp_names, conf=conf, alpha=1.0,
                               radius=4, kp_colors=kp_colors)
    return out


def _build_title_panel(kp_names) -> np.ndarray:
    img = np.zeros((CELL_H, CELL_W, 3), np.uint8)
    img = draw.stage_title(img, "2D Keypoint")
    img = draw.label(img, TITLE_LINE2, (48, 128), scale=draw.TITLE_SCALE,
                      color=(255, 255, 255), thickness=draw.TITLE_THICKNESS)
    img = draw.label(img, TITLE_CAPTION, (48, 168), scale=draw.CAPTION_SCALE)
    speed = _speed_factor()
    img = draw.label(img, f"800 fps -> 1/{speed:.1f} speed",
                      (48, 220), scale=draw.CAPTION_SCALE)
    y = 220
    for name, color in legend_entries(kp_names):
        y += 24
        img = draw.label(img, name, (64, y), scale=draw.SMALL_SCALE, color=color)
    return img


def _leg_jitter_px(uv_cam: np.ndarray, kp_names) -> float:
    """uv_cam: (T,K,2) reprojected/2D px for ONE camera, T = number of RENDERED
    frames (in output-frame order, i.e. exactly what gets drawn on screen).
    Returns the median magnitude of the second difference (frame-to-frame
    jitter) over all leg keypoints and all interior frames."""
    idx = sorted({i for chain in leg_chains(kp_names).values() for i in chain})
    p = uv_cam[:, idx, :]                      # (T, Kleg, 2)
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]           # (T-2, Kleg, 2)
    mag = np.linalg.norm(d2, axis=-1)
    return float(np.nanmedian(mag))


def render(clip: str = clip_io.CLIP_DEFAULT) -> Path:
    d = clip_io.out_dirs(clip)

    # --- load data3D.csv, verify order + units before touching anything else
    csv_path = str(Path(clip) / "data3D.csv")
    xyz_det_mm, conf_det, csv_names = clip_io.load_shipped_kp3d_mm(csv_path)
    det_names = clip_io.detector_kp_names()
    if csv_names != det_names:
        raise ValueError(
            "data3D.csv header order != clip_io.detector_kp_names() -- "
            "refusing to reproject with a possible keypoint-order mismatch "
            f"(csv: {csv_names[:5]}..., detector: {det_names[:5]}...)")
    print(f"[order check] data3D.csv header == detector_kp_names(): PASS "
          f"({len(csv_names)} keypoints)")

    idx_d2m = clip_io.detector_to_model_index()
    model_names = clip_io.model_kp_names()
    xyz_mm = xyz_det_mm[:, idx_d2m]             # (T,50,3) mm, MODEL order
    conf = conf_det[:, idx_d2m]                 # (T,50)   MODEL order
    T_csv = xyz_mm.shape[0]

    i_ant = model_names.index("Antenna_Base")
    i_abd = model_names.index("Abd_tip")
    body_len = np.linalg.norm(xyz_mm[:, i_ant] - xyz_mm[:, i_abd], axis=-1)
    median_body_len = float(np.nanmedian(body_len))
    print(f"[units check] median Antenna_Base<->Abd_tip = {median_body_len:.4f} mm "
          f"(expected ~= {EXPECTED_BODY_LENGTH_MM} mm)")
    if abs(median_body_len - EXPECTED_BODY_LENGTH_MM) > 0.05:
        raise ValueError(
            f"median body length {median_body_len:.4f} mm is far from the "
            f"expected ~{EXPECTED_BODY_LENGTH_MM} mm -- likely a unit or "
            "keypoint-order bug; refusing to render")

    if WINDOW_START + WINDOW_LEN > T_csv:
        raise ValueError(
            f"WINDOW_START+WINDOW_LEN ({WINDOW_START + WINDOW_LEN}) exceeds "
            f"data3D.csv's {T_csv} frames")

    # --- DLTs + camera order, matched against 02_kp2d.npz's panel order ----
    z = np.load(d["predictions"] / "02_kp2d.npz", allow_pickle=True)
    kp2d, conf_det2d = z["kp2d"], z["conf"]      # (N,C,K,2), (N,C,K) MODEL order
    cam_names = [str(c) for c in z["cam_names"]]
    kp_names = [str(n) for n in z["kp_names"]]
    if kp_names != model_names:
        raise ValueError("02_kp2d.npz kp_names != clip_io.model_kp_names() "
                          "-- keypoint order mismatch")
    N = kp2d.shape[0]

    cam_mats, dlt_names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    if list(dlt_names) != cam_names:
        raise ValueError(
            f"DLT camera order {list(dlt_names)} != 02_kp2d.npz cam_names "
            f"{cam_names}")
    if WINDOW_START + WINDOW_LEN > N:
        raise ValueError(
            f"WINDOW_START+WINDOW_LEN ({WINDOW_START + WINDOW_LEN}) exceeds "
            f"02_kp2d.npz's {N} frames (used for crop centring)")

    elev_deg = _elevations_deg(cam_names, clip)
    kp_colors = jarvis_kp_colors(kp_names)
    # DISPLAY ONLY (task-28): "Camera N" panel labels; every lookup above/
    # below (cam_names, DLT order) keeps using the real Cam20128xx strings.
    disp_name = clip_io.display_names(cam_names)

    t_for_f = np.array(
        [WINDOW_START + int(f * WINDOW_LEN / N_OUT) for f in range(N_OUT)],
        np.int64)
    assert np.all(np.diff(t_for_f) > 0), "expected a strictly increasing map"

    # reproject data3D.csv for every rendered output frame, all 7 cameras
    uv_csv = np.stack(
        [clip_io.project(cam_mats, xyz_mm[int(t)]) for t in t_for_f])  # (N_OUT,C,K,2)
    conf_csv_f = conf[t_for_f]                                          # (N_OUT,K)

    # --- in-frame sanity check (native 1936x448 video frame) ---------------
    print("[in-frame check] fraction of reprojected keypoints inside "
          f"{NATIVE_FRAME_W}x{NATIVE_FRAME_H} per camera, over the "
          f"{N_OUT}-frame rendered window:")
    for ci, cam in enumerate(cam_names):
        pts = uv_csv[:, ci]                     # (N_OUT,K,2)
        valid = np.all(np.isfinite(pts), axis=-1)
        inside = (valid & (pts[..., 0] >= 0) & (pts[..., 0] < NATIVE_FRAME_W)
                  & (pts[..., 1] >= 0) & (pts[..., 1] < NATIVE_FRAME_H))
        frac = float(inside.sum()) / float(pts.shape[0] * pts.shape[1])
        print(f"    {cam}: {frac:.4f}")

    # --- jitter comparison: data3D.csv reprojection vs 04_kp3d_filt.npz -----
    kp3d_z = np.load(d["predictions"] / "04_kp3d_filt.npz", allow_pickle=True)
    k3 = kp3d_z["kp3d"]
    kp3d_names = [str(n) for n in kp3d_z["kp_names"]]
    if kp3d_names != kp_names:
        raise ValueError("04_kp3d_filt.npz kp_names != 02_kp2d.npz kp_names")
    uv_filt = np.stack(
        [clip_io.project(cam_mats, k3[int(t)]) for t in t_for_f])       # (N_OUT,C,K,2)

    print("[jitter check] median leg-keypoint 2nd-difference jitter (px), "
          f"over the same {N_OUT} rendered frames, per camera:")
    for ci, cam in enumerate(cam_names):
        j_csv = _leg_jitter_px(uv_csv[:, ci], kp_names)
        j_filt = _leg_jitter_px(uv_filt[:, ci], kp_names)
        smoother = "04_kp3d_filt" if j_filt < j_csv else "data3D.csv"
        print(f"    {cam}: data3D.csv={j_csv:.3f}  04_kp3d_filt={j_filt:.3f}  "
              f"smoother={smoother}")

    # --- crops: SAME as Act 1 (centred on the detector 2D), for direct
    # frame-for-frame comparability between the two videos -------------------
    src_frame_w = 1936
    x0_by_cam = {cam: _smoothed_crop_x0(kp2d[:, ci], src_frame_w)
                 for ci, cam in enumerate(cam_names)}
    crops = _preload_crops(clip, cam_names, x0_by_cam, t_for_f)

    out_dir = d["frames"] / "act1_data3d_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    title_panel = _build_title_panel(kp_names)

    for f in range(N_OUT):
        t = int(t_for_f[f])
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)

        for ci, cam in enumerate(cam_names):
            row, col = divmod(ci, GRID_COLS)
            cellx, celly = col * CELL_W, row * CELL_H
            x0 = x0_by_cam[cam][t]

            panel_native = crops[cam][f]
            panel_native = _draw_overlay(
                panel_native, uv_csv[f, ci], conf_csv_f[f], x0, kp_names,
                kp_colors)
            panel_native = draw.scale_bar_mm(
                panel_native, px_per_mm=PX_PER_MM, mm=1.0,
                origin=(10, panel_native.shape[0] - 14))
            panel = cv2.resize(panel_native, (PANEL_W, PANEL_H),
                                interpolation=cv2.INTER_LINEAR)

            py, px = celly + MARGIN + LABEL_H, cellx + MARGIN
            canvas[py:py + PANEL_H, px:px + PANEL_W] = panel
            label_text = f"{disp_name[cam]}  elev {elev_deg[ci]:+.1f} deg"
            canvas = draw.label(canvas, label_text,
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

    out_path = Path(clip) / "ik_explainer" / "act1_data3d_test.mp4"

    def _frames_iter():
        for f in range(N_OUT):
            yield cv2.imread(str(out_dir / f"f{f:05d}.png"))

    write_video(str(out_path), _frames_iter(), fps=int(OUT_FPS), macro_block_size=1)
    print(f"wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    a = ap.parse_args()
    render(a.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
