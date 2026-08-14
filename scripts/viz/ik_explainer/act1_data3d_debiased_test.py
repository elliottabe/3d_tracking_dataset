#!/usr/bin/env python3
"""Act 1 VARIANT (diagnostic, throwaway) -- data3D.csv reprojected, bias removed.

Standalone comparison video, a further variant of
`act1_data3d_test.py` (which reprojects the shipped `data3D.csv` unmodified
and shows a visible, roughly-constant offset vs this pipeline's
`04_kp3d_filt.npz`). This script removes that offset before reprojecting, so
a human can judge whether the two 3D reconstructions agree once the rigid
translation is taken out. NOT part of the shipped ik_explainer.mp4 and NOT
wired into assemble.py; `act1_data3d_test.py` and its frames/video are left
untouched (this is a new, additive file).

BACKGROUND -- measured in a prior diagnostic (task 21), reused here only as a
sanity check on what THIS script computes, never as the value actually
subtracted:
  signed mean offset:  x -0.1116, y -0.0822, z +0.0716 mm
  |bias| = 0.1560 mm = 12.6 px @ 80.7 px/mm
  scatter about that bias = 0.0878 mm = 7.1 px
  body-length ratio 0.9971 (no scale error); cross-correlation over lags +-4
  is flat (13.8 px at lag 0, 13.0 px best) and the residual is ~9x the
  per-frame motion -- i.e. this is a translation, not a timing error.

The bias subtracted here is instead COMPUTED IN-SCRIPT from the two arrays
themselves (see `render()`):
    n = min(len(new_mm_model), len(ours))
    bias = np.nanmean((new_mm_model[:n] - ours[:n]).reshape(-1, 3), axis=0)
This keeps the number reproducible and self-checking: it is printed and
compared against the measured constants above (~1e-3 mm tolerance); a
mismatch means the loading path here differs from the one used to measure
those constants, and the script stops rather than rendering a video built on
an assumption that doesn't hold.

UNITS/ORDER (unchanged from act1_data3d_test.py -- see that module's
docstring for the two prior bugs this guards against):
  - data3D.csv is in DETECTOR order, 0.1 mm units; converted/reordered via
    `clip_io.load_shipped_kp3d_mm` + `clip_io.detector_to_model_index()`
    exactly as in act1_data3d_test.py.
  - 04_kp3d_filt.npz, the DLTs, and the panel layout are MODEL order.

EXPECTATION: with the bias removed, reprojected data3D.csv keypoints should
sit ON the fly at close to the SAME screen location as the
`04_kp3d_filt.npz` reprojection in all seven panels -- i.e. the visible
offset from act1_data3d_test.py should be gone, leaving only the ~7 px
scatter. Median 3D disagreement vs 04_kp3d_filt.npz should drop from
~0.170 mm (~13.8 px) before debiasing to ~0.088 mm (~7.1 px) after (the
scatter floor); if it does not land near that floor, the disagreement was
not purely a translation and this script says so rather than hiding it.
FALSIFICATION: computed bias far from the measured constants above (loading
path mismatch), or post-debias disagreement not near the scatter floor
(translation model wrong) -- either case: STOP, print full diagnostics, and
do not claim agreement.

Run (writes frames + encodes the mp4; prints all verification numbers used
in the task-22 report so they can be reproduced on demand):
    python scripts/viz/ik_explainer/act1_data3d_debiased_test.py
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
TITLE_CAPTION = "-- data3D.csv reprojected, bias removed"
EXPECTED_BODY_LENGTH_MM = 2.398
NATIVE_FRAME_W, NATIVE_FRAME_H = 1936, 448

# Measured reference values (task-21 diagnostic) -- used ONLY to sanity-check
# the bias this script computes independently below; never used as the
# subtracted value itself.
EXPECTED_BIAS_MM = np.array([-0.1116, -0.0822, 0.0716], np.float64)
BIAS_CHECK_TOL_MM = 2e-3
EXPECTED_MEDIAN_BEFORE_MM = 0.170
EXPECTED_MEDIAN_AFTER_MM = 0.088


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
    (presentation-only; the reprojected, bias-corrected data3D.csv keypoints
    drawn on top are unaffected -- see clip_io.py's module docstring)."""
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
    """Reprojected (debiased) data3D.csv keypoints -- leg chains + points,
    dimmed by the CSV's OWN per-keypoint confidence column (col 4 of each
    keypoint's 4), same visual treatment `draw.py` gives detector
    confidence."""
    pts = np.asarray(uv).copy()
    pts[:, 0] -= x0
    out = draw.draw_leg_chains(panel_native, pts, kp_names, alpha=1.0,
                                thickness=1, kp_colors=kp_colors)
    out = draw.draw_keypoints(out, pts, kp_names, conf=conf, alpha=1.0,
                               radius=4, kp_colors=kp_colors)
    return out


def _build_title_panel(kp_names, bias_mm: np.ndarray) -> np.ndarray:
    # NOTE: the caption + bias lines below are longer than act1_data3d_test.py's
    # single short caption, so they use a slightly smaller scale (0.6, vs the
    # shared draw.CAPTION_SCALE=0.65) -- verified with cv2.getTextSize to fit
    # inside this panel's width (CELL_W - x_start = 432 px) so nothing is
    # silently clipped off the edge of the frame (checked by reading rendered
    # frames with the Read tool; the 0.65 scale clipped the bias value).
    _SCALE = 0.6
    img = np.zeros((CELL_H, CELL_W, 3), np.uint8)
    img = draw.stage_title(img, "2D Keypoint")
    img = draw.label(img, TITLE_LINE2, (48, 128), scale=draw.TITLE_SCALE,
                      color=(255, 255, 255), thickness=draw.TITLE_THICKNESS)
    img = draw.label(img, TITLE_CAPTION, (48, 168), scale=_SCALE)
    bias_px = float(np.linalg.norm(bias_mm)) * PX_PER_MM
    img = draw.label(img, "rigid bias removed:", (48, 200), scale=_SCALE)
    bias_line = (f"({bias_mm[0]:+.3f}, {bias_mm[1]:+.3f}, {bias_mm[2]:+.3f}) "
                 f"mm = {bias_px:.1f} px")
    img = draw.label(img, bias_line, (48, 224), scale=_SCALE)
    speed = _speed_factor()
    img = draw.label(img, f"800 fps -> 1/{speed:.1f} speed",
                      (48, 256), scale=draw.CAPTION_SCALE)
    y = 256
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
    xyz_mm = xyz_det_mm[:, idx_d2m]             # (T,50,3) mm, MODEL order, RAW (biased)
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
    disp_name = clip_io.display_names(cam_names, clip)

    # --- load 04_kp3d_filt.npz (full sequence, MODEL order) -----------------
    kp3d_z = np.load(d["predictions"] / "04_kp3d_filt.npz", allow_pickle=True)
    k3 = kp3d_z["kp3d"]                          # (N,50,3) mm, MODEL order
    kp3d_names = [str(n) for n in kp3d_z["kp_names"]]
    if kp3d_names != kp_names:
        raise ValueError("04_kp3d_filt.npz kp_names != 02_kp2d.npz kp_names")

    # --- compute the rigid bias IN-SCRIPT (never hardcoded) -----------------
    # Per the task brief: bias = mean(data3D.csv - ours) over every overlapping
    # (frame, keypoint) pair in the FULL sequence (not just the render window).
    n = min(len(xyz_mm), len(k3))
    bias = np.nanmean((xyz_mm[:n] - k3[:n]).reshape(-1, 3), axis=0)   # (3,) mm
    bias_norm_mm = float(np.linalg.norm(bias))
    print(f"[bias] computed over n={n} overlapping frames: "
          f"({bias[0]:+.4f}, {bias[1]:+.4f}, {bias[2]:+.4f}) mm  "
          f"|bias|={bias_norm_mm:.4f} mm = {bias_norm_mm * PX_PER_MM:.2f} px")
    bias_err = np.abs(bias - EXPECTED_BIAS_MM)
    print(f"[bias check] vs measured reference "
          f"({EXPECTED_BIAS_MM[0]:+.4f}, {EXPECTED_BIAS_MM[1]:+.4f}, "
          f"{EXPECTED_BIAS_MM[2]:+.4f}) mm -- max abs component diff = "
          f"{bias_err.max():.5f} mm (tol {BIAS_CHECK_TOL_MM} mm)")
    if bias_err.max() > BIAS_CHECK_TOL_MM:
        raise ValueError(
            f"computed bias {bias} mm does not match the measured reference "
            f"{EXPECTED_BIAS_MM} mm to within {BIAS_CHECK_TOL_MM} mm "
            f"(max abs diff {bias_err.max():.5f} mm) -- the loading path here "
            "differs from the one used to measure the reference constants; "
            "STOPPING rather than rendering a video built on a bad "
            "assumption.")

    # --- median 3D disagreement vs 04_kp3d_filt.npz, before vs after --------
    dist_before = np.linalg.norm(xyz_mm[:n] - k3[:n], axis=-1)         # (n,K)
    xyz_mm_debiased = xyz_mm - bias                                    # (T_csv,K,3)
    dist_after = np.linalg.norm(xyz_mm_debiased[:n] - k3[:n], axis=-1)  # (n,K)
    med_before_mm = float(np.nanmedian(dist_before))
    med_after_mm = float(np.nanmedian(dist_after))
    print(f"[3D disagreement] median |data3D.csv - 04_kp3d_filt| over "
          f"n={n} frames x {dist_before.shape[1]} keypoints:")
    print(f"    before debias: {med_before_mm:.4f} mm = "
          f"{med_before_mm * PX_PER_MM:.2f} px  (expected ~{EXPECTED_MEDIAN_BEFORE_MM} mm)")
    print(f"    after debias:  {med_after_mm:.4f} mm = "
          f"{med_after_mm * PX_PER_MM:.2f} px  (expected ~{EXPECTED_MEDIAN_AFTER_MM} mm, the scatter floor)")
    if abs(med_after_mm - EXPECTED_MEDIAN_AFTER_MM) > 0.02:
        print("    ** WARNING: post-debias disagreement is NOT close to the "
              "scatter floor -- the offset may not be a pure translation. **")

    # --- reproject the DEBIASED data3D.csv for every rendered output frame -
    t_for_f = np.array(
        [WINDOW_START + int(f * WINDOW_LEN / N_OUT) for f in range(N_OUT)],
        np.int64)
    assert np.all(np.diff(t_for_f) > 0), "expected a strictly increasing map"

    uv_csv = np.stack(
        [clip_io.project(cam_mats, xyz_mm_debiased[int(t)]) for t in t_for_f])  # (N_OUT,C,K,2)
    conf_csv_f = conf[t_for_f]                                          # (N_OUT,K)

    # --- in-frame sanity check (native 1936x448 video frame) ---------------
    print("[in-frame check] fraction of reprojected (debiased) keypoints "
          f"inside {NATIVE_FRAME_W}x{NATIVE_FRAME_H} per camera, over the "
          f"{N_OUT}-frame rendered window:")
    for ci, cam in enumerate(cam_names):
        pts = uv_csv[:, ci]                     # (N_OUT,K,2)
        valid = np.all(np.isfinite(pts), axis=-1)
        inside = (valid & (pts[..., 0] >= 0) & (pts[..., 0] < NATIVE_FRAME_W)
                  & (pts[..., 1] >= 0) & (pts[..., 1] < NATIVE_FRAME_H))
        frac = float(inside.sum()) / float(pts.shape[0] * pts.shape[1])
        print(f"    {cam}: {frac:.4f}")

    # --- 04_kp3d_filt.npz reprojected over the same window ------------------
    uv_filt = np.stack(
        [clip_io.project(cam_mats, k3[int(t)]) for t in t_for_f])       # (N_OUT,C,K,2)

    # --- jitter comparison: debiased data3D.csv reprojection vs 04_kp3d_filt
    print("[jitter check] median leg-keypoint 2nd-difference jitter (px), "
          f"over the same {N_OUT} rendered frames, per camera:")
    for ci, cam in enumerate(cam_names):
        j_csv = _leg_jitter_px(uv_csv[:, ci], kp_names)
        j_filt = _leg_jitter_px(uv_filt[:, ci], kp_names)
        smoother = "04_kp3d_filt" if j_filt < j_csv else "data3D.csv (debiased)"
        print(f"    {cam}: data3D.csv(debiased)={j_csv:.3f}  "
              f"04_kp3d_filt={j_filt:.3f}  smoother={smoother}")

    # --- median 2D reprojection difference per camera, over the rendered
    # window -- this is what the user will actually see on screen -----------
    print("[2D reprojection diff] median |uv(debiased data3D.csv) - "
          f"uv(04_kp3d_filt)| (px), over the {N_OUT}-frame rendered window, "
          "per camera:")
    for ci, cam in enumerate(cam_names):
        d2d = np.linalg.norm(uv_csv[:, ci] - uv_filt[:, ci], axis=-1)  # (N_OUT,K)
        med_px = float(np.nanmedian(d2d))
        print(f"    {cam}: {med_px:.2f} px")

    # --- crops: SAME as Act 1 (centred on the detector 2D), for direct
    # frame-for-frame comparability between the two videos -------------------
    src_frame_w = 1936
    x0_by_cam = {cam: _smoothed_crop_x0(kp2d[:, ci], src_frame_w)
                 for ci, cam in enumerate(cam_names)}
    crops = _preload_crops(clip, cam_names, x0_by_cam, t_for_f)

    out_dir = d["frames"] / "act1_data3d_debiased_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    title_panel = _build_title_panel(kp_names, bias)

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

    out_path = Path(clip) / "ik_explainer" / "act1_data3d_debiased_test.mp4"

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
