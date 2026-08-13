import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import clip_io

CLIP = clip_io.CLIP_DEFAULT
pytestmark = pytest.mark.skipif(
    not Path(CLIP).exists(), reason=f"source clip not present: {CLIP}")


def test_load_dlt_returns_seven_affine_cameras():
    cam_mats, names = clip_io.load_dlt(os.path.join(CLIP, "calibration"))
    assert cam_mats.shape == (7, 4, 3)
    assert names[0] == "Cam2012630" and len(names) == 7
    # affine/telecentric: P[2,:3] == 0, P[2,3] == 1  (P = cam_mat.T)
    P = np.swapaxes(cam_mats, -1, -2)                      # (7,3,4)
    assert np.allclose(P[:, 2, :3], 0.0)
    assert np.allclose(P[:, 2, 3], 1.0)


def test_shipped_kp3d_is_millimetres_and_detector_order():
    xyz, conf, names = clip_io.load_shipped_kp3d_mm(
        clip_io.shipped_csv_path(CLIP))
    assert xyz.shape[1] == 50 and conf.shape[1] == 50
    assert names[:4] == ["Antenna_Base", "EyeL", "EyeR", "Scutellum"]
    # body length Antenna_Base -> Abd_tip is ~2.4 mm for Drosophila.
    i0, i1 = names.index("Antenna_Base"), names.index("Abd_tip")
    body_mm = np.nanmedian(np.linalg.norm(xyz[:, i0] - xyz[:, i1], axis=-1))
    assert 1.5 < body_mm < 4.0, f"body length {body_mm:.2f} mm — unit error"


def test_projection_lands_in_frame_and_never_wildly_outside():
    """Guards the 0.1mm/mm unit trap.

    MEASURED ground truth for this clip: 99.958% of all 322,700 keypoint
    projections land inside 1936x448. The 134 that do not are ALL on
    Cam2012630 (the vertical/top camera) and are all distal right-leg tips
    (T2R_TaTip, T1R_TaTip, T1R_TaT3, T2R_TaT3) leaving the bottom edge by at
    most 15.7 px -- genuine field-of-view clipping on a 448-px-tall strip, not
    a calibration error.

    The excursion bound is what makes this a unit-trap test: feeding the raw
    0.1mm CSV puts u in [7927, 10705], thousands of px outside, so the 32 px
    margin fails instantly.
    """
    cam_mats, _ = clip_io.load_dlt(os.path.join(CLIP, "calibration"))
    xyz, _, _ = clip_io.load_shipped_kp3d_mm(clip_io.shipped_csv_path(CLIP))
    uv = np.stack([clip_io.project(cam_mats, xyz[t]) for t in range(xyz.shape[0])])
    u, v = uv[..., 0], uv[..., 1]
    inside = (u >= 0) & (u < 1936) & (v >= 0) & (v < 448)
    assert inside.mean() > 0.999, f"only {100 * inside.mean():.3f}% in frame"
    margin = 32.0
    assert u.min() > -margin and u.max() < 1936 + margin
    assert v.min() > -margin and v.max() < 448 + margin


def test_camera_view_dirs_form_a_30_degree_arc():
    cam_mats, names = clip_io.load_dlt(os.path.join(CLIP, "calibration"))
    d = clip_io.camera_view_dirs(cam_mats)
    assert d.shape == (7, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    ang = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
    off = ang[~np.eye(7, dtype=bool)]
    # every pair is a multiple of 30 deg within 1.5 deg
    assert np.all(np.abs(off - np.round(off / 30.0) * 30.0) < 1.5)


def test_frame_counts_agree_within_one():
    cam_mats, names = clip_io.load_dlt(os.path.join(CLIP, "calibration"))
    n_vid = clip_io.n_video_frames(clip_io.video_path(CLIP, names[0]))
    xyz, _, _ = clip_io.load_shipped_kp3d_mm(clip_io.shipped_csv_path(CLIP))
    assert abs(n_vid - xyz.shape[0]) <= 1
    assert n_vid == 921


def test_read_frames_returns_requested_frames():
    _, names = clip_io.load_dlt(os.path.join(CLIP, "calibration"))
    imgs = clip_io.read_frames(clip_io.video_path(CLIP, names[0]), [0, 5, 900])
    assert imgs.shape == (3, 448, 1936, 3)
    assert imgs.dtype == np.uint8
