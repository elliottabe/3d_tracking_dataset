"""Unit tests for jarvis_jax.scripts.render_bout_reproj -- CPU-only, fast, no
real video decode / GPU. Video decode (cv2 seek+read) and JARVIS camera-order
geometry (session_geometry) are exercised only by the live render command
documented in the module docstring, not here.
"""
import csv
import json
import math
import os

import numpy as np
import pytest

from jarvis_jax.scripts.render_bout_reproj import (
    reproject_points, project_and_filter, read_fly_csv, dense_by_frame,
    find_bout_in_manifest, camera_video_path, _norm_cam_name,
)


# --------------------------------------------------------------------------
# reprojection
# --------------------------------------------------------------------------
def test_reproject_points_identity_w1():
    # M s.t. ph@M = [x, y, 1] (z-column contributes nothing, w-bias = 1)
    M = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]])
    p3d = np.array([5.0, 7.0, 100.0])
    out = reproject_points(p3d, M)
    assert out.shape == (2,)
    np.testing.assert_allclose(out, [5.0, 7.0])


def test_reproject_points_perspective_divide():
    # M s.t. ph@M = [x, y, z] -> divide by w=z
    M = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0]])
    p3d = np.array([10.0, 20.0, 2.0])
    out = reproject_points(p3d, M)
    np.testing.assert_allclose(out, [5.0, 10.0])


def test_reproject_points_batched_shape():
    M = np.eye(4, 3)
    p3d = np.random.randn(3, 50, 3)      # (T, J, 3)
    out = reproject_points(p3d, M)
    assert out.shape == (3, 50, 2)


def test_reproject_points_nan_propagates():
    M = np.eye(4, 3)
    p3d = np.array([1.0, np.nan, 3.0])
    out = reproject_points(p3d, M)
    assert np.isnan(out).all()


def test_project_and_filter_drops_nan_and_low_conf():
    M = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]])
    kp = np.array([[1.0, 2.0, 0.0],
                  [np.nan, np.nan, np.nan],
                  [4.0, 5.0, 0.0]])
    conf = np.array([0.9, 0.5, 0.1])
    out = project_and_filter(kp, conf, M, min_conf=0.2)
    # joint 1 dropped (NaN), joint 2 dropped (low conf) -> only joint 0 kept
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], [1.0, 2.0])


# --------------------------------------------------------------------------
# fly CSV reader
# --------------------------------------------------------------------------
def _write_fly_csv(path, joint_names, conf_label, rows):
    """rows: list[(frame, kp(J,3)|None, conf(J,)|None)]."""
    nj = len(joint_names)
    h1 = ["frame"]
    h2 = ["frame"]
    for n in joint_names:
        h1 += [n, n, n, n]
        h2 += ["x", "y", "z", conf_label]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(h1)
        w.writerow(h2)
        for frame, kp, conf in rows:
            if kp is None:
                w.writerow([frame] + ["nan"] * (nj * 4))
                continue
            row = [frame]
            for j in range(nj):
                row += [float(kp[j, 0]), float(kp[j, 1]), float(kp[j, 2]),
                       float(conf[j])]
            w.writerow(row)


NAMES = ["Antenna_Base", "EyeL"]


@pytest.mark.parametrize("conf_label", ["confidence", "conf"])
def test_read_fly_csv_roundtrip(tmp_path, conf_label):
    # header-2 label ("confidence" from session_io.write_fly_csv vs. "conf"
    # seen in real production output) must NOT matter -- columns are read
    # positionally.
    rows = [
        (100, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([0.9, 0.8])),
        (101, None, None),
    ]
    p = tmp_path / "fly0.csv"
    _write_fly_csv(str(p), NAMES, conf_label, rows)

    joint_names, frames, kp, conf = read_fly_csv(str(p))
    assert joint_names == NAMES
    np.testing.assert_array_equal(frames, [100, 101])
    assert kp.shape == (2, 2, 3)
    assert conf.shape == (2, 2)
    np.testing.assert_allclose(kp[0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(conf[0], [0.9, 0.8])
    assert math.isnan(kp[1, 0, 0])
    assert math.isnan(conf[1, 0])


def test_read_fly_csv_rejects_bad_header(tmp_path):
    p = tmp_path / "bad.csv"
    with open(p, "w") as f:
        f.write("not,a,valid,header\n1,2,3,4\n")
    with pytest.raises(ValueError):
        read_fly_csv(str(p))


# --------------------------------------------------------------------------
# dense_by_frame
# --------------------------------------------------------------------------
def test_dense_by_frame_fills_gaps_with_nan():
    frames = np.array([10, 12])              # gap at 11
    kp = np.array([[[1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0]]])   # (2,1,3)
    conf = np.array([[0.9], [0.7]])
    out_kp, out_conf = dense_by_frame(frames, kp, conf, start=10, num_frames=3)
    assert out_kp.shape == (3, 1, 3)
    np.testing.assert_allclose(out_kp[0, 0], [1.0, 1.0, 1.0])
    assert np.isnan(out_kp[1]).all()
    np.testing.assert_allclose(out_kp[2, 0], [3.0, 3.0, 3.0])
    assert out_conf[0, 0] == 0.9 and math.isnan(out_conf[1, 0])


def test_dense_by_frame_drops_out_of_range_frames():
    frames = np.array([5, 10, 999])
    kp = np.zeros((3, 1, 3))
    conf = np.zeros((3, 1))
    out_kp, _ = dense_by_frame(frames, kp, conf, start=10, num_frames=2)
    assert out_kp.shape == (2, 1, 3)
    # frame 5 and 999 both fall outside [10,12) -> both dropped, only frame
    # 10 (idx 0) lands
    assert not np.isnan(out_kp[0]).all()
    assert np.isnan(out_kp[1]).all()


# --------------------------------------------------------------------------
# manifest.json best-effort lookup
# --------------------------------------------------------------------------
def test_find_bout_in_manifest_hit(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"bouts": [{"bout_idx": 4, "start": 100, "num_frames": 50},
                                       {"bout_idx": 7, "start": 500, "num_frames": 20}]}))
    b = find_bout_in_manifest(str(p), 7)
    assert b == {"bout_idx": 7, "start": 500, "num_frames": 20}


def test_find_bout_in_manifest_miss_returns_none(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"bouts": [{"bout_idx": 4, "start": 100, "num_frames": 50}]}))
    assert find_bout_in_manifest(str(p), 999) is None


def test_find_bout_in_manifest_missing_file_returns_none(tmp_path):
    assert find_bout_in_manifest(str(tmp_path / "nope.json"), 1) is None


# --------------------------------------------------------------------------
# camera-name helpers (the Cam-prefix gotcha: session_geometry's camera_names
# already include "Cam", e.g. "Cam2012853" -- verified against the real
# red_data_unified_V3 instances_val.json calibrations dict on 2026-07-18)
# --------------------------------------------------------------------------
def test_camera_video_path_name_already_prefixed():
    assert camera_video_path("/sess", "Cam2012853") == "/sess/Cam2012853.mp4"


def test_camera_video_path_bare_numeric_name():
    assert camera_video_path("/sess", "2012853") == "/sess/Cam2012853.mp4"


def test_norm_cam_name():
    assert _norm_cam_name("Cam2012853") == "2012853"
    assert _norm_cam_name("2012853") == "2012853"


# --------------------------------------------------------------------------
# write_camera_video codec guarantee (reproj_video.py edit) -- ffprobe-verified
# --------------------------------------------------------------------------
def test_write_camera_video_produces_h264_yuv420p(tmp_path):
    import shutil
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not available")
    import subprocess
    from jarvis_jax.tracking.reproj_video import write_camera_video

    T = 3
    frames = [np.zeros((32, 48, 3), np.uint8) for _ in range(T)]
    mesh2d = [np.array([[5.0, 5.0]]) for _ in range(T)]
    kp2d = [np.array([[6.0, 6.0]]) for _ in range(T)]
    out = str(tmp_path / "cam0.mp4")
    write_camera_video(out, frames_rgb_iter=iter(frames), mesh2d_by_frame=mesh2d,
                       kp2d_by_frame=kp2d, fps=10)
    assert os.path.exists(out) and os.path.getsize(out) > 0

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt",
        "-of", "default=noprint_wrappers=1", out],
        capture_output=True, text=True, check=True)
    info = dict(line.split("=", 1) for line in result.stdout.strip().splitlines())
    assert info["codec_name"] == "h264"
    assert info["pix_fmt"] == "yuv420p"
