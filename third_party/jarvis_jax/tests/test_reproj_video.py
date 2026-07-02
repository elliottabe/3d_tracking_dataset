import os
import numpy as np
import pytest


def test_draw_overlay_frame_returns_uint8_same_size():
    from jarvis_jax.cse.reproj_video import draw_overlay_frame
    raw = np.zeros((32, 40, 3), np.uint8)
    mesh2d = np.array([[5.0, 5.0], [10.0, 8.0], [20.0, 15.0]])
    kp2d = np.array([[6.0, 6.0], [25.0, 20.0]])
    out = draw_overlay_frame(raw, mesh2d, kp2d)
    assert out.dtype == np.uint8
    assert out.shape == (32, 40, 3)
    # something was drawn (not all-black anymore)
    assert out.max() > 0


def test_write_camera_video_streams_and_counts(tmp_path):
    import imageio
    from jarvis_jax.cse.reproj_video import write_camera_video
    T = 2
    frames = [np.zeros((32, 40, 3), np.uint8) for _ in range(T)]
    mesh2d = [np.array([[5.0, 5.0], [10.0, 8.0]]) for _ in range(T)]
    kp2d = [np.array([[6.0, 6.0]]) for _ in range(T)]
    out = str(tmp_path / "cam0.mp4")
    p = write_camera_video(out, frames_rgb_iter=iter(frames),
                           mesh2d_by_frame=mesh2d, kp2d_by_frame=kp2d, fps=10)
    assert p == out
    assert os.path.exists(out) and os.path.getsize(out) > 0
    rdr = imageio.get_reader(out)
    n = rdr.count_frames() if hasattr(rdr, "count_frames") else sum(1 for _ in rdr)
    rdr.close()
    assert n == T
