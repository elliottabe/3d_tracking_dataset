"""Phase 7 / component 8: per-camera reprojection overlay video.

Streams each camera's raw Frame_*.jpg through imageio.get_writer -- ONE decoded
frame in memory at a time (mirrors stac-mjx/stac_mjx/stac.py:735,745) -- drawing
the projected posed mesh (subset) + keypoints (+ optional SAM mask contour).
Never buffers all frames.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_overlay_frame(raw_rgb, mesh2d, kp2d, *, contour=None):
    """Rasterize raw_rgb with mesh (cyan) + kp (magenta) [+ contour (yellow)]
    overlays to a uint8 RGB array via an Agg matplotlib figure."""

    raw = np.asarray(raw_rgb)
    if raw.dtype != np.uint8:
        raw = np.clip(raw * (255.0 if raw.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    H, W = raw.shape[:2]
    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(raw)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    if contour is not None and len(contour):
        cc = np.asarray(contour)
        ax.plot(cc[:, 0], cc[:, 1], c="yellow", lw=0.8)
    if len(mesh2d):
        m = np.asarray(mesh2d)
        ax.scatter(m[:, 0], m[:, 1], s=2, c="cyan", linewidths=0)
    if len(kp2d):
        k = np.asarray(kp2d)
        ax.scatter(k[:, 0], k[:, 1], s=10, c="magenta", linewidths=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    # resize-safe: crop/pad to (H,W) if the Agg canvas rounded differently
    bh, bw = buf.shape[:2]
    if (bh, bw) != (H, W):
        out = np.zeros((H, W, 3), np.uint8)
        out[:min(H, bh), :min(W, bw)] = buf[:min(H, bh), :min(W, bw)]
        buf = out
    return buf.astype(np.uint8)


def write_camera_video(out_path, *, frames_rgb_iter, mesh2d_by_frame,
                       kp2d_by_frame, contour_by_frame=None, fps=30,
                       codec="libx264", pixelformat="yuv420p",
                       macro_block_size=16):
    """Stream overlay frames to an mp4 via imageio (one frame in memory).

    ``codec``/``pixelformat`` are passed through explicitly (imageio's
    ffmpeg plugin already defaults to libx264/yuv420p, but pinning them here
    guarantees VS-Code/most-players playback regardless of imageio version
    or installed ffmpeg build -- see render_bout_reproj.py, the first real
    caller that needs this guarantee)."""
    import os
    import imageio
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec=codec,
                            pixelformat=pixelformat,
                            macro_block_size=macro_block_size) as video:
        for t, raw in enumerate(frames_rgb_iter):
            contour = None if contour_by_frame is None else contour_by_frame[t]
            frame = draw_overlay_frame(raw, mesh2d_by_frame[t], kp2d_by_frame[t],
                                       contour=contour)
            video.append_data(frame)
    return out_path
