"""Phase 7 / component 8: per-camera reprojection overlay video.

Streams each camera's raw Frame_*.jpg through imageio.get_writer -- ONE decoded
frame in memory at a time (mirrors stac-mjx/stac_mjx/stac.py:735,745) -- drawing
the projected posed mesh (subset) + keypoints (+ optional SAM mask contour)
[+ optional skeleton bone line segments]. Never buffers all frames.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def draw_overlay_frame(raw_rgb, mesh2d, kp2d, *, contour=None,
                       mesh_edges=None, kp_edges=None, edge_lw=1.2):
    """Rasterize raw_rgb with mesh (cyan) + kp (magenta) [+ contour (yellow)]
    [+ skeleton bone segments] overlays to a uint8 RGB array via an Agg
    matplotlib figure.

    Parameters
    ----------
    mesh_edges, kp_edges : array_like, shape (E, 2, 2), optional
        Pixel-space line-segment endpoint pairs (``[[x0,y0],[x1,y1]]`` per
        edge) to draw as thin "bone" lines, in the SAME color as the
        corresponding dots (``mesh_edges`` cyan like ``mesh2d``, ``kp_edges``
        magenta like ``kp2d``) so the overlay reads as a connected skeleton
        rather than a bare point cloud. Callers are expected to have already
        dropped edges with an invalid (NaN / low-confidence) endpoint --
        e.g. via ``render_bout_reproj.skeleton_segments`` -- this function
        draws whatever segments it is given, unconditionally.
    edge_lw : float
        Line width (matplotlib points) for both edge sets.
    """

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
    # bone lines drawn UNDER the joint dots (dots drawn after, so they sit on top)
    if mesh_edges is not None and len(mesh_edges):
        ax.add_collection(LineCollection(np.asarray(mesh_edges), colors="cyan",
                                         linewidths=edge_lw))
    if kp_edges is not None and len(kp_edges):
        ax.add_collection(LineCollection(np.asarray(kp_edges), colors="magenta",
                                         linewidths=edge_lw))
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
                       kp2d_by_frame, contour_by_frame=None,
                       mesh_edges_by_frame=None, kp_edges_by_frame=None,
                       fps=30, codec="libx264", pixelformat="yuv420p",
                       macro_block_size=16):
    """Stream overlay frames to an mp4 via imageio (one frame in memory).

    ``codec``/``pixelformat`` are passed through explicitly (imageio's
    ffmpeg plugin already defaults to libx264/yuv420p, but pinning them here
    guarantees VS-Code/most-players playback regardless of imageio version
    or installed ffmpeg build -- see render_bout_reproj.py, the first real
    caller that needs this guarantee).

    ``mesh_edges_by_frame``/``kp_edges_by_frame`` (optional): per-frame
    ``(E_t, 2, 2)`` skeleton bone line-segment arrays, see
    ``draw_overlay_frame``'s ``mesh_edges``/``kp_edges``. ``None`` (default)
    draws no bone lines -- existing dot-only callers are unaffected."""
    import os
    import imageio
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec=codec,
                            pixelformat=pixelformat,
                            macro_block_size=macro_block_size) as video:
        for t, raw in enumerate(frames_rgb_iter):
            contour = None if contour_by_frame is None else contour_by_frame[t]
            mesh_edges = None if mesh_edges_by_frame is None else mesh_edges_by_frame[t]
            kp_edges = None if kp_edges_by_frame is None else kp_edges_by_frame[t]
            frame = draw_overlay_frame(raw, mesh2d_by_frame[t], kp2d_by_frame[t],
                                       contour=contour, mesh_edges=mesh_edges,
                                       kp_edges=kp_edges)
            video.append_data(frame)
    return out_path
