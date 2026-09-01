"""Phase 7 / component 8: per-camera reprojection overlay video.

Streams each camera's raw frames through imageio.get_writer -- ONE decoded
frame in memory at a time (mirrors stac-mjx/stac_mjx/stac.py:735,745) -- drawing
the projected posed mesh (subset) + keypoints (+ optional SAM mask contour)
[+ optional skeleton bone line segments]. Never buffers all frames.

COST, measured per frame on a real bout (Session0 2025_10_20_13_20_04 bout 28,
1936x448 h264, 2007 frames x 7 cameras = 11m25s):
    matplotlib Agg draw  ~34 ms   (26 ms of it is imshow's resample of the raw
                                   frame: the axes' xlim/ylim put the image on
                                   a half-pixel offset, so it is a real
                                   interpolation and cannot be skipped or
                                   downgraded to 'nearest' without changing
                                   pixels -- both were measured to differ)
    h264 decode          ~18 ms
    libx264 encode       ~10 ms
So the three things that pay off WITHOUT touching a single output pixel are
(a) reusing one Agg figure for a whole camera (``OverlayRenderer``),
(b) overlapping the decode with the draw (``_prefetch``), and (c) rendering the
cameras concurrently, one isolated subprocess each (``render_camera_overlays``)
-- the cameras are completely independent. All three are pixel-identical by
construction: the same frames, in the same order, through the same rasteriser.
Measured on that bout: 97.9 -> 72 s per camera for (a)+(b), and 685 -> 195 s
for all seven with (c) at 7 workers (on a node already at load ~130, so that
3.5x is a floor, not a ceiling).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

_EMPTY_PTS = np.zeros((0, 2))
_EMPTY_SEGS = np.zeros((0, 2, 2))


class OverlayRenderer:
    """One reusable Agg figure for a whole camera's worth of overlay frames.

    ``draw`` returns byte-for-byte what ``draw_overlay_frame`` returns
    (verified over random frames/point counts in
    tests/test_reproj_video_fast.py): the artists are created once, in the same
    order (so the same z-order), and only their data is swapped per frame.
    Saves the per-frame Figure/Axes/imshow construction (~6 of ~40 ms).
    """

    def __init__(self, H, W, *, edge_lw=1.2, dpi=100.0):
        self.H, self.W = int(H), int(W)
        fig = plt.figure(figsize=(self.W / dpi, self.H / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        self._im = ax.imshow(np.zeros((self.H, self.W, 3), np.uint8))
        ax.set_xlim(0, self.W); ax.set_ylim(self.H, 0); ax.axis("off")
        # creation order == draw_overlay_frame's: contour, mesh edges, kp edges,
        # mesh dots, kp dots (dots on top of the bones).
        self._contour, = ax.plot([], [], c="yellow", lw=0.8)
        self._mesh_edges = ax.add_collection(
            LineCollection(_EMPTY_SEGS, colors="cyan", linewidths=edge_lw))
        self._kp_edges = ax.add_collection(
            LineCollection(_EMPTY_SEGS, colors="magenta", linewidths=edge_lw))
        self._mesh = ax.scatter([], [], s=2, c="cyan", linewidths=0)
        self._kp = ax.scatter([], [], s=10, c="magenta", linewidths=0)
        self.fig = fig
        self.ax = ax

    def draw(self, raw_rgb, mesh2d, kp2d, *, contour=None,
             mesh_edges=None, kp_edges=None):
        raw = np.asarray(raw_rgb)
        if raw.dtype != np.uint8:
            raw = np.clip(raw * (255.0 if raw.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        H, W = raw.shape[:2]
        self._im.set_data(raw)
        if contour is not None and len(contour):
            cc = np.asarray(contour)
            self._contour.set_data(cc[:, 0], cc[:, 1])
        else:
            self._contour.set_data([], [])
        self._mesh_edges.set_segments(
            np.asarray(mesh_edges) if mesh_edges is not None and len(mesh_edges)
            else _EMPTY_SEGS)
        self._kp_edges.set_segments(
            np.asarray(kp_edges) if kp_edges is not None and len(kp_edges)
            else _EMPTY_SEGS)
        self._mesh.set_offsets(np.asarray(mesh2d) if len(mesh2d) else _EMPTY_PTS)
        self._kp.set_offsets(np.asarray(kp2d) if len(kp2d) else _EMPTY_PTS)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
        # resize-safe: crop/pad to (H,W) if the Agg canvas rounded differently
        bh, bw = buf.shape[:2]
        if (bh, bw) != (H, W):
            out = np.zeros((H, W, 3), np.uint8)
            out[:min(H, bh), :min(W, bw)] = buf[:min(H, bh), :min(W, bw)]
            buf = out
        return buf.astype(np.uint8)

    def close(self):
        plt.close(self.fig)


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

    Single-frame convenience wrapper around ``OverlayRenderer``; a caller
    drawing many frames of the same size should hold one renderer instead
    (``write_camera_video`` does).
    """
    raw = np.asarray(raw_rgb)
    r = OverlayRenderer(raw.shape[0], raw.shape[1], edge_lw=edge_lw)
    try:
        return r.draw(raw, mesh2d, kp2d, contour=contour,
                      mesh_edges=mesh_edges, kp_edges=kp_edges)
    finally:
        r.close()


def _prefetch(it, size=8):
    """Yield `it` from a background thread, up to `size` items ahead.

    Decoding the next frame (h264, ~18 ms) happens while the current one is
    being rasterised (~34 ms), so the decode stops being serial time. Order,
    values and exceptions are preserved exactly -- this only changes WHEN the
    work happens. ~21 MB of buffered frames at the default size for this rig's
    1936x448 frames.
    """
    import queue
    import threading
    done = object()
    q: "queue.Queue" = queue.Queue(maxsize=size)

    def _run():
        try:
            for item in it:
                q.put(item)
        except BaseException as exc:                    # noqa: BLE001 -- re-raised below
            q.put(exc)
        else:
            q.put(done)

    threading.Thread(target=_run, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def write_camera_video(out_path, *, frames_rgb_iter, mesh2d_by_frame,
                       kp2d_by_frame, contour_by_frame=None,
                       mesh_edges_by_frame=None, kp_edges_by_frame=None,
                       fps=30, codec="libx264", pixelformat="yuv420p",
                       macro_block_size=16, prefetch=8):
    """Stream overlay frames to an mp4 via imageio (one frame in memory).

    ``codec``/``pixelformat`` are passed through explicitly (imageio's
    ffmpeg plugin already defaults to libx264/yuv420p, but pinning them here
    guarantees VS-Code/most-players playback regardless of imageio version
    or installed ffmpeg build -- see render_bout_reproj.py, the first real
    caller that needs this guarantee).

    ``mesh_edges_by_frame``/``kp_edges_by_frame`` (optional): per-frame
    ``(E_t, 2, 2)`` skeleton bone line-segment arrays, see
    ``draw_overlay_frame``'s ``mesh_edges``/``kp_edges``. ``None`` (default)
    draws no bone lines -- existing dot-only callers are unaffected.

    ``prefetch``: how many source frames to decode ahead on a background
    thread (0 disables). Output is identical either way; see ``_prefetch``.
    """
    import os
    import imageio
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames = _prefetch(frames_rgb_iter, prefetch) if prefetch else frames_rgb_iter
    renderer = None
    try:
        with imageio.get_writer(out_path, fps=fps, codec=codec,
                                pixelformat=pixelformat,
                                macro_block_size=macro_block_size) as video:
            for t, raw in enumerate(frames):
                contour = None if contour_by_frame is None else contour_by_frame[t]
                mesh_edges = None if mesh_edges_by_frame is None else mesh_edges_by_frame[t]
                kp_edges = None if kp_edges_by_frame is None else kp_edges_by_frame[t]
                if renderer is None:
                    raw0 = np.asarray(raw)
                    renderer = OverlayRenderer(raw0.shape[0], raw0.shape[1])
                frame = renderer.draw(raw, mesh2d_by_frame[t], kp2d_by_frame[t],
                                      contour=contour, mesh_edges=mesh_edges,
                                      kp_edges=kp_edges)
                video.append_data(frame)
    finally:
        if renderer is not None:
            renderer.close()
    return out_path


# ---------------------------------------------------------------------------
# One camera per process: the cameras share nothing, and the per-frame cost is
# ~60 ms of CPU that no amount of numpy vectorisation can remove (matplotlib
# rasterisation + h264). Rendering them concurrently is the only large win
# available that leaves every output byte untouched.
#
# Each camera runs as an ISOLATED SUBPROCESS (``python -m
# jarvis_jax.tracking.reproj_video <job.npz>``), the same pattern
# scripts/run_bout.py already uses for the side-by-side viz -- NOT
# multiprocessing:
#   * fork is unsafe: the caller holds jax's threads;
#   * spawn (ProcessPoolExecutor(mp_context='spawn')) re-executes the PARENT's
#     __main__ in every child. For run_bout.py that means re-importing
#     jax + mujoco + hydra + stac_mjx per worker (~20 s and >1 GB each), and
#     for any caller WITHOUT an `if __name__ == "__main__"` guard it means the
#     children re-run the caller's top-level work -- observed here as seven
#     children all rendering the same camera into the same temp file.
# The subprocess imports numpy/matplotlib/imageio/cv2 only: 0.4 s, ~300 MB,
# no jax, no GPU.
# ---------------------------------------------------------------------------

_JOB_KEYS = ("cam", "session_dir", "start", "T", "H", "W", "out_path", "fps",
             "use_sync_plan")


def _save_job(job, path):
    """Serialise one render job to an npz (no pickle).

    The per-frame 2-D point arrays are ragged (a frame with an all-NaN mesh
    contributes zero points), so they go in flat + offsets. They are stored as
    float64: the projections arrive as float32 and widening is exact, and
    matplotlib transforms in float64 either way, so the rasterised pixels are
    unchanged (asserted by tests/test_reproj_video_fast.py's byte comparison of
    the sequential vs subprocess renders).
    """
    import json
    meta = {k: job[k] for k in _JOB_KEYS if k in job}
    arrays = {"meta": np.frombuffer(json.dumps(meta).encode(), np.uint8)}
    for name in ("mesh2d", "kp2d"):
        parts = [np.asarray(a, np.float64).reshape(-1, 2) for a in job[name]]
        arrays[f"{name}_flat"] = (np.concatenate(parts) if parts
                                  else np.zeros((0, 2)))
        arrays[f"{name}_off"] = np.cumsum([0] + [len(p) for p in parts])
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def _load_job(path):
    import json
    with np.load(path) as z:
        job = json.loads(bytes(z["meta"]).decode())
        for name in ("mesh2d", "kp2d"):
            flat, off = z[f"{name}_flat"], z[f"{name}_off"]
            job[name] = [flat[off[t]:off[t + 1]] for t in range(len(off) - 1)]
    return job


def render_one_camera(job):
    """Decode one camera, draw its overlays, write its mp4 atomically.

    ``job``: ``cam, session_dir, start, T, H, W, mesh2d, kp2d, out_path, fps``
    (+ optional ``use_sync_plan``, default True).

    Returns ``(cam, out_path, err)``; ``err`` is None on success or a
    "``Type: message``" string, because a per-camera overlay is cosmetic QC
    and must never fail the bout.
    """
    import os
    cam = job["cam"]
    out_path = job["out_path"]
    try:
        from jarvis_jax.predict.synced_reader import load_plan, read_one_cam
        H, W = int(job["H"]), int(job["W"])
        session_dir = job["session_dir"]
        plan = load_plan(session_dir) if job.get("use_sync_plan", True) else None

        def _frames():
            # read_one_cam yields (frame_rgb|None, present); a dropped slot still
            # needs a same-size placeholder so the frame count stays in lockstep
            # with mesh2d/kp2d (imageio's writer rejects a size change).
            for fr, _present in read_one_cam(session_dir, cam, plan,
                                             int(job["start"]), int(job["T"])):
                yield fr if fr is not None else np.zeros((H, W, 3), np.uint8)

        # Same atomic-write contract as scripts/run_bout.py::atomic_write_mp4,
        # including its ".tmp.mp4" (not bare ".tmp") suffix: imageio picks its
        # backend by sniffing the URI extension, and a bare ".tmp" makes it fall
        # back to the wrong plugin and crash on the `fps` kwarg.
        tmp = (out_path[:-4] + ".tmp.mp4") if out_path.endswith(".mp4") else out_path + ".tmp"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        write_camera_video(tmp, frames_rgb_iter=_frames(),
                           mesh2d_by_frame=job["mesh2d"],
                           kp2d_by_frame=job["kp2d"], fps=int(job["fps"]))
        if not os.path.exists(tmp):
            # imageio never creates the file when zero frames are appended
            return cam, out_path, "no frames written (empty frame window?)"
        os.replace(tmp, out_path)
        return cam, out_path, None
    except Exception as exc:                            # noqa: BLE001 -- QC artifact
        return cam, out_path, f"{type(exc).__name__}: {exc}"


def render_camera_overlays(jobs, *, max_workers=4, tmp_dir=None):
    """Render several cameras' overlay videos, up to `max_workers` at a time.

    ``jobs``: one dict per camera, see ``render_one_camera``.
    Returns ``{cam: None | error_string}`` -- never raises for a failed camera.

    ``max_workers <= 1`` runs them in-process, sequentially: byte-identical
    (asserted in tests), and the escape hatch on a memory- or core-starved
    node. Above that, each camera is an isolated subprocess holding one decoded
    frame, one Agg canvas and one ffmpeg pipe (~300 MB).
    """
    jobs = list(jobs)
    if not jobs:
        return {}
    if max_workers is not None and max_workers <= 1:
        return {j["cam"]: render_one_camera(j)[2] for j in jobs}

    import os
    import subprocess
    import sys
    import tempfile
    # PYTHONPATH: the directory the `jarvis_jax` package lives in, so `-m`
    # resolves regardless of how the parent found it (installed, .pth, or a
    # sys.path insert the child does not inherit).
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(
               [pkg_root] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
           "MPLBACKEND": "Agg"}
    out = {}
    with tempfile.TemporaryDirectory(dir=tmp_dir, prefix="overlay_jobs_") as td:
        pending = []
        for i, job in enumerate(jobs):
            jp = os.path.join(td, f"job{i:03d}.npz")
            _save_job(job, jp)
            pending.append((job["cam"], jp))
        running = []                       # [(cam, Popen)]
        queue = list(pending)
        while queue or running:
            while queue and len(running) < max_workers:
                cam, jp = queue.pop(0)
                proc = subprocess.Popen(
                    [sys.executable, "-m", "jarvis_jax.tracking.reproj_video", jp],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True)
                running.append((cam, proc))
            cam, proc = running.pop(0)      # jobs are equal-length; FIFO is fine
            log = proc.communicate()[0] or ""
            if proc.returncode != 0:
                out[cam] = f"render subprocess exit {proc.returncode}: {log.strip()[-500:]}"
            else:
                err = None
                for line in log.splitlines():
                    if line.startswith("RENDER_ERROR "):
                        err = line[len("RENDER_ERROR "):]
                out[cam] = err
    return out


def _cli(argv):
    """``python -m jarvis_jax.tracking.reproj_video <job.npz>`` -- render the
    one camera described by the job file. Prints ``RENDER_ERROR <msg>`` and
    still exits 0 for an in-render failure (the parent turns that into a
    per-camera warning); a nonzero exit means the worker itself broke."""
    if len(argv) != 1:
        print("usage: python -m jarvis_jax.tracking.reproj_video <job.npz>")
        return 2
    cam, out_path, err = render_one_camera(_load_job(argv[0]))
    if err:
        print(f"RENDER_ERROR {err}")
    else:
        print(f"RENDER_OK {cam} -> {out_path}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(_cli(_sys.argv[1:]))
