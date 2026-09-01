"""The overlay renderer's speedups must not move a single output pixel.

reproj_video.py was changed three ways to cut the 11m25s per-bout overlay
render (7 cameras x 2007 frames): a persistent Agg figure instead of one per
frame, a decode-prefetch thread, and one process per camera. All three are
supposed to be PIXEL-IDENTICAL -- same frames, same order, same rasteriser --
so that is what is asserted here, with `==` and not a similarity metric.
"""
import numpy as np
import pytest

from jarvis_jax.tracking.reproj_video import (
    OverlayRenderer, _prefetch, draw_overlay_frame, write_camera_video)

H, W = 64, 120


def _ref_draw(raw, mesh2d, kp2d, contour=None, mesh_edges=None, kp_edges=None):
    """The pre-2026-09-01 draw_overlay_frame, verbatim: a fresh figure per
    frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    raw = np.asarray(raw)
    if raw.dtype != np.uint8:
        raw = np.clip(raw * (255.0 if raw.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    h, w = raw.shape[:2]
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(raw)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    if contour is not None and len(contour):
        cc = np.asarray(contour)
        ax.plot(cc[:, 0], cc[:, 1], c="yellow", lw=0.8)
    if mesh_edges is not None and len(mesh_edges):
        ax.add_collection(LineCollection(np.asarray(mesh_edges), colors="cyan",
                                         linewidths=1.2))
    if kp_edges is not None and len(kp_edges):
        ax.add_collection(LineCollection(np.asarray(kp_edges), colors="magenta",
                                         linewidths=1.2))
    if len(mesh2d):
        m = np.asarray(mesh2d)
        ax.scatter(m[:, 0], m[:, 1], s=2, c="cyan", linewidths=0)
    if len(kp2d):
        k = np.asarray(kp2d)
        ax.scatter(k[:, 0], k[:, 1], s=10, c="magenta", linewidths=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    bh, bw = buf.shape[:2]
    if (bh, bw) != (h, w):
        out = np.zeros((h, w, 3), np.uint8)
        out[:min(h, bh), :min(w, bw)] = buf[:min(h, bh), :min(w, bw)]
        buf = out
    return buf.astype(np.uint8)


def _frames(n=12, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        yield dict(
            raw=rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
            mesh2d=rng.uniform(-10, W + 10, (int(rng.integers(0, 40)), 2)),
            kp2d=rng.uniform(0, W, (int(rng.integers(0, 8)), 2)),
            contour=None if i % 3 else rng.uniform(0, W, (9, 2)),
            mesh_edges=None if i % 2 else rng.uniform(0, W, (6, 2, 2)),
            kp_edges=None if i % 4 else rng.uniform(0, W, (3, 2, 2)),
        )


def test_persistent_renderer_is_pixel_identical():
    """A REUSED figure must rasterise the same bytes as a fresh one -- including
    frames with zero mesh points, zero keypoints, and no contour/edges (those
    artists exist but hold no data, instead of not existing)."""
    r = OverlayRenderer(H, W)
    try:
        for i, f in enumerate(_frames()):
            ref = _ref_draw(**f)
            got = r.draw(f["raw"], f["mesh2d"], f["kp2d"], contour=f["contour"],
                         mesh_edges=f["mesh_edges"], kp_edges=f["kp_edges"])
            assert got.shape == ref.shape
            n_diff = int((got != ref).sum())
            assert n_diff == 0, f"frame {i}: {n_diff} channel values differ"
    finally:
        r.close()


def test_draw_overlay_frame_still_matches_its_old_self():
    f = next(iter(_frames(1, seed=3)))
    assert (draw_overlay_frame(f["raw"], f["mesh2d"], f["kp2d"],
                               contour=f["contour"], mesh_edges=f["mesh_edges"],
                               kp_edges=f["kp_edges"]) == _ref_draw(**f)).all()


def test_prefetch_preserves_order_and_values():
    src = list(range(50))
    assert list(_prefetch(iter(src), 4)) == src
    assert list(_prefetch(iter([]), 4)) == []


def test_prefetch_reraises_producer_exception():
    def boom():
        yield 1
        raise RuntimeError("decode failed at frame 1")
    with pytest.raises(RuntimeError, match="decode failed"):
        list(_prefetch(boom(), 4))


def test_prefetch_on_off_gives_the_same_video(tmp_path):
    """prefetch is a scheduling change only: the mp4 bytes must be identical."""
    rng = np.random.default_rng(5)
    raws = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(8)]
    mesh = [rng.uniform(0, W, (20, 2)) for _ in range(8)]
    kp = [rng.uniform(0, W, (5, 2)) for _ in range(8)]
    paths = []
    for pf in (0, 8):
        p = tmp_path / f"pf{pf}.mp4"
        write_camera_video(str(p), frames_rgb_iter=iter(raws),
                           mesh2d_by_frame=mesh, kp2d_by_frame=kp,
                           fps=10, prefetch=pf)
        paths.append(p)
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_render_camera_overlays_sequential_matches_parallel(tmp_path):
    """Rendering the cameras in worker processes must give byte-identical mp4s
    to rendering them in-process (max_workers=1)."""
    import imageio
    from jarvis_jax.tracking.reproj_video import render_camera_overlays
    rng = np.random.default_rng(11)
    n_frames = 6
    session = tmp_path / "session"
    session.mkdir()
    cams = ["CamA", "CamB", "CamC"]
    for cam in cams:
        with imageio.get_writer(str(session / f"{cam}.mp4"), fps=10,
                                codec="libx264", pixelformat="yuv420p",
                                macro_block_size=16) as v:
            for _ in range(n_frames):
                v.append_data(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))

    def jobs(outdir):
        out = []
        for cam in cams:
            out.append(dict(cam=cam, session_dir=str(session), start=0,
                            T=n_frames, H=H, W=W,
                            mesh2d=[rng.uniform(0, W, (15, 2)) for _ in range(n_frames)],
                            kp2d=[rng.uniform(0, W, (4, 2)) for _ in range(n_frames)],
                            out_path=str(outdir / f"{cam}.mp4"), fps=10))
        return out

    seq_dir = tmp_path / "seq"; seq_dir.mkdir()
    par_dir = tmp_path / "par"; par_dir.mkdir()
    js = jobs(seq_dir)
    errs_seq = render_camera_overlays(js, max_workers=1)
    js_par = [dict(j, out_path=str(par_dir / f"{j['cam']}.mp4")) for j in js]
    errs_par = render_camera_overlays(js_par, max_workers=3)
    assert set(errs_seq) == set(errs_par) == set(cams)
    assert all(v is None for v in errs_seq.values()), errs_seq
    assert all(v is None for v in errs_par.values()), errs_par
    for cam in cams:
        a = (seq_dir / f"{cam}.mp4").read_bytes()
        b = (par_dir / f"{cam}.mp4").read_bytes()
        assert a == b, f"{cam}: parallel render differs from sequential"


def test_render_camera_overlays_reports_error_without_raising(tmp_path):
    """A broken job comes back as an error STRING (a cosmetic QC render must
    never fail the bout) and leaves no half-written output at the real path."""
    from jarvis_jax.tracking.reproj_video import render_camera_overlays
    import imageio
    rng = np.random.default_rng(13)
    (tmp_path / "s").mkdir()
    with imageio.get_writer(str(tmp_path / "s" / "CamX.mp4"), fps=10,
                            codec="libx264", pixelformat="yuv420p",
                            macro_block_size=16) as v:
        for _ in range(4):
            v.append_data(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
    out = tmp_path / "broken.mp4"
    errs = render_camera_overlays(
        [dict(cam="CamX", session_dir=str(tmp_path / "s"), start=0, T=4, H=H, W=W,
              mesh2d=[np.zeros((0, 2))],           # too short for T=4 -> IndexError
              kp2d=[np.zeros((0, 2))] * 4,
              out_path=str(out), fps=10)],
        max_workers=1)
    assert errs["CamX"] is not None and "IndexError" in errs["CamX"], errs
    assert not out.exists(), "a failed render must not publish a partial mp4"


def test_missing_source_video_still_writes_black_frames(tmp_path):
    """Pre-existing contract (unchanged): read_one_cam yields None for a slot it
    cannot read and the caller substitutes a black placeholder, so a missing
    camera video produces an all-black overlay rather than an error. Pinned here
    because the parallel worker now owns that placeholder logic."""
    from jarvis_jax.tracking.reproj_video import render_camera_overlays
    out = tmp_path / "black.mp4"
    errs = render_camera_overlays(
        [dict(cam="Nope", session_dir=str(tmp_path), start=0, T=3, H=H, W=W,
              mesh2d=[np.zeros((0, 2))] * 3, kp2d=[np.zeros((0, 2))] * 3,
              out_path=str(out), fps=10)],
        max_workers=1)
    assert errs["Nope"] is None and out.exists()
