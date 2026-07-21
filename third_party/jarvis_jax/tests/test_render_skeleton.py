"""Unit tests for skeleton-edge drawing added to the bout reprojection
render driver (render_bout_reproj.skeleton_segments + reproj_video's
draw_overlay_frame mesh_edges/kp_edges). CPU-only, fast, no real video
encode/decode -- Agg-rasterizes a small synthetic frame only.
"""
import numpy as np

from jarvis_jax.scripts.render_bout_reproj import skeleton_segments


# --------------------------------------------------------------------------
# skeleton_segments -- pure numpy, the "which edges get drawn" logic
# --------------------------------------------------------------------------
def test_skeleton_segments_basic_pairing():
    pts2d = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    valid = np.array([True, True, True])
    ei = np.array([0, 1])
    ej = np.array([1, 2])
    segs = skeleton_segments(pts2d, valid, ei, ej)
    assert segs.shape == (2, 2, 2)
    np.testing.assert_allclose(segs[0], [[0.0, 0.0], [10.0, 0.0]])
    np.testing.assert_allclose(segs[1], [[10.0, 0.0], [10.0, 10.0]])


def test_skeleton_segments_skips_nan_endpoint():
    # joint 1 is NaN (e.g. dropped by project_and_filter's NaN check) ->
    # its "valid" entry is False; edge (0,1) referencing it must be skipped,
    # while edge (0,2) between two valid joints is kept.
    pts2d = np.array([[0.0, 0.0], [np.nan, np.nan], [10.0, 10.0]])
    valid = np.array([True, False, True])
    ei = np.array([0, 0])
    ej = np.array([1, 2])
    segs = skeleton_segments(pts2d, valid, ei, ej)
    assert segs.shape == (1, 2, 2)
    np.testing.assert_allclose(segs[0], [[0.0, 0.0], [10.0, 10.0]])


def test_skeleton_segments_skips_low_conf_endpoint():
    # joint 1 is finite but below min_conf -> also marked invalid upstream;
    # both edges touching it must be dropped.
    pts2d = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    valid = np.array([True, False, True])
    ei = np.array([0, 1])
    ej = np.array([1, 2])
    segs = skeleton_segments(pts2d, valid, ei, ej)
    assert segs.shape == (0, 2, 2)


def test_skeleton_segments_no_edges_returns_empty():
    pts2d = np.zeros((3, 2))
    valid = np.array([True, True, True])
    segs = skeleton_segments(pts2d, valid, np.zeros(0, dtype=np.int64),
                             np.zeros(0, dtype=np.int64))
    assert segs.shape == (0, 2, 2)


def test_skeleton_segments_all_valid_keeps_all_edges():
    pts2d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    valid = np.array([True, True, True, True])
    ei = np.array([0, 1, 2])
    ej = np.array([1, 2, 3])
    segs = skeleton_segments(pts2d, valid, ei, ej)
    assert segs.shape == (3, 2, 2)


# --------------------------------------------------------------------------
# draw_overlay_frame actually rasterizes the given segments as lines (not
# just dots) -- and skips segments the caller never gave it (the low-conf/
# NaN-endpoint-skip contract is enforced by skeleton_segments upstream;
# here we confirm the rasterizer draws exactly what it's handed).
# --------------------------------------------------------------------------
def test_draw_overlay_frame_draws_edge_pixels():
    from jarvis_jax.tracking.reproj_video import draw_overlay_frame

    H, W = 40, 60
    raw = np.zeros((H, W, 3), np.uint8)
    # a horizontal cyan "bone" segment along y=20, well clear of image edges
    mesh_edges = np.array([[[5.0, 20.0], [55.0, 20.0]]])
    # a lone magenta kp dot far away, so it can't bleed into our edge check
    kp2d = np.array([[6.0, 35.0]])

    out_no_edges = draw_overlay_frame(raw, mesh2d=np.zeros((0, 2)), kp2d=kp2d)
    out_edges = draw_overlay_frame(raw, mesh2d=np.zeros((0, 2)), kp2d=kp2d,
                                   mesh_edges=mesh_edges, edge_lw=3.0)

    assert out_no_edges.dtype == np.uint8 and out_edges.dtype == np.uint8
    assert out_no_edges.shape == (H, W, 3) and out_edges.shape == (H, W, 3)

    # nothing drawn near (x=30, y=20) without the edge -> stays black
    assert out_no_edges[20, 30].max() == 0
    # with the edge, the segment's midpoint pixel lights up cyan-ish
    # (green+blue channels dominate red)
    mid = out_edges[20, 30].astype(int)
    assert mid[1] + mid[2] > mid[0] + 40

    # a point untouched by either the edge or the (far-away) dot stays black
    # in both renders
    assert out_edges[2, 2].max() == 0
    assert out_no_edges[2, 2].max() == 0


def test_draw_overlay_frame_kp_edges_are_magenta():
    from jarvis_jax.tracking.reproj_video import draw_overlay_frame

    H, W = 40, 60
    raw = np.zeros((H, W, 3), np.uint8)
    kp_edges = np.array([[[5.0, 10.0], [55.0, 10.0]]])
    out = draw_overlay_frame(raw, mesh2d=np.zeros((0, 2)), kp2d=np.zeros((0, 2)),
                             kp_edges=kp_edges, edge_lw=3.0)
    mid = out[10, 30].astype(int)
    # magenta ~= (255, 0, 255): red+blue dominate green
    assert mid[0] + mid[2] > mid[1] + 40


def test_draw_overlay_frame_skips_edge_never_passed_in():
    """End-to-end: skeleton_segments drops the low-conf-endpoint edge, and
    draw_overlay_frame -- given only what skeleton_segments produced -- draws
    nothing where that dropped edge would have been."""
    from jarvis_jax.tracking.reproj_video import draw_overlay_frame

    H, W = 40, 60
    raw = np.zeros((H, W, 3), np.uint8)
    pts2d = np.array([[5.0, 20.0], [55.0, 20.0]])   # would draw across y=20
    valid = np.array([True, False])                  # endpoint 1 invalid
    ei = np.array([0])
    ej = np.array([1])

    segs = skeleton_segments(pts2d, valid, ei, ej)
    assert segs.shape == (0, 2, 2)

    out = draw_overlay_frame(raw, mesh2d=np.zeros((0, 2)), kp2d=np.zeros((0, 2)),
                             mesh_edges=segs, edge_lw=3.0)
    # exclude the last pixel row: a pre-existing Agg-canvas edge artifact
    # (present even with ZERO overlays of any kind -- verified separately,
    # unrelated to skeleton edges) always renders it white.
    assert out[:-1].max() == 0
