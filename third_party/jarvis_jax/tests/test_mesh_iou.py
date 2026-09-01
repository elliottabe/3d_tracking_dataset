# tests/test_mesh_iou.py -- was test_run_silhouette_polish.py.
# The run_polish driver was deleted with the silhouette; these four IoU tests
# came with the helpers, which the live per-bout QC report still calls.
import os
import numpy as np
import pytest


def test_iou_of_projected_verts_known_overlap():
    from jarvis_jax.tracking.mesh_iou import iou_of_projected_verts
    # ref mask: filled 10x10 square in a 20x20 image.
    ref = np.zeros((20, 20), dtype=bool); ref[5:15, 5:15] = True
    # projected verts exactly filling the same square -> IoU == 1.
    yy, xx = np.mgrid[5:15, 5:15]
    verts2d = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)  # (x,y)
    iou = iou_of_projected_verts(verts2d, (20, 20), ref)
    assert abs(iou - 1.0) < 1e-9
    # a disjoint square -> IoU == 0.
    verts_off = verts2d + np.array([10.0, 0.0])  # shift x by 10 -> no overlap kept in-bounds
    verts_off = verts_off[(verts_off[:, 0] < 20)]
    iou0 = iou_of_projected_verts(verts_off, (20, 20), ref)
    assert iou0 < 0.2


def test_iou_ignores_out_of_bounds_verts():
    from jarvis_jax.tracking.mesh_iou import iou_of_projected_verts
    ref = np.zeros((10, 10), dtype=bool); ref[2:8, 2:8] = True
    verts = np.array([[100.0, 100.0], [-5.0, -5.0], [4.0, 4.0]])  # 2 OOB, 1 inside
    iou = iou_of_projected_verts(verts, (10, 10), ref)
    assert 0.0 <= iou <= 1.0  # no crash on OOB; finite


def test_soft_iou_of_verts_higher_when_verts_fill_mask():
    """Eval-only soft-IoU (baseline-comparable): verts densely filling the mask
    give a higher soft-IoU than verts sitting outside it. Bounded in [0,1]."""
    from jarvis_jax.tracking.mesh_iou import soft_iou_of_verts
    mask = np.zeros((30, 30), dtype=bool); mask[8:22, 8:22] = True
    yy, xx = np.mgrid[8:22, 8:22]
    inside = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)  # (x,y)
    outside = inside + np.array([25.0, 0.0])  # shifted well off the mask
    s_in = soft_iou_of_verts(inside, mask, sigma=1.3)
    s_out = soft_iou_of_verts(outside, mask, sigma=1.3)
    assert 0.0 <= s_in <= 1.0 and 0.0 <= s_out <= 1.0
    assert s_in > s_out
    # empty mask -> 0 (no crash)
    assert soft_iou_of_verts(inside, np.zeros((30, 30), dtype=bool), sigma=1.3) == 0.0


def test_filled_tri_iou_perfect_and_partial():
    # a single triangle covering a known region vs a matching mask
    verts = np.array([[10, 10], [40, 10], [10, 40]], float)
    faces = np.array([[0, 1, 2]])
    import cv2
    from jarvis_jax.tracking.mesh_iou import filled_tri_iou
    ref = np.zeros((50, 50), np.uint8)
    cv2.fillPoly(ref, [verts.astype(np.int32)], 1)
    ref = ref.astype(bool)
    assert filled_tri_iou(verts, faces, ref.shape, ref) > 0.99
    empty = np.zeros((50, 50), bool)
    assert filled_tri_iou(verts, faces, empty.shape, empty) == 0.0


