# tests/test_run_silhouette_polish.py
import os
import numpy as np
import pytest


def test_iou_of_projected_verts_known_overlap():
    from jarvis_jax.cse.run_silhouette_polish import iou_of_projected_verts
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
    from jarvis_jax.cse.run_silhouette_polish import iou_of_projected_verts
    ref = np.zeros((10, 10), dtype=bool); ref[2:8, 2:8] = True
    verts = np.array([[100.0, 100.0], [-5.0, -5.0], [4.0, 4.0]])  # 2 OOB, 1 inside
    iou = iou_of_projected_verts(verts, (10, 10), ref)
    assert 0.0 <= iou <= 1.0  # no crash on OOB; finite


def test_soft_iou_of_verts_higher_when_verts_fill_mask():
    """Eval-only soft-IoU (baseline-comparable): verts densely filling the mask
    give a higher soft-IoU than verts sitting outside it. Bounded in [0,1]."""
    from jarvis_jax.cse.run_silhouette_polish import soft_iou_of_verts
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


def test_run_polish_is_importable_and_has_cli():
    import jarvis_jax.cse.run_silhouette_polish as m
    assert hasattr(m, "run_polish")
    assert hasattr(m, "main")
    assert callable(m.run_polish)


IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present (GPU-only heavy run)")
def test_run_polish_report_keys_smoke(tmp_path):
    """Tiny 2-frame smoke (CPU) exercising the report assembly. The scientific
    magnitude comes from the coordinator GPU run, not this test."""
    from jarvis_jax.cse.run_silhouette_polish import run_polish
    XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
    MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
    ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
    rep = run_polish(
        "2026_03_18_15_31_22", ik_h5=IK, model_xml=XML, mesh_npz=MESH, root=ROOT,
        split="val", n_points=32, silhouette_weight=0.3, max_frames=2, n_iter=10,
        out_dir=str(tmp_path),
    )
    for k in ("iou_before", "iou_after", "soft_iou_before", "soft_iou_after",
              "reproj_px_before", "reproj_px_after",
              "marker_resid_before", "marker_resid_after", "n_frames"):
        assert k in rep
    assert rep["n_frames"] == 2
