import numpy as np
import jax.numpy as jnp
from jarvis_jax.geometry.center3d import quantize_center3d, mask_centroids, centroids_to_fullpx, triangulate_dlt_batched, estimate_center3d_from_masks
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.data.v3_3d import V3FramesetDataset

CALIB = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3/calib_params/2026_01_13_18_47_45"


def test_quantize_matches_v3_recipe():
    # midrange of non-zero coords, int-snapped to the grid (gs=1)
    pts = np.array([[10.4, -3.0, 6.6], [2.0, -1.0, 8.0]], dtype=np.float32)
    # axis0 nz: max10.4 min2 -> mid=6.2 -> int=6 ; axis1: max-1 min-3 -> mid=-2 -> int=-2
    # axis2: max8 min6.6 -> mid=7.3 -> int=7
    out = quantize_center3d(pts, grid_spacing=1)
    assert out.tolist() == [6.0, -2.0, 7.0]


def test_quantize_single_point_is_lattice_snap():
    p = np.array([[12.7, -4.2, 0.9]], dtype=np.float32)  # one triangulated centroid
    out = quantize_center3d(p, grid_spacing=1)            # mid = p itself -> int(p)
    assert out.tolist() == [12.0, -4.0, 0.0]


def test_quantize_empty_returns_zeros():
    out = quantize_center3d(np.zeros((0, 3), np.float32), grid_spacing=1)
    assert out.tolist() == [0.0, 0.0, 0.0]


def test_mask_centroids_single_blob():
    crops = jnp.zeros((1, 2, 448, 448, 4), jnp.uint8)
    crops = crops.at[0, 0, 100, 200, 3].set(1)   # cam0 mask pixel at (y=100,x=200)
    crops = crops.at[0, 1, 50, 60, 3].set(1)     # cam1 at (y=50,x=60)
    cent, valid = mask_centroids(crops)
    assert bool(valid[0, 0]) and bool(valid[0, 1])
    assert jnp.allclose(cent[0, 0], jnp.array([200.0, 100.0]))  # [x, y]
    assert jnp.allclose(cent[0, 1], jnp.array([60.0, 50.0]))


def test_mask_centroids_empty_marks_invalid():
    crops = jnp.zeros((1, 1, 448, 448, 4), jnp.uint8)  # all-zero mask
    cent, valid = mask_centroids(crops)
    assert not bool(valid[0, 0])


def test_centroids_to_fullpx():
    cent = jnp.array([[[224.0, 224.0]]])      # crop center
    centerHM = jnp.array([[[500.0, 300.0]]])  # crop center in full px
    full = centroids_to_fullpx(cent, centerHM, crop=448)
    assert jnp.allclose(full[0, 0], jnp.array([500.0, 300.0]))  # 224 + 500 - 224


def test_batched_dlt_recovers_known_point():
    rt = ReprojectionTool(CALIB)
    cm = rt.camera_matrices.astype(np.float32)        # (nc,4,3)
    nc = cm.shape[0]
    X = np.array([3.0, -5.0, 12.0])                   # known 3D point
    pts2d = rt.reproject_point(X)                     # (nc,2) full px
    B = 1
    cmb = cm[None]                                    # (1,nc,4,3)
    p2 = pts2d.astype(np.float32)[None]               # (1,nc,2)
    valid = np.ones((1, nc), bool)
    out = np.asarray(triangulate_dlt_batched(p2, cmb, valid))[0]
    assert np.linalg.norm(out - X) < 1e-2, out
    # matches the reference reconstruct_point
    ref = rt.reconstruct_point(pts2d)
    assert np.linalg.norm(out - ref) < 1e-3


def test_batched_dlt_drops_invalid_camera():
    rt = ReprojectionTool(CALIB)
    cm = rt.camera_matrices.astype(np.float32); nc = cm.shape[0]
    X = np.array([1.0, 2.0, 9.0]); pts2d = rt.reproject_point(X).astype(np.float32)
    valid = np.ones((1, nc), bool); valid[0, 0] = False     # drop cam 0
    p2 = pts2d[None].copy(); p2[0, 0] = 9999.0              # garbage in dropped cam
    out = np.asarray(triangulate_dlt_batched(p2, cm[None], valid))[0]
    assert np.linalg.norm(out - X) < 1e-1, out             # still recovered


ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"


def test_estimate_center3d_close_to_gt():
    ds = V3FramesetDataset(ROOT, "val")
    s = ds[0]
    crops4 = s["crops4"][None]            # (1,nc,448,448,4)
    centerHM = s["centerHM"][None].astype("float32")
    cams = s["cameraMatrices"][None].astype("float32")
    center3D, n_valid = estimate_center3d_from_masks(crops4, centerHM, cams)
    assert int(n_valid[0]) >= 2
    err = float(np.linalg.norm(np.asarray(center3D)[0] - s["center3D"]))
    # mask-centroid center vs GT keypoint-midrange center: expect within ~one ROI half-cube
    assert err < 24.0, f"center3D too far from GT: {err}"


def test_project_center_to_cameras_inverts_triangulation():
    from jarvis_jax.geometry.center3d import project_center_to_cameras
    CALIB = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3/calib_params/2026_01_13_18_47_45"
    rt = ReprojectionTool(CALIB)
    cm = rt.camera_matrices.astype(np.float32)            # (nc,4,3)
    X = np.array([4.0, -6.0, 11.0], np.float32)
    px = project_center_to_cameras(X, cm)                  # (nc,2)
    assert px.shape == (cm.shape[0], 2)
    back = np.asarray(triangulate_dlt_batched(px[None], cm[None], np.ones((1, cm.shape[0]), bool)))[0]
    assert np.linalg.norm(back - X) < 1e-2, back
