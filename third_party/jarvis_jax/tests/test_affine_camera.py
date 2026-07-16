import numpy as np
from jarvis_jax.tracking.affine_camera import factor_affine, reconstruct_affine

# A real telecentric calibration matrix (Cam2012630, 2026_03_18_15_31_22).
P_REAL = np.array([
    [8.1001000000000012, 0.0074869000000000012, -0.031773000000000003, -2.8279999999999998],
    [0.0093308000000000002, -8.0787999999999993, -0.17912, 462.77999999999997],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def test_factor_reconstruct_roundtrip():
    K2, R, t = factor_affine(P_REAL)
    P2 = reconstruct_affine(K2, R, t)
    assert np.allclose(P2, P_REAL, atol=1e-9), f"roundtrip mismatch:\n{P2}\nvs\n{P_REAL}"


def test_factor_R_is_rotation():
    K2, R, t = factor_affine(P_REAL)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "R not orthonormal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9), "det(R) != 1"


def test_factor_K2_upper_triangular_positive_diag():
    K2, R, t = factor_affine(P_REAL)
    assert abs(K2[1, 0]) < 1e-12, "K2 not upper triangular"
    assert K2[0, 0] > 0 and K2[1, 1] > 0, "K2 diagonal not positive"


def test_project_affine_matches_reprojection_tool(tmp_path):
    import cv2
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.affine_camera import project_affine
    # Write P_REAL as a one-camera calib dir and compare projections.
    d = tmp_path / "calib"; d.mkdir()
    fs = cv2.FileStorage(str(d / "Cam0001.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("projectionMatrix", P_REAL); fs.release()
    rt = ReprojectionTool(str(d))
    X = np.array([1.5, -0.7, 12.0])
    uv_tool = rt.reproject_point(X)[0]            # (2,)
    uv_aff = project_affine(P_REAL, X)            # (2,)
    assert np.allclose(uv_tool, uv_aff, atol=1e-9), f"{uv_tool} vs {uv_aff}"


def test_project_from_params_matches_project_affine():
    import jax.numpy as jnp
    from jarvis_jax.tracking.affine_camera import factor_affine, project_affine, project_from_params
    K2, R, t = factor_affine(P_REAL)
    X = np.array([[1.5, -0.7, 12.0], [0.2, 0.3, 9.0]])
    uv_np = project_affine(P_REAL, X)                                   # (2,2)
    uv_jx = np.asarray(project_from_params(jnp.asarray(K2), jnp.asarray(R),
                                           jnp.asarray(t), jnp.asarray(X)))
    assert np.allclose(uv_np, uv_jx, atol=1e-6), f"{uv_np} vs {uv_jx}"
