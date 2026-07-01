import numpy as np
from jarvis_jax.cse.affine_camera import factor_affine, reconstruct_affine

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
