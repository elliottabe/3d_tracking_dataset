"""Affine (telecentric / orthographic) camera math.

A telecentric camera's 3x4 DLT matrix has 3rd row [0,0,0,1], so projection is
affine:  uv = M @ X + t,  M = P[:2,:3] (2x3), t = P[:2,3] (2,).
We factor M = K2 @ R[:2,:] where K2 is 2x2 upper-triangular (magnification /
shear / aspect) and R in SO(3) is the camera orientation. This separates the
"intrinsic" shape (K2) from the pose (R, t) for bundle adjustment.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import rq


def factor_affine(P: np.ndarray):
    """Factor a 3x4 affine DLT matrix into (K2, R, t). See module docstring."""
    P = np.asarray(P, dtype=np.float64)
    assert P.shape == (3, 4)
    M = P[:2, :3]                      # (2,3)
    t = P[:2, 3].copy()               # (2,)
    K2, Q = rq(M, mode="economic")    # M = K2 @ Q ; K2 (2,2) upper-tri, Q (2,3) orthonormal rows
    # Normalize signs so K2 has a positive diagonal (absorb sign flips into Q).
    signs = np.sign(np.diag(K2))
    signs[signs == 0] = 1.0
    S = np.diag(signs)
    K2 = K2 @ S
    Q = S @ Q
    # Extend the two orthonormal rows to a full right-handed rotation.
    r2 = np.cross(Q[0], Q[1])
    R = np.vstack([Q, r2])
    if np.linalg.det(R) < 0:         # enforce det(R) = +1
        R[2] = -R[2]
    return K2, R, t


def reconstruct_affine(K2: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse of factor_affine: build the 3x4 affine DLT matrix."""
    M = np.asarray(K2, np.float64) @ np.asarray(R, np.float64)[:2, :]   # (2,3)
    P = np.zeros((3, 4), dtype=np.float64)
    P[:2, :3] = M
    P[:2, 3] = np.asarray(t, np.float64)
    P[2, 3] = 1.0
    return P


def project_affine(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Affine projection uv = P[:2,:3] @ X + P[:2,3] (no perspective divide)."""
    P = np.asarray(P, np.float64); X = np.asarray(X, np.float64)
    return X @ P[:2, :3].T + P[:2, 3]


def project_from_params(K2, R_mat, t, X):
    """JAX-friendly affine projection from (K2 (2,2), R_mat (3,3), t (2,), X (...,3))."""
    M = K2 @ R_mat[:2, :]                      # (2,3)
    return X @ M.T + t                         # (...,2)
