"""Building MuJoCo cameras that match the real rig.

The rig's calibration is AFFINE -- every camera matrix's third row is
[0,0,0,1], apparent size is exactly depth-independent, and an RQ decomposition
into K/R/C is singular. The MuJoCo counterpart is therefore an ORTHOGRAPHIC
camera, whose fovy is the FULL visible height in length units (verified against
a rendered scene: fovy=2.0 showed 1.984 units).
"""
from __future__ import annotations

import numpy as np
import pytest

from viz.core.mjcam import (affine_camera_rows, mat_to_quat,
                            mujoco_camera_from_affine,
                            project_with_mujoco_camera, similarity_from_points)

IMG = (1936, 448)


def _affine_cam(right, down, origin, ku=8.0, kv=8.0):
    """A (4,3) camera in the `ph @ M` convention: u = ku*right.(X-o) + W/2."""
    W, H = IMG
    m0 = ku * np.asarray(right, float)
    m1 = kv * np.asarray(down, float)
    M = np.zeros((4, 3))
    M[:3, 0] = m0; M[3, 0] = W / 2.0 - m0 @ origin
    M[:3, 1] = m1; M[3, 1] = H / 2.0 - m1 @ origin
    M[3, 2] = 1.0
    return M


def _project(M, X):
    ph = np.concatenate([np.asarray(X, float), np.ones((len(X), 1))], axis=1)
    p = ph @ M
    return p[:, :2] / p[:, 2:3]


# ------------------------------------------------------------------ parsing

def test_a_perspective_camera_is_rejected_not_silently_mishandled():
    M = _affine_cam([1, 0, 0], [0, 1, 0], np.zeros(3))
    M[:3, 2] = [0.01, 0.02, 0.03]          # a real depth divide
    with pytest.raises(ValueError, match="not affine"):
        affine_camera_rows(M)


def test_affine_rows_round_trip():
    M = _affine_cam([1, 0, 0], [0, 1, 0], np.array([3.0, 4.0, 5.0]))
    m0, o0, m1, o1 = affine_camera_rows(M)
    X = np.random.default_rng(0).normal(size=(20, 3))
    uv = np.column_stack([X @ m0 + o0, X @ m1 + o1])
    assert np.allclose(uv, _project(M, X))


# --------------------------------------------------------------- similarity

def test_similarity_recovers_a_known_transform():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(60, 3))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    s_true, t_true = 7.3, np.array([10.0, -4.0, 2.0])
    B = s_true * (Q @ A.T).T + t_true
    s, R, t = similarity_from_points(A, B)
    assert s == pytest.approx(s_true, rel=1e-6)
    assert np.allclose(R, Q, atol=1e-8)
    assert np.allclose(t, t_true, atol=1e-6)


def test_similarity_never_returns_a_reflection():
    """A mirrored fit must come back as a proper rotation: det(R) = +1."""
    rng = np.random.default_rng(2)
    A = rng.normal(size=(50, 3))
    B = A * np.array([1.0, 1.0, -1.0])       # reflected
    _, R, _ = similarity_from_points(A, B)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)


def test_mat_to_quat_round_trips():
    rng = np.random.default_rng(3)
    for _ in range(20):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Q *= np.sign(np.linalg.det(Q))
        q = mat_to_quat(Q)
        w, x, y, z = q
        Rb = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(Rb, Q, atol=1e-8)


# ------------------------------------------------------------------ camera

@pytest.mark.parametrize("right,down", [
    ([1, 0, 0], [0, 0, -1]),
    ([0, 1, 0], [0, 0, -1]),
    ([0.6, 0.8, 0], [0, 0, -1]),
])
def test_built_camera_reproduces_the_affine_projection(right, down):
    """The whole point: MuJoCo must draw a point where the real camera does."""
    rng = np.random.default_rng(4)
    origin = np.array([2.0, 3.0, 1.0])
    M = _affine_cam(right, down, origin)
    X = rng.normal(scale=0.05, size=(40, 3)) + origin
    s, R, t = 1.0, np.eye(3), np.zeros(3)          # model frame == world frame
    pos, quat, fovy = mujoco_camera_from_affine(M, IMG, s, R, t, X.mean(0))
    got = project_with_mujoco_camera(X, pos, quat, fovy, IMG)
    want = _project(M, X)
    assert np.max(np.linalg.norm(got - want, axis=1)) < 1e-6


def test_it_holds_through_a_model_to_world_similarity():
    """The MuJoCo scene is in model units; the calibration is in world mm."""
    rng = np.random.default_rng(5)
    origin = np.array([50.0, 40.0, 60.0])
    M = _affine_cam([1, 0, 0], [0, 0, -1], origin)
    Xm = rng.normal(scale=0.05, size=(40, 3))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3))); Q *= np.sign(np.linalg.det(Q))
    s, R, t = 86.0, Q, origin
    Xw = s * (R @ Xm.T).T + t
    pos, quat, fovy = mujoco_camera_from_affine(M, IMG, s, R, t, Xm.mean(0))
    got = project_with_mujoco_camera(Xm, pos, quat, fovy, IMG)
    want = _project(M, Xw)
    assert np.max(np.linalg.norm(got - want, axis=1)) < 1e-4


def test_fovy_is_the_full_visible_height():
    M = _affine_cam([1, 0, 0], [0, 0, -1], np.zeros(3), ku=8.0, kv=8.0)
    _, _, fovy = mujoco_camera_from_affine(M, IMG, 1.0, np.eye(3), np.zeros(3),
                                           np.zeros(3))
    assert fovy == pytest.approx(IMG[1] / 8.0)


def test_standoff_distance_cannot_change_what_is_drawn():
    """Orthographic: moving the camera along its own axis must not rescale."""
    M = _affine_cam([1, 0, 0], [0, 0, -1], np.zeros(3))
    X = np.random.default_rng(6).normal(scale=0.05, size=(30, 3))
    a = project_with_mujoco_camera(
        X, *mujoco_camera_from_affine(M, IMG, 1.0, np.eye(3), np.zeros(3),
                                      X.mean(0), back_off=1.0), IMG)
    b = project_with_mujoco_camera(
        X, *mujoco_camera_from_affine(M, IMG, 1.0, np.eye(3), np.zeros(3),
                                      X.mean(0), back_off=50.0), IMG)
    assert np.allclose(a, b, atol=1e-9)
