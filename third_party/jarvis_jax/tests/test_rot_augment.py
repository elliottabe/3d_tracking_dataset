import numpy as np
import pytest

from jarvis_jax.data.rot_augment import sample_rotation, augment_sample


def test_sampled_rotations_are_orthogonal_with_unit_determinant():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = sample_rotation(rng)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_tilt_is_bounded():
    """Full SO(3) would destroy the gravity prior ('legs point down'), which is
    real free information for a fly on the arena floor."""
    rng = np.random.default_rng(0)
    up = np.array([0.0, 0.0, 1.0])
    for _ in range(200):
        R = sample_rotation(rng, tilt_deg=30.0, axis=up)
        cos = float(np.clip((R @ up) @ up, -1, 1))
        assert np.degrees(np.arccos(cos)) <= 30.0 + 1e-4


def test_zero_tilt_keeps_the_gravity_axis_fixed():
    rng = np.random.default_rng(0)
    up = np.array([0.0, 0.0, 1.0])
    R = sample_rotation(rng, tilt_deg=0.0, axis=up)
    np.testing.assert_allclose(R @ up, up, atol=1e-6)


def test_augment_rotates_labels_about_the_grid_centre():
    """The label must move with the grid, or the net learns a wrong mapping."""
    kp = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], np.float32)
    c = np.array([0.0, 0.0, 0.0], np.float32)
    th = np.pi / 2
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
    out = augment_sample({"kp3d": kp, "center3D": c, "vis": np.ones(2, bool)}, R)
    np.testing.assert_allclose(out["kp3d"][0], [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out["kp3d"][1], [-2.0, 0.0, 0.0], atol=1e-5)


def test_augment_preserves_pairwise_distances():
    """A rotation is rigid — bone lengths must not change."""
    rng = np.random.default_rng(1)
    kp = rng.normal(0, 5, (50, 3)).astype(np.float32)
    c = np.zeros(3, np.float32)
    R = sample_rotation(rng)
    out = augment_sample({"kp3d": kp, "center3D": c, "vis": np.ones(50, bool)}, R)
    d0 = np.linalg.norm(kp[1:] - kp[:-1], axis=-1)
    d1 = np.linalg.norm(out["kp3d"][1:] - out["kp3d"][:-1], axis=-1)
    np.testing.assert_allclose(d0, d1, atol=1e-4)


def test_invisible_keypoints_are_left_alone():
    kp = np.zeros((3, 3), np.float32)
    vis = np.array([True, False, True])
    rng = np.random.default_rng(0)
    out = augment_sample({"kp3d": kp, "center3D": np.zeros(3, np.float32),
                          "vis": vis}, sample_rotation(rng))
    np.testing.assert_array_equal(out["kp3d"][1], np.zeros(3, np.float32))
