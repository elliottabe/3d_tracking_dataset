import numpy as np
import pytest

from jarvis_jax.data.rot_augment import sample_rotation, augment_sample, gravity_axis


def test_gravity_axis_recovers_shared_up_direction_from_a_high_agreement_rig():
    """All cameras agree 'up' (image-y) is +Z, with small per-camera noise.

    gravity_axis must recover the SHARED direction the rig agrees on, not the
    axis of maximum disagreement between cameras. A previous implementation
    took the SVD of the mean-CENTERED residuals, whose top singular vector is
    the axis cameras disagree about most -- for a 7-camera rig with 2% noise
    around +Z that returned an axis ~90 degrees off (cosine ~0.007 to +Z)."""
    rng = np.random.default_rng(0)
    n_cams = 7
    cam_mats = np.zeros((n_cams, 3, 4))
    for i in range(n_cams):
        y = np.array([0.0, 0.0, 1.0]) + rng.normal(0, 0.02, 3)
        y /= np.linalg.norm(y)
        cam_mats[i, :3, 1] = y
        cam_mats[i, :3, 0] = [1.0, 0.0, 0.0]
        cam_mats[i, :3, 2] = [0.0, 1.0, 0.0]

    axis = gravity_axis(cam_mats)
    cos = float(np.clip(axis @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
    degrees_off = np.degrees(np.arccos(cos))
    assert degrees_off <= 5.0, (
        f"recovered axis {axis} is {degrees_off:.1f} degrees off +Z; "
        "gravity_axis must return the rig's SHARED up direction"
    )


def test_gravity_axis_is_a_unit_vector_with_nonnegative_z():
    rng = np.random.default_rng(1)
    n_cams = 5
    cam_mats = np.zeros((n_cams, 3, 4))
    for i in range(n_cams):
        y = np.array([0.0, 0.0, -1.0]) + rng.normal(0, 0.01, 3)
        y /= np.linalg.norm(y)
        cam_mats[i, :3, 1] = y

    axis = gravity_axis(cam_mats)
    assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6)
    assert axis[2] >= 0.0
    # Cameras agree on -Z as image-y "up"; the returned axis is sign-normed
    # to have z >= 0, so it should land near +Z (the flipped shared axis).
    cos = float(np.clip(axis @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
    assert np.degrees(np.arccos(cos)) <= 5.0


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
    """A vis-ignoring implementation must fail this test.

    center3D and the invisible keypoint are both non-zero and non-coincident,
    so rotating the invisible row (instead of masking it) actually changes its
    value -- unlike the degenerate origin-about-origin case, which is a no-op
    under ANY rotation and so passes even a broken, vis-ignoring
    implementation."""
    kp = np.array([[5.0, 1.0, 2.0], [3.0, -4.0, 6.0], [0.0, 2.0, -1.0]], np.float32)
    vis = np.array([True, False, True])
    c = np.array([2.0, -1.0, 3.0], np.float32)
    rng = np.random.default_rng(0)
    R = sample_rotation(rng)
    out = augment_sample({"kp3d": kp.copy(), "center3D": c, "vis": vis}, R)
    np.testing.assert_array_equal(out["kp3d"][1], kp[1])
