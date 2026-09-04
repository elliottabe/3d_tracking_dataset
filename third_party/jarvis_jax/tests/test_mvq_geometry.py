import numpy as np
import jax.numpy as jnp
import pytest

# a real telecentric calibration (Cam2012630, 2026_03_18_15_31_22), see tests/test_affine_camera.py
P_REAL = np.array([
    [8.1001, 0.0074869, -0.031773, -2.828],
    [0.0093308, -8.0788, -0.17912, 462.78],
    [0.0, 0.0, 0.0, 1.0]], np.float64)


def _cam_mats(n=3, seed=0):
    """(n,4,3) P.T stacks: P_REAL rotated about z by different yaws."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        th = rng.uniform(0, 2 * np.pi)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        P = P_REAL.copy(); P[:2, :3] = P[:2, :3] @ Rz
        out.append(P.T.astype(np.float32))
    return np.stack(out)


def test_affine_rows_and_projection_match_reprojection_tool_formula():
    from jarvis_jax.models.mvq.geometry import affine_rows, project_local, local_offset
    cm = _cam_mats()
    M, t = affine_rows(jnp.asarray(cm))
    assert M.shape == (3, 2, 3) and t.shape == (3, 2)
    X = np.array([1.5, -0.7, 12.0], np.float32)
    ph = np.concatenate([X, [1.0]])
    uv_ref = np.stack([(ph @ cm[c])[:2] / (ph @ cm[c])[2] for c in range(3)])   # ReprojectionTool math
    center = jnp.zeros(3); origin = jnp.zeros((3, 2))
    uv = project_local(jnp.asarray(X), M, local_offset(M, t, center, origin))
    np.testing.assert_allclose(np.asarray(uv), uv_ref, atol=1e-3)


def test_affine_rows_rejects_pinhole():
    from jarvis_jax.models.mvq.geometry import affine_rows
    cm = _cam_mats(1); cm[0, 2, 2] = 0.5    # P[2,2] != 0
    with pytest.raises(ValueError):
        affine_rows(jnp.asarray(cm))


def test_ray_from_pixel_projects_back_for_every_depth():
    from jarvis_jax.models.mvq.geometry import affine_rows, ray_from_pixel, project_local, local_offset
    M, t = affine_rows(jnp.asarray(_cam_mats()))
    tl = local_offset(M, t, jnp.array([3.0, -2.0, 5.0]), jnp.full((3, 2), 100.0))
    uv = jnp.asarray(np.random.default_rng(1).uniform(0, 448, size=(3, 5, 2)).astype(np.float32))
    p0, d = ray_from_pixel(M, tl, uv)
    assert p0.shape == (3, 5, 3) and d.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(d), axis=-1), 1.0, atol=1e-5)
    for s in (-30.0, 0.0, 17.5):
        X = p0 + s * d[:, None, :]                                    # (C,N,3)
        for c in range(3):
            back = project_local(X[c], M, tl)[:, c]                   # (N,2) in camera c
            np.testing.assert_allclose(np.asarray(back), np.asarray(uv[c]), atol=1e-2)


def test_warp_cameras_is_exact_for_affine_image_warp():
    from jarvis_jax.models.mvq.geometry import affine_rows, project_local, local_offset, warp_cameras
    M, t = affine_rows(jnp.asarray(_cam_mats()))
    tl = local_offset(M, t, jnp.zeros(3), jnp.zeros((3, 2)))
    th = 0.3; A = jnp.asarray(np.tile(0.9 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]), (3, 1, 1)))
    b = jnp.asarray(np.array([[5.0, -3.0]] * 3))
    X = jnp.asarray(np.random.default_rng(2).normal(size=(7, 3)).astype(np.float32) * 5)
    uv = project_local(X, M, tl)                                       # (7,C,2)
    uv_w = jnp.einsum("cij,ncj->nci", A, uv) + b                       # warp the 2D labels
    M2, tl2 = warp_cameras(M, tl, A, b)
    np.testing.assert_allclose(np.asarray(project_local(X, M2, tl2)), np.asarray(uv_w), atol=1e-3)


def test_rotate_world_keeps_projection():
    from jarvis_jax.models.mvq.geometry import affine_rows, project_local, local_offset, rotate_world
    M, t = affine_rows(jnp.asarray(_cam_mats()))
    tl = local_offset(M, t, jnp.zeros(3), jnp.zeros((3, 2)))
    a = 0.7; R = jnp.asarray(np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32))
    X = jnp.asarray(np.random.default_rng(3).normal(size=(4, 3)).astype(np.float32))
    np.testing.assert_allclose(np.asarray(project_local(X @ R.T, rotate_world(M, R), tl)),
                               np.asarray(project_local(X, M, tl)), atol=1e-3)


def test_mirror_world_flips_every_view():
    from jarvis_jax.models.mvq.geometry import affine_rows, project_local, local_offset, mirror_world
    M, t = affine_rows(jnp.asarray(_cam_mats()))
    tl = local_offset(M, t, jnp.zeros(3), jnp.zeros((3, 2)))
    X = jnp.asarray(np.random.default_rng(4).normal(size=(4, 3)).astype(np.float32))
    uv = np.asarray(project_local(X, M, tl))
    S = jnp.asarray(np.diag([-1.0, 1.0, 1.0]).astype(np.float32))
    M2, tl2 = mirror_world(M, tl, 448)
    uv2 = np.asarray(project_local(X @ S, M2, tl2))                    # reflected world, mirrored cams
    np.testing.assert_allclose(uv2[..., 0], 447.0 - uv[..., 0], atol=1e-3)
    np.testing.assert_allclose(uv2[..., 1], uv[..., 1], atol=1e-3)


def test_token_centres_and_px_scale():
    from jarvis_jax.models.mvq.geometry import token_pixel_centres, px_scale, affine_rows
    c = np.asarray(token_pixel_centres(28, 28, 16))
    assert c.shape == (784, 2) and tuple(c[0]) == (8.0, 8.0) and tuple(c[1]) == (24.0, 8.0)
    M, _ = affine_rows(jnp.asarray(_cam_mats()))
    assert 7.5 < float(px_scale(M)) < 8.5          # ~8.1 px per world unit for this rig
