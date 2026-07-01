import numpy as np
from jarvis_jax.cse.affine_camera import reconstruct_affine, factor_affine, project_affine
from jarvis_jax.cse.bundle_adjust import assemble_observations, initial_points

P_REAL = np.array([
    [8.1001, 0.0074869, -0.031773, -2.828],
    [0.0093308, -8.0788, -0.17912, 462.78],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def _two_cam_rig():
    # camera 0 = P_REAL; camera 1 = P_REAL with a 25-degree yaw applied to R.
    K2, R, t = factor_affine(P_REAL)
    th = np.deg2rad(25.0)
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    return [P_REAL, reconstruct_affine(K2, Ry @ R, t)]


def test_assemble_and_triangulate_recovers_points():
    cams = _two_cam_rig()
    rng = np.random.default_rng(0)
    pts = rng.uniform([-3, -3, 8], [3, 3, 14], size=(5, 3))   # 5 world points
    F, C, K = 1, 2, 5
    kp2d = np.zeros((F, C, K, 3))
    for c, P in enumerate(cams):
        uv = project_affine(P, pts)                            # (5,2)
        kp2d[0, c, :, :2] = uv; kp2d[0, c, :, 2] = 1.0
    obs = assemble_observations(kp2d, min_cams=2)
    assert obs.n_points == 5 and obs.n_cams == 2 and obs.uv.shape[0] == 10
    X0 = initial_points(obs, cams)
    assert np.allclose(X0, pts, atol=1e-6), f"triangulation off:\n{X0}\nvs\n{pts}"
