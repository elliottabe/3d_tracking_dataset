import numpy as np
from jarvis_jax.cse.courtship_triangulate import triangulate_keypoints


def _cam(P):  # P is (3,4); center3d expects (4,3)=P.T
    return P.T.astype(np.float32)


def test_triangulates_known_point_and_nan_when_underdetermined():
    # two orthogonal pinhole-ish cameras
    P0 = np.array([[1000, 0, 320, 0], [0, 1000, 240, 0], [0, 0, 1, 5.0]], float)
    P1 = np.array([[0, 0, 1000, 0], [0, 1000, 240, 0], [-1, 0, 0, 5.0]], float)
    cam_mats = np.stack([_cam(P0), _cam(P1)])            # (2,4,3)
    X = np.array([0.3, -0.2, 1.0])
    def proj(P):
        x = P @ np.append(X, 1.0); return x[:2] / x[2]
    kp2d = np.zeros((1, 2, 1, 2), np.float32)            # T=1,C=2,K=1
    kp2d[0, 0, 0] = proj(P0); kp2d[0, 1, 0] = proj(P1)
    conf = np.ones((1, 2, 1), np.float32)
    p3d, c3d = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    assert np.allclose(p3d[0, 0], X, atol=1e-2)
    assert c3d[0, 0] > 0
    # only one confident view -> NaN + zero conf
    conf1 = conf.copy(); conf1[0, 1, 0] = 0.0
    p3d1, c3d1 = triangulate_keypoints(kp2d, conf1, cam_mats, conf_thresh=0.3)
    assert np.isnan(p3d1[0, 0]).all() and c3d1[0, 0] == 0.0


def test_multipoint_each_tk_recovers_its_own_3d():
    # Test with T=2, K=2 to catch transpose/axis-order regressions.
    # The transpose(0,2,1,3) in triangulate_keypoints is a no-op with T=K=1,
    # so we need multi-point/multi-timestep to verify correct axis ordering.
    P0 = np.array([[1000, 0, 320, 0], [0, 1000, 240, 0], [0, 0, 1, 5.0]], float)
    P1 = np.array([[0, 0, 1000, 0], [0, 1000, 240, 0], [-1, 0, 0, 5.0]], float)
    cam_mats = np.stack([_cam(P0), _cam(P1)])            # (2,4,3)
    # Four distinct 3-D points: pts[t,k] is the 3-D world point for (time t, keypoint k)
    pts = np.array([[[0.30, -0.20, 1.00], [0.10, 0.05, 1.20]],
                    [[-0.15, 0.10, 0.90], [0.25, -0.05, 1.10]]], dtype=float)
    def proj(P, X):
        x = P @ np.append(X, 1.0); return x[:2] / x[2]
    kp2d = np.zeros((2, 2, 2, 2), np.float32)                     # (T=2,C=2,K=2)
    for t in range(2):
        for k in range(2):
            kp2d[t, 0, k] = proj(P0, pts[t, k])
            kp2d[t, 1, k] = proj(P1, pts[t, k])
    conf = np.ones((2, 2, 2), np.float32)
    p3d, c3d = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    # Each (t,k) must recover its own distinct 3-D point; wrong axis ordering would misassign.
    for t in range(2):
        for k in range(2):
            assert np.allclose(p3d[t, k], pts[t, k], atol=1e-2), \
                f"(t={t},k={k}) expected {pts[t,k]}, got {p3d[t,k]}"
    assert (c3d > 0).all()
