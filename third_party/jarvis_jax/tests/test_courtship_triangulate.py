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
