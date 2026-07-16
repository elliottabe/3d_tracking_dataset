import numpy as np
from jarvis_jax.tracking import ik_error as ike


def test_sigma_kp_higher_for_lower_conf():
    kp3d = np.zeros((5, 3, 3))
    conf = np.array([[1.0, 0.5, 0.0]] * 5)          # (T=5, K=3)
    s = ike.sigma_kp_from_conf(conf, kp3d, floor_mm=1.0, scale_mm_per_lowconf=10.0)
    assert s.shape == (3,)
    assert s[0] < s[1] < s[2]                        # lower conf -> larger sigma
    assert s.min() >= 1.0                             # floor respected


def test_sigma2_diag_shape_and_values():
    s = np.array([2.0, 3.0])                         # K=2
    S2 = ike.sigma2_diag(s)
    assert S2.shape == (6, 6)
    assert np.allclose(np.diag(S2), [4, 4, 4, 9, 9, 9])   # sigma^2 repeated xyz
