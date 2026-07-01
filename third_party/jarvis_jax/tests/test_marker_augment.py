import numpy as np
from jarvis_jax.cse.marker_augment import augment_wing_markers, WING_MARKER_IDS


def test_augment_overrides_wing_markers_and_boosts_weight():
    T, n_kp = 3, 50
    kp = np.zeros((T, n_kp, 3)); w = np.ones(n_kp * 3)
    tips = [
        {"left": (np.array([1.0, 2.0, 3.0]), 3), "right": None},   # frame 0: left tip, 3 cams
        {"left": None, "right": None},                              # frame 1: none
        {"left": None, "right": (np.array([4.0, 5.0, 6.0]), 2)},    # frame 2: right tip, 2 cams
    ]
    kp2, w2 = augment_wing_markers(kp, w, tips, wing_weight=5.0, min_cams=2)
    # frame 0 left markers (7,8) set to the left tip
    assert np.allclose(kp2[0, 7], [1, 2, 3]) and np.allclose(kp2[0, 8], [1, 2, 3])
    # frame 2 right markers (29,30) set to the right tip
    assert np.allclose(kp2[2, 29], [4, 5, 6]) and np.allclose(kp2[2, 30], [4, 5, 6])
    # frame 1 untouched (still zero)
    assert np.allclose(kp2[1, 7], [0, 0, 0])
    # weights boosted for all four wing markers' coords
    for idx in (7, 8, 29, 30):
        assert np.allclose(w2[idx * 3:idx * 3 + 3], 5.0)
    # a non-wing marker weight unchanged
    assert np.allclose(w2[3 * 3:3 * 3 + 3], 1.0)
    # inputs not mutated
    assert np.allclose(kp[0, 7], [0, 0, 0]) and np.allclose(w[7 * 3], 1.0)


def test_min_cams_gate():
    kp = np.zeros((1, 50, 3)); w = np.ones(150)
    tips = [{"left": (np.array([1.0, 1.0, 1.0]), 1), "right": None}]   # only 1 cam
    kp2, w2 = augment_wing_markers(kp, w, tips, min_cams=2)
    assert np.allclose(kp2[0, 7], [0, 0, 0])          # not applied (below min_cams)
    assert np.allclose(w2[7 * 3:7 * 3 + 3], 1.0)
