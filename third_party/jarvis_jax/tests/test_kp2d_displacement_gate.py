import numpy as np
from jarvis_jax.tracking.kp2d_displacement_gate import (
    resolve_thresholds_px, displacement_gate_kp2d)


def test_resolve_thresholds_per_keypoint_not_global():
    names = ["Scutellum", "WingL_V13", "T1L_TaTip"]
    thr = resolve_thresholds_px(names, default_px=40.0,
                                per_keypoint_px={"WingL_V13": 150.0})
    assert thr.shape == (3,)
    assert thr[0] == 40.0            # falls back to default
    assert thr[1] == 150.0           # explicit override
    assert thr[2] == 40.0            # falls back to default
    # a single global number is exactly what this must NOT collapse to:
    # two different keypoints got two different thresholds
    assert thr[0] != thr[1]


def test_resolve_thresholds_rejects_unknown_keypoint_name():
    import pytest
    with pytest.raises(ValueError):
        resolve_thresholds_px(["Scutellum"], default_px=40.0,
                              per_keypoint_px={"NotAKeypoint": 999.0})


def _toggle_series(T, cluster_a, cluster_b, conf_val=0.93):
    """(T,1,1,2) kp2d + (T,1,1) conf toggling between two clusters every
    frame, both at high confidence -- the measured WingL_V13 failure mode."""
    kp2d = np.zeros((T, 1, 1, 2), np.float32)
    for t in range(T):
        kp2d[t, 0, 0] = cluster_a if t % 2 == 0 else cluster_b
    conf = np.full((T, 1, 1), conf_val, np.float32)
    return kp2d, conf


def test_synthetic_hop_beyond_threshold_is_dropped():
    """A synthetic landmark hop larger than threshold must be flagged (conf
    zeroed) on the arrival frame of every such transition. Without the gate,
    nothing in this pipeline drops these -- both clusters are high
    confidence, single-camera, so there is no confidence/consensus signal."""
    T = 10
    kp2d, conf = _toggle_series(T, (100.0, 100.0), (300.0, 100.0))  # 200px hop
    out = displacement_gate_kp2d(
        kp2d, conf, ["WingL_V13"], default_px=40.0, conf_thresh=0.3)
    # every transition after frame 0 is a 200px hop > 40px threshold
    assert np.all(out[1:, 0, 0] == 0.0)
    assert out[0, 0, 0] == conf[0, 0, 0]     # first frame never touched (no predecessor)


def test_fast_but_plausible_sweep_is_kept():
    """A smooth, fast, real sweep (monotonic motion within the keypoint's
    own real-motion budget) must survive untouched -- this is the
    preservation half of the contract, not just the suppression half."""
    T = 30
    kp2d = np.zeros((T, 1, 1, 2), np.float32)
    kp2d[:, 0, 0, 0] = np.linspace(0.0, 300.0, T)   # ~10.3px/frame, well under 40px
    conf = np.full((T, 1, 1), 0.9, np.float32)
    out = displacement_gate_kp2d(
        kp2d, conf, ["WingL_V13"], default_px=40.0, conf_thresh=0.3)
    assert np.allclose(out, conf)


def test_disabled_mode_is_noop_bit_identical():
    """The run_bout.py wiring only calls this function when
    cfg.detector.kp2d_displacement_gate.enabled is true; this test locks the
    other half of that contract -- when the gate itself runs with an
    effectively-infinite threshold (the 'disabled' shape), it must return
    conf unchanged, matching what run_bout.py's own default-off branch
    produces (conf simply never touched)."""
    T = 20
    kp2d, conf = _toggle_series(T, (100.0, 100.0), (300.0, 100.0))
    out = displacement_gate_kp2d(
        kp2d, conf, ["WingL_V13"], default_px=1e12, conf_thresh=0.3)
    assert np.array_equal(out, conf)
    assert out is not conf                       # copy, not aliasing -- but bit-identical


def test_low_confidence_endpoint_is_never_evaluated():
    """A big jump FROM a low-confidence (e.g. empty-crop zero-fill) frame
    must not flag the arrival frame -- see the module docstring: comparing a
    real detection to a zero-filled gap would read as a spurious hop and
    wrongly drop a perfectly good detection."""
    T = 3
    kp2d = np.zeros((T, 1, 1, 2), np.float32)
    kp2d[0, 0, 0] = (0.0, 0.0)      # empty-crop zero-fill
    kp2d[1, 0, 0] = (500.0, 500.0)  # first real, confident detection
    kp2d[2, 0, 0] = (505.0, 505.0)  # small real move
    conf = np.array([[[0.0]], [[0.9]], [[0.9]]], np.float32)   # frame 0 below conf_thresh
    out = displacement_gate_kp2d(
        kp2d, conf, ["WingL_V13"], default_px=40.0, conf_thresh=0.3)
    assert out[1, 0, 0] == conf[1, 0, 0]           # NOT flagged despite the 707px "hop"


def test_multi_keypoint_threshold_is_per_keypoint():
    """WingL_V13's fast threshold must not leak onto a slow thorax keypoint
    sharing the same call -- the core of the 'per-keypoint, not global'
    requirement, exercised end to end through displacement_gate_kp2d."""
    T = 4
    names = ["Scutellum", "WingL_V13"]
    kp2d = np.zeros((T, 1, 2, 2), np.float32)
    conf = np.full((T, 1, 2), 0.9, np.float32)
    # Scutellum: a 60px hop (small keypoint, over the 40px default)
    kp2d[:, 0, 0, 0] = [0.0, 60.0, 60.0, 60.0]
    # WingL_V13: a 60px hop too, but under its 150px override
    kp2d[:, 0, 1, 0] = [0.0, 60.0, 60.0, 60.0]
    out = displacement_gate_kp2d(
        kp2d, conf, names, default_px=40.0,
        per_keypoint_px={"WingL_V13": 150.0}, conf_thresh=0.3)
    assert out[1, 0, 0] == 0.0                     # Scutellum: flagged
    assert out[1, 0, 1] == conf[1, 0, 1]            # WingL_V13: kept
