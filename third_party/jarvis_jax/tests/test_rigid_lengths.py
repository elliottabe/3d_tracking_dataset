"""Properties the rigid-length projection must have -- especially the ones that
distinguish it from the temporal smoothing wings were EXCLUDED from."""
import numpy as np
import pytest

from jarvis_jax.tracking.rigid_lengths import (estimate_bone_lengths,
                                               enforce_bone_lengths,
                                               segment_length_report)

NAMES = ["Scutellum", "WingL_base", "WingL_V12", "WingL_V13"]
EDGES = np.array([[0, 1], [1, 2], [2, 3]])


def _chain(T=50, jitter=0.0, seed=0):
    """Straight chain along +x with unit spacing, optional radial jitter on the
    LAST point only (the measured failure: V12/V13 dragged along the vein)."""
    rng = np.random.default_rng(seed)
    kp = np.zeros((T, 4, 3), float)
    for i in range(4):
        kp[:, i, 0] = i
    if jitter:
        kp[:, 3, 0] += rng.normal(0, jitter, T)
    return kp, np.ones((T, 4), float)


def test_enforced_lengths_hit_the_target():
    """split='distal' (the default) must be EXACT in a single sweep on a chain."""
    kp, conf = _chain(jitter=0.3, seed=1)
    targets = {0: 1.0, 1: 1.0, 2: 1.0}
    out, info = enforce_bone_lengths(kp, conf, NAMES, EDGES, targets, iters=1)
    for (a, b) in EDGES:
        L = np.linalg.norm(out[:, a] - out[:, b], axis=-1)
        assert np.allclose(L, 1.0, atol=1e-6), f"edge {a}-{b} -> {L[:3]}"
    assert info["n_applied"] > 0


def test_direction_is_never_changed():
    """Only the RADIUS may change. If the direction moves, the projection is
    doing something other than what it claims."""
    kp, conf = _chain(jitter=0.4, seed=2)
    kp[:, 3, 1] = 0.7                              # give the last edge a real 3-D direction
    targets = {0: 1.0, 1: 1.0, 2: 2.0}
    out, _ = enforce_bone_lengths(kp, conf, NAMES, EDGES, targets, iters=1)
    for (a, b) in EDGES[2:]:
        u_in = kp[:, b] - kp[:, a]; u_in /= np.linalg.norm(u_in, axis=-1, keepdims=True)
        u_out = out[:, b] - out[:, a]; u_out /= np.linalg.norm(u_out, axis=-1, keepdims=True)
        assert np.allclose(u_in, u_out, atol=1e-9)


def test_fast_motion_survives_exactly():
    """THE point of a per-frame projection. A smoother would attenuate this;
    wings were excluded from smoothing because it halved the song's speed."""
    T = 200
    kp = np.zeros((T, 4, 3), float)
    for i in range(4):
        kp[:, i, 0] = i
    # last point swings fast and wide, at constant radius from its parent
    th = np.linspace(0, 40 * np.pi, T)
    kp[:, 3, 0] = 2.0 + np.cos(th)
    kp[:, 3, 1] = np.sin(th)
    conf = np.ones((T, 4))
    targets = {0: 1.0, 1: 1.0, 2: 1.0}
    out, _ = enforce_bone_lengths(kp, conf, NAMES, EDGES, targets, iters=3)
    sp_in = np.linalg.norm(np.diff(kp[:, 3], axis=0), axis=-1)
    sp_out = np.linalg.norm(np.diff(out[:, 3], axis=0), axis=-1)
    # already at the target radius -> motion must be untouched, not merely close
    assert np.allclose(sp_out, sp_in, atol=1e-9), (sp_in[:3], sp_out[:3])


def test_lower_confidence_endpoint_absorbs_the_correction():
    kp = np.zeros((1, 4, 3)); kp[0, 2, 0] = 1.0; kp[0, 3, 0] = 3.0   # edge 2-3 too long
    conf = np.ones((1, 4)); conf[0, 2] = 0.99; conf[0, 3] = 0.01
    out, _ = enforce_bone_lengths(kp, conf, NAMES, EDGES, {2: 1.0}, iters=1,
                                 anchor_patterns=(), split="confidence")
    moved_2 = abs(out[0, 2, 0] - kp[0, 2, 0])
    moved_3 = abs(out[0, 3, 0] - kp[0, 3, 0])
    assert moved_3 > 10 * moved_2, (moved_2, moved_3)


def test_anchored_keypoints_never_move():
    kp, conf = _chain(jitter=0.5, seed=3)
    out, _ = enforce_bone_lengths(kp, conf, NAMES, EDGES, {0: 0.5, 1: 1.0, 2: 1.0},
                                 iters=3, anchor_patterns=("Scutellum",))
    assert np.allclose(out[:, 0], kp[:, 0]), "Scutellum was anchored but moved"


def test_nan_endpoints_are_skipped_not_propagated():
    kp, conf = _chain()
    kp[5, 3] = np.nan
    out, _ = enforce_bone_lengths(kp, conf, NAMES, EDGES, {0: 1.0, 1: 1.0, 2: 1.0})
    assert np.isnan(out[5, 3]).all()
    assert np.isfinite(out[5, :3]).all(), "a NaN endpoint must not poison its neighbours"
    assert np.isfinite(out[6]).all()


def test_max_shift_refuses_a_wild_correction():
    """A wildly mistriangulated point must NOT be dragged onto the target and
    thereby made to look plausible."""
    kp, conf = _chain()
    kp[7, 3, 0] = 50.0
    out, info = enforce_bone_lengths(kp, conf, NAMES, EDGES, {2: 1.0}, iters=1,
                                    max_shift=5.0)
    assert np.allclose(out[7, 3], kp[7, 3]), "wild point should be left alone"
    assert info["n_skipped_max_shift"] > 0


def test_estimator_refuses_an_edge_it_cannot_measure():
    kp, conf = _chain(T=10)
    targets, rep = estimate_bone_lengths(kp, conf, NAMES, EDGES, min_frames=50)
    assert all(v is None for v in targets.values())
    assert "usable frames" in rep["WingL_V12->WingL_V13"]["reason"]


def test_estimator_refuses_a_non_rigid_edge():
    """An edge whose own length varies wildly cannot define a target; measuring
    one anyway is how a projection ends up confidently wrong."""
    kp, conf = _chain(T=400, jitter=2.0, seed=4)
    targets, rep = estimate_bone_lengths(kp, conf, NAMES, EDGES,
                                         min_frames=10, max_cv=0.2)
    assert targets[2] is None
    assert "not rigid" in rep["WingL_V12->WingL_V13"]["reason"]
    assert targets[0] is not None, "a genuinely rigid edge must still be measured"


def test_estimator_recovers_a_known_length():
    kp, conf = _chain(T=300, jitter=0.05, seed=5)
    targets, rep = estimate_bone_lengths(kp, conf, NAMES, EDGES, min_frames=10)
    assert targets[0] == pytest.approx(1.0, abs=0.02)
    assert rep["Scutellum->WingL_base"]["n"] == 300


def test_report_is_tautological_after_enforcement():
    """Documents WHY length CV is not the acceptance test: it is ~0 by
    construction once enforced, whatever the target was."""
    kp, conf = _chain(jitter=0.4, seed=6)
    out, _ = enforce_bone_lengths(kp, conf, NAMES, EDGES,
                                 {0: 1.0, 1: 1.0, 2: 99.0}, iters=5)
    rep = segment_length_report(out, NAMES, EDGES)
    mean, cv = rep["WingL_V12->WingL_V13"]
    assert mean == pytest.approx(99.0, abs=1e-6) and cv < 1e-9


def test_confidence_split_needs_iterations_on_a_chain_but_distal_does_not():
    """Documents why 'distal' is the default: on a chain, a distal edge's
    correction perturbs the shared middle point and breaks the edge proximal to
    it, so 'confidence' converges only approximately."""
    kp, conf = _chain(jitter=0.3, seed=7)
    targets = {0: 1.0, 1: 1.0, 2: 1.0}
    d, _ = enforce_bone_lengths(kp, conf, NAMES, EDGES, targets, iters=1,
                                split="distal")
    c, _ = enforce_bone_lengths(kp, conf, NAMES, EDGES, targets, iters=1,
                                split="confidence", anchor_patterns=())
    err = lambda x: max(abs(np.linalg.norm(x[:, a] - x[:, b], axis=-1) - 1.0).max()
                        for a, b in EDGES)
    assert err(d) < 1e-9
    assert err(c) > 1e-3


def test_unknown_split_is_rejected():
    kp, conf = _chain()
    with pytest.raises(ValueError, match="distal.*confidence"):
        enforce_bone_lengths(kp, conf, NAMES, EDGES, {2: 1.0}, split="magic")
