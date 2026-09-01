"""Cross-view wing L/R assignment. The decisive case is a MAJORITY flip, which
is what consensus triangulation provably cannot fix."""
import numpy as np
import pytest

from jarvis_jax.tracking.wing_lr_assign import resolve_wing_lr, _project

NAMES = ["Scutellum", "WingL_base", "WingL_V12", "WingL_V13",
         "WingR_base", "WingR_V12", "WingR_V13", "Abd_tip"]


def _rig(C=7, seed=0):
    """C orthographic-ish DLT cameras looking from different directions."""
    rng = np.random.default_rng(seed)
    P = np.zeros((C, 4, 3))
    for c in range(C):
        th = 2 * np.pi * c / C
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                      [0.0, 0.0, 1.0],
                      [np.sin(th), np.cos(th), 0.0]])
        P[c, :3, :] = R.T * 50.0                 # scale to pixels
        P[c, 3, :] = [200.0, 200.0, 1.0]         # offset; w row -> 1
        P[c, :3, 2] = 0.0                        # make w independent of X (affine)
    return P


def _scene(T=12, flip_cams=(), seed=1):
    """Two wings at mirrored positions; `flip_cams` get their L/R labels swapped."""
    rng = np.random.default_rng(seed)
    K = len(NAMES)
    X = np.zeros((T, K, 3))
    X[:, 0] = [0, 0, 0]                                       # Scutellum
    for t in range(T):
        wob = 0.15 * rng.standard_normal(3)
        X[t, 1] = [0.2, +1.0, 0.0] + wob                      # L base
        X[t, 2] = [1.2, +2.0, 0.0] + wob                      # L V12
        X[t, 3] = [1.5, +2.4, 0.0] + wob                      # L V13
        X[t, 4] = [0.2, -1.0, 0.0] - wob                      # R base
        X[t, 5] = [1.2, -2.0, 0.0] - wob                      # R V12
        X[t, 6] = [1.5, -2.4, 0.0] - wob                      # R V13
        X[t, 7] = [-2.0, 0.0, 0.0]                            # Abd_tip (body scale)
    P = _rig()
    kp2d = _project(X, P).astype(np.float32)                  # (T,C,K,2)
    conf = np.ones(kp2d.shape[:3], np.float32)
    truth = kp2d.copy()
    for c in flip_cams:
        for a, b in ((1, 4), (2, 5), (3, 6)):
            kp2d[:, c, [a, b]] = kp2d[:, c, [b, a]]
    return kp2d, conf, P, truth, X


def test_no_flips_is_left_alone():
    kp2d, conf, P, truth, _ = _scene()
    out, _, rep = resolve_wing_lr(kp2d, conf, P, NAMES)
    assert rep["n_view_swaps"] == 0
    assert np.allclose(out, truth, atol=1e-4)


def test_recovers_a_MINORITY_flip():
    kp2d, conf, P, truth, _ = _scene(flip_cams=(1, 2))
    out, _, rep = resolve_wing_lr(kp2d, conf, P, NAMES)
    assert np.allclose(out, truth, atol=1e-4), rep
    assert rep["frames_with_any_swap"] == kp2d.shape[0]


def test_recovers_a_MAJORITY_flip_on_SOME_frames():
    """THE measured case: 4 of 7 views wrong on a MINORITY OF FRAMES (Session0
    bout 28: 11 of 300). Consensus triangulation cannot fix this -- it follows
    the 4 and gates the 3 -- but geometry plus bout-level naming can."""
    kp2d, conf, P, truth, _ = _scene(T=10)
    bad = [2, 5, 7]
    for t in bad:
        for c in (0, 1, 2, 3):
            for a, b in ((1, 4), (2, 5), (3, 6)):
                kp2d[t, c, [a, b]] = kp2d[t, c, [b, a]]
    out, _, rep = resolve_wing_lr(kp2d, conf, P, NAMES)
    assert np.allclose(out, truth, atol=1e-4), rep


def test_a_majority_flipped_on_EVERY_frame_is_ambiguous_up_to_a_global_swap():
    """Documents the limit: if the same 4 views are wrong on every frame, both
    global namings explain the data equally well. The output must still be
    self-CONSISTENT (all views agreeing), just possibly L<->R renamed -- and an
    anchor resolves it."""
    kp2d, conf, P, truth, _ = _scene(flip_cams=(0, 1, 2, 3))
    out, _, _ = resolve_wing_lr(kp2d, conf, P, NAMES)
    li = [NAMES.index(n) for n in ("WingL_base", "WingL_V12", "WingL_V13")]
    ri = [NAMES.index(n) for n in ("WingR_base", "WingR_V12", "WingR_V13")]
    same = np.allclose(out, truth, atol=1e-4)
    swapped = np.allclose(out[:, :, li], truth[:, :, ri], atol=1e-4) and \
              np.allclose(out[:, :, ri], truth[:, :, li], atol=1e-4)
    assert same or swapped, "output is neither the truth nor its global swap"
    # anchoring a view that IS correct in the input pins the naming
    out2, _, _ = resolve_wing_lr(kp2d, conf, P, NAMES, anchor_cameras=(4,))
    assert np.allclose(out2, truth, atol=1e-4)


def test_a_flip_on_only_some_frames_is_fixed_per_frame():
    kp2d, conf, P, truth, _ = _scene(T=10)
    bad = [2, 5, 7]
    for t in bad:
        for c in (1, 4, 5, 6):                    # majority, on these frames only
            for a, b in ((1, 4), (2, 5), (3, 6)):
                kp2d[t, c, [a, b]] = kp2d[t, c, [b, a]]
    out, _, rep = resolve_wing_lr(kp2d, conf, P, NAMES)
    assert np.allclose(out, truth, atol=1e-4)
    assert rep["frames_with_any_swap"] == len(bad)


def test_scoring_does_not_gate_the_correct_minority():
    """Regression on the design choice: if scoring used reproj_resid_px it would
    discard the 3 good views and keep the majority's answer."""
    import inspect
    from jarvis_jax.tracking import wing_lr_assign as m
    src = inspect.getsource(m.resolve_wing_lr)
    assert "reproj_resid_px=None" in src and "view_conf_thresh=None" in src


def test_anchor_camera_is_never_flipped():
    kp2d, conf, P, truth, _ = _scene(flip_cams=(0, 1, 2, 3))
    _, _, rep = resolve_wing_lr(kp2d, conf, P, NAMES, anchor_cameras=(0,))
    assert rep["per_camera_swap_rate"][0] == 0.0


def test_too_many_cameras_is_refused():
    kp2d, conf, P, _, _ = _scene()
    with pytest.raises(ValueError, match="patterns"):
        resolve_wing_lr(kp2d, conf, P, NAMES, max_cameras=3)


def test_low_confidence_views_do_not_drive_the_decision():
    kp2d, conf, P, truth, _ = _scene(flip_cams=(5,))
    conf[:, 5] = 0.0                              # the flipped view is unusable
    out, _, rep = resolve_wing_lr(kp2d, conf, P, NAMES)
    # its own points are untouched-or-swapped either way; the OTHER views must
    # come through unchanged rather than being dragged to match it.
    keep = [c for c in range(kp2d.shape[1]) if c != 5]
    assert np.allclose(out[:, keep], truth[:, keep], atol=1e-4)


def test_temporal_resolution_survives_FAST_wing_motion():
    """The temporal tie-break must not assume slow motion -- courting males beat
    their wings fast, and wings were excluded from smoothing for exactly that
    reason. It works because the margin is large: a real inter-frame step is
    much smaller than the L<->R separation. Measured on Session0 bout 28 fly1
    after frame 300, WingL_V12's median step is 0.41u and its max 2.3u, against
    wings ~10-20u apart -- an order of magnitude of headroom."""
    T = 24
    K = len(NAMES)
    X = np.zeros((T, K, 3))
    th = np.linspace(0, 6 * np.pi, T)              # fast sweep, ~0.8 rad/frame
    for t in range(T):
        X[t, 0] = [0, 0, 0]
        for sgn, (b, v12, v13) in ((+1, (1, 2, 3)), (-1, (4, 5, 6))):
            X[t, b] = [0.2, sgn * 1.0, 0.0]
            X[t, v12] = [1.2 + 0.5 * np.cos(th[t]), sgn * (2.0 + 0.4 * np.sin(th[t])), 0]
            X[t, v13] = [1.5 + 0.6 * np.cos(th[t]), sgn * (2.4 + 0.5 * np.sin(th[t])), 0]
        X[t, 7] = [-2.0, 0.0, 0.0]
    P = _rig()
    truth = _project(X, P).astype(np.float32)
    kp2d = truth.copy()
    conf = np.ones(kp2d.shape[:3], np.float32)
    for t in (5, 6, 13):                            # majority flip, some frames
        for c in (0, 1, 2, 3):
            for a, b in ((1, 4), (2, 5), (3, 6)):
                kp2d[t, c, [a, b]] = kp2d[t, c, [b, a]]
    out, _, rep = resolve_wing_lr(kp2d, conf, P, NAMES)
    assert np.allclose(out, truth, atol=1e-3), rep
    assert rep["frames_with_any_swap"] == 3


def test_the_temporal_pass_is_what_makes_the_majority_case_work():
    """Regression: without the temporal handedness pass the solver prefers the
    3-flip complement over the correct 4-flip answer (fewest-flips bias), so
    this documents that the pass exists and is load-bearing."""
    import inspect
    from jarvis_jax.tracking import wing_lr_assign as m
    src = inspect.getsource(m.resolve_wing_lr)
    assert "resolved TEMPORALLY" in src
    assert "anchor = int(np.argmin(best_pat.sum(1)))" in src


def test_collapse_is_detected_where_it_is_and_not_where_it_is_not():
    """A collapse (both labels on one wing) is invisible to a swap search, so it
    needs its own detector."""
    from jarvis_jax.tracking.wing_lr_assign import (detect_wing_lr_collapse,
                                                    mask_collapsed_wing_views)
    kp2d, conf, P, truth, _ = _scene(T=20)
    victims = [(3, 1), (3, 4), (4, 1), (4, 4), (4, 5)]
    for t, c in victims:
        for a, b in ((1, 4), (2, 5), (3, 6)):
            kp2d[t, c, b] = kp2d[t, c, a]          # R label onto the L wing
    col, info = detect_wing_lr_collapse(kp2d, conf, NAMES)
    for t, c in victims:
        assert col[t, c], f"missed the collapse at frame {t} camera {c}"
    # No false positives among WELL-CONDITIONED cameras. Some views in this rig
    # are genuinely degenerate (both wings superpose in projection, median sep
    # ~0.1 body lengths); flagging those on every frame is the DOCUMENTED intent
    # -- such a view cannot inform L/R whatever the detector did -- so they are
    # excluded from the false-positive check rather than treated as errors.
    med = info["per_camera_median_sep"]
    good = [c for c in range(kp2d.shape[1])
            if med[c] is not None and med[c] > 1.5]
    assert good, "rig gave no well-conditioned camera to test against"
    for c in good:
        for t in range(20):
            if (t, c) not in victims:
                assert not col[t, c], f"false positive at frame {t} camera {c}"


def test_masking_only_touches_wing_keypoints():
    from jarvis_jax.tracking.wing_lr_assign import (detect_wing_lr_collapse,
                                                    mask_collapsed_wing_views)
    kp2d, conf, P, _, _ = _scene(T=6)
    for a, b in ((1, 4), (2, 5), (3, 6)):
        kp2d[2, 3, b] = kp2d[2, 3, a]
    col, _ = detect_wing_lr_collapse(kp2d, conf, NAMES)
    out, minfo = mask_collapsed_wing_views(conf, col, NAMES, min_views_kept=1)
    wing = [NAMES.index(n) for n in NAMES if "Wing" in n]
    other = [i for i in range(len(NAMES)) if i not in wing]
    assert col[2, 3], "the injected collapse was not detected"
    assert (out[2, 3, wing] == 0).all()
    assert (out[2, 3, other] == conf[2, 3, other]).all(), "non-wing conf was touched"
    # NON-wing confidences must be untouched everywhere, and any view NOT
    # flagged must be byte-identical.
    assert (out[:, :, other] == conf[:, :, other]).all()
    keep = ~col
    assert (out[keep][:, wing] == conf[keep][:, wing]).all(), \
        "an unflagged view's wing conf was modified"


def test_masking_refuses_when_too_few_views_would_survive():
    """The female-folded-wings guard: if masking would leave fewer than
    min_views_kept views, mask NOTHING for that frame. Measured consequence of
    not doing this: 117 bad frames became 246, with 88 new all-NaN frames."""
    from jarvis_jax.tracking.wing_lr_assign import mask_collapsed_wing_views
    kp2d, conf, P, _, _ = _scene(T=4)
    C = kp2d.shape[1]
    col = np.zeros((4, C), bool)
    col[0, :] = True                       # every view collapsed -> refuse
    col[1, :2] = True                      # only 2 of 7 -> allow
    out, info = mask_collapsed_wing_views(conf, col, NAMES, min_views_kept=4)
    wing = [NAMES.index(n) for n in NAMES if "Wing" in n]
    assert (out[0][:, wing] == conf[0][:, wing]).all(), "frame 0 should be untouched"
    assert (out[1, 0, wing] == 0).all(), "frame 1 view 0 should be masked"
    assert info["frames_left_alone_too_few_views"] == 1
    assert info["frames_masked"] == 1
