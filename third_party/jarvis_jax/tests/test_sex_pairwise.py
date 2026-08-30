import numpy as np
import pytest

from jarvis_jax.tracking.sex_pairwise import (
    pair_features, fit_pairwise, predict_male_slot, clip_index)

NAMES = ["Scutellum", "Abd_A4", "Abd_tip", "WingL_base", "WingL_V12",
         "EyeL", "EyeR", "T1L_FeTi", "T1L_TiTa"]


def _fly(scale=1.0, abd=1.0):
    """Synthetic fly: `scale` sets overall size, `abd` the abdomen extension."""
    kp = np.zeros((len(NAMES), 3), np.float32)
    kp[NAMES.index("Scutellum")] = [0, 0, 0]
    kp[NAMES.index("Abd_A4")] = [0, -6 * scale, 0]
    kp[NAMES.index("Abd_tip")] = [0, -13 * scale * abd, 0]
    kp[NAMES.index("WingL_base")] = [1 * scale, -1 * scale, 0]
    kp[NAMES.index("WingL_V12")] = [3 * scale, -18 * scale, 0]
    kp[NAMES.index("EyeL")] = [-2 * scale, 4 * scale, 0]
    kp[NAMES.index("EyeR")] = [2 * scale, 4 * scale, 0]
    kp[NAMES.index("T1L_FeTi")] = [-3 * scale, 1 * scale, 0]
    kp[NAMES.index("T1L_TiTa")] = [-5 * scale, -2 * scale, 0]
    return kp


def test_pair_features_are_exactly_antisymmetric():
    """The property the whole design rests on: swapping the pair must negate
    the features, so P(a male) == 1 - P(b male) by construction."""
    a, b = _fly(1.0), _fly(1.25, abd=1.1)
    fab = pair_features(a, b, NAMES)
    fba = pair_features(b, a, NAMES)
    np.testing.assert_allclose(fab, -fba, atol=1e-6)


def test_identical_flies_give_zero_features():
    a = _fly(1.0)
    np.testing.assert_allclose(pair_features(a, a.copy(), NAMES), 0.0, atol=1e-6)


def test_features_are_not_all_scale_normalised_away():
    """Size IS the signal (females are larger). A feature vector that is
    invariant to overall scale has thrown away the main cue."""
    small, big = _fly(1.0), _fly(1.3)
    f = pair_features(small, big, NAMES)
    assert np.abs(f).max() > 1e-3


def test_prediction_is_swap_consistent():
    rng = np.random.default_rng(0)
    X, y, g = [], [], []
    for i in range(40):
        male = _fly(1.0 + rng.normal(0, .02), abd=1.0)
        female = _fly(1.25 + rng.normal(0, .02), abd=1.15)
        # alternate which slot holds the male so the label is not slot-correlated
        if i % 2:
            X.append(pair_features(male, female, NAMES)); y.append(0)
        else:
            X.append(pair_features(female, male, NAMES)); y.append(1)
        g.append(f"clip{i // 8}")
    m = fit_pairwise(np.array(X), np.array(y), np.array(g))
    male, female = _fly(1.0), _fly(1.25, abd=1.15)
    s0, p0 = predict_male_slot(m, male, female, NAMES)
    s1, p1 = predict_male_slot(m, female, male, NAMES)
    assert s0 == 0 and s1 == 1, (s0, s1)
    assert abs(p0 - p1) < 1e-6, "swap must give the mirrored probability"


def test_model_has_no_intercept():
    """An intercept would break antisymmetry: the model could prefer slot 0."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4)); y = (X[:, 0] > 0).astype(int)
    g = np.array([f"c{i//5}" for i in range(30)])
    m = fit_pairwise(X, y, g)
    assert float(np.abs(m["intercept"])) == 0.0


def test_cv_groups_by_clip_never_by_frameset():
    """Adjacent framesets in a clip are near-duplicates. Grouping CV by
    frameset would leak exactly the way the dataset split used to."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3)); y = (X[:, 0] > 0).astype(int)
    g = np.array([f"clip{i//10}" for i in range(40)])
    m = fit_pairwise(X, y, g)
    assert set(m["cv_groups"]) == {"clip0", "clip1", "clip2", "clip3"}
    assert len(m["fold_accuracy"]) == 4


def test_clip_index_splits_on_frame_gaps():
    merged = {"framesets": {}}
    for f in [10, 11, 12, 900, 901]:
        merged["framesets"][f"rec_a/Frame_{f:06d}/fly0"] = {
            "recording": "rec_a", "fly_id": 0, "frames": [], "ann_ids": []}
    clips = clip_index(merged, gap=100)
    assert len(clips) == 2
    assert sorted(len(v) for v in clips.values()) == [2, 3]


def test_nan_keypoints_do_not_poison_features():
    a, b = _fly(1.0), _fly(1.25)
    b[NAMES.index("WingL_V12")] = np.nan
    f = pair_features(a, b, NAMES)
    assert np.all(np.isfinite(f)), "NaN keypoints must degrade, not propagate"


def test_scalars_raises_when_majority_of_named_segments_do_not_resolve():
    """Under a wrong keypoint-name ordering (this repo has a documented
    tracking-order vs XML/model-order gotcha), most named _SEGMENTS fail to
    resolve and contribute 0.0 -- but the whole-cloud `extent` feature
    survives regardless (it does no name lookup). That means the model
    trains on degraded-but-not-chance features and reports plausible,
    quietly-wrong accuracy with no error -- worse than clean chance for
    detectability. Must raise loudly instead of silently zeroing out."""
    from jarvis_jax.tracking.sex_pairwise import _scalars, _SEGMENTS

    # Only EyeL/EyeR and T1L_FeTi/T1L_TiTa resolve: 2 of 6 _SEGMENTS.
    bad_names = ["EyeL", "EyeR", "T1L_FeTi", "T1L_TiTa", "Abd_tip"]
    kp = np.zeros((len(bad_names), 3), np.float64)
    with pytest.raises(ValueError, match="_SEGMENTS"):
        _scalars(kp, bad_names)


def test_scalars_does_not_raise_at_exactly_half_resolved():
    """Exactly half resolving (the boundary, not 'fewer than half') must
    still be allowed."""
    from jarvis_jax.tracking.sex_pairwise import _scalars, _SEGMENTS

    assert len(_SEGMENTS) == 6
    # Scutellum/Abd_tip, Scutellum/Abd_A4, Abd_A4/Abd_tip resolve: 3 of 6.
    ok_names = ["Scutellum", "Abd_tip", "Abd_A4", "WingL_base", "EyeL"]
    kp = np.zeros((len(ok_names), 3), np.float64)
    out = _scalars(kp, ok_names)
    assert out.shape == (len(_SEGMENTS) + 2,)
