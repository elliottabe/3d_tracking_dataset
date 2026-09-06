# tests/test_mvq_perkp.py
"""`scripts/benchmark/mvq_perkp_bouts.py` -- per-keypoint bout stability
metrics, promoted from the scratchpad `perkp_all_bouts.py` /
`perkp_jump_probe.py` (see docs/benchmark/2026-09-mvq/p3b-notes.md's "P3b on
the mask-free route" / "re-lift bouts 1, 4, 28" tables, whose numbers this
module must keep reproducing).

Definitions under test (kept verbatim with the scratchpad scripts so the
numbers stay comparable to the P3b tables -- see the module docstring):
  * pose_jump: a frame has >= `min_kp` MALE keypoints whose frame-to-frame
    step exceeds `jump_mm` (0.5mm default = 5 world units, MM_PER_UNIT=0.1).
  * straddle: a frame has >= `min_kp` MALE keypoints nearer the FEMALE's body
    centroid than the male's own centroid -- but ONLY when the female
    centroid is itself finite (CLAUDE.md / P3b bout-1 lesson: bout 1's r2
    straddle number was "clean" only because the female was NaN on 76% of
    frames there, i.e. blind, not clean -- an unmeasurable frame must not
    read as a non-straddle frame).
  * female_missing: the female's 3D centroid is NaN.
  * head_tail_flips: the Antenna_Base->Abd_tip axis unit vector reverses
    (cos angle to the previous frame < 0), keypoints found BY NAME (never by
    a bare integer -- CLAUDE.md's keypoint-order-trap history), so a
    permuted `kp_names` must not change the count.

Each test states the expectation before asserting it (CLAUDE.md).
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from scripts.benchmark.mvq_perkp_bouts import perkp_bout_metrics  # noqa: E402

K = 12
NAMES = ["Antenna_Base", "EyeL", "EyeR", "Scutellum", "WingL_base", "WingL_V12",
         "WingR_base", "WingR_V12", "T1L_FeTi", "T1R_FeTi", "T2L_FeTi", "Abd_tip"]
IA, IT = NAMES.index("Antenna_Base"), NAMES.index("Abd_tip")


def _still_track(T, K, *, centre=(0.0, 0.0, 0.0), rng=None, spread=5.0):
    """(T,K,3): K points on a small rigid body around `centre`, plus tiny
    (<< jump/straddle thresholds) per-frame noise so it is not perfectly
    static -- a real track never is -- while remaining far below any of the
    metrics' triggering thresholds."""
    rng = rng or np.random.default_rng(0)
    base = rng.normal(scale=spread, size=(K, 3))
    body = np.asarray(centre) + base
    noise = rng.normal(scale=0.05, size=(T, K, 3))   # 0.05 units << 5-unit jump threshold
    return body[None] + noise


def _male_female(T=40, seed=0, female_centre=(200.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    male = _still_track(T, K, centre=(0.0, 0.0, 0.0), rng=rng)
    female = _still_track(T, K, centre=female_centre, rng=rng)
    return male, female


def test_no_motion_scores_zero_pose_jump():
    """EXPECTATION: a bout where nothing moves beyond the per-frame noise
    floor scores pose_jump_frac exactly 0 -- there is no frame with >= 5
    keypoints stepping more than 0.5mm."""
    male, female = _male_female()
    r = perkp_bout_metrics({0: female, 1: male}, NAMES, male_fly=1)
    assert r["pose_jump_frac"] == 0.0
    assert r["T"] == male.shape[0]


def test_injected_jump_scores_exactly_one_frame_and_names_the_six_keypoints():
    """EXPECTATION: stepping 6 of the male's keypoints by 0.8mm (8 units, >
    the 5-unit/0.5mm threshold) on ONE frame only trips pose_jump on that
    ONE frame (fraction exactly 1/T), and `per_kp_jump_frac` is nonzero at
    exactly those 6 keypoint indices, found by their NAMES."""
    male, female = _male_female(T=20)
    T = male.shape[0]
    jump_frame = 10
    jumped_kps = [0, 2, 4, 6, 8, 10]   # 6 keypoints, arbitrary subset
    male = male.copy()
    male[jump_frame:, jumped_kps, 0] += 8.0   # +8 units = +0.8mm, persists (a real step)
    r = perkp_bout_metrics({0: female, 1: male}, NAMES, male_fly=1)
    assert r["pose_jump_frac"] == pytest.approx(1.0 / T)
    per_kp = r["per_kp_jump_frac"]
    assert list(r["per_kp_names"]) == NAMES
    for k in range(K):
        if k in jumped_kps:
            assert per_kp[k] == pytest.approx(1.0 / T), (k, per_kp[k])
        else:
            assert per_kp[k] == 0.0, (k, per_kp[k])


def test_straddle_when_six_male_keypoints_sit_on_the_female_centroid():
    """EXPECTATION: moving 6 of the male's keypoints onto the female's
    centroid for exactly one frame trips straddle on that ONE frame only
    (straddle_frac == 1/T) -- those keypoints are (by construction) nearer
    the female centroid than the male's own."""
    male, female = _male_female(T=20, female_centre=(200.0, 0.0, 0.0))
    T = male.shape[0]
    strad_frame = 5
    strad_kps = [1, 3, 5, 7, 9, 11]
    fem_centroid_at_frame = np.nanmean(female[strad_frame], axis=0)
    male = male.copy()
    male[strad_frame, strad_kps, :] = fem_centroid_at_frame
    r = perkp_bout_metrics({0: female, 1: male}, NAMES, male_fly=1)
    assert r["straddle_frac"] == pytest.approx(1.0 / T)


def test_nan_female_frame_counts_as_missing_not_straddle():
    """EXPECTATION (the P3b bout-1 blindness lesson): a frame where the
    female's centroid is entirely unmeasurable (NaN) must count in
    `female_missing_frac`, and must NOT ALSO be read as a clean ("no
    straddle") frame nor a straddling one -- it is unmeasurable, not clean.
    Even if we additionally place 6 male keypoints far from the male's own
    centroid on that same NaN frame, straddle must stay 0 there because the
    female reference itself does not exist."""
    male, female = _male_female(T=20)
    T = male.shape[0]
    nan_frame = 8
    female = female.copy()
    female[nan_frame] = np.nan
    male = male.copy()
    male[nan_frame, [0, 1, 2, 3, 4, 5], :] += 500.0   # would-be "straddle-shaped" outliers
    r = perkp_bout_metrics({0: female, 1: male}, NAMES, male_fly=1)
    assert r["female_missing_frac"] == pytest.approx(1.0 / T)
    assert r["straddle_frac"] == 0.0


def test_head_tail_flip_found_by_name_with_permuted_kp_names():
    """EXPECTATION: reversing the Antenna_Base->Abd_tip axis (a >90deg turn
    in one frame) counts exactly one flip, found BY NAME -- permuting
    `kp_names` (and permuting the keypoint arrays to match) must not change
    the count, since the lookup is never a bare integer index."""
    T = 10
    rng = np.random.default_rng(3)
    male = _still_track(T, K, centre=(0.0, 0.0, 0.0), rng=rng)
    female = _still_track(T, K, centre=(200.0, 0.0, 0.0), rng=rng)
    male = male.copy()
    # Antenna_Base fixed at origin; Abd_tip walks along +x, then flips to -x
    # from frame 5 onward -- axis direction reverses exactly once.
    male[:, IA, :] = 0.0
    male[:5, IT, :] = np.array([10.0, 0.0, 0.0])
    male[5:, IT, :] = np.array([-10.0, 0.0, 0.0])
    r = perkp_bout_metrics({0: female, 1: male}, NAMES, male_fly=1)
    assert r["head_tail_flips"] == 1

    perm = rng.permutation(K)
    names_p = [NAMES[i] for i in perm]
    male_p = male[:, perm, :]
    female_p = female[:, perm, :]
    r_p = perkp_bout_metrics({0: female_p, 1: male_p}, names_p, male_fly=1)
    assert r_p["head_tail_flips"] == 1


def test_contact_and_missing_fractions_are_named_not_indexed():
    """Sanity: contact_frac and female_missing_frac are well-formed
    fractions in [0,1] and per_kp_names is returned alongside per_kp_jump_frac
    so a caller can never read the array by a bare position without also
    having the name at hand."""
    male, female = _male_female(T=15, female_centre=(5.0, 0.0, 0.0))  # close by default
    r = perkp_bout_metrics({0: female, 1: male}, NAMES, male_fly=1, contact_units=15.0)
    assert 0.0 <= r["contact_frac"] <= 1.0
    assert 0.0 <= r["female_missing_frac"] <= 1.0
    assert len(r["per_kp_names"]) == len(r["per_kp_jump_frac"]) == K
