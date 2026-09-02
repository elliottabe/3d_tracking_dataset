"""Per-(bout, fly) tracking quality from the SAM masks, and the validity gate.

WHY THIS EXISTS. Session0 bout 28 fly0 (the female) is well tracked for frames
0-1499 and badly tracked for 1500-2006: her SAM mask area collapses from
15 862 px to 4 882 px and 40% of the masks that are still flagged `valid` are
slivers (the female pressed against a wall). `min_present_cameras: 3` is blind
to that -- the masks are PRESENT, just tiny -- and the wing-pitch fit then
solves against a third of a fly and produces a 130 deg swing that every
band-limited parameterisation renders as a SMOOTH wrong answer.

Two things are tested here and they are not the same thing:

  * the BOUT-LEVEL score, which is mask-only (area, validity, centroids) and
    ranks (bout, fly) pairs so a well-tracked bout can be chosen without
    running any pose stage;
  * the PER-FRAME GATE, which is POSE-AWARE. An area test cannot catch a
    wrong-fly or a merged-fly mask -- those have perfectly normal area -- so
    the gate also asks how much of the projected BODY lands inside the mask.

And the gate's output is an ENVELOPE, not a boolean: in `spline`/`lowpass`
zeroing `present` does NOT freeze a frame (the knots interpolate across it),
so the gate has to multiply the correction to zero on the frames it skips.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis_jax.tracking.mask_quality import (
    area_reference, bout_fly_quality, camera_validity, frame_keep,
    gate_envelope, mask_areas_from_masks, mask_areas_from_packed,
    body_inside_fraction,
)


# ---------------------------------------------------------------------------
# areas
# ---------------------------------------------------------------------------
def test_areas_from_packed_count_set_bits_and_come_back_as_T_by_C():
    """`sam3_masks.npz` stores (C, T, H, ceil(W/8)) uint8 per fly; every
    downstream array in this pipeline is (T, C). A silent transpose here would
    put camera 0's areas on frame 0 -- the same index-space error that has
    produced confident, self-consistent, completely wrong numbers before."""
    C, T, H, Wp = 3, 4, 2, 2
    packed = np.zeros((C, T, H, Wp), np.uint8)
    packed[0, 0] = 0xFF                       # 2 rows x 2 bytes x 8 bits = 32
    packed[2, 3, 0, 0] = 0b1010_0000          # 2 bits
    area = mask_areas_from_packed(packed)
    assert area.shape == (T, C)
    assert area[0, 0] == 32
    assert area[3, 2] == 2
    assert area.sum() == 34


def test_areas_from_masks_and_from_packed_agree():
    rng = np.random.default_rng(0)
    C, T, H, W = 3, 5, 8, 13
    m = rng.random((T, C, H, W)) < 0.3
    packed = np.packbits(m.transpose(1, 0, 2, 3), axis=-1)
    assert np.array_equal(mask_areas_from_masks(m),
                          mask_areas_from_packed(packed, width=W))


def test_packed_areas_ignore_the_pad_bits_when_width_is_given():
    """np.packbits pads the last byte with zeros, so ignoring `width` is safe
    for a real mask -- but a caller that packs a mask whose last bits are set
    would over-count. `width` makes the crop explicit."""
    C, T, H, W = 1, 1, 1, 4
    m = np.ones((T, C, H, W), bool)
    packed = np.packbits(m.transpose(1, 0, 2, 3), axis=-1)     # 0b1111_0000
    assert mask_areas_from_packed(packed, width=W)[0, 0] == 4
    assert mask_areas_from_packed(packed)[0, 0] == 4           # pad bits are 0


# ---------------------------------------------------------------------------
# the reference area, and slivers relative to it
# ---------------------------------------------------------------------------
def test_the_area_reference_is_per_camera_and_survives_a_bad_quarter():
    """The reference is the fly's OWN healthy area, per camera, and bout 28
    fly0 spends its last quarter collapsed. A MEDIAN reference would be dragged
    down by a long enough bad stretch and the slivers would then look normal;
    the p75 default is what keeps the reference on the healthy population."""
    T, C = 400, 2
    area = np.full((T, C), 1000.0)
    area[300:] = 100.0                       # a bad quarter
    valid = np.ones((T, C), bool)
    ref = area_reference(area, valid)
    assert ref.shape == (C,)
    assert np.allclose(ref, 1000.0)
    assert np.median(area[:, 0]) == 1000.0   # (a median would survive 25% ...)
    area[199:] = 100.0                       # ... but not 50%
    assert np.median(area[:, 0]) == 100.0
    assert np.allclose(area_reference(area, valid, pct=75.0), 1000.0)


def test_invalid_frames_do_not_enter_the_reference():
    T, C = 10, 1
    area = np.full((T, C), 5.0)
    area[:5] = 1000.0
    valid = np.ones((T, C), bool)
    valid[:5] = False
    assert np.allclose(area_reference(area, valid), 5.0)


def test_a_camera_with_no_valid_frame_gets_a_nan_reference_not_a_zero():
    """A zero reference makes `area / ref` infinite and every sliver look
    healthy. NaN propagates to "not usable", which is the truthful answer."""
    area = np.zeros((4, 1))
    valid = np.zeros((4, 1), bool)
    assert np.isnan(area_reference(area, valid)[0])


# ---------------------------------------------------------------------------
# the per-(frame, camera) validity gate
# ---------------------------------------------------------------------------
def test_camera_validity_rejects_slivers_relative_to_that_cameras_own_reference():
    """Cameras see the fly at very different scales (bout 28 fly0: 14 238 px on
    Cam2012861 against 20 156 px on Cam2012853), so an ABSOLUTE area floor
    would reject a whole camera or accept a sliver depending on which one it
    is. The test is a fraction of that camera's own healthy area."""
    area = np.array([[20000.0, 14000.0],
                     [20000.0, 2000.0],       # cam 1 sliver (14% of its own)
                     [4000.0, 14000.0]])      # cam 0 sliver (20% of its own)
    valid = np.ones((3, 2), bool)
    ref = np.array([20000.0, 14000.0])
    ok = camera_validity(valid, area, ref, min_area_frac=0.25)
    assert ok.tolist() == [[True, True], [True, False], [False, True]]


def test_the_gate_is_POSE_AWARE_and_that_is_not_the_same_as_area():
    """THE POINT OF THE WHOLE GATE. A wrong-fly or merged-fly mask has a
    perfectly normal area; only its DISAGREEMENT WITH THE POSE gives it away.
    Frame 1 below is full-size and would pass any area test."""
    area = np.full((3, 1), 20000.0)
    valid = np.ones((3, 1), bool)
    ref = np.array([20000.0])
    inside = np.array([[0.95], [0.30], [0.88]])       # frame 1: the wrong fly
    assert camera_validity(valid, area, ref, min_area_frac=0.25
                           ).ravel().tolist() == [True, True, True]
    ok = camera_validity(valid, area, ref, min_area_frac=0.25,
                         body_inside=inside, min_body_inside=0.5)
    assert ok.ravel().tolist() == [True, False, True]


def test_an_invalid_camera_stays_invalid_however_good_its_area_looks():
    ok = camera_validity(np.zeros((2, 1), bool), np.full((2, 1), 20000.0),
                         np.array([20000.0]), min_area_frac=0.25)
    assert not ok.any()


def test_frame_keep_counts_usable_cameras_not_valid_ones():
    ok = np.array([[True, True, True, False],
                   [True, True, False, False],
                   [False, False, False, False]])
    assert frame_keep(ok, min_cameras=3).tolist() == [True, False, False]
    assert frame_keep(ok, min_cameras=2).tolist() == [True, True, False]


# ---------------------------------------------------------------------------
# the envelope -- the half of the gate that a band-limited mode needs
# ---------------------------------------------------------------------------
def test_the_envelope_is_exactly_zero_on_every_skipped_frame():
    """"Skip" has to mean the STAC pose, not "decay toward it". In `spline`
    mode zeroing `present` only stops the frame contributing to the COST; the
    knots still interpolate across it, so without this multiplier the skipped
    frames keep a fitted correction."""
    keep = np.array([True] * 10 + [False] * 5 + [True] * 10)
    for ramp in (0, 4, 32):
        env = gate_envelope(keep, ramp=ramp)
        assert env.shape == keep.shape
        assert np.all(env[~keep] == 0.0), f"ramp={ramp}"
        assert env.min() >= 0.0 and env.max() <= 1.0


def test_the_envelope_ramps_no_faster_than_the_knot_spacing():
    """A HARD 0/1 step would be broadband -- exactly the content `spline` and
    `lowpass` exist to keep out of the song band. The ramp is linear over one
    knot spacing, so the envelope can add nothing faster than the basis can
    already express."""
    keep = np.ones(200, bool)
    keep[100] = False
    env = gate_envelope(keep, ramp=32)
    assert env[100] == 0.0
    assert np.max(np.abs(np.diff(env))) <= 1.0 / 32 + 1e-6
    assert env[100 - 32] == pytest.approx(1.0) and env[100 + 32] == pytest.approx(1.0)
    assert env[100 + 16] == pytest.approx(0.5, abs=1e-6)


def test_ramp_zero_is_a_hard_gate_which_is_what_free_mode_wants():
    """In `param_mode='free'` the correction is already independent per frame,
    so there is nothing to smear and nothing to protect: a hard gate is both
    correct and the minimum change to the arm the negative result was measured
    on."""
    keep = np.array([True, True, False, True])
    assert gate_envelope(keep, ramp=0).tolist() == [1.0, 1.0, 0.0, 1.0]


def test_an_all_kept_bout_has_an_all_ones_envelope():
    assert np.all(gate_envelope(np.ones(50, bool), ramp=32) == 1.0)


def test_an_all_skipped_bout_has_an_all_zero_envelope():
    assert np.all(gate_envelope(np.zeros(50, bool), ramp=32) == 0.0)


# ---------------------------------------------------------------------------
# the pose-aware signal itself
# ---------------------------------------------------------------------------
def test_body_inside_fraction_counts_projected_vertices_landing_in_the_mask():
    T, C, H, W = 2, 2, 10, 10
    masks = np.zeros((T, C, H, W), bool)
    masks[:, :, 2:8, 2:8] = True
    uv = np.zeros((T, C, 4, 2), np.float32)
    uv[0, 0] = [[3, 3], [4, 4], [5, 5], [6, 6]]        # all inside
    uv[0, 1] = [[3, 3], [4, 4], [0, 0], [9, 9]]        # half inside
    uv[1, 0] = [[0, 0], [9, 0], [0, 9], [9, 9]]        # none inside
    uv[1, 1] = [[3, 3], [np.nan, np.nan], [4, 4], [5, 5]]   # NaN dropped
    valid = np.ones((T, C), bool)
    f = body_inside_fraction(uv, masks, valid)
    assert f.shape == (T, C)
    assert f[0, 0] == pytest.approx(1.0)
    assert f[0, 1] == pytest.approx(0.5)
    assert f[1, 0] == pytest.approx(0.0)
    assert f[1, 1] == pytest.approx(1.0), "a non-finite vertex must be dropped, not counted"


def test_body_inside_fraction_is_nan_where_the_camera_is_invalid():
    masks = np.ones((1, 1, 4, 4), bool)
    uv = np.zeros((1, 1, 2, 2), np.float32)
    assert np.isnan(body_inside_fraction(uv, masks, np.zeros((1, 1), bool))[0, 0])


# ---------------------------------------------------------------------------
# the bout-level ranking
# ---------------------------------------------------------------------------
def test_bout_fly_quality_separates_a_clean_fly_from_a_collapsed_one():
    T, C = 400, 7
    good_area = np.full((T, C), 20000.0)
    good_valid = np.ones((T, C), bool)
    bad_area = good_area.copy()
    bad_valid = good_valid.copy()
    bad_area[300:, :5] = 1500.0              # 5 of 7 cameras collapse
    bad_valid[300:, 5:] = False              # the other 2 drop out
    cents = np.zeros((T, C, 2), np.float32)

    # window=100 == the length of the bad stretch, so a fully-bad block exists
    g = bout_fly_quality(good_area, good_valid, cents, window=100)
    b = bout_fly_quality(bad_area, bad_valid, cents, window=100)
    assert g["frac_frames_ok"] == 1.0 and g["sliver_frac"] == 0.0
    assert b["frac_frames_ok"] == pytest.approx(0.75)
    # 5 cameras x 100 frames of sliver, over the 2600 (frame, camera) pairs
    # that are still flagged valid
    assert b["sliver_frac"] == pytest.approx(500 / 2600)
    assert g["score"] > b["score"]
    # and the WORST WINDOW is reported, because a bout-average hides a bad
    # quarter -- which is exactly how fly0's 1500-2006 stayed unscored.
    assert b["worst_window_frac_ok"] == 0.0
    assert g["worst_window_frac_ok"] == 1.0


def test_bout_fly_quality_reports_the_per_frame_keep_flags_it_scored():
    T, C = 100, 4
    area = np.full((T, C), 1000.0)
    area[50:60] = 10.0
    valid = np.ones((T, C), bool)
    q = bout_fly_quality(area, valid, np.zeros((T, C, 2), np.float32))
    keep = q["frame_ok"]
    assert keep.shape == (T,)
    assert keep[:50].all() and not keep[50:60].any() and keep[60:].all()


# ---------------------------------------------------------------------------
# the whole gate, in one call
# ---------------------------------------------------------------------------
def test_wing_fit_validity_gate_separates_its_two_rejection_reasons():
    """"Sliver" and "disagrees with the pose" send a reader to different places
    -- SAM coverage vs the marker solve -- so they are counted separately, the
    same way `n_thin_frames` and `n_no_bridge_frames` already are."""
    from jarvis_jax.tracking.mask_quality import wing_fit_validity_gate
    T, C, H, W = 4, 3, 20, 20
    masks = np.zeros((T, C, H, W), bool)
    masks[:, :, 4:16, 4:16] = True                     # 144 px everywhere
    masks[1, 0] = False
    masks[1, 0, 4:6, 4:6] = True                       # frame 1 cam 0: a sliver
    valid = np.ones((T, C), bool)
    uv = np.full((T, C, 4, 2), 10.0, np.float32)       # all inside
    uv[2, 1] = 0.0                                     # frame 2 cam 1: wrong fly
    g = wing_fit_validity_gate(masks, valid, body_uv=uv, min_cameras=3)
    assert g["n_sliver_camera_frames"] == 1
    assert g["n_pose_reject_camera_frames"] == 1
    assert g["camera_ok"][1, 0] == False and g["camera_ok"][2, 1] == False
    assert g["frame_keep"].tolist() == [True, False, False, True]


def test_the_gate_without_a_pose_is_area_only_and_says_so_by_missing_the_wrong_fly():
    """Recorded as a test because it is the failure mode the design note names:
    a wrong-fly mask has perfectly normal area."""
    from jarvis_jax.tracking.mask_quality import wing_fit_validity_gate
    T, C, H, W = 2, 2, 20, 20
    masks = np.zeros((T, C, H, W), bool)
    masks[:, :, 4:16, 4:16] = True
    valid = np.ones((T, C), bool)
    uv = np.full((T, C, 4, 2), 10.0, np.float32)
    uv[1, 0] = 0.0
    assert wing_fit_validity_gate(masks, valid, min_cameras=2
                                  )["frame_keep"].all()
    assert not wing_fit_validity_gate(masks, valid, body_uv=uv, min_cameras=2
                                      )["frame_keep"][1]
