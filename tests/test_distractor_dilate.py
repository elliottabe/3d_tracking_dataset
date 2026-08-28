"""Distractor gray-fill must cover the other fly's LIMBS, not just its body.

A SAM3 mask is the body silhouette; legs, wings and antennae fall outside it
and survived the fill, leaving a fly-shaped object in the crop. Measured on
Session0/2025_10_20_13_20_04 bout_00028, while the female is edge-on against
the wall (her mask falls to 2-5% of its area on five cameras), the detector
labelled that leftover male as the target on 87-100% of frames -- on cameras
where HER mask was 100% valid -- collapsing the two 3D tracks onto one animal.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from jarvis_jax.predict.session_frameset import _dilate, build_frameset  # noqa: E402


def test_dilate_is_a_noop_at_radius_zero():
    m = np.zeros((20, 20), bool); m[8:12, 8:12] = True
    assert np.array_equal(_dilate(m, 0), m)


def test_dilate_grows_the_mask_and_swallows_an_adjacent_thin_limb():
    m = np.zeros((60, 60), bool)
    m[25:35, 25:35] = True                  # body
    limb = np.zeros((60, 60), bool)
    limb[29:31, 35:45] = True               # a 2px-wide leg sticking out
    assert not (_dilate(m, 0) & limb).any()          # body mask misses it
    covered = (_dilate(m, 9) & limb).sum() / limb.sum()
    assert covered > 0.5, f"r=9 only covered {covered:.0%} of the limb"
    assert _dilate(m, 15)[limb].all(), "r=15 should cover the whole limb"


def _frameset(distractor_dilate, gap=200):
    """One camera, target and distractor `gap` px apart, each with a limb."""
    H = W = 448
    NC = 2                                    # build_frameset needs >= 2 valid cams
    frame = np.full((NC, H, W, 3), 200, np.uint8)
    target = np.zeros((NC, H, W), bool)
    dist = np.zeros((NC, H, W), bool)
    cx = W // 2
    dx = cx + gap
    # SAM3 masks cover the BODY ONLY -- that is the whole point. The limbs are
    # dark pixels in the frame that lie OUTSIDE the mask, which is why the
    # plain fill leaves them behind.
    target[:, 200:248, cx - 24:cx + 24] = True            # target body (masked)
    dist[:, 200:248, dx - 24:dx + 24] = True              # distractor body (masked)
    tgt_limb = np.s_[:, 222:226, cx + 24:cx + 70]         # NOT in the mask
    dst_limb = np.s_[:, 222:226, dx - 70:dx - 24]         # NOT in the mask
    for c in range(NC):
        frame[c][target[c]] = 10
        frame[c][dist[c]] = 10
    frame[tgt_limb] = 10
    frame[dst_limb] = 10
    cams = np.tile(np.eye(4, 3, dtype=np.float32)[None], (NC, 1, 1))
    centroids = np.tile(np.array([[cx, 224]], np.float32), (NC, 1))
    valid = np.ones(NC, bool)
    return frame, target, dist, centroids, valid, cams


def test_the_distractors_limb_is_filled_only_once_dilation_is_on(monkeypatch):
    """The regression itself: at 0 the limb survives, at 15 it is filled."""
    import jarvis_jax.predict.session_frameset as sf
    frame, target, dist, centroids, valid, cams = _frameset(0)
    monkeypatch.setattr(sf, "triangulate_dlt_batched",
                        lambda *a, **k: np.zeros((1, 3), np.float32))
    monkeypatch.setattr(sf, "project_center_to_cameras",
                        lambda *a, **k: np.tile(np.array([[224.0, 224.0]], np.float32), (2, 1)))

    limb = np.zeros_like(dist[0]); limb[222:226, 224 + 200 - 70:224 + 200 - 24] = True

    def filled_frac(radius):
        c4, _, _ = sf.build_frameset(frame, target, centroids, valid, cams,
                                     distractor_masks=dist,
                                     distractor_dilate=radius)
        rgb = c4[0, :, :, :3]
        # the limb pixels started at 10; "filled" means they were overwritten
        return float((rgb[limb][:, 0] != 10).mean())

    # The limb is 46 px long, so dilating by r closes the r px nearest the
    # body -- the coverage is geometric, not all-or-nothing. Assert that
    # relationship rather than a hoped-for number.
    assert filled_frac(0) < 0.05, "at r=0 the distractor's limb should survive (the bug)"
    fracs = [filled_frac(r) for r in (0, 9, 15, 25, 46)]
    assert all(b >= a for a, b in zip(fracs, fracs[1:])), f"not monotonic: {fracs}"
    assert fracs[-1] > 0.95, f"a radius >= the limb length should cover it: {fracs[-1]:.2f}"
    assert fracs[2] > 0.25, f"r=15 should close the limb nearest the body: {fracs[2]:.2f}"


def test_the_targets_own_limb_is_never_filled(monkeypatch):
    """Dilation is symmetric: excluding the DILATED target protects its limbs."""
    import jarvis_jax.predict.session_frameset as sf
    frame, target, dist, centroids, valid, cams = _frameset(0, gap=120)
    monkeypatch.setattr(sf, "triangulate_dlt_batched",
                        lambda *a, **k: np.zeros((1, 3), np.float32))
    monkeypatch.setattr(sf, "project_center_to_cameras",
                        lambda *a, **k: np.tile(np.array([[224.0, 224.0]], np.float32), (2, 1)))
    for radius in (0, 9, 15, 25):
        c4, _, _ = sf.build_frameset(frame, target, centroids, valid, cams,
                                     distractor_masks=dist, distractor_dilate=radius)
        rgb = c4[0, :, :, :3]
        tgt = target[0]
        assert (rgb[tgt][:, 0] == 10).all(), f"target pixels filled at r={radius}"


def test_target_mask_channel_is_untouched_by_dilation(monkeypatch):
    import jarvis_jax.predict.session_frameset as sf
    frame, target, dist, centroids, valid, cams = _frameset(0)
    monkeypatch.setattr(sf, "triangulate_dlt_batched",
                        lambda *a, **k: np.zeros((1, 3), np.float32))
    monkeypatch.setattr(sf, "project_center_to_cameras",
                        lambda *a, **k: np.tile(np.array([[224.0, 224.0]], np.float32), (2, 1)))
    a, _, _ = sf.build_frameset(frame, target, centroids, valid, cams,
                                distractor_masks=dist, distractor_dilate=0)
    b, _, _ = sf.build_frameset(frame, target, centroids, valid, cams,
                                distractor_masks=dist, distractor_dilate=15)
    assert np.array_equal(a[..., 3], b[..., 3]), "channel 3 must stay the raw target mask"
