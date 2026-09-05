# tests/test_lift_mask_containment.py
"""`jarvis_jax.tracking.lift_mvq.mask_containment_filter` -- the per-keypoint
filter that deletes the keypoints of one fly which are sitting on the OTHER
fly, plus its gate/CLI wiring.

WHAT IT IS FOR. Identity in the masked-bout lifter is decided per FRAME
(`pick_mask_pair`) and the collapse guard is a per-frame test, so neither can
see the failure measured on the 30-bout r2 lift of 2025_10_20_13_20_04: in
contact frames -- 264 of the 391 pose-jump frames have the FEMALE's slot empty
-- the male instance absorbs PART of her body (bout 1 frame 364: 17 male
keypoints on the female). The instance is mostly correct, so the fix has to be
per keypoint.

WHAT IS LOAD-BEARING HERE:

  * THE CONJUNCTION. A keypoint is dropped only when it is inside the OTHER
    fly's mask AND outside its own (dilated). Dropping merely for being
    outside its own mask would delete real tarsi -- legs routinely extend past
    a SAM mask -- so `test_outside_both_masks_is_kept` is the check that this
    filter is not a silhouette gate in disguise.
  * THE VIEW QUORUM. One or two bad SAM masks must not be able to delete a
    keypoint; `min_views` cameras have to agree.
  * THE SPIKE RULE'S CONSERVATISM. A 0.3 mm step is ordinary fly motion at
    800 Hz and must survive; only a pure spike (> 2x the threshold) or a step
    into/out of a containment failure is dropped.
  * THE GATE. The filter DELETES keypoints from kp3d.npz, so a filtered and an
    unfiltered lift are different files and the gate string must say which.
  * NO SILENT CHANGE WHEN OFF. `identity="sex"` cannot run the filter (it has
    no human id review to say whose body is whose), so that path must be
    byte-identical to what it was.

CAMERA ORDER (CLAUDE.md). The fake store's camera axis and `cam_mats` are the
SAME canonical order by construction here; `mask_containment_filter` refuses a
kp2d whose camera count disagrees, which is the one order error a pure-geometry
test can detect.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]

CAMS = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
        "Cam2012857", "Cam2012861", "Cam2012862"]

# The same inline affine rig as `tests/test_coarse_centres.py::_affine_rig`
# (7 cameras, each rotating ONE real camera's projection submatrix about a
# DIFFERENT mix of x and y so no world axis sits in a null direction), scaled
# down to a small image so a whole (H,W) mask is cheap to build in a test.
_P_REAL = np.array([[8.1001, 0.0074869, -0.031773, 900.0],
                    [0.0093308, -8.0788, -0.17912, 300.0],
                    [0.0, 0.0, 0.0, 1.0]], np.float64)

H_IMG, W_IMG = 240, 320


def _affine_rig(n_cam=7):
    """(n_cam,4,3) DLT matrices (`P.T`, `p_h @ M`, as
    `ReprojectionTool.camera_matrices`), retargeted onto a 320x240 image."""
    cams = []
    for i in range(n_cam):
        ax = 2 * np.pi * i / n_cam
        ay = 4 * np.pi * i / n_cam
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(ax), -np.sin(ax)],
                       [0, np.sin(ax), np.cos(ax)]])
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                       [0, 1, 0],
                       [-np.sin(ay), 0, np.cos(ay)]])
        P = _P_REAL.copy()
        P[:2, :3] = P[:2, :3] @ (Ry @ Rx)
        P[0, 3], P[1, 3] = W_IMG / 2.0, H_IMG / 2.0
        cams.append(P.T.astype(np.float32))
    return np.stack(cams)


def _project(cam_mats, xyz):
    """(K,3) -> (C,K,2), the convention `project_points` uses."""
    from jarvis_jax.tracking.lift_mvq import project_points
    return project_points(cam_mats, xyz)


class FakeStore:
    """A `BoutMaskStore` with the packed SAM3 npz replaced by two DISCS.

    Per camera and frame, each fly's mask is the disc of radius `radius` px
    around that fly's own body centre reprojected into that camera -- the
    "two blobs per camera" the brief asks for, and geometrically consistent
    with the rig, so a keypoint placed at a fly's centre really does land
    inside that fly's blob in every view.
    """

    def __init__(self, cam_mats, centres, *, radius=14.0, valid=None,
                 cameras=CAMS, H=H_IMG, W=W_IMG):
        self.cam_mats = np.asarray(cam_mats, np.float64)
        self.centres = np.asarray(centres, np.float64)          # (A,T,3)
        self.n_flies, self.T = self.centres.shape[:2]
        self.cameras = list(cameras)
        self.H, self.W = int(H), int(W)
        self.radius = float(radius)
        C = len(self.cameras)
        self.valid = (np.ones((self.n_flies, C, self.T), bool) if valid is None
                      else np.asarray(valid, bool))
        self.n_unpacks = 0
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        self._yy, self._xx = yy.astype(np.float64), xx.astype(np.float64)

    def valid_at(self, fly, t):
        return np.asarray(self.valid[fly, :, t], bool)

    def centroid_at(self, fly, t):
        return _project(self.cam_mats, self.centres[fly, t][None])[:, 0]

    def mask_at(self, fly, cam_i, t):
        self.n_unpacks += 1
        uv = _project(self.cam_mats, self.centres[fly, t][None])[int(cam_i), 0]
        if not np.isfinite(uv).all():
            return np.zeros((self.H, self.W), bool)
        return ((self._xx - uv[0]) ** 2 + (self._yy - uv[1]) ** 2) <= self.radius ** 2


def _two_fly_scene(K=10, T=1, sep=6.0):
    """Two flies `sep` world units apart, and a male whose keypoints all start
    on his OWN body. Returns (cam_mats, store, kp3d (2,T,K,3))."""
    cam_mats = _affine_rig()
    male_c = np.array([0.0, 0.0, 0.0])
    fem_c = np.array([sep, 0.0, 0.0])
    centres = np.zeros((2, T, 3))
    centres[0, :] = fem_c            # fly0 = FEMALE
    centres[1, :] = male_c           # fly1 = MALE
    store = FakeStore(cam_mats, centres)
    kp3d = np.zeros((2, T, K, 3), np.float32)
    # Male keypoints tightly around his own centre; female's around hers.
    rng = np.random.default_rng(0)
    for f, c in ((0, fem_c), (1, male_c)):
        kp3d[f, :] = (c[None, None] + 0.25 * rng.standard_normal((T, K, 3))).astype(np.float32)
    return cam_mats, store, kp3d


def _kp2d_for(cam_mats, kp3d):
    """(2,T,C,K,2) reprojections of `kp3d` -- what the lifter's kp2d holds."""
    A, T, K = kp3d.shape[:3]
    C = cam_mats.shape[0]
    out = np.full((A, T, C, K, 2), np.nan, np.float32)
    for f in range(A):
        for t in range(T):
            out[f, t] = _project(cam_mats, kp3d[f, t])
    return out


# ---------------------------------------------------------------- rule 1
def test_five_male_keypoints_inside_the_female_blob_are_dropped_and_only_those():
    """The brief's core case: 5 of the male's 10 keypoints are moved ONTO the
    female's body. Exactly those 5 must be NaN in kp3d, in kp2d in EVERY view
    and in conf; the other 5 must be bit-for-bit untouched; the FEMALE must
    lose nothing (her keypoints are on her own body)."""
    from jarvis_jax.tracking.lift_mvq import mask_containment_filter
    cam_mats, store, kp3d = _two_fly_scene(K=10, T=1)
    bad = np.arange(5)
    kp3d[1, 0, bad] = store.centres[0, 0][None] + 0.1 * np.arange(5)[:, None] * 0.0
    kp2d = _kp2d_for(cam_mats, kp3d)
    conf = np.full(kp2d.shape[:-1], 0.8, np.float32)

    out3, out2, outc, rep = mask_containment_filter(kp3d, kp2d, store, cam_mats,
                                                    min_views=3, own_margin=6.0,
                                                    conf_by_fly=conf)

    dropped = ~np.isfinite(out3[1, 0]).all(-1)
    assert np.array_equal(np.flatnonzero(dropped), bad), (
        f"expected exactly keypoints {list(bad)} dropped, got {np.flatnonzero(dropped)}")
    assert not np.isfinite(out2[1, 0, :, bad]).any(), "kp2d must be NaN in EVERY view"
    assert np.isnan(outc[1, 0, :, bad]).all(), "conf must be NaN for a dropped keypoint"
    keep = np.arange(5, 10)
    assert np.array_equal(out3[1, 0, keep], kp3d[1, 0, keep])
    assert np.array_equal(out2[1, 0][:, keep], kp2d[1, 0][:, keep])
    assert np.array_equal(outc[1, 0][:, keep], conf[1, 0][:, keep])
    assert np.isfinite(out3[0]).all(), "the female lost keypoints she should keep"
    assert rep["n_kp_dropped_other_mask"] == {"fly0": 0, "fly1": 5}, rep
    assert rep["n_kp_dropped_spike"] == {"fly0": 0, "fly1": 0}, rep
    assert rep["frac_kp_dropped"]["fly1"] == pytest.approx(0.5)
    assert rep["per_frame"]["n_other_mask"][1][0] == 5


def test_a_keypoint_outside_both_masks_is_kept():
    """Legs extend past a SAM mask and occluded parts project outside it, so
    "outside its own mask" alone must never drop a keypoint. This is the check
    that the filter is a WHOSE-BODY test, not a silhouette gate."""
    from jarvis_jax.tracking.lift_mvq import mask_containment_filter
    cam_mats, store, kp3d = _two_fly_scene(K=6, T=1)
    kp3d[1, 0, 0] = np.array([-14.0, -8.0, 5.0], np.float32)     # far from both flies
    kp2d = _kp2d_for(cam_mats, kp3d)
    out3, _o2, _oc, rep = mask_containment_filter(kp3d, kp2d, store, cam_mats,
                                                  min_views=3, own_margin=6.0)
    # sanity: it really is outside its OWN mask in every view (else the test
    # would pass for the wrong reason)
    uv = _project(cam_mats, kp3d[1, 0, :1])
    n_in_own = sum(bool(store.mask_at(1, c, 0)[int(round(uv[c, 0, 1])),
                                               int(round(uv[c, 0, 0]))])
                   for c in range(len(CAMS))
                   if 0 <= round(uv[c, 0, 0]) < W_IMG and 0 <= round(uv[c, 0, 1]) < H_IMG)
    assert n_in_own == 0, "the probe point was inside its own mask -- test is vacuous"
    assert np.isfinite(out3[1, 0, 0]).all(), "a keypoint outside BOTH masks was dropped"
    assert rep["n_kp_dropped_other_mask"]["fly1"] == 0


def test_a_keypoint_inside_the_female_blob_in_only_two_views_is_kept():
    """The view quorum. Only 2 cameras have BOTH masks valid, so even a
    keypoint sitting squarely on the female cannot reach `min_views=3` -- one
    or two bad SAM masks must not be able to delete a keypoint."""
    from jarvis_jax.tracking.lift_mvq import mask_containment_filter
    cam_mats, store, kp3d = _two_fly_scene(K=6, T=1)
    kp3d[1, 0, 0] = store.centres[0, 0]
    store.valid[0, 2:, 0] = False              # the female's mask valid in 2 cams only
    kp2d = _kp2d_for(cam_mats, kp3d)
    out3, _o2, _oc, rep = mask_containment_filter(kp3d, kp2d, store, cam_mats,
                                                  min_views=3, own_margin=6.0)
    assert np.isfinite(out3[1, 0, 0]).all()
    assert rep["n_kp_dropped_other_mask"]["fly1"] == 0
    assert rep["n_frames_testable"] == 0, "a 2-camera frame is not testable at min_views=3"
    # ... and with all 7 valid, the SAME keypoint is dropped: the quorum is
    # what saved it, not the geometry.
    store.valid[:] = True
    out3b, _b2, _bc, repb = mask_containment_filter(kp3d, kp2d, store, cam_mats,
                                                    min_views=3, own_margin=6.0)
    assert not np.isfinite(out3b[1, 0, 0]).all()
    assert repb["n_kp_dropped_other_mask"]["fly1"] == 1


def _uniform_rig(n_cam=7):
    """7 copies of ONE orthographic-ish camera (`_P_REAL`'s row 2 is
    [0,0,0,1]), so world x scales to px by the SAME 8.1 px/unit in every view.

    Degenerate for triangulation and deliberately so: this filter never
    triangulates, and a uniform scale is what makes "a point 1.9 units from
    its own centre is 15.4 px out in every view" an exact statement instead of
    a per-camera coin flip.
    """
    P = _P_REAL.copy()
    P[0, 3], P[1, 3] = W_IMG / 2.0, H_IMG / 2.0
    return np.stack([P.T.astype(np.float32)] * n_cam)


def test_the_own_mask_is_dilated_so_a_point_just_outside_it_survives():
    """A keypoint one leg segment past its own mask edge, but overlapping the
    female's blob, must survive at `own_margin=6` and be dropped at
    `own_margin=0` -- the dilation is doing real work, not decoration. Legs
    extend past a SAM mask, so this is the margin that stops the filter
    deleting real tarsi near a contacting fly.

    Geometry, at 8.1 px/world unit and a 14-px blob: the flies are 3.0 units
    (24.3 px) apart, the probe sits 1.9 units (15.4 px) from the male -- just
    OUTSIDE his 14-px blob, INSIDE hers (8.9 px from her centre), and inside
    his blob dilated by 6 px (20 px).
    """
    from jarvis_jax.tracking.lift_mvq import mask_containment_filter
    cam_mats = _uniform_rig()
    centres = np.zeros((2, 1, 3))
    centres[0, 0] = [3.0, 0.0, 0.0]                     # fly0 = FEMALE
    centres[1, 0] = [0.0, 0.0, 0.0]                     # fly1 = MALE
    store = FakeStore(cam_mats, centres, radius=14.0)
    kp3d = np.tile(centres[:, :, None, :], (1, 1, 4, 1)).astype(np.float32)
    kp3d[1, 0, 0] = [1.9, 0.0, 0.0]
    kp2d = _kp2d_for(cam_mats, kp3d)
    strict = mask_containment_filter(kp3d, kp2d, store, cam_mats,
                                     min_views=3, own_margin=0.0)[0]
    lenient = mask_containment_filter(kp3d, kp2d, store, cam_mats,
                                      min_views=3, own_margin=6.0)[0]
    assert not np.isfinite(strict[1, 0, 0]).all(), (
        "the probe point is not outside the undilated own mask -- test is vacuous")
    assert np.isfinite(lenient[1, 0, 0]).all(), (
        "a 6-px dilation of the own mask did not save a point just past its edge")


# ---------------------------------------------------------------- rule 2
def test_the_spike_rule_drops_a_lone_1p5mm_jump_and_keeps_a_0p3mm_step():
    """0.3 mm/frame is ordinary fly motion at 800 Hz; 1.5 mm is not. Neither
    keypoint is on the other fly, so this is the temporal rule alone."""
    from jarvis_jax.tracking.lift_mvq import mask_containment_filter
    cam_mats, store, kp3d = _two_fly_scene(K=2, T=3)
    kp3d[1, :, :] = store.centres[1, :, None, :]        # both keypoints at rest
    kp3d[1, 1, 0] += np.array([15.0, 0.0, 0.0], np.float32)   # 1.5 mm spike, frame 1
    kp3d[1, 1:, 1] += np.array([3.0, 0.0, 0.0], np.float32)   # 0.3 mm step, frame 1 on
    kp2d = _kp2d_for(cam_mats, kp3d)
    out3, _o2, _oc, rep = mask_containment_filter(kp3d, kp2d, store, cam_mats,
                                                  min_views=3, own_margin=6.0,
                                                  max_step_units=5.0)
    assert not np.isfinite(out3[1, 1, 0]).all(), "the 1.5 mm spike frame survived"
    assert np.isfinite(out3[1, :, 1]).all(), "the 0.3 mm step was dropped"
    assert rep["n_kp_dropped_spike"]["fly1"] >= 1
    assert rep["n_kp_dropped_other_mask"]["fly1"] == 0


def test_a_moderate_step_is_dropped_only_when_it_also_fails_containment():
    """A step of 1.2x `max_step` is SUSPECT, not proof. It survives on its own
    and is dropped when the frame it lands on is also on the other fly -- and
    then BOTH frames of the pair go, which is the point of the pair rule."""
    from jarvis_jax.tracking.lift_mvq import mask_containment_filter
    cam_mats, store, kp3d = _two_fly_scene(K=2, T=2, sep=6.0)
    kp3d[1, :, :] = store.centres[1, :, None, :]
    # keypoint 0: a 6-unit step to a point that is NOT on the female
    kp3d[1, 1, 0] = store.centres[1, 1] + np.array([0.0, 0.0, 6.0], np.float32)
    kp2d = _kp2d_for(cam_mats, kp3d)
    plain = mask_containment_filter(kp3d, kp2d, store, cam_mats, min_views=3,
                                    own_margin=6.0, max_step_units=5.0)[0]
    assert np.isfinite(plain[1, :, 0]).all(), (
        "a 0.6 mm step with no containment failure must survive")

    # same step magnitude, but frame 1 lands ON the female
    kp3d2 = kp3d.copy()
    kp3d2[1, 1, 0] = store.centres[0, 1]
    kp2d2 = _kp2d_for(cam_mats, kp3d2)
    out3, _o2, _oc, rep = mask_containment_filter(kp3d2, kp2d2, store, cam_mats,
                                                  min_views=3, own_margin=6.0,
                                                  max_step_units=5.0)
    assert not np.isfinite(out3[1, 1, 0]).all(), "the landing frame is on the female"
    assert not np.isfinite(out3[1, 0, 0]).all(), (
        "the frame the keypoint jumped OFF must go too -- that is the pair rule")
    assert rep["n_kp_dropped_other_mask"]["fly1"] == 1
    assert rep["n_kp_dropped_spike"]["fly1"] == 1


def test_a_whole_fly_is_never_nand_by_this_filter():
    """Frame-level identity belongs to `pick_mask_pair` and the collapse
    guard. Even with EVERY male keypoint sitting on the female, the filter
    reports keypoint drops and leaves the female untouched -- it must not
    start deciding frames."""
    from jarvis_jax.tracking.lift_mvq import mask_containment_filter
    cam_mats, store, kp3d = _two_fly_scene(K=8, T=2)
    kp3d[1] = store.centres[0][:, None, :]              # the whole male on the female
    kp2d = _kp2d_for(cam_mats, kp3d)
    out3, _o2, _oc, rep = mask_containment_filter(kp3d, kp2d, store, cam_mats,
                                                  min_views=3, own_margin=6.0)
    assert not np.isfinite(out3[1]).any()
    assert np.isfinite(out3[0]).all()
    assert rep["n_kp_dropped_other_mask"]["fly1"] == 16
    assert rep["enabled"] is True


def test_the_female_slot_may_be_empty_and_the_male_is_still_tested():
    """The measured defect (264 of 391 pose-jump frames) has the female's
    KEYPOINTS missing while her MASK is fine. The test must key on the masks,
    not on the other fly's keypoints, or it is blind exactly where it matters."""
    from jarvis_jax.tracking.lift_mvq import mask_containment_filter
    cam_mats, store, kp3d = _two_fly_scene(K=6, T=1)
    kp3d[0] = np.nan                                     # the female slot is empty
    kp3d[1, 0, :3] = store.centres[0, 0]                 # 3 male keypoints on her body
    kp2d = _kp2d_for(cam_mats, kp3d)
    out3, _o2, _oc, rep = mask_containment_filter(kp3d, kp2d, store, cam_mats,
                                                  min_views=3, own_margin=6.0)
    assert rep["n_kp_dropped_other_mask"]["fly1"] == 3
    assert np.isfinite(out3[1, 0, 3:]).all()


def test_camera_axis_disagreement_is_refused():
    """kp2d's camera axis and `cam_mats` must be the SAME canonical order --
    the CLAUDE.md trap. A count mismatch is the one such error pure geometry
    can detect, and it must raise rather than broadcast."""
    from jarvis_jax.tracking.lift_mvq import mask_containment_filter
    cam_mats, store, kp3d = _two_fly_scene(K=4, T=1)
    kp2d = _kp2d_for(cam_mats, kp3d)[:, :, :5]           # 5 cameras, not 7
    with pytest.raises(ValueError, match="canonical order"):
        mask_containment_filter(kp3d, kp2d, store, cam_mats, min_views=3, own_margin=6.0)


def test_the_filter_is_pure():
    """The inputs must come back unmodified -- `lift_masked_bout` rebinds the
    returned arrays, and an in-place filter would also have mutated the
    caller's `kp3d_mvq` in the returned bookkeeping dict."""
    from jarvis_jax.tracking.lift_mvq import mask_containment_filter
    cam_mats, store, kp3d = _two_fly_scene(K=6, T=2)
    kp3d[1, :, :2] = store.centres[0][:, None, :]
    kp2d = _kp2d_for(cam_mats, kp3d)
    conf = np.full(kp2d.shape[:-1], 0.8, np.float32)
    k3, k2, cf = kp3d.copy(), kp2d.copy(), conf.copy()
    mask_containment_filter(kp3d, kp2d, store, cam_mats, min_views=3,
                            own_margin=6.0, conf_by_fly=conf)
    assert np.array_equal(np.nan_to_num(kp3d, nan=-7), np.nan_to_num(k3, nan=-7))
    assert np.array_equal(np.nan_to_num(kp2d, nan=-7), np.nan_to_num(k2, nan=-7))
    assert np.array_equal(conf, cf)


# ---------------------------------------------------------------- gate string
def _fake_checkpoint(tmp_path, name="final"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 1}')
    (d / "_METADATA").write_text('{"tree": "fake"}')
    return str(d)


def test_the_gate_string_differs_with_containment_on_and_off():
    """The filter DELETES keypoints from kp3d.npz, so a filtered and an
    unfiltered lift are different files: a run that turns it on must not reuse
    bouts lifted without it."""
    from jarvis_jax.tracking.lift_mvq import mvq_gate_signature, mvq_gate_string
    ck = _fake_checkpoint(Path(os.environ.get("PYTEST_TMP", "/tmp")) / "mvqgate")
    on = mvq_gate_string(ck, exist_thresh=0.5, identity="mask", containment="on")
    off = mvq_gate_string(ck, exist_thresh=0.5, identity="mask", containment="off")
    assert on != off
    assert json.loads(on)["containment"] is True
    assert json.loads(off)["containment"] is False
    # `containment` is the EFFECTIVE bool: identity="sex" has no human id
    # review to say whose body is whose, so the filter cannot run there.
    assert mvq_gate_signature(ck, identity="sex", containment="on")["containment"] is False
    # thresholds are deliberately NOT enrolled (mvq_meta.json carries them)
    assert set(json.loads(on)) == {"lifter", "checkpoint", "step", "sha256",
                                   "exist_thresh", "identity", "containment"}


def test_resolved_containment_accepts_the_yaml_boolean():
    """`containment: on` in YAML is the BOOLEAN True under PyYAML 1.1 rules,
    not the string "on" -- a config written the obvious way must not raise."""
    from jarvis_jax.tracking.lift_mvq import resolved_containment
    assert resolved_containment(None) is True                # DEFAULT_CONTAINMENT
    assert resolved_containment(True) is True
    assert resolved_containment(False) is False
    assert resolved_containment("on") is True
    assert resolved_containment("off") is False
    with pytest.raises(ValueError, match="containment"):
        resolved_containment("maybe")
    cfg = OmegaConf.load(REPO / "configs" / "mvq" / "p3a.yaml")
    assert resolved_containment(cfg.containment) is True
    assert int(cfg.containment_min_views) == 3


def test_run_bout_and_the_cli_agree_on_the_gate():
    """`run_bout.stage_b_gate_signature` recomputes the gate from the config
    and REFUSES a bout whose kp3d.npz disagrees, so the config default and the
    CLI default have to produce the same string -- otherwise every freshly
    lifted bout is rejected at Stage B."""
    import ast
    import importlib.util
    from jarvis_jax.tracking.lift_mvq import mvq_gate_string
    ck = _fake_checkpoint(Path(os.environ.get("PYTEST_TMP", "/tmp")) / "mvqgate2")
    cfg = OmegaConf.create({
        "pipeline": {"lifter": "mvq"},
        "mvq": dict(OmegaConf.load(REPO / "configs" / "mvq" / "p3a.yaml"),
                    **{"checkpoint": ck}),
    })
    spec = importlib.util.spec_from_file_location("_rb", REPO / "scripts" / "run_bout.py")
    src = open(REPO / "scripts" / "run_bout.py").read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "stage_b_gate_signature")
    ns = {"json": json, "os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<rb>", "exec"), ns)
    got = ns["stage_b_gate_signature"](cfg)
    # the CLI's own default is --containment on
    want = mvq_gate_string(ck, step=None, exist_thresh=0.5, identity="mask",
                           containment="on")
    assert got == want, f"config gate {got} != CLI gate {want}"
    assert spec is not None


def test_the_cli_exposes_containment_on_by_default():
    """`--containment` must default ON, and the threshold flags must match the
    module constants -- a CLI default that drifted from the config default is
    a Stage-B refusal on every bout."""
    import importlib.util
    from jarvis_jax.tracking import lift_mvq
    spec = importlib.util.spec_from_file_location(
        "_mvq_cli", REPO / "scripts" / "mvq_lift_bout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.build_parser().parse_args(
        ["--session-dir", "/x", "--predictions-dir", "/y", "--out", "/z",
         "--run", "/w", "--bout", "1"])
    assert args.containment == "on"
    assert args.containment_min_views == lift_mvq.CONTAINMENT_MIN_VIEWS
    assert args.containment_own_margin_px == lift_mvq.CONTAINMENT_OWN_MARGIN_PX
    assert args.containment_max_step_units == lift_mvq.CONTAINMENT_MAX_STEP_UNITS
    assert mod.build_parser().parse_args(
        ["--session-dir", "/x", "--predictions-dir", "/y", "--out", "/z",
         "--run", "/w", "--bout", "1", "--containment", "off"]).containment == "off"


# ------------------------------------------------- inside lift_masked_bout
HUMAN_MASKS = {"method": "human_id_review", "male_slot": 1}


class PlacedRunner:
    """A `FakeRunner` whose instances are placed at CHOSEN world positions.

    `test_lift_masked_bout.FakeRunner` puts every instance at
    `100*slot + centre_x`, which is nowhere near a mask; the containment
    filter needs instances that really do sit on a given fly's blob. This one
    reads its output from a `(2,T,K,3)` table indexed by [typed slot, frame],
    so a test can put N male keypoints on the female at frame `t` and then
    assert exactly those N are gone from the written kp3d.npz.

    One window per frame (the two flies always merge) and `batch=1`, so
    `infer` is called once per frame in order -- that is what makes the frame
    counter here line up with the bout's time axis.
    """

    from jarvis_jax.tracking.lift_mvq import MVQRunner as _R
    read_typed = _R.read_typed
    to_pipeline = _R.to_pipeline
    gates_signature = _R.gates_signature
    gates_string = _R.gates_string
    del _R

    def __init__(self, checkpoint, *, kp_names, cam_mats, table, cameras=CAMS,
                 identity="mask", containment=None):
        from jarvis_jax.tracking.lift_mvq import resolved_containment
        self.checkpoint = os.path.abspath(checkpoint)
        self.step, self.step_label = None, "final"
        self.identity = identity
        self.containment = resolved_containment(containment)
        self.kp_names = list(kp_names)
        self.K = len(self.kp_names)
        self.I = 4
        self.cameras = list(cameras)
        self.C = len(self.cameras)
        self.cam_mats = np.asarray(cam_mats, np.float32)
        self.batch = 1
        self.exist_thresh = 0.5
        self.meta = {"keypoint_names": list(self.kp_names)}
        self._gates = None
        self.table = np.asarray(table, np.float32)          # (2,T,K,3): [female, male]
        self._n_windows = 0
        self._n_infer = 0

    def windows(self, frames, present, centres, prompt_mask=None):
        centres = np.atleast_2d(np.asarray(centres, np.float64))
        b = centres.shape[0]
        self._n_windows += 1
        return {"crops": np.zeros((b, 1, self.C, 4, 4, 3), np.uint8),
                "cam_valid": np.ones((b, 1, self.C), bool),
                "M": np.zeros((b, self.C, 2, 3), np.float32),
                "t_local": np.zeros((b, 1, self.C, 2), np.float32),
                "origin": np.zeros((b, self.C, 2), np.int32),
                "centres": centres.astype(np.float32)}

    def infer(self, w, *, prompt_on=None):
        from jarvis_jax.train.matching import SLOT_FEMALE, SLOT_MALE
        b = np.asarray(w["centres"]).shape[0]
        assert b == 1, "PlacedRunner assumes one merged window per frame"
        t = self._n_infer
        self._n_infer += 1
        kp3d = np.zeros((b, self.I, self.K, 3), np.float32)
        kp3d[0, int(SLOT_FEMALE)] = self.table[0, t]
        kp3d[0, int(SLOT_MALE)] = self.table[1, t]
        exist = np.zeros((b, self.I), np.float32)
        exist[:, int(SLOT_FEMALE)] = 0.9
        exist[:, int(SLOT_MALE)] = 0.8
        sex_prob = np.full((b, self.I), 0.5, np.float32)
        sex_prob[:, int(SLOT_FEMALE)] = 0.9
        sex_prob[:, int(SLOT_MALE)] = 0.1
        return {"kp3d": kp3d,
                "kp2d": np.zeros((b, self.I, self.C, self.K, 2), np.float32),
                "vis": np.full((b, self.I, self.C, self.K), 0.8, np.float32),
                "exist": exist, "sex_prob": sex_prob,
                "conf_raw": np.full((b, self.I, self.K), 0.05, np.float32),
                "xyz": np.zeros((b, self.I, self.K, 3), np.float32)}


def _model_names(n=10):
    """`n` distinct keypoint names, EyeL/EyeR among them so the lifter's rigid
    EyeL-EyeR permutation invariant is actually exercised."""
    return ["EyeL", "EyeR"] + [f"KP{i}" for i in range(n - 2)]


def _placed_scene(T=3, K=10, contaminate=None):
    """A two-fly bout with the male's keypoints `contaminate[t]` moved onto the
    female's body at frame t. Returns (cam_mats, store, table, centres)."""
    cam_mats = _affine_rig()
    centres = np.zeros((2, T, 3))
    centres[0, :] = [6.0, 0.0, 0.0]           # fly0 = FEMALE
    centres[1, :] = [0.0, 0.0, 0.0]           # fly1 = MALE
    store = FakeStore(cam_mats, centres)
    rng = np.random.default_rng(1)
    table = np.zeros((2, T, K, 3), np.float32)
    for f in range(2):
        table[f] = centres[f][:, None, :] + 0.25 * rng.standard_normal((T, K, 3))
    for t, idx in (contaminate or {}).items():
        table[1, t, list(idx)] = centres[0, t]
    return cam_mats, store, table, centres


def test_lift_masked_bout_applies_the_filter_and_reports_it(tmp_path):
    """End to end: three male keypoints sit on the female at frame 1 of 3.

    Expectation. Frame 1 loses exactly those three to the CONTAINMENT rule.
    Frames 0 and 2 lose the same three to the PAIR rule -- the keypoint
    travels 6 units (0.6 mm) onto her and back, which is suspect at
    `max_step_units=5` and fails containment at the other end of both pairs,
    so the frame it jumped off and the frame it came back to go with it. The
    other seven keypoints are untouched in every frame, and the female loses
    nothing. If the wiring were wrong rather than the filter, the npz would be
    complete and `containment_report["enabled"]` False."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    K, T = 10, 3
    bad = [0, 1, 2]
    cam_mats, store, table, centres = _placed_scene(T=T, K=K, contaminate={1: bad})
    names = _model_names(K)
    r = PlacedRunner(_fake_checkpoint(tmp_path), kp_names=names, cam_mats=cam_mats,
                     table=table)
    out = tmp_path / "bout"
    res = lift_masked_bout(
        r, [(np.zeros((7, 8, 8, 3), np.uint8), np.ones(7, bool)) for _ in range(T)],
        centres, np.ones((2, T), bool), out_dir=str(out), model_names=names,
        merge_dist_units=100.0, mask_store=store, containment="on",
        mask_sex_meta=HUMAN_MASKS, verbose=False)

    assert res["containment"] is True
    rep = res["containment_report"]
    assert rep["enabled"] is True
    assert rep["n_kp_dropped_other_mask"] == {"fly0": 0, "fly1": 3}, rep
    with np.load(out / "fly1" / "kp3d.npz") as z:
        k3, gates = z["kp3d"], str(z["gates"])
        wname = [str(n) for n in z["kp_names"]]
    assert json.loads(gates)["containment"] is True
    for t in range(T):
        miss = [wname[i] for i in np.flatnonzero(~np.isfinite(k3[t]).all(-1))]
        assert sorted(miss) == sorted(names[i] for i in bad), (t, miss)
    assert rep["n_kp_dropped_spike"] == {"fly0": 0, "fly1": 6}, (
        f"the pair rule should take the same 3 keypoints in frames 0 and 2: {rep}")
    with np.load(out / "fly1" / "kp2d.npz") as z:
        assert not np.isfinite(z["kp2d"][1, :, [wname.index(names[i]) for i in bad]]).any()
        assert np.isnan(z["conf"][1, :, [wname.index(names[i]) for i in bad]]).all()
    with np.load(out / "fly0" / "kp3d.npz") as z:
        assert np.isfinite(z["kp3d"]).all(), "the female lost keypoints"
    meta = json.loads((out / "mvq_meta.json").read_text())
    assert meta["containment"] is True
    assert meta["containment_min_views"] == 3
    assert meta["containment_report"]["n_kp_dropped_other_mask"]["fly1"] == 3
    assert meta["containment_report"]["per_frame"]["n_other_mask"][1] == [0, 3, 0]
    assert meta["containment_report"]["per_frame"]["n_spike"][1] == [3, 0, 3]


def test_containment_off_leaves_the_written_bout_untouched(tmp_path):
    """`--containment off` must reproduce the pre-Task-9 arrays exactly -- the
    filter is the only thing that changed, so the off arm is the control the
    before/after comparison rests on."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    K, T = 10, 3
    cam_mats, store, table, centres = _placed_scene(T=T, K=K, contaminate={1: [0, 1, 2]})
    names = _model_names(K)
    r = PlacedRunner(_fake_checkpoint(tmp_path), kp_names=names, cam_mats=cam_mats,
                     table=table, containment="off")
    out = tmp_path / "bout_off"
    res = lift_masked_bout(
        r, [(np.zeros((7, 8, 8, 3), np.uint8), np.ones(7, bool)) for _ in range(T)],
        centres, np.ones((2, T), bool), out_dir=str(out), model_names=names,
        merge_dist_units=100.0, mask_store=store, containment="off",
        mask_sex_meta=HUMAN_MASKS, verbose=False)
    assert res["containment"] is False
    with np.load(out / "fly1" / "kp3d.npz") as z:
        assert np.isfinite(z["kp3d"]).all(), "containment=off deleted keypoints"
        assert json.loads(str(z["gates"]))["containment"] is False
    assert store.n_unpacks == 0, "containment=off must not unpack a single mask"


def test_identity_sex_is_byte_identical_with_containment_on_or_off(tmp_path):
    """The filter needs the masks' human id review to know whose body is
    whose, so `identity="sex"` cannot run it. That path must therefore be
    BYTE-identical either way -- including the gate string, which is what
    stops a sex-head run being invalidated by a knob that cannot touch it."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    K, T = 10, 3
    cam_mats, store, table, centres = _placed_scene(T=T, K=K, contaminate={1: [0, 1, 2]})
    names = _model_names(K)
    ck = _fake_checkpoint(tmp_path)
    outs = {}
    for arm in ("on", "off"):
        r = PlacedRunner(ck, kp_names=names, cam_mats=cam_mats, table=table,
                         identity="sex", containment=arm)
        d = tmp_path / f"sex_{arm}"
        with pytest.warns(RuntimeWarning) if arm == "on" else _nullctx():
            lift_masked_bout(
                r, [(np.zeros((7, 8, 8, 3), np.uint8), np.ones(7, bool)) for _ in range(T)],
                centres, np.ones((2, T), bool), out_dir=str(d), model_names=names,
                merge_dist_units=100.0, mask_store=store, containment=arm,
                identity="sex", mask_sex_meta=HUMAN_MASKS, verbose=False)
        outs[arm] = d
    for fly in ("fly0", "fly1"):
        for f in ("kp3d.npz", "kp2d.npz"):
            a = (outs["on"] / fly / f).read_bytes()
            b = (outs["off"] / fly / f).read_bytes()
            assert a == b, f"identity=sex {fly}/{f} changed with containment"
    with np.load(outs["on"] / "fly1" / "kp3d.npz") as z:
        assert json.loads(str(z["gates"]))["containment"] is False
    assert store.n_unpacks == 0, "identity=sex must not unpack a single mask"


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
