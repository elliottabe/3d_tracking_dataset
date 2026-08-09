"""Tests for SAM3 visibility bookkeeping + mid-bout gap repair.

`valid=False` conflates "outside this camera's field of view" (nothing to fix)
with "in frame and SAM3 missed it" (fixable). `in_frame_codes` separates them;
`find_gap_cameras` picks the fixable ones worth a re-segmentation pass;
`_merge_fill_camera` fills the holes WITHOUT discarding masks SAM3 already got
right (unlike the outlier repair, which replaces a camera wholesale because its
masks are wrong).

Root cause being repaired: SAM3VideoTracker prompts at frame_index 0 and
propagates 'forward' only, so a fly entering a camera's view mid-bout is never
picked up there. See docs and configs/sam3/default.yaml.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from jarvis_jax.predict.sam3_driver import (
    IN_FRAME_NO,
    IN_FRAME_UNKNOWN,
    IN_FRAME_YES,
    _merge_fill_camera,
    find_gap_cameras,
    in_frame_codes,
)


def make_cam_mats(n=4, radius=500.0, height=120.0, W=640, H=480):
    """(C,4,3) DLT matrices for a look-at ring aimed at the origin.

    Real projective cameras, not an identity map: with P = [I | offset] the
    depth coordinate is unconstrained, the DLT null space is 2-dimensional and
    triangulation is degenerate -- so a "simple" stub silently tests nothing.
    """
    K = np.array([[800.0, 0, W / 2], [0, 800.0, H / 2], [0, 0, 1.0]])
    mats = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        C = np.array([radius * np.cos(ang), radius * np.sin(ang), height])
        f = -C / np.linalg.norm(C)
        right = np.cross(f, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, f)
        R = np.stack([right, up, f])
        P = np.hstack([R, (-R @ C)[:, None]])
        mats.append((K @ P).T)                       # (4,3)
    return np.stack(mats)


class FakeRepro:
    """Synthetic ReprojectionTool exposing BOTH the matrices and the NumPy
    point API, so it drives the batched path and the per-point helpers alike."""

    def __init__(self, n_cam=4, W=640, H=480):
        self.camera_matrices = make_cam_mats(n_cam, W=W, H=H)
        self.num_cameras = n_cam
        self.cameras = {f"Cam{i}": object() for i in range(n_cam)}

    def project(self, X):
        Xh = np.concatenate([np.asarray(X, float), [1.0]])
        p = np.einsum("i,cij->cj", Xh, self.camera_matrices)
        return p[:, :2] / p[:, 2:3]

    def reproject_point(self, p3d):
        return self.project(p3d)

    def reconstruct_point(self, points2d, cams_to_use=None):
        pts = np.asarray(points2d, float)
        use = list(range(self.num_cameras)) if cams_to_use is None else list(cams_to_use)
        if len(use) < 2:
            return np.zeros(3)
        rows = []
        for k in use:
            P = self.camera_matrices[k].T                 # (3,4)
            u, v = pts[k]
            rows.append(u * P[2] - P[0])
            rows.append(v * P[2] - P[1])
        _, _, Vh = np.linalg.svd(np.stack(rows))
        Xh = Vh[-1]
        return (Xh / Xh[3])[:3]


TRUE_PT = np.array([12.0, -7.0, 3.0])          # a point all ring cameras see


def obs_for(rt, X=TRUE_PT):
    """(C,2) observations of X in every camera."""
    return rt.project(X)


# ---------------------------------------------------------------------------
# in_frame_codes
# ---------------------------------------------------------------------------

def _cent_from(rt, val, X=TRUE_PT):
    """(A,C,T,2) centroids consistent with X, given a (A,C,T) validity mask."""
    A, C, T = val.shape
    uv = rt.project(X)                                   # (C,2)
    cent = np.zeros((A, C, T, 2))
    cent[:] = uv[None, :, None, :]
    return cent


def test_valid_views_are_in_frame_by_definition():
    rt = FakeRepro(4)
    val = np.ones((1, 4, 3), bool)
    codes = in_frame_codes(_cent_from(rt, val), val, rt, W=640, H=480)
    assert (codes == IN_FRAME_YES).all()


def test_missing_view_inside_bounds_is_a_recoverable_miss():
    rt = FakeRepro(4)
    val = np.ones((1, 4, 1), bool); val[0, 3, 0] = False
    codes = in_frame_codes(_cent_from(rt, val), val, rt, W=640, H=480)
    assert codes[0, 3, 0] == IN_FRAME_YES


def test_missing_view_outside_bounds_is_out_of_fov():
    # Same geometry, but an image so small that the observation falls off it:
    # the fly is elsewhere in the world, not merely unsegmented.
    rt = FakeRepro(4)
    val = np.ones((1, 4, 1), bool); val[0, 3, 0] = False
    codes = in_frame_codes(_cent_from(rt, val), val, rt, W=2, H=2)
    assert codes[0, 3, 0] == IN_FRAME_NO


def test_fewer_than_two_valid_cameras_is_unknown_not_a_guess():
    rt = FakeRepro(4)
    val = np.zeros((1, 4, 1), bool); val[0, 0, 0] = True      # only ONE valid
    codes = in_frame_codes(_cent_from(rt, val), val, rt, W=640, H=480)
    assert codes[0, 0, 0] == IN_FRAME_YES                     # the valid one
    assert (codes[0, 1:, 0] == IN_FRAME_UNKNOWN).all()


def test_invalid_cameras_own_centroid_is_never_used():
    # An invalid view's stored centroid is meaningless; poisoning it must not
    # change any code, since invalid cameras contribute zero rows to the DLT.
    rt = FakeRepro(4)
    val = np.ones((1, 4, 1), bool); val[0, 3, 0] = False
    clean = _cent_from(rt, val)
    dirty = clean.copy(); dirty[0, 3, 0] = [9e4, 9e4]
    a = in_frame_codes(clean, val, rt, W=640, H=480)
    b = in_frame_codes(dirty, val, rt, W=640, H=480)
    assert np.array_equal(a, b)


def test_in_frame_codes_rejects_wrong_shapes():
    with pytest.raises(ValueError, match=r"\(A,C,T,2\)"):
        in_frame_codes(np.zeros((2, 3, 4)), np.zeros((2, 3, 4), bool),
                       FakeRepro(3), 10, 10)


def test_bout22_shape_three_of_seven_cameras_still_resolves():
    # The real bout 22 geometry: 3 cameras see her, 4 do not. With >=2 valid
    # sources every missing view must RESOLVE (yes/no), never stay unknown --
    # that is what lets the bout be judged instead of silently dropped.
    rt = FakeRepro(7)
    val = np.zeros((1, 7, 5), bool); val[0, :3] = True
    codes = in_frame_codes(_cent_from(rt, val), val, rt, W=640, H=480)
    assert (codes[0, :3] == IN_FRAME_YES).all()
    assert (codes[0, 3:] != IN_FRAME_UNKNOWN).all()


def test_batched_result_matches_a_per_point_reference():
    # The batched form replaced a per-(fly,frame,camera) loop; it must agree
    # with the straightforward implementation, not merely run faster.
    rt = FakeRepro(5)
    rng = np.random.default_rng(0)
    A, C, T = 2, 5, 6
    val = rng.random((A, C, T)) > 0.35
    cent = _cent_from(rt, val) + rng.normal(0, 0.01, (A, C, T, 2))
    # min_support=2 to match the reference's `len(src) < 2` rule; the default
    # of 3 is a separate policy decision, covered by its own tests.
    got = in_frame_codes(cent, val, rt, W=640, H=480, min_support=2)

    exp = np.full((A, C, T), IN_FRAME_UNKNOWN, np.int8)
    for f in range(A):
        for t in range(T):
            src = [i for i in range(C) if val[f, i, t]]
            for k in range(C):
                if val[f, k, t]:
                    exp[f, k, t] = IN_FRAME_YES
                    continue
                if len(src) < 2:
                    continue
                pts = np.zeros((C, 2))
                for i in src:
                    pts[i] = cent[f, i, t]
                rp = rt.reproject_point(rt.reconstruct_point(pts, cams_to_use=src))[k]
                exp[f, k, t] = (IN_FRAME_YES if (0 <= rp[0] < 640 and 0 <= rp[1] < 480)
                                else IN_FRAME_NO)
    assert np.array_equal(got, exp)


# ---------------------------------------------------------------------------
# find_gap_cameras
# ---------------------------------------------------------------------------

def test_gap_found_when_in_frame_but_invalid():
    A, C, T = 1, 3, 100
    val = np.ones((A, C, T), bool); val[0, 1, :50] = False
    codes = np.full((A, C, T), IN_FRAME_YES, np.int8)
    gaps = find_gap_cameras(val, codes, min_frames=30, min_frac=0.02)
    assert gaps == [(0, 1, 50)]


def test_out_of_fov_frames_are_not_a_gap():
    # This is the distinction that matters: an unobservable fly must never
    # trigger a (pointless, GPU-expensive) re-segmentation pass.
    A, C, T = 1, 3, 100
    val = np.ones((A, C, T), bool); val[0, 1, :50] = False
    codes = np.full((A, C, T), IN_FRAME_YES, np.int8)
    codes[0, 1, :50] = IN_FRAME_NO
    assert find_gap_cameras(val, codes, min_frames=10, min_frac=0.0) == []


def test_unknown_frames_are_not_a_gap():
    A, C, T = 1, 3, 100
    val = np.ones((A, C, T), bool); val[0, 1, :50] = False
    codes = np.full((A, C, T), IN_FRAME_YES, np.int8)
    codes[0, 1, :50] = IN_FRAME_UNKNOWN
    assert find_gap_cameras(val, codes, min_frames=10, min_frac=0.0) == []


def test_small_gaps_are_ignored_by_both_thresholds():
    A, C, T = 1, 2, 1000
    val = np.ones((A, C, T), bool)
    val[0, 1, :20] = False                     # 20 frames: below min_frames=30
    codes = np.full((A, C, T), IN_FRAME_YES, np.int8)
    assert find_gap_cameras(val, codes, min_frames=30, min_frac=0.0) == []
    val2 = np.ones((A, C, T), bool)
    val2[0, 1, :15] = False                    # 1.5% of the bout: below min_frac
    assert find_gap_cameras(val2, codes, min_frames=10, min_frac=0.02) == []


def test_gaps_sorted_largest_first():
    A, C, T = 2, 3, 200
    val = np.ones((A, C, T), bool)
    val[0, 1, :40] = False
    val[1, 2, :90] = False
    codes = np.full((A, C, T), IN_FRAME_YES, np.int8)
    gaps = find_gap_cameras(val, codes, min_frames=30, min_frac=0.0)
    assert [g[2] for g in gaps] == [90, 40]
    assert gaps[0][:2] == (1, 2)


def test_find_gap_cameras_shape_mismatch_raises():
    with pytest.raises(ValueError, match="!="):
        find_gap_cameras(np.ones((1, 2, 3), bool), np.ones((1, 2, 4), np.int8))


# ---------------------------------------------------------------------------
# _merge_fill_camera
# ---------------------------------------------------------------------------

class FakeBM:
    def __init__(self, n_cam, T, masks, identity_map):
        self.num_cameras = n_cam
        self.num_frames = T
        self.masks = masks
        self.identity_map = identity_map


def _m(val=True):
    a = np.zeros((4, 4), bool)
    if val:
        a[1:3, 1:3] = True
    return a


def test_merge_fill_preserves_existing_masks():
    # The critical difference from repair_outlier_cameras, which replaces a
    # camera wholesale: a gap fill must not discard frames SAM3 got right.
    T = 4
    original = _m()
    masks = [[{7: {"mask": original, "centroid": np.array([1.5, 1.5]),
                   "score": 0.9}} if t < 2 else {} for t in range(T)]]
    bm = FakeBM(1, T, masks, [{7: 0}])
    filled = {0: [_m() for _ in range(T)]}
    n_added = _merge_fill_camera(bm, 0, num_animals=1, T=T, filled=filled)
    assert n_added == 2                        # only the two empty frames
    for t in range(2):
        assert bm.masks[0][t][0]["mask"] is original     # untouched
        assert bm.masks[0][t][0]["score"] == 0.9
    for t in (2, 3):
        assert 0 in bm.masks[0][t]


def test_merge_fill_rekeys_identity_map_to_fly_index():
    T = 2
    masks = [[{7: {"mask": _m(), "centroid": np.array([1.5, 1.5]), "score": 1.0}}
              for _ in range(T)]]
    bm = FakeBM(1, T, masks, [{7: 0}])
    _merge_fill_camera(bm, 0, num_animals=1, T=T, filled={})
    assert bm.identity_map[0] == {0: 0}
    assert 0 in bm.masks[0][0]                 # re-keyed from obj_id 7 to fly 0


def test_merge_fill_ignores_empty_candidate_masks():
    T = 3
    bm = FakeBM(1, T, [[{} for _ in range(T)]], [{}])
    filled = {0: [np.zeros((4, 4), bool), None, _m()]}
    n_added = _merge_fill_camera(bm, 0, num_animals=1, T=T, filled=filled)
    assert n_added == 1                        # only the non-empty one
    assert bm.masks[0][2][0]["mask"].any()


def test_merge_fill_drops_flies_beyond_num_animals():
    T = 1
    masks = [[{7: {"mask": _m(), "centroid": np.array([1.5, 1.5]), "score": 1.0},
               9: {"mask": _m(), "centroid": np.array([2.5, 2.5]), "score": 1.0}}]]
    bm = FakeBM(1, T, masks, [{7: 0, 9: 5}])   # obj 9 maps outside num_animals
    _merge_fill_camera(bm, 0, num_animals=2, T=T, filled={})
    assert set(bm.masks[0][0]) == {0}


# ---------------------------------------------------------------------------
# as_numpy_repro: two ReprojectionTool classes, two incompatible APIs
# ---------------------------------------------------------------------------

from jarvis_jax.predict.sam3_driver import (  # noqa: E402
    _camera_matrices, as_numpy_repro)


class FakeJarvisTool:
    """Mimics jarvis.utils.reprojection.ReprojectionTool's torch API.

    Camera k observes (x + k, y): a shift per camera, so reprojection is
    invertible and camera SELECTION (via zeroed maxvals) is observable.
    """

    def __init__(self, n_cam=4):
        self.num_cameras = n_cam
        self.cameras = {f"Cam{i}": object() for i in range(n_cam)}
        self.device = "cpu"
        self.last_maxvals = None
        # The real torch tool carries these; _camera_matrices reads them
        # through the adapter, so the stub must too.
        self.cameraMatrices = make_cam_mats(n_cam)

    def reprojectPoint(self, point3D):
        # The real tool ends with `[:, :2].permute(0, 2, 1).squeeze()`, so a
        # SINGLE point comes back as (C,2), not (N,3,C). The first version of
        # this fake returned the un-squeezed intermediate shape -- the tests
        # passed and production raised IndexError. A fake that does not match
        # the real return contract tests nothing.
        import torch
        p = point3D.reshape(-1, 3)
        n = p.shape[0]
        out = torch.zeros((n, self.num_cameras, 2))
        for k in range(self.num_cameras):
            out[:, k, 0] = p[:, 0] + k
            out[:, k, 1] = p[:, 1]
        return out.squeeze()

    def reconstructPoint(self, points, maxvals):
        import torch
        self.last_maxvals = maxvals.detach().cpu().numpy().reshape(-1)
        w = maxvals.reshape(-1)
        idx = torch.nonzero(w > 0).flatten()
        xs = torch.stack([points[0, k] - k for k in idx])
        ys = torch.stack([points[1, k] for k in idx])
        return torch.stack([xs.mean(), ys.mean(), torch.tensor(0.0)])


def test_numpy_tool_passes_through_unchanged():
    rt = FakeRepro(3)
    assert as_numpy_repro(rt) is rt


def test_jarvis_tool_is_adapted_to_the_numpy_api():
    ad = as_numpy_repro(FakeJarvisTool(4))
    assert hasattr(ad, "reconstruct_point") and hasattr(ad, "reproject_point")
    assert ad.num_cameras == 4


def test_adapter_roundtrips_a_point():
    ad = as_numpy_repro(FakeJarvisTool(4))
    obs = np.stack([[10.0 + k, 20.0] for k in range(4)])       # (C,2)
    X = ad.reconstruct_point(obs)
    assert np.allclose(X[:2], [10.0, 20.0], atol=1e-4)
    rp = ad.reproject_point(X)
    assert rp.shape == (4, 2)
    assert np.allclose(rp, obs, atol=1e-4)


def test_adapter_camera_selection_zeroes_excluded_maxvals():
    # JARVIS selects cameras by ZEROING maxvals, not by an index list -- the
    # leave-one-out residual depends on that translation being right.
    tool = FakeJarvisTool(4)
    ad = as_numpy_repro(tool)
    obs = np.stack([[10.0 + k, 20.0] for k in range(4)])
    ad.reconstruct_point(obs, cams_to_use=[0, 2])
    assert list(tool.last_maxvals) == [1.0, 0.0, 1.0, 0.0]


def test_adapter_returns_zeros_below_two_cameras():
    ad = as_numpy_repro(FakeJarvisTool(4))
    obs = np.zeros((4, 2))
    assert np.allclose(ad.reconstruct_point(obs, cams_to_use=[1]), 0.0)


def test_unknown_tool_raises_instead_of_failing_later_in_a_try_except():
    # Both repair paths are best-effort try/except, so an API mismatch that
    # only shows up mid-run reads as a warning and silently disables the
    # feature -- which is exactly how repair_outlier_cameras went unnoticed.
    class Nothing:
        pass
    with pytest.raises(TypeError, match="neither the NumPy"):
        as_numpy_repro(Nothing())


def test_in_frame_codes_works_through_the_adapter():
    # The batched path reads camera matrices, which must resolve through the
    # adapter to the wrapped torch tool's `cameraMatrices`.
    tool = FakeJarvisTool(4)
    ad = as_numpy_repro(tool)
    ref = FakeRepro(4)
    val = np.ones((1, 4, 2), bool); val[0, 3] = False
    cent = np.zeros((1, 4, 2, 2))
    cent[:] = ref.project(TRUE_PT)[None, :, None, :]
    codes = in_frame_codes(cent, val, ad, W=640, H=480)
    assert (codes[0, :3] == IN_FRAME_YES).all()
    assert (codes[0, 3] != IN_FRAME_UNKNOWN).all()


def test_camera_matrices_resolve_through_the_adapter():
    from jarvis_jax.predict.sam3_driver import _camera_matrices
    tool = FakeJarvisTool(4)
    direct = _camera_matrices(tool)
    viaad = _camera_matrices(as_numpy_repro(tool))
    assert direct.shape == (4, 4, 3)
    assert np.allclose(direct, viaad)


def test_camera_matrices_missing_raises():
    from jarvis_jax.predict.sam3_driver import _camera_matrices
    class Nothing:
        pass
    with pytest.raises(AttributeError, match="camera_matrices"):
        _camera_matrices(Nothing())


def test_adapter_reproject_returns_C_by_2():
    # Pins the shape contract that broke in production: the real tool squeezes
    # to (C,2) for a single point.
    ad = as_numpy_repro(FakeJarvisTool(7))
    out = ad.reproject_point(np.array([1.0, 2.0, 3.0]))
    assert out.shape == (7, 2)


REAL_CAL = ('/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/'
            'Session0/2025_10_20_13_20_04/calibration')


@pytest.mark.skipif(not os.path.isdir(REAL_CAL), reason="calibration not present")
def test_adapter_matches_the_numpy_tool_on_real_calibration():
    """Both ReprojectionTool flavours must agree on real calibration.

    The unit tests above run against stubs; this pins the two REAL
    implementations together, which is where the API mismatch actually bit.
    """
    import glob
    sys.path.insert(0, '/mmfs1/gscratch/portia/eabe/Research/MyRepos/'
                       '3d_tracking_dataset/third_party/JARVIS-HybridNet')
    from jarvis.utils.reprojection import ReprojectionTool as JarvisRT
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool as JaxRT
    names = {os.path.splitext(os.path.basename(p))[0]: os.path.basename(p)
             for p in sorted(glob.glob(REAL_CAL + '/Cam*.yaml'))}
    ad = as_numpy_repro(JarvisRT(root_dir=REAL_CAL, calib_paths=names, device='cpu'))
    ref = JaxRT(REAL_CAL)

    X = np.array([200.0, 15.0, 20.0])
    a, b = ad.reproject_point(X), ref.reproject_point(X)
    assert a.shape == b.shape == (ref.num_cameras, 2)
    assert np.abs(a - b).max() < 1e-2
    assert np.abs(ad.reconstruct_point(b) - ref.reconstruct_point(b)).max() < 1e-2
    assert np.abs(ad.reconstruct_point(b, cams_to_use=[0, 2, 4])
                  - ref.reconstruct_point(b, cams_to_use=[0, 2, 4])).max() < 1e-2
    assert np.allclose(_camera_matrices(ad), _camera_matrices(ref))


# ---------------------------------------------------------------------------
# min_support: two views are not enough to judge visibility
# ---------------------------------------------------------------------------

def test_two_views_are_unknown_by_default():
    # With exactly 2 valid cameras the DLT is a ray-ray intersection whose
    # depth is barely constrained, so the reprojection is not trustworthy.
    rt = FakeRepro(5)
    val = np.zeros((1, 5, 1), bool); val[0, :2, 0] = True
    codes = in_frame_codes(_cent_from(rt, val), val, rt, W=640, H=480)
    assert (codes[0, 2:, 0] == IN_FRAME_UNKNOWN).all()


def test_three_views_resolve():
    rt = FakeRepro(5)
    val = np.zeros((1, 5, 1), bool); val[0, :3, 0] = True
    codes = in_frame_codes(_cent_from(rt, val), val, rt, W=640, H=480)
    assert (codes[0, 3:, 0] != IN_FRAME_UNKNOWN).all()


def test_min_support_is_tunable_down_to_two():
    rt = FakeRepro(5)
    val = np.zeros((1, 5, 1), bool); val[0, :2, 0] = True
    codes = in_frame_codes(_cent_from(rt, val), val, rt, W=640, H=480,
                           min_support=2)
    assert (codes[0, 2:, 0] != IN_FRAME_UNKNOWN).all()


def test_min_support_never_drops_below_two():
    # One view cannot triangulate at all, whatever min_support says.
    rt = FakeRepro(5)
    val = np.zeros((1, 5, 1), bool); val[0, 0, 0] = True
    codes = in_frame_codes(_cent_from(rt, val), val, rt, W=640, H=480,
                           min_support=1)
    assert (codes[0, 1:, 0] == IN_FRAME_UNKNOWN).all()


def test_low_support_never_manufactures_gap_work():
    # The cost guard: unreliable classifications must not send the GPU chasing
    # frames that are probably not there (bout 22 rerun: 3201 phantom frames).
    A, C, T = 1, 5, 200
    val = np.zeros((A, C, T), bool); val[0, :2] = True     # only 2 supporting
    rt = FakeRepro(C)
    codes = in_frame_codes(_cent_from(rt, val), val, rt, W=640, H=480)
    assert find_gap_cameras(val, codes, min_frames=10, min_frac=0.0) == []
