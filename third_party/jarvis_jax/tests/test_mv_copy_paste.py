# tests/test_mv_copy_paste.py
import numpy as np
import jax.numpy as jnp
import pytest
from mvq_fixtures import make_v12_root, REC


def _two_samples(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    # manifest_n_flies=1: composite()'s target precondition requires
    # unlabelled_sex == SEX_UNKNOWN (no untracked animal in the scene) -- the
    # default 2-fly manifest reports fly1 as present-but-unlabelled at frame 0,
    # which is a legitimate real-world state but not a legal copy-paste TARGET
    # (that's the loader hook's job to filter for, next task). This knob only
    # changes manifest metadata, not any pixel/keypoint/mask array.
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1, manifest_n_flies=1)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    tgt = ds[ds.windows.index((REC, 0, 0))]        # single labelled fly (frame 0)
    src = ds[ds.windows.index((REC, 0, 2))]        # donor: host of frame 2
    return ds, tgt, src


def test_view_shifts_are_exact_affine_translations():
    from jarvis_jax.data.mv_copy_paste import view_shifts
    from jarvis_jax.models.mvq.geometry import project_local
    rng = np.random.default_rng(0)
    M = rng.normal(size=(3, 2, 3)).astype(np.float32); ts = rng.normal(size=(3, 2)).astype(np.float32)
    tt = rng.normal(size=(3, 2)).astype(np.float32); D = np.array([3.0, -2.0, 1.0], np.float32)
    X = rng.normal(size=(5, 3)).astype(np.float32)
    uv_src = np.asarray(project_local(jnp.asarray(X), jnp.asarray(M), jnp.asarray(ts)))          # (K,C,2)
    uv_tgt = np.asarray(project_local(jnp.asarray(X + D), jnp.asarray(M), jnp.asarray(tt)))
    s = view_shifts(M, D, ts, tt)
    np.testing.assert_allclose(uv_src + s[None], uv_tgt, atol=1e-4)


def test_body_plane_axes_are_orthonormal_and_span_the_spread():
    from jarvis_jax.data.mv_copy_paste import body_plane_axes
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 3)) * np.array([10.0, 4.0, 0.5])
    ax = body_plane_axes(pts.astype(np.float32), np.ones(50, bool))
    np.testing.assert_allclose(ax @ ax.T, np.eye(2), atol=1e-5)
    assert abs(ax[:, 2]).max() < 0.2                       # thin axis (z) is the normal, not in-plane


def test_sample_offset_respects_separation_and_plane():
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, sample_offset
    ax = np.array([[1.0, 0, 0], [0, 1.0, 0]], np.float32)
    rng = np.random.default_rng(1)
    for _ in range(50):
        D = sample_offset(rng, ax, CopyPasteParams())
        assert abs(D[2]) < 1e-6 and 8.0 <= np.linalg.norm(D) <= 60.0


def test_composite_labels_are_geometrically_exact_and_host_is_occluded(tmp_path):
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    from jarvis_jax.models.mvq.geometry import project_local
    ds, tgt, src = _two_samples(tmp_path)
    D = np.array([12.0, 0.0, 0.0], np.float32)                      # contact range: donor overlaps the host
    params = CopyPasteParams()
    out = composite(tgt, src, D, params)
    assert out is not None
    assert out["fly_valid"].tolist() == [True, True] and out["fly_sex"][1] == src["fly_sex"][0]
    assert int(out["unlabelled_sex"]) == -1
    # pasted 3D reprojects onto pasted 2D in every valid camera
    uv = np.asarray(project_local(jnp.asarray(out["kp3d_local"][1, 0]), jnp.asarray(out["M"]), jnp.asarray(out["t_local"][0])))
    has = out["has3d"][1, 0]
    for c in range(7):
        np.testing.assert_allclose(uv[has, c], out["kp2d"][1, 0, c][has], atol=1e-3)
    # the donor's pixels landed where its labels say: the crop is brighter under the pasted mask centre
    c = 0; k = np.where(out["vis2d"][1, 0, c])[0][0]
    u, v = np.round(out["kp2d"][1, 0, c, k]).astype(int)
    assert out["crops"][0, c, v, u].max() > tgt["crops"][0, c, v, u].max() or out["crops"][0, c].mean() != tgt["crops"][0, c].mean()
    # EXACT check: pick the donor keypoint nearest the donor mask's own centroid (well
    # inside the blob, so alpha == 1 there even after the 1-px feather) and require the
    # pasted crop equal the donor's own (gain-scaled) pixel there, within +/-2 levels --
    # not just "brighter than before".
    mask_c = src["prompt_mask"][0, c]
    ys, xs = np.nonzero(mask_c)
    centroid = np.array([xs.mean(), ys.mean()])
    donor_uv_local = src["kp2d"][0, 0, c]                              # (K,2), pre-shift
    donor_vis = np.where(src["vis2d"][0, 0, c])[0]
    k2 = donor_vis[np.argmin(np.linalg.norm(donor_uv_local[donor_vis] - centroid, axis=1))]
    lo, hi = params.gain_clip
    # subsampled medians match composite()'s own gain computation exactly (::4)
    gain = float(np.clip((np.median(tgt["crops"][0, c, ::4, ::4]) + 1.0)
                          / (np.median(src["crops"][0, c, ::4, ::4]) + 1.0), lo, hi))
    u_s, v_s = np.round(donor_uv_local[k2]).astype(int)
    assert mask_c[v_s, u_s]                    # the chosen keypoint really is inside the donor mask
    donor_val = np.clip(src["crops"][0, c, v_s, u_s].astype(np.float64) * gain, 0, 255)
    u2, v2 = np.round(out["kp2d"][1, 0, c, k2]).astype(int)
    np.testing.assert_allclose(out["crops"][0, c, v2, u2].astype(np.float64), donor_val, atol=2)
    # host keypoints under the donor mask are now invisible; the host prompt mask lost those pixels
    assert out["vis2d"][0].sum() <= tgt["vis2d"][0].sum()
    assert out["prompt_mask"].sum() <= tgt["prompt_mask"].sum()
    # host labels untouched
    np.testing.assert_array_equal(out["kp2d"][0], tgt["kp2d"][0]); np.testing.assert_array_equal(out["kp3d_local"][0], tgt["kp3d_local"][0])


def test_composite_rejects_when_donor_leaves_crop_or_cameras_missing(tmp_path):
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    ds, tgt, src = _two_samples(tmp_path)
    assert composite(tgt, src, np.array([500.0, 0.0, 0.0], np.float32), CopyPasteParams()) is None
    src2 = dict(src); cv = src["cam_valid"].copy(); cv[0, 0] = False; src2["cam_valid"] = cv
    assert composite(tgt, src2, np.array([12.0, 0.0, 0.0], np.float32), CopyPasteParams()) is None


def test_composite_raises_for_two_fly_target(tmp_path):
    """composite()'s target precondition (exactly one labelled fly, no
    unlabelled animal) is enforced, not silently ignored: a target that
    already has fly 1 labelled must raise, not overwrite it."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1, manifest_n_flies=1)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    tgt2 = ds[ds.windows.index((REC, 0, 1))]        # two-fly frame: fly_valid == [True, True]
    src = ds[ds.windows.index((REC, 0, 2))]
    assert tgt2["fly_valid"].tolist() == [True, True]
    with pytest.raises(ValueError):
        composite(tgt2, src, np.array([12.0, 0.0, 0.0], np.float32), CopyPasteParams())


def test_composite_at_zero_offset_occludes_host_keypoints(tmp_path):
    """D=0: the donor lands exactly on top of the host. Some host keypoints
    must be occluded (cleared from vis2d) -- and every one of them must be a
    keypoint that actually falls under the donor's shifted mask, not just
    'fewer than before' by coincidence."""
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite, view_shifts, _translate
    ds, tgt, src = _two_samples(tmp_path)
    D = np.zeros(3, np.float32)
    out = composite(tgt, src, D, CopyPasteParams())
    assert out is not None
    assert out["vis2d"][0].sum() < tgt["vis2d"][0].sum()          # strictly fewer, not just <=
    shifts = view_shifts(tgt["M"], D, src["t_local"][0], tgt["t_local"][0])
    tv = tgt["cam_valid"][0]
    cleared = tgt["vis2d"][0, 0] & ~out["vis2d"][0, 0]             # (C,K) host keypoints newly invisible
    assert cleared.sum() == 14                                    # measured for this fixture at D=0
    hk = np.round(tgt["kp2d"][0, 0]).astype(int)                  # (C,K,2)
    for c in range(7):
        if not cleared[c].any():
            continue
        assert tv[c]
        m = _translate(src["prompt_mask"][0, c].astype(np.uint8), shifts[c], nearest=True).astype(bool)
        for k in np.where(cleared[c])[0]:
            u, v = hk[c, k]
            assert m[v, u]                                        # cleared only because the donor mask covers it


def test_composite_prompt_mask_and_gain_on_hand_built_samples():
    """Hand-built minimal T=1 samples (C=1, K=2, 64x64) isolate prompt_mask
    subtraction and the gain-matched paste from the real fixture's geometry:
    a uniform-grey (100) target with a 10x10 host mask, and a uniform-grey
    (50) donor with a 10x10 mask placed to overlap the host mask by half when
    pasted at D=0 (identical M/t_local -> shift == 0)."""
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    from jarvis_jax.train.matching import SEX_UNKNOWN

    H = W = 64
    M = np.eye(2, 3, dtype=np.float32)[None]                       # (C=1,2,3)
    t_local = np.zeros((1, 1, 2), np.float32)                       # (T=1,C=1,2), same for tgt & src

    host_mask = np.zeros((H, W), bool); host_mask[20:30, 20:30] = True     # rows 20:30, cols 20:30
    donor_mask = np.zeros((H, W), bool); donor_mask[20:30, 25:35] = True   # rows 20:30, cols 25:35
    overlap = host_mask & donor_mask
    assert host_mask.sum() == 100 and overlap.sum() == 50           # half of the host mask, by construction

    def _sample(crop_value, mask, fly1_valid=False):
        return {
            "crops": np.full((1, 1, H, W, 3), crop_value, np.uint8),
            "cam_valid": np.ones((1, 1), bool),
            "M": M, "t_local": t_local,
            "kp2d": np.zeros((2, 1, 1, 2, 2), np.float32),
            "vis2d": np.zeros((2, 1, 1, 2), bool),
            "kp3d_local": np.zeros((2, 1, 2, 3), np.float32),
            "has3d": np.zeros((2, 1, 2), bool),
            "fly_valid": np.array([True, fly1_valid]),
            "fly_sex": np.array([0, -1], np.int8),
            "unlabelled_sex": np.int8(SEX_UNKNOWN),
            "prompt_mask": mask[None, None],
        }

    tgt = _sample(100, host_mask)
    # host kp0 inside the host/donor overlap (row=25,col=27) -- must be occluded;
    # host kp1 far from any mask (row=5,col=5) -- must stay visible.
    tgt["kp2d"][0, 0, 0, 0] = [27, 25]; tgt["kp2d"][0, 0, 0, 1] = [5, 5]
    tgt["vis2d"][0, 0, 0] = [True, True]

    src = _sample(50, donor_mask)
    src["kp2d"][0, 0, 0, 0] = [30, 25]; src["kp2d"][0, 0, 0, 1] = [30, 25]
    src["vis2d"][0, 0, 0] = [True, True]
    src["has3d"][0, 0] = [True, True]

    out = composite(tgt, src, np.zeros(3, np.float32), CopyPasteParams())
    assert out is not None

    # (i) the host prompt mask lost exactly the overlap pixels
    assert out["prompt_mask"][0, 0].sum() == host_mask.sum() - overlap.sum()

    # (ii) gain = clip(101/51, 0.7, 1.4) == 1.4 (clipped): a pixel 3px inside the
    # donor mask (row=25, col=29 -- >=3px from every donor-mask edge) equals
    # round(50*1.4) == 70, past the 1-px feather.
    lo, hi = CopyPasteParams().gain_clip
    gain = float(np.clip((np.median(tgt["crops"][0, 0, ::4, ::4]) + 1.0)
                          / (np.median(src["crops"][0, 0, ::4, ::4]) + 1.0), lo, hi))
    assert gain == pytest.approx(1.4)
    assert donor_mask[25, 29] and all(donor_mask[25 + dr, 29 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    assert out["crops"][0, 0, 25, 29].tolist() == [70, 70, 70]

    # (iii) host vis2d cleared for the keypoint under the donor mask, kept for the other
    assert out["vis2d"][0, 0, 0].tolist() == [False, True]


def test_composite_rejects_donor_with_no_mask_content_anywhere():
    """A donor whose `prompt_mask` is empty in EVERY target-valid camera has
    nothing to composite: the old code skipped every camera, painted nothing,
    and still wrote the donor's 2D/3D labels into the sample -- a fly claimed
    with zero pixel evidence in any view, the exact invariant `composite`'s
    docstring promises never to break. It must reject the draw instead."""
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    from jarvis_jax.train.matching import SEX_UNKNOWN

    H = W = 64
    M = np.eye(2, 3, dtype=np.float32)[None]
    t_local = np.zeros((1, 1, 2), np.float32)

    def _sample(mask):
        return {
            "crops": np.full((1, 1, H, W, 3), 100, np.uint8),
            "cam_valid": np.ones((1, 1), bool),
            "M": M, "t_local": t_local,
            "kp2d": np.zeros((2, 1, 1, 2, 2), np.float32),
            "vis2d": np.zeros((2, 1, 1, 2), bool),
            "kp3d_local": np.zeros((2, 1, 2, 3), np.float32),
            "has3d": np.zeros((2, 1, 2), bool),
            "fly_valid": np.array([True, False]),
            "fly_sex": np.array([0, -1], np.int8),
            "unlabelled_sex": np.int8(SEX_UNKNOWN),
            "prompt_mask": mask[None, None].copy(),
        }

    host_mask = np.zeros((H, W), bool); host_mask[20:30, 20:30] = True
    tgt = _sample(host_mask)
    src = _sample(np.zeros((H, W), bool))              # donor mask empty everywhere
    src["vis2d"][0, 0, 0] = [True, True]
    src["has3d"][0, 0] = [True, True]
    assert not src["prompt_mask"].any()
    assert composite(tgt, src, np.zeros(3, np.float32), CopyPasteParams()) is None
    # sanity: the SAME donor with mask content is accepted, so the rejection is
    # about the empty mask and nothing else
    src["prompt_mask"] = host_mask[None, None].copy()
    assert composite(tgt, src, np.zeros(3, np.float32), CopyPasteParams()) is not None


# --- T=2 (mvq-v2 plan B task 2): composite() accepts T >= 1 windows -----------------------
#
# NOTE on scope: these tests exercise `mv_copy_paste.composite` ALONE, so they build T=2
# samples straight from `V12WindowDataset(..., T=2, ...)` or by hand rather than through
# `paste_window`. The loader wiring that feeds it (`pair_deltas`, `delta`/`donor_delta`,
# the spacing-matched donor pool, and `__getitem__` pasting at T>1) landed with Task 1 and
# is covered by `tests/test_v12_windows.py::test_t2_copy_paste_pairs_a_donor_of_the_same_spacing`.

def _duplicate_frame_axis(sample):
    """Turn a T=1 sample dict into a T=2 dict whose second frame is a byte-for-byte
    duplicate of the first -- used to show the T>=1 `composite()` path collapses to its
    own T=1 result when both frames of a window are identical."""
    out = dict(sample)
    for k in ("crops", "cam_valid", "t_local", "prompt_mask"):
        out[k] = np.concatenate([sample[k], sample[k]], axis=0)
    for k in ("kp2d", "vis2d", "kp3d_local", "has3d"):
        out[k] = np.concatenate([sample[k], sample[k]], axis=1)
    return out


def test_composite_t2_is_the_t1_path_applied_per_frame(tmp_path):
    """SELF-CONSISTENCY guard (spec §7 risk 3): a T=2 window whose two frames
    are identical must composite to two frames identical to what the SAME
    function returns for the T=1 window -- i.e. the per-frame loop adds no
    cross-frame coupling and treats frame 1 exactly as frame 0.

    It is NOT a byte-equality test against the pre-T2 implementation (both
    sides here run today's code); parity with the original P3a path was
    established by the Task-2 reviewer's line-by-line trace of the loop, and
    this test is what keeps the two frames interchangeable from here on."""
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    ds1, tgt1, src1 = _two_samples(tmp_path)
    D = np.array([12.0, 0.0, 0.0], np.float32)
    out1 = composite(tgt1, src1, D, CopyPasteParams())
    assert out1 is not None
    tgt2, src2 = _duplicate_frame_axis(tgt1), _duplicate_frame_axis(src1)
    out2 = composite(tgt2, src2, D, CopyPasteParams())
    assert out2 is not None
    for k in ("crops", "prompt_mask"):
        np.testing.assert_array_equal(out2[k][0], out1[k][0])
        np.testing.assert_array_equal(out2[k][1], out1[k][0])
    for k in ("kp2d", "vis2d", "kp3d_local", "has3d"):
        np.testing.assert_array_equal(out2[k][:, 0], out1[k][:, 0])
        np.testing.assert_array_equal(out2[k][:, 1], out1[k][:, 0])


def _t2_samples(tmp_path):
    """Real T=2 (consecutive-frame) windows out of V12WindowDataset -- no donor/host
    overlap in EITHER window (host: fly0 frames 0,1; donor: fly0 frames 2,3, the
    2nd-fly frame 3 does not touch fly0's own labels)."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=5, two_fly_frame=3, manifest_n_flies=1)
    ds = V12WindowDataset(root, "train", T=2, train=False)
    tgt = ds[ds.windows.index((REC, 0, 0))]
    src = ds[ds.windows.index((REC, 0, 2))]
    return ds, tgt, src


def test_donor_keeps_its_own_motion_across_the_pair(tmp_path):
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    from jarvis_jax.models.mvq.geometry import project_local
    ds, tgt, src = _t2_samples(tmp_path)
    D = np.array([12.0, 0.0, 0.0], np.float32)          # contact range, as the T=1 fixture tests use
    out = composite(tgt, src, D, CopyPasteParams())
    assert out is not None and out["kp3d_local"].shape[1] == 2
    # the pasted fly moved by exactly the donor's own frame-to-frame motion, not by
    # zero (frozen at frame 0) and not by the host's motion -- checked on keypoints
    # alive in BOTH frames (a keypoint dead in one frame but not the other, per the
    # per-frame rejection rule, legitimately has kp3d_local zeroed only there).
    both_alive = out["has3d"][1, 0] & out["has3d"][1, 1]
    assert both_alive.sum() >= 40                     # sanity: the check below isn't vacuous
    np.testing.assert_allclose(out["kp3d_local"][1, 1][both_alive] - out["kp3d_local"][1, 0][both_alive],
                               src["kp3d_local"][0, 1][both_alive] - src["kp3d_local"][0, 0][both_alive], atol=1e-4)
    # and its 3D still reprojects onto its written 2D in BOTH frames
    for t in range(2):
        uv = np.asarray(project_local(jnp.asarray(out["kp3d_local"][1, t]), jnp.asarray(out["M"]),
                                      jnp.asarray(out["t_local"][t])))
        has = out["has3d"][1, t]
        for c in range(out["crops"].shape[1]):
            np.testing.assert_allclose(uv[has, c], out["kp2d"][1, t, c][has], atol=1e-3)


def test_composite_rejects_donor_of_a_different_window_length(tmp_path):
    """`src` must have the same T as `tgt`: paste_window's donor pool is only
    length-matched, not spacing-matched, so a length mismatch has to be an
    explicit reject rather than an out-of-bounds index."""
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    ds1, tgt1, src1 = _two_samples(tmp_path)
    tgt2 = _duplicate_frame_axis(tgt1)
    D = np.array([12.0, 0.0, 0.0], np.float32)
    assert composite(tgt2, src1, D, CopyPasteParams()) is None       # tgt T=2, src T=1
    assert composite(tgt1, _duplicate_frame_axis(src1), D, CopyPasteParams()) is None  # tgt T=1, src T=2


def test_composite_rejects_the_whole_pair_when_either_frame_has_nothing_to_paste():
    """A donor with mask content in frame 0 but none in frame 1 (in the only
    target-valid camera) must void the WHOLE pair -- not paste frame 0 alone
    and silently drop frame 1's labels."""
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    from jarvis_jax.train.matching import SEX_UNKNOWN

    H = W = 64
    M = np.eye(2, 3, dtype=np.float32)[None]
    t_local = np.zeros((2, 1, 2), np.float32)

    def _sample(masks, fly1_valid=False):
        return {
            "crops": np.full((2, 1, H, W, 3), 100, np.uint8),
            "cam_valid": np.ones((2, 1), bool),
            "M": M, "t_local": t_local,
            "kp2d": np.zeros((2, 2, 1, 2, 2), np.float32),
            "vis2d": np.zeros((2, 2, 1, 2), bool),
            "kp3d_local": np.zeros((2, 2, 2, 3), np.float32),
            "has3d": np.zeros((2, 2, 2), bool),
            "fly_valid": np.array([True, fly1_valid]),
            "fly_sex": np.array([0, -1], np.int8),
            "unlabelled_sex": np.int8(SEX_UNKNOWN),
            "prompt_mask": np.stack(masks)[:, None],
        }

    host_mask = np.zeros((H, W), bool); host_mask[20:30, 20:30] = True
    donor_mask_frame0 = np.zeros((H, W), bool); donor_mask_frame0[20:30, 20:30] = True
    donor_mask_frame1_empty = np.zeros((H, W), bool)             # nothing to paste this frame

    tgt = _sample([host_mask, host_mask])
    src = _sample([donor_mask_frame0, donor_mask_frame1_empty])
    src["vis2d"][0] = True
    src["has3d"][0] = True
    assert composite(tgt, src, np.zeros(3, np.float32), CopyPasteParams()) is None
    # sanity: the same donor with content in BOTH frames is accepted
    src["prompt_mask"][1, 0] = donor_mask_frame0
    assert composite(tgt, src, np.zeros(3, np.float32), CopyPasteParams()) is not None
