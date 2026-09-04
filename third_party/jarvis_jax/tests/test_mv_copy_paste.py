# tests/test_mv_copy_paste.py
import numpy as np
import jax.numpy as jnp
from mvq_fixtures import make_v12_root, REC


def _two_samples(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1)
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
    gain = float(np.clip((np.median(tgt["crops"][0, c]) + 1.0) / (np.median(src["crops"][0, c]) + 1.0), lo, hi))
    u_s, v_s = np.round(donor_uv_local[k2]).astype(int)
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
