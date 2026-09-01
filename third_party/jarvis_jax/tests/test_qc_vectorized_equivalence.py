"""The vectorized QC metrics must reproduce the pre-2026-09-01 Python loops.

qc.py / qc_perframe.py / mesh_iou.py were vectorized to cut qc.json (3m39s) and
qc_perframe.npz (2m52s) on a 2007-frame, 7-camera bout. That is only legitimate
if the NUMBERS do not move, so the reference implementations below are the
verbatim loops that were replaced, and each test asserts equality against them.

Expected tolerances (and why they differ):
  * reprojection-error metrics, hard IoU  -> EXACT (`==`). Batched reprojection
    and stacked SVD are bit-identical, and hard IoU is integer pixel counts.
  * soft IoU                              -> 1e-12 relative. The splat grid is
    bit-identical; only the two final REDUCTIONS are rearranged
    (sum over the bbox + |mask| instead of a full-frame sum), which costs 1-2
    ulp. Measured on real data: max rel 5.9e-16.
"""
import numpy as np
import pytest

from jarvis_jax.tracking import qc as qcmod
from jarvis_jax.tracking import mesh_iou as miou

CAL = np.array([
    [[8.1001, 0.0074869, -0.031773, -2.828],
     [0.0093308, -8.0788, -0.17912, 462.78],
     [0.0, 0.0, 0.0, 1.0]],
    [[-0.15977, 8.0916, -0.20825, -60.121],
     [-0.19148, -0.21031, -8.0742, 421.42],
     [0.0, 0.0, 0.0, 1.0]],
    [[-8.0669, -0.5271, 0.1174, 986.55],
     [-0.11122, 0.20323, -8.0855, 425.63],
     [0.0, 0.0, 0.0, 1.0]],
    [[0.42219, -8.0698, 0.29397, 1041.6],
     [0.20885, 0.32217, -8.0679, 424.19],
     [0.0, 0.0, 0.0, 1.0]],
    [[5.7, 5.7, 0.11, 120.0],
     [0.13, -0.19, -8.09, 430.0],
     [0.0, 0.0, 0.0, 1.0]],
], dtype=np.float64)


class RT:
    """The scalar-only ReprojectionTool API, verbatim from
    jarvis_jax.geometry.reprojection_tool (kept independent of it on purpose:
    these tests must not be able to pass by both sides sharing a bug)."""

    def __init__(self, mats):
        self.mats = np.asarray(mats, np.float64)
        self.num_cameras = len(self.mats)

    def reproject_point(self, p3d):
        p3d = np.asarray(p3d, dtype=np.float64)
        p3d_h = np.concatenate([p3d, [1.0]])
        pts2d = np.zeros((self.num_cameras, 2), dtype=np.float64)
        for i in range(self.num_cameras):
            proj = self.mats[i].dot(p3d_h)
            pts2d[i] = (proj / proj[2])[:2]
        return pts2d

    def reconstruct_point(self, points2d, cams_to_use=None):
        if cams_to_use is None:
            cams_to_use = list(range(self.num_cameras))
        if len(cams_to_use) < 2:
            return np.zeros(3, dtype=np.float64)
        points2d = np.asarray(points2d, dtype=np.float64)
        n = len(cams_to_use)
        A = np.zeros((2 * n, 4), dtype=np.float64)
        for i, cam_idx in enumerate(cams_to_use):
            P = self.mats[cam_idx]
            uv = points2d[cam_idx]
            A[2 * i:2 * i + 2] = uv.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]
        Vh = np.linalg.svd(A)[2]
        X_h = Vh[-1] / Vh[-1][3]
        return X_h[:3]


class BatchRT(RT):
    """Same tool, with the batched primitives qc.py prefers."""

    def reproject_points(self, p3d):
        p3d = np.asarray(p3d, dtype=np.float64)
        ph = np.concatenate([p3d, np.ones((*p3d.shape[:-1], 1))], axis=-1)
        proj = np.einsum('...k,cjk->...cj', ph, self.mats)
        return proj[..., :2] / proj[..., 2:3]

    def reconstruct_points(self, points2d, cams):
        points2d = np.asarray(points2d, float)
        cams = np.asarray(cams, int)
        N, n = cams.shape
        if n < 2:
            return np.zeros((N, 3))
        P = self.mats[cams]
        rows = points2d[..., None] * P[:, :, 2:3, :] - P[:, :, 0:2, :]
        Vh = np.linalg.svd(rows.reshape(N, 2 * n, 4))[2]
        X_h = Vh[:, -1, :]
        return X_h[:, :3] / X_h[:, 3:4]


# ---------------------------------------------------------------------------
# Reference (pre-vectorization) implementations, verbatim.
# ---------------------------------------------------------------------------

def ref_per_camera_reproj_error(rt, kp3d_mm, kp2d_by_cam, vis_by_cam):
    kp3d_mm = np.asarray(kp3d_mm, float)
    out = {}
    for c, kp2d in kp2d_by_cam.items():
        vis = np.asarray(vis_by_cam.get(c, np.ones(len(kp3d_mm), bool)), bool)
        errs = []
        for j in range(len(kp3d_mm)):
            if not vis[j]:
                continue
            xyz = kp3d_mm[j]
            obs = np.asarray(kp2d[j], float)
            if not (np.all(np.isfinite(xyz)) and np.all(np.isfinite(obs))):
                continue
            uv = rt.reproject_point(xyz)[c]
            if not np.all(np.isfinite(uv)):
                continue
            errs.append(float(np.linalg.norm(uv - obs)))
        if errs:
            out[c] = float(np.median(errs))
    return out


def ref_loo_reproj(rt, kp2d_by_cam, vis_by_cam):
    cams = sorted(kp2d_by_cam)
    if not cams:
        return {"per_kp": np.zeros(0), "median": float("nan"), "n": 0}
    n_kp = len(next(iter(kp2d_by_cam.values())))
    per_kp = np.full(n_kp, np.nan)
    all_errs = []
    for j in range(n_kp):
        vis_cams = [c for c in cams
                    if np.asarray(vis_by_cam.get(c, np.ones(n_kp, bool)))[j]]
        if len(vis_cams) < 3:
            continue
        obs = np.zeros((rt.num_cameras, 2))
        for c in vis_cams:
            obs[c] = np.asarray(kp2d_by_cam[c][j], float)
        kp_errs = []
        for held in vis_cams:
            others = [c for c in vis_cams if c != held]
            if len(others) < 2:
                continue
            X = rt.reconstruct_point(obs, cams_to_use=others)
            proj = rt.reproject_point(X)[held]
            kp_errs.append(float(np.linalg.norm(proj - obs[held])))
        if kp_errs:
            per_kp[j] = float(np.mean(kp_errs))
            all_errs.extend(kp_errs)
    return {"per_kp": per_kp,
            "median": float(np.median(all_errs)) if all_errs else float("nan"),
            "n": len(all_errs)}


def ref_ik_reproj_samples(rt, kp3d_mm, kp2d_by_cam, vis_by_cam):
    kp3d_mm = np.asarray(kp3d_mm, float)
    n_kp = len(kp3d_mm)
    idx_out, err_out = [], []
    for c, kp2d in kp2d_by_cam.items():
        vis = np.asarray(vis_by_cam.get(c, np.ones(n_kp, bool)), bool)
        for j in range(n_kp):
            if not vis[j]:
                continue
            xyz = kp3d_mm[j]
            obs = np.asarray(kp2d[j], float)
            if not (np.all(np.isfinite(xyz)) and np.all(np.isfinite(obs))):
                continue
            uv = rt.reproject_point(xyz)[c]
            if not np.all(np.isfinite(uv)):
                continue
            idx_out.append(j)
            err_out.append(float(np.linalg.norm(uv - obs)))
    return np.asarray(idx_out, int), np.asarray(err_out, float)


def ref_iou_of_projected_verts(verts2d, mask_shape, ref_mask) -> float:
    H, W = mask_shape
    pred = np.zeros((H, W), dtype=bool)
    v = np.asarray(verts2d)
    xs = np.round(v[:, 0]).astype(int)
    ys = np.round(v[:, 1]).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    pred[ys[ok], xs[ok]] = True
    ref = np.asarray(ref_mask, dtype=bool)
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return float(inter / union) if union > 0 else 0.0


def ref_soft_iou_of_verts(verts2d, mask, *, sigma=1.3, splat_k=2) -> float:
    mask = np.asarray(mask).astype(np.float32)
    H, W = mask.shape
    if mask.sum() == 0:
        return 0.0
    v = np.asarray(verts2d, dtype=np.float64)
    grid = np.zeros((H, W), dtype=np.float64)
    bx = np.floor(v[:, 0]).astype(int)
    by = np.floor(v[:, 1]).astype(int)
    for di in range(-splat_k, splat_k + 1):
        for dj in range(-splat_k, splat_k + 1):
            cx = bx + dj
            cy = by + di
            w = np.exp(-(((cx + 0.5 - v[:, 0]) ** 2 + (cy + 0.5 - v[:, 1]) ** 2))
                       / (2 * sigma ** 2))
            valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            np.add.at(grid, (np.clip(cy, 0, H - 1), np.clip(cx, 0, W - 1)),
                      np.where(valid, w, 0.0))
    soft = 1.0 - np.exp(-grid)
    inter = float((soft * mask).sum())
    union = float((soft + mask - soft * mask).sum())
    return inter / union if union > 0 else 0.0


def ref_mesh_mask_iou_report(rt, mesh_mm, masks_by_cam):
    mesh_mm = np.asarray(mesh_mm, float)
    hard, soft = {}, {}
    for c, mask in masks_by_cam.items():
        if mask is None or len(mesh_mm) == 0:
            continue
        mask = np.asarray(mask)
        uv = np.stack([rt.reproject_point(mesh_mm[k])[c]
                       for k in range(len(mesh_mm))], 0)
        hard[c] = ref_iou_of_projected_verts(uv, mask.shape, mask)
        soft[c] = ref_soft_iou_of_verts(uv, mask)
    hm = float(np.mean(list(hard.values()))) if hard else float("nan")
    sm = float(np.mean(list(soft.values()))) if soft else float("nan")
    return {"hard": hard, "soft": soft, "hard_mean": hm, "soft_mean": sm}


def ref_per_frame_qc(rt, *, mesh_by_frame, kp3d_by_frame, kp2d_by_frame,
                     vis_by_frame, masks_by_frame):
    T = len(mesh_by_frame)
    soft = np.full(T, np.nan); hard = np.full(T, np.nan)
    reproj = np.full(T, np.nan); ncam = np.zeros(T, int)
    for t in range(T):
        masks_c = {c: m for c, m in masks_by_frame[t].items() if m is not None}
        ncam[t] = len(masks_c)
        if masks_c:
            rep = ref_mesh_mask_iou_report(rt, mesh_by_frame[t], masks_c)
            hv = [v for v in rep["hard"].values() if v is not None]
            sv = [v for v in rep["soft"].values() if v is not None]
            if hv: hard[t] = float(np.median(hv))
            if sv: soft[t] = float(np.median(sv))
        errs = ref_per_camera_reproj_error(rt, kp3d_by_frame[t],
                                          kp2d_by_frame[t], vis_by_frame[t])
        if errs:
            reproj[t] = float(np.median(list(errs.values())))
    return {"soft_iou": soft, "hard_iou": hard, "reproj_px": reproj,
            "n_cams": ncam.astype(np.float32)}


# ---------------------------------------------------------------------------
# A small but nasty synthetic bout: NaN 3-D, NaN 2-D, partial visibility,
# a frame with no visible camera at all, an empty mask, a missing mask.
# ---------------------------------------------------------------------------

H, W = 60, 90
T, K, V = 6, 9, 40


def make_bout(seed=0, nan_obs_visible=True):
    rng = np.random.default_rng(seed)
    C = len(CAL)
    rt = RT(CAL)
    kp3d = rng.normal(0, 4, (T, K, 3)) + np.array([2.0, -3.0, 25.0])
    mesh = rng.normal(0, 5, (T, V, 3)) + np.array([2.0, -3.0, 25.0])
    kp3d[2] = np.nan                                  # whole frame unmeasured
    kp3d[3, 4] = np.nan                               # one bad keypoint
    mesh[4, :3] = np.nan                              # NaN mesh verts
    mesh[5, 0] = [1e6, 1e6, 25.0]                     # projects far out of frame
    kp2d = np.stack([rt.reproject_point(kp3d[t, j]) if np.isfinite(kp3d[t, j]).all()
                     else np.full((C, 2), np.nan)
                     for t in range(T) for j in range(K)]).reshape(T, K, C, 2)
    kp2d = np.moveaxis(kp2d, 2, 1)                    # (T,C,K,2)
    kp2d = kp2d + rng.normal(0, 1.5, kp2d.shape)
    kp2d[0, 1, 2] = np.nan                            # a NaN observation
    vis = rng.random((T, C, K)) > 0.25
    vis[1] = False                                    # frame with nothing visible
    if not nan_obs_visible:
        # Real data never has a NaN 2-D observation at conf > thresh, and
        # loo_reproj (the SVD) raises LinAlgError on one -- both before and
        # after this vectorization (see
        # test_loo_reproj_still_raises_on_nan_observation).
        vis &= np.isfinite(kp2d).all(-1)
    masks = rng.random((T, C, H, W)) > 0.7
    masks[3, 2] = False                               # empty mask
    kp2d_by_frame = [{c: kp2d[t, c] for c in range(C)} for t in range(T)]
    vis_by_frame = [{c: vis[t, c] for c in range(C)} for t in range(T)]
    masks_by_frame = [{c: (None if (t == 4 and c == 1) else masks[t, c])
                       for c in range(C)} for t in range(T)]
    return dict(rt=rt, brt=BatchRT(CAL),
                kp3d_by_frame=[kp3d[t] for t in range(T)],
                mesh_by_frame=[mesh[t] for t in range(T)],
                kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                masks_by_frame=masks_by_frame)


@pytest.fixture(scope="module")
def bout():
    return make_bout()


@pytest.fixture(scope="module")
def bout_clean():
    """Same bout, minus the one NaN-but-visible 2-D observation: everything
    that triangulates (loo_reproj, and qc_report through it) needs a solvable
    DLT system, exactly as real data provides."""
    return make_bout(nan_obs_visible=False)


def test_per_camera_reproj_error_exact(bout):
    for t in range(T):
        ref = ref_per_camera_reproj_error(bout["rt"], bout["kp3d_by_frame"][t],
                                          bout["kp2d_by_frame"][t], bout["vis_by_frame"][t])
        for rt_ in (bout["rt"], bout["brt"]):        # scalar fallback AND batched
            got = qcmod.per_camera_reproj_error(rt_, bout["kp3d_by_frame"][t],
                                                bout["kp2d_by_frame"][t],
                                                bout["vis_by_frame"][t])
            assert got.keys() == ref.keys(), f"frame {t}: camera set changed"
            for c in ref:
                assert got[c] == ref[c], f"frame {t} cam {c}: {got[c]} != {ref[c]}"


def test_ik_reproj_samples_exact(bout):
    for t in range(T):
        i_r, e_r = ref_ik_reproj_samples(bout["rt"], bout["kp3d_by_frame"][t],
                                         bout["kp2d_by_frame"][t], bout["vis_by_frame"][t])
        i_g, e_g = qcmod._ik_reproj_samples(bout["brt"], bout["kp3d_by_frame"][t],
                                            bout["kp2d_by_frame"][t], bout["vis_by_frame"][t])
        assert (i_g == i_r).all() and (e_g == e_r).all(), f"frame {t}"


def test_loo_reproj_exact(bout_clean):
    bout = bout_clean
    for t in range(T):
        ref = ref_loo_reproj(bout["rt"], bout["kp2d_by_frame"][t], bout["vis_by_frame"][t])
        for rt_ in (bout["rt"], bout["brt"]):
            got = qcmod.loo_reproj(rt_, bout["kp2d_by_frame"][t], bout["vis_by_frame"][t])
            assert got["n"] == ref["n"], f"frame {t}: sample count changed"
            assert (np.isnan(got["median"]) and np.isnan(ref["median"])) or \
                   got["median"] == ref["median"], f"frame {t} median"
            np.testing.assert_array_equal(got["per_kp"], ref["per_kp"])


def test_hard_iou_exact_and_soft_iou_within_ulps(bout):
    for t in range(T):
        ref = ref_mesh_mask_iou_report(bout["rt"], bout["mesh_by_frame"][t],
                                       bout["masks_by_frame"][t])
        got = qcmod.mesh_mask_iou_report(bout["brt"], bout["mesh_by_frame"][t],
                                         bout["masks_by_frame"][t])
        assert got["hard"].keys() == ref["hard"].keys(), f"frame {t}"
        for c in ref["hard"]:
            assert got["hard"][c] == ref["hard"][c], (
                f"frame {t} cam {c} HARD IoU {got['hard'][c]} != {ref['hard'][c]}")
            assert got["soft"][c] == pytest.approx(ref["soft"][c], rel=1e-12, abs=1e-15), (
                f"frame {t} cam {c} SOFT IoU {got['soft'][c]} != {ref['soft'][c]}")


def test_mesh_iou_public_helpers_match_dense_reference():
    rng = np.random.default_rng(7)
    mask = rng.random((40, 70)) > 0.6
    for verts in (rng.uniform(-5, 75, (30, 2)),              # partly off-frame
                  rng.uniform(10, 30, (30, 2)),              # fully inside
                  rng.uniform(200, 300, (30, 2)),            # fully outside
                  np.full((5, 2), np.nan)):
        h_ref = ref_iou_of_projected_verts(verts, mask.shape, mask)
        s_ref = ref_soft_iou_of_verts(verts, mask)
        assert miou.iou_of_projected_verts(verts, mask.shape, mask) == h_ref
        assert miou.soft_iou_of_verts(verts, mask) == pytest.approx(
            s_ref, rel=1e-12, abs=1e-15)
        h, s = miou.hard_soft_iou_of_verts(verts, mask)
        assert h == h_ref and s == pytest.approx(s_ref, rel=1e-12, abs=1e-15)
    # empty mask short-circuits to 0.0, as before
    assert miou.soft_iou_of_verts(np.zeros((3, 2)), np.zeros((5, 5), bool)) == 0.0


def test_mesh_iou_float_mask_takes_its_own_sum():
    """hard_soft_iou_of_verts shares ONE |mask| reduction between the two
    metrics, which is only provably exact for a bool mask -- a non-bool one
    must still match the reference (it gets its own sum)."""
    rng = np.random.default_rng(21)
    mask = (rng.random((30, 50)) > 0.5).astype(np.float32)
    verts = rng.uniform(0, 50, (25, 2))
    h_ref = ref_iou_of_projected_verts(verts, mask.shape, mask)
    s_ref = ref_soft_iou_of_verts(verts, mask)
    h, s = miou.hard_soft_iou_of_verts(verts, mask)
    assert h == h_ref
    assert s == pytest.approx(s_ref, rel=1e-12, abs=1e-15)


def test_per_frame_qc_matches_reference(bout):
    ref = ref_per_frame_qc(bout["rt"], mesh_by_frame=bout["mesh_by_frame"],
                           kp3d_by_frame=bout["kp3d_by_frame"],
                           kp2d_by_frame=bout["kp2d_by_frame"],
                           vis_by_frame=bout["vis_by_frame"],
                           masks_by_frame=bout["masks_by_frame"])
    from jarvis_jax.tracking.qc_perframe import per_frame_qc
    got = per_frame_qc(bout["brt"], mesh_by_frame=bout["mesh_by_frame"],
                       kp3d_by_frame=bout["kp3d_by_frame"],
                       kp2d_by_frame=bout["kp2d_by_frame"],
                       vis_by_frame=bout["vis_by_frame"],
                       masks_by_frame=bout["masks_by_frame"])
    assert got.keys() == ref.keys()
    np.testing.assert_array_equal(got["n_cams"], ref["n_cams"])
    np.testing.assert_array_equal(got["reproj_px"], ref["reproj_px"])
    np.testing.assert_array_equal(got["hard_iou"], ref["hard_iou"])
    np.testing.assert_allclose(got["soft_iou"], ref["soft_iou"],
                               rtol=1e-12, atol=1e-15)


def test_shared_frame_metrics_changes_nothing(bout_clean):
    """Passing a precomputed frame_metrics must give the SAME artifacts as
    letting each function compute its own (that sharing is the 2x in Stage E)."""
    from jarvis_jax.tracking.qc_perframe import per_frame_qc
    bout = bout_clean
    kw = dict(kp3d_by_frame=bout["kp3d_by_frame"], mesh_by_frame=bout["mesh_by_frame"],
              kp2d_by_frame=bout["kp2d_by_frame"], vis_by_frame=bout["vis_by_frame"],
              masks_by_frame=bout["masks_by_frame"])
    fm = qcmod.frame_metrics(bout["brt"], **kw)
    a = qcmod.qc_report(bout["brt"], **kw)
    b = qcmod.qc_report(bout["brt"], frame_metrics=fm, **kw)
    assert a == b
    pa = per_frame_qc(bout["brt"], **kw)
    pb = per_frame_qc(bout["brt"], frame_metrics=fm, **kw)
    for k in pa:
        np.testing.assert_array_equal(pa[k], pb[k])


def test_qc_report_aggregates_match_reference(bout_clean):
    """End-to-end qc.json values against a report assembled from the reference
    per-frame loops (the aggregation itself is unchanged, so only the medians
    of the vectorized metrics are under test here)."""
    bout = bout_clean
    kw = dict(kp3d_by_frame=bout["kp3d_by_frame"], mesh_by_frame=bout["mesh_by_frame"],
              kp2d_by_frame=bout["kp2d_by_frame"], vis_by_frame=bout["vis_by_frame"],
              masks_by_frame=bout["masks_by_frame"])
    got = qcmod.qc_report(bout["brt"], **kw)

    per_cam_all, loo_all, hard_all, soft_all = [], [], [], []
    for t in range(T):
        per_cam_all.extend(ref_per_camera_reproj_error(
            bout["rt"], bout["kp3d_by_frame"][t], bout["kp2d_by_frame"][t],
            bout["vis_by_frame"][t]).values())
        lo = ref_loo_reproj(bout["rt"], bout["kp2d_by_frame"][t], bout["vis_by_frame"][t])
        if lo["n"]:
            loo_all.append(lo["median"])
        sr = ref_mesh_mask_iou_report(bout["rt"], bout["mesh_by_frame"][t],
                                      bout["masks_by_frame"][t])
        if np.isfinite(sr["hard_mean"]):
            hard_all.append(sr["hard_mean"])
        if np.isfinite(sr["soft_mean"]):
            soft_all.append(sr["soft_mean"])

    def agg(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.median(xs)) if xs else float("nan")

    assert got["n_frames"] == T
    assert got["per_camera_reproj_px"]["n"] == len(per_cam_all)
    assert got["per_camera_reproj_px"]["median"] == agg(per_cam_all)
    assert got["loo_reproj_px"]["n_frames"] == len(loo_all)
    assert got["loo_reproj_px"]["median"] == agg(loo_all)
    assert got["mesh_mask_iou"]["n_frames"] == len(hard_all)
    assert got["mesh_mask_iou"]["hard_median"] == agg(hard_all)
    assert got["mesh_mask_iou"]["soft_median"] == pytest.approx(
        agg(soft_all), rel=1e-12, abs=1e-15)


def test_loo_reproj_still_raises_on_nan_observation(bout):
    """A NaN 2-D at vis=True makes LAPACK fail to converge -- the scalar
    version did too, and this batches the same failure rather than silently
    inventing a number for it."""
    t = 0                                    # kp2d[0, 1, 2] is NaN, vis True
    with pytest.raises(np.linalg.LinAlgError):
        ref_loo_reproj(bout["rt"], bout["kp2d_by_frame"][t], bout["vis_by_frame"][t])
    with pytest.raises(np.linalg.LinAlgError):
        qcmod.loo_reproj(bout["brt"], bout["kp2d_by_frame"][t], bout["vis_by_frame"][t])
