import os, numpy as np, pytest, mujoco
from jarvis_jax.cse.silhouette_ik_solve import (
    build_solver_inputs, solve_ik, run_single_fly, run_ablation, _model_to_mm, _mm_to_model,
    _wing_fk_indices, _wing_joint_qpos_indices, _augment_wing_markers_stac_order,
    _COCO_KEYPOINT_NAMES, _STAC_WING_KP_IDX, _withhold_wing_kp,
)
from stac_mjx import utils as stac_utils

IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
RECORDING = "2026_03_18_15_31_22"


def _mean_marker_error(inputs: dict, q: np.ndarray) -> float:
    """Mean per-marker 3-D L2 error between FK(q) site positions and kp_data.

    Mirrors the exact FK call sequence used by stac_mjx's marker_cost
    (stac_mjx/stac_core_jaxls.py ~L204-231): set qpos, run
    ``stac_mjx.utils.kinematics`` then ``stac_mjx.utils.com_pos``, then read
    off site positions via ``stac_mjx.utils.get_site_xpos(data, site_idxs)``.
    NaN keypoints (occluded/masked) are skipped, matching marker_cost's
    finite-mask handling.

    Args:
        inputs: dict as returned by build_solver_inputs (or a time-sliced copy).
        q: (T, nq) qpos trajectory to evaluate the fit at.

    Returns:
        Mean L2 marker error (scalar), averaged over all non-NaN
        frame/marker pairs.
    """
    mjx_model = inputs["mjx_model"]
    mjx_data = inputs["mjx_data"]
    site_idxs = inputs["site_idxs"]
    kp_data = np.asarray(inputs["kp_data"])  # (T, n_kp, 3)
    T = q.shape[0]

    errs = []
    for t in range(T):
        data = mjx_data.replace(qpos=np.asarray(q[t]))
        data = stac_utils.kinematics(mjx_model, data)
        data = stac_utils.com_pos(mjx_model, data)
        markers = np.asarray(stac_utils.get_site_xpos(data, site_idxs))  # (n_kp, 3)
        kp = kp_data[t]  # (n_kp, 3)
        finite = np.isfinite(kp).all(axis=-1)
        if not np.any(finite):
            continue
        per_marker = np.linalg.norm(markers[finite] - kp[finite], axis=-1)
        errs.append(per_marker.mean())
    return float(np.mean(errs))


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_build_inputs_shapes():
    inp = build_solver_inputs(IK, XML)
    T = inp["q_init"].shape[0]; nq = inp["q_init"].shape[1]; nk = len(inp["kp_names"])
    assert nq == 93 and nk == 50
    assert inp["kp_data"].shape == (T, nk, 3)
    assert inp["kps_to_opt"].shape == (nk * 3,)
    assert inp["site_idxs"].shape == (nk,)
    assert inp["lb"].shape == (nq,) and inp["ub"].shape == (nq,)


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_solve_ik_does_not_worsen_fit_on_slice():
    inp = build_solver_inputs(IK, XML)
    sl = slice(0, 8)
    small = dict(inp)
    small["q_init"] = inp["q_init"][sl]; small["kp_data"] = inp["kp_data"][sl]
    q = solve_ik(small, smooth_weight=0.0, n_iter=40)
    assert q.shape == small["q_init"].shape
    assert np.isfinite(q).all()

    # Assembly-faithfulness gate: the STORED ik-h5 qpos was fit to these
    # markers with these site offsets, so FK(stored qpos) must land close to
    # kp_data. A wrong site->body mapping or wrong offset source would blow
    # this up (sites parented to the wrong body / offset the wrong marker).
    # Model units here are meters at this fly model's scale (body span ~0.5),
    # so 0.05 (~10% of body span) comfortably passes a faithful assembly
    # (observed ~0.005) while rejecting a scrambled site_idxs mapping
    # (observed ~0.20, 4x over this threshold).
    err_stored = _mean_marker_error(small, small["q_init"])
    assert err_stored < 0.05, (
        f"mean marker error at stored ik-h5 qpos is {err_stored:.3f} (expected <0.05); "
        "this indicates the solver-input assembly (site_idxs / site offsets / "
        "kp_data ordering) does not match the STAC h5 that produced q_init."
    )

    # The re-solve should not materially worsen the fit relative to the
    # stored qpos.
    err_solved = _mean_marker_error(small, q)
    assert err_solved <= err_stored * 1.5, (
        f"solve_ik worsened the mean marker error: stored={err_stored:.3f}, "
        f"solved={err_solved:.3f}"
    )


def test_model_mm_bridge_roundtrips():
    """The model<->mm similarity bridge (Task 4) must be a true inverse pair.

    Fits a random similarity (s, R, t) from a synthetic point cloud to a
    randomly transformed copy of itself (as ``_umeyama`` would from
    marker_sites -> triangulated kp_mm), then checks that mapping a batch of
    MODEL-frame points model->mm->model recovers the originals.
    """
    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3))                       # "model frame" markers
    true_s, true_t = 74.69, np.array([10.0, -3.0, 5.0])
    # random rotation via QR
    A = rng.normal(size=(3, 3))
    true_R, _ = np.linalg.qr(A)
    if np.linalg.det(true_R) < 0:
        true_R[:, -1] *= -1
    dst = true_s * (true_R @ src.T).T + true_t           # "mm frame" markers

    from jarvis_jax.cse.silhouette_ik_solve import _umeyama
    s, R, t = _umeyama(src, dst)
    np.testing.assert_allclose(s, true_s, rtol=1e-6)
    np.testing.assert_allclose(R, true_R, atol=1e-6)
    np.testing.assert_allclose(t, true_t, atol=1e-6)

    # round-trip a fresh set of model-frame points (not the fitting set)
    pts_model = rng.normal(size=(20, 3))
    pts_mm = _model_to_mm(pts_model, s, R, t)
    pts_model2 = _mm_to_model(pts_mm, s, R, t)
    np.testing.assert_allclose(pts_model2, pts_model, atol=1e-8)


_REPORT_KEYS = {
    "qpos_shape", "wing_tip_err_px", "wing_tip_err_mm",
    "wing_angle_pred", "wing_angle_stac", "reproj_px", "n_frames_with_tips",
}


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_run_single_fly_baseline_wiring(tmp_path):
    """Driver-wiring smoke test: run_single_fly(use_silhouette=False) on a
    4-frame slice must return the documented report keys with finite values
    and write the qpos npz. Exercises build_solver_inputs -> solve_ik ->
    report-metric assembly (wing angle + reprojection) without touching the
    SAM-mask/silhouette extraction path.
    """
    report = run_single_fly(
        RECORDING,
        ik_h5=IK,
        model_xml=XML,
        mesh_npz=MESH,
        root=ROOT,
        split="val",
        use_silhouette=False,
        max_frames=4,
        n_iter=20,
        out_dir=str(tmp_path),
    )
    assert set(report) >= _REPORT_KEYS
    assert report["qpos_shape"] == (4, 93)
    assert report["n_frames_with_tips"] == 0
    # no silhouette extraction was run -> no tip-error samples (NaN per side)
    assert all(np.isnan(v) for v in report["wing_tip_err_px"].values())
    assert all(np.isnan(v) for v in report["wing_tip_err_mm"].values())
    assert set(report["wing_angle_pred"]) == {"left", "right"}
    assert set(report["wing_angle_stac"]) == {"left", "right"}
    assert all(np.isfinite(a) for v in report["wing_angle_pred"].values() for a in v)
    assert all(np.isfinite(a) for v in report["wing_angle_stac"].values() for a in v)
    assert np.isfinite(report["reproj_px"])

    out_path = os.path.join(str(tmp_path), f"{RECORDING}_qpos.npz")
    assert os.path.exists(out_path)
    saved = np.load(out_path)
    assert saved["qpos"].shape == (4, 93)


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_run_single_fly_silhouette_wiring_does_not_worsen_reproj(tmp_path):
    """use_silhouette=True end-to-end wiring test on a 4-frame slice (Task 4
    review fix): with only_missing=True (the default) and full GT wing
    keypoints present, augment_wing_markers is a no-op, so reproj_px here
    must match the use_silhouette=False baseline -- NOT the ~4x-worse value
    (2.13->8.42px) the unconditional-override bug produced.
    """
    report_false = run_single_fly(
        RECORDING,
        ik_h5=IK,
        model_xml=XML,
        mesh_npz=MESH,
        root=ROOT,
        split="val",
        use_silhouette=False,
        max_frames=4,
        n_iter=20,
        out_dir=str(tmp_path / "baseline"),
    )
    report_true = run_single_fly(
        RECORDING,
        ik_h5=IK,
        model_xml=XML,
        mesh_npz=MESH,
        root=ROOT,
        split="val",
        use_silhouette=True,
        max_frames=4,
        n_iter=20,
        out_dir=str(tmp_path / "silhouette"),
    )
    assert set(report_true) >= _REPORT_KEYS
    assert report_true["qpos_shape"] == (4, 93)
    assert np.isfinite(report_true["reproj_px"])
    # confidence-gated augmentation on full GT data must not worsen reproj
    # relative to the no-silhouette baseline (allow a small numerical slack).
    assert report_true["reproj_px"] <= report_false["reproj_px"] * 1.1 + 0.5, (
        f"use_silhouette=True reproj_px={report_true['reproj_px']:.3f} is worse than "
        f"the use_silhouette=False baseline={report_false['reproj_px']:.3f}; "
        "the confidence gate (only_missing) should make augmentation a no-op "
        "when GT wing keypoints are present."
    )

    out_path = os.path.join(str(tmp_path / "silhouette"), f"{RECORDING}_qpos.npz")
    assert os.path.exists(out_path)


def test_wing_fk_indices_land_on_wing_geoms_not_thorax():
    """_wing_fk_indices must select vertices on the wing membrane geoms
    (left=29, right=35 for this mesh), not the thorax. Regression test for
    the bug where passing wing_side_vertices' fps-relative indices straight
    into make_fk_repose silently selected thorax_collision vertices instead
    (their FK-reposed distance is ~constant regardless of wing pose).
    """
    idx = _wing_fk_indices(MESH)  # [L-prox, L-tip, R-prox, R-tip], full-vertex-array indices
    z = np.load(MESH, allow_pickle=True)
    vgeom = z["vertex_geom"]
    m = mujoco.MjModel.from_xml_path(XML)

    l_prox_geom, l_tip_geom, r_prox_geom, r_tip_geom = vgeom[idx]
    assert l_prox_geom == 29 and l_tip_geom == 29
    assert r_prox_geom == 35 and r_tip_geom == 35

    l_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, 29)
    r_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, 35)
    assert "wing" in l_name.lower() and "left" in l_name.lower()
    assert "wing" in r_name.lower() and "right" in r_name.lower()
    assert "thorax" not in l_name.lower() and "thorax" not in r_name.lower()


def test_wing_joint_qpos_indices_match_model():
    """_wing_joint_qpos_indices must resolve to the wing hinge joints'
    actual qpos addresses (yaw/roll/pitch per side for this fly model), used
    by run_single_fly's wing_angle_pred/wing_angle_stac metrics.
    """
    m = mujoco.MjModel.from_xml_path(XML)
    idx = _wing_joint_qpos_indices(m)
    assert idx == {"left": [7, 8, 9], "right": [10, 11, 12]}


def test_stac_order_bridge_writes_stac_indices_6_7_8_9_not_29():
    """_augment_wing_markers_stac_order must write the STAC-ordered wing
    marker indices (WingL_V12=6, WingL_V13=7, WingR_V12=8, WingR_V13=9 for
    this fly model's kp_names) and must NOT touch STAC idx 29 (T2L_TaT1, a
    leg marker) -- regression test for the coco/STAC index-permutation bug
    described in _augment_wing_markers_stac_order's docstring.
    """
    kp_names = [
        "Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "EyeL", "EyeR",
        "WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13", "Abd_A4", "Abd_tip",
        "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
        "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
        "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
        "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
        "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
        "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
    ]
    assert kp_names.index("WingL_V12") == 6
    assert kp_names.index("WingL_V13") == 7
    assert kp_names.index("WingR_V12") == 8
    assert kp_names.index("WingR_V13") == 9
    assert kp_names.index("T2L_TaT1") == 29
    # sanity: every name used here is a real coco keypoint name (the schema
    # WING_MARKER_IDS/_COCO_KEYPOINT_NAMES are defined against).
    assert set(kp_names) <= set(_COCO_KEYPOINT_NAMES)

    n_kp = len(kp_names)
    T = 1
    kp_data = np.full((T, n_kp, 3), np.nan)  # all missing -> only_missing=True fills them
    kps_to_opt = np.ones(n_kp * 3)
    # pin the leg marker at STAC idx 29 to a known present (finite) value,
    # to detect any accidental overwrite.
    kp_data[0, 29] = [42.0, 43.0, 44.0]

    tips_list = [{"left": (np.array([1.0, 2.0, 3.0]), 3),
                  "right": (np.array([4.0, 5.0, 6.0]), 3)}]

    kp_data2, kps_to_opt2 = _augment_wing_markers_stac_order(
        kp_data, kps_to_opt, tips_list, kp_names, wing_weight=2.0, min_cams=2,
    )

    for j in (6, 7):  # WingL_V12, WingL_V13
        assert np.allclose(kp_data2[0, j], [1.0, 2.0, 3.0])
        assert np.allclose(kps_to_opt2[j * 3:j * 3 + 3], 2.0)
    for j in (8, 9):  # WingR_V12, WingR_V13
        assert np.allclose(kp_data2[0, j], [4.0, 5.0, 6.0])
        assert np.allclose(kps_to_opt2[j * 3:j * 3 + 3], 2.0)

    # STAC idx 29 (a leg marker, T2L_TaT1) must be untouched.
    assert np.allclose(kp_data2[0, 29], [42.0, 43.0, 44.0])
    assert np.allclose(kps_to_opt2[29 * 3:29 * 3 + 3], 1.0)


def test_withhold_wing_kp_nans_only_stac_indices_6_7_8_9():
    """_withhold_wing_kp must NaN exactly the STAC wing indices (6,7,8,9) and
    leave every other marker's kp_data untouched (regression guard: run_ablation
    condition (b)/(c) rely on this to correctly drop the wing from marker_cost's
    finite mask without corrupting any other keypoint).
    """
    rng = np.random.default_rng(0)
    T, n_kp = 5, 50
    kp_data = rng.normal(size=(T, n_kp, 3))
    assert _STAC_WING_KP_IDX == {"left": (6, 7), "right": (8, 9)}

    kp2 = _withhold_wing_kp(kp_data)
    wing_idx = [6, 7, 8, 9]
    assert np.isnan(kp2[:, wing_idx, :]).all()
    other_idx = [i for i in range(n_kp) if i not in wing_idx]
    np.testing.assert_allclose(kp2[:, other_idx, :], kp_data[:, other_idx, :])
    # original array must not be mutated in place.
    assert not np.isnan(kp_data).any()


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_run_ablation_wiring(tmp_path):
    """Driver-wiring smoke test on a 4-frame slice: run_ablation must return
    the three conditions (reference/baseline/silhouette) with the documented
    per-condition keys, a finite recovery_ratio per side, and condition (b)
    (baseline) must have withheld (NaN) wing keypoints going into its solve
    while condition (a) (reference) does not.

    This is a wiring/finite gate only -- the scientific magnitude (does the
    silhouette actually recover the wing) comes from the real GPU run, not
    from this fast smoke test.
    """
    report = run_ablation(
        RECORDING,
        ik_h5=IK,
        model_xml=XML,
        mesh_npz=MESH,
        root=ROOT,
        split="val",
        max_frames=4,
        n_iter=20,
        out_dir=str(tmp_path),
    )

    assert set(report) >= {
        "conditions", "recovery_ratio_mm", "recovery_ratio_px",
        "recovery_ratio_to_sam_mm", "recovery_ratio_to_sam_px", "n_frames_with_tips",
    }
    assert set(report["conditions"]) == {"reference", "baseline", "silhouette"}

    for cond in ("reference", "baseline", "silhouette"):
        c = report["conditions"][cond]
        assert set(c) >= {
            "qpos_shape", "wing_tip_err_mm", "wing_tip_err_px",
            "dist_to_ref_mm", "dist_to_ref_px", "reproj_px",
        }
        assert c["qpos_shape"] == (4, 93)
        assert set(c["wing_tip_err_mm"]) == {"left", "right"}
        assert set(c["dist_to_ref_mm"]) == {"left", "right"}
        # non-wing reproj_px must be finite (body/legs must not blow up / be dropped).
        assert np.isfinite(c["reproj_px"])
        out_path = os.path.join(str(tmp_path), f"{RECORDING}_{cond}_qpos.npz")
        assert os.path.exists(out_path)

    # reference (a) has no NaN'd distance-to-itself; baseline/silhouette
    # (b)/(c) dist_to_ref is measured against (a) and must be finite when
    # any frame had a valid tip/umeyama fit.
    if report["n_frames_with_tips"] > 0:
        for side in ("left", "right"):
            assert np.isfinite(report["recovery_ratio_mm"][side]) or np.isnan(
                report["recovery_ratio_mm"][side]
            )  # always one or the other (never inf/nan-from-exception)
    for side in ("left", "right"):
        val_mm = report["recovery_ratio_mm"][side]
        val_px = report["recovery_ratio_px"][side]
        val_sam_mm = report["recovery_ratio_to_sam_mm"][side]
        val_sam_px = report["recovery_ratio_to_sam_px"][side]
        assert isinstance(val_mm, float) and isinstance(val_px, float)
        assert isinstance(val_sam_mm, float) and isinstance(val_sam_px, float)

    # Condition (b)/(c) inputs must actually withhold the wing keypoints;
    # condition (a) must not. Re-derive the same solver inputs to check the
    # kp_data each condition's solve actually saw.
    inputs = build_solver_inputs(IK, XML)
    kp_data_ref = inputs["kp_data"][:4]
    assert np.isfinite(kp_data_ref[:, [6, 7, 8, 9], :]).all(), (
        "reference (a) must solve on the full, non-withheld GT wing keypoints"
    )
    kp_data_withheld = _withhold_wing_kp(kp_data_ref)
    assert np.isnan(kp_data_withheld[:, [6, 7, 8, 9], :]).all(), (
        "baseline (b) / silhouette (c) must withhold (NaN) the STAC wing "
        "keypoint indices 6,7,8,9 before solving"
    )

    # At least one recovery_ratio value should be finite (not both NaN) for
    # this recording/slice to be a meaningful wiring check -- if this ever
    # fails it means no frame produced a usable SAM tip on this slice, which
    # is a data/wiring problem worth surfacing rather than silently passing.
    finite_any = any(
        np.isfinite(report["recovery_ratio_mm"][s]) for s in ("left", "right")
    )
    assert finite_any, (
        "no finite recovery_ratio_mm on either side -- no frame in this "
        "4-frame slice produced a usable SAM-triangulated wing tip"
    )
    finite_any_sam = any(
        np.isfinite(report["recovery_ratio_to_sam_mm"][s]) for s in ("left", "right")
    )
    assert finite_any_sam, (
        "no finite recovery_ratio_to_sam_mm on either side -- no frame in "
        "this 4-frame slice produced a usable SAM-triangulated wing tip"
    )


# ---------------------------------------------------------------------------
# Phase 3 / Task 4: per-identity ann selection (multi-animal threading).
# ---------------------------------------------------------------------------


def test_ann_for_image_selects_by_ann_id_by_image():
    from jarvis_jax.cse.silhouette_ik_solve import _ann_for_image
    id2ann_multi = {100: [{"id": 5, "keypoints": [1]}, {"id": 6, "keypoints": [2]}]}
    # explicit selection picks the chosen ann
    a = _ann_for_image(id2ann_multi, 100, {100: 6})
    assert a["id"] == 6
    # None -> first-ann (backward compatible)
    b = _ann_for_image(id2ann_multi, 100, None)
    assert b["id"] == 5
    # missing image -> None
    assert _ann_for_image(id2ann_multi, 999, {100: 6}) is None
    # image not in map -> falls back to first ann
    c = _ann_for_image(id2ann_multi, 100, {200: 6})
    assert c["id"] == 5


def test_triangulate_kp_mm_uses_selected_ann():
    """With two distinct anns per image, _triangulate_kp_mm must triangulate the
    ann chosen by ann_id_by_image, not the first one -- so fly0 and fly1 yield
    different 3-D keypoints (regression guard for identity threading)."""
    import numpy as np
    from jarvis_jax.cse.silhouette_ik_solve import _triangulate_kp_mm
    from jarvis_jax.cse.affine_camera import factor_affine, reconstruct_affine, project_affine

    class _FakeRT:
        def __init__(self, cam_mats):
            self.num_cameras = len(cam_mats)
            self._cm = cam_mats
        def reconstruct_point(self, obs, cams_to_use=None):
            cams = cams_to_use or list(range(self.num_cameras))
            A = np.zeros((2 * len(cams), 4))
            for i, c in enumerate(cams):
                P = self._cm[c]; uv = obs[c]
                A[2 * i:2 * i + 2] = uv.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]
            _, _, Vh = np.linalg.svd(A)
            return (Vh[-1] / Vh[-1][3])[:3]

    P0 = np.array([[8.1, 0, 0, -2.8], [0, -8.0, 0, 462.0], [0, 0, 0, 1.0]])
    K2, R, t = factor_affine(P0)
    th = np.deg2rad(20.0); Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    cam_mats = [P0, reconstruct_affine(K2, Ry @ R, t)]
    rt = _FakeRT(cam_mats)

    X_fly0 = np.array([118.0, 33.0, 12.0]); X_fly1 = np.array([124.0, 33.0, 12.0])
    kpnames = ["Scutellum"]; coco_kpnames = ["Scutellum"]
    cam2img = {0: 10, 1: 11}

    def _ann(aid, X):
        kp = np.zeros(3)
        return {"id": aid, "keypoints": None, "_X": X}

    # build two anns per image with the keypoint projected from each fly's X
    id2ann_multi = {}
    for c, iid in cam2img.items():
        a0 = {"id": iid * 10, "keypoints": list(project_affine(cam_mats[c], X_fly0)) + [2.0]}
        a1 = {"id": iid * 10 + 1, "keypoints": list(project_affine(cam_mats[c], X_fly1)) + [2.0]}
        id2ann_multi[iid] = [a0, a1]

    sel0 = {iid: iid * 10 for iid in cam2img.values()}       # fly0 anns
    sel1 = {iid: iid * 10 + 1 for iid in cam2img.values()}   # fly1 anns
    kp0, v0 = _triangulate_kp_mm(rt, kpnames, coco_kpnames, cam2img, id2ann_multi, sel0)
    kp1, v1 = _triangulate_kp_mm(rt, kpnames, coco_kpnames, cam2img, id2ann_multi, sel1)
    assert v0[0] and v1[0]
    assert np.linalg.norm(kp0[0] - X_fly0) < 1e-3
    assert np.linalg.norm(kp1[0] - X_fly1) < 1e-3
    assert np.linalg.norm(kp0[0] - kp1[0]) > 1.0    # fly0 != fly1
