import os, numpy as np, pytest
from jarvis_jax.cse.silhouette_ik_solve import (
    build_solver_inputs, solve_ik, run_single_fly, _model_to_mm, _mm_to_model,
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


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_run_single_fly_baseline_wiring(tmp_path):
    """Driver-wiring smoke test: run_single_fly(use_silhouette=False) on a
    4-frame slice must return the documented report keys with finite values
    and write the qpos npz. Exercises build_solver_inputs -> solve_ik ->
    report-metric assembly (FK wing length + reprojection) without touching
    the SAM-mask/silhouette extraction path.
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
    assert set(report) >= {
        "qpos_shape", "wing_len_pred", "wing_len_stac", "reproj_px", "n_frames_with_tips",
    }
    assert report["qpos_shape"] == (4, 93)
    assert report["n_frames_with_tips"] == 0
    assert np.isfinite(report["wing_len_pred"])
    assert np.isfinite(report["wing_len_stac"])
    assert np.isfinite(report["reproj_px"])

    out_path = os.path.join(str(tmp_path), f"{RECORDING}_qpos.npz")
    assert os.path.exists(out_path)
    saved = np.load(out_path)
    assert saved["qpos"].shape == (4, 93)
