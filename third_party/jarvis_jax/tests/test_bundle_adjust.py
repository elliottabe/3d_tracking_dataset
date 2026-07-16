import numpy as np
from jarvis_jax.tracking.affine_camera import reconstruct_affine, factor_affine, project_affine
from jarvis_jax.tracking.bundle_adjust import assemble_observations, initial_points

P_REAL = np.array([
    [8.1001, 0.0074869, -0.031773, -2.828],
    [0.0093308, -8.0788, -0.17912, 462.78],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def _two_cam_rig():
    # camera 0 = P_REAL; camera 1 = P_REAL with a 25-degree yaw applied to R.
    K2, R, t = factor_affine(P_REAL)
    th = np.deg2rad(25.0)
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    return [P_REAL, reconstruct_affine(K2, Ry @ R, t)]


def test_assemble_and_triangulate_recovers_points():
    cams = _two_cam_rig()
    rng = np.random.default_rng(0)
    pts = rng.uniform([-3, -3, 8], [3, 3, 14], size=(5, 3))   # 5 world points
    F, C, K = 1, 2, 5
    kp2d = np.zeros((F, C, K, 3))
    for c, P in enumerate(cams):
        uv = project_affine(P, pts)                            # (5,2)
        kp2d[0, c, :, :2] = uv; kp2d[0, c, :, 2] = 1.0
    obs = assemble_observations(kp2d, min_cams=2)
    assert obs.n_points == 5 and obs.n_cams == 2 and obs.uv.shape[0] == 10
    X0 = initial_points(obs, cams)
    assert np.allclose(X0, pts, atol=1e-6), f"triangulation off:\n{X0}\nvs\n{pts}"


def test_solve_recovers_perturbed_cameras():
    from jarvis_jax.tracking.bundle_adjust import solve_bundle_adjust, mean_reproj_error, initial_points
    true_cams = _two_cam_rig()
    rng = np.random.default_rng(1)
    pts = rng.uniform([-3, -3, 8], [3, 3, 14], size=(40, 3))
    F, C, K = 1, 2, 40
    kp2d = np.zeros((F, C, K, 3))
    for c, P in enumerate(true_cams):
        kp2d[0, c, :, :2] = project_affine(P, pts); kp2d[0, c, :, 2] = 1.0
    obs = assemble_observations(kp2d, min_cams=2)

    # Perturb camera 1's orientation by ~3 degrees -> reprojection error appears.
    K2, R, t = factor_affine(true_cams[1])
    th = np.deg2rad(3.0)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    init_cams = [true_cams[0], reconstruct_affine(K2, Rz @ R, t)]

    err_before = mean_reproj_error(obs, init_cams, initial_points(obs, init_cams))
    refined = solve_bundle_adjust(obs, init_cams, refine="pose", n_iter=120)
    err_after = mean_reproj_error(obs, refined, initial_points(obs, refined))
    assert err_before > 1.0, f"expected a real perturbation, got {err_before}"
    assert err_after < 0.2 * err_before, f"BA did not reduce error: {err_before} -> {err_after}"


def test_refine_reverts_when_no_improvement():
    from jarvis_jax.tracking.bundle_adjust import refine_calibration
    # Perfect data + perfect cameras: BA cannot improve -> must return factory cams.
    cams = _two_cam_rig()
    rng = np.random.default_rng(2)
    pts = rng.uniform([-3, -3, 8], [3, 3, 14], size=(30, 3))
    kp2d = np.zeros((1, 2, 30, 3))
    for c, P in enumerate(cams):
        kp2d[0, c, :, :2] = project_affine(P, pts); kp2d[0, c, :, 2] = 1.0
    obs = assemble_observations(kp2d, min_cams=2)
    refined, report = refine_calibration(obs, cams, refine="pose")
    assert report["err_before"] < 1e-4
    assert not report["improved"]
    for a, b in zip(refined, cams):
        assert np.allclose(a, b), "should revert to factory cameras"


def test_load_kp2d_orders_cameras_like_reprojection_tool(tmp_path):
    import json, cv2
    from jarvis_jax.tracking.run_bundle_adjust import load_kp2d_from_coco
    root = tmp_path; rec = "RECX"
    (root / "annotations").mkdir(parents=True)
    (root / "calib_params" / rec).mkdir(parents=True)
    for nm in ("Cam0002", "Cam0001"):                   # intentionally unsorted on disk
        fs = cv2.FileStorage(str(root / "calib_params" / rec / f"{nm}.yaml"), cv2.FILE_STORAGE_WRITE)
        fs.write("projectionMatrix", P_REAL); fs.release()
    # 2 cameras of the same frame; each annotation carries 2 keypoints (x,y,vis).
    images = [
        {"id": 1, "file_name": f"{rec}/Cam0001/F0.jpg"},
        {"id": 2, "file_name": f"{rec}/Cam0002/F0.jpg"},
    ]
    anns = [
        {"image_id": 1, "keypoints": [10, 20, 2, 30, 40, 2]},
        {"image_id": 2, "keypoints": [11, 21, 2, 31, 41, 2]},
    ]
    coco = {"keypoint_names": ["a", "b"], "images": images, "annotations": anns,
            "framesets": {f"{rec}/F0": {"datasetName": rec, "frames": [1, 2]}}}
    (root / "annotations" / "instances_val.json").write_text(json.dumps(coco))
    kp2d, cam_names = load_kp2d_from_coco(str(root), "val", rec)
    assert cam_names == ["Cam0001", "Cam0002"]          # ReprojectionTool sorted order
    assert kp2d.shape == (1, 2, 2, 3)
    assert list(kp2d[0, 0, 0, :2]) == [10, 20]          # cam0 = Cam0001
    assert list(kp2d[0, 1, 0, :2]) == [11, 21]          # cam1 = Cam0002
