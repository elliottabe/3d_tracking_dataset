import numpy as np
from jarvis_jax.cse.silhouette_targets import build_silhouette_targets


def test_build_targets_shapes_and_nan_for_missing(monkeypatch):
    from jarvis_jax.cse import silhouette_targets as st
    from jarvis_jax.cse.silhouette_joint_ik import SIL_PER_CAM, unpack_sil_value
    import jax.numpy as jnp

    n_cam, n_points, T = 3, 32, 2
    P = [np.array([[8.0, 0, 0, -2.0], [0, -8.0, 0, 460.0], [0, 0, 0, 1.0]]) for _ in range(n_cam)]

    # --- stub the calibration + coco + mask I/O so the test is offline ---
    class _FakeCam:
        def __init__(self, P): self.cameraMatrix = P
    class _FakeRT:
        def __init__(self, calib_dir):
            self.cameras = {f"cam{i}": None for i in range(n_cam)}
            self._camera_list = [_FakeCam(P[i]) for i in range(n_cam)]
            self.num_cameras = n_cam
    monkeypatch.setattr(st, "ReprojectionTool", _FakeRT)

    # frame 0: cam0 and cam1 have a mask, cam2 does not. frame 1: only cam0.
    square = np.zeros((30, 30), dtype=bool); square[8:20, 8:20] = True
    def _fake_cam2img(row, id2file, cam_names): return dict(row)
    def _fake_ann(id2ann_multi, iid, sel): return {"id": iid}
    def _fake_mask(root, split, fn, ann_id):
        return square if ann_id in (0, 1, 10) else None
    def _fake_id2file(*a, **k): return {}
    monkeypatch.setattr(st, "_cam2img_for_frame", _fake_cam2img)
    monkeypatch.setattr(st, "_ann_for_image", _fake_ann)
    monkeypatch.setattr(st, "_load_sam_mask", _fake_mask)
    monkeypatch.setattr(st, "_load_coco_index", lambda root, split: ({}, {}, ["cam0", "cam1", "cam2"]))

    # fs_imgids: per frame, {cam_idx: image_id}. Use the ids the fake mask keys on.
    fs_imgids = [{0: 0, 1: 1, 2: 99}, {0: 10, 1: 98, 2: 97}]

    sil_data, meta = build_silhouette_targets(
        root="X", split="val", fs_imgids=fs_imgids, calib_dir="Y",
        n_points=n_points, seed=0,
    )
    assert meta["n_cam"] == n_cam and meta["n_pts"] == n_points
    assert sil_data.shape == (T, n_cam * SIL_PER_CAM(n_points))

    Ms0, ts0, pts0 = unpack_sil_value(jnp.asarray(sil_data[0]), n_cam, n_points)
    # cam0, cam1 present -> finite boundary; cam2 missing -> NaN boundary.
    assert np.isfinite(np.asarray(pts0[0])).all()
    assert np.isfinite(np.asarray(pts0[1])).all()
    assert np.isnan(np.asarray(pts0[2])).all()
    # camera matrix blocks are always finite (calibration constant).
    assert np.isfinite(np.asarray(Ms0)).all()
    # frame 1: only cam0 present.
    _, _, pts1 = unpack_sil_value(jnp.asarray(sil_data[1]), n_cam, n_points)
    assert np.isfinite(np.asarray(pts1[0])).all()
    assert np.isnan(np.asarray(pts1[1])).all()
    assert np.isnan(np.asarray(pts1[2])).all()
