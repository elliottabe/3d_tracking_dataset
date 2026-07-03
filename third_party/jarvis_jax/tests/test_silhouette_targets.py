import numpy as np
import jarvis_jax.cse.silhouette_targets as tgt


class _FakeCam:
    def __init__(self):
        self.cameraMatrix = np.array([[2.0, 0, 0, 5.0], [0, 3.0, 0, 7.0], [0, 0, 0, 1.0]])


class _FakeRT:
    def __init__(self, calib_dir):
        self.cameras = {"CamA": 1, "CamB": 2}
        self.num_cameras = 2
        self._camera_list = [_FakeCam(), _FakeCam()]


def _square_mask():
    m = np.zeros((60, 60), bool); m[20:40, 20:40] = True
    return m


def test_build_targets_shapes_and_erosion(monkeypatch):
    monkeypatch.setattr(tgt, "ReprojectionTool", _FakeRT)
    monkeypatch.setattr(tgt, "_load_coco_index",
                        lambda root, split: ({10: "f0", 11: "f1"}, {10: [{"id": 1}], 11: [{"id": 2}]}, None))
    monkeypatch.setattr(tgt, "_cam2img_for_frame", lambda row, id2f, cams: {0: 10, 1: 11})
    monkeypatch.setattr(tgt, "_ann_for_image", lambda m, iid, a: {"id": iid})
    monkeypatch.setattr(tgt, "_load_sam_mask", lambda root, split, fn, aid: _square_mask())

    out = tgt.build_silhouette_targets("r", "val", [None, None], "cd", n_points=32, erode_px=3)
    assert out["boundary"].shape == (2, 2, 32, 2)
    assert out["conf_p"].shape == (2, 2, 32)
    assert out["present"].all()
    assert out["cam_Ms"].shape == (2, 2, 3) and out["cam_ts"].shape == (2, 2)
    # eroded boundary lies strictly inside the raw 20..39 square (halo stripped inward)
    b = out["boundary"][0, 0]
    assert b[:, 0].min() >= 20 and b[:, 0].max() <= 39
    assert b[:, 0].min() > 20    # erosion pulled the boundary inward from the raw edge


def test_missing_camera_is_nan_absent(monkeypatch):
    monkeypatch.setattr(tgt, "ReprojectionTool", _FakeRT)
    monkeypatch.setattr(tgt, "_load_coco_index",
                        lambda root, split: ({10: "f0"}, {10: [{"id": 1}]}, None))
    monkeypatch.setattr(tgt, "_cam2img_for_frame", lambda row, id2f, cams: {0: 10})  # cam1 missing
    monkeypatch.setattr(tgt, "_ann_for_image", lambda m, iid, a: {"id": iid})
    monkeypatch.setattr(tgt, "_load_sam_mask", lambda root, split, fn, aid: _square_mask())

    out = tgt.build_silhouette_targets("r", "val", [None], n_points=16, erode_px=0, calib_dir="cd")
    assert out["present"][0, 0] and not out["present"][0, 1]
    assert np.isnan(out["boundary"][0, 1]).all()
    assert (out["conf_p"][0, 1] == 0).all()
