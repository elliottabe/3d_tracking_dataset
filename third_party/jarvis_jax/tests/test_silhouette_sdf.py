import numpy as np
from jarvis_jax.tracking.silhouette_sdf import _mask_bbox, _mask_to_sdf_crop
import jarvis_jax.tracking.silhouette_sdf as sdfmod


def test_mask_bbox_expands_by_margin():
    mask = np.zeros((100, 200), bool)
    mask[40:60, 80:120] = True          # 20 tall x 40 wide, center (100,50)
    x0, y0, x1, y1 = _mask_bbox(mask, 0.5)
    # width 40 -> +/-20 margin -> x in [60,140]; height 20 -> +/-10 -> y in [30,70]
    assert (x0, x1) == (60.0, 140.0)
    assert (y0, y1) == (30.0, 70.0)


def test_mask_bbox_empty_returns_none():
    assert _mask_bbox(np.zeros((10, 10), bool), 0.4) is None


def test_sdf_crop_signs_and_transform():
    mask = np.zeros((100, 100), bool)
    mask[30:70, 30:70] = True           # 40x40 square
    bbox = (20.0, 20.0, 80.0, 80.0)     # 60x60 crop
    sdf, gs, go = _mask_to_sdf_crop(mask, bbox, (120, 120))
    assert sdf.shape == (120, 120)
    # grid_offset is the integer crop origin; grid_scale maps 60 orig px -> 120 grid px
    assert np.allclose(go, [20.0, 20.0])
    assert np.allclose(gs, [120.0 / 60.0, 120.0 / 60.0])
    # center of the square is deep inside -> negative; a point well outside -> positive
    cx = int((50 - go[0]) * gs[0]); cy = int((50 - go[1]) * gs[1])
    assert sdf[cy, cx] < 0
    corner = int((22 - go[0]) * gs[0])  # near crop corner, outside the square
    assert sdf[corner, corner] > 0
    # magnitude is in ORIGINAL px: center is ~20 orig px from the nearest edge
    assert abs(abs(sdf[cy, cx]) - 20.0) < 3.0


def test_sdf_crop_empty_returns_none():
    assert _mask_to_sdf_crop(np.zeros((50, 50), bool), (0, 0, 50, 50), (32, 32)) is None


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


def test_build_sdf_stack_present_fill_and_shapes(monkeypatch):
    monkeypatch.setattr(sdfmod, "ReprojectionTool", _FakeRT)
    monkeypatch.setattr(sdfmod, "_load_coco_index",
                        lambda root, split: ({10: "f0", 11: "f1"}, {10: [{"id": 1}], 11: [{"id": 2}]}))
    monkeypatch.setattr(sdfmod, "_cam2img_for_frame", lambda row, id2f, cams: {0: 10})  # cam 1 absent
    monkeypatch.setattr(sdfmod, "_ann_for_image", lambda m, iid, a: {"id": iid})
    monkeypatch.setattr(sdfmod, "_load_sam_mask", lambda root, split, fn, aid: _square_mask())
    out = sdfmod.build_sdf_stack("r", "val", [None], "cd", out_hw=(32, 32))
    assert out["sdf"].shape == (1, 2, 32, 32)
    assert out["grid_scale"].shape == (1, 2, 2) and out["grid_offset"].shape == (1, 2, 2)
    assert out["cam_Ms"].shape == (2, 2, 3) and out["cam_ts"].shape == (2, 2)
    assert out["present"][0, 0] and not out["present"][0, 1]
    assert np.allclose(out["sdf"][0, 1], 1e4)        # absent camera -> +1e4 fill
    assert out["sdf"][0, 0].min() < 0                # present camera -> some inside (negative) SDF
