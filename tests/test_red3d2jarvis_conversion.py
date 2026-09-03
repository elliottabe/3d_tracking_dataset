"""Guards on the raw red3d -> JARVIS conversion rule.

The two things here that silently corrupt everything downstream, and that no
residual/NaN/IoU check would catch, are the VERTICAL FLIP and the KEYPOINT
ORDER. Both are pinned to synthetic fixtures so a future edit to
scripts/data_prep/red3d2jarvis.py cannot quietly drop either.

Synthetic throughout: no dependency on the real label export.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data_prep.red3d2jarvis import (
    MISSING,
    convert,
    jarvis_skeleton,
    load_fly50,
    read_raw_csv,
)

H, W = 448, 1936
SKEL = "/home/tuthill/juan/skeletons/fly50.json"


def _write_raw(path: Path, ndim: int, rows: dict[int, np.ndarray]) -> None:
    with open(path, "w") as f:
        f.write(SKEL + "\n")
        for frame, arr in rows.items():
            parts = [str(frame)]
            for k, v in enumerate(arr):
                parts.append(str(k))
                parts += [repr(float(x)) for x in v]
            f.write(",".join(parts) + "\n")


def _dlt(path: Path) -> None:
    # affine DLT: u = 80*X + 1, v = -80*Y + 400  (row 2 = [0,0,0,1])
    path.write_text("\n".join(["80", "0", "0", "1", "0", "-80", "0", "400",
                               "0", "0", "0"]) + "\n")


def _yaml(path: Path) -> None:
    path.write_text(f"%YAML:1.0\n---\nimage_width: {W}\nimage_height: {H}\n")


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    names, _ = load_fly50()
    rec = tmp_path / "2099_01_01_00_00_00_male"
    (rec / "calibration").mkdir(parents=True)
    rng = np.random.default_rng(0)
    uv = rng.uniform(50, 400, size=(len(names), 2))
    uv[:, 0] += 500
    uv[3] = [1e7, 1e7]                       # Scutellum not labelled
    for cam in ("Cam1", "Cam2"):
        _write_raw(rec / f"{cam}.csv", 2, {7: uv})
        _dlt(rec / "calibration" / f"{cam}_dlt.csv")
        _yaml(rec / "calibration" / f"{cam}.yaml")
    _write_raw(rec / "keypoints3d.csv", 3,
               {7: rng.uniform(0, 3, size=(len(names), 3))})
    return rec


def _run(rec: Path, out: Path):
    convert(rec, out, "sub", "REC", "male", "test", None,
            scale_10x=True, link_from=[], video_dir=None)
    return json.load(open(out / "annotations" / "instances_train.json"))


def test_reader_places_values_at_their_stated_index(tmp_path):
    """A row that lists indices out of order must not shift the keypoint axis."""
    p = tmp_path / "c.csv"
    p.write_text(SKEL + "\n5,2,20.0,21.0,0,0.0,1.0,1,10.0,11.0\n")
    _, rows = read_raw_csv(p, 2)
    assert np.allclose(rows[5], [[0, 1], [10, 11], [20, 21]])


def test_vertical_flip_and_sentinel(recording, tmp_path):
    blob = _run(recording, tmp_path / "out")
    names, _ = load_fly50()
    _, raw = read_raw_csv(recording / "Cam1.csv", 2)
    uv = raw[7]
    ann = next(a for a in blob["annotations"]
               if blob["images"][a["image_id"]]["file_name"].startswith("REC/Cam1"))
    kp = np.array(ann["keypoints"]).reshape(-1, 3)
    vis = (np.abs(uv) < MISSING).all(axis=1)

    # THE FLIP: y is measured from the BOTTOM in the raw CSV.
    assert np.array_equal(kp[vis, 0], np.floor(uv[vis, 0]).astype(int))
    assert np.array_equal(kp[vis, 1], np.floor(H - uv[vis, 1]).astype(int))
    # and it is genuinely a flip, not a no-op, on this fixture
    assert not np.array_equal(kp[vis, 1], np.floor(uv[vis, 1]).astype(int))

    # sentinel -> COCO "not labelled"
    scut = names.index("Scutellum")
    assert not vis[scut]
    assert list(kp[scut]) == [0, 0, 0]
    assert kp[vis, 2].min() == 1


def test_bbox_is_unrounded_flipped_extent_with_no_margin(recording, tmp_path):
    blob = _run(recording, tmp_path / "out")
    _, raw = read_raw_csv(recording / "Cam1.csv", 2)
    uv = raw[7]
    vis = (np.abs(uv) < MISSING).all(axis=1)
    x, y = uv[vis, 0], H - uv[vis, 1]
    ann = next(a for a in blob["annotations"]
               if blob["images"][a["image_id"]]["file_name"].startswith("REC/Cam1"))
    assert ann["bbox"] == pytest.approx(
        [x.min(), y.min(), x.max() - x.min(), y.max() - y.min()])


def test_keypoint_order_is_fly50_and_matches_the_shipped_root(recording, tmp_path):
    blob = _run(recording, tmp_path / "out")
    names, edges = load_fly50()
    assert blob["keypoint_names"] == names
    assert names[:6] == ["Antenna_Base", "EyeL", "EyeR", "Scutellum",
                         "Abd_A4", "Abd_tip"]
    # the two landmarks the 2026-08-31 order bug swapped
    assert names.index("WingR_base") == 28
    assert names.index("T2L_TiTa") == 18
    assert blob["skeleton"] == jarvis_skeleton(names, edges)
    assert len(blob["skeleton"]) == len(edges) == 44


def test_no_train_val_split_is_performed(recording, tmp_path):
    """DEFECT 5: the split belongs to build_generalmodel_split, not here."""
    out = tmp_path / "out"
    _run(recording, out)
    assert (out / "annotations" / "instances_train.json").exists()
    assert not (out / "annotations" / "instances_val.json").exists()


def test_framesets_group_all_cameras_of_one_frame(recording, tmp_path):
    blob = _run(recording, tmp_path / "out")
    assert list(blob["framesets"]) == ["REC/Frame_7"]
    assert len(blob["framesets"]["REC/Frame_7"]["frames"]) == 2


def test_scale_10x_scales_only_the_world_block_of_the_first_two_rows(recording, tmp_path):
    cv2 = pytest.importorskip("cv2")
    out = tmp_path / "out"
    _run(recording, out)
    fs = cv2.FileStorage(str(out / "calib_params" / "REC" / "Cam1.yaml"),
                         cv2.FILE_STORAGE_READ)
    P = fs.getNode("projectionMatrix").mat()
    assert fs.getNode("scale").real() == 10
    fs.release()
    np.testing.assert_allclose(P[0], [8.0, 0, 0, 1.0])
    np.testing.assert_allclose(P[1], [0, -8.0, 0, 400.0])
    np.testing.assert_allclose(P[2], [0, 0, 0, 1.0])


def test_a_foreign_skeleton_is_refused(recording, tmp_path):
    """fly50.json is missing on this machine, so its NAME is the only guard
    that these 50 names describe this data."""
    p = recording / "Cam1.csv"
    p.write_text(p.read_text().replace("fly50.json", "fly31.json"))
    with pytest.raises(SystemExit, match="fly50.json"):
        _run(recording, tmp_path / "out")


def test_duplicate_frame_rows_are_refused(recording, tmp_path):
    p = recording / "Cam1.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[-1]]) + "\n")
    with pytest.raises(ValueError, match="appears twice"):
        _run(recording, tmp_path / "out")
