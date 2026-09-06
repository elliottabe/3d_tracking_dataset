"""The v12-format pseudo-label export writer (spec 2026-09-05 §3.2).

Transcribed from the task-2 brief. Two deviations from the brief's literal
snippets, both forced by the fixture as Task 1's review round left it:
  * `CAMS` is 7 cameras now (it was 3 when the brief was written), so the
    `reprojected_labels` case builds its `vis` as `(len(CAMS), K)` instead of
    the brief's hard-coded `(3, 6)`.
  * The loader needs a real calibration directory to build a
    `ReprojectionTool`, so the round-trip case writes one with the fixture's
    `make_calib` rather than pointing `calib_dir` at a path that does not
    exist.
"""
import json

import numpy as np
from pseudo_fixtures import CAMS, KP_NAMES, make_bout, make_calib


def _records(a, r, rec="rec0"):
    from jarvis_jax.data.pseudo_export import PseudoRecord
    out = []
    for t in np.flatnonzero(r.frame):
        out.append(PseudoRecord(recording=rec, frame=1000 + int(t), host_fly=1,
                                kp3d=a.kp3d[:, t], kp2d=a.kp2d[:, t], vis=a.conf[:, t] >= 0.5,
                                sex=np.array([0, 1], np.int8),
                                stratum={"host_sex": "male", "contact": False, "wall": False},
                                gates={"exist": 0.95}, partners={1: 1001 + int(t)}, role="anchor"))
    return out


def test_export_round_trips_through_the_loader(tmp_path):
    from jarvis_jax.data.pseudo_export import write_pseudo_export
    from jarvis_jax.data.pseudo_gates import GateThresholds, admit_bout, load_bout_arrays
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.tracking.lift_mvq import BoutMaskStore
    d, npz, cm = make_bout(tmp_path, T=40)
    a = load_bout_arrays(d, cameras=CAMS)
    r = admit_bout(a, BoutMaskStore(npz, CAMS), cm, GateThresholds())
    out = str(tmp_path / "pseudo")
    frames = lambda rec, f: np.zeros((len(CAMS), 80, 120, 3), np.uint8)   # black frames are fine here
    export_names = list(reversed(KP_NAMES))                                # a DIFFERENT order on purpose
    write_pseudo_export(out, _records(a, r), export_names=export_names, cameras=CAMS,
                        recordings={"rec0": {"calib_dir": make_calib(tmp_path / "calib", cm),
                                             "fly_sex": {"fly0": "female", "fly1": "male"}}},
                        checkpoint="/fake/final", gates=GateThresholds(), frame_reader=frames)
    man = json.load(open(f"{out}/manifest.json"))
    assert man["source"] == "pseudo" and man["weight"] == 0.3
    coco = json.load(open(f"{out}/annotations/instances_train.json"))
    assert coco["keypoint_names"] == export_names
    fs = next(iter(coco["framesets"].values()))
    assert fs["role"] == "anchor" and fs["source"] == "pseudo" and "1" in fs["partners"]
    ds = V12WindowDataset(out, "train", T=1, train=False)
    assert len(ds) == len(coco["framesets"])


def test_val_split_is_written_and_empty(tmp_path):
    """Nothing pseudo-labelled may ever enter validation: the headline number
    is measured on human labels only (spec §7, confirmation bias). The file
    exists so a loader asked for the val split gets an empty set, not a
    FileNotFoundError that a caller might 'fix' by pointing it at train."""
    from jarvis_jax.data.pseudo_export import write_pseudo_export
    from jarvis_jax.data.pseudo_gates import GateThresholds, admit_bout, load_bout_arrays
    from jarvis_jax.tracking.lift_mvq import BoutMaskStore
    d, npz, cm = make_bout(tmp_path, T=40)
    a = load_bout_arrays(d, cameras=CAMS)
    r = admit_bout(a, BoutMaskStore(npz, CAMS), cm, GateThresholds())
    out = str(tmp_path / "pseudo")
    write_pseudo_export(out, _records(a, r), export_names=KP_NAMES, cameras=CAMS,
                        recordings={"rec0": {"calib_dir": make_calib(tmp_path / "calib", cm)}},
                        checkpoint="/fake/final", gates=GateThresholds(),
                        frame_reader=lambda rec, f: np.zeros((len(CAMS), 80, 120, 3), np.uint8))
    val = json.load(open(f"{out}/annotations/instances_val.json"))
    assert val["framesets"] == {} and val["images"] == [] and val["annotations"] == []
    assert val["keypoint_names"] == list(KP_NAMES)


def test_written_jpeg_round_trips_the_rgb_channels(tmp_path):
    """The writer hands cv2 a BGR view; the loader reads back with PIL as RGB.
    A swapped flip would put blue flies in the training set and no downstream
    metric would see it."""
    from PIL import Image
    from jarvis_jax.data.pseudo_export import write_pseudo_export
    from jarvis_jax.data.pseudo_gates import GateThresholds, admit_bout, load_bout_arrays
    from jarvis_jax.tracking.lift_mvq import BoutMaskStore
    d, npz, cm = make_bout(tmp_path, T=40)
    a = load_bout_arrays(d, cameras=CAMS)
    r = admit_bout(a, BoutMaskStore(npz, CAMS), cm, GateThresholds())
    rgb = np.zeros((len(CAMS), 80, 120, 3), np.uint8)
    rgb[:, :, :, 0] = 200                                    # a RED frame
    rgb[:, :, :, 2] = 10
    out = str(tmp_path / "pseudo")
    write_pseudo_export(out, _records(a, r)[:1], export_names=KP_NAMES, cameras=CAMS,
                        recordings={"rec0": {"calib_dir": make_calib(tmp_path / "calib", cm)}},
                        checkpoint="/fake/final", gates=GateThresholds(),
                        frame_reader=lambda rec, f: rgb)
    coco = json.load(open(f"{out}/annotations/instances_train.json"))
    fn = coco["images"][0]["file_name"]
    back = np.asarray(Image.open(f"{out}/images/{fn}").convert("RGB"), np.uint8)
    assert back[..., 0].mean() > 150 and back[..., 2].mean() < 60


def test_keypoints_are_written_in_EXPORT_order_not_model_order(tmp_path):
    """The CLAUDE.md trap: the campaign npz is in model order, the export's
    keypoint axis is `annotations/keypoint_names.json`. A by-index write would
    put Abd_tip's pixels under Antenna_Base and every metric would still look
    fine."""
    from jarvis_jax.data.pseudo_export import to_export_order
    arr = np.arange(len(KP_NAMES))[None, :, None].astype(np.float32)      # (1,K,1)
    export_names = list(reversed(KP_NAMES))
    got = to_export_order(arr, KP_NAMES, export_names)
    assert [KP_NAMES[int(v)] for v in got[0, :, 0]] == export_names


def test_export_order_is_applied_on_the_KEYPOINT_axis_of_every_rank(tmp_path):
    """(C,K,2) pixels, (C,K) visibility and (K,3) world points must all be
    permuted on their OWN keypoint axis. An axis picked by rank alone would
    permute cameras on one of them and still 'look fine'."""
    from jarvis_jax.data.pseudo_export import to_export_order
    K, C = len(KP_NAMES), len(CAMS)
    export_names = list(reversed(KP_NAMES))
    uv = np.tile(np.arange(K, dtype=np.float32)[None, :, None], (C, 1, 2))     # (C,K,2)
    vis = np.tile(np.arange(K)[None, :], (C, 1))                                # (C,K)
    xyz = np.tile(np.arange(K, dtype=np.float32)[:, None], (1, 3))              # (K,3)
    for got in (to_export_order(uv, KP_NAMES, export_names)[0, :, 0],
                to_export_order(vis, KP_NAMES, export_names)[0],
                to_export_order(xyz, KP_NAMES, export_names)[:, 0]):
        assert [KP_NAMES[int(v)] for v in got] == export_names


def test_dlt_of_the_written_2d_reproduces_the_written_3d(tmp_path):
    """The loader triangulates the 2D labels (`V12WindowDataset._dlt`); it never
    reads a 3D field. Writing the REPROJECTION of the gated 3D makes that
    round trip exact -- the gate only guarantees the 2D HEAD is within 3 px."""
    from jarvis_jax.data.pseudo_export import reprojected_labels
    from pseudo_fixtures import cam_mats, project
    cm = cam_mats()
    X = np.random.default_rng(0).normal(size=(6, 3)) * 5
    uv, vis = reprojected_labels(X, cm, np.ones((len(CAMS), 6), bool), (80, 120))
    np.testing.assert_allclose(uv, project(cm, X), atol=1e-6)
    assert vis.shape == (len(CAMS), 6)


def test_a_keypoint_outside_the_frame_is_written_invisible(tmp_path):
    """`vis` is the gate's per-view visibility AND inside-the-image: a
    reprojection off the sensor must not be handed to the loader as a
    labelled pixel (it would be triangulated as if observed)."""
    from jarvis_jax.data.pseudo_export import reprojected_labels
    from pseudo_fixtures import cam_mats
    cm = cam_mats()
    X = np.zeros((3, 3), np.float64)
    X[1] = [1e4, 0.0, 0.0]                                   # far outside an 80x120 frame
    X[2] = [np.nan, np.nan, np.nan]
    uv, vis = reprojected_labels(X, cm, np.ones((len(CAMS), 3), bool), (80, 120))
    assert vis[:, 0].all() and not vis[:, 1].any() and not vis[:, 2].any()
    assert np.isfinite(uv).all()                             # NaN never reaches the JSON
