import json, os
import numpy as np
import pytest
from mvq_fixtures import make_v12_root, CAMS, REC, K


def test_window_census_t1_and_t2(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1)
    d1 = V12WindowDataset(root, "train", T=1)
    d2 = V12WindowDataset(root, "train", T=2)
    assert len(d1) == 4          # fly0 x3 frames + fly1 x1
    assert len(d2) == 2          # fly0: (0,1), (1,2); fly1 has no consecutive pair
    assert d1.keypoint_names == json.load(open(os.path.join(root, "annotations", "keypoint_names.json")))


def test_sample_shapes_and_instances(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    i = ds.windows.index((REC, 0, 1))              # host fly0, frame 1 (two-fly frame)
    s = ds[i]
    assert set(s) == set(WINDOW_KEYS)
    assert s["crops"].shape == (1, 7, 448, 448, 3) and s["crops"].dtype == np.uint8
    assert s["cam_valid"].shape == (1, 7) and s["cam_valid"].all()
    assert s["M"].shape == (7, 2, 3) and s["t_local"].shape == (1, 7, 2)
    assert s["kp3d_local"].shape == (2, 1, K, 3) and s["kp2d"].shape == (2, 1, 7, K, 2)
    assert s["fly_valid"].tolist() == [True, True]
    assert ds.n_flies(i) == 2 and ds.is_female(i)
    j = ds.windows.index((REC, 0, 0))
    assert ds[j]["fly_valid"].tolist() == [True, False]


def test_labels_are_consistent_with_geometry(tmp_path):
    """GT 3D (local) reprojected through (M, t_local) must land on the GT 2D
    crop coords -- the invariant the reprojection loss relies on."""
    import jax.numpy as jnp
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.models.mvq.geometry import project_local
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    s = ds[0]
    uv = np.asarray(project_local(jnp.asarray(s["kp3d_local"][0, 0]), jnp.asarray(s["M"]),
                                  jnp.asarray(s["t_local"][0])))            # (K,C,2)
    vis = s["vis2d"][0, 0]                                                    # (C,K)
    gt = s["kp2d"][0, 0]                                                      # (C,K,2)
    err = np.linalg.norm(uv.transpose(1, 0, 2) - gt, axis=-1)[vis]
    assert err.max() < 0.5
    assert s["has3d"][0, 0].all()
    # the host's visible keypoints lie inside the crop
    assert (gt[vis] >= 0).all() and (gt[vis] <= 447).all()
    assert 7.5 < float(s["px_scale"]) < 8.5
    # crop_origin (C,2), full-frame px, per WINDOW (shared by every T frame):
    # kp2d (crop-local) + crop_origin must reconstruct the fixture's own raw
    # (full-frame) 2D labels for every visible keypoint.
    assert s["crop_origin"].shape == (7, 2) and s["crop_origin"].dtype == np.int32
    coco = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    img_by_id = {im["id"]: im for im in coco["images"]}
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    fs0 = coco["framesets"][f"{REC}/Frame_0/fly0"]
    cam_names = ds.camera_names(0)
    full = np.zeros((7, K, 2), np.float32)
    for img_id, ann_id in zip(fs0["frames"], fs0["ann_ids"]):
        if ann_id is None:
            continue
        c = cam_names.index(img_by_id[img_id]["file_name"].split("/")[1])
        full[c] = np.asarray(ann_by_id[ann_id]["keypoints"], np.float32).reshape(-1, 3)[:, :2]
    recon = s["kp2d"][0, 0] + s["crop_origin"][:, None, :]
    np.testing.assert_allclose(recon[vis], full[vis], atol=1e-3)
    # assemble() must reconstruct the SAME full-frame coordinates from uv=kp2d.
    from jarvis_jax.models.mvq.model import assemble
    out = {"xyz": s["kp3d_local"][None, 0:1], "conf_logit": np.zeros((1, 1, 1, K), np.float32),
          "exist_logit": np.array([[10.0]], np.float32), "uv": s["kp2d"][None, 0:1],
          "vis_logit": np.zeros((1, 1, 1, 7, K), np.float32)}
    _, _, kp2d_full = assemble(out, center3D=s["center3D"][None], crop_origin=s["crop_origin"][None])
    np.testing.assert_allclose(kp2d_full[0, 0, 0][vis], full[vis], atol=1e-3)


def test_jitter_only_in_train_mode_and_bounded(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    a = V12WindowDataset(root, "train", T=1, train=False)[0]["center3D"]
    b = V12WindowDataset(root, "train", T=1, train=False)[0]["center3D"]
    c = V12WindowDataset(root, "train", T=1, train=True, jitter_units=3.0, seed=1)[0]["center3D"]
    np.testing.assert_array_equal(a, b)
    assert 0 < np.abs(c - a).max() <= 3.0


def test_none_slot_marks_camera_invalid(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    p = os.path.join(root, "annotations", "instances_train.json")
    coco = json.load(open(p))
    fs = coco["framesets"][f"{REC}/Frame_0/fly0"]
    fs["ann_ids"][2] = None                         # third listed camera unresolved
    json.dump(coco, open(p, "w"))
    ds = V12WindowDataset(root, "train", T=1, train=False)
    s = ds[ds.windows.index((REC, 0, 0))]
    assert s["cam_valid"].sum() == 6
    assert (~s["vis2d"][0, 0][~s["cam_valid"][0]]).all()


def test_t2_window_shares_center_and_has_both_frames(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=2, train=False)
    s = ds[ds.windows.index((REC, 0, 0))]
    assert s["crops"].shape[0] == 2 and s["has3d"][0].all()
    # frame 1 has the second fly, frame 0 does not: fly 1 valid, with vis only in frame 1
    assert s["fly_valid"][1] and not s["vis2d"][1, 0].any() and s["vis2d"][1, 1].any()


def test_batches_stack_and_weights(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset, window_batches
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    w = np.array([1.0, 0.0, 0.0, 0.0])
    b = next(window_batches(ds, 2, shuffle=True, seed=0, weights=w, num_workers=2))
    assert b["crops"].shape == (2, 1, 7, 448, 448, 3)
    np.testing.assert_array_equal(b["center3D"][0], b["center3D"][1])   # only index 0 has weight


def test_jitter_is_per_sample_reproducible_and_epoch_varying(tmp_path):
    """Regression for concurrent-draw corruption: window_batches fetches
    samples from a ThreadPoolExecutor, so a shared numpy Generator is unsafe.
    Jitter must be a pure function of (seed, index, epoch)."""
    from jarvis_jax.data.v12_windows import V12WindowDataset, window_batches
    root = make_v12_root(tmp_path, n_frames=4)      # >=4 T=1 windows (fly0 x4)

    d_a = V12WindowDataset(root, "train", T=1, train=True, jitter_units=3.0, seed=7)
    d_b = V12WindowDataset(root, "train", T=1, train=True, jitter_units=3.0, seed=7)
    np.testing.assert_array_equal(d_a[0]["center3D"], d_b[0]["center3D"])   # (a) same (seed,i,epoch)

    d_a.epoch = 1
    assert not np.array_equal(d_a[0]["center3D"], d_b[0]["center3D"])       # (b) epoch changes it

    ds = V12WindowDataset(root, "train", T=1, train=True, jitter_units=3.0, seed=3)
    assert len(ds) >= 4
    serial = next(window_batches(ds, 4, shuffle=False, seed=3, num_workers=1))["center3D"]
    ds2 = V12WindowDataset(root, "train", T=1, train=True, jitter_units=3.0, seed=3)
    parallel = next(window_batches(ds2, 4, shuffle=False, seed=3, num_workers=4))["center3D"]
    np.testing.assert_array_equal(serial, parallel)                        # (c) thread-count invariant
