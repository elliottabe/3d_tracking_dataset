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
          "exist_logit": np.array([[10.0]], np.float32), "sex_logit": np.array([[0.0]], np.float32),
          "uv": s["kp2d"][None, 0:1], "vis_logit": np.zeros((1, 1, 1, 7, K), np.float32)}
    _, _, kp2d_full, _ = assemble(out, center3D=s["center3D"][None], crop_origin=s["crop_origin"][None])
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


def test_fly_sex_and_unlabelled_sex_keys(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS
    from jarvis_jax.train.matching import SEX_FEMALE, SEX_MALE
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    assert "fly_sex" in WINDOW_KEYS and "unlabelled_sex" in WINDOW_KEYS
    both = ds.windows.index((REC, 0, 1))           # frame 1 has fly0 + fly1 labelled
    s = ds[both]
    assert s["fly_sex"].dtype == np.int8 and s["fly_sex"].tolist() == [SEX_FEMALE, SEX_MALE]
    assert s["unlabelled_sex"].dtype == np.int8 and int(s["unlabelled_sex"]) == -1
    only_host = ds.windows.index((REC, 0, 0))      # frame 0: fly1 present per manifest, not labelled
    s0 = ds[only_host]
    assert s0["fly_sex"].tolist() == [SEX_FEMALE, -1]
    assert int(s0["unlabelled_sex"]) == SEX_MALE
    assert ds.unlabelled_sex(only_host) == SEX_MALE and ds.unlabelled_sex(both) == -1
    # a fly1-host window in the two-fly frame: fly0 (female) is the OTHER labelled fly
    host1 = ds.windows.index((REC, 1, 1))
    assert ds[host1]["fly_sex"].tolist() == [SEX_MALE, SEX_FEMALE]


def test_sex_resolution_is_per_window_and_annotation_first(tmp_path):
    """Sex is resolved PER WINDOW (not one collapsed value per (recording,
    fly), which was last-frameset-wins) and ANNOTATION-FIRST.

    Real case: `2025_10_20_13_20_04` fly0 has 677 framesets annotated MALE
    (subset `courtship_20_04_male`) and 15 annotated FEMALE
    (`20_04_female_climbing`), while the manifest says fly0 = female for all
    692. The user judged a full-frame render against known-sex reference
    recordings on 2026-09-04: the two subsets really do label DIFFERENT
    animals under one fly id, so the annotators' `sex` is right for both
    blocks and the manifest's single per-fly value cannot represent the
    recording.

    Fixture: fly0 is annotated MALE at frame 2 only, manifest `fly_sex.fly0 =
    female`, `sex: "mixed"`, `n_flies: 2`. (a) the annotation wins per
    window, so frame 2 is male (with a FEMALE unlabelled animal) and frame 0
    female (with a MALE unlabelled animal); (b) with the frame-2 annotation
    `sex` blanked the manifest is used instead."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.train.matching import SEX_FEMALE, SEX_MALE
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1, fly0_sex_by_frame={2: "male"})
    ds = V12WindowDataset(root, "train", T=1, train=False)
    i_ann_male = ds.windows.index((REC, 0, 2))        # fly0 ANNOTATED male here
    i_ann_female = ds.windows.index((REC, 0, 0))
    # (a) the annotation wins over the conflicting manifest, PER WINDOW
    assert ds.manifest[REC]["fly_sex"]["fly0"] == "female"      # manifest disagrees
    assert not ds.is_female(i_ann_male) and ds.is_female(i_ann_female)
    assert ds[i_ann_male]["fly_sex"][0] == SEX_MALE
    assert ds[i_ann_female]["fly_sex"][0] == SEX_FEMALE
    assert ds.window_fly_sex(i_ann_male) == [SEX_MALE]
    assert ds.window_fly_sex(i_ann_female) == [SEX_FEMALE]
    # mixed 2-fly recording, one fly labelled -> the unlabelled animal is the
    # OPPOSITE of THIS window's host, not the manifest's missing fly id
    assert ds.manifest[REC]["sex"] == "mixed" and ds.manifest[REC]["n_flies"] == 2
    assert ds.unlabelled_sex(i_ann_male) == SEX_FEMALE
    assert ds.unlabelled_sex(i_ann_female) == SEX_MALE
    assert int(ds[i_ann_male]["unlabelled_sex"]) == SEX_FEMALE
    assert int(ds[i_ann_female]["unlabelled_sex"]) == SEX_MALE

    # (b) with no annotation sex on frame 2, the manifest's per-fly value is used
    p = os.path.join(root, "annotations", "instances_train.json")
    coco = json.load(open(p))
    fs2 = coco["framesets"][f"{REC}/Frame_2/fly0"]
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    for aid in fs2["ann_ids"]:
        if aid is not None:
            ann_by_id[aid]["sex"] = "unknown"
    json.dump(coco, open(p, "w"))
    ds2 = V12WindowDataset(root, "train", T=1, train=False)
    assert ds2.is_female(i_ann_male) and ds2.is_female(i_ann_female)
    assert ds2[i_ann_male]["fly_sex"][0] == SEX_FEMALE


def test_sex_disagreement_warning_is_printed_once(tmp_path, capsys):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1, fly0_sex_by_frame={2: "male"})
    V12WindowDataset(root, "train", T=1, train=False)
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "disagree on annotation sex" in l]
    assert len(lines) == 1, out
    assert REC in lines[0] and "fly0" in lines[0] and "'female'" in lines[0]
    assert "'female': 2" in lines[0] and "'male': 1" in lines[0]
    # names which value is actually used (the annotation) and which disagrees (the manifest)
    assert "ANNOTATION sex WINS" in lines[0] and "disagrees" in lines[0]
    # a self-consistent root warns about nothing
    (tmp_path / "clean").mkdir()
    clean = make_v12_root(tmp_path / "clean", n_frames=3, two_fly_frame=1)
    V12WindowDataset(clean, "train", T=1, train=False)
    assert "disagree on annotation sex" not in capsys.readouterr().out


def test_fly_centroids_match_sample_labels(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    i = ds.windows.index((REC, 0, 1))
    c = ds.fly_centroids(i)
    s = ds[i]
    assert c.shape == (2, 3)
    for f in range(2):
        has = s["has3d"][f, 0]
        ref = (s["kp3d_local"][f, 0][has]).mean(0) + s["center3D"]
        np.testing.assert_allclose(c[f], ref, atol=1e-3)


def test_fixture_masks_are_written_and_match_host_blob(tmp_path):
    """The synthetic fixture now writes a mask npz per image (masks/<rec>/<cam>/
    Frame_<f>.npz) so `prompt_mask` is non-empty: for the host fly in the
    two-fly frame, the mask must be non-empty in every camera and lie
    entirely within the host's own bright 24x24 blob (mask == blob)."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    i = ds.windows.index((REC, 0, 1))          # host fly0, two-fly frame
    s = ds[i]
    for c in range(7):
        assert s["prompt_mask"][0, c].any()
        assert s["crops"][0, c][s["prompt_mask"][0, c]].min() > 200


def test_copy_paste_hook_is_deterministic_and_skips_unlabelled_present(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1)
    # manifest says 2 flies; frames 0 and 2 have only fly0 labelled -> unlabelled_sex = male -> never pasted
    ds = V12WindowDataset(root, "train", T=1, train=True, copy_paste=CopyPasteParams(p=1.0))
    i = ds.windows.index((REC, 0, 0))
    assert ds[i]["fly_valid"].tolist() == [True, False]
    # rewrite the manifest to n_flies=1 so those windows become fully labelled donors/targets
    import json, os
    man = json.load(open(os.path.join(root, "manifest.json"))); man["recordings"][REC]["n_flies"] = 1
    man["recordings"][REC]["fly_sex"] = {"fly0": "female"}
    json.dump(man, open(os.path.join(root, "manifest.json"), "w"))
    ds = V12WindowDataset(root, "train", T=1, train=True, copy_paste=CopyPasteParams(p=1.0, max_tries=20))
    s1 = ds[i]; s2 = ds[i]
    assert s1["fly_valid"].tolist() == [True, True]
    np.testing.assert_array_equal(s1["crops"], s2["crops"])            # same (seed, i, epoch) -> same paste
    ds.epoch = 1
    s3 = ds[i]
    assert not np.array_equal(s1["kp2d"][1], s3["kp2d"][1])           # a new epoch draws a new paste
    r = ds.paste_window(i, np.random.default_rng(0))
    assert r is not None and set(r[1]) == {"donor", "D", "sep", "contact"}
    val = V12WindowDataset(root, "val", T=1, train=False, copy_paste=CopyPasteParams(p=1.0))
    assert val[i]["fly_valid"].tolist() == [True, False]               # never in eval mode


def test_paste_window_falls_back_to_other_sex_pool(tmp_path):
    """host_sex's own pool may be empty (after excluding i itself) while a donor
    of the OTHER sex is available in the same calib group -- paste_window must
    fall back rather than give up on the first empty draw."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1, manifest_n_flies=1)
    import json, os
    man = json.load(open(os.path.join(root, "manifest.json")))
    man["recordings"][REC]["fly_sex"] = {"fly0": "female"}
    json.dump(man, open(os.path.join(root, "manifest.json"), "w"))
    ds = V12WindowDataset(root, "train", T=1, train=True,
                          copy_paste=CopyPasteParams(p=1.0, opposite_sex_p=0.0, max_tries=4))
    i = ds.windows.index((REC, 0, 0))
    j = ds.windows.index((REC, 1, 1))               # fly1, the two-fly frame: annotation sex "male"
    ds._donors = {("A", 0): [i], ("A", 1): [j]}     # female pool == {i} only; usable donor is male j
    r = ds.paste_window(i, np.random.default_rng(0))
    assert r is not None and r[1]["donor"] == j
    ds._donors = {("A", 0): [i]}                    # no donor of either sex once i excludes itself
    assert ds.paste_window(i, np.random.default_rng(0)) is None


def test_center_shift_moves_window_by_exact_amount(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    ds0 = V12WindowDataset(root, "val", T=1, train=False)
    ds5 = V12WindowDataset(root, "val", T=1, train=False, center_shift_units=5.0)
    i = ds0.windows.index((REC, 0, 1))
    a, b = ds0[i], ds5[i]
    d = b["center3D"] - a["center3D"]
    assert abs(np.linalg.norm(d) - 5.0) < 1e-4 and abs(d[2]) < 1e-6          # exact magnitude, in-plane
    # the labels describe the same world points: local + centre is invariant
    np.testing.assert_allclose(b["kp3d_local"][0, 0] + b["center3D"], a["kp3d_local"][0, 0] + a["center3D"], atol=1e-3)
    assert np.array_equal(ds5[i]["center3D"], b["center3D"])                    # deterministic
    dtr = V12WindowDataset(root, "val", T=1, train=True, center_shift_units=5.0, jitter_units=0.0)
    np.testing.assert_allclose(dtr[i]["center3D"], a["center3D"], atol=1e-6)    # train mode ignores it


def test_sex_label_overrides_beat_the_annotation_and_manifest(tmp_path):
    """P3b `train.sex_label_overrides`: an operator statement that the EXPORT is
    wrong for a whole (recording, fly). It must sit ABOVE both steps of the
    resolution chain -- the fixture's fly0 is `female` in the manifest AND
    annotated `male` in frame 1, and the override must win in both frames --
    and it must reach everything the resolved sex feeds: `is_female`,
    `fly_sex_code`, the sample's `fly_sex`, and the donor index."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1, fly0_sex_by_frame={1: "male"})
    base = V12WindowDataset(root, "train", T=1, train=False)
    i_ann = base.windows.index((REC, 0, 1))       # annotation says male here
    i_man = base.windows.index((REC, 0, 0))       # no annotation disagreement: manifest says female
    assert not base.is_female(i_ann) and base.is_female(i_man)

    ov = V12WindowDataset(root, "train", T=1, train=False,
                          sex_overrides={REC: {"fly0": "female"}})
    assert ov.is_female(i_ann) and ov.is_female(i_man)
    assert ov.fly_sex_code(REC, 0, 1) == 0 and ov.fly_sex_code(REC, 0, 0) == 0
    assert int(ov[i_ann]["fly_sex"][0]) == 0
    # int and "0" keys are accepted too, and the donor index is built from the override
    ov2 = V12WindowDataset(root, "train", T=1, train=True, copy_paste=CopyPasteParams(p=1.0),
                           sex_overrides={REC: {0: "male"}})
    assert not ov2.is_female(i_man)
    assert all(k[1] != 0 for k in ov2._donors), "no window should still be indexed as female"

    # default: no override, nothing changes
    assert V12WindowDataset(root, "train", T=1, train=False).is_female(i_man)


def test_sex_label_overrides_reject_a_typo(tmp_path):
    """A bad sex string must fail at construction, not silently resolve to
    "unknown" (which would suppress that fly's sex AND existence targets)."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    with pytest.raises(ValueError, match="expected 'female' or 'male'"):
        V12WindowDataset(root, "train", T=1, train=False, sex_overrides={REC: {"fly0": "F"}})
