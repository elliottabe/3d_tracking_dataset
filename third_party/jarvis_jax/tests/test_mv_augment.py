# tests/test_mv_augment.py
import os

import numpy as np
import jax, jax.numpy as jnp
import pytest
from mvq_fixtures import make_v12_root

_ANATOMY_V1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "..", "configs", "anatomy", "v1.yaml")


def _batch(tmp_path, B=2):
    from jarvis_jax.data.v12_windows import V12WindowDataset, window_batches
    ds = V12WindowDataset(make_v12_root(tmp_path), "train", T=1, train=False)
    b = next(window_batches(ds, B, shuffle=False, num_workers=1))
    return {k: jnp.asarray(v) for k, v in b.items()}, ds.keypoint_names


def _reproj_err(b):
    from jarvis_jax.models.mvq.geometry import project_local
    errs = []
    for i in range(b["crops"].shape[0]):
        for t in range(b["crops"].shape[1]):
            uv = project_local(b["kp3d_local"][i, 0, t], b["M"][i], b["t_local"][i, t])  # (K,C,2)
            d = jnp.linalg.norm(jnp.swapaxes(uv, 0, 1) - b["kp2d"][i, 0, t], axis=-1)   # (C,K)
            m = b["vis2d"][i, 0, t] & b["has3d"][i, 0, t][None]
            errs.append(float(jnp.where(m, d, 0.0).max()))
    return max(errs)


def test_identity_when_disabled(tmp_path):
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path)
    out = augment_window(jax.random.PRNGKey(0), b, MVAugParams(enabled=False), build_lr_swap(names))
    for k in b:
        np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(b[k]))


@pytest.mark.parametrize("params", [
    dict(rot_deg=30.0, world_yaw=False, world_tilt_deg=0.0, mirror_p=0.0, cam_drop_p=0.0),
    dict(rot_deg=0.0, scale_min=1.0, scale_max=1.0, translate_frac=0.0, world_yaw=True, world_tilt_deg=30.0, mirror_p=0.0, cam_drop_p=0.0),
    dict(rot_deg=0.0, scale_min=1.0, scale_max=1.0, translate_frac=0.0, world_yaw=False, world_tilt_deg=0.0, mirror_p=1.0, cam_drop_p=0.0),
], ids=["per-view-affine", "world-rotation", "mirror"])
def test_each_geometric_aug_keeps_labels_consistent(tmp_path, params):
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path)
    p = MVAugParams(brightness=0, contrast=0, gamma=0, blur_max=0, noise_scale=0, pc_color=0, **params)
    out = augment_window(jax.random.PRNGKey(3), b, p, build_lr_swap(names))
    assert _reproj_err(out) < 0.05          # 3D->2D consistency survives exactly


def test_mirror_swaps_left_right_and_flips_pixels(tmp_path):
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path)
    swap = build_lr_swap(names)
    p = MVAugParams(rot_deg=0, scale_min=1, scale_max=1, translate_frac=0, world_yaw=False,
                    world_tilt_deg=0, mirror_p=1.0, cam_drop_p=0, brightness=0, contrast=0,
                    gamma=0, blur_max=0, noise_scale=0, pc_color=0)
    out = augment_window(jax.random.PRNGKey(0), b, p, swap)
    np.testing.assert_array_equal(np.asarray(out["crops"]), np.asarray(b["crops"][..., ::-1, :]))
    l = names.index("EyeL"); r = names.index("EyeR")
    np.testing.assert_allclose(np.asarray(out["kp2d"][..., l, 0]), 447.0 - np.asarray(b["kp2d"][..., r, 0]), atol=1e-4)
    np.testing.assert_allclose(np.asarray(out["kp3d_local"][..., l, :]),
                               np.asarray(b["kp3d_local"][..., r, :]) * np.array([-1, 1, 1]), atol=1e-5)


def test_per_view_affine_moves_pixels_with_labels(tmp_path):
    """Paint the EyeL label pixel white in one view; after warping, the
    transformed label must land on a white pixel."""
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path, B=1)
    l = names.index("EyeL")
    crops = np.asarray(b["crops"]).copy(); crops[:] = 0
    u, v = np.round(np.asarray(b["kp2d"][0, 0, 0, :, l])).astype(int).T        # (C,)
    for c in range(7):
        crops[0, 0, c, max(v[c]-2, 0):v[c]+3, max(u[c]-2, 0):u[c]+3] = 255
    b["crops"] = jnp.asarray(crops)
    p = MVAugParams(rot_deg=25, scale_min=0.9, scale_max=1.1, translate_frac=0.05, world_yaw=False,
                    world_tilt_deg=0, mirror_p=0, cam_drop_p=0, brightness=0, contrast=0, gamma=0,
                    blur_max=0, noise_scale=0, pc_color=0)
    out = augment_window(jax.random.PRNGKey(7), b, p, build_lr_swap(names))
    uv = np.asarray(out["kp2d"][0, 0, 0, :, l]); vis = np.asarray(out["vis2d"][0, 0, 0, :, l])
    img = np.asarray(out["crops"][0, 0])
    for c in np.where(vis)[0]:
        x, y = np.round(uv[c]).astype(int)
        assert img[c, y, x].max() > 128, f"cam {c}: label off the painted pixel"


def test_per_view_affine_recomputes_px_scale(tmp_path):
    """px_scale is a per-sample scalar summary of M (geometry.py::px_scale).
    The per-view affine rewrites M (rotation+scale per camera), so a stale
    px_scale carried over from the UN-augmented sample would mis-scale every
    px_scale-weighted loss term (losses_mvq.py) relative to the geometry the
    model actually trained on this step."""
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    from jarvis_jax.models.mvq.geometry import px_scale
    b, names = _batch(tmp_path)
    p = MVAugParams(rot_deg=20, scale_min=1.4, scale_max=1.6, translate_frac=0.05, world_yaw=False,
                    world_tilt_deg=0, mirror_p=0, cam_drop_p=0, brightness=0, contrast=0, gamma=0,
                    blur_max=0, noise_scale=0, pc_color=0)
    out = augment_window(jax.random.PRNGKey(5), b, p, build_lr_swap(names))
    for i in range(b["crops"].shape[0]):
        expected = float(px_scale(out["M"][i]))
        assert abs(float(out["px_scale"][i]) - expected) < 1e-5
        assert abs(float(out["px_scale"][i]) - float(b["px_scale"][i])) > 1e-3   # scale_min/max != 1 -> must differ


def test_all_geometric_augs_composed_keep_labels_consistent(tmp_path):
    """All geometric ops enabled at their MVAugParams DEFAULTS (photometric
    zeroed) at once -- not one at a time, as test_each_geometric_aug_keeps_
    labels_consistent checks -- must still keep GT 3D reprojecting onto GT 2D
    exactly: per-view affine, world rotation, mirror and camera dropout all
    rewrite the SAME M/t_local/kp2d/kp3d_local, in sequence, and a bug in how
    any two compose (not just each alone) would show up here as reprojection
    drift."""
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path)
    p = MVAugParams(brightness=0, contrast=0, gamma=0, blur_max=0, noise_scale=0, pc_color=0)
    out = augment_window(jax.random.PRNGKey(11), b, p, build_lr_swap(names))
    assert _reproj_err(out) < 0.05


def test_camera_dropout_never_below_three(tmp_path):
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path)
    p = MVAugParams(cam_drop_p=1.0, cam_drop_max=6, rot_deg=0, scale_min=1, scale_max=1, translate_frac=0,
                    world_yaw=False, world_tilt_deg=0, mirror_p=0, brightness=0, contrast=0, gamma=0,
                    blur_max=0, noise_scale=0, pc_color=0)
    out = augment_window(jax.random.PRNGKey(1), b, p, build_lr_swap(names))
    cv = np.asarray(out["cam_valid"])
    assert (cv.sum(-1) >= 3).all() and (cv.sum(-1) < 7).any()
    assert not np.asarray(out["vis2d"])[..., ~cv[0, 0], :][0].any()


def test_camera_dropout_keys_off_every_frame_valid(tmp_path):
    """Frame 1 has camera index 2 unresolved (None slot, like
    test_none_slot_marks_camera_invalid). Before this fix, both the "invalid
    sorts last" penalty and the n_valid-3 floor were keyed on frame 0's
    cam_valid alone (`cam_valid[:, 0]`) -- camera 2, invalid ONLY in frame 1,
    would look like a perfectly good (frame-0-valid) drop candidate there,
    risking frame 1 (cam_valid ANDed with the drop mask, same drop mask
    every frame) ending with fewer than 3 valid cameras once its own
    already-invalid camera stacks with the drop. Keying on
    `cam_valid.all(axis=1)` (valid in EVERY frame) instead fixes both: every
    frame keeps >= 3 valid cameras after aggressive dropout, and camera 2
    (invalid in frame 1) is never among the cameras THIS augmentation drops
    in any frame."""
    import json, os
    from mvq_fixtures import REC
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    root = make_v12_root(tmp_path, n_frames=3)
    p = os.path.join(root, "annotations", "instances_train.json")
    coco = json.load(open(p))
    fs = coco["framesets"][f"{REC}/Frame_1/fly0"]
    fs["ann_ids"][2] = None                          # third listed camera unresolved, frame 1 only
    json.dump(coco, open(p, "w"))
    ds = V12WindowDataset(root, "train", T=2, train=False)
    s = ds[ds.windows.index((REC, 0, 0))]
    b = {k: jnp.asarray(v)[None] for k, v in s.items()}
    before = np.asarray(b["cam_valid"][0])                       # (T,C) before augmentation
    assert not before[1, 2] and before[0, 2]        # fixture sanity: only frame 1's cam 2 is invalid
    params = MVAugParams(cam_drop_p=1.0, cam_drop_max=6, rot_deg=0, scale_min=1, scale_max=1,
                         translate_frac=0, world_yaw=False, world_tilt_deg=0, mirror_p=0,
                         brightness=0, contrast=0, gamma=0, blur_max=0, noise_scale=0, pc_color=0)
    out = augment_window(jax.random.PRNGKey(2), b, params, build_lr_swap(ds.keypoint_names))
    cv = np.asarray(out["cam_valid"][0])                          # (T,C) after augmentation
    assert (cv.sum(-1) >= 3).all()
    dropped = before & ~cv                                        # (T,C) True where THIS aug turned a valid cam off
    assert not dropped[:, 2].any()                                # never drops the already-sometimes-invalid camera


def test_lr_swap_covers_every_anatomy_pair():
    """`build_lr_swap` must pair every L/R landmark in the real anatomy
    keypoint order (not just the synthetic test fixture's), and
    `assert_lr_swap_covers` must accept that full coverage rather than
    raising a false positive."""
    from omegaconf import OmegaConf
    from jarvis_jax.data.augment import assert_lr_swap_covers, build_lr_swap
    names = [str(n) for n in OmegaConf.load(_ANATOMY_V1).model.KP_NAMES]
    swap = build_lr_swap(names)
    assert_lr_swap_covers(names, required=names)
    moved = [i for i in range(len(names)) if swap[i] != i]
    assert len(moved) == 2 * sum(1 for n in names if n.startswith(("WingL", "T1L", "T2L", "T3L", "EyeL")))
    for i in moved:
        assert names[i].replace("L", "R", 1) == names[swap[i]] or names[swap[i]].replace("L", "R", 1) == names[i]


def test_lr_swap_refuses_a_half_pair():
    """A landmark named only on one side (its mirror missing from `names`)
    must raise, naming the missing partner -- this is what would otherwise
    let a horizontal flip silently relabel a left leg as itself."""
    from jarvis_jax.data.augment import assert_lr_swap_covers
    with pytest.raises(ValueError, match="WingR_base"):
        assert_lr_swap_covers(["WingL_base", "Scutellum"], required=["WingL_base", "WingR_base", "Scutellum"])


def test_t2_augmentation_keeps_gt3d_on_gt2d_in_both_frames(tmp_path):
    """Every geometric op must update M/t_local so the labels stay consistent --
    at T=2 the mirror's t_local broadcast and the camera dropout's all-frames
    AND are the two places that can silently break one frame only."""
    from jarvis_jax.data.augment import build_lr_swap
    from jarvis_jax.data.mv_augment import MVAugParams, augment_window
    from jarvis_jax.data.v12_windows import V12WindowDataset, window_batches
    from jarvis_jax.models.mvq.geometry import project_local
    root = make_v12_root(tmp_path, n_frames=5)
    ds = V12WindowDataset(root, "train", T=2, pair_deltas=(1,), train=False)
    b = next(window_batches(ds, 2, shuffle=False, num_workers=1, drop_last=False))
    jb = {k: jnp.asarray(v) for k, v in b.items()}
    # `_mirror` (mv_augment.py) broadcasts frame 0's t_local to every frame instead of
    # transforming each frame's own -- correct ONLY because the loader gives every frame
    # of a window the same crop origin (v12_windows.py's `_build`). Pin that invariant on
    # the RAW, pre-augmentation batch: if the loader ever gives frames their own origin,
    # this must fail FIRST, before the reprojection check below goes looking for the bug
    # in the wrong place.
    np.testing.assert_array_equal(np.asarray(jb["t_local"][:, 0]), np.asarray(jb["t_local"][:, 1]))
    p = MVAugParams(cam_drop_p=1.0, cam_drop_max=2, mirror_p=1.0)
    a = augment_window(jax.random.PRNGKey(0), jb, p, build_lr_swap(ds.keypoint_names))
    for t in range(2):
        uv = jax.vmap(lambda X, M, tl: project_local(X, M, tl))(a["kp3d_local"][:, 0, t], a["M"], a["t_local"][:, t])
        d = np.linalg.norm(np.asarray(uv) - np.moveaxis(np.asarray(a["kp2d"][:, 0, t]), 1, 2), axis=-1)
        m = np.asarray(a["vis2d"][:, 0, t])
        assert float(d[np.moveaxis(m, 1, 2)].max()) < 0.5      # px
    cv = np.asarray(a["cam_valid"])                             # (B,T,C)
    assert cv.sum(-1).min() >= 3                                # >= 3 cameras survive, in every frame
    assert np.array_equal(cv[:, 0], cv[:, 1])
