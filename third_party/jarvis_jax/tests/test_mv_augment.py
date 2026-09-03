# tests/test_mv_augment.py
import numpy as np
import jax, jax.numpy as jnp
import pytest
from mvq_fixtures import make_v12_root


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
