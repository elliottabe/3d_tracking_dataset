import numpy as np
from jarvis_jax.data.augment import build_lr_swap

V3_NAMES = ['Antenna_Base', 'EyeL', 'EyeR', 'Scutellum', 'Abd_A4', 'Abd_tip',
            'WingL_base', 'WingL_V12', 'WingL_V13', 'T1L_ThxCx', 'T1L_Tro',
            'T1L_FeTi', 'T1L_TiTa', 'T1L_TaT1', 'T1L_TaT3', 'T1L_TaTip',
            'T2L_Tro', 'T2L_FeTi', 'T2L_TiTa', 'T2L_TaT1', 'T2L_TaT3', 'T2L_TaTip',
            'T3L_Tro', 'T3L_FeTi', 'T3L_TiTa', 'T3L_TaT1', 'T3L_TaT3', 'T3L_TaTip',
            'WingR_base', 'WingR_V12', 'WingR_V13', 'T1R_ThxCx', 'T1R_Tro',
            'T1R_FeTi', 'T1R_TiTa', 'T1R_TaT1', 'T1R_TaT3', 'T1R_TaTip',
            'T2R_Tro', 'T2R_FeTi', 'T2R_TiTa', 'T2R_TaT1', 'T2R_TaT3', 'T2R_TaTip',
            'T3R_Tro', 'T3R_FeTi', 'T3R_TiTa', 'T3R_TaT1', 'T3R_TaT3', 'T3R_TaTip']


def test_build_lr_swap_pairs_and_involution():
    sw = build_lr_swap(V3_NAMES)
    idx = {n: i for i, n in enumerate(V3_NAMES)}
    # L<->R pairs
    assert sw[idx['EyeL']] == idx['EyeR'] and sw[idx['EyeR']] == idx['EyeL']
    assert sw[idx['T1L_TaTip']] == idx['T1R_TaTip']
    assert sw[idx['WingL_base']] == idx['WingR_base']
    # midline keypoints map to themselves
    for m in ('Antenna_Base', 'Scutellum', 'Abd_A4', 'Abd_tip'):
        assert sw[idx[m]] == idx[m]
    # involution covering all K
    assert np.array_equal(sw[sw], np.arange(len(V3_NAMES)))
    assert sw.dtype == np.int32


import jax
import jax.numpy as jnp
from jarvis_jax.data.augment import (
    affine_warp_image, affine_transform_kp, affine_batch)


def test_affine_identity_is_noop():
    img = jnp.asarray(np.random.RandomState(0).randint(0, 256, (16, 16, 4)), jnp.float32)
    out = affine_warp_image(img, theta=0.0, s=1.0, tx=0.0, ty=0.0)
    assert jnp.allclose(out, img, atol=1e-4)
    kp = jnp.asarray([[3.0, 5.0], [10.0, 2.0]])
    kpo = affine_transform_kp(kp, 0.0, 1.0, 0.0, 0.0, heatmap_size=16)
    assert jnp.allclose(kpo, kp, atol=1e-4)


def test_affine_rotation_image_and_kp_consistent():
    """A bright pixel and a keypoint placed at the SAME location move together
    under a 90deg rotation (image is 2x the keypoint resolution)."""
    hm = 16
    W = 2 * hm                                   # 32
    img = jnp.zeros((W, W, 4), jnp.float32)
    # bright pixel at image coord matching keypoint (kx,ky)=(12,4) -> 2x = (24,8)
    kx, ky = 12.0, 4.0
    img = img.at[int(2 * ky), int(2 * kx), 0].set(255.0)   # [row=y, col=x]
    theta = jnp.pi / 2                            # 90 deg
    warped = affine_warp_image(img, theta=theta, s=1.0, tx=0.0, ty=0.0)
    kp = jnp.asarray([[kx, ky]])
    kpo = affine_transform_kp(kp, theta, 1.0, 0.0, 0.0, heatmap_size=hm)[0]
    # the brightest warped pixel should sit at 2x the transformed keypoint
    flat = int(jnp.argmax(warped[..., 0]))
    py, px = flat // W, flat % W
    assert abs(px - 2 * float(kpo[0])) <= 2.0 and abs(py - 2 * float(kpo[1])) <= 2.0


def test_affine_batch_offcrop_sets_vis_false():
    rng = np.random.RandomState(0)
    img = jnp.asarray(rng.randint(0, 256, (2, 32, 32, 4), dtype=np.uint8))
    # one kp near an edge so a big translation pushes it off-crop
    kp = jnp.asarray(np.tile([[1.0, 1.0]], (2, 50, 1)), jnp.float32)
    vis = jnp.ones((2, 50), bool)
    # deterministic large negative translation via a fixed key + max range
    out_img, out_kp, out_vis = affine_batch(
        jax.random.PRNGKey(0), img, kp, vis, rot_deg=0.0, scale_min=1.0,
        scale_max=1.0, translate_frac=0.5, heatmap_size=16)
    assert out_img.shape == img.shape and out_img.dtype == jnp.uint8
    assert out_kp.shape == kp.shape
    # at least some keypoints fall off-crop and are marked invisible
    assert bool((~out_vis).any())


from jarvis_jax.data.augment import flip_batch, cutout_batch, photometric_batch


def test_flip_batch_mirrors_x_and_swaps_indices():
    sw = build_lr_swap(V3_NAMES)
    K = len(V3_NAMES)
    img = jnp.asarray(np.random.RandomState(1).randint(0, 256, (1, 8, 8, 4), dtype=np.uint8))
    kp = jnp.asarray(np.random.RandomState(2).uniform(0, 16, (1, K, 2)).astype(np.float32))
    vis = jnp.ones((1, K), bool)
    # flip_p=1.0 -> always flip (deterministic)
    fi, fk, fv = flip_batch(jax.random.PRNGKey(0), img, kp, vis, sw, 1.0, heatmap_size=16)
    # image mirrored along width
    assert jnp.array_equal(fi[0], img[0, :, ::-1, :])
    idxL = V3_NAMES.index("EyeL"); idxR = V3_NAMES.index("EyeR")
    # the flipped EyeL slot holds the mirrored original EyeR x
    assert abs(float(fk[0, idxL, 0]) - (15.0 - float(kp[0, idxR, 0]))) < 1e-3
    assert abs(float(fk[0, idxL, 1]) - float(kp[0, idxR, 1])) < 1e-3


def test_cutout_zeros_rgb_keeps_mask_and_kp():
    img = jnp.full((1, 16, 16, 4), 100, jnp.uint8)
    out = cutout_batch(jax.random.PRNGKey(0), img, n=2, frac=0.5)
    assert out.shape == img.shape
    assert int((out[..., :3] == 0).sum()) > 0          # some RGB erased
    assert jnp.array_equal(out[..., 3], img[..., 3])   # mask channel intact


def test_photometric_changes_rgb_within_range_keeps_mask():
    img = jnp.asarray(np.random.RandomState(3).randint(0, 256, (2, 16, 16, 4), dtype=np.uint8))
    out = photometric_batch(jax.random.PRNGKey(0), img, 0.3, 0.3, 0.3)
    assert out.shape == img.shape and out.dtype == jnp.uint8
    assert int(out[..., :3].max()) <= 255 and int(out[..., :3].min()) >= 0
    assert not jnp.array_equal(out[..., :3], img[..., :3])   # RGB changed
    assert jnp.array_equal(out[..., 3], img[..., 3])         # mask intact
