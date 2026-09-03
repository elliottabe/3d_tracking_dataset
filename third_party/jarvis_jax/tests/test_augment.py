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
    """Legacy path (keep_kp_in_bounds=False): the in-bounds AND still fires."""
    rng = np.random.RandomState(0)
    img = jnp.asarray(rng.randint(0, 256, (2, 32, 32, 4), dtype=np.uint8))
    # one kp near an edge so a big translation pushes it off-crop
    kp = jnp.asarray(np.tile([[1.0, 1.0]], (2, 50, 1)), jnp.float32)
    vis = jnp.ones((2, 50), bool)
    # deterministic large negative translation via a fixed key + max range
    out_img, out_kp, out_vis = affine_batch(
        jax.random.PRNGKey(0), img, kp, vis, rot_deg=0.0, scale_min=1.0,
        scale_max=1.0, translate_frac=0.5, heatmap_size=16, keep_kp_in_bounds=False)
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


from jarvis_jax.data.augment import AugParams, augment_batch


def test_augment_disabled_is_identity():
    rng = np.random.RandomState(0)
    img = jnp.asarray(rng.randint(0, 256, (2, 32, 32, 4), dtype=np.uint8))
    kp = jnp.asarray(rng.uniform(0, 16, (2, 50, 2)).astype(np.float32))
    vis = jnp.ones((2, 50), bool)
    sw = build_lr_swap(V3_NAMES)
    oi, ok, ov = augment_batch(jax.random.PRNGKey(0), img, kp, vis,
                               AugParams(enabled=False), sw, heatmap_size=16)
    assert jnp.array_equal(oi, img) and jnp.array_equal(ok, kp) and jnp.array_equal(ov, vis)


def test_augment_enabled_preserves_shapes_and_dtypes():
    rng = np.random.RandomState(0)
    img = jnp.asarray(rng.randint(0, 256, (2, 32, 32, 4), dtype=np.uint8))
    kp = jnp.asarray(rng.uniform(2, 14, (2, 50, 2)).astype(np.float32))
    vis = jnp.ones((2, 50), bool)
    sw = build_lr_swap(V3_NAMES)
    oi, ok, ov = augment_batch(jax.random.PRNGKey(0), img, kp, vis,
                               AugParams(enabled=True), sw, heatmap_size=16)
    assert oi.shape == img.shape and oi.dtype == jnp.uint8
    assert ok.shape == kp.shape and ov.shape == vis.shape and ov.dtype == jnp.bool_


def test_aug_config_composes():
    from hydra import initialize_config_dir, compose
    from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers
    register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["paths=hyak", "aug=default"])
    assert cfg.aug.enabled is True and cfg.aug.rot_deg == 30 and cfg.aug.flip_p == 0.5
    # new colour/noise robustness params present
    assert cfg.aug.blur_max == 0.5 and cfg.aug.noise_scale == 0.02 and cfg.aug.pc_color == 0.2


# --- new colour/noise augmentations (blur / gaussian-noise / per-channel colour) ---
from jarvis_jax.data.augment import (
    gaussian_blur_batch, gaussian_noise_batch, per_channel_multiply_batch)


def _rgb_img(seed=7):
    rng = np.random.RandomState(seed)
    return jnp.asarray(rng.randint(0, 256, (3, 24, 24, 4), dtype=np.uint8))


def test_gaussian_blur_disabled_is_identity_and_keeps_mask():
    img = _rgb_img()
    assert jnp.array_equal(gaussian_blur_batch(jax.random.PRNGKey(0), img, 0.0), img)
    out = gaussian_blur_batch(jax.random.PRNGKey(1), img, 0.5)
    assert out.shape == img.shape and out.dtype == jnp.uint8
    assert not jnp.array_equal(out[..., :3], img[..., :3])   # RGB changed
    assert jnp.array_equal(out[..., 3], img[..., 3])          # mask intact
    # blur reduces high-frequency energy (per-pixel variance) on average
    assert float(out[..., :3].astype(jnp.float32).var()) <= float(img[..., :3].astype(jnp.float32).var())


def test_gaussian_noise_disabled_is_identity_and_keeps_mask():
    img = _rgb_img(8)
    assert jnp.array_equal(gaussian_noise_batch(jax.random.PRNGKey(0), img, 0.0), img)
    out = gaussian_noise_batch(jax.random.PRNGKey(2), img, 0.05)
    assert out.shape == img.shape and out.dtype == jnp.uint8
    assert int(out[..., :3].max()) <= 255 and int(out[..., :3].min()) >= 0
    assert not jnp.array_equal(out[..., :3], img[..., :3])
    assert jnp.array_equal(out[..., 3], img[..., 3])


def test_per_channel_multiply_disabled_is_identity_and_keeps_mask():
    img = _rgb_img(9)
    assert jnp.array_equal(per_channel_multiply_batch(jax.random.PRNGKey(0), img, 0.0), img)
    out = per_channel_multiply_batch(jax.random.PRNGKey(3), img, 0.2)
    assert out.shape == img.shape and out.dtype == jnp.uint8
    assert not jnp.array_equal(out[..., :3], img[..., :3])
    assert jnp.array_equal(out[..., 3], img[..., 3])


# --- affine bounds-fitting: the transform must not push annotated keypoints
# --- out of the crop (measured 2026-09-02: the wall crop kept only 82.7% of
# --- its 49 annotated keypoints per draw under configs/aug/default.yaml, and
# --- the six most-dropped were ALL tarsal tips).
from jarvis_jax.data.augment import fit_affine_to_bounds

DEF = dict(rot_deg=30.0, scale_min=0.8, scale_max=1.25, translate_frac=0.1)


def _cloud(cx, cy, spread, K=50, seed=11):
    """A (K,2) keypoint cloud centred at (cx,cy) in heatmap units."""
    rng = np.random.RandomState(seed)
    return (np.stack([cx + rng.uniform(-spread, spread, K),
                      cy + rng.uniform(-spread, spread, K)], -1).astype(np.float32))


def _kept_fraction(kp, hm=224, ndraw=200, keep=True, **prm):
    """Mean fraction of the annotated keypoints that survive affine_batch's
    in-bounds AND, over `ndraw` independent draws."""
    K = kp.shape[0]
    img = jnp.zeros((1, 2 * hm, 2 * hm, 4), jnp.uint8)
    kp_b = jnp.asarray(kp)[None]
    vis_b = jnp.ones((1, K), bool)
    fr = []
    for d in range(ndraw):
        _, _, v = affine_batch(jax.random.PRNGKey(7919 * (d + 1)), img, kp_b, vis_b,
                               heatmap_size=hm, keep_kp_in_bounds=keep, **prm)
        fr.append(float(np.asarray(v).mean()))
    return float(np.mean(fr))


def test_affine_keeps_visible_kp_in_bounds_on_a_wall_like_crop():
    """A fly pressed against the crop edge (centroid 80 hm-units off-centre,
    the measured wall case) must not lose supervision to the augmentation."""
    kp = _cloud(224 / 2 - 80, 224 / 2 - 40, spread=40)
    assert _kept_fraction(kp, keep=True, **DEF) == 1.0
    # and the defect is real in the legacy path (guards against a tautology)
    assert _kept_fraction(kp, keep=False, **DEF) < 0.95


def test_affine_never_touches_the_sampled_rotation():
    """Bounds-fitting shrinks scale and translation only: rotation diversity,
    the augmentation that actually teaches orientation invariance, is intact."""
    hm = 224
    kp = jnp.asarray(_cloud(hm / 2 - 80, hm / 2 - 40, spread=40))[None]
    vis = jnp.ones((1, kp.shape[1]), bool)
    for d in range(25):
        key = jax.random.PRNGKey(1234 + d)
        th = jnp.deg2rad(jax.random.uniform(jax.random.split(key, 4)[0], (1,),
                                            minval=-DEF["rot_deg"], maxval=DEF["rot_deg"]))
        th2, _, _, _, _ = fit_affine_to_bounds(kp, vis, th, jnp.ones(1) * 1.25,
                                               jnp.zeros(1), jnp.zeros(1), hm, 0.0)
        assert float(abs(th2[0] - th[0])) < 1e-6


def test_affine_rotates_about_the_visible_keypoint_centroid():
    """One visible keypoint => the transform is centred ON it, so with s=1 and
    zero translation it cannot move no matter what rotation was drawn."""
    hm = 32
    kp = jnp.asarray([[[6.0, 8.0]]], jnp.float32)          # (1,1,2), far off-centre
    vis = jnp.ones((1, 1), bool)
    img = jnp.zeros((1, 2 * hm, 2 * hm, 4), jnp.uint8)
    for d in range(8):
        _, k2, v2 = affine_batch(jax.random.PRNGKey(d), img, kp, vis, rot_deg=180.0,
                                 scale_min=1.0, scale_max=1.0, translate_frac=0.0,
                                 heatmap_size=hm, keep_kp_in_bounds=True)
        assert bool(v2[0, 0])
        assert float(jnp.abs(k2[0, 0] - kp[0, 0]).max()) < 1e-3
    # legacy path rotates about the crop centre, so it DOES move
    moved = max(float(jnp.abs(affine_batch(
        jax.random.PRNGKey(d), img, kp, vis, rot_deg=180.0, scale_min=1.0,
        scale_max=1.0, translate_frac=0.0, heatmap_size=hm,
        keep_kp_in_bounds=False)[1][0, 0] - kp[0, 0]).max()) for d in range(8))
    assert moved > 1.0


def test_affine_image_warp_shares_the_keypoint_centre():
    """The image must be warped about the SAME centre as the keypoints: a
    bright pixel colocated with the single visible keypoint stays put."""
    hm = 32
    W = 2 * hm
    kx, ky = 6.0, 8.0
    img = np.zeros((1, W, W, 4), np.uint8)
    img[0, int(2 * ky), int(2 * kx), 0] = 255
    kp = jnp.asarray([[[kx, ky]]], jnp.float32)
    vis = jnp.ones((1, 1), bool)
    for d in range(6):
        i2, _, _ = affine_batch(jax.random.PRNGKey(100 + d), jnp.asarray(img), kp, vis,
                                rot_deg=180.0, scale_min=1.0, scale_max=1.0,
                                translate_frac=0.0, heatmap_size=hm,
                                keep_kp_in_bounds=True)
        flat = int(jnp.argmax(i2[0, ..., 0]))
        py, px = flat // W, flat % W
        assert abs(px - 2 * kx) <= 1.5 and abs(py - 2 * ky) <= 1.5


def test_affine_bounds_fit_leaves_a_centred_crop_uncorrected():
    """A fly in the middle of its crop never violates the bound, so neither
    corrective term may fire: the sampled scale and translation come back
    EXACTLY as drawn. (The transform centre still moves to the keypoint
    centroid -- that is the fix, not a correction, and is tested above.)"""
    hm = 224
    kp = jnp.asarray(_cloud(hm / 2, hm / 2, spread=35))[None]
    vis = jnp.ones((1, kp.shape[1]), bool)
    img = jnp.asarray(np.random.RandomState(5).randint(0, 256, (1, 2 * hm, 2 * hm, 4),
                                                       dtype=np.uint8))
    for d in range(20):
        key = jax.random.PRNGKey(31337 + d)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        th = jnp.deg2rad(jax.random.uniform(k1, (1,), minval=-DEF["rot_deg"],
                                            maxval=DEF["rot_deg"]))
        s0 = jax.random.uniform(k2, (1,), minval=DEF["scale_min"], maxval=DEF["scale_max"])
        tx0 = jax.random.uniform(k3, (1,), minval=-DEF["translate_frac"],
                                 maxval=DEF["translate_frac"])
        ty0 = jax.random.uniform(k4, (1,), minval=-DEF["translate_frac"],
                                 maxval=DEF["translate_frac"])
        _, s1, tx1, ty1, _ = fit_affine_to_bounds(kp, vis, th, s0, tx0, ty0, hm, 0.01)
        assert float(s1[0]) == float(s0[0])
        assert abs(float(tx1[0]) - float(tx0[0])) < 1e-7
        assert abs(float(ty1[0]) - float(ty0[0])) < 1e-7
        # and no supervision is lost either way
        assert bool(affine_batch(key, img, kp, vis, heatmap_size=hm,
                                 keep_kp_in_bounds=True, **DEF)[2].all())
        assert bool(affine_batch(key, img, kp, vis, heatmap_size=hm,
                                 keep_kp_in_bounds=False, **DEF)[2].all())


def test_fit_affine_to_bounds_shrinks_scale_by_the_exact_needed_factor():
    """Hand-computed: two visible kp 12 units apart in a 16-unit grid
    (bounds [0,15]). s=1.5 would span 18 > 15, so s must land on 15/12=1.25
    and the translation on the single feasible value, -0.5 heatmap units."""
    hm = 16
    kp = jnp.asarray([[[2.0, 8.0], [14.0, 8.0]]], jnp.float32)
    vis = jnp.ones((1, 2), bool)
    th, s, tx, ty, cen = fit_affine_to_bounds(
        kp, vis, jnp.zeros(1), jnp.ones(1) * 1.5, jnp.zeros(1), jnp.zeros(1), hm, 0.0)
    assert abs(float(s[0]) - 1.25) < 1e-5
    assert abs(float(tx[0]) * hm - (-0.5)) < 1e-4
    assert abs(float(cen[0, 0]) - 8.0) < 1e-5 and abs(float(cen[0, 1]) - 8.0) < 1e-5
    assert abs(float(ty[0])) < 1e-6          # y span is zero -> unconstrained


def test_fit_affine_to_bounds_ignores_invisible_keypoints():
    """An annotation already off-crop (vis=False) carries no supervision, so it
    must not constrain the transform -- otherwise one bad label collapses the
    augmentation for the whole sample."""
    hm = 16
    kp = jnp.asarray([[[8.0, 8.0], [-500.0, 8.0]]], jnp.float32)
    vis = jnp.asarray([[True, False]])
    _, s, tx, _, cen = fit_affine_to_bounds(
        kp, vis, jnp.zeros(1), jnp.ones(1) * 1.25, jnp.ones(1) * 0.05,
        jnp.zeros(1), hm, 0.0)
    assert abs(float(s[0]) - 1.25) < 1e-6          # unshrunk
    assert abs(float(tx[0]) - 0.05) < 1e-6         # unclipped
    assert abs(float(cen[0, 0]) - 8.0) < 1e-5      # centroid is the visible kp only


def test_affine_bounds_fit_survives_a_fully_invisible_sample():
    """No visible keypoint => nothing to protect; must not divide by zero."""
    hm = 32
    kp = jnp.asarray(_cloud(hm / 2, hm / 2, 8, K=5))[None]
    vis = jnp.zeros((1, 5), bool)
    img = jnp.zeros((1, 2 * hm, 2 * hm, 4), jnp.uint8)
    i2, k2, v2 = affine_batch(jax.random.PRNGKey(0), img, kp, vis, heatmap_size=hm,
                              keep_kp_in_bounds=True, **DEF)
    assert bool(jnp.isfinite(k2).all()) and not bool(v2.any())


def test_affine_bounds_fit_is_jittable_and_per_sample():
    """Must survive the nnx.jit train step, and must fit each sample on its own
    (a batch mixing a wall crop with a centred one may not shrink both)."""
    hm = 224
    kp = jnp.asarray(np.stack([_cloud(hm / 2 - 80, hm / 2 - 40, 40),
                               _cloud(hm / 2, hm / 2, 35, seed=3)]))
    vis = jnp.ones((2, kp.shape[1]), bool)
    img = jnp.zeros((2, 2 * hm, 2 * hm, 4), jnp.uint8)
    f = jax.jit(lambda k, i, p, v, keep: affine_batch(
        k, i, p, v, heatmap_size=hm, keep_kp_in_bounds=keep, **DEF), static_argnums=4)
    _, k2, v2 = f(jax.random.PRNGKey(0), img, kp, vis, True)
    assert bool(v2.all())
    _, kb, _ = f(jax.random.PRNGKey(0), img, kp, vis, False)
    # per-sample: the wall crop is corrected by much more than the centred one
    d_wall = float(jnp.abs(k2[0] - kb[0]).max())
    d_centred = float(jnp.abs(k2[1] - kb[1]).max())
    assert d_wall > 20.0 and d_centred < 5.0


# --- targeted cutout (PROPOSED, default-off). Measured 2026-09-02 over 60
# --- real V5 train crops: the fly silhouette is 9.6% +- 1.7% of a 448-px crop
# --- and only 10.4% of the pixels cutout erases land on it -- i.e. no better
# --- than chance, because the box centres are uniform over the crop.

def _old_cutout(key, img4_u8, n, frac):
    """The pre-2026-09-02 cutout, inlined as the regression reference."""
    B, H, W, _ = img4_u8.shape
    side = max(1, int(round(frac * W)))
    out = img4_u8
    xs = jnp.arange(W)[None, :]
    ys = jnp.arange(H)[None, :]
    for j in range(int(n)):
        k1, k2 = jax.random.split(jax.random.fold_in(key, j))
        cx = jax.random.randint(k1, (B,), 0, W)
        cy = jax.random.randint(k2, (B,), 0, H)
        x0 = (cx - side // 2)[:, None]; x1 = (cx + side // 2)[:, None]
        y0 = (cy - side // 2)[:, None]; y1 = (cy + side // 2)[:, None]
        box = ((ys >= y0) & (ys < y1))[:, :, None] & ((xs >= x0) & (xs < x1))[:, None, :]
        out = jnp.concatenate([out[..., :3] * (~box)[..., None].astype(out.dtype),
                               out[..., 3:]], axis=-1)
    return out


def _crop_with_blob(cx, cy, r, W=64, seed=1):
    """(1,W,W,4) uint8 crop: uniform RGB plus a disc silhouette in channel 3."""
    rng = np.random.RandomState(seed)
    img = rng.randint(40, 210, (1, W, W, 4)).astype(np.uint8)
    yy, xx = np.mgrid[0:W, 0:W]
    img[0, ..., 3] = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8)
    return jnp.asarray(img)


def _on_fly_fraction(img_before, img_after):
    """Fraction of the pixels cutout erased that sit on the silhouette."""
    er = (np.asarray(img_after)[..., :3].sum(-1) == 0) & \
         (np.asarray(img_before)[..., :3].sum(-1) != 0)
    m = np.asarray(img_before)[..., 3] > 0
    return float((er & m).sum()) / max(int(er.sum()), 1)


def test_cutout_defaults_reproduce_the_old_implementation_exactly():
    """Both new options default off, so nothing that trains today may move --
    including the RNG stream (the extra draws are taken only when enabled)."""
    img = _crop_with_blob(20, 20, 8)
    for j in range(5):
        k = jax.random.PRNGKey(j)
        assert jnp.array_equal(cutout_batch(k, img, 2, 0.25), _old_cutout(k, img, 2, 0.25))
        assert jnp.array_equal(
            cutout_batch(k, img, 3, 0.3, mask_target_p=0.0, size_rel_fly=0.0),
            _old_cutout(k, img, 3, 0.3))


def test_cutout_mask_targeting_moves_the_boxes_onto_the_fly():
    """The whole point: with mask_target_p=1 nearly every erased pixel that
    can be on the animal is, instead of ~its area share of the crop."""
    img = _crop_with_blob(16, 16, 9)                  # blob is ~6% of the crop
    uni, tgt = [], []
    for j in range(30):
        k = jax.random.PRNGKey(500 + j)
        uni.append(_on_fly_fraction(img, cutout_batch(k, img, 2, 0.25)))
        tgt.append(_on_fly_fraction(img, cutout_batch(k, img, 2, 0.25,
                                                      mask_target_p=1.0)))
    assert np.mean(tgt) > 4.0 * np.mean(uni)
    assert np.mean(uni) < 0.15


def test_cutout_mask_targeting_stays_a_mixture_at_p_half():
    """p=0.5 must sit strictly between the two, or it is not a mixture and has
    simply replaced one bias with another."""
    img = _crop_with_blob(16, 16, 9)
    f = lambda p: np.mean([_on_fly_fraction(img, cutout_batch(
        jax.random.PRNGKey(900 + j), img, 2, 0.25, mask_target_p=p))
        for j in range(30)])
    lo, mid, hi = f(0.0), f(0.5), f(1.0)
    assert lo < mid < hi


def test_cutout_size_rel_fly_scales_the_box_with_the_animal():
    """A fly twice as long gets a box twice as wide -- 4x the erased area --
    so the occlusion is a constant fraction of the ANIMAL, not of the crop."""
    small = _crop_with_blob(32, 32, 6)                # bbox side ~12 px
    big = _crop_with_blob(32, 32, 12)                 # bbox side ~24 px
    area = lambda im: np.mean([
        float((np.asarray(cutout_batch(jax.random.PRNGKey(j), im, 1, 0.25,
                                       size_rel_fly=0.5))[..., :3].sum(-1) == 0).sum())
        for j in range(20)])
    r = area(big) / max(area(small), 1.0)
    assert 3.0 < r < 5.0


def test_cutout_targeting_degrades_to_uniform_on_an_empty_mask():
    """Crops whose SAM mask failed must not blow up or produce NaN centres."""
    img = _crop_with_blob(16, 16, 0)                  # r=0 -> empty silhouette
    out = cutout_batch(jax.random.PRNGKey(3), img, 2, 0.25, mask_target_p=1.0,
                       size_rel_fly=0.5)
    assert out.shape == img.shape and out.dtype == jnp.uint8
    assert int((out[..., :3] == 0).sum()) > 0
    assert jnp.array_equal(out[..., 3], img[..., 3])
    # and it must degrade to UNIFORM, not pile every box on one cell. The eps
    # floor inside _sample_mask_centre is what does that (log(0)=-inf on every
    # cell makes the categorical argmax degenerate), so test it directly.
    from jarvis_jax.data.augment import _sample_mask_centre
    empty = jnp.zeros((1, 128, 128), bool)
    cs = [tuple(float(v[0]) for v in _sample_mask_centre(jax.random.PRNGKey(j),
                                                         empty, 8))
          for j in range(25)]
    assert all(np.isfinite(c).all() for c in cs)
    assert len({tuple(np.round(c).astype(int)) for c in cs}) > 15
    assert np.ptp(np.asarray(cs)[:, 0]) > 40.0


def test_cutout_targeting_still_leaves_the_mask_channel_intact():
    """BY DEFAULT cutout writes channels 0-2 only. For a mask-ON model that
    means an RGB-occluded limb is still visible in channel 3 -- the occlusion
    is much weaker than it looks. `punch_mask=True` is the opt-in fix; this
    pins the historical default."""
    img = _crop_with_blob(16, 16, 9)
    out = cutout_batch(jax.random.PRNGKey(4), img, 2, 0.25, mask_target_p=1.0)
    assert jnp.array_equal(out[..., 3], img[..., 3])


def test_cutout_punch_mask_erases_channel_3_without_moving_the_boxes():
    """`punch_mask` must change ONLY channel 3: same RNG stream, same boxes,
    byte-identical RGB. If the RGB moved too, the flag would be confounding
    the very A/B it exists to clean up."""
    img = _crop_with_blob(16, 16, 9)
    k = jax.random.PRNGKey(4)
    plain = cutout_batch(k, img, 2, 0.25, mask_target_p=1.0)
    punched = cutout_batch(k, img, 2, 0.25, mask_target_p=1.0, punch_mask=True)
    assert jnp.array_equal(plain[..., :3], punched[..., :3])
    assert not jnp.array_equal(plain[..., 3], punched[..., 3])
    # channel 3 is erased exactly where RGB was
    erased = (plain[..., 0] == 0) & (plain[..., 1] == 0) & (plain[..., 2] == 0)
    assert jnp.all(punched[..., 3][erased] == 0)


def test_cutout_punch_mask_is_an_exact_noop_on_a_mask_ablated_batch():
    """THE PROPERTY THAT MAKES IT SAFE ON BOTH ARMS OF A MASK A/B. A mask-OFF
    arm reaches augmentation with channel 3 already zeroed by
    ZeroMaskDataset, so punching zeros into zeros must change nothing at all
    -- enabling the flag uniformly leaves that arm bit-identical and removes
    the free hint only from the mask-ON arm."""
    img = _crop_with_blob(16, 16, 9).at[..., 3].set(0)
    k = jax.random.PRNGKey(4)
    plain = cutout_batch(k, img, 2, 0.25)
    punched = cutout_batch(k, img, 2, 0.25, punch_mask=True)
    assert jnp.array_equal(plain, punched)


def test_affine_n_fit_ignores_distractor_rows_but_warps_them():
    """Rows beyond n_fit (the other fly's keypoints) must not change the fitted
    affine: the primary rows come out identical with or without them, and the
    extra rows go through the same warp (a distractor far off-crop stays
    invisible rather than dragging the crop toward itself)."""
    import jax
    from jarvis_jax.data.augment import affine_batch
    key = jax.random.PRNGKey(3)
    B, K, hm = 2, 3, 224
    img = jnp.zeros((B, 448, 448, 4), jnp.uint8)
    prim = jnp.asarray(np.random.RandomState(0).uniform(60, 160, (B, K, 2)).astype("float32"))
    dist = jnp.asarray(np.array([[[500.0, 500.0], [10.0, 10.0], [-300.0, 100.0]]] * B, "float32"))
    vis_p = jnp.ones((B, K), bool)
    kw = dict(rot_deg=30, scale_min=0.8, scale_max=1.25, translate_frac=0.1, heatmap_size=hm)
    _, kp_alone, vis_alone = affine_batch(key, img, prim, vis_p, **kw)
    _, kp_both, vis_both = affine_batch(
        key, img, jnp.concatenate([prim, dist], 1), jnp.ones((B, 2 * K), bool), n_fit=K, **kw)
    np.testing.assert_allclose(np.asarray(kp_both[:, :K]), np.asarray(kp_alone), atol=1e-4)
    np.testing.assert_array_equal(np.asarray(vis_both[:, :K]), np.asarray(vis_alone))
    # the far-off distractor row is out of bounds after the warp -> not visible
    assert not bool(vis_both[:, K].any())
    # WITHOUT n_fit the distractor rows would have been fitted in: different affine
    _, kp_fit_all, _ = affine_batch(
        key, img, jnp.concatenate([prim, dist], 1), jnp.ones((B, 2 * K), bool), **kw)
    assert not np.allclose(np.asarray(kp_fit_all[:, :K]), np.asarray(kp_alone), atol=1e-3)
