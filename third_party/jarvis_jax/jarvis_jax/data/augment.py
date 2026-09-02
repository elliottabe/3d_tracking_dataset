"""On-device (jitted) train-time data augmentation for ViTPose 2D.

Geometric affine + horizontal flip (with a left/right keypoint-index swap) +
cutout + photometric, applied to a (img4_u8, kp_xy, vis) batch before heatmap
rendering. Keypoints are transformed in lockstep with the image. Train-only;
`AugParams.enabled == False` is an exact identity.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


def _mirror_name(n):
    """The left/right mirror of a keypoint name, or None if it is midline.
    The side token is the last char of the segment before the first '_'
    (e.g. 'EyeL'->'EyeR', 'WingL_base'->'WingR_base', 'T1L_TaTip'->'T1R_TaTip')."""
    head = n.split("_", 1)[0]
    if head.endswith("L"):
        return n.replace(head, head[:-1] + "R", 1)
    if head.endswith("R"):
        return n.replace(head, head[:-1] + "L", 1)
    return None


def build_lr_swap(names):
    """Permutation index array mapping each keypoint to its L/R mirror (midline
    -> itself). Returns int32 (K,). Asserts the result is an involution."""
    idx = {n: i for i, n in enumerate(names)}
    swap = list(range(len(names)))
    for i, n in enumerate(names):
        m = _mirror_name(n)
        if m is not None and m in idx:
            swap[i] = idx[m]
    arr = np.asarray(swap, dtype=np.int32)
    assert np.array_equal(arr[arr], np.arange(len(names))), \
        "lr_swap is not an involution — check keypoint L/R naming"
    return arr


def _rot(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])


def affine_warp_image(img, theta, s, tx, ty, cx=None, cy=None):
    """Inverse-sample warp of (H,W,C) float32 about (cx, cy) -- the geometric
    center when they are None. Output =
    c + s*R(theta)*(src-c) + (tx,ty)*W  (so src is solved inversely).
    Bilinear, out-of-bounds -> 0.

    ``cx``/``cy`` are in IMAGE pixels and must be the same point that
    ``affine_transform_kp`` is given in heatmap units, or the keypoints stop
    tracking the pixels (which no loss curve would show -- see
    ``scripts/viz/vit_augmentation_preview.py``)."""
    H, W, C = img.shape
    cx = W / 2.0 if cx is None else cx
    cy = H / 2.0 if cy is None else cy
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    ox = xs.astype(jnp.float32) - cx - tx * W
    oy = ys.astype(jnp.float32) - cy - ty * W
    inv = _rot(-theta) / s
    src_x = inv[0, 0] * ox + inv[0, 1] * oy + cx
    src_y = inv[1, 0] * ox + inv[1, 1] * oy + cy

    def samp(ch):
        return jax.scipy.ndimage.map_coordinates(
            ch, [src_y, src_x], order=1, mode="constant", cval=0.0)

    return jnp.stack([samp(img[..., k]) for k in range(C)], axis=-1)


def affine_transform_kp(kp, theta, s, tx, ty, heatmap_size, cx=None, cy=None):
    """Forward affine on (K,2) keypoints (x,y) in heatmap coords, about
    (cx, cy) in the same units (the grid center when they are None)."""
    c = heatmap_size / 2.0
    cx = c if cx is None else cx
    cy = c if cy is None else cy
    R = _rot(theta)
    rel_x = kp[:, 0] - cx
    rel_y = kp[:, 1] - cy
    nx = R[0, 0] * rel_x + R[0, 1] * rel_y
    ny = R[1, 0] * rel_x + R[1, 1] * rel_y
    out_x = cx + s * nx + tx * heatmap_size
    out_y = cy + s * ny + ty * heatmap_size
    return jnp.stack([out_x, out_y], axis=-1)



def fit_affine_to_bounds(kp, vis, theta, s, tx, ty, heatmap_size, margin=0.0,
                         center=None):
    """Shrink a sampled affine, in closed form, until every VISIBLE keypoint
    lands inside [margin, heatmap_size-1-margin] -- and report the center it
    must be applied about.

    Why this exists. ``affine_batch`` ANDs an in-bounds test into ``vis``, and
    that AND is correct: a Gaussian target centred outside the heatmap has no
    peak to supervise. The defect is that the sampled transform CREATES the
    out-of-bounds condition, and does so unevenly. Measured over 200 draws per
    preset on real V5 crops (``figures/2026-09-02-vit-augmentations/
    augmentation_stats.json``), a centred female crop kept 100% of its 50
    annotated keypoints while a female pressed against the chamber wall kept
    82.7% (aug=default) / 74.4% (aug=heavy), worst draw 10/49. The six
    most-dropped keypoints were ALL tarsal tips (T2R_TaTip 72%, T2L_TaTip 68%,
    T3R/T3L_TaTip 46%, T1L/T1R_TaTip 43-44%) -- the same keypoints the
    detector A/B measured worst (T2R_TaTip 60 px). So the augmentation starved
    supervision exactly where the model then failed. Fix the transform, not
    the visibility rule.

    Three closed-form corrections, applied in order, each chosen so the more
    valuable augmentation survives:

    1. **Center on the visible-keypoint centroid, not the crop center.** This
       is the whole cause. ``crop_origin`` clamps the 448-px window to the
       image, so a fly at the arena wall sits ~80 heatmap-units off centre;
       rotating about the crop centre then swings it through an arc of
       2*80*sin(rot/2) units before the fly's own ~40-unit reach is counted.
       Rotating about the fly removes that lever arm entirely and costs no
       diversity at all. On a centred crop the two centres coincide, so those
       samples are untouched bit-for-bit.
    2. **Clip the translation into the feasible window.** Translation is the
       cheapest of the three to give up (the crop is already re-centred on a
       detection at inference).
    3. **Only if no translation can fit the rotated keypoint bbox, shrink the
       scale** to the exact factor that does. Rotation is never touched: it is
       the augmentation that teaches orientation invariance, which is what a
       heatmap model most needs, so it is the last thing to sacrifice -- and
       in practice it is never sacrificed.

    ``vis`` is used only to decide which keypoints must be protected. This
    dataset's visibility flags are unreliable as OCCLUSION labels (1864/1871
    val annotations are marked fully visible), but that is not what they are
    read for here: ``transforms.transform_keypoints`` sets vis=False for
    annotations already outside the crop, and those genuinely carry no
    supervision, so letting them constrain the transform would collapse the
    augmentation for the whole sample over a label that trains nothing.

    Args:
      kp: (B,K,2) keypoints in heatmap units, vis: (B,K) bool,
      theta/s/tx/ty: (B,) sampled affine (tx,ty are fractions of heatmap_size),
      margin: keep keypoints this far inside the border (heatmap units). The
        caller's default is 0.01 -- 0.02 crop px, ~5000x the measured float32
        round-trip noise of 2e-6 units and geometrically nothing, but without
        it ~0.06% of keypoints land a rounding error outside and are dropped.
      center: (B,2) override for the transform centre, in heatmap units. Only
        for A/B'ing the centring choice (``scripts/viz/vit_augmentation_
        preview.py --fix-ablation``): passing the crop centre reproduces the
        old centring with only the clip/shrink corrections, which the
        measurement showed pins the translation to a window endpoint on most
        wall draws. None (the default) uses the visible-keypoint centroid.
    Returns:
      (theta, s_fit, tx_fit, ty_fit, center) -- center is (B,2) in heatmap
      units and MUST be passed to both ``affine_transform_kp`` and (scaled to
      image pixels) ``affine_warp_image``.
    """
    hm = float(heatmap_size)
    lo = float(margin)
    hi = (hm - 1.0) - float(margin)
    avail = hi - lo

    w = vis.astype(jnp.float32)
    n = w.sum(axis=-1)
    any_vis = n > 0
    if center is None:
        cen = (kp * w[..., None]).sum(axis=1) / jnp.where(any_vis, n, 1.0)[:, None]
        cen = jnp.where(any_vis[:, None], cen, hm / 2.0)
    else:
        cen = jnp.broadcast_to(jnp.asarray(center, jnp.float32), (kp.shape[0], 2))

    # keypoints rotated about that centre at unit scale, zero translation
    c, sn = jnp.cos(theta)[:, None], jnp.sin(theta)[:, None]
    rx = kp[..., 0] - cen[:, 0:1]
    ry = kp[..., 1] - cen[:, 1:2]
    ux = c * rx - sn * ry
    uy = sn * rx + c * ry

    BIG = jnp.float32(1e9)
    umin_x = jnp.min(jnp.where(vis, ux, BIG), axis=1)
    umax_x = jnp.max(jnp.where(vis, ux, -BIG), axis=1)
    umin_y = jnp.min(jnp.where(vis, uy, BIG), axis=1)
    umax_y = jnp.max(jnp.where(vis, uy, -BIG), axis=1)

    # widest scale whose rotated bbox still fits the grid (eps: a single
    # visible keypoint has zero span and must not divide by zero)
    span_x = jnp.maximum(umax_x - umin_x, 1e-6)
    span_y = jnp.maximum(umax_y - umin_y, 1e-6)
    s_fit = jnp.where(any_vis, jnp.minimum(s, jnp.minimum(avail / span_x,
                                                          avail / span_y)), s)

    # feasible translation window, in heatmap units; non-empty by construction
    tx_fit = jnp.clip(tx * hm, lo - cen[:, 0] - s_fit * umin_x,
                      hi - cen[:, 0] - s_fit * umax_x) / hm
    ty_fit = jnp.clip(ty * hm, lo - cen[:, 1] - s_fit * umin_y,
                      hi - cen[:, 1] - s_fit * umax_y) / hm
    tx_fit = jnp.where(any_vis, tx_fit, tx)
    ty_fit = jnp.where(any_vis, ty_fit, ty)
    return theta, s_fit, tx_fit, ty_fit, cen


def affine_batch(key, img4_u8, kp_xy, vis, *, rot_deg, scale_min, scale_max,
                 translate_frac, heatmap_size, keep_kp_in_bounds=True,
                 bounds_margin=0.01, fit_center=None):
    """Per-sample random affine over a batch. Warps the image (mask channel
    re-binarized), transforms keypoints, and ANDs an in-bounds mask into vis.

    With ``keep_kp_in_bounds`` (the default) the sampled affine is first passed
    through :func:`fit_affine_to_bounds`, which re-centres it on the visible
    keypoints and shrinks translation (then, only if it must, scale) so that
    the AND has nothing left to drop. Rotation is never modified. Set it False
    for the pre-2026-09-02 behaviour, which threw away 17-26% of the annotated
    keypoints on wall-adjacent crops -- read that function's docstring first.
    """
    B, H, W = img4_u8.shape[0], img4_u8.shape[1], img4_u8.shape[2]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    theta = jnp.deg2rad(jax.random.uniform(k1, (B,), minval=-rot_deg, maxval=rot_deg))
    s = jax.random.uniform(k2, (B,), minval=scale_min, maxval=scale_max)
    tx = jax.random.uniform(k3, (B,), minval=-translate_frac, maxval=translate_frac)
    ty = jax.random.uniform(k4, (B,), minval=-translate_frac, maxval=translate_frac)

    if keep_kp_in_bounds:
        theta, s, tx, ty, cen = fit_affine_to_bounds(
            kp_xy, vis, theta, s, tx, ty, heatmap_size, bounds_margin, fit_center)
    else:
        cen = jnp.full((B, 2), heatmap_size / 2.0, jnp.float32)
    # same point, expressed in image pixels (the crop is 2x the heatmap)
    cimg_x = cen[:, 0] * (W / float(heatmap_size))
    cimg_y = cen[:, 1] * (H / float(heatmap_size))

    warped = jax.vmap(affine_warp_image)(img4_u8.astype(jnp.float32), theta, s,
                                         tx, ty, cimg_x, cimg_y)
    warped = warped.at[..., 3].set((warped[..., 3] > 0.5).astype(jnp.float32))
    img_out = jnp.clip(jnp.round(warped), 0, 255).astype(jnp.uint8)

    kp_out = jax.vmap(
        lambda k, th, ss, a, b, cx, cy: affine_transform_kp(
            k, th, ss, a, b, heatmap_size, cx, cy)
    )(kp_xy, theta, s, tx, ty, cen[:, 0], cen[:, 1])
    inb = ((kp_out[..., 0] >= 0) & (kp_out[..., 0] < heatmap_size)
           & (kp_out[..., 1] >= 0) & (kp_out[..., 1] < heatmap_size))
    return img_out, kp_out, (vis & inb)


def flip_batch(key, img4_u8, kp_xy, vis, lr_swap, flip_p, heatmap_size):
    """Per-sample horizontal flip with L/R keypoint-index swap."""
    B = img4_u8.shape[0]
    do = jax.random.bernoulli(key, p=flip_p, shape=(B,))
    img_f = img4_u8[:, :, ::-1, :]
    img_out = jnp.where(do[:, None, None, None], img_f, img4_u8)
    kp_sw = kp_xy[:, lr_swap, :]
    vis_sw = vis[:, lr_swap]
    kp_x = (heatmap_size - 1) - kp_sw[..., 0]
    kp_flip = jnp.stack([kp_x, kp_sw[..., 1]], axis=-1)
    kp_out = jnp.where(do[:, None, None], kp_flip, kp_xy)
    vis_out = jnp.where(do[:, None], vis_sw, vis)
    return img_out, kp_out, vis_out


def _mask_bbox_side(mask, axis_len_h, axis_len_w):
    """(B,) long side of the SAM silhouette's bbox, in pixels; 0 when empty."""
    rows = mask.any(axis=2)                                   # (B,H)
    cols = mask.any(axis=1)                                   # (B,W)
    ar = jnp.arange(axis_len_h)[None, :]
    ac = jnp.arange(axis_len_w)[None, :]
    span_y = (jnp.max(jnp.where(rows, ar, -1), axis=1)
              - jnp.min(jnp.where(rows, ar, axis_len_h), axis=1))
    span_x = (jnp.max(jnp.where(cols, ac, -1), axis=1)
              - jnp.min(jnp.where(cols, ac, axis_len_w), axis=1))
    return jnp.where(mask.any(axis=(1, 2)),
                     jnp.maximum(span_x, span_y).astype(jnp.float32), 0.0)


def _sample_mask_centre(key, mask, pool):
    """(cx, cy) drawn from the SAM silhouette in channel 3, jit-friendly.

    Mean-pools the mask to a (H/pool, W/pool) grid and draws one cell with
    ``jax.random.categorical`` on its log-weights, then jitters uniformly
    inside the cell. Categorical over ~3k cells is far cheaper than an
    inverse-CDF over 200k pixels, and `pool` (8 px) is an order of magnitude
    below the box side, so the quantisation is invisible. An empty mask gives
    uniform log-weights, i.e. it degrades to the uniform sampler.
    """
    B, H, W = mask.shape
    ph, pw = H // pool, W // pool
    pooled = mask[:, :ph * pool, :pw * pool].astype(jnp.float32)
    pooled = pooled.reshape(B, ph, pool, pw, pool).mean(axis=(2, 4))
    k1, k2 = jax.random.split(key)
    flat = jax.random.categorical(k1, jnp.log(pooled.reshape(B, -1) + 1e-6))
    u = jax.random.uniform(k2, (B, 2))
    cx = ((flat % pw).astype(jnp.float32) + u[:, 0]) * pool
    cy = ((flat // pw).astype(jnp.float32) + u[:, 1]) * pool
    return cx, cy


def cutout_batch(key, img4_u8, n, frac, *, mask_target_p=0.0, size_rel_fly=0.0,
                 mask_pool=8):
    """Zero up to n random square boxes (side ~frac*W) in RGB (channels 0-2).

    Defaults are an EXACT no-op change: uniform box centres, side = frac*W,
    and the same RNG stream as before ``mask_target_p``/``size_rel_fly``
    existed (both extra draws are taken only when their option is enabled).

    Why the options exist (measured 2026-09-02 over 60 real V5 train crops,
    ``figures/2026-09-02-vit-augmentations/augmentation_stats.json``): the fly
    silhouette covers only a few percent of its 448-px crop, so with uniform
    centres the great majority of the erased pixels land on empty arena floor.
    The augmentation costs its full compute and occludes the animal almost
    never.

      ``mask_target_p``  probability, per box, of drawing the centre from the
        SAM silhouette (channel 3) instead of uniformly. A MIXTURE on purpose:
        at 1.0 every sample would have an occluded fly, which replaces one
        bias with another. Note the mask is used at DATA-PREP time only -- it
        works just as well for the promoted mask-off checkpoint, whose
        channel-3 input weights are zeroed so the network cannot see it.
      ``size_rel_fly``   if > 0, box side = this fraction of the fly's
        mask-bbox long side instead of ``frac`` of the crop width, so the
        occlusion is the same fraction of the ANIMAL across cameras and
        distances (apparent fly size varies a lot between the 7 views).
        Falls back to ``frac * W`` for a crop with no mask.

    Note that cutout writes channels 0-2 only: the silhouette in channel 3 is
    NOT punched out, so for a mask-ON model an RGB-occluded limb is still
    visible in the mask channel and the occlusion is much weaker than it
    looks. That is moot for the promoted mask-off checkpoint.
    """
    B, H, W, _ = img4_u8.shape
    side_u = max(1, int(round(frac * W)))
    out = img4_u8
    xs = jnp.arange(W)[None, :]
    ys = jnp.arange(H)[None, :]
    fly_mask = img4_u8[..., 3] > 0
    if size_rel_fly > 0.0:
        fly = _mask_bbox_side(fly_mask, H, W)
        side = jnp.where(fly > 0, jnp.clip(size_rel_fly * fly, 4.0, float(W)),
                         float(side_u))                        # (B,)
    else:
        side = jnp.full((B,), float(side_u))
    half = jnp.floor(side / 2.0)[:, None]                      # matches side//2
    for j in range(int(n)):
        k1, k2 = jax.random.split(jax.random.fold_in(key, j))
        cx = jax.random.randint(k1, (B,), 0, W).astype(jnp.float32)
        cy = jax.random.randint(k2, (B,), 0, H).astype(jnp.float32)
        if mask_target_p > 0.0:
            km, kb = jax.random.split(jax.random.fold_in(key, 1000 + j))
            mcx, mcy = _sample_mask_centre(km, fly_mask, mask_pool)
            on = (jax.random.bernoulli(kb, p=mask_target_p, shape=(B,))
                  & fly_mask.any(axis=(1, 2)))
            cx = jnp.where(on, mcx, cx)
            cy = jnp.where(on, mcy, cy)
        x0 = cx[:, None] - half; x1 = cx[:, None] + half
        y0 = cy[:, None] - half; y1 = cy[:, None] + half
        mx = (xs >= x0) & (xs < x1)        # (B,W)
        my = (ys >= y0) & (ys < y1)        # (B,H)
        box = my[:, :, None] & mx[:, None, :]   # (B,H,W)
        keep = (~box)[..., None].astype(out.dtype)
        rgb = out[..., :3] * keep
        out = jnp.concatenate([rgb, out[..., 3:]], axis=-1)
    return out


def photometric_batch(key, img4_u8, brightness, contrast, gamma):
    """Per-sample brightness/contrast/gamma jitter on RGB (channels 0-2)."""
    B = img4_u8.shape[0]
    k1, k2, k3 = jax.random.split(key, 3)
    b = jax.random.uniform(k1, (B, 1, 1, 1), minval=-brightness, maxval=brightness)
    c = jax.random.uniform(k2, (B, 1, 1, 1), minval=1 - contrast, maxval=1 + contrast)
    g = jax.random.uniform(k3, (B, 1, 1, 1), minval=1 - gamma, maxval=1 + gamma)
    rgb = img4_u8[..., :3].astype(jnp.float32) / 255.0
    mean = rgb.mean(axis=(1, 2, 3), keepdims=True)
    rgb = (rgb - mean) * c + mean + b
    rgb = jnp.clip(rgb, 0.0, 1.0) ** g
    rgb = jnp.clip(jnp.round(rgb * 255.0), 0, 255).astype(img4_u8.dtype)
    return jnp.concatenate([rgb, img4_u8[..., 3:]], axis=-1)


def _gauss_kernel2d(ksize, sigma):
    ax = jnp.arange(ksize) - (ksize - 1) / 2.0
    g = jnp.exp(-(ax ** 2) / (2.0 * sigma ** 2)); g = g / g.sum()
    return jnp.outer(g, g)                                   # (k,k)


def gaussian_blur_batch(key, img4_u8, blur_max, ksize=5, sigma=1.0):
    """Per-sample variable-strength Gaussian blur on RGB (channels 0-2). A fixed
    (ksize,sigma) blur is lerped in by a per-sample amount alpha in [0, blur_max]
    (alpha=0 -> identity), so a single kernel gives a range of blur without a
    per-sample variable kernel. Mask channel untouched. Robustness to motion
    blur / defocus on fast courtship frames."""
    if blur_max <= 0.0:
        return img4_u8
    B = img4_u8.shape[0]
    rgb = img4_u8[..., :3].astype(jnp.float32)
    k2 = _gauss_kernel2d(ksize, sigma)
    ker = jnp.broadcast_to(k2[:, :, None, None], (ksize, ksize, 1, 3))
    blurred = jax.lax.conv_general_dilated(
        rgb, ker, window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=3)
    alpha = jax.random.uniform(key, (B, 1, 1, 1), minval=0.0, maxval=blur_max)
    out = (1.0 - alpha) * rgb + alpha * blurred
    rgb_u8 = jnp.clip(jnp.round(out), 0, 255).astype(img4_u8.dtype)
    return jnp.concatenate([rgb_u8, img4_u8[..., 3:]], axis=-1)


def gaussian_noise_batch(key, img4_u8, noise_scale):
    """Per-sample additive Gaussian sensor noise on RGB (channels 0-2). Per-sample
    std in [0, noise_scale] (in [0,1] image units; matches JARVIS scale ~0.02).
    Mask channel untouched."""
    if noise_scale <= 0.0:
        return img4_u8
    B = img4_u8.shape[0]
    k1, k2 = jax.random.split(key)
    scale = jax.random.uniform(k1, (B, 1, 1, 1), minval=0.0, maxval=noise_scale)
    rgb = img4_u8[..., :3].astype(jnp.float32) / 255.0
    rgb = jnp.clip(rgb + jax.random.normal(k2, rgb.shape) * scale, 0.0, 1.0)
    rgb_u8 = jnp.clip(jnp.round(rgb * 255.0), 0, 255).astype(img4_u8.dtype)
    return jnp.concatenate([rgb_u8, img4_u8[..., 3:]], axis=-1)


def per_channel_multiply_batch(key, img4_u8, pc_color):
    """Per-sample per-channel colour multiply on RGB (channels 0-2): each channel
    scaled by an independent factor in [1-pc_color, 1+pc_color] (JARVIS
    PER_CHANNEL_MULTIPLY, [0.8,1.2] -> pc_color=0.2). Mask channel untouched."""
    if pc_color <= 0.0:
        return img4_u8
    B = img4_u8.shape[0]
    f = jax.random.uniform(key, (B, 1, 1, 3), minval=1.0 - pc_color, maxval=1.0 + pc_color)
    rgb = img4_u8[..., :3].astype(jnp.float32) * f
    rgb_u8 = jnp.clip(jnp.round(rgb), 0, 255).astype(img4_u8.dtype)
    return jnp.concatenate([rgb_u8, img4_u8[..., 3:]], axis=-1)


@dataclasses.dataclass(frozen=True)
class AugParams:
    enabled: bool = True
    rot_deg: float = 30.0
    scale_min: float = 0.8
    scale_max: float = 1.25
    translate_frac: float = 0.1
    flip_p: float = 0.5
    # Constrain the sampled affine so it cannot push an annotated keypoint out
    # of the crop (see fit_affine_to_bounds). Default True as of 2026-09-02:
    # False is the old behaviour and loses 17-26% of the labels on
    # wall-adjacent crops, almost all of it on the tarsal tips.
    keep_kp_in_bounds: bool = True
    cutout_n: int = 2
    cutout_frac: float = 0.25
    # Targeted cutout (see cutout_batch). Both 0.0 = the historical uniform
    # box, bit-for-bit; PROPOSED, not yet A/B-trained as of 2026-09-02.
    cutout_mask_target_p: float = 0.0
    cutout_size_rel_fly: float = 0.0
    brightness: float = 0.2
    contrast: float = 0.2
    gamma: float = 0.2
    # colour/noise robustness (match JARVIS-HybridNet's blur/noise/per-channel set)
    blur_max: float = 0.5      # max lerp toward a Gaussian-blurred copy (0 disables)
    noise_scale: float = 0.02  # max additive-noise std in [0,1] image units (0 disables)
    pc_color: float = 0.2      # per-channel multiply half-range (0 disables)


def augment_batch(key, img4_u8, kp_xy, vis, params, lr_swap, heatmap_size=224):
    """Apply the full augmentation pipeline to a batch. Identity when
    params.enabled is False. Order: affine -> flip -> cutout -> photometric."""
    if not params.enabled:
        return img4_u8, kp_xy, vis
    kg, kf, kc, kp_, kb, kn, kpc = jax.random.split(key, 7)
    img, kp, vis = affine_batch(
        kg, img4_u8, kp_xy, vis, rot_deg=params.rot_deg,
        scale_min=params.scale_min, scale_max=params.scale_max,
        translate_frac=params.translate_frac, heatmap_size=heatmap_size,
        keep_kp_in_bounds=params.keep_kp_in_bounds)
    img, kp, vis = flip_batch(kf, img, kp, vis, lr_swap, params.flip_p, heatmap_size)
    img = cutout_batch(kc, img, params.cutout_n, params.cutout_frac,
                       mask_target_p=params.cutout_mask_target_p,
                       size_rel_fly=params.cutout_size_rel_fly)
    img = photometric_batch(kp_, img, params.brightness, params.contrast, params.gamma)
    img = gaussian_blur_batch(kb, img, params.blur_max)
    img = gaussian_noise_batch(kn, img, params.noise_scale)
    img = per_channel_multiply_batch(kpc, img, params.pc_color)
    return img, kp, vis
