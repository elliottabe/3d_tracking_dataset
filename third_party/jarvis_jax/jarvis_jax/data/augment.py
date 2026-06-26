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


def affine_warp_image(img, theta, s, tx, ty):
    """Inverse-sample warp of (H,W,C) float32 about its center. Output =
    center + s*R(theta)*(src-center) + (tx,ty)*W  (so src is solved inversely).
    Bilinear, out-of-bounds -> 0."""
    H, W, C = img.shape
    cx = W / 2.0
    cy = H / 2.0
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


def affine_transform_kp(kp, theta, s, tx, ty, heatmap_size):
    """Forward affine on (K,2) keypoints (x,y) in heatmap coords."""
    c = heatmap_size / 2.0
    R = _rot(theta)
    rel = kp - jnp.array([c, c])
    nx = R[0, 0] * rel[:, 0] + R[0, 1] * rel[:, 1]
    ny = R[1, 0] * rel[:, 0] + R[1, 1] * rel[:, 1]
    out_x = c + s * nx + tx * heatmap_size
    out_y = c + s * ny + ty * heatmap_size
    return jnp.stack([out_x, out_y], axis=-1)


def affine_batch(key, img4_u8, kp_xy, vis, *, rot_deg, scale_min, scale_max,
                 translate_frac, heatmap_size):
    """Per-sample random affine over a batch. Warps the image (mask channel
    re-binarized), transforms keypoints, and ANDs an in-bounds mask into vis."""
    B = img4_u8.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    theta = jnp.deg2rad(jax.random.uniform(k1, (B,), minval=-rot_deg, maxval=rot_deg))
    s = jax.random.uniform(k2, (B,), minval=scale_min, maxval=scale_max)
    tx = jax.random.uniform(k3, (B,), minval=-translate_frac, maxval=translate_frac)
    ty = jax.random.uniform(k4, (B,), minval=-translate_frac, maxval=translate_frac)

    warped = jax.vmap(affine_warp_image)(img4_u8.astype(jnp.float32), theta, s, tx, ty)
    warped = warped.at[..., 3].set((warped[..., 3] > 0.5).astype(jnp.float32))
    img_out = jnp.clip(jnp.round(warped), 0, 255).astype(jnp.uint8)

    kp_out = jax.vmap(
        lambda k, th, ss, a, b: affine_transform_kp(k, th, ss, a, b, heatmap_size)
    )(kp_xy, theta, s, tx, ty)
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


def cutout_batch(key, img4_u8, n, frac):
    """Zero up to n random square boxes (side ~frac*W) in RGB (channels 0-2)."""
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


@dataclasses.dataclass(frozen=True)
class AugParams:
    enabled: bool = True
    rot_deg: float = 30.0
    scale_min: float = 0.8
    scale_max: float = 1.25
    translate_frac: float = 0.1
    flip_p: float = 0.5
    cutout_n: int = 2
    cutout_frac: float = 0.25
    brightness: float = 0.2
    contrast: float = 0.2
    gamma: float = 0.2


def augment_batch(key, img4_u8, kp_xy, vis, params, lr_swap, heatmap_size=224):
    """Apply the full augmentation pipeline to a batch. Identity when
    params.enabled is False. Order: affine -> flip -> cutout -> photometric."""
    if not params.enabled:
        return img4_u8, kp_xy, vis
    kg, kf, kc, kp_ = jax.random.split(key, 4)
    img, kp, vis = affine_batch(
        kg, img4_u8, kp_xy, vis, rot_deg=params.rot_deg,
        scale_min=params.scale_min, scale_max=params.scale_max,
        translate_frac=params.translate_frac, heatmap_size=heatmap_size)
    img, kp, vis = flip_batch(kf, img, kp, vis, lr_swap, params.flip_p, heatmap_size)
    img = cutout_batch(kc, img, params.cutout_n, params.cutout_frac)
    img = photometric_batch(kp_, img, params.brightness, params.contrast, params.gamma)
    return img, kp, vis
