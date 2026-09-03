"""Multi-view augmentation for mvq windows. Every geometric op updates the
affine cameras so GT 3D still reprojects onto GT 2D exactly (tested)."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jarvis_jax.data.augment import (photometric_batch, gaussian_blur_batch,
                                     gaussian_noise_batch, per_channel_multiply_batch)
from jarvis_jax.models.mvq.geometry import rotate_world, mirror_world

CROP = 448


@dataclasses.dataclass
class MVAugParams:
    enabled: bool = True
    rot_deg: float = 30.0
    scale_min: float = 0.8
    scale_max: float = 1.25
    translate_frac: float = 0.1
    world_yaw: bool = True
    world_tilt_deg: float = 30.0
    mirror_p: float = 0.5
    cam_drop_p: float = 0.3
    cam_drop_max: int = 2
    brightness: float = 0.2
    contrast: float = 0.2
    gamma: float = 0.2
    blur_max: float = 0.5
    noise_scale: float = 0.02
    pc_color: float = 0.2


def _forward_affine(theta, s, tx, ty, c=(CROP - 1) / 2.0):
    """Forward map p' = A p + b for a rotation by theta and scale s about the
    crop centre followed by a translation (tx, ty) in px."""
    ct, st = jnp.cos(theta), jnp.sin(theta)
    A = s * jnp.array([[ct, -st], [st, ct]])
    b = jnp.array([c, c]) - A @ jnp.array([c, c]) + jnp.array([tx, ty])
    return A, b


def _warp_rgb(img, A, b):
    """Warp (H,W,3) uint8 by the FORWARD map (A, b): output pixel q samples input A^-1 (q - b)."""
    H, W = img.shape[:2]
    Ai = jnp.linalg.inv(A)
    ys, xs = jnp.meshgrid(jnp.arange(H, dtype=jnp.float32), jnp.arange(W, dtype=jnp.float32), indexing="ij")
    q = jnp.stack([xs, ys], -1) - b
    src = q @ Ai.T                                                    # (H,W,2) x,y
    coords = [src[..., 1], src[..., 0]]
    out = jnp.stack([jax.scipy.ndimage.map_coordinates(img[..., ch].astype(jnp.float32), coords,
                                                       order=1, mode="constant", cval=0.0)
                     for ch in range(img.shape[-1])], -1)
    return jnp.clip(jnp.round(out), 0, 255).astype(jnp.uint8)


def _per_view_affine(key, b, p):
    B, T, C = b["crops"].shape[:3]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    theta = jnp.deg2rad(jax.random.uniform(k1, (B, C), minval=-p.rot_deg, maxval=p.rot_deg))
    s = jax.random.uniform(k2, (B, C), minval=p.scale_min, maxval=p.scale_max)
    tx = jax.random.uniform(k3, (B, C), minval=-p.translate_frac, maxval=p.translate_frac) * CROP
    ty = jax.random.uniform(k4, (B, C), minval=-p.translate_frac, maxval=p.translate_frac) * CROP
    A, bb = jax.vmap(jax.vmap(_forward_affine))(theta, s, tx, ty)      # (B,C,2,2), (B,C,2)
    warp = jax.vmap(jax.vmap(jax.vmap(_warp_rgb, in_axes=(0, 0, 0)), in_axes=(0, None, None)))
    crops = warp(b["crops"], A, bb)                                    # over B, T, C
    pm = jax.vmap(jax.vmap(jax.vmap(lambda m, A_, b_: _warp_rgb(m[..., None].astype(jnp.uint8) * 255, A_, b_)[..., 0] > 127,
                                    in_axes=(0, 0, 0)), in_axes=(0, None, None)))(b["prompt_mask"], A, bb)
    # warp_cameras(M, t, A, b) per sample; t_local is per (sample, frame), so write it out:
    M = jnp.einsum("bcij,bcjk->bcik", A, b["M"])
    tl = jnp.einsum("bcij,btcj->btci", A, b["t_local"]) + bb[:, None, :, :]
    kp = jnp.einsum("bcij,bftckj->bftcki", A, b["kp2d"]) + bb[:, None, None, :, None, :]
    inb = ((kp >= 0) & (kp <= CROP - 1)).all(-1)
    return {**b, "crops": crops, "prompt_mask": pm, "M": M, "t_local": tl, "kp2d": kp,
            "vis2d": b["vis2d"] & inb}


def _rot_mats(key, B, p):
    ky, kt = jax.random.split(key)
    yaw = jax.random.uniform(ky, (B,), minval=0.0, maxval=2 * jnp.pi) if p.world_yaw else jnp.zeros((B,))
    tilt = jnp.deg2rad(jax.random.uniform(kt, (B,), minval=-p.world_tilt_deg, maxval=p.world_tilt_deg))
    cz, sz, cx, sx = jnp.cos(yaw), jnp.sin(yaw), jnp.cos(tilt), jnp.sin(tilt)
    Rz = jnp.stack([jnp.stack([cz, -sz, 0 * cz], -1), jnp.stack([sz, cz, 0 * cz], -1),
                    jnp.stack([0 * cz, 0 * cz, 1 + 0 * cz], -1)], -2)
    Rx = jnp.stack([jnp.stack([1 + 0 * cx, 0 * cx, 0 * cx], -1), jnp.stack([0 * cx, cx, -sx], -1),
                    jnp.stack([0 * cx, sx, cx], -1)], -2)
    return Rz @ Rx                                                         # (B,3,3)


def _world_rotation(key, b, p):
    R = _rot_mats(key, b["crops"].shape[0], p)
    M = jax.vmap(rotate_world)(b["M"], R)
    X = jnp.einsum("bij,bftkj->bftki", R, b["kp3d_local"])
    return {**b, "M": M, "kp3d_local": X}


def _mirror(key, b, p, lr_swap):
    B = b["crops"].shape[0]
    do = jax.random.bernoulli(key, p.mirror_p, (B,))
    M2, tl2 = jax.vmap(lambda M_, t_: mirror_world(M_, t_, CROP), in_axes=(0, 0))(
        b["M"], b["t_local"][:, 0])
    # t_local is identical across frames only if origins are shared per window (they are);
    # apply the same b-shift to every frame:
    tl2 = jnp.broadcast_to(tl2[:, None], b["t_local"].shape)
    sel = lambda a, m: jnp.where(do.reshape((B,) + (1,) * (a.ndim - 1)), m, a)
    X = b["kp3d_local"][..., lr_swap, :] * jnp.array([-1.0, 1.0, 1.0])
    kp = b["kp2d"][..., lr_swap, :].at[..., 0].set(CROP - 1 - b["kp2d"][..., lr_swap, 0])
    return {**b,
            "crops": sel(b["crops"], b["crops"][..., ::-1, :]),
            "prompt_mask": sel(b["prompt_mask"], b["prompt_mask"][..., ::-1]),
            "M": sel(b["M"], M2), "t_local": sel(b["t_local"], tl2),
            "kp3d_local": sel(b["kp3d_local"], X), "has3d": sel(b["has3d"], b["has3d"][..., lr_swap]),
            "kp2d": sel(b["kp2d"], kp), "vis2d": sel(b["vis2d"], b["vis2d"][..., lr_swap])}


def _camera_dropout(key, b, p):
    B, T, C = b["cam_valid"].shape
    k1, k2, k3 = jax.random.split(key, 3)
    do = jax.random.bernoulli(k1, p.cam_drop_p, (B,))
    n_drop = jax.random.randint(k2, (B,), 1, p.cam_drop_max + 1)
    score = jax.random.uniform(k3, (B, C)) + (~b["cam_valid"][:, 0]).astype(jnp.float32)   # invalid sort last
    order = jnp.argsort(score, axis=1)
    rank = jnp.argsort(order, axis=1)                                     # rank of each cam
    n_valid = b["cam_valid"][:, 0].sum(1)
    n_drop = jnp.minimum(n_drop, jnp.maximum(n_valid - 3, 0))
    drop = (rank < n_drop[:, None]) & do[:, None]                          # (B,C)
    cv = b["cam_valid"] & ~drop[:, None, :]
    return {**b, "cam_valid": cv, "vis2d": b["vis2d"] & cv[:, None, :, :, None]}


def _photometric(key, b, p):
    B, T, C = b["crops"].shape[:3]
    flat = b["crops"].reshape((B * T * C,) + b["crops"].shape[3:])
    flat4 = jnp.concatenate([flat, jnp.zeros(flat.shape[:-1] + (1,), flat.dtype)], -1)   # reuse 4-ch helpers
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x = photometric_batch(k1, flat4, p.brightness, p.contrast, p.gamma)
    x = gaussian_blur_batch(k2, x, p.blur_max)
    x = gaussian_noise_batch(k3, x, p.noise_scale)
    x = per_channel_multiply_batch(k4, x, p.pc_color)
    return {**b, "crops": x[..., :3].reshape(b["crops"].shape)}


def augment_window(key, b, params: MVAugParams, lr_swap):
    if not params.enabled:
        return b
    lr_swap = jnp.asarray(lr_swap)
    ka, kr, km, kd, kp = jax.random.split(key, 5)
    if params.rot_deg > 0 or params.scale_min != 1 or params.scale_max != 1 or params.translate_frac > 0:
        b = _per_view_affine(ka, b, params)
    if params.world_yaw or params.world_tilt_deg > 0:
        b = _world_rotation(kr, b, params)
    if params.mirror_p > 0:
        b = _mirror(km, b, params, lr_swap)
    if params.cam_drop_p > 0:
        b = _camera_dropout(kd, b, params)
    if any(v > 0 for v in (params.brightness, params.contrast, params.gamma, params.blur_max,
                           params.noise_scale, params.pc_color)):
        b = _photometric(kp, b, params)
    return b
