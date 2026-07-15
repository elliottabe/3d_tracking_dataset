"""Heatmap decode variants for the 2D-noise diagnostic and the sharpened
production decode. Pure functions over (B,H,W,K) heatmap arrays.

- soft_centroid : windowed mass-centroid (sharpen=1.0 == eval.mpjpe.heatmaps_to_keypoints)
- hard_argmax   : quantized peak pixel (EfficientTrack-style diagnostic baseline)
- gaussian      : log-parabolic subpixel through the 3 pixels at the peak
"""
import numpy as np
import jax
import jax.numpy as jnp


def decode_soft_centroid(hm, *, in_size=448, radius=7, sharpen=1.0):
    """(B,H,W,K) -> (B,K,2) windowed mass-centroid in in_size px. `sharpen`
    raises the windowed positive mass to a power before the centroid, pulling
    the estimate toward the peak (reduces diffuse-tail sensitivity)."""
    b, h, w, k = hm.shape
    if h != w:
        raise ValueError(f"expected square heatmaps, got {h}x{w}")
    p = jax.nn.relu(hm)
    idx = jnp.argmax(p.reshape(b, h * w, k), axis=1)          # (B,K)
    py = (idx // w)[:, None, None, :]
    px = (idx % w)[:, None, None, :]
    ys = jnp.arange(h)[None, :, None, None]
    xs = jnp.arange(w)[None, None, :, None]
    win = ((jnp.abs(ys - py) <= radius) & (jnp.abs(xs - px) <= radius)).astype(p.dtype)
    pw = (p * win) ** sharpen                                  # 0 outside window
    z = pw.sum(axis=(1, 2)) + 1e-8
    gx = jnp.arange(w, dtype=hm.dtype)[None, None, :, None]
    gy = jnp.arange(h, dtype=hm.dtype)[None, :, None, None]
    x = (pw * gx).sum(axis=(1, 2)) / z
    y = (pw * gy).sum(axis=(1, 2)) / z
    scale = in_size / float(h)
    return jnp.stack([x * scale, y * scale], axis=-1)          # (B,K,2)


def decode_hard_argmax(hm, *, in_size=448):
    """(B,H,W,K) -> (B,K,2) hard argmax (quantized), EfficientTrack-style."""
    b, h, w, k = hm.shape
    p = jax.nn.relu(hm)
    idx = jnp.argmax(p.reshape(b, h * w, k), axis=1)           # (B,K)
    py = (idx // w).astype(hm.dtype)
    px = (idx % w).astype(hm.dtype)
    scale = in_size / float(h)
    return jnp.stack([px * scale, py * scale], axis=-1)


def decode_gaussian(hm, *, in_size=448):
    """(B,H,W,K) -> (B,K,2) log-parabolic (Gaussian) subpixel about the peak
    (numpy). A parabola fit to log-intensity == a Gaussian fit; robust to the
    asymmetric diffuse tails a windowed centroid drifts with."""
    hm = np.asarray(hm)
    b, h, w, k = hm.shape
    p = np.maximum(hm, 0.0)
    idx = p.reshape(b, h * w, k).argmax(axis=1)                # (B,K)
    py = idx // w; px = idx % w
    eps = 1e-6
    bi = np.arange(b)[:, None]; ki = np.arange(k)[None, :]
    pxm = np.clip(px - 1, 0, w - 1); pxp = np.clip(px + 1, 0, w - 1)
    pym = np.clip(py - 1, 0, h - 1); pyp = np.clip(py + 1, 0, h - 1)

    def _off(vm1, v0, vp1):
        a = np.log(vm1 + eps); c0 = np.log(v0 + eps); c = np.log(vp1 + eps)
        denom = a - 2 * c0 + c
        off = np.where(np.abs(denom) > eps, 0.5 * (a - c) / denom, 0.0)
        return np.clip(off, -1.0, 1.0)

    dx = _off(p[bi, py, pxm, ki], p[bi, py, px, ki], p[bi, py, pxp, ki])
    dy = _off(p[bi, pym, px, ki], p[bi, py, px, ki], p[bi, pyp, px, ki])
    scale = in_size / float(h)
    out = np.zeros((b, k, 2), np.float64)
    out[..., 0] = (px + dx) * scale
    out[..., 1] = (py + dy) * scale
    return out


def decode_heatmaps(hm, *, method="soft_centroid", in_size=448, radius=7, sharpen=1.0):
    """Dispatch to a decoder by name; always returns a numpy (B,K,2)."""
    if method == "soft_centroid":
        return np.asarray(decode_soft_centroid(hm, in_size=in_size, radius=radius, sharpen=sharpen))
    if method == "hard_argmax":
        return np.asarray(decode_hard_argmax(hm, in_size=in_size))
    if method == "gaussian":
        return decode_gaussian(hm, in_size=in_size)
    raise ValueError(f"unknown decode method {method!r}")
