"""Keypoint decoding (argmax + windowed-centroid) and MPJPE."""
import jax
import jax.numpy as jnp


def heatmaps_to_keypoints(hm, *, in_size=448, radius=7, sharpen=1.0):
    """(B,H,W,K) heatmaps -> (B,K,2) keypoints in `in_size` pixel coords.

    Argmax locates each keypoint's peak, then a local mass-centroid within a
    +/-radius window refines it to sub-pixel. This is robust to diffuse positive
    background (a global centroid would be swamped by it and collapse to image
    center).
    """
    b, h, w, k = hm.shape
    if h != w:
        raise ValueError(
            f"heatmaps_to_keypoints expects square heatmaps, got {h}x{w}")
    p = jax.nn.relu(hm)
    idx = jnp.argmax(p.reshape(b, h * w, k), axis=1)       # (B,K)
    py = (idx // w)[:, None, None, :]                       # (B,1,1,K)
    px = (idx % w)[:, None, None, :]
    ys = jnp.arange(h)[None, :, None, None]
    xs = jnp.arange(w)[None, None, :, None]
    win = ((jnp.abs(ys - py) <= radius) & (jnp.abs(xs - px) <= radius)).astype(p.dtype)
    pw = (p * win) ** sharpen        # sharpen>1 concentrates mass near the peak,
    z = pw.sum(axis=(1, 2)) + 1e-8   # reducing diffuse-tail drift; 1.0 == original
    gx = jnp.arange(w, dtype=hm.dtype)[None, None, :, None]
    gy = jnp.arange(h, dtype=hm.dtype)[None, :, None, None]
    x = (pw * gx).sum(axis=(1, 2)) / z
    y = (pw * gy).sum(axis=(1, 2)) / z
    scale = in_size / float(h)
    return jnp.stack([x * scale, y * scale], axis=-1)       # (B,K,2)


def mpjpe(pred_kp, gt_kp, vis):
    """Mean per-joint position error over visible keypoints (pixels)."""
    d = jnp.linalg.norm(pred_kp - gt_kp, axis=-1)          # (B,K)
    w = vis.astype(d.dtype)
    return (d * w).sum() / jnp.maximum(w.sum(), 1.0)
