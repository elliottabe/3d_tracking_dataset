"""Keypoint decoding (normalized-mass centroid) and MPJPE."""
import jax
import jax.numpy as jnp


def heatmaps_to_keypoints(hm, *, in_size=448):
    """(B,H,W,K) heatmaps -> (B,K,2) keypoints in `in_size` pixel coords."""
    b, h, w, k = hm.shape
    if h != w:
        raise ValueError(f"heatmaps_to_keypoints expects square heatmaps, got {h}x{w}")
    p = jax.nn.relu(hm)
    z = p.sum(axis=(1, 2), keepdims=True) + 1e-8          # (B,1,1,K)
    p = p / z
    ys = jnp.arange(h, dtype=hm.dtype)
    xs = jnp.arange(w, dtype=hm.dtype)
    gy, gx = jnp.meshgrid(ys, xs, indexing="ij")          # (H,W)
    x = (p * gx[None, :, :, None]).sum(axis=(1, 2))        # (B,K)
    y = (p * gy[None, :, :, None]).sum(axis=(1, 2))        # (B,K)
    scale = in_size / float(h)
    return jnp.stack([x * scale, y * scale], axis=-1)      # (B,K,2)


def mpjpe(pred_kp, gt_kp, vis):
    """Mean per-joint position error over visible keypoints (pixels)."""
    d = jnp.linalg.norm(pred_kp - gt_kp, axis=-1)          # (B,K)
    w = vis.astype(d.dtype)
    return (d * w).sum() / jnp.maximum(w.sum(), 1.0)
