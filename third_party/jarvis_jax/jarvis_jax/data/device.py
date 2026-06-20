"""On-device (jitted) preprocessing: image normalization + Gaussian heatmap
rendering. Moves the heavy per-batch work off the CPU loader and avoids
transferring dense heatmaps over PCIe. The Gaussian renderer is a JAX port of
jarvis_jax.data.transforms.gaussian_heatmaps (same sigma)."""
import jax.numpy as jnp

IMAGENET_MEAN_J = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
IMAGENET_STD_J = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)


def normalize_image(img4_u8):
    """(...,448,448,4) uint8 -> float32. RGB ImageNet-normalized; mask kept 0/1."""
    img = img4_u8.astype(jnp.float32)
    rgb = (img[..., :3] / 255.0 - IMAGENET_MEAN_J) / IMAGENET_STD_J
    mask = img[..., 3:4]
    return jnp.concatenate([rgb, mask], axis=-1)


def render_heatmaps(kp_xy, vis, heatmap_size=224, sigma=7.0):
    """(B,K,2) heatmap-coord keypoints + (B,K) vis -> (B,H,W,K) Gaussians."""
    ys = jnp.arange(heatmap_size, dtype=jnp.float32)
    xs = jnp.arange(heatmap_size, dtype=jnp.float32)
    gy, gx = jnp.meshgrid(ys, xs, indexing="ij")          # (H,W)
    gx = gx[None, :, :, None]                              # (1,H,W,1)
    gy = gy[None, :, :, None]
    cx = kp_xy[:, None, None, :, 0]                        # (B,1,1,K)
    cy = kp_xy[:, None, None, :, 1]
    hm = jnp.exp(-(((gx - cx) ** 2) + ((gy - cy) ** 2)) / (2.0 * sigma * sigma))
    return hm * vis[:, None, None, :].astype(jnp.float32)
