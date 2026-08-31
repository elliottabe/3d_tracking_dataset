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


def normalize_image_center_channel(img4_u8):
    """(...,448,448,4) uint8 -> float32, for the ``center_channel`` instance-cue
    ablation arm (see ``jarvis_jax/data/center_channel.py::CenterChannelDataset``).

    Differs from ``normalize_image`` ONLY in how the 4th channel is scaled.
    The SAM-mask channel that `normalize_image` serves is stored as literal
    0/1 uint8 values (``_load_mask(...).astype(np.uint8)`` in
    ``data/v3.py``/``data/v5_2d.py``) -- passed through unchanged is already
    the right {0,1} float. `CenterChannelDataset` instead stores a continuous
    Gaussian in [0,1] SCALED to the uint8 range (``round(g*255)``), the only
    way to keep the blob's spatial width intact through a uint8 buffer (a
    literal-0/1 encoding of a continuous Gaussian would truncate every
    non-peak pixel to 0, collapsing the blob to one hot pixel and destroying
    the whole point of a WIDE instance cue) -- so it must be divided back by
    255 here to land back in [0,1]. A separate function rather than a branch
    inside `normalize_image`: this keeps the existing (already-running
    mask-ablation) training arms' numerics byte-identical, and makes the
    encoding difference explicit at every call site instead of an implicit
    per-batch flag.
    """
    img = img4_u8.astype(jnp.float32)
    rgb = (img[..., :3] / 255.0 - IMAGENET_MEAN_J) / IMAGENET_STD_J
    center = img[..., 3:4] / 255.0
    return jnp.concatenate([rgb, center], axis=-1)


def render_heatmaps(kp_xy, vis, heatmap_size=224, sigma=7.0):
    """(B,K,2) heatmap-coord keypoints + (B,K) vis -> (B,H,W,K) Gaussians.

    ``sigma`` may be a scalar (all channels) or a per-channel ``(K,)`` array — the
    latter lets densely-packed channels (e.g. wing vertices) use a tighter Gaussian
    so neighbouring peaks stay resolvable instead of merging into one blob.
    """
    ys = jnp.arange(heatmap_size, dtype=jnp.float32)
    xs = jnp.arange(heatmap_size, dtype=jnp.float32)
    gy, gx = jnp.meshgrid(ys, xs, indexing="ij")          # (H,W)
    gx = gx[None, :, :, None]                              # (1,H,W,1)
    gy = gy[None, :, :, None]
    cx = kp_xy[:, None, None, :, 0]                        # (B,1,1,K)
    cy = kp_xy[:, None, None, :, 1]
    s = jnp.asarray(sigma, dtype=jnp.float32)
    if s.ndim == 1:                                        # per-channel (K,) -> (1,1,1,K)
        s = s[None, None, None, :]
    hm = jnp.exp(-(((gx - cx) ** 2) + ((gy - cy) ** 2)) / (2.0 * s * s))
    return hm * vis[:, None, None, :].astype(jnp.float32)
