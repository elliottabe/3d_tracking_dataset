"""Keypoint training losses (JAX)."""
import jax
import jax.numpy as jnp


def heatmap_mse(pred, gt, vis):
    """Masked heatmap MSE. pred,gt: (B,H,W,K); vis: (B,K) bool.

    Mean over spatial dims per keypoint, then mean over visible keypoints.
    """
    per_kp = jnp.mean((pred - gt) ** 2, axis=(1, 2))     # (B,K)
    w = vis.astype(pred.dtype)                            # (B,K)
    denom = jnp.maximum(w.sum(), 1.0)
    return (per_kp * w).sum() / denom


def mask_containment(pred, mask224, eps=1e-6):
    """Fraction of positive predicted heatmap mass outside the mask.

    pred: (B,H,W,K); mask224: (B,H,W) float {0,1}. Returns a scalar in [0,1].
    """
    p = jax.nn.relu(pred)                                 # positive mass
    outside = p * (1.0 - mask224[..., None])
    return outside.sum() / (p.sum() + eps)
