"""Keypoint training losses (JAX)."""
import jax
import jax.numpy as jnp


def heatmap_mse(pred, gt, vis, bg_weight=0.1, fg_thresh=0.01, eps=1e-6):
    """Foreground-weighted heatmap MSE. pred,gt: (B,H,W,K); vis: (B,K) bool.

    Plain mean-MSE over all H*W pixels dilutes the sparse keypoint-foreground
    gradient ~50000x, so the model collapses to predicting all-zeros and the
    peak never forms. Instead average the squared error separately over
    foreground (gt > fg_thresh) and background pixels per keypoint, then combine
    as mse_fg + bg_weight * mse_bg. This gives the peak a full-strength gradient.
    Averaged over visible keypoints only.
    """
    d = (pred - gt) ** 2
    fg = (gt > fg_thresh).astype(pred.dtype)
    nfg = fg.sum(axis=(1, 2)) + eps
    nbg = (1.0 - fg).sum(axis=(1, 2)) + eps
    per_kp = (d * fg).sum(axis=(1, 2)) / nfg + bg_weight * (
        (d * (1.0 - fg)).sum(axis=(1, 2)) / nbg)          # (B,K)
    w = vis.astype(pred.dtype)
    return (per_kp * w).sum() / jnp.maximum(w.sum(), 1.0)


def mask_containment(pred, mask224, eps=1e-6):
    """Fraction of positive predicted heatmap mass outside the mask.

    pred: (B,H,W,K); mask224: (B,H,W) float {0,1}. Returns a scalar in [0,1].
    """
    p = jax.nn.relu(pred)                                 # positive mass
    outside = p * (1.0 - mask224[..., None])
    return outside.sum() / (p.sum() + eps)
