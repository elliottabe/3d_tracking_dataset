"""3D MPJPE evaluation metric (JAX)."""
from __future__ import annotations

import jax.numpy as jnp


def mpjpe_3d(
    pred: jnp.ndarray,   # (B, K, 3)
    gt: jnp.ndarray,     # (B, K, 3)
    valid: jnp.ndarray,  # (B, K) bool
) -> jnp.ndarray:
    """Mean per-joint position error over valid joints (world units).

    Args:
        pred:  ``(B, K, 3)`` predicted 3D keypoints (world coords).
        gt:    ``(B, K, 3)`` ground-truth 3D keypoints (world coords).
        valid: ``(B, K)`` bool mask of valid joints.

    Returns:
        Scalar mean L2 error over valid joints.
    """
    d = jnp.linalg.norm(pred - gt, axis=-1)   # (B, K)
    w = valid.astype(d.dtype)
    return (d * w).sum() / jnp.maximum(w.sum(), 1.0)
