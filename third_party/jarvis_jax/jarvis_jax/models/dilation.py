"""Binary/float mask dilation via a single reduce_window max-pool (JAX, jit-safe).

Mirrors the PyTorch mask-gating reference
    F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
(jarvis/efficienttrack/loss.py:25-59, jarvis/prediction/sam3_masker.py:25-38) with
no scipy and no Python loop in the traced step, so it runs inside nnx.jit for the
train-time containment loss (Task 2) and the inference gate (Task 3). `k` is the
static square structuring-element side; k <= 1 is the identity.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def dilate_mask_jax(mask, k):
    """Dilate a {0,1}/float mask by a k x k square element (max-pool, stride 1).

    mask: (H,W) | (B,H,W) | (B,H,W,C). Returns the SAME shape/dtype. k is a static
    Python int; k <= 1 (or 0) returns `mask` unchanged (identity). Padding='SAME'
    (window centered for odd k, matching PyTorch padding=k//2).
    """
    k = int(k)
    if k <= 1:
        return mask
    x = jnp.asarray(mask)
    win = (k, k)
    strides = (1, 1)
    # reduce_window needs a window entry per axis; pad the spatial window with 1s
    # on the leading/trailing non-spatial axes so B and C are untouched.
    if x.ndim == 2:                       # (H,W)
        wd, ws, pad = win, strides, "SAME"
    elif x.ndim == 3:                     # (B,H,W) or (H,W,C) -> treat leading as batch-like
        wd, ws, pad = (1, *win), (1, *strides), "SAME"
    elif x.ndim == 4:                     # (B,H,W,C)
        wd, ws, pad = (1, *win, 1), (1, *strides, 1), "SAME"
    else:
        raise ValueError(f"dilate_mask_jax expects 2/3/4-D mask, got shape {x.shape}")
    neg_inf = jnp.array(-jnp.inf, dtype=x.dtype)
    out = jax.lax.reduce_window(x, neg_inf, jax.lax.max, wd, ws, pad)
    return out
