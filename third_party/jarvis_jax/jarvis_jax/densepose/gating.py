"""Inference-time dilated-mask gating for keypoint/dense heatmaps (JAX).

Mirrors jarvis/prediction/jarvis3D_multi.py:331-333: expand the SAM fly mask by a
dilation kernel (kp slack for wings/legs that extend past the mask edge), then
zero all heatmap mass outside it before decoding keypoints. Peaks inside the
dilated mask are preserved; spurious off-fly peaks are removed so the argmax /
soft-argmax lands on the fly. Opt-in at prediction time (Phase 5).
"""
from __future__ import annotations

import jax.image
import jax.numpy as jnp

from jarvis_jax.models.dilation import dilate_mask_jax


def gate_heatmaps(hm, mask, dilate=21):
    """Zero heatmap mass outside the dilated SAM mask.

    hm:   (B,H,W,K) heatmaps. mask: (B,Hm,Wm) float {0,1} (any res). dilate: static
    px kernel (0/1 -> gate by the raw mask). Returns (B,H,W,K), same dtype as hm.
    """
    hm = jnp.asarray(hm)
    B, H, W, K = hm.shape
    m = dilate_mask_jax(jnp.asarray(mask, dtype=hm.dtype), dilate)
    m_hw = jax.image.resize(m, (B, H, W), method="nearest")
    m_hw = (m_hw > 0.5).astype(hm.dtype)
    return hm * m_hw[..., None]
