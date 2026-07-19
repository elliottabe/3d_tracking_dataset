# tests/test_hybridnet_fusion.py
"""Task 7: mask-fusion hook wired into HybridNet3D.__call__.

- ``fusion_mode='none'`` (default) must be byte-identical to the pre-Task-7
  forward -- masks are IGNORED entirely (critical regression guard).
- ``fusion_mode='carve'`` gates the pre-V2VNet volume by cross-camera mask
  consistency (Task 6's ``mask_consistency_volume`` + ``soft_gate``).
"""
import numpy as np
import jax.numpy as jnp
import pytest
from flax import nnx

from jarvis_jax.hybridnet.v2vnet import V2VNet


def _tiny_model(mode):
    from jarvis_jax.hybridnet.model import HybridNet3D

    class FrontEnd(nnx.Module):
        def predict_heatmaps(self, crops):  # (B,cam,H,W,J) constant heatmaps
            B, cam = crops.shape[0], crops.shape[1]
            return jnp.ones((B, cam, 224, 224, 4), "float32")

    class Cfg:
        num_keypoints = 4
        sharpen = 1.0
        fusion_mode = mode
        gate_temperature = 0.5
        gate_floor = 0.0

    return HybridNet3D(FrontEnd(), V2VNet(4, 4, rngs=nnx.Rngs(0)), Cfg())


def _inputs():
    B, cam = 1, 3
    crops = jnp.zeros((B, cam, 448, 448, 3), "float32")
    c3 = jnp.zeros((B, 3), "float32")
    # centerHM=113 (heatmap_size/2) centers the projected FOV in the 226-px
    # frame. NOTE: a bare jnp.eye(4, 3) camera (X/Z, Y/Z with Z running
    # through 0 across the grid, centerHM=0) reprojects this constant
    # front-end heatmap to an ALL-ZERO pre-gate volume (every grid voxel's
    # projected pixel saturates the reproject clamp onto the heatmap's
    # zero-padded border) -- confirmed directly against reproject_heatmaps
    # in isolation. Gating an all-zero volume is 0*anything==0 regardless of
    # the gate, so it can never differ under fusion_mode='carve' and is
    # unusable for test_carve_runs_and_differs. This camera instead uses a
    # simple W==1 (no perspective divide) projection with a scale (10x) big
    # enough that the projected Y range spans across the mask's carved
    # row-120 threshold (verified empirically: ~21% of voxels land in the
    # carved region, base != gated post-V2VNet).
    cHM = jnp.full((B, cam, 2), 113.0, "float32")
    M = np.zeros((cam, 4, 3), dtype=np.float32)
    for c in range(cam):
        M[c, 0, 0] = 10.0  # x -> val1 (column)
        M[c, 1, 1] = 10.0  # y -> val2 (row)
        M[c, 3, 2] = 1.0   # translation row supplies constant w = 1
    cams = jnp.tile(jnp.asarray(M)[None], (B, 1, 1, 1))
    masks = jnp.ones((B, cam, 226, 226), "float32")
    return crops, c3, cHM, cams, masks


def test_none_equals_baseline():
    crops, c3, cHM, cams, masks = _inputs()
    m = _tiny_model("none")
    a = m(crops, c3, cHM, cams, use_running_average=True)[1]
    b = m(crops, c3, cHM, cams, masks=masks, use_running_average=True)[1]
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))  # masks ignored in 'none'


def test_carve_runs_and_differs():
    crops, c3, cHM, cams, masks = _inputs()
    m = _tiny_model("carve")
    partial = masks.at[:, :, 120:, :].set(0.0)   # carve half the silhouette
    base = m(crops, c3, cHM, cams, masks=jnp.ones_like(masks), use_running_average=True)[0]
    gated = m(crops, c3, cHM, cams, masks=partial, use_running_average=True)[0]
    assert not np.allclose(np.asarray(base), np.asarray(gated))
