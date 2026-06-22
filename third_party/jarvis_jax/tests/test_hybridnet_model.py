# tests/test_hybridnet_model.py
"""Tests for soft_argmax_3d and HybridNet3D."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jarvis_jax.hybridnet.model import soft_argmax_3d, HybridNet3D
from jarvis_jax.hybridnet.v2vnet import V2VNet
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.config import ViTPoseConfig


# ---------------------------------------------------------------------------
# Primary gate: soft_argmax_3d recovers the peak grid position
# ---------------------------------------------------------------------------

def test_soft_argmax_recovers_peak():
    G = 24
    vol = np.zeros((1, 1, G, G, G), dtype=np.float32)
    vol[0, 0, 5, 10, 15] = 10.0           # peak at grid index (5, 10, 15)
    pts, conf = soft_argmax_3d(jnp.asarray(vol), grid_spacing=1, roi_cube=48)
    # grid index 5,10,15 -> world = idx*grid_spacing*2 - roi_cube
    expect = np.array([5, 10, 15]) * 1 * 2 - 48
    assert np.allclose(np.asarray(pts)[0, 0], expect, atol=1.0), np.asarray(pts)[0, 0]
    assert float(conf[0, 0]) > 0


def test_soft_argmax_shape():
    B, J, G = 2, 5, 24
    vol = jnp.ones((B, J, G, G, G), dtype=jnp.float32)
    pts, conf = soft_argmax_3d(vol, grid_spacing=1, roi_cube=48)
    assert pts.shape == (B, J, 3)
    assert conf.shape == (B, J)


def test_soft_argmax_finite():
    rng = np.random.RandomState(42)
    vol = jnp.asarray(rng.randn(2, 10, 24, 24, 24).astype(np.float32))
    pts, conf = soft_argmax_3d(vol, grid_spacing=1, roi_cube=48)
    assert bool(jnp.isfinite(pts).all()), "points3D contains non-finite values"
    assert bool(jnp.isfinite(conf).all()), "confidences contain non-finite values"


def test_soft_argmax_conf_range():
    """Confidence should be in (0, 1]."""
    rng = np.random.RandomState(7)
    vol = jnp.asarray(rng.randn(1, 3, 24, 24, 24).astype(np.float32))
    pts, conf = soft_argmax_3d(vol, grid_spacing=1, roi_cube=48)
    conf_np = np.asarray(conf)
    assert (conf_np >= 0).all(), f"conf min={conf_np.min()}"
    assert (conf_np <= 1.0 + 1e-5).all(), f"conf max={conf_np.max()}"


# ---------------------------------------------------------------------------
# End-to-end smoke: HybridNet3D produces correct shapes + finite outputs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hybridnet_model():
    """Build a tiny HybridNet3D (random weights) for smoke tests."""
    cfg = ViTPoseConfig()
    rngs = nnx.Rngs(0)
    vitpose = ViTPose(cfg, rngs=rngs)
    v2vnet = V2VNet(cfg.num_keypoints, cfg.num_keypoints, rngs=rngs)
    model = HybridNet3D(vitpose, v2vnet, cfg)
    return model


def test_hybridnet3d_predict_heatmaps_shape(hybridnet_model):
    """predict_heatmaps should return (B, num_cam, 224, 224, J)."""
    B, num_cam = 1, 2
    J = 50
    rng = np.random.RandomState(0)
    crops = jnp.asarray(rng.randint(0, 255, (B, num_cam, 448, 448, 4), dtype=np.uint8))
    hm = hybridnet_model.predict_heatmaps(crops)
    assert hm.shape == (B, num_cam, 224, 224, J), hm.shape


def test_hybridnet3d_forward_shapes(hybridnet_model):
    """Full forward pass: vol (B,J,24,24,24), points3D (B,J,3), conf (B,J)."""
    B, num_cam = 1, 2
    J = 50
    rng = np.random.RandomState(1)
    crops = jnp.asarray(rng.randint(0, 255, (B, num_cam, 448, 448, 4), dtype=np.uint8))
    center3D = jnp.zeros((B, 3), dtype=jnp.float32)
    centerHM = jnp.zeros((B, num_cam, 2), dtype=jnp.float32)
    cam_mat = jnp.broadcast_to(
        jnp.eye(4, 3, dtype=jnp.float32)[None, None],
        (B, num_cam, 4, 3),
    )
    vol, pts, conf = hybridnet_model(
        crops, center3D, centerHM, cam_mat,
        use_running_average=True,
    )
    assert vol.shape == (B, J, 24, 24, 24), vol.shape
    assert pts.shape == (B, J, 3), pts.shape
    assert conf.shape == (B, J), conf.shape
    assert bool(jnp.isfinite(pts).all()), "points3D has non-finite values"
