"""Parity test: JAX reproject_heatmaps must match PyTorch ReprojectionLayer."""
import os
import numpy as np
import jax.numpy as jnp
import pytest
from jarvis_jax.hybridnet.reproject import reproject_heatmaps

FIX = os.path.join(os.path.dirname(__file__), "..",
                   "jarvis_jax", "convert", "reproject_fixture.npz")
skip = pytest.mark.skipif(not os.path.exists(FIX), reason="parity fixture absent")


@skip
def test_reproject_matches_pytorch():
    z = np.load(FIX)
    out = np.asarray(reproject_heatmaps(
        jnp.asarray(z["heatmaps"]), jnp.asarray(z["center3D"]),
        jnp.asarray(z["centerHM"]), jnp.asarray(z["cameraMatrices"]),
        grid_size=int(z["grid_size"]), grid_spacing=int(z["grid_spacing"]),
        heatmap_size=int(z["heatmap_size"])))
    ref = z["heatmaps3D"]
    assert out.shape == ref.shape, (out.shape, ref.shape)
    # gather/average of identical heatmap values -> should match closely; the
    # only expected drift is trilinear-interp/rounding at borders.
    assert np.abs(out - ref).max() < 1e-2, float(np.abs(out - ref).max())
