import jax.numpy as jnp
import numpy as np
import pytest

from jarvis_jax.hybridnet.reproject import reproject_heatmaps, _build_half_grid


def _inputs(seed=0, B=2, C=7, J=3, hm=64):
    rng = np.random.default_rng(seed)
    return dict(
        heatmaps=jnp.asarray(rng.random((B, C, J, hm, hm), np.float32)),
        center3D=jnp.asarray(rng.normal(0, 5, (B, 3)).astype(np.float32)),
        centerHM=jnp.asarray(rng.uniform(100, 900, (B, C, 2)).astype(np.float32)),
        camera_matrices=jnp.asarray(rng.normal(0, 1, (B, C, 4, 3)).astype(np.float32)),
    )


def test_identity_rotation_is_byte_identical_to_default():
    """Guards the refactor: the shipped path must not move."""
    a = reproject_heatmaps(**_inputs(), grid_size=16, heatmap_size=64)
    b = reproject_heatmaps(**_inputs(), grid_size=16, heatmap_size=64,
                           rotation=jnp.eye(3, dtype=jnp.float32))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_rotation_changes_the_volume():
    theta = np.pi / 3
    R = jnp.asarray(np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]], np.float32))
    a = reproject_heatmaps(**_inputs(), grid_size=16, heatmap_size=64)
    b = reproject_heatmaps(**_inputs(), grid_size=16, heatmap_size=64, rotation=R)
    assert not np.allclose(np.asarray(a), np.asarray(b))


def test_grid_spacing_scales_the_sampled_extent():
    """A finer spacing must span a proportionally smaller world extent —
    this is what makes stage-2 refinement 4x higher resolution."""
    g1 = np.asarray(_build_half_grid(48, 1.0))
    g4 = np.asarray(_build_half_grid(48, 0.25))
    assert np.isclose(g1.max() / g4.max(), 4.0, rtol=1e-5)


def test_rotation_is_orthogonality_checked():
    bad = jnp.asarray(np.diag([2.0, 1.0, 1.0]).astype(np.float32))
    with pytest.raises(ValueError, match="orthogonal"):
        reproject_heatmaps(**_inputs(), grid_size=16, heatmap_size=64, rotation=bad)
