import jax.numpy as jnp
import numpy as np

from jarvis_jax.hybridnet.model import soft_argmax_3d


def _one_hot(G, idx):
    v = np.zeros((1, 1, G, G, G), np.float32)
    v[0, 0, idx[0], idx[1], idx[2]] = 1.0
    return jnp.asarray(v)


def test_spacing_one_unchanged():
    G = 8
    pts, _ = soft_argmax_3d(_one_hot(G, (5, 2, 6)), grid_spacing=1.0, roi_cube=8.0)
    expect = np.array([5, 2, 6], np.float32) * 1.0 * 2 - 8.0 / 2
    np.testing.assert_allclose(np.asarray(pts)[0, 0], expect, atol=1e-4)


def test_quarter_spacing_gives_quarter_extent():
    """The whole point of stage 2: same voxel index, 1/4 the world offset."""
    G = 8
    pts, _ = soft_argmax_3d(_one_hot(G, (5, 2, 6)), grid_spacing=0.25, roi_cube=2.0)
    expect = np.array([5, 2, 6], np.float32) * 0.25 * 2 - 2.0 / 2
    np.testing.assert_allclose(np.asarray(pts)[0, 0], expect, atol=1e-4)


def test_subvoxel_interpolation_on_a_two_voxel_blob():
    """A peak split evenly between neighbours must land BETWEEN them —
    this is the sub-voxel precision stage 2 exists to exploit."""
    G = 8
    v = np.zeros((1, 1, G, G, G), np.float32)
    v[0, 0, 4, 4, 4] = 1.0
    v[0, 0, 5, 4, 4] = 1.0
    pts, _ = soft_argmax_3d(jnp.asarray(v), grid_spacing=0.25, roi_cube=2.0)
    assert 4.4 < (np.asarray(pts)[0, 0, 0] + 1.0) / 0.5 < 4.6


def test_conf_unaffected_by_spacing():
    G = 8
    _, c1 = soft_argmax_3d(_one_hot(G, (5, 2, 6)), grid_spacing=1.0, roi_cube=8.0)
    _, c2 = soft_argmax_3d(_one_hot(G, (5, 2, 6)), grid_spacing=0.25, roi_cube=2.0)
    np.testing.assert_allclose(np.asarray(c1), np.asarray(c2), atol=1e-6)
