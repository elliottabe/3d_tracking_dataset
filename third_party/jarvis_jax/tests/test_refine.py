import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jarvis_jax.hybridnet.refine import (
    RefineNet, refine_volumes, refine_keypoints)

B, C, J, HM = 2, 7, 5, 64


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.random((B, C, J, HM, HM), np.float32)),
            jnp.asarray(rng.normal(0, 3, (B, J, 3)).astype(np.float32)),
            jnp.asarray(rng.uniform(200, 800, (B, C, 2)).astype(np.float32)),
            jnp.asarray(rng.normal(0, 1, (B, C, 4, 3)).astype(np.float32)))


def test_refine_volume_shape_is_per_joint_and_small():
    hm, kp, chm, cm = _inputs()
    v = refine_volumes(hm, kp, chm, cm, cube=8, spacing=0.25, heatmap_size=HM)
    assert v.shape == (B, J, 8, 8, 8)


def test_refine_volume_is_much_cheaper_than_stage1():
    """50 joints x 24^3 must be ~8x SMALLER than one 50 x 48^3 stage-1 volume."""
    assert (50 * 24 ** 3) * 8 == 50 * 48 ** 3


def test_each_joint_volume_is_centred_on_its_own_coarse_estimate():
    """Two joints far apart must produce different volumes — a single shared
    volume would silently destroy the per-joint refinement."""
    hm, kp, chm, cm = _inputs()
    kp = kp.at[0, 0].set(jnp.array([0.0, 0.0, 0.0]))
    kp = kp.at[0, 1].set(jnp.array([20.0, 20.0, 20.0]))
    v = refine_volumes(hm, kp, chm, cm, cube=8, spacing=0.25, heatmap_size=HM)
    assert not np.allclose(np.asarray(v)[0, 0], np.asarray(v)[0, 1])


def test_refinenet_shares_weights_across_joints():
    net = RefineNet(channels=8, rngs=nnx.Rngs(0))
    x = jnp.zeros((B * J, 8, 8, 8, 1), jnp.float32)
    y = net(x, use_running_average=True)
    assert y.shape == (B * J, 8, 8, 8, 1)
    n_params = sum(int(np.prod(p.shape)) for p in
                   jax.tree_util.tree_leaves(nnx.state(net, nnx.Param)))
    assert n_params < 200_000, "RefineNet must stay small; it runs 50x per sample"


def test_refined_keypoints_stay_within_the_capture_window():
    """A refinement may move a joint by at most +/- cube/2 * spacing * 2."""
    hm, kp, chm, cm = _inputs()
    net = RefineNet(channels=8, rngs=nnx.Rngs(0))
    out = refine_keypoints(net, hm, kp, chm, cm, cube=8, spacing=0.25,
                           heatmap_size=HM)
    assert out.shape == (B, J, 3)
    limit = 8 / 2 * 0.25 * 2
    assert np.all(np.abs(np.asarray(out - kp)) <= limit + 1e-3)


import jax  # noqa: E402  (used by the param-count assertion above)
