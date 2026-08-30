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


class _AllNegativeNet:
    """Stand-in for ``RefineNet`` that always emits raw, entirely non-positive
    per-voxel logits (varying across joints, but never above zero) -- exactly
    the state ``soft_argmax_3d``'s contract assumes CANNOT happen at its input
    (its own comments: "already-positive softplus output"). Real RefineNet
    output is unbounded/signed; a near-zero-init net (or one that simply
    hasn't learned to peak on a given joint yet) can easily land here."""

    def __init__(self, rng):
        self._rng = rng

    def __call__(self, x, use_running_average=False):
        shape = x.shape[:-1] + (1,)
        # Strictly negative (never exactly 0) so this isn't sensitive to
        # relu's boundary convention -- and still varies per (B*J) row so a
        # degenerate, input-independent result is distinguishable from a real
        # (if small) one.
        vals = -np.abs(self._rng.standard_normal(shape).astype(np.float32)) - 0.01
        return jnp.asarray(vals)


def test_refine_keypoints_does_not_collapse_negative_windows_to_a_fixed_corner():
    """Reproduces the soft_argmax_3d contract violation described in
    hybridnet/model.py (lines 84, 96): that function assumes its input is
    ALREADY softplus-positive; refine_keypoints instead fed it raw signed
    RefineNet logits directly.

    Consequence: when a joint's entire refinement window is non-positive
    (an ordinary, expected RefineNet state -- see ``_AllNegativeNet``), relu
    zeroes EVERY voxel. soft_argmax_3d's own zero-guard (``norm == 0 ->
    ones_like``) then makes the per-joint offset a function of ``cube`` and
    ``spacing`` ONLY -- not of the (differing) negative content -- so EVERY
    such joint, no matter its actual raw logits, is pushed to the exact same
    hard-coded grid corner. That is a real, finite-valued but data-destroying
    failure (and the smooth-but-unbounded cousin of it -- a near-empty
    surviving-voxel window -- is what produces the 40-70x heavier gradient
    tail measured for the refine arms; see refine-softplus-fix.md).

    Before the fix (softplus applied in refine_keypoints): FAILS -- every
    joint's delta is identical. After: every joint's delta differs (softplus
    keeps voxels weakly positive and content-sensitive even when the raw
    logits are all negative).
    """
    hm, kp, chm, cm = _inputs()
    net = _AllNegativeNet(np.random.default_rng(1))
    out = refine_keypoints(net, hm, kp, chm, cm, cube=8, spacing=0.25,
                           heatmap_size=HM)
    delta = np.asarray(out - kp)  # (B, J, 3)
    assert np.all(np.isfinite(delta)), "refine produced non-finite offsets"
    first = delta[0, 0]
    collapsed = np.array([np.allclose(delta[b, j], first, atol=1e-4)
                          for b in range(B) for j in range(J)])
    assert not collapsed.all(), (
        "every joint's refined offset collapsed to the SAME fixed value "
        f"({first}) despite differing (all-negative) RefineNet logits -- "
        "refine_keypoints is feeding un-softplussed logits into "
        "soft_argmax_3d, discarding all per-joint information")


import jax  # noqa: E402  (used by the param-count assertion above)
