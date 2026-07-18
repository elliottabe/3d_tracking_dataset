import os, numpy as np, jax.numpy as jnp, pytest
FIX = os.path.join(os.path.dirname(__file__), "..", "jarvis_jax", "convert",
                   "fixtures", "efficienttrack_large.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(FIX), reason="generate fixture (Task 1)")


def _load_full(z):
    from flax import nnx
    from jarvis_jax.models.efficienttrack import EfficientTrack, load_efficienttrack_from_npz
    net = EfficientTrack(num_joints=int(z["num_joints"]), rngs=nnx.Rngs(0))
    return load_efficienttrack_from_npz(net, z)  # loads backbone + bifpn + head


def test_res2_heatmaps_match_pytorch():
    z = np.load(FIX, allow_pickle=True)
    x = jnp.asarray(np.transpose(z["input_nchw"], (0, 2, 3, 1)))
    net = _load_full(z)
    res2 = net(x)                                   # (N,224,224,J)
    ref = np.transpose(z["res2"], (0, 2, 3, 1))     # NCHW->NHWC
    assert res2.shape == ref.shape, (res2.shape, ref.shape)
    np.testing.assert_allclose(np.asarray(res2), ref, atol=2e-4, rtol=2e-4)


def test_res1_heatmaps_match_pytorch():
    z = np.load(FIX, allow_pickle=True)
    x = jnp.asarray(np.transpose(z["input_nchw"], (0, 2, 3, 1)))
    net = _load_full(z)
    res1, res2 = net.forward_both(x)
    ref1 = np.transpose(z["res1"], (0, 2, 3, 1))
    assert res1.shape == ref1.shape, (res1.shape, ref1.shape)
    np.testing.assert_allclose(np.asarray(res1), ref1, atol=2e-4, rtol=2e-4)
