import os, numpy as np, jax.numpy as jnp, pytest
FIX = os.path.join(os.path.dirname(__file__), "..", "jarvis_jax", "convert",
                   "fixtures", "efficienttrack_large.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(FIX), reason="generate fixture (Task 1)")


def test_backbone_feature_maps_match_pytorch():
    from flax import nnx
    from jarvis_jax.models.efficientnet import EfficientNetB3, load_backbone_from_npz
    z = np.load(FIX, allow_pickle=True)
    x = jnp.asarray(np.transpose(z["input_nchw"], (0, 2, 3, 1)))  # NCHW->NHWC
    net = EfficientNetB3(rngs=nnx.Rngs(0))
    net = load_backbone_from_npz(net, z)
    p3, p4, p5 = net(x)
    for got, key in [(p3, "feat_p3"), (p4, "feat_p4"), (p5, "feat_p5")]:
        ref = np.transpose(z[key], (0, 2, 3, 1))  # NCHW->NHWC
        assert got.shape == ref.shape, (key, got.shape, ref.shape)
        np.testing.assert_allclose(np.asarray(got), ref, atol=1e-4, rtol=1e-4)
