import os, numpy as np, jax.numpy as jnp, pytest
FIX = os.path.join(os.path.dirname(__file__), "..", "jarvis_jax", "convert",
                   "fixtures", "efficienttrack_large.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(FIX), reason="generate fixture (Task 1)")


def _load_full(z):
    from flax import nnx
    from jarvis_jax.models.efficienttrack import EfficientTrack, load_efficienttrack_from_npz
    net = EfficientTrack(num_joints=int(z["num_joints"]), rngs=nnx.Rngs(0))
    return load_efficienttrack_from_npz(net, z)  # loads backbone + bifpn + head


def _assert_parity(pred, ref, *, name):
    """Guarded EfficientTrack heatmap parity vs PyTorch.

    The residual is NOT a structural bug: it is zero-mean float32 rounding
    noise amplified by 1/sqrt(eps)~316x in InstanceNorm on near-constant
    ("dead") BiFPN channels (~10/50 output channels). Verified in
    .superpowers/sdd/task-3-report.md and independently in review:
    mean signed diff ~-5e-7 (no systematic bias), corr 0.9999999996.
    The mean/signed-mean bounds are the load-bearing regression guard (wrong
    weight/axis/norm-eps would produce a systematic offset); the max/corr/
    violation bounds guard against the noise floor silently growing.
    """
    diff = np.asarray(pred) - ref
    absdiff = np.abs(diff)
    mean_abs = absdiff.mean()
    signed_mean = diff.mean()
    max_abs = absdiff.max()
    corr = np.corrcoef(np.asarray(pred).ravel(), ref.ravel())[0, 1]
    viol = absdiff > (2e-4 + 2e-4 * np.abs(ref))
    viol_rate = viol.mean()

    print(f"[{name}] mean_abs_diff={mean_abs:.3e} signed_mean={signed_mean:.3e} "
          f"max_abs_diff={max_abs:.3e} corr={corr:.10f} violation_rate={viol_rate:.3%}")

    assert mean_abs < 3e-4,            f"mean abs diff {mean_abs:.2e}"
    assert abs(signed_mean) < 5e-5,    f"signed mean {signed_mean:.2e} (systematic bias?)"
    assert max_abs < 6e-3,             f"max abs diff {max_abs:.2e}"
    assert corr > 0.99999,             f"correlation {corr:.10f}"
    assert viol_rate < 0.02,           f"violation rate {viol_rate:.3%}"


def test_res2_heatmaps_match_pytorch():
    z = np.load(FIX, allow_pickle=True)
    x = jnp.asarray(np.transpose(z["input_nchw"], (0, 2, 3, 1)))
    net = _load_full(z)
    res2 = net(x)                                   # (N,224,224,J)
    ref = np.transpose(z["res2"], (0, 2, 3, 1))     # NCHW->NHWC
    assert res2.shape == ref.shape, (res2.shape, ref.shape)
    _assert_parity(res2, ref, name="res2")


def test_res1_heatmaps_match_pytorch():
    z = np.load(FIX, allow_pickle=True)
    x = jnp.asarray(np.transpose(z["input_nchw"], (0, 2, 3, 1)))
    net = _load_full(z)
    res1, res2 = net.forward_both(x)
    ref1 = np.transpose(z["res1"], (0, 2, 3, 1))
    assert res1.shape == ref1.shape, (res1.shape, ref1.shape)
    _assert_parity(res1, ref1, name="res1")
