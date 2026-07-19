"""Parity gate: JAX/NNX EfficientNetB3Std vs torchvision efficientnet_b3
(ImageNet weights), golden fixture from
jarvis_jax/convert/export_effnet_b3_imagenet_fixture.py.

P3=feat_3 (48,56,56), P4=feat_5 (136,28,28), P5=feat_7 (384,14,14) for a
448x448 input (see the fixture export script's printed stride/channel table).
"""
import os

import numpy as np
import jax.numpy as jnp
import pytest

FIX = os.path.join(os.path.dirname(__file__), "..", "jarvis_jax", "convert",
                    "fixtures", "effnet_b3_imagenet.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(FIX), reason="generate fixture (export_effnet_b3_imagenet_fixture.py)")


def test_effnet_b3_std_taps_match_torchvision():
    from flax import nnx
    from jarvis_jax.models.effnet_b3_std import EfficientNetB3Std, load_b3std_from_npz

    z = np.load(FIX)
    x = jnp.asarray(np.transpose(z["input_nchw"], (0, 2, 3, 1)))  # NCHW -> NHWC

    net = EfficientNetB3Std(rngs=nnx.Rngs(0))
    net = load_b3std_from_npz(net, z)
    p3, p4, p5 = net(x, use_running_average=True)

    # Plain allclose (no guarded/loosened criterion needed -- observed max
    # abs diffs are ~6e-4..8e-4 across all three taps, comfortably inside
    # atol=2e-3; float32 accumulation across ~26 MBConv blocks + 7 BatchNorms
    # of rescaling does not require a looser bound here).
    print()
    for got, key, expect_c in [(p3, "feat_3", 48), (p4, "feat_5", 136), (p5, "feat_7", 384)]:
        ref = np.transpose(z[key], (0, 2, 3, 1))  # NCHW -> NHWC
        assert got.shape == ref.shape, (key, got.shape, ref.shape)
        assert got.shape[-1] == expect_c, (key, got.shape)
        got_np = np.asarray(got)
        abs_diff = np.abs(got_np - ref)
        print(f"{key}: shape={got.shape} max_abs_diff={float(abs_diff.max()):.6g} "
              f"mean_abs_diff={float(abs_diff.mean()):.6g}")
        np.testing.assert_allclose(got_np, ref, atol=2e-3, rtol=2e-3)
