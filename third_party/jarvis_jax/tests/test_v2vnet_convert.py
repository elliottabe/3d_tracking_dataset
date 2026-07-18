"""Convert v2vNet.* from HybridNet-large_final.pth -> Orbax ckpt -> load -> shape check.

Forward-parity gate: this asserts the converted module runs and preserves shape
(B,C_in,48,48,48)->(B,C_out,24,24,24) (channels-last: (B,48,48,48,C_in)->(B,24,24,24,C_out)).
There is no standalone v2vNet golden fixture; numeric parity is covered end-to-end
by Task 5's full-HybridNet fixture.
"""
import os
import jax.numpy as jnp
import pytest

PTH = os.path.join(os.path.dirname(__file__), "..", "..", "JARVIS-HybridNet",
                    "projects", "unified_V3_masked", "models", "HybridNet",
                    "Run_20260620-173554", "HybridNet-large_final.pth")
pytestmark = pytest.mark.skipif(not os.path.exists(PTH), reason="needs HybridNet .pth")


def test_convert_v2vnet_shape(tmp_path):
    from jarvis_jax.convert.load_v2vnet_torch import convert_v2vnet_pth, load_v2vnet_ckpt

    out = tmp_path / "v2v"
    convert_v2vnet_pth(PTH, in_ch=50, out_ch=50, out_dir=str(out))
    v2v = load_v2vnet_ckpt(str(out), in_ch=50, out_ch=50)
    y = v2v(jnp.zeros((1, 48, 48, 48, 50)), use_running_average=True)
    assert y.shape == (1, 24, 24, 24, 50)
