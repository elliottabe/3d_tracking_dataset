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


def test_v2vnet_orbax_roundtrip_no_shape_mismatch(tmp_path):
    """Regression test for the c13ea16 review fix.

    A prior commit added ``transpose_kernel=True`` to ``Upsample3DBlock`` in
    the *shared* runtime module ``jarvis_jax/hybridnet/v2vnet.py``. That flips
    the expected Orbax-restore kernel shape for
    ``decoder_upsample1.deconv`` -- (out,in) vs (in,out) axis order -- and
    would have broken restoring any pre-existing JAX-native-trained V2VNet
    Orbax checkpoint (e.g. HybridNet's "run4") whose deconv kernel was saved
    under the *unmodified* (transpose_kernel=False) shape convention. This
    save -> eval_shape -> restore round trip on a freshly constructed
    (non-torch-converted) V2VNet is exactly that failure mode: it must not
    raise a shape-mismatch error.
    """
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import nnx

    from jarvis_jax.hybridnet.v2vnet import V2VNet

    in_ch = out_ch = 8  # small, distinct-enough from in==out to catch axis swaps
    model = V2VNet(in_ch, out_ch, rngs=nnx.Rngs(0))
    _, state = nnx.split(model)

    out_dir = tmp_path / "v2v_roundtrip"
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(str(out_dir), state)
    ckptr.wait_until_finished()

    m_abstract = nnx.eval_shape(lambda: V2VNet(in_ch, out_ch, rngs=nnx.Rngs(0)))
    gdef, abstract_state = nnx.split(m_abstract)
    restored_state = ckptr.restore(str(out_dir), target=abstract_state)
    restored = nnx.merge(gdef, restored_state)

    y = restored(jnp.zeros((1, 8, 8, 8, in_ch)), use_running_average=True)
    assert y.shape == (1, 4, 4, 4, out_ch)
