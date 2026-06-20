"""Backbone forward-parity: NNX ViT (MAE weights) vs timm forward_features.

The timm reference is produced once in the torch env via
``scripts/make_ref_features.py``, which writes a *shared* input
``/tmp/parity_rgb_nhwc.npy`` and the reference ``/tmp/ref_feats.npy``. The test
LOADS the shared input (does NOT regenerate it) so both frameworks see identical
pixels, then appends a zero 4th (mask) channel -- which is a no-op because the
loader zero-inits the 4th patch-embed channel.

Generate refs (torch/jarvis env)::

    /gscratch/portia/eabe/miniconda3/envs/jarvis/bin/python scripts/make_ref_features.py

Run the test (jax/3d_tracking env)::

    MAE_NPZ=/tmp/mae_vitb.npz REF_NPY=/tmp/ref_feats.npy \
    RGB_NPY=/tmp/parity_rgb_nhwc.npy \
    python -m pytest tests/test_parity.py -v -s
"""
import os
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_default_matmul_precision", "highest")

from flax import nnx
from jarvis_jax import ViTPoseConfig
from jarvis_jax.models.vit import ViT
from jarvis_jax.convert.load_weights import load_vit_from_npz

NPZ = os.environ.get("MAE_NPZ", "/tmp/mae_vitb.npz")
REF_NPY = os.environ.get("REF_NPY", "/tmp/ref_feats.npy")
RGB_NPY = os.environ.get("RGB_NPY", "/tmp/parity_rgb_nhwc.npy")


def _run_nnx(cfg, rgb_nhwc):
    """Load weights into a fresh ViT and run it on a 4-channel (zero mask) input."""
    vit = ViT(cfg, rngs=nnx.Rngs(0))
    vit = load_vit_from_npz(vit, NPZ, cfg)
    b, h, w, _ = rgb_nhwc.shape
    x = jnp.concatenate([jnp.asarray(rgb_nhwc), jnp.zeros((b, h, w, 1))], axis=-1)
    return np.asarray(vit(x))


def _rel(feats, ref):
    return float(np.abs(feats - ref).max() / (np.abs(ref).max() + 1e-6))


@pytest.mark.skipif(not os.path.exists(NPZ), reason="run Task 8 to produce the npz")
@pytest.mark.skipif(
    not os.path.exists(REF_NPY) or not os.path.exists(RGB_NPY),
    reason="run scripts/make_ref_features.py in the torch env first",
)
def test_backbone_parity_vs_reference():
    """448 parity: passes < 5e-2 (loosened for 14->28 pos-embed interpolation)."""
    cfg = ViTPoseConfig()  # img_size=448
    rgb = np.load(RGB_NPY)
    assert rgb.shape == (1, cfg.img_size, cfg.img_size, 3), rgb.shape
    feats = _run_nnx(cfg, rgb)
    ref = np.load(REF_NPY)
    assert feats.shape == ref.shape, (feats.shape, ref.shape)
    rel = _rel(feats, ref)
    print(f"\n[448] max relative backbone diff: {rel:.3e}")
    assert rel < 5e-2, f"448 parity {rel:.3e} >= 5e-2"


REF224 = "/tmp/ref_feats_224.npy"
RGB224 = "/tmp/parity_rgb_nhwc_224.npy"


@pytest.mark.skipif(not os.path.exists(NPZ), reason="run Task 8 to produce the npz")
@pytest.mark.skipif(
    not os.path.exists(REF224) or not os.path.exists(RGB224),
    reason="run IMG=224 RGB_NPY=/tmp/parity_rgb_nhwc_224.npy "
    "REF_NPY=/tmp/ref_feats_224.npy scripts/make_ref_features.py",
)
def test_backbone_parity_224_diagnostic():
    """224 diagnostic: pos-embed interp is ~identity (14->14), so this is tight.

    A loose number here would indicate the qkv/LN/MLP mapping is wrong (not the
    448 interpolation drift).
    """
    cfg = dataclasses.replace(ViTPoseConfig(), img_size=224)
    rgb = np.load(RGB224)
    assert rgb.shape == (1, 224, 224, 3), rgb.shape
    feats = _run_nnx(cfg, rgb)
    ref = np.load(REF224)
    assert feats.shape == ref.shape, (feats.shape, ref.shape)
    rel = _rel(feats, ref)
    print(f"\n[224] max relative backbone diff: {rel:.3e}")
    assert rel < 5e-3, f"224 diagnostic {rel:.3e} >= 5e-3 -> mapping is wrong"
