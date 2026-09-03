import numpy as np
import jax, jax.numpy as jnp
import pytest
from flax import nnx

TINY = dict(patch=16, embed_dim=32, depth=2, num_heads=4, mlp_ratio=2.0,
            num_registers=4, rope_base=100.0, layer_norm_eps=1e-5, in_ch=3)


def test_forward_shapes_patch_tokens_only():
    from jarvis_jax.models.dinov3 import DINOv3, DINOv3Config
    cfg = DINOv3Config(**TINY)
    m = DINOv3(cfg, rngs=nnx.Rngs(0))
    x = jnp.zeros((2, 64, 48, 3), jnp.float32)          # non-square on purpose
    toks = m(x)
    assert toks.shape == (2, (64 // 16) * (48 // 16), 32)
    assert m.forward_all(x).shape == (2, 1 + 4 + 12, 32)


def test_rope_angles_shape_and_tile():
    from jarvis_jax.models.dinov3 import rope_cos_sin
    cos, sin = rope_cos_sin(h=4, w=3, head_dim=8, base=100.0)
    assert cos.shape == sin.shape == (12, 8)
    # tile(2): second half of the last axis repeats the first half
    np.testing.assert_allclose(np.asarray(cos)[:, :4], np.asarray(cos)[:, 4:])


def test_presets_match_official_sizes():
    from jarvis_jax.models.dinov3 import DINOv3Config
    b, l = DINOv3Config.vitb16(), DINOv3Config.vitl16()
    assert (b.embed_dim, b.depth, b.num_heads, b.num_registers) == (768, 12, 12, 4)
    assert (l.embed_dim, l.depth, l.num_heads, l.num_registers) == (1024, 24, 16, 4)


def _write_hf_style_safetensors(tmp_path, cfg, seed=0):
    """Emit a checkpoint in the HuggingFace DINOv3ViTModel key layout with
    random values, and return (dir, dict_of_numpy)."""
    from safetensors.numpy import save_file
    rng = np.random.default_rng(seed)
    D, H = cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio)
    t = {}
    t["embeddings.cls_token"] = rng.normal(size=(1, 1, D)).astype(np.float32)
    t["embeddings.mask_token"] = rng.normal(size=(1, 1, D)).astype(np.float32)
    t["embeddings.register_tokens"] = rng.normal(size=(1, cfg.num_registers, D)).astype(np.float32)
    t["embeddings.patch_embeddings.weight"] = rng.normal(size=(D, cfg.in_ch, cfg.patch, cfg.patch)).astype(np.float32)
    t["embeddings.patch_embeddings.bias"] = rng.normal(size=(D,)).astype(np.float32)
    for i in range(cfg.depth):
        p = f"layer.{i}."
        for n in ("q_proj", "v_proj", "o_proj"):
            t[p + f"attention.{n}.weight"] = rng.normal(size=(D, D)).astype(np.float32)
            t[p + f"attention.{n}.bias"] = rng.normal(size=(D,)).astype(np.float32)
        t[p + "attention.k_proj.weight"] = rng.normal(size=(D, D)).astype(np.float32)   # no k bias
        t[p + "mlp.up_proj.weight"] = rng.normal(size=(H, D)).astype(np.float32)
        t[p + "mlp.up_proj.bias"] = rng.normal(size=(H,)).astype(np.float32)
        t[p + "mlp.down_proj.weight"] = rng.normal(size=(D, H)).astype(np.float32)
        t[p + "mlp.down_proj.bias"] = rng.normal(size=(D,)).astype(np.float32)
        t[p + "layer_scale1.lambda1"] = rng.normal(size=(D,)).astype(np.float32)
        t[p + "layer_scale2.lambda1"] = rng.normal(size=(D,)).astype(np.float32)
        for n in ("norm1", "norm2"):
            t[p + f"{n}.weight"] = rng.normal(size=(D,)).astype(np.float32)
            t[p + f"{n}.bias"] = rng.normal(size=(D,)).astype(np.float32)
    t["norm.weight"] = rng.normal(size=(D,)).astype(np.float32)
    t["norm.bias"] = rng.normal(size=(D,)).astype(np.float32)
    d = tmp_path / "ckpt"; d.mkdir()
    save_file(t, str(d / "model.safetensors"))
    return str(d), t


def test_loader_consumes_every_key_and_maps_layouts(tmp_path):
    from jarvis_jax.models.dinov3 import DINOv3, DINOv3Config, load_dinov3_safetensors
    cfg = DINOv3Config(**TINY)
    d, t = _write_hf_style_safetensors(tmp_path, cfg)
    m = load_dinov3_safetensors(DINOv3(cfg, rngs=nnx.Rngs(0)), d)
    # Linear: torch (out,in) -> flax kernel (in,out)
    np.testing.assert_allclose(np.asarray(m.layer[0].attention.q_proj.kernel[...]),
                               t["layer.0.attention.q_proj.weight"].T)
    # Conv: torch (O,I,kh,kw) -> flax (kh,kw,I,O)
    np.testing.assert_allclose(np.asarray(m.embeddings.patch_embeddings.kernel[...]),
                               t["embeddings.patch_embeddings.weight"].transpose(2, 3, 1, 0))
    # LayerNorm weight -> scale
    np.testing.assert_allclose(np.asarray(m.norm.scale[...]), t["norm.weight"])
    assert m.layer[0].attention.k_proj.bias is None   # k has no bias in DINOv3


def test_loader_refuses_unconsumed_key(tmp_path):
    from safetensors.numpy import save_file
    from jarvis_jax.models.dinov3 import DINOv3, DINOv3Config, load_dinov3_safetensors
    cfg = DINOv3Config(**TINY)
    d, t = _write_hf_style_safetensors(tmp_path, cfg)
    t["layer.0.attention.extra.weight"] = np.zeros((1,), np.float32)
    save_file(t, str(tmp_path / "ckpt" / "model.safetensors"))
    with pytest.raises(RuntimeError, match="unconsumed"):
        load_dinov3_safetensors(DINOv3(cfg, rngs=nnx.Rngs(0)), d)


def test_registered_backbone_slices_to_rgb():
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.backbone import build_backbone, BACKBONES
    assert "dinov3_b16" in BACKBONES and "dinov3_l16" in BACKBONES
    # tiny override path: factory honours cfg.embed_dim/depth/num_heads when
    # they are SMALLER than the preset (test-only knob) so CPU tests stay fast
    cfg = ViTPoseConfig(img_size=64, patch=16, in_ch=4, embed_dim=32, depth=2, num_heads=4)
    bb = build_backbone("dinov3_b16", cfg, rngs=nnx.Rngs(0))
    toks = bb(jnp.zeros((1, 64, 64, 4)))
    assert toks.shape == (1, 16, 32)


@pytest.mark.gpu
def test_parity_against_transformers_reference():
    """Runs only where `transformers` (>=4.56, DINOv3ViTModel) is installed:
    `pip install transformers` in the 3d_tracking env. Same random weights,
    same input, outputs agree to 3e-3 (RoPE runs in bf16 in both)."""
    transformers = pytest.importorskip("transformers")
    import torch, tempfile
    from safetensors.torch import save_file
    from jarvis_jax.models.dinov3 import DINOv3, DINOv3Config, load_dinov3_safetensors
    hfcfg = transformers.DINOv3ViTConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                         num_attention_heads=4, hidden_act="gelu",
                                         use_gated_mlp=False, num_register_tokens=4, patch_size=16)
    ref = transformers.DINOv3ViTModel(hfcfg).eval()
    cfg = DINOv3Config(patch=16, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0, num_registers=4)
    with tempfile.TemporaryDirectory() as d:
        save_file(ref.state_dict(), f"{d}/model.safetensors")
        m = load_dinov3_safetensors(DINOv3(cfg, rngs=nnx.Rngs(0)), d)
    x = np.random.default_rng(0).normal(size=(2, 3, 64, 64)).astype(np.float32)
    with torch.inference_mode():
        y_ref = ref(pixel_values=torch.tensor(x)).last_hidden_state.numpy()
    y = np.asarray(m.forward_all(jnp.asarray(x.transpose(0, 2, 3, 1))))
    np.testing.assert_allclose(y, y_ref, atol=3e-3, rtol=1e-4)
