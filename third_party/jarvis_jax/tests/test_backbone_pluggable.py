import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jarvis_jax.config import ViTPoseConfig


TINY = dict(img_size=64, patch=16, embed_dim=48, depth=2, num_heads=4,
            num_keypoints=5, heatmap_size=32)   # CPU-friendly


def test_registry_has_vit_and_builds_it():
    from jarvis_jax.models.backbone import BACKBONES, build_backbone, Backbone
    from jarvis_jax.models.vit import ViT
    assert "vit" in BACKBONES
    cfg = ViTPoseConfig(**TINY)
    bb = build_backbone("vit", cfg, rngs=nnx.Rngs(0))
    assert isinstance(bb, ViT)
    assert isinstance(bb, Backbone)                        # structural protocol match
    toks = bb(jnp.zeros((1, 64, 64, 4)))
    assert toks.shape == (1, (64 // 16) ** 2, 48)          # (B, num_tokens, embed_dim)


def test_build_backbone_passthrough_and_unknown():
    from jarvis_jax.models.backbone import build_backbone
    from jarvis_jax.models.vit import ViT
    cfg = ViTPoseConfig(**TINY)
    prebuilt = ViT(cfg, rngs=nnx.Rngs(1))
    assert build_backbone(prebuilt, cfg, rngs=nnx.Rngs(0)) is prebuilt   # passthrough
    with pytest.raises(KeyError):
        build_backbone("does_not_exist", cfg, rngs=nnx.Rngs(0))


def test_register_backbone_and_use_in_vitpose():
    from jarvis_jax.models.backbone import register_backbone, BACKBONES
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.models.vit import ViT

    calls = {"n": 0}
    def _wrapped(cfg, *, rngs):
        calls["n"] += 1
        return ViT(cfg, rngs=rngs)
    register_backbone("vit_counted", _wrapped)
    assert "vit_counted" in BACKBONES
    cfg = ViTPoseConfig(**TINY)
    m = ViTPose(cfg, backbone="vit_counted", rngs=nnx.Rngs(0))
    assert calls["n"] == 1
    out = m(jnp.zeros((1, 64, 64, 4)))
    assert out.shape == (1, 32, 32, 5)


def test_vitpose_default_backward_compat():
    # the pre-existing hardcoded-ViT construction path is unchanged.
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.models.vit import ViT
    cfg = ViTPoseConfig(**TINY)
    m = ViTPose(cfg, rngs=nnx.Rngs(0))                     # no backbone= given
    assert isinstance(m.backbone, ViT)
    out = m(jnp.zeros((1, 64, 64, 4)))
    assert out.shape == (1, 32, 32, 5)
    assert jnp.isfinite(out).all()


def test_default_param_paths_unchanged():
    # ViTPose(cfg) and ViTPose(cfg, backbone="vit") must yield the SAME param-tree
    # structure (so existing checkpoints restore into either).
    #
    # NOTE (adapted from brief): this nnx version's `State.flat_state()` yields
    # (path_tuple, value) pairs where `path_tuple` is a plain tuple of str/int
    # keys -- not a jax.tree_util keypath object -- so `jax.tree_util.keystr`
    # does not apply here. Stringifying the raw path tuples gives the same
    # "identical param tree structure" assertion the brief intends.
    from jarvis_jax.models.vitpose import ViTPose
    cfg = ViTPoseConfig(**TINY)
    a = ViTPose(cfg, rngs=nnx.Rngs(0))
    b = ViTPose(cfg, backbone="vit", rngs=nnx.Rngs(0))
    ka = sorted(str(p) for p, _ in nnx.split(a)[1].flat_state())
    kb = sorted(str(p) for p, _ in nnx.split(b)[1].flat_state())
    assert ka == kb
