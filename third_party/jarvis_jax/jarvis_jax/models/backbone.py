"""Pluggable backbone registry for ViTPose (Phase 5).

Sets up the abstraction a future SAM3 ViTDet-trunk port would register into,
WITHOUT porting it (that ViT-L trunk is an explicit non-goal — spec §7/§10). The
default backbone is the existing JAX ViT-B ('vit'), so every current construction
path and the on-disk cse_vit350 checkpoints are unchanged.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vit import ViT


@runtime_checkable
class Backbone(Protocol):
    """A vision trunk: image (B,H,W,in_ch) -> patch tokens (B,num_tokens,embed_dim)."""
    def __call__(self, x): ...


def _vit_factory(cfg: ViTPoseConfig, *, rngs: nnx.Rngs):
    return ViT(cfg, rngs=rngs)


BACKBONES = {"vit": _vit_factory}


def register_backbone(name, factory):
    """Register a backbone factory `(cfg, *, rngs) -> nnx.Module` under `name`."""
    existing = BACKBONES.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(f"backbone '{name}' already registered to a different factory")
    BACKBONES[name] = factory


def build_backbone(name_or_module, cfg, *, rngs):
    """Return a backbone module. str -> look up + build; module -> passthrough."""
    if isinstance(name_or_module, str):
        try:
            factory = BACKBONES[name_or_module]
        except KeyError as e:
            raise KeyError(
                f"unknown backbone '{name_or_module}'; registered: {sorted(BACKBONES)}") from e
        return factory(cfg, rngs=rngs)
    return name_or_module      # already a built module


def _register_optional():
    # DINOv3 lives in its own module; registering here keeps `BACKBONES`
    # the single lookup table without importing it at package import.
    from jarvis_jax.models import dinov3
    dinov3.register()

_register_optional()
