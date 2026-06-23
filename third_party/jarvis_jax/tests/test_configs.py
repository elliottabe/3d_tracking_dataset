"""Hydra config composition tests (CPU-only, no model build)."""
import os
import dataclasses
import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass

register_resolvers()


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


def test_base_config_resolves():
    cfg = _compose(["paths=hyak"])
    # Full resolution must not raise (catches missing interpolations).
    OmegaConf.resolve(cfg)
    assert cfg.paths.runs_root
    assert cfg.paths.cache_dir
    assert cfg.run_id


def test_build_dataclass_filters_unknown_keys():
    @dataclasses.dataclass(frozen=True)
    class C:
        a: int = 1
        b: int = 2
    node = OmegaConf.create({"a": 10, "b": 20, "extra": 99})
    out = build_dataclass(C, node)
    assert out == C(a=10, b=20)


def test_model_data_cache_groups_resolve():
    cfg = _compose(["paths=hyak", "model=hybridnet", "data=v3", "cache=default"])
    OmegaConf.resolve(cfg)
    assert cfg.model.roi_cube == 48
    assert cfg.model.grid_spacing == 1
    assert cfg.model.num_cameras == 7
    assert cfg.model.sharpen == 3.0
    assert cfg.model.vitpose.num_keypoints == 50
    assert cfg.data.num_joints == 50
    assert cfg.cache.split in ("train", "val")


def test_model_vitpose_standalone_resolves():
    # model=vitpose puts the ViT fields directly at cfg.model.* (for 2D training).
    cfg = _compose(["paths=hyak", "model=vitpose"])
    OmegaConf.resolve(cfg)
    assert cfg.model.num_keypoints == 50
    assert cfg.model.img_size == 448


def test_model_hybridnet_composes_shared_vit():
    # model=hybridnet composes the SAME vitpose config, mounted under model.vitpose,
    # plus v2vnet + 3D params — one ViT definition, no duplication.
    cfg = _compose(["paths=hyak", "model=hybridnet"])
    OmegaConf.resolve(cfg)
    assert cfg.model.roi_cube == 48
    assert cfg.model.grid_spacing == 1
    assert cfg.model.num_cameras == 7
    assert cfg.model.sharpen == 3.0
    assert cfg.model.vitpose.num_keypoints == 50
    assert cfg.model.vitpose.img_size == 448
    assert cfg.model.v2vnet.in_channels == 50
