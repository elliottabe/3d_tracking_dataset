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
