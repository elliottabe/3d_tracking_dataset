"""Hydra helpers shared by all jarvis_jax entrypoints."""
import os
import dataclasses
from omegaconf import OmegaConf

# configs/ is a sibling of the jarvis_jax package dir:
#   third_party/jarvis_jax/jarvis_jax/hydra_utils.py -> third_party/jarvis_jax/configs
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def register_resolvers() -> None:
    """Register jarvis_jax OmegaConf resolvers (idempotent)."""
    OmegaConf.register_new_resolver(
        "eq", lambda x, y: str(x).lower() == str(y).lower(),
        use_cache=False, replace=True)
    OmegaConf.register_new_resolver(
        "divide", lambda x, y: x // y, use_cache=False, replace=True)
    OmegaConf.register_new_resolver(
        "resolve_default", lambda default, arg: default if arg == "" else arg,
        use_cache=False, replace=True)


def build_dataclass(dc_type, cfg_node, extra: dict | None = None):
    """Construct a frozen dataclass from a config node, dropping unknown keys."""
    fields = {f.name for f in dataclasses.fields(dc_type)}
    data = {k: v for k, v in OmegaConf.to_container(cfg_node, resolve=True).items()
            if k in fields}
    if extra:
        data.update({k: v for k, v in extra.items() if k in fields})
    return dc_type(**data)


def run_dir_for(cfg) -> str:
    """Absolute run directory: <runs_root>/<run_id>."""
    return os.path.join(cfg.paths.runs_root, cfg.run_id)
