from pathlib import Path
from omegaconf import OmegaConf

# Create a parent config with paths defined
_parent_cfg = OmegaConf.create({
    "paths": {
        "body_model_dir": "/gscratch/portia/eabe/Research/MyRepos/fly_neuromech/fruitfly_body_models"
    }
})

# Monkey-patch OmegaConf.load to handle interpolations from the anatomy configs
_original_load = OmegaConf.load

def _patched_load(config_path):
    """Load config and merge with parent to resolve interpolations."""
    cfg = _original_load(config_path)
    # Check raw value (without resolving) to see if it has interpolations
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(raw_cfg, dict) and "mjcf_path" in raw_cfg and "${paths" in str(raw_cfg.get("mjcf_path", "")):
        # Merge the loaded config with parent that has paths defined
        cfg = OmegaConf.merge(_parent_cfg, {"_": cfg})["_"]
    return cfg

OmegaConf.load = _patched_load
