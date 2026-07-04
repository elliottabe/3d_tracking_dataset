import os
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

CFG = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"


def test_detector_finetune_config_resolves():
    os.environ.setdefault("USER", "eabe")
    import stac_mjx  # noqa: F401  (registers the multirun_save_dir resolver)
    with initialize_config_dir(version_base=None, config_dir=CFG):
        c = compose(config_name="detector_finetune")
        d = OmegaConf.to_container(c, resolve=True)
    assert d["gate"]["consensus_px"] == 25.0
    assert d["val_recordings"]
    # regression: unquoted 2026_05_27_11_56_05 parses as a YAML int
    # (underscore digit-separators) unless quoted in the config -- these must
    # stay strings for V3Dataset(..., recordings=...) matching.
    assert all(isinstance(r, str) for r in d["val_recordings"])
    assert "2026_05_27_11_56_05" in d["val_recordings"]
    assert d["out_dir"].endswith("v4_kp_silbootstrap/final")
    assert d["real_root"].endswith("red_data_unified_V3")   # resolves via paths.red_data_v3_root
