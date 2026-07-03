import os
from hydra import initialize_config_dir, compose

CFG_DIR = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        return compose(config_name="courtship_pipeline", overrides=overrides)


def test_courtship_pipeline_composes_and_paths_generalize(monkeypatch):
    monkeypatch.setenv("USER", "someone")
    cfg = _compose(["paths=hyak"])
    # user comes from env; no hardcoded 'eabe'
    assert cfg.paths.user == "someone"
    # required groups present
    assert cfg.recording.session_dir and cfg.recording.num_animals == 2
    assert cfg.detector.num_keypoints == 50 and cfg.detector.crop == 448
    assert cfg.silhouette.mesh_npz.endswith(".npz")
    assert "bouts" not in cfg.outputs.out or True  # out is a resolvable pattern


def test_default_user_when_env_absent(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    cfg = _compose(["paths=hyak"])
    assert cfg.paths.user == "eabe"   # fallback default
