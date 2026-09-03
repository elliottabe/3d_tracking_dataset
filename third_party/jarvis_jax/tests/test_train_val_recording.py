"""train.val_recording must reach run_training.

Before this key existed, run_training's per-cohort val recording was a
hardcoded default (2026_05_27_11_56_05). On the v12 root that recording was
re-filed under its true capture id, so the cohort matched nothing; the whole
v12_bal_maskoff run logged 'female ...: 0.000px', and after the empty-cohort
guard landed the same launch would RAISE at dataset construction. The key
has to be plumbed from the hydra config, not left to the default.
"""
import os
from hydra import initialize_config_dir, compose

from jarvis_jax.scripts import train_keypoints as tk

CFG_DIR = os.path.join(os.path.dirname(tk.__file__), "..", "..", "configs")


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(CFG_DIR)):
        return compose(config_name="config", overrides=overrides)


def test_val_recording_override_reaches_run_training(monkeypatch):
    seen = {}

    def fake_run_training(root, **kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(tk, "run_training", fake_run_training)
    cfg = _compose(["model=vitpose", "train=vit2d", "paths=hyak",
                    "run_id=t", "paths.runs_root=/tmp/x", "paths.data_root=/tmp/y",
                    "train.val_recording='2026_06_09_15_46_55'"])
    tk.main_from_cfg(cfg)
    assert seen["val_recording"] == "2026_06_09_15_46_55"


def test_val_recording_default_is_declared_in_config():
    cfg = _compose(["model=vitpose", "train=vit2d", "paths=hyak",
                    "run_id=t", "paths.runs_root=/tmp/x", "paths.data_root=/tmp/y"])
    assert "val_recording" in cfg.train


def test_unquoted_val_recording_is_refused():
    import pytest
    cfg = _compose(["model=vitpose", "train=vit2d", "paths=hyak",
                    "run_id=t", "paths.runs_root=/tmp/x", "paths.data_root=/tmp/y",
                    "train.val_recording=2026_06_09_15_46_55"])   # parses as an int
    with pytest.raises(ValueError, match="quote"):
        tk.main_from_cfg(cfg)
