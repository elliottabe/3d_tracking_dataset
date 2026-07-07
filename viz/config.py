# viz/config.py
"""Resolve recording paths + KP_NAMES for the courtship pipeline via Hydra."""
import os
from hydra import initialize_config_dir, compose

_CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

def courtship_recording(config_name="courtship_pipeline", overrides=None):
    os.environ.setdefault("USER", "eabe")
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(_CFG_DIR)):
        c = compose(config_name=config_name, overrides=overrides or [])
    return {
        "calib_dir": str(c.recording.calib_dir),
        "session_dir": str(c.recording.session_dir),
        "predictions_dir": str(c.recording.predictions_dir),
        "cameras": list(c.recording.cameras),
        "kp_names": list(c.model.KP_NAMES),
    }
