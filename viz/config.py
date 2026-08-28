# viz/config.py
"""Resolve recording paths + KP_NAMES for the courtship pipeline via Hydra."""
import os
from hydra import initialize_config_dir, compose

# Registers the `basename` OmegaConf resolver (on import) that
# configs/outputs/default.yaml AND recording/session{0,1}.yaml's
# predictions_dir depend on. Without it, composing `pipeline` here raises
# UnsupportedInterpolationType -- viz composes its own config independently of
# scripts/run_bout.py, which registers the resolver itself.
from utils import path_utils as _path_utils  # noqa: F401

_CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

def courtship_recording(config_name="pipeline", overrides=None):
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


def resolve_body_model_xml(path):
    """Map a body-model path recorded in an old run onto the current checkout.

    Runs embed `cfg.model.MJCF_PATH` at solve time, and the body models moved
    out of `<repo>/models/` into the SIBLING `fruitfly_body_models` clone -- so
    a run from before the move points at a path that no longer exists. Same
    fallback as scripts/estimate_recording_scale.py and
    scripts/figures/export_fig4_bundle.py. Returns the path unchanged when it
    exists, so a current run is unaffected.
    """
    import os
    p = str(path)
    if os.path.exists(p):
        return p
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sib = os.path.join(os.path.dirname(repo), "fruitfly_body_models")
    marker = os.sep + "models" + os.sep
    if marker in p.replace("//", "/"):
        tail = p.replace("//", "/").split(marker, 1)[1]
        cand = os.path.join(sib, tail)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"body model XML not found: {path!r} (also tried the sibling checkout "
        f"{sib!r}). The models moved out of <repo>/models/; see the "
        f"body-model checkout notes.")
