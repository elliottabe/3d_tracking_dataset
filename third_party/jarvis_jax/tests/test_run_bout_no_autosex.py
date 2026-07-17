import os, sys
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def test_main_from_cfg_does_not_auto_sex(monkeypatch):
    import run_bout
    calls = []
    monkeypatch.setattr(run_bout, "_log_gpu_env", lambda: None)
    monkeypatch.setattr(run_bout, "resolve_bout_ids", lambda cfg: [1])
    monkeypatch.setattr(run_bout, "process_bout_fly",
                        lambda cfg, b, f: calls.append(("fly", b, f)))
    monkeypatch.setattr(run_bout, "_canonicalize_bout_sex",
                        lambda cfg, b: calls.append(("sex", b)))
    cfg = OmegaConf.create({"recording": {"num_animals": 2}})
    run_bout.main_from_cfg(cfg)
    assert ("fly", 1, 0) in calls and ("fly", 1, 1) in calls   # both flies processed
    assert not any(c[0] == "sex" for c in calls)               # NO auto-sexing
