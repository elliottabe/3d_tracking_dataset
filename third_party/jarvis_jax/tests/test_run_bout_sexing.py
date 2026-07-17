import os
import sys
import numpy as np
import pytest
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

KP = ["Scutellum", "Abd_tip", "WingL_base", "WingL_V13", "WingR_base", "WingR_V13"]


def _make(T, wing_osc):
    K = len(KP)
    kp = np.zeros((T, K, 3), float)
    conf = np.ones((T, K), float)
    kp[:, 1] = [1, 0, 0]                                   # Abd_tip
    kp[:, 4] = [0.5, 0, 0]; kp[:, 5] = [1.4, 0.02, 0]      # WingR static
    t = np.linspace(0, 4 * np.pi, T)
    ang = np.radians(45.0 + wing_osc * np.sin(t))
    kp[:, 2] = [0.5, 0, 0]
    kp[:, 3, 0] = 0.5 + np.cos(ang); kp[:, 3, 1] = np.sin(ang)
    return kp, conf


def _write_bout(bout_dir, male_fly, T=200):
    for fly in (0, 1):
        d = os.path.join(bout_dir, f"fly{fly}"); os.makedirs(d)
        kp, c = _make(T, 40 if fly == male_fly else 1)
        np.savez(os.path.join(d, "kp3d.npz"), kp3d=kp, conf3d=c)
        open(os.path.join(d, f"orig_fly{fly}"), "w").close()


def test_canonicalize_bout_sex_swaps_and_skips_single_animal(tmp_path):
    import run_bout
    out = str(tmp_path / "pose")
    os.makedirs(os.path.join(out, "bouts"))
    bdir = os.path.join(out, "bouts", "bout_00001")
    _write_bout(bdir, male_fly=0)                          # male currently fly0
    cfg = OmegaConf.create({
        "outputs": {"out": out},
        "recording": {"predictions_dir": str(tmp_path / "preds"), "num_animals": 2},
        "model": {"KP_NAMES": KP},
        "sexing": {"ratio_thr": 1.5, "high_ratio": 2.5, "conf_min": 0.2, "min_frames": 20},
    })
    run_bout._canonicalize_bout_sex(cfg, 1)
    assert os.path.exists(os.path.join(bdir, "fly1", "orig_fly0"))   # swapped
    assert os.path.exists(os.path.join(bdir, "sex.json"))
