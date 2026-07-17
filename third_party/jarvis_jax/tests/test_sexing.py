import numpy as np
from jarvis_jax.tracking import sexing

KP = ["Scutellum", "Abd_tip", "WingL_base", "WingL_V13", "WingR_base", "WingR_V13"]
IDX = {n: i for i, n in enumerate(KP)}


def _make(T, wing_osc):
    """(T,K,3) kp3d + (T,K) conf. Body axis +x. Left wing angle = 45 +/- wing_osc deg
    (oscillating); right wing static folded near the body axis."""
    K = len(KP)
    kp = np.zeros((T, K, 3), float)
    conf = np.ones((T, K), float)
    kp[:, IDX["Scutellum"]] = [0, 0, 0]
    kp[:, IDX["Abd_tip"]] = [1, 0, 0]
    kp[:, IDX["WingR_base"]] = [0.5, 0, 0]
    kp[:, IDX["WingR_V13"]] = [1.4, 0.02, 0]          # ~1.3 deg from +x, static
    t = np.linspace(0, 4 * np.pi, T)
    ang = np.radians(45.0 + wing_osc * np.sin(t))
    kp[:, IDX["WingL_base"]] = [0.5, 0, 0]
    kp[:, IDX["WingL_V13"], 0] = 0.5 + np.cos(ang)    # wing vec = [cos(ang), sin(ang), 0]
    kp[:, IDX["WingL_V13"], 1] = np.sin(ang)
    return kp, conf


def test_wing_song_cv_oscillating_gt_static():
    kp_m, c = _make(200, wing_osc=40)     # singing male
    kp_f, _ = _make(200, wing_osc=1)      # static female
    cv_m = sexing.wing_song_cv(kp_m, c, IDX)
    cv_f = sexing.wing_song_cv(kp_f, c, IDX)
    assert cv_m > cv_f
    assert cv_m > 0.05


def test_wing_song_cv_min_frames_and_low_conf():
    kp, c = _make(10, wing_osc=40)
    assert np.isnan(sexing.wing_song_cv(kp, c, IDX, min_frames=20))   # too few frames
    kp, c = _make(200, wing_osc=40)
    c[:, IDX["WingL_V13"]] = 0.0
    c[:, IDX["WingR_V13"]] = 0.0                                       # both tips low-conf
    assert np.isnan(sexing.wing_song_cv(kp, c, IDX, conf_min=0.2))


def test_sex_decision_clear_high():
    d = sexing.sex_bout_from_pose(0.10, 0.40)         # ratio 4.0
    assert d["male_fly"] == 1 and d["confidence"] == "high" and d["method"] == "pose_wing_cv"


def test_sex_decision_clear_medium():
    d = sexing.sex_bout_from_pose(0.20, 0.36)         # ratio 1.8
    assert d["male_fly"] == 1 and d["confidence"] == "medium"


def test_sex_decision_male_fly0():
    d = sexing.sex_bout_from_pose(0.40, 0.10)         # fly0 higher
    assert d["male_fly"] == 0 and d["confidence"] == "high"


def test_sex_decision_weak_with_mask_fallback():
    d = sexing.sex_bout_from_pose(0.15, 0.16, mask_sex_meta={"song_cv": [0.1, 0.2]})  # ratio ~1.07
    assert d["male_fly"] == 1 and d["confidence"] == "low" and d["method"] == "mask_fallback"


def test_sex_decision_weak_no_mask_unknown():
    d = sexing.sex_bout_from_pose(0.15, 0.16, mask_sex_meta=None)
    assert d["male_fly"] is None and d["confidence"] == "unknown" and d["method"] == "unresolved"


import json
import os


def _write_bout(bout_dir, male_fly, T=200):
    """Create bout_dir/fly0,fly1 each with kp3d.npz (male fly has oscillating wing)
    plus an 'orig_flyN' marker file so tests can tell which physical dir moved."""
    for fly in (0, 1):
        d = os.path.join(bout_dir, f"fly{fly}")
        os.makedirs(d)
        osc = 40 if fly == male_fly else 1
        kp, c = _make(T, wing_osc=osc)
        np.savez(os.path.join(d, "kp3d.npz"), kp3d=kp, conf3d=c)
        open(os.path.join(d, f"orig_fly{fly}"), "w").close()
    return bout_dir


def test_canonicalize_swaps_male_to_fly1(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=0)     # male currently fly0
    res = sexing.canonicalize_bout(bd, KP)
    assert res["applied_swap"] is True
    assert res["male_fly"] == 1 and res["original_male_fly"] == 0
    assert os.path.exists(os.path.join(bd, "fly1", "orig_fly0"))   # was fly0, now fly1
    assert os.path.exists(os.path.join(bd, "fly0", "orig_fly1"))
    assert json.load(open(os.path.join(bd, "sex.json")))["male_fly"] == 1


def test_canonicalize_no_swap_when_male_already_fly1(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=1)
    res = sexing.canonicalize_bout(bd, KP)
    assert res["applied_swap"] is False and res["male_fly"] == 1
    assert os.path.exists(os.path.join(bd, "fly1", "orig_fly1"))


def test_canonicalize_idempotent(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=0)
    sexing.canonicalize_bout(bd, KP)                 # swaps
    res2 = sexing.canonicalize_bout(bd, KP)          # male now fly1 -> no-op
    assert res2["applied_swap"] is False and res2["male_fly"] == 1


def test_canonicalize_dry_run_moves_nothing(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=0)
    res = sexing.canonicalize_bout(bd, KP, dry_run=True)
    assert res["applied_swap"] is True and res["original_male_fly"] == 0
    assert os.path.exists(os.path.join(bd, "fly0", "orig_fly0"))    # NOT moved
    assert not os.path.exists(os.path.join(bd, "sex.json"))         # NOT written


def test_canonicalize_missing_kp3d_unknown(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=0)
    os.remove(os.path.join(bd, "fly1", "kp3d.npz"))
    res = sexing.canonicalize_bout(bd, KP)
    assert res["confidence"] == "unknown" and res["applied_swap"] is False


def test_swap_recovers_partial_after_step1(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=1)
    f0 = os.path.join(bd, "fly0")
    tmp = os.path.join(bd, ".fly_swap_tmp")
    os.rename(f0, tmp)                    # simulate crash after step 1 (fly0->tmp done)
    sexing._swap_fly_dirs(bd)            # recover + complete
    assert os.path.exists(os.path.join(bd, "fly0", "orig_fly1"))
    assert os.path.exists(os.path.join(bd, "fly1", "orig_fly0"))
    assert not os.path.exists(tmp)


def test_swap_recovers_partial_after_step2(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=1)
    f0 = os.path.join(bd, "fly0")
    f1 = os.path.join(bd, "fly1")
    tmp = os.path.join(bd, ".fly_swap_tmp")
    os.rename(f0, tmp)                    # step 1: fly0 -> tmp
    os.rename(f1, f0)                     # step 2: fly1 -> fly0  (fly1 now missing)
    sexing._swap_fly_dirs(bd)            # recover: tmp -> fly1
    assert os.path.exists(os.path.join(bd, "fly1", "orig_fly0"))
    assert os.path.exists(os.path.join(bd, "fly0", "orig_fly1"))
    assert not os.path.exists(tmp)
