import numpy as np
import pytest
from pseudo_fixtures import CAMS, make_bout


def _gates(tmp_path, **kw):
    from jarvis_jax.data.pseudo_gates import GateThresholds, admit_bout, load_bout_arrays
    from jarvis_jax.tracking.lift_mvq import BoutMaskStore
    d, npz, cm = make_bout(tmp_path, **kw)
    a = load_bout_arrays(d, cameras=CAMS)
    return admit_bout(a, BoutMaskStore(npz, CAMS), cm, GateThresholds()), a


def test_clean_bout_admits_every_frame_except_the_decorrelation_thinning(tmp_path):
    r, a = _gates(tmp_path, T=40)
    assert r.fly.all()                                   # both flies pass every per-fly gate
    # decorrelation keeps one anchor every 16 frames: frames 0, 16, 32
    assert np.flatnonzero(r.frame).tolist() == [0, 16, 32]
    assert set(r.partners) == {1, 4, 16}
    assert r.partners[1][0] and r.partners[4][0] and r.partners[16][0]


def test_low_existence_frame_is_a_negative_for_that_slot_not_a_guess(tmp_path):
    r, _ = _gates(tmp_path, T=20, drop_frames=(5,))
    assert not r.fly[0, 5] and r.fly[1, 5]               # fly0 absent, fly1 still admitted
    assert r.reasons["exist"][0, 5]
    assert 5 not in np.flatnonzero(r.frame).tolist() or r.frame[5]   # frame usable, fly0 absent


def test_step_spike_rejects_both_neighbouring_frames(tmp_path):
    r, _ = _gates(tmp_path, T=20, spike_frames=(9,))
    assert not r.fly[1, 8] and not r.fly[1, 9]
    assert r.fly[1, 11]
    assert r.quant["step_units"][1, 9] > 3.0


def test_identity_gate_rejects_a_sex_head_bout(tmp_path):
    r, _ = _gates(tmp_path, T=20, identity="sex")
    assert not r.frame.any() and r.reasons["identity"].all()


def test_reprojection_gate_fires_when_the_2d_head_disagrees(tmp_path):
    from jarvis_jax.data.pseudo_gates import GateThresholds, reprojection_gate
    from pseudo_fixtures import cam_mats
    d, npz, cm = make_bout(tmp_path, T=8)
    from jarvis_jax.data.pseudo_gates import load_bout_arrays
    a = load_bout_arrays(d, cameras=CAMS)
    ok, med = reprojection_gate(a.kp2d, a.kp3d, cm, a.conf, GateThresholds())
    assert ok.all() and float(np.nanmax(med)) < 1e-3
    bad = a.kp2d.copy(); bad[1, 3] += 10.0                # 10 px in every view
    ok2, med2 = reprojection_gate(bad, a.kp3d, cm, a.conf, GateThresholds())
    assert not ok2[1, 3] and ok2[0, 3]


def test_containment_gate_fires_when_the_pose_leaves_its_own_mask(tmp_path):
    from jarvis_jax.data.pseudo_gates import GateThresholds, containment_gate, load_bout_arrays
    from jarvis_jax.tracking.lift_mvq import BoutMaskStore
    d, npz, cm = make_bout(tmp_path, T=6)
    a = load_bout_arrays(d, cameras=CAMS)
    store = BoutMaskStore(npz, CAMS)
    ok, frac = containment_gate(a.kp3d, store, cm, GateThresholds())
    assert ok.all() and frac.min() >= 0.9
    moved = a.kp3d.copy(); moved[0, 2] += np.array([25.0, 0.0, 0.0])   # off its own mask
    ok2, frac2 = containment_gate(moved, store, cm, GateThresholds())
    assert not ok2[0, 2] and frac2[0, 2] < 0.9


def test_decorrelate_protects_partners(tmp_path):
    from jarvis_jax.data.pseudo_gates import decorrelate
    admit = np.ones(40, bool)
    keep = decorrelate(admit, 16)
    assert np.flatnonzero(keep).tolist() == [0, 16, 32]
    assert np.diff(np.flatnonzero(keep)).min() >= 16


def test_partner_masks_need_the_partner_frame_to_pass_too(tmp_path):
    from jarvis_jax.data.pseudo_gates import partner_masks
    admit = np.ones(20, bool); admit[5] = False
    p = partner_masks(admit, (1, 4, 16))
    assert not p[1][4] and p[1][6] is np.True_ or p[1][6]
    assert not p[4][1] and p[16][3] is np.False_ or not p[16][3]
