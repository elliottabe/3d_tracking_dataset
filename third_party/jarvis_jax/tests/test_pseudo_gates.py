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
    assert not r.fly[1, 10]              # the spike reverts after frame 9, so frame 10's
                                          # incoming step is ALSO large -- rejected too
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
    admit = np.ones(20, bool); admit[19] = False          # kills frame 19 as a partner ENDPOINT
    p = partner_masks(admit, (1, 4, 16))
    # each anchor whose partner IS frame 19 loses the pair, at every delta
    assert not p[1][18]                                   # anchor 18, delta=1 -> partner 19
    assert not p[4][15]                                   # anchor 15, delta=4 -> partner 19
    assert not p[16][3]                                   # anchor 3,  delta=16 -> partner 19
    # the neighbouring anchor one frame earlier is unaffected (its partner isn't 19)
    assert p[1][17] and p[4][14] and p[16][2]


def test_anchors_decorrelated_alone_partners_available_at_every_anchor(tmp_path):
    """Anchors and partners are separate: anchors are thinned to >= 16 frames
    apart with NO exemption for partner frames (partners never chain into
    new anchors), while every anchor still has a usable T=2 partner (forward
    or backward) at each configured delta, derived straight from
    `partners[delta]` -- exactly how Task 2 is expected to read them."""
    r, _ = _gates(tmp_path, T=40)
    anchors = np.flatnonzero(r.frame).tolist()
    assert anchors == [0, 16, 32]
    assert np.diff(anchors).min() >= 16                  # decorrelation spacing invariant holds
    T = r.fly.shape[1]
    for a in anchors:
        for d in (1, 4, 16):
            fwd = a + d < T and bool(r.partners[d][a])
            bwd = a - d >= 0 and bool(r.partners[d][a - d])
            assert fwd or bwd, f"anchor {a} has no delta={d} partner in either direction"


def test_identity_gate_rejects_mixed_frames_individually(tmp_path):
    """`identity_source` can vary frame-by-frame within one bout: only the
    specific frames with a non-mask head are rejected, not the whole bout."""
    T = 20
    identity = ["mask"] * T
    identity[3] = "sex"; identity[14] = "sex"
    r, _ = _gates(tmp_path, T=T, identity=identity)
    assert r.reasons["identity"][3] and r.reasons["identity"][14]
    others = [t for t in range(T) if t not in (3, 14)]
    assert not r.reasons["identity"][others].any()
    assert not r.fly[:, 3].any() and not r.fly[:, 14].any()
    assert r.fly[:, others].all()
