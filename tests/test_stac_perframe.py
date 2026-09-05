"""Per-frame multi-start STAC IK: the pure pieces (warm start, chained starts,
wing-weighted Viterbi penalty, h5 schema). Solver behaviour is pinned in
stac-mjx's tests; the end-to-end run is validated on bout 28 in
docs/benchmark/2026-09-04-wing-ik-spike/notes.md."""
from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from jarvis_jax.tracking.stac_perframe import (
    STAC_H5_KEYS,
    chained_starts,
    qvel_from_qpos,
    warm_start,
    wing_dof_weights,
    write_stac_h5,
)
from jarvis_jax.tracking.stac_polish import dof_mask, viterbi_select

NQ_NAMES = ["root"] * 7 + ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                           "wing_yaw_right", "wing_roll_right", "wing_pitch_right",
                           "abdomen", "coxa_T1_left", "femur_T1_left"]
KP = ["Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "T1L_FeTi"]


def test_warm_start_puts_root_on_the_root_keypoint_and_orients_from_trunk():
    T = 3
    kp = np.zeros((T, len(KP), 3))
    kp[:, KP.index("Scutellum")] = [0, 0, 0]; kp[:, KP.index("Antenna_Base")] = [1, 0, 0]
    kp[:, KP.index("WingL_base")] = [0, 0.3, 0]; kp[:, KP.index("WingR_base")] = [0, -0.3, 0]
    kp[:, KP.index("T1L_FeTi")] = [0.2, 0.2, -0.1]
    kp[1, :, 0] += 5.0                                   # frame 1 translated in x
    cfg = OmegaConf.create({"ROOT_OPTIMIZATION_KEYPOINT": "Scutellum",
                            "JAXLS_ORIENTATION_KEYPOINTS": {"rear": "Scutellum", "left": "WingL_base",
                                                            "right": "WingR_base", "front": "Antenna_Base"}})
    q_base = np.zeros(len(NQ_NAMES)); q_base[3] = 1.0; q_base[7] = 0.7   # a non-zero hinge in qpos0 must survive
    q = warm_start(kp.reshape(T, -1), KP, cfg, q_base)
    assert q.shape == (T, len(NQ_NAMES))
    np.testing.assert_allclose(q[:, :3], kp[:, KP.index("Scutellum")])
    np.testing.assert_allclose(np.linalg.norm(q[:, 3:7], axis=1), 1.0, atol=1e-6)
    # fly along +x, left along +y: body frame == world frame -> identity quaternion
    np.testing.assert_allclose(np.abs(q[0, 3]), 1.0, atol=1e-6)
    assert np.all(q[:, 7] == 0.7) and np.all(q[:, 8:] == 0.0)


def test_chained_starts_reset_only_the_named_pitch_from_the_solved_pose():
    T = 4; nq = len(NQ_NAMES)
    q = np.random.default_rng(1).normal(size=(T, nq))
    spring = np.zeros(nq); ipl, ipr = NQ_NAMES.index("wing_pitch_left"), NQ_NAMES.index("wing_pitch_right")
    spring[ipl] = spring[ipr] = -1.0
    s = chained_starts(q, NQ_NAMES, spring, ["wing_rest_left", "wing_rest_right", "wing_rest_both"])
    other = np.ones(nq, bool); other[[ipl, ipr]] = False
    for v in s.values():
        np.testing.assert_array_equal(v[:, other], q[:, other])          # everything else is the SOLVED pose
    assert np.all(s["wing_rest_left"][:, ipl] == -1.0) and np.all(s["wing_rest_left"][:, ipr] == q[:, ipr])
    assert np.all(s["wing_rest_right"][:, ipr] == -1.0) and np.all(s["wing_rest_right"][:, ipl] == q[:, ipl])
    assert np.all(s["wing_rest_both"][:, [ipl, ipr]] == -1.0)
    with pytest.raises(ValueError, match="unknown start"):
        chained_starts(q, NQ_NAMES, spring, ["zero"])


def test_wing_weighted_viterbi_penalises_a_wing_flip_more_than_the_same_leg_motion():
    nq = len(NQ_NAMES); T = 30; m = np.ones(nq, bool)
    w = wing_dof_weights(NQ_NAMES, 20.0)
    assert w[NQ_NAMES.index("wing_pitch_right")] == 20.0 and w[NQ_NAMES.index("femur_T1_left")] == 1.0
    # candidates A/B differ by 0.5 rad on ONE dof; costs favour alternating by a hair
    costs = np.full((2, T), 1e-3); costs[0, ::2] -= 1e-5; costs[1, 1::2] -= 1e-5
    qA = np.zeros((T, nq)); qB_wing = qA.copy(); qB_wing[:, NQ_NAMES.index("wing_pitch_right")] = 0.5
    qB_leg = qA.copy(); qB_leg[:, NQ_NAMES.index("femur_T1_left")] = 0.5
    # a weight at which the LEG flip is still worth it (penalty 0.25*w_sw < 1e-5) but the WING flip is not
    w_sw = 1e-5   # leg flip penalty 0.25*w_sw = 2.5e-6 < the 5e-6/frame gain of alternating; wing: 20x that
    idx_leg = viterbi_select(costs, [qA, qB_leg], m, w_sw, dof_weights=w)
    idx_wing = viterbi_select(costs, [qA, qB_wing], m, w_sw, dof_weights=w)
    assert np.sum(idx_leg[1:] != idx_leg[:-1]) == T - 1        # leg: flips every frame (cheap)
    assert np.sum(idx_wing[1:] != idx_wing[:-1]) == 0          # wing: held (20x more expensive)


def test_write_stac_h5_matches_the_pipeline_schema(tmp_path):
    import h5py
    T, nq, K, nb = 3, len(NQ_NAMES), len(KP), 4
    p = tmp_path / "stac_ik.h5"
    write_stac_h5(str(p), cfg=OmegaConf.create({"a": 1}), kp_names=KP, names_qpos=NQ_NAMES, names_xpos=[f"b{i}" for i in range(nb)],
                  kp_data=np.zeros((T, K * 3)), marker_sites=np.zeros((T, K, 3)), offsets=np.zeros((K, 3)),
                  qpos=np.zeros((T, nq)), xpos=np.zeros((T, nb, 3)), xquat=np.zeros((T, nb, 4)), qvel=np.zeros((T, nq - 1)),
                  extra={"ik_iterations": np.arange(T)}, attrs={"ik_solver": "per_frame"})
    with h5py.File(p) as f:
        assert set(STAC_H5_KEYS) <= set(f.keys())
        assert f["qpos"].dtype == np.float32 and f["kp_names"].dtype.kind == "S"
        assert [n.decode() for n in f["names_qpos"][:]] == NQ_NAMES
        assert f.attrs["ik_solver"] == "per_frame" and "ik_iterations" in f


def test_qvel_from_qpos_matches_stac_mjx_reference():
    """The numpy twin must reproduce stac_mjx.utils.compute_velocity_from_kinematics
    (which loops per frame in JAX; 118 s for 1500 frames) to float32 precision."""
    import jax.numpy as jnp
    from stac_mjx import utils
    rng = np.random.default_rng(0); T, nq = 40, 15
    q = rng.normal(scale=0.3, size=(T, nq)).cumsum(0); q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    ref = np.asarray(utils.compute_velocity_from_kinematics(jnp.asarray(q), dt=0.00125, freejoint=True))
    mine = qvel_from_qpos(q, 0.00125, True)
    assert mine.shape == ref.shape == (T, nq - 1)
    np.testing.assert_allclose(mine, ref, rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(qvel_from_qpos(q, 0.00125, False), np.clip(np.diff(np.vstack([q, q[-1:]]), axis=0) / 0.00125, -20, 20), atol=1e-9)
