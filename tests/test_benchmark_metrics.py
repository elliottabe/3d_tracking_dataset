"""Tests for scripts/benchmark/metrics.py (pure metric functions, synthetic data)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark.metrics import (
    jitter_series, joint_bounds, joint_limit_violation_rate, kp_group,
    proximity_bl, reproj_series_by_group,
)


def test_kp_group():
    assert kp_group("T1L_FeTi") == "leg"
    assert kp_group("T3R_TaT5") == "leg"
    assert kp_group("WingL_V12") == "wing"
    assert kp_group("WingL_base") == "trunk"
    assert kp_group("Scutellum") == "trunk"
    assert kp_group("Abd_tip") == "trunk"
    assert kp_group("Head") == "other"


def test_jitter_series_flags_noisy_legs():
    T = 100
    names = ["free"] * 7 + ["coxa_T1_left", "wing_yaw_left"]
    smooth = np.zeros((T, 9))
    smooth[:, 7] = np.linspace(0, 1, T)          # smooth leg ramp
    noisy = smooth.copy()
    rng = np.random.default_rng(0)
    noisy[:, 7] += rng.normal(0, 0.05, T)         # jittery leg
    noisy[:, 8] += rng.normal(0, 0.5, T)          # noisy WING: must not count
    js, jn = jitter_series(smooth, names), jitter_series(noisy, names)
    assert js.shape == (T - 2,)
    assert np.median(jn) > 10 * max(np.median(js), 1e-12)


def test_joint_limits_synthetic_model():
    import mujoco
    xml = """<mujoco><worldbody><body><joint name="free" type="free"/>
      <body><joint name="hinge_lim" type="hinge" range="-0.5 0.5" limited="true"/>
        <geom size="0.01"/></body><geom size="0.01"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    lb, ub = joint_bounds(m)
    assert lb.shape == (m.nq,)
    assert np.isinf(lb[:7]).all()                 # free joint unbounded
    # MuJoCo converts XML range in degrees to radians internally
    assert lb[7] == pytest.approx(-0.5 * np.pi / 180) and ub[7] == pytest.approx(0.5 * np.pi / 180)
    qpos = np.zeros((10, m.nq))
    qpos[5:, 7] = 0.6                             # 5 of 10 frames beyond limit
    assert joint_limit_violation_rate(qpos, lb, ub) == pytest.approx(0.5)


def test_reproj_series_by_group_zero_for_perfect_projection():
    rng = np.random.default_rng(1)
    T, C, K = 4, 3, 4
    kp_names = ["T1L_FeTi", "T2R_Tro", "WingL_V12", "Scutellum"]
    kp3d = rng.normal(0, 5, (T, K, 3))
    # camera matrices (C,4,3): uv_h = [X 1] @ P
    cams = []
    for c in range(C):
        A = np.eye(3, 4)                          # simple projective rows
        A[2, 3] = 10.0 + c                        # keep depth positive
        cams.append(A.T)                          # (4,3)
    cam_mats = np.stack(cams)
    Xh = np.concatenate([kp3d, np.ones((T, K, 1))], axis=-1)      # (T,K,4)
    uvh = np.einsum("tkf,cfe->tcke", Xh, cam_mats)                # (T,C,K,3)
    kp2d = uvh[..., :2] / uvh[..., 2:3]
    conf = np.ones((T, C, K))
    out = reproj_series_by_group(kp3d, kp2d, conf, cam_mats, kp_names)
    assert set(out) == {"leg", "wing", "trunk"}
    for series in out.values():
        assert series.shape == (T,)
        assert np.nanmax(series) < 1e-6


def test_reproj_low_conf_ignored():
    T, C, K = 2, 2, 2
    kp_names = ["T1L_FeTi", "T1R_FeTi"]
    kp3d = np.zeros((T, K, 3))
    cam_mats = np.stack([np.eye(3, 4).T + [[0], [0], [0], [1e-9]] for _ in range(C)])
    cam_mats[..., 2] = np.array([0, 0, 0, 1.0])   # depth row -> w=1
    kp2d = np.full((T, C, K, 2), 100.0)           # wildly wrong 2D
    conf = np.zeros((T, C, K))                    # ...but zero confidence
    out = reproj_series_by_group(kp3d, kp2d, conf, cam_mats, kp_names)
    assert np.isnan(out["leg"]).all()             # nothing confident -> NaN


def test_proximity_bl_scales_with_distance():
    T, K = 5, 10
    rng = np.random.default_rng(2)
    body = rng.normal(0, 1.0, (T, K, 3))          # spread ~ its own size
    near = body + np.array([1.0, 0, 0])
    far = body + np.array([50.0, 0, 0])
    assert np.median(proximity_bl(body, far)) > 10 * np.median(proximity_bl(body, near))
