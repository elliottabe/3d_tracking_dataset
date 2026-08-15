"""Tests for scripts/benchmark/metrics.py (pure metric functions, synthetic data)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark.metrics import (
    jitter_series, joint_bounds, joint_limit_violation_rate, kp3d_spike_rate,
    kp_group, proximity_bl, reproj_series_by_group,
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


def _spike_fixture(T=100, K=4, body=10.0):
    """Smooth-moving keypoints with a known Antenna_Base<->Abd_tip length."""
    kp_names = ["Antenna_Base", "Abd_tip"] + [f"kp{i}" for i in range(K - 2)]
    kp3d = np.zeros((T, K, 3))
    kp3d[:, :, 0] = np.linspace(0, 5, T)[:, None]     # smooth fast motion
    kp3d[:, 1, 1] = body                              # fixed body length
    return kp3d, kp_names


def test_kp3d_spike_rate_counts_isolated_jumps():
    # Median-style jitter metrics are blind to the tail this measures (the
    # 2026-08-14 jax-vs-jarvis A/B found identical medians but a 5x spike
    # gap): one keypoint jumping for one frame must register, smooth motion
    # must not.
    T, K = 100, 4
    kp3d, kp_names = _spike_fixture(T, K)
    assert kp3d_spike_rate(kp3d, kp_names) == 0.0
    spiky = kp3d.copy()
    spiky[50, 2, 1] += 1.0                            # 0.1 body-length jump
    r = kp3d_spike_rate(spiky, kp_names)
    # a 1-frame jump bends 3 consecutive second differences of that keypoint
    assert np.isclose(r, 3 / ((T - 2) * K))


def test_kp3d_spike_rate_is_unit_free():
    # The frozen benchmark tree triangulates in ~0.1 mm units, the Session6
    # clip in mm: the SAME motion must score the SAME in any unit.
    kp3d, kp_names = _spike_fixture()
    kp3d[50, 2, 1] += 1.0
    assert np.isclose(kp3d_spike_rate(kp3d, kp_names),
                      kp3d_spike_rate(kp3d * 11.3, kp_names))


def test_kp3d_spike_rate_ignores_nan_gaps():
    kp3d, kp_names = _spike_fixture(T=50, K=2)
    kp3d[10:20, 0] = np.nan                           # occlusion gap, no spike
    assert kp3d_spike_rate(kp3d, kp_names) == 0.0


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
