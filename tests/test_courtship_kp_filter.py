"""Tests for the courtship pre-STAC keypoint filter wrapper
(jarvis_jax.cse.courtship_filter): it must build sane skeleton edges, be a
no-op when disabled, and remove a foot-tip spike while preserving shape."""
import numpy as np
import pytest
from omegaconf import OmegaConf

from jarvis_jax.cse.courtship_filter import (
    courtship_skeleton_edges, filter_bout_kp3d,
)

# minimal but realistic KP order: body + one full leg chain
KP_NAMES = [
    "Scutellum", "Antenna_Base", "EyeL", "EyeR", "Abd_A4", "Abd_tip",
    "WingL_base", "WingL_V12", "WingL_V13",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
]
TIP = KP_NAMES.index("T1L_TaTip")


def _full_cfg(enabled=True):
    return OmegaConf.create({
        "enabled": enabled,
        "confidence": {"enabled": True, "threshold": 0.3, "exclude_keypoint_patterns": ["Wing"]},
        "bone_length": {"enabled": True, "threshold_std": 5.0, "exclude_keypoint_patterns": ["Wing"]},
        "centroid_jump": {"enabled": False},
        "identity_relink": {"enabled": False},
        "isolated_spike": {"enabled": True, "threshold_factor": 10.0, "max_iterations": 1},
        "medfilt_despike": {"enabled": False},
        "medfilt": {"enabled": False},
        "interpolation": {"enabled": True, "use_spline": True,
                          "max_edge_extrap_frames": 5, "edge_fit_window": 5},
        "confidence_smooth": {"enabled": False},
        "savgol": {"enabled": True, "window": 7, "polyorder": 2},
    })


def _smooth_bout(T=60, seed=0):
    """Smoothly drifting keypoints (no spikes)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-2, 2, size=(len(KP_NAMES), 3))
    t = np.linspace(0, 2 * np.pi, T)[:, None, None]
    drift = 0.3 * np.sin(t + rng.uniform(0, 1, size=(1, len(KP_NAMES), 1)))
    return (base[None] + drift).astype(np.float64)


def test_skeleton_edges_cover_leg_and_body():
    edges = courtship_skeleton_edges(KP_NAMES)
    assert edges.ndim == 2 and edges.shape[1] == 2
    pairs = {tuple(e) for e in edges}
    i = {n: k for k, n in enumerate(KP_NAMES)}
    # a leg chain link and a leg-root->Scutellum link must be present
    assert (i["T1L_TaT3"], i["T1L_TaTip"]) in pairs
    assert (i["Scutellum"], i["T1L_ThxCx"]) in pairs
    # no self edges, all indices in range
    assert all(a != b for a, b in pairs)
    assert edges.max() < len(KP_NAMES) and edges.min() >= 0


def test_disabled_is_noop_copy():
    kp = _smooth_bout()
    out = filter_bout_kp3d(kp, np.ones(kp.shape[:2]), KP_NAMES, _full_cfg(enabled=False))
    assert out.shape == kp.shape
    np.testing.assert_allclose(out, kp)
    assert out is not kp  # a copy, safe to mutate


def test_removes_foot_tip_spike_preserves_shape():
    kp = _smooth_bout()
    clean_tip = kp[:, TIP, :].copy()
    kp[30, TIP, :] += np.array([50.0, -40.0, 30.0])  # single-frame outlier that reverses
    conf = np.ones(kp.shape[:2])
    out = filter_bout_kp3d(kp, conf, KP_NAMES, _full_cfg())

    assert out.shape == kp.shape
    assert np.isfinite(out).all()
    # the spike frame must be pulled back near the underlying smooth trajectory
    resid_before = np.linalg.norm(kp[30, TIP] - clean_tip[30])
    resid_after = np.linalg.norm(out[30, TIP] - clean_tip[30])
    assert resid_before > 40.0
    assert resid_after < 5.0, f"spike not removed: residual {resid_after:.2f} mm"
    # frame-to-frame foot-tip acceleration must drop sharply
    acc = lambda a: np.nanmax(np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=-1))
    assert acc(out[:, TIP]) < 0.5 * acc(kp[:, TIP])


def test_coverage_never_below_raw_input():
    """Smoothing must never delete a keypoint the raw triangulation had.
    A bone-length outlier in a TRAILING run (which interpolation cannot
    extrapolate) must fall back to the raw value, not be left NaN -- otherwise
    it would NaN the STAC fit / bridge for that frame."""
    kp = _smooth_bout(T=60)
    # stretch the foot tip far on the last 4 frames -> bone-length outliers in a
    # trailing run the edge-extrapolation cap (5) still can't safely fill.
    kp[-4:, TIP, :] += np.array([30.0, 0.0, 0.0])
    conf = np.ones(kp.shape[:2])
    out = filter_bout_kp3d(kp, conf, KP_NAMES, _full_cfg())

    raw_finite = np.isfinite(kp).all(-1)      # (T, K)
    out_finite = np.isfinite(out).all(-1)
    # every keypoint finite in the raw input stays finite after filtering
    assert (out_finite[raw_finite]).all(), "filtering dropped a keypoint the input had"
    # and the last frame's foot tip specifically is finite (not a NaN edge run)
    assert np.isfinite(out[-1, TIP]).all()


def test_wings_preserve_raw_motion_but_legs_still_smoothed():
    """preserve_raw_patterns=['Wing'] must leave wing keypoints byte-for-byte at
    the raw triangulation (fast wing song is real, not noise) while legs still
    get smoothed."""
    WING = KP_NAMES.index("WingL_V12")
    rng = np.random.default_rng(1)
    kp = _smooth_bout(T=60)
    # rapid wing oscillation (real high-freq motion) + noisy leg tip
    t = np.arange(60)
    kp[:, WING, 0] += 4.0 * np.sin(t * 1.7)            # fast wing beat
    kp[:, TIP, :] += rng.normal(0, 0.6, size=(60, 3))  # leg jitter
    conf = np.ones(kp.shape[:2])
    cfg = _full_cfg()
    cfg.preserve_raw_patterns = ["Wing"]
    out = filter_bout_kp3d(kp, conf, KP_NAMES, cfg)

    # wing keypoint is untouched (raw preserved) -> full motion kept
    np.testing.assert_allclose(out[:, WING], kp[:, WING], atol=1e-9)
    spd = lambda a: np.median(np.linalg.norm(np.diff(a, axis=0), axis=-1))
    assert spd(out[:, WING]) > 0.9 * spd(kp[:, WING])   # wing motion preserved
    # leg tip IS smoothed (jitter reduced)
    assert spd(out[:, TIP]) < 0.8 * spd(kp[:, TIP])
