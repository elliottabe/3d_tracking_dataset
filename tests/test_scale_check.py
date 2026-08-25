"""Synthetic tests for scripts/viz/scale_check.py.

No real recordings are used. The only real input is the small, checked-in
v1 MuJoCo XML (skipped if absent).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# Sibling Brunton-Lab/fruitfly_body_models checkout == paths.body_model_dir.
MODEL_XML = REPO_ROOT.parent / "fruitfly_body_models" / "fruitfly_v1" / "fruitfly_v1_free.xml"


def _require_model():
    if not MODEL_XML.exists():
        pytest.skip(f"model xml not found: {MODEL_XML}")


def _kp_names_and_ref():
    """Real v1 KP_NAMES + their rest-pose ``tracking[...]`` site positions
    (NaN where a name has no matching site -- none expected for v1, but keep
    this robust)."""
    import mujoco

    from scripts.estimate_recording_scale import _load_anatomy_cfg

    cfg = _load_anatomy_cfg("configs/anatomy/v1.yaml")
    kp_names = list(cfg.model.KP_NAMES)

    mj = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(mj)
    mujoco.mj_forward(mj, d)
    site_idx = {}
    for i in range(mj.nsite):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and name.startswith("tracking[") and name.endswith("]"):
            site_idx[name[len("tracking["):-1]] = i

    ref = np.full((len(kp_names), 3), np.nan)
    for i, name in enumerate(kp_names):
        if name in site_idx:
            ref[i] = d.site_xpos[site_idx[name]]
    return kp_names, ref, str(MODEL_XML)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula -- avoids a scipy dependency for a test."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                 [axis[2], 0, -axis[0]],
                 [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _span(pts: np.ndarray) -> float:
    return float(np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum()))


# ---------------------------------------------------------------------------
# 1. rigid alignment does not change size
# ---------------------------------------------------------------------------

def test_rigid_alignment_preserves_size():
    _require_model()
    from scripts.viz.scale_check import _rigid_align

    _, ref, _ = _kp_names_and_ref()
    finite = np.all(np.isfinite(ref), axis=1)
    pts = ref[finite]
    assert pts.shape[0] >= 4

    rot = _rotation_matrix(np.array([0.3, 0.7, -0.2]), 1.1)
    t = np.array([0.4, -0.25, 0.15])
    scale_true = 1.3
    moved = scale_true * (rot @ pts.T).T + t

    R, tt = _rigid_align(moved, pts)
    aligned = (R @ moved.T).T + tt

    orig_span = _span(pts)
    aligned_span = _span(aligned)
    assert aligned_span == pytest.approx(scale_true * orig_span, rel=0.01)
    # sanity: alignment actually did something (didn't just no-op)
    assert not np.allclose(R, np.eye(3), atol=1e-3)


# ---------------------------------------------------------------------------
# 2. span ratios are correct for a known scale
# ---------------------------------------------------------------------------

def test_span_ratio_matches_known_scale(tmp_path):
    _require_model()
    from scripts.viz.scale_check import render_scale_check

    kp_names, ref, model_xml = _kp_names_and_ref()
    s_true = 0.0123
    T = 4
    kp3d = np.tile((ref / s_true)[None], (T, 1, 1))

    ratios = render_scale_check(kp3d, kp_names, model_xml, [s_true],
                                [0, 1, 2, 3], tmp_path / "grid.png")
    assert len(ratios) == 1
    assert ratios[0]["trunk"] == pytest.approx(1.0, abs=0.01)
    assert ratios[0]["leg"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3. wing markers are excluded from the alignment / do not perturb trunk+leg
# ---------------------------------------------------------------------------

def test_wing_corruption_does_not_move_trunk_leg_ratio(tmp_path):
    _require_model()
    from scripts.benchmark.metrics import kp_group
    from scripts.viz.scale_check import render_scale_check

    kp_names, ref, model_xml = _kp_names_and_ref()
    s_true = 0.0123
    kp3d_clean = np.tile((ref / s_true)[None], (2, 1, 1))

    ratios_clean = render_scale_check(kp3d_clean, kp_names, model_xml, [s_true],
                                      [0, 1], tmp_path / "clean.png")

    wing_idx = [i for i, n in enumerate(kp_names) if kp_group(n) == "wing"]
    assert wing_idx, "expected at least one wing marker in v1 KP_NAMES"
    kp3d_corrupt = kp3d_clean.copy()
    kp3d_corrupt[:, wing_idx, :] += 50.0  # wildly wrong, well off the body

    ratios_corrupt = render_scale_check(kp3d_corrupt, kp_names, model_xml, [s_true],
                                        [0, 1], tmp_path / "corrupt.png")

    assert ratios_corrupt[0]["trunk"] == pytest.approx(ratios_clean[0]["trunk"], abs=0.01)
    assert ratios_corrupt[0]["leg"] == pytest.approx(ratios_clean[0]["leg"], abs=0.01)


# ---------------------------------------------------------------------------
# 4. NaN frames are handled, not raised
# ---------------------------------------------------------------------------

def test_nan_frame_does_not_raise(tmp_path):
    _require_model()
    from scripts.viz.scale_check import render_scale_check

    kp_names, ref, model_xml = _kp_names_and_ref()
    s_true = 0.0123
    kp3d = np.tile((ref / s_true)[None], (3, 1, 1))
    kp3d[1] = np.nan  # entire frame missing
    kp3d[2, :5] = np.nan  # partial frame missing

    ratios = render_scale_check(kp3d, kp_names, model_xml, [s_true],
                                [0, 1, 2], tmp_path / "nan.png")
    assert len(ratios) == 1
    # frame 0 alone is fully finite and should still drive a sane ratio
    assert ratios[0]["trunk"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 5. PNG is written, non-empty, and roughly the expected tiled size
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. fitted-pose mode (qpos)
# ---------------------------------------------------------------------------

def test_qpos_rest_pose_reproduces_rest_ratios(tmp_path):
    """Passing the model's own rest qpos explicitly must reproduce exactly
    what qpos=None gives -- the rest-pose codepath is just qpos=mj.qpos0."""
    _require_model()
    import mujoco

    from scripts.viz.scale_check import render_scale_check

    kp_names, ref, model_xml = _kp_names_and_ref()
    mj = mujoco.MjModel.from_xml_path(model_xml)
    s_true = 0.0123
    T = 3
    kp3d = np.tile((ref / s_true)[None], (T, 1, 1))
    frames = [0, 1, 2]

    ratios_none = render_scale_check(kp3d, kp_names, model_xml, [s_true], frames,
                                     tmp_path / "none.png")
    qpos_rest = np.tile(mj.qpos0[None], (T, 1))
    ratios_rest = render_scale_check(kp3d, kp_names, model_xml, [s_true], frames,
                                     tmp_path / "rest.png", qpos=qpos_rest)

    assert ratios_rest[0]["trunk"] == pytest.approx(ratios_none[0]["trunk"], abs=1e-6)
    assert ratios_rest[0]["leg"] == pytest.approx(ratios_none[0]["leg"], abs=1e-6)


def test_qpos_posed_moves_leg_ratio_not_trunk_ratio(tmp_path):
    """A non-rest qpos (legs bent away from rest) changes the leg span ratio
    but leaves the trunk ratio essentially unchanged -- trunk sites are rigid
    relative to the thorax root, which is exactly the property that makes
    the trunk row pose-invariant and therefore trustworthy at rest pose."""
    _require_model()
    import mujoco

    from scripts.benchmark.metrics import LEG_QPOS_PREFIXES
    from scripts.viz.scale_check import render_scale_check

    kp_names, ref, model_xml = _kp_names_and_ref()
    mj = mujoco.MjModel.from_xml_path(model_xml)
    s_true = 0.0123
    T = 2
    kp3d = np.tile((ref / s_true)[None], (T, 1, 1))
    frames = [0, 1]

    leg_cols = []
    for j in range(mj.njnt):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and name.startswith(LEG_QPOS_PREFIXES):
            leg_cols.append(mj.jnt_qposadr[j])
    assert leg_cols, "expected leg joints in the v1 model"

    qpos_rest = np.tile(mj.qpos0[None], (T, 1))
    qpos_posed = qpos_rest.copy()
    qpos_posed[:, leg_cols] += 0.4  # bend every leg joint well away from rest

    ratios_rest = render_scale_check(kp3d, kp_names, model_xml, [s_true], frames,
                                     tmp_path / "rest.png", qpos=qpos_rest)
    ratios_posed = render_scale_check(kp3d, kp_names, model_xml, [s_true], frames,
                                      tmp_path / "posed.png", qpos=qpos_posed)

    # trunk pinned: rigid relative to the thorax root, unaffected by leg angle
    assert ratios_posed[0]["trunk"] == pytest.approx(ratios_rest[0]["trunk"], abs=0.02)
    # leg ratio must actually move -- otherwise this test is not exercising posed mode
    assert abs(ratios_posed[0]["leg"] - ratios_rest[0]["leg"]) > 0.03


def test_missing_stac_h5_falls_back_to_rest_with_warning(tmp_path, capsys):
    from scripts.viz.scale_check import _resolve_qpos

    qpos, desc = _resolve_qpos(tmp_path, "stac")
    assert qpos is None
    assert "missing" in desc
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_png_written_and_tiled_size(tmp_path):
    _require_model()
    import imageio.v2 as imageio

    from scripts.viz.scale_check import render_scale_check

    kp_names, ref, model_xml = _kp_names_and_ref()
    kp3d = np.tile((ref / 0.0123)[None], (2, 1, 1))
    out_png = tmp_path / "grid.png"
    size = (200, 200)

    render_scale_check(kp3d, kp_names, model_xml, [0.010, 0.0123],
                       [0, 1], out_png, labels=["a", "b"], size=size)

    assert out_png.exists()
    assert out_png.stat().st_size > 0

    img = imageio.imread(out_png)
    h, w = img.shape[:2]
    # exactly 2x tall (row labels only add width, not height)
    assert h == pytest.approx(2 * size[1], rel=0.1)
    # at least 2x wide, generously bounded above for the label strip
    assert 2 * size[0] * 0.9 <= w <= 2 * size[0] * 2.5
