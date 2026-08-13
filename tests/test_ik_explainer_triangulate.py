import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io, triangulate3d

CLIP = clip_io.CLIP_DEFAULT
pytestmark = pytest.mark.skipif(
    not Path(CLIP).exists(), reason="source clip not present")


def test_round_trip_recovers_known_3d_in_mm():
    """Project known mm 3D into 7 views, triangulate back, expect ~0 error.

    This is the guarantee that lets the explainer drop the 0.1 unit factor:
    triangulating our own 2D yields mm directly.
    """
    cam_mats, _ = clip_io.load_dlt(str(Path(CLIP) / "calibration"))
    xyz, _, _ = clip_io.load_shipped_kp3d_mm(clip_io.shipped_csv_path(CLIP))
    X = xyz[:50]                                        # (T,K,3) mm
    ph = np.concatenate([X, np.ones((*X.shape[:2], 1))], -1)
    proj = np.einsum("tkj,cjm->tckm", ph, cam_mats)
    uv = proj[..., :2] / proj[..., 2:3]
    conf = np.ones(uv.shape[:3], np.float32)
    kp3d, _ = triangulate3d.triangulate(uv, conf, cam_mats)
    err = np.linalg.norm(kp3d - X, axis=-1)
    assert np.nanmax(err) < 1e-3, f"max round-trip error {np.nanmax(err):.2e} mm"


def test_affine_projection_has_unit_depth():
    """The rig is telecentric, so the perspective divide must be a no-op."""
    cam_mats, _ = clip_io.load_dlt(str(Path(CLIP) / "calibration"))
    X = np.array([[10.0, 3.0, 1.5], [15.0, 2.0, 1.0]])
    ph = np.concatenate([X, np.ones((2, 1))], -1)
    proj = np.einsum("kj,cjm->ckm", ph, cam_mats)
    assert np.allclose(proj[..., 2], 1.0)


def test_filter_never_reduces_coverage():
    rng = np.random.default_rng(0)
    names = clip_io.model_kp_names()
    kp3d = rng.normal(size=(60, 50, 3)) * 0.2 + 10.0
    conf = np.ones((60, 50), np.float32)
    out = triangulate3d.filter_kp3d(kp3d, conf, names)
    assert out.shape == kp3d.shape
    finite_in = np.isfinite(kp3d).all(-1)
    finite_out = np.isfinite(out).all(-1)
    assert np.all(finite_out[finite_in]), "filter deleted a keypoint it was given"
