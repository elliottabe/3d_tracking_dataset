import numpy as np
import pytest

from jarvis_jax.tracking.triangulate import refine_from_seed


def _rig(n_cam=7, seed=0):
    rng = np.random.default_rng(seed)
    cams = []
    for i in range(n_cam):
        th = 2 * np.pi * i / n_cam
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        # Camera CENTER on a real ring (radius 50, z=-200) so views have an
        # actual baseline. A pure z-rotation applied to a z-aligned t (the
        # earlier draft of this fixture used t=(0,0,200) for every camera)
        # leaves every camera's optical center at the same point -- z is the
        # rotation axis, so it's invariant under R -- which makes triangulation
        # of depth impossible no matter how many "views" you add: every camera
        # sees a different image-plane orientation of the SAME ray, so any
        # DLT solve (this one included) can only return an arbitrary point
        # along that ray, not the true depth. Verified against
        # triangulate_dlt_batched: it reproduces the same degenerate failure
        # on the single-center rig, so this is a fixture bug, not a
        # refine_from_seed bug.
        C = np.array([50.0 * np.cos(th), 50.0 * np.sin(th), -200.0])
        t = -R @ C
        P = np.hstack([8.0 * R, (8.0 * t)[:, None]])
        cams.append(P.T.astype(np.float32))          # (4,3), p_h @ M convention
    return np.stack(cams)


def _project(cam_mats, X):
    h = np.append(np.asarray(X, np.float64), 1.0)
    out = []
    for M in cam_mats:
        p = h @ M
        out.append([p[0] / p[2], p[1] / p[2]])
    return np.asarray(out)


def test_clean_views_recover_the_true_point():
    cams = _rig()
    X = np.array([1.0, 2.0, 3.0])
    xy = _project(cams, X)[None, :, None, :]                 # (1,C,1,2)
    conf = np.ones((1, cams.shape[0], 1), np.float32)
    seed = (X + 0.4)[None, None, :].astype(np.float32)       # (1,1,3)
    out, _, nv = refine_from_seed(seed, xy.astype(np.float32), conf, cams,
                                  gate_px=15.0)
    np.testing.assert_allclose(out[0, 0], X, atol=1e-3)
    assert nv[0, 0] == cams.shape[0]


def test_a_swapped_view_is_gated_out():
    """The failure the whole gate exists for: one camera's tarsal tip jumps to
    the wrong leg at conf 0.4-0.8 and drags the DLT."""
    cams = _rig()
    X = np.array([1.0, 2.0, 3.0])
    xy = _project(cams, X)
    xy[3] += 250.0                                            # gross swap
    conf = np.ones((1, cams.shape[0], 1), np.float32)
    conf[0, 3, 0] = 0.6
    seed = (X + 0.4)[None, None, :].astype(np.float32)
    out, _, nv = refine_from_seed(seed, xy[None, :, None, :].astype(np.float32),
                                  conf, cams, gate_px=15.0)
    np.testing.assert_allclose(out[0, 0], X, atol=1e-2)
    assert nv[0, 0] == cams.shape[0] - 1


def test_seed_is_returned_when_too_few_views_survive():
    cams = _rig()
    X = np.array([1.0, 2.0, 3.0])
    xy = _project(cams, X) + 500.0                            # every view gated out
    conf = np.ones((1, cams.shape[0], 1), np.float32)
    seed = np.array([[[9.0, 9.0, 9.0]]], np.float32)
    out, _, nv = refine_from_seed(seed, xy[None, :, None, :].astype(np.float32),
                                  conf, cams, gate_px=5.0, min_views=2)
    np.testing.assert_allclose(out[0, 0], seed[0, 0], atol=1e-6)
    assert nv[0, 0] < 2


def test_output_is_continuous_not_quantized():
    """Stage 3 exists to escape the ~0.17 mm voxel staircase — two seeds a
    tenth of a voxel apart must give distinguishable, non-identical output."""
    cams = _rig()
    outs = []
    for d in (0.0, 0.1):
        X = np.array([1.0 + d, 2.0, 3.0])
        xy = _project(cams, X)[None, :, None, :].astype(np.float32)
        conf = np.ones((1, cams.shape[0], 1), np.float32)
        seed = np.zeros((1, 1, 3), np.float32)
        out, _, _ = refine_from_seed(seed, xy, conf, cams, gate_px=1e6)
        outs.append(out[0, 0, 0])
    assert 0.05 < abs(outs[1] - outs[0]) < 0.15


def test_low_conf_views_are_dropped():
    cams = _rig()
    X = np.array([1.0, 2.0, 3.0])
    xy = _project(cams, X)[None, :, None, :].astype(np.float32)
    conf = np.ones((1, cams.shape[0], 1), np.float32)
    conf[0, :3, 0] = 0.05
    seed = X[None, None, :].astype(np.float32)
    _, _, nv = refine_from_seed(seed, xy, conf, cams, gate_px=15.0,
                                conf_thresh=0.3)
    assert nv[0, 0] == cams.shape[0] - 3
