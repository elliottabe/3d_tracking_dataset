"""ReprojectionTool's BATCHED primitives must equal the scalar loops BIT for BIT.

qc.py reprojects/triangulates through `reproject_points`/`reconstruct_points`
instead of per-point Python loops (7M scalar calls per bout). qc.json is a
continuous time series across runs, so "close enough" is not good enough here:
a one-ulp drift would show up as unexplained movement in every bout's numbers.

Expectation if the vectorization is correct: `==`, not `allclose`.
(np.einsum is used rather than `@`/tensordot precisely because the BLAS-backed
products reassociate the 4-term dot and drift ~1e-13 px -- asserted below.)
"""
import numpy as np
import pytest

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

# Real telecentric projection matrices from this rig (Cam2012630/631/853/855,
# 2026_03_18_15_31_22-style calibration): P[2, :3] == 0, w == 1.
P_REAL = [
    [[8.1001, 0.0074869, -0.031773, -2.828],
     [0.0093308, -8.0788, -0.17912, 462.78],
     [0.0, 0.0, 0.0, 1.0]],
    [[-0.15977, 8.0916, -0.20825, -60.121],
     [-0.19148, -0.21031, -8.0742, 421.42],
     [0.0, 0.0, 0.0, 1.0]],
    [[-8.0669, -0.5271, 0.1174, 986.55],
     [-0.11122, 0.20323, -8.0855, 425.63],
     [0.0, 0.0, 0.0, 1.0]],
    [[0.42219, -8.0698, 0.29397, 1041.6],
     [0.20885, 0.32217, -8.0679, 424.19],
     [0.0, 0.0, 0.0, 1.0]],
]


@pytest.fixture(scope="module")
def rt(tmp_path_factory):
    cv2 = pytest.importorskip("cv2")
    d = tmp_path_factory.mktemp("calib")
    for i, P in enumerate(P_REAL):
        fs = cv2.FileStorage(str(d / f"Cam{i:04d}.yaml"), cv2.FILE_STORAGE_WRITE)
        fs.write("projectionMatrix", np.asarray(P, np.float64))
        fs.release()
    return ReprojectionTool(str(d))


def test_reproject_points_bit_identical_to_loop(rt):
    rng = np.random.default_rng(0)
    P = rng.normal(0, 30, size=(400, 3))
    ref = np.stack([rt.reproject_point(p) for p in P])
    assert (rt.reproject_points(P) == ref).all(), "batched reprojection drifted"


def test_reproject_points_keeps_leading_shape_and_nan(rt):
    P = np.full((3, 5, 3), np.nan)
    P[0, 0] = [1.0, 2.0, 3.0]
    out = rt.reproject_points(P)
    assert out.shape == (3, 5, rt.num_cameras, 2)
    assert (out[0, 0] == rt.reproject_point(P[0, 0])).all()
    assert np.isnan(out[1, 2]).all(), "NaN must propagate, not become a number"


def test_matmul_would_have_drifted(rt):
    """Why einsum: the BLAS-backed product is NOT bit-identical."""
    rng = np.random.default_rng(1)
    P = rng.normal(0, 30, size=(400, 3))
    ref = np.stack([rt.reproject_point(p) for p in P])
    ph = np.concatenate([P, np.ones((len(P), 1))], 1)
    Mt = np.ascontiguousarray(rt._cam_mats_f64.transpose(0, 2, 1))
    proj = np.moveaxis(ph @ Mt, 0, 1)
    blas = proj[..., :2] / proj[..., 2:3]
    assert np.allclose(blas, ref, atol=1e-9)
    assert not (blas == ref).all(), (
        "BLAS matmul happened to match here; keep using einsum anyway")


def test_reconstruct_points_bit_identical_to_loop(rt):
    rng = np.random.default_rng(2)
    X = rng.normal(0, 30, size=(200, 3))
    obs_full = rt.reproject_points(X)                       # (N, C, 2)
    # every 3-camera subset order the LOO test can produce
    cams = np.stack([np.sort(rng.permutation(rt.num_cameras)[:3])
                     for _ in range(len(X))])
    ref = np.stack([rt.reconstruct_point(obs_full[i], cams_to_use=list(cams[i]))
                    for i in range(len(X))])
    got = rt.reconstruct_points(
        np.stack([obs_full[i][cams[i]] for i in range(len(X))]), cams)
    assert (got == ref).all(), "batched triangulation drifted"
    assert np.abs(got - X).max() < 1e-6, "triangulation is not even correct"


def test_reconstruct_points_under_two_cameras_is_zeros(rt):
    out = rt.reconstruct_points(np.zeros((4, 1, 2)), np.zeros((4, 1), int))
    assert out.shape == (4, 3) and (out == 0).all()
