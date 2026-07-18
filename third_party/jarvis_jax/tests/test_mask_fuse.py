"""Tests for the SAM3 mask-fusion core (`jarvis_jax.hybridnet.mask_fuse`).

Covers:
- soft_gate: floor=1.0 exact identity; low-temperature attenuation outside
  the silhouette.
- mask_input_crops: background zeroing (Option B).
- mask_consistency_volume: robustness to a single all-zero ("bad") camera
  mask under `mean` and `topk` aggregation -- a genuine voxel seen by the
  other cameras must NOT be carved to zero (this is why aggregation must
  never be a strict product across cameras).
"""
import numpy as np
import jax.numpy as jnp

from jarvis_jax.hybridnet.mask_fuse import (
    soft_gate,
    mask_input_crops,
    mask_consistency_volume,
)


def test_soft_gate_identity_when_floor_one():
    vol = jnp.asarray(np.random.RandomState(0).rand(2, 5, 8, 8, 8).astype("float32"))
    cons = jnp.asarray(np.random.RandomState(1).rand(2, 8, 8, 8).astype("float32"))
    out = soft_gate(vol, cons, floor=1.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(vol), atol=1e-6)


def test_soft_gate_attenuates_outside_silhouette():
    vol = jnp.ones((1, 3, 4, 4, 4), "float32")
    cons = jnp.zeros((1, 4, 4, 4), "float32").at[0, 0, 0, 0].set(1.0)  # only one voxel consistent
    out = np.asarray(soft_gate(vol, cons, temperature=0.1, floor=0.0))
    assert out[0, 0, 0, 0, 0] > out[0, 0, 1, 1, 1]                     # inside > outside


def test_mask_input_zeroes_background():
    crops = jnp.ones((1, 2, 6, 6, 3), "float32")
    masks = jnp.zeros((1, 2, 6, 6), "float32").at[:, :, :3, :].set(1.0)
    out = np.asarray(mask_input_crops(crops, masks))
    assert out[0, 0, 4, 0, 0] == 0.0 and out[0, 0, 0, 0, 0] == 1.0


def _degenerate_camera_matrices(num_cam):
    """Build (num_cam, 4, 3) camera matrices with constant depth (w == 1) and a
    position-independent x/y mapping (z ignored).

    Simplification / rationale (see task-6-report.md): the exact index each
    grid voxel maps to under the full DLT + clamp + trilinear-upsample chain
    in `reproject._reproject_single` is intricate and is already exercised by
    `tests/test_reproject_parity.py` / `test_reprojection_tool.py`. This test
    is about the CROSS-CAMERA AGGREGATION behaviour of
    `mask_consistency_volume`, not the reprojection geometry itself. Using a
    degenerate camera (w constant, so no division-by-zero regardless of grid
    position) combined with spatially-UNIFORM per-camera masks (all-ones vs.
    all-zeros) means every voxel gathers that camera's constant fill value no
    matter which index the geometry computes -- so the test result depends
    only on the aggregation logic, not on precise projection arithmetic.
    """
    mats = np.zeros((num_cam, 4, 3), dtype=np.float32)
    for c in range(num_cam):
        mats[c, 0, 0] = 1.0  # x -> val1
        mats[c, 1, 1] = 1.0  # y -> val2
        mats[c, 3, 2] = 1.0  # translation row supplies constant w = 1
    return mats


def test_mask_consistency_volume_robust_to_one_bad_camera_mean():
    num_cam, hm, grid = 3, 8, 8
    # cameras 0,1 "see" the object everywhere (all-ones mask); camera 2 is a
    # bad/blank mask (all-zeros). A genuine voxel is one the good cameras
    # agree on, and with uniform-fill masks that is every voxel.
    masks_np = np.ones((1, num_cam, hm, hm), dtype=np.float32)
    masks_np[:, 2] = 0.0
    masks = jnp.asarray(masks_np)

    center3D = jnp.zeros((1, 3), jnp.float32)
    centerHM = jnp.zeros((1, num_cam, 2), jnp.float32)
    cameraMatrices = jnp.asarray(_degenerate_camera_matrices(num_cam))[None]

    vol_mean = mask_consistency_volume(
        masks, center3D, centerHM, cameraMatrices,
        grid_size=grid, heatmap_size=hm, aggregate="mean",
    )
    assert vol_mean.shape == (1, grid, grid, grid)
    assert bool(jnp.isfinite(vol_mean).all())
    # 2 good (=1) + 1 bad (=0) cameras -> mean == 2/3, strictly > 0 everywhere.
    np.testing.assert_allclose(np.asarray(vol_mean), np.full((1, grid, grid, grid), 2.0 / 3.0), atol=1e-5)
    assert bool((vol_mean > 0).all())


def test_mask_consistency_volume_robust_to_one_bad_camera_topk():
    num_cam, hm, grid = 3, 8, 8
    masks_np = np.ones((1, num_cam, hm, hm), dtype=np.float32)
    masks_np[:, 2] = 0.0
    masks = jnp.asarray(masks_np)

    center3D = jnp.zeros((1, 3), jnp.float32)
    centerHM = jnp.zeros((1, num_cam, 2), jnp.float32)
    cameraMatrices = jnp.asarray(_degenerate_camera_matrices(num_cam))[None]

    vol_topk = mask_consistency_volume(
        masks, center3D, centerHM, cameraMatrices,
        grid_size=grid, heatmap_size=hm, aggregate="topk",
    )
    assert vol_topk.shape == (1, grid, grid, grid)
    # default topk drops the single worst camera (the bad/blank one) and
    # averages the remaining num_cam-1=2 good cameras -> exactly 1.0.
    np.testing.assert_allclose(np.asarray(vol_topk), np.ones((1, grid, grid, grid)), atol=1e-5)
    assert bool((vol_topk > 0).all())
