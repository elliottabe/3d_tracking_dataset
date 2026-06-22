"""Tests for data/repro_cache.py — writer + reader round-trip (Task 2) and
GPU-resident mesh-sharded cache loader + batch iterator (Task 3).

Task 2 tests:
    test_write_read_roundtrip — synthetic volumes streamed via write_cache,
        reload via load_cache; checks shapes, dtype (fp16), and meta round-trip.
    test_load_cache_readonly — load_cache returns a memmap in read-only mode.

Task 3 tests:
    test_to_device_sharded_shapes — to_device_sharded returns sharded JAX
        arrays with correct shapes; n trimmed to device multiple.
    test_to_device_sharded_trim — to_device_sharded trims n when not divisible.
    test_cached_batches_shapes — cached_batches yields batches with correct shapes.
    test_to_device_sharded_and_batch — combined smoke test from brief.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jarvis_jax.data.repro_cache import write_cache, load_cache, to_device_sharded, cached_batches
from jarvis_jax.sharding import data_parallel_mesh


def _make_volumes_iter(n, rng):
    """Yield n individual (50,48,48,48) fp16 volumes."""
    for _ in range(n):
        yield rng.rand(50, 48, 48, 48).astype(np.float16)


def test_write_read_roundtrip(tmp_path):
    n = 3
    rng = np.random.RandomState(0)

    vols_list = [rng.rand(50, 48, 48, 48).astype(np.float16) for _ in range(n)]
    kp3d = rng.rand(n, 50, 3).astype(np.float32)
    c3d = rng.rand(n, 3).astype(np.float32)
    vis = np.ones((n, 50), dtype=bool)

    meta = {
        "vitpose_ckpt": "/fake/ckpt",
        "grid_size": 48,
        "grid_spacing": 1,
        "roi_cube": 48,
        "heatmap_size": 226,
        "num_cameras": 6,
        "n": n,
        "split": "val",
    }

    write_cache(
        str(tmp_path), "val",
        volumes_iter=iter(vols_list),
        n=n,
        kp3d=kp3d,
        center3D=c3d,
        vis=vis,
        meta=meta,
    )

    # Check expected files exist
    assert os.path.isfile(str(tmp_path / "val_volumes.f16"))
    assert os.path.isfile(str(tmp_path / "val_labels.npz"))
    assert os.path.isfile(str(tmp_path / "val_meta.json"))

    # load_cache round-trip
    c = load_cache(str(tmp_path), "val")

    # Shape + dtype of volumes
    assert c["volumes"].shape == (n, 50, 48, 48, 48), (
        f"Expected (n,50,48,48,48) got {c['volumes'].shape}")
    assert c["volumes"].dtype == np.float16, (
        f"Expected float16 got {c['volumes'].dtype}")

    # Labels round-trip
    np.testing.assert_array_equal(c["kp3d"], kp3d)
    np.testing.assert_array_equal(c["center3D"], c3d)
    np.testing.assert_array_equal(c["vis"], vis)

    # Meta round-trip
    assert c["meta"]["grid_size"] == 48
    assert c["meta"]["vitpose_ckpt"] == "/fake/ckpt"
    assert c["meta"]["n"] == n
    assert c["meta"]["split"] == "val"

    # Volume values match what was written (fp16 precision)
    for i in range(n):
        np.testing.assert_array_equal(c["volumes"][i], vols_list[i])


def test_load_cache_readonly(tmp_path):
    """load_cache must return a memmap in read-only mode (mode='r')."""
    n = 2
    rng = np.random.RandomState(1)
    vols = [rng.rand(50, 48, 48, 48).astype(np.float16) for _ in range(n)]
    meta = {"vitpose_ckpt": "x", "grid_size": 48, "n": n, "split": "train"}

    write_cache(
        str(tmp_path), "train",
        volumes_iter=iter(vols),
        n=n,
        kp3d=np.zeros((n, 50, 3), dtype=np.float32),
        center3D=np.zeros((n, 3), dtype=np.float32),
        vis=np.zeros((n, 50), dtype=bool),
        meta=meta,
    )

    c = load_cache(str(tmp_path), "train")
    assert isinstance(c["volumes"], np.memmap), "volumes must be a np.memmap"
    # In read-only mode, writes should raise
    with pytest.raises((ValueError, TypeError, OSError)):
        c["volumes"][0, 0, 0, 0, 0] = np.float16(999.0)


# ---------------------------------------------------------------------------
# Task 3: GPU-resident mesh-sharded cache helpers
# ---------------------------------------------------------------------------

def _fake_cache(n):
    """Build a synthetic in-memory cache dict with n framesets."""
    return {
        "volumes": np.zeros((n, 50, 48, 48, 48), np.float16),
        "kp3d": np.zeros((n, 50, 3), np.float32),
        "center3D": np.zeros((n, 3), np.float32),
        "vis": np.ones((n, 50), bool),
        "meta": {"n": n},
    }


def test_to_device_sharded_shapes():
    """to_device_sharded returns JAX arrays with correct shapes and sharding."""
    mesh = data_parallel_mesh()
    nd = jax.device_count()
    n = 4 * nd
    dev = to_device_sharded(_fake_cache(n), mesh)

    # All four arrays must be present and have correct shapes
    assert dev["volumes"].shape == (n, 50, 48, 48, 48), (
        f"volumes shape mismatch: {dev['volumes'].shape}")
    assert dev["kp3d"].shape == (n, 50, 3), f"kp3d shape mismatch: {dev['kp3d'].shape}"
    assert dev["center3D"].shape == (n, 3), f"center3D shape mismatch: {dev['center3D'].shape}"
    assert dev["vis"].shape == (n, 50), f"vis shape mismatch: {dev['vis'].shape}"

    # All arrays must carry sharding metadata (JAX Arrays have .sharding)
    for key in ("volumes", "kp3d", "center3D", "vis"):
        assert hasattr(dev[key], "sharding"), f"{key} missing .sharding attribute"

    # volumes must stay fp16 on device
    assert dev["volumes"].dtype == jnp.float16, (
        f"volumes dtype should be float16, got {dev['volumes'].dtype}")

    # n_used returned
    assert dev["n_used"] == n


def test_to_device_sharded_trim():
    """to_device_sharded trims n to the largest device-count multiple."""
    mesh = data_parallel_mesh()
    nd = jax.device_count()
    # Give n that is NOT divisible by device count (add 1 extra)
    n_raw = 4 * nd + 1
    dev = to_device_sharded(_fake_cache(n_raw), mesh)
    n_used = dev["n_used"]

    assert n_used == 4 * nd, f"Expected n_used={4*nd}, got {n_used}"
    assert dev["volumes"].shape[0] == n_used, (
        f"volumes axis-0 ({dev['volumes'].shape[0]}) != n_used ({n_used})")
    # Confirm it's divisible
    assert n_used % nd == 0, f"n_used={n_used} is not divisible by nd={nd}"


def test_cached_batches_shapes():
    """cached_batches yields dicts with correct shapes (no host round-trip needed)."""
    mesh = data_parallel_mesh()
    nd = jax.device_count()
    n = 4 * nd
    dev = to_device_sharded(_fake_cache(n), mesh)

    batch_size = 2 * nd
    batches = list(cached_batches(dev, batch_size=batch_size, shuffle=False))

    # With n=4*nd and batch_size=2*nd, expect exactly 2 batches
    assert len(batches) == 2, f"Expected 2 batches, got {len(batches)}"

    for b in batches:
        assert b["volumes"].shape == (batch_size, 50, 48, 48, 48), (
            f"volumes batch shape: {b['volumes'].shape}")
        assert b["kp3d"].shape == (batch_size, 50, 3), (
            f"kp3d batch shape: {b['kp3d'].shape}")
        assert b["center3D"].shape == (batch_size, 3), (
            f"center3D batch shape: {b['center3D'].shape}")
        assert b["vis"].shape == (batch_size, 50), (
            f"vis batch shape: {b['vis'].shape}")


def test_to_device_sharded_and_batch():
    """Combined smoke test from the task brief."""
    mesh = data_parallel_mesh()
    nd = jax.device_count()
    dev = to_device_sharded(_fake_cache(4 * nd), mesh)
    assert hasattr(dev["volumes"], "sharding")
    b = next(cached_batches(dev, batch_size=2 * nd, shuffle=False))
    assert b["volumes"].shape == (2 * nd, 50, 48, 48, 48)
    assert b["kp3d"].shape == (2 * nd, 50, 3)
