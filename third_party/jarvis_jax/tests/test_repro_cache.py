"""Tests for data/repro_cache.py — writer + reader round-trip.

Tests:
    test_write_read_roundtrip — synthetic volumes streamed via write_cache,
        reload via load_cache; checks shapes, dtype (fp16), and meta round-trip.
    test_load_cache_readonly — load_cache returns a memmap in read-only mode.
"""
import json
import os

import numpy as np
import pytest

from jarvis_jax.data.repro_cache import write_cache, load_cache


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
