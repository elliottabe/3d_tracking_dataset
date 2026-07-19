# tests/test_v3_batches_parallel.py
"""Parallel (thread-pool) sample fetch in v3.batches() must be numerically
identical to the serial path for the same seed/index-selection -- only the
sample-fetch is parallelized, not the index selection itself. Uses a tiny
synthetic stub dataset (no real red_data / I/O)."""
import numpy as np

from jarvis_jax.data.v3 import batches


class _StubDS:
    """Minimal stand-in for V3Dataset.__getitem__: deterministic synthetic
    (img4, kp, vis) derived purely from the index `i`, so two fetches of the
    same index are guaranteed byte-identical regardless of fetch order/thread."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img4 = np.full((8, 8, 4), fill_value=i % 256, dtype=np.uint8)
        kp = np.full((3, 2), fill_value=float(i), dtype=np.float32)
        vis = np.array([i % 2 == 0, True, i % 3 == 0], dtype=np.bool_)
        return img4, kp, vis


def _collect(ds, **kwargs):
    return list(batches(ds, 4, **kwargs))


def _assert_batches_equal(a, b):
    assert len(a) == len(b)
    for (ai, ak, av), (bi, bk, bv) in zip(a, b):
        assert np.array_equal(ai, bi)
        assert np.array_equal(ak, bk)
        assert np.array_equal(av, bv)


def test_shapes_and_count():
    ds = _StubDS(23)
    out = _collect(ds, shuffle=True, seed=0, num_workers=4)
    # drop_last=True (default) -> floor(23/4)=5 batches of 4
    assert len(out) == 5
    for imgs, kps, viss in out:
        assert imgs.shape == (4, 8, 8, 4) and imgs.dtype == np.uint8
        assert kps.shape == (4, 3, 2) and kps.dtype == np.float32
        assert viss.shape == (4, 3) and viss.dtype == np.bool_


def test_equivalence_shuffle_serial_vs_parallel():
    ds = _StubDS(37)
    serial = _collect(ds, shuffle=True, seed=0, num_workers=1)
    parallel = _collect(ds, shuffle=True, seed=0, num_workers=8)
    _assert_batches_equal(serial, parallel)


def test_equivalence_no_shuffle_serial_vs_parallel():
    ds = _StubDS(29)
    serial = _collect(ds, shuffle=False, seed=0, num_workers=1)
    parallel = _collect(ds, shuffle=False, seed=0, num_workers=8)
    _assert_batches_equal(serial, parallel)


def test_equivalence_weighted_serial_vs_parallel():
    ds = _StubDS(20)
    rng = np.random.default_rng(1)
    w = rng.random(len(ds))
    w = w / w.sum()
    serial = _collect(ds, weights=w, seed=0, num_workers=1)
    parallel = _collect(ds, weights=w, seed=0, num_workers=8)
    _assert_batches_equal(serial, parallel)


def test_default_num_workers_matches_serial():
    """Default (num_workers=8) must match the explicit serial (num_workers=1)
    path -- parallelism must never change results even when the caller doesn't
    pass num_workers explicitly."""
    ds = _StubDS(31)
    serial = _collect(ds, shuffle=True, seed=0, num_workers=1)
    default = _collect(ds, shuffle=True, seed=0)
    _assert_batches_equal(serial, default)


def test_drop_last_false_preserves_remainder_batch():
    ds = _StubDS(10)
    serial = _collect(ds, shuffle=False, seed=0, drop_last=False, num_workers=1)
    parallel = _collect(ds, shuffle=False, seed=0, drop_last=False, num_workers=8)
    assert len(serial) == 3  # 4, 4, 2
    _assert_batches_equal(serial, parallel)
