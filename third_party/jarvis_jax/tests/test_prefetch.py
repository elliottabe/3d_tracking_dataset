import numpy as np
import jax
import pytest
from jarvis_jax.sharding import data_parallel_mesh
from jarvis_jax.data.prefetch import prefetch


def _fake_batches(n, nd):
    for k in range(n):
        # batch divisible by device count
        yield (np.full((nd, 4, 4, 4), k, dtype=np.uint8),
               np.full((nd, 5, 2), k, dtype=np.float32),
               np.ones((nd, 5), dtype=bool))


def test_prefetch_preserves_order_and_values():
    mesh = data_parallel_mesh()
    nd = len(jax.devices())
    got = list(prefetch(_fake_batches(6, nd), mesh, depth=2))
    assert len(got) == 6
    for k, (img, kp, vis) in enumerate(got):
        assert int(np.asarray(img).flat[0]) == k          # order preserved
        assert np.asarray(kp).shape == (nd, 5, 2)
        # device-resident (sharded) arrays
        assert hasattr(img, "sharding")


def test_prefetch_propagates_worker_exception():
    from jarvis_jax.sharding import data_parallel_mesh
    mesh = data_parallel_mesh()
    nd = len(jax.devices())
    def bad():
        yield (np.zeros((nd, 4, 4, 4), np.uint8),
               np.zeros((nd, 5, 2), np.float32), np.ones((nd, 5), bool))
        raise ValueError("boom")
    gen = prefetch(bad(), mesh, depth=1)
    next(gen)                      # first batch ok
    with pytest.raises(ValueError, match="boom"):
        next(gen)                  # second pull surfaces the worker error
