"""Process-based loader workers (`data/loader_workers.py`) and the thread-pool
lifetime of `window_batches`.

The two things that must hold for the process path to be a drop-in for the
thread path on a live training run:

  1. A batch is BYTE-IDENTICAL between the paths for the same (indices, seed) --
     including with train=True, centre jitter and copy-paste on. That is only
     true because every per-sample RNG in `V12WindowDataset` is seeded from
     (dataset seed, index, epoch) with no worker-local state, and because
     `window_batches` forwards the epoch with every task. If either stopped
     being true, the augmentation a worker draws would silently diverge from the
     one the sampler thinks it drew, and nothing else in the pipeline would
     notice.
  2. Neither path leaks OS threads. `window_batches` used to hold its
     `ThreadPoolExecutor` in a `with` inside the generator body, which releases
     the threads only when the generator is DRAINED; a consumer that stops
     mid-epoch left `num_workers` threads alive until the generator was garbage
     collected (and the GC at interpreter shutdown died inside `Thread.join`).
"""
import threading
import time

import numpy as np
import pytest
from mvq_fixtures import make_v12_root

from jarvis_jax.data.concat_windows import ConcatWindowDataset
from jarvis_jax.data.loader_workers import (ConcatSpec, ProcessSampleLoader, V12Spec,
                                            build_dataset, dataset_spec)
from jarvis_jax.data.mv_copy_paste import CopyPasteParams
from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS, window_batches


def _root(base, **kw):
    base.mkdir(parents=True, exist_ok=True)
    return make_v12_root(base, **kw)


def _same(a, b, what):
    assert set(a) == set(b) == set(WINDOW_KEYS)
    for k in WINDOW_KEYS:
        assert a[k].shape == b[k].shape, f"{what}: {k} shape {a[k].shape} vs {b[k].shape}"
        assert a[k].dtype == b[k].dtype, f"{what}: {k} dtype {a[k].dtype} vs {b[k].dtype}"
        assert np.array_equal(a[k], b[k]), f"{what}: {k} differs between the two loader paths"


def _drain(ds, **kw):
    return list(window_batches(ds, 2, seed=3, shuffle=True, **kw))


@pytest.mark.parametrize("train,copy_paste", [(False, None),
                                              (True, CopyPasteParams(p=1.0, max_tries=8))])
def test_process_batches_are_byte_identical_to_the_thread_path(tmp_path, train, copy_paste):
    """Same dataset, same seed -> the same bytes, augmentation included."""
    root = _root(tmp_path / "a")
    def make():
        return V12WindowDataset(str(root), "train", T=1, train=train, seed=0,
                                jitter_units=10.0, copy_paste=copy_paste)
    thr = _drain(make())
    ds = make()
    proc = _drain(ds, workers="processes", num_workers=2)
    getattr(ds, "_mvq_loader_pool").close()
    assert len(thr) == len(proc) and thr
    for n, (a, b) in enumerate(zip(thr, proc)):
        _same(a, b, f"batch {n} (train={train}, copy_paste={copy_paste is not None})")


def test_the_epoch_reaches_the_workers(tmp_path):
    """Jitter is epoch-dependent, so a worker that never saw `ds.epoch` would
    return epoch-0 samples for every epoch. Two seeds must differ on BOTH paths
    and agree ACROSS paths -- the first half of that is what makes the second
    half evidence rather than a tautology."""
    root = _root(tmp_path / "a")
    def batch(seed, **kw):
        ds = V12WindowDataset(str(root), "train", T=1, train=True, seed=0, jitter_units=10.0)
        # shuffle=False: the index list is identical for both seeds, so any
        # difference is the epoch reaching (or not reaching) `__getitem__`.
        out = next(window_batches(ds, 2, seed=seed, shuffle=False, **kw))
        pool = getattr(ds, "_mvq_loader_pool", None)
        if pool is not None:
            pool.close()
        return out
    t0, t1 = batch(0), batch(1)
    p0, p1 = batch(0, workers="processes", num_workers=2), batch(1, workers="processes", num_workers=2)
    assert not np.array_equal(t0["center3D"], t1["center3D"]), "fixture jitter is epoch-invariant"
    _same(t0, p0, "epoch 0")
    _same(t1, p1, "epoch 1")


def test_concat_spec_round_trips_including_a_forced_sample_weight(tmp_path):
    """`ConcatWindowDataset.worker_spec` + `build_dataset` must rebuild the same
    windows in the worker, and a view that CHANGES a sample (the negatives
    root's forced `sample_weight`) must survive the trip."""
    a = V12WindowDataset(str(_root(tmp_path / "a")), "train", T=1, seed=0, train=False)
    b = V12WindowDataset(str(_root(tmp_path / "b")), "train", T=1, seed=0, train=False)
    ds = ConcatWindowDataset([a, b], names=["real", "extra"])
    spec = dataset_spec(ds)
    assert isinstance(spec, ConcatSpec) and all(isinstance(s, V12Spec) for s in spec.specs)
    rebuilt = build_dataset(spec)
    assert len(rebuilt) == len(ds) and rebuilt.names == ds.names
    for i in (0, len(ds) - 1):
        _same({k: np.asarray(ds[i][k])[None] for k in WINDOW_KEYS},
              {k: np.asarray(rebuilt[i][k])[None] for k in WINDOW_KEYS}, f"rebuilt sample {i}")

    forced = build_dataset(ConcatSpec(
        specs=(spec.specs[0], type(spec.specs[1])(**{**spec.specs[1].__dict__,
                                                     "force_sample_weight": 1.0}),),
        names=spec.names, allow_calib_mismatch=False))
    j = len(a)                                        # first window of the second root
    assert float(forced[j]["sample_weight"]) == 1.0
    assert forced.weight(j) == 1.0


def test_a_pool_serves_several_datasets_by_key(tmp_path):
    """One pool, two keys -- how the T=1 and T=2 mvq streams share 24 workers."""
    root = str(_root(tmp_path / "a"))
    specs = {T: V12WindowDataset(root, "train", T=T, train=False, seed=0).worker_spec()
             for T in (1, 2)}
    with ProcessSampleLoader(specs, 2) as pool:
        for T in (1, 2):
            (_, samples), = pool.map_batches([[0, 1]], epoch=0, key=T)
            assert samples[0]["crops"].shape[0] == T


def _threads():
    with open("/proc/self/status") as f:
        return int(f.read().split("Threads:")[1].split()[0])


def _settle(base, tries=100):
    for _ in range(tries):
        if _threads() <= base:
            return _threads()
        time.sleep(0.05)
    return _threads()


class _TinyDS:
    """A dataset whose samples cost nothing: this test is about the pool's
    lifetime, not about window assembly."""
    epoch = 0
    def __len__(self):
        return 32
    def __getitem__(self, i):
        return {k: np.zeros((1,), np.float32) for k in WINDOW_KEYS}


def test_window_batches_does_not_leak_threads_across_epochs(tmp_path):
    ds = _TinyDS()
    base = _threads()
    for e in range(20):
        for _ in window_batches(ds, 4, seed=e, num_workers=8):
            pass
        assert _settle(base) <= base, f"epoch {e}: threads grew to {_threads()} from {base}"
    assert threading.active_count() >= 1


def test_an_abandoned_epoch_releases_its_pool_without_waiting_for_gc(tmp_path):
    """The leak that mattered: a consumer that stops mid-epoch. Closing the
    generator must return the threads, with no garbage-collection round trip."""
    ds = _TinyDS()
    base = _threads()
    gens = []
    for e in range(10):
        g = window_batches(ds, 4, seed=e, num_workers=8)
        next(g)
        gens.append(g)
    assert _threads() > base, "the fixture never started any worker threads -- test is vacuous"
    for g in gens:
        g.close()                       # explicit close, NOT `del` + gc.collect()
    assert _settle(base) <= base, f"threads stuck at {_threads()} (base {base}) after close()"


def test_close_returns_promptly_with_batches_still_in_flight(tmp_path):
    """`run_training` closes the pool after the last training step, with up to
    `inflight` batches per stream still being produced. A shutdown that blocks
    there would strand a finished 40 000-step run just before its final eval and
    checkpoint, so `close()` is bounded by construction: it must return, and it
    must report that it reaped the workers."""
    root = str(_root(tmp_path / "a"))
    ds = V12WindowDataset(root, "train", T=1, train=False, seed=0)
    n = len(ds)
    pool = ProcessSampleLoader({None: ds.worker_spec()}, 2, inflight=3)
    it = pool.map_batches([[i % n for i in range(b * 2, b * 2 + 2)] for b in range(12)], epoch=0)
    next(it)                                     # one batch out, the rest in flight
    t0 = time.time()
    reaped = pool.close(timeout=30.0)
    assert reaped, "close() had to abandon its workers"
    assert time.time() - t0 < 30, "close() blocked on the in-flight batches"
    assert pool.close() is True                  # idempotent
