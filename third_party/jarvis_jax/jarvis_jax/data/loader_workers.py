"""Process-based sample workers for `v12_windows.window_batches`.

WHY PROCESSES. `V12WindowDataset.__getitem__` is mostly pure-Python/numpy work
-- window assembly, centre jitter, copy-paste compositing -- and only the JPEG
decode releases the GIL. Measured on the 8-GPU batch-32 mvq-v2 run
(`mvq_t2_v2_20260906`), the in-process `ThreadPoolExecutor(24)` used 1.5-4 of
the node's 32 cores and delivered ~11 samples/s against the ~15/s eight L40S
eat, so the depth-2 prefetch drained and GPU utilisation cycled ~6 s at 100 %
then ~3 s at 0 %. Processes give each sample its own interpreter, so the
Python half scales with cores instead of with one GIL.

SPAWN, NEVER FORK. By the time the loader starts the parent has JAX and (on a
training run) eight CUDA contexts initialised. `fork` would copy the CUDA
driver's state and locks into a child that cannot legally use them; the child
either deadlocks inside the driver or corrupts the parent's context. The pool
therefore uses `multiprocessing.get_context("spawn")`, which means a child
inherits NOTHING -- it rebuilds the dataset from a picklable SPEC (`V12Spec` /
`ConcatSpec`, below) in the pool initializer, once per worker, and then serves
`__getitem__` for the index lists the parent sends.

SAMPLING STAYS IN THE PARENT. `_balanced_weights`, the epoch permutation and
`ds.epoch = seed` are parent-side exactly as before; only `__getitem__` moves.
The epoch travels with every task, and every per-sample RNG in
`V12WindowDataset` is seeded as `SeedSequence([self.seed, i, self.epoch, ...])`
-- a pure function of (dataset seed, index, epoch), with no worker-local
state -- so a worker reproduces the same jitter and the same copy-paste donor
the thread path would have drawn. Batches are consequently BYTE-IDENTICAL
between the two paths, augmentation included (asserted by
`tests/test_loader_workers.py`, train=True and copy_paste on).

This module must stay JAX-FREE: it is imported by the spawned children, and a
worker that imports JAX would try to claim a GPU the parent already owns.
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import multiprocessing
import os


# --------------------------------------------------------------------- specs
@dataclasses.dataclass(frozen=True)
class V12Spec:
    """Everything `V12WindowDataset.__init__` needs, as picklable values.

    `force_sample_weight` is not a constructor argument: it reproduces
    `train_mvq._ForceSampleWeight` (the negatives root trains at
    `sample_weight` 1.0 whatever its manifest says) without importing
    `train_mvq` -- which would pull JAX into the worker.
    """
    root: str
    split: str
    T: int
    pair_deltas: tuple
    max_flies: int
    jitter_units: float
    seed: int
    train: bool
    recordings: tuple | None
    copy_paste: object                     # CopyPasteParams | None (a module-level dataclass)
    center_shift_units: float
    sex_overrides: dict
    force_sample_weight: float | None = None


@dataclasses.dataclass(frozen=True)
class ConcatSpec:
    specs: tuple
    names: tuple
    allow_calib_mismatch: bool


class _ForcedWeightView:
    """Worker-side twin of `train_mvq._ForceSampleWeight`.

    Deliberately duplicated rather than imported: `train_mvq` imports JAX, and
    the whole point of the spawn pool is that a worker never touches a device.
    `epoch` needs an explicit property for the same reason it does there --
    `ConcatWindowDataset`'s epoch setter fans out over its `datasets` list, and
    the write has to reach the real dataset rather than land on the view.
    """

    def __init__(self, ds, value):
        self._ds = ds
        self._value = float(value)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        import numpy as np
        s = dict(self._ds[i])
        s["sample_weight"] = np.float32(self._value)
        return s

    def weight(self, i):
        return self._value

    @property
    def epoch(self):
        return self._ds.epoch

    @epoch.setter
    def epoch(self, value):
        self._ds.epoch = int(value)

    def __getattr__(self, k):
        if k == "_ds":
            raise AttributeError(k)
        return getattr(self._ds, k)


def build_dataset(spec):
    """Rebuild a window dataset from its spec. Imports are local so that
    importing this module (which the parent does before spawning) does not drag
    in the dataset modules' own dependencies until a worker actually needs them."""
    if isinstance(spec, ConcatSpec):
        from jarvis_jax.data.concat_windows import ConcatWindowDataset
        return ConcatWindowDataset([build_dataset(s) for s in spec.specs],
                                   names=list(spec.names),
                                   allow_calib_mismatch=spec.allow_calib_mismatch)
    if not isinstance(spec, V12Spec):
        raise TypeError(f"not a window-dataset spec: {type(spec).__name__}")
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ds = V12WindowDataset(spec.root, spec.split, T=spec.T, pair_deltas=tuple(spec.pair_deltas),
                          max_flies=spec.max_flies, jitter_units=spec.jitter_units,
                          seed=spec.seed, train=spec.train,
                          recordings=(set(spec.recordings) if spec.recordings is not None else None),
                          copy_paste=spec.copy_paste,
                          center_shift_units=spec.center_shift_units,
                          sex_overrides=dict(spec.sex_overrides or {}))
    if spec.force_sample_weight is not None:
        ds = _ForcedWeightView(ds, spec.force_sample_weight)
    return ds


def dataset_spec(ds):
    """The spec for `ds`, asking the object itself (`worker_spec`).

    Every wrapper in the stack (`train_mvq._DatasetView` and its subclasses)
    implements `worker_spec`, so a view is transparent here exactly as it is to
    the sampler. A dataset that cannot describe itself raises rather than
    silently loading something else in the worker."""
    fn = getattr(ds, "worker_spec", None)
    if fn is None:
        raise TypeError(f"{type(ds).__name__} has no worker_spec(): it cannot be rebuilt in a "
                        f"loader worker process (use workers='threads')")
    return fn()


# ------------------------------------------------------------------- workers
_WORKER_DS = {}


_NO_GPU_ENV = {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}


def _init_worker(specs):
    # Belt to the braces of `_child_env` below: a worker must never claim a GPU.
    for k, v in _NO_GPU_ENV.items():
        os.environ.setdefault(k, v)
    global _WORKER_DS
    # Every worker rebuilds the same roots, so it would re-print the same
    # sex-disagreement and calibration-mismatch banners the PARENT already
    # printed -- 24 copies of them, ahead of step 0, in the run log. Only the
    # build is silenced (stderr and any exception still surface), so a worker
    # that fails to build still says why.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _WORKER_DS = {k: build_dataset(s) for k, s in specs.items()}


def _get_item(task):
    key, epoch, i = task
    ds = _WORKER_DS[key]
    ds.epoch = int(epoch)          # so jitter/copy-paste RNG matches the parent's epoch
    return ds[int(i)]


@contextlib.contextmanager
def _child_env():
    """Spawn children snapshot `os.environ` at exec, so pin them to the CPU
    HERE rather than in the initializer.

    A spawn child re-imports the parent's `__main__` module (as `__mp_main__`)
    before the pool's initializer ever runs -- and this loader's `__main__` is
    `jarvis_jax.scripts.train_mvq`, whose import chain pulls in JAX. That import
    happens too early for `_init_worker` to guard, and a worker that resolved a
    GPU backend would sit on a device the parent already owns. The parent's own
    backend is initialised long before any pool is built (`data_parallel_mesh`),
    so setting and restoring these two variables around the spawn does not touch
    it. The `__main__` guard in the entry script is what stops the re-import
    from re-running the trainer; this only stops it from taking a device."""
    old = {k: os.environ.get(k) for k in _NO_GPU_ENV}
    os.environ.update(_NO_GPU_ENV)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ProcessSampleLoader:
    """A spawn pool of workers, each holding its own copy of the dataset(s).

    `specs` is {key: spec}; the key selects which dataset a task addresses, so
    the T=1 and T=2 streams of an mvq run share ONE pool (and therefore one
    slice of the node's cores) instead of running 2 x num_workers processes
    against 32 cores.

    `map_batches` keeps `inflight` batches submitted at all times so the
    workers stay busy while the parent collates and the GPU step runs; results
    come back in submission order, so the batch a consumer sees is the batch
    the sampler drew.
    """

    def __init__(self, specs, num_workers, inflight=3):
        if not isinstance(specs, dict):
            specs = {None: specs}
        self.keys = tuple(specs)
        self.num_workers = max(1, int(num_workers))
        self.inflight = max(1, int(inflight))
        ctx = multiprocessing.get_context("spawn")
        with _child_env():
            self._pool = ctx.Pool(self.num_workers, initializer=_init_worker, initargs=(specs,))
        self._closed = False

    def map_batches(self, index_lists, epoch, key=None):
        pending = collections.deque()
        it = iter(index_lists)

        def submit():
            try:
                nxt = [int(i) for i in next(it)]
            except StopIteration:
                return False
            pending.append((nxt, self._pool.map_async(
                _get_item, [(key, int(epoch), i) for i in nxt], chunksize=1)))
            return True

        for _ in range(self.inflight):
            if not submit():
                break
        while pending:
            bidx, res = pending.popleft()
            samples = res.get()
            submit()
            yield bidx, samples

    def close(self):
        if not self._closed:
            self._closed = True
            self._pool.terminate()
            self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
