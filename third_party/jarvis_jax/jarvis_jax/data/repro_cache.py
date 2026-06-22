"""Reprojected-volume cache: writer and reader.

Stores per-frameset reprojected volumes (fp16) and labels so the v2vNet
trainer can skip frozen ViTPose + reproject every step.

Layout on disk
--------------
<cache_dir>/
    <split>_volumes.f16      np.memmap (n, 50, 48, 48, 48) float16
    <split>_labels.npz       kp3d (n,50,3) f32, center3D (n,3) f32, vis (n,50) bool
    <split>_meta.json        dict with vitpose_ckpt, grid params, n, split

Public API
----------
write_cache(cache_dir, split, *, volumes_iter, n, kp3d, center3D, vis, meta)
    Stream n volumes from `volumes_iter` (each (50,48,48,48) fp16 or f32)
    to the memmap WITHOUT holding all data in RAM, then write labels.npz
    and meta.json.

load_cache(cache_dir, split) -> dict
    Returns:
        volumes   : np.memmap (n,50,48,48,48) float16 (mode='r')
        kp3d      : np.ndarray (n,50,3) float32
        center3D  : np.ndarray (n,3)   float32
        vis       : np.ndarray (n,50)  bool
        meta      : dict

to_device_sharded(cache, mesh) -> dict
    Place cache arrays as JAX arrays sharded on axis 0 across mesh.
    Trims n to the largest multiple of device count.
    Returns:
        volumes   : jnp array (n_used,50,48,48,48) float16, sharded P("data")
        kp3d      : jnp array (n_used,50,3) float32, sharded P("data")
        center3D  : jnp array (n_used,3)   float32, sharded P("data")
        vis       : jnp array (n_used,50)  bool,    sharded P("data")
        n_used    : int

cached_batches(dev_cache, batch_size, *, shuffle=True, seed=0, drop_last=True) -> iterator
    Yield {volumes, kp3d, center3D, vis} batches from device-resident arrays.
    Each yielded batch is re-sharded with P("data") to stay device-resident.
    batch_size must be divisible by device count.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

import numpy as np

# JAX imports deferred to function bodies so this module stays importable even
# without a JAX installation (e.g. during write-only cache-building jobs).
# They are imported at the top of each function that needs them.

# Fixed volume shape constants — kept in sync with HybridNet3D.reproject_volume
_NUM_JOINTS = 50
_GRID = 48  # spatial dimension of the reprojected volume

_VOL_SHAPE = (_NUM_JOINTS, _GRID, _GRID, _GRID)  # per-frameset


def _vol_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"{split}_volumes.f16")


def _labels_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"{split}_labels.npz")


def _meta_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"{split}_meta.json")


# ---------------------------------------------------------------------------
# write_cache
# ---------------------------------------------------------------------------

def write_cache(
    cache_dir: str,
    split: str,
    *,
    volumes_iter: Iterable[np.ndarray],
    n: int,
    kp3d: np.ndarray,        # (n, 50, 3) float32
    center3D: np.ndarray,    # (n, 3)     float32
    vis: np.ndarray,         # (n, 50)    bool
    meta: dict,
) -> None:
    """Write reprojected volumes and labels to cache_dir.

    Volumes are streamed one-by-one from *volumes_iter* to a memmap so that
    the full (n, 50, 48, 48, 48) array (up to ~29 GB for n~3 k) is never
    held in RAM simultaneously.

    Args:
        cache_dir:     Directory to write files into (created if absent).
        split:         Dataset split name, e.g. 'train' or 'val'.
        volumes_iter:  Iterable yielding n arrays of shape (50,48,48,48).
                       Each element may be float16 or float32; stored as fp16.
        n:             Total number of framesets expected from the iterator.
        kp3d:          ``(n, 50, 3)`` float32 triangulated 3-D keypoints.
        center3D:      ``(n, 3)`` float32 3-D ROI centres.
        vis:           ``(n, 50)`` bool visibility flags.
        meta:          Dict written to meta.json (must include 'vitpose_ckpt'
                       and 'grid_size'; 'n' and 'split' are injected/overridden).
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Open memmap for writing — shape (n, J, G, G, G) fp16
    mm = np.memmap(
        _vol_path(cache_dir, split),
        dtype=np.float16,
        mode="w+",
        shape=(n, *_VOL_SHAPE),
    )

    written = 0
    for vol in volumes_iter:
        if written >= n:
            break
        vol = np.asarray(vol)
        if vol.dtype != np.float16:
            vol = vol.astype(np.float16)
        mm[written] = vol
        written += 1

    if written != n:
        raise ValueError(
            f"write_cache: volumes_iter yielded {written} items but n={n}")

    # Flush to disk (no-op for w+ mode but explicit is clearer)
    mm.flush()
    del mm  # release file handle

    # Write labels
    np.savez(
        _labels_path(cache_dir, split),
        kp3d=np.asarray(kp3d, dtype=np.float32),
        center3D=np.asarray(center3D, dtype=np.float32),
        vis=np.asarray(vis, dtype=bool),
    )

    # Write meta — inject / normalise n and split
    meta_out = dict(meta)
    meta_out["n"] = n
    meta_out["split"] = split
    with open(_meta_path(cache_dir, split), "w") as f:
        json.dump(meta_out, f, indent=2)


# ---------------------------------------------------------------------------
# load_cache
# ---------------------------------------------------------------------------

def load_cache(cache_dir: str, split: str) -> dict:
    """Load a previously-written cache.

    Returns a dict with:
        volumes   : np.memmap (n, 50, 48, 48, 48) float16, mode='r'
        kp3d      : np.ndarray (n, 50, 3) float32
        center3D  : np.ndarray (n, 3)     float32
        vis       : np.ndarray (n, 50)    bool
        meta      : dict
    """
    meta_file = _meta_path(cache_dir, split)
    if not os.path.isfile(meta_file):
        raise FileNotFoundError(
            f"Cache meta not found: {meta_file}. "
            f"Run precompute_repro_cache.py first.")

    with open(meta_file) as f:
        meta = json.load(f)

    n = int(meta["n"])

    mm = np.memmap(
        _vol_path(cache_dir, split),
        dtype=np.float16,
        mode="r",
        shape=(n, *_VOL_SHAPE),
    )

    with np.load(_labels_path(cache_dir, split)) as f:
        kp3d = f["kp3d"].copy()
        center3D = f["center3D"].copy()
        vis = f["vis"].copy()

    return {
        "volumes": mm,
        "kp3d": kp3d,
        "center3D": center3D,
        "vis": vis,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# to_device_sharded
# ---------------------------------------------------------------------------

def to_device_sharded(cache: dict, mesh) -> dict:
    """Place cache arrays as JAX device-resident arrays sharded on axis 0.

    Trims the first axis (framesets) to the largest multiple of the mesh
    device count so that axis-0 shards evenly.  Logs the number of dropped
    framesets if any are discarded.

    Args:
        cache: Dict as returned by :func:`load_cache` (or a synthetic dict with
               the same keys: ``volumes``, ``kp3d``, ``center3D``, ``vis``).
               ``volumes`` may be a np.memmap or np.ndarray (fp16 or fp32).
        mesh:  A :class:`jax.sharding.Mesh` with axis name ``"data"`` (e.g. from
               :func:`jarvis_jax.sharding.data_parallel_mesh`).

    Returns:
        Dict with keys ``volumes`` (fp16), ``kp3d``, ``center3D``, ``vis`` as
        sharded JAX arrays, plus ``n_used`` (int).
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    nd = len(mesh.devices)
    n = cache["volumes"].shape[0]
    n_used = (n // nd) * nd

    if n_used < n:
        dropped = n - n_used
        print(
            f"to_device_sharded: trimming {n} -> {n_used} framesets "
            f"(dropped {dropped} to align to {nd} devices)"
        )

    sharding = NamedSharding(mesh, P("data"))

    def _put(arr, dtype=None):
        # Read the memmap slice into RAM first (avoids repeated memmap seeks),
        # then hand to device_put with the target sharding.
        np_arr = np.asarray(arr[:n_used])
        if dtype is not None:
            np_arr = np_arr.astype(dtype)
        return jax.device_put(jnp.asarray(np_arr), sharding)

    volumes = _put(cache["volumes"], dtype=np.float16)
    kp3d    = _put(cache["kp3d"])
    center3D = _put(cache["center3D"])
    vis     = _put(cache["vis"])

    return {
        "volumes":  volumes,
        "kp3d":     kp3d,
        "center3D": center3D,
        "vis":      vis,
        "n_used":   n_used,
    }


# ---------------------------------------------------------------------------
# cached_batches
# ---------------------------------------------------------------------------

def cached_batches(
    dev_cache: dict,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> Iterator[dict]:
    """Yield batches from GPU-resident sharded cache arrays.

    Generates index batches (shuffled if requested), gathers rows from the
    device-resident arrays, then re-shards each batch with ``P("data")`` so
    the result stays device-resident.  No explicit host transfer occurs per
    step: JAX gathers across shards and the re-shard keeps the result on GPU.

    Args:
        dev_cache:   Dict returned by :func:`to_device_sharded`.
        batch_size:  Number of framesets per batch.  Must be divisible by the
                     device count (the sharding requires even splitting).
        shuffle:     Whether to shuffle frameset indices before batching.
        seed:        RNG seed used when *shuffle* is True.
        drop_last:   If True (default), drop the final partial batch.

    Yields:
        Dicts with keys ``volumes`` ``(B,50,48,48,48)`` fp16, ``kp3d``
        ``(B,50,3)`` f32, ``center3D`` ``(B,3)`` f32, ``vis`` ``(B,50)`` bool;
        each array sharded on axis 0 with ``P("data")``.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    n_used = dev_cache["n_used"]
    nd = len(dev_cache["volumes"].sharding.mesh.devices)

    if batch_size % nd != 0:
        raise ValueError(
            f"cached_batches: batch_size={batch_size} must be divisible by "
            f"device count nd={nd}"
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(n_used)
    if shuffle:
        rng.shuffle(indices)

    # Determine which mesh / sharding to use from the stored arrays
    sharding = dev_cache["volumes"].sharding

    keys = ("volumes", "kp3d", "center3D", "vis")

    n_batches = n_used // batch_size
    if not drop_last and (n_used % batch_size) != 0:
        n_batches += 1

    for i in range(n_batches):
        idx = indices[i * batch_size : (i + 1) * batch_size]
        if len(idx) < batch_size and drop_last:
            break
        # Gather rows from each sharded array then re-shard the result.
        # arr[idx] triggers a JAX gather; re-shard with device_put keeps it
        # on the same devices without a host round-trip.
        batch = {}
        for k in keys:
            gathered = dev_cache[k][idx]
            batch[k] = jax.device_put(gathered, sharding)
        yield batch
