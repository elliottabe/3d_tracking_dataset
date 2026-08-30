"""Precompute reprojected volumes for the HybridNet cached trainer.

Sweeps all framesets in the requested split, runs a frozen 2-D front-end +
reproject (HybridNet3D.reproject_volume), and streams fp16 volumes to a
memmap so the v2vNet trainer can skip the expensive front-end on every step.

The frameset dataset class is SELECTABLE via ``cache.dataset_version``
(default ``"v5"``): ``"v5"`` loads ``jarvis_jax.data.v5_3d.
V5FramesetDataset`` (red_data_3d_v5, per-fly framesets, calibration by
group); ``"v3"`` preserves the legacy ``jarvis_jax.data.v3_3d.
V3FramesetDataset`` path (red_data_unified_V3). See
``run_precompute``'s ``dataset_version`` docstring for details.

The 2-D front-end is SELECTABLE via ``cache.front_end`` (default
``"vitpose"``, byte-identical to the pre-selector behavior):

  - ``front_end=vitpose`` (default): loads ``paths.vitpose_ckpt`` via
    ``load_vitpose``, exactly as before.
  - ``front_end=efficienttrack_bn``: restores an ``EfficientTrackBN``
    checkpoint from ``cache.frontend_ckpt`` (an Orbax "final" dir, e.g.
    ``.../jax_efficienttrack_runs/et2d_bn_imagenet/final``), the SAME
    eval_shape + StandardCheckpointer + nnx.merge pattern as
    ``jarvis_jax.scripts.eval_keypoints_2d.restore_model``. Because that
    checkpoint was trained by the 2-D trainer's ``loss_fn`` (which does
    ``img = normalize_image(img4_u8)`` before calling the model --
    ImageNet-normalizes RGB, passes the mask channel through raw), the
    HybridNet3D built here is constructed with ``normalize_frontend=True`` so
    ``reproject_volume``/``predict_heatmaps`` feeds it the SAME
    normalize_image-preprocessed input it saw at 2-D training time (NOT the
    raw-float path used by the faithful-port plain ``EfficientTrack``, whose
    default stays byte-identical -- see ``jarvis_jax/hybridnet/model.py``).

Usage (Hydra)
-------------
    # ViTPose arm (default, unchanged)
    python scripts/precompute_repro_cache.py \\
        paths=hyak cache=default \\
        cache.split=val \\
        cache.batch=8 \\
        cache.limit=0 \\
        cache.force=false

    # EfficientTrackBN arm (own cache_dir -- see paths.cache_dir override)
    python scripts/precompute_repro_cache.py \\
        paths=hyak cache=default \\
        cache.front_end=efficienttrack_bn \\
        cache.frontend_ckpt=/gscratch/portia/$USER/data/Johnson_lab/jax_efficienttrack_runs/et2d_bn_imagenet/final \\
        paths.cache_dir=/gscratch/portia/$USER/data/Johnson_lab/jax_repro_cache/v3_etbn \\
        cache.split=val

Per-arm cache directory: there is no separate config group for this -- rely
on overriding ``paths.cache_dir=<dir>`` per arm (e.g. one dir per front_end)
so the ViTPose and EfficientTrackBN caches never share a directory.

Idempotent: if <split>_meta.json exists and the 'front_end', 'frontend_ckpt'
(generic keys; back-compat also reads the legacy 'vitpose_ckpt' key when
'front_end' is absent), 'grid_size', 'grid_spacing', 'roi_cube',
'heatmap_size' keys match (and --limit matches n), and the volume file has
the expected size, the precompute is skipped unless cache.force=true is
given. A front-end/ckpt MISMATCH (e.g. requesting efficienttrack_bn against a
cache built with vitpose) is treated as stale and forces a rebuild -- it is
never silently reused.
"""

from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# Add repo root to path when run directly (not as a module)
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import hydra
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass

register_resolvers()

# Fixed volume shape constants — kept in sync with repro_cache.py
_NUM_JOINTS = 50
_GRID = 48


# Grid params written into meta — single source of truth for the staleness check
_META_GRID_PARAMS = {
    "grid_size": 48,
    "grid_spacing": 1,
    "roi_cube": 48,
    # heatmap_size is BOUNDING_BOX/2 + 2 = 448/2 + 2 = 226
    "heatmap_size": 226,
}


def _cache_is_valid(
    cache_dir: str,
    split: str,
    front_end: str,
    frontend_ckpt: str | None,
    n: int,
) -> bool:
    """Return True if a valid, matching, complete cache already exists.

    Checks:
      - meta.json matches front_end + frontend_ckpt, n, and all grid params
      - volume file exists and has the expected byte size (guards truncated files)

    Back-compat: caches written before the front-end selector existed only
    ever wrote a bare 'vitpose_ckpt' key (no 'front_end' key at all, since
    ViTPose was the only option). Such a cache is treated as
    front_end='vitpose' with frontend_ckpt taken from the legacy key, so
    existing ViTPose caches remain valid without a forced rebuild. Any
    front_end/ckpt MISMATCH (including a legacy cache being reused for
    front_end='efficienttrack_bn') is treated as stale -> rebuild, never
    silently reused.
    """
    meta_path = os.path.join(cache_dir, f"{split}_meta.json")
    vol_path = os.path.join(cache_dir, f"{split}_volumes.f16")
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        meta_front_end = meta.get("front_end", "vitpose")
        # Generic key first; fall back to the legacy vitpose-only key.
        meta_ckpt = meta.get("frontend_ckpt", meta.get("vitpose_ckpt"))
        if meta_front_end != front_end:
            return False
        if meta_ckpt != frontend_ckpt:
            return False
        if int(meta.get("n", -1)) != n:
            return False
        for key, expected in _META_GRID_PARAMS.items():
            if meta.get(key) != expected:
                return False
        # Guard against truncated volume files (e.g. killed mid-write)
        expected_bytes = n * _NUM_JOINTS * _GRID * _GRID * _GRID * 2  # fp16 = 2 bytes
        if not os.path.isfile(vol_path):
            return False
        if os.path.getsize(vol_path) != expected_bytes:
            return False
        return True
    except Exception:
        return False


_FRONT_ENDS = ("vitpose", "efficienttrack_bn")


def _build_front_end(front_end: str, *, vitpose_ckpt, vitpose_cfg, frontend_ckpt):
    """Construct + restore the frozen 2-D front-end for `front_end`.

    Returns (front_end_module, ckpt_used, normalize_frontend):
      - front_end='vitpose': loads `vitpose_ckpt` via load_vitpose (exactly
        the pre-selector path). normalize_frontend=False -- HybridNet3D
        always ImageNet-normalizes ViTPose input via its own `_is_vitpose`
        branch, independent of this flag.
      - front_end='efficienttrack_bn': restores `frontend_ckpt` (an Orbax
        "final" dir) via `eval_keypoints_2d.restore_model`'s eval_shape +
        StandardCheckpointer + nnx.merge pattern -- the SAME restore path
        used to load trained EfficientTrackBN checkpoints elsewhere.
        normalize_frontend=True is REQUIRED: this checkpoint was trained by
        the 2-D trainer's loss_fn, which normalizes input
        (`img = normalize_image(img4_u8)`) before calling the model, so
        HybridNet3D must be told to do the same at cache-build time (see
        `jarvis_jax/hybridnet/model.py`'s `normalize_frontend` flag) --
        feeding it raw crops (the plain-EfficientTrack HybridNetBackbone-
        parity default) would silently mismatch train/inference
        normalization.
    """
    if front_end not in _FRONT_ENDS:
        raise ValueError(
            f"unknown front_end {front_end!r} (expected one of {_FRONT_ENDS})")

    if front_end == "vitpose":
        if not vitpose_ckpt:
            raise ValueError("front_end='vitpose' requires a vitpose_ckpt "
                              "(paths.vitpose_ckpt)")
        from jarvis_jax.convert.build_checkpoint import load_vitpose
        print(f"[precompute] Loading ViTPose from {vitpose_ckpt}")
        model = load_vitpose(vitpose_ckpt, vitpose_cfg)
        return model, vitpose_ckpt, False

    # front_end == "efficienttrack_bn"
    if not frontend_ckpt:
        raise ValueError("front_end='efficienttrack_bn' requires a "
                          "frontend_ckpt (cache.frontend_ckpt=...)")
    from jarvis_jax.scripts.eval_keypoints_2d import restore_model
    print(f"[precompute] Restoring EfficientTrackBN from {frontend_ckpt}")
    model = restore_model(frontend_ckpt, "efficienttrack_bn", vitpose_cfg)
    return model, frontend_ckpt, True


def run_precompute(
    *,
    root: str,
    cache_dir: str,
    split: str,
    batch: int,
    limit: int,
    force: bool,
    vitpose_ckpt: str | None = None,
    front_end: str = "vitpose",
    frontend_ckpt: str | None = None,
    vitpose_cfg=None,
    dataset_version: str = "v5",
):
    """Precompute reprojected volumes and write them to a memmap cache.

    Args:
        root:        Root of the dataset (v5: contains annotations/, images/,
                     masks/, calibrations/, manifest.json; v3: contains
                     annotations/, train/, val/) -- see `dataset_version`.
        cache_dir:   Output directory for the cache files. Callers building
                     BOTH a ViTPose and an EfficientTrackBN cache MUST pass a
                     different cache_dir per arm -- this function does not
                     namespace cache_dir by front_end itself (see module
                     docstring); the staleness check (_cache_is_valid) will
                     force a rebuild rather than silently reuse a
                     different-front-end cache in the SAME cache_dir, but two
                     concurrent/successive arms sharing one cache_dir will
                     still clobber each other's files on disk.
        split:       Dataset split: 'train' or 'val'.
        batch:       Framesets per batch.
        limit:       Cache only the first N framesets; 0 = all.
        force:       Overwrite existing cache even if meta matches.
        vitpose_ckpt: Orbax ViTPose checkpoint directory. Required when
                     front_end='vitpose' (the default); ignored otherwise.
        front_end:   'vitpose' (default) | 'efficienttrack_bn'. Selects the
                     frozen 2-D front-end used to build the cache.
        frontend_ckpt: Orbax checkpoint directory for a non-vitpose front_end
                     (e.g. an EfficientTrackBN 'final' dir). Required when
                     front_end != 'vitpose'; ignored otherwise.
        vitpose_cfg: Optional ViTPoseConfig instance (doubles as the generic
                     num_keypoints/in_ch container for EfficientTrackBN, same
                     convention as eval_keypoints_2d.py); built from defaults
                     if None.
        dataset_version: 'v5' (default) | 'v3'. Selects the frameset dataset
                     class: 'v5' loads `jarvis_jax.data.v5_3d.
                     V5FramesetDataset` (red_data_3d_v5, per-fly framesets,
                     calibration looked up by group); 'v3' preserves the
                     legacy `jarvis_jax.data.v3_3d.V3FramesetDataset` path
                     (red_data_unified_V3). Both expose an identical
                     `__getitem__` schema (see v5_3d.py module docstring) so
                     nothing else in this file needs to branch on it.
    """
    from flax import nnx

    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.data.repro_cache import load_cache, write_cache
    from jarvis_jax.hybridnet.model import HybridNet3D
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.sharding import data_parallel_mesh, replicate, shard_batch

    if front_end not in _FRONT_ENDS:
        raise ValueError(
            f"unknown front_end {front_end!r} (expected one of {_FRONT_ENDS})")

    if str(dataset_version) == "v5":
        from jarvis_jax.data.v5_3d import V5FramesetDataset as FramesetDataset
        from jarvis_jax.data.v5_3d import frameset_batches  # noqa: F401
    else:
        from jarvis_jax.data.v3_3d import V3FramesetDataset as FramesetDataset
        from jarvis_jax.data.v3_3d import frameset_batches  # noqa: F401

    if vitpose_cfg is None:
        vitpose_cfg = ViTPoseConfig()

    # ------------------------------------------------------------------
    # Load dataset to determine n
    # ------------------------------------------------------------------
    print(f"[precompute] Loading dataset: {root} split={split} "
          f"(dataset_version={dataset_version})")
    ds = FramesetDataset(root, split)
    total = len(ds)
    n = min(limit, total) if limit > 0 else total
    print(f"[precompute] {total} framesets in split; will cache {n}")

    # ------------------------------------------------------------------
    # Idempotency check (cheap: resolve which ckpt this arm uses without
    # constructing/restoring any model yet)
    # ------------------------------------------------------------------
    ckpt_for_check = vitpose_ckpt if front_end == "vitpose" else frontend_ckpt
    if not force and _cache_is_valid(cache_dir, split, front_end, ckpt_for_check, n):
        print(f"[precompute] Cache already valid at {cache_dir} "
              f"(split={split}, n={n}, front_end={front_end}). "
              f"Use cache.force=true to recompute.")
        return

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    front_end_model, ckpt_used, normalize_frontend = _build_front_end(
        front_end, vitpose_ckpt=vitpose_ckpt, vitpose_cfg=vitpose_cfg,
        frontend_ckpt=frontend_ckpt)
    v2vnet = V2VNet(vitpose_cfg.num_keypoints, vitpose_cfg.num_keypoints, rngs=nnx.Rngs(0))
    model = HybridNet3D(front_end_model, v2vnet, vitpose_cfg,
                         normalize_frontend=normalize_frontend)

    # ------------------------------------------------------------------
    # Replicate model across devices
    # ------------------------------------------------------------------
    mesh = data_parallel_mesh()
    num_devices = len(jax.devices())
    print(f"[precompute] {num_devices} device(s); replicating model")
    graphdef, state = nnx.split(model)
    state = replicate(state, mesh)
    model = nnx.merge(graphdef, state)

    # JIT the reproject_volume call for performance
    @jax.jit
    def _reproject(crops4, center3D, centerHM, cameraMatrices):
        return model.reproject_volume(crops4, center3D, centerHM, cameraMatrices)

    # ------------------------------------------------------------------
    # Build meta dict
    # ------------------------------------------------------------------
    meta = dict(_META_GRID_PARAMS)
    # Generic front-end stamp -- distinguishes a ViTPose cache from an
    # EfficientTrackBN cache and lets _cache_is_valid catch a mismatch.
    meta["front_end"] = front_end
    meta["frontend_ckpt"] = ckpt_used
    # Informational only -- NOT part of _cache_is_valid's staleness gate (that
    # gate is keyed on front_end/ckpt/n/grid params, unchanged here). Records
    # which frameset dataset class built this cache (v5's V5FramesetDataset
    # vs v3's V3FramesetDataset) for provenance when reading meta.json later.
    meta["dataset_version"] = str(dataset_version)
    # Back-compat: also write the legacy 'vitpose_ckpt' key when this IS a
    # ViTPose cache, so any older reader that only knows that key (e.g.
    # train_3d_cached.py's informational wandb log) keeps working unchanged.
    # Deliberately NOT written for front_end='efficienttrack_bn' -- that
    # front-end is not a ViTPose, and writing a non-ViTPose path under a key
    # named 'vitpose_ckpt' would be misleading.
    if front_end == "vitpose":
        meta["vitpose_ckpt"] = ckpt_used
    meta["num_cameras"] = int(ds[0]["cameraMatrices"].shape[0])
    # Carry the skeleton so the cached trainer can wire the graph-Laplacian
    # bone prior (train_3d_cached reads meta["keypoint_names"]/["skeleton"]).
    # Without this the cached run silently trains with zero edges.
    meta["keypoint_names"] = list(ds.keypoint_names)
    meta["skeleton"] = ds.skeleton

    # ------------------------------------------------------------------
    # Preallocate label accumulators (filled as volumes are generated)
    # ------------------------------------------------------------------
    kp3d_acc = np.zeros((n, _NUM_JOINTS, 3), dtype=np.float32)
    c3d_acc = np.zeros((n, 3), dtype=np.float32)
    vis_acc = np.zeros((n, _NUM_JOINTS), dtype=bool)

    # ------------------------------------------------------------------
    # Generator: yields per-frameset volumes, fills label accumulators
    # as a side effect so write_cache can consume it fully before labels
    # are passed.  Labels are preallocated above and passed by reference;
    # write_cache consumes the generator (filling them), then writes labels.
    # ------------------------------------------------------------------
    indices = np.arange(n)  # first n, in order (shuffle=False)
    batch_size = batch
    t0 = time.perf_counter()
    written_count = [0]  # mutable counter accessible inside generator

    def _volumes_generator():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sel = indices[start:end]
            B = len(sel)

            # Load samples (manual index sweep to enforce --limit exactly)
            samples = [ds[int(i)] for i in sel]
            batch_data = {key: np.stack([s[key] for s in samples], axis=0)
                         for key in samples[0]}

            # Save original labels before padding inputs
            kp3d_batch = batch_data["kp3d"]
            c3d_batch = batch_data["center3D"]
            vis_batch = batch_data["vis"]

            # Pad inputs on axis 0 to device-multiple if necessary (fixes IndivisibleError)
            # on partial batches. shard_batch requires axis-0 divisible by len(jax.devices()).
            nd = len(jax.devices())
            pad = (-B) % nd
            if pad > 0:
                batch_data["crops4"] = np.concatenate(
                    [batch_data["crops4"], np.repeat(batch_data["crops4"][-1:], pad, axis=0)], axis=0)
                batch_data["center3D"] = np.concatenate(
                    [batch_data["center3D"], np.repeat(batch_data["center3D"][-1:], pad, axis=0)], axis=0)
                batch_data["centerHM"] = np.concatenate(
                    [batch_data["centerHM"], np.repeat(batch_data["centerHM"][-1:], pad, axis=0)], axis=0)
                batch_data["cameraMatrices"] = np.concatenate(
                    [batch_data["cameraMatrices"], np.repeat(batch_data["cameraMatrices"][-1:], pad, axis=0)], axis=0)

            # Shard inputs across devices
            with mesh:
                crops4 = shard_batch(jnp.asarray(batch_data["crops4"]), mesh)
                center3D_b = shard_batch(jnp.asarray(batch_data["center3D"]), mesh)
                centerHM = shard_batch(jnp.asarray(batch_data["centerHM"]), mesh)
                camMat = shard_batch(jnp.asarray(batch_data["cameraMatrices"]), mesh)

                vol = _reproject(crops4, center3D_b, centerHM, camMat)

            # Pull to host; vol shape: (B_padded, 50, 48, 48, 48)
            vol_np = np.asarray(vol)
            # Slice off padding to restore original batch size
            vol_np = vol_np[:B]

            # Accumulate labels for this batch (use original unpadded batch)
            kp3d_acc[start:end] = kp3d_batch
            c3d_acc[start:end] = c3d_batch
            vis_acc[start:end] = vis_batch

            # Yield individual frameset volumes (fp16)
            for b in range(B):
                vol_b = vol_np[b].astype(np.float16)
                written_count[0] += 1
                elapsed = time.perf_counter() - t0
                fps = written_count[0] / elapsed
                print(f"[precompute] {written_count[0]}/{n} framesets  "
                      f"({fps:.1f} fs/s)  vol_range=["
                      f"{float(vol_b.min()):.3f}, {float(vol_b.max()):.3f}]")
                yield vol_b

    # ------------------------------------------------------------------
    # write_cache: consumes the generator (filling label accumulators),
    # then writes labels + meta in one place (no duplicated logic here)
    # ------------------------------------------------------------------
    write_cache(
        cache_dir,
        split,
        volumes_iter=_volumes_generator(),
        n=n,
        kp3d=kp3d_acc,
        center3D=c3d_acc,
        vis=vis_acc,
        meta=meta,
    )

    elapsed = time.perf_counter() - t0
    vol_path = os.path.join(cache_dir, f"{split}_volumes.f16")
    vol_size_gb = os.path.getsize(vol_path) / 1e9
    print(f"\n[precompute] Done in {elapsed:.1f}s")
    print(f"[precompute] {vol_path}  ({vol_size_gb:.2f} GB)")
    print(f"[precompute] labels: {split}_labels.npz")
    print(f"[precompute] meta:   {split}_meta.json")

    # ------------------------------------------------------------------
    # Quick load_cache round-trip check
    # ------------------------------------------------------------------
    print("\n[precompute] Round-trip check via load_cache ...")
    c = load_cache(cache_dir, split)
    assert c["volumes"].shape == (n, _NUM_JOINTS, _GRID, _GRID, _GRID), (
        f"Unexpected shape: {c['volumes'].shape}")
    assert c["volumes"].dtype == np.float16
    vols = np.asarray(c["volumes"])
    finite_frac = float(np.isfinite(vols).mean())
    vmin = float(vols.min())
    vmax = float(vols.max())
    print(f"[precompute] volumes shape={c['volumes'].shape}  "
          f"dtype={c['volumes'].dtype}  finite={finite_frac:.6f}  "
          f"range=[{vmin:.4f}, {vmax:.4f}]")
    assert finite_frac == 1.0, f"Non-finite values in cached volumes: {1-finite_frac:.2%}"
    print("[precompute] Round-trip OK.")


def main_from_cfg(cfg):
    """Map a composed Hydra config into run_precompute kwargs and run.

    ``cache.front_end`` (default 'vitpose') / ``cache.frontend_ckpt`` (default
    None) are read via ``.get`` with the pre-selector defaults so composing
    against a ``cache`` group that predates these keys still works
    byte-identically (front_end='vitpose', frontend_ckpt=None -> unused).

    ``cache.dataset_version`` (default 'v5') selects the frameset dataset
    class -- see `run_precompute`'s `dataset_version` docstring. A `cache`
    group composed before this key existed still defaults to 'v5' via
    ``.get``; pass ``cache.dataset_version=v3`` to build against the legacy
    red_data_unified_V3 tree instead.
    """
    from jarvis_jax.config import ViTPoseConfig
    vitpose_cfg = build_dataclass(ViTPoseConfig, cfg.model.vitpose)
    return run_precompute(
        root=cfg.paths.data_root,
        vitpose_ckpt=cfg.paths.vitpose_ckpt,
        cache_dir=cfg.paths.cache_dir,
        split=cfg.cache.split,
        batch=cfg.cache.batch,
        limit=cfg.cache.limit,
        force=cfg.cache.force,
        front_end=cfg.cache.get("front_end", "vitpose"),
        frontend_ckpt=cfg.cache.get("frontend_ckpt", None),
        vitpose_cfg=vitpose_cfg,
        dataset_version=cfg.cache.get("dataset_version", "v5"),
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
