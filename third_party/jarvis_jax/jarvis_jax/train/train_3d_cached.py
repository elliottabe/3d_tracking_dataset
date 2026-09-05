"""Cached v2vNet 3D trainer.

Trains ONLY the v2vNet on a GPU-resident precomputed reprojected-volume cache.
No per-step ViTPose inference, reprojection, or host-to-device transfer.

Per-step transform (matches HybridNet3D.__call__ post-reproject path exactly)
-----------------------------------------------------------------------------
  batch['volumes']  (B, J, 48, 48, 48) fp16        ← cached reproject_volume output
  → astype(f32)
  → transpose (0,2,3,4,1) → (B, 48, 48, 48, J)    ← channels-last for V2VNet
  → v2vnet(use_running_average=False)
  → (B, 24, 24, 24, J)
  → transpose (0,4,1,2,3) → (B, J, 24, 24, 24)    ← joint-first for soft_argmax
  → softplus                                         ← match model.py step 9
  → soft_argmax_3d(grid_spacing=1, roi_cube=48)
  → heatmap3d_mse + lw * graph_laplacian

CLI (Hydra; see configs/):
    python -m jarvis_jax.train.train_3d_cached run_id=myrun train=cached3d \\
        train.total_steps=20000 train.sharpen=3 paths=hyak

LEGACY: cached3d/HybridNet lost the A/B to ViTPose->DLT/IK (see
docs/benchmark ab-hybridnet-vs-dlt-ik); its repro-volume cache and run dirs
were deleted in the 2026-09-05 storage cleanup and `paths.cache_dir` no
longer exists. Pass `paths.cache_dir=<dir>` explicitly and rebuild the cache
(scripts/precompute_repro_cache.py) if this trainer is ever revived.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx.transforms.autodiff import DiffState

from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass, run_dir_for

register_resolvers()

from jarvis_jax.data.repro_cache import load_cache, to_device_sharded, cached_batches
from jarvis_jax.eval.mpjpe_3d import mpjpe_3d
from jarvis_jax.hybridnet.model import soft_argmax_3d
from jarvis_jax.hybridnet.v2vnet import V2VNet
from jarvis_jax.sharding import data_parallel_mesh, replicate
from jarvis_jax.train.checkpoint import make_manager, save_step, restore_latest
from jarvis_jax.train.losses_3d import (
    heatmap3d_mse, graph_laplacian, build_skeleton_edges,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CachedConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    total_steps: int = 20000
    batch_size: int = 64
    laplacian_weight: float = 0.0   # OFF by default; enable via --laplacian-weight
    sigma: float = 2.0
    sharpen: float = 3.0            # soft-argmax sharpening exponent (center-bias fix)
    seed: int = 0


# ---------------------------------------------------------------------------
# Path filter: selects only Param variables in the V2VNet
# (the v2vNet IS the whole trained model here — no ViTPose sub-path —
# so every nnx.Param is a v2vNet param; the filter is kept for symmetry
# with train_3d.py and in case the optimizer is ever attached to a wrapper)
# ---------------------------------------------------------------------------

def _is_v2vnet_param(path, var):
    """Return True iff *var* is an nnx.Param.

    For the cached trainer the model IS just V2VNet, so all Params are
    v2vNet params.  The path-check is intentionally omitted here (unlike
    train_3d.py where we must filter out ViTPose), but kept as a named
    filter for clarity and future-proofing.
    """
    return isinstance(var, nnx.Param)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def make_v2v_optimizer(v2v: V2VNet, cfg: CachedConfig) -> nnx.Optimizer:
    """AdamW + warmup-cosine schedule over the V2VNet parameters.

    Args:
        v2v: V2VNet module to optimise.
        cfg: CachedConfig with lr, weight_decay, warmup_steps, total_steps.

    Returns:
        nnx.Optimizer wrapping v2v with AdamW + cosine LR schedule.
    """
    decay_steps = max(cfg.total_steps, cfg.warmup_steps + 1)
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=decay_steps,
        end_value=0.0,
    )
    tx = optax.adamw(sched, weight_decay=cfg.weight_decay)
    return nnx.Optimizer(v2v, tx, wrt=_is_v2vnet_param)


# ---------------------------------------------------------------------------
# Cached train step
# ---------------------------------------------------------------------------

def make_cached_step(
    laplacian_weight: float,
    ei: np.ndarray,
    ej: np.ndarray,
    grid_spacing: int = 1,
    roi_cube: int = 48,
    sigma: float = 2.0,
    sharpen: float = 1.0,
):
    """Return an nnx.jit-compiled train step for the cached v2vNet trainer.

    Per-step transform (matches HybridNet3D.__call__ post-reproject path):
      1. volumes (B,J,48,48,48) fp16 → f32
      2. transpose (0,2,3,4,1) → (B,48,48,48,J)   channels-last for V2VNet
      3. v2vnet(use_running_average=False) → (B,24,24,24,J)
      4. transpose (0,4,1,2,3) → (B,J,24,24,24)   joint-first
      5. softplus                                    matches model.py step 9
      6. soft_argmax_3d → points (B,J,3)
      7. loss = heatmap3d_mse + lw * graph_laplacian

    Args:
        laplacian_weight: Weight for graph-Laplacian bone-shape term (0 = off).
        ei: (E,) int32 source joint indices for skeleton edges.
        ej: (E,) int32 destination joint indices for skeleton edges.
        grid_spacing: World units per grid step (default 1).
        roi_cube:     Full cube side-length in world units (default 48).
        sigma:        Gaussian sigma in grid units for the 3D heatmap target.

    Returns:
        step(v2v, opt, batch_dict) -> scalar loss
    """
    lw = float(laplacian_weight)
    ei_jnp = jnp.asarray(ei)
    ej_jnp = jnp.asarray(ej)

    def loss_fn(v2v: V2VNet, volumes, kp3d, vis, center3D):
        # 1. Cast fp16 → f32
        vol = volumes.astype(jnp.float32)           # (B, J, 48, 48, 48)

        # 2. Transpose to channels-last for V2VNet
        vol = jnp.transpose(vol, (0, 2, 3, 4, 1))  # (B, 48, 48, 48, J)

        # 3. V2VNet
        vol = v2v(vol, use_running_average=False)   # (B, 24, 24, 24, J)

        # 4. Transpose to joint-first for soft_argmax
        vol = jnp.transpose(vol, (0, 4, 1, 2, 3))  # (B, J, 24, 24, 24)

        # 5. Softplus (matches HybridNet3D.__call__ step 9)
        vol = jax.nn.softplus(vol)                  # (B, J, 24, 24, 24)

        # pred_vol is now (B, J, G, G, G) — used for heatmap loss
        pred_vol = vol

        # 6. Soft-argmax → cube-local points; add center3D for world coords
        points_local, _ = soft_argmax_3d(pred_vol, grid_spacing=grid_spacing,
                                          roi_cube=roi_cube, sharpen=sharpen)
        points3D = points_local + center3D[:, None, :]  # (B, J, 3) world

        # 7. Losses
        loss = heatmap3d_mse(
            pred_vol, kp3d, vis,
            grid_spacing=grid_spacing,
            roi_cube=roi_cube,
            center3D=center3D,
            sigma=sigma,
        )
        if lw > 0.0 and ei_jnp.shape[0] > 0:
            loss = loss + lw * graph_laplacian(points3D, kp3d, vis, ei_jnp, ej_jnp)
        return loss

    @nnx.jit
    def step(v2v: V2VNet, opt: nnx.Optimizer, batch: dict):
        volumes  = batch["volumes"]
        kp3d     = batch["kp3d"]
        vis      = batch["vis"]
        center3D = batch["center3D"]

        loss, grads = nnx.value_and_grad(
            loss_fn,
            argnums=DiffState(0, _is_v2vnet_param),
        )(v2v, volumes, kp3d, vis, center3D)
        opt.update(v2v, grads)
        return loss

    return step


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def make_eval_step_jitted(grid_spacing: int = 1, roi_cube: int = 48, sharpen: float = 1.0):
    """Return a jit-compiled eval step for the per-batch forward pass.

    Args:
        grid_spacing: World units per grid step.
        roi_cube:     Full cube side-length in world units.
        sharpen:      soft-argmax sharpening exponent (center-bias fix).

    Returns:
        eval_step(v2v, batch) -> (points3D, vis) where points3D is world coords
    """
    @nnx.jit
    def eval_step(v2v: V2VNet, volumes, center3D):
        # 1. Cast fp16 → f32
        vol = volumes.astype(jnp.float32)               # (B, J, 48, 48, 48)

        # 2. Transpose to channels-last for V2VNet
        vol = jnp.transpose(vol, (0, 2, 3, 4, 1))      # (B, 48, 48, 48, J)

        # 3. V2VNet with eval mode (use_running_average=True)
        vol = v2v(vol, use_running_average=True)         # (B, 24, 24, 24, J)

        # 4. Transpose to joint-first for soft_argmax
        vol = jnp.transpose(vol, (0, 4, 1, 2, 3))      # (B, J, 24, 24, 24)

        # 5. Softplus (matches HybridNet3D.__call__ step 9)
        vol = jax.nn.softplus(vol)                      # (B, J, 24, 24, 24)

        # 6. Soft-argmax → cube-local points; add center3D for world coords
        points_local, _ = soft_argmax_3d(vol, grid_spacing=grid_spacing,
                                         roi_cube=roi_cube, sharpen=sharpen)
        points3D = points_local + center3D[:, None, :]  # (B, J, 3) world

        return points3D

    return eval_step


def eval_mpjpe_3d_cached(
    v2v: V2VNet,
    dev_cache: dict,
    batch_size: int,
    *,
    grid_spacing: int = 1,
    roi_cube: int = 48,
    sharpen: float = 1.0,
) -> float:
    """Mean 3D MPJPE over the cached val set (model in eval mode).

    Args:
        v2v:         V2VNet module to evaluate.
        dev_cache:   Device-resident cache dict from :func:`to_device_sharded`.
        batch_size:  Samples per eval batch.
        grid_spacing: World units per grid step.
        roi_cube:     Full cube side-length in world units.
        sharpen:      soft-argmax sharpening exponent (center-bias fix).

    Returns:
        Mean 3D MPJPE in world units, or 0.0 if no valid joints.
    """
    v2v.eval()
    eval_step = make_eval_step_jitted(grid_spacing=grid_spacing, roi_cube=roi_cube,
                                      sharpen=sharpen)
    total, count = 0.0, 0

    for batch in cached_batches(dev_cache, batch_size, shuffle=False, drop_last=False):
        kp3d = batch["kp3d"]
        vis = batch["vis"]

        # Call the jit-compiled per-batch forward + soft_argmax
        pts = eval_step(v2v, batch["volumes"], batch["center3D"])

        n = int(vis.sum())
        if n == 0:
            continue
        err = float(mpjpe_3d(pts, kp3d, vis)) * n
        total += err
        count += n

    v2v.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# run_cached_training
# ---------------------------------------------------------------------------

def save_run_config(run_dir: str, config: dict) -> str:
    """Persist a run's effective config for easy later recovery.

    Writes two files into ``run_dir``:
      - ``run_config.json``: the latest effective config (overwritten each
        launch, so it always reflects the most recent settings — e.g. the
        extended ``total_steps`` after a resume).
      - ``run_config_history.jsonl``: one JSON line appended per launch, so a
        resume never erases the original config (full provenance of the run).

    Args:
        run_dir: Directory to write the config files into (created if absent).
        config:  JSON-serialisable dict of the effective run configuration.

    Returns:
        Path to the written ``run_config.json``.
    """
    os.makedirs(run_dir, exist_ok=True)
    latest = os.path.join(run_dir, "run_config.json")
    with open(latest, "w") as f:
        json.dump(config, f, indent=2, default=str)
    with open(os.path.join(run_dir, "run_config_history.jsonl"), "a") as f:
        f.write(json.dumps(config, default=str) + "\n")
    return latest


def run_cached_training(
    cache_dir: str,
    *,
    out_dir: str,
    ckpt_dir: str = None,
    tcfg: CachedConfig = None,
    save_every: int = 500,
    log_every: int = 50,
    eval_every: int = 500,
) -> dict:
    """Train v2vNet on GPU-resident reprojected-volume cache.

    No per-step ViTPose inference or host-to-device transfer.

    Args:
        cache_dir:  Directory containing the precomputed cache files
                    (train_volumes.f16, train_labels.npz, etc.).
        out_dir:    Directory for the final Orbax checkpoint.
        ckpt_dir:   If given, use CheckpointManager for auto-resume.
        tcfg:       CachedConfig; defaults constructed if None.
        save_every: Checkpoint every N steps.
        log_every:  Print loss every N steps.
        eval_every: Evaluate 3D MPJPE every N steps.

    Returns:
        Dict with keys: first_loss, final_loss, val_mpjpe_3d, steps.
    """
    tcfg = tcfg or CachedConfig()

    n_dev = len(jax.devices())
    if tcfg.batch_size % n_dev != 0:
        raise ValueError(
            f"batch_size ({tcfg.batch_size}) must be divisible by the JAX "
            f"device count ({n_dev}) for data-parallel sharding.")

    # --- Load caches ---
    print(f"Loading train cache from {cache_dir} ...")
    train_cache = load_cache(cache_dir, "train")
    J = train_cache["volumes"].shape[1]
    print(f"  train: {train_cache['volumes'].shape[0]} samples, {J} joints")

    val_cache = None
    try:
        val_cache = load_cache(cache_dir, "val")
        print(f"  val:   {val_cache['volumes'].shape[0]} samples")
    except FileNotFoundError:
        print("  (no val cache found; skipping val MPJPE eval)")

    # --- Place on device ---
    mesh = data_parallel_mesh()
    train_dev = to_device_sharded(train_cache, mesh)
    val_dev = to_device_sharded(val_cache, mesh) if val_cache is not None else None

    # --- Skeleton edges from cache meta ---
    ei = np.zeros((0,), dtype=np.int32)
    ej = np.zeros((0,), dtype=np.int32)
    meta = train_cache.get("meta", {})
    kp_names = meta.get("keypoint_names", [])
    skeleton = meta.get("skeleton", [])
    if kp_names and skeleton:
        ei, ej = build_skeleton_edges(kp_names, skeleton)
        print(f"Skeleton edges wired: {len(ei)} "
              f"(graph-Laplacian active when laplacian_weight={tcfg.laplacian_weight} > 0)")
    else:
        # Try to read from the dataset JSON alongside the cache
        json_path = os.path.join(cache_dir, "train_dataset.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                ds_json = json.load(f)
            cats = ds_json.get("categories", [{}])
            cat = cats[0] if cats else {}
            kp_names = cat.get("keypoints", [])
            skeleton = cat.get("skeleton", [])
            if kp_names and skeleton:
                ei, ej = build_skeleton_edges(kp_names, skeleton)
                print(f"Skeleton edges (from train_dataset.json): {len(ei)}")
        if len(ei) == 0:
            print("Skeleton edges: none found — graph-Laplacian will be zero.")

    # --- Build model ---
    v2v = V2VNet(J, J, rngs=nnx.Rngs(tcfg.seed))

    # --- Optimizer ---
    opt = make_v2v_optimizer(v2v, tcfg)

    # --- CheckpointManager for auto-resume ---
    mngr = make_manager(ckpt_dir) if ckpt_dir else None
    start = 0
    if mngr is not None:
        v2v, opt, start = restore_latest(mngr, v2v, opt)
        if start:
            print(f"Resuming from checkpoint at step {start}")

    # --- Persist the effective run config for easy later recovery ---
    # run_dir is the directory that holds ckpt/ and final/ (their parent).
    run_dir = os.path.dirname((ckpt_dir or out_dir).rstrip("/")) or "."
    config_record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        **dataclasses.asdict(tcfg),
        "save_every": save_every,
        "log_every": log_every,
        "eval_every": eval_every,
        "cache_dir": cache_dir,
        "out_dir": out_dir,
        "ckpt_dir": ckpt_dir,
        "resumed_from_step": int(start),
        "n_train_samples": int(train_cache["volumes"].shape[0]),
        "n_val_samples": int(val_cache["volumes"].shape[0]) if val_cache is not None else 0,
        "n_joints": int(J),
        "n_devices": int(n_dev),
        "n_skeleton_edges": int(len(ei)),
        "graph_laplacian_active": bool(len(ei) > 0 and tcfg.laplacian_weight > 0),
        "cache_vitpose_ckpt": meta.get("vitpose_ckpt"),
        "cache_grid_size": meta.get("grid_size"),
    }
    print(f"Wrote run config -> {save_run_config(run_dir, config_record)}")

    # --- Replicate across mesh (critical: must happen AFTER restore) ---
    gdef_v, st_v = nnx.split(v2v)
    v2v = nnx.merge(gdef_v, replicate(st_v, mesh))
    gdef_o, st_o = nnx.split(opt)
    opt = nnx.merge(gdef_o, replicate(st_o, mesh))

    # --- Train step ---
    step_fn = make_cached_step(
        laplacian_weight=tcfg.laplacian_weight,
        ei=ei, ej=ej,
        grid_spacing=1, roi_cube=48, sigma=tcfg.sigma,
        sharpen=tcfg.sharpen,
    )
    print(f"soft-argmax sharpen exponent: {tcfg.sharpen} (center-bias fix; 1.0=off)")

    first_loss = None
    final_loss = 0.0

    # Infinite epoch loop over the cache
    step_idx = start
    epoch = 0
    while step_idx < tcfg.total_steps:
        for batch in cached_batches(
            train_dev, tcfg.batch_size,
            shuffle=True, seed=tcfg.seed + epoch,
        ):
            if step_idx >= tcfg.total_steps:
                break

            final_loss = float(step_fn(v2v, opt, batch))
            # Capture first_loss only at the true first step of this run
            if step_idx == start:
                first_loss = final_loss

            if (step_idx + 1) % log_every == 0:
                print(f"step {step_idx + 1}/{tcfg.total_steps}  loss {final_loss:.5f}")

            if (step_idx + 1) % eval_every == 0 and val_dev is not None:
                val_e = eval_mpjpe_3d_cached(v2v, val_dev, tcfg.batch_size,
                                             sharpen=tcfg.sharpen)
                print(f"  val 3D MPJPE {val_e:.3f}")

            if mngr is not None and (step_idx + 1) % save_every == 0:
                save_step(mngr, step_idx + 1, v2v, opt)

            step_idx += 1

        epoch += 1

    # Final checkpoint
    if mngr is not None:
        save_step(mngr, tcfg.total_steps, v2v, opt)
        mngr.wait_until_finished()

    # Final eval
    val_mpjpe = 0.0
    if val_dev is not None:
        val_mpjpe = eval_mpjpe_3d_cached(v2v, val_dev, tcfg.batch_size,
                                         sharpen=tcfg.sharpen)
        print(f"Final val 3D MPJPE {val_mpjpe:.3f}")

    # Save final checkpoint with StandardCheckpointer
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(out_dir, nnx.split(v2v)[1], force=True)
    ckptr.wait_until_finished()

    return {
        "first_loss": first_loss if first_loss is not None else final_loss,
        "final_loss": final_loss,
        "val_mpjpe_3d": val_mpjpe,
        "steps": tcfg.total_steps,
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main_from_cfg(cfg):
    """Map a composed Hydra config into CachedConfig + run cached training."""
    cache_dir = cfg.paths.get("cache_dir", None)
    if cache_dir is None:
        raise SystemExit(
            "paths.cache_dir was removed 2026-09-05 (LEGACY cached3d/HybridNet "
            "cache deleted). Pass paths.cache_dir=<dir> explicitly and rebuild "
            "it with scripts/precompute_repro_cache.py.")
    tcfg = build_dataclass(CachedConfig, cfg.train)
    run_dir = run_dir_for(cfg)
    return run_cached_training(
        cache_dir,
        out_dir=os.path.join(run_dir, "final"),
        ckpt_dir=os.path.join(run_dir, "ckpt"),
        tcfg=tcfg,
        save_every=cfg.train.save_every,
        log_every=cfg.train.log_every,
        eval_every=cfg.train.eval_every,
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
