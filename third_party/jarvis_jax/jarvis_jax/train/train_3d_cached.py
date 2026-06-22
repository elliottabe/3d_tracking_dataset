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

CLI:
    python -m jarvis_jax.train.train_3d_cached \\
        --cache-dir /path/to/repro_cache \\
        --out /path/to/output \\
        --steps 20000 --batch 64 --lr 3e-4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx.transforms.autodiff import DiffState

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
        points_local, _ = soft_argmax_3d(pred_vol, grid_spacing=grid_spacing, roi_cube=roi_cube)
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

def eval_mpjpe_3d_cached(
    v2v: V2VNet,
    dev_cache: dict,
    batch_size: int,
    *,
    grid_spacing: int = 1,
    roi_cube: int = 48,
) -> float:
    """Mean 3D MPJPE over the cached val set (model in eval mode).

    Args:
        v2v:         V2VNet module to evaluate.
        dev_cache:   Device-resident cache dict from :func:`to_device_sharded`.
        batch_size:  Samples per eval batch.
        grid_spacing: World units per grid step.
        roi_cube:     Full cube side-length in world units.

    Returns:
        Mean 3D MPJPE in world units, or 0.0 if no valid joints.
    """
    v2v.eval()
    total, count = 0.0, 0

    for batch in cached_batches(dev_cache, batch_size, shuffle=False, drop_last=False):
        vol = batch["volumes"].astype(jnp.float32)
        kp3d = batch["kp3d"]
        vis = batch["vis"]
        center3D = batch["center3D"]

        # Apply same post-reproject transform as the train step (eval=no dropout)
        vol = jnp.transpose(vol, (0, 2, 3, 4, 1))              # (B, 48, 48, 48, J)
        vol = v2v(vol, use_running_average=True)                 # (B, 24, 24, 24, J)
        vol = jnp.transpose(vol, (0, 4, 1, 2, 3))              # (B, J, 24, 24, 24)
        vol = jax.nn.softplus(vol)

        points_local, _ = soft_argmax_3d(vol, grid_spacing=grid_spacing, roi_cube=roi_cube)
        pts = points_local + center3D[:, None, :]               # world coords

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

def run_cached_training(
    cache_dir: str,
    *,
    out_dir: str,
    ckpt_dir: str = None,
    tcfg: CachedConfig = None,
    save_every: int = 500,
    vitpose_ckpt_for_meta_check: str = None,
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
        vitpose_ckpt_for_meta_check: Unused; accepted for API symmetry.
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
    )

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
            if first_loss is None:
                first_loss = final_loss

            if (step_idx + 1) % log_every == 0:
                print(f"step {step_idx + 1}/{tcfg.total_steps}  loss {final_loss:.5f}")

            if (step_idx + 1) % eval_every == 0 and val_dev is not None:
                val_e = eval_mpjpe_3d_cached(v2v, val_dev, tcfg.batch_size)
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
        val_mpjpe = eval_mpjpe_3d_cached(v2v, val_dev, tcfg.batch_size)
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

def main():
    ap = argparse.ArgumentParser(
        description="Train v2vNet on GPU-resident reprojected-volume cache (no ViTPose).")
    ap.add_argument("--cache-dir", required=True,
                    help="Directory containing precomputed cache files")
    ap.add_argument("--out", required=True,
                    help="Output directory for the final checkpoint")
    ap.add_argument("--ckpt-dir", default=None,
                    help="CheckpointManager directory for auto-resume")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--laplacian-weight", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    tcfg = CachedConfig(
        total_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        laplacian_weight=args.laplacian_weight,
    )
    run_cached_training(
        args.cache_dir,
        out_dir=args.out,
        ckpt_dir=args.ckpt_dir,
        tcfg=tcfg,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
