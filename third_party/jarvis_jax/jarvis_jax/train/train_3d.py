"""3D HybridNet training: frozen ViTPose front-end + trainable v2vNet.

Only the v2vNet sub-module is optimised (ViTPose weights are frozen):
  1. stop_gradient is applied to the per-camera heatmaps produced by ViTPose
     before they flow into the reprojection / v2vNet path — so no gradient
     ever reaches the 86M-param frozen backbone.
  2. The optimizer is built with a path-based wrt-filter that selects ONLY
     nnx.Param variables inside the ``model.v2vnet`` subtree, so AdamW never
     updates any ViTPose parameter.

CLI:
    python -m jarvis_jax.train.train_3d \\
        --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3 \\
        --out /tmp/hybridnet3d_ckpt \\
        --vitpose-ckpt /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_8gpu_20260620/final \\
        --steps 2000 --batch 4 --lr 1e-4
"""
import dataclasses
import os
import queue
import threading

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx.transforms.autodiff import DiffState

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.convert.build_checkpoint import load_vitpose
from jarvis_jax.data.v3_3d import V3FramesetDataset, frameset_batches
from jarvis_jax.eval.mpjpe_3d import mpjpe_3d
from jarvis_jax.hybridnet.model import HybridNet3D, soft_argmax_3d
from jarvis_jax.hybridnet.reproject import reproject_heatmaps
from jarvis_jax.hybridnet.v2vnet import V2VNet
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass, run_dir_for
from jarvis_jax.sharding import data_parallel_mesh, shard_batch, replicate
from jarvis_jax.train.checkpoint import make_manager, save_step, restore_latest
from jarvis_jax.train.losses_3d import (
    heatmap3d_mse, graph_laplacian, build_skeleton_edges,
)

register_resolvers()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HybridNetConfig:
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 200
    total_steps: int = 2000
    batch_size: int = 4
    laplacian_weight: float = 0.0  # OFF by default; enable via --laplacian-weight
    sigma: float = 2.0
    seed: int = 0


# ---------------------------------------------------------------------------
# Path filter: selects only Param variables in the v2vnet subtree
# ---------------------------------------------------------------------------

def _is_v2vnet_param(path, var):
    """Return True iff *var* is an nnx.Param in the v2vnet subtree.

    Inside nnx.value_and_grad / nnx.Optimizer the path is a tuple of string
    attribute names, e.g. ``('v2vnet', 'encoder_decoder', 'decoder_res1', ...)``.
    We select only variables whose root is 'v2vnet'.
    """
    return isinstance(var, nnx.Param) and len(path) > 0 and path[0] == "v2vnet"


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def make_v2v_optimizer(model: HybridNet3D, cfg: HybridNetConfig) -> nnx.Optimizer:
    """AdamW + warmup-cosine schedule, updating ONLY the v2vNet params.

    The wrt-filter ``_is_v2vnet_param`` restricts the optimizer state to the
    v2vnet subtree, so ViTPose parameters are never touched by AdamW regardless
    of the computed gradients.
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
    return nnx.Optimizer(model, tx, wrt=_is_v2vnet_param)


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

def make_train_step_3d(
    laplacian_weight: float,
    ei: np.ndarray,
    ej: np.ndarray,
    grid_spacing: int = 1,
    roi_cube: int = 48,
    sigma: float = 2.0,
):
    """Return an nnx.jit-compiled train step for 3D HybridNet.

    The step:
      1. Runs HybridNet3D.__call__ but applies stop_gradient to the ViTPose
         heatmaps via a custom forward path.
      2. Computes loss = heatmap3d_mse + laplacian_weight * graph_laplacian.
      3. Differentiates wrt ONLY the v2vnet Params (``_is_v2vnet_param`` filter).
      4. Updates the optimizer (which also only tracks v2vnet Params).

    Args:
        laplacian_weight: Weight for the graph-Laplacian bone-shape term.
        ei: (E,) int32 source joint indices for skeleton edges.
        ej: (E,) int32 destination joint indices for skeleton edges.
        grid_spacing: World units per grid step (default 1).
        roi_cube:     Full cube side-length in world units (default 48).
        sigma:        Gaussian sigma in grid units for the 3D heatmap target.

    Returns:
        step(model, optimizer, batch_dict) -> scalar loss
    """
    lw = float(laplacian_weight)
    ei_jnp = jnp.asarray(ei)
    ej_jnp = jnp.asarray(ej)

    def loss_fn(
        model: HybridNet3D,
        crops4_u8,
        center3D,
        centerHM,
        cameraMatrices,
        kp3d,
        vis,
    ):
        # Run ViTPose (frozen, eval mode) and immediately stop gradient.
        # This prevents any gradient from flowing back through the 86M-param
        # ViTPose front-end while still using its heatmaps as input to v2vNet.
        hm = model.predict_heatmaps(crops4_u8)         # (B, nc, 224, 224, J)
        hm = jax.lax.stop_gradient(hm)                 # << freeze boundary

        # Pad 224 → 226 and transpose to (B, nc, J, 226, 226) for reprojection
        hm = jnp.pad(hm, [(0, 0), (0, 0), (1, 1), (1, 1), (0, 0)])
        hm = jnp.transpose(hm, (0, 1, 4, 2, 3))

        # Reproject → V2VNet → soft-argmax (via the v2vnet sub-path)
        # NOTE: this is a parallel forward path to HybridNet3D.__call__ in model.py,
        # inserting stop_gradient after ViTPose. Keep grid/pad/transpose constants
        # (grid_size=48, grid_spacing=1, heatmap_size=226, pad=(1,1,1,1)) in sync.
        vol3d = reproject_heatmaps(
            hm, center3D, centerHM, cameraMatrices,
            grid_size=48, grid_spacing=1, heatmap_size=226,
        )                                               # (B, J, 48, 48, 48)
        vol3d = vol3d / 255.0
        vol3d = jnp.transpose(vol3d, (0, 2, 3, 4, 1)) # (B, 48, 48, 48, J)
        vol3d = model.v2vnet(vol3d, use_running_average=False)
        vol3d = jnp.transpose(vol3d, (0, 4, 1, 2, 3)) # (B, J, 24, 24, 24)
        vol3d = jax.nn.softplus(vol3d)

        points_local, _ = soft_argmax_3d(vol3d, grid_spacing=grid_spacing, roi_cube=roi_cube)
        points3D = points_local + center3D[:, None, :]  # (B, J, 3) world

        loss = heatmap3d_mse(
            vol3d, kp3d, vis,
            grid_spacing=grid_spacing,
            roi_cube=roi_cube,
            center3D=center3D,
            sigma=sigma,
        )
        if lw > 0.0 and ei_jnp.shape[0] > 0:
            loss = loss + lw * graph_laplacian(points3D, kp3d, vis, ei_jnp, ej_jnp)
        return loss

    @nnx.jit
    def step(model, optimizer, batch):
        crops4_u8 = batch["crops4"]
        center3D = batch["center3D"]
        centerHM = batch["centerHM"]
        cameraMatrices = batch["cameraMatrices"]
        kp3d = batch["kp3d"]
        vis = batch["vis"]

        loss, grads = nnx.value_and_grad(
            loss_fn,
            argnums=DiffState(0, _is_v2vnet_param),
        )(model, crops4_u8, center3D, centerHM, cameraMatrices, kp3d, vis)
        optimizer.update(model, grads)
        return loss

    return step


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def eval_mpjpe_3d(
    model: HybridNet3D,
    ds: V3FramesetDataset,
    batch_size: int,
) -> float:
    """Average 3D MPJPE over the dataset (model in eval mode).

    Args:
        model:      HybridNet3D (uses use_running_average=True internally).
        ds:         V3FramesetDataset to evaluate on.
        batch_size: Number of framesets per eval batch.

    Returns:
        Mean 3D MPJPE in world units (mm by default), or 0.0 if no valid joints.
    """
    model.eval()
    total, count = 0.0, 0
    for batch in frameset_batches(ds, batch_size, shuffle=False, drop_last=False):
        crops4_u8 = jnp.asarray(batch["crops4"])
        center3D = jnp.asarray(batch["center3D"])
        centerHM = jnp.asarray(batch["centerHM"])
        cameraMatrices = jnp.asarray(batch["cameraMatrices"])
        kp3d = jnp.asarray(batch["kp3d"])
        vis = jnp.asarray(batch["vis"])

        _, pts, _ = model(
            crops4_u8, center3D, centerHM, cameraMatrices,
            use_running_average=True,
        )
        n = int(vis.sum())
        if n == 0:
            continue
        err = float(mpjpe_3d(pts, kp3d, vis)) * n
        total += err
        count += n
    model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Dict prefetcher (dict batches instead of tuple batches)
# ---------------------------------------------------------------------------

def _dict_prefetch(batch_iter, mesh, depth=2):
    """Yield device-resident sharded dict batches from a numpy dict iterator."""
    q = queue.Queue(maxsize=depth)

    def worker():
        try:
            for batch in batch_iter:
                dev_batch = {
                    k: shard_batch(jnp.asarray(v), mesh)
                    for k, v in batch.items()
                }
                q.put(("ok", dev_batch))
        except Exception as e:
            q.put(("err", e))
            return
        q.put(("end", None))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        tag, payload = q.get()
        if tag == "ok":
            yield payload
        elif tag == "end":
            return
        else:
            raise payload


# ---------------------------------------------------------------------------
# Infinite epoch stream
# ---------------------------------------------------------------------------

def _epochs_3d(ds, batch_size, base_seed):
    """Infinite stream of dict batches, reshuffled each epoch."""
    epoch = 0
    while True:
        yield from frameset_batches(
            ds, batch_size, shuffle=True, seed=base_seed + epoch)
        epoch += 1


# ---------------------------------------------------------------------------
# run_training_3d
# ---------------------------------------------------------------------------

def run_training_3d(
    root: str,
    *,
    out_dir: str,
    vitpose_ckpt: str,
    ckpt_dir: str = None,
    tcfg: HybridNetConfig = None,
    save_every: int = 500,
    val_recording: str = "2026_05_27_11_56_05",
    log_every: int = 50,
    eval_every: int = 500,
) -> dict:
    """Build HybridNet3D, train v2vNet (ViTPose frozen), checkpoint + eval.

    Mirrors ``train_keypoints.run_training`` in structure.

    Args:
        root:           Root of the V3 dataset.
        out_dir:        Directory for the final Orbax checkpoint.
        vitpose_ckpt:   Directory of the pre-trained ViTPose checkpoint.
        ckpt_dir:       If given, use CheckpointManager for auto-resume.
        tcfg:           HybridNetConfig; defaults constructed if None.
        save_every:     Checkpoint every N steps.
        val_recording:  Recording name for the validation split.
        log_every:      Print loss every N steps.
        eval_every:     Evaluate 3D MPJPE every N steps.

    Returns:
        dict with keys: first_loss, final_loss, val_mpjpe_3d, steps.
    """
    tcfg = tcfg or HybridNetConfig()
    cfg = ViTPoseConfig()

    n_dev = len(jax.devices())
    if tcfg.batch_size % n_dev != 0:
        raise ValueError(
            f"batch_size ({tcfg.batch_size}) must be divisible by the JAX device "
            f"count ({n_dev}) for data-parallel sharding.")

    # --- Build model ---
    vitpose = load_vitpose(vitpose_ckpt, cfg)
    v2vnet = V2VNet(cfg.num_keypoints, cfg.num_keypoints, rngs=nnx.Rngs(tcfg.seed))
    model = HybridNet3D(vitpose, v2vnet, cfg)

    # --- Optimizer (v2vnet params only) ---
    opt = make_v2v_optimizer(model, tcfg)

    # --- CheckpointManager for auto-resume ---
    mngr = make_manager(ckpt_dir) if ckpt_dir else None
    start = 0
    if mngr is not None:
        model, opt, start = restore_latest(mngr, model, opt)
        if start:
            print(f"resuming from checkpoint at step {start}")

    # --- Replicate across mesh (critical for resume; harmless on fresh start) ---
    mesh = data_parallel_mesh()
    gdef_m, st_m = nnx.split(model)
    model = nnx.merge(gdef_m, replicate(st_m, mesh))
    gdef_o, st_o = nnx.split(opt)
    opt = nnx.merge(gdef_o, replicate(st_o, mesh))

    # --- Datasets ---
    train_ds = V3FramesetDataset(root, "train")
    val_ds = V3FramesetDataset(root, "val", recordings=[val_recording])

    # --- Skeleton edges from V3 dataset COCO annotations ---
    ei, ej = build_skeleton_edges(train_ds.keypoint_names, train_ds.skeleton)
    print(f"skeleton edges wired: {len(ei)} (graph-Laplacian prior active when "
          f"laplacian_weight={tcfg.laplacian_weight} > 0)")

    # --- Train step ---
    step_fn = make_train_step_3d(
        laplacian_weight=tcfg.laplacian_weight,
        ei=ei, ej=ej,
        grid_spacing=1, roi_cube=48, sigma=tcfg.sigma,
    )

    # --- Prefetched batch stream ---
    host_stream = _epochs_3d(train_ds, tcfg.batch_size, tcfg.seed)
    dev_stream = _dict_prefetch(host_stream, mesh, depth=2)

    first_loss = None
    final_loss = 0.0

    for i in range(start, tcfg.total_steps):
        batch = next(dev_stream)
        final_loss = float(step_fn(model, opt, batch))
        if first_loss is None:
            first_loss = final_loss
        if (i + 1) % log_every == 0:
            print(f"step {i+1}/{tcfg.total_steps}  loss {final_loss:.5f}")
        if (i + 1) % eval_every == 0 and len(val_ds) > 0:
            val_e = eval_mpjpe_3d(model, val_ds, tcfg.batch_size)
            print(f"  val 3D MPJPE {val_e:.3f}")
        if mngr is not None and (i + 1) % save_every == 0:
            save_step(mngr, i + 1, model, opt)

    if mngr is not None:
        save_step(mngr, tcfg.total_steps, model, opt)
        mngr.wait_until_finished()

    val_mpjpe = eval_mpjpe_3d(model, val_ds, tcfg.batch_size) if len(val_ds) > 0 else 0.0
    print(f"final val 3D MPJPE {val_mpjpe:.3f}")

    ckptr = ocp.StandardCheckpointer()
    ckptr.save(out_dir, nnx.split(model)[1], force=True)
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
    """Map a composed Hydra config into HybridNetConfig + run inline-3D training."""
    tcfg = build_dataclass(HybridNetConfig, cfg.train)
    run_dir = run_dir_for(cfg)
    return run_training_3d(
        cfg.paths.data_root,
        out_dir=os.path.join(run_dir, "final"),
        vitpose_ckpt=cfg.paths.vitpose_ckpt,
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
