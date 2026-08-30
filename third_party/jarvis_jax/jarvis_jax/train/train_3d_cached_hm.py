"""Coarse-to-fine / rotation-augmented cached 3D trainer (Task 15).

WHY THIS FILE EXISTS (not train_3d_cached.py). train_3d_cached.py trains
V2VNet on a cache of ALREADY-REPROJECTED volumes (fixed axis-aligned grid,
fixed grid_size=48/spacing=1, baked in at cache-build time -- see
scripts/precompute_repro_cache.py). Two of Task 15's arm features are
IMPOSSIBLE to build from that cache no matter how it is resampled
afterward, because they are properties of the REPROJECTION step itself, not
of its output:

  1. ROTATION AUGMENTATION (``train.rot_augment``): rotating the grid BASIS
     before the DLT projection (jarvis_jax/hybridnet/reproject.py's
     ``rotation`` kwarg on ``reproject_heatmaps``) -- rot_augment.py's own
     module docstring is explicit that this is a grid-basis rotation, not a
     post-hoc relabeling.
  2. COARSE-TO-FINE REFINEMENT (``model.refine.enabled``,
     jarvis_jax/hybridnet/refine.py): re-reprojects a small per-joint window
     at spacing=0.25 (vs. stage-1's spacing=1) directly from the RAW 2-D
     heatmaps -- "matched to the 2-D detail that ALREADY EXISTS" (refine.py
     docstring). At spacing=1 each stage-1 voxel already truncates its
     heatmap lookup to whole-pixel granularity, so that sub-voxel detail is
     NOT present in a spacing=1 cached volume; interpolating it more finely
     would manufacture detail, not recover it. The SAME is true of A5_hires
     (grid_size=96/spacing=0.5 at the SAME physical FOV as the default
     48/spacing=1): its finer sampling is only real if it comes from a fresh
     reprojection of the raw heatmaps.

So this trainer reads a cache of PER-CAMERA 2-D HEATMAPS
(scripts/precompute_heatmap_cache.py's ``<split>_heatmaps.f16`` +
``<split>_hmlabels.npz``/``_hmmeta.json``) instead of pre-baked volumes, and
redoes the (cheap: a DLT matmul + bilinear coordinate upsample + integer
gather -- no front-end forward pass) reprojection FRESH every step with
whatever grid_size/grid_spacing/rotation THIS arm needs. This is what makes
every one of Task 15's 8 arms buildable from ONE cache.

Rotation-augmentation math (validated empirically against real cached
heatmaps -- see task-15-report.md; NOT the untested rot_augment.augment_sample
convention, which turned out to need the opposite rotation direction for
this use):
  - reproject.py's ``_build_half_grid`` maps grid index ``p`` to world point
    ``center3D + R @ base_offset(p)`` when ``rotation=R``.
  - So a stage-1 prediction living in that same base-offset space converts
    to WORLD coordinates via ``center3D + R @ points_local`` (confirmed:
    using ``R`` here reproduces the SAME physical peak, to within one voxel,
    as the unrotated (R=I) reprojection of the identical real heatmaps --
    using ``R.T`` does not).
  - The Gaussian heatmap-MSE target (``losses_3d.heatmap3d_mse``) instead
    needs the GT expressed in the ROTATED grid's OWN local-offset space:
    ``center3D + R.T @ (gt_kp - center3D)`` (the inverse direction).
  - A single rotation is sampled PER TRAINING STEP (shared across the whole
    batch, not per-sample) -- still real per-step augmentation variance
    across ~20k steps, and lets the whole batch share one ``reproject_heatmaps``
    call with a (3,3) (not (B,3,3)) rotation.

Per-step transform (stage 1)
-----------------------------------------------------------------------------
  heatmaps (B,num_cam,J,224,224) fp16 cached -> f32
  -> pad 224->226 each spatial dim (matches HybridNet3D.reproject_volume)
  -> transpose to (B,num_cam,J,226,226)
  -> reproject_heatmaps(grid_size, grid_spacing, rotation) -> (B,J,G,G,G) /255
  -> transpose channels-last -> V2VNet(use_running_average=False) -> (B,G/2,...,J)
  -> transpose joint-first -> softplus -> soft_argmax_3d(sharpen)
  -> points_local (B,J,3); world = center3D + R@points_local (or +points_local if R=None)

Stage 2 (optional, model.refine.enabled): reuses hybridnet/refine.py's
RefineNet + refine_keypoints UNCHANGED (never modified -- see repo
constraints) on the RAW (unpadded, heatmap_size=224) cached heatmaps,
re-centred on the stage-1 WORLD-frame estimate. Deliberately axis-aligned
(no rotation) even under rot_augment: its window is a tiny +/-3-world-unit
local patch (refine.py's own docstring) with weights shared across joints
("local peak sharpening... stage 1 already did the localization"), so the
world-orientation bias rotation-augmentation targets barely applies there --
and refine.py cannot be modified to add one anyway.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.data.rot_augment import gravity_axis, sample_rotation
from jarvis_jax.eval.mpjpe_3d import mpjpe_3d
from jarvis_jax.hybridnet.model import soft_argmax_3d
from jarvis_jax.hybridnet.reproject import _reproject_single
from jarvis_jax.hybridnet.refine import RefineNet, refine_keypoints
from jarvis_jax.hybridnet.v2vnet import V2VNet
from jarvis_jax.train.checkpoint import make_manager, save_step, restore_latest


def reproject_heatmaps(heatmaps, center3D, centerHM, camera_matrices, *,
                        grid_size, grid_spacing, heatmap_size, rotation=None):
    """Batched reprojection -- vmaps ``reproject._reproject_single`` directly
    instead of going through ``reproject.reproject_heatmaps``.

    WHY. The public wrapper does an eager ``bool(jnp.allclose(...))``
    orthogonality/determinant check on ``rotation`` -- correct for an
    eager/Python-level R, but it raises ``TracerBoolConversionError`` the
    moment ``rotation`` is itself a jit-traced array (which it must be here:
    a NEW random rotation is sampled every training step and passed into an
    ``nnx.jit``-compiled step -- making it a Python-level jit *static* arg
    would force a recompile every step). ``rotation`` here is ALWAYS built by
    ``rot_augment.sample_rotation`` (Rodrigues' formula composed from a
    normalized axis), which is a proper rotation (orthogonal, det=+1) BY
    CONSTRUCTION -- the check is validating an invariant that already holds,
    not catching a real failure mode for this caller, so skipping it (by
    calling the same private ``_reproject_single`` the public wrapper itself
    vmaps, unmodified) costs nothing. ``reproject.py`` is not modified.
    """
    rotation = None if rotation is None else jnp.asarray(rotation, jnp.float32)
    rot_in_axis = None if (rotation is None or rotation.ndim == 2) else 0
    _single = lambda hm, c3, cHM, cM, rot: _reproject_single(
        hm, c3, cHM, cM, grid_size=grid_size, grid_spacing=grid_spacing,
        heatmap_size=heatmap_size, rotation=rot)
    return jax.vmap(_single, in_axes=(0, 0, 0, 0, rot_in_axis))(
        heatmaps, center3D, centerHM, camera_matrices, rotation)

_NUM_JOINTS = 50
_HM = 224          # raw (unpadded) cached heatmap size
_HM_PAD = 226      # stage-1 padded heatmap size (matches reproject_volume)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HMCachedConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    total_steps: int = 20000
    batch_size: int = 32
    sigma: float = 2.0
    sharpen: float = 3.0
    seed: int = 0
    laplacian_weight: float = 0.0   # kept for CLI parity; unused (repo default OFF)

    # Stage-1 grid (A5_hires overrides grid_size/grid_spacing; SAME physical
    # FOV as the default 48/1.0 -- see module docstring).
    grid_size: int = 48
    grid_spacing: float = 1.0

    # Rotation augmentation (A2/A4/A5/A6/A7/A8).
    rot_augment: bool = False
    tilt_deg: float = 30.0           # rot_augment.sample_rotation's own default

    # Stage-2 coarse-to-fine refinement (A3/A4/A6/A7/A8).
    refine_enabled: bool = False
    refine_cube: int = 24
    refine_spacing: float = 0.25
    refine_channels: int = 32
    refine_sharpen: float = 3.0
    # heatmap3d_mse lives on an O(0.1-1) density-MSE scale; a raw
    # squared-world-unit refine error is O(10-100) at this dataset's
    # coordinate scale, so weighting it 1:1 would let refine swamp the
    # stage-1 gradient entirely (empirically: unweighted loss ~145 vs
    # ~0.4 for stage-1 alone). 0.01 brings the two terms to a comparable
    # magnitude so stage 1 still does the localization refine.py's own
    # docstring assigns it ("stage 1 already did the localization").
    refine_loss_weight: float = 0.01

    # Data-selection knobs.
    exclude_second_fly: bool = False   # A6: drop the 264 two-fly framesets' 2nd fly
    female_weight: float = 1.0         # A7: oversample female framesets (see v5_3d.py)

    # Gradient clipping (optax.clip_by_global_norm, composed BEFORE AdamW).
    # A7_c2f_aug_femwt (refine_enabled=True) diverged to NaN at step
    # ~850/1500 with unclipped AdamW (loss 0.02313 at step 800 -> nan at
    # 850, never recovers -- a single bad step permanently poisons AdamW's
    # moment estimates); A3_c2f and A6_c2f_aug_norecover (both also
    # refine_enabled=True) later did the same at steps ~3300 and ~3750 of a
    # separate 20000-step run. Not a rotation-convention/magnitude bug
    # (rot_augment-only arms A1/A2/A5 never diverged). The instability
    # tracks `refine_enabled`, not rotation augmentation.
    #
    # Threshold chosen from a 1000-step instrumented reproduction of
    # A7_c2f_aug_femwt's own config on the real cache (this did NOT
    # reproduce the NaN itself -- GPU float non-associativity makes the
    # exact divergence step non-reproducible even from the same seed/data
    # order, consistent with it being a rare, stochastic bad-batch event
    # rather than a steady drift -- but it DOES show refine's healthy
    # operating range, which is what a clip threshold must sit above):
    # global grad-norm p50=2.0, p90=18.4, p95=41.0, p99=100.0,
    # p99.5=143.7, max=193.5 over 1000 steps, all finite/healthy. A
    # matched non-refine (refine_enabled=False, same seed/rot_augment)
    # reproduction over the same 1000 steps was far calmer (p50=0.03,
    # p99=0.3, max=1.9 by step 400) -- refine's gradients have a
    # substantially heavier tail, consistent with the divergence tracking
    # `refine_enabled`. 150.0 sits above every percentile of the observed
    # HEALTHY refine distribution (only the single largest of 1000 steps,
    # 193.5, would have clipped) so it does not touch normal training, while
    # still being orders of magnitude below what an actual unbounded
    # explosion would produce. CAVEAT: if the true NaN precursor is a
    # forward-pass numerical hazard (e.g. refine_keypoints's soft_argmax_3d
    # dividing by a near-zero, sharpen=3-cubed volume mass -- refine.py
    # skips the softplus stage-1 applies before its own relu, so it is the
    # only path that can produce a volume of near-all-zero voxels) rather
    # than purely a large-but-finite gradient, clipping cannot rescue it: a
    # NaN loss produces a NaN grad global-norm, and `clip_by_global_norm`
    # passes NaN through unchanged (its `<` comparison against a NaN norm is
    # always False, selecting the "already NaN" clip branch). See
    # `nan_guard` below for the actual safety net in that case; a real fix
    # would need an epsilon/guard inside refine.py or model.py, which this
    # trainer may not modify. None (or <=0) DISABLES clipping and must be
    # byte-identical to the pre-clipping optimizer (same
    # `optax.adamw(...)` object, not `optax.chain(...)` wrapping a no-op --
    # a chain changes the optimizer-state pytree shape even when the
    # clip itself never engages, which would break Orbax restore of
    # checkpoints saved by the pre-clipping code). See make_optimizer.
    grad_clip_norm: float | None = 150.0

    # NaN guard -- the PRIMARY safety net, not a secondary one: gradient
    # clipping only helps a large-but-FINITE gradient; if the true NaN
    # precursor is a forward-pass numerical hazard (see grad_clip_norm's
    # docstring above) clipping cannot rescue it at all, and this is the
    # only thing that stops the waste. A7 ran 650 more steps to a NaN
    # result before anyone noticed; A3_c2f and A6_c2f_aug_norecover each
    # burned thousands more (20000-step budget) the same way. When True, a
    # non-finite step loss is printed loudly (with the step index) and
    # training stops immediately instead of continuing to the configured
    # total_steps.
    nan_guard: bool = True


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_hm_cache(cache_dir: str, split: str) -> dict:
    meta_path = os.path.join(cache_dir, f"{split}_hmmeta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    n = int(meta["n"])
    num_cam = int(meta["num_cameras"])
    hm = np.memmap(os.path.join(cache_dir, f"{split}_heatmaps.f16"), dtype=np.float16,
                    mode="r", shape=(n, num_cam, _NUM_JOINTS, _HM, _HM))
    lbl = np.load(os.path.join(cache_dir, f"{split}_hmlabels.npz"))
    return {
        "heatmaps": hm,
        "kp3d": lbl["kp3d"], "vis": lbl["vis"], "center3D": lbl["center3D"],
        "centerHM": lbl["centerHM"], "cameraMatrices": lbl["cameraMatrices"],
        "fly_id": lbl["fly_id"], "calib_group": lbl["calib_group"],
        "is_female": lbl["is_female"], "meta": meta, "n": n, "num_cam": num_cam,
    }


def _select_train_indices(cache: dict, *, exclude_second_fly: bool,
                          female_weight: float) -> np.ndarray:
    """Index pool for one epoch: optional fly_id!=0 drop (A6), then optional
    female oversampling by repetition (A7) -- mirrors v5_3d.py's
    frameset_batches female_weight convention exactly."""
    idx = np.arange(cache["n"])
    if exclude_second_fly:
        idx = idx[cache["fly_id"][idx] == 0]
    reps = np.where(cache["is_female"][idx], max(int(round(female_weight)), 1), 1)
    return np.repeat(idx, reps)


def _gather_batch(cache: dict, idx: np.ndarray) -> dict:
    # Stay fp16 host-side (halves the memcpy/host->device transfer per step
    # vs. upcasting here); the jitted forward casts to f32 on-device, which
    # is fast and where it belongs.
    hm = np.ascontiguousarray(cache["heatmaps"][idx])   # (B,cam,J,224,224) fp16
    return {
        "heatmaps": hm,
        "kp3d": cache["kp3d"][idx].astype(np.float32),
        "vis": cache["vis"][idx],
        "center3D": cache["center3D"][idx].astype(np.float32),
        "centerHM": cache["centerHM"][idx].astype(np.float32),
        "cameraMatrices": cache["cameraMatrices"][idx].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Model container (V2VNet + optional RefineNet, trained jointly)
# ---------------------------------------------------------------------------

class C2FModel(nnx.Module):
    def __init__(self, num_joints: int, *, refine_enabled: bool, refine_channels: int,
                 rngs: nnx.Rngs):
        self.v2v = V2VNet(num_joints, num_joints, rngs=rngs)
        self.refine = RefineNet(channels=refine_channels, rngs=rngs) if refine_enabled else None


def _build_tx(cfg: HMCachedConfig):
    """The raw optax transform `make_optimizer` wraps into an `nnx.Optimizer`
    -- split out so tests can exercise the clip-vs-plain-AdamW behavior
    directly on synthetic gradient pytrees, without building a model."""
    decay_steps = max(cfg.total_steps, cfg.warmup_steps + 1)
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=cfg.lr, warmup_steps=cfg.warmup_steps,
        decay_steps=decay_steps, end_value=0.0)
    adamw = optax.adamw(sched, weight_decay=cfg.weight_decay)
    if cfg.grad_clip_norm is None or cfg.grad_clip_norm <= 0:
        # DISABLED: literally the same `adamw` transform used before gradient
        # clipping existed -- NOT `optax.chain(identity, adamw)`, which would
        # change the optimizer-state pytree (an extra EmptyState leaf) even
        # though the clip never engages, silently breaking Orbax restore of
        # any checkpoint an already-running/older arm saved with this path.
        return adamw
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip_norm), adamw)


def make_optimizer(model: C2FModel, cfg: HMCachedConfig) -> nnx.Optimizer:
    return nnx.Optimizer(model, _build_tx(cfg), wrt=nnx.Param)


# ---------------------------------------------------------------------------
# Stage-1 forward (shared by train/eval; ``training`` selects BN mode)
# ---------------------------------------------------------------------------

def _stage1_forward(v2v: V2VNet, heatmaps, center3D, centerHM, camera_matrices, *,
                     grid_size: int, grid_spacing: float, sharpen: float,
                     rotation, training: bool):
    heatmaps = heatmaps.astype(jnp.float32)   # on-device upcast (host stayed fp16)
    hm = jnp.pad(heatmaps, [(0, 0), (0, 0), (0, 0), (1, 1), (1, 1)])   # 224->226
    hm = jnp.transpose(hm, (0, 1, 2, 3, 4))  # already (B,cam,J,226,226)
    vol = reproject_heatmaps(hm, center3D, centerHM, camera_matrices,
                              grid_size=grid_size, grid_spacing=grid_spacing,
                              heatmap_size=_HM_PAD, rotation=rotation)
    vol = vol / 255.0                                          # (B,J,G,G,G)
    vol = jnp.transpose(vol, (0, 2, 3, 4, 1))                   # channels-last
    vol = v2v(vol, use_running_average=not training)
    vol = jnp.transpose(vol, (0, 4, 1, 2, 3))                   # joint-first
    vol = jax.nn.softplus(vol)
    roi = grid_size * grid_spacing
    points_local, _ = soft_argmax_3d(vol, grid_spacing=grid_spacing, roi_cube=roi,
                                      sharpen=sharpen)
    return points_local, vol


def _rotate_points(points_local, R, center3D):
    if R is None:
        return points_local + center3D[:, None, :]
    # R shared across the batch: world = center3D + R @ points_local (validated
    # direction -- see module docstring).
    world = jnp.einsum("ij,bkj->bki", R, points_local)
    return world + center3D[:, None, :]


def _rotate_gt_for_grid_target(kp3d, R, center3D):
    """GT expressed in the ROTATED grid's own local-offset space -- the
    OPPOSITE rotation direction from `_rotate_points` (see module docstring:
    ``center3D + R.T @ (gt - center3D)``)."""
    if R is None:
        return kp3d
    rel = kp3d - center3D[:, None, :]
    rel_grid = jnp.einsum("ji,bkj->bki", R, rel)   # R.T @ rel, per point
    return rel_grid + center3D[:, None, :]


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

def make_train_step(cfg: HMCachedConfig):
    from jarvis_jax.train.losses_3d import heatmap3d_mse

    grid_size, grid_spacing = cfg.grid_size, cfg.grid_spacing
    sharpen = cfg.sharpen
    sigma = cfg.sigma
    refine_enabled = cfg.refine_enabled
    refine_cube, refine_spacing = cfg.refine_cube, cfg.refine_spacing
    refine_sharpen = cfg.refine_sharpen
    refine_loss_weight = cfg.refine_loss_weight

    def loss_fn(model: C2FModel, batch: dict, R):
        heatmaps = batch["heatmaps"]           # (B,cam,J,224,224) f32
        center3D = batch["center3D"]
        centerHM = batch["centerHM"]
        camMat = batch["cameraMatrices"]
        kp3d = batch["kp3d"]
        vis = batch["vis"]

        points_local, pred_vol = _stage1_forward(
            model.v2v, heatmaps, center3D, centerHM, camMat,
            grid_size=grid_size, grid_spacing=grid_spacing, sharpen=sharpen,
            rotation=R, training=True)

        gt_grid_target = _rotate_gt_for_grid_target(kp3d, R, center3D)
        loss = heatmap3d_mse(pred_vol, gt_grid_target, vis, grid_spacing=grid_spacing,
                              roi_cube=grid_size * grid_spacing, center3D=center3D,
                              sigma=sigma)

        if refine_enabled:
            coarse_world = _rotate_points(points_local, R, center3D)
            # refine.py's own heatmap_size default (224, UNPADDED) -- see
            # module docstring: stage 2 needs raw heatmap resolution, no pad.
            hm_t = heatmaps.astype(jnp.float32)              # (B,cam,J,224,224)
            refined = refine_keypoints(model.refine, hm_t, coarse_world, centerHM,
                                       camMat, cube=refine_cube, spacing=refine_spacing,
                                       heatmap_size=_HM, sharpen=refine_sharpen,
                                       use_running_average=False)
            w = vis.astype(refined.dtype)
            refine_err = (((refined - kp3d) ** 2).sum(-1) * w).sum() / jnp.maximum(w.sum(), 1.0)
            loss = loss + refine_loss_weight * refine_err

        return loss

    @nnx.jit
    def step(model: C2FModel, opt: nnx.Optimizer, batch: dict, R):
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch, R)
        opt.update(model, grads)
        return loss

    return step


def make_eval_fn(cfg: HMCachedConfig):
    grid_size, grid_spacing = cfg.grid_size, cfg.grid_spacing
    sharpen = cfg.sharpen
    refine_enabled = cfg.refine_enabled
    refine_cube, refine_spacing = cfg.refine_cube, cfg.refine_spacing
    refine_sharpen = cfg.refine_sharpen

    @nnx.jit
    def eval_step(model: C2FModel, batch: dict):
        heatmaps = batch["heatmaps"]
        center3D = batch["center3D"]
        centerHM = batch["centerHM"]
        camMat = batch["cameraMatrices"]

        points_local, _ = _stage1_forward(
            model.v2v, heatmaps, center3D, centerHM, camMat,
            grid_size=grid_size, grid_spacing=grid_spacing, sharpen=sharpen,
            rotation=None, training=False)   # eval is always UNROTATED (true world)
        world = points_local + center3D[:, None, :]

        if refine_enabled:
            hm_t = heatmaps.astype(jnp.float32)
            world = refine_keypoints(model.refine, hm_t, world, centerHM, camMat,
                                     cube=refine_cube, spacing=refine_spacing,
                                     heatmap_size=_HM, sharpen=refine_sharpen,
                                     use_running_average=True)
        return world

    return eval_step


# ---------------------------------------------------------------------------
# run_cached_hm_training
# ---------------------------------------------------------------------------

def _check_nan_guard(loss: float, step_idx: int, total_steps: int, enabled: bool) -> bool:
    """Returns True iff a non-finite `loss` should halt training NOW (and, in
    that case, prints a loud, step-numbered report first) -- see A7_c2f_aug_femwt's
    divergence: loss was finite (0.02313) at step 800 and `nan` by 850, and
    the run then burned 650 more steps to a NaN result before anyone noticed.
    A no-op (returns False, prints nothing) whenever `enabled` is False or
    `loss` is finite."""
    if not enabled or np.isfinite(loss):
        return False
    print(f"!!! NON-FINITE LOSS at step {step_idx + 1}/{total_steps} "
          f"(loss={loss}) -- halting training early (nan_guard); "
          f"{total_steps - step_idx - 1} steps NOT run.", flush=True)
    return True


def save_run_config(run_dir: str, config: dict) -> str:
    os.makedirs(run_dir, exist_ok=True)
    latest = os.path.join(run_dir, "run_config.json")
    with open(latest, "w") as f:
        json.dump(config, f, indent=2, default=str)
    with open(os.path.join(run_dir, "run_config_history.jsonl"), "a") as f:
        f.write(json.dumps(config, default=str) + "\n")
    return latest


def _batches(cache, idx_pool, batch_size, *, shuffle, seed):
    idx = idx_pool.copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    stop = (len(idx) // batch_size) * batch_size
    for s in range(0, stop, batch_size):
        yield _gather_batch(cache, idx[s:s + batch_size])


def run_cached_hm_training(
    cache_dir: str, *, out_dir: str, ckpt_dir: str = None, cfg: HMCachedConfig = None,
    v5_root: str = None, save_every: int = 1000, log_every: int = 50,
    eval_every: int = 1000, max_ckpt_to_keep: int = 3,
) -> dict:
    cfg = cfg or HMCachedConfig()

    print(f"Loading train heatmap cache from {cache_dir} ...")
    train_cache = load_hm_cache(cache_dir, "train")
    val_cache = load_hm_cache(cache_dir, "val")
    print(f"  train: {train_cache['n']} samples   val: {val_cache['n']} samples")

    train_idx_pool = _select_train_indices(
        train_cache, exclude_second_fly=cfg.exclude_second_fly,
        female_weight=cfg.female_weight)
    print(f"  train index pool (post filter/oversample): {len(train_idx_pool)}")

    rngs = nnx.Rngs(cfg.seed)
    model = C2FModel(_NUM_JOINTS, refine_enabled=cfg.refine_enabled,
                     refine_channels=cfg.refine_channels, rngs=rngs)
    opt = make_optimizer(model, cfg)

    # max_ckpt_to_keep default (3) pruned A7's own pre-divergence checkpoint
    # before anyone could look at it (Task 15: A7 went NaN ~step 850, but by
    # the time that was noticed only steps 1200/1350/1500 -- all already
    # NaN -- survived Orbax's rolling retention). Callers investigating an
    # unstable arm should raise this.
    mngr = make_manager(ckpt_dir, max_to_keep=max_ckpt_to_keep) if ckpt_dir else None
    start = 0
    if mngr is not None:
        model, opt, start = restore_latest(mngr, model, opt)
        if start:
            print(f"Resuming from checkpoint at step {start}")

    run_dir = os.path.dirname((ckpt_dir or out_dir).rstrip("/")) or "."
    config_record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        **dataclasses.asdict(cfg),
        "cache_dir": cache_dir, "out_dir": out_dir, "ckpt_dir": ckpt_dir,
        "resumed_from_step": int(start),
        "n_train_samples": int(train_cache["n"]), "n_val_samples": int(val_cache["n"]),
        "n_train_pool": int(len(train_idx_pool)),
    }
    print(f"Wrote run config -> {save_run_config(run_dir, config_record)}")

    step_fn = make_train_step(cfg)
    eval_fn = make_eval_fn(cfg)

    # Representative gravity axis per calib group (all 3 rig calibrations
    # measured within ~2 deg of +Z -- see task-15-report.md); only computed
    # when rotation augmentation is on.
    axis_by_group = {}
    if cfg.rot_augment:
        for g in sorted(set(train_cache["calib_group"].tolist())):
            gi = np.nonzero(train_cache["calib_group"] == g)[0][0]
            axis_by_group[g] = gravity_axis(train_cache["cameraMatrices"][gi])
        print(f"rot_augment ON; gravity axes by group: {axis_by_group}")

    rng_np = np.random.default_rng(cfg.seed + 12345)

    def _sample_R(batch_groups):
        if not cfg.rot_augment:
            return None
        # One shared rotation per step (see module docstring); axis from the
        # DOMINANT calib group in this batch (cheap, and the 3 groups agree
        # to within ~2 deg anyway).
        vals, counts = np.unique(batch_groups, return_counts=True)
        axis = axis_by_group[vals[np.argmax(counts)]]
        R = sample_rotation(rng_np, tilt_deg=cfg.tilt_deg, axis=axis)
        return jnp.asarray(R)

    first_loss = None
    final_loss = 0.0
    step_idx = start
    epoch = 0
    diverged = False
    while step_idx < cfg.total_steps:
        idx = train_idx_pool.copy()
        np.random.default_rng(cfg.seed + epoch).shuffle(idx)
        stop = (len(idx) // cfg.batch_size) * cfg.batch_size
        for s in range(0, stop, cfg.batch_size):
            if step_idx >= cfg.total_steps:
                break
            sel = idx[s:s + cfg.batch_size]
            batch_np = _gather_batch(train_cache, sel)
            batch = {k: jax.device_put(v) for k, v in batch_np.items()}
            R = _sample_R(train_cache["calib_group"][sel])

            final_loss = float(step_fn(model, opt, batch, R))
            if step_idx == start:
                first_loss = final_loss

            if _check_nan_guard(final_loss, step_idx, cfg.total_steps, cfg.nan_guard):
                diverged = True
                break

            if (step_idx + 1) % log_every == 0:
                print(f"step {step_idx + 1}/{cfg.total_steps}  loss {final_loss:.5f}")

            if (step_idx + 1) % eval_every == 0:
                val_e = _eval_mpjpe(eval_fn, model, val_cache, cfg.batch_size)
                print(f"  val 3D MPJPE {val_e:.3f}")

            if mngr is not None and (step_idx + 1) % save_every == 0:
                save_step(mngr, step_idx + 1, model, opt)

            step_idx += 1
        epoch += 1
        if diverged:
            break

    if mngr is not None:
        # step_idx == cfg.total_steps on a normal completion; on an early
        # nan_guard exit it is the true (lower) number of completed steps --
        # using it (not cfg.total_steps) keeps this checkpoint's key honest.
        save_step(mngr, step_idx, model, opt)
        mngr.wait_until_finished()

    # ------------------------------------------------------------------
    # Final val pass: overall + by_calib_group + by_sex MPJPE, and the
    # Task-16 arrays (val_pred.npz here; val_gt.npz/keypoint_names.json
    # once, shared across arms).
    # ------------------------------------------------------------------
    val_mpjpe, val_pred, breakdown = _eval_full(eval_fn, model, val_cache, cfg.batch_size)
    print(f"Final val 3D MPJPE {val_mpjpe:.3f}")
    print(f"  by_calib_group: {breakdown['by_calib_group']}")
    print(f"  by_sex: {breakdown['by_sex']}")

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "..", "val_pred.npz"), kp3d=val_pred)

    val_mpjpe_path = os.path.join(out_dir, "..", "val_mpjpe.json")
    with open(val_mpjpe_path, "w") as f:
        json.dump({"overall": val_mpjpe, **breakdown}, f, indent=2)

    if v5_root is not None:
        ann_dir = os.path.join(v5_root, "annotations")
        gt_path = os.path.join(ann_dir, "val_gt.npz")
        names_path = os.path.join(ann_dir, "keypoint_names.json")
        if not os.path.isfile(gt_path):
            os.makedirs(ann_dir, exist_ok=True)
            np.savez(gt_path, kp3d=val_cache["kp3d"])
            print(f"wrote {gt_path}")
        if not os.path.isfile(names_path):
            with open(os.path.join(v5_root, "annotations", "instances_val.json")) as f:
                inst = json.load(f)
            with open(names_path, "w") as f:
                json.dump(inst.get("keypoint_names", []), f)
            print(f"wrote {names_path}")

    ckptr = ocp.StandardCheckpointer()
    ckptr.save(out_dir, nnx.split(model)[1], force=True)
    ckptr.wait_until_finished()

    return {
        "first_loss": first_loss if first_loss is not None else final_loss,
        "final_loss": final_loss,
        "val_mpjpe_3d": val_mpjpe,
        "steps": step_idx,
        "diverged": diverged,
    }


def _eval_mpjpe(eval_fn, model, cache, batch_size) -> float:
    total, count = 0.0, 0
    n = cache["n"]
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        idx = np.arange(s, e)
        batch_np = _gather_batch(cache, idx)
        batch = {k: jax.device_put(v) for k, v in batch_np.items()}
        pts = eval_fn(model, batch)
        vis = jnp.asarray(batch_np["vis"])
        kp3d = jnp.asarray(batch_np["kp3d"])
        nv = int(vis.sum())
        if nv == 0:
            continue
        err = float(mpjpe_3d(pts, kp3d, vis)) * nv
        total += err
        count += nv
    return total / max(count, 1)


def _eval_full(eval_fn, model, cache, batch_size):
    n = cache["n"]
    preds = np.zeros((n, _NUM_JOINTS, 3), np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        idx = np.arange(s, e)
        batch_np = _gather_batch(cache, idx)
        batch = {k: jax.device_put(v) for k, v in batch_np.items()}
        pts = np.asarray(eval_fn(model, batch))
        preds[s:e] = pts

    kp3d = cache["kp3d"]
    vis = cache["vis"]
    d = np.linalg.norm(preds - kp3d, axis=-1)   # (n,J)
    w = vis.astype(np.float64)

    def _mpjpe(mask):
        ww = w[mask]
        dd = d[mask]
        denom = ww.sum()
        return float((dd * ww).sum() / denom) if denom > 0 else float("nan")

    overall = _mpjpe(np.ones(n, bool))
    by_group = {}
    for g in sorted(set(cache["calib_group"].tolist())):
        by_group[g] = _mpjpe(cache["calib_group"] == g)
    by_sex = {
        "female": _mpjpe(cache["is_female"]),
        "male": _mpjpe(~cache["is_female"]),
    }
    return overall, preds, {"by_calib_group": by_group, "by_sex": by_sex}
