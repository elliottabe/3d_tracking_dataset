# Multi-view query lifter: DINOv3 encoder + D4RT-style query decoder

**Date:** 2026-09-03
**Status:** design approved in conversation, awaiting implementation plan
**Short name:** `mvq` (multi-view query)
**Replaces, when promoted:** Stage A (ViTPose heatmaps) + Stage B (robust DLT)
in `scripts/run_bout.py`. Everything downstream (Stage B2 filter, STAC IK,
polish, viz) is unchanged.
**Related:** `docs/specs/2026-08-29-coarse-to-fine-3d-design.md` (the
volumetric HybridNet line this design retires),
`docs/benchmark/2026-09-02-vitpose-maskoff-ab/` (current detector),
commit `019a52b` (distractor supervision in 2D).

---

## 1. Why

The pipeline today is ViTPose 2D per camera -> robust DLT -> STAC IK. Its
measured failure is the female fly at the wall, in occlusion, and in contact
with the male: a single-view leg swap drags the DLT, a small or edge-on
female loses her keypoints to the male's leftover silhouette after gray-fill,
and per-frame heatmaps carry no temporal or cross-view context. The
volumetric HybridNet line fuses views but quantizes at 0.2 mm per grid step
on a 0.16 mm tarsal segment, is locked to the training world frame, and
mean-pools cameras.

This design takes one idea from D4RT (Zhang et al., CVPR 2026, arXiv
2512.08924): a heavy encoder builds a shared token bank once, and a
lightweight cross-attention decoder answers independent queries against it.
D4RT itself is a monocular, scene-scale, 1B-parameter model with no notion of
a named landmark, no multi-camera input, and no released weights; a port was
evaluated and rejected (spike, 2026-09-03). What transfers is the decoder
architecture and the loss structure. The encoder is DINOv3 ViT-B/16, whose
JAX port exists in `jax-ml/bonsai` and whose gated weights we now have access
to.

### Facts from the codebase survey that shape the design

1. **The cameras are affine.** Every shipped `Cam*.yaml` has projection row 3
   equal to `[0,0,0,1]`; projection is `uv = M X + t` with `M` 2x3, `t` 2, no
   camera centre, no distortion (`jarvis_jax/tracking/affine_camera.py`).
   Reprojection is linear, and every geometric augmentation below is exact.
2. **There is no 3D ground truth independent of the calibration.** `kp3d`
   is DLT of the human 2D labels, computed in `v5_3d.py.__getitem__`; a joint
   labelled in fewer than two cameras has no 3D label. Per-camera 2D
   reprojection is therefore the primary supervision, and 3D L1 is auxiliary.
3. **Data is small and skewed.** Live root `red_data_3d_v12_export0902`:
   2,661 train framesets, 153 val; images 1936x448; two-fly framesets are
   1.7% of train and 24% of val; calibration group C appears only in val,
   group B only in train.
4. **Temporal labels are thin but exist.** About 735 consecutive labelled
   frame pairs in train, mostly in `2026_02_13_13_44_49`; 15 pairs in val.
5. **Train and inference crops differ on the 3D path.** Training centres
   each camera's crop on its own GT bbox; inference (`build_frameset`)
   centres all seven on one triangulated mask centroid and gray-fills the
   other fly. The new loader uses the inference convention.
6. **World units:** 0.1 mm per world unit (`scripts/estimate_recording_scale.py`,
   three anchors and a guard). Mean projection scale is about 8 px per unit.
7. **Output contract:** `kp3d.npz` = `{kp3d (T,50,3) f32, conf3d (T,50) f32,
   gates str}` with NaN as the no-estimate sentinel, keypoint axis in
   `cfg.model.KP_NAMES` order; `kp2d.npz` = `{kp2d (T,7,50,2), conf (T,7,50)}`
   in full-frame px, canonical camera order.

## 2. Decisions taken (with the user, 2026-09-03)

| decision | choice |
|---|---|
| Output | 3D directly (ROI-local) plus 2D per view; replaces Stages A+B |
| Acceptance | bout 28 figures (female wall/occlusion) and overall generalization: v12 val cohorts incl. calibration group C, frozen 13-bout benchmark |
| Temporal | T-aware from day one; train on T=1 and T=2 human labels; long windows via pseudo-labels are a later spec |
| 3D frame | ROI-local offsets from center3D; calibration enters as per-token geometry embeddings; no world-frame identity |
| Identity | promptable: learned instance queries by default (set prediction), optional mask-derived prompt token dropped with p=0.5 in training |
| Backbone | DINOv3 ViT-B/16, full fine-tune, 4 GPUs, one node |
| Architecture | A: per-view encoder, light interleaved cross-view fusion, D4RT-style decoder; B (no fusion) is the first ablation |

## 3. Non-goals

- Mask-free localisation of the crop. Version 1 still takes center3D from
  the SAM3 mask centroids. A coarse localiser on full frames is separable.
- Long-range identity through crossings, and sex canonicalisation. Both stay
  with SAM3 ids, `identity_link.py`, and the wing-song step.
- Pseudo-labelled long windows from the current pipeline. Phase 2, own spec.
- Retiring robust DLT. `pipeline.lifter` keeps `dlt` selectable; promotion
  is gated on the benchmark.
- Fixing STAC IK. Prior work found IK to be the end-to-end bottleneck; Phase
  P4 figures will show whether IK absorbs the 3D gain.

## 4. Architecture

### 4.1 Inputs per sample

- `crops` (T, C=7, 448, 448, 3) uint8, each centred on the projection of
  the window's reference `center3D` into that camera, clamped like
  `build_frameset`. No mask channel.
- `cam_valid` (T, C) bool. Absent cameras (None frameset slots, dropped
  views) are masked out of attention and all losses.
- `M` (C, 2, 3), `t` (C, 2) affine rows in full-frame px; `crop_origin`
  (T, C, 2); `center3D` (3,), one per window (the host fly's).
- Optional `prompt_mask` (T, C, 448, 448) bool for the prompted mode.

### 4.2 Backbone: DINOv3 ViT-B/16

`jarvis_jax/models/dinov3.py`, registered in `models/backbone.py`. Vendored
and adapted from `bonsai/models/dinov3/{modeling,params}.py` (Apache-2.0):
patch 16, width 768, 12 layers, 12 heads, GELU MLP x4, 4 register tokens,
axial RoPE (base 100, coords normalised per axis, applied in bf16 as the
reference does), LayerScale, LayerNorm eps 1e-5. Inputs NHWC, ImageNet
mean/std. Weights loaded from `facebook/dinov3-vitb16-pretrain-lvd1689m`
safetensors via the bonsai key map; a loader test asserts every checkpoint
key is consumed.

All 7*T crops of a batch run as one image batch with shared weights and
per-block `nnx.remat`. Only the 784 patch tokens per crop enter the bank;
cls and registers are dropped. 448 px is handled natively by RoPE.

### 4.3 Geometry embedding (affine ray tokens)

For an affine camera every pixel back-projects to a line with the same
direction `d = null(M)` (unit 3-vector) through a point `p0(uv)` that is
linear in the pixel (least-norm solution of `M p0 = uv - t_local`, where
`t_local = M center3D + t - crop_origin` folds the ROI origin and crop into
the offset). For each token, `(p0, d)` in ROI-local coordinates is
Fourier-embedded (8 bands) and linearly projected to width 768, then added
to the token. The decoder can triangulate by finding where lines meet;
nothing names a world frame or calibration group.

Also added per token: a learned relative-frame embedding (0..T-1, max 8) and,
behind `camera_slot_embed: true`, a learned per-camera-slot embedding for
per-camera appearance (slots are constant across calibration groups).

### 4.4 Fusion stack

Four blocks trained from scratch, LayerScale initialised to zero so the
pretrained features pass through unchanged at step 0. Interleaved:

- **local**: self-attention within each view's 784 tokens;
- **global**: self-attention over a 2x2 average-pooled bank (196 tokens per
  view, ~1.4k for T=1, ~2.7k for T=2), update broadcast back to the four
  children.

`n_global: 0` yields Approach B for the ablation. Output: bank
(T*7*784, 768) with validity mask.

### 4.5 Prompt token

Masked mean of the target fly's patch tokens over its valid views (mask
downsampled to the 28x28 token grid), projected to 768. Present with
probability `prompt_p` in training (annealed 1.0 -> 0.5 over the first 2k
steps), always present in the shipped prompted mode, absent in unprompted
mode.

### 4.6 Decoder

**Instance queries.** `n_instances` learned embeddings: 3 for courtship
(num_animals + 1 slack), 2 for single-fly. The prompt token is added to
instance 0 when present. Existence head: mean of an instance's 3D query
features -> linear -> logit.

**3D keypoint queries** (heavy path). For each (instance i, keypoint k,
frame t): `q = E_inst[i] + E_kp[k] + E_frame[t]`; 300 queries for T=2.
Eight pre-norm blocks of width 768, 12 heads: self-attention over all 3D
queries in the window, cross-attention to the bank (validity-masked), MLP x4.
The self-attention is a deliberate departure from D4RT's independent queries
(skeleton prior within an instance, repulsion between instances; D4RT
dropped it only for dense-decoding cost). Head: linear -> 3-vector offset in
ROI-local world units scaled by 1/24, plus a confidence logit.

**2D view queries** (light path). For each (i, k, camera c, t):
`q = proj(final 3D query feature) + E_geom[c] + E_frame[t]`, where `E_geom`
is the Fourier embedding of the camera's affine rows (not a slot index).
Four cross-attention-only blocks, D4RT style; 2,100 queries for T=2. Head:
(u, v) in crop-normalised [0,1] plus a visibility logit.

**Refinement pass** (`refine_passes: 1`, default on). For every 3D query,
gather what pass 1 committed to: bilinear samples of the token grid and a
9x9 RGB patch at the reprojected (u, v) in each valid view, averaged over
views, plus Fourier-embedded coordinates. Add to the query, re-run the same
decoder weights with a pass embedding, predict residuals on 3D and 2D. This
is the direct reuse of D4RT's (u, v, t) point query and the sub-pixel
mechanism regression heads otherwise lack.

**Assembly.** `kp3d_world = center3D + 24 * offset`; `kp2d_full = crop_origin
+ 448 * uv`; `conf3d = sigmoid(conf logit)`. A keypoint with no valid view,
or belonging to an instance with existence below `exist_thresh` (0.5), is
emitted as NaN with conf3d 0.

## 5. Losses and matching

All errors are in pixels so weights are comparable; 3D errors are converted
with the recording's mean projection scale `s = mean ||M||_F / sqrt(2)`
(about 8 px/unit). Masked means over valid entries; mean over the batch.

**Matching.** A prompted instance is assigned to the prompted fly. Otherwise
instances are matched to labelled flies per window by minimum cost over all
injective assignments, enumerated in JAX (at most 6 for N=3, two flies).
Cost = mean reprojection error + 3D L1 where 3D exists. Unmatched instances
receive only the existence loss.

| # | term | on | form | weight |
|---|---|---|---|---|
| 1 | reprojection (primary) | each labelled view, v>0 | Huber(delta=8 px) of `M kp3d + t` vs human 2D | 1.0 |
| 2 | 3D L1 | joints with >=2 labelled views | `s * |offset - offset_gt|_1` | 0.5 |
| 3 | 2D view head | each labelled view, v>0 | Huber(8 px) of view (u,v) vs human 2D | 0.5 |
| 4 | visibility BCE | each view query | target = label v flag; absent cameras masked | 0.1 |
| 5 | confidence (D4RT) | each 3D query | `c * err_reproj - lambda log c`, c = sigmoid | 0.2 |
| 6 | existence BCE | each instance | matched 1, unmatched 0 | 1.0 |
| 7 | repulsion | two-fly frames | hinge `max(0, r - dist)` to the other fly's same-part labels (15-part index from `data/distractor.py`): r = 20 px per view, 2.5 world units in 3D | 0.5 |

**Deep supervision.** Terms 1-3 also on pass-1 outputs at weight 0.5 and on
every second decoder block through the shared heads at weight 0.3.

**Not losses, always metrics:** rigid segment length spread within a window
(wing veins, femora, tibiae, eye spacing), left/right symmetry, 2D-head vs
reprojection disagreement (successor of the residual gate), error by sex,
calibration group, and one-/two-fly. `rigid_length_weight` exists, default
0.0.

**Balance.** The manifest-driven balanced sampler over `<behavior>_<sex>`
(alpha 0.5) reused from `v5_2d.py`; `female_weight` per-sample loss weight,
default 1.0. All weights above are starting points recorded in
`configs/train/mvq.yaml`; P3 includes a short sweep on reprojection vs 3D
weight.

## 6. Data

### 6.1 Windows (`jarvis_jax/data/v12_windows.py`)

A sample is (recording, host fly, start frame, T). T=1: every frameset
(2,661 train). T=2: every consecutive labelled frame pair for the host
(~735 train, 15 val). Batches are bucketed by T; each T has its own jitted
step; the trainer alternates buckets. The host fly defines `center3D` =
midrange of its triangulated GT, jittered by +-3 world units to mimic
mask-centroid noise. Every other labelled fly whose keypoints fall inside
the crop is a second instance; its keypoints outside a given view get v=0
there. None frameset slots -> `cam_valid = False`. Keypoint order is
asserted equal to `annotations/keypoint_names.json`.

### 6.2 Strip cache (`jarvis_jax/data/strip_cache.py`,
`scripts/precompute_strip_cache.py`)

Full fine-tuning rules out feature caching, and decoding ~450 JPEGs of
1936x448 per step would starve 4 GPUs. A one-off script stores a 576-wide
full-height strip per (frameset, camera) in a uint8 memmap (~14 GB train)
with the strip origin, and the same for masks. The 448 crop is cut on device;
the centre jitter is absorbed by the wider strip. Cache meta records the
dataset root and version; a mismatch forces a rebuild.

### 6.3 Augmentation (`jarvis_jax/data/mv_augment.py`, on device)

Geometric, all exact under affine cameras:

- **Per-view affine** (rotation +-30 deg, scale 0.8-1.25, translation
  0.1): warp the crop by (A, b) and set `M' = A M`, `t' = A t + b`. Geometry
  tokens use the warped cameras. The bounds-fit from `augment.py`
  (rotate about the visible-keypoint centroid) is reused.
- **World rotation** (yaw 360 deg, tilt +-30 deg; `rot_augment: true`):
  images untouched, labels rotated, `M' = M R^T`.
- **Global mirror** (`mirror_p: 0.5`): flip all views, reflect the world,
  swap L/R keypoint names via `build_lr_swap` (asserted involution), update
  every M.
- **No per-view flips** (an unrealisable camera).

Robustness: photometric per view from `augment.py` defaults, cutout, camera
dropout (`cam_drop_p: 0.3`, 1-2 views).

Two-fly synthesis: distractor gray-fill with `fill_p` (15/60 dilation rule
from `data/distractor.py`); multi-view copy-paste of a donor frameset at a
3D translation Delta, each view shifted by `M Delta`, same calibration group
only (`copy_paste_p`, default 0.0 in the first run, on for the two-fly
ablation).

### 6.4 Split

The v12 recording-level split unchanged. Group C is val-only and reported
as its own cohort. Temporal metrics also run on bout 28 because val T=2 pairs
are thin.

## 7. Training

`jarvis_jax/train/train_mvq.py`, Hydra `configs/model/mvq.yaml`,
`configs/train/mvq.yaml`, launched through the existing SLURM training
launcher on one 4-GPU node (data-parallel sharding). AdamW, warmup-cosine
(warmup 500, 30k steps), weight decay 0.05, grad clip, bf16 compute / fp32
params, batch 8 per GPU (fallback 4 with accumulation). Two LR groups:
backbone 0.1x, fusion+decoder 1x. Weight EMA (`ema: 0.999`, on). Orbax
checkpoints, auto-resume.

Eval every 2k steps on v12 val in both prompted and unprompted modes: 3D
MPJPE (world units and mm), per-view 2D px error, cohorts (sex, calibration
group, one-/two-fly), instance existence precision/recall on two-fly frames,
rigid length spread, 2D-head vs reprojection disagreement. The trainer raises
if a named cohort is empty.

Compute placement per CLAUDE.md: compute nodes only; on an idle GPU node run
directly, otherwise `scripts/slurm/submit_task.sh`.

## 8. Pipeline integration

`pipeline.lifter: dlt | mvq` in `scripts/run_bout.py`. The `mvq` path
(`jarvis_jax/tracking/lift_mvq.py`) replaces Stages A and B: synchronized
frames via `read_window`, crops from mask-centroid `center3D` exactly as
`build_frameset`, windows of T frames, and it writes the same artifacts:

- `kp2d.npz`: view-head output in full-frame px, `conf` = view visibility
  sigmoid; keypoint axis in v12 order (== detector order), so
  `reorder_detector_to_model` and `verify_detector_kp_order` (via the run's
  `.hydra/overrides.yaml`) apply unchanged;
- `kp3d.npz`: `kp3d`, `conf3d`, and a `gates` signature that includes the
  checkpoint hash, T, and prompt mode so the stale-artifact check keeps
  working. NaN sentinel.

Prompted mode is the version-1 default (prompt token from the fly's masks).
Unprompted mode (instance nearest the mask `center3D`) is evaluated offline
and on bout 28 and is not shipped until it matches prompted mode. Stage B2,
STAC, polish and all viz see identical inputs. Throughput: ~1 TFLOP per T=2
window forward, a 3,000-frame bout well under a minute on an L40S.

## 9. Testing (TDD)

- Backbone parity vs bonsai's reference module on identical input;
  loaded-weight coverage (every safetensors key consumed).
- Geometry: `p0 + s d` projects to the token pixel for all s; per-view
  affine warp + camera update keeps GT 3D reprojecting onto warped 2D labels
  exactly; world rotation leaves the loss unchanged; mirror composes with
  the L/R swap; the swap is an involution.
- Matching: brute-force assignment equals `scipy.optimize.linear_sum_assignment`
  on random costs; prompted instance always pinned.
- Losses: zero at ground truth; masked means ignore invalid entries;
  repulsion exactly zero with no second fly; confidence term lowers c where
  error is high.
- Loader: T=2 window count equals the frameset-key census; None slots
  invalid; keypoint order asserted; two-fly framesets yield two instances;
  crops byte-identical to `build_frameset` on a synthetic frame.
- Output contract: shapes, dtypes, NaN sentinel, model keypoint order,
  gate signature accepted by the bout driver (extend
  `tests/test_bout_artifacts_order.py`).
- Jitted steps for T=1, T=2, and camera dropout.

## 10. Figure gates (expectation stated in each script before rendering)

1. **Geometry sanity, 1k steps:** reprojected pass-1 3D on val frames lands
   on the fly in all views. If not, the geometry tokens are wrong before
   anything else is.
2. **Bout 28 female wall/occlusion frames, 3+ cameras,** new vs ViTPose+DLT
   via `python -m viz overlay --compare`: tarsal tips stay on the leg
   through the wall band; no keypoint sits on the male.
3. **Rigid invariant traces per frame,** wing veins and femora, both flies,
   baseline vs new: flat lines, spread reported in mm.
4. **Two-fly val frames,** instance-coloured, prompted vs unprompted.
5. **Calibration group C val error next to A** (generalization).
6. **Benchmark scorecard delta** (`scripts/benchmark/`) plus STAC
   joint-angle traces on bout 28.

Figures under `figures/2026-09-mvq/`; scorecards and notes under
`docs/benchmark/2026-09-mvq/`. Every figure uses keypoint, camera and fly
names and units; nothing is indexed by integer position.

## 11. Code placement

```
third_party/jarvis_jax/jarvis_jax/
  models/dinov3.py                 backbone + safetensors loader (from bonsai)
  models/mvq/geometry.py           affine ray tokens, camera updates for aug
  models/mvq/fusion.py             local/global interleaved fusion
  models/mvq/decoder.py            instance/3D/2D queries, refinement pass
  models/mvq/model.py              MVQ module, assembly, NaN policy
  train/train_mvq.py               trainer, eval cohorts
  train/losses_mvq.py              terms 1-7, deep supervision
  train/matching.py                enumerated assignment
  data/v12_windows.py              window dataset
  data/strip_cache.py              memmap strips
  data/mv_augment.py               exact multi-view augmentation
  tracking/lift_mvq.py             pipeline lifter (Stages A+B replacement)
scripts/precompute_strip_cache.py
scripts/viz/mvq_overlay.py         promoted figure script (gates 1, 2, 4)
configs/model/mvq.yaml, configs/train/mvq.yaml
```

## 12. Risks

- **Regression precision below heatmaps.** The refinement pass exists for
  this. Contingency: a small local heatmap readout in the 2D head.
- **Overfitting 2.7k framesets.** EMA, deep supervision, exact geometric
  augmentation; watch the train/val gap and the group-C cohort.
- **Unstable matching early.** Prompt probability anneals 1.0 -> 0.5 over
  the first 2k steps.
- **Memory.** 14 crops per sample with remat; fallback batch 4 + grad
  accumulation. Prior note: 8-way JAX concurrency on one node has failed;
  this is one process over 4 GPUs.
- **Two-fly scarcity in train (1.7%).** The copy-paste ablation answers it.
- **DINOv3 features at macro scale.** Unknown; the P2 T=1 result against
  DLT and the A4 arm is the checkpoint before spending on T=2 and ablations.

## 13. Phases (for the implementation plan)

- **P0** Backbone port, weight loader, parity and coverage tests.
- **P1** Strip cache, window loader, geometry module, augmentation, tests.
- **P2** Model, losses, matching, T=1 training on 4 GPUs; gate 1; val vs
  DLT and the A4 arm; gates 3 and 5 on val.
- **P3** T=2 buckets; ablations one variable each: A vs B (`n_global`),
  refinement on/off, prompted vs unprompted, copy-paste on/off, reprojection
  vs 3D weight sweep.
- **P4** Pipeline lifter, bout 28 end-to-end, gates 2, 4, 6; benchmark
  scorecard; promotion decision.
- **Phase 2 (separate spec):** pseudo-labelled long windows from
  gated pipeline outputs (raw robust-DLT, per-keypoint validity from
  consensus residual, rigid invariants, view count; balanced by sex, wall
  proximity, and contact), validated only on human consecutive pairs and
  bout 28.

## 14. Provenance

- D4RT: Zhang et al., "Efficiently Reconstructing Dynamic Scenes One D4RT
  at a Time", arXiv 2512.08924; project page d4rt-paper.github.io. No
  official code or weights.
- Reimplementations inspected (scratchpad, throwaway):
  `Lijiaxin0111/Open-d4rt` (PyTorch, Apache-2.0, ViT-g weights on HF) and
  `jiangyurong609/d4rt-pytorch` (PyTorch, Apache-2.0). Their decoder
  (cross-attention-only blocks), query embedder (Fourier uv + learned
  timestep embeddings + 9x9 RGB patch), heads and loss structure informed
  sections 4.6 and 5.
- DINOv3 JAX: `jax-ml/bonsai/bonsai/models/dinov3/` (Apache-2.0, merged
  2025-12-23), parity-tested against `transformers.DINOv3ViTModel`.
- Codebase survey 2026-09-03 (this conversation): affine calibration,
  load-time DLT 3D labels, v12 counts, consecutive-pair census, crop
  convention mismatch, `kp3d.npz` contract, distractor loss implementations.
