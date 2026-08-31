# Coarse-to-fine 3D keypoints for fly wing/leg kinematics

**Date:** 2026-08-29
**Status:** design approved, awaiting implementation plan
**Goal:** clean wing and leg kinematics for BOTH flies on Session0
`2025_10_20_13_20_04` bout 28, delivered end-to-end through STAC IK.
**Extends:** `docs/specs/2026-08-07-tracking-upgrade-program-design.md`

---

## 1. Why

The pipeline today is ViTPose 2D -> robust DLT -> STAC IK -> polish. A
HybridNet-style learned 3D lifter exists in `jarvis_jax` (`hybridnet/`,
`train/train_3d_cached.py`, `submit_3d_vitpose.sh`) and was trained
(`v2v_vitpose`, val 3D MPJPE 1.083) but never shipped. Prior A/Bs on it
contradicted each other: median metrics called learned fusion "a wash with
DLT", while tail metrics showed it cutting jitter spikes 5x (834 -> 156).

Five measurements taken during this design resolve the contradiction and
identify the real defects.

### 1.1 The 48^3 voxel grid is too coarse for a fly (primary defect)

The shipped grid is `grid_size: 48, grid_spacing: 1`, so **1 voxel = 1 world
unit**. Triangulated GT from `courtship_V2` (120 framesets) gives:

| segment | voxels |
|---|---|
| `Scutellum -> Abd_tip` (body) | 12.80 |
| `WingL_base -> WingL_V12` | 17.47 |
| `WingL_V12 -> WingL_V13` | 4.92 |
| `T1L_FeTi -> T1L_TiTa` (tibia) | 4.43 |
| `T1L_TiTa -> T1L_TaT1` | 1.98 |
| `T1L_TaT1 -> T1L_TaT3` | 2.17 |
| `T1L_TaT3 -> T1L_TaTip` | **1.59** |
| `EyeL -> EyeR` | 4.32 |

Calibrating to physical units (body 12.8 u ~ 2.2 mm; head width 4.32 u ~ 0.8
mm) gives **1 voxel ~ 0.17 mm**. The distal tarsal segment -- the thing that
*is* leg kinematics -- is about one and a half voxels. JARVIS's 48^3 was tuned
for rats and primates, where a limb segment spans dozens of voxels; it was
never re-derived for a 2 mm fly.

Measured against the 2D it consumes: **1 voxel = 2.7-3.2 heatmap pixels**
(per-camera, `px_per_unit` probe). The 3D stage samples the heatmaps ~3x
coarser than their own resolution -- it discards 2D detail it already has.

This explains the contradictory A/Bs: **DLT is continuous and sub-pixel
(precise, brittle); HybridNet fuses all views (robust, quantized at ~0.17 mm)**.
They fail orthogonally. Neither alone is what the kinematics need.

### 1.2 The 2D detector is trained to produce diffuse heatmaps

`jarvis_jax/data/transforms.py:47` sets target `sigma=7.0` heatmap px. A
tarsal segment is ~4.6 heatmap px (1.59 voxels x ~2.9 px/voxel), so the target
Gaussian's sigma alone is ~1.5x longer than the whole segment; adjacent tarsal
targets overlap almost completely. This is the direct cause of the previously
measured heatmap peak concentration of **0.352**. Soft-argmax can interpolate
sub-voxel only on a sharp peak, so 1.1 and 1.2 compound.

sigma/heatmap_size = 3.1% matches the COCO convention, but COCO was tuned for
humans, whose limb segments span a large fraction of the frame. Against the
fly's ~37 heatmap-px body length, sigma=7 is ~19% -- about 5x too wide.

**CORRECTION (2026-08-30, after the Phase 2 retrains): the prescription in
this section was WRONG. sigma=7.0 is correct and sigma=2.0 is a 6.5x
regression.** The anatomical measurement above is right; the inference drawn
from it does not follow.

Controlled test, same dataset / sampling / augmentation, first eval at step
1500 -- the only variable is sigma:

| target sigma | val MPJPE @1500 |
|---|---|
| 7.0 | 28.98 px |
| 2.0 | 86.17 px |

Trained to completion, on the new leakage-free split (female =
`2026_05_27_11_56_05`):

| run | data | sigma | sampling | aug | val MPJPE | female |
|---|---|---|---|---|---|---|
| `v4_8gpu_20260808` | old | 7.0 | -- | default | 8.553 px | 16.524 px |
| `v5_sigma2_bal` | new | **2.0** | balanced | heavy | **55.528 px** | 62.382 px |
| `v5_s70_bal` | new | 7.0 | balanced | heavy | 6.448 px | 20.678 px |
| `v5_s70_bal_augdef` | new | 7.0 | balanced | default | **5.915 px** | **14.239 px** |

**Why the reasoning failed: sigma sets the optimisation basin, not the
resolution ceiling.** This section compared sigma to the *animal* (7 px against
a 4.6 px tarsal segment, hence "5x too wide"). But the target Gaussian's width
determines how far a mispredicted peak can be and still receive gradient toward
the right answer -- it does not cap how sharply the trained model can peak. The
~3% sigma/heatmap convention is an optimisation constant, and sigma=7/224 =
3.1% is textbook where sigma=2/224 = 0.9% is far below anything standard. At
sigma=2 the model still learned crisp peaks (concentration 0.885, up from
0.352) -- it just fired them **on the wrong legs**: 70.6% of its >30 px misses
land closer to the mirror keypoint's GT than to their own. Sharpening the
target bought peak quality and lost the correspondence that makes a peak mean
anything.

**The tarsal-resolution problem in 1.1 is real and unaffected by this.** It
must be fixed downstream, in coarse-to-fine stage 2, not by tightening the 2D
target. The measurement in 1.1 stands; only 1.2's remedy is withdrawn.

Two consequences for what follows. First, Phase 2's sigma change is reversed
(see below) -- the retrain is still worth doing, but for the dataset, not the
sigma. Second, one earlier test is downgraded rather than refuted: the
flip-augmentation A/B (72.9 vs 73.6 px) ran at sigma=2, where **both** arms
were broken, so it was inconclusive about flip and was never rerun at sigma=7.

### 1.3 The train/val split leaks, so no prior 3D number is trustworthy

The split is **within-recording and frame-level**: `general_model/<subset>/`
holds `train/` and `val/` directories over the *same* recording. Fraction of
val framesets having a train frameset within +/-3 frames:

| | leak |
|---|---|
| `S8_male_R_amp`, `courtship_25_51_female` | 100% |
| `courtship_V3` | 86% |
| `headless_24_04_1` | 80% |
| `grooming` | 64% |
| overall | ~50% |

**CORRECTION (2026-08-29, after Task 5): this leakage does NOT reach the
trained models, and the two claims that followed from it were WRONG.**

The frame-level split above is real, but it lives only in
`general_model/<subset>/annotations/`. Neither trainer reads those files.
Measured directly:

| dataset | consumed by | train recs | val recs | shared |
|---|---|---|---|---|
| `red_data_unified_V3` | 3D trainer (`paths.data_root`) | 13 | 4 | **0** |
| `red_data_unified_V4` | detector `v4_8gpu_20260808` | 11 | 9 | **0** |
| `red_data_unified_V4_femclimb` | — | 12 | 9 | **0** |

All three are RECORDING-level with zero overlap: the unified builders re-split
by recording, so `general_model`'s frame-level leak never propagated.

Therefore: **val MPJPE 1.083 / 0.708 and the laplacian ablation are NOT
invalidated by leakage**, and **the shipped detector is NOT contaminated**.
Retrain it for the sigma fix in 1.2 (a real, independent measurement), not for
contamination. What survives from this section is narrower but still worth
doing: our split additionally preserves scarce female data via guard bands and
covers 26 recordings rather than 17. That is an improvement, not a defect fix.

### 1.4 264 complete framesets are silently discarded

`jarvis_jax/data/v3_3d.py:122-128` maps `image_id -> first annotation`, with
the comment "shouldn't happen for single-fly". It does happen, on four two-fly
courtship recordings:

| recording | 2-fly framesets | dropped anns | 2nd-fly reproj |
|---|---|---|---|
| 2026_04_08_14_59_45 | 214 | 1374 | 0.39 px |
| 2026_04_07_11_33_33 | 30 | 181 | 0.40 px |
| 2026_06_11_13_58_43 | 17 | 98 | 0.39 px |
| 2026_06_11_13_58_45 | 3 | 20 | 0.39 px |
| **total** | **264** | **1673** | |

The dropped fly triangulates at 0.39 px -- identical to the kept fly. It is
clean, complete, 7-camera data, 9.3% of the dataset, and since these are
courtship recordings roughly half are female.

A chimera hypothesis (that "keep first" mixes fly A in one camera with fly B in
another) was **tested and refuted**: two-fly framesets triangulate at 0.39 px
median, same as single-fly controls. Annotation order is consistent across
cameras. The defect is data loss, not corruption.

### 1.5 The calibration domain-shift story on record is wrong

Prior notes attribute the V3 HybridNet collapse on Session0 to "Session0 is
2025, all training is 2026". Content-hashing every calibration refutes this:

| group | labeled recordings | framesets |
|---|---|---|
| **A** (= bout 28's) | courtship_V2/V3/V4, 20_04_female_climbing, **2026_04_08**, **2026_04_07** | **974** |
| B | 18 recordings (S6male, amputees, headless, grooming, female, wall_frames, courtship_25_51/28_34, 2026_03_22, 2026_06_11 x2) | 2532 |
| C | courtship_11_50 female + male | 30 |

Note `2026_04_08_14_59_45` (251 framesets, **both flies labeled**) is in bout
28's own calibration group -- the single most valuable recording here.

Session0's calibration is **byte-identical** to `courtship_V3`/`V2`/`V4` and
numerically identical to `20_04_female_climbing`. The rig was recalibrated
*between 2026 recordings*. Group differences are large (max relative difference
1.6-1.9 in the projection matrices) -- genuinely different world frames.

`hybridnet/reproject.py` shows the mechanism precisely: `grid = base_grid +
center3D` (grid is **axis-aligned to the world frame**, only translated), and
cameras are **mean-pooled** (line ~180, so camera order/count are abstracted
away). The V2VNet's only geometric dependency is **world-frame orientation** --
it learns "dorsal is roughly +Z" for whatever calibration dominates training.
Bout 28's group is 32% of fly-samples once the second flies are recovered
(244 of the 264 land in group A), so this is a real but second-order risk, not
the collapse cause.

**This section supersedes the `v3-hybridnet-regression` note's root cause.**

---

## 2. Non-goals

- Fixing the STAC leg-chain. Phase 4 will show whether IK absorbs the 3D gain;
  if it does, that is a separate project.
- Retiring robust DLT from the production pipeline. This work produces a
  candidate; promotion is a later decision gated on the benchmark.
- Re-labeling. Every gain here comes from data already labeled.

---

## 3. Design

### Phase 0 -- Baseline capture

Run bout 28 through the current pipeline end-to-end; keep `kp2d.npz`,
`kp3d.npz`, `qpos`, and per-camera reprojection overlays for both flies. This
is the "before" for every later comparison.

### Phase 1 -- Dataset rebuild -> `red_data_3d_v5`

A fresh, correctly organized root. **The split is metadata, not directory
layout** -- the central fix, since today regenerating a split means moving
image files.

```
/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5/
├── manifest.json                 # source of truth: per recording -> source subset,
│                                 #   sex, behavior, calib_group, n_framesets, split
├── calibrations/
│   ├── calibA/Cam*.yaml          # deduplicated by content hash: 3 groups, not 21 copies
│   ├── calibB/…
│   └── calibC/…
├── images/<recording>/<Cam>/Frame_%06d.jpg     # symlinks, stored ONCE, no train/val dirs
├── masks/<recording>/<Cam>/Frame_%06d_fly<k>.npz
├── annotations/
│   ├── instances.json            # single source of truth; every ann carries fly_id/sex/behavior
│   ├── split.json                # recording+frame -> train|val; regenerable, moves zero files
│   ├── instances_train.json      # DERIVED from the two above
│   └── instances_val.json        # DERIVED
├── contact_sheets/<recording>.png
└── build_report.json
```

Content:
- **26 recordings** = union of `general_model` (21) and the V3-only extras (5).
  Nine recordings were never in 3D training: both amputee sets, all five
  headless, `wall_frames`, `20_04_female_climbing`.
- **3,536 framesets -> 3,800 fly-samples**: 2,847 from `general_model` + 689
  from the five V3-only recordings, plus the 264 recovered second flies. By
  calibration group the fly-samples are A 1,218 / B 2,552 / C 30.
- **Per-fly identity**: annotations and framesets carry `fly_id`, so both flies
  in the four two-fly recordings survive (fixes 1.4).
- **Calibrations deduplicated** by content hash into 3 groups, making the
  calibration-group split a first-class, queryable property.
- **`categories`** name corrected from `"Rat"` to fly.
- Images symlinked (source is stable on gscratch; saves 4.4 GB). `--copy`
  available.

**Split policy** (female data is the binding constraint -- only 199 confirmed
female framesets, 7% of the data):
- Male/general recordings: **whole-recording holdout**, zero leakage.
- The five female recordings: **all stay in training**; val is a guard-banded
  contiguous tail block (~10%, +/-50-frame guard).

Accepted limit: a within-recording female val measures "new pose, same fly and
session" and will read optimistic against bout 28, which is a new session. The
real female test is bout 28 under GT-free metrics plus rendered overlays.

**Sex recovery**: 54% of framesets are `sex: unknown`, including the three big
Group-A courtship recordings (V2/V3/V4 = 677 framesets in bout 28's exact
calibration). Sex is a per-recording property (per-fly-slot for the four
two-fly recordings), so ~30 judgements. The build emits one **contact-sheet
PNG per recording** -- the labeled fly with keypoints overlaid, several frames
x 2 cameras -- for a manual pass writing `sex` into `manifest.json`. No GUI; if
the sheets prove ambiguous we escalate.

**Figure gate:** the contact sheets themselves, plus a split-audit plot proving
zero train/val temporal overlap for whole-recording holdouts and >= 50-frame
separation for the female guard bands.

### Phase 1b -- SAM3 masks for the 9 new recordings

The detector is 4-channel (RGB + mask) and `sam3_masks/` covers only V3's 17
recordings. The 9 additions need a mask pass (~5,160 images). `mask_report.json`
records `extra_masks_total: 9657` -- second-fly masks already exist for the
two-fly recordings but were never matched to annotations; the rebuild must
match them by IoU against the annotation bbox.

**Figure gate:** mask overlays on `20_04_female_climbing` and `wall_frames`,
the two OOD recordings we are adding specifically for the female.

### Phase 2 -- ViTPose retrain with sharp targets

- ~~Target **sigma 7.0 -> 2.0** heatmap px (fixes 1.2; ~2.5 px is the
  anatomical scale of a tarsal segment, so 2.0 is deliberately just inside
  it).~~ **REVERSED 2026-08-30: sigma stays 7.0.** See the correction in 1.2 --
  sigma=2.0 measured 55.528 px val MPJPE against sigma=7.0's 6.448 px on
  identical data. `transforms.py` and `configs/train/vit2d.yaml` keep
  `target_sigma: 7.0`.
- Trained on the leakage-free split -- **this, not sigma, is what Phase 2
  delivers.** New data at sigma=7.0 with balanced sampling and `aug=default`
  reaches 5.915 px val MPJPE vs the shipped detector's 8.553 px (31% better),
  and 14.239 px vs 16.524 px on the female (14% better). `aug=heavy` was
  actively hurting the female (20.678 px) and is not used.
- **One 8-GPU data-parallel run**, like `v4_8gpu_20260808`. This is the
  critical path and wants all 32 CPUs feeding one process's JPEG decode.
- The sigma=7.0 control is free: `v4_8gpu_20260808` itself. Its val number is
  contaminated, but bout 28 was never in any training set, so it is a valid
  comparison **for the bout-28 figures only**, labeled as such.

**Figure gate:** heatmap peak concentration before/after (0.352 -> target
>0.7); 2D overlays on female wall/occlusion frames across >= 3 cameras.

### Phase 3 -- Coarse-to-fine 3D

- **Stage 1** -- V2VNet at 48^3, spacing 1. Exactly JARVIS: robust
  localization and cross-view disambiguation. Unchanged.
- **Stage 2** -- per-joint refinement volumes, **24^3 at spacing 0.25**,
  recentered on the stage-1 estimate. Gives a +/-3 unit capture window against
  a ~1 unit stage-1 error, and 1 voxel ~ 0.8 heatmap px -- correctly matched to
  the 2D resolution (fixes 1.1). Cost is 50 x 24^3 = 691k voxels vs stage 1's
  5.5M, i.e. **8x cheaper than stage 1**, not equal. Weights shared across
  joints (joint-as-batch).
- **Stage 3** -- continuous refine, post-hoc at inference: project the stage-2
  estimate into each camera, gate views by agreement, robust-DLT the survivors
  for sub-voxel output. Reuses the consensus code in
  `jarvis_jax/tracking/triangulate.py`. Applies to **every** arm, so it costs
  no arm.
- **Rotation augmentation** behind a flag, default bounded: 360 deg yaw about
  the gravity axis, +/-30 deg tilt. Rotate the grid basis and the GT labels by
  the same R. Justified less by calibration invariance (Group A is 24% of
  framesets) than as a free data multiplier at 3,150 framesets. Honest
  counter-argument: the world frame carries a real gravity prior ("legs point
  down"), so augmentation is expected to help the female on walls and cost a
  little on the easy male -- which is why A1/A2 and A3/A4 measure it.

**Eight parallel arms**, one GPU each, all on the rebuilt dataset and the
sigma=2.0 detector. Features are pre-cached, so host RAM per process is low.

| arm | config | isolates |
|---|---|---|
| A1 `base` | stage-1 48^3 sp1, no aug | **JARVIS reproduced on a clean split** -- the control |
| A2 `base+aug` | + rotation aug | rotation augmentation |
| A3 `c2f` | coarse-to-fine, no aug | the refinement stage |
| A4 `c2f+aug` | recommended config | -- |
| A5 `hires` | single-stage 96^3 sp0.5 | is coarse-to-fine better than simply going finer? |
| A6 `c2f+aug -recovered` | A4 minus the 264 second flies | the data recovery |
| A7 `c2f+aug +femwt` | female-upweighted sampling | the stated female goal, directly |
| A8 `c2f+aug seed2` | A4, different seed | **the noise floor** -- which deltas are real |

A1 matters most: stock JARVIS 3D has never been measured on a non-leaky split.
Whatever ships is compared against it honestly.

**Figure gate:** per-joint 3D error against voxel size (the 1.1 hypothesis,
falsifiable -- if error does not concentrate on the sub-2-voxel segments, the
resolution story is wrong); tarsal-tip and wing-vein 3D traces, baseline vs
new, female and male.

### Phase 4 -- bout 28 end-to-end and acceptance

Run the winning arm through STAC IK to final kinematics.

**Acceptance figures** (all with real keypoint/camera/fly names and units, per
`CLAUDE.md`):
1. Wing and leg joint-angle traces (deg vs frame), baseline vs new, **both
   flies** -- the actual goal.
2. Per-camera reprojection overlays on >= 3 cameras including an occlusion
   frame and a wall frame, female included.
3. Scorecard delta on the frozen 13-bout benchmark (`scripts/benchmark/`), so
   nothing else regresses.

**Stated expectation, to be written before the figures are generated:** if the
resolution hypothesis is right, tarsal-tip traces lose their ~0.17 mm
staircase quantization and leg-angle traces become smooth without added
lag; if it is wrong, traces stay equally jagged and the remaining error is
upstream (2D) or downstream (IK).

---

## 4. Code placement

New dataset builder and training entry points inside `jarvis_jax`, reusing the
verified-faithful ported `hybridnet/reproject.py` and `hybridnet/v2vnet.py`.
Not a fresh package -- a head-to-head against reference PyTorch HybridNet on
this very bout confirmed the port, and rewriting it spends budget for nothing.

- `jarvis_jax/data/build_v5.py` -- dataset builder (new)
- `jarvis_jax/data/v5_3d.py` -- per-fly frameset loader (replaces `v3_3d.py`)
- `jarvis_jax/hybridnet/refine.py` -- stage-2 per-joint refinement (new)
- `jarvis_jax/hybridnet/reproject.py` -- add `rotation` and `spacing` args
- `jarvis_jax/tracking/triangulate.py` -- stage-3 gated continuous refine
- `scripts/viz/contact_sheet.py` -- promoted, committed (regenerable figure)
- `scripts/viz/voxel_resolution.py` -- promoted, committed (the 1.1 gate)

Figures under `figures/2026-08-29-c2f-3d/`; small text artifacts (scorecards,
notes) under `docs/benchmark/2026-08-29-c2f-3d/`.

---

## 5. Testing

TDD throughout. Tests that must exist:

- **Split has zero leakage.** For whole-recording holdouts, train and val
  recording sets are disjoint; for female guard bands, min frame distance
  between train and val >= 50. This is the test that would have caught 1.3.
- **Second-fly recovery is exactly 264 framesets / 1673 annotations**, 244 of
  them in calibration group A, and each recovered fly triangulates below 2 px.
- **`reproject` parity**: with `rotation=I, spacing=1`, output is byte-identical
  to the current implementation. Guards the refactor.
- **Rotation equivariance**: reprojecting with rotation R and rotating the GT
  by R gives the same loss as the unrotated pair.
- **Stage-2 capture**: given a stage-1 estimate perturbed by up to 3 units, the
  true point lies inside the refinement volume.
- **Calibration grouping**: content hashing yields exactly 3 groups, with bout
  28's calibration in group A.

---

## 6. Risks

1. **Attribution across phases.** Phase 3's fan-out isolates every 3D choice,
   but Phase 1 (dataset) and Phase 2 (detector) land together. Partly covered
   by A6 (data recovery) and the sigma=7.0 checkpoint (bout-28 figures only).
   Every intermediate artifact is kept, so isolating a cause is a re-run of one
   stage, not a redesign.
2. **8-way JAX concurrency is a known failure mode here** -- a prior 8-way
   attempt on a 128 GB cgroup died with `CUDA_ERROR_UNKNOWN`; the working cap
   was ~4. We have 256 GB (32 GB/process) and cached features keep host RAM
   low. Mitigation: ramp 4 -> 8, set `XLA_PYTHON_CLIENT_MEM_FRACTION`, do not
   assume.
3. **A5 memory**: 96^3 x 50 joints ~ 177 MB/sample vs 22 MB today. May need
   batch 4 or bf16 on a 46 GB L40S. If it will not fit, A5 is dropped -- it is
   the least important arm.
4. **The IK may absorb the 3D gain** -- previously measured reprojecting 30-60
   px off the DLT 3D it consumed. Phase 4 shows this plainly if it happens; per
   scope, fixing it is a follow-up.
5. **Allocation walltime**: ~2d14h left of a 3-day job on `g3102`. Training
   auto-resumes via Orbax, so preemption is survivable, but phases are ordered
   so bout 28 gets a result before the wall.
6. **Rotation augmentation may cost the male.** A1/A2 and A3/A4 measure it; the
   flag makes it the cheapest thing to toggle first if results disappoint.

---

## 7. Provenance

Probes written during this design (scratchpad, outputs quoted above):
`chimera_probe.py` (refuted the chimera hypothesis; found the 1,673 dropped
annotations), `dropped_probe.py` (characterized them), `voxel_scale.py`
(segment lengths in voxels), `px_per_unit.py` (heatmap px per voxel).
Calibration hashing and split-leakage counts are inline `python3 -c` runs.
The two promoted scripts in section 4 make the load-bearing measurements
(1.1 and the contact sheets) reproducible.
