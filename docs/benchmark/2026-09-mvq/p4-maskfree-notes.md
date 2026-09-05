# P4a mask-free front end -- benchmark notes

## Centre-shift spike (2026-09-04)

Task 1 of `.superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b` (spec §6). The
mvq lifter (`mvq_t1_b16_p3a_20260904/final`, val loss/step: EMA at
`total_steps=10000`) was trained with window centres jittered by only
+-0.3mm (`jitter_units=3.0`, `MM_PER_UNIT=0.1`) around the host fly's own
labelled-3D bbox midpoint. The mask-free front end (P4b) will instead place
windows from a coarse track interpolated to 800fps, whose centre can be
~0.6mm off the true fly centre. This spike measures how the UNPROMPTED
typed-slot policy (`policy_instance(..., prompted=False, has_mask=False)` --
the real inference-time instance choice, no ground truth) degrades as the
window centre is displaced by a known, exact amount on the full val split
(153 windows), via the new `V12WindowDataset(..., center_shift_units=...)`
loader option (eval-mode only, deterministic per window, in-plane;
`jarvis_jax/data/v12_windows.py`).

Command:
```
PYTHONPATH=third_party/jarvis_jax:. CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  python scripts/benchmark/mvq_centre_shift.py \
  --run /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p3a_20260904/final
```

### Results (n=153 val windows at every shift)

| shift (mm) | policy MPJPE (mm) | vs unshifted | miss fraction |
|---:|---:|---:|---:|
| 0.0 | 0.0934 | -- (baseline) | 0.000 |
| 0.5 | 0.0970 | +3.9% | 0.0065 |
| 1.0 | 0.1084 | **+16.1%** | 0.0065 |
| 2.0 | 0.4166 | +346% (4.46x) | 0.0654 |
| 3.0 | 0.8715 | +833% (9.33x) | 0.2222 |

Decision lines (per the docstring EXPECTATION / §6 rule): error within 10% of
the unshifted value (0.1027mm) and miss fraction under 2% (0.02), both **up
to and including the 1mm (10 unit) shift** -- the magnitude the coarse-track
front end is expected to sit near.

### Figure reading

`figures/2026-09-mvq/p4_maskfree/centre_shift.png` (JSON alongside):
two panels, error (mm, left) and miss fraction (right) vs centre shift (mm),
each with its decision line dashed in gray, title naming the checkpoint
(`mvq_t1_b16_p3a_20260904`, `step=final(total_steps=10000)`).

- **Left (error):** flat from 0 to 0.5mm, then the point at 1.0mm sits
  visibly just above the 1.1x dashed line -- the numeric check above confirms
  it (+16.1%, over the +10% allowance). From 1mm to 2mm the curve turns into
  a steep, near-linear climb (0.108mm -> 0.417mm, +0.31mm for the next 1mm of
  shift alone) and keeps climbing to 0.87mm at 3mm.
- **Right (miss fraction):** both the 0.5mm and 1.0mm points sit clearly
  below the 0.02 line (0.0065, no change between them at all -- the same
  10/153 windows miss at both). The curve crosses the 0.02 line between 1mm
  and 2mm and is at 0.065 (>3x the line) by 2mm, 0.222 by 3mm.

Both panels show the same shape: near-flat through 0.5mm, a borderline miss
of the error line right at 1mm, then a cliff starting immediately past 1mm.
This is the failure signature the docstring's EXPECTATION describes as
"climbs steeply" near the 1mm boundary, not a gradual degradation that stays
inside tolerance out to 3mm.

### Decision (§6 rule)

**Retrain needed: YES.** At the boundary shift the coarse-track front end is
expected to operate near (~0.6mm nominal, with 1mm named in the spec as the
shift the retrain should cover), the policy MPJPE already exceeds the 10%
tolerance (+16.1% at 1mm vs the +10% allowance, 0.1084mm vs the 0.1027mm
line) even though the miss fraction is still comfortably under 2% (0.65%) at
that point. Immediately beyond 1mm the degradation is not a soft margin but
a cliff: by 2mm the error is 4.46x the unshifted value and the miss fraction
(6.5%) is already more than 3x over its own 2% line; by 3mm error is 9.33x
and 22% of windows return no instance at all. A model trained on +-0.3mm
jitter has essentially no headroom past a 1mm centre error, which is exactly
the displacement the mask-free coarse track can produce -- the §6 retrain
with 1mm jitter (`jitter_units` an order of magnitude larger than today's
3.0) is warranted before Task 4 runs the mask-free front end on real
interpolated coarse-track centres. Per the task brief: STOP after Task 1 and
report this; Tasks 2-3 (which do not depend on the retrained checkpoint) can
proceed, and the controller should schedule the retrain before Task 4.

## Window cost (2026-09-04)

Task 2 of `.superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b` (spec §7). One
batched, single-pass `MVQRunner.infer` on 32 real windows built from ONE
synced frame set of bout 28 (`Session0/2025_10_20_13_20_04`, frame 446975,
all 7 cameras present), centres jittered +-3mm around fly0's mvq centroid on
that frame; 2 warm-ups then 20 timed iterations
(`scripts/benchmark/mvq_window_cost.py`, JSON beside the figure dir). The
checkpoint is `mvq_t1_b16_p3a_20260904/final`, its own `attn_impl: cudnn`,
fp32 crops (uint8 in, `normalize_crops` to fp32 -- the same dtype path
training used).

| GPU | batch of 32 | infer | + window build (CPU crop) | total | §7 decision |
|---|---:|---:|---:|---:|---|
| **L40S** (g3106, the acceptance card) | 1148 +- 27 ms | 35.9 ms/window | 0.6 | **36.4 ms/window** | **stride 1** |
| A40 (g3051, first run) | 2094 +- 40 ms | 65.4 ms/window | 1.7 | 67.2 ms/window | stride 2 |

**Decision: the fine pass runs at STRIDE 1.** The rule in §7 is stride 1 at or
below 60 ms/window, and the acceptance criterion is "under 3 GPU-hours per
recording on one L40S" -- on that card the measured cost is 36.4 ms/window,
below the 40 ms the §7 budget assumed. Scaled to a recording (498k frames):
coarse at stride 16 is ~31k frames x ~1.5 windows ~= 28 min, the fine pass
~30 bouts x 2000 frames x ~1.5 windows ~= 55 min, so ~1.4 GPU-h of mvq plus
CenterDetect -- inside the 3 GPU-hour acceptance with room to spare.

The first measurement landed on an **A40** (submit_task.sh's constraint list
is `h200|a100|l40s|l40|a40`, and ckpt-all gave it an A40), where the same
code costs 1.85x as much and would flip the decision to stride 2. The number
is therefore card-dependent and only the L40S row answers the spec's
question; a run of the mask-free front end scheduled onto an A40 should
either use stride 2 or expect ~2.6 GPU-h of mvq. Pin the card
(`--constraint=l40s`) when re-measuring.

Not measured: bf16 crops. cuDNN flash attention accepted the fp32 path
without complaint (the same path training used), so there was no forcing
reason to change the checkpoint's numerics for a timing run; if the fine
pass ever needs more headroom, a bf16 arm is the first thing to try and it
is a numerics change that has to be re-validated against the P3a gates, not
a flag.

Commands:
```
scripts/slurm/submit_task.sh --time 0:30:00 --mem 32 mvq_wincost \
  "export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 HF_TOKEN= \
   PYTHONPATH=third_party/jarvis_jax:. && \
   python scripts/benchmark/mvq_window_cost.py \
     --run /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p3a_20260904/final \
     --session-dir /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04 \
     --frame 446975"
# and the same command in an sbatch with --constraint=l40s (job 39595689)
```
Artifacts: `figures/2026-09-mvq/p4_maskfree/window_cost.json` (A40) and
`figures/2026-09-mvq/p4_maskfree/l40s/window_cost.json` (L40S).

### Bout-video refactor equivalence (same task)

`scripts/viz/mvq_bout_video.py` now goes through `MVQRunner`. Pre- vs
post-refactor on bout 28, `--n 4 --attn_impl xla` on CPU, checkpoint
`mvq_t1_b16_p3a_20260904/final` (the reference had to be regenerated: the
earlier smoke used `--step latest` = step 7000, and orbax has since kept
only steps 8000-10000, so the pre-refactor script from git `HEAD` was re-run
at `final/` for the comparison): **max |difference| = 0.0 exactly** on
`kp3d`, `kp2d` and the per-view `conf`, for both flies in both the prompted
and unprompted modes, with identical NaN patterns and an identical
`per_frame` block (slot, fallback, has_mask, exist, sex_prob) in
`mvq_meta.json`.

One deliberate, documented change to the artifacts: `kp3d.npz`'s `conf3d` is
now the pipeline-facing confidence (the per-view visibility sigmoid averaged
over cameras, spec §2) instead of the raw mvq confidence head, which is
preserved unchanged beside it as `conf3d_mvq_raw` (verified equal to the old
`conf3d` to 0.0, and the new `conf3d` equal to the old `conf`'s mean over
cameras to 0.0). `kp3d.npz` also gained the Stage-B `gates` string. The
keypoint order of this script's own files is unchanged (mvq order --
`to_pipeline(..., model_names=mvq_names)` is the identity permutation here;
the fine pass is what writes `cfg.model.KP_NAMES` order).

### Centre-shift curve on the jitter retrain, step 3000 (interim; ckpt-all job 39604340)

`scripts/benchmark/mvq_centre_shift.py --run .../mvq_t1_b16_p4_jitter10_20260904 --step 3000` (val, 153 windows, unprompted policy):

| shift (mm) | P3a final: MPJPE mm / miss | jitter-10 @3000: MPJPE mm / miss |
|---|---|---|
| 0.0 | 0.093 / 0.0 % | 0.083 / 0.0 % |
| 0.5 | 0.097 / 0.7 % | 0.094 / 0.0 % |
| 1.0 | 0.108 / 0.7 % | 0.102 / 0.0 % |
| 2.0 | 0.417 / 6.5 % | 0.362 / 0.7 % |
| 3.0 | 0.872 / 22 % | 0.663 / 0.0 % |

Reading (controller): after 3000 steps of 1 mm jitter the MISS behaviour is already robust (0.7 % at 2 mm, none at
3 mm, vs 6.5 % / 22 %) and the absolute error at 1 mm (0.102) is below the P3a model's centred error, but relative
to its own centred value (+23 % at 1 mm) the §6 "within 10 %" rule is not yet met; the 2 mm cliff persists because
the training jitter is +-10 units per axis (radial reach ~1.4 mm), so 2 mm is outside the trained range by design.
Re-run on the final checkpoint decides; if still over the rule at 1 mm, the follow-up is jitter 15-20 units or a
tighter coarse placement (stride 8), not a redesign.

## Coarse pass (task 4, spec §4.3) -- 2026-09-04

`jarvis_jax/tracking/coarse_track.py` + `scripts/coarse_pass_mvq.py`: the
mask-free replacement for `scripts/coarse_pass.py`. Writes
`coarse_tracks.npz` in the SAM3 coarse-pass schema plus the mvq fields, so
`scripts/coarse_pass_gates.py::load_tracks` opens either file (asserted in
`tests/test_coarse_track.py`, which runs the gates' own
`compute_gate_signals`/`apply_gates` on an mvq file).

### Coarse pass timing

| stage | wall clock | notes |
|---|---|---|
| model load (mvq + CenterDetect) | PENDING | printed as `[coarse] models loaded in Xs` |
| coarse pass, 20_04 @ stride 16 | PENDING | printed as `[coarse] done: N coarse frames in X min` |

**Real run status: NOT RUN.** The retrained checkpoint
`mvq_t1_b16_p4_jitter10_20260904/final` did not exist at the end of this task
(training was at step 4000 of its schedule and still held all 8 GPUs at
~41 GB each), and the brief forbids waiting on it. Command to run, once the
`final/` appears and the GPUs are free:

```
module load cuda/12.9.1
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset LD_LIBRARY_PATH JAX_PLATFORMS
export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 HF_TOKEN= \
       CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
       PYTHONPATH=third_party/jarvis_jax:.
python scripts/coarse_pass_mvq.py \
  --session-dir /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04 \
  --calib-dir   /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04/calibration \
  --run  /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p4_jitter10_20260904/final \
  --centerdetect /gscratch/portia/eabe/data/Johnson_lab/jax_centerdetect_runs/cd_focal_bg30/ckpt/epoch_004 \
  --stride 16 --resume \
  --out /gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/coarse_mvq/coarse_tracks.npz
```

`--resume` is safe to repeat: the pass writes a complete, gates-readable
`coarse_tracks.partial.npz` every `--partial-every` (default 2000) coarse
frames and continues after its last frame (restoring the last detected/reused
window centres, `last_centres`, so a blank frame right after the resume
boundary still reuses them instead of going NaN), so the run can be split
across several foreground calls without redoing work. The floor plane and
every feature are recomputed over the whole concatenated track on each write,
so a resumed run and a single-shot run have IDENTICAL FRAME COVERAGE AND
CENTRE REUSE -- not a byte-identical file: earlier chunks' `kp3d` is read back
from its `float16` on-disk storage, so `wing_angle_deg` (and anything else
derived from `kp3d`) on a resumed chunk is recomputed from that f16-quantised
value, an inherent ~0.05 mm rounding difference from a true single-shot run.

### Deviation: the floor-plane sign rule (figure-gated)

The plan's wording was "least-squares plane on the first 2000 finite coarse
centroids, sign chosen so the median fly height is positive". Implemented
literally that is not just under-determined, it is BACKWARDS on real data,
and the A/B figure says so:

* a least-squares plane runs through the MEAN of the points, so "height above
  it" measures the mean height of the FLIES, not the glass, and the median
  residual is ~0 by construction;
* flies rest on the floor and occasionally climb, so the cloud is
  bottom-heavy and that median residual is NEGATIVE -- the rule then flips
  the normal DOWNWARD, and every height reads mirrored.

Measured on a synthetic arena (90 % of centroids within 0.3 mm of the glass,
10 % up a wall at 1-4 mm, plane tilted ~11 deg):

| | dot(normal, true up) | median height | max abs height error |
|---|---|---|---|
| literal rule | **-1.000** | +2.25 units | **75.8 units (7.6 mm)** |
| shipped `fit_floor` | +1.000 | +1.69 units | 0.36 units (0.036 mm) |

The literal rule's fitted heights lie on a slope **-1** line against the
truth: a fly 4 mm up a wall reads as 3.6 mm BELOW the floor. Figure (and its
generating script + npz): `figures/2026-09-04-mvq-coarse-floor/floor_sign_ab.png`.

What is kept: the least-squares fit over the first 2000 finite centroids, and
positive fly heights. What is fixed: "up" is the direction the cloud is
bottom-heavy in (mean along the normal above the median), and the offset sits
at the 3rd percentile of the along-normal coordinate, so the plane is the
floor the flies stand on rather than their average altitude. Guarded by
`test_fit_floor_recovers_a_known_tilted_plane_with_positive_heights`, which
also checks the mirrored arena.

**Fix round 1 robustness addendum.** Two gaps in the first cut: (1) every
finite centroid was fit with no confidence gate, so low-`exist` rows (a typed
slot barely firing, or firing on background) counted as real geometry; (2) a
single total-least-squares SVD over floor+wall points together tilts the
normal toward the wall in proportion to the wall's point FRACTION (every
point gets equal SVD weight, wall points are the ones farthest from
co-planar), so the fit's accuracy was one occlusion-heavy recording away from
degrading. Fixed: `fit_floor` now takes an `exist` argument and fits only
`exist >= 0.5` centroids (`coarse_pass_mvq.py` passes `tr["exist"]`); after
the first SVD, a SECOND SVD refits the normal on only the bottom 50 % of
points by that first pass's height, which is dominated by the true floor
regardless of the wall fraction; the offset moved from the 1st to the 3rd
percentile of the (refit) heights, a little less exposed to a single
below-floor outlier. Guarded by
`test_fit_floor_ignores_low_exist_points_and_refits_normal_on_lowest_quantile`
(30 % wall points plus a handful of `exist=0.1` sub-floor outliers; normal
recovered within 2 deg, offset within 1 unit of the true floor).

### Signals that change meaning in an mvq file, before §5 (bout detection)

* **`area` is all-NaN in an mvq file** (there are no masks). The SAM3 gates'
  area-ratio test therefore can never pass on one -- the file OPENS and every
  shape is right, but `apply_gates` returns an empty bout table. That is
  asserted in the test so it is discovered here and not as a mysteriously
  empty CSV. §5's detector uses the mvq features (`dist`, `wing_angle_deg`,
  `speed`, `heading_deg`, `height`, `exist`) instead.
* **`valid`/`n_valid_cams` mean "the reprojected 3D centroid lands inside
  camera c's image", not "camera c saw the fly".** There is exactly one 3D
  point (the typed slot's keypoint mean) reprojected to every camera -- no
  per-camera detection or mask -- so `n_valid_cams >= 3` is NOT an occlusion
  signal here the way it can be read off a SAM3 file's mask-derived `valid`.
  A camera can show `valid=True` while the fly is fully behind a wall in that
  view, as long as the 3D point still projects inside the frame.
* **`in_frame` is 0/1 ONLY.** SAM3's third state, `IN_FRAME_UNKNOWN = 2`
  ("< 2 other valid cameras -- position not determinable", `sam3_driver.py`),
  is collapsed to `0` (`IN_FRAME_NO`) in an mvq file, because there is one 3D
  point and no notion of "not enough OTHER cameras" to be unsure against. The
  fine pass's gap-repair idiom, which runs on runs of `IN_FRAME_YES`/
  `IN_FRAME_NO` codes, must NOT be applied unchanged to an mvq file -- it will
  read every out-of-frame case as a confident "no", never "unknown".
* **`heading_deg` is measured from the male's ANTERIOR axis** (Abd_tip ->
  Scutellum), so 0 deg = the male is pointed straight at the female. The wing
  angle uses the posterior axis (Scutellum -> Abd_tip) because that is the
  axis a wing is held relative to. Both are documented in
  `coarse_features`; a heading whose zero meant "facing away" would be read
  backwards by every downstream gate.
