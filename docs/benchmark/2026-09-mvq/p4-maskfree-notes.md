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
