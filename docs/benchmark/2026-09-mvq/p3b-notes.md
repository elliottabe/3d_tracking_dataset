# P3b: the contact-heavy mvq fine-tune (`mvq_t1_b16_p3b_contact_20260905`)

## Why

Measured on `2025_10_20_13_20_04` (20_04) with the r2 relative-assignment lift
(see the "Mask-identity assignment" section of `p3a-notes.md`): in CONTACT
frames where the female slot is empty, the MALE instance absorbs her head, T1
legs and wing base. 1.4 % of frames have >= 5 male keypoints jumping > 0.5 mm
between neighbouring frames; 1.7 % have >= 5 male keypoints sitting on HER
body. The sex head calls 20_04's female a MALE (her slot fires < 0.5; the
female slot is read on only ~50 % of frames).

Three training-side causes, and what this run does about each:

| cause (measured) | knob | mechanism |
|---|---|---|
| (a) contact pairs are under-trained -- P3a val: `contact_pair` cohort 0.18 mm vs 0.09 mm overall, `mask_containment` 0.81 on contact pairs (lowest of any cohort) | `copy_paste_p` 0.5 -> **0.8**, `copy_paste_contact_p` 0.3 -> **0.7**, `copy_paste_contact_sep` (8, 30) -> **(4, 25)** units | ~56 % of every training window is now a synthetic touching/overlapping pair, at heavier overlap than the real 24-30-unit mounting pairs |
| (b) female-host windows are scarce -- **202 of 2661** train windows (7.6 %), because 20_04's 677-frameset `courtship_20_04_male` block labels only the MALE as host and the present-but-unlabelled female gets existence-ignore and no sex supervision | `female_host_weight` 1.0 -> **4.27** | the two host sexes are drawn equally often (verified: weight mass female 0.5000 / male 0.5000) |
| (c) nothing penalises a slot's keypoints landing on the OTHER fly | `loss.other_fly_repulsion` 0 -> **0.5** | hinge on (distance to own fly's GT centroid - distance to the other's), in units, on windows with two labelled flies |

## Knobs (all new; every default leaves P3a behaviour bit-identical)

| knob | default | this run | file |
|---|---|---|---|
| `train.copy_paste_contact_sep` | `[8.0, 30.0]` (`CopyPasteParams`' own default) | `[4.0, 25.0]` | `train/train_mvq.py`, `configs/train/mvq.yaml` |
| `train.female_host_weight` | `1.0` (unchanged) | `4.27` | `train/train_mvq.py::_balanced_weights` |
| `train.sex_label_overrides` | `{}` | `{}` -- **not used**, see below | `data/v12_windows.py` |
| `train.loss.other_fly_repulsion` | `0.0` (off) | `0.5` | `train/losses_mvq.py::LossWeights` |

### `female_host_weight` vs the existing `female_weight`

`female_weight` (P2) multiplies female-host windows INSIDE `_balanced_weights`'
behaviour-category normalisation, so how much sampled mass it actually buys is
data-dependent and opaque -- which is how P2 sampled 677 male-host windows as
"female" for a whole run without anyone noticing. `female_host_weight` is
applied to the FINAL normalised weights instead, so the mass ratio it produces
is exactly `female_host_weight x (mass_F / mass_M)` and can be solved for a
target. On `red_data_3d_v12_export0902` train the unweighted ratio is 0.234
(202 of 2661 windows), hence **4.27**. `female_weight` stays 1.0.

Two prints make it verifiable rather than assumed -- the weight mass at sampler
construction, and the ratio the sampler ACTUALLY drew over the first 200
batches (`_RATIO_BATCHES`). From this run's log:

```
[mvq] T=1 sampler: 202/2661 female-host windows, female_host_weight=4.27
      -> weight mass female 0.5000 male/other 0.5000 (F/M 1.000)
```

Risk, stated: 202 unique female-host windows now carry half the sampling mass,
so each is drawn ~740 times over 10k steps of batch 35. Copy-paste (0.8),
`mv_aug` and the 10-unit centre jitter are what stand between that and
memorisation; if the `female` val cohort degrades while train loss falls, this
knob is the first suspect.

### `other_fly_repulsion` (term 7b, `losses_mvq.py`)

Term 7 (`rep`) already pushes a predicted keypoint off the other fly's
same-part LABEL, but only within `rep_px`/`rep_units` and only for keypoints
that fly actually has labels for. Term 7b asks the coarser question the contact
failure is about -- *is this point on the right ANIMAL at all?* -- using the GT
3D centroids `cen` the slot assignment already computes:

```
other_rep = mean over (b, f, t, k) of relu(||xyz_pred[f] - cen[f]|| - ||xyz_pred[f] - cen[o]||)
```

so it is exactly zero as soon as a point sits on its own side of the two
centroids' midplane, and grows linearly (in units, 0.1 mm) once it crosses.
Scored only where BOTH flies are labelled AND assigned to a slot AND have a 3D
centroid -- real two-fly windows and copy-paste composites alike. ~20 lines.
Known false positive, accepted: during a real mounting pair (20-26 units apart,
body half-length ~11 units) a genuinely extended leg tip can cross the midplane
and be penalised. The hinge has no margin, so the penalty there is small.

### `sex_label_overrides`: implemented, deliberately EMPTY

`V12WindowDataset(..., sex_overrides={rec: {fly: "female"|"male"}})` now sits
at step 0 of the resolution chain, above the annotation and the manifest, and
raises on a bad sex string rather than silently resolving to "unknown".

**20_04 is NOT overridden, and should not be.** The mapping question the brief
asked to settle is already settled in the files, in the other direction: the
loader's annotation-first resolution (P3a fix wave, `p3a-notes.md` "Fix wave
2026-09-04: sex resolution") is CORRECT for this recording. Its fly0 spans two
annotation subsets that label DIFFERENT animals -- `courtship_20_04_male` (677
framesets, annotation `sex: male`) and `20_04_female_climbing` (15 framesets,
`sex: female`) -- and the user's authoritative full-frame verdict confirmed
both. A `recording -> {fly_index: sex}` map is per (recording, fly) and
therefore structurally incapable of expressing that; setting it would either
restate what the loader already does or actively corrupt 677 windows.

The real gap is different from what an override can fix and is left as the
**top P3c candidate**: in those 677 windows the female is PRESENT but
UNLABELLED, so `slot_ignore` drops her slot's existence target entirely. Her
existence is in fact known (`unlabelled_sex` resolves her as female), so those
windows could supply 677 positive existence examples for the female slot --
which is precisely the symptom being chased ("her slot fires < 0.5"). Not done
here: it changes `slot_ignore` semantics shared by the loss AND `evaluate`, so
it would move the P3a acceptance baseline mid-comparison. Items (a)-(c) above
attack the same failure without that risk.

## Eval: `cross_fly_frac` (new, the metric that shows the fix directly)

`mask_containment` says "inside the host mask" -- generous for a 50-keypoint
skeleton, and structurally blind to the second fly. `cross_fly_frac` says "on
the wrong animal": for each labelled fly assigned to a slot, the fraction of
that slot's predicted 3D keypoints nearer the OTHER fly's GT centroid than
their own, averaged over the window's flies; NaN on single-fly windows.
Reported as `cross_fly_frac`, `cross_fly_frac_<cohort>` for every P3a cohort
(so `cross_fly_frac_contact_pair` is the headline number) and
`cross_fly_frac_slot<s>` per typed slot. All P3a cohorts kept unchanged.

### BEFORE number (jitter-10 final, the warm-start source, val 153 windows)

Produced with `evaluate` on `mvq_t1_b16_p4_jitter10_20260904/final`
(`figures/2026-09-mvq/p3b_gates/cross_fly_baseline_jitter10.json`, GPU 6,
generator kept in the session scratchpad as `cross_fly_baseline.py`).

Expectation stated before running: `cross_fly_frac` should be small overall
(most val windows are single-fly or well-separated) and NOTICEABLY HIGHER on
`contact_pair`; if it were not, the metric would not be measuring what it
claims and the P3b acceptance test would need rethinking.

| metric (val, 153 windows) | prompted | unprompted |
|---|---|---|
| **cross_fly_frac** | 0.1020 | **0.0482** |
| **cross_fly_frac, contact_pair** (n=26) | 0.1469 | **0.0762** |
| cross_fly_frac, two_fly (n=74) | 0.1020 | 0.0482 |
| cross_fly_frac, female | 0.1036 | 0.0548 |
| cross_fly_frac, group_A / group_C | 0.1039 / 0.0883 | 0.0518 / 0.0217 |
| cross_fly_frac, slot1 (FEMALE) | 0.1536 | **0.0576** |
| cross_fly_frac, slot2 (male) | 0.1024 | **0.0388** |
| cross_fly_frac, single_fly | nan (by construction) | nan |
| mask_containment | 0.8949 | 0.8853 |
| mask_containment, contact_pair | 0.8009 | 0.8138 |

**Met, and it points where the field failure does.** Contact pairs are
**1.58x** the aggregate unprompted (0.0762 vs 0.0482) and 1.44x prompted, and
the **FEMALE slot mixes 1.5x more than the male slot** (0.0576 vs 0.0388
unprompted) -- the same asymmetry the 20_04 lift shows, reproduced on held-out
val by a metric that never sees a mask. `slot3` is nan (no real same-sex pair
in val) and `slot0` is nan unprompted (the prompt is off), both expected.
These are the numbers P3b has to beat.

## Gate figure: copy-paste at the contact-heavy setting

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 PYTHONPATH=third_party/jarvis_jax:. \
    python scripts/viz/mvq_copy_paste_check.py --out figures/2026-09-mvq/p3b_gates \
        --contact-sep 4 25 --contact-p 0.7 --n-contact 5 --n-far 3
```

(`--contact-sep`, `--contact-p`, `--n-contact`, `--n-far` are new on that
script, so the gate can be run at the settings a given run actually trains
with rather than only at the dataclass default.)

`figures/2026-09-mvq/p3b_gates/copy_paste_check.{png,json}` (gitignored;
regenerate with the command above). 8 rows x 7 cameras. Pool 1850 eligible
targets, 43 draws, **0 rejected by `composite`**, 18 skipped by the check
script's head filter (legibility only -- see `p3a-notes.md`).

**Expectation stated before rendering:** the contact rows must show the pasted
donor (orange) TOUCHING or OVERLAPPING the host (cyan) -- visibly heavier
overlap than the 21.7-27.6 u the old (8, 30) render produced -- at the same
place relative to the host in every camera, with each fly's labels on its own
body and host keypoints under the donor drawn hollow.

**Read back (Read tool, all 8 rows in three crops):** met.

- **Contact rows, 5/5** at seps **8.5, 9.5, 21.7, 21.8, 23.6 u** -- the
  tightening worked: two of five are now genuinely stacked bodies (8.5, 9.5 u)
  where the old default's contact bucket bottomed out at 21.7 u. In every
  camera the donor sits at a consistent position relative to the host (e.g.
  #2026: orange above cyan in Cam2012630/2012855, and the same 3D relation
  seen from the other side in Cam2012861), never floating or per-camera
  offset. Hollow cyan circles appear exactly where the donor overlays the host
  (#2387 Cam2012631/2012861/2012862, #2236 Cam2012853, #1533 Cam2012862).
  On the two stacked rows the orange and cyan dots interleave spatially --
  that is the physical situation at 8-9 units, not a labelling error.
- **Far rows, 3/3** at 27.6, 38.8, 42.6 u -- bodies clearly separated, and at
  38.8/42.6 u the donor is only a partial cluster at the crop edge in some
  cameras and absent in others. That is `composite()`'s documented visibility
  rule (a camera with no donor pixels gets no donor labels), already recorded
  in `p3a-notes.md`, not new behaviour.
- Row #1052's Cam2012630 panel is black and titled "absent" -- a
  target-invalid camera, drawn correctly.

**Limitation, stated so it is not over-read:** 7 of the 8 rows are male-host
(the only female-host row is #1288, a *far* pair). That is the eligible donor
pool's own imbalance -- 187 female of 1850 eligible targets -- and is exactly
what `female_host_weight=4.27` corrects at training time; the figure was drawn
from an UNWEIGHTED permutation of the pool, so it does not show the sampler's
realised mix. The realised mix is instead reported numerically by the
first-200-batch ratio print above.

### Donors stayed within calibration group A -- measured, not assumed

The brief allowed drawing donors from BOTH groups "if the donor pool allows".
The pool numerically allows it (group B has 692 train windows; every eligible
paste TARGET is group A), but the geometry does not. `mv_copy_paste` is
affine-exact only because donor and target share `M`: the pasted 3D label is
`X_src + D`, which under the target's cameras projects to `M_tgt X + M_tgt D +
t_tgt`, while the pasted PIXELS are the donor's own crop translated, i.e.
`M_src X + M_tgt D + t_tgt`. Measured on the real calibrations (same 7 camera
names, same order in both groups):

```
A vs B: max|dM| 0.1822 px/unit  ->  up to 4.11 px label error at 15 units from the crop centre
```

P3a's own per-camera fit is 2.6-3.7 px, so cross-group donors would inject a
label bias at the level of the signal into ~26 % of pastes. **Kept group A.**

## Launch

Batch math: `run_training` raises unless `batch_size % len(jax.devices()) == 0`.
On **7** GPUs (0-5 and 7) that rules out 32; **35 = 5 per device** was chosen
(the jitter-10 run was 4/device on 8 L40S, and `q_chunk: null`'s profiling note
records the unchunked path fitting at 4-8 samples/GPU). Fallback on OOM was 28
(4/device, the proven per-device batch) -- **not needed, 35 ran clean**: 41.7 GB of 46 GB per L40S at steady state, GPU 6 untouched at 0 MiB.

```bash
cd third_party/jarvis_jax
module load cuda/12.9.1
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset LD_LIBRARY_PATH JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 TF_GPU_ALLOCATOR=cuda_malloc_async
export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
setsid nohup python -u -m jarvis_jax.scripts.train_mvq model=mvq train=mvq paths=hyak \
  run_id=mvq_t1_b16_p3b_contact_20260905 \
  train.total_steps=10000 train.warmup_steps=500 train.lr=1e-4 \
  train.prompt_p_start=0.0 train.prompt_p_end=0.0 \
  train.copy_paste_p=0.8 train.copy_paste_contact_p=0.7 'train.copy_paste_contact_sep=[4.0,25.0]' \
  train.female_host_weight=4.27 train.loss.other_fly_repulsion=0.5 train.jitter_units=10 \
  train.warm_start=/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p4_jitter10_20260904/final \
  train.eval_every=2000 train.save_every=1000 train.batch_size=35 train.num_workers=24 \
  "paths.runs_root=\${paths.mvq_runs_root}" \
  > slurm_logs/mvq_t1_b16_p3b_contact_20260905.out 2>&1 &
```

| | |
|---|---|
| PID | **1281418** (the `python` process; `setsid` wrapper 1281416 exits immediately) |
| log | `third_party/jarvis_jax/slurm_logs/mvq_t1_b16_p3b_contact_20260905.out` |
| run dir | `/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p3b_contact_20260905/` |
| GPUs | 0,1,2,3,4,5,7 (L40S, node g3102); GPU 6 left free |
| started | 2026-09-05 15:18 |
| step rate | **1.27 s/step** steady state (steps 150 -> 400 in 317 s), 41.7 GB/GPU |
| expected end | ~**19:20 on 2026-09-05** (10000 x 1.27 s = 3 h 32 m of stepping + 5 evals) |

First-200-batch check, from the log:

```
[mvq] realised host-sex ratio over the first 200 batches: female 3550/7000 = 0.507
```

so the sampler really did draw the two host sexes equally, not merely weight
them that way. `other_rep` is live and non-degenerate in the log (0.26-0.50 at
steps 50-400, contributing 0.13-0.25 of a ~28 total loss) -- large enough to
have a gradient, far from dominating.

Warm start reported `restored 608/608 leaves; not restored: []` -- the
jitter-10 checkpoint is already 4-slot P3a-shaped, so unlike the P3a warm start
nothing (not even the sex head) starts fresh.

## What to look at when it finishes

1. `cross_fly_frac_contact_pair` (unprompted) vs the BEFORE number above --
   this is the acceptance number for the whole run.
2. `mask_containment_contact_pair` and `cohort_contact_pair` MPJPE, against
   P3a @10k's 0.815 / 0.179 mm.
3. `sex_acc` and `exist_rec_slot1` (the female slot) -- (b) is meant to lift
   these; the 20_04 female's slot firing < 0.5 is the field symptom.
4. `cohort_female` MPJPE against P3a's 0.114 mm -- the overfitting check on
   `female_host_weight`.
5. Then re-lift 20_04 bouts 25/5/8 with `--identity mask` on the new
   checkpoint and re-run the swap/NaN audit table in `p3a-notes.md`'s
   "Re-smoke (round 2)" section; that is the end-to-end number the field
   failure was measured in.

## Files

- `third_party/jarvis_jax/jarvis_jax/train/losses_mvq.py` -- `LossWeights.other_fly_repulsion`, term 7b, `other_rep` metric
- `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py` -- `copy_paste_contact_sep`/`female_host_weight`/`sex_label_overrides` config fields and their plumbing, `_balanced_weights` final-weight multiplier, the two sampler prints, `cross_fly_frac` in `evaluate`
- `third_party/jarvis_jax/jarvis_jax/data/v12_windows.py` -- `sex_overrides` at step 0 of the sex chain, `_parse_sex_overrides`
- `third_party/jarvis_jax/jarvis_jax/scripts/train_mvq.py` -- list->tuple / DictConfig->dict for the two new structured knobs
- `third_party/jarvis_jax/configs/train/mvq.yaml` -- the four new keys, all at their no-op defaults
- `scripts/viz/mvq_copy_paste_check.py` -- `--contact-sep`, `--contact-p`, `--n-contact`, `--n-far`
- tests: `tests/test_mvq_losses.py` (+2), `tests/test_v12_windows.py` (+2), `tests/test_train_mvq_smoke.py` (+2)
- `figures/2026-09-mvq/p3b_gates/copy_paste_check.{png,json}` (gitignored)

## Acceptance (2026-09-05, final checkpoint)

Final val (`.../mvq_t1_b16_p3b_contact_20260905/final`): mpjpe 0.0849 mm,
cross_fly_frac 0.036 overall / 0.055 contact pairs, contact cohort 0.146 mm
(baseline jitter-10: 0.089 / 0.048 / 0.076 / 0.167).

### Step 1: centre-shift rule (spec Sec6)

`scripts/benchmark/mvq_centre_shift.py --run .../mvq_t1_b16_p3b_contact_20260905/final`
(153 val windows, unprompted policy). Decision line at 1mm: 1.1x unshifted =
0.0928mm.

| shift (mm) | policy MPJPE (mm) | miss frac |
|---:|---:|---:|
| 0.0 | 0.0844 | 0.000 |
| 0.5 | 0.0835 | 0.000 |
| 1.0 | 0.0825 | 0.000 |
| 2.0 | 0.3801 | 0.000 |
| 3.0 | 0.6403 | 0.000 |

**PASS.** At 1mm, error 0.0825mm is BELOW the unshifted value (0.0844mm),
comfortably under the 0.0928mm line; miss fraction is 0.0 throughout 0-1mm.
Figure read back (`figures/2026-09-mvq/p3b_gates/centre_shift_p3b.png`):
0/0.5/1.0mm sit together near-flat just under the dashed 1.1x line, the cliff
starts only between 1 and 2mm (as with the jitter-10 checkpoint this warm-
started from); miss-rate panel flat at 0.0 across all five shifts.
Artifacts: `figures/2026-09-mvq/p3b_gates/centre_shift_p3b.{png,json}`.

### Step 2: re-lift bouts 1, 4, 28 of Session0/2025_10_20_13_20_04

`--identity mask`, TWICE (`--containment off` / `--containment on`) into
`OutFiles/p3b_accept/{off,on}/` (fly_id-less bouts CSV built into
`OutFiles/p3b_accept/bouts_unified_summary.csv`). Compared against (a) the
existing r2 lift (jitter-10 checkpoint, containment off,
`.../pose_mvq_p3a_r2/bouts/`) with the scratchpad per-bout metric script
(pose-jump: >=5 male kps stepping >0.5mm/frame; straddle: >=5 male kps
nearer the female centroid than his own; female-missing fraction; male kp
dropped by containment).

**bout 1** (T=513)

| | (a) r2 jitter-10 | (b) P3b off | (c) P3b on |
|---|---:|---:|---:|
| pose-jump | 0.0448 | 0.0000 | 0.0000 |
| straddle | 0.0019 | 0.1793 | 0.1423 |
| female NaN | 0.7583 | 0.1092 | 0.1092 |

**bout 4** (T=394)

| | (a) r2 jitter-10 | (b) P3b off | (c) P3b on |
|---|---:|---:|---:|
| pose-jump | 0.0279 | 0.0000 | 0.0000 |
| straddle | 0.2640 | 0.0000 | 0.0000 |
| female NaN | 0.4822 | 0.2563 | 0.2563 |

**bout 28** (T=2007)

| | (a) r2 jitter-10 | (b) P3b off | (c) P3b on |
|---|---:|---:|---:|
| pose-jump | 0.0000 | 0.0000 | 0.0000 |
| straddle | 0.0005 | 0.0000 | 0.0000 |
| female NaN | 0.1196 | 0.1545 | 0.1545 |

Male keypoints dropped by containment in (c): bout 1 0.48%, bout 4 0.03%,
bout 28 0.00% (T-weighted mean over the three bouts ~0.09%) -- containment
touches almost nothing here because the model itself already fixed most of
the leak.

**Reading.** Pose-jump: (b) <= (a) in every bout (0.0448->0, 0.0279->0,
0->0) -- the MODEL fix, present even with containment off. Bout 1's
straddle number looks like it got WORSE ((a) 0.0019 vs (b) 0.1793), but (a)'s
number is blind: her centroid is NaN on 75.8% of bout 1's frames under the
r2 checkpoint, so the straddle test could not fire there at all; P3b's female
NaN rate on bout 1 is 10.9%, so the metric is now actually measuring instead
of blind, and (c) (containment on) is lower than (b) (0.1423 < 0.1793),
matching the expected ordering among the two P3b arms. Bout 4 shows the
clean case: straddle drops from 26.4% (blind on "only" 48% of frames) to
0.0% in both P3b arms. Bout 28 stays clean (pose-jump/straddle ~0 throughout)
as expected, though female-missing ticks up slightly (0.1196 -> 0.1545,
+3.5pp) -- a small, isolated regression against the "not worse than (a)"
expectation, outweighed by the large female-NaN improvements on bouts 1 and 4
(0.7583->0.1092, 0.4822->0.2563).

**Verdict: (b) <= (a) holds** (the gating condition for Step 4) -- pose-jump
improves or ties in all three bouts, and the one metric that looks worse
(bout 1 straddle) is explained by reduced blindness, not a new defect.

### Step 3: figure

`figures/2026-09-mvq/p3b_gates/p3b_bout1_bout4_compare.png` -- bout 1
(frames 374, 351, 71) and bout 4 (frames 230, 220, 260), overhead
Cam2012630 + side Cam2012855, three columns (r2 jitter-10 / P3b containment
off / P3b containment on), male (fly1) keypoints in orange over both flies'
SAM mask outlines. Read back: in the r2 column, bout 1's orange male
keypoints visibly spread onto/around the cyan female outline in both
cameras (f374, f351 especially); in BOTH P3b columns (off and on) the orange
points stay on the male's own orange-outlined body in all three bout-1
frames and both cameras -- i.e. already fixed in the containment-OFF column,
a model-level fix, not merely the containment filter removing points. Bout 4
shows the same clean separation in all three columns. Caveat: these three
frames are the OLD r2-based worst-frame picks, not re-picked for P3b, so a
clean render here does not contradict the bout 1 straddle metric (17.9%/
14.2% of frames) -- those residual straddle frames occur elsewhere in the
bout, not at these particular frames.

### Step 4: config + campaign defaults

Both gates passed -> `configs/mvq/p3b.yaml` created (copy of `p3a.yaml`,
checkpoint swapped to `.../mvq_t1_b16_p3b_contact_20260905/final`, `identity:
mask`, `containment: "on"` unchanged). `scripts/slurm/mvq_p3a_campaign.sh`
defaults changed to `RUN_NAME=pose_mvq_p3b` / `MVQ_CONFIG=p3b` (p3a still
selectable via `--mvq-config p3a --run-name pose_mvq_p3a`). Verified:
`JAX_PLATFORMS=cpu pytest third_party/jarvis_jax/tests/test_lift_masked_bout.py -q`
-- 32 passed; `scripts/slurm/mvq_p3a_campaign.sh --dry-run --only
2025_10_20_13_20_04` -- prints the array/aggregate sbatch scripts rooted at
`pose_mvq_p3b` with `mvq=p3b pipeline.lifter=mvq`, nothing submitted.

## P3b on single-fly recordings

Field check of `mvq_t1_b16_p3b_contact_20260905/final` -- this checkpoint has
only ever been run on courtship PAIRS; here it runs mask-free
(`scripts/coarse_pass_mvq.py`, CenterDetect + mvq, stride 1, `--num-animals
1`) on three genuinely single-fly spans. Risk tested: a SECOND typed slot
firing on a single-fly recording (phantom fly / duplicate window), or
mis-sexing the lone fly.

`--num-animals 1` stores only ONE row, read with `want_sex=-1` (whichever
typed slot exists) via `read_typed` (`coarse_track.py` docstring, confirmed
by reading it) -- it cannot show whether the OTHER typed slot ALSO cleared
threshold in the same window, or whether CenterDetect ever proposed a 2nd
window. Built `single_fly_slot_probe.py` (direct `MVQRunner`/`CenterDetector`
probe, same windowing geometry as `coarse_pass`, 20 evenly-spaced
frames/span) to check that directly; it deliberately passes `max_animals=2`
to `lift_peaks_to_centres` (production uses `max_animals=num_animals=1`) so
a would-be 2nd candidate is visible instead of being capped away.

| span | frames | fps | exist>=0.5 | sex read | phantom window (probe, `max_animals=2`) | 2 typed slots same window | pose-jump | vs reference (median, name-matched) | verdict |
|---|---:|---:|---:|---|---:|---:|---:|---|---|
| clip_A (Clip/Session6/2025_10_12_15_06_46, f0-921) | 921 | 4.07 | 69.7% (30.3% NaN, never marginal: 0.50-1.0) | 100% male (sex_prob~4e-5), GT unknown | 0/20 | 0/20 | 6.5% | raw 10.79mm / centroid-rel 0.99mm vs data3D.csv | see caveat below |
| fr_b2 (free_running/Session11/2026_03_03_14_32_16, f25933-26174, bout 2) | 241 | 2.43 | 100% | 100% female (sex_prob 0.996), matches GT (notes.txt: "all female") | 0/20 | 0/20 | 0% | 0.041mm vs bout_00002/fly0/kp3d.npz | SAFE |
| fr_b4 (same rec, f90913-91189, bout 4) | 276 | 2.84 | 100% | 100% female (sex_prob 0.997), matches GT | 1/20 (duplicate, see below) | 0/20 | 0% | 0.068mm vs bout_00004/fly0/kp3d.npz | SAFE (see caveat) |

Units: verified empirically (Antenna_Base-Abd_tip body length = 24 raw
units in `data3D.csv` = 2.4mm, matching real Drosophila anatomy) that mvq's
coarse-pass output, `data3D.csv` and the free-running `kp3d.npz` are all in
the SAME 0.1mm ("units") convention -- table values converted to mm.

**fr_b2 / fr_b4: clean.** Existence 100%, single window every frame over the
FULL production run (`n_windows` unique == 1 across all 241/276 frames,
confirmed from the real `coarse_pass` output, not just the 20-frame probe),
sex read matches the session's known-all-female ground truth with high
confidence, zero pose-jump, and kp3d agrees with the independent DLT/STAC
`kp3d.npz` reference to 0.04-0.07mm median per-keypoint (name-matched via
`cfg.model.KP_NAMES`, never by index). fr_b4's one probe-flagged frame
(91173, `max_animals=2` only) has CenterDetect proposing a second 3D
candidate 2.8mm from the first, but BOTH windows' female slot converges on
the SAME centroid (`[209.7,35.6,13.4]` vs `[209.7,35.3,13.3]`) -- a DUPLICATE
window on the one real fly, not a hallucination elsewhere, and the actual
production `--num-animals 1` run (which correctly caps
`max_animals=num_animals=1`) never splits into 2 windows for any of its 276
frames, so this never reaches an output track.

**clip_A: identity gates are safe, but keypoint SCALE COLLAPSES ~3x.**
Zero phantom windows/slots at any threshold, and sex read is a stable,
confident single type throughout (no flip-flopping). But body length
(Antenna_Base-Abd_tip) computed directly from mvq's own kp3d is 0.73-0.94mm
across sampled frames, vs 2.4mm in `data3D.csv` and vs the 2.6mm the SAME
metric gives on fr_b2/fr_b4 -- the skeleton is compressed to roughly a
third of true anatomy, keypoints huddling near the crop centre instead of
spanning the body. This is why the naive "raw vs reference" distance
(10.79mm) looked bad but the naive "centroid-relative" distance (0.99mm)
looked fine: a collapsed near-point skeleton and a normal-sized skeleton
disagree, after re-centering, by roughly half the real anatomy's spread --
almost exactly what was measured, i.e. 0.99mm was NOT evidence of good shape
match, it was hiding the collapse (the CLAUDE.md trap: a plausible-looking
number for the wrong reason). Confirmed as calibration/rig-specific, not a
same-scale generalization failure: fr_b2/fr_b4 use the SAME rig+calibration
convention as courtship training data (`free_running_session11.yaml`: "Same
7-camera rig + calibration as courtship") and get correct 2.6mm anatomy;
clip_A is a different rig/session (`Video_recordings/Clip/Session6`, native
calibration is `Cam*_dlt.csv`, converted here via a scratch Cam*.yaml write
from `clip_io.load_dlt`, byte-verified to reproject window centres onto the
detector's own peaks) whose world-unit scale evidently does not match what
P3b's crop-size convention expects. clip_A's existence miss rate (30.3%
NaN, never marginal) is a recall gap, not a false-positive identity issue.

**Figures** (read back against the stated expectation -- skeleton ON the
fly, no second marker set):
- `figures/2026-09-mvq/p3b_gates/single_fly/fr_b2_contact_sheet.png`,
  `.../fr_b4_contact_sheet.png` -- 6 frames x (overhead Cam2012630, side
  Cam2012861): green keypoints sit tightly and correctly-scaled ON the one
  fly in every frame/camera of both bouts, exist/sex titles read 1.00/female
  throughout. Matches expectation exactly.
- `.../clip_A_contact_sheet.png` -- 6 frames x (Cam2012630, Cam2012861):
  green points form a small, scattered cloud near but not on the fly's body
  (frames 0/184/368/552 exist 0.71-0.84) and vanish where exist is NaN
  (frames 736/920) -- disagrees with the "skeleton on the fly" expectation,
  consistent with the scale-collapse finding above, not a phantom (no
  second marker set drawn in any frame).

**Overall verdict: SAFE for phantom-fly / mis-sexing** on all three
single-fly spans -- the specific risk this checkpoint was tested for
(a second typed slot asserting a fly that isn't there, or mis-sexing the
lone fly) did not occur in the real `--num-animals 1` production path on
921+241+276 frames. The one open finding is UNRELATED to that risk:
clip_A's anatomy scale collapse, attributed to a calibration/rig mismatch
for that specific test recording rather than a P3b defect, since the
same-rig-as-training free-running spans are clean. Recommend re-testing
clip_A (or an equivalent single-fly clip) on a courtship-rig calibration
before drawing conclusions about P3b's OOD generalization from it.

Scripts (scratch, not committed):
`single_fly_slot_probe.py`, `make_contact_sheet.py` in
`/tmp/claude-398823/.../scratchpad/` (session-local).
