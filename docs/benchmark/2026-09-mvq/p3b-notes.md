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

## P3b on the mask-free route (2026-09-05)

Re-evaluates the mask-free front end (`scripts/coarse_pass_mvq.py`,
`jarvis_jax.tracking.coarse_track`) on Session0/2025_10_20_13_20_04 with the
P3b contact-heavy checkpoint (`mvq_t1_b16_p3b_contact_20260905/final`),
against the jitter-10 checkpoint results recorded in `p4-maskfree-notes.md`
and `gate-bouts-probe-2026-09-05.md`. GPU gate observed: waited for the
8 concurrent `mvq_lift_bout.py` campaign workers (Session1) to clear and
both GPUs to read 0 MiB before starting; coarse pass ran on GPU0, the bout-117
fine pass on GPU1, in parallel, both uncontended thereafter.

### Coarse pass (stride 16, 31125 coarse frames)

Wall clock **98.2 min** (5.28 frames/s end-to-end, uncontended) vs
jitter-10's recorded 221.2 min (2.35 frames/s) -- **not a fair speed
comparison**, since the jitter-10 number was measured under a concurrent
4-worker GPU-lift campaign and this run was not; the reader/compute path is
unchanged between checkpoints.

**Floor-block defect found and fixed.** The on-disk `.meta.json` first wrote
`orientation: "skew"`, `skew: 0.144` -- ABOVE the 0.1 marginal threshold,
and unlike jitter-10 (whose skew=0.077 heuristic pick was independently
confirmed correct in the earlier real-run wave) **the automatic sign here
was WRONG**: as-is, median trackable height was -53.4 units with 100% of
trackable frames negative -- physically impossible for a floor-anchored
height. Checked both signs directly (`fit_floor` with `up_hint=+/-normal`
on the run's own `X3d`/`exist`): the flipped sign exactly matches jitter-10's
independently-confirmed floor normal and gives median height 5.8, p5 -7.7,
p95 23.7, only 28.7% negative -- physically sane. Fixed by rerunning
`coarse_pass_mvq.py --resume --up-hint <jitter-10's confirmed normal>`
(a ~0 s recompute-only pass, no re-inference needed); corrected meta now
reads `orientation: "hint"`, median trackable height **17.13** units (p5
3.62, p95 35.04, 1.2% negative) -- comparable to jitter-10's 11.29 but on the
high side, both dragged up by the same wall-climbing tail. This is a new
operational hazard of this checkpoint's centroid distribution, not evidence
about mask-free tracking quality itself, but confirms the CLAUDE.md rule
("do not trust the sign silently") caught a real flip.

### Comparison table (jitter-10 vs P3b, mask-free route, Session0 20_04)

| metric | jitter-10 | P3b | verdict |
|---|---:|---:|---|
| female frac_trackable (exist>=0.5) | 47.8% | 45.4% | unchanged (~2pp worse) |
| male frac_trackable | 98.1% | 98.4% | unchanged |
| collapsed coarse frames | 0 | 748 (2.4%) | **regression** |
| floor orientation / skew | hint / 0.077 (confirmed) | skew / 0.144 (wrong sign; fixed w/ hint) | new hazard, fixed |
| gate bouts, DEFAULT thresholds | 542 windows, recall 0.90, precision 0.083 | 521 windows, recall 0.967 (29/30), precision 0.083 (43/521) | recall improved, precision flat |
| gate bouts, CALIBRATED (max_gap=100, min_dur=25) | 32 windows, recall 0.967, precision 0.531 (17/32) | 28 windows, recall 0.967 (29/30), precision 0.643 (18/28) | **precision improved** |
| male wing angle >=30deg frac / median | 91.3% / 40.4deg | 91.6% / 40.0deg | **unchanged -- feature-definition problem, not model** |
| bout117 (frames 111488-114688) female missing (per-kp NaN) | 4.81% | 9.34% | **regression** |
| bout117 male missing (per-kp NaN) | 2.16% | 1.41% | improved |
| bout117 collapsed (female/male) | 0/0 | 0/0 | unchanged |
| bout117 straddle frac (>=5 male kp closer to female centroid than own) | 0.2269 | 0.2334 | **unchanged** (contact-frame tangling persists) |

The unmatched GT bout is the same one in both runs (bout_idx 8, per
`coarse_pass_gates.py`'s `--ground-truth` validation), not a new gap.

### Figures

Neither `coarse_tracks_check.png` nor `coarse_centres_check.png` had been
generated yet for the real jitter-10 run (the earlier SDD wave deferred that
step); both were generated fresh here from the existing jitter-10
`coarse_tracks.npz` so the comparison below is a real image diff, not just
numbers.

- `figures/2026-09-mvq/p4_maskfree_p3b/coarse_tracks_check.png` (+.json) vs
  `figures/2026-09-mvq/p4_maskfree/coarse_tracks_check.png` (+.json, newly
  generated jitter-10 counterpart). **Read back:** the existence panel looks
  visually near-identical between the two -- cyan (female) flickers rapidly
  above/below the 0.5 line throughout the whole 10.4-minute recording in
  BOTH checkpoints; **her trace is NOT continuous in either**, answering the
  task's stated question directly. Male wing-angle panels are
  near-identical (baseline ~30-50deg, frequent spikes to 150-170deg) in
  both. The gate-bout row is marginally sparser for P3b (521 vs 542
  windows, one more visible gap around minute 3-4) but still very dense in
  both. Both report **0/30 reviewed bouts unexplained** by the tracks
  (bout-28 zoom panel: distance/wing-angle both step cleanly at the
  reviewed-bout boundary in both checkpoints).
- `figures/2026-09-mvq/p4_maskfree_p3b/coarse_centres_check.png` vs
  `figures/2026-09-mvq/p4_maskfree/coarse_centres_check.png` (same 12
  frames, deterministic by bout index). **Read back:** centres land
  correctly inside each fly's own body in both cameras, in both
  checkpoints, at every frame where a fly is present (e.g. frames 13648,
  252176, 373760, 475984) -- no centre on a wall or reflection in either.
  This gate is dominated by shared CenterDetect + 3D-centroid merge logic
  (unaffected by which mvq checkpoint lifts keypoints), so near-identical
  results are expected; only frame 497984's side camera shows a cyan
  (female) circle in P3b that jitter-10 lacks, a single-frame difference,
  not a pattern.
- `figures/2026-09-mvq/p4_maskfree_p3b/fine_bout117/bout117_maskfree_contact_sheet.png`
  (6-frame overhead+side sheet, gate-bout-117 candidate, frames
  111488-114687). **Read back:** in "apart" frames (112768, 113408, 114048)
  identity is correctly separated -- orange (male) and cyan (female)
  skeletons sit on two distinct, separate fly bodies in both cameras. In the
  two contact/mounting frames (112128, 114687) the orange and cyan skeletons
  overlap and tangle on what looks like a single fused body in both
  cameras -- the same qualitative failure the straddle metric measures,
  present at essentially the same rate as jitter-10.

### Verdict

**Improved on gate quality, unchanged on the two core mask-free blockers.**
P3b's contact-heavy retrain measurably helps the *gate* layer built on top of
the mask-free tracks -- default-threshold recall rose from 0.90 to 0.967
(27/30 -> 29/30 reviewed bouts matched) and, more importantly, the
calibrated-threshold operating point's precision rose from 0.531 to 0.643
at equal recall and fewer windows (32 -> 28) -- a real, if modest, gain for
the gate-bouts-probe's recommended `max_gap=100` setting. But the two
failures that actually drive the mask-free front end's poor recall/precision
in the first place are **unchanged**: female trackability stays at ~45-48%
(the root cause the gate probe traced 81% of inter-bout gaps to), and the
male wing-angle gate stays saturated at ~91% regardless of checkpoint,
confirming this is a **feature-definition problem** (the rest baseline
sits at 27-40deg, already close to/above the 30deg threshold) and not
something a lifter retrain can fix. On the one bout where P3b's contact
training should show up most directly (gate-bout 117, a sustained
close-proximity + wing-extension candidate), cross-fly tangling is
**unchanged** (straddle 0.2269 -> 0.2334) and female per-keypoint
missingness got **worse** (4.8% -> 9.3%), and the full coarse pass now
produces 748 collapsed frames where jitter-10 had zero. **What remains for
Plan 2**: the mask-free front end needs (1) an independent fix for female
localizability (CenterDetect or the merge/typed-slot assignment, not the
mvq keypoint lifter -- P3b already targeted the lifter and moved neither
number), and (2) the wing-angle behaviour gate redefined against a
per-recording or per-fly rest baseline rather than a fixed 30deg threshold,
since it is saturated at ~91% under BOTH checkpoints.

### Timings

| stage | wall clock | rate |
|---|---:|---:|
| coarse pass, P3b, stride 16, 31125 frames, GPU0 | 98.2 min | 5.28 frames/s |
| fine pass, P3b, bout 117, stride 1, 3200 frames, GPU1 (parallel w/ coarse) | 11.6 min | 4.59 frames/s |
| floor-sign fix (--resume --up-hint, recompute only) | <1 s | n/a |

### Files

- `/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/coarse_mvq_p3b/coarse_tracks.npz` (+`.meta.json`)
- `/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/coarse_mvq_p3b/bouts_mvq_gates.csv` (default thresholds), `bouts_mvq_gates_calibrated.csv` (max_gap=100, min_duration=25)
- `/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/coarse_mvq_p3b/fine_bout117/fine_tracks.npz` (+`.meta.json`), converted bout at `.../fine_bout117/bouts/bout_00117/fly{0,1}/{kp2d,kp3d}.npz`
- `figures/2026-09-mvq/p4_maskfree_p3b/{coarse_tracks_check,coarse_centres_check}.{png,json}`
- `figures/2026-09-mvq/p4_maskfree_p3b/fine_bout117/bout117_maskfree_{preview,preview_small}.mp4`, `bout117_maskfree_preview_header.json`, `bout117_maskfree_contact_sheet.png`
- `figures/2026-09-mvq/p4_maskfree/{coarse_tracks_check,coarse_centres_check}.{png,json}` (jitter-10 counterparts, newly generated this session for a real image comparison)
- Generating scripts, kept in the session scratchpad per CLAUDE.md (one-off diagnostics, not promoted): `step2_convert_fine_bout_p3b.py`, `step3_render_bout117_p3b.py` (copies of the prior agent's bout-117 scripts repointed at the P3b paths), `straddle_compare.py` (jitter-10 vs P3b straddle/missing-fraction metric, both variants).

## Single-fly pseudo-label pass for mvq v2 (2026-09-06)

Task 4 of `.superpowers/sdd/2026-09-05-mvq-v2-plan-a-pseudolabels`; spec
`docs/specs/2026-09-05-mvq-v2-pseudolabel-t2-design.md` §3.4. Produces the
single-fly half of the v2 pseudo-label set -- free-running FEMALES, so the
female-host stratum the whole effort exists to fix is not starved -- by
running the same P3b checkpoint mask-free at `--num-animals 1` and writing
P3b-SHAPED bout dirs that Task 2's extractor reads unchanged.

**Precondition: not re-run.** The P3b single-fly field check is the section
"P3b on single-fly recordings" above, verdict **SAFE for phantom-fly /
mis-sexing** on 921+241+276 frames (fr_b2 = free_running/Session11/
2026_03_03_14_32_16 among them). This pass cites it rather than repeating it;
what follows are this run's own numbers over 8,000 new frames.

Script: `scripts/pseudo_labels/singlefly_p3b_pass.py` (three modes: the pass,
`--scan` trackability probe, `--contact-sheet` renderer), test
`third_party/jarvis_jax/tests/test_singlefly_pass.py`. Checkpoint
`mvq_t1_b16_p3b_contact_20260905/final`, CenterDetect `cd_focal_bg30/
ckpt/epoch_004`, GPU node g3102 device 7, ~5 frames/s (8,000 frames in 28 min).

**Clip/Session6/2025_10_12_15_06_46 is EXCLUDED** (controller, 2026-09-05):
its keypoints come out at ~1/3 scale -- the calibration/rig mismatch recorded
above under clip_A -- and the cause is unresolved. `--link-cameras` (the
`Cam*_frames_<a>_<b>.mp4` -> `<Cam>.mp4` symlinks that recording needs) is
implemented and unit-tested, but no Clip data enters v2.

### Spans

Both recordings are free_running Session11 (`notes.txt`: "flies: all female
Canton S"), same 7-camera rig and calibration as courtship. Candidate spans
were chosen around walking bouts from `free_running_bout_summary.csv` and
screened with `--scan` (20 evenly spaced probe frames, CenterDetect peaks
lifted to a 3D centre): **all eight candidates scored 1.00 trackable**, so the
first two per recording were taken.

| recording | bout dir | span (canonical slots) | contains walking bout |
|---|---|---|---|
| 2026_03_03_14_32_16 | bout_00000 | 25000-27000 | 2 (25933-26173) -- the field check's fr_b2 |
| 2026_03_03_14_32_16 | bout_00001 | 90200-92200 | 4 (90913-91188) -- fr_b4 |
| 2026_03_03_13_25_13 | bout_00000 | 26300-28300 | 1 and 2 |
| 2026_03_03_13_25_13 | bout_00001 | 121300-123300 | 4 |

Caveat on the scan gate: the 20-frame probe says 1.00, but at stride 1 the
full pass detects its own centre on 84-100 % of frames and REUSES the previous
one on the rest (`centre_source`, the coarse pass's rule). The sparse probe
cannot see that, so the per-bout `frac_centre_detected` below is the number to
read, not the scan's.

### Per-bout numbers (all 2,000 frames each; units = 0.1 mm)

| recording / bout | written | exist | exactly 1 typed slot >= 0.5 | 2nd typed slot exist (mean) | frames w/ both typed slots | 2nd-slot dist med / p95 / max | > 3.0 u | step med / p99 | centre det / reuse (max run) | sex read (p_female) |
|---|---|---|---|---|---|---|---|---|---|---|
| 14_32_16 / 00000 | 100 % | 0.997 | 0.584 | 0.405 | 832 | 6.10 / 32.9 / 47.4 | 493 | 0.33 / 30.5 | 0.841 / 0.159 (46) | female 54 % (0.54) |
| 14_32_16 / 00001 | 100 % | 0.996 | 0.827 | 0.196 | 347 | 7.95 / 35.9 / 44.9 | 231 | 0.19 / 6.3 | 0.851 / 0.149 (45) | female 74 % (0.74) |
| 13_25_13 / 00000 | 100 % | 0.996 | 0.868 | 0.128 | 264 | 2.05 / 9.1 / 46.9 | 100 | 0.19 / 6.1 | 0.948 / 0.052 (61) | female 87 % (0.87) |
| 13_25_13 / 00001 | 100 % | 0.999 | 0.954 | 0.070 | 93 | 47.6 / 49.1 / 49.7 | 93 | 0.14 / 1.3 | 1.000 / 0.000 (0) | female 100 % (1.00) |

Three findings, each with the quantity that separates it from its harmless
look-alike:

1. **Both typed slots fire on the ONE fly, often** (up to 42 % of frames), so
   "fraction of frames with exactly one slot above 0.5" is 0.58-0.95, not
   ~1.0 as the 20-frame probe suggested. Counting live slots cannot tell a
   hedged sex read from a phantom, so the pass also records the MEDIAN
   PER-KEYPOINT 3D DISTANCE between the two typed slots
   (`second_typed_slot_dist_units`) against `pick_typed_pair`'s own same-fly
   rule (`COLLAPSE_DIST_UNITS` = 3.0 u = 0.3 mm). On three bouts the median is
   2-8 u -- the same animal read twice, the model hedging on sex -- while on
   13_25_13/00001 all 93 such frames sit at ~47.6 u (4.8 mm, the far edge of
   the 5.6 mm window): a genuine PHANTOM, on 4.7 % of that bout's frames. It
   never reaches the output (`read_typed(want_sex=-1)` takes the
   higher-existence slot) but it is exactly the material §3.5's
   existence-negatives want.
2. **Stale reused windows fire existence ~1.0 on empty crops.** 348 of the
   8,000 frames (4.4 %) have existence >= 0.98 with EVERY per-view visibility
   below 0.5; the contact sheet of the six worst
   (`figures/2026-09-mvq/v2_pseudo/singlefly_worst_14_32_16_b0.png`, frames
   26749-26757) shows bare arena floor in both cameras with no fly anywhere.
   This is the spec's known failure reproduced on a single-fly recording. The
   visibility head is right where the existence head is wrong, and the
   pseudo-label gates use the visibility (`conf >= 0.5` scores the
   reprojection gate), so **0 of those 348 frames are admitted** (checked
   directly).
3. **The sex head is unreliable here.** Across the four bouts the typed read
   is female on 54 %, 74 %, 87 % and 100 % of written frames (mean p_female
   0.54-1.00) for a fly that is female by the session's own `notes.txt`. The
   pass therefore records the RECORDING'S ground truth
   (`--fly-sex female`, `sex_source: recording_ground_truth`) and keeps the
   model's read beside it in `mvq_meta.json`'s `sex_read`. Training on those
   labels is the corrective signal, not confirmation bias.

### The gates pick the good frames (rigid invariants, all frames vs admitted)

`pseudo_gates.admit_bout(..., store=None, use_identity=False)` -- the real
gate stack the extractor runs -- over each bout. `body` is the RIGID
Antenna_Base-Abd_tip length (~24-27 u = 2.4-2.7 mm on a real fly; ~8 u would
be the clip_A scale collapse) and `eyes` the RIGID EyeL-EyeR spacing:

| recording / bout | admitted | anchors | body CV all -> admitted | eyes CV all -> admitted | reproj p95 all -> admitted (px) | blind frames admitted |
|---|---|---|---|---|---|---|
| 14_32_16 / 00000 | 1306 (65 %) | 94 | 0.114 -> 0.033 | 0.569 -> 0.082 | 3.73 -> 1.19 | 0 of 245 |
| 14_32_16 / 00001 | 1456 (73 %) | 95 | 0.112 -> 0.010 | 0.622 -> 0.028 | 51.6 -> 1.08 | 0 of 42 |
| 13_25_13 / 00000 | 1668 (83 %) | 110 | 0.126 -> 0.010 | 0.133 -> 0.033 | 7.97 -> 0.72 | 0 of 61 |
| 13_25_13 / 00001 | 1933 (97 %) | 125 | 0.010 -> 0.010 | 0.030 -> 0.030 | 0.78 -> 0.77 | 0 of 0 |

Every admitted frame has all 7 cameras seeing at least one keypoint, and the
admitted median body length is 26.1-27.2 u (2.6-2.7 mm) with CV 0.01-0.03 --
correct anatomy, no scale collapse. Rejections are step (67-671 per bout) and
reprojection (0-477); existence rejects 0, which is the point of finding 1:
existence is not the gate that catches the empty windows, visibility is.

### Independent cross-check vs the pipeline's own ViTPose -> DLT track

The only check that does not go through the mvq model: the free-running
pipeline's `Session11_<rec>_bouts/bouts/bout_<n>/fly0/kp3d.npz` over the same
video frames and calibration, compared BY KEYPOINT NAME (the reference npz
stores no `kp_names`; its axis is `cfg.model.KP_NAMES` by construction, which
the matching body length below confirms).

| recording / bout | walking bout | overlap | frame lag | median agreement | body length mine / reference |
|---|---|---|---|---|---|
| 14_32_16 / 00000 | 2 | 241 | 0 | **0.43 u = 0.043 mm** | 26.53 / 26.28 u |
| 14_32_16 / 00001 | 4 | 276 | 0 | **0.61 u = 0.061 mm** | 26.03 / 25.73 u |
| 13_25_13 / 00000 | 1 | 550 | +25 | 3.32 u at lag 0 -> **0.50 u at lag +25** | 27.09 / 27.11 u |
| 13_25_13 / 00000 | 2 | 384 | +26 | 6.55 u at lag 0 -> **0.45 u at lag +26** | 27.12 / 27.07 u |
| 13_25_13 / 00001 | 4 | 502 | +25/26 | 4.08 u at lag 0 -> **0.52 u at lag +25** | 27.21 / 26.92 u |

**The 13_25_13 offset is a frame-index CONVENTION difference, not an error.**
That recording's `sync_plan.json` says `status: reindex` with
`canonical_len 508196` vs 508170 mp4 frames = **26 dropped frames**, while
14_32_16 is `clean` and needs no shift. This pass maps canonical slots through
`predict.synced_reader.load_plan`; the reference bout dirs predate the
dropped-frame gate and index mp4 positions directly, so at frame ~26.5k they
are 25 frames behind. The lag curve is a clean V with its minimum at +25/+26
and the agreement there is 0.045-0.052 mm -- the SAME level as 14_32_16's at
lag 0, and the level the earlier field check measured (0.041/0.068 mm). It
matters that this pass is on the canonical-slot side: the pseudo-label
extractor's frame reader (`SessionFrames`) uses the same `load_plan`, so the
JPEG written for frame f is the frame the keypoints came from.

### Export

`scripts/pseudo_labels/extract_p3b_pseudolabels.py --num-animals 1
--no-identity-gate --target 2000 --min-female 0` over
`OutFiles/v2_singlefly/*/`, into
`/gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_singlefly_20260905`.

- **424 anchors** (every admissible one) + **828 T=2 partners** = **1,252
  framesets**, 8,764 JPEGs, 579 MB, in 160 s. Partner availability 416 / 410 /
  421 at Delta = 1 / 4 / 16.
- Per recording: 13_25_13 698, 14_32_16 554. Per bout 274-374, so the largest
  single bout is 29.9 % of the export -- §3.2's "no bout > 2 %" rule is a
  courtship-campaign rule over 160 bouts and cannot apply to a four-bout
  single-fly set.
- Manifest: `n_flies: 1`, `behavior: free_running`, `sex: female`,
  `sex_source: recording_ground_truth`, `has_masks: false`, `weight: 0.3`,
  `balance.female_host_weight: null` (one-sided -- there is no host-sex ratio
  to restore, and 0.0 would read as "weight these to nothing"). Stratum:
  1,252 female/mid; `contact`/`apart` are two-fly quantities and are 0 by
  construction, wall 1.4 %.
- `verify_export` (the loader round trip `V12WindowDataset` performs, run on
  the real export): **20 windows checked of 1,252, mean 49.9/50 keypoints with
  3D, mean 7.0 valid cameras, keypoint_names_ok true.**
- **Short of the ~2,000 framesets §3.4 asks for.** The binding limit is the
  anchor pool, not the gates: 4 bouts x 2,000 frames with 16-frame
  decorrelation is at most 500 anchors, and 424 of them passed. Two more spans
  per recording -- or a third Session11 recording (all seven have their own
  calibration and the same all-female `notes.txt`) -- would close the gap; the
  pass and the extract are one command each, and the extract is re-runnable
  over the same output root.

### Figures (read back against the stated expectation)

Expectation, stated in the script docstring before rendering: *one fly per
crop, keypoints ON her body (head points at the head end, abdomen at the
abdomen, leg chains along the legs) at the right scale, in BOTH cameras, and
no second skeleton anywhere in the crop.*

- `figures/2026-09-mvq/v2_pseudo/singlefly_check.png` (14_32_16/bout_00000, 12
  frames x Cam2012630 overhead + Cam2012861 side): matches the expectation.
  One fly per crop, no second marker set anywhere; red head points on the
  head, magenta abdomen points on the abdomen, yellow thorax/wing points on
  the thorax and wing bases, and six blue leg chains that follow the real
  legs, including on the wall-climbing frames where the fly hangs upside down.
  The 12th column (f26977, visibility 0.74) has the fly at the crop edge with
  part of the skeleton off her -- the low-visibility tail, below the gates.
- `.../singlefly_check_13_25_13_b0.png` (Cam2012855 + Cam2012862) and
  `.../singlefly_check_13_25_13_b1.png` (Cam2012630 + Cam2012857) and
  `.../singlefly_check_14_32_16_b1.png` (Cam2012631 + Cam2012853): same
  verdict on the other three bouts and on four further cameras chosen by name.
- `.../singlefly_worst_14_32_16_b0.png` (the six LOWEST-visibility written
  frames): empty arena floor in both cameras, existence 1.00, visibility 0.00
  -- finding 2 above, and the reason the default sheet selects on visibility.

### Commands

```bash
# span screening (CenterDetect only)
python scripts/pseudo_labels/singlefly_p3b_pass.py --scan 25000:27000 [--scan ...] \
  --session-dir <VID>/2026_03_03_14_32_16 --centerdetect <CD> --out OutFiles/v2_singlefly/<rec>
# the pass (GPU; module load cuda/12.9.1, LD_PRELOAD libstdc++, unset LD_LIBRARY_PATH JAX_PLATFORMS)
python scripts/pseudo_labels/singlefly_p3b_pass.py \
  --session-dir <VID>/2026_03_03_14_32_16 --run <RUN>/final --centerdetect <CD> \
  --bout 25000:27000 --bout 90200:92200 --num-animals 1 --fly-sex female \
  --out OutFiles/v2_singlefly/2026_03_03_14_32_16
# the figures
python scripts/pseudo_labels/singlefly_p3b_pass.py --contact-sheet \
  OutFiles/v2_singlefly/2026_03_03_14_32_16/bouts/bout_00000 \
  --out-png figures/2026-09-mvq/v2_pseudo/singlefly_check.png \
  --sheet-cameras Cam2012630,Cam2012861 --sheet-frames 12         # add --sheet-worst
# the export
JAX_PLATFORMS=cpu python scripts/pseudo_labels/extract_p3b_pseudolabels.py \
  --runs OutFiles/v2_singlefly/*/ --num-animals 1 --no-identity-gate \
  --target 2000 --min-female 0 --per-rec-frac 1.0 --per-bout-cap 2000 \
  --out /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_singlefly_20260905
```

One-off diagnostic scripts kept in the session scratchpad per CLAUDE.md (not
promoted): `summarise_bouts.py` (the per-bout table), `admitted_check.py` (the
all-vs-admitted invariants), `vs_reference.py` and `shift_probe.py` (the DLT
cross-check and the lag curve).
