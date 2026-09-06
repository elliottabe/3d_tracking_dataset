# MVQ v2 pseudo-label plan (Plan A) -- working notes

## Gate calibration, bout 4

Real bout: `pose_mvq_p3b/bouts/bout_00004`,
`2025_10_20_13_20_04` (Session0), 7 cameras, 394 frames, default
`GateThresholds()`.

```
frames 394
per-fly admitted: fly0 (female) 0.00, fly1 (male) 0.934
anchors (decorrelated) 25
combined per-gate rejection rate (both flies, mean over F,T):
  exist     0.316
  step      0.407
  reproj    0.253
  contain   0.497
  nonfinite 0.231
  identity  0.000
```

fly0 (female, canonical fly0 = female post-canonicalization) is admitted on
0 % of frames in this bout; fly1 (male) at 93.4 %. Per-fly breakdown:

```
fly0: exist_ok=0.37 step_ok=0.21 reproj_ok=0.54 contain_ok=0.01 finite=0.55
fly1: exist_ok=1.00 step_ok=0.97 reproj_ok=0.96 contain_ok=1.00 finite=0.99
```

Investigated before accepting this: is `contain_ok=0.01` for fly0 a gate bug
(e.g. a camera-order mixup, cf. the keypoint/camera-order traps in
CLAUDE.md) or genuine data? Picked a frame with fully-finite fly0 `kp3d`
(t=108) and printed the per-camera containment fraction directly:

```
Cam2012630 frac_inside=0.46  Cam2012853 frac_inside=0.48
Cam2012855 frac_inside=0.60  Cam2012857 frac_inside=0.58
Cam2012861 frac_inside=0.72  Cam2012862 frac_inside=0.68
(Cam2012631 invalid mask this frame, skipped)
```

These are moderate, plausible partial-containment values (not 0 % or 100 %
everywhere, which is what a camera-index bug tends to produce) -- consistent
with a genuinely imperfect IK fit for the harder (female) fly on this bout,
matching the pipeline's documented female-tracking gap (see
`ab-hybridnet-vs-dlt-ik` memory: FEMALE is the hard fly, walls/OOD poses).
Conclusion: the gates are behaving correctly here; this is real data, not a
bug. None of the *combined* (both-fly) per-gate rejection rates exceed 90 %
(worst is `contain` at 49.7 %), so per the brief's "gate rejects > 90 % ->
stop" check, nothing here is a vacuous/broken gate -- but note that
single-fly, single-bout admission can legitimately be 0 % for the harder fly,
which the extractor (Task 2) needs to account for when hitting the 50/50
female-host/male-host stratification target across the full 11-recording,
160-bout campaign (a single bout is not a representative sample).

## Fixture bugs found and fixed (in `tests/pseudo_fixtures.py` / `pseudo_gates.py`)

The brief's fixture/test code, transcribed verbatim, initially failed 6/8
tests. All 6 failures traced to bugs in the *fixture and default-threshold
handling*, not ambiguity in the gate logic itself; fixed in a-priori TDD
order and reverified against every assertion before moving on:

1. **Camera FOV too narrow for `contain_min_cams`/`reproj_min_views`
   defaults.** `GateThresholds()` defaults (`reproj_min_views=5`,
   `contain_min_cams=5`) assume >= 5 real cameras; the fixture only has 3.
   Fixed by clamping both to `min(threshold, C)` where `C` is the actual
   camera count, in `reprojection_gate` and `containment_gate`.
2. **`cam_mats()` scale=2.0 put fly1 (default `sep_units=40`) at u in
   [126,163] in camera0's 80x120 frame -- entirely off-frame for the WHOLE
   bout**, not a boundary case (verified numerically across all t). Fixed by
   changing the affine scale to 1.0, which keeps both flies inside every
   camera with margin across the full default drift range, and leaves
   documented `H,W=80,120` and `sep_units=40` untouched.
3. **`make_masks`' `n_dropped` fixture field was a scalar
   `len(drop_frames) and 1 or 0` broadcast over the WHOLE `[1]*T` array**
   for fly1, marking every frame of the bout as containment-dropped whenever
   `drop_frames` was non-empty (regardless of which frame), which made
   `identity_gate` reject the entire bout. `drop_frames` is meant to
   simulate an existence dip (not a containment drop) -- fixed by keeping
   `n_dropped` all-zero.
4. **`partner_masks` semantics.** The test
   `test_partner_masks_need_the_partner_frame_to_pass_too` uses `delta=16`,
   `T=20`, poisoning index 5 and checking `p[16][3]`; that index is
   strictly INTERIOR to the span `[3,19]`, never an endpoint of any valid
   delta=16 pair in a 20-frame array -- the only way this assertion is
   non-vacuous is if a partner pair requires every frame in the closed span
   `[t, t+delta]` to pass, not just its two endpoints. Re-verified this
   reading against every other assertion in the file (including the `delta=1`
   and `delta=4` cases, where span vs. endpoint cannot be distinguished)
   before changing the implementation. `partner_masks` now requires the
   full span via a cumulative-sum window check.

## Known concern carried into Task 2

`admit_bout`'s `decorrelate(frame_any, thr.decorrelation, protect=None)`
call hardcodes `protect=None`, discarding the `protect` array computed just
above it from `partner_masks`. Per spec §3.2 ("the frame is >= 16 frames
from any admitted frame ... unless it is a T=2 partner") and `decorrelate`'s
own docstring, partner frames should be exempt from the 16-frame spacing
requirement. As written, no test exercises this interaction (the one test
that checks decorrelation, `test_clean_bout_...`, has every frame passing
every gate, so `protect` never has an effect), so this was left as UNFIXED
dead code rather than silently changed without a red test -- flagged here so
Task 2 (or whoever adds the covering test) knows to wire `protect` through
before relying on it.

---

# Task 2: the v12-format writer, the extractor CLI, and the yield census

Date: 2026-09-05. Code: `jarvis_jax/data/pseudo_export.py`,
`scripts/pseudo_labels/{extract_p3b_pseudolabels,census_diagnose}.py`,
`scripts/viz/pseudo_export_check.py`.

## The census STOPS the export (controller ruling 2026-09-05)

Full scan of all 160 campaign bouts / 11 recordings (620 s, 0 bouts skipped),
`GateThresholds()` defaults, one row per admitted (anchor frame, fly) --
which is what a v12 frameset, and therefore a training window, is:

| recording | F/contact | F/apart | F/mid | M/contact | M/apart | M/mid | total |
|---|---|---|---|---|---|---|---|
| 2025_10_20_13_20_04 | 0 | 434 | 0 | 0 | 1404 | 187 | 2025 |
| 2026_04_02_12_11_50 | 0 | 67 | 0 | 0 | 284 | 19 | 370 |
| 2026_04_02_14_54_28 | 0 | 192 | 0 | 0 | 649 | 7 | 848 |
| 2026_04_02_15_25_51 | 0 | 432 | 0 | 0 | 1053 | 3 | 1488 |
| 2026_04_02_15_44_42 | 0 | 100 | 0 | 0 | 226 | 0 | 326 |
| 2026_04_02_16_03_48 | 0 | 42 | 0 | 0 | 239 | 0 | 281 |
| 2026_04_02_16_21_32 | 0 | 75 | 0 | 0 | 558 | 0 | 633 |
| 2026_04_02_16_39_56 | 0 | 208 | 0 | 0 | 654 | 5 | 867 |
| 2026_04_02_16_56_37 | 0 | 86 | 0 | 0 | 178 | 0 | 264 |
| 2026_04_02_17_28_34 | 0 | 191 | 0 | 0 | 473 | 2 | 666 |
| 2026_04_02_17_52_50 | 0 | 78 | 0 | 0 | 122 | 0 | 200 |
| **TOTAL** | **0** | **1905** | 0 | **0** | **5840** | 223 | **7968** |

("mid" = the other fly is missing on that frame, so the pair separation is
undefined; `--contact-units 15`.)

Two spec §3.2 targets are unreachable as written, for two DIFFERENT reasons:

1. **Female-host supply is 1,905 anchors, below the 3,000 floor** -- so the
   run stopped after writing `census.json` rather than rebalancing onto the
   male. A 50/50 export could be at most 3,810 framesets, not 10,000.
2. **The contact stratum is EMPTY -- and it is the definition, not the
   gates.** `figures/2026-09-mvq/v2_pseudo/census_diagnosis.png` (33 bouts,
   3 per recording): over EVERY frame with both flies finite, the minimum
   inter-fly CENTROID separation is ~20 units; 0 of 18,295 frames are under
   the 15-unit (1.5 mm) cut, and the admitted anchors' separation histogram
   is indistinguishable from all frames' (1.1/19.5/62.9/15.0 % vs
   1.3/20.8/63.5/13.0 % in the 20/25/30/40-unit bins). Two fly centroids
   cannot approach to 1.5 mm because a fly is ~2.5-3 mm long. The physical
   measure does show contact in abundance: the MINIMUM inter-fly KEYPOINT
   distance is under 5 units (0.5 mm) on **32.9 %** of all frames and
   **31.3 %** of admitted anchors -- again gate-neutral, and comfortably
   above the >= 25 % contact target. Recommendation: redefine `contact` as
   min inter-fly keypoint distance < 5 units.

Why the female is thin (same 33 bouts, per FLY, fraction of fly-frames
rejected by each gate; a frame can fail several):

| gate | female | male |
|---|---|---|
| exist | 0.098 | 0.000 |
| step | 0.204 | 0.127 |
| reproj | 0.107 | 0.004 |
| **contain** | **0.724** | 0.178 |
| nonfinite | 0.120 | 0.098 |
| identity | 0.000 | 0.000 |
| **admitted** | **24.5 %** | **69.7 %** |

Containment dominates, and it is NOT the ">= 5 valid mask cameras" clause:
spot-checking 4 bouts, both flies have 6-7 valid mask cameras on essentially
every frame, and it is the FRACTION that falls short -- the female's 5th-best
per-camera containment runs 0.60-0.86 against the 0.90 threshold (bout 4 of
20_04: female median 0.600, male 0.960). The female's pose genuinely sits
partly outside her own SAM3 mask, exactly the case Task 1's docstring says
the gate is meant to reject. Relaxing `contain_frac` would buy female volume
by admitting the least trustworthy poses of the hardest fly.

Other census facts:
- Partner availability is high: 7,578 / 7,406 / 7,370 of 7,968 anchors have a
  Delta = 1 / 4 / 16 partner (95 / 93 / 92 %).
- The per-bout 2 % cap would not have bound: the largest bout supplies 170
  anchors against a cap of 200.
- Wall frames are essentially absent from the campaign (7 of 7,968, in two
  recordings), so "wall frames in proportion" is satisfied trivially and wall
  diversity has to keep coming from the real v12 wall subsets.

## The writer

`write_pseudo_export` produces the v12 layout `V12WindowDataset` already
reads -- `annotations/instances_train.json` (+ an intentionally EMPTY
`instances_val.json`), `annotations/keypoint_names.json`,
`images/<rec>/<cam>/Frame_<n>.jpg`, `masks/<rec>/<cam>/Frame_<n>.npz`,
`calibrations/<group>/Cam*.yaml` symlinks, `manifest.json` -- plus the
pseudo-only fields (`source`, `checkpoint`, `gates`, `weight`, `role`,
`stratum`, `partners`, `bout`).

Decisions worth recording:
- **2D labels are the reprojection of the gated 3D.** The loader never reads
  a 3D field; it triangulates the written 2D. Writing the model's own 2D head
  would hand training a pose up to `reproj_px` (3 px) away from the one that
  passed the gates. `test_the_loaders_dlt_reproduces_the_gated_3d_in_EXPORT_
  keypoint_order` asserts the round trip to 1e-3 units through the real
  loader; the projection uses the cameras' float64 matrices, not the float32
  `camera_matrices`, so the DLT inverts exactly what was projected.
- **One frameset per (frame, host fly), written only for flies that were
  SAMPLED.** Frameset count == window count, so the stratified draw is what
  the trainer sees; an admitted-but-unsampled fly stays a present-but-
  unlabelled animal, which the loader's `unlabelled_sex` already handles.
- **A fly visible in fewer than 2 cameras is dropped, not written** -- the
  loader's DLT would give it no 3D and centre its window on the origin.
- **JPEGs are written with cv2 (2.2 ms vs PIL's 8.0 ms per 1936x448 frame)**,
  which takes BGR; `test_written_jpeg_round_trips_the_rgb_channels` guards the
  flip, since a swapped export would train on blue flies invisibly.
- **Video reads decode THROUGH gaps up to 300 frames instead of seeking**: a
  `cap.set(POS_FRAMES)` costs ~195 ms on these H.264 files against 0.6 ms for
  a sequential read (measured on 20_04/Cam2012630), which is the difference
  between ~1.4 s and ~40 ms of video time per written frameset.

## Read the export back

`scripts/viz/pseudo_export_check.py --root <export> --out figures/.../check.png`
renders sampled framesets (female host, contact, wall, partner, random) in
two cameras each with the written keypoints by group colour, the leg chains,
and the annotation's own SAM3 mask outline. On the 2-bout real dry run
(`OutFiles/pseudo_dryrun2`, 20_04 bouts 4-5, 23 framesets / 161 images) the
panels show head points on the head, abdomen points on the abdomen, wing
points at the wing tips, leg chains running outward from the thorax, the mask
outline wrapping the same animal, and natural fly colouring (a BGR swap would
have shown blue flies on an orange floor).

## Round 2: decisions applied (2026-09-05)

User + controller rulings, all implemented in
`scripts/pseudo_labels/extract_p3b_pseudolabels.py`:

1. **`contact` := minimum inter-fly KEYPOINT distance < 5 units**
   (`--contact-kp-units`, default 5.0). The spec's centroid rule stays in the
   code as `--contact-units` and is recorded per frameset
   (`stratum.sep_units`) and in `census.json`
   (`by_stratum_centroid`, `by_host_sex_cell_centroid`) so the measurement
   that retired it stays visible. Guarded by
   `test_contact_is_the_min_keypoint_distance_not_the_centroid_separation`.
2. **Gates unchanged; the two sides are drawn separately.** The female side
   takes EVERY admissible anchor (`--female-n` unset). The male side is capped
   (`--male-n 4000`) and drawn round-robin over recordings with a hard
   per-recording share (`--per-rec-frac 0.20`) and per-bout cap
   (`--per-bout-cap 200`), >= 25 % contact and >= 25 % apart where the yield
   allows. The bout counter is PER SIDE (the ruling's "the 2 % rule applies
   per side"): sharing it would let the uncapped female side spend the male
   side's per-bout budget before the male side is drawn at all.
   `test_the_caps_bound_a_drawn_side` covers both caps on a two-recording
   fixture.
3. **The export is deliberately not 50/50** and says so: `manifest.balance`
   carries `female_n`, `male_n` and
   `female_host_weight = male_n / female_n`, the multiplier that restores
   50/50 at training time (`train.female_host_weight`).
4. **Post-write verification runs on the REAL export** (`verify_export`,
   skippable with `--no-verify`): `V12WindowDataset` opens the written root,
   reads 20 random windows and asserts the keypoint axis matches
   `annotations/keypoint_names.json`, the 2D/3D are finite, the host has
   triangulated 3D, >= 2 cameras are valid and the crops are not all black
   (an unwritten or misnamed JPEG). `summary.md` beside the manifest carries
   the per-side, per-stratum, per-recording and partner tables.

Two writer changes came out of this round:
- JPEGs are written only for cameras some frameset actually resolves (the
  loader never opens the others), which keeps a ~150k-file tree from carrying
  dead frames.
- `write_pseudo_export` now REFUSES masks whose (H, W) differs from the
  frame's: the loader slices a mask with the image's crop origin, so a
  differently-sized mask is not a smaller mask, it is a mask of somewhere
  else.

## Task 5: empty-window negatives (2026-09-06)

`scripts/pseudo_labels/extract_empty_windows.py` (+ `tests/test_empty_windows.py`),
spec §3.5. Two centroid sources per recording (controller ruling 2026-09-05,
extended by a coordinator ruling 2026-09-06 for T=2 partners):

- **(a) `--tracks` (mask-free coarse pass)**: only Session0/2025_10_20_13_20_04
  has a `coarse_tracks.npz`. Arena-edge candidates are CenterDetect reads with
  `exist < 0.2` near the tracked-centroid convex hull -- the real file has
  **zero** such reads (`exist` ranges 0.500-1.000, never below): `MVQRunner.
  read_typed`/`coarse_pass` treat a below-threshold read as "no read" and never
  write it, so the file this pipeline produces cannot carry the low-confidence
  "false peak" rows the spec's edge stratum was written against. This is
  reported as a CONCERN below, not silently worked around; the edge_frac
  machinery is intact and will populate itself the moment a source ever
  carries such rows (verified with a synthetic file in
  `test_edge_frac_draws_from_the_false_peak_band`).
- **(b) `--lift-root` (masked p3b bout lifts)**: per-frame centroid = nanmean
  over keypoints of `kp3d.npz`, placed at the bout's `frame_start`, only
  inside bouts. Adjacent campaign bouts can overlap by a handful of frames at
  their boundary (measured: `2026_04_02_15_25_51` bout_00013/14 share 6
  frames, `2026_04_02_16_21_32` bout_00011/12 share 8) -- kept the FIRST
  bout's read for the shared frames and counted them, rather than aborting
  the recording.
- Newer `pose_mvq_p3b` used where its bout count matched or exceeded the
  `pose_mvq_p3b.pre_ownwindow_0905` sibling (Session0 + the first 3 Session1
  recordings, all fully re-lifted); the `pre_ownwindow_0905` root used
  otherwise (Session1 lifts still re-running at run time; 3D is identical
  between the two, only confidences differ).

### T=2 partners (coordinator ruling 2026-09-06)

Every drawn negative ("anchor") also gets partner framesets at Delta in
{1, 4, 16} video frames FORWARD (`f0 + delta`), same world centre, written
only where the SAME clearance gates (>= 60 units from every tracked centroid,
>= 6 units above the floor, inside >= 5 cameras) hold AT THE PARTNER'S OWN
FRAME -- never assumed from the anchor. A delta that fails is omitted, never
guessed; `_link_partners` also refuses a partner frame already claimed by
another negative or partner (`collision`). The anchor's frameset carries
`partners: {"1": frame, "4": frame, "16": frame}` for whichever deltas
cleared (`pseudo_export`'s existing convention, unmodified); partners are
written as independent framesets with `role: "partner"`, so
`write_pseudo_export`'s own `per_role`/`partner_availability` summaries
separate anchors from partners with no changes to that module.

**Stride-16 gap on source (a).** `2025_10_20_13_20_04`'s tracked-centroid
coverage (`frame_to_cent`) only exists at the coarse-pass's own stride-16
samples, so `f0+1` and `f0+4` are NEVER a known frame there --
`_clears_gates` returns `no_coverage` for both, every time (201/201 anchors,
both deltas). Only `f0+16` lands exactly on the next coarse sample and
resolves (195/201, the rest lost to `dist`/one `collision`). This is a
genuine data-coverage limit of the mask-free coarse pass, not a bug --
confirmed both by the real run's numbers and by a dedicated unit test
(`test_partners_resolve_and_share_the_anchor_centre`) that asserts delta=16
is exactly the one that resolves on a stride-16 synthetic source. Every
`--lift-root` recording has PER-FRAME coverage inside its bouts, so all
three deltas mostly resolve there (65-99 % taken per delta; `collision` --
two negatives landing within 16 frames of each other -- is the next largest
loss, `no_coverage` mostly means "delta lands outside every bout").

### Real run (2026-09-06)

```
python scripts/pseudo_labels/extract_empty_windows.py \
  --tracks 2025_10_20_13_20_04=.../coarse_mvq_p3b/coarse_tracks.npz \
  --lift-root <rec>=.../pose_mvq_p3b[.pre_ownwindow_0905] (x10) \
  --out /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_negatives_20260905 \
  --n 2000
```

11 recordings, ~97 min wall time (video-seek-bound: a random frame across a
~500k-frame recording costs ~0.4-2.5 s per camera under this run's I/O
contention with a concurrent campaign export). **2,000 anchor negatives +
5,312 T=2 partners = 7,312 framesets, 51,184 images.** `per_role`:
`{"negative": 2000, "partner": 5312}`. `partner_availability`:
`{"1": 1729, "4": 1721, "16": 1862}` (out of a 6,000-delta ceiling, 3 per
anchor -- the stride-16 recording's 402 structural `no_coverage` losses on
deltas 1/4 account for most of the gap from ceiling).

| recording | source | anchors | partners | shortfall | dominant rejection |
|---|---|---:|---:|---:|---|
| 2025_10_20_13_20_04 | tracks | 201 | 195 | 0 | dist/height/cams even |
| 2026_04_02_12_11_50 | lift_root | 200 | 544 | 0 | dist |
| 2026_04_02_14_54_28 | lift_root | 200 | 582 | 0 | dist |
| 2026_04_02_15_25_51 | lift_root | 200 | 585 | 0 | dist |
| 2026_04_02_15_44_42 | lift_root | 200 | 557 | 0 | dist |
| 2026_04_02_16_03_48 | lift_root | 200 | 566 | 0 | dist |
| 2026_04_02_16_21_32 | lift_root | 200 | 583 | 0 | dist |
| 2026_04_02_16_39_56 | lift_root | 200 | 584 | 0 | dist |
| 2026_04_02_16_56_37 | lift_root | 200 | 549 | 0 | dist |
| 2026_04_02_17_28_34 | lift_root | 199 | 567 | 0 | dist |
| 2026_04_02_17_52_50 | lift_root | **0** | 0 | 181 | dist+height (see below) |

**2026_04_02_17_52_50 legitimately yields zero.** Its own tracked-centroid
bounding box is only 5.6 units TALL (z: 8.58-14.17), below `--min-height-units`
(6) by itself -- a real property of this recording (3 short bouts, the fly
never climbed), not a bug: confirmed by inspecting `RecData.bbox_hi -
bbox_lo` directly (`[164.7, 36.8, 5.58]` vs 60-90 units tall for every other
recording). Its 181-anchor shortfall was redistributed to the 10 recordings
that hit their own quota with room to spare (one bumped-quota re-draw per
donor, `bonus = shortfall // n_donors`), landing the total exactly at the
requested 2,000.

**Verify** (`verify_negative_export`, the negative-specific round trip --
`extract_p3b_pseudolabels.verify_export` asserts a real host's triangulated
3D, which is false by construction for every window here): 20/7312 windows
sampled, `is_negative` True, `fly_valid` all-False, finite `center3D`, mean
7.0/7 valid cameras, non-black crops -- `{"all_negative": true}`.

**Figure** (`figures/2026-09-mvq/v2_pseudo/negatives_check.png`, read back
with the Read tool): 12 panels, all "interior" stratum (the edge stratum's
pool is 0 -- see the exist<0.2 finding above), spanning 6 different
recordings and both cameras per negative. Every panel shows bare textured
arena floor or the dark arena-wall/edge band; NO fly body, wing, or leg is
visible in any panel. Matches the stated expectation cleanly.

### Concerns / deviations

1. **Edge stratum never populated from a real `coarse_tracks.npz`.** The
   file's `exist` field never goes below `TRACKABLE_EXIST` (0.5) because
   `MVQRunner.read_typed` treats a sub-threshold read as "no read" and never
   writes it -- there is no low-confidence row LEFT to mine for false peaks.
   The `--edge-frac`/`--exist-edge-thresh`/`--edge-units` machinery is
   implemented and tested against a synthetic file that DOES carry such
   rows, but produced 0 edge windows on the real run. A real edge/false-peak
   stratum needs either a lowered existence-write threshold in the coarse
   pass itself, or a separate CenterDetect-peaks-only source -- out of this
   task's scope. **RESOLVED 2026-09-06** by the CenterDetect-peaks source; see
   "Fix round 1" at the end of this file for the replacement, the two rules
   that make it honest, and the measured 0.36 % false-peak rate.
2. **T=2 partners are forward-only** (`f0+delta`), unlike the positive
   pseudo-labels' bidirectional endpoint search -- a coordinator
   simplification (an empty window has no motion to prefer a direction
   from), not an oversight.
3. **`pseudo_export.write_pseudo_export` always keys a negative frameset
   `<rec>/Frame_<n>/neg0`** (not `neg<k>`, despite the brief's illustrative
   `neg<k>` wording) -- harmless here because every negative (anchor or
   partner) this script emits already has a UNIQUE frame per recording
   (enforced by `used_frames`), so no two ever collide on one key; flagged
   in case a future caller assumes `neg<k>` numbering.
4. Wall time (~97 min for 51,184 images) was video-seek-bound under I/O
   contention from a concurrent campaign export on the same filesystem; a
   quieter node should be substantially faster (uncontended single-seek cost
   measured at ~0.4 s/frame/camera on this recording's own mp4s).

## Real export (post own-window re-lift), 2026-09-06

Everything above this heading was measured on the FIRST P3b lift, which had a
per-view visibility defect. The campaign was re-lifted with the own-window fix
(`mvq_meta.gates.window_pref = "own"`, same lifter checkpoint
`mvq_t1_b16_p3b_contact_20260905/final`, sha `b54cdaef64696345`) and the old
trees were renamed `pose_mvq_p3b.pre_ownwindow_0905`. The extractor's
discovery glob is the exact name `.../courtship/*/*/pose_mvq_p3b`, so the
renamed siblings are invisible to it; `manifest.source_roots` lists the 11
`pose_mvq_p3b` roots it actually read and `n_source_bouts` 160, which is the
whole re-lifted campaign (160 bouts x 2 flies = 320 bout-flies, LOO median
0.60-0.68 px per recording in `pose_mvq_p3b/qc/session_qc.json`).

The stale export (17,812 framesets drawn from the defective lift) was deleted
and the root re-created from scratch.

### Census: the own-window fix roughly DOUBLES the female-host yield

Same gates, same contact definition, same 160 bouts, 0 bouts skipped in either
run. One row = one admitted (anchor frame, host fly) = one v12 frameset.

| recording | F pre | F post | M pre | M post |
|---|---|---|---|---|
| 2025_10_20_13_20_04 | 434 | 530 | 1591 | 1679 |
| 2026_04_02_12_11_50 | 67 | 157 | 303 | 324 |
| 2026_04_02_14_54_28 | 192 | 462 | 656 | 692 |
| 2026_04_02_15_25_51 | 432 | 883 | 1056 | 1121 |
| 2026_04_02_15_44_42 | 100 | 278 | 226 | 306 |
| 2026_04_02_16_03_48 | 42 | 179 | 239 | 291 |
| 2026_04_02_16_21_32 | 75 | 500 | 558 | 612 |
| 2026_04_02_16_39_56 | 208 | 434 | 659 | 695 |
| 2026_04_02_16_56_37 | 86 | 139 | 178 | 208 |
| 2026_04_02_17_28_34 | 191 | 492 | 475 | 601 |
| 2026_04_02_17_52_50 | 78 | 153 | 122 | 149 |
| **TOTAL** | **1,905** | **4,207** (+121 %) | **6,063** | **6,678** (+10 %) |

No recording lost anchors on either side. The female-host stratum is now above
the controller's 3,000 floor on its own, so the census no longer stops the
export; the run was still given `--min-female 1905` so that a yield BELOW the
pre-fix number would have halted it before any frame was decoded.

Post-fix cells (contact := min inter-fly KEYPOINT distance < 5 units):
female 1,107 contact / 3,100 apart; male 1,476 contact / 4,956 apart / 246 mid
(the other fly missing, separation undefined). The spec's original CENTROID
rule still admits nothing anywhere (`by_host_sex_cell_centroid`: 4,207 female
and 6,432 male all "apart"), exactly as before the re-lift -- the fix did not
change that verdict. Partner availability 10,704 / 10,661 / 10,595 of 10,885
anchors at Delta = 1 / 4 / 16 (98.3 / 98.0 / 97.3 %, up from 95 / 93 / 92 %).
Wall anchors 6 in the whole census (2 recordings), so wall diversity still has
to come from the real v12 wall subsets. Aggregate gate rejections over 238,558
fly-frames are unchanged in shape -- containment 55,048, step 33,925, nonfinite
21,710, reproj 18,465, exist 8,314, identity 0 -- containment is still the
dominant filter, it just rejects far fewer female frames than it did.

### Draw and export

Decided parameters, unchanged: every admissible female-host anchor, male-host
capped at 4,000 with per-recording <= 20 % and per-bout <= 200, partners at
Delta 1/4/16 with endpoint semantics, loss weight 0.3, seed 0.

| side | pool | taken | contact | apart | wall | max recording share | max bout |
|---|---|---|---|---|---|---|---|
| female | 4,207 | 4,207 (all) | 1,107 (26.3 %) | 3,100 (73.7 %) | 6 | 21.0 % | 131 |
| male | 6,678 | 4,000 (capped) | 1,172 (29.3 %) | 2,752 (68.8 %) | 0 | 11.4 % | 100 |

No shortfalls on either side (both >= 25 % contact and >= 25 % apart came out
of the pool, not out of a relaxation). The male side respected both caps
(11.4 % <= 20 %, 100 <= 200); the female side is `mode: all`, so its 21.0 %
recording share is the uncapped side's own composition, not a cap violation.

**`manifest.balance.female_host_weight` is now 0.9508** (= 4,000 / 4,207), not
the 2.0997 of the stale export: the female side has grown to the size of the
male cap, so the export is very nearly 50/50 by itself. `train.female_host_weight`
in `configs/train/mvq_v2.yaml` (4.27 at the time of writing) has to be reset to
this number, or the trainer will over-weight female-host windows by ~4.5x.

Totals: **25,382 framesets** (8,207 anchors + 17,175 partners), **127,001
images**, 177,674 annotations, 177,674 mask rows, 0 dropped for < 2 cameras.
Realised composition: female-host 51.3 %, contact 27.8 %, apart 71.3 %, wall
0.07 %, largest single bout 2.63 % of the export, all 11 recordings present.
Per-stratum framesets: female/apart 9,246, female/contact 3,248, male/apart
9,065, male/contact 3,552, male/mid 271. Partner framesets per delta: 8,078 /
8,051 / 7,989 at Delta = 1 / 4 / 16.

On disk: **42 GB** (images 26 GB, masks 16 GB, annotations 273 MB), written in
3,666 s (61 min) single-process on g3102 under load from another user's job;
the census scan before it took 660 s.

**Verify** (`extract_p3b_pseudolabels.verify_export`, the loader round trip on
the real export): the in-run check passed 20/25,382 windows, and a second run
with a larger sample and a different seed (n=100, seed=7) passed too --
`{"windows_checked": 100, "n_windows": 25382, "mean_kp_with_3d": 49.83,
"mean_valid_cameras": 7.0, "keypoint_names_ok": true}`. Every sampled window
has finite `kp3d_local`/`center3D`/`kp2d`, a host fly with triangulated 3D,
7/7 valid cameras and non-black crops, and the instances' keypoint axis equals
`annotations/keypoint_names.json`.

**Figure** `figures/2026-09-mvq/v2_pseudo/ownwindow_export_check.png` (12
panels, `scripts/viz/pseudo_export_check.py`, read back with the Read tool).
Expectation stated first: on a re-lifted female-host or contact frameset the
written keypoints must sit ON the host fly -- red head points at the head,
magenta abdomen points on the abdomen, yellow wing points at the wing tips,
cyan leg chains running outward from the thorax -- with the annotation's own
SAM3 outline wrapping THAT animal; if the own-window re-lift had broken host
assignment, a female-host panel would show the skeleton on the male or split
between the two flies. What the panels show: `[female host]`
15_44_42/Frame_301155/fly0 (contact, kpdist 4.56 u) and `[female+contact]`
Frame_420504/fly0 (kpdist 5.00 u) are anatomically correct in both Cam2012630
and Cam2012631, outline on the same fly; `[male+contact]`
16_03_48/Frame_465128/fly1 at kpdist **0.28 u** -- the two flies touching, the
partner's wing inside the crop -- keeps every keypoint on the outlined host;
`[wall]` 20_04/Frame_207422/fly0 is correct on Cam2012630 and shows only 14 of
50 keypoints on Cam2012631, which is the wall view's genuine occlusion (the
export needs >= 2 valid cameras, not 7); `[partner]` and `[random]` panels are
correct on both cameras. Fly colouring is natural (a BGR swap would show blue
flies on an orange floor).

### Exact commands

```bash
rm -rf /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_p3b_20260905
mkdir -p /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_p3b_20260905

# census only (660 s, 160 bouts, no frame decoded)
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 nice -n 10 python scripts/pseudo_labels/extract_p3b_pseudolabels.py \
    --out /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_p3b_20260905 --census-only

# draw + export + verify (3,666 s)
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 nice -n 10 python scripts/pseudo_labels/extract_p3b_pseudolabels.py \
    --out /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_p3b_20260905 \
    --male-n 4000 --per-rec-frac 0.20 --per-bout-cap 200 --contact-kp-units 5.0 \
    --weight 0.3 --seed 0 --min-female 1905 --figures-dir figures/2026-09-mvq/v2_pseudo

# read it back
python scripts/viz/pseudo_export_check.py \
    --root /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_p3b_20260905 \
    --out figures/2026-09-mvq/v2_pseudo/ownwindow_export_check.png --n 6 --cams 2
```

### Concerns

1. **`female_host_weight` moved from 2.0997 to 0.9508.** Any config or run
   that hard-codes the old value (`configs/train/mvq_v2.yaml` has 4.27) now
   over-weights the female host. The number lives in
   `manifest.balance.female_host_weight` -- read it from there.
2. **The export cannot be sharded by recording.** `extract_p3b_pseudolabels.py`
   has no `--only`/`--recordings` flag, and `--runs` shards DISCOVERY, which
   would break the global draw (the male side's per-recording 20 % share, the
   balance weight and the single manifest). It was therefore run
   single-process; 61 min for 127,001 images is acceptable and no code change
   was made.
3. **Male anchors are still capped at 4,000 out of a 6,678 pool** while the
   female side takes all 4,207. Raising the male cap would need a matching
   rethink of the balance weight; left at the decided value.
4. **Wall frames remain vanishingly rare** (6 anchors campaign-wide, 0.07 % of
   the export). Wall/OOD diversity has to keep coming from the real v12 wall
   subsets -- the pseudo set does not supply it.

## Review gallery (2026-09-06)

Generated the real human-review gallery from the P3b export (spec §3.3 /
§6 item 2), per task-3 brief Step 4.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python scripts/pseudo_labels/pseudolabel_gallery.py \
    --export /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_p3b_20260905 \
    --n 300 --out figures/2026-09-mvq/v2_pseudo/gallery
```

Wrote 30 pages (`page_00.png`..`page_29.png`, 10 framesets/page, overhead
`Cam2012630` left / side `Cam2012855` right) and `gallery/review.csv` (300
rows, header as spec'd, every `verdict` blank). Draw is anchors only
(`--include-partners` not passed), stratified over
`(host_sex, contact, wall)` with the script's default seed.

### Draw composition

Population (anchors, from the run's own per-cell pool sizes) vs. what the
n=300 stratified draw actually pulled (from `review.csv`):

| cell | pool | drawn |
|---|---:|---:|
| female/apart/floor | 3,094 | 113 |
| male/apart/floor | 2,828 | 103 |
| female/contact/floor | 1,107 | 40 |
| male/contact/floor | 1,172 | 43 |
| female/apart/wall | 6 | 1 |
| **total** | **8,207** | **300** |

**The wall cell is under-represented relative to the brief's stated
expectation.** The brief text said "the 6 wall anchors should all be in";
`stratified_alloc`'s actual guarantee (per its own docstring) is only a
proportional floor share plus a minimum of 1 per non-empty cell when there is
room, clipped to the cell's own pool -- not full inclusion of a small pool.
With `wall` pool = 6 out of 8,207 anchors and `n = 300`, the proportional
share floors to 0, the minimum-1 rule bumps it to 1, and the remainder-round
that fills out the rest of the 300 goes to cells with far larger raw
remainders (the four floor/contact cells, each in the hundreds to
low-thousands). So only 1 of the 6 wall anchors landed in this sample
(`2025_10_20_13_20_04/Frame_392159/fly0`, page 11) -- this is a gap between
the task's framing and the implemented allocator, not a bug in the draw
itself; flagging it rather than silently treating "1 of 6" as satisfying
the brief.

### Page-by-page read-back

Six pages read with the Read tool: two contact/female, one contact/male, one
apart/female, one apart/male, and the page holding the one wall anchor drawn
(no page in this 300-sample holds more than one wall anchor). Expectation
(from the script's docstring / CLAUDE.md): every panel shows ONE fly's
keypoints on that fly's own body inside its own grey mask outline in both
cameras; head red at the antennae, abdomen magenta at the tail, leg chains
not crossing to the partner fly; a straddling skeleton or one on the arena
floor is a reject.

- **page_05 (female/apart/floor, pure, 10/10 panels)** -- recordings
  `2026_04_02_15_44_42` / `2026_04_02_16_03_48`. Every panel: the female host
  (cyan label) sits fully inside her own white mask outline in both cameras,
  red head-cluster at the antennae end, magenta abdomen at the tail, yellow
  thorax/wing points, blue leg rays radiating outward without reaching the
  separate unlabeled partner fly a body-length or more away. 0/10 suspicious.
- **page_18 (male/apart/floor, pure, 10/10 panels)** -- recordings
  `2026_04_02_15_25_51` / `2026_04_02_15_44_42`. Same pattern with the male
  host correctly in orange text; keypoints and mask contour track only the
  host in every panel, partner fly clearly separate. 0/10 suspicious.
- **page_12 (female/contact/floor, pure, 10/10 panels)** -- recordings
  `2026_04_02_14_54_28` / `2026_04_02_15_25_51`. Keypoints/mask stay on the
  host only in all 10 panels, correct head/abdomen colors, no crossing to the
  partner. Worth flagging for interpretation, not as a reject: despite the
  `contact` label, the two flies in these panels sit with a visible ~1
  body-length gap in most frames (`sep` 28-32u) rather than touching bodies
  -- "contact" here reads as a proximity/pose-similarity threshold, not
  literal body contact. 0/10 suspicious.
- **page_27 (male/contact/floor, pure, 10/10 panels)** -- recordings
  `2026_04_02_15_25_51` / `2026_04_02_15_44_42` / `2026_04_02_16_03_48`. Same
  containment pattern as page_12 (its male counterpart): host-only
  keypoints/mask, correct anatomy colors, partner separate. 0/10 suspicious.
- **page_13 (female/contact/floor, pure, continuation of page_12's cell,
  10/10 panels)** -- recordings `2026_04_02_16_03_48` / `2026_04_02_16_21_32`.
  Re-examined at 2x zoom (rows 6-10, including the closest-proximity frames
  on the page) specifically because two flies overlap in several of these
  frames; in every case the host mask/keypoints stay bounded to the host's
  own dark body and the partner fly (visible, unlabeled, semi-transparent
  wings) remains outside the mask outline even where the two bodies nearly
  touch on screen. 0/10 suspicious.
- **page_11 (the wall page: mixed female/apart/floor + the one drawn
  female/apart/wall anchor + female/contact/floor, 10/10 panels)** --
  recordings `2026_04_02_17_52_50` (3 apart/floor), `2025_10_20_13_20_04` (the
  1 wall anchor + 2 contact/floor), `2026_04_02_12_11_50` (2 contact/floor),
  `2026_04_02_14_54_28` (1 contact/floor). The 9 non-wall panels match every
  other page (host-only containment, correct anatomy colors). The wall panel
  (row 4, `2025_10_20_13_20_04/Frame_392159/fly0`, sep=107.7u) is SUSPICIOUS:
  in the overhead camera (`Cam2012630`) the keypoints/mask sit correctly on
  the host's own body against the wall; but in the side camera
  (`Cam2012855`) the keypoint cluster renders cut off in the bottom-right
  corner of the panel, disconnected from the only visible fly body in that
  frame (which itself carries no keypoints). Flagging as a likely reject --
  either the side-camera projection places the host's own keypoints off the
  visible body for this steep wall geometry, or the visible fly in that view
  is not actually the host frame's fly.

### Suspicious panels (pre-screen only, no verdicts written)

1. `2025_10_20_13_20_04/Frame_392159/fly0` (female/apart/wall, page 11,
   row 4) -- side camera (`Cam2012855`) keypoints render disconnected from
   the visible fly body, cut off in the panel's bottom-right corner. Likely
   reject candidate; the overhead camera for the same frameset looks correct.

That is 1 suspicious panel out of the 60 framesets / 120 camera panels read
back across the 6 pages above. No verdicts were filled into `review.csv` --
per task instruction, that pass is Elliott's.

### Handoff

`review.csv` (300 rows, `verdict` blank) is versioned at
`docs/benchmark/2026-09-mvq/v2-review/review.csv`, byte-identical to the
generated
`figures/2026-09-mvq/v2_pseudo/gallery/review.csv` (gitignored, regenerate
with the command above). Pages are under
`figures/2026-09-mvq/v2_pseudo/gallery/page_00.png`..`page_29.png` (not
committed). After Elliott fills in `verdict` (`accept`/`reject`) --
including a decision on the one suspicious wall panel above -- apply with:

```bash
python scripts/pseudo_labels/pseudolabel_gallery.py \
    --export /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_p3b_20260905 \
    --apply-review figures/2026-09-mvq/v2_pseudo/gallery/review.csv
```

(point it at the reviewed copy of the CSV -- either the gallery-dir original
once edited in place, or a copy of `docs/benchmark/2026-09-mvq/v2-review/review.csv`
with verdicts filled in, so long as the header/row order from generation is
preserved). Per spec §3.3: overall reject fraction > 3% aborts and writes
nothing (exit 2, with a per-gate breakdown of the rejected rows printed for
tightening); otherwise it drops rejected framesets plus any whole stratum
whose own reject rate exceeds 3%, and gates training on the resulting
`review_summary.json`.

### Fix round 1 (2026-09-06): edge negatives from CenterDetect false peaks

The committed run's edge stratum mined `coarse_tracks.npz` for reads with
`exist < 0.2` and drew **zero** (`MVQRunner.read_typed` returns None below
`exist_thresh`, so a sub-threshold read is never written; the real file's
`exist` never leaves 0.500-1.000). Replaced with the source the spec actually
asks for -- CenterDetect run over sampled frames, its peaks lifted with
`coarse_centres.lift_peaks_to_centres`, the centres the two real flies do not
explain kept as candidates. Two rules make that honest, and both were
measured, not assumed:

1. **`--cd-peaks` must exceed 2.** `CenterDetector.peaks` hard-codes k=2 and
   with two flies in frame those two peaks per camera ARE the flies, so
   `max_animals >= 3` over that array returns nothing new: on
   `2026_04_02_16_21_32`, 32/32 centres lifted that way sat within 12 units
   of a tracked fly. `CenterDetectPeaks` runs the SAME checkpoint,
   preprocessing (`_centerdetect_preprocess`, PIL BILINEAR) and decode with
   k=8, and `false_peak_centres` then does the production top-2 read first,
   consumes every peak it explains, and re-lifts the survivors round by round
   (`_compact_peaks` exists because `_seed_candidates` seeds only from peak
   slots 0/1). Yield saturates at k=8 -- k=16 and k=24 return the identical
   candidate set, `extract_top_k_peaks(suppression_radius=15)` cannot find
   more separated maxima on a 160x160 heatmap.
2. **The scan skips frames where any fly is untracked** (`--edge-allow-untracked`
   opts out; ON by default). Without it, a 300-frame scan of
   `2025_10_20_13_20_04` returns 26 candidates -- and **25 of them sit on
   frames where the female is untracked, every one 2.3-39.3 units (median
   11.9) from HER OWN nearest tracked position before/after the gap**. They
   are her, re-found by CenterDetect after the typed read dropped her. Writing
   those as "no fly here" would have trained the existence head to suppress
   the hardest fly in the dataset -- the exact failure the figure exists to
   catch, arriving through a stratum labelled "arena edge". Cost of the rule
   is small: 97-100 % of the lift-root recordings' in-bout frames have both
   flies, and 44 % (13,652/31,125) of 20_04's coarse frames.

Both strata now go through `_clears_gates` in `sample_negatives` (the review's
IMPORTANT finding: the edge loop called `_accept` directly, so the gates that
define "no fly in this window" were enforced for interior draws only).
`test_edge_frac_draws_from_the_false_peak_band` re-checks the distance /
height / camera rule on edge rows against the recording's own trajectory, and
`test_edge_candidate_on_a_fly_is_rejected_by_the_gates` feeds a pool whose
rows sit exactly on a tracked fly and asserts all 20 are rejected on `dist`.
`--edge-units` changed meaning with the source: it now caps how far OUTSIDE
the tracked-centroid hull a candidate may sit (signed distance,
`_hull_signed_distance`), default 100 units; a candidate INSIDE the hull is an
empty-floor window and is never rejected for it.

**Measured false-peak rate: 28 qualifying candidates in 7,808 scanned frames
(0.36 %).** 12,218 residual centres were lifted; 12,140 (99.3 %) were rejected
on distance -- they are secondary CenterDetect responses on the flies' own
bodies, 2.7-32 units away -- 46 on height, 4 on cameras, 0 outside the hull.
So a 25 % edge share is not reachable at any quota here: it would need ~140k
scanned frames (~12 GPU-hours) and there is no evidence the rate rises. Per
coordinator ruling 2026-09-06 the realised yield is accepted and the
remainder of `--n 2000` is interior.

That rate is itself the finding. The mask-free empty-window failures the
stratum was written for (gate bouts 1/69/335) came from **stale REUSED
windows and single-fly frames** -- both of which this scan deliberately
excludes, the first because a reused centre is not a CenterDetect peak at
all, the second because a frame with a fly unaccounted for cannot support the
claim "this window is empty". With both flies tracked and accounted for,
CenterDetect essentially does not fire on empty arena. The existence head's
empty-crop supervision therefore comes mostly from the interior stratum plus
the stale-window frames the single-fly pass already flags.

#### Real run (2026-09-06, g3102 GPU 6, same sources and `--n 2000` as commit 0fa9733)

```
--n 2000 --edge-frac 0.25 --edge-scan-frames 1500 --edge-scan-block 64 --edge-oversample 3
```

**2,000 anchors (28 edge / 1,972 interior) + 5,302 T=2 partners = 7,302
framesets, 51,114 images**, 5,630 s total of which 2,090 s was the
CenterDetect scan. The 28 edge anchors carry 78 partners of their own, so
106 of the 7,302 framesets are edge-stratum (verified by reading the written
`instances_train.json` back: 7,302 framesets, all `negative: true`,
`fly_id: -1`, `weight: 1.0`, zero non-negative annotations). `per_role {"negative": 2000, "partner": 5302}`.
`partner_availability {"1": 1731, "4": 1702, "16": 1869}`.
**`weight` is 1.0** in the manifest, in every `recordings[*]` block and on
every frameset (coordinator ruling: a negative's only supervision is an
existence target of 0, and that target is a geometric fact about the window,
not a checkpoint's guess -- nothing to down-weight). `source` stays
`"pseudo"`, `role` stays `"negative"`/`"partner"`; asserted in
`test_negative_record_survives_write_pseudo_export`.

| recording | src | anchors | edge | interior | partners | scanned | lifted | cand |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2025_10_20_13_20_04 | tracks | 201 | **0** | 201 | 194 | 964 | 790 | 0 |
| 2026_04_02_12_11_50 | lift_root | 200 | 5 | 195 | 564 | 569 | 721 | 5 |
| 2026_04_02_14_54_28 | lift_root | 200 | 3 | 197 | 583 | 776 | 1157 | 3 |
| 2026_04_02_15_25_51 | lift_root | 200 | 4 | 196 | 570 | 789 | 1357 | 4 |
| 2026_04_02_15_44_42 | lift_root | 200 | 1 | 199 | 567 | 539 | 670 | 1 |
| 2026_04_02_16_03_48 | lift_root | 200 | 2 | 198 | 550 | 669 | 1232 | 2 |
| 2026_04_02_16_21_32 | lift_root | 200 | 2 | 198 | 580 | 1011 | 1879 | 2 |
| 2026_04_02_16_39_56 | lift_root | 200 | 7 | 193 | 574 | 674 | 1019 | 7 |
| 2026_04_02_16_56_37 | lift_root | 200 | **0** | 200 | 540 | 552 | 865 | 0 |
| 2026_04_02_17_28_34 | lift_root | 199 | 4 | 195 | 580 | 561 | 850 | 4 |
| 2026_04_02_17_52_50 | lift_root | **0** | 0 | 0 | 0 | 704 | 1678 | 0 |

`2025_10_20_13_20_04` -- the recording whose gate bout 1 motivated the whole
stratum -- yielded **0 from 964 all-tracked frames**, all 790 residual centres
rejected on distance. `2026_04_02_17_52_50` again yields 0 anchors (its
tracked footprint is 5.6 units tall, below `--min-height-units` by itself);
its 181-anchor share was redistributed as before, so the total is exactly
2,000. The cross-recording edge top-up found no donor with spare candidates
(every pool was fully consumed), which is now logged rather than silent.

**Edge rejections inside `sample_negatives`: 0 in every recording** -- the
scan's gate and the sampler's authoritative re-check agree, as they must. The
re-check is kept anyway: it is what makes the invariant hold for any pool,
including a test fake.

**Partners** (5,302 of a 6,000 ceiling): delta 1 taken 1,731 (no_coverage 206,
collision 60, dist 3); delta 4 taken 1,702 (no_coverage 211, collision 81,
dist 6); delta 16 taken 1,869 (no_coverage 44, collision 73, dist 14). The
stride-16 `no_coverage` structural loss on deltas 1/4 for
`2025_10_20_13_20_04` is unchanged (182/182 each).

**Verify**: `{"windows_checked": 20, "n_windows": 7302, "mean_valid_cameras":
7.0, "all_negative": true}`. The `figures/` copy of `negatives_report.json` is
now written AFTER verify (review's minor finding), so the copy that survives
beside the PNG carries the loader round trip.

**Figure** `figures/2026-09-mvq/v2_pseudo/negatives_check.png`, 16 panels,
**8 edge** (orange titles, rows 1-2, from `12_11_50`, `15_44_42`, `16_03_48`,
`16_21_32`) and 8 interior (rows 3-4, from `20_04`, `12_11_50`, `15_25_51`,
`16_39_56`), 2 cameras per negative.

*Expectation stated before looking*: every panel bare substrate or an arena
fixture, no fly; a fly in an EDGE panel specifically would mean the false-peak
candidates are still bypassing the gates.

*Read back*: all 16 panels show machined arena floor (diagonal scratch
pattern, yellowish dirt flecks) and/or the dark chamber-wall band. No fly
body, wing or leg in any panel. Two edge panels
(`16_21_32` Frame_344654, `15_44_42` Frame_421509) carried faint round specks
too small to resolve at thumbnail size; both were re-rendered at full 448-px
crop across all 7 cameras with the window centre marked
(`figures/2026-09-mvq/v2_pseudo/edge_zoom_*.png`) and read back -- the specks
are out-of-focus dust on the substrate, no elongated body, no legs, and the
nearest tracked fly is 75.4 u / 76.2 u away with both flies tracked at those
frames. Matches the expectation; the gate gap is closed.

**Interior stratum already carries arena-edge appearance.** Rebuilding each
recording's tracked-centroid hull and measuring the drawn interior centres
against it: 38-81 % of interior anchors per recording sit OUTSIDE that hull
(median signed distance -2.2 to +9.4 units), i.e. at the chamber periphery --
which is why most figure panels show the wall band. What the edge stratum adds
is the "CenterDetect fired here" provenance, not the appearance.
