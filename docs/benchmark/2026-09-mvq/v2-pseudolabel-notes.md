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
   task's scope.
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
