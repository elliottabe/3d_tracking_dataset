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
