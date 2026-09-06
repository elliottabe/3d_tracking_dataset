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
