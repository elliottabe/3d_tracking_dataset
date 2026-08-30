# Bout-28 fly0 dropout diagnosis (Task 19, Phase 5)

**Diagnosis only.** No pipeline source, config, or `stac-mjx` changes; the
pipeline was not re-run. Everything below reads `sam3_masks.npz` and the
archived Task-1 baseline arrays (`figures/2026-08-29-c2f-3d/phase0-baseline/
arrays/bout_00028/`) via `scripts/qc/diagnose_mask_dropout.py`, which is
committed alongside this note. Regenerate all of it with:

```
micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 \
  python scripts/qc/diagnose_mask_dropout.py all
```

Outputs land in `figures/2026-08-29-c2f-3d/phase5-dropout/` (gitignored):
`dropout_report.json` (Step 2 table + per-frame arrays), `repair_check.json`
(Step 3), `reproject_check.json` (Step 4), `step5_attrition.png` (Step 5),
`step1_valid_overlay.png` (Step 1).

## Headline

Session0 `2025_10_20_13_20_04` bout 28, fly0 (female): frames 1502-2006 read
as "NaN 3D" downstream (`qc_perframe.npz` `reproj_px` is NaN for a clean,
contiguous 100% of exactly these 505 frames). **This is a
confidence/coverage-gate cascade, not an identity or mask-tracking failure.**
Task 1's fix-round conclusion ("the SAM3 mask drifts onto empty background")
does not hold: the masks are smooth, correctly separated from fly1, and
`in_frame` (the pipeline's own independent out-of-view check) agrees with
`valid` bit-for-bit. Three distinct, complementary mechanisms combine to
produce the observed 100%-NaN block, none of them an identity bug:

1. **`masks.min_views: 4`** (`configs/pipeline.yaml`), evaluated on the
   mask-agreement-adjusted view count (`masks.kp_mask_agree_fly_lengths:
   3.0`), blanks the *entire* frame's `kp3d` to NaN in **66.9%** (338/505) of
   after-split frames outright — even on frames where the 3 permanently-good
   cameras (`Cam2012853`, `Cam2012855`, `Cam2012630`) alone would clear DLT's
   own >=2-view minimum.
2. **`view_conf_thresh: 0.6`** (`configs/detector/vitpose_v3.yaml`) removes
   further views on marginal-confidence detections — median per-view
   confidence sits at 0.44-0.59 in this stretch even on cameras whose mask is
   100% valid and whose keypoints agree with it — so even the 167 frames that
   *pass* `min_views` mostly end up with an incomplete (not all-50) keypoint
   set. Only 20/505 (4.0%) after-split frames get all 50 keypoints.
3. **STAC's all-or-nothing consumer** (`scripts/run_bout.py`
   `finite_frame_mask` + `NAN_SOLVE_MIN_SEG=30` after
   `NAN_SOLVE_MAX_GAP=10`-frame gap-fill) requires a run of >=30
   *consecutive* fully-complete frames before it will even attempt a solve.
   The 20 survivors are scattered singletons/pairs; **the only usable
   segment in the entire 2007-frame bout is `[0, 1502)`** (verified by
   running the real gap-fill + segment-finder over the archived `kp3d.npz`).
   That is why the final pose (and `reproj_px`) reads as 100% NaN for the
   *whole* tail even though raw DLT triangulation actually recovers a
   partial signal (any keypoint finite) in 18.2% of those frames.

And underneath all three: **the female is genuinely leaving 4 of 7 cameras'
fields of view.** Step 4's independent reprojection check (male sanity-check
+ female trunk-keypoint reprojection) and Step 3's repair-trigger check
(the pipeline's own mask-centroid-based geometric test) agree on this by two
different routes.

## Step 1 — settled: absent-and-flagged, not drifted

Re-rendered `Cam*` raw frames at bout-offsets 1550/1700/1900, drawing fly0's
mask (cyan, `PALETTE["fly0"]`) **only where `valid=True`**, and — going one
step further than the brief asked, because it removes all ambiguity — also
drawing fly1's mask (orange) so any visible animal is unambiguously
identified. `figures/2026-08-29-c2f-3d/phase5-dropout/step1_valid_overlay.png`
(crops read at 2x zoom per camera/frame).

**What I looked for:** a panel with `valid=False` and a cyan mask anyway
(drift), vs. a panel with `valid=False` and nothing female-shaped in frame at
all (absent-and-flagged).

**What it showed:** every `valid=False` panel for fly0 (`Cam2012631` at
offsets 1700 and 1900; `Cam2012857`, `Cam2012861`, `Cam2012862` at offset
1900) contains exactly one visible fly, and it is unambiguously colored
**orange** (fly1) — there is no cyan pixel anywhere in the frame. The visible
animal is the male, not a missed female. In the `valid=True` panels near this
boundary (`Cam2012630`, `Cam2012853`, sometimes `Cam2012855`), fly0's mask is
correctly a small, partial cyan sliver near the image edge next to the
male — consistent with her being close to the boundary of that camera's
view, not a confident, centered presence.

**Verdict: absent-and-flagged holds; "drifted onto background" does not.**

*Reconciling Task 1's contradictory reading.* I could not inspect Task 1's
original PNGs (not committed, and its own `figures/.../fix-round-1/`
directory is gitignored and no longer present), so I cannot say with
certainty what its "persistent mask blob" was. What I can say from the
current code and the current archived data: (a) `viz/core/overlays.py`'s
`draw_points`/`draw_mask` already filter non-finite points and gate the mask
draw on `masks["valid"]`, so a literal re-run of Task 1's own command
(`python -m viz overlay --fly 0 --frame {1550,1700,1900}`) against this data
should not reproduce a drifted-mask illusion; (b) `outputs.h5`'s
`mesh_mm`/`kp3d_mm`/`qpos` for fly0 are 100% NaN at exactly these three
frames in the very run Task 1 examined, so any rendered "mesh" content there
cannot have come from a real fitted pose; (c) `viz/core/colors.py`'s
`PALETTE["detector"]` (the always-finite raw 2D keypoints, per Task 1's own
Q2 finding) is the *identical* color to `PALETTE["fly0"]` — a busy montage
mixing those two categories is an easy place for a human to misread "the
detector's own keypoints, which land somewhere real but wrong" as "the
mask." I did not chase this further; it is not needed to answer the brief's
central question, and Step 1's own images settle that question directly.

## Step 2 — per-camera per-stage table (dead frame 1600, live frame 1400)

Gates, in order: mask present (`valid`) -> keypoints emitted (any finite in
`kp2d`) -> per-keypoint conf (`conf_thresh=0.3`) -> per-view median conf
(`view_conf_thresh=0.6`) -> mask agreement (`kp_mask_agree_fly_lengths=3.0`,
not in the brief's original list but a real, load-bearing gate — see below)
-> 3D emitted (finite in `kp3d.npz`).

**Dead frame, offset 1600** (`views_raw=6`, `views_agree=4`, **not**
min_views-gated — kp3d is actually 49/50 finite here, see caveat below):

| camera | mask valid | area px | any kp finite | any conf>=0.3 | view median conf | view_conf_pass (>=0.6) | centroid disagree px | mask agree |
|---|---|---|---|---|---|---|---|---|
| Cam2012630 | True | 10651 | True | True | 0.612 | **True** | 12.8 | True |
| Cam2012631 | **False** | 0 | True | True(46/50) | 0.444 | False | 1597.8 | False |
| Cam2012853 | True | 11891 | True | True(49/50) | 0.637 | **True** | 35.2 | True |
| Cam2012855 | True | 5336 | True | True | 0.558 | False | 37.1 | True |
| Cam2012857 | True | 2294 | True | True(47/50) | 0.468 | False | 167.9 | **False** |
| Cam2012861 | True | 3075 | True | True(48/50) | 0.468 | False | 29.2 | True |
| Cam2012862 | True | 677 | True | True(47/50) | 0.519 | False | 267.8 | **False** |

`views_agree=4` (630, 853, 855, 861) clears `min_views=4` — this frame is NOT
gated, and 49/50 keypoints do triangulate. **Important correction to the
brief's own worked example:** offset 1600 is not itself a raw-DLT-NaN frame.
It sits inside the "dead" window only in the *downstream* sense (STAC never
gets a 30-frame run to solve it in) — see Step 5.

**Live frame, offset 1400** (`views_raw=7`, `views_agree=7`, `kp3d`
50/50 finite):

| camera | mask valid | area px | any kp finite | any conf>=0.3 | view median conf | view_conf_pass | centroid disagree px | mask agree |
|---|---|---|---|---|---|---|---|---|
| Cam2012630 | True | 9816 | True | True | 0.608 | True | 11.8 | True |
| Cam2012631 | True | 9298 | True | True | 0.529 | False | 25.0 | True |
| Cam2012853 | True | 11647 | True | True | 0.722 | True | 29.1 | True |
| Cam2012855 | True | 12529 | True | True | 0.593 | False | 58.9 | True |
| Cam2012857 | True | 14196 | True | True(48/50) | 0.585 | False | 31.3 | True |
| Cam2012861 | True | 15176 | True | True | 0.709 | True | 33.4 | True |
| Cam2012862 | True | 11315 | True | True | 0.637 | True | 54.4 | True |

Note even the "live" frame only has 3/7 cameras above `view_conf_thresh=0.6`
— confidence is already marginal well before the split; it is coverage
(views_agree dropping below 4), not confidence alone, that flips the frame
NaN.

**Which gate removes the 3-good-camera case?** Exact reconciliation against
the archived `kp3d.npz` (`dropout_report.json`'s per-frame arrays): modeling
*only* `conf_thresh` + `view_conf_thresh` (no frame-level gate, no reproj
consensus) reproduces the archived per-keypoint finiteness with **zero
mismatches** on the 167/505 after-split frames where `views_agree >= 4`
(8350/8350 keypoint-slots match exactly). On the remaining 338/505 frames
where `views_agree < 4`, the archived `kp3d` is **100% NaN in every one**,
independent of what those same per-keypoint gates would have produced (57.1%
of them would have triangulated *something* pre-gate). That is
`gate_low_coverage_frames`/`masks.min_views` firing exactly and only where
predicted, with no residual unexplained variance — **this is the single
gate that discards the "3 good cameras are enough to triangulate" case**,
and it is a coverage/agreement threshold, not an identity check.

The `reproj_resid_px=10.0` consensus gate (`TRIGGER_FACTOR=3.0`,
`MIN_CONSENSUS_VIEWS=3`) makes **no additional difference** in this window —
the zero-mismatch reconciliation above already accounts for every frame
without modeling it, so it never further removes a view here.

A second, smaller mechanism accounts for the other 75/505 (14.9%) all-NaN
frames that are *not* `min_views`-gated: e.g. offset 1522 has `views_raw=7`,
`views_agree=7` (every mask valid and agreeing) yet only **1 of 7** cameras
(`Cam2012853`, 0.641) clears `view_conf_thresh=0.6` — nowhere near the >=2
views DLT needs, so all 50 keypoints go NaN even with a perfect mask picture.
This is a pure detector-confidence deficit, not masks or identity at all.

## Step 3 — repair triggers: (b), not (a) or (c)

`repair_check.json`: `repair_outliers_fired=False` (no `suspect_cameras` in
the npz), `repair_missing_fired=False` (no `gap_repair` record),
`in_frame_equals_valid=True` bit-for-bit, `n_in_frame_yes_but_invalid=0`
across the *entire* bout (not just the dead window).

- **`repair_missing`** (`jarvis_jax/predict/sam3_driver.py:
  repair_missing_cameras` -> `find_gap_cameras`) only treats a
  `(fly,camera)` as a "gap" when `in_frame_codes` resolves it to
  `IN_FRAME_YES` while `valid=False`. `in_frame_codes` (same file,
  L923-976) triangulates fly0 from whichever cameras *are* valid at that
  frame (min_support=3 — and `views_raw` never drops below 3 in this
  window, so it is never `IN_FRAME_UNKNOWN`; only 0/1 values appear in the
  saved array), reprojects into the invalid camera, and tests image bounds.
  For every one of fly0's 1010 invalid views in this bout, that test came
  back `IN_FRAME_NO`. `find_gap_cameras` therefore found **zero** qualifying
  gaps — `repair_missing` never had a candidate to act on. This is
  option **(b)**, not (a): it did not fail `repair_missing_accept_resid:
  25.0`, it never attempted a re-segment in the first place.
- **`repair_outliers`** is inapplicable for a different reason, also (b):
  its per-camera leave-one-out residual (`_loo_residual`) is undefined (NaN)
  for a camera with no mask at all, so an absent view never even enters its
  candidate pool (masks that are *wrong* trigger it; masks that are *absent*
  cannot). No `suspect_cameras` were written, consistent with that.
- Option **(c)** ("structurally blind because most cameras are right") is
  real but does not apply *here*: it is `repair_outliers`'s own explicit
  `len(outliers) > C//2` bail-out (`sam3_driver.py:743-748`), which requires
  those cameras to have been flagged as outliers in the first place — they
  never were, for the reason above. `repair_missing` has no analogous
  majority-assumption; it is a `min_support>=3` geometric threshold, which
  was met throughout.

## Step 4 — genuinely out of view, corroborated two independent ways

`reproject_check.json`: reprojected fly1's Scutellum 3D position (sanity
check on the tool itself) and fly0's nearest-finite Scutellum 3D position
into all 7 cameras at offsets 1550/1700/1900.

**Sanity check passes:** fly1 (100% finite everywhere) reprojects
**in-bounds on all 7 cameras at all three offsets** — the calibration /
reprojection math is not the source of any of the numbers below.

**Fly0:**

| camera | 1550 (u,v) | 1700 (u,v) | 1900 (u,v) | trend |
|---|---|---|---|---|
| Cam2012630 | (1811, 151) in | (1801, 129) in | (1830, 106) in | stays in bounds |
| Cam2012853 | (1824, 369) in | (1814, 381) in | (1842, 429) in | stays in bounds |
| Cam2012631 | (1762, **-159**) | (1752, **-224**) | (1781, **-350**) | monotonically further above frame |
| Cam2012857 | (1833, **541**) | (1823, **604**) | (1853, **736**) | monotonically further below frame (H=448) |
| Cam2012861 | (1820, **-85**) | (1810, **-148**) | (1838, **-281**) | monotonically further above frame |
| Cam2012862 | (1819, **-71**) | (1808, **-121**) | (1835, **-207**) | monotonically further above frame |

Bold = out of the `1936x448` image bounds. The four cameras that flag
`valid=False`/`in_frame=0` during this window are exactly the four whose
reprojection is out of bounds, and the magnitude of the overshoot **grows
smoothly across 1550->1700->1900** (e.g. `Cam2012631`: -159 -> -224 -> -350
px) — the signature of a real trajectory carrying her further outside each
camera's frustum, not measurement noise (noise would not be monotonic across
150-frame gaps). Two of the four clip the top of frame (negative v) and one
clips the bottom (v > 448), consistent with a single coherent 3D position
near a wall/corner viewed from very different camera angles, not a random
scatter.

**One honest caveat:** `Cam2012855` is one of the three cameras that stays
100% `valid` throughout, yet its reprojected position (489/532/636 px, all
> H=448) is also out of bounds. I read this as the estimator, not the fly:
fly0's Scutellum 3D in this window is triangulated from as few as 2-3
marginal-confidence views (that's the whole diagnosis, after all), so a
single trunk keypoint's position carries real error, particularly along the
weakly-constrained depth axis for a narrow camera-baseline subset. The other
four cameras' near-linear, growing-with-time pattern is far too consistent
to be explained by that same noise, but `Cam2012855`'s single anomalous
number is a reminder that this reprojection is corroborating evidence, not a
certified ground truth.

**Conclusion: for `Cam2012631`, `Cam2012857`, `Cam2012861`, `Cam2012862`,
the evidence supports a genuine, recoverably-*un*recoverable exit from frame,
not a recoverable mis-track** — matching Step 3's independent
mask-centroid-based `in_frame_codes` finding exactly.

## Step 5 — attrition figure

`figures/2026-08-29-c2f-3d/phase5-dropout/step5_attrition.png`, three panels
sharing a frame axis, split marked at 1502.

**Stated expectation (from the brief):** coverage -> mask-present drops to
~3 at 1502, keypoint/conf curves crash to 0, locating the loss downstream of
the masks; masks-are-the-whole-story -> mask-present itself drops to 0.

**What it shows:** top panel — `views_raw` (mask-present) does **not** drop
to 0 anywhere; it declines from a flat 7 starting around offset ~1420-1450
and settles into a 2-7 range after 1502 (histogram: {3:127, 4:30, 5:152,
6:108, 7:88} — median 5, i.e. well above the brief's "~3" guess but clearly
and permanently below 7). `views_agree` (orange, after the mask-agreement
gate) sits consistently at or below `views_raw`, crossing the `min_views=4`
line (grey dashed) far more often. Middle panel — `kp3d.npz` keypoints
finite crashes from 50 to mostly 0 almost exactly at 1502, with sporadic
47-50 spikes. Bottom panel — "STAC-usable" (all 50 finite) frames are dense
green before 1502 and reduce to a handful of hairline spikes after, none
wide enough to see at this scale (they are the same 20 scattered singleton
frames from Step 2/3).

**This is the brief's "coverage" shape, not its "masks are the whole
story" shape** — confirming the headline: the loss is real and located
downstream of the raw masks (in the per-view confidence gate, the
mask-agreement/min_views frame gate, and STAC's segment-length requirement),
not in SAM3 losing track of her.

One nuance the figure also makes visible: the decline starts gradually
around offset ~1420-1450, well before the frame-level NaN block begins at
1502. The 1502 boundary is not "the moment something broke" — it is where
the *cumulative* effect of small, scattered single-keypoint gaps (the first
non-all-50-finite frame is offset 1420, while masks are still 100% valid
everywhere) finally exceeds `NAN_SOLVE_MAX_GAP=10`/`NAN_SOLVE_MIN_SEG=30`'s
tolerance and the one long usable segment `[0, 1502)` ends. Frames
1420-1501 already have occasional missing keypoints; they just weren't
*enough* missing, yet, to break the run.

## Step 6 — commit

```
git add scripts/qc/diagnose_mask_dropout.py \
        docs/benchmark/2026-08-29-c2f-3d/dropout-diagnosis.md
git commit -m "diag(masks): locate the bout-28 fly0 dropout in the stage chain"
```

## Step 7 — recommendation for Task 20

**Do not scope Task 20 as an identity/mask-tracking fix — that framing is
refuted by Steps 1-4.** The single gate most directly responsible for
converting "partially recoverable" into "wholesale NaN" is
**`masks.min_views: 4`** (as evaluated on the `kp_mask_agree_fly_lengths`
-adjusted view count): it accounts for 338/505 (66.9%) of after-split
frames' complete NaN-out with zero unexplained discrepancy against the
archived data, discarding frames where the three permanently-good cameras
alone would have cleared DLT's real minimum. The remaining share is a
detector-confidence deficit (`view_conf_thresh: 0.6` against genuine median
confidences of 0.44-0.59 for this fly in this stretch), and the reason *no*
frame in the tail ever produces final output is `scripts/run_bout.py`'s
all-or-nothing STAC input criterion (`finite_frame_mask` +
`NAN_SOLVE_MIN_SEG=30`) discarding a real, if incomplete and noisy, partial
3D signal because it is scattered across singleton frames rather than one
long complete run.

Underneath the gates, the female really is walking towards a wall/corner and
leaving 4 of 7 cameras' fields of view over ~100-400 frames (Step 4's
monotonic, physically-coherent reprojection drift) — a `min_views` relaxation
or a friendlier STAC-input criterion cannot manufacture views that don't
exist; camera coverage in this stretch genuinely bottoms out at 2-3
cameras for a sustained run, which the `min_views` comment's own numbers
(elsewhere in the corpus) warn is measurably ill-conditioned for full-body
DLT. A fix should therefore target **graceful degradation of the partial
signal that already exists** (e.g., let STAC accept a per-frame partial
keypoint set — or a lower per-frame keypoint-count bar in place of literal
all-50 — for frames that clear a relaxed, still-principled view/consensus
bar, rather than either (a) globally lowering `min_views` for the whole
corpus, which the config's own comment already measured causes bones to
flex 20-50%, or (b) chasing an identity/segmentation bug that does not
exist here), scoped narrowly enough to be tested against the frozen 13-bout
benchmark rather than assumed safe for the whole corpus from this one bout.
Full recovery of frames where only 2 cameras genuinely see her may simply be
out of reach; the realistic target is "less total loss," not "zero loss."

## Files

- `scripts/qc/diagnose_mask_dropout.py` (committed)
- `figures/2026-08-29-c2f-3d/phase5-dropout/{dropout_report.json,
  repair_check.json, reproject_check.json, step1_valid_overlay.png,
  step5_attrition.png}` (gitignored, regenerate with the command above)
