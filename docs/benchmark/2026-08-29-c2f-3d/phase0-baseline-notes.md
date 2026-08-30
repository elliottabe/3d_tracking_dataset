# Phase 0 baseline expectation

Running bout 28 through the CURRENT pipeline (ViTPose v4_8gpu_20260808 -> robust
DLT -> STAC IK -> polish). This is the "before", not a result.

Expected from prior measurements:
- male (fly1) reprojection ~7 px, female (fly0) ~42 px, ~87 px when flies are close
- female silhouette IoU ~0
- leg-angle traces visibly jagged; tarsal-tip 3D traces show ~0.17 mm staircase
  quantization if the resolution hypothesis holds

If the female reprojection comes out near the male's, the benchmark baseline does
not describe this bout and the acceptance thresholds in Task 17 must be re-derived
from THIS run rather than from docs/benchmark/2026-08-07-baseline-scorecard.json.

## Observed

Ran bout 28 (Session0 `2025_10_20_13_20_04`, frames 446306-448312, 2007 frames,
7 cameras) through `scripts/run_bout.py` (anatomy=v1, nq=93). Both fly0 and
fly1 were already `DONE` from a prior run of this exact codebase state
(2026-08-27 22:29-23:48); re-running `run_bout.py` exercised the resume path
and skipped both (`bout 28 fly0: already DONE, skipping` / same for fly1) —
see "Procedural notes" below for why this still counts as a valid capture of
the CURRENT pipeline. Run root:
`/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/pose`.

### Per-fly numbers

**qc.json** (pipeline's own QC, per fly):

| metric | fly0 (female) | fly1 (male) |
|---|---|---|
| `per_camera_reproj_px.median` | NaN (n=12245) | 6.054 px (n=14049) |
| `loo_reproj_px.median` | 6.259 px (n_frames=1879/2007) | 2.651 px (n_frames=2007/2007) |
| `silhouette_iou.hard_median` | 0.0287 | 0.0251 |
| `silhouette_iou.soft_median` | 0.1715 | 0.1522 |

**qc_perframe.npz `reproj_px`** (all-camera per-frame aggregate, the metric
the figures below are consistent with):

| stat | fly0 (female) | fly1 (male) |
|---|---|---|
| median | 10.26 px | 6.02 px |
| mean | 16.48 px | 6.08 px |
| % frames NaN | 25.2% | 0.0% |
| p90 / p95 / p99 | 37.13 / 45.69 / 62.74 px | 6.59 / 6.81 / 7.13 px |
| max (frame offset) | 90.26 px (offset 1498, abs 447804) | 7.56 px (offset 1810, abs 448116) |
| `hard_iou` median/mean | 0.0279 / 0.0216 | 0.0246 / 0.0244 |

### Comparison to the frozen benchmark scorecard (bout 28 is NOT one of the 13
frozen bouts: those are 1,2,8,12,17,22,26,29)

- Cohort courtship_female: `reproj_px_median`=42.30 px, `reproj_px_close`=86.90 px.
- Cohort courtship_male: `reproj_px_median`=7.20 px.
- Bout 28 fly0's **typical** performance (median 10.26 px) is much better than
  the cohort female median (42.30 px) — only ~1.7x worse than fly1 (6.02 px),
  not the ~5.9x female/male gap implied by the cohort medians (42.30/7.20).
- But bout 28 fly0's **worst-case** (90.26 px, at the proximity event around
  offset 1484-1501) lands almost exactly on the cohort's `reproj_px_close`
  (86.90 px) — the "close" failure mode is real and matches the benchmark;
  it's just localized to a ~20-frame window, not this bout's typical state.

**This triggers the brief's own fallback clause**: bout 28's *typical* female
reprojection (10.26 px) comes out much nearer the male's (6.02 px) than the
frozen benchmark cohort implies. The benchmark baseline scorecard describes
this bout's worst-case window well but NOT its typical state. Task 17's
acceptance thresholds should be re-derived from bout 28 directly (both the
10.26 px typical and 90.26 px close-window numbers) rather than assumed equal
to the cohort's 42.30/86.90 px.

### Figures — what each one actually showed (opened with Read, not assumed)

Working commands discovered (see "Ruling R1 commands" below for the full
set). All renders below used `--run <OUT>` = the run root above.

**`overlay` — typical frame (offset 200, abs 446506)**
- `overlay_bout28_fly0_f200.png`: cyan mesh+detector cloud (fly0's PALETTE
  color IS cyan, same as "detector") sits tightly on the female's body/wings/
  legs in all 7 camera montage tiles. Good agreement, no drift. Consistent
  with fly0's median-region performance (this offset falls just before
  fly0's actual best10 frames, offsets 205-212 at ~8 px).
- `overlay_bout28_fly1_f200.png`: orange mesh (fly1's color) + cyan detector
  points track the male's extended-wing courtship pose closely across all 7
  cameras. Tight orange/cyan agreement everywhere, consistent with the
  ~6 px error.

**`overlay` — close/hard-case frame (offset 1490, abs 447796, inside the
90 px spike window)**
- `overlay_bout28_fly0_f1490_close.png`: **dramatic, unambiguous failure.**
  In 6 of 7 camera views (Cam2012630/853/855/857/861/862) the real female fly
  is clearly visible in the raw frame, sharply in focus, but the fitted mesh
  (grey blob) + cyan keypoint cloud is rendered floating in empty space,
  detached from the real animal — in Cam2012861 the mesh hangs in mid-air
  with no fly visible under it at all. This is a direct visual confirmation
  of the reprojection blow-up, not just a number: the IK output has locked
  onto the wrong target during the close-proximity event.

**`legskel` — typical frame (offset 200)**
- `legskel_bout28_fly0_f200.png` / `legskel_bout28_fly1_f200.png`: cyan
  (detector) and green (fit) leg chains overlap closely on all 6 legs across
  all 7 cameras for BOTH flies. No visible detachment at this frame for
  either fly.

**`legskel` — close frame (offset 1490)**
- `legskel_bout28_fly0_f1490_close.png`: chains are grossly wrong. In
  Cam2012862 both cyan and green chains stretch as long diagonal lines all
  the way across the frame to a point far from the visible fly (which sits
  in the left third of the image); in Cam2012857 the chains form a huge
  triangle spanning most of the frame height with no fly visible under it;
  the remaining views show tangled, anatomically-implausible crossed chains
  instead of the clean per-leg "star" pattern seen at offset 200. Same
  failure as the overlay figure, now shown at the joint-chain level.

**`legskel --compare` smoke test** (Task 1 has no second run yet, so this was
a **self-compare** sanity check only, not a real A/B):
- `legskel_bout28_fly0_f200_selfcompare_smoketest.png`: `--compare <same run
  root>` adds an orange "compare=..." chain layer that lands exactly on top
  of the green fit layer (as it must, comparing a run against itself). This
  confirms the `--compare` flag's file-path plumbing and legend work; it
  proves nothing about fit quality since there's no second arm yet.

**`compare_stac_fits.py` — typical frame (200)**
- `compare_stac_fits_f200.png`: WHITE (observed) and CYAN (fitted markers)
  agree closely with each other and mostly sit on the ORANGE anatomical mesh
  sites for both flies. The one consistent deviation is at leg tips (tarsal
  claws), where white/cyan dots hang a small, visible gap below the last
  rendered leg segment for BOTH flies (fly0 gap metric 0.0158, fly1 0.0161 —
  nearly identical). This reads as a small, systematic marker-offset residual
  at the leg tips, not a per-fly defect (both flies show the same magnitude).
- fly1's right panel clearly shows the male's wing raised in a courtship
  display pose; fly0's left panel shows both wings folded — anatomically
  sensible for both.

**`compare_stac_fits.py` — close frame (1490)**
- `compare_stac_fits_f1490_close.png`: **fly0 (left panel)** shows white/cyan
  markers scattered visibly OFF the mesh near the head/front-leg region — a
  disconnected cloud of white and cyan dots hovering above the thorax/head
  with no mesh underneath them, while the mesh itself sits in an odd
  raised-front-leg pose. This is the same failure as overlay/legskel, seen
  here as "the IK found a self-consistent pose, but it does not match where
  the (corrupted) observations actually are." **fly1 (right panel)** at the
  identical frame looks clean — wing extended, markers sitting near the legs
  with only the same small tip-gap seen at frame 200. The gap metric itself
  (fly0 0.0155, fly1 0.0161) barely moves between the two frames — it
  measures cyan-vs-orange self-consistency of the IK model, NOT
  observed-vs-real accuracy, so it does not surface this failure; only the
  WHITE dots floating off the mesh do. Noted as a limitation of that
  specific diagnostic for later tasks.

**Tarsal-tip staircase check** (no command given in the brief for this part
of Step 5; produced a one-off scratchpad script per CLAUDE.md's "one-off
diagnostics" guidance, since Step 5 explicitly asks about it) —
`tarsal_staircase_check.png`: plotted `T1L_TaTip` raw triangulated (kp3d.npz,
pre-filter) position, mean-subtracted, over each fly's longest all-finite
"plausible-walking" window (max per-frame step < 1.5 mm: fly0 offsets
157-307, fly1 offsets 1741-1891), plus a whole-bout frame-to-frame |diff|
histogram. **Result: NO staircase quantization observed.** Both traces are
smooth continuous curves at mm-scale zoom (male shows a clean single-swing
leg-step arc up to ~11 mm of travel); the diff histograms decay smoothly from
near-zero with no bump/kink at the hypothesized 0.17 mm step (modal step
0.0075 mm fly0 / 0.0025 mm fly1, both far below 0.17 mm). **This disagrees
with the brief's stated resolution-hypothesis expectation** for this
keypoint/bout: if a coarse-decode staircase artifact exists, it isn't visible
in bout 28's raw triangulated tarsal-tip trace at this zoom.

**Leg-angle jaggedness check** (same rationale/one-off script) —
`leg_angle_jaggedness_check.png`: plotted `tibia_T1_left` qpos (deg) over the
same two walking windows. **Result: smooth, not visibly jagged**, within
those windows (median frame-to-frame step 0.28 deg fly0 / 0.15 deg fly1).
However the WHOLE-BOUT step-size percentiles are dramatically different by
fly: fly0 p99=30.3 deg, max=71.3 deg vs fly1 p99=1.9 deg, max=3.1 deg — i.e.
fly0's leg angles DO jump violently, but only during the close-proximity
failure window identified above, not generically. **Partial disagreement
with the brief**: "visibly jagged" is not a pervasive property of the
current pipeline's leg-angle traces; it is confined to the same
proximity-failure window that also blows up reprojection error, and is
specific to fly0.

### Bottom line vs. the stated expectation

- Male (fly1) reprojection ~7 px: **confirmed** (qc.json per-camera median
  6.05 px; qc_perframe median 6.02 px, tight distribution 5.7-7.6 px
  throughout).
- Female (fly0) reprojection ~42 px, ~87 px when close: **NOT confirmed as a
  typical-frame number** — bout 28's typical fly0 error is 10.26 px, far
  below the cohort's 42.30 px. **Confirmed at the close-proximity event
  specifically** — 90.26 px peak vs. cohort's 86.90 px "close" number.
- Female silhouette IoU ~0: **not distinctly worse than male here** — fly0
  hard IoU median 0.0279 vs fly1 0.0246, essentially the same order and
  fly0 is not lower.
- Tarsal-tip ~0.17 mm staircase: **not observed** (see above).
- Leg-angle jaggedness: **not observed in typical frames**; **observed only
  in the fly0 close-proximity window** (see above).
- **Consequence for Task 17**: per the brief's own fallback clause, bout 28's
  acceptance thresholds should be derived from THIS run (10.26 px typical /
  90.26 px close-window fly0; 6.02 px fly1), not assumed identical to the
  frozen 13-bout scorecard's female/male cohort medians — the cohort
  describes this bout's worst moments well but overstates its typical
  female-vs-male gap by roughly 3.5x (5.9x cohort gap vs 1.7x this bout's
  typical gap).

### Ruling R1 commands (working invocations, for later tasks to reuse)

```bash
# overlay: mesh+kp+mask+axis montage across all 7 cameras, one (bout,fly,frame)
python -m viz overlay --run "$OUT" --bout 28 --fly 0 --frame 200 \
  --out figures/2026-08-29-c2f-3d/phase0-baseline/overlay_bout28_fly0_f200.png
# --compare <other_run_root> overlays a second run's mesh for the same tile (not used here)

# legskel: detector-2D (cyan) vs fitted-3D (green) leg chains, all 7 cameras
python -m viz legskel --run "$OUT" --bout 28 --fly 0 --frame 200 \
  --out figures/2026-08-29-c2f-3d/phase0-baseline/legskel_bout28_fly0_f200.png
# --compare <other_run_root> adds a third orange chain layer from that run
python -m viz legskel --run "$OUT" --bout 28 --fly 0 --frame 200 \
  --compare "$OUT" \
  --out figures/2026-08-29-c2f-3d/phase0-baseline/legskel_bout28_fly0_f200_selfcompare_smoketest.png

# compare_stac_fits.py: white=observed, cyan=fitted marker sites, orange=anatomical
# sites, rendered through MuJoCo. --frame indexes INTO the per-bout stac_ik.h5
# (0..2006 for bout 28), NOT an absolute video frame. No --bout flag exists;
# each --fit LABEL=PATH points directly at a per-fly stac_ik.h5.
XML=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml
ANAT=configs/anatomy/v1.yaml
python scripts/viz/compare_stac_fits.py \
  --fit fly0="$OUT/bouts/bout_00028/fly0/stac_ik.h5" \
  --fit fly1="$OUT/bouts/bout_00028/fly1/stac_ik.h5" \
  --xml "$XML" --anatomy "$ANAT" \
  --out figures/2026-08-29-c2f-3d/phase0-baseline/compare_stac_fits_f200.png --frame 200
```

`$OUT` = `/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/pose`
(the value of `cfg.outputs.out` for `paths=hyak recording=session0`).

### Procedural notes / deviations from the brief

1. **Step 2's `+bout_ids=28` fails** — `pipeline.yaml` already defines
   `bout_ids: ''`, so hydra's `+` (append-only) override errors
   (`Could not append to config. An item is already at 'bout_ids'`). Used
   `bout_ids=28` (no `+`), matching existing repo convention/memory. This
   ran cleanly and printed `[courtship] processing 1 bout(s): [28]`.
2. **Step 3's `hydra.compose("config", ...)` fails** — `configs/config.yaml`
   has no `recording`/`outputs` defaults group, so overriding
   `recording=session0` errors (`Could not override 'recording'`). Used
   `hydra.compose("pipeline", ...)` instead (the actual `config_name` of
   `scripts/run_bout.py`'s `@hydra.main`), which resolves
   `cfg.outputs.out` correctly and matches the real run root.
3. **Both flies were already `DONE`** from a prior run of this exact
   codebase state (arrays dated 2026-08-27 22:29-23:48, i.e. ~2 days before
   this task). `scripts/run_bout.py` has one uncommitted local diff (a
   Stage-A `centroids` variable-scoping fix) that is a no-op here because
   Stage A is skipped once `kp2d.npz` exists; no other uncommitted change
   (`build_detector_dataset.py`, `tests/...`, `viz/views/sidebyside.py`)
   touches Stages A-E's numerics. Re-running `run_bout.py` exercised the
   resume/skip path exactly as designed and confirms these arrays represent
   the current pipeline; it did not perform fresh Stage A-E computation.
   Flagging this because Task 1's premise ("the only artifact that cannot be
   regenerated after configs change") assumes a fresh run — this baseline is
   current-state-valid but not freshly executed today.

---

## Fix round 1 correction (2026-08-29)

Two critical findings from an independent review of this report required
investigation and correction. No pipeline source changes, no submodule
changes, no re-running the pipeline — see the "Fix round 1" section of
`.superpowers/sdd/2026-08-29-coarse-to-fine-3d/task-1-report.md` for the
full evidence trail; summary below.

**1. The fly0 NaN block (offsets 1502-2006, 505/2007 frames) is a permanent
loss for the last quarter of the bout, not a localized ~20-frame proximity
spike, and it is deeper than "the female is lost":**

- Raw triangulation (`kp3d.npz`) is only *partially* destroyed in this span:
  413/505 frames (82%) have every keypoint NaN, but 92/505 (18%) have most or
  ALL 50 keypoints present and finite (20 frames are 100% complete). Yet
  `qpos_refined.npz`/`outputs.h5` (Stage C STAC-IK output) is **fully NaN
  for all 505 frames without exception**, including the 20 with perfectly
  complete raw input. `qc_perframe.reproj_px` reprojects the FITTED FK sites
  (not raw triangulation), so it goes NaN in lockstep with the fitted qpos.
- This aligns with a STAC clip boundary (`configs/stac/courtship.yaml`:
  `n_frames_per_clip: 500` → clip starts at 0/500/1000/1500/2000; frame 1500
  is exactly a clip start, and is where the archived qpos first goes NaN).
  `stac_mjx/stac_core_jaxls.py` already contains comments describing
  anti-poisoning protection against "one NaN frame" freezing a whole clip's
  batched LM solve; that protection does not fully hold for bout 28's
  sustained bad stretch. Not confirmed by re-running/instrumenting (out of
  scope here) — reported as the most parsimonious explanation, worth
  confirming computationally in a later task.
- Visually (mask overlays rendered+read at offsets 1550/1700/1900, all 7
  cameras): in several camera views (most consistently `Cam2012630`,
  `Cam2012861`) the SAM3 mask sits on empty background with no fly under it,
  while the real female remains clearly visible and in focus elsewhere in
  the same frame — an identity/mask-tracking failure, not occlusion, not a
  wall, and not a handoff to the male. `Cam2012862` tracks the real animal
  correctly throughout. At least one camera (`Cam2012853`) stays
  "confidently" above `view_conf_thresh=0.6` while intermittently showing
  the same fly-less phantom blob — a "confidently wrong" view the gate does
  not catch (the ~5% blind spot the config's own docstring documents).
- **Corrected language**: fly0's 10.26 px median is supported by only
  1501/2007 frames (74.8% of the bout) — it is silent on the remaining
  25.2%, which is *absent*, not merely worse. The "1.7x vs. cohort 5.9x"
  comparison above describes only that 74.8%; fly1 tracks normally
  (median 6.16 px) through the exact span fly0 has no pose at all. Whether
  that missing quarter should count toward the female/male gap is an
  acceptance-criteria choice for Task 17, not a number this report can
  settle.

**2. The stac-mjx submodule is dirty, but confirmed inert for these
arrays.** `git submodule status` shows `+25a098b...` (checked-out HEAD ahead
of what the superproject records, `763b6f6`). The submodule's own reflog
shows zero commits between `763b6f6` (2026-08-15) and `25a098b` (2026-08-28
14:43 -0700, ~16h AFTER these arrays were written on 2026-08-27); the
current uncommitted diff on top of `25a098b` (`stac.py`, `stac_core.py`,
`stac_core_jaxls.py`) is later still (mtimes 2026-08-28 18:24, matching
`slurm_logs/` wing-rest-prior A/B job timestamps 14:23-22:54 that day). The
diff adds a wing rest-pose regularizer (`JAXLS_Q_REG_TO_REST`/
`JAXLS_REG_GATE`) that is `getattr(cfg.model, KEY, None)`-gated and
short-circuits to inert when the key is absent — confirmed absent from
every config in the repo (`grep -rln` over `configs/`), including
`configs/anatomy/v1.yaml` used here. Net: Stage C/D numerics for bout 28
are byte-identical whether measured against `763b6f6` (what produced these
arrays) or the current dirty tree — but this conclusion is scoped to this
run's config and does not generalize. Exact blob hashes for
`stac.py`/`stac_core.py`/`stac_core_jaxls.py` at working-tree/`25a098b`/
`763b6f6` are recorded in `task-1-report.md`'s Fix round 1 section for
Task 17 to reproduce or invalidate.
