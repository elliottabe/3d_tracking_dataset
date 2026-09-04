# P3a Task 9: pre-launch figure gates (sex labels, copy-paste, contact_pair overlay)

## Pre-launch gates (2026-09-04)

Two gate figures required before the mvq trainer's next run launches: does the
v12 manifest's per-fly sex label agree with anatomy in the four mixed-sex
recordings, and does the multi-view copy-paste compositor produce
geometrically consistent, plausible pairs on real frames. Both were generated
and READ (Read tool, cropped into row-groups for legibility) against the
expectation stated in each script's docstring before writing this section.

Regeneration:

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/.claude/worktrees/mvq-p3a
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 PYTHONPATH=third_party/jarvis_jax:. \
    python scripts/viz/mvq_sex_label_check.py --out figures/2026-09-mvq/p3a_gates
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 PYTHONPATH=third_party/jarvis_jax:. \
    python scripts/viz/mvq_copy_paste_check.py --out figures/2026-09-mvq/p3a_gates
```

Outputs: `figures/2026-09-mvq/p3a_gates/{sex_label_check,copy_paste_check}.png`
+ matching `.json` beside each (gitignored, not committed).

### Gate 1: sex-label check (`sex_label_check.png`, `sex_label_check.json`)

> **Superseded in part** by "Fix wave 2026-09-04: sex resolution" at the end
> of this file. This section's per-recording verdicts still stand, but its
> 2025_10_20_13_20_04 row selection could not see that recording's two
> annotation subsets, and the script's expectation text and panel titles have
> since changed (per-window resolved sex + manifest value, one row per
> subset). Read the later section for the current reading of that recording.

Expectation (script docstring, verbatim, AT THE TIME): *"for each mixed recording, the fly
the manifest calls MALE (orange) is the smaller body with the dark abdomen
tip; the FEMALE (cyan) is larger with a pointed, pale-striped abdomen. In
2025_10_20_13_20_04 only the female is labelled: the unlabelled fly in the
crop must be the smaller, darker one. If any recording shows the reverse, its
`fly_sex` convention is wrong and the export must be fixed before training."*

12 rows (4 recordings x 3 host-fly0 windows spread across each recording) x 2
camera columns (the 2 cameras with the best coverage of the non-host fly, or
of fly0 alone when fly1 is never labelled in the recording). Row/window
selection: for the 3 mixed-both-labelled recordings, EVERY host-fly0 window
was built (44/15/22 windows respectively -- cheap, no jitter change needed)
and ranked by how many of fly1's keypoints actually land inside the 448x448
crop in their best camera, because `n_flies(i) > 1` (labelled somewhere in the
recording) routinely allows fly1 to be entirely cropped out of a given
window's frame -- the first draft of this script picked windows this way and
produced panels where fly1 was a single stray point at the crop edge.

Per-recording verdict:

- **2025_10_20_13_20_04** (train, calib group B, only fly0=female labelled):
  in 2/3 sampled windows the crop shows fly0 alone (no second fly enters this
  particular 448px crop). In the 3rd window (Cam2012861/Cam2012857) a second,
  unlabelled fly IS visible: smaller, lighter/tan-bodied, with clearly visible
  red eyes, next to the larger labelled cyan female with a darker striped
  abdomen. Consistent with the manifest (female labelled, presumed-male
  unlabelled is the smaller one) -- **not reversed**, though evidence is thin
  since the male rarely enters this crop.
- **2026_04_02_17_28_34** (train, calib group A, both labelled): all 3
  sampled windows land on the same tandem courtship pose (head-to-tail, males
  behind/beside the female) with BOTH flies clearly inside every panel. The
  orange (fly1=male) fly is visibly smaller with a compact, dark abdomen tip;
  the cyan (fly0=female) fly is longer with a pointed, pale-striped abdomen.
  **Not reversed** -- this recording gives the strongest, cleanest confirming
  evidence of the four.
- **2026_04_02_12_11_50** (val, calib group C, both labelled): fly1 is weakly
  represented in this recording's windows (only 3/15 host-fly0 windows have
  ANY fly1 keypoint inside the crop, best case 20/~48 keypoints) -- the
  sampled windows show fly0 (cyan, female) large and centred, with fly1
  (orange) only a small, dark, partial silhouette at the crop's edge. No
  reversal evidence, but the visual check here is **weak/inconclusive**
  by coverage, not contradictory.
- **2026_04_02_15_25_51** (val, calib group A, both labelled): a clear
  mounting pose -- fly1 (orange, male) on top, smaller and compact; fly0
  (cyan, female) below, larger, with a striped abdomen visible extending past
  the male's body. **Not reversed.**

**No recording shows a reversed `fly_sex` convention.** All four are
consistent with the manifest's fly0=female / fly1=male labelling; two
(17_28_34, 15_25_51) give strong direct visual confirmation, one
(2025_10_20) gives weak but consistent confirmation, and one (12_11_50)
gives inconclusive-but-not-contradictory evidence due to fly1 rarely
entering the crop. Proceeding with training is **not blocked** by this gate.

### Gate 2: copy-paste check (`copy_paste_check.png`, `copy_paste_check.json`)

Expectation (script docstring, verbatim): *"in every one of the 7 cameras the
pasted donor (orange labels) sits at the same place relative to the host
(cyan) -- never floating, offset, or missing in one view -- its labels lie on
its own body, host keypoints under the donor are drawn hollow (occluded), and
the contact rows (sep <= 15 units) show the two bodies touching or
overlapping like a mounting pair."*

6 rows (3 contact, sep<=30u; 3 far, sep>30u; naturally included one
male-donor/male-host same-sex pair, so no forced substitution was needed) x 7
camera columns, drawn from `V12WindowDataset(root, "train", T=1, train=True,
copy_paste=CopyPasteParams(p=1.0, max_tries=30))` and `ds.paste_window(i,
rng)` over random single-fly windows (`n_flies(i)==1 and
unlabelled_sex(i)==-1`, the documented guard). `paste_window` never rejected
(0/25 draws returned `None`) -- no rejection-pattern concern to report. (This
run is AFTER Ruling B's `contact_sep` widening below; row seps and counts
below reflect the re-run, not the original 15-unit figure.)

**Mid-task finding (user-flagged), then corrected by the controller: headless
donors are by design, not a defect.** The first render put a real anomalous
window in row 0: donor window `(2026_06_09_15_38_35, fly0, frame 412)` has
ZERO visible Antenna/Eye keypoints in ALL 7 of its own cameras (`has3d` false
at every head landmark, independent of copy-paste). Verified directly
against the dataset (not just the figure):
`ds[2608]["vis2d"][0,0][:, head_idx].sum(0)` is `[0,0,0,0,0,0,0]` across all
7 cameras for the 3 head keypoints, vs. 7/7 for every other window checked.
Confirmed the compositor itself was NOT at fault (view_shifts consistent,
keypoints landed at the geometrically correct shifted location in every
camera that had donor mask content). **Controller ruling: this is not a
label/data defect.** The manifest carries `behavior: "headless"` for 5
recordings (verified directly -- `2026_06_09_15_00_41`, `2026_06_09_15_21_14`,
`2026_06_09_15_38_35`, `2026_06_09_15_46_55`, `2026_06_10_15_05_02`; the
offending donor's recording, `2026_06_09_15_38_35`, is one of them) -- a real
experimental condition, so a headless donor is a legitimate fly appearance
and the production loader/augmentation applies NO head filter. Kept the
check-script-only `HEAD_NAMES`/`_has_head` filter purely for figure
legibility (a check figure meant to demonstrate placement geometry is
clearer without an anatomically ambiguous specimen in it); 12/25 draws in the
re-run below were skipped for this reason. Re-checked the cause of that
fraction directly (not just accepted the controller's framing): of 11 skips
logged in a separate instrumented run, 8 had their skipped fly (donor or
host) drawn from one of the 5 headless-behaviour recordings above, and 3 came
from otherwise-ordinary recordings where that one window's 7 crops simply
didn't happen to contain a visible head (ordinary per-window
framing/occlusion, not a labelling gap either way). So the ~44% figure is
mostly, not entirely, "recording mix of the donor pool" -- the remainder is
ordinary single-window head-visibility variance, and neither component is
label loss. No QC pass or upstream filter is warranted; this is purely a
check-script legibility choice.

Observations against the expectation, on the head-filtered, `contact_sep=(8,
30)` render (post-Ruling-B):

- **Contact rows (3/3, sep 27.6u/26.7u/21.7u)**: in every camera the donor
  (orange) appears directly adjacent to/overlapping the host (cyan) at a
  consistent position -- no floating or per-camera offset. Host keypoints
  under the donor are drawn as hollow cyan circles where the two bodies
  overlap (e.g. row 1, Cam2012857/Cam2012861), and solid cyan elsewhere. At
  this wider (still-realistic, see Ruling B) separation the two bodies read
  as two DISTINGUISHABLE flies in contact/touching rather than one fused
  blob, which is if anything a closer match to the real mounting poses seen
  in gate 1 (`2026_04_02_17_28_34`, `2026_04_02_15_25_51`) than the original
  9-14u renders were. All three contact rows still read as a plausible
  mounting/contact pair.
- **Far rows (3/3, sep 42.6u/41.4u/35.2u)**: bodies clearly separated, not
  overlapping -- expected, and clearly distinct from the contact rows'
  separations even with the widened threshold. At the largest separation
  (42.6u) the donor appears as only a partial cluster of points at the frame
  edge in some cameras and is absent in others -- **this is the correct
  behaviour of `composite()`, not floating/offset geometry**: those cameras'
  panels show no donor pixels because the SHIFTED donor mask/keypoints
  landed outside the 448px crop for that specific view, and `composite()`'s
  own visibility rule (`vis_d & inside & tv & painted`) intentionally leaves
  a camera with no evidence blank rather than fabricate a position. Docstring
  literally says "never ... missing in one view", but that reading only
  holds for pairs close enough that both bodies fit in every crop; the "far"
  bucket exists precisely to also cover far, asymmetric-per-camera framing.
- No offset/mirrored/floating donor was seen in any of the 42 camera panels
  (6 rows x 7 cams) with the head filter in place. The `contact`/`far` split
  still reads sensibly at the new threshold -- re-verified by re-reading the
  regenerated PNG, not assumed from the JSON alone.

**Verdict: the copy-paste compositor is geometrically consistent within its
own documented visibility rule.** No blocking defect found; see Ruling B
below for the corrected `contact_sep`/`CONTACT_UNITS` threshold and its
effect on real-data selection.

## Ruling B (2026-09-04): the 15-unit contact threshold was wrong -- corrected to 30

The controller's independent check of real mating-pair windows (val
`#60`-`#67`, the step-14000 failure cohort the `contact_pair` cohort exists
for) found centroid separations of ~20-26 units -- ABOVE the original 15-unit
"contact" cutoff used in three places, so the cohort/case that exists
specifically to surface those failures was silently selecting nothing.
Fixed by widening the threshold to 30 units (3mm) everywhere it appears,
with a single source of truth:

1. `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py`: new module-level
   `CONTACT_UNITS = 30.0` (comment: "3 mm; real mounting pairs have centroid
   gaps of ~24-30 units, see p3a-notes.md"); `_cohorts`'s `_contact` now
   compares against it instead of a bare `15.0`.
2. `scripts/viz/mvq_overlay.py`: `_is_contact` now imports and uses that same
   `CONTACT_UNITS` from `jarvis_jax.train.train_mvq` (decoupled from
   `mv_copy_paste.CopyPasteParams` -- the diagnostic threshold and the
   augmentation's own `contact_sep` are now independent knobs, per the
   controller's explicit ask, even though they happen to share a value
   today).
3. `third_party/jarvis_jax/jarvis_jax/data/mv_copy_paste.py`:
   `CopyPasteParams.contact_sep` default `(8.0, 15.0)` -> `(8.0, 30.0)`,
   comment updated to explain the stacked-pair regime now brackets the real
   24-30-unit mounting-pair range (with the low end still giving heavier
   overlap than reality is ever this close).

**Real-data census re-run with the new threshold** (`_is_contact` over the
val split, 153 windows, same code path `mvq_overlay.py` uses):

```
CONTACT_UNITS 30.0
n_windows(val) 153   n_contact_pairs 26
```

All 26 hits are in `2026_04_02_15_25_51` (the same recording gate 1 flagged
as the clearest mounting-pose confirmation) at window indices **42-67**,
seps 20.95-25.97 units, i.e. exactly the mating-pair failure cohort the
`contact_pair` case exists for -- **val #60-#67 ARE selected**, as the
controller expected:

```
i=42  frame416375  sep=25.97   i=43  frame416375  sep=25.97
i=44  frame416376  sep=25.92   i=45  frame416376  sep=25.92
i=46  frame416377  sep=25.83   i=47  frame416377  sep=25.83
i=48  frame416378  sep=25.73   i=49  frame416378  sep=25.73
i=50  frame416379  sep=25.65   i=51  frame416379  sep=25.65
i=52  frame416380  sep=25.50   i=53  frame416380  sep=25.50
i=54  frame416381  sep=25.46   i=55  frame416381  sep=25.46
i=56  frame416382  sep=25.42   i=57  frame416382  sep=25.42
i=58  frame416408  sep=25.64   i=59  frame416408  sep=25.64
i=60  frame416555  sep=23.09   i=61  frame416555  sep=23.09
i=62  frame416576  sep=22.26   i=63  frame416576  sep=22.26
i=64  frame416617  sep=20.95   i=65  frame416617  sep=20.95
i=66  frame416662  sep=22.56   i=67  frame416662  sep=22.56
```

(Windows come in host0/host1 pairs at the same frame, hence the doubled
indices -- both hosts of the same two-fly frame separately qualify.) At the
old 15-unit threshold this count was 0.

**Effect on `mvq_copy_paste_check.png`**: re-ran with the widened
`contact_sep=(8, 30)` default (see "Observations" above, now updated in
place) -- contact rows now sit at 21.7-27.6u instead of 9.5-13.8u, reading as
two distinguishable touching/mounting bodies rather than one fused blob
(arguably closer to the real courtship poses in gate 1), and the contact/far
split remains visually unambiguous.

**Tests**: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu pytest
tests/test_mv_copy_paste.py tests/test_train_mvq_smoke.py -q` -> **18 passed**
(193s), pristine. `test_sample_offset_respects_separation_and_plane` (asserts
`8.0 <= |D| <= 60.0` across both contact_sep and far_sep draws) and
`test_composite_labels_are_geometrically_exact_and_host_is_occluded` (uses a
fixed `D=12.0`, inside both the old and new contact range, and doesn't read
`contact_sep` at all) both still hold as anticipated.

**Spec updated**: `docs/specs/2026-09-04-mvq-p3a-identity-existence-design.md`
§6 (`sep ~ U(8, 30)`) and §7 (`contact_pair`: "closer than 30 units"), each
with a parenthetical "(amended 2026-09-04 after the gate: real mounting
pairs are 24-30 units apart)".

## Overlay script changes (`scripts/viz/mvq_overlay.py`)

- Added `contact_pair` case: `sel = [r for r in rows if r["contact"]]`, where
  `contact` is computed per-window via a new `_is_contact(ds, i)` helper using
  `ds.fly_centroids(i)` and `CONTACT_UNITS` (see Ruling B -- imported from
  `jarvis_jax.train.train_mvq`, the same constant `_cohorts`'s
  `contact_pair` val cohort uses, decoupled from `mv_copy_paste`).
- Added `slot` (the oracle instance index already computed as `inst`) and
  `sex_prob` (`sigmoid(out["sex_logit"][0, inst])`) to each row dict and to
  the y-axis row label: `f"#{i} {F/M} grp{g}\nslot{inst}
  pF={sex_prob:.2f}\n{mm:.2f}mm"`.
- Default `--cases` changed to `female,two_fly,contact_pair,worst`.
- Not run against a real checkpoint in this task (no trained mvq run exists
  yet for P3a). Verified import-clean
  (`python -c "import ast; ast.parse(open('scripts/viz/mvq_overlay.py').read())"`)
  and `python scripts/viz/mvq_overlay.py --help` both pass. `_is_contact` was
  additionally smoke-tested directly against the real dataset (train+val, no
  model needed) -- with `CONTACT_UNITS=30` it now selects the real
  `#60`-`#67` mating-pair cohort in val (see Ruling B census above), unlike
  the original 15-unit version which selected nothing in either split.

## Files

- `scripts/viz/mvq_sex_label_check.py` (new)
- `scripts/viz/mvq_copy_paste_check.py` (new)
- `scripts/viz/mvq_overlay.py` (modified: `contact_pair` case, `slot`/`sex_prob` row fields + label, new default `--cases`, `CONTACT_UNITS` import)
- `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py` (modified, Ruling B: `CONTACT_UNITS = 30.0` module constant, `_cohorts`'s `_contact` uses it)
- `third_party/jarvis_jax/jarvis_jax/data/mv_copy_paste.py` (modified, Ruling B: `CopyPasteParams.contact_sep` default `(8.0, 15.0)` -> `(8.0, 30.0)`)
- `docs/specs/2026-09-04-mvq-p3a-identity-existence-design.md` (modified, Ruling B: §6/§7 contact range/rule amended to 30 units)
- `figures/2026-09-mvq/p3a_gates/{sex_label_check,copy_paste_check}.{png,json}` (gitignored, not committed; regenerate with the commands above -- copy_paste_check regenerated post-Ruling-B, sex_label_check unaffected by either ruling)

## Fix wave 2026-09-04: sex resolution

Whole-branch review finding C1: `V12WindowDataset.__init__` collapsed sex to
ONE value per `(recording, fly)` in a loop over framesets -- last one wins,
i.e. a silent coin flip whenever a `(rec, fly)` pair spans annotation subsets
that disagree. Verified on the real export
(`red_data_3d_v12_export0902`, train split): exactly one pair does.

```
2025_10_20_13_20_04 fly0
  courtship_20_04_male    677 framesets  annotation sex = male    frames  84143-439478
  20_04_female_climbing    15 framesets  annotation sex = female  frames 446642-447638
  manifest: sex "mixed", n_flies 2, fly_sex {fly0: female, fly1: male}, sex_source "dirname"
```

Every other `(recording, fly)` in train and val is single-subset and
self-consistent (checked exhaustively, both splits).

### Gate figure reading (`figures/2026-09-mvq/p3a_gates/sex_label_check.png`)

Regenerate:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 PYTHONPATH=third_party/jarvis_jax:. \
    python scripts/viz/mvq_sex_label_check.py --out figures/2026-09-mvq/p3a_gates
```

The script now (a) titles every panel with BOTH the per-window resolved sex
(`ds=`, what training consumes) and the manifest value (`man=`), flagging
disagreements with `[ds!=man]`, and (b) guarantees one row per annotation
SUBSET of each recording, so 20_04's 15-frameset `20_04_female_climbing`
block is sampled alongside its 677-frameset `courtship_20_04_male` block
instead of being reachable only by luck of frame-position sampling. Rows
drawn: 3 x `2025_10_20_13_20_04` (1 male-block, 2 female-block), 3 each of
`17_28_34`, `12_11_50`, `15_25_51`.

**What the PNG shows (read with the Read tool, rows 0-2 cropped for
legibility), stated plainly.** In BOTH 20_04 blocks the labelled (cyan) host
is the same animal in appearance: a compact dark-abdomen fly. In the
`20_04_female_climbing` rows a second, UNLABELLED fly is visible at the upper
left of every panel -- larger, with a visibly pale-banded, longer abdomen. In
the `courtship_20_04_male` rows the labelled fly is alone in the crop. So the
two blocks are not visually distinguishable from each other, which is
consistent with one fly (one sex) across all 692 framesets -- and
inconsistent with the annotation `sex` field flipping between them.

**User verdict 2026-09-04: all 20_04 windows show a female; the annotation
`sex` in subset `courtship_20_04_male` is wrong; the manifest is
authoritative.** So `is_female` for 20_04 stays `female` for all 692 windows,
the earlier female-host census (879/2661 = 0.33) and the P2 balanced sampler
were therefore CORRECT, and the 677 windows were never mislabelled in
training -- what was wrong was only that the value came out of an unordered
last-one-wins loop rather than a stated rule.

**Concern to carry forward (my measurements disagree with the visual
verdict, so it is recorded rather than buried).** Two size invariants,
computed from labels only (no images), over the whole train split:

| recording (block) | manifest sex | n | Antenna_Base-Abd_tip (units) | EyeL-EyeR (units) |
|---|---|---|---|---|
| 2026_01_29_14_09_33 | female | 88 | 28.63 +- 0.50 | 5.65 +- 0.43 |
| 2026_04_02_17_28_34 fly0 | female | 44 | 26.94 +- 0.58 | 5.44 +- 0.45 |
| 2026_01_13_18_47_45 | male | 490 | 24.29 +- 0.43 | 4.91 +- 0.50 |
| 2026_02_09_22_26_25 | male | 301 | 23.46 +- 0.51 | 5.09 +- 0.39 |
| 2026_04_02_17_28_34 fly1 | male | 44 | 23.37 +- 2.31 | 4.69 +- 0.31 |
| **20_04 (`courtship_20_04_male` block)** | female (manifest) | 677 | **23.22 +- 0.73** | **4.31 +- 0.42** |
| **20_04 (`20_04_female_climbing` block)** | female (manifest) | 15 | **23.65 +- 0.54** | **4.87 +- 0.35** |

Both 20_04 blocks measure at or below the known-MALE range on both metrics,
and are statistically indistinguishable from each other (which does support
"one fly throughout"). The cross-group scale confound was checked and ruled
out: calibration px/unit is 8.073 (A), 8.043 (B), 8.027 (C) -- a 0.4 %
difference, so a world unit means the same thing in 20_04's group B as in the
group A references. I could not reconcile this with the visual verdict; the
ruling stands (manifest wins) and this table is here so the disagreement is
not lost. If the 20_04 host is in fact the male, the `female` cohort and
`female_weight` are inflated by 677/879 of their content, which would matter.

### What changed (`jarvis_jax/data/v12_windows.py`)

- Sex is resolved PER WINDOW by `_resolve_fs_sex(rec, frame, fly)`, order:
  manifest `fly_sex["fly<id>"]` -> that frameset's own annotation `sex` ->
  recording `sex` -> unknown. This deliberately INVERTS
  `data/v5_3d._resolve_sex` (annotation-first), which is unchanged for its
  own callers; the inversion is the ruling above.
- `is_female(i)` reads the per-window value; so do `fly_sex` in the sample,
  the donor index, `paste_window`'s `host_sex`, and the new
  `window_fly_sex(i)`. `fly_sex_code` now takes `(rec, fly, frame)` --
  the frame is required, because the annotation step of the chain is per
  frameset. The other labelled fly is resolved from ITS own frame-0
  frameset, falling back to the manifest by fly id.
- `unlabelled_sex(i)` keeps the manifest-fly-id logic: 20_04's unlabelled
  animal is the male in all 692 windows.
- `__init__` prints ONE warning per `(rec, fly)` whose framesets carry more
  than one KNOWN annotation sex, naming the counts, which value wins, and
  that the export should be checked. Observed on the real export:

```
[v12_windows] 2025_10_20_13_20_04 fly0: framesets disagree on annotation sex
{'female': 15, 'male': 677} -- the manifest's fly_sex='female' WINS
(manifest-first resolution, see the module docstring); check the export's
annotation `sex` for this recording
```

### Corrected census (train, resolved host sex, per window)

```
female 879 / 2661  (0.330)      male 1782 / 2661
```

Unchanged from the pre-fix numbers, by the ruling. (Had the annotation won
instead, it would have been female 202 / 2661 = 0.076.)

### P3b follow-up

- **Fix the `sex` field of the `courtship_20_04_male` annotations in the
  v12 export** (677 framesets of `2025_10_20_13_20_04` fly0 say `male`; the
  fly is a female per the user's read of the gate figure). Until then
  `v12_windows` masks it by resolving manifest-first, and the init warning
  names it on every load. Re-check the size table above when the export is
  fixed -- if it stays male-sized, the manifest entry is what needs revisiting.

## Fix wave 2026-09-04: tolerant checkpoint loading, one shared policy, cohort metrics

### Warm start against the real 30k checkpoint (verified output)

`warm_start_partial` into a 4-slot P3a model, source
`/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_local8_20260904/final`
(verified in the whole-branch review):

```
[mvq] warm start from .../mvq_t1_b16_local8_20260904/final: restored 605/608 leaves;
not restored: ['decoder/e_inst (partial rows 0:3)', 'decoder/heads/sex/bias',
               'decoder/heads/sex/kernel']
```

THREE entries, not two: `e_inst` is reported because it is only partially
restored (its 3 source rows land in rows 0-2, row 3 keeps its fresh init).
Spec §8 said "exactly those two" and has been amended.

`load_mvq_model` is now tolerant through the SAME merge
(`train/checkpoint.py::merge_state_by_path`, also used by
`warm_start_partial`; `replicated_abstract_tree` + `restore_own_tree` build
the target from the CHECKPOINT's own stored tree). Before this it built the
target from the fresh model and raised on every pre-P3a checkpoint -- so no
figure or benchmark script could open the checkpoint that IS the P3a
baseline. Loading the 30k run as ITSELF (its own `mvq_run.json`, so
`n_instances = 3`) reports only the sex head:

```
[mvq] load .../mvq_t1_b16_local8_20260904/final: 2 leaf/leaves NOT restored
(kept at fresh init): ['decoder/heads/sex/bias', 'decoder/heads/sex/kernel']
```

`meta["_unrestored_leaves"]` carries that list to callers, and
`mvq_overlay.py` uses it to report `sex_prob = nan` rather than a number from
a randomly-initialised head.

### 30k baseline: `mask_containment` on val (the P3a acceptance reference)

Produced by `scripts/viz/mvq_overlay.py` (which now computes containment with
the SAME shared `models/mvq/policy.mask_containment` `train_mvq.evaluate`
uses, so this is directly comparable to a run's own val numbers):

```bash
module load cuda/12.9.1; export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset LD_LIBRARY_PATH JAX_PLATFORMS
export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 HF_TOKEN= \
       CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
       PYTHONPATH=third_party/jarvis_jax:.
RUN=/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_local8_20260904/final
python scripts/viz/mvq_overlay.py --run $RUN --split val --n 6 \
    --cases female,two_fly,contact_pair,worst \
    --out figures/2026-09-mvq/mvq_t1_b16_local8_final30k/unprompted
python scripts/viz/mvq_overlay.py --run $RUN --split val --n 6 \
    --cases female,two_fly,contact_pair,worst --prompted \
    --out figures/2026-09-mvq/mvq_t1_b16_local8_final30k/prompted
```

| metric (val, 153 windows) | prompted | unprompted |
|---|---|---|
| **mask_containment (mean)** | **0.8768** (153/153 scored) | **0.8323** (49/153 scored) |
| mask_containment, contact_pair | 0.7766 (n=26) | 0.7862 (n=25) |
| mask_containment, two_fly | 0.8310 (n=74) | 0.8323 (n=49) |
| mask_containment, single_fly | 0.9197 (n=79) | n/a (0 scored) |
| mask_containment, female | 0.8541 (n=67) | 0.8126 (n=25) |
| mpjpe3d oracle / policy (units) | 1.096 / 1.096 | 1.082 / 1.594 |
| policy_miss_frac | 0.000 | 0.680 |
| policy_miss_frac, single_fly | 0.000 | **1.000** |
| policy_miss_frac, contact_pair | 0.000 | 0.038 |

The unprompted column only scores containment where the policy named an
instance at all, which is why its n is 49, not 153. `policy_miss_frac` 1.000
on single-fly windows reproduces the step-14000 finding (spec §1: "misses
99 % of single-fly ... val windows") on the FINAL checkpoint -- it did not
improve over training. These are the numbers P3a's acceptance targets
(unprompted miss < 5 % on single-fly; containment HIGHER on `contact_pair`)
are measured against. Note the containment baseline is high in absolute terms
(0.78-0.92): the host mask is generous relative to a 50-keypoint skeleton, so
the informative comparison is the DELTA, not the level.

`sex_prob` is `nan` everywhere in this baseline (the 30k run has no trained
sex head); its slots are labelled `untyped, legacy` in the figures because
`n_instances = 3`.

### Contact-pair figure reading (unprompted)

`figures/2026-09-mvq/mvq_t1_b16_local8_final30k/unprompted/gate1_contact_pair.png`,
6 rows (val #42-#47, all `2026_04_02_15_25_51` mounting pairs, seps
20.9-26.0 units) x 7 cameras. Rows 0-1 and 4-5 read with the Read tool.

Expectation stated before looking: on a real mounting pair each existing
slot's points should stay on ONE animal; a slot straddling both bodies is the
step-14000 mixing failure. For a LEGACY untyped 3-slot model the weaker
expectation is just "each existing slot is a coherent single body".

**What the PNG shows.** In every row exactly one slot sits on the host fly
and sits on it well -- dots coincide with the white human-label x's at
2.6-3.7 px per camera (panel titles), e.g. #42 cyan slot1, #43 magenta slot0,
#46 cyan slot1, #47 magenta slot0. Slot 2 (orange), which carries the HIGHEST
existence probability in all 26 contact windows (0.95, vs 0.58-0.61 for slot
0 and 0.41-0.44 for slot 1), is a degenerate instance: its points start near
the host's head or abdomen tip and trail off into the crop's blank
out-of-frame padding, in every one of the 7 cameras, in all four rows read.
It is not on the other fly -- it is off the image. That is the uncalibrated,
meaning-free slot P3a's typed slots and label-driven existence targets exist
to remove, and it is exactly what makes the unprompted policy a coin flip:
the always-on slot's centroid competes with the real one on distance to the
ROI origin.

Limitation of this figure, stated so it is not over-read: the white labels
drawn are the HOST fly's only (`vis2d[0]`), so it cannot show whether a slot
lands on the SECOND animal. The per-sample `mask_containment` in
`summary.json` is the quantity that answers that, and on `contact_pair` it is
the lowest of any cohort (0.777 prompted / 0.786 unprompted vs 0.920 on
single-fly) -- consistent with mixing being worst exactly there.

### Spec facts corrected here rather than in the spec

- **Slot 3 has no real positives in train.** Spec §1/§11 treat
  `2026_06_09_15_38_35` as "two females, both labelled, 31 train framesets".
  Verified against the export: only **fly0** is labelled in that recording
  (31 single-fly windows, manifest `n_flies: 2`, `fly_sex {fly0: female,
  fly1: female}`), so its windows carry `unlabelled_sex = female` and are the
  same-sex-pair case with ONE label. Slot 3's only positives in training are
  therefore same-sex COPY-PASTE windows (30 % of pastes); there is no real
  labelled same-sex pair anywhere in train. Spec §11's "slot 3 starvation"
  risk is real and slightly worse than written.
- **Copy-paste never enters a group B or C crop.** Eligible targets (one
  labelled fly AND no unlabelled animal) are 1850 windows, ALL in calibration
  group A. Group B's 692 windows are the 20_04 recording, whose second fly is
  always present-but-unlabelled, so they are ineligible as targets; group C
  has no train windows at all. Donors are drawn from the target's own group,
  so donors are group A only. Spec §11's "donors of both sexes into every
  recording's crops decouple sex from background" therefore holds only within
  group A.

### Other fixes in this wave

- **I1**: `slot_ignore`'s unlabelled-animal mask could ignore a slot that
  HOLDS a labelled fly (a female host in slot 1 with an unlabelled female
  present -- i.e. all 31 `15_38_35` windows). `ignore & ~slot_target` now
  applies in BOTH `mvq_loss` and `evaluate`'s per-slot counting, so the one
  existence answer known for certain is supervised and eval counts the same
  slots the loss did.
- **I2**: `evaluate` emits `sex_acc_{cohort}`, `mask_containment_{cohort}`
  and `policy_miss_frac_{cohort}` for every cohort, and `_cohorts` gains
  `single_fly`. Spec §8 grades `sex_acc` on group C, containment on
  `contact_pair` and the miss fraction on single-fly windows; no aggregate
  could answer those.
- **I3/I4**: new `jarvis_jax/models/mvq/policy.py` holds the ONE
  `policy_instance` (typed slots 1..I-1 for a 4-slot model, all slots for a
  legacy one) and `mask_containment`. `evaluate`, `mvq_overlay.py` and
  `mvq_val_baselines.py` all call them; the three inline copies (two of them
  commented "exactly train_mvq.evaluate's policy" while already differing)
  are gone.
- **I5**: `composite` rejects a donor with no mask content in ANY
  target-valid camera. It used to paint nothing and still write the donor's
  labels -- a fly claimed with zero pixel evidence in every view.
- **Launch hygiene**: `run_training` skips `warm_start_partial` when its own
  `ckpt/` already has a step (resume beats warm start, so a requeue was
  reading the source only to discard it). The skip is printed.

### Tests

```bash
cd third_party/jarvis_jax
JAX_PLATFORMS=cpu pytest tests/test_dinov3.py tests/test_mvq_geometry.py \
  tests/test_mvq_model.py tests/test_mvq_losses.py tests/test_mvq_matching.py \
  tests/test_mvq_attention.py tests/test_v12_windows.py tests/test_mv_augment.py \
  tests/test_mv_copy_paste.py tests/test_train_mvq_smoke.py \
  tests/test_checkpoint_warm_start_partial.py -q -m "not gpu"
# 102 passed, 4 deselected (was 96 passed / 4 deselected before this wave)
```

New tests: per-window manifest-first sex resolution + the disagreement
warning (`test_v12_windows.py`), `load_mvq_model` on a pre-P3a checkpoint and
the warm-start skip on requeue (`test_train_mvq_smoke.py`), a labelled fly's
slot is never ignored (`test_mvq_losses.py`), empty-mask donor rejection
(`test_mv_copy_paste.py`).

### Files (fix wave)

- `third_party/jarvis_jax/jarvis_jax/data/v12_windows.py` (per-window manifest-first sex, `fly_sex_code(rec, fly, frame)`, `window_fly_sex`, disagreement warning)
- `third_party/jarvis_jax/jarvis_jax/models/mvq/policy.py` (new: `policy_instance`, `typed_candidates`, `mask_containment`, `EXIST_THRESH`)
- `third_party/jarvis_jax/jarvis_jax/models/mvq/checkpoint.py` (tolerant `load_mvq_model` both branches, `_unrestored_leaves`)
- `third_party/jarvis_jax/jarvis_jax/train/checkpoint.py` (`merge_state_by_path`, `replicated_abstract_tree`, `restore_own_tree`; `warm_start_partial` now a one-liner over them)
- `third_party/jarvis_jax/jarvis_jax/train/losses_mvq.py` (I1)
- `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py` (shared policy/containment, I1 in eval, per-cohort metrics, `single_fly` cohort, warm-start skip)
- `third_party/jarvis_jax/jarvis_jax/data/mv_copy_paste.py` (I5)
- `scripts/viz/mvq_sex_label_check.py` (per-window + manifest sex in titles, one row per annotation subset)
- `scripts/viz/mvq_overlay.py` (all existing slots drawn per-slot-coloured, `mask_containment` in summary.json, legacy-checkpoint safe)
- `scripts/benchmark/mvq_val_baselines.py` (shared policy)
- `docs/specs/2026-09-04-mvq-p3a-identity-existence-design.md` (§6 rejection/label + donor-pool rules, §8 three unrestored leaves + tolerant loader, §9 test list)
- `figures/2026-09-mvq/p3a_gates/{sex_label_check.{png,json},sex_20_04_zoom.png}` and `figures/2026-09-mvq/mvq_t1_b16_local8_final30k/{prompted,unprompted}/{gate1_*.png,summary.json}` (gitignored; regenerate with the commands above)
