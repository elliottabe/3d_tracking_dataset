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

Expectation (script docstring, verbatim): *"for each mixed recording, the fly
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
