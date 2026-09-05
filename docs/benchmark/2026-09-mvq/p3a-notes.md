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

> **Superseded for 2025_10_20_13_20_04** by "Fix wave 2026-09-04: sex
> resolution" at the end of this file. Its verdicts for the other three
> recordings still stand, but its 20_04 row selection could not see that
> recording's two annotation subsets, and its conclusion there ("female
> labelled, presumed-male unlabelled") is WRONG for the 677-frameset
> `courtship_20_04_male` block: the labelled fly is the male there. The
> script's expectation text and panel titles have also changed (per-window
> resolved sex + manifest value, `[ds!=man]` flag, one row per subset).

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

**What the PNG shows (read with the Read tool, rows cropped into groups for
legibility), stated plainly.** The `courtship_20_04_male` row (f84143) shows
the labelled (cyan) host alone in the crop: a compact fly with a dark, blunt
abdomen and the wings folded over it, and the panel is flagged `[ds!=man]`
(`ds=male man=female`). The `20_04_female_climbing` rows (f446642, f446923)
show the labelled host on the arena wall with a SECOND, unlabelled fly at the
upper left of every panel -- that companion is the larger one, with a longer,
visibly pale-banded abdomen. My own eye could not separate the two blocks'
labelled hosts from each other in these 448 px crops (both read compact and
dark, the climbing one partly obscured by its wall pose), so this figure on
its own does not settle the question either way; what it does show is that
the two blocks are DIFFERENT scenes -- one a lone fly, one a pair on a wall.

**FINAL user verdict 2026-09-04 (from a full-frame render with known-sex
reference recordings, which is the authoritative call): the labelled fly is
the MALE in the `courtship_20_04_male` block (677 framesets) and the FEMALE
in the `20_04_female_climbing` block (15 framesets).** The two annotation
subsets label DIFFERENT ANIMALS under the same fly id, so the annotators'
own `sex` field is right for both blocks and the manifest's single per-fly
value (`{fly0: female, fly1: male}`, `sex_source: "dirname"`) cannot
represent this recording at all. Resolution is therefore per window and
ANNOTATION-FIRST.

**Consequence for the P2 runs: their balanced sampler treated 677 male-host
windows as female.** The collapsed per-(rec, fly) value resolved to `female`
for all 692 windows, so `female_weight` and the `female` cohort counted the
courtship block's male as a female for every P2/30k training run. Val is
unaffected (no val recording disagrees; still 67/153 female windows), so the
30k baseline table in the next section stands as measured.

**Supporting evidence: two size invariants, labels only, over the whole train
split.** These AGREE with the verdict for the 677 (male-sized) and are
inconclusive for the 15 (also male-sized by this metric, but n=15 in a wall
pose, and the user's visual call from full frames is authoritative):

| recording (block) | ruled sex | n | Antenna_Base-Abd_tip (units) | EyeL-EyeR (units) |
|---|---|---|---|---|
| 2026_01_29_14_09_33 | female | 88 | 28.63 +- 0.50 | 5.65 +- 0.43 |
| 2026_04_02_17_28_34 fly0 | female | 44 | 26.94 +- 0.58 | 5.44 +- 0.45 |
| 2026_01_13_18_47_45 | male | 490 | 24.29 +- 0.43 | 4.91 +- 0.50 |
| 2026_02_09_22_26_25 | male | 301 | 23.46 +- 0.51 | 5.09 +- 0.39 |
| 2026_04_02_17_28_34 fly1 | male | 44 | 23.37 +- 2.31 | 4.69 +- 0.31 |
| **20_04 `courtship_20_04_male`** | **male** (annotation) | 677 | **23.22 +- 0.73** | **4.31 +- 0.42** |
| **20_04 `20_04_female_climbing`** | **female** (annotation) | 15 | **23.65 +- 0.54** | **4.87 +- 0.35** |

The 677-frameset block sits squarely in the known-MALE range on both metrics
-- direct independent support for the ruling, and for the earlier
manifest-first reading having been wrong. The 15-frameset block measures
male-sized too, which the ruling does not follow; that block is small (15
windows, 0.6 % of train) and its pose is a wall climb, so the invariant is
weak there. The cross-calibration-group scale confound was checked and ruled
out: px per world unit is 8.073 (A), 8.043 (B), 8.027 (C) -- 0.4 %, so a unit
means the same thing in 20_04's group B as in the group A references.

### What changed (`jarvis_jax/data/v12_windows.py`)

- Sex is resolved PER WINDOW by `_resolve_fs_sex(rec, frame, fly)`, order:
  that frameset's OWN annotation `sex` -> manifest `fly_sex["fly<id>"]` ->
  recording `sex` -> unknown. That is `data/v5_3d._resolve_sex`'s chain
  (called directly), applied to the window's own frameset instead of once
  per (recording, fly).
- `is_female(i)` reads the per-window value; so do `fly_sex` in the sample,
  the donor index, `paste_window`'s `host_sex`, and the new
  `window_fly_sex(i)`. `fly_sex_code` now takes `(rec, fly, frame)` -- the
  frame is REQUIRED, because the annotation step of the chain is per
  frameset. The other labelled fly is resolved from ITS own frame-0
  frameset, falling back to the manifest by fly id when it has none.
- `unlabelled_sex(i)`: in a manifest-`mixed`, `n_flies == 2` recording with
  exactly one labelled fly, the unlabelled animal is the OPPOSITE of that
  window's own host sex (`1 - host`; `SEX_PRESENT_UNKNOWN` if the host's sex
  is unknown). Reading the manifest's missing fly id instead would claim a
  male unlabelled animal in the 677 windows whose LABELLED animal is that
  male. Every other case keeps the manifest-id logic.
- `__init__` prints ONE warning per `(rec, fly)` whose framesets carry more
  than one KNOWN annotation sex, naming the counts, that the ANNOTATION
  wins, and the manifest value that disagrees. Observed on the real export:

```
[v12_windows] 2025_10_20_13_20_04 fly0: framesets disagree on annotation sex
{'female': 15, 'male': 677} -- the ANNOTATION sex WINS per window
(annotation-first resolution, see the module docstring); the manifest's single
fly_sex='female' disagrees and is NOT used for these framesets
```

### Corrected census (train, resolved host sex, per window)

```
female  202 / 2661  (0.0759)        male 2459 / 2661  (0.9241)
```

Recomputed on the real export after the flip. Under the old collapsed
behaviour it was **female 879 / 2661 (0.330)** -- the 677 courtship-block
male-host windows were counted as female, which is exactly what the P2
balanced sampler and `female` cohort consumed. Val is unchanged (67 / 153
female windows; no val recording disagrees).

### P3b follow-up

- **The v12 export's manifest cannot express `2025_10_20_13_20_04`.** Its
  `fly_sex` is a single value per fly id, but this recording's two annotation
  subsets track different animals under fly id 0. The loader now handles it
  correctly per window; the export would be cleaner with per-subset
  `fly_sex`, or with the two subsets split into separate recording entries.
  Until then the init warning names the recording on every load, and any code
  reading `manifest[rec]["fly_sex"]` directly (rather than through
  `V12WindowDataset`) is wrong for those 677 framesets -- the gate script
  `scripts/viz/mvq_sex_label_check.py` is the reference for showing both
  values side by side.
- Re-check the size table above if the export changes: the 15-frameset
  `20_04_female_climbing` block measures male-sized on both invariants, so
  either it is a third labelling wrinkle or the size metric is unreliable in
  a wall-climb pose.

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

New tests: per-window annotation-first sex resolution + the disagreement
warning (`test_v12_windows.py`), `load_mvq_model` on a pre-P3a checkpoint and
the warm-start skip on requeue (`test_train_mvq_smoke.py`), a labelled fly's
slot is never ignored (`test_mvq_losses.py`), empty-mask donor rejection
(`test_mv_copy_paste.py`).

### Files (fix wave)

- `third_party/jarvis_jax/jarvis_jax/data/v12_windows.py` (per-window annotation-first sex, `fly_sex_code(rec, fly, frame)`, `window_fly_sex`, disagreement warning)
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

## P3a run `mvq_t1_b16_p3a_20260904` — result (2026-09-04, 10k steps, warm start from the 30k final, 8x L40S local, ~1.05 s/step)

Validation every 2000 steps, both modes, 153 val windows; the 30k final checkpoint is the baseline
(`figures/2026-09-mvq/mvq_t1_b16_local8_final30k_p2code/*/summary.json` for containment/contact-pair,
`final/mvq_run.json` for the rest). Full trend table: `p3a_val_trend.txt` beside this file.

Spec §8 acceptance (unprompted = the mask-free typed-slot route; oracle instance unless stated):

| criterion | target | 30k baseline | P3a @10k | pass |
|---|---|---|---|---|
| unprompted policy miss fraction, single-fly windows | < 5 % | 100 % | 0.0 % | yes |
| existence precision / recall, female slot | > 0.9 | n/a | 1.00 / 0.98 | yes |
| existence precision / recall, male slot | > 0.9 | n/a | 1.00 / 0.96 | yes |
| sex accuracy, group C (held-out calibration) | > 0.95 | n/a | 1.00 | yes |
| mask containment, contact_pair cohort (26 windows) | higher than baseline | 0.786 | 0.815 | yes |
| oracle MPJPE not worse anywhere by > 5 % | | all 0.109 mm | all 0.092 mm; group C 0.090 vs 0.093 | yes |

Cohort MPJPE (oracle, mm, 30k -> P3a): two-fly 0.151 -> 0.116; contact pairs 0.275 -> 0.179 (-35 %);
female 0.132 -> 0.114; group A 0.113 -> 0.092; group C 0.093 -> 0.090. Unprompted POLICY error
0.160 -> 0.094 mm (misses 68 % -> 0 %). Prompted oracle 0.111 -> 0.084 mm, but prompted POLICY (slot 0)
0.111 -> 0.107 mm: the prompted slot is now the weaker route (its sex head 0.985 vs 1.00; when
prompted the model under-reports the second fly: male-slot recall 0.84, female-slot recall 0.56).

Curve shape: geometry recovered to baseline by step 500 and plateaued from ~4000; existence
calibrated by 2000; sex accuracy jumped 0.60 -> 0.985 between 2000 and 4000 and reached 1.00 by
8000 (the early male bias resolved without a sampler change). Final-step numbers are within noise
of step 8000.

Bout-28 end-to-end (step-7000 checkpoint, unprompted, `scripts/viz/mvq_bout_video.py`):
identity held across all 2007 frames (fly0 -> female slot, fly1 -> male slot, no swaps observed in
the sampled frames); fly0 missing in 111 frames (local 1648-1772) where she sits at the arena edge,
out of frame in one camera and clipped in three -- the female-slot existence drops to 0.23 there
while ViTPose+DLT emits scattered junk; the prompted pass had 0 missing frames. Video:
`figures/2026-09-mvq/bout28_p3a/bout28_mvq_unprompted_step7000.mp4`.

P3b follow-ups from this run: (1) prompted-slot semantics (prompt a typed slot, or retire the prompt
path); (2) mask-aware presence / prompted fallback for edge frames; (3) mask-free ROI placement
(centre-jitter robustness test on this checkpoint, then a centre detector); (4) per-camera depth
ordering and a normal-direction offset for copy-paste stacking; (5) export fix for the 20_04
annotation sex; (6) log a running-mean loss.

## Bout 28 through the IK pipeline on mvq keypoints (2026-09-04, jobs 39593529 + 39593671)

Inputs staged from the step-7000 unprompted run (`pose_mvq/bouts/bout_00028/unprompted/`): local frames
0-1500 (the female leaves the arena edge-wards after ~1650), keypoints permuted BY NAME to `model.KP_NAMES`
(EyeL-EyeR invariant checked identical before/after), pipeline-facing conf = view visibility (mvq's D4RT
conf3d ~0.05 is not a probability; raw kept as `conf3d_mvq_raw`), a 1500-frame companion masks npz
(`run_bout.py` takes T from the masks), `pipeline.allow_stale_kp3d=true`, `recording.bouts_csv` pointed at a
one-row CSV (the session's unified CSV is a broken symlink). Run root `pose_mvq_ik/`; the ViTPose+DLT fit is
the `_courtship_backup` pose tree (older pipeline version -- its qc.json lacks the newer keys).

| fly | metric | ViTPose+DLT | mvq |
|---|---|---|---|
| female | LOO reprojection, median px | 6.26 | 0.87 |
| female | fitted-keypoint frame-to-frame RMS (outputs.h5 units) | 1.53 | 0.20 |
| female | fitted EyeL-EyeR CV | 6.2 % | 0.7 % |
| male | LOO reprojection, median px | 2.65 | 0.69 |
| male | fitted-keypoint frame-to-frame RMS | 0.385 | 0.234 |
| male | fitted EyeL-EyeR CV | 0.5 % | 0.4 % |
| both | NaN frames after IK | 0 | 0 |
| both | IK fitted vs measured reprojection (mvq only), median px | - | 4.5 / 0.94 (F), 3.6 / 0.88 (M) |

Fitted eye spacing differs between the runs (3.3 vs 4.7 units for the female; each run fits its own body
scale): mvq's matches its observed spacing (4.86 units); the ViTPose fit shrank it -- the marker-offset
absorption failure CLAUDE.md warns about. To be looked at with the scale QC before promotion.

Figures read: `figures/2026-09-mvq/bout28_p3a/ik_compare_fly0_f{669,1337}.png` (compare_stac_fits; mvq
observations sit on the fitted markers, the ViTPose fit has floating observations off the legs/head) and both
`sidebyside_still.png` (mesh pose matches the frame in all three rig views; the male's wing extension is
reproduced by the fit). The female's fit is the one that improved most (7x smoother), consistent with the P2
finding that the female is the hard fly.

## Masked-bout mvq campaign (P3a final)

Task 7 of the mask-free front end, but it is not mask-free: it is the route that takes the 160
courtship bouts SAM3 has already segmented and replaces the pipeline's ViTPose+DLT front half
(Stages A and B) with the P3a mvq lifter, leaving the keypoint filter, body-scale precompute, STAC
IK, polish and viz stages untouched. Checkpoint
`/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p3a_20260904/final` (the 10000-step
save; `configs/mvq/p3a.yaml`).

### Design

* **Windows.** Per frame, each fly's SAM3 mask centroids are triangulated (>= 2 valid views) and the
  two centres become ONE 448-px multi-view crop when they are within 30 units (3 mm) -- the same
  `coarse_centres.plan_windows` rule the coarse pass uses -- else one crop each. A frame with no
  usable centre is NaN for both flies and is counted as `no_centre` in `mvq_meta.json`. Inference is
  UNPROMPTED (`prompt_on=None`); the masks place the crop and nothing else.
* **Typed slots, not mask slots.** `fly0` is the model's FEMALE typed slot (slot 1) and `fly1` its
  MALE typed slot (slot 2), read per frame from the window with the highest existence for THAT slot.
  `exist < 0.5` NaNs that fly's frame -- there is no fallback to an untyped slot, so a frame written
  as "the female" is never quietly some other slot. This is the canonical identity
  `sexing.canonicalize_bout` would enforce, so the lifter writes the bout's `sex.json` itself
  (`male_fly: 1`, `method: "mvq_sex_head"`, per-fly mean `sex_prob`/`exist`) and `canonicalize_bout`
  now treats that method as authoritative -- no swap, no wing-song CV -- below a HUMAN review only.
  A mask `sex_meta` or id-review entry that disagrees is logged in `mvq_meta.json`, never acted on.
* **Confidences.** `conf3d` = the mean per-view visibility sigmoid over cameras and `conf` (2D) = the
  per-view visibility, because mvq's own `conf` head is a D4RT score of ~0.02-0.2 that the pipeline's
  0.3-0.5 gates would read as garbage; the raw head is kept beside them as `conf3d_mvq_raw`.
* **Gates string.** `{"checkpoint": ".../mvq_t1_b16_p3a_20260904/final", "exist_thresh": 0.5,
  "lifter": "mvq", "sha256": "7608843ecbd1a10c", "step": "final"}` (sorted-key JSON) is stamped into
  every kp3d.npz and is byte-equal to what `run_bout.py::stage_b_gate_signature` computes for
  `pipeline.lifter=mvq mvq=p3a` -- so Stage B accepts these files WITHOUT
  `pipeline.allow_stale_kp3d`, and refuses them the moment the config points at other weights. The
  same string makes the lift idempotent.
* **Keypoint/camera order.** The written axis is `cfg.model.KP_NAMES`, permuted BY NAME from the
  model's detector order, asserted equal to KP_NAMES and checked against the rigid EyeL-EyeR spacing
  before anything is written. The camera axis is the canonical `cfg.recording.cameras` order, which
  the mask npz is permuted into by name -- on Session0 bout 28 the npz order really is different
  (`['Cam2012853','Cam2012862','Cam2012855','Cam2012857','Cam2012861','Cam2012630','Cam2012631']`,
  permutation `[5,6,0,2,3,4,1]`), so this is load-bearing, not ceremonial.
* **Out root** `<processed>/courtship/<Session>/<recording>/pose_mvq_p3a/`. Nothing is written into
  `pose/` (the ViTPose+DLT baseline), `pose_mvq/` (the bout-28 render npz) or `pose_mvq_ik/` (the
  1500-frame bout-28 IK test).
* **Chain order.** The lift array runs BEFORE precompute, so `scale.json` pools its rigid-segment
  body scale over the WHOLE recording instead of the single bout precompute seeds -- the
  scale-from-first-bout defect, measured 15.5 % low on Session0. The lifter's `sex.json` is also what
  makes `estimate_recording_scale._determine_identity` return `canonical`, so `scale_by_fly` is
  populated and `scaling.allow_shared_scale` can stay false.

### Window plan on real data (Session0 bout 28, CPU pre-check)

`OutFiles/mvq_task7_smoke/precheck_bout28.py`: 2007 frames, both flies have a usable 3D mask centre
in ALL of them, so no frame is dropped for want of a centre. The two flies are a median 3.71 mm
apart (p10 3.06, p90 4.24), so only 137/2007 frames (6.8 %) merge into one crop and 1870 plan two --
3877 windows in total against the 4014 the old one-window-per-fly path ran. The merge is therefore a
small saving here, not the point; the point is that on the frames where the flies DO touch (the
mounting frames, which is where cross-fly mixing happens) the model sees both animals in one crop
and can use its two-instance decoder rather than being asked to pick a fly out of a crop centred
between them.

### Smoke test: Session0 bout 28 (job 39606799, one A40, 2026-09-04)

`OutFiles/mvq_task7_smoke/smoke_bout28.sh` -- lift the whole bout, rebuild a SAME-CHECKPOINT
reference with the pre-existing `scripts/viz/mvq_bout_video.py --no-render` path, compare, then run
`scripts/run_bout.py ... pipeline.lifter=mvq mvq=p3a` on the result.

**Lift.** 2007 frames in 453 s (4.4 frames/s end to end, 4.7 steady-state; 3877 windows). Missing
frames fly0 121, fly1 0; 0 frames with no mask centre. `sex.json`: `male_fly 1`, `method
mvq_sex_head`, `confidence high` -- mean P(female) 0.965 (fly0) vs 0.0017 (fly1), mean existence
0.948 / 0.998 -- and no `identity_disagreements`, since the masks' human review also says male =
slot 1. Written keypoint order is `KP_NAMES` (`['Scutellum','WingL_base','WingR_base',...]`) and the
observed EyeL-EyeR span is 0.485 mm (fly0) / 0.427 mm (fly1), which matches the 4.86-unit observed
spacing recorded for the step-7000 run above.

**ARM 1 -- new lifter vs the old path, SAME checkpoint, whole bout** (this is the acceptance gate;
the brief's threshold is a median per-keypoint 3D difference < 0.03 mm on frames both call finite):

| fly | frames | median | p90 | max | NaN frac new / ref |
|---|---|---|---|---|---|
| fly0 female | 2007 | **0.0 mm** | 0.0 mm | 6.18 mm | 0.060 / 0.041 |
| fly1 male | 2007 | **0.0 mm** | 0.0 mm | 4.55 mm | 0.000 / 0.000 |

Broken out by what the two code paths actually do differently:

| subset | fly0 median | fly1 median |
|---|---|---|
| 1870 two-window frames (crops IDENTICAL) | 0.0 mm | 0.0 mm |
| 137 merged frames (crop moved to the midpoint) | 0.152 mm | 0.081 mm |

So on every frame where the two paths place the same crop the output is bit-identical, and the merge
moves a keypoint by ~0.1 mm -- a tenth of a leg segment, and the intended behaviour (a crop centred
between two touching flies is what lets the two-instance decoder see both).

The two non-zero *maxima* on the identical-crop subset are the OTHER deliberate difference: the typed
slot is taken from whichever window has the highest existence for that slot, so on 46/1870 frames
(fly0) and 21/1870 (fly1) -- 2.5 % and 1.1 % -- the winner is the OTHER fly's crop, and in ALL of
those frames the chosen window index is indeed the other fly's (checked). The NaN fractions move for
the matching reason: the old path falls back to an untyped slot when the typed one is weak (190
female frames did that), the new one refuses and NaNs; 69 of those 190 are recovered by the
highest-existence rule and 121 stay NaN. Stricter by design.

**ARM 2 -- context, NOT a gate.** The stored `pose_mvq/.../unprompted` arrays were produced by step
7000, which has since been rotated out of the run's `ckpt/`; the campaign uses `final/` (10000
steps). Different weights, so this is a model delta: median 0.011 mm (fly0) / 0.005 mm (fly1), p90
0.262 / 0.016 mm. The two checkpoints agree closely on this bout -- worth knowing before reading the
P3a-vs-P4 comparisons as if `final` were a different model.

**Pipeline continuation (`run_bout.py ... pipeline.lifter=mvq mvq=p3a`).** Stages A and B were
skipped -- the log contains none of their prints (`gray-fill`, `view-gate`, `kp-mask-agree`,
`wing-collapse`, `rigid-repair`, `mask-coverage`: 0 lines) and the Stage-B gate check passed with no
`allow_stale_kp3d`. `sex.json` was left exactly as the lifter wrote it (`male_fly 1`, `applied_swap
false`), which is also what the human ID review for this bout says, so no fly dirs moved. Body scale
came out of the pooled path the lifter's sex.json unlocks:

    [scale] pooled over 2 bout-flies -> 0.011808  identity=canonical
            scale_by_fly={'0': 0.011647, '1': 0.011968}

(the recording's pooled reference is 0.011738, and `scaling.allow_shared_scale` stayed false).

| fly | LOO reproj median px | IK fitted / measured px | ratio | mesh-mask IoU (hard) |
|---|---|---|---|---|
| fly0 female | 0.97 (1884 frames) | 8.11 / 1.11 | 7.3 | 0.029 |
| fly1 male | 0.68 (2007 frames) | 3.65 / 0.78 | 4.7 | 0.025 |

CAVEAT (final fix wave, 2026-09-05): this table was measured on the PRE-COLLAPSE-GUARD lift (the one
`track_qc.json` below flags at 90/2007 frames). Re-lifting bout 28 after the collapse guard landed
(e31b7d1) triggered on 0/2007 frames, so nothing here was actually a same-fly-read-twice collapse and
the numbers stand unchanged.

For scale: the ViTPose+DLT arm on this bout measured 6.26 px (female) / 2.65 px (male) LOO, so the
observations are 4-6x better; the IK is still the bottleneck (fitted/measured 7.3x and 4.7x), exactly
the P2 conclusion. These numbers cover the WHOLE 2007-frame bout, including the tail after ~1650
where the female is at the arena edge, so they are worse than the 1500-frame step-7000 run recorded
above (4.5/0.94 F, 3.6/0.88 M) and are not directly comparable to it.

`track_qc.json` reports the two tracks collapsing on 90/2007 frames (4.8 %): min separation 5.34 vs a
22.4-unit body length. CORRECTED 2026-09-05 (final fix wave): this was NOT a QC signal about the
input that the typed-slot route left unaddressed -- it was the absence, at the time, of the collapse
guard `lift_masked_bout._flush` now has (landed e31b7d1, both typed slots chosen independently can
read the same physical fly on a merged window; the guard keeps the higher-exist slot and NaNs the
other, flagging the frame). Re-lifting bout 28 under the guard triggers it on 0/2007 frames, so this
90-frame figure was the pre-guard defect, not an input property -- see the caveat on the LOO/IK table
above.

**Figure read back** (`pose_mvq_p3a/bouts/bout_00028/fly0/sidebyside_still.png`, three rig views,
opened with the Read tool). Expectation stated before looking: the 2D skeleton should sit on the
FEMALE inside her SAM mask in all three views with nothing on the other fly, and the MuJoCo render
should reproduce her pose. What it shows: yes for the body and legs -- head, thorax, abdomen and six
leg chains all inside the blue mask contour in the left/top/right views, the second fly (visible at
the top-left of two panels) untouched, and the rendered mesh's orientation matching the video in each
view. The weakest part is the WINGS: the render splays them wider than the video's wing outline
suggests in the two oblique views, with the magenta wing markers out at the spread tips. That is a
wing-fit question for the IK (cf. the wing DOF convention notes), not evidence about the lifter, but
it should be compared against the DLT arm before the campaign is trusted on wing kinematics.

**Wall clock** (Session0 bout 28, 2007 frames, 2 flies):

| stage | time | where |
|---|---|---|
| mvq lift (3877 windows) | 453 s (4.4 frames/s) | job 39606799, one A40 |
| same-checkpoint reference via the old path | ~11 min | job 39606799 |
| `run_bout.py` up to the viz crash (both STAC solves + polish for fly0) | 36 min | job 39606799 |
| resumed QC + viz for fly0 and all of fly1 | 20 min | local, `CUDA_VISIBLE_DEVICES=7` |

Job `39606785` was an earlier submission of the same script that died in 2 s: its driver lived in the
node-local session scratchpad under `/tmp`, which does not exist on a compute node. The script now
lives in `OutFiles/mvq_task7_smoke/` on shared storage.

### Running the campaign

    # inspect every sbatch command first
    scripts/slurm/mvq_p3a_campaign.sh --dry-run

    # pure queue mode: 11 chains, each mvq-lift array -> precompute -> IK array -> aggregate
    scripts/slurm/mvq_p3a_campaign.sh

    # lift on an interactive node (4 workers, one GPU each) and queue only the rest
    scripts/slurm/mvq_p3a_campaign.sh --local-gpus 4

`--only <timestamp>` restricts it to one recording. The lift is idempotent on the gates string, so a
re-run costs nothing for bouts already done, and `--local-gpus` refuses to start while a training
process matching `--guard-pattern` is alive.

## Mask-identity assignment (2026-09-05)

The campaign's `identity="sex"` lifter took `fly0` from the model's FEMALE typed slot and `fly1` from
its MALE one, each out of whichever window read that slot most confidently. On
`Session0/2025_10_20_13_20_04` (20_04) the sex head types that female as a MALE, so
(`.superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b/female-miss-diagnosis.md`):

- her own mask window's male slot fires 0.87-1.00 **on her body** while the female slot reads
  0.01-0.46, which NaN'd `fly0` on 40 % of the recording (97.7 % of bout 25);
- whenever that read beat the male window's, `pick_typed_pair` handed the male track **her body** --
  2.5 % of 20_04 frames, 21 % in bout 25, 19 % in bout 5, and 0-2 % in every Session1 recording. The
  collapse guard cannot see it: the two written flies are different bodies, just the wrong way round.

### Design

`lift_masked_bout(..., identity=)` now has two modes, and `--identity {mask,sex}` on
`scripts/mvq_lift_bout.py` selects them (`mask` is the default; `configs/mvq/p3a.yaml` sets
`identity: mask`).

- **`sex`** -- unchanged: `pick_typed_pair`, the model's typed slots, `sex.json` method
  `mvq_sex_head`.
- **`mask`** -- the SAM3 masks carry a **human id review** (`sex_meta.method ==
  "human_id_review"`, `male_slot: 1` on all 160 campaign bouts), which is the top of this
  pipeline's identity precedence (`sexing.canonicalize_bout`: human > mvq > wing-song CV). So
  `fly{f}` IS mask fly `f`, and the model is asked only *which instance is on this mask?*
  Per frame, per mask fly `f`, in the window `frame_windows` put that mask's centre in
  (`pick_mask_pair`): among the 4 slots with `exist >= exist_thresh`, keep those whose keypoint
  centroid is within `--mask-assign-units` (default **10 units = 1 mm**) of that mask's
  triangulated centre and take the nearest; the typed slot for that fly's sex wins when it also
  qualifies (ties by existence). Nothing inside the radius -> **NaN**, never "the nearest thing in
  the crop". A bout whose masks carry no human review falls back to `sex` **with a warning** and
  records it. `male_slot != 1` under `mask` is **refused** -- `fly{f}` IS mask fly `f` there, so
  writing `male_fly: 1` would name the wrong fly.

Collapse guard kept, and it is what protects the merged-window case (both masks resolving to the
same instance): two written flies within `COLLAPSE_DIST_UNITS` (3 u median per keypoint) keep the
one **nearer its own mask** and NaN the other (tie keeps fly0).

`sex.json` keeps `male_fly: 1` but `method`/`authority` become `mask_human_id_review`, with
`confidence: "user"` (a human decided, not the sex head). `sexing.LIFTER_SEX_METHODS` puts that
method in `canonicalize_bout`'s authoritative no-swap set beside `mvq_sex_head`, and
`bout_lift_is_current` accepts either. `mvq_meta.json` gains `identity` (requested),
`identity_resolved`, `mask_assign_units`, `sex_head_disagree_frac` per fly, and per frame
`identity_source`, `slot_used`, `sex_head_agrees`, `mask_dist_units`.

`identity` is in `mvq_gate_signature`/`mvq_gate_string` and `run_bout.stage_b_gate_signature`
passes `cfg.mvq.identity`, so a run switched between the modes refuses the other's bouts instead of
reusing them -- **every existing `pose_mvq_p3a` lift is now stale by design** (its gates string has
no `identity` key). The gate names the mode the bout **actually ran** -- `resolve_mask_identity` is
called before the gate string is built, so a bout that fell back to `sex` is stamped `sex` and a
`mask` run does not accept it. `mask_assign_units` is deliberately NOT in the signature, for the
same reason as `collapse_dist_units` -- recorded in `mvq_meta.json` instead.

### Smoke: 20_04 bouts 25 and 5 re-lifted with `--identity mask`

    OUT=OutFiles/mvq_identity_smoke
    PYTHONPATH=third_party/jarvis_jax:. python -u scripts/mvq_lift_bout.py \
        --session-dir  $VID/courtship/Session0/2025_10_20_13_20_04 \
        --predictions-dir $PROC/courtship/Session0/2025_10_20_13_20_04/sam3_masks \
        --out $OUT --bout 25 --bout 5 \
        --run $RUNS/mvq_t1_b16_p3a_20260904/final \
        --exist-thresh 0.5 --batch 8 --merge-dist-units 30.0 \
        --identity mask --mask-assign-units 10.0 \
        --anatomy configs/anatomy/v1.yaml --recording-cfg configs/recording/session0.yaml \
        --bouts-csv $PROC/.../pose_mvq_p3a/bouts_unified_summary.csv

Audit (`figures/2026-09-mvq/p3a_campaign_female_misses/identity_mask_smoke.json`; the method is
`scratchpad/identity_check.py`'s -- written fly centroid vs the two mask centres, "swapped" = > 8 u
closer to the OTHER mask):

| bout | metric | before (`identity=sex`, `pose_mvq_p3a`) | after (`identity=mask`) |
|---|---|---|---|
| 25 | fly0 (female) NaN | **0.977** (419/429) | **0.210** (90/429) |
| 25 | fly1 (male) NaN | 0.000 | 0.000 |
| 25 | fly1 swapped onto her | **0.214** | **0.000** |
| 25 | fly0 swapped | 0.000 | 0.000 |
| 25 | median dist to OWN mask, fly0 / fly1 | 4.25 / 5.37 u | 4.46 / 4.98 u |
| 25 | `sex_head_disagree_frac` fly0 / fly1 | -- | 0.982 / 0.000 |
| 5 | fly0 (female) NaN | **0.659** (520/789) | **0.070** (55/789) |
| 5 | fly1 (male) NaN | 0.010 (8/789) | 0.199 (157/789) |
| 5 | fly1 swapped onto her | **0.188** | **0.000** |
| 5 | fly0 swapped | 0.026 | 0.000 |
| 5 | median dist to OWN mask, fly0 / fly1 | 5.61 / 6.37 u | 4.54 / 6.21 u |
| 5 | `sex_head_disagree_frac` fly0 / fly1 | -- | 0.778 / 0.000 |

Both swap fractions go to **zero** and the female's NaN fraction drops 4.7x (bout 25) and 9.4x
(bout 5), with the median distance to her own mask **unchanged at ~4.5 u** -- the radius is not
letting a different instance in.

**Where the two residual numbers come from** (`scratchpad/why_missing.py`, `merged_detail.py`):

- bout 25's remaining 90 fly0 NaNs are **all** in the 98 MERGED frames (both flies in one crop). On
  90 of them exactly ONE slot is above threshold, slot 2 at exist ~0.95, sitting 6.4 u from the
  MALE's mask and 24.3 u (median) from hers -- the model simply does not emit an instance on her in
  a merged crop. On the other 8, slot 3 ("other") fires on her and she IS written from it
  (`slot_used[fly0] == 3`). So this is a model limitation, not an assignment one, and the expected
  "fly0 NaN falls to near the male's 0.000" is only partly met: it falls to 0.210, and the rest is
  contact frames.
- bout 5's fly1 NaN rise 0.010 -> 0.199 is the SWAP being converted into an honest NaN, not new
  loss. All 157 misses are merged frames; of the 149 frames that were written before and are NaN
  now, **98 were flagged swapped** by the > 8 u criterion and the remainder are the same merged
  frames with the two mask centres 24.5-30.0 u apart (right at the 30 u merge boundary), where the
  live instances all sit on her side.

**Radius A/B** (`OutFiles/mvq_identity_smoke_r15`, `--mask-assign-units 15`): bout 25 is
**identical** (90/0), confirming her merged-frame misses are not a radius artefact; bout 5 recovers
a little (fly0 55 -> 28, fly1 157 -> 127) but **81 frames (10.3 %) start hitting the collapse
guard** -- the wider radius lets both masks claim the same instance -- against **0 collapses at
10 u** in either bout. 10 units stays the default. (Calibration on the sex-head lifts, over frames
where identity was right: distance from a written fly's keypoint centroid to its own mask centre is
median 5.4 u / p95 8.0 u / p99 12.5 u for the male, so 10 u costs ~3 % of good male frames in the
worst case -- `scratchpad/radius_calib.py`.)

### Figure

`figures/2026-09-mvq/p3a_campaign_female_misses/identity_mask_bout25.png`, regenerated by

    PYTHONPATH=third_party/jarvis_jax:. python scripts/viz/mvq_identity_overlay.py \
        --session-dir $VID/courtship/Session0/2025_10_20_13_20_04 \
        --masks-dir   $PROC/courtship/Session0/2025_10_20_13_20_04/sam3_masks \
        --compare before_sex_head=$PROC/.../pose_mvq_p3a \
        --run     after_mask_identity=OutFiles/mvq_identity_smoke \
        --bout 25 --frame 149 --cameras Cam2012630,Cam2012861 \
        --out figures/2026-09-mvq/p3a_campaign_female_misses/identity_mask_bout25.png

Frame 149 was chosen (`scratchpad/pick_frame.py`) as a HARD frame, not a flattering one: one of the
89 bout-25 frames where the old run both NaN'd the female AND put the male on her body, with the two
mask centres 45.8 u apart so the bodies read unambiguously.

**Expectation stated before rendering:** cyan written keypoints inside the cyan (human-reviewed
FEMALE) mask outline and orange inside the orange (MALE) one, in both the overhead Cam2012630 and
the side Cam2012861.

**Read back (Read tool, both rows):** met. Top row (`identity=sex`): the orange-outlined fly --
left, darker, wings folded -- carries NO keypoints at all in either camera, while every orange dot
("written fly1 = male") sits on the CYAN-outlined fly, and there are no cyan dots anywhere
(fly0 NaN). That is the swap, visible. Bottom row (`identity=mask`): orange dots follow the
orange-outlined fly's head, thorax, abdomen and leg tips in both cameras, cyan dots follow the
cyan-outlined fly, and neither fly's dots appear on the other's outline.

### Tests

`third_party/jarvis_jax/tests/test_lift_masked_bout.py`: **30 passed** (19 pre-existing + 11 new),
`JAX_PLATFORMS=cpu`. The new ones cover the female-mask window whose typed slot is dead
(`slot_used[fly0] == 2`, `sex_head_agrees[fly0] == False`), byte-identical output between the modes
when the sex head is right, the no-swap-vs-swap contrast with the OLD behaviour pinned, the
10-unit-radius NaN, the merged-window collapse guard keeping the nearer mask, the no-review
fallback, the `male_slot != 1` refusal, the gate string differing between modes (and
`bout_lift_is_current` rejecting the other mode's string) against run_bout's REAL
`stage_b_gate_signature`, `canonicalize_bout`'s no-swap for `mask_human_id_review`, and the
CLI/slurm plumbing. Also green: `test_lift_mvq`, `test_sexing`, `test_run_bout_sexing`,
`test_sam3_sexing`, `test_recanonicalize_masks`, `test_slurm_courtship_array`,
`test_coarse_pass_gates`, `test_coarse_pass_timeline_mvq_refusal` (61 passed),
`test_coarse_track`/`test_coarse_centres`/`test_coarse_figure_gates` (37), repo
`tests/test_run_bout_pipeline_structure.py` (40). `test_run_bout_no_autosex` fails on an
unrelated pre-existing drift (`main_from_cfg` reads `cfg.outputs.out`, added 2026-08-27, and the
test's cfg has no `outputs`); nothing in this change touches `main_from_cfg`.

### Follow-ups

1. **Re-lift the whole campaign.** Every existing `pose_mvq_p3a` bout is stale on the new gate
   string, and the 0-2 % male swaps exist in every recording, not only 20_04.
2. The 20_04 female is still lost on contact frames (bout 25: 0.210, all merged). That needs the
   model to emit an instance on her in a merged crop -- the P3b sex-label fine-tune, or the
   prompted branch with her mask as the prompt.
3. `mask_assign_units` is CLI/meta-only, not gated. If it is ever tuned per recording, that choice
   will not invalidate a kp3d.npz -- deliberate, but worth revisiting if it stops being a constant.

### Fix round 1 (review): gate on the RESOLVED identity, not the requested one

The first cut built the gate string from the requested `identity` and only then resolved the
human-review fallback, so a bout whose masks carry no review ran the sex head but was stamped
`identity: mask`. `bout_lift_is_current` would then answer True for the `mask` gate, and the moment
someone canonicalized a review into those masks the re-lift that should now produce mask identities
would be **skipped as already current** -- a sex-head bout inside a run labelled `mask`, with
nothing downstream able to tell.

`resolve_mask_identity(identity, mask_sex_meta)` is now the one place the fallback is decided
(returning `(mode, fallback_message_or_None)`, and raising on `male_slot != 1`), and every caller
that builds a gate string resolves first:

- `lift_masked_bout` resolves **before** the gate string and the currency/skip check; the stamped
  gates, `mvq_meta.json`'s `identity_source`/`identity_resolved` and `sex.json`'s method are all the
  resolved mode.
- `scripts/mvq_lift_bout.py` reads each bout's `sex_meta` (one small npz member) **before** its
  per-bout skip check and gates on the resolved mode -- previously one gate string for the whole
  invocation.
- `slurm_bout_array`'s `--mvq-lift skip` verification does the same per bout (`_gates_for(i)`), so
  it neither calls a finished fallback lift missing nor calls a fallback lift "done" for a mask run.

Consequence, deliberate and documented in `mvq_gate_signature`: `run_bout.stage_b_gate_signature`
has no bout index and never opens the mask npz, so with `mvq.identity: mask` a fallback bout is
**refused at Stage B** rather than silently accepted. The fix for such a bout is to give its masks
the human review (then re-lift, which is no longer skipped) or to run that recording with
`mvq.identity=sex`. No campaign bout is affected -- all 160 carry `human_id_review` + `male_slot: 1`.

Verified on real data: `mvq_lift_bout.py` re-run over 20_04 bouts 5 and 25 now prints
`skip (kp3d.npz already carries these gates, identity mask)` for both, and
`run_bout.stage_b_gate_signature` with the real `configs/mvq/p3a.yaml` reproduces the smoke lift's
stamped string byte-for-byte while refusing the old sex-head campaign lift.

Tests: `test_lift_masked_bout.py` **31 passed** (+1: a fallback bout is stamped `sex`, is current
for the `sex` gate, is NOT current for the `mask` gate, and re-lifts -- not skips -- once the review
lands). Four pre-existing tests that exercise the sex-head route with no mask `sex_meta` now say
`identity="sex"` explicitly instead of relying on the default label. `test_slurm_courtship_array`
4 passed; `test_lift_mvq`/`test_sexing`/`test_run_bout_sexing`/`test_sam3_sexing`/
`test_recanonicalize_masks`/`test_coarse_pass_gates`/`test_coarse_pass_timeline_mvq_refusal`/
`test_coarse_track` 80 passed.
