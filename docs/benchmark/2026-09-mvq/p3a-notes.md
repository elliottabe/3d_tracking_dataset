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

6 rows (3 contact, sep<=15u; 3 far, sep>15u; naturally included one
male-donor/male-host same-sex pair, so no forced substitution was needed) x 7
camera columns, drawn from `V12WindowDataset(root, "train", T=1, train=True,
copy_paste=CopyPasteParams(p=1.0, max_tries=30))` and `ds.paste_window(i,
rng)` over random single-fly windows (`n_flies(i)==1 and
unlabelled_sex(i)==-1`, the documented guard). `paste_window` never rejected
(0/18 draws returned `None`) -- no rejection-pattern concern to report.

**Mid-task finding (user-flagged) -- a genuinely headless donor, not a
compositing bug.** The first render (before the fix below) put a real
anomalous window in row 0: donor window `(2026_06_09_15_38_35, fly0, frame
412)` has ZERO visible Antenna/Eye keypoints in ALL 7 of its own cameras
(`has3d` false at every head landmark, independent of copy-paste) -- an
actual amputated/headless-looking specimen in the labelled data, not an
artifact of the paste. Verified directly against the dataset (not just the
figure): `ds[2608]["vis2d"][0,0][:, head_idx].sum(0)` is `[0,0,0,0,0,0,0]`
across all 7 cameras for the 3 head keypoints, vs. 7/7 for every other
window checked. Confirmed the compositor itself was NOT at fault (view_shifts
consistent, keypoints landed at the geometrically correct shifted location
in every camera that had donor mask content) -- the fly in the source data is
simply missing its head. **Fix**: `mvq_copy_paste_check.py` now requires both
host and donor to have at least one visible Antenna/Eye keypoint in some
camera (`HEAD_NAMES`/`_has_head`) before accepting a draw; 8/18 draws in the
final run were skipped for this reason (~44% -- worth noting as a real data
quality signal: a non-trivial fraction of single-fly windows in this dataset
have no visible head in any view, so a production run of the augmentation
itself has no such filter and will occasionally paste one of these). This is
a note for the P3a controller, not a blocking defect in `composite()`.

Observations against the expectation, on the head-filtered final render:

- **Contact rows (3/3, sep 9.5u/13.8u/10.6u)**: in every camera the donor
  (orange) appears directly overlapping/adjacent to the host (cyan) at a
  consistent position -- no floating or per-camera offset. Host keypoints
  under the donor are drawn as hollow cyan circles exactly where the two
  bodies overlap (e.g. row 2, Cam2012855/Cam2012857), and solid cyan
  elsewhere. All three contact rows read as a plausible mounting/contact
  pair, matching the expectation.
- **Far rows (3/3, sep 26.7u/27.6u/42.6u)**: the two bodies are clearly
  separated, not overlapping -- expected. At the largest separation
  (42.6u) the donor appears as only a partial cluster of points at the
  frame edge in 4/7 cameras and is absent (no donor content at all) in the
  other 3 (Cam2012630, Cam2012853, Cam2012855) -- **this is the correct
  behaviour of `composite()`, not floating/offset geometry**: those cameras'
  panels show no donor pixels because the SHIFTED donor mask/keypoints
  landed outside the 448px crop for that specific view, and `composite()`'s
  own visibility rule (`vis_d & inside & tv & painted`) intentionally leaves
  a camera with no evidence blank rather than fabricate a position. Docstring
  literally says "never ... missing in one view", but that reading only
  holds for pairs close enough that both bodies fit in every crop; the "far"
  bucket exists precisely to also cover far, asymmetric-per-camera framing.
- No offset/mirrored/floating donor was seen in any of the 42 camera panels
  (6 rows x 7 cams) once the headless-donor issue above was fixed.

**Verdict: the copy-paste compositor is geometrically consistent within its
own documented visibility rule.** No blocking defect found. Two follow-ups
for the controller, neither blocking this launch: (1) the headless-fly
fraction in the underlying labelled data (~44% of a small random sample hit
it) may be worth a dedicated QC pass or an upstream filter in
`mv_copy_paste`/`V12WindowDataset` itself, since the production augmentation
path has no such guard; (2) `_is_contact`'s 15-unit rule (added to
`mvq_overlay.py`'s `contact_pair` case, see below) never fires on REAL
labelled data in either split -- the closest two real labelled flies in any
window are 24.3 units apart (train split, 88 two-fly windows checked; `sep
<= 15` count = 0, `sep <= 25` count = 2) -- so `--cases contact_pair` will
print "no samples" against real val/train checkpoints until a future
diagnostic run also covers copy-pasted windows. This is consistent with,
not contradictory to, copy-paste's whole reason for existing (real contact
pairs are essentially absent from the labelled set).

## Overlay script changes (`scripts/viz/mvq_overlay.py`)

- Added `contact_pair` case: `sel = [r for r in rows if r["contact"]]`, where
  `contact` is computed per-window via a new `_is_contact(ds, i)` helper using
  `ds.fly_centroids(i)` and the SAME 15-unit threshold copy-paste calls
  "contact" (`CopyPasteParams().contact_sep[1]`, single source of truth,
  imported rather than re-hardcoded).
- Added `slot` (the oracle instance index already computed as `inst`) and
  `sex_prob` (`sigmoid(out["sex_logit"][0, inst])`) to each row dict and to
  the y-axis row label: `f"#{i} {F/M} grp{g}\nslot{inst}
  pF={sex_prob:.2f}\n{mm:.2f}mm"`.
- Default `--cases` changed to `female,two_fly,contact_pair,worst`.
- Not run against a real checkpoint in this task (no trained mvq run exists
  yet for P3a). Verified import-clean
  (`python -c "import ast; ast.parse(open('scripts/viz/mvq_overlay.py').read())"`)
  and `python scripts/viz/mvq_overlay.py --help` both pass. `_is_contact` was
  additionally smoke-tested directly against the real dataset (train+val,
  no model needed) -- see the "never fires on real data" note above; the
  function itself runs cleanly, it simply has nothing to select in either
  split of real (non-copy-pasted) data at inference time.

## Files

- `scripts/viz/mvq_sex_label_check.py` (new)
- `scripts/viz/mvq_copy_paste_check.py` (new)
- `scripts/viz/mvq_overlay.py` (modified: `contact_pair` case, `slot`/`sex_prob` row fields + label, new default `--cases`)
- `figures/2026-09-mvq/p3a_gates/{sex_label_check,copy_paste_check}.{png,json}` (gitignored, not committed; regenerate with the commands above)
