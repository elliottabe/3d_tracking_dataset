# Content-keyed, leak-free split for the ViTPose 2D data (2026-09-02)

`red_data_3d_v5_valfix` splits by **recording name**. A recording name does not
identify footage in this dataset: the same capture was ingested more than once
under different session names, so the name-keyed split could not see that it
was putting the same pixels on both sides.

Measured over all 39,536 upstream jpgs (md5, complete, **no sampling**;
488 files/s at 192 threads on gpfs, 81 s):

| | |
|---|---|
| upstream images | 39,536 |
| unique contents | 20,125 |
| redundant copies | 19,411 |
| images reachable from `red_data_3d_v5_valfix` | 24,752 -> 20,125 unique |
| **val images byte-identical to a train image** | **511 / 1,687 unique val contents (30.3%)** |
| val annotations sitting on one | 620 / 1,871 (33.1%) |

## The six recording-level aliases

Byte identity only — frame-number proximity was tested and **rejected** as
evidence (counters free-run per session; 2026_01_29_14_09_33 female and
2026_06_01_15_34_04 grooming male reach min |dframe| = 4 while sharing zero
bytes).

| component | mechanism | shared frames |
|---|---|---|
| `2026_01_13_18_47_45` == `2026_03_22_12_07_40` | V3 re-ingest of `S6male` | 2,709 (100% of the second) |
| `2026_03_09_14_39_40` == `2026_04_07_11_33_33` == `2026_04_08_14_59_45` | `courtship_V2` re-labelled two-fly in V3 | 1,183 / 154 / 1,029 |
| `2026_06_11_13_58_43` == `_45` == `2026_06_15_12_12_33` == `_34` | `courtship_28_34_male` / `_female` + V3 two-fly | 322 (100%) |
| `2026_05_27_11_56_05` == `2026_05_27_11_57_05` | `courtship_11_50_female` / `_male` | 105 (100%) |
| `2026_06_18_19_23_03` == `2026_06_19_11_09_36` | `courtship_25_51_male` / `_female` | 161 (100%) |
| `2026_06_09_15_38_35` == `2026_06_10_15_05_02` | `headless_56_42` / `_1` chunks | 7 |

Three of the four `VAL_RECORDINGS` sit in such a component with their twin in
train. Only `2026_03_18_15_31_22` was clean.

The dominant mechanism is **one capture, one recording name per fly**: the
`courtship_*_male` / `courtship_*_female` subsets are the same frames labelled
once per animal and filed under names one second apart. `split_v5` already
enforced "both flies of a frame go to the same side" — but only *within* a
recording, so the two names defeated it.

## The unit

Cross-recording duplicates carry one frame number and one camera, and all 7
cameras of a duplicated frame duplicate together (1,225 / 1,225 groups, zero
partial). The split atom is therefore

    capture = (alias component, frame number)

which subsumes content identity, the 7-camera frameset, and the two-flies rule.

## New root

`/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v6_contentsplit`
(new artifact; `red_data_3d_v5_valfix`, `red_data_3d_v5`, `red_data_unified_V3`
and `general_model/` were not modified). `images/`, `masks/` and
`calibrations/` are symlinks back at the v5 tree.

    python third_party/jarvis_jax/scripts/build_content_split.py \
        --src  .../red_data_3d_v5_valfix \
        --out  .../red_data_3d_v6_contentsplit --hash-cache <json>

Policy: held out WHOLE (unseen fly + session) = `2026_03_18_15_31_22` (group B,
courtship male), `2026_06_18_19_23_03`+`2026_06_19_11_09_36` (group A,
courtship M+F), `2026_05_27_11_56_05`+`2026_05_27_11_57_05` (group C,
courtship M+F). The last two are female-inclusive and are spent deliberately
via `val_components_force`: they are the two smallest such components (38 of
569 female framesets) and without them there is no unseen-session female
number and no group-C val at all. Every other female-inclusive component keeps
the guarded-tail policy. Guard raised 50 -> 200 (see below).

## Composition, old vs new

| | old (v5_valfix) | new (v6_contentsplit) |
|---|---|---|
| train framesets | 3,437 | 3,335 |
| val framesets | 277 | 357 |
| val unique image contents | 1,687 | 1,799 |
| **val contents also in train** | **511 (30.3%)** | **0** |
| honest (unleaked) val contents | 1,176 | 1,799 (+53%) |
| val male / female framesets | 229 / 48 | 261 / 96 |
| val two-fly frames | 30 | 32 |
| val behaviour | courtship 267, general F 10 | courtship 337, general F 10, headless 8, climbing F 2 |
| train female framesets | 518 | 463 (-10.6%) |
| dropped to guard band | 3 | 67 |

Val val splits into two tiers that measure different things and should be
reported separately: **unseen session** 212 framesets (174 M / 38 F, 1,225
images) and **unseen pose, same session** 145 framesets (87 M / 58 F, 574
images).

## Verification

Independent re-hash from disk of only the shipped `instances_train.json` /
`instances_val.json`:

    NEW  train 21,455 paths / 17,479 contents; val 2,275 paths / 1,799 contents
         ASSERT train_content & val_content == set()  ->  overlap = 0   PASSED
    OLD  train 22,449 paths / 18,557 contents; val 1,729 paths / 1,687 contents
         overlap = 511  (30.3% of val)

Residual exact duplicates (harmless — both copies are on the same side by
construction, they only oversample): 3,990 within train, 476 within val.

Near-duplicates, measured as RMS grey difference of a 64x16 thumbnail (0-255).
Calibration: frames **1 apart** in the same recording score 0.39; random pairs
96.6. An 8x8 dHash was tried first and **rejected** — it is degenerate on these
wide arena strips (6,618 distinct values for 20,125 images) and reported 12,325
phantom pairs.

| | old | new |
|---|---|---|
| min val->train RMS | 0.00 | 1.58 |
| val images < 2.0 RMS | 558 | 3 |

Guard 50 -> 200 was chosen from this: at 50 the floor was 1.39 with 14 val
images under 2.0; 200 lifts it to 1.58 with 3, costs 42 more train framesets,
and 500 buys nothing further.

## Which corpus is canonical

**Neither, as a whole — and it does not matter for the split, because the key
is the content hash, which is corpus-independent.** Evidence: `general_model`
holds 5,152 unique contents V3 lacks; `red_data_unified_V3` holds 791
`general_model` lacks. So "canonical" cannot mean "drop the other".

* **V3 is canonical for annotations**: it is the only source carrying `sex`,
  it is the id space `sam3_masks` is keyed by, and it is the only source of the
  two-fly labels (`2026_04_07`, `2026_04_08`, the `2026_06_11` pair exist
  nowhere else). This is what `build_v5.discover_sources` already does.
* **`general_model` is canonical for image storage**: more pixels, and it is
  where v5 already links from.

## Both per-fly annotations are KEPT

On an aliased capture the two ingests carry two different things:

* the **same fly labelled twice** — median 0.00 px apart, mean 0.70, p95 5.00,
  p99 13.04, max 56.08 px over 193,779 compared keypoints.
* a **second fly present on only one side** — 2,525 annotations. Real data.

Both are kept. Dropping the redundant same-fly copy would prejudge which of two
disagreeing labels is right; that is a human call and is what the adjudication
CSVs below exist for. Redundancy oversamples those captures, it does not leak.

## Label disagreement (adjudication inputs)

Matching is by **median keypoint displacement**, not bbox IoU: three alias
groups are the same courtship footage labelled once per fly, and a courting
pair overlaps at IoU 0.35-0.64, so an IoU matcher pairs the two *different*
flies and reports a 325 px "disagreement". The displacement distribution is
bimodal with nothing in between — best match <1 px for 77%, <5 px for 84%,
jumping to 186 px at p85, and the second-best match is >=212 px at p5.

3,876 same-fly pairs; **1,110 disagree** on >=1 keypoint (>0.5 px).
14 pairs were rejected for carrying **conflicting `sex` labels despite being
co-located to <20 px median** — one of those two labels is wrong; not resolved
here.

Dominant keypoints (share of the 1,110 disagreeing pairs):

| keypoint | % of pairs | median px | max px |
|---|---|---|---|
| T1L_TaT3 | 93.2 | 5.38 | 10.82 |
| T3L_TaT3 | 91.2 | 7.07 | 17.20 |
| T3R_TaT3 | 90.5 | 6.40 | 14.77 |
| T2L_TaT3 | 88.8 | 5.66 | 21.95 |
| T1R_TaT3 | 87.7 | 5.38 | 13.45 |
| T2R_TaT3 | 85.9 | 5.00 | 21.00 |
| Scutellum | 74.0 | 8.25 | 26.48 |
| WingL_base | 52.3 | 6.71 | 24.17 |
| WingR_base | 51.6 | 8.54 | 26.57 |

i.e. the distal tarsal chain, the midline landmark, and the wing bases — the
anatomy the detector is worst at. Median 0.00 with this tail says these are
copied labels with a subset re-fit, not an independent human pass.

**This bounds the detector A/B.** mask-on vs mask-off was 5.291 vs 5.483 px
MPJPE, a 0.19 px effect. The dataset's own labels disagree with themselves by
a mean of 0.70 px and p95 of 5.00 px on identical pixels. The effect is well
inside the label-noise floor, and it was measured on a split with 30.3% of val
leaked, shared by both arms.

## Files

* `same_fly_label_disagreements.csv` — one row per disagreeing same-fly pair,
  worst-first, with both resolved source paths, both corpora and splits, the
  top 5 disagreeing keypoints, and empty `verdict` / `notes` columns.
  The `figure` column names the PNG for the 48 pairs that have one.
* `same_fly_label_disagreements_keypoints.csv` — one row per differing keypoint
  instance (23,138). Join to the pair CSV on `(content_md5, ann_id_a, ann_id_b)`.
* `disagreement_by_keypoint.csv` — the 50-row roll-up (the table above).
* Figures: `figures/2026-09-02-dataset-dedup/label_adjudication/<recording_a>_<camera>_Frame_<n>__<md5[:8]>.png`,
  matching the `figure` column. 48 of them: every alias pair gets its worst 6
  first (so the female `headless_56_42` pair is covered), the rest worst-first.
* `figures/2026-09-02-dataset-dedup/split_leak_check.png` + `.json` — the
  acceptance figure.

Regenerate:

    python scripts/analysis/label_disagreement.py --hash-cache <json>
    python scripts/viz/split_leak_check.py --root <v6> --baseline <v5_valfix> \
        --hash-cache <json> --thumbs <npy> --thumb-keys <pkl> \
        --out figures/2026-09-02-dataset-dedup

## What must be re-measured

Every val number on `red_data_3d_v5` or `red_data_3d_v5_valfix` is optimistic:
the mask-on/mask-off detector A/B (5.291 vs 5.483, 8.2% female), the v4 gain,
and any MPJPE quoted against these roots. Both A/B arms shared the contaminated
split so the leak is largely common-mode and the direction may hold, but the
magnitudes do not — and the effect size is below the label-noise floor either
way. Re-run on `red_data_3d_v6_contentsplit`, reporting the unseen-session and
same-session val tiers separately.
