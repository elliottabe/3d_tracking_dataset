# 2026-09-02 courtship re-export: conversion, and the V2/V3/V4 leak it retires

Raw source: `/gscratch/portia/eabe/data/Johnson_lab/red_data/coutship_label_2026_09_02`
(9 recordings; the directory name is misspelled upstream and was left alone).
Converter: `scripts/data_prep/red3d2jarvis.py` (corrected copy of the user's
script, which was not modified).
New root: `/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v9_courtship0902`
Source root: `.../red_data/general_model_2026_09_02` (symlink farm; general_model untouched)
Figure: `figures/2026-09-02-courtship-labels/conversion_check.png`
(regenerate: `python scripts/viz/courtship_label_conversion_check.py`)

## The headline: V2, V3 and V4 were never three recordings

They are three arbitrary, non-overlapping frame ranges of ONE capture,
`2025_10_20_13_20_04` (Session0), filed under three fabricated recording ids.

| | frames | annotated images |
|---|---:|---:|
| courtship_V2 (2026_03_09_14_39_40) | 463 | 3,224 |
| courtship_V3 (2026_03_18_15_31_22) | 137 | 958 |
| courtship_V4 (2026_04_01_16_23_08) | 77 | 539 |
| **union** | **677** | **4,721** |
| new `2025_10_20_13_20_04_male` | **677** | **4,739** |

The image sets are EQUAL: `new \ union = 0` and `union \ new = 0` on 4,739
references. All three carry byte-identical calibration.

**Consequence.** The v8 split held V3 out "whole" as an unseen session while
training on V2 and V4 — the same fly, the same session, minutes away. That was
**958 of 1,778 val images, 54% of the val set**. Any courtship-male val number
from `red_data_3d_v8_gm_only` is optimistic by an unknown amount and must not be
compared against a number from this root.

The honest val set did not shrink: v8 val 1,778 − 958 leaky = 820; v9 val = 819.

## Is the new data actually complete? Measured, not assumed

| recording | framesets | cams/frameset | images | annotated | unannotated |
|---|---:|---:|---:|---:|---:|
| **2025_10_20_13_20_04_male** (used) | 677 | 7–7 | 4,739 | **4,739** | **0** |
| old courtship_V2 | 463 | 7–7 | 3,241 | 3,224 | 17 |
| old courtship_V3 | 137 | 7–7 | 959 | 958 | 1 |
| old courtship_V4 | 77 | 7–7 | 539 | 539 | 0 |
| old wall_frames | 11 | 7–7 | 77 | **33** | **44** |
| 2025_10_12_10_56_07_male_climbing (wall replacement) | 31 | 7–7 | 217 | 196 | **21** |

The premise holds **exactly** for V2/V3/V4: the new export annotates all 7
cameras of all 677 framesets and fills precisely the 18 images the old subsets
left blank (framesets 94198, 149426, 247000, 372364 — each 49/50 keypoints).

It does **not** hold for `wall_frames` — see BLOCKED below.

## Conversion recipe, and how it was proven

`x = floor(u_raw)`, `y = floor(image_height − v_raw)`, `v=1`; `|coord| ≥ 1e6`
(the CSVs' `1e+07`) → `(0,0,0)`; bbox = min/max of the UNROUNDED flipped
coordinates over visible keypoints, **no margin**.

Raw CSV `v` is bottom-origin; the shipped jpgs are the native top-left-origin
video frames (MAE 0.36–0.42 grey levels vs a decoded frame; 35–104 flipped).

Three independent proofs, because a keypoint-order error here is silent:

1. **Exact reproduction.** The rule reproduces the existing `general_model`
   annotations for all 7 recordings both sources hold — **59,331 labelled
   keypoints, 100.000% exact on x, y and visibility**, bbox to 4 decimals.
   On the converted subset: **0/4,721 keypoint arrays differ, 0 bbox differ**.
2. **Cross-modal.** Index *k* of `keypoints3d.csv` projects onto index *k* of
   every camera's 2D CSV at a **median 0.001–0.003 px** through the recording's
   own raw DLT.
3. **Rigid invariants** (677 frames): left/right homologous leg segments agree
   to **0.03–2.0%** across all 15 pairs; wing veins near-constant (CV 3.3–5.3%);
   leg chains monotonic proximal→distal with T1 < T2 < T3; Antenna_Base→Abd_tip
   = 2.32 mm.

Order is fly50 (`data/fly50.json`), byte-identical to the v8 root's
`keypoint_names.json`. Keypoints and cameras resolved BY NAME.

## `--scale_10x`

Multiplies `projectionMatrix[0:2, 0:3]` by 0.1 — i.e. re-expresses the DLT for
world coordinates 10× larger than the raw `_dlt.csv` was fitted in. Row 2 of
these DLTs is `[0,0,0,1]` (affine), so this IS a consistent change of world
units, not the broken half-rescale it would be for a projective DLT.
**Used TRUE**: the emitted yaml is then byte-identical, all 7 cameras, to the
calibration already shipped in `general_model` for the same recording.

## New root composition

19 subsets, 16 recordings, 19,929 image refs → 19,334 unique contents
(both unchanged from v8 — the pixel corpus is identical; only annotations moved).

| | train | val |
|---|---:|---:|
| images | 18,123 | 819 |
| annotations | 18,230 | 1,066 |
| framesets | 2,630 | 154 |
| two-fly images | 246 | 249 |

val = 4.32% of images, 5.52% of annotations. Annotations 19,340 → **19,358 (+18)**.

Split audit: `cross_recording_leaks 0`, `cross_fly_leaked_frames 0`,
`cross_capture_leaks 0`, `content_overlap 0` — re-verified independently by
md5-ing all 18,942 shipped image files: **train ∩ val = 0**.

Policy changes: V3's holdout removed (it was the leak); `2025_10_20_13_20_04`
forced to TRAIN (one session of one animal — 24% of the corpus, and courtship-male
val is already covered by the 11_50 and 25_51 components from genuinely unseen
sessions).

## Female share

**9.62% → 9.47%.** The decline is **entirely the wall_frames sex correction**
(male, not female — 28 annotations leave the female budget), not the new data:
holding sex fixed the share would be 9.61%. Female annotations 1,827
(train 1,357 / val 470; val = 25.7% of all female).

`wall_frames` no longer contributes to the female budget at all, and the
previous justification for keeping it in train — "100% of the corpus's
wall-adjacent FEMALE supervision" — is void. It survives on merits unchanged by
sex (~28 annotations cannot separate a model change; it is the only wall
footage, so a val-only placement would leave the model never trained on
wall-adjacent poses).

## BLOCKED: wall_frames cannot be replaced

The user maps `general_model/wall_frames` ← `2025_10_12_10_56_07_male_climbing`.
That recording **cannot be converted**:

* **No video exists** for `2025_10_12_10_56_07` anywhere on this machine
  (searched all of `Video_recordings` and `/gscratch/portia/eabe/data`), and the
  raw export ships CSVs only. There are no pixels to annotate.
* The requested **content check is impossible**: the new export's 31 labelled
  frames `{0, 4012, 4121, 6321, 6403, 487938, …}` are **disjoint** from
  wall_frames' 11 frames `{4031, 4056, 6708, 6822, 6884, 6933, 8800, 8839, 8863,
  8930, 8943}`. The two labellings cover *different frames of the same capture*,
  so there are no identical pixels to compare on. The frame numbers interleave
  (4012 vs 4031, 6403 vs 6708) and the calibration is byte-identical, which is
  consistent with the user's mapping but is not the byte-level proof requested.
* Even as labels it is **not strictly better**: 196/217 annotated (21 gaps),
  mean 40.4/50 keypoints, min 0, only 60 images fully labelled.

`wall_frames` was therefore **kept**. Deleting it would lose 33 annotations of
the corpus's only wall footage and gain nothing. Needs a user decision.

## Not replaced, deliberately

The export's other 6 recordings duplicate subsets already in `general_model`
that the user did not ask to replace, and for `25_51` the new export is **not** a
superset — it drops frameset `431686` (7 cameras × both flies) while adding a
few others. Left alone.

## Known defect, pre-existing

`20_04_female_climbing` is the SAME capture as `courtship_20_04_male`
(2025_10_20_13_20_04) but its subset-internal recording id is the fabricated
`2026_08_26_16_05_15`, so the builder cannot see the relationship. Both are
forced to TRAIN, which is what stops that hidden relationship becoming a
cross-side leak. Do not infer subset→source mappings from these ids.
