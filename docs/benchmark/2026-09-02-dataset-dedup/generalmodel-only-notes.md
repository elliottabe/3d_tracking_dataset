# general_model-only, content-keyed split — measurements

Companion to `notes.md` (the general_model + V3 content-split pass). This one
covers the ruling to **source frames from `red_data/general_model` only** and
drop `red_data_unified_V3`.

New root: `/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v7_generalmodel`
Build: `third_party/jarvis_jax/scripts/build_generalmodel_split.py`
Figure: `figures/2026-09-02-dataset-dedup/generalmodel_only_decision.png`
(regenerate with the scripts saved beside it).

Nothing was mutated: `general_model`, `red_data_unified_V3`, `red_data_3d_v5`,
`red_data_3d_v5_valfix`, `red_data_3d_v6_contentsplit` were read-only
throughout, and the new root's `images/`/`masks/` are symlinks into
`general_model`.

## The corpus, by content

All 39,536 upstream jpgs md5'd (49 s at 192 threads, no sampling; the cache is
kept at `figures/2026-09-02-dataset-dedup/hash_cache.json`).

| | general_model | red_data_unified_V3 |
|---|---|---|
| image references | 19,929 | 19,593 |
| unique contents | 19,334 | 14,973 |
| annotations | 19,382 | 20,751 |

Contents: general_model-only **5,152**, shared **14,182**, V3-only **791**.

## Precondition 1 — two-fly coverage. **FAILS.**

`general_model` files a two-fly capture as two subsets under two recording
timestamps, one per animal. Merging by content recovers those:

| merged pair | contents with 2 annotations | bbox-centroid separation (px) |
|---|---|---|
| `courtship_28_34_female + _male` | 250 | median 315, min 8, max 492 |
| `courtship_25_51_female + _male` | 146 | median 242, min 9, max 483 |
| `courtship_11_50_female + _male` | 103 | median 393, min 294, max 610 |
| **genuinely two-fly** | **499** | |
| `courtship_28_34` same animal twice | 14 | 5.7 – 9.1 px — a labelling defect |
| `headless_56_42 + _56_42_1` | 7 | 2.3 px — one fly, two ingests |

The separation column is the check, not decoration: two labels on ONE animal
sit under 10 px, two animals sit at 200–700 px. The distribution is bimodal
with nothing in between.

The V3-based root carries **1,673** two-annotation images, and every one of
them comes from a V3-only recording (`2026_04_08_14_59_45` 1,374,
`2026_04_07_11_33_33` 181, `2026_06_11_13_58_43/_45` 118). Of those:

| | images |
|---|---|
| general_model also labels two-fly (the `courtship_28_34` alias) | 118 |
| general_model has the pixels but labels ONE fly (`courtship_V2`) | 1,096 |
| general_model does not have the pixels at all | 459 |

So **499 vs 1,673**. The two sets are not nested — 395 of the 499 are captures
the V3-based root files as two separate single-fly recordings — but the
courtship_V2 capture's second animal is unrecoverable from general_model.

## Precondition 2 — what the 791 V3-only contents are

| V3 recording | V3-only contents | annotations | of which two-fly |
|---|---|---|---|
| `2026_04_08_14_59_45` | 728 | 1,102 | 404 |
| `2026_04_07_11_33_33` | 63 | 113 | 50 |

Every other V3 recording is 100% covered by general_model: byte-identical
pixels AND labels that match to under 5 px.

At annotation level (matched by median visible-keypoint displacement, < 50 px;
the distribution is bimodal — 95% of best matches under 5 px, 0.08% in the
5–186 px gap):

| | annotations |
|---|---|
| V3 annotations matched by a general_model annotation | 18,430 |
| **V3 annotations with no general_model counterpart** | **2,321** |
| … of which sit on a two-fly image (the second animal) | 2,014 |
| … of which sit on pixels general_model does not have | 1,225 |

All 2,321 come from `2026_04_07_11_33_33` (244) and `2026_04_08_14_59_45`
(2,077). All carry `sex: "unknown"` and `behavior: "unknown"` in V3's raw
annotation files, and they are spread evenly over all seven cameras
(299–355 each) — no camera- or view-specific coverage is lost.

**Is any of it the only coverage of something?** Not of a behaviour
(courtship is the corpus's largest behaviour and `courtship_V2` — the same
footage — stays), not of a calibration group (both recordings are group B, and
groups A/B/C all survive with val coverage), not of a sex *category*. What it
IS the only source of is **mass** on the two hardest axes:

| | v5_valfix | v6_contentsplit | v7_generalmodel |
|---|---|---|---|
| two-fly images, TRAIN | 1,487 | 1,453 | **221** (−85%) |
| two-fly images, VAL | 181 | 215 | **274** (+27%) |
| FEMALE framesets, TRAIN | 518 | 446 | **183** (−59%) |
| FEMALE framesets, VAL | 48 | 96 | 62 |

The female training loss is exactly `2026_04_08_14_59_45` (216 framesets) +
`2026_04_07_11_33_33` (28) = 244; the `2026_06_11` pair's 19 are recovered
through `courtship_28_34_female`. The measurement side is unharmed — val
two-fly coverage actually improves — so what the ruling costs is TRAINING
exposure to overlapping flies and to the female, which CLAUDE.md names as this
pipeline's failure mode.

## Precondition 3 — sex. **PASSES, completely.**

V3's annotations carry a non-`"unknown"` `sex` for exactly 8 recordings. The
general_model subset names encode the same 8 and agree on **8/8**:

| recording | V3 annotation sex | general_model subset | subset-derived |
|---|---|---|---|
| `2026_01_13_18_47_45` | male | `S6male` | male |
| `2026_01_29_14_09_33` | female | `female` | female |
| `2026_05_27_11_56_05` | female | `courtship_11_50_female` | female |
| `2026_05_27_11_57_05` | male | `courtship_11_50_male` | male |
| `2026_06_15_12_12_33` | male | `courtship_28_34_male` | male |
| `2026_06_15_12_12_34` | female | `courtship_28_34_female` | female |
| `2026_06_18_19_23_03` | male | `courtship_25_51_male` | male |
| `2026_06_19_11_09_36` | female | `courtship_25_51_female` | female |

Every other recording's sex in the v5 manifest — `courtship_V2/V3/V4`,
`grooming`, all five `headless_*`, `wall_frames`, both amputation subsets,
`20_04_female_climbing` — is `"unknown"` in V3's annotations too. Those labels
are **human**, applied through `build_v5.apply_sex_labels`, and were never
V3-derived. Dropping V3 therefore loses **zero** sex information.

`courtship_V2/V3/V4` and `grooming` are the four not sex-encoded by their
subset name (all male, 1,330 framesets). They are carried forward from
`red_data_3d_v5_valfix/manifest.json` and each recording records its provenance
in `manifest.json["recordings"][rec]["sex_source"]`
(`subset_name` | `carried_from_manifest`). Metadata is carried, not frames.

Trap worth naming: **"female" contains "male"**, so the parser must test
female first. Pinned by `test_subset_sex_female_is_not_matched_as_male`.

## Precondition 4 — `sam3_masks`. Dead weight for the current recipe.

`red_data_unified_V3/sam3_masks` (19,594 npz, 17 recordings) is the only mask
store the v5 roots use, keyed by V3's annotation ids via `src_ann_id`.
`general_model` holds masks for two subsets only (`S8_male_R_amp` 1,218,
`headless_22_50` 126), which the v5 build never picked up; the new root links
them (1,344 files, 6.7% of images — measured non-zero on 24/300 random train
samples, 1/300 val).

Nothing in the promoted path needs the rest. `configs/detector/vitpose_v3.yaml`
points at `v5vf_maskoff`, trained with `train.mask_ablation=true`, and sets
`zero_mask_channel: true` — the checkpoint's 4th input channel is provably all
zeros, so mask content cannot affect it. Masks are still used at inference for
the crop centre and the distractor gray-fill, but those come from SAM3 run on
the pipeline's own video, never from this training root. The one real loss is
optionality: the mask-on arm cannot be re-run on this root.

## The new root

| | v5_valfix | v6_contentsplit | **v7_generalmodel** |
|---|---|---|---|
| train / val framesets | 3,437 / 277 | 3,293 / 357 | 2,519 / 240 |
| guard-band framesets | 3 | 67 | 24 |
| train images / unique contents | 22,449 / 18,557 | 21,455 / 17,479 | **17,374 / 17,374** |
| val images / unique contents | 1,729 / 1,687 | 2,275 / 1,799 | 1,393 / 1,393 |
| val annotations | 1,871 | 2,467 | 1,666 |
| **val contents also in train** | **511 (30.3%)** | 0 | **0** |
| val male / female framesets | 229 / 48 | 261 / 96 | 178 / 62 |
| val two-fly images | 181 | 215 | 274 |
| val behaviour | courtship 206, unknown 61, general 10 | courtship 272, unknown 65, general 10, headless 8, climbing 2 | courtship 220, general 10, headless 8, climbing 2 |
| val calib groups | A/B/C | A/B/C | A 71 / B 139 / C 30 |

v7's train side has **zero internal duplication** (17,374 paths, 17,374
contents) where v6 carries 3,976 redundant copies — the content merge removes
the oversampling that the two-corpus roots inherit.

Split policy is v6's, unchanged, so the roots are comparable: hold out
`2026_03_18_15_31_22` whole; force the two smallest female-inclusive
components (`2026_06_18_19_23_03` = the 25_51 pair, `2026_05_27_11_56_05` = the
11_50 pair) to val so there is an unseen-session female number and any group-C
val at all; every other female-inclusive component keeps its guarded 10% tail.

### Zero-overlap assertion, re-derived from the shipped jsons

Independent pass: read only `instances_{train,val}.json`, resolve each image
through the root's own `images/` symlinks, md5 the real file. v5_valfix is kept
as the positive control — a verifier that reports zero on everything is not
verifying anything.

```
red_data_3d_v7_generalmodel      train  17374 image paths /  17374 unique contents
red_data_3d_v7_generalmodel      val     1393 image paths /   1393 unique contents
red_data_3d_v7_generalmodel      ASSERT train_content & val_content == set()  ->  overlap = 0  (0.0% of val)
red_data_3d_v7_generalmodel      assert passed
red_data_3d_v6_contentsplit      ASSERT ...                                   ->  overlap = 0  (0.0% of val)
red_data_3d_v6_contentsplit      assert passed
red_data_3d_v5_valfix            ASSERT ...                                   ->  overlap = 511  (30.3% of val)
red_data_3d_v5_valfix            ASSERT FAILED (expected for v5_valfix)
```

The builder's own `write_derived` guard also fired clean:
`cross_fly_leaked_frames 0, cross_capture_leaks 0, content_overlap 0`.

### Identity, and the 21 slots recorded ABSENT

Fly identity comes from the SUBSET, not from position within a camera's
annotation list, so the chimera failure CONTROLLER RULING R15 exists to prevent
cannot occur here. R15's spirit still applies once: on 21 (camera, content)
slots the two subsets' boxes land on the SAME animal (median 2.3–9.1 px), so
neither can be attributed and BOTH are recorded absent — 14 on
`courtship_28_34` frames 621799 and 621803, 7 on the single
`headless_56_42`/`_56_42_1` shared capture. That costs 42 annotation boxes
(0.2%) and is listed in `build_report.json → ambiguous_examples`.

## Missing camera annotations (the side question)

Definition: a frameset whose `ann_ids` do not resolve on all 7 cameras.

**Per-sex RATE, on `red_data_3d_v5_valfix` (3,717 framesets):**

| sex | framesets | incomplete | rate |
|---|---|---|---|
| female | 569 | 167 | **29.3%** |
| male | 3,148 | 284 | **9.0%** |
| all | 3,717 | 451 | 12.1% |

The raw counts say male (284) > female (167). The **rate says the opposite,
3.3×**. Sex is resolved per fly (`fly_sex` first, then recording `sex`), which
matters: the two-fly recordings' two flies get different labels.

At source level, over `general_model`'s own 2,847 per-subset framesets, the gap
is much smaller — female 33/199 = 16.6%, male 148/1,112 = 13.3% (1.25×). The
built root's larger gap comes from the V3 two-fly recordings, where R15 marks a
camera absent for BOTH flies whenever the per-camera count disagrees, and those
recordings are 40–75% incomplete.

**Clustering by camera** — yes, strongly, but not on the cameras suspected:

| camera | missing views (v5_valfix) |
|---|---|
| Cam2012630 | 276 (46%) |
| Cam2012631 | 108 |
| Cam2012853 | 106 |
| Cam2012862 | 64 |
| Cam2012855 | 23 |
| Cam2012857 | 16 |
| Cam2012861 | **7** |

`Cam2012630` alone holds 46% of all 600 missing views; `Cam2012861` is the
*best* camera, not a problem one.

**Clustering by recording** — also yes, and more strongly than by sex:

| recording | incomplete | rate | sex |
|---|---|---|---|
| `2026_07_30_13_28_99` (wall_frames) | 7/7 | 100% | male |
| `2026_06_11_13_58_43` | 24/32 | 75% | two-fly |
| `2026_04_07_11_33_33` | 44/61 | 72% | two-fly |
| `2026_06_15_12_12_34` (courtship_28_34_female) | 24/43 | 56% | female |
| `2026_04_08_14_59_45` | 184/458 | 40% | two-fly |
| `2026_06_18_19_23_03` / `2026_06_19_11_09_36` | 4/22, 4/23 | 18%, 17% | male, female |
| `2026_03_22_12_07_40`, `2026_01_13_18_47_45` | 57/371, 72/490 | 15% | male |

So the driver is **two-fly courtship footage plus `courtship_28_34_female`**,
which is where the female framesets live — the sex effect is largely a
recording effect.

**Single-camera framesets.** None survive into a built root: `MIN_CAMS = 3`
drops them (v5_valfix's minimum resolved-camera count is 3). At source level,
over the 3,536 distinct (recording, frame) pairs in `general_model` + V3, 51
have exactly one annotated camera and 17 have none — close to the 54/3,519
reported from a slightly different enumeration. Those are dropped before
triangulation ever sees them, so the triangulation risk is contained; the cost
is silently discarded data.

Reported, not fixed, per the brief.

## Concerns

* **Precondition 1 fails.** 499 two-fly images vs 1,673, and −85% two-fly /
  −59% female on the TRAINING side. The val side is fine or better. This root
  is built and verified but **no config points at it and no training run has
  used it** — the trade needs a human ruling first.
* The 14 `courtship_28_34` contents where both subsets labelled the same animal
  are a genuine labelling defect nobody has adjudicated.
* `manifest["recordings"]["2026_06_09_15_38_35"]["n_flies"] = 2` is the union
  of slots seen across that component's captures, not a claim that two animals
  are annotated — no frameset there carries a `fly1`.
* Annotation-level `behavior` is left `"unknown"` (as in every prior root) so
  `V5Dataset.balanced_weights` is unchanged; the real behaviour is on the
  manifest entry only.
* Mask coverage drops to 6.7% of images. Harmless for `v5vf_maskoff`, fatal
  for any mask-on arm.
