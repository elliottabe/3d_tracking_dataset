# Class-balanced sampling, wired for the wall frames

2026-09-02

## The defect

`configs/sampling/balanced.yaml` (`sampling=balanced`) has existed since the V4
roots and is wired end to end: `train_keypoints.run_training` reads
`sampling.balance_key`, calls `V3Dataset.balanced_weights(key=...)`, and hands
the result to `batches(..., weights=...)`. On every `general_model`-derived
root it could not be used at all.

`build_generalmodel_split.py` deliberately leaves `behavior` as `"unknown"` on
every annotation — writing it would silently re-weight `balanced_weights` for
roots already trained against — and emits no `category` field. So
`class_counts` returned `{"unknown": 18426}` and `run_training` raised:

    sampling.balance_key='category' but every train annotation is 'unknown'
    on that axis ... balanced sampling would silently degrade to uniform.

That guard did its job; the axis it guards was simply unreachable. The 196 new
wall annotations would have trained at their raw 1.06% share.

## The fix

`data/v5_2d._resolve_behavior` reads the regime from `manifest.json`, where the
builder actually writes it (`recordings[<rec>]["behavior"]`), falling back to
the annotation's own field first. This is the same trick `_resolve_sex` already
uses one function up, for the same reason: the human labels live on the
manifest, not on the annotations.

`_resolve_category` then crosses the resolved behaviour with the resolved sex —
`courtship_female`, `wall_male` — reproducing the shape of the hand-written V4
taxonomy (`scripts/build_detector_dataset.SUBSET_CATEGORY`) without restating
it. That table is stale (it still names `headless_22_50`, renamed 2026-09-02,
and knows nothing of `courtship_20_04_male` or `wall_frames_15_06_46_male`) and
a lookup table edited by hand on every rename is the metadata-drifts-from-data
failure `sex.json` exists to prevent. The cross is also strictly finer where it
matters: V4 lumped headless males and females into one bucket.

## Measured on red_data_3d_v10_wall0902 train (18,426 annotations)

`balance_key=category`, `alpha=0.5`, `max_repeat=20`:

| class | anns | % of data | % of draws | repeat |
|---|---:|---:|---:|---:|
| courtship_male | 5041 | 27.36 | 19.62 | 0.72x |
| grooming_male | 4592 | 24.92 | 18.72 | 0.75x |
| amputation_male | 3415 | 18.53 | 16.15 | 0.87x |
| general_male | 3335 | 18.10 | 15.96 | 0.88x |
| general_female | 616 | 3.34 | 6.86 | 2.05x |
| headless_male | 462 | 2.51 | 5.94 | 2.37x |
| headless_female | 385 | 2.09 | 5.42 | 2.59x |
| **courtship_female** | 251 | 1.36 | 4.38 | **3.21x** |
| **wall_male** | 224 | 1.22 | 4.14 | **3.40x** |
| climbing_female | 105 | 0.57 | 2.83 | 4.97x |

**Why `category` and not `behavior`.** Balancing by `behavior` lifts `wall`
3.89x but leaves the 251 female-courtship annotations inside one
5,292-annotation `courtship` class at 0.80x — it does nothing at all for the
hard fly. That contrast is the whole argument for the sex cross, and it is the
right-hand panel of the figure.

**`max_repeat=20` is currently inert.** The class spread here is 48x
(5,041 / 105), not the 139x that motivated the cap, and the largest realised
repeat is 4.97x. Kept as a guard, not an active knob.

Verified through the wrapper a `mask_ablation=true` run actually uses
(`V5Dataset` -> `ZeroMaskDataset` -> `class_counts`/`balanced_weights`): counts
forward unchanged, and an actual `rng.choice` draw of 18,426 indices realises
wall_male at 4.17% against the 4.14% analytic share.

## Figure

`figures/2026-09-02-sampling-balance/sampling_balance.png` (+ the `.json` the
table above is read from). Regenerate:

    python scripts/viz/sampling_balance_check.py \
        --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v10_wall0902 \
        --out figures/2026-09-02-sampling-balance

Expectation the figure is read against: neither axis is a single "unknown"
bucket; blue (draws) exceeds grey (data) for every class under ~5% and falls
below it for the big ones. Bars tracking the grey ones one-for-one would mean
the weights are uniform and the fix did not take.

## What this does NOT do

Nothing is enabled by default. `sampling=balanced` is opt-in per run; without
it training stays uniform and unchanged. No detector has been retrained
against these weights yet — the numbers above are the sampling distribution,
not an accuracy result.
