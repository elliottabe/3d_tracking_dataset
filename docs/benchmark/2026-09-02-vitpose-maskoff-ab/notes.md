# v5vf_maskoff vs v5vf_maskon — 2D detector A/B on the corrected val split

2026-09-02. Numbers in `scorecard.json`; figures in
`figures/2026-09-02-vitpose-maskoff-ab/` (untracked — regenerate with the
commands at the bottom).

Both checkpoints were trained on `red_data_3d_v5_valfix` with an otherwise
identical recipe (`target_sigma=7.0`, `sampling=balanced` on sex at
`alpha=0.7`, `aug=default`, 30k steps); the only difference is
`train.mask_ablation=true`. So the full valfix val split is clean for both,
which is why these two and not the previously-configured
`v5_s70_bal_augdef_full` (trained on `red_data_3d_v5`, where the valfix val
recording `2026_04_07_11_33_33` was TRAIN data).

## Conditions run, and why

| arm | condition evaluated | in-distribution? |
|---|---|---|
| `v5vf_maskoff` | `zeroed` (headline) + `populated` | `zeroed` is in-distribution **by construction** — the 4th channel was 0 for every training sample |
| `v5vf_maskon` | `populated` (headline) + `zeroed` | `populated` is in-distribution; its `zeroed` number carries the harness's OOD caveat |

The harness docstring's caveat — that zeroing a mask-trained checkpoint is
out-of-distribution, so a large increase is ambiguous — applies **only** to
`v5vf_maskon/zeroed` (7.453 px, +36%). It does not apply to
`v5vf_maskoff/zeroed`, which is that model's own training distribution.

## Headline

MPJPE in crop px, each arm under its in-distribution condition, 1871
annotations (304 female / 1567 male), keypoints resolved by name.

| subset | v5vf_maskoff | v5vf_maskon | delta |
|---|---|---|---|
| overall | **5.291** | 5.483 | maskoff −3.6% |
| male (n=1567) | **5.376** | 5.529 | maskoff −2.8% |
| **female (n=304)** | **4.854** | 5.250 | maskoff −8.2% |
| clean_female (n=181) | 3.748 | **3.658** | maskon −2.4% |
| mixed_female (n=123) | **6.480** | 7.592 | maskoff −17.2% |

`clean_*` = recordings whose `manifest.json` split is exactly `val`
(`2026_03_18_15_31_22`, `2026_04_07_11_33_33`, `2026_05_27_11_57_05`,
`2026_06_15_12_12_33`) — they contributed no training annotation.
`mixed_*` = the rest of the val split, drawn from recordings whose *other*
frames were trained on.

## The three findings that matter more than the headline

**1. The `9.93 / 19.08` female numbers in `predict_2d.py` are a
14-annotation artifact.** They reproduce to three decimals — but as the
MPJPE of recording `2026_05_27_11_56_05` alone: 14 of 1871 annotations
(0.7%), from a recording whose manifest split is `mixed`. The overall
numbers (5.29 / 5.49) reproduce correctly as overall numbers. The true
per-sex female gap is 4.854 vs 5.250, an 8.2% difference — not the ~2x the
docstring implies. Corrected in `predict_2d.py` in the same commit.

Note also that `2026_05_27_11_56_05` **exists in both data roots** (14 val
images, manifest sex `female`). It is not a typo for `2026_05_27_11_57_05`,
which is the **male** half of the same courtship pair (105 val images,
manifest sex `male`) — "correcting" the one-digit difference would have
pointed the female metric at a male recording. The harness default was
removed rather than renamed, and `--val-recording` now aborts if it selects
nothing.

**2. `v5vf_maskoff` is mathematically invariant to the mask channel, so
`zero_mask_channel` is a measured no-op for it.** All 196,608 elements of
`patch_embed.proj.kernel[:, :, 3, :]` are exactly `0.0`, and its predictions
are bitwise identical with the mask populated or zeroed (max abs difference
`0.0` across 1871x50x2 values). This is a reproducible property of the
recipe, not a fluke of one run: `mask_ablation_zeroed_v5s70` also has an
all-zero 4th channel, while `v5vf_maskon` and `v5_s70_bal_augdef_full` have
nonzero ones (max 5.0e-2 / 5.2e-2). `v5vf_maskon` genuinely uses the channel
— zeroing it costs +36% overall.

Keep `zero_mask_channel: true` beside the `ckpt`: it is *correct*, and it is
the invariant that protects a future mask-ablation checkpoint that might not
be perfectly invariant. But it does not currently buy any accuracy, and the
claim that omitting it "silently degrades accuracy" is false for this
checkpoint. The SAM mask is still required by the pipeline for the crop
centre and the distractor gray-fill.

**3. The female advantage is a tail effect that vanishes on never-trained
female data.** On the only clean female recording the two arms are a wash
(3.748 vs 3.658, maskon nominally ahead); the entire 8.2% aggregate comes
from mixed recordings (6.480 vs 7.592). And it is a tail, not a shift: the
female *median* slightly favours maskon (3.18 vs 3.28 px) while the p99
favours maskoff (29.2 vs 36.5 px), with `>20px` at 2.1% vs 2.5%.

A caveat this val split cannot escape: the one clean female recording is the
*easiest* recording in the split (3.7 px), and every *hard* female recording
is a mixed one. Neither arm is really tested on hard-and-clean female data.

## Where the two differ, by body part

Per-keypoint delta, female annotations (`per_keypoint_ab_female.png`) is
clearly two-signed:

* maskoff better on distal legs / tarsi: `T3R_TaT3` +2.40, `T3L_TaTip`
  +2.37, `T3R_TaTip` +2.34, `T3R_FeTi` +1.81, `T2L_TaT3` +1.62 px.
* maskon better, by a smaller and very consistent margin, on the wing and
  head landmarks: `WingR_V13` −0.23, `WingL_V12` −0.20, `WingL_V13` −0.20,
  `EyeL` −0.25, `EyeR` −0.18, `T2R_Tro` −0.34 px.

On the full split the wing picture flips (maskoff better on the left wing
veins, `WingL_V13` +0.63, `WingL_V12` +0.61). There is no reliable
wing regression from promoting maskoff, but it is not a clean win on wings
either — worth knowing for the wing-orientation work.

## What the figures show

Every PNG below was opened and read back against the expectation stated in
its generating script's docstring.

* `overlay_close.png` (three two-fly frames at 1.01–1.02 body lengths,
  female): the stated failure signature — the mask-off arm's skeleton
  drifting onto the *other* fly — **does not occur**. Both arms sit on the
  correct fly; per-panel MPJPE 4.8/5.1, 5.3/4.8, 4.1/4.3 px, trading wins.
* `overlay_occlusion.png` (largest inter-fly bbox IoU, 0.13–0.15): same
  result. Both track the target through direct head-to-head contact.
* `overlay_wall.png`: both arms degrade together at the arena wall
  (14.8/13.1 px on the worst row) and fail the *same way* — mid-leg tarsi
  flung tens of px along the dark wall band (`T2L_TaTip` 123 px maskoff /
  90 px maskon).
* `overlay_worst.png` (worst female frames for maskoff): the two arms are
  near-indistinguishable, missing the *same* keypoints by the *same*
  amounts — `T2R_TaTip` 60 vs 60 px, `T2R_TaT3` 57 vs 57 px, `T2R_TaT1` 49
  vs 49 px. A shared representational limit on a raised middle-right leg,
  not a mask-channel effect.
* `overlay_median.png`: the control is good for both (4.2/4.5, 4.2/3.9,
  4.2/3.8 px), so the hard-case comparison is meaningful. Leg chains attach
  to the thorax, wing veins run from the wing base, head lands at the eye —
  an implicit anatomy check that no keypoint-order scramble is present.
* `per_group_mpjpe.png`: maskoff's populated and zeroed bars are identical
  in every group (the invariance, visually). Against maskon on the female,
  maskoff wins only on `legs` (4.9 vs 5.4) and `tarsi` (4.8 vs 5.7);
  `wings` is a tie (4.8 vs 4.7).
* `error_tail.png`: the curves are on top of each other to p90 and separate
  only above p95.

`overlay_occlusion.png` originally ranked by GT visibility and selected
fully-visible flies — on this split 1864 of 1871 annotations carry all 50
keypoints marked visible, so the annotators labelled through occlusion and
the flag carries no occlusion signal. The selector now ranks by inter-fly
bbox IoU instead, and refuses to fall back to visibility.

## Recommendation

**Keep the shipped `v5vf_maskoff` promotion; correct its stated
justification.** It is nominally better overall (−3.6%) and in the tail
(p99 41.5 vs 50.3 px; `>20px` 2.8% vs 3.2%), better on the tarsi that feed
leg IK, and — the argument the val numbers do not capture — a detector whose
first layer discards the mask channel cannot be hurt by a SAM mask dropout
at inference, which is a recurring failure in this pipeline.

What must change is the *claim*: the config comment's "mask channel roughly
DOUBLING female error" rests on 14 annotations and should be replaced with
the measured +8.2% per-sex figure, together with the fact that the effect
does not survive on never-trained female data.

Do not present this as a female breakthrough. On the axis that matters most
— clean, never-trained female frames — the two checkpoints are equivalent,
and every hard-case figure shows them failing identically.

## Reproduce

```bash
unset LD_LIBRARY_PATH
for run in maskoff:zeroed,populated maskon:populated,zeroed; do
  name=${run%%:*}; conds=${run##*:}
  python scripts/analysis/mask_channel_eval.py \
    --ckpt-dir /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v5vf_$name/final \
    --data-root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5_valfix \
    --conditions $conds --val-recording 2026_04_07_11_33_33 --batch-size 4 \
    --out figures/2026-09-02-vitpose-maskoff-ab/v5vf_$name.npz
done

python scripts/analysis/mask_channel_report.py \
  --npz figures/2026-09-02-vitpose-maskoff-ab/v5vf_maskoff.npz --label v5vf_maskoff \
  --npz figures/2026-09-02-vitpose-maskoff-ab/v5vf_maskon.npz  --label v5vf_maskon \
  --data-root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5_valfix \
  --arm v5vf_maskoff/zeroed --arm v5vf_maskon/populated \
  --out-dir figures/2026-09-02-vitpose-maskoff-ab

python scripts/viz/detector_ckpt_overlay.py \
  --arm maskoff=figures/2026-09-02-vitpose-maskoff-ab/v5vf_maskoff.npz:zeroed \
  --arm maskon=figures/2026-09-02-vitpose-maskoff-ab/v5vf_maskon.npz:populated \
  --data-root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5_valfix \
  --case close --case occlusion --case wall --case worst --case median --n 3 \
  --out-dir figures/2026-09-02-vitpose-maskoff-ab
```

`--batch-size 4` is required: the eager (unjitted) eval forward at batch 16
asks XLA for a 34 GiB allocation and OOMs a 46 GB GPU.
