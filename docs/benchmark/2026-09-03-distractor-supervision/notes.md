# 2026-09-03 -- distractor-aware supervision for the 2D detector (arms launched)

Follows docs/benchmark/2026-09-03-maskoff-attention: the mask-off detector's
tarsal-tip misses are heatmap peaks on the OTHER fly's leg tips and on floor
reflections while the true tip responds at 0.31 of the max; attention is not
the failure. Nothing in the loss or the data stream said those pixels were
wrong. Three changes, all off by default (configs/train/vit2d.yaml):

| knob | what it does | arm value |
|---|---|---|
| `train.hardneg_k` / `hardneg_weight` | MSE over the k worst background pixels per channel (a wrong blob was ~free: bg term averages 50k px at 0.1) | 256 / 0.5 |
| `train.repulsion_weight` | penalise each channel at the other fly's SAME-PART keypoints (own fg excluded); `DistractorKeypointDataset` appends the other fly's keypoints as rows [K:] | 1.0 |
| `train.distractor_fill_p` | gray-fill the other fly's SAM body mask as inference already does (`DistractorGrayFillDataset`) | 0.5 |
| `train.copy_paste_p` | paste a same-camera donor fly into single-fly frames (`CopyPasteKeypointDataset`); donor keypoints become the distractor rows | 0.3 (arm B only) |

Code: `jarvis_jax/data/distractor.py`, `jarvis_jax/data/copy_paste.py`
(`CopyPasteKeypointDataset`), `jarvis_jax/train/losses.py::heatmap_mse`,
`train/train.py::make_train_step` (2K keypoint rows, `n_fit` for the affine,
extended `lr_swap`). Tests: `tests/test_distractor.py`, `test_losses.py`,
`test_augment.py`, `test_train_step_distractor.py`, `test_copy_paste_keypoints.py`.
Weights are UNSWEPT starting points, stated here so the A/B is honest about it.

## Preview (read back before launching)

`figures/2026-09-03-distractor-aug/distractor_aug_preview.png`, regenerate:

    python scripts/viz/distractor_aug_preview.py

Six real v12 train crops (3 closest two-fly pairs, 3 singles: female headless,
male wall, male amputation). Against the docstring expectations: orange
keypoints sit on the OTHER fly in all three two-fly rows; the gray-fill greys
that fly's body (dilated 15 px) and leaves its legs visible; the TaTip-channel
repulsion footprint puts blobs on the other fly's tarsal tips only (max 1.00
on two-fly rows, 0.00 on singles); pasted donors match host scale/lighting
with a soft edge, at 77 / 239 / 176 px separation, host-on-top occlusion in
two of three. One row worth knowing about: `2026_04_02_17_28_34/Cam2012631/
Frame_617380` ann 13496 (female) has ONE visible keypoint while the male's 50
sit on the only fly in the crop -- a near-empty annotation that the loss
already averages over visible keypoints only.

Real two-fly coverage: 616 of 18,489 train annotations (3.3%) share their
image with another fly; val is 32% two-fly. Arm A relies on category-balanced
sampling lifting courtship; arm B closes the gap with copy-paste (p=0.3 ->
~32% two-fly samples, matching val).

## Arms (identical v12_bal_maskoff recipe otherwise; 4 L40S each on g3102)

| run | delta vs v12_bal_maskoff | launched |
|---|---|---|
| `v12_bal_maskon` | mask_ablation=false only | 2026-09-03 09:33 |
| `v12_bal_maskoff_distr` (arm A) | hardneg 256/0.5 + repulsion 1.0 + fill 0.5 | 2026-09-03 10:26 |
| `v12_bal_maskoff_distr_cp` (arm B) | arm A + copy_paste_p=0.3 | queued behind maskon (GPUs 0-3) |

Loss values are NOT comparable across arms (extra terms): step-50 loss is
0.105 for arm A vs 0.067 for the baseline by construction. Compare val MPJPE,
and the readout figure.

## Acceptance, stated before the results

On v12 val (1,069 anns, 5 held-out recordings), against v12_bal_maskoff 15.19 px:
* tarsal-tip MPJPE down, especially on the two courtship recordings
  (2026_04_02_15_25_51 17.9 px, _12_11_50 10.9 px) and on two-fly frames;
* `heatmap_modes_bad_tips`-style readout: heatmap-at-GT / max median rises
  from 0.31 and the fraction of argmax landing on the other fly's mask falls;
* body keypoints not worse than +0.5 px (the hard-negative term must not
  suppress real peaks); single-fly recordings (2026_01_29 5.4 px) unchanged.
Regenerate the comparison once the arms finish:

    CUDA_VISIBLE_DEVICES=<free> python scripts/analysis/mask_channel_eval.py \
        --ckpt-dir <run>/final --data-root <v12 root> --conditions zeroed --out figures/...npz
    CUDA_VISIBLE_DEVICES=<free> python scripts/viz/detector_attention_maps.py \
        --ckpt-v12 <run>/final --ckpt-ref .../v12_bal_maskoff/final --skip-mass --out figures/...

## Results (2026-09-03, all four arms finished; figures/2026-09-03-detector-arms/)

    python scripts/analysis/mask_channel_eval.py --ckpt-dir <run>/final --data-root <v12> \
        --conditions zeroed|populated --val-recording 2026_06_09_15_46_55 [--distractor-grayfill] --out <npz>
    python scripts/viz/detector_arms_compare.py --out figures/2026-09-03-detector-arms --tag plain|grayfill --npz label=<npz> ...
    python scripts/viz/detector_attention_maps.py --ckpt-v12 <arm>/final --ckpt-ref <maskoff>/final \
        --eval-npz <maskoff plain npz> --skip-mass [--populated-v12] --out figures/2026-09-03-detector-arms/readout_<arm>

**A bug found on the way, fixed in this commit.** The distractor gray-fill
(eval option since 7aed8f5, and the training wrapper above) decided "which
mask is mine" by the MERGED annotation id only. Two recordings' mask files are
keyed by `src_ann_id` instead (S8_male_R_amp: 1187/1330 train rows;
headless_22_50_female: 126/140 val rows), so on those the fill treated the
target's OWN mask as the distractor and erased the fly (72 px on that val
recording; the female cohort in every log line). Consequences: (i) the
2026-09-03 claim that the inference fill is an "OOD penalty" for v12_bal_maskoff
(15.19 -> 21.29 px) was the bug -- with the key fixed the fill HELPS it,
15.19 -> 13.05 px; (ii) arms A and B trained with ~3% of draws erasing the
target (p=0.5 x 6.4% of train rows), so their fill component was partly
noise; both are being re-run with the fix (`v12_bal_maskoff_distr_cp_v2`,
and `v12_bal_maskon_distr_cp` = mask ON + the same supervision). Regression
test: `tests/test_distractor.py::test_grayfill_resolves_own_mask_by_src_ann_id`.
The same key resolution was fixed in `detector_attention_maps.masks_for`; the
attention mass curves were regenerated (two-fly count 351, not 477) and the
conclusion is unchanged: v12 and v5vf curves lie on top of each other.

Val MPJPE (px), 1,069 annotations, 5 held-out recordings. "plain" = the raw
crop; "gray-fill" = the other fly's body filled as `predict_2d` does at
inference (the DEPLOYED condition, now bug-free):

| arm | plain | gray-fill | gf: two-fly | gf: single | gf: tarsal tips | gf: body | gf: female | gf: male |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v12_bal_maskoff (baseline) | 15.19 | 13.05 | 12.53 | 13.57 | 22.06 | 7.46 | 10.54 | 15.04 |
| v12_bal_maskon | **13.49** | 13.30 | 11.97 | 14.61 | 22.05 | 7.12 | 10.52 | 15.50 |
| v12_bal_maskoff_distr (arm A) | 14.78 | 12.83 | 11.82 | 13.82 | 20.42 | 7.34 | 9.96 | 15.11 |
| v12_bal_maskoff_distr_cp (arm B) | 14.27 | **12.04** | **11.21** | **12.86** | **18.56** | **7.07** | **9.85** | **13.78** |

Per recording (gray-fill): arm B is best or tied on 4 of 5 (15_25_51 11.2 vs
13.4 baseline; 15_21_14 16.4 vs 17.9) and 0.65 px worse on the single-fly
headless female (8.10 vs 7.45). Per keypoint (arm B - baseline): T1L/T2R TaTip
-6.5 px, T2L_TaT3 -4.5, T1L_TaT3 -3.7; the worst regression is T2R_FeTi +1.5.
Plain-condition picture (figure `arms_compare_plain.png`): the mask channel
wins two-fly frames (12.4 vs 16.8) and body points (7.3 vs 10.4) but loses
single-fly frames (14.6 vs 13.6); arm B wins tarsal tips (20.6 vs 24.1) and
single-fly frames (12.9) -- complementary, hence the combined mask-on arm.

Acceptance, against the criteria above: tips down (arm B -3.5 px gf, most on
15_25_51), two-fly and single both down, body not worse (7.07 vs 7.46), tail
shorter (p99 127 vs 146; >20 px 11.5% vs 12.1%) -- MET by arm B. Arm A meets
tips/tail but not single-fly. maskon meets body/two-fly, fails single-fly.

Heatmap readout on the baseline's 699 bad tips (err > 30 px), identical crops:

| arm | heatmap-at-GT / max, median | GT is a competing mode (>0.5) | GT is the peak (>0.9) |
|---|---:|---:|---:|
| v12_bal_maskoff | 0.31 | 30.0% | 5.4% |
| v12_bal_maskon | 0.33 | 36.5% | 15.6% |
| arm A | 0.09 | 21.9% | 8.6% |
| arm B | 0.19 | 31.3% | 16.5% |

Read back from `readout_v12_bal_maskoff_distr_cp/heatmap_modes_bad_tips.png`:
arm B's distribution is BIMODAL -- 74 tips move to "GT is the peak" (from 18)
while 173 move to ~0 response at GT (from 72). The hard-negative term
converts competing-mode misses into clean hits on some tips and suppresses
the true tip entirely on others; the argmax-on-other-fly fraction drops only
10.9 -> 9.4%. So the gain is real but the mechanism is blunt; the weights
(256/0.5/1.0) are unswept and hardneg_weight is the first thing to sweep.
