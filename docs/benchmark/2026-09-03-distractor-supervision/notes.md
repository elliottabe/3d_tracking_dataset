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
