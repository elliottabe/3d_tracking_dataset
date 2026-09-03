# 2026-09-03 -- v12_bal_maskoff: where the network looks when it misses a tarsal tip

Question: the v12 mask-off detector trained worse than before (train loss 0.0069
vs 0.0037 at 30k; honest held-out val 15.19 px). Do per-layer attention maps
show WHERE it fails, and is the new root fittable at all?

Figures: `figures/2026-09-03-maskoff-attention/` (gitignored). Regenerate on a
GPU node:

    CUDA_VISIBLE_DEVICES=<free> python scripts/viz/detector_train_fit_by_recording.py
    CUDA_VISIBLE_DEVICES=<free> python scripts/viz/detector_attention_maps.py \
        [--ckpt-v12 <run>/final --ckpt-ref <other run>/final]

Small artifacts here: `train_fit_by_recording.json` (per-recording MPJPE on the
checkpoint's OWN split, L/R tarsal-tip error), `attention_summary.json`
(per-layer attention mass and heatmap GT/max ratios).

Both arms are mask-ablation checkpoints; channel 3 is zeroed throughout.
Keypoint order verified for v12; v5vf's training root is deleted so its guard is
warn-only (evidence it matches: 5.42 vs 5.43 px on the shared control recording).

## 1. The training data IS fittable -- no label defect, no L/R defect

`train_fit_by_recording.png`: the finished v12_bal_maskoff checkpoint on 120
random annotations from each of its 12 TRAIN recordings scores 2.2-3.8 px
everywhere (p90 3.8-6.8 px), with the LEFT-minus-RIGHT tarsal-tip gap within
+-1.4 px on every recording. Nothing stands out, so the 1.85x higher training
loss is the cost of a larger, more diverse, balanced-sampled set under
augmentation, not an export the net cannot fit. The flip table
(`augment.build_lr_swap`) was also checked against the 50 names: an involution,
every L<->R pair correct.

The "unexplained left bias" from the 2026-09-03 A/B is real but LOCAL: it sits
almost entirely in val recording `2026_06_09_15_21_14` (headless male; tips L
41.2 px vs R 22.4 px). The other four val recordings are within 4 px L/R. It is
a hard-pose/floor-reflection recording (see frame 843 below), not a flip bug.

Val per recording (same run): 2026_01_29 5.4, 06_09_15_46_55 7.5,
04_02_12_11_50 10.9, 06_09_15_21_14 16.0, 04_02_15_25_51 17.9 px.

## 2. Attention maps do NOT localise the failure -- both arms look at the same places

Setup: for the worst tarsal-tip frames, the attention row of the token under the
GT tip (head-mean) at layers 1/4/8/12 plus attention rollout, for v12 and for
v5vf_maskoff on the identical crop. Expectation (A): on v12's misses the tip
token attends to the other fly / wall band and v5vf's does not.

Result: (A) is refuted. On frame `2026_04_02_15_25_51/Cam2012630/Frame_416555`
(v12 T2L_TaTip 344 px off, v5vf 3 px) the four layer maps and the rollout are
visually indistinguishable between arms, with the same mass on the target
(0.11 / 0.09 / 0.28 / 0.34 vs 0.10 / 0.10 / 0.28 / 0.35). Same on
`Frame_416617` (324 px vs 3 px) and on the single-fly floor frame
`2026_06_09_15_21_14/Cam2012861/Frame_12574` (243 px vs 4 px), where both arms
attend along the floor-reflection band at layer 1 and converge onto the GT tip
by layers 8-12. Layer 1 attends broadly to background, layer 4 to sparse
structural tokens (crop edges, leg lines), layers 8-12 to a compact patch at
the queried tip -- for BOTH models.

`attn_mass_by_layer.png` (160 frames per group): bad frames (tip err > 30 px)
do put more mass on the other fly (0.14-0.21 vs 0.07-0.12 for good frames)
and less on the target, but the v12 and v5vf curves lie on top of each other in
every group and every layer. The attention pattern is a property of the frame
(crowding), not of which model is failing.

## 3. The failure is in the heatmap readout: the GT tip barely responds

`heatmap_modes_bad_tips.png`, 698 bad tips from 320 frames, both arms on the
same crops:

| | v12_bal_maskoff | v5vf_maskoff |
|---|---:|---:|
| heatmap at GT / heatmap max, median | 0.31 | 1.00 |
| GT is a competing mode (ratio > 0.5) | 30% | 92% |
| GT is the peak (ratio > 0.9) | 5% | 84% |

So this is not mostly a peak-selection problem where the true tip nearly wins:
in 70% of v12's misses the response at the true tip is under half the max. The
heatmap instead fires on tarsus-like structure elsewhere -- the other fly's leg
tips (frames 416555/416617), the floor-reflection line (four modes along it on
frame 12574), a wing edge (frame 416617's T3L). v5vf has the same weak
secondary blobs on these frames, but its GT peak dominates -- which is recall
of recordings it trained on.

## Recommendation

* Attention visualisation is not the tool for this failure; do not spend more on
  it. The backbone routes information the same way in the model that succeeds
  and the model that fails; the difference is in the features/decoder that
  turn those tokens into a confident tip response on unseen recordings.
* The failure signature -- weak GT response, spurious tarsus-like modes on
  other-fly legs and floor reflections -- points at (i) the distractor
  gray-fill that inference applies and training never did (measured male
  16.9 -> 15.2 px under OOD fill, 2026-09-03), (ii) the mask channel, whose
  arm `v12_bal_maskon` is training now with the identical recipe, and (iii)
  multi-view consistency at triangulation rather than the 2D detector.
  Re-run `detector_attention_maps.py --ckpt-v12 <v12_bal_maskon>/final` once
  it finishes to compare the mask-on heatmaps on the same frames.
* Do not chase the L/R bias as a flip/label bug; it is one hard val recording.
