# mvq v2: pseudo-labelled T=2 training from scratch (design)

Date: 2026-09-05. Status: draft for review. Author: Claude (with Elliott Abe).
Predecessors: `2026-09-03-mvq-dinov3-query-decoder-design.md` (architecture, P0-P2),
`2026-09-04-mvq-p3a-identity-existence-design.md` (typed slots, copy-paste, existence ignore),
`2026-09-04-mvq-maskfree-frontend-design.md` (coarse pass, gates, fine pass). Evidence:
`docs/benchmark/2026-09-mvq/{p3a-notes.md, p3b-notes.md, p4-maskfree-notes.md, gate-bouts-probe-2026-09-05.md}`.

## 1. Why

The production checkpoint (P3b, `mvq_t1_b16_p3b_contact_20260905/final`) is the fourth warm-start in a chain
(30k prompted base -> P3a typed slots -> jitter-10 -> P3b contact). It reaches 0.085 mm on the 153 human
validation framesets and, with masks, tracks courtship pairs without identity jumps. What still fails, and why:

| failure (measured 2026-09-05) | root cause |
|---|---|
| mask-free tracking unstable in contact frames (bout 117: 28 % of frames with >= 5 male keypoints jumping > 0.5 mm; 23 % with male keypoints nearer the female) while the SAME checkpoint is stable with mask-centred windows | every frame is solved independently (T=1); windows come from per-frame CenterDetect peaks; the typed read picks whichever window is most confident, which flips frame to frame |
| existence fires (0.95-1.0) on empty windows: stale reused centres (gate bout 69: 121 consecutive coarse frames), CenterDetect false peaks at the arena edge (gate bout 1) | the existence head never saw a window without a fly |
| the 20_04 female was called male until P3b; contact pairs are the weakest cohort (0.146 mm vs 0.085 overall; cross-fly fraction 5.5 % vs 3.6 %) | 2,661 labelled framesets; 202 female-host windows; real contact pairs unlabelled (the 20_04 female is present-but-unlabelled in 677 windows); contact seen only through copy-paste composites |
| male wing angle >= 30 deg on 91 % of coarse frames; female median 39 deg on bout 117 | wing landmarks or the angle built from them have a ~40 deg resting offset; wings are the weakest landmarks |
| the headline number cannot see any of the above | validation 3D = triangulated 2D annotations; per-frame metrics; no temporal or per-keypoint metric |

No new human annotation is available. The masked p3b campaign (160 courtship bouts, identity fixed by the
human-reviewed masks, cleaned by the containment filter, per-frame QC) and the single-fly recordings
(free_running Session11, Clip Session6) are the only new signal. They are consecutive-frame video, which is
exactly what T=2 needs and what the labelled set lacks (~735 consecutive labelled pairs).

## 2. Decisions

1. **Train from scratch**, not another warm start: T=2 changes the input; the recipe elements that helped
   (jitter, contact copy-paste, female-host balancing, cross-fly repulsion at weight 20, prompting off) go in
   from step 0; P3b stays production until v2 passes acceptance.
2. **Pseudo-labels from the masked campaign**, human-reviewed before training (§4), weighted below real labels.
3. **T=2 with variable spacing**: the second frame is Delta in {1, 4, 16} video frames away (1.25-20 ms), sampled
   per example, so the temporal term sees real motion including contact transitions; real single framesets
   train as T=1 (or as a pair with an unlabelled neighbour and consistency-only supervision when the source
   video is available).
4. **Empty-window negatives** for the existence head, sampled from the same videos where the coarse pass shows
   no fly, plus arena-edge windows.
5. **Bouts** (from the earlier brainstorm, decision "c"): a permissive compute gate from the coarse pass, then
   strict behavioural bouts derived from the fine tracks; the proximity criterion is dropped (the male sings and
   approaches from a distance); the wing-angle feature is re-derived after v2 (rest baseline must be ~0).
   Bout definition is its own follow-up spec; v2 must deliver the stable tracks it needs.

## 3. Data

### 3.1 Real labels (unchanged)
v12 export: 2,661 train / 153 val framesets, annotation-first sex resolution, calibration groups A/B train, C val.
Weight 1.0. The ~735 consecutive labelled pairs form real T=2 pairs (Delta = 1).

### 3.2 Pseudo-labels (new)
Source: the p3b campaign run roots `<processed>/courtship/<Session>/<rec>/pose_mvq_p3b/bouts/bout_*/fly{0,1}/`
(kp3d in model order, kp2d per camera, existence, `mvq_meta.json` with identity source, containment drops,
collapse flags) for 160 bouts / 11 recordings, and, for single-fly diversity, mask-free P3b tracks on
free_running Session11 and Clip Session6 (§3.4).

Admission gates, all per frame, both flies:
- existence >= 0.8 for every fly present; a fly absent on the frame is admitted only as "absent" (an existence
  negative for its slot), never guessed;
- identity from human review (`identity_source == "mask"`), zero containment drops, `collapsed == False`;
- per-keypoint frame-to-frame step <= 0.3 mm on both neighbours (no spikes);
- reprojection self-consistency: per-view 2D head vs reprojected 3D <= 3 px median over keypoints;
- mask containment (fraction of reprojected keypoints inside the own mask, dilated 6 px) >= 0.9 in >= 5 cameras;
- the frame is >= 16 frames from any admitted frame of the same bout (decorrelation) unless it is a T=2 partner.

Target size: **~10,000 courtship framesets** (about 4x the real set), stratified: 50 % female-host / 50 %
male-host windows; >= 25 % contact frames (inter-fly centroid distance < 1.5 mm) and >= 25 % separated; all 11
recordings; no bout > 2 % of the set; wall/edge frames included in proportion. Plus **~2,000 single-fly
framesets** (§3.4). Each admitted frameset carries its T=2 partners at Delta = 1, 4, 16 when those frames also
pass the gates (else the pair is dropped, the single frame kept). Weight 0.3 (loss multiplier), tuned once by a
weight-sweep at 0.1 / 0.3 / 1.0 on a 5k-step probe if the first run shows validation regression.

Pseudo-labels are stored as a v12-format export (`red_data_3d_v12_pseudo_p3b_<date>/`) with the same manifest
schema plus `source: "pseudo"`, `checkpoint`, `gates` per frameset, so the existing loader, sampler and cohort
code apply unchanged.

### 3.3 Human review gate (required before training)
A stratified **gallery of ~300 admitted framesets** rendered in two cameras (overhead + one side), keypoints and
mask outlines, grouped by stratum, with an editable accept/reject CSV, plus histograms of every gate quantity.
Threshold: if > 3 % of the gallery is rejected, the offending gate is tightened and the set regenerated; a
rejected stratum (e.g. "contact, female host") is dropped entirely rather than thinned. The reviewed accept list
is committed beside the export manifest.

### 3.4 Single-fly data
free_running Session11 (7 recordings, single fly, same rig) and Clip Session6/2025_10_12_15_06_46 (921 frames,
own calibration, JARVIS `data3D.csv` reference). Pseudo-labels from the mask-free P3b pass (`--num-animals 1`,
single-fly typed rule) with the same gates minus identity; they provide a different arena/lighting, climbing
poses, and natural "one fly only" windows. Preconditions: the P3b single-fly field check passes (no phantom
second fly, keypoints stable); otherwise single-fly pseudo-labels wait for v2's first pass over them.

### 3.5 Negatives
Empty windows: for each recording, windows centred on random floor locations >= 6 mm from every tracked fly
centroid (coarse pass), plus windows on arena-edge fixtures where CenterDetect produced false peaks
(gate bout 1 type). Label: every slot absent. ~2,000 windows. They enter only the existence loss.

## 4. Model and training

Architecture unchanged (DINOv3 ViT-B/16 per view, affine ray tokens, cross-view fusion, 4-slot query decoder;
crops `(T, C, 448, 448, 3)` are already T-aware). Changes:

- **T=2 input** end to end: dataset yields pairs at Delta in {1,4,16} (real pairs Delta = 1); copy-paste
  extended to T=2 (the donor is pasted in both frames with its own two-frame motion, same affine shift per
  camera, contact placement as P3b); the loader's current `T` restriction to consecutive labelled framesets is
  generalised to "labelled at both frames, spacing Delta".
- **Identity-persistence loss**: the slot assignment is made on frame 0 and REUSED on frame 1 (no re-matching);
  a hinge on the two frames' per-slot centroid displacement beyond what the GT moved (units), and the existing
  cross-fly repulsion on both frames.
- **Existence negatives** (§3.5) in the existence loss with target 0 for every slot; ignore rules for
  present-but-unlabelled flies unchanged.
- **Recipe**: prompting off; jitter 1 mm (10 units); copy-paste p 0.8, contact p 0.7, contact sep (4, 25);
  female-host weight so the host-sex ratio is 0.5; cross-fly repulsion 20; camera dropout (each camera absent
  with p 0.1); horizontal flip with left/right keypoint relabelling; wing keypoint loss weight x2 and wing
  visibility weight x2; batch 32 across 8 devices (4/device -- the 7-device layout hung), 40k steps, lr 3e-4
  warmup 1k, eval every 2k, save every 2k.
- **Calibration after training**: temperature scaling of existence and per-view visibility on the validation
  set so that 0.5 means 0.5; the calibrated temperatures are stored in `final/mvq_run.json` and applied by
  `MVQRunner`.

## 5. Acceptance (all must hold before v2 replaces P3b)

| check | threshold |
|---|---|
| validation (153 human framesets, unprompted, oracle and policy) | mpjpe <= 0.085 mm; policy misses 0; sex_acc >= 0.99; contact_pair <= 0.146 mm; cross_fly_frac <= 0.036 overall and <= 0.055 contact pairs; single_fly cohort misses 0 |
| centre-shift rule (spec P4 §6) | error at 1 mm within +10 %; misses < 2 % |
| masked field checks, bouts 1/4/28 of 20_04 (containment OFF) | pose-jump frames 0 %; straddle frames <= P3b; female-missing <= P3b |
| mask-free field checks, bout 117 stride 1 | pose-jump frames <= 5 % (P3b 28 %); straddle <= 10 % (P3b 23 %); no identity swap in the contact sheet |
| mask-free coarse pass, 20_04 | female trackable >= 0.85 of coarse frames (jitter-10: 0.48); no existence >= 0.5 on the stale-window bout 69 frames; gate bout 1 false peaks not asserted |
| single fly (free_running 2 bouts, Clip Session6) | exactly one fly asserted on >= 99 % of frames; keypoint stability as bouts 28; agreement with the DLT/JARVIS reference within the courtship LOO band |
| existence calibration | reliability curve within 0.05 of the diagonal on validation |
| figures read back | contact sheets for each field check; validation overlay of the female cohort |

## 6. Deliverables and order

1. `scripts/pseudo_labels/extract_p3b_pseudolabels.py` (gates, stratified sampling, v12-format export, gate
   histograms) + `scripts/pseudo_labels/pseudolabel_gallery.py` (review gallery + accept/reject CSV) + tests.
2. Human review of the gallery (Elliott); regeneration if needed; the accept list committed.
3. Single-fly pseudo-labels (after the P3b single-fly field check) and empty-window negatives.
4. Loader/sampler: T=2 with Delta, pseudo/real weights, negatives; copy-paste for T=2; tests.
5. Losses: identity persistence; negatives in existence; wing weights; tests.
6. Camera dropout and flip augmentation; tests.
7. Training run `mvq_t2_v2_<date>` from scratch on 8 GPUs (~1 day); the watcher reports every eval.
8. Calibration; acceptance suite (§5) as one script that writes a scorecard; figures read back; notes.
9. If accepted: `configs/mvq/v2.yaml`, campaign defaults, re-lift both sessions once more.

## 7. Risks

- Pseudo-label confirmation bias: mitigated by the human gallery gate, the 0.3 weight, hard admission gates,
  validation on human labels only, and the field checks that use frames never admitted.
- T=2 doubles per-step compute; batch stays 32 pairs (64 frames) -> ~2x memory; if it does not fit, batch 16
  pairs with gradient accumulation of 2.
- Copy-paste for T=2 is new code; a byte-equality test against the T=1 path for Delta = 0 guards it.
- Single-fly pseudo-labels depend on the field check; if P3b hallucinates a second fly there, single-fly data
  enters only as negatives for the extra slots.
- Wing landmarks may be biased in the pseudo-labels themselves (they come from P3b); the wing weight increase
  only sharpens what is already there. A wing-specific fix needs the resting-angle audit (bout-level figure of
  wing keypoints on rested flies) before v2's weights are frozen.
