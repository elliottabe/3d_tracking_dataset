# mvq P3a: sex-typed instance slots, existence targets, multi-view copy-paste

Amends `docs/specs/2026-09-03-mvq-dinov3-query-decoder-design.md` (§4.6
instance queries, §5 matching and term 6, §6.3 augmentation, §8 assembly).
Where the two disagree, this document wins. Approved by the user 2026-09-04
after the step-14000 review of run `mvq_t1_b16_local8_20260904`
(`docs/benchmark/2026-09-mvq/p2-t1-notes.md`, "Step-14000 check").

## 1. Why

Three findings from the step-14000 check, all with one root cause: an
instance slot has no fixed meaning.

1. **The existence head is uncalibrated** (sigmoid ~0.45 on every slot;
   the unprompted policy misses 99 % of single-fly and 34 % of two-fly val
   windows). Matching is by geometric cost only, unmatched slots get no
   geometric gradient, so slots duplicate each other, the argmin among
   near-ties is a coin flip, and the BCE optimum for a coin-flip target is
   its marginal frequency (1/3 on single-fly crops, 2/3 on two-fly crops).
   Train `exist_acc` has been flat at ~0.82 since step 2000.
2. **Stacked mating pairs mix the two flies** (val #60-#67, 0.5-1.2 mm,
   one slot's keypoints split across both bodies), and the mask prompt does
   not fix it. A slot has no concept of *which* animal it is looking for.
3. **26 % of training windows contain an unlabelled fly.** Recording
   `2025_10_20_13_20_04` (692 train framesets, courtship pair, only the
   host labelled, no masks) puts a visible male in the crop while the loss
   tells every unmatched slot "nothing here" and never penalises a slot that
   gathers his keypoints.

The user's goal beyond the fixes: the network should know what a female
and a male are, so identity in courtship is a property of the output, not
of a later tracking step.

**Data facts that shape the design** (v12 export `red_data_3d_v12_export0902`):
- Every labelled fly in all 16 recordings has a sex (13 recordings from the
  directory name, 3 from user statements on 2026-09-02); none is unknown.
  For the 4 mixed recordings the directory-name convention is fly0 =
  female, fly1 = male -- a convention, so it gets a figure gate (§7).
- Same-sex pairs exist: `2026_06_09_15_38_35` is two females (31 train
  framesets, both labelled).
- Two-fly-labelled framesets: train 88 + 31 (`17_28_34` mixed, `15_38_35`
  F/F), val 74 (`12_11_50` group C, `15_25_51` group A, both mixed).
- Host masks exist for every window (val 100 %, train 100 % on a 120-window
  sample); the manifest's `has_masks` flag is stale and is not used.
- Cameras are affine, so a 3D translation is an exact per-camera 2D
  translation (`geometry.py`): multi-view copy-paste is exact.

## 2. Decisions (user, 2026-09-04)

- P3 is two specs. **P3a** (this one): identity, existence, copy-paste,
  eval plumbing, one fresh T=1 run. **P3b** (later): 30k evaluation on bout
  28 and the frozen benchmark, refinement/fusion ablations, T=2.
- The P3a run is queued on `ckpt-g2` (requeue) as soon as the code is
  ready; the running 30k job is not touched.
- Gray-fill is dropped: typed slots want the second fly visible and
  labelled, and the unlabelled-male recording has no mask to fill anyway.

## 3. Slot semantics

`n_instances = 4`, fixed meaning by index:

| slot | meaning | prompt token added |
|---|---|---|
| 0 | the prompted fly (whatever its sex) | yes, when `prompt_on` |
| 1 | the female | no |
| 2 | the male | no |
| 3 | other (second fly of an already-taken sex, or unknown sex) | no |

`assign_slots(fly_sex, fly_valid, prompt_on, dist)` in `train/matching.py`
replaces cost-based `match`. It is a pure function of labels -- no
prediction enters, so no coin flips anywhere:

1. Flies are processed host first (fly 0), then the others by increasing
   `dist` = distance of the fly's labelled-3D centroid from the ROI origin.
2. The host goes to slot 0 when `prompt_on`.
3. Otherwise a fly goes to its sex slot (female 1, male 2) if that slot is
   still free, else to slot 3. Unknown sex goes to slot 3.
4. Invalid flies (`fly_valid` false) get `-1`.

Outputs: `assign (B,F)` slot per fly, `slot_target (B,I)` = 1 where a fly
was assigned, `slot_ignore (B,I)` (§4). `enumerate_assignments`/`match`
are deleted with their tests; nothing else calls them.

Consequences: on a single-female crop, unprompted, exactly slot 1 exists;
prompted, exactly slot 0 exists and slot 1 does not. On a mixed pair
unprompted, slots 1 and 2 exist. On the F/F recording the nearer female
takes slot 1, the other slot 3. Every target is a deterministic function of
the image content plus the prompt flag.

The decoder's `e_inst` embeddings are unchanged in form; only their
supervision changes. Self-attention among 3D queries already lets the
female and male queries see each other.

## 4. Existence, ignore, and sex targets

**New batch keys** (all from the loader, §6):
- `fly_sex (F,) int8`: 0 female, 1 male, -1 unknown.
- `unlabelled_sex () int8`: -1 = every animal in the window is labelled;
  else the sex code of the one unlabelled animal (2 = present, sex
  unknown). From the manifest: `n_flies(recording) - len(flies labelled in
  this window)`; at most one, since `n_flies <= 2 = max_flies`.

**Existence** (term 6). Target = `slot_target`. `slot_ignore[i]` is true
for the unlabelled animal's sex slot and for slot 3 when `unlabelled_sex
>= 0` (for code 2, slots 1-3 are all ignored). Slot 0 is never ignored:
its target is exactly `prompt_on`. BCE is a masked mean over non-ignored
slots; `exist_acc` likewise. Copy-paste windows carry `unlabelled_sex = -1`
(they are only built from fully labelled windows, §6).

**Sex head** (new term 8). `Heads.sex = Linear(D, 1)` on the same pooled
instance feature as `exist`; output `sex_logit (B,I)`. BCE (female = 1)
over assigned slots whose fly has a known sex, weight `sex: 0.5`. For the
typed slots this is a consistency check; for slots 0 and 3 it is the only
sex readout, and it is what the pipeline reads for the prompted fly.

**Everything else** (terms 1-5, 7, deep supervision, prompt annealing,
balanced sampler) is unchanged. Term 7 (repulsion to the other fly's same
part) now fires on every copy-paste window as well as the labelled pairs.

## 5. Model and assembly

- `MVQConfig.n_instances` default 4 (`configs/model/mvq.yaml`); the tiny
  test preset follows. The 3D-query count rises 150 -> 200 (T=1); the
  measured step-time delta is recorded in the notes, expected under 10 %
  since the backbone dominates.
- `assemble()` returns a fourth array `sex_prob (B,I)` and applies the
  same existence gate as today. The docstring states the slot table.
- `load_mvq_model` looks for `mvq_run.json` in the RUN directory first,
  then `final/` (backwards compatible). `run_training` writes
  `<run_dir>/mvq_run.json` (model, train, keypoint_names, `val: null`)
  before step 0, so any mid-run checkpoint loads without a staging
  directory. `final/mvq_run.json` keeps its final `val`.

## 6. Multi-view copy-paste (`data/mv_copy_paste.py`, in the loader, CPU)

Applied in `V12WindowDataset.__getitem__` when `train` and `T == 1`, with
probability `copy_paste_p` (default 0.5), only to windows that have ONE
labelled fly and `unlabelled_sex == -1`. Deterministic per
`(seed, i, epoch)` like the jitter.

**Source.** Another TRAIN window `j` of the same calibration group (same
`M`), drawn from a per-group index built at init; its host fly is the
donor: crop pixels, host mask, `kp2d`, `vis2d`, `kp3d_local`, `has3d`,
sex. Donor sex: opposite of the host with probability 0.7, else same. A
donor whose valid cameras do not cover the target's valid cameras is
rejected (up to 8 draws, then no paste). Val windows are never donors.

**Placement.** `D` (3, ROI-local world units) is sampled in the host's
body plane -- the plane of the two largest principal axes of the host's
labelled 3D points -- at a random in-plane direction and separation
`sep`: with probability 0.3 "contact" `sep ~ U(8, 15)` (0.8-1.5 mm, the
stacked-pair regime), else `sep ~ U(15, 60)`. Pasted labels that would
leave the crop in a target-valid camera reject the draw.

**Exactness.** With `t_local` from both samples, the per-camera shift in
crop px is `shift_c = M_c D + t_local_src[c] - t_local_tgt[c]` (real
valued). Donor crop and mask are translated by `shift_c` with bilinear
(mask: nearest) warps; `kp2d_paste = kp2d_src + shift_c`, `kp3d_paste =
kp3d_local_src + D`. Test: `project_local(kp3d_paste)` equals
`kp2d_paste` to 1e-3 px wherever the donor's own labels were consistent.

**Compositing.** Donor pixels replace target pixels under the shifted
mask, gain-matched by the ratio of crop medians clipped to [0.7, 1.4], with
a 1-px feathered edge. The donor is on top: host `vis2d[c,k]` is cleared
where the host keypoint falls inside the shifted donor mask, and the host
`prompt_mask` has the donor mask subtracted. The donor becomes fly 1 with
`fly_valid` true and its own sex. The pasted fly's mask is not shipped in
the batch (the prompt targets the host only).

**Not done.** T=2 windows (P3b), donors from other calibration groups,
photometric augmentation of the donor beyond the gain (the on-device
photometric aug applies to the whole composite afterwards).

## 7. Evaluation

`evaluate` changes:
- Unprompted policy: among typed slots 1-3 with existence >= 0.5, the one
  whose predicted centroid is nearest the ROI origin (slot 0 is never a
  candidate unprompted). Prompted policy: slot 0, falling back per sample
  to the unprompted policy when that window has no usable mask.
- New metrics: existence precision/recall PER SLOT (excluding ignored
  slots), `sex_acc` over assigned slots with known sex, and
  `mask_containment`: fraction of the policy instance's reprojected
  keypoints that fall inside the host's mask, over views where the mask is
  non-empty and the keypoint is labelled visible. This is the direct
  measure of cross-fly mixing; the 30k run's value at its final checkpoint
  is the baseline.
- New cohort `contact_pair`: two-fly windows whose labelled centroids are
  closer than 15 units. The step-14000 failures live here.

Figure gates, expectation stated in each script's docstring before it
renders, PNGs read back with the Read tool, regeneration command in the
notes:

1. `scripts/viz/mvq_sex_label_check.py` (before training): for the four
   mixed recordings, three framesets each, two cameras, fly0 and fly1
   labels in the shared palette with their manifest sex. Expectation: the
   fly labelled male is the one with the smaller body and dark abdomen
   tip; the female is larger with a pointed pale abdomen. A recording
   where this is reversed is fixed in the export before any training.
2. `scripts/viz/mvq_copy_paste_check.py` (before training): six pasted
   windows x 7 cameras, both flies' labels drawn, contact and far
   placements, mixed and same-sex donors. Expectation: the donor sits at
   the same place relative to the host in every camera (no camera shows it
   floating or offset), its labels are on its body in all views, and host
   keypoints under the donor are drawn as invisible.
3. `scripts/viz/mvq_overlay.py` at step 10k of the new run (mid-run, via
   the run-dir `mvq_run.json`): same three cases as gate 1 plus a
   `contact_pair` case. Expectation: on #60-#67 each typed slot's points
   stay on one animal; the female slot on the female.

## 8. Training run: warm start from the 30k run (user decision 2026-09-04)

The 30k T=1 run's final EMA (`mvq_t1_b16_local8_20260904/final/`, expected
2026-09-04 ~11:15) seeds run `mvq_t1_b16_p3a_<date>`. Only two leaves change
shape: `decoder.e_inst` (3,768) -> (4,768) and the new `heads.sex`. A
shape-tolerant `warm_start_partial(model, src_dir)` in
`train/checkpoint.py` restores every leaf whose path AND shape match, copies
the three old slot rows into rows 0-2 of `e_inst` (row 3 keeps its fresh
init), leaves `heads.sex` fresh, and prints the list of leaves it did not
restore (must be exactly those two). Fresh optimizer, zero-seeded EMA, step
0; a requeue then resumes from the fine-tune's own `ckpt/` (resume beats
warm start, as `warm_start_restore`'s docstring already rules).
`MVQTrainConfig.warm_start: str | None` carries the source dir.

| | value |
|---|---|
| steps / warmup | 10,000 / 500 |
| peak LR head / backbone | 1e-4 / 1e-5 (`lr 1e-4`, `backbone_lr_mult 0.1`) |
| prompt_p | 0.5 constant (`prompt_p_start = prompt_p_end = 0.5`) |
| copy_paste_p | 0.5 from step 0 |
| n_instances / loss.sex | 4 / 0.5 |
| batch, EMA, aug, eval_every, save_every | as the 30k run (32, 0.999, on, 2000, 1000) |

Submitted to `ckpt-g2` with requeue (topology-independent resume,
fd2e0f2); the running 30k job is not touched.

**Fallback / ablation.** If at step 5k `sex_acc` on group C is below 0.9
or `mask_containment` on `contact_pair` has not improved over the 30k
baseline, launch the same config as a FRESH 30k run (`warm_start: null`,
`lr 3e-4`, prompt anneal as before). The pair then answers whether the
warm start held the typed slots back.

Notes go in `docs/benchmark/2026-09-mvq/p3a-notes.md` with the
unrestored-leaf list, the step-time delta, the gate figures' readings, and
the same-joint DLT comparison at the end.

Acceptance (versus the 30k run's final checkpoint on the same val split):
unprompted policy miss fraction under 5 % on single-fly windows; per-slot
existence precision and recall above 0.9 on slots 1-2; `sex_acc` above
0.95 on group C; `mask_containment` higher on `contact_pair`; oracle MPJPE
not worse anywhere by more than 5 %. Failing the containment target does
not block the run's use; it triggers the assembly-time mask gate as the
next P3b item.

## 9. Testing (TDD, CPU)

- `test_mvq_matching.py` (replaces the three `match`/`enumerate` tests):
  slot assignment for single female / single male / mixed pair / F/F pair /
  unknown sex, prompted and unprompted; tie-break by distance; invalid
  flies; ignore mask for `unlabelled_sex` in {-1, 0, 1, 2}; jit-able.
- `test_mvq_losses.py`: existence BCE ignores ignored slots (changing an
  ignored slot's logit changes nothing); sex term only on assigned known-sex
  slots; loss near zero at ground truth with I=4; existing tests updated to
  I=4 and the new keys.
- `test_mv_copy_paste.py`: exactness (reprojection identity above), shift
  formula against a brute-force full-frame construction, host visibility
  cleared under the donor mask, prompt mask subtraction, rejection when the
  donor cameras do not cover the target, determinism per (seed, i, epoch),
  `unlabelled_sex` windows never pasted.
- `test_v12_windows.py`: new keys present with the right dtypes;
  `unlabelled_sex` is -1 for fully labelled windows and the manifest sex
  code for the courtship-host windows.
- `test_mvq_model.py`: `sex_logit` shape, assemble returns `sex_prob`,
  prompt still touches only slot 0.
- `test_checkpoint_warm_start_partial.py`: a 3-slot tiny model's final/ warm-starts a
  4-slot tiny model -- matching leaves equal, `e_inst[:3]` equal, row 3 and
  `heads.sex` untouched, the unrestored list is exactly those two paths.
- `test_train_mvq_smoke.py`: run-dir `mvq_run.json` written before step 0
  and loadable mid-run; evaluate emits per-slot existence, `sex_acc`,
  `mask_containment`, and the `contact_pair` cohort; prompted policy falls
  back when a window has no mask.

## 10. Code placement

- `jarvis_jax/train/matching.py`: `assign_slots`, `slot_ignore`; old
  functions removed.
- `jarvis_jax/train/losses_mvq.py`: uses `assign_slots`; term 8;
  `LossWeights.sex`.
- `jarvis_jax/models/mvq/decoder.py`, `model.py`: sex head, `n_instances`
  4, assemble.
- `jarvis_jax/models/mvq/checkpoint.py`: run-dir json lookup.
- `jarvis_jax/train/checkpoint.py`: `warm_start_partial`.
- `jarvis_jax/data/v12_windows.py`: `fly_sex`, `unlabelled_sex`, donor
  index, copy-paste hook; `WINDOW_KEYS` extended.
- `jarvis_jax/data/mv_copy_paste.py` (new): pure numpy planning, warp,
  composite.
- `jarvis_jax/train/train_mvq.py`: run-dir json at start, evaluate changes,
  `copy_paste_p` and `warm_start` in `MVQTrainConfig`.
- `configs/model/mvq.yaml`, `configs/train/mvq.yaml`.
- `scripts/viz/mvq_sex_label_check.py`, `scripts/viz/mvq_copy_paste_check.py`
  (new, repo root); `scripts/viz/mvq_overlay.py` gains the `contact_pair`
  case and shows the slot index and sex probability per row.

## 11. Risks

- **Sex from recording appearance, not anatomy.** 12 training recordings
  is few subjects. Group C in val is the honest check; copy-paste donors of
  both sexes into every recording's crops decouple sex from background.
- **Directory-name sex convention wrong for one recording.** Gate 1 in §7
  catches it before training.
- **Copy-paste realism.** Gain-matched hard pastes have edge artefacts the
  network could key on; the photometric augmentation runs after
  compositing, and the F/F and mixed real pairs remain in the data. If the
  `contact_pair` metrics improve only on pasted-looking windows, P3b adds
  Poisson blending.
- **Slot 3 starvation.** It is positive only on same-sex pairs (31 real
  framesets plus 30 % of pastes). Acceptable: its job is to exist rarely
  and correctly.
- **Step-time.** +50 3D queries per window; measured at launch.

## 12. Out of scope

T=2 training, refinement/fusion ablations, bout-28 evaluation of the 30k
run, assembly-time mask gating, gray-fill, Poisson blending, donors across
calibration groups, pseudo-labels (Phase 2 spec).
