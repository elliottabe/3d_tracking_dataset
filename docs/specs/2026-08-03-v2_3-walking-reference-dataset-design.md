# v2.3 walking reference dataset — design

**Date:** 2026-08-03
**Goal:** Re-run IK over all free-running walking bouts with the `fruitfly_v2.3`
anatomy and pack the result into a single batched HDF5 matching the format of
`/gscratch/portia/eabe/fly_neuromech/data/datasets/Fruitfly_v1_walk_1000hz_interp_padded.h5`.

**Output:** `/gscratch/portia/eabe/fly_neuromech/data/datasets/Fruitfly_v2_3_walk_1000hz_interp_padded.h5`

---

## 1. Scope

This is **not** a full vision-pipeline rerun. 3D keypoints for every walking bout
already exist on disk. The work is:

1. make `v2.3` a first-class anatomy (it is not usable today),
2. re-run preprocessing → STAC IK → FK/postprocess → combine under that anatomy,
3. replace the notebook repack step with a committed script.

### Data source

The 372 walking bouts of the **old (JARVIS) route**, spread over 22 session dirs:

| session group | dirs | bout-summary filename |
|---|---|---|
| `free_running/session1_7/Predictions_3D_*` | 7 | `free_walking_bouts_summary.csv` |
| `free_running/session10/Predictions_3D_*` | 9 | `free_walking_bouts_summary.csv` |
| `free_running/session11/Predictions_3D_*` | 6 | `free_running_bout_summary.csv` |

372 bouts (162 / 157 / 53 by group), 134,863 frames, per-bout T ∈ [121, 1514].
All 22 dirs have `data3D.csv` and `info.yaml`. 19/22 also have
`walking_bouts_summary.curated.modified.json`, but no script or config on this
route reads it (verified by grep), so the 3 dirs without it are not a risk —
only `bouts_csv` drives bout selection.

**Explicitly out of scope:** the 68 bouts from the newer SAM3/ViTPose per-bout
pipeline (`free_running/Session11_*_bouts/`). They cover the same Session11
footage with a better front-end, but there is no aggregator from the per-bout
layout to the postprocess input format, and mixing front-ends would make the
dataset inhomogeneous. Revisit as a separate A/B.

### Reference format (target)

Measured from the v1 file: 198 clips × 1588 frames.

| dataset | shape | notes |
|---|---|---|
| `qpos` | (N, T, 93) | `[xyz(3), quat wxyz(4), 86 hinges]`, model units |
| `qvel` | (N, T, 92) | `[lin(3), ang(3), 86]`, forward-difference at dt=1e-3, last frame 0 |
| `xpos` | (N, T, 68, 3) | MuJoCo body frames from FK |
| `xquat` | (N, T, 68, 4) | |
| `kp_data` | (N, T, 150) | flat 50×3 tracked keypoints |
| `clip_lengths` | (N,) int32 | |
| `qpos_names/<i>` | 93 scalar strings | group, one per **column**; first 7 are `'free'` |

Under v2.3 the widths become **qpos 101, qvel 100, xpos/xquat 74 bodies,
qpos_names 101**. `kp_data` stays 150 (the 50 keypoints are identical across
anatomies).

### Decisions taken

- **Model:** `fruitfly_muscles_warp.xml` is the canonical v2.3 source.
- **`clip_lengths`:** store **true unpadded** lengths. The reference file stores
  the *padded* length (all 198 entries = 1588), a bug that `fly_mimic` works
  around with `unpadded_clip_lengths()` sniffing repeated trailing frames.
  Storing true lengths still loads correctly in `HDF5ReferenceClips` and makes
  `use_unpadded_clip_length` unnecessary.
- **Segment calibration stays OFF.** `preprocessing.calibration.enabled: False`
  (the current default) — no per-segment model morphing for this run. Only the
  single global Procrustes scale applies. See §2.4 for why this also removes a
  silent-failure mode.
- **Adhesion actuators are commented out, not deleted**, matching how v2.1
  handles the same 8 actuators.
- **Keypoint inputs:** re-run preprocessing under `anatomy=v2_3` (rather than
  reusing the v1 preprocessed files), so the global Procrustes scale is fit to
  the v2.3 rest pose instead of v1's.
- **Packing script home:** `3d_tracking_dataset/scripts/export/`.

---

## 2. Why v2.3 does not work today

Three blockers, all verified empirically on a compute node.

### 2.1 MJX cannot load the v2.3 model

Both v2.3 variants carry 8 `<adhesion>` actuators (2 labrum + 6 tarsal claw),
which are `mjTRN_BODY` transmissions that MJX does not implement. Measured:

```
v2.3/assets/fruitfly.xml        as-is                 nu=70  body_trn=8 -> MJX_FAIL [mjTRN_BODY] not supported
v2.3/assets/fruitfly.xml        adhesion stripped     nu=62  body_trn=0 -> MJX_OK
v2.3/fruitfly_muscles_warp.xml  as-is                 nu=272 body_trn=8 -> MJX_FAIL [mjTRN_BODY] not supported
v2.3/fruitfly_muscles_warp.xml  adhesion stripped     nu=264 body_trn=0 -> MJX_OK
v2.1/fruitfly_v2.1_muscles.xml  as-is                 nu=197 body_trn=0 -> MJX_OK
```

In all cases `nq=101, nv=100, nbody=74` is unchanged by the strip — the
kinematic tree is untouched. This is not muscles-specific: it hits both
variants identically. v2.1 works only because its 8 adhesion actuators are
commented out (`fruitfly_v2.1_muscles.xml:2293-2300`); v2.3 re-enabled them.

Impact if unfixed: `postprocess_stac_data.py:818 mjx.put_model(mj_model)` hard-fails.

### 2.2 The v2.3 model has no tracking sites

v2.3 has 1059 sites, **all** `mu_*` muscle sites — 0 named `tracking[...]`.
v2.1 has 50.

**STAC itself does not need them** — `stac.py:214-275 _create_body_sites` adds all
50 marker sites at runtime from `KEYPOINT_MODEL_PAIRS` + `KEYPOINT_INITIAL_OFFSETS`.
But two other stages read them out of the XML, so they must be added anyway:

- **Preprocessing (Stage 3) — hard dependency.**
  `preprocess_keypoints_for_ik.py:256` selects sites via `'tracking[' in name`,
  strips the wrapper, and matches skeleton nodes against the result to define
  both the keypoint column order (`reorder_to_xml_site_order`) and the rest-pose
  Procrustes reference. With 0 tracking sites `tracking_names_clean` is empty,
  all 50 nodes land in `unmatched`, and the reorder degenerates — it prints a
  warning but does not raise.
- **Postprocess (Stage 5).** `postprocess_stac_data.py:826` selects by
  `'tracking' in site.name` → `site_indices = []` → empty `(T, 0, 3)`
  `xpos_egocentric` / `site_xpos`. Silent garbage, not a crash.

Site names are `tracking[<KP_NAME>]`, e.g. `tracking[Scutellum]`. v2.1 and v2.3
have **identical body names**, so the sites transfer directly.

**Source of truth: `KEYPOINT_INITIAL_OFFSETS`, not the v2.1 XML.** Generating the
sites from the anatomy config guarantees the XML sites and STAC's runtime sites
agree. Compared numerically, the two sources match on 46/50 sites exactly, differ
by sub-mm rounding on `Antenna_Base` / `EyeL` / `EyeR`, and genuinely disagree on
one:

```
T1L_TaTip xml=+0.005  cfg=+0.005      T1R_TaTip xml=-0.005  cfg=-0.005
T2L_TaTip xml=+0.005  cfg=+0.005      T2R_TaTip xml=-0.005  cfg=-0.005
T3L_TaTip xml=+0.005  cfg=-0.005  <-- config breaks the L/R mirror
T3R_TaTip xml=-0.005  cfg=-0.005
```

The v2.1 XML follows the left/right pattern; `v2_muscles.yaml` has the left T3
tarsal tip mirrored onto the right side. These are *initial* offsets that STAC
subsequently fits, so the cost is a poorer starting point for one keypoint rather
than a wrong result — but `configs/anatomy/v2_3.yaml` must carry the corrected
`T3L_TaTip: 0 0.005 0` rather than copy the bug forward.

### 2.3 Model files are not where the configs look

`configs/paths/hyak.yaml:25` sets `body_model_dir: ${paths.cwd_dir}/models/`, and
`models/` contains only `fruitfly_v1` and `fruitfly_cse`. All v2.x models live in
`fly_neuromech/fruitfly_body_models/`. (Note `configs/anatomy/v2.yaml:10` points at
`fruitfly_v2.1/fruitfly_v2.1.xml`, which does not exist — copy `v2_muscles.yaml`,
not `v2.yaml`.)

### 2.4 Non-blockers, confirmed

- **Segment scales:** `configs/preprocessing/default.yaml:62` has
  `calibration.enabled: False`, and none of the 22 preprocessed files contain
  `segment_scales`. Verified: each has exactly
  `info/{clip_lengths,fly_ids,source_flies}`. So the v1-body-name morph hazard is
  inert. (It would be live if calibration were enabled: `rescale.py:125-131`
  skips unknown body names **silently**, so v1 `tarsus_*`/`claw_*` names against a
  v2.3 model would half-morph the model with no warning.)
- **Floor alignment:** the v1 `claw_T1_left` names still resolve under v2.3 by
  accident, because `postprocess_stac_data.py:590-593` uses a **substring** match
  and `'claw_T1_left' in 'tarsal_claw_T1_left'` is True. We set the names
  explicitly rather than depend on that.
- **Keypoint order:** `KP_NAMES` and `KEYPOINT_MODEL_PAIRS` key order are
  identical across v1/v2/v2_muscles. This matters because
  `prune_model_to_available` early-returns without reordering when the keypoint
  sets match exactly (`keypoint_prune.py:50-53`), after which columns are matched
  **positionally** against `KEYPOINT_MODEL_PAIRS.keys()` order.

---

## 3. Architecture

Seven stages. Each writes a distinct artifact and is independently verifiable.

```
Stage 0  model prep        -> models/fruitfly_v2.3/fruitfly_muscles_warp.xml (edited in place)
Stage 1  anatomy config    -> configs/anatomy/v2_3.yaml
Stage 2  postproc config   -> v2.3 floor-alignment end effectors
Stage 3  preprocessing     -> 22 x preprocessing/preprocessed_bout_v2_3_free_running.h5
Stage 4  STAC IK           -> 22 x stac/Fruitfly_ik_v2_3_free_running.h5
Stage 5  postprocess FK    -> 22 x postprocessing/ik_output_v2_3_free_running.h5
Stage 6  combine           -> ik_output_combined_v2_3_free_running{,_interpolated}.h5
Stage 7  pack              -> Fruitfly_v2_3_walk_1000hz_interp_padded.h5
```

### Stage 0 — v2.3 IK/FK model

`models/fruitfly_v2.3` → symlink to
`fly_neuromech/fruitfly_body_models/fruitfly_v2.3`, so `${paths.body_model_dir}`
resolves.

Two edits to `fruitfly_muscles_warp.xml`, applied **in place** in the body-model
tree (this modifies a file in the `fly_neuromech` repo):

1. **Comment out** the 8 `<adhesion>` actuators — 4 two-line blocks at lines
   2392-2393, 2539-2540, 2628-2629, 2723-2724 — wrapping them in `<!-- -->`
   exactly as `fruitfly_v2.1_muscles.xml:2293-2300` already does. Commenting
   rather than deleting keeps the muscle model's adhesion definition recoverable
   for dynamics work, where MJX is not in the loop. Leave the
   `<default class="adhesion*">` blocks alone: defaults create no actuators, and
   the `adhesion-collision` geom class is needed for FK geometry.
2. **Add the 50 `tracking[<KP_NAME>]` sites**, generated from
   `configs/anatomy/v2_3.yaml`'s `KEYPOINT_MODEL_PAIRS` + `KEYPOINT_INITIAL_OFFSETS`
   (see §2.2 for why the config, not the v2.1 XML, is the source of truth), each
   added to its mapped parent body.

Step 2 is scripted (`scripts/models/add_v2_3_tracking_sites.py`) rather than
hand-written, so it stays reproducible if v2.3 is updated upstream and so the
XML cannot drift from the anatomy config. Step 1 is a literal 8-line comment-out.

**Ordering:** step 2 reads `configs/anatomy/v2_3.yaml`, so Stage 1 must be
authored first. Stages are numbered by data flow, not execution order; the build
order is 1 → 0 → 2 → … The `mjcf_path` in the config points at the XML that
step 2 edits, which is fine — the config is only read for its
`KEYPOINT_MODEL_PAIRS` / `KEYPOINT_INITIAL_OFFSETS`, and the model is not
compiled until Stage 0's verification.

**Verification:** compile and assert `nq=101, nv=100, nbody=74, nu=264`,
`actuator_trntype` contains no `mjTRN_BODY`, exactly 50 `tracking[...]` sites
whose names equal `KP_NAMES` and whose parent bodies equal
`KEYPOINT_MODEL_PAIRS`, and `mjx.put_model` succeeds.

### Stage 1 — `configs/anatomy/v2_3.yaml`

Copy `v2_muscles.yaml` and change:

- `name: v2_3`
- `mjcf_path: ${paths.body_model_dir}/fruitfly_v2.3/fruitfly_muscles_warp.xml`
- `arena_path: ${paths.body_model_dir}/fruitfly_v2.3/floor.xml`
- `joint_names`: v2.1's 82 **plus** the 13 v2.3 additions —
  `antenna_{left,right}`, `antenna_abduct_{left,right}`,
  `antenna_twist_{left,right}`, `haltere_{left,right}`, `haustellum`,
  `haustellum_abduct`, `labrum_{left,right}`, `rostrum`.

`body_names`, `KP_NAMES`, `KEYPOINT_MODEL_PAIRS`, `KEYPOINT_INITIAL_OFFSETS`,
`SITES_TO_REGULARIZE` carry over **verbatim, order preserved** (see §2.4), with
one deliberate correction: `T3L_TaTip: 0 0.005 0`, restoring the left/right
mirror that `v2_muscles.yaml` breaks (§2.2).

### Stage 2 — postprocessing config

Set explicitly for this run:

```yaml
floor_alignment.end_effector_names:
  [tarsal_claw_T1_left, tarsal_claw_T1_right,
   tarsal_claw_T2_left, tarsal_claw_T2_right,
   tarsal_claw_T3_left, tarsal_claw_T3_right]
```

Interpolation stays `source_hz: 800.0 → target_hz: 1000.0`, `method: cubic`;
floor `percentile: 5.0`, `target_z: -0.125`.

### Stage 3 — preprocessing

Run `preprocess_keypoints_for_ik.py` per session dir with `anatomy=v2_3
dataset=free_running`. `bouts_csv` defaults to
`${dataset.name}_bouts_summary.csv` = `free_running_bouts_summary.csv`, which
matches **nothing** on disk, so it must be overridden per group:

- `session1_7`, `session10` → `preprocessing.bouts_csv=free_walking_bouts_summary.csv`
- `session11` → `preprocessing.bouts_csv=free_running_bout_summary.csv`

Output: `preprocessing/preprocessed_bout_v2_3_free_running.h5` per dir — which
satisfies the downstream filename templates natively, with no further overrides.

**Verification gate:** total bouts across the 22 files must be **372**, matching
the v1 route. A mismatch means bout selection changed and must be reconciled
before continuing — a different bout set would break comparability with the v1
and v2_muscles datasets.

### Stage 4 — STAC IK

`batch_run_stac.py --anatomy v2_3 --dataset free_running`, once per session root
(`find_preprocessed_files` uses a non-recursive `glob`, so it needs each of
`session1_7/`, `session10/`, `session11/` separately).

**Patch required:** `batch_run_stac.py:440-444` restricts `--dataset` to
`['', 'courtship', 'stationary', 'amputation']`; add `free_running`.

Output: `stac/Fruitfly_ik_v2_3_free_running.h5` (+ `Fruitfly_fit_...h5`) per dir.

### Stage 5 — postprocess

`batch_postprocess_predictions.py --anatomy v2_3 --dataset free_running`
(`rglob`, so one call covers all 22). Does floor alignment, 800→1000 Hz cubic
interp, MJX FK for `xpos`/`xquat`/`site_xpos`/`xpos_egocentric`, and finite-diff
`qvel`.

Output: `postprocessing/ik_output_v2_3_free_running.h5` per dir.

### Stage 6 — combine

`combine_data.py anatomy=v2_3 dataset=free_running +base_dir=<free_running root>`
→ `ik_output_combined_v2_3_free_running.h5` and `..._interpolated.h5`, with
`bout_000..bout_371` groups plus `info/`.

### Stage 7 — pack

New `scripts/export/pack_reference_clips.py`, replacing cells 75–77 of
`fly_neuromech/fly_mimic/notebooks/Walking_data_cleaning.ipynb`.

Reads the combined interpolated h5, and for each of `qpos, qvel, xpos, xquat,
kp_data`:

- pad every bout to `T_max` by **repeating the final frame** (`np.tile` of
  `arr[-1:]`) — matching the reference convention, not zero-padding;
- stack along a new leading clip axis;
- write `clip_lengths` = the **true** per-bout lengths (§1);
- write `qpos_names` as a group of 101 scalar strings, one per column, taken
  from `info/names_qpos` (stac_mjx `_part_names`, so the first 7 are `'free'` —
  this is the per-column convention `data_utils.py:122-155` detects to set
  `qpos_root_offset=0, qvel_root_offset=-1`);
- record provenance in root attrs: source combined h5, anatomy `v2_3`, model
  XML path, `source_hz`/`target_hz`, git SHA, and an explicit
  `clip_lengths_are_true: True` flag noting the deviation from the v1 file.

CLI: `--input <combined_interp.h5> --output <dataset.h5>`, so the same script
serves future anatomies.

---

## 4. Testing

**Unit (fast, no cluster):**

- Stage-0 model: commenting the adhesion actuators drops `nu` 272→264 and leaves
  `nq/nv/nbody` at 101/100/74; no `mjTRN_BODY` remains; `mjx.put_model` succeeds;
  exactly 50 `tracking[...]` sites, names == `KP_NAMES`, parents ==
  `KEYPOINT_MODEL_PAIRS`, positions == `KEYPOINT_INITIAL_OFFSETS`.
- `configs/anatomy/v2_3.yaml`: every `KEYPOINT_MODEL_PAIRS` body resolves in the
  compiled model; `KP_NAMES` order is byte-identical to `v1.yaml`'s; joint set
  equals the model's minus the free joint; `T3L_TaTip` mirrors `T3R_TaTip`
  (regression test for the §2.2 bug).
- `pack_reference_clips.py`: on a synthetic 3-bout dict with known lengths —
  padding repeats the last frame, `clip_lengths` are the true lengths, shapes are
  `(3, T_max, ...)`, `qpos_names` is a 101-entry group.

**Integration gates, in order:**

1. Stage 3 → 372 bouts total (hard gate; see §3 Stage 3).
2. Stage 4 → one bout's `stac_ik.h5` has `qpos (T, 101)`, `xpos (T, 74, 3)`.
3. Stage 5 → `xpos_egocentric` is `(T, 50, 3)` and **not** `(T, 0, 3)` — this is
   the §2.2 silent-garbage check.
4. Stage 7 → final file opens through `fly_mimic`'s `HDF5ReferenceClips`, and
   `remap_to_model` succeeds against a compiled v2.3 model.

**Sanity comparison against v1:** per-bout root-translation traces and IK
residuals should be close between the v1 and v2.3 runs on the same bouts. A
large systematic divergence points at the rest-pose rescale (Stage 3) or the
keypoint-order hazard (§2.4), not at real anatomy differences.

---

## 5. Risks

| risk | mitigation |
|---|---|
| Re-running preprocessing yields ≠372 bouts | Hard gate at Stage 3; reconcile before continuing |
| v2.3 rest-pose rescale shifts IK vs v1 | Expected and desired; quantify via the §4 sanity comparison |
| Keypoint column order silently wrong | Unit test asserts `KP_NAMES` order matches v1 exactly |
| Empty egocentric arrays | Explicit Stage 5 shape gate |
| Someone enables `calibration.enabled` later | Out of scope by decision (§1); documented in §2.4 — v1 segment names would half-morph a v2.3 model silently |
| Cross-repo XML edit is lost or diverges | Commented (not deleted) so it is self-documenting; tracking-site insertion is scripted and re-runnable; Stage-0 assertions catch a reverted file |
| Login-node saturation | All heavy stages run on compute nodes |

---

## 6. Deliverables

**New files**
- `scripts/models/add_v2_3_tracking_sites.py`
- `models/fruitfly_v2.3` (symlink)
- `configs/anatomy/v2_3.yaml`
- `scripts/export/pack_reference_clips.py`
- tests per §4

**Modified**
- `fly_neuromech/fruitfly_body_models/fruitfly_v2.3/fruitfly_muscles_warp.xml`
  — adhesion actuators commented out; 50 tracking sites added (cross-repo edit)
- `scripts/batch_run_stac.py` — add `free_running` to `--dataset` choices
- `configs/postprocessing/` — v2.3 floor-alignment end effectors
- `docs/running_the_pipeline.md` — replace the "anatomy=v2_muscles does NOT work"
  section with the working v2.3 recipe

**Output**
- `/gscratch/portia/eabe/fly_neuromech/data/datasets/Fruitfly_v2_3_walk_1000hz_interp_padded.h5`
  — 372 clips, `qpos (372, T_max, 101)`, `xpos (372, T_max, 74, 3)`
