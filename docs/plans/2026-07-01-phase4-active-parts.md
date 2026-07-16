# Phase 4: Active-Parts (Headless / Amputation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Solve silhouette-landmark IK for flies with a missing body part (an amputated front leg, or a headless fly) using ONE anatomy model per body type (V1 `fruitfly_v1_free.xml`, unchanged for every condition) plus a per-recording **active-parts mask** that (a) drops the off-part's markers, (b) locks + post-solve-clamps the off-part's joints to rest, and (c) excludes the off-part's mesh geoms from the silhouette landmark extraction. Prove "no phantom residuals": the off-part joints stay pinned at rest in the RETURNED qpos across all frames, AND the present-marker fit is unchanged versus an unmasked run. Run this end-to-end on REAL general-model recordings (amputee `S8_male_R_amp`, headless `headless_22_50`) — generating the missing SAM3 masks + STAC artifacts first — and validate PER-FRAME (the labeled framesets are temporally sparse; do not rely on smoothness).

**Architecture:** A new anatomy-agnostic `active_parts.py` config module keyed by body-part name (never by hardcoded index) declares, per part, its model joint qpos indices, its canonical keypoint names, and its mesh segment ids. `derive_active_parts(kp_names)` auto-derives which parts are "off" by diffing a recording's coco `keypoint_names` against the canonical 50 (with an explicit override). `build_active_mask(...)` turns the off-part set into a concrete mask dict against the real model+mesh: which marker coords to zero (`kps_to_opt`), which joint DOFs to lock (`qs_to_opt=False`), the `locked_qpos_idx` + `rest_qpos` used by the post-solve clamp, and the excluded mesh `seg_ids`/`vertex_idx`. The IK backbone (`stac_mjx.stac_core_jaxls.JaxlsBatchSolver`, `run_stac_bout.run`, `run_single_fly`/`run_ablation`) is reused UNMODIFIED except that the mask is applied in the cse layer: markers zeroed + joints locked BEFORE the solve, and — critically — a **post-solve clamp** (`clamp_locked_qpos`) overwrites the returned `qpos[:, locked_idx]` back to rest, because a `qs_to_opt=False` DOF still drifts as an LM variable under the smoothness/limit costs and that drift otherwise appears in the returned qpos (the "phantom"). A prerequisite reduced-schema patch to `cse_labels.build_bout` (intersect the model kp order with the recording's present names, mirroring `stac_mjx.keypoint_prune.prune_model_to_available`) lets the amputee/headless coco (44/47 names) flow through triangulation without a `KeyError`, and the wing augment/withhold path is made name-based so its indices shift correctly under the reduced schema.

**Tech Stack:** Python, JAX, NumPy, jaxls (LM, reused via `stac_mjx.stac_core_jaxls`), MuJoCo-MJX, `stac_mjx` (io + solver + `run_stac` + `keypoint_prune`), `jarvis_jax` (`reprojection_tool`, `affine_camera`, `silhouette_landmarks`, `silhouette_ik`, `marker_augment`, `cse_labels`, `run_stac_bout`, `silhouette_ik_solve`), SAM3 (PyTorch front-end, `JARVIS-HybridNet/tools/sam3_label_masks.py`), h5py, pytest.

## Global Constraints

- **One anatomy per body type, unchanged:** V1 `/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml` (nq=93 = 7 free root + 86 hinges) for ALL conditions. No per-case XMLs. Mesh `/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz` (65 segments; `vertex_segment` len 61666).
- **Anatomy-agnostic part table:** `PART_TABLE` keyed by part name; grouping is by name convention (legs `T{1,2,3}{L,R}` / joint suffix `_{left,right}`; wings `wing_*_{left,right}`; head kps `Antenna*`/`Eye*`; abdomen `Abd_*` / joints `abdomen*`). Index by JOINT name, not body name (`tarsus5_T1_right` moves body `claw_T1_right`).
- **Affine / telecentric cameras** (spec decision 8): per-camera 3×4 DLT, 3rd row `[0,0,0,1]`; projection has no perspective divide. `ReprojectionTool.reconstruct_point`/`reproject_point` are affine-consistent (the `/proj[2]` divide is a no-op).
- **3-D observation model** (spec decision 7): triangulate to 3-D, fit model markers to 3-D. Silhouette landmarks enter as confidence-gated 3-D markers via `marker_augment.augment_wing_markers` (`only_missing=True`, default `wing_weight=0.5`). No 2-D IK term (that is Phase 6).
- **Reuse, do NOT fork:** the solver `stac_mjx.stac_core_jaxls.JaxlsBatchSolver.solve_trajectory`, `run_stac_bout.run`, and `silhouette_ik_solve.run_single_fly`/`run_ablation` are used unchanged in spirit — the active-parts mask is applied in the cse layer (marker zeroing + joint locking pre-solve **and the post-solve clamp**), threaded through `run_single_fly` via a new `active_parts` kwarg (like Phase-3 threaded `ann_id_by_image`).
- **Off-parts auto-derived from `keypoint_names`** (user decision 2): `derive_active_parts(kp_names)` diffs the recording's coco `keypoint_names` against the canonical 50; omitted names identify off-part(s). Explicit `override` argument available. The missing part is encoded STRUCTURALLY (names OMITTED), NOT a v=0 visibility flag and NOT a condition field.
- **"No phantom residuals" (the acceptance definition):** the off-part joints are pinned to rest in the RETURNED qpos (post-solve clamp: `max|qpos[:,locked] - rest| < eps` across ALL frames) AND the present-marker reprojection/fit is within eps of the unmasked run (the mask must not perturb the rest of the fly). Locking `qs_to_opt=False` ALONE is insufficient — the LM variable still drifts; the clamp is what removes the phantom.
- **general_model = labeled FRAMES only** (user): the actual videos are NOT stored; SAM3 + STAC + IK run over ONLY the labeled framesets on disk (amputee: val=20 / train=174; headless: val=2 / train=18). These are temporally SPARSE / possibly non-contiguous, so smoothness across "adjacent" framesets is weak. Phase-4 validation is **PER-FRAME** (each frameset is well-posed via 7-view triangulation), NOT reliant on temporal continuity. **Prefer the TRAIN split** for a larger, more stable STAC offset fit (headless val=2 is too few — train=18 required there; amputee val=20 is borderline, train=174 preferred).
- **Environment (opposite `LD_LIBRARY_PATH` per stage):** SAM3 mask stage needs `export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib`. The JAX/STAC stage needs `unset LD_LIBRARY_PATH`. `unset LD_LIBRARY_PATH` BETWEEN them. Run GPU work DIRECTLY on the node (background bash for >600s); do NOT sbatch-and-idle.
- **CPU logic tests:** `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest ...`. Real-data GPU tests are gated by `@pytest.mark.skipif(not os.path.exists(...))` on the concrete artifact paths and run under the activated env (no `JAX_PLATFORMS=cpu`). There is no registered `@pytest.mark.gpu` marker and no `conftest.py` in `third_party/jarvis_jax/tests/`; follow the existing `skipif(os.path.exists(...))` gating convention used by `test_silhouette_ik_solve.py` / `test_multifly_bout.py`.
- **New code** under `third_party/jarvis_jax/jarvis_jax/cse/`; **tests** under `third_party/jarvis_jax/tests/`.
- TDD, DRY, YAGNI, frequent commits; each task: failing test → run (fail) → minimal impl → run (pass) → commit by EXPLICIT path (never `git add -A`). Do NOT commit `cse_work/`/`sam3_masks/` data outputs.
- **Branch:** `elliottabe/paper_update_062026` (rolling multi-phase branch; work directly on it).
- **Dependency note (honest):** **T3 (reduced-schema `build_bout` patch) MUST land before T6** — without it, `build_bout` on the 44/47-name coco raises `KeyError` in `_reorder_index` and no bout can be built. T6 (SAM3 + STAC on real data) is the environment-sensitive stage (opposite `LD_LIBRARY_PATH`), and T7 depends on T6's artifacts.

---

## File structure

NEW under `third_party/jarvis_jax/jarvis_jax/cse/`:

- `active_parts.py` **(new, Tasks 1–2, 5)** — the config/mask. `CANONICAL_KP_NAMES` (the 50 STAC names) + `PART_TABLE` (part → {joint qpos idxs, canonical kp names, mesh seg ids}); `derive_active_parts(kp_names, override=None)`; `build_active_mask(kp_names, model_xml, mesh_npz, off_parts)`; `clamp_locked_qpos(qpos, mask)`; `apply_active_mask_to_inputs(inputs, mask)`; and (Task 5) `excluded_vertex_indices(mask, mesh_npz)`.
- `run_active_parts_ik.py` **(new, Task 7)** — thin driver/CLI: given a general_model recording, derive active parts from its coco, run silhouette-IK with the mask via `run_single_fly(..., active_parts=...)`, report present-marker `reproj_px` + the no-phantom check.

MODIFY:

- `cse_labels.py` **(Task 3)** — reduced-schema patch: intersect the model `kp_order` with the recording's present `keypoint_names` in `build_bout`/`triangulate_recording` before `_reorder_index`, and write the shorter `kp_order` into the bout h5 (mirrors `stac_mjx.keypoint_prune.prune_model_to_available`). Full-schema path unchanged.
- `silhouette_ik_solve.py` **(Tasks 4–5)** — (i) make the wing augment/withhold path NAME-based so its STAC indices shift correctly under a reduced schema; (ii) thread an optional `active_parts` mask dict through `run_single_fly`/`run_ablation` (drop off markers + lock joints pre-solve + `clamp_locked_qpos` post-solve); (iii) geom-exclusion: exclude off-part vertices from the silhouette wing-tip extraction / coverage path.
- `silhouette_landmarks.py` **(Task 5)** — accept an optional `exclude_vertex_idx`/`exclude_seg_ids` so `wing_side_vertices` and any coverage metric ignore the off-part's vertices.

TESTS under `third_party/jarvis_jax/tests/`:

- `test_active_parts.py` **(new, Tasks 1–2, 5)** — part table, derive, mask build, clamp (CPU/model-load), and the geom-exclusion vertex-set check.
- Reduced-schema `build_bout` test added to `test_cse_labels_reduced_schema.py` **(new, Task 3)** — real-data-gated on the amputee coco.
- Name-based wing + mask-threading + SIMULATED no-phantom tests added to `test_silhouette_ik_solve.py` **(Task 4)**.
- Real data-prep command sequence + artifacts-exist check documented and scripted in `test_active_parts_realdata.py` **(new, Tasks 6–7)** — data-prep is NOT a pytest (documented reproducible commands + an artifacts-exist assertion helper); T7 is a real-data-gated GPU pytest.

**Shared test/path constants** (used across the real-data-gated tests):

```python
GM = "/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
ANATOMY = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs/anatomy/v1.yaml"
STAC_CFG = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs"
# amputee (right front leg distal off; T1R_ThxCx kept):
AMP = f"{GM}/S8_male_R_amp"; AMP_REC = "2026_02_13_13_44_49"       # 44 kp; val=20, train=174
# headless (head off):
HL = f"{GM}/headless_22_50"; HL_REC = "2026_06_09_15_46_55"        # 47 kp; val=2, train=18
# a NORMAL recording with existing cse+masks, for the deterministic SIMULATED no-phantom test:
NORM_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
NORM_REC = "2026_03_18_15_31_22"
NORM_IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"
```

**Verified anatomy facts** (transcribed from the real model + mesh + anatomy yaml — do not re-derive, use directly):

- Model `nq=93`, `nkey=1`; `key_qpos[0]` is all-zero for every hinge → the rest value to clamp locked joints to is `mujoco.MjModel.from_xml_path(XML).key_qpos[0]` (all leg/wing hinges rest at 0.0).
- Canonical 50 kp order == `stac-mjx/configs/anatomy/v1.yaml` `model.KEYPOINT_MODEL_PAIRS` keys: `Scutellum`=0, `WingL_base`=1, `WingR_base`=2, `Antenna_Base`=3, `EyeL`=4, `EyeR`=5, `WingL_V12`=6, `WingL_V13`=7, `WingR_V12`=8, `WingR_V13`=9, `Abd_A4`=10, `Abd_tip`=11, `T1L_*`=12–18 (ThxCx,Tro,FeTi,TiTa,TaT1,TaT3,TaTip), `T1R_*`=19–25, `T2L_*`=26–31, `T2R_*`=32–37, `T3L_*`=38–43, `T3R_*`=44–49.
- Joint→qpos: free root 0–6; `wing_{yaw,roll,pitch}_left`=7,8,9; `..._right`=10,11,12; abdomen 13–26; **T1L 27–37, T1R 38–48**, T2L 49–59, T2R 60–70, T3L 71–81, T3R 82–92. **Head has NO joints** (rigid weld to thorax). Each leg = 11 hinges.
- Mesh `seg_ids`/`seg_names`: thorax=1, **head=2** (+ subparts rostrum=3, haustellum=4, labrum_left=5, labrum_right=6, antenna_left=7, antenna_right=8), **wing_left=9, wing_right=10**, abdomen 11–17, **T1L legs seg 20–27** (coxa,femur,tibia,tarsus,tarsus2,tarsus3,tarsus4,claw), **T1R legs seg 28–35**, T2L 36–43, T2R 44–51, T3L 52–59, T3R 60–67.
- **`_STAC_WING_KP_IDX = {"left":(6,7),"right":(8,9)}` is only valid for the FULL 50 schema.** Headless drops kp 3,4,5 → the wing kps shift to STAC indices 3,4,5,6; amputee drops the tail T1R distal → indices before the wings are unchanged but the total shrinks. This is why the wing indices must be looked up by NAME (Task 4).
- amputee `S8_male_R_amp` coco (val + train) both have exactly 44 `keypoint_names`, missing `T1R_Tro,T1R_FeTi,T1R_TiTa,T1R_TaT1,T1R_TaT3,T1R_TaTip` (`T1R_ThxCx` KEPT). headless `headless_22_50` coco both 47 names, missing `Antenna_Base,EyeL,EyeR`.

---

## Task 1: `active_parts.py` PART_TABLE + `derive_active_parts`

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/active_parts.py`
- Test: `third_party/jarvis_jax/tests/test_active_parts.py`

**Interfaces:**
- Consumes: nothing external (pure Python + the fixed canonical schema). `CANONICAL_KP_NAMES: list[str]` (the 50 STAC names, above) is the single source of truth for the diff.
- Produces:
  - `CANONICAL_KP_NAMES: list[str]` — the canonical 50 keypoint names, in STAC/`KEYPOINT_MODEL_PAIRS` order.
  - `PART_TABLE: dict[str, dict]` — part name → `{"joints": list[str] (joint NAMES), "kp_names": list[str], "seg_ids": list[int]}`. Parts: `"head"`, `"wing_left"`, `"wing_right"`, `"legT1L"`,`"legT1R"`,`"legT2L"`,`"legT2R"`,`"legT3L"`,`"legT3R"`. (Root/thorax/abdomen are never toggled off in Phase 4 and are intentionally absent.) Joint NAMES are stored (not qpos indices) so the table is anatomy-agnostic; qpos indices are resolved against the real model in Task 2.
  - `derive_active_parts(kp_names: list[str], override: list[str] | None = None) -> dict[str, list[str]]` — returns `{"off": [part,...], "on": [part,...]}`. If `override` is given, `off = override` (validated against `PART_TABLE` keys). Otherwise: a part is "off" iff ALL of its `kp_names` that are also in `CANONICAL_KP_NAMES` are ABSENT from `kp_names`. (For a leg, "off" requires every one of its distal markers missing; `T1R_ThxCx` present but all T1R distal missing → `legT1R` off — because `legT1R.kp_names` for the diff excludes ThxCx, see note.) `on` = all `PART_TABLE` keys not in `off`.

Note on the amputee ambiguity (verified): T1 legs carry 7 kps incl. `T{n}{L/R}_ThxCx`; T2/T3 carry 6 (no ThxCx). The amputee keeps `T1R_ThxCx` (kp 19) but drops the 6 distal T1R markers. So `PART_TABLE["legT1R"]["kp_names"]` used for the OFF-diff is the 6 DISTAL names only (`T1R_Tro..T1R_TaTip`) — ThxCx is deliberately excluded from the diff set so the derive fires correctly, while `PART_TABLE["legT1R"]["joints"]` still locks ALL 11 T1R hinges (the ThxCx marker legitimately constrains the coxa root; locking is about the meaningless distal joint angles).

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_active_parts.py`:

```python
import os
import numpy as np
import pytest

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"

# The three real schemas (transcribed from the real coco keypoint_names):
FULL_50 = [
    "Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "EyeL", "EyeR",
    "WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13", "Abd_A4", "Abd_tip",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
]
AMPUTEE_44 = [n for n in FULL_50 if n not in
              ("T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip")]
HEADLESS_47 = [n for n in FULL_50 if n not in ("Antenna_Base", "EyeL", "EyeR")]


def test_canonical_kp_names_match_stac_order():
    from jarvis_jax.tracking.active_parts import CANONICAL_KP_NAMES
    assert CANONICAL_KP_NAMES == FULL_50
    assert len(CANONICAL_KP_NAMES) == 50


def test_part_table_shape_and_names():
    from jarvis_jax.tracking.active_parts import PART_TABLE
    assert set(PART_TABLE) == {"head", "wing_left", "wing_right",
                               "legT1L", "legT1R", "legT2L", "legT2R", "legT3L", "legT3R"}
    # head has kps but NO joints (rigid weld) and its own mesh segs.
    assert PART_TABLE["head"]["joints"] == []
    assert set(PART_TABLE["head"]["kp_names"]) == {"Antenna_Base", "EyeL", "EyeR"}
    assert set(PART_TABLE["head"]["seg_ids"]) == {2, 3, 4, 5, 6, 7, 8}
    # legT1R: 11 hinge joints, 6 DISTAL kp names (ThxCx excluded from the diff set), 8 segs.
    assert len(PART_TABLE["legT1R"]["joints"]) == 11
    assert "T1R_ThxCx" not in PART_TABLE["legT1R"]["kp_names"]
    assert set(PART_TABLE["legT1R"]["kp_names"]) == {
        "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip"}
    assert set(PART_TABLE["legT1R"]["seg_ids"]) == set(range(28, 36))
    # wings: 3 joints (yaw/roll/pitch), 2 kps, 1 seg.
    assert len(PART_TABLE["wing_left"]["joints"]) == 3
    assert set(PART_TABLE["wing_left"]["kp_names"]) == {"WingL_V12", "WingL_V13"}
    assert PART_TABLE["wing_left"]["seg_ids"] == [9]
    assert PART_TABLE["wing_right"]["seg_ids"] == [10]


def test_derive_active_parts_amputee_headless_full():
    from jarvis_jax.tracking.active_parts import derive_active_parts
    amp = derive_active_parts(AMPUTEE_44)
    assert amp["off"] == ["legT1R"]
    assert "legT1R" not in amp["on"] and "head" in amp["on"]
    hl = derive_active_parts(HEADLESS_47)
    assert hl["off"] == ["head"]
    assert "head" not in hl["on"] and "legT1R" in hl["on"]
    full = derive_active_parts(FULL_50)
    assert full["off"] == []
    assert set(full["on"]) == {"head", "wing_left", "wing_right",
                               "legT1L", "legT1R", "legT2L", "legT2R", "legT3L", "legT3R"}


def test_derive_active_parts_override():
    from jarvis_jax.tracking.active_parts import derive_active_parts
    d = derive_active_parts(FULL_50, override=["wing_left"])
    assert d["off"] == ["wing_left"]
    with pytest.raises(ValueError):
        derive_active_parts(FULL_50, override=["not_a_part"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_active_parts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.tracking.active_parts'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/active_parts.py`:

```python
"""Per-recording active-parts config + mask (Phase 4: headless / amputation).

ONE anatomy per body type (V1 fruitfly_v1_free.xml, unchanged); a per-recording
active-parts mask expresses a missing body part by (a) dropping that part's
markers, (b) locking + post-solve-clamping that part's joints to rest, and (c)
excluding that part's mesh geoms from the silhouette. The mask is applied in the
cse layer only -- the stac_core_jaxls solver / run_stac_bout / run_single_fly are
reused unchanged. Parts are keyed by NAME so the table is anatomy-agnostic; joint
NAMES (not qpos indices) are stored and resolved against the real model at
build_active_mask time. Index by joint NAME (tarsus5_T1_right moves body
claw_T1_right -- a body/joint name mismatch that would break body-name indexing).
"""
from __future__ import annotations

# Canonical 50 keypoint names in STAC / KEYPOINT_MODEL_PAIRS order (verified
# against stac-mjx/configs/anatomy/v1.yaml). Single source of truth for the diff.
CANONICAL_KP_NAMES = [
    "Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "EyeL", "EyeR",
    "WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13", "Abd_A4", "Abd_tip",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
]


def _leg_joints(prefix, side):
    """The 11 hinge joint NAMES of one leg (verified against fruitfly_v1_free.xml).

    prefix in {T1,T2,T3}; side in {left,right}. The distal tip joint is
    tarsus5_{prefix}_{side} (it moves body claw_{prefix}_{side}).
    """
    return [
        f"coxa_abduct_{prefix}_{side}", f"coxa_twist_{prefix}_{side}", f"coxa_{prefix}_{side}",
        f"femur_twist_{prefix}_{side}", f"femur_{prefix}_{side}", f"tibia_{prefix}_{side}",
        f"tarsus_{prefix}_{side}", f"tarsus2_{prefix}_{side}", f"tarsus3_{prefix}_{side}",
        f"tarsus4_{prefix}_{side}", f"tarsus5_{prefix}_{side}",
    ]


def _leg_kps(prefix, sideR):
    """The 6 DISTAL kp names of a leg (ThxCx excluded from the OFF-diff set)."""
    s = "R" if sideR else "L"
    return [f"{prefix}{s}_{suf}" for suf in ("Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip")]


def _leg_segs(base):
    """8 mesh seg ids of a leg: coxa,femur,tibia,tarsus,tarsus2,tarsus3,tarsus4,claw."""
    return list(range(base, base + 8))


PART_TABLE = {
    "head": {"joints": [], "kp_names": ["Antenna_Base", "EyeL", "EyeR"],
             "seg_ids": [2, 3, 4, 5, 6, 7, 8]},
    "wing_left": {"joints": ["wing_yaw_left", "wing_roll_left", "wing_pitch_left"],
                  "kp_names": ["WingL_V12", "WingL_V13"], "seg_ids": [9]},
    "wing_right": {"joints": ["wing_yaw_right", "wing_roll_right", "wing_pitch_right"],
                   "kp_names": ["WingR_V12", "WingR_V13"], "seg_ids": [10]},
    "legT1L": {"joints": _leg_joints("T1", "left"), "kp_names": _leg_kps("T1", False),
               "seg_ids": _leg_segs(20)},
    "legT1R": {"joints": _leg_joints("T1", "right"), "kp_names": _leg_kps("T1", True),
               "seg_ids": _leg_segs(28)},
    "legT2L": {"joints": _leg_joints("T2", "left"), "kp_names": _leg_kps("T2", False),
               "seg_ids": _leg_segs(36)},
    "legT2R": {"joints": _leg_joints("T2", "right"), "kp_names": _leg_kps("T2", True),
               "seg_ids": _leg_segs(44)},
    "legT3L": {"joints": _leg_joints("T3", "left"), "kp_names": _leg_kps("T3", False),
               "seg_ids": _leg_segs(52)},
    "legT3R": {"joints": _leg_joints("T3", "right"), "kp_names": _leg_kps("T3", True),
               "seg_ids": _leg_segs(60)},
}


def derive_active_parts(kp_names, override=None):
    """Auto-derive off/on parts from a recording's coco keypoint_names.

    A part is OFF iff EVERY one of its (canonical) diff kp_names is absent from
    kp_names. With override given, off = override (validated). See Interfaces.
    """
    present = set(kp_names)
    if override is not None:
        bad = [p for p in override if p not in PART_TABLE]
        if bad:
            raise ValueError(f"override parts not in PART_TABLE: {bad}")
        off = list(override)
    else:
        off = []
        for part, spec in PART_TABLE.items():
            diff_kps = [n for n in spec["kp_names"] if n in CANONICAL_KP_NAMES]
            if diff_kps and all(n not in present for n in diff_kps):
                off.append(part)
    on = [p for p in PART_TABLE if p not in off]
    return {"off": off, "on": on}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_active_parts.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/active_parts.py third_party/jarvis_jax/tests/test_active_parts.py
git commit -m "feat(cse): active-parts PART_TABLE + derive_active_parts (schema-diff auto-derive)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `build_active_mask` + `clamp_locked_qpos` (against the real model/mesh)

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/active_parts.py` (add `build_active_mask`, `clamp_locked_qpos`, `apply_active_mask_to_inputs`)
- Test: `third_party/jarvis_jax/tests/test_active_parts.py` (add real-model/mesh mask tests)

**Interfaces:**
- Consumes: Task 1 `PART_TABLE`, `derive_active_parts`; `mujoco.MjModel.from_xml_path(model_xml)` (`mj_name2id(m, mjOBJ_JOINT, name)`, `jnt_qposadr`, `nq`, `key_qpos[0]`); `numpy.load(mesh_npz)["vertex_segment"]` (len 61666) + `["seg_ids"]`.
- Produces:
  - `build_active_mask(kp_names: list[str], model_xml: str, mesh_npz: str, off_parts: list[str]) -> dict` — resolves the off-part set against the real model+mesh and returns a mask dict with EXACT keys:
    - `"kp_names"`: `list[str]` — the recording's kp_names (echoed, so consumers know the marker order the masks index).
    - `"kps_to_opt_mask"`: `np.ndarray (len(kp_names)*3,) float32` — 1.0 everywhere, 0.0 on the 3 coords of every off-part marker that IS present in `kp_names` (for a leg amputation the distal markers are simply absent from `kp_names`, so this is mostly a no-op; it matters when a part is toggled off via override while its markers are still present, e.g. the SIMULATED test).
    - `"qs_to_opt_mask"`: `np.ndarray (nq,) bool` — True everywhere, False on the qpos indices of every off-part joint (resolved by joint NAME via `jnt_qposadr`). Root qpos 0–6 always True.
    - `"locked_qpos_idx"`: `np.ndarray (n_locked,) int32` — the sorted qpos indices set False above.
    - `"rest_qpos"`: `np.ndarray (nq,) float64` — `model.key_qpos[0]` (the rest pose; all hinges 0.0). The clamp target.
    - `"excluded_seg_ids"`: `list[int]` — union of every off-part's `seg_ids`.
    - `"excluded_vertex_idx"`: `np.ndarray (n_excluded,) int64` — indices into the full 61666-vertex array whose `vertex_segment` is in `excluded_seg_ids`.
    - `"off_parts"`: `list[str]` (echoed).
  - `clamp_locked_qpos(qpos: np.ndarray, mask: dict) -> np.ndarray` — returns a COPY of `qpos (T,nq)` with `qpos[:, mask["locked_qpos_idx"]] = mask["rest_qpos"][mask["locked_qpos_idx"]]` (the post-solve clamp that removes the phantom drift). No-op (returns an equal copy) when `locked_qpos_idx` is empty.
  - `apply_active_mask_to_inputs(inputs: dict, mask: dict) -> dict` — returns a shallow COPY of a `build_solver_inputs`-style dict with `kps_to_opt` multiplied by `mask["kps_to_opt_mask"]` and `qs_to_opt` AND-ed with `mask["qs_to_opt_mask"]` (never touches root 0–6). Also NaNs `kp_data[:, off-marker, :]` for off-part markers that are present in the STAC kp order (belt-and-suspenders with the weight zeroing; the marker_cost drops a coord on either kps_to_opt=0 OR NaN). Used by Task 4 pre-solve.

- [ ] **Step 1: Write the failing test**

Add to `third_party/jarvis_jax/tests/test_active_parts.py`:

```python
_HAVE_MODEL = os.path.exists(XML) and os.path.exists(MESH)


@pytest.mark.skipif(not _HAVE_MODEL, reason="V1 model / mesh not present")
def test_build_active_mask_amputee_locks_t1r_38_48():
    from jarvis_jax.tracking.active_parts import build_active_mask, derive_active_parts
    off = derive_active_parts(AMPUTEE_44)["off"]         # ["legT1R"]
    mask = build_active_mask(AMPUTEE_44, XML, MESH, off)
    # T1R hinges are qpos 38..48 inclusive (11 DOF).
    assert set(mask["locked_qpos_idx"].tolist()) == set(range(38, 49))
    assert mask["qs_to_opt_mask"].shape == (93,)
    assert not mask["qs_to_opt_mask"][38:49].any()       # all locked
    assert mask["qs_to_opt_mask"][:38].all() and mask["qs_to_opt_mask"][49:].all()
    assert mask["qs_to_opt_mask"][:7].all()              # root never locked
    # excluded mesh segs are exactly T1R's 8 leg segs (28..35).
    assert set(mask["excluded_seg_ids"]) == set(range(28, 36))
    # excluded vertex set == vertices whose segment is a T1R seg.
    seg = np.load(MESH, allow_pickle=True)["vertex_segment"]
    expect = np.where(np.isin(seg, list(range(28, 36))))[0]
    assert np.array_equal(np.sort(mask["excluded_vertex_idx"]), np.sort(expect))
    assert len(mask["excluded_vertex_idx"]) > 0
    # rest_qpos is the model rest (all hinges 0).
    assert np.allclose(mask["rest_qpos"][38:49], 0.0)


@pytest.mark.skipif(not _HAVE_MODEL, reason="V1 model / mesh not present")
def test_build_active_mask_headless_no_joints_but_geoms_excluded():
    from jarvis_jax.tracking.active_parts import build_active_mask, derive_active_parts
    off = derive_active_parts(HEADLESS_47)["off"]        # ["head"]
    mask = build_active_mask(HEADLESS_47, XML, MESH, off)
    # head has NO joints -> nothing locked.
    assert mask["locked_qpos_idx"].size == 0
    assert mask["qs_to_opt_mask"].all()
    # but head-region geoms (segs 2..8) ARE excluded.
    assert set(mask["excluded_seg_ids"]) == {2, 3, 4, 5, 6, 7, 8}
    seg = np.load(MESH, allow_pickle=True)["vertex_segment"]
    expect = np.where(np.isin(seg, [2, 3, 4, 5, 6, 7, 8]))[0]
    assert np.array_equal(np.sort(mask["excluded_vertex_idx"]), np.sort(expect))


@pytest.mark.skipif(not _HAVE_MODEL, reason="V1 model / mesh not present")
def test_build_active_mask_full_schema_is_identity():
    from jarvis_jax.tracking.active_parts import build_active_mask
    mask = build_active_mask(FULL_50, XML, MESH, [])
    assert mask["qs_to_opt_mask"].all()
    assert mask["locked_qpos_idx"].size == 0
    assert np.allclose(mask["kps_to_opt_mask"], 1.0)
    assert mask["excluded_seg_ids"] == [] and mask["excluded_vertex_idx"].size == 0


@pytest.mark.skipif(not _HAVE_MODEL, reason="V1 model / mesh not present")
def test_build_active_mask_zeroes_present_off_markers_via_override():
    # SIMULATED case: full 50-name schema but legT1R toggled off by override ->
    # its distal markers ARE present, so their kps_to_opt coords must be zeroed.
    from jarvis_jax.tracking.active_parts import build_active_mask, CANONICAL_KP_NAMES
    mask = build_active_mask(FULL_50, XML, MESH, ["legT1R"])
    for nm in ("T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip"):
        j = FULL_50.index(nm)
        assert np.allclose(mask["kps_to_opt_mask"][j * 3:j * 3 + 3], 0.0)
    # T1R_ThxCx (present, kept -- constrains the coxa root) NOT zeroed.
    j = FULL_50.index("T1R_ThxCx")
    assert np.allclose(mask["kps_to_opt_mask"][j * 3:j * 3 + 3], 1.0)


def test_clamp_locked_qpos_pins_to_rest():
    from jarvis_jax.tracking.active_parts import clamp_locked_qpos
    mask = {"locked_qpos_idx": np.array([38, 39, 48], np.int32),
            "rest_qpos": np.zeros(93)}
    q = np.random.default_rng(0).normal(size=(5, 93))
    q2 = clamp_locked_qpos(q, mask)
    assert np.allclose(q2[:, [38, 39, 48]], 0.0)          # locked -> rest
    keep = [i for i in range(93) if i not in (38, 39, 48)]
    assert np.allclose(q2[:, keep], q[:, keep])           # everything else untouched
    assert not np.shares_memory(q2, q)                    # returns a copy


def test_apply_active_mask_to_inputs_gates_weights_and_dofs():
    from jarvis_jax.tracking.active_parts import apply_active_mask_to_inputs
    inp = {"kps_to_opt": np.ones(6, np.float32), "qs_to_opt": np.ones(4, bool),
           "kp_data": np.ones((3, 2, 3), np.float32)}
    mask = {"kps_to_opt_mask": np.array([1, 1, 1, 0, 0, 0], np.float32),
            "qs_to_opt_mask": np.array([True, True, False, True]),
            "off_marker_stac_idx": np.array([1])}
    out = apply_active_mask_to_inputs(inp, mask)
    assert np.allclose(out["kps_to_opt"], [1, 1, 1, 0, 0, 0])
    assert out["qs_to_opt"].tolist() == [True, True, False, True]
    assert np.isnan(out["kp_data"][:, 1, :]).all()        # off marker NaN'd
    assert np.isfinite(out["kp_data"][:, 0, :]).all()     # present marker kept
    assert not np.shares_memory(out["kps_to_opt"], inp["kps_to_opt"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_active_parts.py -k "mask or clamp or apply" -v`
Expected: FAIL with `ImportError: cannot import name 'build_active_mask'` (and `clamp_locked_qpos`/`apply_active_mask_to_inputs`).

- [ ] **Step 3: Write minimal implementation**

Add to `third_party/jarvis_jax/jarvis_jax/cse/active_parts.py`:

```python
import numpy as np


def build_active_mask(kp_names, model_xml, mesh_npz, off_parts):
    """Resolve an off-part set against the real model+mesh into a mask dict.

    See Interfaces (Task 2). Marker order for kps_to_opt_mask/off_marker_stac_idx
    follows the passed kp_names (the STAC/bout kp order the mask will be applied
    against). Joints are resolved by NAME via mj_name2id, so a body/joint name
    mismatch (tarsus5_T1_right -> body claw_T1_right) is handled correctly.
    """
    import mujoco

    kp_names = list(kp_names)
    name2idx = {n: i for i, n in enumerate(kp_names)}

    m = mujoco.MjModel.from_xml_path(model_xml)
    nq = int(m.nq)
    rest_qpos = np.asarray(m.key_qpos[0], dtype=np.float64) if m.nkey > 0 else np.zeros(nq)

    qs_to_opt_mask = np.ones(nq, dtype=bool)
    locked = []
    kps_to_opt_mask = np.ones(len(kp_names) * 3, dtype=np.float32)
    off_marker_stac_idx = []
    excluded_seg_ids = []

    for part in off_parts:
        spec = PART_TABLE[part]
        # (b) lock the part's joints (by joint NAME -> qpos adr).
        for jname in spec["joints"]:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"joint '{jname}' (part {part}) not found in {model_xml}")
            adr = int(m.jnt_qposadr[jid])
            qs_to_opt_mask[adr] = False
            locked.append(adr)
        # (a) zero the part's markers that are actually PRESENT in kp_names.
        for kn in spec["kp_names"]:
            j = name2idx.get(kn)
            if j is not None:
                kps_to_opt_mask[j * 3:j * 3 + 3] = 0.0
                off_marker_stac_idx.append(j)
        # (c) collect the part's mesh seg ids.
        excluded_seg_ids.extend(spec["seg_ids"])

    # never lock the free root (qpos 0..6).
    qs_to_opt_mask[:7] = True
    locked = sorted(set(i for i in locked if i >= 7))

    excluded_seg_ids = sorted(set(excluded_seg_ids))
    if excluded_seg_ids:
        seg = np.load(mesh_npz, allow_pickle=True)["vertex_segment"]
        excluded_vertex_idx = np.where(np.isin(seg, excluded_seg_ids))[0].astype(np.int64)
    else:
        excluded_vertex_idx = np.zeros(0, dtype=np.int64)

    return {
        "kp_names": kp_names,
        "kps_to_opt_mask": kps_to_opt_mask,
        "qs_to_opt_mask": qs_to_opt_mask,
        "locked_qpos_idx": np.asarray(locked, dtype=np.int32),
        "rest_qpos": rest_qpos,
        "excluded_seg_ids": excluded_seg_ids,
        "excluded_vertex_idx": excluded_vertex_idx,
        "off_marker_stac_idx": np.asarray(sorted(set(off_marker_stac_idx)), dtype=np.int64),
        "off_parts": list(off_parts),
    }


def clamp_locked_qpos(qpos, mask):
    """Post-solve clamp: pin every locked joint's qpos back to rest (removes the
    phantom LM-variable drift). Returns a copy; no-op when nothing is locked."""
    q = np.array(qpos, copy=True)
    idx = np.asarray(mask["locked_qpos_idx"])
    if idx.size:
        q[:, idx] = np.asarray(mask["rest_qpos"])[idx]
    return q


def apply_active_mask_to_inputs(inputs, mask):
    """Pre-solve: gate marker weights + DOFs and NaN off-part markers. Copy-safe."""
    out = dict(inputs)
    out["kps_to_opt"] = (np.asarray(inputs["kps_to_opt"], np.float32)
                         * np.asarray(mask["kps_to_opt_mask"], np.float32))
    out["qs_to_opt"] = np.asarray(inputs["qs_to_opt"], bool) & np.asarray(mask["qs_to_opt_mask"], bool)
    off = np.asarray(mask.get("off_marker_stac_idx", np.zeros(0, np.int64)))
    if off.size and "kp_data" in inputs:
        kp = np.array(inputs["kp_data"], dtype=np.float64, copy=True)
        kp[:, off, :] = np.nan
        out["kp_data"] = kp
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_active_parts.py -v`
Expected: PASS (all Task-1 + Task-2 tests; 10 passed). The model/mesh-gated tests run because both files exist (verified on disk).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/active_parts.py third_party/jarvis_jax/tests/test_active_parts.py
git commit -m "feat(cse): build_active_mask + clamp_locked_qpos + apply_active_mask_to_inputs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Reduced-schema patch to `cse_labels.build_bout` (prerequisite for T6)

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/cse_labels.py` (`_reorder_index`, `triangulate_recording`, `build_bout`)
- Test: `third_party/jarvis_jax/tests/test_cse_labels_reduced_schema.py` **(new)**

**Interfaces:**
- Consumes: `coco["keypoint_names"]` (44 for amputee / 47 for headless / 50 full); `model_kp_order(anatomy_yaml) -> list[str]` (the canonical 50, unchanged).
- Produces (changed behavior, backward compatible):
  - `_reorder_index(src_names: list[str], dst_names: list[str]) -> np.ndarray` — UNCHANGED signature/semantics, but callers now pass a `dst_names` already intersected with `src_names` (so no absent name is looked up). Still `src[idx] == dst`.
  - `triangulate_recording(coco, rec, rt, kp_order, id2img, id2ann, by_rec) -> (kp3d (n_fs,K',3), vis (n_fs,K'), fs_keys, fs_imgids)` — now uses `kp_present = [n for n in kp_order if n in set(coco["keypoint_names"])]` (model order, intersected), builds `reorder` against `kp_present`, and returns `K' = len(kp_present)` columns (44/47/50). Also returns/echoes `kp_present` so `build_bout` writes the correct names (add `kp_present` as a 5th return value).
  - `build_bout(coco_path, calib_root, rec, anatomy_yaml, model_xml, out_h5) -> (out_h5, s)` — writes `kp_names` = the intersected `kp_present` (44/47/50) instead of the full 50, so the bout h5 is self-describing and `run_stac_bout`'s `prune_model_to_available` sees a consistent reduced set. `model_rest_keypoints`/`umeyama_scale` operate on the intersected order. Full-schema path is byte-for-byte unchanged (intersection == full when nothing is missing).

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_cse_labels_reduced_schema.py`:

```python
import os
import h5py
import numpy as np
import pytest

GM = "/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
ANATOMY = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs/anatomy/v1.yaml"
AMP = f"{GM}/S8_male_R_amp"
AMP_REC = "2026_02_13_13_44_49"
AMP_COCO = f"{AMP}/annotations/instances_train.json"       # train=174 framesets
AMP_CALIB = f"{AMP}/calib_params"

_HAVE = all(os.path.exists(p) for p in (AMP_COCO, XML, ANATOMY))


def test_reorder_index_intersection_no_keyerror():
    """The core blocker: _reorder_index must be fed a dst intersected with src,
    so an absent canonical name never triggers pos[n] KeyError."""
    from jarvis_jax.densepose.cse_labels import _reorder_index
    src = ["A", "B", "D"]                 # recording (missing C)
    dst_full = ["A", "B", "C", "D"]       # model order
    # intersecting first is the required pattern:
    dst = [n for n in dst_full if n in set(src)]
    idx = _reorder_index(src, dst)
    assert [src[i] for i in idx] == dst   # src[idx] == dst
    # and the OLD unguarded call would KeyError -- assert we no longer do that path
    with pytest.raises(KeyError):
        _reorder_index(src, dst_full)     # documents why the intersection is needed


@pytest.mark.skipif(not _HAVE, reason="amputee coco / model / anatomy not present")
def test_build_bout_amputee_writes_44_kp_names(tmp_path):
    from jarvis_jax.densepose.cse_labels import build_bout
    out = str(tmp_path / f"{AMP_REC}_bout.h5")
    p, s = build_bout(AMP_COCO, AMP_CALIB, AMP_REC, ANATOMY, XML, out)
    assert os.path.exists(p)
    with h5py.File(p, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f["kp_names"][()]]
        assert len(names) == 44, f"expected 44 kp_names, got {len(names)}"
        # the 6 T1R distal markers are absent; T1R_ThxCx kept.
        for miss in ("T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip"):
            assert miss not in names
        assert "T1R_ThxCx" in names
        assert f["keypoints"].shape[1] == 44
        assert f["vis"].shape[1] == 44
        assert f["keypoints"].shape[0] == f["fs_imgids"].shape[0] > 0
    assert np.isfinite(s) and s > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_cse_labels_reduced_schema.py -v`
Expected: FAIL — `test_build_bout_amputee_writes_44_kp_names` raises `KeyError` from `_reorder_index` (the current `build_bout` calls `_reorder_index(src_names, kp_order)` on the full 50 `kp_order`, and the amputee `src_names` lacks the 6 T1R distal names → `pos[n]` KeyError). `test_reorder_index_intersection_no_keyerror` passes already (it documents the pattern).

- [ ] **Step 3: Write minimal implementation**

In `third_party/jarvis_jax/jarvis_jax/cse/cse_labels.py`, change `triangulate_recording` to intersect the model order with the recording's present names before building the reorder, and return the intersected names. Replace the top of `triangulate_recording` (currently lines ~104–108):

```python
    src_names = coco["keypoint_names"]
    src_set = set(src_names)
    # Reduced-schema (amputation/headless): the recording's coco keypoint_names
    # OMITS the missing part's names. Intersect the model order with the present
    # names BEFORE building the reorder so _reorder_index never looks up an
    # absent name (KeyError). Mirrors stac_mjx.keypoint_prune.prune_model_to_
    # available's present/dropped pattern; a no-op when every model kp is present.
    kp_present = [n for n in kp_order if n in src_set]
    dropped = [n for n in kp_order if n not in src_set]
    if dropped:
        print(f"  [build_bout] {len(kp_present)}/{len(kp_order)} model keypoints present; "
              f"dropped absent from recording ({len(dropped)}): {dropped}")
    reorder = _reorder_index(src_names, kp_present)  # dst(present model) -> src
    K = len(kp_present)
```

and change its `return` (line ~127) to also return the present names:

```python
    return np.asarray(kp3d), np.asarray(vis), fs_keys, fs_imgids, kp_present
```

Then update `build_bout` (line ~140) to consume the 5th return value and write the reduced names. Replace the `triangulate_recording(...)` call + the `rest`/`kp_names` write:

```python
    kp3d, vis, fs_keys, fs_imgids, kp_present = triangulate_recording(
        coco, rec, rt, kp_order, id2img, id2ann, by_rec)
    if len(kp3d) == 0:
        raise RuntimeError(f"no complete framesets for {rec}")

    model = mujoco.MjModel.from_xml_path(model_xml)
    rest = model_rest_keypoints(model, kp_present)       # intersected order
```

and change the h5 `kp_names` dataset (line ~156) to `np.array(kp_present, dtype="S20")`. (The `keypoints`/`vis`/`scale`/`fs_*` writes are unchanged; they now carry `K'` columns.)

Note the `_reorder_index` body itself is left unchanged — the fix is that every caller now passes an intersected `dst`. `_reorder_index`'s existing `pos[n]` will still `KeyError` if mis-called on a non-intersected list, which the first unit test documents on purpose.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_cse_labels_reduced_schema.py -v`
Expected: PASS (2 passed).

Then run a full-schema regression to prove the normal path is unchanged (this is the existing Phase-2/3 recording; CPU-safe):

```bash
cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -c "
from jarvis_jax.densepose.cse_labels import build_bout
import h5py
p, s = build_bout(
  '/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3/annotations/instances_val.json',
  '/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3/calib_params',
  '2026_03_18_15_31_22',
  '/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs/anatomy/v1.yaml',
  '/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml',
  '/tmp/claude-398823/regress_bout.h5')
with h5py.File(p) as f:
  n=[x.decode() if isinstance(x,bytes) else x for x in f['kp_names'][()]]
  assert len(n)==50, len(n); print('OK full-schema unchanged: 50 kp_names, scale', round(s,4))
"
```
Expected: `OK full-schema unchanged: 50 kp_names, scale ...`. (Delete `/tmp/claude-398823/regress_bout.h5` after; do NOT commit it.)

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/cse_labels.py third_party/jarvis_jax/tests/test_cse_labels_reduced_schema.py
git commit -m "fix(cse): reduced-schema build_bout (intersect model kp order with present names)

Amputee/headless coco omit the missing part's keypoint_names; _reorder_index
KeyErrored on the absent canonical names. Intersect kp_order with the recording's
present names before building the reorder and write the shorter kp_names into the
bout h5 (mirrors keypoint_prune.prune_model_to_available). Full-schema unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Name-based wing augment/withhold + mask threading into `run_single_fly` + SIMULATED no-phantom test

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py` (`_stac_wing_idx` name-based helper; `_augment_wing_markers_stac_order`, `_withhold_wing_kp` made name-based; `run_single_fly`/`run_ablation` gain an `active_parts` kwarg + apply mask pre-solve + `clamp_locked_qpos` post-solve)
- Test: `third_party/jarvis_jax/tests/test_silhouette_ik_solve.py` (add name-based-wing + mask-threading + SIMULATED no-phantom tests)

**Interfaces:**
- Consumes: Task 2 `build_active_mask(kp_names, model_xml, mesh_npz, off_parts)`, `derive_active_parts(kp_names, override)`, `apply_active_mask_to_inputs(inputs, mask)`, `clamp_locked_qpos(qpos, mask)`.
- Produces (new/changed):
  - `_stac_wing_idx(kp_names: list[str]) -> dict[str, tuple[int, int]]` — returns `{"left": (i_V12, i_V13), "right": (i_V12, i_V13)}` by looking up `WingL_V12/V13`, `WingR_V12/V13` by NAME in `kp_names` (the STAC/bout kp order), NOT the hardcoded `(6,7)/(8,9)`. Raises if a wing name is missing (wings are never off in Phase 4). Correct under any reduced schema (headless shifts wings to 3,4,5,6).
  - `_withhold_wing_kp(kp_data: np.ndarray, kp_names: list[str]) -> np.ndarray` — NEW required 2nd arg `kp_names`; NaNs the wing rows resolved by `_stac_wing_idx(kp_names)` instead of the module-constant `_STAC_WING_KP_IDX`. (Update `run_ablation`'s call site to pass `kp_names`.)
  - `_augment_wing_markers_stac_order(...)` — UNCHANGED public behavior; internally its coco-slot bridge already looks up by NAME (`_COCO_KEYPOINT_NAMES` + `stac_idx_of_coco_slot`), so it already tolerates a reduced STAC kp order as long as the wing names are present. Add a regression test proving it writes the SHIFTED wing indices under a headless-like reordering.
  - `run_single_fly(..., active_parts: list[str] | dict | None = None) -> dict` — new trailing kwarg. If `active_parts` is a `list[str]` it is treated as the off-part list; if `None` it is AUTO-DERIVED from the recording's coco `keypoint_names` via `derive_active_parts` (so the driver just works on an amputee/headless recording); if it's a dict it is treated as an already-built mask. When any part is off: build the mask, `apply_active_mask_to_inputs` BEFORE `solve_ik`, and `clamp_locked_qpos(qpos, mask)` AFTER. Off markers are excluded from the reprojection metric. Default `None` + full-schema recording → empty off set → behavior byte-for-byte unchanged.
  - `run_ablation(..., active_parts: list[str] | dict | None = None) -> dict` — same threading (mask applied to all three conditions' inputs; clamp applied to all three solved qpos). Kept minimal; T7 uses `run_single_fly`.

- [ ] **Step 1: Write the failing test**

Add to `third_party/jarvis_jax/tests/test_silhouette_ik_solve.py`:

```python
def test_stac_wing_idx_name_based_full_and_shifted():
    from jarvis_jax.tracking.silhouette_ik_solve import _stac_wing_idx, _COCO_KEYPOINT_NAMES
    # full STAC order: WingL_V12=6, WingL_V13=7, WingR_V12=8, WingR_V13=9.
    from jarvis_jax.tracking.active_parts import CANONICAL_KP_NAMES
    idx = _stac_wing_idx(CANONICAL_KP_NAMES)
    assert idx == {"left": (6, 7), "right": (8, 9)}
    # headless-like: drop Antenna_Base/EyeL/EyeR -> wings shift to 3,4,5,6.
    hl = [n for n in CANONICAL_KP_NAMES if n not in ("Antenna_Base", "EyeL", "EyeR")]
    idxh = _stac_wing_idx(hl)
    assert idxh == {"left": (3, 4), "right": (5, 6)}


def test_withhold_wing_kp_name_based_shifted_order():
    import numpy as np
    from jarvis_jax.tracking.silhouette_ik_solve import _withhold_wing_kp
    from jarvis_jax.tracking.active_parts import CANONICAL_KP_NAMES
    hl = [n for n in CANONICAL_KP_NAMES if n not in ("Antenna_Base", "EyeL", "EyeR")]
    kp = np.ones((3, len(hl), 3))
    out = _withhold_wing_kp(kp, hl)
    for j in (3, 4, 5, 6):
        assert np.isnan(out[:, j, :]).all()       # wings withheld at shifted idx
    kept = [j for j in range(len(hl)) if j not in (3, 4, 5, 6)]
    assert np.isfinite(out[:, kept, :]).all()


def test_augment_wing_markers_writes_shifted_indices_under_headless_order():
    """Regression: the coco-slot bridge must fill the SHIFTED STAC wing indices
    (3,4,5,6) under a headless-like kp order, not the full-schema 6,7,8,9."""
    import numpy as np
    from jarvis_jax.tracking.silhouette_ik_solve import _augment_wing_markers_stac_order
    from jarvis_jax.tracking.active_parts import CANONICAL_KP_NAMES
    hl = [n for n in CANONICAL_KP_NAMES if n not in ("Antenna_Base", "EyeL", "EyeR")]
    T = 2
    kp = np.full((T, len(hl), 3), np.nan)         # all missing -> only_missing fills
    w = np.ones(len(hl) * 3)
    tips = [{"left": (np.array([1.0, 2.0, 3.0]), 3), "right": (np.array([4.0, 5.0, 6.0]), 3)}
            for _ in range(T)]
    kp2, w2 = _augment_wing_markers_stac_order(kp, w, tips, hl, wing_weight=0.5, only_missing=True)
    assert np.allclose(kp2[:, 3, :], [1, 2, 3]) and np.allclose(kp2[:, 4, :], [1, 2, 3])  # left
    assert np.allclose(kp2[:, 5, :], [4, 5, 6]) and np.allclose(kp2[:, 6, :], [4, 5, 6])  # right
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_ik_solve.py -k "stac_wing_idx or withhold_wing_kp_name_based or augment_wing_markers_writes_shifted" -v`
Expected: FAIL — `ImportError: cannot import name '_stac_wing_idx'`; `_withhold_wing_kp` currently takes 1 arg (TypeError on the 2-arg call); the shifted-index augment test fails because the current bridge only works when the STAC wing names resolve (it will actually PASS if the bridge is already name-based — if so, keep the test as a guard and note it green).

- [ ] **Step 3: Write minimal implementation**

In `silhouette_ik_solve.py`, add the name-based helper near `_STAC_WING_KP_IDX` (line ~942):

```python
def _stac_wing_idx(kp_names):
    """STAC/bout-order indices of the distal wing markers, resolved by NAME.

    Returns {"left": (WingL_V12_idx, WingL_V13_idx), "right": (WingR_V12_idx,
    WingR_V13_idx)}. Name-based so it is correct under a reduced schema
    (headless drops kp 3,4,5 -> wings shift from 6,7,8,9 to 3,4,5,6). The full
    _STAC_WING_KP_IDX = {"left":(6,7),"right":(8,9)} constant is retained only as
    the documented full-schema reference / for existing full-schema tests.
    """
    kp_names = list(kp_names)
    pos = {n: i for i, n in enumerate(kp_names)}
    try:
        return {"left": (pos["WingL_V12"], pos["WingL_V13"]),
                "right": (pos["WingR_V12"], pos["WingR_V13"])}
    except KeyError as e:
        raise ValueError(f"wing marker {e} absent from kp_names (wings are never "
                         f"an off-part in Phase 4)") from None
```

Change `_withhold_wing_kp` (line ~945) to be name-based:

```python
def _withhold_wing_kp(kp_data: np.ndarray, kp_names) -> np.ndarray:
    """Return a copy of kp_data (T, n_kp, 3, STAC order) with the four distal
    wing marker rows (resolved by NAME via _stac_wing_idx) NaN'd, so the STAC
    marker_cost finite-mask drops them. kp_names required (reduced-schema safe)."""
    kp2 = np.array(kp_data, dtype=np.float64, copy=True)
    widx = _stac_wing_idx(kp_names)
    wing_idx = [i for ids in widx.values() for i in ids]
    kp2[:, wing_idx, :] = np.nan
    return kp2
```

Update `run_ablation`'s call site (line ~1105): `kp_data_withheld = _withhold_wing_kp(kp_data_ref, kp_names)`. Also change `run_ablation`'s `wing_stac_idx` (line ~1154) to `_stac_wing_idx(kp_names)`-derived set instead of the constant.

Thread `active_parts` through `run_single_fly` (signature line ~683 add `active_parts=None`). After `inputs = build_solver_inputs(...)` and the `q_init`/`kp_data`/`kps_to_opt` slicing (line ~775), and after the silhouette augmentation block (line ~803, so the mask is applied to the augmented arrays), build + apply the mask:

```python
    # --- Phase 4: active-parts mask (headless / amputation) ---
    from jarvis_jax.tracking.active_parts import (
        derive_active_parts, build_active_mask, apply_active_mask_to_inputs, clamp_locked_qpos)
    kp_names_stac = list(inputs["kp_names"])
    if active_parts is None:
        off_parts = derive_active_parts(kp_names_stac)["off"]
        active_mask = build_active_mask(kp_names_stac, model_xml, mesh_npz, off_parts) if off_parts else None
    elif isinstance(active_parts, dict):
        active_mask = active_parts            # already-built mask
    else:
        active_mask = build_active_mask(kp_names_stac, model_xml, mesh_npz, list(active_parts))
```

Then build `small` (line ~807) and, if `active_mask` is not None, apply it before solving; wrap the solve + clamp:

```python
    small = dict(inputs)
    small["q_init"] = q_init
    small["kp_data"] = kp_data
    small["kps_to_opt"] = kps_to_opt
    if active_mask is not None:
        small = apply_active_mask_to_inputs(small, active_mask)

    qpos = solve_ik(small, smooth_weight=smooth_weight, n_iter=n_iter)
    if active_mask is not None:
        qpos = clamp_locked_qpos(qpos, active_mask)   # post-solve clamp: no phantom
```

In the reprojection metric loop (line ~880), skip off-part markers so the metric reflects only present markers:

```python
        off_stac = set(active_mask["off_marker_stac_idx"].tolist()) if active_mask is not None else set()
        for j, nm in enumerate(kp_names):
            if j in off_stac:
                continue
            ci = name2coco.get(nm)
            ...
```

Add `active_parts` to the returned report: `off_parts=(active_mask["off_parts"] if active_mask else [])` and `locked_qpos_idx=(active_mask["locked_qpos_idx"].tolist() if active_mask else [])`. Apply the identical threading to `run_ablation` (build the mask once, `apply_active_mask_to_inputs` on `small_a/b/c`, `clamp_locked_qpos` on `qpos_a/b/c`). Keep every default `None`; on a full-schema recording `derive_active_parts` returns `off=[]` → `active_mask=None` → path unchanged.

- [ ] **Step 4: Run the name-based unit tests + the full CPU suite (regression)**

```bash
cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest \
  tests/test_silhouette_ik_solve.py -k "stac_wing_idx or withhold or augment or ann_for_image or triangulate_kp_mm" -v
```
Expected: PASS (name-based tests + the existing Phase-3 CPU threading tests). The existing full-schema `test_withhold_wing_kp_*` tests call `_withhold_wing_kp(kp)` with one arg — UPDATE them in this same step to pass `_COCO_KEYPOINT_NAMES`-derived STAC names (or the anatomy kp_names), since the signature now requires `kp_names`. Grep for the call sites first: `grep -n "_withhold_wing_kp(" tests/test_silhouette_ik_solve.py` and fix each. Re-run until green.

- [ ] **Step 5: The SIMULATED no-phantom test (deterministic dev gate — GPU)**

Add to `third_party/jarvis_jax/tests/test_silhouette_ik_solve.py`:

```python
NORM_IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"


@pytest.mark.skipif(not os.path.exists(NORM_IK), reason="normal recording cse not present")
def test_simulated_no_phantom_t1r_off_on_normal_recording(tmp_path):
    """On the NORMAL recording (full 50-kp cse), simulate T1R-off via override.
    NO-PHANTOM ACCEPTANCE:
      (1) solved off-leg joints (qpos 38..48) stay at rest across ALL frames;
      (2) present-marker reproj_px is within eps of the unmasked run (the mask
          does not perturb the rest of the fly)."""
    from jarvis_jax.tracking.silhouette_ik_solve import run_single_fly
    common = dict(ik_h5=NORM_IK, model_xml=XML, mesh_npz=MESH, root=ROOT,
                  split="val", use_silhouette=False, max_frames=8, n_iter=40,
                  out_dir=str(tmp_path))
    base = run_single_fly(RECORDING, **common)                       # unmasked
    masked = run_single_fly(RECORDING, active_parts=["legT1R"], **common)

    import numpy as np
    q = np.load(os.path.join(str(tmp_path), f"{RECORDING}_qpos.npz"))["qpos"]
    # (1) the RETURNED qpos has T1R (38..48) pinned to rest (0) on every frame.
    assert np.max(np.abs(q[:, 38:49])) < 1e-6, "phantom: T1R joints drifted off rest"
    assert masked["locked_qpos_idx"] == list(range(38, 49))
    assert masked["off_parts"] == ["legT1R"]
    # (2) present-marker reproj essentially unchanged vs the unmasked solve.
    assert np.isfinite(base["reproj_px"]) and np.isfinite(masked["reproj_px"])
    assert abs(masked["reproj_px"] - base["reproj_px"]) < 0.75, (
        f"mask perturbed present-marker fit: base={base['reproj_px']:.3f} "
        f"masked={masked['reproj_px']:.3f} px")
```

Run under the activated env (GPU; `NORM_IK` exists on disk):

```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
cd third_party/jarvis_jax && python -m pytest \
  tests/test_silhouette_ik_solve.py::test_simulated_no_phantom_t1r_off_on_normal_recording -v -s
```
Expected: PASS. `max|q[:,38:49]| < 1e-6` (clamp), and `|Δreproj_px| < 0.75px` (T1R markers were the only thing removed; the rest of the body fit is unaffected because those markers are among 50 and T1R DOFs don't pull other bodies once clamped). If `|Δreproj_px|` exceeds the bound, DEBUG (superpowers:systematic-debugging) — likely a mask being applied to the wrong indices — do NOT loosen the threshold to pass.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py third_party/jarvis_jax/tests/test_silhouette_ik_solve.py
git commit -m "feat(cse): name-based wing indices + active-parts mask threading + post-solve clamp

Wing augment/withhold resolves WingL/R_V12/V13 by name (reduced-schema safe).
run_single_fly/run_ablation accept active_parts (auto-derived): apply the mask
pre-solve and clamp locked joints to rest post-solve (no phantom). Simulated
T1R-off no-phantom gate on the normal recording.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Geom-exclusion in the silhouette path

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_landmarks.py` (`wing_side_vertices` gains optional `exclude_seg_ids`)
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py` (`extract_tips_for_frames` accepts `active_parts`/mask and passes the excluded set through; `run_single_fly` forwards it)
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/active_parts.py` (add `excluded_vertex_indices(mask, mesh_npz)` convenience — already computed in `build_active_mask`; expose a helper for the fps-relative subset)
- Test: `third_party/jarvis_jax/tests/test_active_parts.py` (geom-exclusion vertex-set + coverage checks)

**Interfaces:**
- Consumes: Task 2 `mask["excluded_seg_ids"]`, `mask["excluded_vertex_idx"]`; the mesh npz `vertex_segment` + `fps_300` subset.
- Produces:
  - `wing_side_vertices(mesh_npz, exclude_seg_ids: list[int] | None = None) -> dict` — UNCHANGED default behavior; when `exclude_seg_ids` is given, the fps-subset points whose segment is in `exclude_seg_ids` are removed from consideration before the tip/prox argmax/argmin, so a masked-off part's vertices never become a wing tip/prox. (Wings themselves are never excluded in Phase 4, so this is defensive for head/leg exclusion that could otherwise leak a stray vertex into the thorax-centroid or side selection.)
  - `excluded_fps_indices(mask: dict, mesh_npz: str) -> np.ndarray` (in `active_parts.py`) — the positions within `fps_300` (0..299) whose full-vertex `vertex_segment` is in `mask["excluded_seg_ids"]`. This is the fps-relative form the silhouette landmark path indexes (matching `_wing_fk_indices`' `fps_300[idx]` convention).
  - `extract_tips_for_frames(..., active_parts=None)` — new trailing kwarg (list/dict/None like `run_single_fly`); builds/receives the mask and, for a head/leg-off recording, passes `exclude_seg_ids` into the `wing_side_vertices` call inside `_wing_fk_indices` (thread an `exclude_seg_ids` arg through `_wing_fk_indices`). Since the silhouette landmark in Phase 4 is still the WING tip, and wings are never off, the practical effect is defensive; the test asserts the excluded vertex set is correctly computed and honored.

- [ ] **Step 1: Write the failing test**

Add to `third_party/jarvis_jax/tests/test_active_parts.py`:

```python
@pytest.mark.skipif(not _HAVE_MODEL, reason="V1 model / mesh not present")
def test_excluded_fps_indices_match_seg_ids():
    from jarvis_jax.tracking.active_parts import build_active_mask, excluded_fps_indices, derive_active_parts
    off = derive_active_parts(HEADLESS_47)["off"]        # ["head"]
    mask = build_active_mask(HEADLESS_47, XML, MESH, off)
    fps_ex = excluded_fps_indices(mask, MESH)
    z = np.load(MESH, allow_pickle=True)
    fps = z["fps_300"]
    seg = z["vertex_segment"]
    # every excluded fps position maps to a full-vertex whose seg is head (2..8).
    assert len(fps_ex) > 0
    assert np.isin(seg[fps[fps_ex]], [2, 3, 4, 5, 6, 7, 8]).all()
    # and no NON-excluded fps position is a head vertex.
    keep = np.setdiff1d(np.arange(len(fps)), fps_ex)
    assert not np.isin(seg[fps[keep]], [2, 3, 4, 5, 6, 7, 8]).any()


@pytest.mark.skipif(not _HAVE_MODEL, reason="V1 model / mesh not present")
def test_wing_side_vertices_ignores_excluded_segs():
    from jarvis_jax.tracking.silhouette_landmarks import wing_side_vertices
    base = wing_side_vertices(MESH)
    # excluding a LEG's segs must not change the WING tip/prox selection
    # (wings and that leg are disjoint) -> defensive no-op for wings.
    excl = wing_side_vertices(MESH, exclude_seg_ids=list(range(28, 36)))  # T1R
    assert base == excl
    # the excluded-vertex fps positions are never returned as a wing tip/prox.
    z = np.load(MESH, allow_pickle=True)
    seg = z["vertex_segment"]; fps = z["fps_300"]
    for side in ("left", "right"):
        for k in ("tip", "prox"):
            assert seg[fps[excl[side][k]]] not in list(range(28, 36))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_active_parts.py -k "excluded_fps or wing_side_vertices_ignores" -v`
Expected: FAIL — `ImportError: cannot import name 'excluded_fps_indices'`; `wing_side_vertices()` currently takes no `exclude_seg_ids` (TypeError).

- [ ] **Step 3: Write minimal implementation**

In `active_parts.py` add:

```python
def excluded_fps_indices(mask, mesh_npz):
    """fps_300-relative positions whose full-vertex segment is in the mask's
    excluded_seg_ids (the fps subset the silhouette landmark path indexes)."""
    z = np.load(mesh_npz, allow_pickle=True)
    fps = z["fps_300"] if "fps_300" in z.files else z[f"fps_{len(z['vertex_segment'])}"]
    seg = z["vertex_segment"]
    excl = set(mask["excluded_seg_ids"])
    if not excl:
        return np.zeros(0, dtype=np.int64)
    return np.where(np.isin(seg[fps], list(excl)))[0].astype(np.int64)
```

In `silhouette_landmarks.py`, change `wing_side_vertices` (line 12) to accept `exclude_seg_ids`:

```python
def wing_side_vertices(mesh_npz, exclude_seg_ids=None):
    z = np.load(mesh_npz, allow_pickle=True)
    fps = z["fps_300"] if "fps_300" in z.files else z[f"fps_{len(z['vertex_segment'])}"]
    seg = z["vertex_segment"][fps]; C = z["vertices"][fps]
    id2n = {int(s): (n.decode() if isinstance(n, bytes) else n)
            for s, n in zip(z["seg_ids"], z["seg_names"])}
    names = [id2n[int(s)].lower() for s in seg]
    excl = set(exclude_seg_ids or [])
    keep_mask = np.array([int(s) not in excl for s in seg])
    thorax_c = C[np.array([("thorax" in n) for n in names]) & keep_mask].mean(0)
    out = {}
    for side in ("left", "right"):
        si = np.where(np.array([("wing" in n and side in n) for n in names]) & keep_mask)[0]
        d = np.linalg.norm(C[si] - thorax_c, axis=1)
        out[side] = {"tip": int(si[d.argmax()]), "prox": int(si[d.argmin()])}
    return out
```

In `silhouette_ik_solve.py`, thread `exclude_seg_ids` through `_wing_fk_indices(mesh_npz, exclude_seg_ids=None)` (pass into its `wing_side_vertices(mesh_npz, exclude_seg_ids)` call) and give `extract_tips_for_frames` an `active_parts=None` kwarg that, when a mask has `excluded_seg_ids`, forwards them into `_wing_fk_indices`. Forward `active_parts` from `run_single_fly`/`run_ablation` into `extract_tips_for_frames`. Default None → no exclusion → behavior unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_active_parts.py -v`
Expected: PASS (all Task-1/2/5 tests). Also re-run the silhouette-landmarks suite to confirm no regression: `JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_landmarks.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/active_parts.py third_party/jarvis_jax/jarvis_jax/cse/silhouette_landmarks.py third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py third_party/jarvis_jax/tests/test_active_parts.py
git commit -m "feat(cse): geom-exclusion in the silhouette path (off-part vertices ignored)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Real data-prep for the amputee + headless recordings (SAM3 masks + build_bout + run_stac_bout)

**PREREQUISITE: Task 3 MUST be committed first** (build_bout would KeyError on the reduced coco otherwise). This task is NOT a pytest — it is a documented, reproducible command sequence (env-sensitive) plus an artifacts-exist assertion. A thin sbatch/CLI wrapper is committed; the produced `sam3_masks/` + `cse_work/` are DATA and are NOT committed.

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/dataprep_active_parts.sh` **(new, committed)** — the reproducible command sequence for one condition/split (both env stages).
- Test: `third_party/jarvis_jax/tests/test_active_parts_realdata.py` **(new)** — `assert_dataprep_artifacts_exist(...)` helper + a `skipif`-gated existence test (NOT the heavy generation).

**Interfaces:**
- Consumes: `JARVIS-HybridNet/tools/sam3_label_masks.py` CLI (`--data-root`, `--splits`, `--confidence`, `--qc-thresh`, `--resolution`; out defaults to `<data-root>/sam3_masks/<split>/<rec>/<cam>/Frame_*.npz`); `python -m jarvis_jax.densepose.cse_labels build-bout` (Task-3-patched); `python -m jarvis_jax.tracking.run_stac_bout` (existing).
- Produces (artifacts on disk, PER CONDITION, using the TRAIN split for a larger sample):
  - `<COND>/sam3_masks/train/<rec>/<Cam*>/Frame_*.npz`
  - `<COND>/cse_work/<rec>_bout.h5` (Task-3 reduced schema: 44/47 kp_names)
  - `<COND>/cse_work/<rec>/Fruitfly_ik_v1_cse.h5` (STAC ik; qpos (T,93); reduced kp_names)

- [ ] **Step 1: Commit the reproducible data-prep script**

Create `third_party/jarvis_jax/jarvis_jax/cse/dataprep_active_parts.sh` (run on a GPU node; args: COND REC SPLIT DATASET_CFG):

```bash
#!/usr/bin/env bash
# Phase-4 active-parts data-prep for ONE general_model condition/split.
# Usage: dataprep_active_parts.sh <COND_DIR> <REC> <SPLIT> <DATASET_CFG>
#   e.g. dataprep_active_parts.sh \
#        /gscratch/portia/eabe/data/Johnson_lab/red_data/general_model/S8_male_R_amp \
#        2026_02_13_13_44_49 train amputation
# headless: ... general_model/headless_22_50 2026_06_09_15_46_55 train BDN2_headless
set -euo pipefail
COND="$1"; REC="$2"; SPLIT="$3"; DSCFG="$4"
XML=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml
ANATOMY=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs/anatomy/v1.yaml
STAC_CFG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs
HN=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet
WORK="$COND/cse_work"; mkdir -p "$WORK"

source ~/.bashrc; micromamba activate 3d_tracking

# --- Stage A: SAM3 masks (LD_PRELOAD + cu13 LD_LIBRARY_PATH) ---
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib
python "$HN/tools/sam3_label_masks.py" \
  --data-root "$COND" --splits "$SPLIT" \
  --confidence 0.5 --qc-thresh 0.6 --resolution 1008
unset LD_PRELOAD

# --- Stage B: build bout (CPU; Task-3 reduced-schema patch REQUIRED) ---
unset LD_LIBRARY_PATH
python -m jarvis_jax.densepose.cse_labels build-bout \
  --coco "$COND/annotations/instances_${SPLIT}.json" \
  --calib-root "$COND/calib_params" --rec "$REC" \
  --anatomy "$ANATOMY" --model-xml "$XML" \
  --out "$WORK/${REC}_bout.h5"

# --- Stage C: STAC solve (JAX; LD_LIBRARY_PATH MUST be unset) ---
python -m jarvis_jax.tracking.run_stac_bout \
  --bout "$WORK/${REC}_bout.h5" \
  --out "$WORK/${REC}/Fruitfly_ik_v1_cse.h5" \
  --stac-config-dir "$STAC_CFG" \
  --overrides paths=hyak anatomy=v1 dataset="$DSCFG"
echo "[dataprep] done: $WORK/${REC}/Fruitfly_ik_v1_cse.h5"
```

Commit it:
```bash
git add third_party/jarvis_jax/jarvis_jax/cse/dataprep_active_parts.sh
git commit -m "feat(cse): reproducible Phase-4 data-prep script (SAM3 -> build_bout -> STAC)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 2: Run the data-prep on the amputee (TRAIN split), in the background (env-sensitive, GPU, may exceed 600s)**

The SAM3 stage (train = 174 fs × 7 cams ≈ 1218 imgs) is minutes; STAC is minutes. Run detached on the node:

```bash
bash third_party/jarvis_jax/jarvis_jax/cse/dataprep_active_parts.sh \
  /gscratch/portia/eabe/data/Johnson_lab/red_data/general_model/S8_male_R_amp \
  2026_02_13_13_44_49 train amputation
```
Run this via a BACKGROUND bash (it persists across turns). Watch its output file. HONEST NOTE: the two stages need OPPOSITE `LD_LIBRARY_PATH` — the script sets cu13 for SAM3, then `unset`s it before build_bout/STAC. If SAM3 import fails with a libstdc++/CUDA error, the `LD_PRELOAD`/cu13 env is the culprit; if STAC fails with a cuDNN/JAX error, a stale `LD_LIBRARY_PATH` leaked in — confirm the `unset` ran.

- [ ] **Step 3: Run the data-prep on the headless recording (TRAIN split — val=2 is too few)**

```bash
bash third_party/jarvis_jax/jarvis_jax/cse/dataprep_active_parts.sh \
  /gscratch/portia/eabe/data/Johnson_lab/red_data/general_model/headless_22_50 \
  2026_06_09_15_46_55 train BDN2_headless
```
Also background. headless train = 18 fs × 7 = 126 imgs (fast).

- [ ] **Step 4: Assert the artifacts exist (this is the pytest for T6)**

Create `third_party/jarvis_jax/tests/test_active_parts_realdata.py`:

```python
import os, glob
import h5py
import numpy as np
import pytest

GM = "/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model"
AMP = f"{GM}/S8_male_R_amp"; AMP_REC = "2026_02_13_13_44_49"
HL = f"{GM}/headless_22_50"; HL_REC = "2026_06_09_15_46_55"
SPLIT = "train"


def _artifacts(cond, rec, n_kp_expected):
    masks = glob.glob(f"{cond}/sam3_masks/{SPLIT}/{rec}/*/Frame_*.npz")
    bout = f"{cond}/cse_work/{rec}_bout.h5"
    ik = f"{cond}/cse_work/{rec}/Fruitfly_ik_v1_cse.h5"
    return masks, bout, ik, n_kp_expected


@pytest.mark.parametrize("cond,rec,n_kp", [(AMP, AMP_REC, 44), (HL, HL_REC, 47)])
def test_dataprep_artifacts_exist(cond, rec, n_kp):
    masks, bout, ik, n_kp = _artifacts(cond, rec, n_kp)
    if not (masks and os.path.exists(bout) and os.path.exists(ik)):
        pytest.skip(f"run dataprep_active_parts.sh for {os.path.basename(cond)} first")
    assert len(masks) > 0, "no SAM3 mask npz written"
    with h5py.File(bout, "r") as f:
        names = [x.decode() if isinstance(x, bytes) else x for x in f["kp_names"][()]]
        assert len(names) == n_kp, f"bout has {len(names)} kp_names, expected {n_kp}"
    import stac_mjx.io_dict_to_hdf5 as ioh5
    d = ioh5.load(ik)
    assert np.asarray(d["qpos"]).shape[1] == 93
    ik_names = [x.decode() if isinstance(x, bytes) else x for x in d["kp_names"]]
    assert len(ik_names) == n_kp, f"ik_h5 has {len(ik_names)} kp_names, expected {n_kp}"
```

Run (CPU-safe; skips until the artifacts exist):
```bash
cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest \
  tests/test_active_parts_realdata.py::test_dataprep_artifacts_exist -v
```
Expected: after Steps 2–3 finish, PASS for both `(S8_male_R_amp, 44)` and `(headless_22_50, 47)`. Before they finish, SKIP with the "run dataprep first" reason. SUCCESS CRITERION: SAM3 mask npz present per camera, `<rec>_bout.h5` with 44 (amputee) / 47 (headless) kp_names, `Fruitfly_ik_v1_cse.h5` with qpos (T,93) and matching reduced kp_names.

- [ ] **Step 5: Commit (the test only; NOT the data)**

```bash
git add third_party/jarvis_jax/tests/test_active_parts_realdata.py
git commit -m "test(cse): Phase-4 data-prep artifacts-exist gate (amputee + headless)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Do NOT `git add` `general_model/*/sam3_masks/` or `general_model/*/cse_work/` (data outputs).

---

## Task 7: Real end-to-end active-parts silhouette-IK validation (per-frame)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/run_active_parts_ik.py` **(new)** — thin driver/CLI.
- Test: `third_party/jarvis_jax/tests/test_active_parts_realdata.py` (add the per-frame validation gate)

**Interfaces:**
- Consumes: Task 4 `run_single_fly(recording, *, ik_h5, model_xml, mesh_npz, root, split, calib_dir, use_silhouette, wing_weight, max_frames, n_iter, out_dir, active_parts) -> dict`; Task 1 `derive_active_parts`; the coco `keypoint_names`.
- Produces:
  - `run_active_parts_ik(recording: str, *, cond_root: str, model_xml: str, mesh_npz: str, split: str = "train", calib_dir: str | None = None, use_silhouette: bool = True, max_frames: int = 0, n_iter: int = 50, out_dir: str, override: list[str] | None = None) -> dict` — reads `<cond_root>/annotations/instances_<split>.json` for the recording's `keypoint_names`, derives off-parts, resolves `ik_h5 = <cond_root>/cse_work/<rec>/Fruitfly_ik_v1_cse.h5`, `calib_dir = <cond_root>/calib_params/<rec>` (default), runs `run_single_fly(..., active_parts=off_parts_or_override)`, and returns `{"off_parts": [...], "report": <run_single_fly dict>}`. CLI mirrors these args.

- [ ] **Step 1: Write the failing test**

Add to `third_party/jarvis_jax/tests/test_active_parts_realdata.py`:

```python
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"


def _ik_ready(cond, rec):
    return os.path.exists(f"{cond}/cse_work/{rec}/Fruitfly_ik_v1_cse.h5")


@pytest.mark.parametrize("cond,rec,expect_off,locked", [
    (AMP, AMP_REC, ["legT1R"], list(range(38, 49))),
    (HL, HL_REC, ["head"], []),
])
@pytest.mark.skipif(not (os.path.exists(XML) and os.path.exists(MESH)),
                    reason="model / mesh not present")
def test_run_active_parts_ik_per_frame(cond, rec, expect_off, locked, tmp_path):
    """PER-FRAME validation (framesets are sparse -- no smoothness reliance):
      - off-parts auto-derived from the coco schema;
      - present-marker reproj_px below threshold (7-view triangulation per frame);
      - off-part joints pinned to rest across ALL frames (no phantom);
      - the missing part is not hallucinated (locked qpos == rest)."""
    if not _ik_ready(cond, rec):
        pytest.skip(f"run T6 dataprep for {os.path.basename(cond)} first")
    from jarvis_jax.tracking.run_active_parts_ik import run_active_parts_ik
    out = run_active_parts_ik(
        rec, cond_root=cond, model_xml=XML, mesh_npz=MESH, split=SPLIT,
        use_silhouette=True, max_frames=0, n_iter=50, out_dir=str(tmp_path))
    assert out["off_parts"] == expect_off
    rep = out["report"]
    assert rep["qpos_shape"][1] == 93
    assert rep["off_parts"] == expect_off
    assert rep["locked_qpos_idx"] == locked
    # present-marker reprojection is clean (per-frame, over 7 views). Honest
    # threshold: general_model is a fresh recording w/ factory calib (no Phase-1
    # BA here) -> allow up to 15 px; tighten if BA is later applied.
    assert np.isfinite(rep["reproj_px"]) and rep["reproj_px"] < 15.0, \
        f"present-marker reproj_px={rep['reproj_px']:.2f} too high"
    # no phantom: the RETURNED qpos has every locked joint pinned to rest.
    q = np.load(os.path.join(str(tmp_path), f"{rec}_qpos.npz"))["qpos"]
    if locked:
        assert np.max(np.abs(q[:, locked[0]:locked[-1] + 1])) < 1e-6, \
            "phantom: locked joints drifted off rest in the returned qpos"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_active_parts_realdata.py::test_run_active_parts_ik_per_frame -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.tracking.run_active_parts_ik'` (if the T6 artifacts exist) or SKIP (if not yet generated). Either confirms the test is wired; the import failure is the RED we drive.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/run_active_parts_ik.py`:

```python
"""Phase-4 driver: active-parts silhouette-IK on a general_model recording.

Given a general_model condition (amputee / headless), read the recording's coco
keypoint_names, auto-derive the off body-part(s), and run the Phase-2/3
run_single_fly with the active-parts mask (markers dropped + joints locked +
post-solve clamp + off-part geoms excluded from the silhouette). Reports the
present-marker reprojection error and the no-phantom check (locked joints pinned
to rest). Per-frame; the labeled framesets are temporally sparse.
"""
from __future__ import annotations

import json
import os

from jarvis_jax.tracking.active_parts import derive_active_parts
from jarvis_jax.tracking.silhouette_ik_solve import run_single_fly


def run_active_parts_ik(recording, *, cond_root, model_xml, mesh_npz, split="train",
                        calib_dir=None, use_silhouette=True, max_frames=0, n_iter=50,
                        out_dir, override=None):
    coco = json.load(open(os.path.join(cond_root, "annotations", f"instances_{split}.json")))
    off_parts = derive_active_parts(coco["keypoint_names"], override=override)["off"]
    ik_h5 = os.path.join(cond_root, "cse_work", recording, "Fruitfly_ik_v1_cse.h5")
    if calib_dir is None:
        calib_dir = os.path.join(cond_root, "calib_params", recording)
    report = run_single_fly(
        recording, ik_h5=ik_h5, model_xml=model_xml, mesh_npz=mesh_npz,
        root=cond_root, split=split, calib_dir=calib_dir,
        use_silhouette=use_silhouette, max_frames=max_frames, n_iter=n_iter,
        out_dir=out_dir, active_parts=(override if override is not None else off_parts))
    return {"off_parts": off_parts, "report": report}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-root", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model-xml",
                    default="/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml")
    ap.add_argument("--mesh",
                    default="/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-silhouette", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--override", nargs="*", default=None)
    a = ap.parse_args()
    out = run_active_parts_ik(
        a.rec, cond_root=a.cond_root, model_xml=a.model_xml, mesh_npz=a.mesh,
        split=a.split, use_silhouette=not a.no_silhouette, max_frames=a.max_frames,
        n_iter=a.n_iter, out_dir=a.out_dir, override=a.override)
    print(f"[run_active_parts_ik] off={out['off_parts']} "
          f"reproj_px={out['report']['reproj_px']:.2f} "
          f"locked={out['report']['locked_qpos_idx']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Real GPU run (background), then the per-frame gate.** Under the activated env (T6 artifacts must exist):

```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
cd third_party/jarvis_jax && python -m pytest \
  tests/test_active_parts_realdata.py::test_run_active_parts_ik_per_frame -v -s
```
Run via BACKGROUND bash (both parametrizations solve the full train bout; may exceed 600s). SUCCESS CRITERION per condition:
- `off_parts` auto-derived correctly (`["legT1R"]` for amputee, `["head"]` for headless);
- present-marker `reproj_px < 15px` (per-frame, 7-view; honest — factory calib, no Phase-1 BA in this task);
- amputee: `q[:, 38:49]` all `< 1e-6` (T1R pinned to rest — no phantom); headless: no joints locked (`locked_qpos_idx == []`) and the head-region mesh is excluded from the silhouette (the fit does not try to reach a head that isn't there).

HONEST FALLBACK: if `train` (174/18 fs) gives an unstable STAC offset fit or the reproj is above threshold, first confirm it is not a phantom/mask bug (superpowers:systematic-debugging), then report the real numbers. If the number is genuinely borderline (e.g. 15–20px) due to factory calibration, note it and flag Phase-1 BA on this recording as the remedy — do NOT loosen the threshold silently. If `val` was used and its tiny frameset count (amputee 20 / headless 2) makes the offset fit unstable, re-run on `train`.

- [ ] **Step 5: Commit (driver + test only; NOT data/qpos)**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/run_active_parts_ik.py third_party/jarvis_jax/tests/test_active_parts_realdata.py
git commit -m "feat(cse): active-parts silhouette-IK driver + real per-frame validation (amputee + headless)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

### Spec-coverage check (each Phase-4 spec requirement → task)

| Phase-4 spec / decision / brief requirement | Task |
|---|---|
| Component 5: anatomy + active-parts config (declare model+mesh + active joint/marker/geom masks) | **Task 1** (PART_TABLE + derive) + **Task 2** (build_active_mask) |
| Decision 3: one model per anatomy + per-recording active-parts mask that (a) drops markers, (b) locks joints, (c) excludes geoms | (a)+(b) **Task 2/4**, (c) **Task 5** |
| Off-parts auto-derived from coco keypoint_names diff vs canonical 50 + explicit override (user decision 2) | **Task 1** `derive_active_parts` |
| §6 IK objective: inactive joints held at rest; marker_cost weighted by active-part mask; reuse `stac_core_jaxls` unchanged | **Task 4** (apply_active_mask_to_inputs pre-solve + clamp_locked_qpos post-solve; solver untouched) |
| "Verify no phantom residuals" (spec §11 Phase 4): off-part must not create spurious IK residuals/drift | **Task 4** SIMULATED gate (`max|qpos_locked-rest|<1e-6` AND `|Δreproj_px|<0.75`) + **Task 7** real per-frame gate |
| Reduced-schema kp_data flows through (44/47 markers) — bout build must not KeyError | **Task 3** (build_bout intersect patch; prerequisite for T6) |
| Phase-2/3 hardcoded 50-schema wing indices break under shifted schema → name-based | **Task 4** `_stac_wing_idx` / name-based `_withhold_wing_kp` / augment regression |
| Silhouette excludes off-part geoms (amputation: those geoms; headless: head mesh) | **Task 5** (`wing_side_vertices(exclude_seg_ids)` + `excluded_fps_indices`) |
| Real end-to-end incl. silhouette on REAL amputee/headless data (user decision 1) — SAM3 masks + STAC artifacts generated | **Task 6** (data-prep) + **Task 7** (IK validation) |
| general_model = labeled frames only; validate PER-FRAME (sparse framesets, no long video) | **Task 6** (train split, T = #framesets) + **Task 7** (per-frame reproj, no smoothness reliance) |
| SAM3 stage LD_PRELOAD+cu13 vs JAX/STAC unset LD_LIBRARY_PATH; on-node not sbatch-and-idle | **Task 6** (`dataprep_active_parts.sh` toggles env between stages; background bash) |
| Affine cameras / 3-D obs model / reuse run_single_fly + run_stac_bout unmodified | Global Constraints; **Tasks 4/6/7** (mask in cse layer only) |
| amputation / BDN2_headless dataset configs (cosmetic naming) | **Task 6** (`--overrides dataset=amputation` / `dataset=BDN2_headless`) |

### Placeholder scan

- No "TBD"/"implement later"/"handle edge cases later" tokens. Every code step has runnable code; every run step has an exact command + expected output.
- No unresolved path tokens: `ANATOMY`/`STAC_CFG`/`XML`/`MESH` are concrete absolute paths (verified on disk), not `<path>` placeholders. The stac config dir is `stac-mjx/configs` (verified: has `anatomy/v1.yaml` + `dataset/{amputation,BDN2_headless,}.yaml`) — NOT the top-level `configs/`.
- One deliberate call-out (Task 4 Step 4): the existing full-schema `test_withhold_wing_kp_*` tests must be updated to the new 2-arg `_withhold_wing_kp(kp, kp_names)` signature — flagged explicitly (grep + fix), not hidden.
- Honest limitations flagged rather than papered over: (a) T3 must precede T6 (build_bout KeyError); (b) T6 opposite-`LD_LIBRARY_PATH` fragility with concrete failure-mode diagnosis; (c) headless val=2 is too few → train required; (d) Task-7 reproj threshold is 15px because no Phase-1 BA is applied here (factory calib) — with the remedy (BA) named, not the threshold loosened.

### Type-consistency check across tasks

- **Mask dict** produced by Task 2 `build_active_mask` has keys `{kp_names, kps_to_opt_mask (f32 (K*3,)), qs_to_opt_mask (bool (nq,)), locked_qpos_idx (int32), rest_qpos (f64 (nq,)), excluded_seg_ids (list[int]), excluded_vertex_idx (int64), off_marker_stac_idx (int64), off_parts (list[str])}`. Consumed by: Task 2 `clamp_locked_qpos` (reads `locked_qpos_idx`, `rest_qpos`) and `apply_active_mask_to_inputs` (reads `kps_to_opt_mask`, `qs_to_opt_mask`, `off_marker_stac_idx`); Task 4 `run_single_fly`/`run_ablation` (reads `off_marker_stac_idx`, `off_parts`, `locked_qpos_idx`, passes whole dict to the two helpers); Task 5 `extract_tips_for_frames`/`excluded_fps_indices` (reads `excluded_seg_ids`, `excluded_vertex_idx`); Task 7 asserts `report["locked_qpos_idx"]`/`report["off_parts"]`. Keys match across every producer/consumer.
- **`derive_active_parts` output** `{"off": list[str], "on": list[str]}` feeds `build_active_mask(..., off_parts=off)` (a `list[str]` of PART_TABLE keys) in Tasks 4 & 7. Consistent: `off` is always a subset of `PART_TABLE` keys.
- **`run_single_fly`'s new `active_parts` kwarg** type is `list[str] | dict | None` (list → off-parts; dict → prebuilt mask; None → auto-derive). Task 7's driver passes a `list[str]` (`off_parts` or `override`); Task 4's SIMULATED test passes `["legT1R"]` (list); the internal auto-derive path passes the derived `list[str]` into `build_active_mask`. All three branches converge on a mask dict (or None), and the None/empty-off path is byte-for-byte the pre-Phase-4 behavior.
- **`clamp_locked_qpos(qpos (T,nq) float, mask) -> (T,nq)`** returns the same shape (a copy); Task 4 applies it to `solve_ik`'s `(T,93)` output; Task 4/7 assert `q[:, locked]` == rest and `qpos_shape[1] == 93`. Consistent.
- **Task 3 `triangulate_recording` return arity** changes from 4-tuple to 5-tuple (adds `kp_present`); its only in-module caller is `build_bout` (updated in the same task). No other module imports `triangulate_recording` (module-private-by-convention; the CLI goes through `build_bout`). `_reorder_index` signature is unchanged; only its callers pass intersected `dst_names`.
- **`_stac_wing_idx(kp_names) -> {"left":(int,int),"right":(int,int)}`** (Task 4) is consumed by `_withhold_wing_kp` and `run_ablation`'s `wing_stac_idx`; both build the same 4-index set. The full-schema constant `_STAC_WING_KP_IDX` is retained only for reference/existing tests; no runtime path depends on its hardcoded values after Task 4.
- **Bout h5 schema** written by Task 3 (`keypoints (T,K')`, `kp_names (K',) S20`, `vis (T,K')`, `scale`, `recording`, `fs_keys`, `fs_imgids`) is exactly what `run_stac_bout.run` reads (`kp_names` + `keypoints`, then `prune_model_to_available`), and what `run_single_fly` reads for `fs_imgids` — consistent with reduced K' (44/47/50).
