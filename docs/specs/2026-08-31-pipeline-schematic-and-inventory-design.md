# Pipeline schematic + component inventory — design

**Status:** design approved in chat 2026-08-31 (sections 1-3). Not yet an
implementation plan. Written mid-discussion so the schematic is not lost.

**Goal.** One comprehensive schematic from NEW RAW VIDEO (no bout summary, no
segmentation) through to kinematics, usable as a working guide; plus a
classified inventory of every component so cleanup becomes a series of
reviewed decisions rather than a guess.

**Decisions taken (user):**
1. Staged: Phase 1 coarse pass -> automatic bout windows; Phase 2 throughput
   rework for full-rate-everywhere, designed only when needed (YAGNI).
2. A bout must pass BOTH a behaviour gate and a trackability gate.
3. The coarse pass is SAM3 at a reduced frame rate (not classical CV, not a
   new trained detector) — same mask semantics, no new failure modes.
4. This pass produces the schematic + inventory ONLY. No deletions.
5. Approach C: emit today's bout-summary CSV contract AND persist the
   per-frame coarse tracks.

---

## 1. Schematic

`[NEW]` = Phase 1 work. Everything else exists today.

```
RAW INPUT   per recording: Cam<id>.mp4 x7, 800 fps, ~497,993 frames (~10.4 min)
            calibration/  Cam<id>.yaml + Cam<id>_dlt.csv   <- ORTHOGRAPHIC DLT
                          camera order = directory glob, NEVER a config
   |
   +-- frame-sync gate ......... sync_plan.json (canonical slots; dropped-frame realign)
   |                            sync_policy.stale_invalidation_enabled gates deletion
   |
   +-- [NEW] COARSE PASS  SAM3 @ 1/16 rate (50 fps) over the WHOLE recording
   |     |                217,871 camera-frames = 1.10x today's per-bout SAM3 cost
   |     +-- recording-level TRACKS (persisted, Approach C)
   |     |     per (coarse_frame, camera, slot): centroid, area, in_frame,
   |     |     border distance, inter-fly separation, wing proxy
   |     +-- GATE 1 behaviour ...... wing extension / proximity
   |     +-- GATE 2 trackability ... area, border, n_cams, separability
   |     +-- bout summary CSV  fly_id, bout_idx, start_frame, end_frame, source_fly
   |           ^ SAME contract run_bout.py consumes (recording.bouts_csv + bout_ids)
   |
   +-- sexing / canonicalization ... sex.json per bout (male_fly; enforces male = fly1)
                                     gates per-fly scale; manual-GUI reviewed today
   v
PER-BOUT  (run_bout.py, SLURM array over bouts; DONE markers + resume)

  SAM3 full-rate .... sam3_masks.npz   packed, valid, centroids, in_frame, cameras
  Stage A  ViTPose 2D on mask crops ......... kp2d.npz      (kp2d, conf)
  Stage B  DLT triangulation ................ kp3d.npz      (kp3d, conf3d)
             + reproj_resid_px=10 consensus gate (3x trigger, >=3 views survive)
  Stage B2 filter ........................... kp3d_filt.npz  <- FEEDS EVERYTHING BELOW
             conf -> bone-length MAD -> spike -> spline -> savgol
             preserve_raw_patterns: ['Wing']  (wings carry real ~120-135 Hz signal)
  scale .... scale.json  RECORDING-level, pooled; scale_by_fly when identity canonical
  offsets .. offsets.h5      STAC marker offsets
  Stage C  STAC ik_only ..................... stac_ik.h5
  Stage D  polish (silhouette weights 0 = OFF) + model->mm BRIDGE
                                              qpos_refined.npz (qpos, bridge_s/R/t/ok)
  Stage E  outputs.h5 + qc.json + qc_perframe.npz
  Stage F  sidebyside viz ................... sidebyside.mp4  (--right rigcam)
   v
SESSION   aggregate / QC dashboard
   v
KINEMATICS   <-- the next part
```

### Annotations (measured this session, keep visible)

- **Units.** 1 world unit ~= 0.1 mm. The `_mm` suffix (`kp3d_mm`, `mesh_mm`)
  overstates by ~10x. Anchors: track-merge body length 22.65 -> 2.265 mm;
  T1 femur 5.05 -> 0.505 mm; T3 femur 6.96 -> 0.696 mm. Ratios/percentages
  and all px figures are unaffected.
- **Stage C/D is the accuracy bottleneck**, and now has TWO distinct targets:
  - male: uniform ~2.1x, a PROPORTION mismatch (trunk implies a scale +7.2%
    off the leg chain on fly1) -> per-segment scaling
  - female: the same, PLUS a wing-specific failure at 5.6x
    (wings fitted 21.33 px vs measured 3.81 px; WingL_V13 31.50 px = 8.81x)
- **`paths.data_root` still defaults to `red_data_unified_V3`**, not the
  leakage-free `red_data_3d_v5`. Same shape as the `target_sigma: 2.0` trap.
- **`bridge_ok` / qpos finiteness is the real per-frame validity signal**, not
  `coverage.json` (which reported fly0 "ok", 7 cams, median_views 7.0 for a
  bout where 25% of her frames have no IK).

---

## 2. Coarse pass

**Rate** 1/16 (50 fps). A fly walking ~20 mm/s moves ~0.4 mm per coarse frame,
under a fifth of a 2.3 mm body — safe for identity tracking. 1/32 (0.55x
today's cost) is the fallback.

**Chunking is already solved.** Reuse `chunk_len: 1000`, `chunk_overlap: 120`
(`configs/sam3/default.yaml`): identity is carried across a boundary by
box-prompting the next chunk at the previous chunk's bboxes plus an overlap IoU
match, each session freed between chunks. A recording at 1/16 is 31,125
frames/camera ~= 31 chunks.

**Identity drift across those boundaries does not matter.** A typical bout is
947 full-rate frames ~= 59 COARSE frames, i.e. 6% of one 1000-frame chunk, so
bouts land inside a single chunk essentially always; and the full-rate per-bout
SAM3 pass re-establishes identity independently afterwards, as today.

**Cost, measured.** Bouts are only 5.7% of a recording (28,417 of 497,993
frames over 30 bouts):

| pass | camera-frames | vs today |
|---|---|---|
| today's per-bout SAM3, whole recording | 198,919 | 1.00x |
| whole recording @ 1/16 (50 fps) | 217,871 | **1.10x** |
| whole recording @ 1/32 (25 fps) | 108,935 | 0.55x |
| full-rate everywhere (Phase 2) | 3,485,951 | 17.5x |

**Gate 1 — behaviour.** Wing extension via area ABOVE that fly's own baseline
(the male's area is genuinely larger when extended; the existing sexing CV
already relies on this), plus centroid separation for proximity/following.

**Gate 2 — trackability.** The same area signal read in the opposite
direction, which is why one cheap measurement serves both gates:

- area **< 50%** of that fly's own rolling median
- centroid border distance **< ~50 px**
- `in_frame` >= 3 cameras (hard floor, `MIN_CAMS`, unchanged)
- flies separable / not overlapping

Thresholds are relative to each fly's OWN baseline, never absolute (healthy
male 670k px vs healthy female 519k px).

**Calibration evidence** — bout 28, median over valid cameras. Both signals
lead `in_frame` by 90-190 frames, and neither fires on the male:

| range | fly0 area | vs 0-1400 | fly0 border px | fly1 area | fly1 border px |
|---|---|---|---|---|---|
| 0-1400 | 518,634 | 0.0% | 138.1 | 670,398 | 179.2 |
| 1400-1500 | 334,230 | -35.6% | 68.0 | 682,752 | 185.2 |
| 1500-1590 | 200,015 | **-61.4%** | **42.1** | 679,242 | 179.6 |
| 1590-1750 | 60,122 | -88.4% | 26.7 | 659,236 | 182.9 |

fly0's IK dies at 1500; `in_frame` only drops at 1590. A -50% area threshold
lands on the real cliff; -35% would trigger at 1400 and discard ~100 usable
frames.

**Validation.** Session0 `2025_10_20_13_20_04` has 30 human-reviewed bouts.
Measure recall, precision, and boundary agreement. Sharpest acceptance test:
the coarse pass should emit **bout 28 ending near frame 1500, not 2007** —
trimming the 506 frames the current external segmentation includes and the IK
cannot fit.

**Placement.** New `scripts/coarse_pass.py` (or
`jarvis_jax/predict/coarse_pass.py`) calling the existing `sam3_driver` with a
stride, emitting the tracks file plus a bout-summary CSV in today's schema.
No changes to `run_bout.py`.

---

## 3. Inventory

Six categories. The last two were forced by evidence, not chosen up front.

| Tag | Meaning | Verified example |
|---|---|---|
| **KEEP** | in the production raw-video -> kinematics path | `run_bout.py` Stages A-F, `sam3_driver`, `estimate_recording_scale` |
| **PARKED** | inactive, deliberately retained, documented re-enable condition | silhouette polish (weights 0; re-enable once leg curl is fixed upstream); `kp2d_displacement_gate` (default off, kept as a negative result) |
| **RESEARCH** | built for an investigation, outside production, kept as the record of a decision | `jarvis_jax/hybridnet/*` — 73 refs, **zero from `run_bout.py`** |
| **LEGACY-IN-USE** | old route still load-bearing; has an ordering constraint | `utils/fly_detection.py` — the only thing writing bout summaries today |
| **SUPERSEDED** | replaced, zero live callers -> deletable | none found in the sample so far |
| **STALE-REFERENCE** | code already gone, but references/defaults still point at it -> fix the pointer, delete nothing | `batch_split_valid_bouts.py` docstring naming 3 already-deleted scripts; **`paths.data_root` -> V3** |

**Evidence standard.** Every row carries callers with file paths, the named
successor, and the exact command that verified it. Without that the inventory
is an opinion: `batch_pair_bouts` / `merge_paired_bouts` / `run_stac_paired`
looked like deletable dead code and the files had **already been removed** —
only a docstring referenced them.

**Ordering constraints (a DAG, not a list):**

```
[independent, do now]  paths.data_root: V3 -> v5
[blocked]              utils/fly_detection.py bout-summary role
                         requires: coarse pass validated vs the 30 known bouts
[blocked]              legacy JARVIS batch route (batch_process_predictions,
                       Predictions_3D_* / data3D_*.csv)
                         requires: the same validation
```

---

## 4. Open items

- **Female orientation (raised 2026-08-31, RESOLVED — see below).** In
  `figures/2026-08-31-ik-metric/bout28_fly0_rigcam_1200-1500.mp4` the user
  reports fly0's fit is flipped/rotated: her dorsal side should face left but
  appears to face the wall. If real, an orientation error would explain a large
  share of her wing residual (a rotated body projects its wings to the wrong
  side) and would NOT be visible to any current metric — the same class of bug
  as the historical keypoint-order scramble, which LOO/IoU was fully blind to.
  Test: build an orthonormal body frame from measured keypoints and from fitted
  sites, and report the rotation between them.
- Phase 2 throughput rework — design only when Phase 1 is validated.
- `kp3d_mm` / `mesh_mm` naming is misleading (see Units); renaming touches
  every consumer and is a separate job.

---

## 5. RESOLVED: the female's orientation flip and her wing failure are ONE bug

User observation: in `bout28_fly0_rigcam_1200-1500.mp4` fly0's dorsal side
faces the wall when it should face left. Confirmed, and traced.

**The IK is not at fault.** Measured-vs-fitted body-frame rotation over offsets
0-1400: fly1 median 1.6 deg, fly0 median 5.4 deg, and **zero flipped axes on
either fly**. The fit follows its input faithfully, so a wrong orientation is
inherited, not introduced.

**Independent check, needing no chamber geometry.** A fly cannot have its dorsal
side pointing toward its own tarsal tips. Angle between the dorsal axis and
(body centroid - mean tarsal tip):

| | dorsal vs +Z (up) | dorsal vs body-above-own-feet |
|---|---|---|
| fly1 (male) | 3.0 deg, 0% pointing down | 19.5 deg, **0%** inverted |
| fly0 (female) | 40.6 deg, 2.1% down | 8.2 deg, **12.9%** inverted |

**Mechanism.** The dorsal axis is `anterior x (WingL_base - WingR_base)`, so a
left/right swap of the wing bases flips it. Sign test:

| | L-R as labelled | L/R swapped |
|---|---|---|
| fly1 (male) | 0/280 = 0.0% inverted | 280/280 = 100.0% |
| fly0 (female) | 36/280 = **12.9%** | 244/280 = 87.1% |

The male's 0/280 proves the test is perfectly sensitive. So this is NOT a global
swap — it is an **intermittent per-frame L/R mirror swap in ~13% of the female's
frames**, exactly the mirror confusion already measured for her (2D mirror rate
53.3% among her >30 px errors; the `reproj_resid_px` consensus gate rejects
100% of the male's mirror views but only 77.4% of hers, because she is
view-starved and consensus needs spare views).

**This unifies two findings previously treated as separate.** When her lateral
axis flips, WingL and WingR swap sides, so her wing markers land on the wrong
side of the body — which is the gross wing displacement seen in the renders and
the 5.60x wing residual (fitted 21.33 px vs measured 3.81 px) against 2.0-2.95x
everywhere else. The orientation flip and the "wing IK failure" are the same bug.

**Consequence for the plan.** The female's fix is UPSTREAM (2D mirror confusion
/ view scarcity), not in the IK and not in per-segment scaling. This supersedes
the earlier framing that her problem was proximity/identity contamination plus a
separate wing-IK defect. The male's uniform ~2.1x proportion mismatch is
untouched by this and remains the per-segment-scaling target.

**Cheap gate this buys us.** `dorsal . (body_centroid - mean_tarsal_tip) < 0` is
a per-frame, self-referential anatomy check that needs no ground truth and no
chamber geometry. It catches exactly the bug class that LOO/IoU, residuals and
NaN checks are all blind to. Worth adding to `qc_perframe` alongside the new
`ik_reproj` metric.
