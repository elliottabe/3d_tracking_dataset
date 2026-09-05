# Courtship bout pipeline — pre-launch audit of the P3a r2 campaign

Read-only audit of `scripts/slurm/mvq_p3a_campaign.sh --run-name pose_mvq_p3a_r2 --local-gpus 4`
(all 11 recordings, 160 bouts, 119 119 frames) at `elliottabe/paper_update_062026` **c578241**.
Nothing was edited, submitted, or run on a GPU.

Effective config composed on CPU with
`hydra.compose(config_name="pipeline", overrides=["pipeline.lifter=mvq","mvq=p3a","recording=session0",
"recording.session_dir=…","outputs.out=…/pose_mvq_p3a_r2"])`; every value below is quoted from that
composition, not from a config file read in isolation.

| campaign fact | value |
|---|---|
| bouts | Session0 30, Session1 130 (11 recordings); **all 160 have `sam3_masks.npz`** |
| frames | 28 417 (S0) + 90 702 (S1) = **119 119**; largest bout 2449 fr (`16_21_32` b11) |
| run roots | fresh `…/<Session>/<rec>/pose_mvq_p3a_r2/`; **Session0's already holds 30 current lifts + `bouts_unified_summary.csv`** |
| lift gate on disk | `{"checkpoint":".../mvq_t1_b16_p3a_20260904/final","exist_thresh":0.5,"identity":"mask","lifter":"mvq","sha256":"7608843ecbd1a10c","step":"final"}` |
| verified | `slurm_bout_array.py --mvq-lift skip … --dry-run` on Session0 prints `mvq lift: skipped (30 bouts already lifted)` — the on-disk gates match `configs/mvq/p3a.yaml` today |

## 0. Campaign wrapper + submitter

| | |
|---|---|
| code | `scripts/slurm/mvq_p3a_campaign.sh`; `scripts/slurm_bout_array.py:492` (main), `:656` (lift-skip check), `:729/:740/:752` (precompute / array / aggregate) |
| effective | `--lifter mvq --mvq-lift skip --mvq-config p3a --slurm ckpt_all`; overrides `recording=session{0,1} recording.session_dir=… [recording.bouts_csv=…] outputs.out=… mvq=p3a pipeline.lifter=mvq` |
| writes | `<run>/bouts_unified_summary.csv` (S0 only, rewritten every invocation), `local-lift-gpu*.log`, then sbatch scripts |
| gating | `bout_lift_is_current` (kp3d gates + `sex.json`) per bout; a stale/partial lift is refused, not silently DLT-re-derived |
| new path? | yes. `slurm=ckpt_all` → `partition ckpt-all`, `constraint h200\|a100\|l40s\|l40\|a40`, `cpus 8`, `mem 48G`, `time 12:00:00`, `requeue` |
| gaps | (a) `wait` with no args always returns 0 → a **crashed local lift worker is not detected** and the chain is submitted anyway (the `--mvq-lift skip` check then catches it and `exit 1`, but only after the previous recordings were already queued); (b) `read_mvq_cfg` reads `checkpoint/step/exist_thresh/merge_dist_units` but **not `identity`** — the local lift hardcodes argparse's `mask` default. Consistent today; flip `configs/mvq/p3a.yaml: identity: sex` and every bout would be refused at Stage B. `mask_assign_margin_units`/`mask_assign_max_units` are likewise defaulted on the local route and are *not* in the gate string. |

Session0's `courtship_bouts_unified_summary.csv` is confirmed a broken symlink into a deleted
`Predictions_3D_34662304/`; the wrapper's fly_id-less rewrite from `courtship_bouts_fly0_summary.csv`
is required and `parse_bouts` (`sam3_driver.py:25`) keeps every row of a fly_id-less CSV. Session1's
per-recording CSVs all exist and carry `fly_id = Session1/<ts>`, matching `session_tag_for`.

## 1. Stage A/B replacement — mvq lift

| | |
|---|---|
| code | `scripts/mvq_lift_bout.py:main`; `jarvis_jax/tracking/lift_mvq.py:1033 lift_masked_bout`, `:331 MVQRunner`, `:218 resolve_mask_identity`, `:305 mvq_gate_string` |
| config | `mvq.checkpoint=…/mvq_t1_b16_p3a_20260904/final`, `step=null`, `exist_thresh=0.5`, `identity=mask`, `batch=8`, `merge_dist_units=30.0`, `attn_impl=null` |
| per-fly? | per bout; writes **both** flies. `fly{f}` = SAM3 mask fly `f` (human review), instance→mask by relative nearness (margin 8 u, cap 60 u) |
| writes | `bouts/bout_XXXXX/fly{0,1}/{kp2d,kp3d}.npz` + `bouts/bout_XXXXX/{sex.json,mvq_meta.json}` |
| gating | `bout_lift_is_current`: both `kp3d.npz` carry the gate string **and** `sex.json` exists |
| new path? | yes — `identity=mask` (not the old `sex` head). Session0's r2 lifts are already `identity mask`, 30/30 with `sex.json`, no WARNING/Traceback in the four worker logs. |
| gaps | `conf` written by the lifter is **per-view visibility** and `conf3d` its mean over cameras (`lift_mvq.py:542 to_pipeline`) — a *fraction of cameras*, not a ViTPose peak confidence. Every downstream threshold tuned on the DLT route (`detector.conf_thresh 0.3`, `filtering.confidence.threshold 0.3`, `stac.offsets_min_conf 0.7`) is being applied to a different quantity. Measured on the r2 lifts: fly1 (male) 99.3 % of frames fully finite, fly0 (female) 77.3 %; frames passing `min(conf3d) ≥ 0.7` — the offsets gate — are 24 211/28 447 (male) and **5 370/28 447 (female)**. Both clear the 500-frame sample, so no crash, but the female's offsets come from her ~9 best bouts. |

## 2. Bout entry, camera order, coverage, sync

`run_bout.py:1429 process_bout_fly`. `bout_complete` = `DONE` file. `check_bout_camera_order`
(`:1470`) is non-mutating and never fatal. `coverage.json` written once per bout dir.
`masks.min_views=4` / `masks.kp_mask_agree_fly_lengths=3.0` are **Stage-B gates only** — they never run
on this route (they live inside `if not stage_done(kp3d_path)`), and they are not in the mvq gate
string. `sync.invalidate_stale=true`; both sessions report `plan status=none` (no `Cam*_meta.csv`), so
nothing is invalidated. `run_bout.py:1544` **refuses** to fall through to ViTPose/DLT if a lift is
missing — the important safety property, verified present.

## 3. Stage B2 — keypoint filter

| | |
|---|---|
| code | `run_bout.py:1865` → `jarvis_jax/tracking/filter.py:67 filter_bout_kp3d` |
| config | `filtering.enabled=true`, `preserve_raw_patterns=['Wing']`, `confidence{enabled,0.3,exclude Wing}`, `bone_length{5.0σ}`, `isolated_spike{10.0,1}`, `interpolation{spline, edge≤5}`, `savgol{7,2}`; `centroid_jump/identity_relink/medfilt*/confidence_smooth` off |
| per-fly? | per (bout, fly) |
| writes | `kp3d_filt.npz`; skipped if present |
| new path? | unchanged by the four commits under audit |
| gaps | the `confidence.threshold 0.3` now means "visible in <30 % of cameras" (see §1). It cannot lose data: `filter.py:94-95` restores any keypoint the filter NaN'd that the raw had. 20.8 % of the female's finite (frame, keypoint) entries are below it and get spline-replaced rather than dropped. |

## 4. Body scale

| | |
|---|---|
| code | `run_bout.py:1897`; `scripts/estimate_recording_scale.py:822 estimate_run_root`, `:660 _determine_identity`, `:225 assert_plausible_body_scale` |
| config | `scaling.scale_keypoints=rigid_segment` (estimator `umeyama` ignored, recorded as `"ignored (rigid_segment)"`), `robust_stat=median`, `allow_shared_scale=false` |
| per-fly? | **per fly** — `scale_by_fly[str(fly)]`; a shared scale raises `RuntimeError` |
| writes | `<run>/scale.json` once per run root, by the precompute job |
| gating | `stage_done(scale.json)`; pooled branch needs `_n_pooled ≥ 2` and `identity=="canonical"` |
| new path? | yes, and it will resolve: all 30 (S0) bout dirs carry `sex.json` with a real `int` `male_fly: 1`, so `_determine_identity` returns `canonical`. v1's run root produced `scale_by_fly {"0":0.011518,"1":0.011530}` over 60 bout-flies — the same code path. |
| gaps | `bout_kp3d_paths` prefers `kp3d_filt.npz`; at precompute time only bout `idxs[0]` fly0 has one, so the pool is 1 filtered + 59 raw. A ~0.1 % inconsistency, not a defect. **Hard dependency**: if any single bout's lift is missing `sex.json`, identity goes `unknown` and the *whole recording* aborts at this line. |

`configs/anatomy/v1.yaml: model.segment_calibration=false` → `segment_scales.json` is not written.

## 5. Marker offsets

| | |
|---|---|
| code | `run_bout.py:2103`; `scripts/offsets_sample.py:180 resolve_offsets_path`, `:88 select_offsets_sample`, `:201 offsets_fit_cfg`; `jarvis_jax/tracking/stac.py:21 fit_offsets_once` |
| config | `stac.offsets_min_conf=0.7`, `stac.n_fit_frames=500`, `stac.allow_shared_offsets=false`, `mad_k=3.0` (hardcoded) |
| per-fly? | **per fly** — `offsets_fly0.h5` / `offsets_fly1.h5` + a `.json` provenance sidecar; shared `offsets.h5` refused unless `allow_shared_offsets` |
| writes | `<run>/offsets_fly<f>.h5` (atomic `.tmp` → `os.replace`), written by precompute; array tasks read-only |
| new path? | **yes** (157648d). v1's run roots contain only a shared `offsets.h5` — proof this is the changed behaviour. `offsets_fit_cfg` deep-copies the cfg and sets `JAXLS_SMOOTH_WEIGHT=0`, `JAXLS_SMOOTH_Q_MULT=None`, `JAXLS_CHUNK_SIZE=0` on the *copy*, so the outer cfg is untouched. |
| gaps | this is the **only** consumer of `anatomy.model.JAXLS_COST_TOLERANCE=1e-10` (353c859) — see §6. Sample selection is deterministic round-robin, so the female's 500 frames come from her best-covered bouts. |

## 6. Stage C — IK

| | |
|---|---|
| code | `run_bout.py:2164`; `jarvis_jax/tracking/stac_perframe.py:146 solve_per_frame_ik` |
| config | `stac.solver=per_frame`; `stac.per_frame`: `starts=[wing_rest_left,wing_rest_right,wing_rest_both]` (+ implicit `zero`), `lambda_initial=5e-4`, `lambda_min=1e-8`, `cost_tolerance=1e-12`, `gradient_tolerance=1e-14`, `parameter_tolerance=1e-16`, `n_iter=500`, `batch=512`, `switch_weight=1e-3`, `switch_wing_weight=20.0`, `save_candidates=true`. `nan_solve.min_keypoints=null` (all-or-nothing frame gate), `NAN_SOLVE_MAX_GAP=10`, `NAN_SOLVE_MIN_SEG=30` |
| per-fly? | per (bout, fly); reads that fly's `offsets_fly<f>.h5` and that fly's `scale` |
| writes | `stac_ik.h5` (via `stac_ik.tmp.h5`), with `attrs ik_solver="per_frame"` + `ik_start_idx`/`ik_iterations`/`qpos_candidates`/`candidate_costs` |
| new path? | **yes — verified selected by default**, not by an override: `configs/stac/default.yaml: solver: per_frame`, inherited by `stac/courtship.yaml`, and `run_bout.py:2213` takes the `per_frame` branch whenever `_ok.any()`. v1's `stac_ik.h5` has *no* attrs (batch); r2's will say `per_frame`. |
| gaps | **`JAXLS_COST_TOLERANCE=1e-10` does NOT reach Stage C.** `solve_per_frame_ik` builds its own `JaxlsBatchSolver` from `stac.per_frame.cost_tolerance=1e-12`; `model.JAXLS_*` only reaches `stac_mjx/stac.py:325`, i.e. the offsets fit. 353c859 is live, but on the offsets fit alone — do not describe it as the Stage-C tolerance. Also: `_restore_unsolved_nan(h5, ok, segs)` (`run_bout.py:1385`) never uses `segs`. A fly with no ≥30-frame finite run writes `unsolvable.json`, returns, and does **not** fail the array task (deliberate). |

## 7. Stage C2 — polish (not run)

`run_bout.py:2297`: `if polish.enabled and solver == "per_frame": pass`. `stac.polish.enabled=true`
with a full weight block is therefore **entirely inert** on this campaign — the per-frame solver already
multi-starts and Viterbi-selects. `jarvis_jax/tracking/stac_polish.py:162 polish_stac_h5` is only
reachable with `stac.solver=batch` (its `dof_mask`/`frame_costs_np`/`viterbi_select` are still imported
and used by `stac_perframe`).

## 8. Stage D — model→mm bridge / D2 wing fit

`run_bout.py:2321` → `jarvis_jax/tracking/bridge.py:50 compute_bridges`, `ik.bridge_mode="keypoint"`
(per-frame Umeyama on FK sites vs kp3d; reads no masks). Writes `qpos_refined.npz`.
`wing_mask_fit.enabled=false` → `_action="none"`, `_pose_sig="none"`, nothing imported or written
(matches v1's `qc.json: pose_source "none"`). Of `configs/ik/default.yaml`, only
`xml`/`mesh_npz`/`mesh_subset`/`bridge_mode` are read by anything — see cleanup C4.

## 9. Stage E — outputs.h5 + QC / overlays / Stage F

| | |
|---|---|
| code | `run_bout.py:2428`; `outputs.py:146 build_fly_outputs`; `qc.py:317 frame_metrics`, `:395 qc_report`; `qc_perframe.per_frame_qc`; overlays `:2536`; sidebyside `:2629` |
| config | `outputs.mesh_subset=fps_500+wing`, **`outputs.overlay=false`**, `overlay_workers=4`, `sidebyside=true`, `sidebyside_frames=300`, `sidebyside_right=rigcam`, `sidebyside_views=left,top,right`, `detector.conf_thresh=0.3` (only used here, for `valid2d`) |
| writes | `outputs.h5`, `qc.json`, `qc_perframe.npz`, `sidebyside.mp4` + `sidebyside.pose_source.json`, then `DONE` |
| new path? | yes — overlays off is 345cdee and **is intended for this campaign** (v1 spent 496–502 s/fly on them; sidebyside, the artifact a reader actually judges, stays on with three rig views) |
| gaps | `sidebyside_pose_args("rigcam","none")` → `--pose refined`, correct. `mesh_mask_iou` on the mvq route is structurally tiny (v1 bout 28: hard 0.028 / soft 0.169 for both flies) — usable as a *relative* A/B number only, not an absolute gate. |

## 10. Aggregate

`build_aggregate_script` (`slurm_bout_array.py:401`) → `session_qc.py:12 aggregate_session_qc` over
`bouts/bout_*/fly*/qc.json`, `--dependency=afterok:<array>`. **Two live bugs**, both confirmed in v1's
`qc/session_qc.json`:
* `iou_hard_median: NaN`, `iou_soft_median: NaN` — it reads `d["silhouette_iou"]`, but `qc.py:450`
  renamed that key to `"mesh_mask_iou"` on 2026-09-01 (`qc.py:181` documents the rename). Not
  mvq-specific: the session IoU has been NaN for every run since that rename, and `iou_by_bout.png`
  is a blank chart.
* every row reads `bout: -1, fly: -1` — `_parse_name` matches `bout_(\d+)_fly(\d)` against
  `os.path.basename(p)`, which is always `qc.json`. The dashboard's x-axis is meaningless and a bad
  row cannot be traced to a bout.

Sexing (`_canonicalize_bout_sex`, `run_bout.py:2745`) is **not auto-invoked** — identity comes from the
lifter's `sex.json`. `check_track_merge` runs per bout and writes `track_qc.json` (never fatal).

## Cleanup list

**must-fix-before-launch**
1. **`session_qc.py:23-24`: `silhouette_iou` → `mesh_mask_iou`.** One-line fix; without it the
   campaign's only session-level silhouette number is NaN for all 11 recordings and the run has no
   aggregate visual-fit metric to compare against v1.
2. **`session_qc.py:8 _parse_name`: parse the bout/fly from the *path*, not the basename**
   (`re.search(r"bout_(\d+)/fly(\d)", p)`). Otherwise every dashboard row is `-1.-1` and a regressed
   bout cannot be identified without re-deriving it.
3. **`mvq_p3a_campaign.sh`: check the local lift workers' exit status.** Collect PIDs and `wait "$pid"`
   per worker (or `wait -n` in a loop) and abort the recording on a non-zero status. Today a worker
   that dies mid-recording is invisible; the failure only surfaces one step later in
   `slurm_bout_array.py:656`, after other recordings have already been queued, and a bout missing
   `sex.json` aborts that whole recording's scale/offsets stage anyway.

**should-fix**
4. `mvq_p3a_campaign.sh: read_mvq_cfg` should also read `identity` (and pass
   `--mask-assign-margin-units`/`--mask-assign-max-units`) so the `--local-gpus` route has the same
   single source of truth the array route already has (`slurm_bout_array.py:702-712`). `identity` is
   in the gate string, so a divergence refuses all 160 bouts; the two assign radii are *not*, so a
   divergence there is silent.
5. Raise `configs/slurm/ckpt_all.yaml: mem: 48` to 64. Measured on v1: `MaxRSS = 0.0197·T + 7.9 GB`;
   the three ≥2100-frame bouts (`16_21_32` b9/b11, `14_54_28` b16) peaked at **50.3 GB against a 48 GB
   request** and survived only because this cluster does not hard-enforce. r2 drops the overlay
   subprocesses (~1.5 GB) but keeps the QC `masks_by_frame`/`mesh_by_frame` lists that dominate.
6. Pass `hydra.run.dir=${outputs.out}/hydra/bout_${...}` from `build_jax_array_script` /
   `build_precompute_script`. All 31 v1 jobs per recording wrote one shared
   `<run>/hydra/analysis/{run_bout.log,.hydra}` — the config snapshot is last-writer-wins and the log
   is interleaved. `configs/pipeline.yaml`'s own comment claims the launchers already do this; they
   do not.
7. `configs/stac/default.yaml`: mark the whole `polish:` block "solver: batch only" at the top (it is
   inert here, §7), and note that `infer_qvels`/`bucketed_ik`/`continuous`/`enable_padding`/
   `ik_only_path` are batch-path-only too.
8. `configs/anatomy/v1.yaml: JAXLS_COST_TOLERANCE` comment says it is the Stage-C fix; correct it to
   "offsets fit only under `stac.solver: per_frame`" (§6).

**cosmetic**
9. `configs/ik/default.yaml`: delete `limit_weight`, `smooth_weight`, `n_iter`, `beta`, `huber_delta`,
   `cg_tolerance_max`, `cg_tolerance_min` — grep shows **zero** consumers, yet the file's own comment
   says "still live: used by the Stage-C STAC path". Same failure mode as the silhouette keys deleted
   above them.
10. `run_bout.py:1385 _restore_unsolved_nan(h5, ok, segs)` — `segs` is never used; drop the parameter
    and the `_segs if not _ok.all() else None` argument at `:2288`.
11. `configs/pipeline.yaml: hydra.job.config.override_dirname.exclude_keys` still lists `silhouette`
    (group deleted) and omits `mvq`/`pipeline`; `override_dirname` is unused because `run.dir` is fixed.
12. `configs/detector/vitpose_v3.yaml` is fully dead on the mvq route except `conf_thresh`; the
    checkpoint is never loaded. Worth a one-line note so nobody tunes it expecting an effect.
13. Stale on disk, safe to leave but not to reuse: the complete v1 campaign under
    `…/<Session>/<rec>/pose_mvq_p3a/` (160/160 bouts DONE, `identity=mvq_sex_head`, shared
    `offsets.h5`, batch STAC) and Session0's `pose_mvq_ik*`/`pose_mvq*` scratch roots. r2 writes to a
    different root, so nothing is silently reused — but keep v1 until the r2 A/B is scored, then
    delete. Nothing in `pose_mvq_p3a_r2/` needs deleting: it holds only the 30 current lifts, the
    rewritten CSV and four lift logs.

## Cost

Measured from v1 (the same 160 bouts, `sacct`): array **56.11 GPU-h** / 160 tasks (mean 21.0 min, max
66 min), precompute **5.31 GPU-h** / 11 jobs (mean 29.0 min), aggregate ~17 s each, 513/513 steps
COMPLETED.

r2 deltas: overlays off saves 0.25 s·frame⁻¹·fly⁻¹ (v1: 502 s for 2007 fr) → **−16.5 GPU-h**;
Stage C batch→per_frame saves ≈0.12 s·frame⁻¹·fly⁻¹ (475→158 s on bout 28 fly0, e6d34a5; batch
measured 6.6 min there) → **−7.8 GPU-h**; per-fly offsets doubles the fit in precompute → **+2.5 GPU-h**.

* IK array ≈ **32 GPU-h**, precompute ≈ **8 GPU-h**, aggregate ≈ 0. **Total ≈ 40 GPU-h.**
* Local mvq lift (Session1 only; Session0's 30 are current and skip): 90 702 frames at the measured
  ~9.9 frames/s aggregate on 4 GPUs → **≈2.5 h wall, ≈10 GPU-h** on the interactive node.
* Wall clock: ~2.5 h blocking local lift, then 11 parallel chains whose critical path is
  precompute (~40 min) + longest array task (~33 min at 2449 fr) + aggregate ≈ **1.3 h of compute**.
  On an empty ckpt-all, ~4–5 h end to end; at the queue depth this partition has been running,
  budget **12–30 h**.
* Longest task ~35 min against a 12 h limit — no risk there.

## Launch readiness: GO with these fixes

The four changes the user wanted are all genuinely on the executed path: `stac.solver=per_frame` is
the shipped default and is the branch taken; `scale_by_fly` and `offsets_fly{0,1}.h5` are both per-fly
and both *refuse* rather than silently share; the mvq gate string on Session0's disk matches
`configs/mvq/p3a.yaml` and Stages A/B are hard-refused rather than skipped. The one correction is
`JAXLS_COST_TOLERANCE=1e-10`, which reaches the **offsets fit**, not Stage C.

Ship the three must-fixes first (all small, none touches numerics: two `session_qc.py` lines and the
wrapper's `wait`), then launch. Items 5–6 are cheap and worth folding in at the same time.
