# Joint Kinematics Analysis — Notebook Guide

A reading and reference guide for `Joint_Kinematics_Analysis.ipynb`.

---

## What this notebook does

This notebook takes the output of the STAC-MJX inverse kinematics (IK) solver — joint angles and body positions for free-walking *Drosophila* — and turns it into publication-ready kinematic analyses. Starting from a single HDF5 file containing walking bouts across multiple flies, it computes per-frame kinematic features (angular velocities, step phases, swing/stance labels, heading and speed), reduces the high-dimensional joint space to a low-dimensional embedding (PCA, optional GPU-UMAP), and produces phase-coordination figures and paper-style plots comparable to Pratt et al. 2024 and DeAngelis et al. 2019.

The main output is a flat Pandas DataFrame (`df_valid`) with ~100 columns per frame (joint angles, derivatives, phase signals, speed, PCA coordinates), plus PDF figures saved to `output/joint_kinematics/`.

---

## Quick-start: how to run

**Minimal pipeline** — run these cells in order, every session:

```
Cell 1  → imports
Cell 2  → configuration (set H5_PATH here)
Cell 3  → define data-loading functions
Cell 4  → define preprocessing functions
Cell 5  → define reduction functions
Cell 6  → define visualisation functions
Cell 7  → define coordination functions
─────────────────────────────────────────── (run once above; below is the actual pipeline)
Cell 8  → load data
Cell 9  → build joint index + leg-tip sites
Cell 10 → build flat DataFrame  →  df
Cell 11 → compute all features  →  df
Cell 12 → feature matrix + NaN filter  →  df_valid
Cell 13 → run PCA  →  pca_result
Cell 14 → add PC1…PC10 to df_valid
```

**Then run any analysis section** (each is independent after Cell 14):

| Section | Cells |
|---|---|
| Distributions overview (speed, heading, height, leg spread) | 15 |
| Step-cycle diagnostic | 16 |
| Phase coordination (onset-based, per step-cycle speed) | 17 |
| Single-bout coordination diagnostic | 18 |
| Standard visualisations (scree, loadings, PCA trajectories, leg coordination) | 19–27 |
| Simple whole-session UMAP | 28–32 |
| Per-fly summaries | 33 |
| Paper figures (speed-binned PCA, ROM, 3-D stance trajectory) | 34–40 |
| Segment UMAP (DeAngelis et al. style) | 41–49 |
| Step-Cycle Joint-Angle UMAP | 50–53 |

---

## Configuration — Cell 2

**This is the only cell most users need to edit.**

| Variable | Default | Purpose |
|---|---|---|
| `H5_PATH` | *(set per user)* | Path to the combined IK output `.h5` file |
| `ACTIVE_JOINT_SET` | `'core'` | Which joints to analyse: `'core'` (4), `'main'` (7), or `'full'` (11 joints/leg) |
| `FPS` | `800` | Camera frame rate in Hz |
| `REFERENCE_LEG` | `'T1_right'` | Leg used as phase reference for coordination plots |
| `HEADING_BODIES` | `['thorax']` | Body used for heading and speed computation |
| `LEGS` | 6-element list | `['T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right']` |
| `OUTPUT_DIR` | `./output/joint_kinematics` | Where PDF figures are saved |

`MODEL_SCALE = 100.0` is defined in Cell 4 for reference (no longer used for speed conversion) — see Units section below.

**Joint sets:**
- `'core'` — coxa_flexion, coxa, femur, tibia (4 joints × 6 legs = 24 features)
- `'main'` — adds coxa_twist, femur_twist, tarsus (7 × 6 = 42)
- `'full'` — adds tarsus2–5 (11 × 6 = 66)

---

## Data flow

```
HDF5 file
  (qpos, xpos, xpos_egocentric per bout)
        │
        ▼ Cell 8  load_ik_bouts()
  bout_dict        ← nested dict: bout_dict['bout_000']['qpos'] etc.
  bout_keys        ← sorted list of 'bout_NNN' strings
  fly_ids          ← array (N_bouts,) of fly identifier strings
  clip_lengths     ← array (N_bouts,) of frames per bout
  names_qpos       ← joint names in qpos column order
  egocentric_site_names  ← 50 site names for xpos_egocentric
        │
        ▼ Cell 9  build_joint_index() + find_leg_tip_site_indices()
  joint_list       ← list of (leg, joint, qpos_column_index) triples
  leg_tip_site_indices  ← {leg: index into xpos_egocentric[:,idx,:]}
        │
        ▼ Cell 10  bouts_to_dataframe()
  df               ← flat DataFrame, all bouts stacked
                      one row per frame; 'bout_id' column groups bouts
        │
        ▼ Cell 11  9-step feature pipeline
  df               ← same DataFrame + all computed columns
        │
        ▼ Cell 12  get_feature_matrix()
  X                ← (N_valid_frames × N_features) float array
  df_valid         ← df with NaN rows removed, row-aligned to X
  valid_mask       ← boolean mask from df → df_valid
        │
        ▼ Cell 13  run_pca()
  pca_result       ← (N_valid_frames × 10) float array
  pca              ← fitted sklearn PCA object
  scaler           ← fitted StandardScaler
        │
        ▼ Cell 14  add columns to df_valid
  df_valid         ← + PC1 … PC10 columns
        │
        ▼ Cells 15–53  Analysis & figures
```

---

## Key variables — where they live

| Variable | Cell | Description |
|---|---|---|
| `bout_dict` | 8 | Full nested dict. Raw data: `bout_dict['bout_000']['qpos']` (T×N_joints), `['xpos']` (T×N_bodies×3), `['xpos_egocentric']` (T×50×3) |
| `bout_keys` | 8 | `['bout_000', 'bout_001', …]` — use to iterate over bouts |
| `fly_ids` | 8 | String array identifying which fly each bout belongs to |
| `clip_lengths` | 8 | Number of frames in each bout |
| `names_qpos` | 8 | E.g. `['free','free',…,'tibia_T1_left',…]` — needed to find joint indices |
| `names_xpos` | 8 | Body names for `xpos` array (e.g. `'thorax'` at index 1); includes `'claw_T1_left'` etc. |
| `egocentric_site_names` | 8 | 50 site names for `xpos_egocentric` (e.g. `'tracking[T1L_TaTip]_fly'`) |
| `joint_list` | 9 | `[(leg, joint, qpos_col_idx), …]` — the canonical joint ordering used throughout |
| `leg_tip_site_indices` | 9 | `{'T1_left': 18, 'T1_right': 25, …}` — index into `xpos_egocentric` axis 1 |
| `df` | 10 | All frames from all bouts in one DataFrame. Key grouping column: `bout_id` |
| `df_valid` | 12 | `df` with rows dropped where any feature is NaN. Row-aligned to `pca_result` |
| `X` | 12 | Numeric feature array fed to PCA; shape `(N_valid_frames, N_features)` |
| `pca_result` | 13 | PC coordinates; shape `(N_valid_frames, 10)` |
| `MODEL_SCALE` | 4 | ≈100; ratio of model units to real fly scale. Kept for reference; not used for speed conversion |

---

## Key DataFrame columns

Columns in `df` and `df_valid`, grouped by when they are added:

**Cell 10 — raw data:**

| Column | Description |
|---|---|
| `frame` | Frame index within the bout (0-based) |
| `bout_id` | String key (e.g. `'bout_042'`); use to group by bout |
| `fly_id` | Fly identifier string |
| `{leg}_{joint}` | Joint angle in **radians** (e.g. `T1_left_tibia`) |
| `{body}_x/y/z` | World-frame body position in model metres (e.g. `thorax_x`) |
| `{leg}_tip_z_ego` | Foot-tip Z in egocentric frame (negative = below thorax; rises during swing) |
| `{leg}_tip_x_ego` | Egocentric X position of leg tip (model-metres) |
| `{leg}_tip_y_ego` | Egocentric Y position of leg tip (model-metres) |
| `{leg}_tip_x/y/z_world` | World-frame leg tip position from MuJoCo `xpos` (`claw_T{1,2,3}_{left,right}` bodies). Model-m; ×10 → real mm |

**Cell 11 — computed features:**

| Column | Added in step | Description |
|---|---|---|
| `{leg}_{joint}_d1` | 1 | Angular velocity (rad/s) — Savitzky-Golay derivative |
| `{leg}_{joint}_d2` | 1 | Angular acceleration (rad/s²) |
| `{leg}_phase` | 2 | **Primary step-cycle phase** [-π, π], Hilbert transform of tibia angle |
| `{leg}_swing_stance` | 3 | 1 = stance, 0 = swing. Derived from `{leg}_tip_z_world` Hilbert phase (`phase < 0` = stance). Correct for all 6 legs |
| `n_legs_stance` | 3 | Count of legs in stance (0–6) per frame |
| `forward_speed` | 4 | Thorax translation speed in **mm/s** (model-m/s × fps × 10; model scale cancels to give mm/s) |
| `heading` | 4 | Fly heading in radians [-π, π] (world frame: 0 = +X direction) |
| `turning_rate` | 4 | Rate of heading change in rad/s |
| `{leg}_phase_rel` | 5 | Phase of leg relative to `REFERENCE_LEG` [-π, π] |
| `mean_abs_vel` | 6 | Mean \|angular velocity\| across all joints, smoothed per bout (rad/s). **Activity proxy for filtering** |
| `{leg}_ego_phase` | 7 | Hilbert phase of foot-tip Z (egocentric height oscillation) |
| `{leg}_lift_phase` | 8 | Stride-detection phase: -π at liftoff, 0 at mid-stance, +π just before next liftoff |
| `step_cycle_id` | 9 | Integer step-cycle index within each bout (0-based, NaN outside first/last swing onset) |
| `step_cycle_mean_speed` | 9 | Mean `forward_speed` over the containing step cycle (mm/s); NaN for partial cycles at bout edges |
| `leg_spread_mm2` | 15 | Area of hexagon formed by 6 leg tips (real mm²); shoelace formula on egocentric XY |
| `heading_dev` | 15 | **Corridor-axis deviation** (degrees, 0–90°): 0° = straight along corridor, 90° = perpendicular. Computed as `min(heading mod 180°, 180° − heading mod 180°)` |

**Cell 14 — PCA:**

| Column | Description |
|---|---|
| `PC1` … `PC10` | PCA coordinates in joint-angle space |

---

## Function reference

### Cell 3 — Data loading

| Function | Returns | Notes |
|---|---|---|
| `load_ik_bouts(h5_path)` | `bout_dict, bout_keys, fly_ids, clip_lengths` | Wraps `ioh5.load()`; prints summary |
| `find_leg_tip_site_indices(site_names, legs)` | `{leg: int or None}` | Searches for `{T1L}_TaTip` pattern in `egocentric_site_names`; tries 4 naming conventions |
| `bouts_to_dataframe(bout_dict, bout_keys, fly_ids, joint_list, names_xpos, heading_bodies, egocentric_site_names, leg_tip_site_indices)` | `df` | Stacks all bouts into one flat DataFrame |
| `build_joint_index(names_qpos, joint_set, legs)` | `joint_index, joint_list` | Maps joint names to qpos column indices; warns on missing joints |

### Cell 4 — Preprocessing

| Function | Output columns | Notes |
|---|---|---|
| `compute_derivatives(df, joint_cols, fps)` | `{col}_d1`, `{col}_d2` | Savitzky-Golay, window=5, polyorder=2 |
| `compute_phases(df, legs, joint='tibia', fps)` | `{leg}_phase` | Bandpass 5–50 Hz + Hilbert; per-bout to avoid edge artifacts |
| `classify_swing_stance(df, legs)` | `{leg}_swing_stance` | Tibia-based fallback: `phase ≥ 0` → stance (1). Used internally by `compute_swing_stance_from_z` |
| `compute_swing_stance_from_z(df, legs, fps)` | `{leg}_swing_stance` | **Primary method (Cell 11 step 3)**. Hilbert phase of `{leg}_tip_z_world`: `phase < 0` → stance (1). Works correctly for all 6 legs. Falls back to tibia-based if Z data missing |
| `compute_swing_stance_z_threshold(df, legs, fps)` | `{leg}_swing_stance` | Alternative: bandpass-filtered Z zero-crossing. `z_filt > 0` → swing (0). Onset-accurate (liftoff ≈ zero crossing) |
| `compute_swing_stance_tip_speed(df, legs, fps, thresh_frac)` | `{leg}_swing_stance` | Alternative: 3-D world-frame tip speed threshold. `speed > thresh_frac × per-bout-max` → swing (0). Sharp onset at liftoff |
| `compute_swing_stance_coxa(df, legs, fps)` | `{leg}_swing_stance` | Alternative: coxa_flexion zero-crossing. Forward protraction (`coxa_filt > 0`) = swing (0) |
| `compute_swing_stance_tibia_perleg(df, legs, fps)` | `{leg}_swing_stance` | Alternative: tibia Hilbert phase with per-leg sign convention (T1 phase ≥ 0 = stance; T2/T3 phase < 0 = stance) |
| `compute_n_legs_stance(df, legs)` | `n_legs_stance` | Row-sum of swing_stance columns |
| `compute_heading_and_velocity(df, fps, body)` | `forward_speed`, `heading`, `turning_rate` | Per-bout; speed in mm/s; Gaussian-smoothed positions |
| `compute_egocentric_leg_phase(df, legs, fps)` | `{leg}_ego_phase` | Bandpass + Hilbert on foot-tip Z height |
| `compute_foot_phase_from_lift(df, legs, fps)` | `{leg}_lift_phase` | Swing-onset detection (threshold at p10 + 30% × amplitude); linear interpolation between onsets |
| `compute_relative_phases(df, legs, reference_leg)` | `{leg}_phase_rel` | Wrapped circular difference vs reference |
| `swing_onsets_from_stance(series)` | frame index array | Returns indices where `swing_stance` transitions 1→0 (stance→swing); used in step-cycle and coordination analyses |
| `compute_step_cycle_speed(df, fps, ref_leg)` | `step_cycle_id`, `step_cycle_mean_speed` | Assigns step-cycle index and mean forward speed per cycle; cycles defined by `ref_leg` swing onsets |
| `compute_phase_offset(onsets_a, onsets_b, fps)` | float array | Onset-based phase of leg A relative to leg B's stride cycle |
| `compute_phase_offset_by_cycle(ref_onsets, leg_onsets, cycle_speeds, q_lo, q_hi)` | float array | Like `compute_phase_offset` but filters to step cycles whose `step_cycle_mean_speed` ∈ [q_lo, q_hi) |
| `mean_resultant_length(phases)` | scalar R ∈ [0,1] | Circular synchrony metric: 1 = perfect, 0 = uniform |

### Cells 5, 6, 7 — Reduction, visualisation & coordination

| Function | Does |
|---|---|
| `get_feature_matrix(df, joint_list, include_derivatives)` | Builds `X`; drops NaN rows; returns `(X, feature_names, valid_mask)` |
| `run_pca(X, n_components, standardize)` | StandardScaler → PCA; returns `(pca_result, pca, scaler)` |
| `run_umap(X, …)` | GPU-optional UMAP; returns `(embedding, reducer, scaler)` |
| `plot_pca_variance(pca)` | Scree plot (individual + cumulative explained variance) |
| `plot_pca_loadings(pca, feature_names, joint_list)` | Heatmap of PC loadings organised by leg and joint |
| `plot_pca_trajectory(pca_result, df_valid, color_by)` | PC1 vs PC2 scatter, coloured by any column |
| `plot_embedding(coords, df_valid, color_by)` | Generic 2-D scatter; handles circular variables and categorical coloring |
| `plot_phase_averaged_pca(df_valid, pca_result, reference_leg)` | Gait cycle in PC space: bins frames by reference-leg phase, plots mean trajectory |
| `plot_leg_coordination(df, legs)` | 6×6 matrix: diagonal = phase histograms, off-diagonal = pairwise phase scatter |
| `compute_tripod_coordination_strength(df, legs)` | TCS metric: fraction of time two tripod groups are in synchronised swing |
| `sample_segments(bout_dict, …, leg_site_indices)` | Randomly samples fixed-length egocentric-position windows for segment UMAP |
| `sample_segments_qpos(bout_dict, …, joint_list)` | Extracts joint-angle windows aligned to the same segment windows |

---

## Section map

| Section | Cells | What runs |
|---|---|---|
| Configuration | 2 | Set paths and parameters |
| Function definitions | 3, 4, 5, 6, 7 | Define all functions (no outputs) |
| **Main pipeline** | **8 → 14** | **Load data → build df → preprocess → PCA. Run this every session.** |
| Distributions overview | 15 | Speed, heading (corridor-axis deviation), height, leg spread — KDE + per-bout strip plots |
| Step-cycle diagnostic | 16 | 4-panel validation: 10 short + 10 long bouts → `step_cycle_diagnostic.pdf` |
| Phase coordination | 17 | Onset-based polar histograms × 4 step-cycle speed quartiles + per-fly R-matrix |
| Single-bout coordination diagnostic | 18 | Set `DIAG_BOUT`; Z-trace + onset timeline and per-cycle polar grid for one bout |
| Standard visualisations | 19–27 | Scree plot (19), loadings heatmap (20), PCA scatter panels (21–23), joint-vs-phase grids (24–25), leg coordination matrix (26), tripod strength (27) |
| Simple whole-session UMAP | 28–32 | GPU imports (28), 3-D UMAP scatter with `COLOR_BY` toggle (29), fly ID encoding (30), UMAP color variables (31), per-bout trajectories (32) |
| Per-fly summaries | 33 | Speed distributions, R-matrix heatmaps, per-fly trajectory overlays |
| **Paper figures** | **34–40** | Speed-binned PCA (34–35), 3-D speed-binned PCA/trajectory (36–37), Range of Motion (38–39), ROM config (40) |
| **Segment UMAP** | **41–49** | DeAngelis-style windowed segments; see detail below |
| **Step-Cycle UMAP** | **50–53** | Step-cycle joint-angle UMAP (Section 11); see detail below |

---

## Choosing a phase signal

Three step-phase signals are available. Pick based on your question:

| Signal | Column | Best for |
|---|---|---|
| **Tibia Hilbert** | `{leg}_phase` | Default. Most reliable; works without egocentric data. Continuous [-π, π]. Use this unless you have a specific reason not to. |
| **Foot-lift** | `{leg}_lift_phase` | When you want phase anchored to a physical event. -π = liftoff, 0 = mid-stance, +π = just before next liftoff. Discrete onsets, gaps at bout edges. |
| **Ego Z Hilbert** | `{leg}_ego_phase` | Continuous Hilbert on foot Z height. Less discriminating than tibia because foot Z is flat during stance. |

---

## Activity filter

Cells 34–35 (paper figures) filter frames using:

```python
_activity_mask = df_valid['mean_abs_vel'] > df_valid['mean_abs_vel'].quantile(0.40)
```

This keeps the **top 60% most-active frames** by mean absolute joint angular velocity.

---

## Units

| Quantity | Unit | Notes |
|---|---|---|
| Joint angles | rad | `<compiler angle="radian"/>` in the MuJoCo XML |
| Angular velocity (`_d1`) | rad/s | |
| Body positions (`xpos`) | model-metres | MuJoCo SI; model is ~100× real fly scale |
| `forward_speed` | **mm/s** | Computed as \|d(xpos)/dt\| × fps × 10. Model-m/s × fps, ÷100 model scale, × 1000 m→mm = ×10 net (mean ≈ 19 mm/s) |
| `mean_abs_vel` | rad/s | Mean \|joint velocity\| across all joints, Gaussian-smoothed (σ=10 frames) per bout |
| `heading`, `turning_rate` | rad, rad/s | Raw heading spans [-π, π]; 0 = fly moving in +X direction |
| `heading_dev` | degrees [0, 90] | Corridor-axis deviation: 0° = straight, 90° = perpendicular. Folds ±180° to 0° |
| `leg_spread_mm2` | mm² | Hexagon area from 6 egocentric leg-tip XY positions |
| Egocentric leg-tip XY (`tip_x/y_ego`) | model-metres | Same scale as xpos; ×10 → mm |
| World-frame leg-tip XYZ (`{leg}_tip_x/y/z_world`) | model-metres | From `xpos` (`claw_T{1,2,3}_{left,right}` bodies); ×10 → real mm |
| `step_cycle_mean_speed` | mm/s | Mean `forward_speed` over one step cycle |

---

## UMAP analyses

There are **three separate UMAP analyses** in the notebook, answering different questions on different inputs.

> **What is UMAP?** UMAP (Uniform Manifold Approximation and Projection) is a nonlinear dimensionality reduction method. Unlike PCA, it preserves local neighborhood structure rather than global variance. All analyses here use the GPU-accelerated version from RAPIDS cuML (requires an NVIDIA GPU).

---

### Analysis 1 — Simple whole-session UMAP (Cells 28–32)

**Question it answers:** What does the full joint-angle space look like? Do different flies, speeds, or gait states separate?

#### What goes in

The same feature matrix `X` that is fed to PCA (Cell 12): joint angles ± their first derivatives, one row per frame, NaN rows removed. Standardized with `StandardScaler` → `X_scaled`.

Shape: `(N_valid_frames, N_features)` — e.g. 78k frames × 48 features for the `'core'` joint set with derivatives.

#### Processing (Cell 28–29)

```
X  →  StandardScaler  →  X_scaled
X_scaled  →  cuML UMAP (n_components=6)  →  umap_result  shape (N, 6)
```

#### What comes out

| Variable | Shape | Description |
|---|---|---|
| `X_scaled` | (N, features) | Standardized feature matrix |
| `reducer_cuml` | cuML object | Fitted UMAP reducer |
| `umap_result` | (N, 6) | 6-D embedding; rows align 1-to-1 with `df_valid` |

#### Cell 29 — COLOR_BY toggle

Cell 29 produces a 3-D scatter (UMAP dimensions 1, 2, 3) with a configurable coloring:

```python
COLOR_BY      = 'cycle_speed'  # 'instant_vel' | 'cycle_speed' | 'n_legs_stance'
                               # 'phase_rel'   | 'phase'
PHASE_REL_LEG = 'T1_right'    # used when COLOR_BY == 'phase_rel'
PHASE_LEG     = 'T1_left'     # used when COLOR_BY == 'phase'
```

| Option | Variable colored | Colormap |
|---|---|---|
| `'instant_vel'` | `forward_speed` at each frame | turbo, data-range |
| `'cycle_speed'` | `step_cycle_mean_speed` (mean speed of the step cycle containing each frame) | turbo, data-range |
| `'n_legs_stance'` | `n_legs_stance` (0–6) | RdYlGn, 0–6 fixed |
| `'phase_rel'` | `{PHASE_REL_LEG}_phase_rel` — phase relative to reference leg | twilight, ±π |
| `'phase'` | `{PHASE_LEG}_phase` — that leg's own tibia Hilbert phase | twilight, ±π |

#### Remaining cells

| Cell | Figure | Coloring |
|---|---|---|
| 30 | Fly ID integer encoding (sets `codes`, `categories`) | — |
| 31 | Multi-panel 2D scatter (UMAP1 vs UMAP2) | n_legs_stance, forward_speed, bout index, fly ID |
| 32 | Grid of per-bout 2D trajectories | Frame index (time within bout) |

#### Parameters

| Parameter | Value | What it controls |
|---|---|---|
| `n_components=6` | 6 | Embedding dimensionality. Visualize any 3 at a time. |
| `n_neighbors=15` | 15 | Size of the local neighborhood. Small = fine clusters; large = broad topology. |
| `min_dist=0.01` | 0.01 | How tightly points pack. Small = tight clusters; large = uniform spread. |
| `metric='euclidean'` | euclidean | Distance metric in input space. |

#### How to interpret

- **Ring or torus:** gait is rhythmically structured. Rotation = progression through stride cycle.
- **Discrete blobs:** distinct gait states. Check `n_legs_stance` — blobs = tripod phases.
- **Fly IDs separate:** substantial inter-individual variability.
- **Cell 32 (per-bout trajectories):** loops = periodic gait; drift = non-stationary behavior.

---

### Analysis 2 — Segment UMAP, DeAngelis style (Cells 41–49)

**Question it answers:** What are the distinct movement motifs in the data, and how are they distributed across gait cycle, speed, and individuals? Introduced by DeAngelis et al. (eLife 2019).

**Key difference from Analysis 1:** Each data point is a **short time window** (200 ms total), capturing temporal dynamics over a full stride cycle rather than a single-frame snapshot.

#### Cell 41 — Configuration

All parameters for the segment UMAP are set here, plus the **shared coloring toggle** used by Cells 44, 46, 47, 48, and 49:

| Parameter | Default | What it controls |
|---|---|---|
| `HALF_WIN_MS = 100` | 100 ms | Half-width of each window. Total = 201 ms ≈ 1–2 stride cycles. |
| `HALF_WIN_FRAMES` | 80 | Computed from HALF_WIN_MS × FPS. |
| `WIN_LENGTH` | 161 | Total frames per segment. |
| `N_SEGMENTS_TARGET` | 100 000 | Number of segments to randomly sample. Reduce if GPU OOM. |
| `SEGMENT_SEED` | 42 | Random seed for reproducibility. |
| `SEG_UMAP_N_NEIGHBORS` | 15 | UMAP neighborhood size. |
| `SEG_UMAP_MIN_DIST` | 0.1 | UMAP cluster packing. |
| `SEG_UMAP_COMPONENTS` | 3 | Embedding dimensionality. |
| `COLOR_BY_JT` | `'cycle_speed'` | **Coloring toggle** for all DeAngelis plotting cells (see below). |
| `PHASE_REL_LEG_JT` | `'T1_right'` | Used when `COLOR_BY_JT == 'phase_rel'`. |
| `PHASE_LEG_JT` | `'T1_left'` | Used when `COLOR_BY_JT == 'phase'`. |

**COLOR_BY_JT options** (same options as Cell 29 `COLOR_BY`):
- `'instant_vel'` — instantaneous forward speed at segment center
- `'cycle_speed'` — step-cycle mean speed for the cycle containing the segment center
- `'n_legs_stance'` — number of legs in stance at center frame (RdYlGn, 0–6)
- `'phase_rel'` — phase of `PHASE_REL_LEG_JT` relative to reference leg (twilight, ±π)
- `'phase'` — `PHASE_LEG_JT` own tibia Hilbert phase (twilight, ±π)

Cell 41 also prints the **memory estimate** (~2 GB raw at 100k segments). Reduce `N_SEGMENTS_TARGET` or `HALF_WIN_MS` if you run out of GPU memory.

#### Cell 42 — `sample_segments()` — extracting windows

Randomly samples `N_SEGMENTS_TARGET` time windows; extracts **egocentric foot-tip (x, y) positions** for 6 legs.

**What comes out:**
- `segments_raw`: shape `(N, 161, 12)` — N segments × 161 frames × 12 variables (6 legs × x,y)
- `seg_meta`: dict with arrays `fly_id`, `bout_id`, `bout_idx`, `center_frame`

Also defines `standardize_segments()` (two-stage normalization; see below).

#### Two-stage standardization

**Stage 1 — subtract per-segment temporal mean:** removes absolute foot position, keeps relative motion within each window.

**Stage 2 — z-score across segments per (timepoint, variable):** equalizes variance across slow and fast joints / all points in the stride cycle.

#### Cell 43 — Guard + run full-body UMAP

Standardizes segments, flattens `(N, 161, 12)` → `(N, 1932)`, then runs cuML UMAP.

**What comes out:**
- `segments_std`: `(N, 161, 12)` — standardized
- `X_seg`: `(N, 1932)` — flattened
- `umap_seg`: `(N, 3)` — 3-D full-body embedding

#### Cell 44 — Full-Body UMAP: Helpers and Figures

Defines `_seg_col(col_or_arr)` helper (maps segment center frame → `df_valid` row for behavioral variable lookup) and computes `_jt_cvals`, `_jt_clabel`, `_jt_cmap`, `_jt_kw` from the `COLOR_BY_JT` toggle.

Produces:
- **Figure 1:** single 3-D scatter colored by the `COLOR_BY_JT` toggle
- **Figure 2:** fixed 2×2 overview panels (limb Z height, fly ID, n_legs_stance, bout index)

#### Cell 45 — `sample_segments_qpos()` — joint-angle windows

Uses the **same** `(bout_idx, center_frame)` pairs from `seg_meta` to extract joint-angle time windows from `qpos`. All three subsequent UMAP analyses (per-leg×joint, per-joint-type) operate on the same segments as the full-body UMAP.

**What comes out:**
- `qpos_segs`: `(N, 161, N_joints)` — joint angles per segment
- `qpos_col_names`: list of `'{leg}_{joint}'` strings

#### Cell 46 — Recompute Colors (lightweight)

**Change `COLOR_BY_JT` in Cell 41, then re-run this cell alone** to update coloring without repeating heavy UMAP computation. Re-defines `_seg_col` and recomputes `_jt_cvals` etc. All plotting cells (47, 48, 49) read these variables.

You can also override the toggle directly in Cell 46 (comment is provided) for one-off changes.

#### Cell 47 — Per-(Leg × Joint) UMAP: Run + Plot

For every (leg, joint) combination, runs a separate **2-D cuML UMAP** on single-channel time series `(N, 161)`. Results plotted as a grid (rows = legs, columns = joint types) colored by `_jt_cvals` from Cell 46.

**What comes out:** `umap_joints`: dict mapping `(leg, joint) → (N, 2)` embedding.

Each panel reveals the kinematic range and structure for that specific joint. A ring shape = clear periodic motion; a cloud = noisy / less structured dynamics.

#### Cell 48 — Per-Joint-Type UMAP: Run (heavy)

For each joint type (e.g., `'tibia'`), pools all 6 legs' time series → shape `(N, 161×6)` → runs **3-D cuML UMAP**.

**What comes out:** `umap_joint_types`: dict mapping `joint_name → (N, 6)` embedding (6-component).

#### Cell 49 — Per-Joint-Type UMAP: Plot (lightweight)

**Re-run after changing `COLOR_BY_JT` and Recompute Colors (Cell 46)** to update plots without re-running the UMAP fit. Produces a 2-row grid of 3-D scatter plots (one panel per joint type) colored by `_jt_cvals`.

By pooling all 6 legs, this reveals the shared kinematic template per joint type regardless of which leg it belongs to.

---

### Analysis 3 — Step-Cycle Joint-Angle UMAP (Cells 50–53)

**Question it answers:** What are the distinct gait patterns organized by complete stride cycles? Uses biologically-grounded segments (one T1_left step cycle per point) and joint angles as features.

**Key difference from Analysis 2:** Segments are **T1_left step cycles** (biologically defined), not random windows. Features are **joint angles** (JOINT_SETS['main']), not egocentric foot positions. Time-normalization (phase-resampling) makes fast and slow cycles directly comparable.

#### Cell 50 — Configuration

| Parameter | Default | What it controls |
|---|---|---|
| `N_PHASE_BINS` | 50 | Phase grid points per cycle. Each cycle resampled to exactly this many points. |
| `JOINT_SET_NAME` | `'main'` | Joint set to use: `'core'` / `'main'` / `'full'` |
| `MIN_CYCLE_FRAMES` | 30 | Skip cycles shorter than this (≈37 ms at 800 Hz; filters partial cycles) |
| `N_CYCLES_MAX` | `None` | Optional cap on cycle count; `None` = use all |
| `SC_UMAP_COMPONENTS` | 3 | Embedding dimensionality |
| `SC_UMAP_N_NEIGHBORS` | 15 | UMAP neighborhood size |
| `SC_UMAP_MIN_DIST` | 0.10 | UMAP cluster packing |

#### Cell 51 — `sample_step_cycle_segments()` — extracting cycles

Iterates T1_left step cycles within each bout (using `swing_onsets_from_stance`). For each cycle `[t0, t1)`:
1. Normalizes time within cycle to [0, 1]
2. For each leg × each joint, interpolates the angle onto the `N_PHASE_BINS`-point phase grid
3. Concatenates into a single feature vector: `N_PHASE_BINS × n_legs × n_joints` dims

NaN values are linearly interpolated; all-NaN channels replaced with zeros.

**What comes out:**
- `X_raw_sc`: `(N_cycles, N_PHASE_BINS × n_legs × n_joints)` float32
- `sc_meta`: list of dicts with `fly_id`, `bout_id`, `speed`, `n_legs_stance`, `duration`

#### Cell 52 — Standardize + Run

Two-stage standardization:
1. **Per-cycle DC removal:** reshape to `(N, n_legs×n_joints, N_PHASE_BINS)`, subtract the phase-axis mean per channel (removes absolute joint angle, keeps oscillatory shape)
2. **Cross-cycle z-score:** z-score each feature across all cycles

Then runs cuML UMAP → `umap_sc`: `(N_cycles, SC_UMAP_COMPONENTS)`.

Also computes metadata arrays: `sc_speeds`, `sc_stances`, `sc_durations`, `sc_fly_ids`.

#### Cell 53 — Visualisation

Produces:
- **3-D scatter grid** (4 panels): colored by step-cycle speed, n_legs_stance, fly ID, cycle duration → `umap_step_cycle_{JOINT_SET_NAME}.pdf`
- **2-D projection grid** (4 rows × 3 pairs): UMAP1×2, UMAP1×3, UMAP2×3 → `umap_step_cycle_{JOINT_SET_NAME}_2d.pdf`

---

### Summary: which UMAP to use for what

| Question | Use |
|---|---|
| Does overall gait structure vary by fly or speed? | Analysis 1, Cell 29 |
| What does the trajectory through kinematic space look like per bout? | Analysis 1, Cell 32 |
| Are there discrete gait states (motifs) in full-body movement? | Analysis 2, full-body (Cells 43–44) |
| Which joints have the most structured rhythmic dynamics? | Analysis 2, per-joint grid (Cell 47) |
| Is the kinematic template for a joint consistent across legs? | Analysis 2, per-joint-type (Cells 48–49) |
| How do complete stride cycles cluster by speed or gait mode? | Analysis 3, step-cycle UMAP (Cells 52–53) |
| Do gait modes form structured clusters when using joint angles? | Analysis 3 vs Analysis 2 comparison |
