# Joint Kinematics Analysis — Notebook Guide

A reading and reference guide for `Joint_Kinematics_Analysis.ipynb`.

---

## What this notebook does

This notebook takes the output of the STAC-MJX inverse kinematics (IK) solver — joint angles and body positions for free-walking *Drosophila* — and turns it into publication-ready kinematic analyses. Starting from a single HDF5 file containing 198 walking bouts across 7 flies, it computes per-frame kinematic features (angular velocities, step phases, swing/stance labels, heading and speed), reduces the high-dimensional joint space to a low-dimensional embedding (PCA, optional GPU-UMAP), and produces phase-coordination figures and paper-style plots comparable to Pratt et al. 2024 and DeAngelis et al. 2019.

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
| Phase coordination (onset-based, per speed) | 17 |
| Standard visualisations (scree, loadings, trajectories) | 18–26 |
| Simple whole-session UMAP | 27–31 |
| Per-fly summaries | 32 |
| Paper figures (phase portraits, activity bins, ROM) | 33–36 |
| Segment UMAP (DeAngelis et al. style) | 37–43 |

---

## Configuration — Cell 3

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
        ▼ Cells 15–43  Analysis & figures
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
| `classify_swing_stance(df, legs)` | `{leg}_swing_stance` | Tibia-based fallback: `phase ≥ 0` → stance (1). Not called directly in Cell 11 — used internally by `compute_swing_stance_from_z` |
| `compute_swing_stance_from_z(df, legs, fps)` | `{leg}_swing_stance` | **Primary method (Cell 11 step 3)**. Hilbert phase of `{leg}_tip_z_world`: `phase < 0` → stance (1). Works correctly for all 6 legs (tibia phase is ~180° out of phase with the step cycle for T2/T3 vs T1). Falls back to tibia-based if Z data missing |
| `compute_n_legs_stance(df, legs)` | `n_legs_stance` | Row-sum of swing_stance columns |
| `compute_heading_and_velocity(df, fps, body)` | `forward_speed`, `heading`, `turning_rate` | Per-bout; speed in mm/s; Gaussian-smoothed positions |
| `compute_egocentric_leg_phase(df, legs, fps)` | `{leg}_ego_phase` | Bandpass + Hilbert on foot-tip Z height |
| `compute_foot_phase_from_lift(df, legs, fps)` | `{leg}_lift_phase` | Swing-onset detection (threshold at p10 + 30% × amplitude); linear interpolation between onsets |
| `compute_relative_phases(df, legs, reference_leg)` | `{leg}_phase_rel` | Wrapped circular difference vs reference |
| `swing_onsets_from_stance(series)` | frame index array | Returns indices where `swing_stance` transitions 1→0 (stance→swing); used in step-cycle and coordination analyses |
| `compute_step_cycle_speed(df, fps, ref_leg)` | `step_cycle_id`, `step_cycle_mean_speed` | Assigns step-cycle index and mean forward speed per cycle; cycles defined by `ref_leg` swing onsets |
| `compute_phase_offset(onsets_a, onsets_b, fps)` | float array | Onset-based phase of leg A relative to leg B's stride cycle; used in Cell 18 restore section |
| `compute_phase_offset_by_cycle(ref_onsets, leg_onsets, cycle_speeds, q_lo, q_hi)` | float array | Like `compute_phase_offset` but filters to step cycles whose `step_cycle_mean_speed` ∈ [q_lo, q_hi); used in Cell 18 accumulation loop |
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
| Swing/stance convention diagnostic | 17 | 5-bout diagnostic: tip Z vs tibia angle vs Hilbert phase vs swing_stance → `step_cycle_swing_diagnostic.pdf` |
| Phase coordination (onset-based, per step-cycle speed) | 18 | Polar histograms × 4 **step-cycle** speed quartiles + per-fly R-matrix; one bout can span multiple quartiles |
| Single-bout coordination diagnostic | 19 | Set `DIAG_BOUT`; produces (1) Z-trace + onset timeline and (2) per-cycle polar grid for one selected bout |
| Standard visualisations | 20–27 | Scree plot, loadings heatmap, PCA scatter panels, joint-vs-phase grids, leg coordination matrix, tripod strength |
| Simple whole-session UMAP | 27–31 | cuML UMAP on joint-angle feature matrix; 3D scatter, per-bout trajectories |
| Per-fly summaries | 32 | Speed distributions, R-matrix heatmaps, per-fly trajectory overlays |
| **Paper figures** | **33–36** | **Phase-colored PCA portrait (Cell 33), own-phase portrait (Cell 34), activity-binned trajectory (Cell 35), range-of-motion bar chart (Cell 36)** |
| Segment UMAP | 37–43 | DeAngelis-style windowed segments: sample (Cell 38), guard (Cell 39), per-joint UMAP (Cell 41–43) |

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

Cells 33–35 (paper figures) filter frames using:

```python
_activity_mask = df_valid['mean_abs_vel'] > df_valid['mean_abs_vel'].quantile(0.40)
```

This keeps the **top 60% most-active frames** by mean absolute joint angular velocity. Using `mean_abs_vel` rather than a raw `forward_speed` threshold is more robust because joint velocity is directly interpretable and unit-consistent across sessions. `mean_abs_vel` is unit-agnostic for the purpose of ranking activity.

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

`MODEL_SCALE = 100.0` is defined in Cell 4 for reference. `forward_speed` is in mm/s: the ÷100 model scale and ×1000 m→mm conversion give a net ×10 factor applied to the raw gradient output.

---

## UMAP analyses

There are **two separate UMAP analyses** in the notebook. They answer different questions and operate on different inputs.

> **What is UMAP?** UMAP (Uniform Manifold Approximation and Projection) is a nonlinear dimensionality reduction method. Unlike PCA, it preserves local neighborhood structure rather than global variance. This makes it better at revealing discrete clusters or curved manifolds. Both analyses here use the GPU-accelerated version from RAPIDS cuML and require an NVIDIA GPU.

---

### Analysis 1 — Simple whole-session UMAP (Cells 27–31)

**Question it answers:** What does the full joint-angle space look like? Do different flies, speeds, or gait states separate?

#### What goes in

The same feature matrix `X` that is fed to PCA (Cell 19): joint angles ± their first derivatives, one row per frame, NaN rows removed.

Before running UMAP, `X` is standardized with `StandardScaler` (zero mean, unit variance per feature) to give `X_scaled`. This is the same transformation PCA already applied internally, but Cell 27 fits a **new, separate scaler** rather than reusing the one from Cell 13. The result is numerically identical — both scalers are fit on the same `X` — but keeping them separate means the UMAP section can be run independently without requiring Cell 13 to have run first. The `scaler_umap` object is only needed if you later want to transform new data into the same space.

Shape: `(N_valid_frames, N_features)` — e.g. 78k frames × 48 features for the `'core'` joint set with derivatives.

#### Processing (Cell 27)

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

#### Figures produced

| Cell | Figure | Coloring |
|---|---|---|
| 28 | 3D scatter (UMAP1, 2, 3) | `mean_abs_vel` (joint angular velocity) |
| 29 | 3D scatter (UMAP1, 2, 3) | Fly ID (discrete colors) |
| 30 | 2×2 panel of 2D scatters (UMAP1 vs UMAP2) | n_legs_stance, forward_speed, bout, fly ID |
| 31 | Grid of per-bout 2D trajectories | Frame index (time within bout) |

#### Parameters

| Parameter | Value | What it controls |
|---|---|---|
| `n_components=6` | 6 | Embedding dimensionality. Visualize any 3 at a time. More components = more information retained. |
| `n_neighbors=15` | 15 | Size of the local neighborhood. **Small** (5–10) = fine local clusters, may fragment. **Large** (30–100) = broad global topology, smoother. |
| `min_dist=0.01` | 0.01 | How tightly points are allowed to pack together. **Small** (0.01) = tight clusters, useful for finding discrete states. **Large** (0.5) = more uniform spread. |
| `metric='euclidean'` | euclidean | Distance metric in the input space. |

#### How to interpret

- **If points form a ring or torus:** the fly's gait is rhythmically structured. Rotation around the ring = progression through the stride cycle. A torus would indicate two coupled cycles (e.g., L and R tripods).
- **If it forms discrete blobs:** there are distinct gait states (e.g., tripod A, tripod B, turns). Check `n_legs_stance` coloring — if blobs correspond to 3/3 alternation they are tripod phases.
- **If fly IDs separate:** there is substantial inter-individual variability in joint kinematics. If they overlap, kinematics are consistent across flies.
- **Cell 31 (per-bout trajectories):** each panel shows one bout's trajectory through UMAP space. If trajectories form loops, the gait is periodic; if they drift, there is non-stationary behavior (acceleration, turns).

---

### Analysis 2 — Segment UMAP (Cells 37–43)

**Question it answers:** What are the distinct movement motifs in the data, and how are they distributed across gait cycle, speed, and individuals? This analysis was introduced by DeAngelis et al. (eLife 2019, paper 46409) for whole-body Drosophila locomotion analysis.

**Key difference from Analysis 1:** Instead of embedding single frames, each data point is a **short time window** (200 ms). This means the embedding captures temporal dynamics — a full stride cycle — not just a snapshot of joint angles.

#### Cell 37 — Configuration

All parameters for the segment UMAP are set here:

| Parameter | Default | What it controls |
|---|---|---|
| `HALF_WIN_MS = 100` | 100 ms | Half-width of each window. Total window = 2×100+1 = 201 ms ≈ 1–2 stride cycles at typical walking speed. The paper used 100 ms at 150 Hz (31 frames); at 800 Hz this gives 161 frames. |
| `HALF_WIN_FRAMES` | 80 | Computed from HALF_WIN_MS × FPS. Each segment spans frames [center−80, center+80]. |
| `WIN_LENGTH` | 161 | Total frames per segment (2 × HALF_WIN_FRAMES + 1). |
| `N_SEGMENTS_TARGET` | 100 000 | Number of segments to randomly sample. More = better coverage of behavior space but more memory. Reduce if GPU OOM. |
| `SEGMENT_SEED` | 42 | Random seed for reproducibility. |
| `SEG_UMAP_N_NEIGHBORS` | 15 | UMAP neighborhood size (same meaning as Analysis 1). |
| `SEG_UMAP_MIN_DIST` | 0.1 | UMAP cluster packing (same meaning as Analysis 1). |
| `SEG_UMAP_COMPONENTS` | 3 | 3D for full-body and per-joint-type; 2D for per-(leg × joint). |

Cell 37 also prints the **memory estimate** for the full-body feature matrix (~2 GB raw at 100k segments). If you run out of GPU memory, decrease `N_SEGMENTS_TARGET` or `HALF_WIN_MS`.

#### Cell 38 — `sample_segments()` — extracting windows

This function randomly samples `N_SEGMENTS_TARGET` time windows from the data and extracts the **egocentric foot-tip (x, y) positions** for each.

**What goes in:**
- `bout_dict`: the raw HDF5 data
- `HALF_WIN_FRAMES`: half-window length
- `leg_tip_site_indices`: maps each leg to its column in `xpos_egocentric`

**What it does:**
1. Collects every valid center frame across all bouts (frames far enough from bout edges that a full window fits)
2. Randomly samples `N_SEGMENTS_TARGET` of them (with fixed seed)
3. For each sampled frame, extracts a `(WIN_LENGTH, 12)` window — 6 legs × (x, y) — from `xpos_egocentric`

**What comes out:**
- `segments_raw`: shape `(N, 161, 12)` — N segments, 161 frames, 12 variables (6 legs × x,y)
- `seg_meta`: dict with arrays `fly_id`, `bout_id`, `bout_idx`, `center_frame` — one entry per segment, used to look up behavioral variables for coloring

Also defined here: `standardize_segments()`, which performs the two-stage normalization described below.

#### Two-stage standardization

Before UMAP, each segment is standardized in two steps (following DeAngelis et al.):

**Stage 1 — subtract per-segment temporal mean:**
Each variable's mean across the 161 frames of that window is subtracted. This removes the absolute position of the fly (where it is in the arena) and keeps only the *relative motion* within the window. After this, all segments are centered around zero regardless of where the fly was standing.

**Stage 2 — z-score across segments per (timepoint, variable):**
At each of the 161 time points, and for each of the 12 variables, the mean and standard deviation are computed across all N segments. Each value is then divided by that std. This equalizes the variance contributed by each variable and each point in the stride cycle, so that slow joints (small amplitude) contribute equally to the UMAP as fast joints (large amplitude).

#### Cell 39 — Guard + run full-body UMAP

After standardization, segments are flattened: `(N, 161, 12)` → `(N, 1932)`. Then cuML UMAP reduces this to 3D.

**What comes out:**
- `segments_std`: shape `(N, 161, 12)` — standardized segments
- `X_seg`: shape `(N, 1932)` — flattened, ready for UMAP
- `umap_seg`: shape `(N, 3)` — 3D full-body segment embedding

#### Cell 40 — Behavioral coloring for full-body UMAP

This cell attaches behavioral labels to each segment using `seg_meta['center_frame']` and `seg_meta['bout_id']` as keys into `df_valid`. Three coloring variables are computed:
- `seg_n_stance`: n_legs_stance at the segment center frame
- `seg_mean_z`: mean foot-tip Z (height) across all 6 legs at center
- `fly_codes`: integer fly ID

Three 3D scatter figures are saved:
- Colored by mean limb Z height
- Colored by fly ID
- Colored by n_legs_stance

#### Cell 41 — `sample_segments_qpos()` — joint-angle windows

Uses the **same** `(bout_idx, center_frame)` pairs from `seg_meta` to extract joint-angle time windows from `qpos`. This ensures all three UMAP analyses (full-body, per-joint, per-type) are computed on the same set of segments and can be directly compared.

**What goes in:** `bout_dict`, `seg_meta`, `joint_list`

**What comes out:**
- `qpos_segs`: shape `(N, 161, N_joints)` — joint angles over time per segment
- `qpos_col_names`: list of `'{leg}_{joint}'` strings (axis-2 labels)

#### Cell 42 — Per-(leg × joint) UMAP

For every combination of leg and joint (e.g., T1_left / tibia, T2_right / coxa), a **separate 2D UMAP** is run.

**What goes in:** For each (leg, joint): a single column from `qpos_segs`, giving shape `(N, 161)` — one time series per segment.

**Processing:** standardize → run 2D cuML UMAP.

**What comes out:**
- `umap_joints`: dict mapping `(leg, joint) → (N, 2)` embedding
- Two grid figures (rows = legs, columns = joint types), colored by n_legs_stance and fly ID

**How to interpret:** Each panel is a 2D map of how that specific joint moves over time. If a joint has a clear periodic motion (e.g., tibia flexion during walking), its embedding will form a ring or arc. Joints with noisier or less structured dynamics appear as a cloud. Coloring by n_legs_stance shows where in stance/swing the joint is at particular parts of its kinematic range.

#### Cell 43 — Per-joint-type UMAP

For each joint type (e.g., `'tibia'`), all 6 legs' time series are combined and a single **3D UMAP** is run.

**What goes in:** For joint type `'tibia'`: columns from `qpos_segs` for all 6 legs, giving shape `(N, 161, 6)` → flattened to `(N, 966)`.

**What comes out:**
- `umap_joint_types`: dict mapping `joint_name → (N, 3)` embedding
- One multi-row figure: rows = joint types, columns = 3 coloring schemes (mean Z, fly ID, n_legs_stance)

**How to interpret:** By pooling all 6 legs, this reveals the **shared kinematic template** for each joint type regardless of which leg it belongs to. If the tibia UMAP forms a ring, all legs go through the same flexion/extension cycle in the same order. Separation by fly ID in this embedding indicates inter-individual differences in the joint's kinematic profile.

---

### Summary: which UMAP to use for what

| Question | Use |
|---|---|
| Does overall gait structure vary by fly or speed? | Analysis 1, Cell 29 |
| What does the trajectory through kinematic space look like per bout? | Analysis 1, Cell 31 |
| Are there discrete gait states (motifs)? | Analysis 2, full-body (Cells 39–40) |
| Which joints have the most structured rhythmic dynamics? | Analysis 2, per-joint grid (Cell 42) |
| Is the kinematic template for a joint consistent across legs? | Analysis 2, per-joint-type (Cell 43) |
