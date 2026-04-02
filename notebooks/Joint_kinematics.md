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
Cell 5  → define Hilbert phase functions
Cell 6  → define reduction functions
Cell 7  → define visualisation functions
Cell 8  → define coordination functions
─────────────────────────────────────────── (run once above; below is the actual pipeline)
Cell 9  → load data
Cell 10 → build joint index + leg-tip sites
Cell 11 → build flat DataFrame  →  df
Cell 12 → compute all features  →  df
Cell 13 → feature matrix + NaN filter  →  df_valid
Cell 14 → run PCA  →  pca_result
Cell 15 → add PC1…PC10 to df_valid
```

**Then run any analysis section** (each is independent after Cell 15):

| Section | Cells |
|---|---|
| Distributions overview (speed, heading, height, leg spread) | 16 |
| Step-cycle diagnostic | 17 |
| Phase coordination (onset-based, per step-cycle speed) | 18 |
| Single-bout coordination diagnostic | 19 |
| Standard visualisations (scree, loadings, PCA trajectories, leg coordination) | 20–28 |
| Simple whole-session UMAP | 29–33 |
| Per-fly summaries | 34 |
| Paper figures (speed-binned PCA, ROM, 3-D stance trajectory) | 35–41 |
| Segment UMAP (DeAngelis et al. style) | 42–50 |
| Step-Cycle Joint-Angle UMAP | 51–54 |
| Paper Figure 2: Walking Dynamics | 55–56 |
| Speed–ROM correlation | 57–62 |

---

## Configuration — Cell 2

**This is the only cell most users need to edit.**

| Variable | Default | Purpose |
|---|---|---|
| `H5_PATH` | *(set per user)* | Path to the combined IK output `.h5` file |
| `ACTIVE_JOINT_SET` | `'core'` | Which joints to analyse: `'core'` (4), `'main'` (7), or `'full'` (11 joints/leg) |
| `FPS` | `800` | Camera frame rate in Hz |
| `SMOOTH_SIGMA` | `6` | Gaussian smoothing sigma for Hilbert phase computation |
| `REFERENCE_LEG` | `'T1_right'` | Leg used as phase reference for coordination plots |
| `HEADING_BODIES` | `['thorax']` | Body used for heading and speed computation |
| `LEGS` | 6-element list | `['T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right']` |
| `OUTPUT_DIR` | `./output/joint_kinematics` | Where PDF figures are saved |

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
        ▼ Cell 9  load_ik_bouts()
  bout_dict        ← nested dict: bout_dict['bout_000']['qpos'] etc.
  bout_keys        ← sorted list of 'bout_NNN' strings
  fly_ids          ← array (N_bouts,) of fly identifier strings
  clip_lengths     ← array (N_bouts,) of frames per bout
  names_qpos       ← joint names in qpos column order
  egocentric_site_names  ← 50 site names for xpos_egocentric
        │
        ▼ Cell 10  build_joint_index() + find_leg_tip_site_indices()
  joint_list       ← list of (leg, joint, qpos_column_index) triples
  leg_tip_site_indices  ← {leg: index into xpos_egocentric[:,idx,:]}
        │
        ▼ Cell 11  bouts_to_dataframe()
  df               ← flat DataFrame, all bouts stacked
                      one row per frame; 'bout_id' column groups bouts
        │
        ▼ Cell 12  8-step feature pipeline
  df               ← same DataFrame + all computed columns
        │
        ▼ Cell 13  get_feature_matrix()
  X                ← (N_valid_frames × N_features) float array
  df_valid         ← df with NaN rows removed, row-aligned to X
  valid_mask       ← boolean mask from df → df_valid
        │
        ▼ Cell 14  run_pca()
  pca_result       ← (N_valid_frames × 10) float array
  pca              ← fitted sklearn PCA object
  scaler           ← fitted StandardScaler
        │
        ▼ Cell 15  add columns to df_valid
  df_valid         ← + PC1 … PC10 columns
        │
        ▼ Cells 16–62  Analysis & figures
```

---

## Key variables — where they live

| Variable | Cell | Description |
|---|---|---|
| `bout_dict` | 9 | Full nested dict. Raw data: `bout_dict['bout_000']['qpos']` (T×N_joints), `['xpos']` (T×N_bodies×3), `['xpos_egocentric']` (T×50×3) |
| `bout_keys` | 9 | `['bout_000', 'bout_001', …]` — use to iterate over bouts |
| `fly_ids` | 9 | String array identifying which fly each bout belongs to |
| `clip_lengths` | 9 | Number of frames in each bout |
| `names_qpos` | 9 | E.g. `['free','free',…,'tibia_T1_left',…]` — needed to find joint indices |
| `names_xpos` | 9 | Body names for `xpos` array (e.g. `'thorax'` at index 1); includes `'claw_T1_left'` etc. |
| `egocentric_site_names` | 9 | 50 site names for `xpos_egocentric` (e.g. `'tracking[T1L_TaTip]_fly'`) |
| `joint_list` | 10 | `[(leg, joint, qpos_col_idx), …]` — the canonical joint ordering used throughout |
| `leg_tip_site_indices` | 10 | `{'T1_left': 18, 'T1_right': 25, …}` — index into `xpos_egocentric` axis 1 |
| `df` | 11 | All frames from all bouts in one DataFrame. Key grouping column: `bout_id` |
| `df_valid` | 13 | `df` with rows dropped where any feature is NaN. Row-aligned to `pca_result` |
| `X` | 13 | Numeric feature array fed to PCA; shape `(N_valid_frames, N_features)` |
| `pca_result` | 14 | PC coordinates; shape `(N_valid_frames, 10)` |

---

## Key DataFrame columns

Columns in `df` and `df_valid`, grouped by when they are added:

**Cell 11 — raw data:**

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

**Cell 12 — computed features:**

| Column | Pipeline step | Description |
|---|---|---|
| `{leg}_{joint}_d1` | 1 | Angular velocity (rad/s) — Savitzky-Golay derivative |
| `{leg}_{joint}_d2` | 1 | Angular acceleration (rad/s²) |
| `{leg}_phase` | 2 | **Primary step-cycle phase** [-π, π]. Hilbert transform of mean-centered 3D tip speed. phase=0: mid-swing (peak speed); phase=±π: mid-stance |
| `{leg}_swing_stance` | 3 | 1 = stance, 0 = swing. Derived from Hilbert phase: `\|phase\| ≥ π/2` → stance. Clean by construction — no speed-threshold artefacts |
| `n_legs_stance` | 3 | Count of legs in stance (0–6) per frame |
| `forward_speed` | 4 | Thorax translation speed in **mm/s** |
| `heading` | 4 | Fly heading in radians [-π, π] (world frame: 0 = +X direction) |
| `turning_rate` | 4 | Rate of heading change in rad/s |
| `{leg}_phase_rel` | 5 | Phase of leg relative to `REFERENCE_LEG` [-π, π] |
| `mean_abs_vel` | 6 | Mean \|angular velocity\| across all joints, smoothed per bout (rad/s). **Activity proxy for filtering** |
| `step_cycle_id` | 7 | Integer step-cycle index within each bout (0-based, NaN outside first/last liftoff) |
| `step_cycle_mean_speed` | 7 | Mean `forward_speed` over the containing step cycle (mm/s); NaN for partial cycles at bout edges |

**Cell 15 — PCA:**

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
| `compute_n_legs_stance(df, legs)` | `n_legs_stance` | Row-sum of `{leg}_swing_stance` columns |
| `compute_heading_and_velocity(df, fps, body)` | `forward_speed`, `heading`, `turning_rate` | Per-bout; speed in mm/s; Gaussian-smoothed positions |
| `compute_relative_phases(df, legs, reference_leg)` | `{leg}_phase_rel` | Wrapped circular difference vs reference leg |
| `swing_onsets_from_stance(series)` | frame index array | Returns indices where `swing_stance` transitions 1→0 (stance→swing) |
| `compute_step_cycle_speed(df, fps, ref_leg)` | `step_cycle_id`, `step_cycle_mean_speed` | Step cycles defined by `liftoff_from_hilbert` on ref leg's Hilbert phase |
| `compute_phase_offset(onsets_a, onsets_b, fps)` | float array | Onset-based phase of leg A relative to leg B's stride cycle |
| `compute_phase_offset_by_cycle(ref_onsets, leg_onsets, cycle_speeds, q_lo, q_hi)` | float array | Like `compute_phase_offset` but filtered to step cycles in speed range [q_lo, q_hi) |
| `mean_resultant_length(phases)` | scalar R ∈ [0,1] | Circular synchrony metric: 1 = perfect, 0 = uniform |

### Cell 5 — Hilbert Phase Functions

| Function | Output | Notes |
|---|---|---|
| `compute_hilbert_phase_bout(positions_xyz, fps, smooth_sigma)` | `phase (T,), speed (T,)` | Hilbert phase from 3D tip speed. Mean-centers speed before transform so phase spans full [-π, π]. phase=0: mid-swing; phase=±π: mid-stance. Per-bout, NaN-safe |
| `swing_onsets_from_hilbert(phase)` | frame index array | Step-cycle boundaries at 2π increments in unwrapped phase (mid-swing → mid-swing) |
| `liftoff_from_hilbert(phase)` | frame index array | Upward crossing of phase = -π/2 → liftoff (swing onset). Used by `compute_step_cycle_speed` and all coordination cells |
| `landing_from_hilbert(phase)` | frame index array | Upward crossing of phase = +π/2 → landing (stance onset) |
| `compute_hilbert_phases(df, legs, fps, smooth_sigma)` | `{leg}_phase` columns | Runs `compute_hilbert_phase_bout` per bout per leg using world-frame tip XYZ |
| `compute_swing_stance_from_hilbert(df, legs)` | `{leg}_swing_stance` columns | Derives binary swing/stance from Hilbert phase: swing if `\|phase\| < π/2`, stance otherwise. Replaces speed-threshold approach |

### Cells 6, 7, 8 — Reduction, visualisation & coordination

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
| Function definitions | 3, 4, 5, 6, 7, 8 | Define all functions (no outputs) |
| **Main pipeline** | **9 → 15** | **Load data → build df → preprocess → PCA. Run this every session.** |
| Distributions overview | 16 | Speed, heading (corridor-axis deviation), height, leg spread — KDE + per-bout strip plots |
| Step-cycle diagnostic | 17 | 4-panel validation: 5 short + 5 long bouts → `step_cycle_diagnostic.pdf` |
| Phase coordination | 18 | Onset-based polar histograms × 3 step-cycle speed tertiles + per-fly R-matrix |
| Single-bout coordination diagnostic | 19 | Set `DIAG_BOUT`; Z-trace + liftoff timeline and per-cycle polar grid for one bout |
| Standard visualisations | 20–28 | Scree (20), loadings heatmap (21), PCA scatter panels (22–24), egocentric step arcs (25–26), joint-vs-phase grid (27–28) |
| Simple whole-session UMAP | 29–33 | GPU imports (29), 3-D UMAP scatter with `COLOR_BY` toggle (30), fly ID encoding (31), UMAP color variables (32), per-bout trajectories (33) |
| Per-fly summaries | 34 | Speed distributions, R-matrix heatmaps, per-fly trajectory overlays |
| **Paper figures** | **35–41** | Speed-binned PCA (35–36), 3-D speed-binned PCA/trajectory (37–38), Range of Motion (39–40), ROM config (41) |
| **Segment UMAP** | **42–50** | DeAngelis-style windowed segments; see detail below |
| **Step-Cycle UMAP** | **51–54** | Step-cycle joint-angle UMAP; see detail below |
| **Paper Figure 2** | **55–56** | Walking Dynamics Overview (config + figure) |
| **Speed–ROM correlation** | **57–62** | Approaches A/B/C + paper figure + scatter |

---

## Phase convention

The Hilbert phase of mean-centered 3D tip speed spans [-π, π]:

| Phase | Event | `swing_stance` |
|---|---|---|
| ≈ -π/2 | **liftoff** (swing onset) | 0→1 transition (→ swing) |
| 0 | **mid-swing** (peak speed) | 0 (swing) |
| ≈ +π/2 | **landing** (stance onset) | 1→0 transition (→ stance) |
| ±π | **mid-stance** (speed trough) | 1 (stance) |

`{leg}_swing_stance = 1` (stance) when `|phase| ≥ π/2`; `= 0` (swing) when `|phase| < π/2`.

`liftoff_from_hilbert(phase)` detects upward crossings of -π/2 — used by Cell 12 pipeline step 7 (`compute_step_cycle_speed`) and all coordination cells (18, 19).

---

## Activity filter

Cells 35–36 (paper figures) filter frames using:

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
| World-frame leg-tip XYZ (`{leg}_tip_x/y/z_world`) | model-metres | From `xpos` (`claw_T{1,2,3}_{left,right}` bodies); ×10 → real mm |
| `step_cycle_mean_speed` | mm/s | Mean `forward_speed` over one step cycle |

---

## UMAP analyses

There are **three separate UMAP analyses** in the notebook, answering different questions on different inputs.

> **What is UMAP?** UMAP (Uniform Manifold Approximation and Projection) is a nonlinear dimensionality reduction method. Unlike PCA, it preserves local neighborhood structure rather than global variance. All analyses here use the GPU-accelerated version from RAPIDS cuML (requires an NVIDIA GPU).

---

### Analysis 1 — Simple whole-session UMAP (Cells 29–33)

**Question it answers:** What does the full joint-angle space look like? Do different flies, speeds, or gait states separate?

#### What goes in

The same feature matrix `X` that is fed to PCA (Cell 13): joint angles ± their first derivatives, one row per frame, NaN rows removed. Standardized with `StandardScaler` → `X_scaled`.

Shape: `(N_valid_frames, N_features)` — e.g. 78k frames × 48 features for the `'core'` joint set with derivatives.

#### Processing (Cells 29–30)

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

#### Cell 30 — COLOR_BY toggle

Cell 30 produces a 3-D scatter (UMAP dimensions 1, 2, 3) with a configurable coloring:

```python
COLOR_BY      = 'cycle_speed'  # 'instant_vel' | 'cycle_speed' | 'n_legs_stance'
                               # 'phase_rel'   | 'phase'
PHASE_REL_LEG = 'T1_right'    # used when COLOR_BY == 'phase_rel'
PHASE_LEG     = 'T1_left'     # used when COLOR_BY == 'phase'
```

| Option | Variable colored | Colormap |
|---|---|---|
| `'instant_vel'` | `forward_speed` at each frame | turbo, data-range |
| `'cycle_speed'` | `step_cycle_mean_speed` | turbo, data-range |
| `'n_legs_stance'` | `n_legs_stance` (0–6) | RdYlGn, 0–6 fixed |
| `'phase_rel'` | `{PHASE_REL_LEG}_phase_rel` | twilight, ±π |
| `'phase'` | `{PHASE_LEG}_phase` — Hilbert phase | twilight, ±π |

#### Remaining cells

| Cell | Figure | Coloring |
|---|---|---|
| 31 | Fly ID integer encoding (sets `codes`, `categories`) | — |
| 32 | Multi-panel 2D scatter (UMAP1 vs UMAP2) | n_legs_stance, forward_speed, bout index, fly ID |
| 33 | Grid of per-bout 2D trajectories | Frame index (time within bout) |

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
- **Cell 33 (per-bout trajectories):** loops = periodic gait; drift = non-stationary behavior.

---

### Analysis 2 — Segment UMAP, DeAngelis style (Cells 42–50)

**Question it answers:** What are the distinct movement motifs in the data, and how are they distributed across gait cycle, speed, and individuals? Introduced by DeAngelis et al. (eLife 2019).

**Key difference from Analysis 1:** Each data point is a **short time window** (200 ms total), capturing temporal dynamics over a full stride cycle rather than a single-frame snapshot.

#### Cell 42 — Configuration

All parameters for the segment UMAP are set here, plus the **shared coloring toggle** used by Cells 45, 47, 48, and 50:

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

**COLOR_BY_JT options** (same as Cell 30 `COLOR_BY`):
- `'instant_vel'`, `'cycle_speed'`, `'n_legs_stance'`, `'phase_rel'`, `'phase'`

Cell 42 also prints the **memory estimate** (~2 GB raw at 100k segments). Reduce `N_SEGMENTS_TARGET` or `HALF_WIN_MS` if you run out of GPU memory.

#### Cell 43 — `sample_segments()` — extracting windows

Randomly samples `N_SEGMENTS_TARGET` time windows; extracts **egocentric foot-tip (x, y) positions** for 6 legs.

**What comes out:**
- `segments_raw`: shape `(N, 161, 12)` — N segments × 161 frames × 12 variables (6 legs × x,y)
- `seg_meta`: dict with arrays `fly_id`, `bout_id`, `bout_idx`, `center_frame`

Also defines `standardize_segments()` (two-stage normalization; see below).

#### Two-stage standardization

**Stage 1 — subtract per-segment temporal mean:** removes absolute foot position, keeps relative motion within each window.

**Stage 2 — z-score across segments per (timepoint, variable):** equalizes variance across slow and fast joints / all points in the stride cycle.

#### Cell 44 — Guard + run full-body UMAP

Standardizes segments, flattens `(N, 161, 12)` → `(N, 1932)`, then runs cuML UMAP.

**What comes out:**
- `segments_std`: `(N, 161, 12)` — standardized
- `X_seg`: `(N, 1932)` — flattened
- `umap_seg`: `(N, 3)` — 3-D full-body embedding

#### Cell 45 — Full-Body UMAP: Helpers and Figures

Defines `_seg_col(col_or_arr)` helper and produces:
- **Figure 1:** single 3-D scatter colored by the `COLOR_BY_JT` toggle
- **Figure 2:** fixed 2×2 overview panels (limb Z height, fly ID, n_legs_stance, bout index)

#### Cell 46 — `sample_segments_qpos()` — joint-angle windows

Uses the **same** `(bout_idx, center_frame)` pairs from `seg_meta` to extract joint-angle time windows from `qpos`.

**What comes out:**
- `qpos_segs`: `(N, 161, N_joints)` — joint angles per segment
- `qpos_col_names`: list of `'{leg}_{joint}'` strings

#### Cell 47 — Recompute Colors (lightweight)

**Change `COLOR_BY_JT` in Cell 42, then re-run this cell alone** to update coloring without repeating heavy UMAP computation.

#### Cell 48 — Per-(Leg × Joint) UMAP: Run + Plot

For every (leg, joint) combination, runs a separate **2-D cuML UMAP** on single-channel time series `(N, 161)`.

**What comes out:** `umap_joints`: dict mapping `(leg, joint) → (N, 2)` embedding.

#### Cell 49 — Per-Joint-Type UMAP: Run (heavy)

For each joint type (e.g., `'tibia'`), pools all 6 legs' time series → shape `(N, 161×6)` → runs **3-D cuML UMAP**.

**What comes out:** `umap_joint_types`: dict mapping `joint_name → (N, 6)` embedding (6-component).

#### Cell 50 — Per-Joint-Type UMAP: Plot (lightweight)

**Re-run after changing `COLOR_BY_JT` and Recompute Colors (Cell 47)** to update plots without re-running the UMAP fit.

---

### Analysis 3 — Step-Cycle Joint-Angle UMAP (Cells 51–54)

**Question it answers:** What are the distinct gait patterns organized by complete stride cycles?

**Key difference from Analysis 2:** Segments are **T1_left step cycles** (defined by `liftoff_from_hilbert`), not random windows. Features are **joint angles**, not egocentric foot positions. Phase-resampling makes fast and slow cycles directly comparable.

#### Cell 51 — Configuration

| Parameter | Default | What it controls |
|---|---|---|
| `N_PHASE_BINS` | 50 | Phase grid points per cycle. Each cycle resampled to exactly this many points. |
| `JOINT_SET_NAME` | `'main'` | Joint set to use: `'core'` / `'main'` / `'full'` |
| `MIN_CYCLE_FRAMES` | 30 | Skip cycles shorter than this (≈37 ms at 800 Hz; filters partial cycles) |
| `N_CYCLES_MAX` | `None` | Optional cap on cycle count; `None` = use all |
| `SC_UMAP_COMPONENTS` | 3 | Embedding dimensionality |
| `SC_UMAP_N_NEIGHBORS` | 15 | UMAP neighborhood size |
| `SC_UMAP_MIN_DIST` | 0.10 | UMAP cluster packing |

#### Cell 52 — `sample_step_cycle_segments()` — extracting cycles

Iterates T1_left step cycles within each bout (using `liftoff_from_hilbert`). For each cycle `[t0, t1)`:
1. Uses T1_left Hilbert phase to define the phase grid [0, 2π] with `N_PHASE_BINS` points
2. For each leg × each joint, interpolates the angle onto the phase grid using `np.interp(..., period=2π)`
3. Concatenates into a single feature vector: `N_PHASE_BINS × n_legs × n_joints` dims

Phase-based resampling (not time-based) makes slow and fast cycles directly comparable.

**What comes out:**
- `X_raw_sc`: `(N_cycles, N_PHASE_BINS × n_legs × n_joints)` float32
- `sc_meta`: list of dicts with `fly_id`, `bout_id`, `speed`, `n_legs_stance`, `duration`

#### Cell 53 — Standardize + Run

Two-stage standardization:
1. **Per-cycle DC removal:** subtract the phase-axis mean per channel
2. **Cross-cycle z-score:** z-score each feature across all cycles

Then runs cuML UMAP → `umap_sc`: `(N_cycles, SC_UMAP_COMPONENTS)`.

#### Cell 54 — Visualisation

Produces:
- **3-D scatter grid** (4 panels): colored by step-cycle speed, n_legs_stance, fly ID, cycle duration
- **2-D projection grid** (4 rows × 3 pairs): UMAP1×2, UMAP1×3, UMAP2×3

---

### Summary: which UMAP to use for what

| Question | Use |
|---|---|
| Does overall gait structure vary by fly or speed? | Analysis 1, Cell 30 |
| What does the trajectory through kinematic space look like per bout? | Analysis 1, Cell 33 |
| Are there discrete gait states (motifs) in full-body movement? | Analysis 2, full-body (Cells 44–45) |
| Which joints have the most structured rhythmic dynamics? | Analysis 2, per-joint grid (Cell 48) |
| Is the kinematic template for a joint consistent across legs? | Analysis 2, per-joint-type (Cells 49–50) |
| How do complete stride cycles cluster by speed or gait mode? | Analysis 3, step-cycle UMAP (Cells 53–54) |
