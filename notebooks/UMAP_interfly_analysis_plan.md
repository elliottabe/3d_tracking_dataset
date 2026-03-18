# UMAP Inter-Fly Variation: Investigation and Analysis Plan

*Generated from session on 2026-03-09. Companion to `Joint_Kinematics_Analysis.ipynb`.*

---

## Context

The `Joint_Kinematics_Analysis.ipynb` notebook runs three UMAP analyses on free-walking *Drosophila* joint-angle data:

- **Analysis 1** (Cells 28–32): whole-session UMAP on the per-frame feature matrix `X` (joint angles ± derivatives), one row per frame
- **Analysis 2** (Cells 41–49): DeAngelis-style segment UMAP on 200 ms windows of egocentric foot positions
- **Analysis 3** (Cells 50–53): step-cycle UMAP on phase-resampled joint angles, one row per complete T1_left stride

The **current** dataset is `ik_output_combined_v1.h5` — 198 bouts, 7 sessions (5 WT males, 2 WT females), ~78k frames total. The **target** dataset is ~20 sessions (10 WT males, 10 WT females), ~220k frames, ~560 bouts. The plan is written to work at both scales; scalability notes are collected in the dedicated section at the end.

**The observed problem:** inter-fly identity is captured as the first UMAP dimension across all three analyses, dominating over gait-related structure (speed, phase, stance count).

---

## Investigation Summary

### Step 1 — Can body size be measured from the IK output?

No. The 50 sites in `xpos_egocentric` are forward-kinematics outputs of the shared STAC MuJoCo model (`fruitfly_v1_free.xml`), which has **fixed segment lengths**. All seven flies return identical inter-landmark distances (e.g., T1 tibia = 0.051, T2 femur = 0.083 model-units for every fly). Body size cannot be assessed from the IK output.

### Step 2 — Raw anatomy from JARVIS data3D CSVs

Distances computed from the raw 3D keypoints **before** preprocessing, across all six sessions:

| Measurement | Males (n≈4) | Females (n=2) | CV% across all | Notes |
|---|---|---|---|---|
| Antenna_Base → Scutellum | ~13.1–13.4 | **14.5–14.9** | 5.7% | Most reliable body-axis proxy |
| Scutellum → EyeL | ~10.5–10.7 | **11.7–12.0** | 5.8% | |
| WingL_base → WingR_base | ~6.4–7.2 | **7.4–7.6** | 5.8% | |
| T1 tibia (Tro→FeTi) | ~4.5–4.9 | ~4.7–4.9 | **2.0%** | Most stable single-leg metric |
| T2 femur (Tro→FeTi) | ~5.4–6.9 | ~6.4–7.3 | **14.0%** | High CV likely due to tracking noise at trochanter |

**Key finding:** Females are ~11% larger than males in head/thorax dimensions. Within-sex size variation is small: ~2–4% in tibias (reliable), rising to 8–14% in femurs (inflated by occlusion noise at the trochanter).

### Step 3 — Do Procrustes scales account for body size?

The preprocessing HDF5 files (`preprocessed_bout_v1_free_walking.h5`) each store `alignment_info/scales` — one scalar per bout. Scale ranges per session:

| Session | Scale range | Direction |
|---|---|---|
| WT female S5 | 0.01126–0.01194 | **smallest** → largest fly |
| WT female S7 | 0.01122–0.01217 | small → large fly |
| WT male S5 | 0.01141–0.01241 | medium |
| WT male S6 | 0.01198–0.01335 | medium |
| WT male S7 | 0.01211–0.01310 | medium |
| WT male S1 | 0.01225–0.01279 | medium |

The direction is correct (larger fly → smaller scale factor to fit fixed model). However, **within-fly scale variation spans 8–11%** — comparable to the between-sex size difference. This means per-bout Procrustes scaling is noisy and does not cleanly absorb body size. The female/male size difference partially survives into the joint-angle space.

### Step 4 — Are joint angle differences too large for body size alone?

The mean T2 femur angle differs by **0.37 rad (~21°)** between sessions in the IK output. To produce this from body-size alone, the tibia would need to be ~37% longer — but tibias vary by only 2–4%. **Body size cannot explain T2 joint angle differences of this magnitude.**

Additional evidence for behavioral origin:
- Forward speed varies ~2× between fastest (session1) and slowest (session7 morning)
- T2 (middle legs) mediates pitch, turning, and load sharing — the joints most sensitive to locomotion strategy, not anatomy
- T1 tibias (most stable anatomy, CV 2%) still show joint angle differences between sessions

---

## Interpretation

The inter-fly UMAP separation has **two stacked contributions** that need to be treated differently:

| Source | Magnitude | Survives Procrustes? | Biologically interesting? |
|---|---|---|---|
| Sex dimorphism (F vs M) | ~11% in body axis | Partially (noisy scale) | Yes — label it, don't remove it |
| Within-sex body size | ~2–4% in tibias | Mostly absorbed | Boring — can safely ignore |
| Mean posture (joint angle offset) | up to 0.37 rad in T2 | Fully present | Yes — reflect locomotion strategy |
| Speed / activity level | ~2× range | Fully present | Yes — most interesting axis |

**The goal** is not to "fix" the inter-fly axis as a nuisance, but to **decompose** it: show what fraction is sex/anatomy vs. posture/gait, then run a gait-centric UMAP that surfaces behavioral structure. Keep fly identity as a coloring variable throughout.

---

## Analysis Plan

All work is in `Joint_Kinematics_Analysis.ipynb`. New cells insert at the points noted. Prerequisites: Cells 1–14 (main pipeline) must have been run.

---

### Part A — Characterize the inter-fly variation (new section, after Cell 33)

**Goal:** Before fixing anything, rigorously describe *what* differs between flies. This becomes a figure in its own right.

#### Cell A1 — Per-fly mean joint angles heatmap

```python
# Per-fly mean of each joint angle, as a heatmap (joints × flies)
import seaborn as sns

joint_cols = [f"{leg}_{jnt}" for leg, jnt, _ in joint_list]
fly_means = df_valid.groupby('fly_id')[joint_cols].mean()

# Normalize each column to [0,1] so joints with different ranges are comparable
fly_means_norm = (fly_means - fly_means.mean()) / fly_means.std()

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    fly_means_norm.T,
    cmap='RdBu_r', center=0, vmin=-2, vmax=2,
    xticklabels=fly_means_norm.index,
    yticklabels=[f"{leg[:2]}{leg[3:5].upper()}_{jnt[:4]}" for leg, jnt, _ in joint_list],
    ax=ax
)
ax.set_title("Per-fly mean joint angle (z-scored across flies)")
ax.set_xlabel("Fly / Session")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'interfly_mean_posture_heatmap.pdf', bbox_inches='tight')
plt.show()
```

What to look for: systematic leg-pair patterns (e.g., all T2 joints shift together → posture shift, not leg-specific artifact). If T1/T2/T3 all shift uniformly → body-size residual. If only T2 → locomotion strategy.

#### Cell A2 — Per-fly speed distributions + posture PC

```python
# Speed distributions per fly (violin)
# Layout scales automatically: ≤10 flies → vertical x-axis labels;
# >10 flies → horizontal violin so labels stay readable
fig, axes = plt.subplots(1, 2, figsize=(max(14, len(fly_means) * 0.7), 5))

fly_order = df_valid.groupby('fly_id')['forward_speed'].median().sort_values().index.tolist()
_n_flies = len(fly_order)

if _n_flies <= 10:
    sns.violinplot(data=df_valid, x='fly_id', y='forward_speed',
                   order=fly_order, ax=axes[0], cut=0, inner='box')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].set_xlabel('Fly / Session')
    axes[0].set_ylabel('Forward speed (mm/s)')
else:
    # Horizontal layout for >10 flies
    sns.violinplot(data=df_valid, y='fly_id', x='forward_speed',
                   order=fly_order, ax=axes[0], cut=0, inner='box',
                   orient='h')
    axes[0].set_ylabel('')
    axes[0].set_xlabel('Forward speed (mm/s)')
axes[0].set_title('Forward speed distribution per fly')

# Right: project fly means onto PC1/PC2 of posture space
_fly_means_arr = fly_means.values  # (n_flies, n_joints)
_posture_pca = PCA(n_components=2).fit(_fly_means_arr)
_posture_coords = _posture_pca.transform(_fly_means_arr)
for i, fid in enumerate(fly_means.index):
    axes[1].scatter(_posture_coords[i, 0], _posture_coords[i, 1], s=120, label=fid)
    axes[1].annotate(fid, (_posture_coords[i, 0], _posture_coords[i, 1]),
                     fontsize=8, ha='left', va='bottom')
axes[1].set_xlabel(f'Posture PC1 ({_posture_pca.explained_variance_ratio_[0]*100:.0f}%)')
axes[1].set_ylabel(f'Posture PC2 ({_posture_pca.explained_variance_ratio_[1]*100:.0f}%)')
axes[1].set_title('Fly mean posture in PC space')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'interfly_speed_posture.pdf', bbox_inches='tight')
plt.show()
```

#### Cell A3 — Quantify how much variance is between- vs. within-fly

With a balanced 10M/10F design, η² alone is sufficient to rank joints by how much they separate flies. For ≥10 flies per sex you can also run a one-way ANOVA per joint and report effect sizes.

```python
# Eta-squared: fraction of total variance explained by fly identity per joint
# Also computes one-way ANOVA F and p-value for significance
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway

fly_codes = LabelEncoder().fit_transform(df_valid['fly_id'])
unique_fly_codes = np.unique(fly_codes)

eta_sq, anova_p = {}, {}
for col in joint_cols:
    vals = df_valid[col].values
    grand_mean = np.nanmean(vals)
    ss_total = np.nansum((vals - grand_mean) ** 2)
    groups = [vals[fly_codes == k] for k in unique_fly_codes]
    ss_between = sum(
        len(g) * (np.nanmean(g) - grand_mean) ** 2 for g in groups
    )
    eta_sq[col] = ss_between / ss_total if ss_total > 0 else 0
    _, anova_p[col] = f_oneway(*[g[~np.isnan(g)] for g in groups])

eta_df = pd.DataFrame({
    'joint':   list(eta_sq.keys()),
    'eta_sq':  list(eta_sq.values()),
    'anova_p': list(anova_p.values()),
}).sort_values('eta_sq', ascending=False)

fig, ax = plt.subplots(figsize=(max(12, len(eta_df) * 0.25), 5))
bars = ax.bar(range(len(eta_df)), eta_df['eta_sq'],
              color=['#d73027' if p < 0.01 else '#4575b4'
                     for p in eta_df['anova_p']])
ax.set_xticks(range(len(eta_df)))
ax.set_xticklabels(eta_df['joint'], rotation=90, fontsize=7)
ax.axhline(0.1, color='k', ls='--', lw=0.8, label='η²=0.10 threshold')
ax.set_ylabel('η² (fraction of variance between flies)')
ax.set_title('Per-joint between-fly variance fraction\n(red = ANOVA p<0.01)')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'interfly_eta_squared.pdf', bbox_inches='tight')
plt.show()

print("Joints with η² > 0.10 (ANOVA p<0.01):")
print(eta_df[(eta_df['eta_sq'] > 0.10) & (eta_df['anova_p'] < 0.01)].to_string())
```

This gives you a precise, quantitative answer to "which joints drive the inter-fly UMAP axis?" With 20 flies the ANOVA is well powered; joints that clear both η²>0.10 and p<0.01 are genuinely between-fly variables.

---

### Part B — Per-fly normalized UMAP (Analysis 1, Cells 28–32)

**Goal:** A UMAP where gait kinematics, not fly identity, drives the first dimension. Run alongside the original for direct comparison.

#### Cell B1 — Per-fly mean subtraction (insert after current Cell 28, before UMAP fit)

```python
# ── Per-fly normalization ─────────────────────────────────────────────────────
# Subtract per-fly mean from X_scaled (already StandardScaler-transformed).
# This removes systematic posture offset between flies while keeping
# within-fly kinematic variation intact.

fly_ids_valid = df_valid['fly_id'].values   # row-aligned to X and X_scaled
X_fly_norm = X_scaled.copy()
for fid in np.unique(fly_ids_valid):
    mask = fly_ids_valid == fid
    X_fly_norm[mask] -= X_fly_norm[mask].mean(axis=0)

# Optional (more aggressive): also divide by per-fly std to normalize amplitude
# Uncomment if fly-ID separation persists after mean subtraction:
# for fid in np.unique(fly_ids_valid):
#     mask = fly_ids_valid == fid
#     sd = X_fly_norm[mask].std(axis=0) + 1e-8
#     X_fly_norm[mask] /= sd

print(f"X_fly_norm shape: {X_fly_norm.shape}")
print("Running UMAP on fly-normalized matrix...")

# ── n_neighbors scaling ───────────────────────────────────────────────────────
# Rule of thumb: n_neighbors ≈ sqrt(N_frames) gives equivalent neighborhood
# density regardless of dataset size.
#   ~78k frames  → sqrt ≈ 280 → use 200 (current)
#   ~220k frames → sqrt ≈ 470 → 50–100 is sufficient and much faster
_n_neighbors_b = min(200, max(50, int(np.sqrt(len(X_fly_norm)))))
print(f"Using n_neighbors={_n_neighbors_b} for {len(X_fly_norm)} frames")

reducer_fly_norm = cuml.manifold.UMAP(
    n_components=6, n_neighbors=_n_neighbors_b, min_dist=0.8,
    metric='euclidean', random_state=70
)
umap_result_fly_norm = np.asarray(reducer_fly_norm.fit_transform(X_fly_norm))
print(f"Embedding shape: {umap_result_fly_norm.shape}")
```

#### Cell B2 — Side-by-side comparison: original vs. normalized

```python
# 2×4 comparison figure
# Row 1: original umap_result  |  Row 2: umap_result_fly_norm
# Columns: fly_id, forward_speed, n_legs_stance, T1_left_phase

fly_codes_arr = np.array([np.unique(fly_ids_valid).tolist().index(f) for f in fly_ids_valid])
_compare_specs = [
    (fly_codes_arr,                              'Fly ID',              'tab10',   {}),
    (df_valid['forward_speed'].values,           'Speed (mm/s)',        'turbo',   {}),
    (df_valid['n_legs_stance'].values,           'N legs stance',       'RdYlGn',  dict(vmin=0, vmax=6)),
    (df_valid['T1_left_phase'].values,           'T1L phase',           'twilight', dict(vmin=-np.pi, vmax=np.pi)),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for col_i, (cvals, clabel, cmap, ckw) in enumerate(_compare_specs):
    for row_i, (emb, title) in enumerate([
        (umap_result,          'Original (no fly correction)'),
        (umap_result_fly_norm, 'Fly-normalized'),
    ]):
        ax = axes[row_i, col_i]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=cvals, cmap=cmap,
                        s=1, alpha=0.4, rasterized=True, **ckw)
        plt.colorbar(sc, ax=ax, fraction=0.04)
        ax.set_title(f"{title}\n{clabel}", fontsize=8)
        ax.set_xlabel('UMAP 1', fontsize=7); ax.set_ylabel('UMAP 2', fontsize=7)
        ax.tick_params(labelsize=6)

plt.suptitle("UMAP comparison: original vs fly-normalized", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'umap_comparison_fly_norm.pdf', bbox_inches='tight')
plt.show()
```

**What to look for:** In the normalized version, fly_id scatter should show all colors interleaved (no separation). Speed and phase should become more structured. If fly blobs persist → upgrade to per-fly z-score (uncomment amplitude normalization in Cell B1).

---

### Part C — Per-fly normalized step-cycle UMAP (Analysis 3, Cells 50–53)

**Goal:** Same correction for the most biologically grounded UMAP. The current Stage 1 (per-cycle DC removal) is already good — just needs one more step.

#### Modify Cell 52 — insert Stage 1b between DC removal and global z-score

Find the block in Cell 52 that reads:
```python
# ── Stage 1: per-cycle DC removal ─────────────────────────────────────────────
X_3d_sc   = X_raw_sc.reshape(_n_cyc_sc, n_legs_sc * n_joints_sc, N_PHASE_BINS)
X_dc_sc   = (X_3d_sc - X_3d_sc.mean(axis=2, keepdims=True)).reshape(_n_cyc_sc, n_feat_sc)

# ── Stage 2: cross-cycle z-score per feature ──────────────────────────────────
_mu_sc  = X_dc_sc.mean(axis=0)
_sd_sc  = X_dc_sc.std(axis=0) + 1e-8
X_z_sc  = ((X_dc_sc - _mu_sc) / _sd_sc).astype(np.float32)
```

Replace with:
```python
# ── Stage 1: per-cycle DC removal ─────────────────────────────────────────────
X_3d_sc   = X_raw_sc.reshape(_n_cyc_sc, n_legs_sc * n_joints_sc, N_PHASE_BINS)
X_dc_sc   = (X_3d_sc - X_3d_sc.mean(axis=2, keepdims=True)).reshape(_n_cyc_sc, n_feat_sc)

# ── Stage 1b: per-fly mean subtraction (between DC removal and global z-score) ─
# Removes residual systematic differences in oscillation shape between flies
# (e.g., consistent mid-leg extension differences) while keeping within-fly
# kinematic structure intact.
_sc_fly_ids_arr = np.array([m['fly_id'] for m in sc_meta])
X_fly_sc = X_dc_sc.copy()
for _fid in np.unique(_sc_fly_ids_arr):
    _mask = _sc_fly_ids_arr == _fid
    X_fly_sc[_mask] -= X_fly_sc[_mask].mean(axis=0)

# ── Stage 2: cross-cycle z-score per feature ──────────────────────────────────
_mu_sc  = X_fly_sc.mean(axis=0)
_sd_sc  = X_fly_sc.std(axis=0) + 1e-8
X_z_sc  = ((X_fly_sc - _mu_sc) / _sd_sc).astype(np.float32)
```

#### Modify Cell 53 — add fly-ID panel + comparison

In Cell 53, the `_sc_color_specs` list already has fly ID as the 3rd entry (`_sc_fly_codes`). After running the modified Cell 52, check whether fly clusters dissolve. Also run the original normalization path in a separate variable for comparison:

```python
# Keep original normalized embedding for comparison
# (re-run Stage 2 on X_dc_sc without fly correction, store as umap_sc_orig)
_mu_orig = X_dc_sc.mean(axis=0)
_sd_orig = X_dc_sc.std(axis=0) + 1e-8
X_z_sc_orig = ((X_dc_sc - _mu_orig) / _sd_orig).astype(np.float32)
_sc_reducer_orig = cuml.manifold.UMAP(
    n_components=SC_UMAP_COMPONENTS, n_neighbors=SC_UMAP_N_NEIGHBORS,
    min_dist=SC_UMAP_MIN_DIST, metric='euclidean', random_state=SEGMENT_SEED_SC
)
umap_sc_orig = np.asarray(_sc_reducer_orig.fit_transform(X_z_sc_orig))

# Then run the fly-corrected version as usual → umap_sc
```

---

### Part D — Sex-aware UMAP: use fly identity as signal, not noise

**Goal:** Instead of removing fly identity, explicitly label by sex and use it as a biological variable. This is the most appropriate treatment for the female/male difference.

#### Cell D1 — Load session metadata CSV and attach sex label

At 7 sessions a hardcoded dict is manageable, but at 20 sessions it becomes error-prone. Maintain a small CSV instead — one row per session, add a row each time a new fly is processed.

**Create `session_metadata.csv`** in the notebook directory (or `data/`):

```
fly_id,sex,genotype,date,notes
session1,male,WT,2026-02-03,
session5,male,WT,2026-02-03,
session5_2,male,WT,2026-02-03,
session6,male,WT,2026-02-02,
session6_afternoon,male,WT,2026-02-02,
session7_afternoon,female,WT,2026-02-04,WT female S7
session7_morning,female,WT,2026-02-09,WT female S5 — confirm mapping
```

> **Note:** Verify `fly_id` strings against `np.unique(df_valid['fly_id'])` before populating. The values above are approximate; confirm against your actual bout_dict keys.

Then in the notebook:

```python
# ── Load session metadata ─────────────────────────────────────────────────────
# session_metadata.csv lives alongside the notebook (or set METADATA_PATH)
METADATA_PATH = Path('./session_metadata.csv')   # adjust if needed
session_meta = pd.read_csv(METADATA_PATH).set_index('fly_id')

# Attach to df_valid
df_valid['sex']      = df_valid['fly_id'].map(session_meta['sex']).fillna('unknown')
df_valid['genotype'] = df_valid['fly_id'].map(session_meta['genotype']).fillna('unknown')

# Attach to step-cycle metadata
for m in sc_meta:
    row = session_meta.loc[m['fly_id']] if m['fly_id'] in session_meta.index else {}
    m['sex']      = row.get('sex',      'unknown')
    m['genotype'] = row.get('genotype', 'unknown')

sc_sex      = np.array([m['sex']      for m in sc_meta])
sc_genotype = np.array([m['genotype'] for m in sc_meta])

print("Sex breakdown:")
print(df_valid['sex'].value_counts())
```

#### Cell D2 — UMAP colored by sex, speed, and posture PC1

```python
# Three-panel: sex | speed | posture PC1
# Uses the fly-normalized embedding umap_result_fly_norm

sex_codes = np.array([0 if s == 'male' else 1 for s in df_valid['sex']])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: sex
sc0 = axes[0].scatter(umap_result_fly_norm[:, 0], umap_result_fly_norm[:, 1],
                       c=sex_codes, cmap='coolwarm', s=1, alpha=0.4, vmin=0, vmax=1)
plt.colorbar(sc0, ax=axes[0], ticks=[0, 1]).set_ticklabels(['male', 'female'])
axes[0].set_title('Sex')

# Panel 2: speed
sc1 = axes[1].scatter(umap_result_fly_norm[:, 0], umap_result_fly_norm[:, 1],
                       c=df_valid['forward_speed'], cmap='turbo', s=1, alpha=0.4)
plt.colorbar(sc1, ax=axes[1], label='mm/s')
axes[1].set_title('Forward speed')

# Panel 3: n_legs_stance
sc2 = axes[2].scatter(umap_result_fly_norm[:, 0], umap_result_fly_norm[:, 1],
                       c=df_valid['n_legs_stance'], cmap='RdYlGn',
                       s=1, alpha=0.4, vmin=0, vmax=6)
plt.colorbar(sc2, ax=axes[2], label='N legs stance')
axes[2].set_title('N legs in stance')

for ax in axes:
    ax.set_xlabel('UMAP 1', fontsize=8); ax.set_ylabel('UMAP 2', fontsize=8)
plt.suptitle('Fly-normalized UMAP — biological variables', fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'umap_fly_norm_sex_speed_stance.pdf', bbox_inches='tight')
plt.show()
```

---

### Part E — Anchor the UMAP axes to biology: speed-binned density

**Goal:** Confirm that after normalization, UMAP dimensions correlate with known gait parameters. Adapted from the existing Paper Figures section (Cells 34–40).

#### Cell E1 — Speed-binned UMAP density

```python
# Bin frames by speed quartile, plot density in UMAP1-UMAP2 space
speed_bins = pd.qcut(df_valid['forward_speed'], q=4,
                      labels=['Q1 (slow)', 'Q2', 'Q3', 'Q4 (fast)'])
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
for i, (label, grp) in enumerate(df_valid.assign(_qbin=speed_bins).groupby('_qbin', observed=True)):
    idx = grp.index
    # Map df_valid index back to embedding row (df_valid has been filtered, use positional)
    pos = np.where(np.isin(np.arange(len(df_valid)), df_valid.index.get_indexer(idx)))[0]
    axes[i].hexbin(umap_result_fly_norm[pos, 0], umap_result_fly_norm[pos, 1],
                   gridsize=50, cmap='YlOrRd', mincnt=1)
    axes[i].set_title(label)
    axes[i].set_xlabel('UMAP 1'); axes[i].set_ylabel('UMAP 2')
plt.suptitle('Speed-binned UMAP density (fly-normalized)', fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'umap_fly_norm_speed_binned_density.pdf', bbox_inches='tight')
plt.show()
```

#### Cell E2 — Spearman correlations: UMAP axes vs. kinematic variables

```python
from scipy.stats import spearmanr

_umap_dims = umap_result_fly_norm[:, :4]  # first 4 UMAP dimensions
_kine_vars = {
    'speed':         df_valid['forward_speed'].values,
    'turning_rate':  df_valid['turning_rate'].values,
    'n_stance':      df_valid['n_legs_stance'].values,
    'T1L_phase':     df_valid['T1_left_phase'].values,
    'mean_abs_vel':  df_valid['mean_abs_vel'].values,
}

print("Spearman ρ: UMAP dimension vs. kinematic variable")
print(f"{'Variable':<18}", end='')
for d in range(_umap_dims.shape[1]):
    print(f"  UMAP{d+1:>2}", end='')
print()
for vname, vvals in _kine_vars.items():
    print(f"{vname:<18}", end='')
    for d in range(_umap_dims.shape[1]):
        rho, p = spearmanr(vvals, _umap_dims[:, d], nan_policy='omit')
        print(f"  {rho:+.2f}{'*' if p < 0.01 else ' '}", end='')
    print()
```

---

### Part G — Sex-stratified UMAP and per-sex comparison (10M + 10F target)

**Goal:** With a balanced 10M/10F design you can ask whether the gait manifold has the same *shape* across sexes (only shifted in posture space) or whether the topology itself differs. Run the fly-normalized UMAP separately for each sex, then overlay with consistent axes.

This part becomes most meaningful once the full dataset is assembled. It can be skipped with the current 5M/2F dataset.

#### Cell G1 — Per-sex joint angle statistics (Mann-Whitney U per joint)

```python
from scipy.stats import mannwhitneyu

males   = df_valid[df_valid['sex'] == 'male']
females = df_valid[df_valid['sex'] == 'female']

mw_results = []
for col in joint_cols:
    m_vals = males[col].dropna().values
    f_vals = females[col].dropna().values
    stat, p = mannwhitneyu(m_vals, f_vals, alternative='two-sided')
    # Common language effect size r = Z / sqrt(N)
    from scipy.stats import norm as _norm
    z = _norm.ppf(1 - p / 2) * np.sign(np.median(f_vals) - np.median(m_vals))
    r = abs(z) / np.sqrt(len(m_vals) + len(f_vals))
    mw_results.append({
        'joint': col,
        'median_male':   np.median(m_vals),
        'median_female': np.median(f_vals),
        'delta_median':  np.median(f_vals) - np.median(m_vals),
        'p_mw':  p,
        'r_eff': r,   # effect size: >0.1 small, >0.3 medium, >0.5 large
    })

mw_df = pd.DataFrame(mw_results).sort_values('r_eff', ascending=False)

# Volcano-style plot: effect size vs -log10(p)
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(mw_df['r_eff'], -np.log10(mw_df['p_mw'] + 1e-300),
                c=mw_df['delta_median'], cmap='RdBu_r', s=60, edgecolors='k', lw=0.3)
plt.colorbar(sc, ax=ax, label='Δ median (female − male, rad)')
ax.axvline(0.3, color='gray', ls='--', lw=0.8, label='r=0.3 (medium effect)')
ax.axhline(-np.log10(0.01), color='gray', ls=':', lw=0.8, label='p=0.01')
for _, row in mw_df[mw_df['r_eff'] > 0.3].iterrows():
    ax.annotate(row['joint'].replace('_', '\n'),
                (row['r_eff'], -np.log10(row['p_mw'] + 1e-300)),
                fontsize=6, ha='left', va='bottom')
ax.set_xlabel('Effect size r')
ax.set_ylabel('−log₁₀(p)')
ax.set_title('Per-joint sex difference (Mann-Whitney U)\ncolor = direction (blue=female larger)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sex_diff_volcano.pdf', bbox_inches='tight')
plt.show()

print("Joints with medium or large sex effect (r>0.3, p<0.01):")
print(mw_df[(mw_df['r_eff'] > 0.3) & (mw_df['p_mw'] < 0.01)]
      [['joint', 'median_male', 'median_female', 'delta_median', 'r_eff', 'p_mw']].to_string())
```

#### Cell G2 — Sex-stratified UMAP overlay

Run the fly-normalized UMAP per sex separately, then overlay in a shared figure. Consistent topology across sexes = gait manifold is universal. Different topology = sex-specific gait organization.

```python
# Build per-sex fly-normalized matrices
_sex_col = df_valid['sex'].values
_umaps_by_sex = {}

for sex in ['male', 'female']:
    _mask_sex = _sex_col == sex
    _X_sex = X_fly_norm[_mask_sex]
    _n_sex = _mask_sex.sum()
    _nn_sex = min(100, max(15, int(np.sqrt(_n_sex))))
    print(f"{sex}: {_n_sex} frames, n_neighbors={_nn_sex}")
    _red = cuml.manifold.UMAP(
        n_components=2, n_neighbors=_nn_sex, min_dist=0.5,
        metric='euclidean', random_state=70
    )
    _umaps_by_sex[sex] = np.asarray(_red.fit_transform(_X_sex))

# Overlay figure: color by speed within each sex
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, sex in zip(axes, ['male', 'female']):
    _mask_sex = _sex_col == sex
    _spd = df_valid.loc[_mask_sex, 'forward_speed'].values
    sc = ax.scatter(_umaps_by_sex[sex][:, 0], _umaps_by_sex[sex][:, 1],
                    c=_spd, cmap='turbo', s=1, alpha=0.4,
                    vmin=np.nanpercentile(_spd, 5), vmax=np.nanpercentile(_spd, 95))
    plt.colorbar(sc, ax=ax, label='Speed (mm/s)')
    ax.set_title(f'{sex.capitalize()} (n={_mask_sex.sum()} frames)')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
plt.suptitle('Per-sex UMAP — colored by speed\n(fly-normalized, independent projections)',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'umap_per_sex_speed.pdf', bbox_inches='tight')
plt.show()
```

**What to look for:**
- Similar ring/manifold shape in both sexes → gait organization is conserved; sex difference is a posture offset
- Different topology (e.g., males have a torus, females have a line) → genuinely different locomotion strategy
- Speed gradient runs in the same direction in both panels → speed axis is consistent across sexes

#### Cell G3 — Combine sexes with shared embedding (UMAP on full fly-normalized data, colored by sex)

This is the complement to G2: one embedding for everything, colored by sex *after* normalization. If normalization worked and the manifold is the same shape, male and female points should intermix uniformly.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sex_codes_arr = np.array([0 if s == 'male' else 1 for s in df_valid['sex']])

# Left: sex coloring on fly-normalized full UMAP
sc0 = axes[0].scatter(umap_result_fly_norm[:, 0], umap_result_fly_norm[:, 1],
                       c=sex_codes_arr, cmap='coolwarm', s=1, alpha=0.3, vmin=0, vmax=1)
cb0 = plt.colorbar(sc0, ax=axes[0], ticks=[0, 1])
cb0.set_ticklabels(['male', 'female'])
axes[0].set_title('Fly-normalized UMAP — sex')

# Right: sex coloring on original (non-normalized) UMAP for comparison
sc1 = axes[1].scatter(umap_result[:, 0], umap_result[:, 1],
                       c=sex_codes_arr, cmap='coolwarm', s=1, alpha=0.3, vmin=0, vmax=1)
cb1 = plt.colorbar(sc1, ax=axes[1], ticks=[0, 1])
cb1.set_ticklabels(['male', 'female'])
axes[1].set_title('Original UMAP — sex')

for ax in axes:
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
plt.suptitle('Sex separation: before and after per-fly normalization', fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'umap_sex_before_after_norm.pdf', bbox_inches='tight')
plt.show()
```

---

### Part F — Derivatives-only UMAP (quick sanity check)

Use only the `_d1` (angular velocity) columns as an alternative normalization. Angular velocities are invariant to absolute pose offset — this is the most conservative normalization. Compare with per-fly mean subtraction.

#### Cell F1

```python
# Build velocity-only feature matrix
_d1_cols = [c for c in df_valid.columns if c.endswith('_d1')]
X_vel = df_valid[_d1_cols].values
_vel_valid = ~np.any(np.isnan(X_vel), axis=1)
X_vel = X_vel[_vel_valid]
df_vel = df_valid[_vel_valid].copy().reset_index(drop=True)

scaler_vel = StandardScaler()
X_vel_sc = scaler_vel.fit_transform(X_vel)

reducer_vel = cuml.manifold.UMAP(
    n_components=6, n_neighbors=200, min_dist=0.8,
    metric='euclidean', random_state=70
)
umap_vel = np.asarray(reducer_vel.fit_transform(X_vel_sc))
print(f"Velocity UMAP shape: {umap_vel.shape}")

# Quick 3-panel: fly_id, speed, n_stance
fly_codes_vel = np.array([
    np.unique(df_vel['fly_id']).tolist().index(f) for f in df_vel['fly_id']
])
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (cvals, label, cmap, ckw) in zip(axes, [
    (fly_codes_vel,                    'Fly ID',      'tab10',  {}),
    (df_vel['forward_speed'].values,   'Speed (mm/s)','turbo',  {}),
    (df_vel['n_legs_stance'].values,   'N stance',    'RdYlGn', dict(vmin=0, vmax=6)),
]):
    sc = ax.scatter(umap_vel[:, 0], umap_vel[:, 1], c=cvals, cmap=cmap,
                    s=1, alpha=0.4, **ckw)
    plt.colorbar(sc, ax=ax)
    ax.set_title(label)
plt.suptitle('Velocity-only UMAP (angular velocities, no absolute angles)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'umap_velocity_only.pdf', bbox_inches='tight')
plt.show()
```

---

## Recommended Execution Order

Run the cells in this order. Each part builds on the previous:

```
Cells 1–14  (main pipeline — load, preprocess, PCA)
│
├── Part A (Cells A1–A3)  ← characterize inter-fly variation first
│                           outputs: heatmap, speed violin, η²+ANOVA bar chart
│
├── Part D (Cell D1)       ← attach sex labels from session_metadata.csv
│                           prerequisite: none; do this before any UMAP
│
├── Part B (Cells B1–B2)  ← fly-normalized Analysis 1 UMAP + comparison
│                           prerequisite: Cell 28 (StandardScaler + X_scaled)
│
├── Part C (Cells 50–53, modified)  ← fly-normalized step-cycle UMAP
│                           modify Cell 52 as described; Cell 53 adds comparison
│
├── Part D (Cell D2)       ← sex-colored UMAP panels
│                           prerequisite: umap_result_fly_norm from Part B
│
├── Part E (Cells E1–E2)  ← anchor UMAP to biology: speed bins + Spearman ρ
│                           prerequisite: umap_result_fly_norm from Part B
│
├── Part G (Cells G1–G3)  ← sex comparison stats + stratified UMAP
│                           prerequisite: Part D labels; meaningful at ≥5 per sex
│
└── Part F (Cell F1)       ← velocity-only UMAP sanity check
                            fully independent, can run anytime after Cell 12
```

---

## Scalability Reference (current 7-fly → target 20-fly)

This section summarizes every change needed when scaling from the current 7-session dataset to ~20 sessions (10M + 10F).

| Component | Current (7 sessions, ~78k frames) | At scale (20 sessions, ~220k frames) | Action needed |
|---|---|---|---|
| Session metadata | Hardcoded `SEX_MAP` dict | `session_metadata.csv` | **Done in Cell D1** — add one row per new session |
| `n_neighbors` (Analysis 1 UMAP) | 200 | 50–100 | **Auto-computed** in Cell B1 via `min(200, max(50, int(sqrt(N))))` |
| `n_neighbors` (velocity UMAP, Cell F1) | 200 (hardcoded) | 50–100 | Apply same formula: `min(200, max(50, int(sqrt(N_vel))))` |
| `n_neighbors` (per-sex UMAP, Cell G2) | — | auto | Already uses `min(100, max(15, sqrt(N_sex)))` |
| Step-cycle count | ~500 cycles | ~1 500 cycles | No change needed; UMAP handles this trivially |
| Segment UMAP (Analysis 2) | capped at 100k | capped at 100k | Already capped — no change |
| η² bar chart width | 12 in | auto | `figsize=(max(12, n_joints * 0.25), 5)` — already in Cell A3 |
| Speed violin layout | vertical (7 labels) | horizontal (20 labels) | Auto-switches in Cell A2 when `n_flies > 10` |
| Heatmap in Cell A1 | 7 columns | 20 columns | No code change; seaborn scales automatically. Increase `figsize` width if crowded: `figsize=(max(14, n_flies * 0.8), 6)` |
| Part G (sex stats + stratified UMAP) | Under-powered (5M/2F) | **Well-powered (10M/10F)** | Run Part G fully only once ≥5 per sex available |
| GPU memory (Analysis 1) | ~30 MB float32 | ~84 MB float32 | Fits easily on any modern GPU; no change |
| GPU memory (Analysis 2 segments) | ~2 GB (100k × 1932) | ~2 GB (capped) | No change |

**The only manual step when adding new sessions:** append a row to `session_metadata.csv`. Everything else auto-adjusts.

### UMAP parameter guidance by dataset size

```
N_frames    n_neighbors   min_dist   notes
──────────────────────────────────────────────────────────────────
  < 50k        200          0.8      fine-grained local structure
 50–150k        100          0.8      current sweet spot
150–300k         50          0.6      reduce further to keep kNN fast
  > 300k         30          0.5      or subsample to 150k + transform
```

For the step-cycle UMAP `SC_UMAP_N_NEIGHBORS=50` is already in the right range regardless of dataset size (few thousand cycles at most).

---

## Key Variables After Running This Plan

| Variable | Created in | Description |
|---|---|---|
| `session_meta` | Cell D1 | DataFrame indexed by fly_id; columns sex, genotype, date |
| `df_valid['sex']` | Cell D1 | Sex label per frame ('male'/'female') |
| `sc_sex` | Cell D1 | Sex label per step cycle |
| `fly_means` | Cell A1 | Per-fly mean joint angles, shape (n_flies, n_joints) |
| `eta_df` | Cell A3 | Per-joint between-fly η² and ANOVA p-value |
| `mw_df` | Cell G1 | Per-joint Mann-Whitney U sex comparison with effect sizes |
| `X_fly_norm` | Cell B1 | Per-fly mean-subtracted feature matrix, shape (N, features) |
| `umap_result_fly_norm` | Cell B1 | Fly-normalized Analysis 1 embedding, shape (N, 6) |
| `X_fly_sc` | Cell C (modified 52) | Per-fly corrected step-cycle matrix |
| `umap_sc` | Cell 53 | Step-cycle UMAP after fly correction |
| `umap_sc_orig` | Cell C | Original step-cycle UMAP (no fly correction) |
| `_umaps_by_sex` | Cell G2 | Dict `{'male': (N_m, 2), 'female': (N_f, 2)}` — per-sex embeddings |
| `umap_vel` | Cell F1 | Velocity-only embedding, shape (N_vel, 6) |

---

## Expected Outcomes and Decision Points

**After Part A:**
- If η² > 0.3 for T2 joints and < 0.1 for T1 joints → confirms behavioral (not anatomical) origin; focus on Part B+C
- If η² is uniformly high across all joints → more likely a global offset (scale residual from Procrustes); also check if female sessions cluster together in the posture PCA (Cell A2)

**After Part B:**
- If fly blobs dissolve → mean subtraction is sufficient
- If blobs persist but shrink → upgrade to per-fly z-score (uncomment amplitude normalization)
- If blobs are exactly male/female → the sex difference is real biology; keep it, just label it (Part D)

**After Part C (step-cycle UMAP):**
- A well-normalized step-cycle UMAP should show a continuous manifold organized by speed (color gradient) and gait phase, not discrete per-fly islands
- If you see 2–3 clusters that align with sex → that's biologically meaningful (different body sizes genuinely change gait kinematics)

**After Part E (Spearman ρ):**
- UMAP1 should correlate most strongly with speed (|ρ| > 0.4) after normalization
- UMAP2 often picks up turning rate or n_legs_stance
- If fly_id still has high ρ after normalization → some residual identity signal; consider also regressing out sex

**After Part G (sex comparison, meaningful at 10M/10F):**
- If Mann-Whitney r > 0.3 for T2 joints → confirmed: sex differences in mid-leg posture are robust and replicable
- If per-sex UMAPs (Cell G2) have similar ring/manifold shape → gait organization is conserved, sex is a posture offset
- If per-sex UMAPs have different topology → genuinely different locomotion strategies; consider separate analyses per sex going forward
- If Cell G3 shows full male/female intermixing after normalization → the fly-normalized UMAP is a clean shared gait embedding usable for all downstream analyses

---

## File Paths for Reference

```python
# IK output (source for main pipeline)
H5_PATH = Path('/home/user/src/JARVIS-HybridNet/projects/fly50_V5/predictions/'
               'predictions3D/Data_analysis/Testing/ik_output_combined_v1.h5')

# Preprocessing HDF5 files (alignment_info/scales stored here)
PREPROC_PATHS = {
    'WT_male_S6':   Path('.../fly50_V5/predictions/predictions3D/Predictions_3D_20260202-171900/preprocessed_bout_v1_free_walking.h5'),
    'WT_male_S1':   Path('.../fly50_V5/predictions/predictions3D/Predictions_3D_20260203-103416/preprocessed_bout_v1_free_walking.h5'),
    'WT_male_S5':   Path('.../fly50_V5/predictions/predictions3D/Predictions_3D_20260203-164328/preprocessed_bout_v1_free_walking.h5'),
    'WT_female_S7': Path('.../fly50_V5/predictions/predictions3D/Predictions_3D_20260204-062628/preprocessed_bout_v1_free_walking.h5'),
    'WT_male_S7':   Path('.../fly50_V5/predictions/predictions3D/Predictions_3D_20260205-111851/preprocessed_bout_v1_free_walking.h5'),
    'WT_female_S5': Path('.../fly50_V5/predictions/predictions3D/Predictions_3D_20260209-094844/preprocessed_bout_v1_free_walking.h5'),
}

# Raw JARVIS 3D keypoints (before preprocessing — for anatomy/body size)
JARVIS_DIRS = {
    'WT_male_S6':   Path('/home/user/src/JARVIS-HybridNet/projects/fly50_V5/predictions/predictions3D/Predictions_3D_20260202-171900/'),
    'WT_male_S1':   Path('/home/user/src/JARVIS-HybridNet/projects/fly50_V5/predictions/predictions3D/Predictions_3D_20260203-103416/'),
    'WT_male_S5':   Path('/home/user/src/JARVIS-HybridNet/projects/fly50_V5/predictions/predictions3D/Predictions_3D_20260203-164328/'),
    'WT_female_S7': Path('/home/user/src/JARVIS-HybridNet/projects/fly50_V5/predictions/predictions3D/Predictions_3D_20260204-062628/'),
    'WT_male_S7':   Path('/home/user/src/JARVIS-HybridNet/projects/fly50_V5/predictions/predictions3D/Predictions_3D_20260205-111851/'),
    'WT_female_S5': Path('/home/user/src/JARVIS-HybridNet/projects/fly50_V5/predictions/predictions3D/Predictions_3D_20260209-094844/'),
}
```

---

## Summary of Analytical Choices

| Choice | Rationale |
|---|---|
| Per-fly mean subtraction rather than full z-score | Mean subtraction removes posture bias while preserving amplitude information (step size, range of motion), which is biologically meaningful. Full z-score is available as an upgrade if needed. |
| `session_metadata.csv` for sex/genotype labels | Scales from 7 to 20+ sessions with no code changes; one row per session. Single source of truth for all metadata. |
| Auto-scaling `n_neighbors` via `sqrt(N)` | Maintains equivalent neighborhood density at any dataset size; prevents expensive kNN graphs at 220k+ frames. |
| Keep sex as an explicit variable | Female/male size difference (~11%) is real biology, not a confound. It should be labeled and used to interpret UMAP axes, not blindly removed. |
| Part G deferred until ≥5 per sex | Mann-Whitney and stratified UMAP are under-powered with 5M/2F; running them early gives misleading statistics. Flag the section and run it once the full dataset is ready. |
| Step-cycle UMAP (Analysis 3) as primary analysis | Most biologically grounded: one point per complete stride, time-normalized, joint-angle features. Per-fly correction here has the cleanest interpretation. |
| η² + ANOVA characterization before normalization | Identifies which joints drive inter-fly separation, informing whether the difference is anatomical (all joints uniform) or behavioral (T2-specific). With 20 flies, ANOVA is well powered. |
| Spearman ρ correlation of UMAP axes | Grounds abstract dimensions in interpretable kinematic variables; confirms normalization is working. |
