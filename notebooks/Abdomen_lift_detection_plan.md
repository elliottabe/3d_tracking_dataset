# Abdomen Lift Detection Plan

*Generated 2026-03-09. Companion to `Joint_Kinematics_Analysis.ipynb`.*

---

## Background

Abdomen lifts are discrete behavioral events (~30 ms, ~24 frames at 800 Hz) visible in raw videos as a marked upward deflection of the abdomen, typically occurring when the fly decelerates or stops momentarily. No automated detector exists yet.

### Relationship to other analyses
- Lifts occur predominantly during **near-stopping** → they are naturally excluded from the walking/running analysis (Cell H5) by the activity mask.
- They shift the **CoM upward** (abdomen is ~20–30% of body mass) → they are potential confounds in the height oscillation analysis (Cell H2) if not excluded.
- They may represent a **postural behavior** (grooming preparation, sensory scanning) worth characterizing in its own right.

---

## Signal

### Available signals
The IK output HDF5 stores, per bout:

| Array | Shape | Description |
|---|---|---|
| `xpos_egocentric` | (T, 50, 3) | Tracking site positions in body frame |
| `kp_data` | (T, 150) | Raw JARVIS 3D keypoints, world frame (reshape to T×50×3) |
| `xpos` | (T, 68, 3) | MuJoCo body positions, world frame |

Abdomen-relevant indices (in egocentric frame, `site_names_egocentric`):
- Index 10: `tracking[Abd_A4]_fly` — base of abdomen (segment A4)
- Index 11: `tracking[Abd_tip]_fly` — distal abdomen tip

Abdomen-relevant indices in `kp_data.reshape(T, 50, 3)` (matching `kp_names`):
- Index 10: `Abd_A4` — world-frame 3D position
- Index 11: `Abd_tip` — world-frame 3D position

### Important caveat: IK joint limits
The MuJoCo model's abdomen joints are clipped at **±0.10–0.15 rad** in qpos, and the abdomen body positions from `xpos` (forward kinematics) will saturate at the same limits. **Do not use `xpos` abdomen bodies as the primary signal.** Use the tracked keypoints instead.

### Recommended primary signal

**`abd_relative_z`** = height of Abd_tip above the thorax, in the egocentric frame:

```python
# Per bout, from HDF5:
abd_tip_ego_z  = xpos_egocentric[:, 11, 2]   # z in body frame (negative = below thorax)
abd_a4_ego_z   = xpos_egocentric[:, 10, 2]   # z of Abd_A4

# Alternative: world-frame relative height (cross-check)
# Requires kp_data reshaped:
kp = kp_data.reshape(T, 50, 3)
abd_tip_world_z = kp[:, 11, 2]   # world z (model units)
thorax_world_z  = xpos[:, 1, 2]  # world z of thorax
abd_relative_z_world = abd_tip_world_z - thorax_world_z
```

The egocentric `abd_tip_ego_z` (mean ≈ −0.051 m, std ≈ 0.006 m, total range ≈ 0.035 m) is the primary signal. A **lift** moves this value less negative (upward).

**Note on coordinate frame:** The egocentric frame uses thorax–head–abdomen to define orientation. The body-axis direction (yaw) is set by the head-abdomen vector, but Z is largely tied to the world vertical. Abd_tip z in this frame is NOT constant (std = 0.006, unlike Scutellum ego z which has std = 0.00001), confirming it captures real abdomen movement.

---

## Detection Algorithm

### Cell I1 — Add abdomen signals to df_valid

First, add abdomen egocentric Z to the per-frame DataFrame. This requires a small modification to `bouts_to_dataframe()` or a post-processing step:

```python
# Post-processing: add abdomen signals from the HDF5 directly to df_valid
# (df_valid is already built; we look up each frame by bout_id and frame index)

import h5py

H5_PATH_STR = str(H5_PATH)

df_valid['abd_tip_ego_z']  = np.nan
df_valid['abd_a4_ego_z']   = np.nan
df_valid['abd_relative_z'] = np.nan   # abd_tip_ego_z - abd_a4_ego_z (tail span)

with h5py.File(H5_PATH_STR, 'r') as f:
    for bid in df_valid['bout_id'].unique():
        idx = df_valid[df_valid['bout_id'] == bid].index
        frames = df_valid.loc[idx, 'frame'].values   # 0-based frame index within bout

        xego = np.array(f[bid]['xpos_egocentric'][:])   # (T, 50, 3)

        df_valid.loc[idx, 'abd_tip_ego_z'] = xego[frames, 11, 2]
        df_valid.loc[idx, 'abd_a4_ego_z']  = xego[frames, 10, 2]
        # Vertical span from A4 to tip (negative = tip below A4)
        df_valid.loc[idx, 'abd_relative_z'] = xego[frames, 11, 2] - xego[frames, 10, 2]

df_valid['abd_tip_ego_z_mm']  = df_valid['abd_tip_ego_z']  * 10.0
df_valid['abd_a4_ego_z_mm']   = df_valid['abd_a4_ego_z']   * 10.0
df_valid['abd_relative_z_mm'] = df_valid['abd_relative_z'] * 10.0

print(f"Abd_tip_ego_z_mm: mean={df_valid['abd_tip_ego_z_mm'].mean():.2f} "
      f"std={df_valid['abd_tip_ego_z_mm'].std():.2f} mm")
print(f"Abd_relative_z_mm: mean={df_valid['abd_relative_z_mm'].mean():.2f} "
      f"std={df_valid['abd_relative_z_mm'].std():.2f} mm")
```

### Cell I2 — Compute per-bout detrended abdomen signal

```python
from scipy.ndimage import uniform_filter1d

BASELINE_FRAMES_ABD = int(0.500 * FPS)   # 400 frames — same as thorax baseline

df_valid['abd_tip_detrended_mm'] = np.nan

for bid, grp in df_valid.groupby('bout_id', sort=False):
    idx = grp.index
    if len(idx) < 30:
        continue
    z_mm = grp['abd_tip_ego_z_mm'].values
    if np.all(np.isnan(z_mm)):
        continue
    # Gaussian smooth then subtract baseline
    from scipy.ndimage import gaussian_filter1d
    z_smooth   = gaussian_filter1d(z_mm, sigma=3)       # ~3.75 ms smoothing
    baseline   = uniform_filter1d(z_smooth, size=BASELINE_FRAMES_ABD, mode='nearest')
    df_valid.loc[idx, 'abd_tip_detrended_mm'] = z_smooth - baseline

print("Abd detrended signal ready.")
print(f"  std={df_valid['abd_tip_detrended_mm'].std():.4f} mm")
```

### Cell I3 — Lift event detector

```python
# ── Detection parameters ──────────────────────────────────────────────────────
LIFT_THRESHOLD_SD     = 1.5    # threshold above baseline in robust SDs
LIFT_MIN_FRAMES       = 8      # min duration: 10 ms at 800 Hz
LIFT_MAX_FRAMES       = 120    # max duration: 150 ms
LIFT_MERGE_FRAMES     = 12     # merge events within 15 ms
LIFT_SPEED_PERCENTILE = 50     # events must occur when speed < this percentile
                               # 50th = below median speed (near-stopping criterion)

from scipy.stats import iqr as scipy_iqr

lift_events = []   # list of dicts, one per detected event

for bid, grp in df_valid.groupby('bout_id', sort=False):
    if len(grp) < 60:
        continue

    z = grp['abd_tip_detrended_mm'].values
    spd = grp['forward_speed'].values
    frames_arr = grp['frame'].values

    if np.all(np.isnan(z)):
        continue

    # Robust SD via IQR
    robust_sd = scipy_iqr(z[~np.isnan(z)]) / 1.35
    if robust_sd < 1e-6:
        continue

    threshold = LIFT_THRESHOLD_SD * robust_sd
    spd_thresh = np.nanpercentile(spd, LIFT_SPEED_PERCENTILE)

    # Binary detection: above threshold AND below speed threshold
    above   = z > threshold
    slow    = spd < spd_thresh
    trigger = above & slow

    # Find contiguous runs above threshold (speed filter applied per-event below)
    from itertools import groupby as _groupby
    runs = []
    for val, run in _groupby(enumerate(trigger), key=lambda x: x[1]):
        if val:
            frames_in_run = [g[0] for g in run]
            runs.append((frames_in_run[0], frames_in_run[-1]))

    # Merge nearby runs
    if not runs:
        continue
    merged = [runs[0]]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= LIFT_MERGE_FRAMES:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Apply duration filter and record events
    fly_id   = grp['fly_id'].iloc[0]
    bout_idx = grp['bout_idx'].iloc[0] if 'bout_idx' in grp.columns else 0

    for start, end in merged:
        dur = end - start + 1
        if not (LIFT_MIN_FRAMES <= dur <= LIFT_MAX_FRAMES):
            continue

        # Event metadata
        peak_z     = np.nanmax(z[start:end+1])
        mean_spd   = np.nanmean(spd[start:end+1])
        # Speed 500 ms before lift
        pre_start  = max(0, start - int(0.500 * FPS))
        pre_spd    = np.nanmean(spd[pre_start:start])
        # Speed 500 ms after lift
        post_end   = min(len(spd), end + int(0.500 * FPS))
        post_spd   = np.nanmean(spd[end:post_end])

        # Gait phase at lift peak
        peak_frame = start + np.nanargmax(z[start:end+1])
        t1l_phase_at_peak = grp['T1_left_phase'].iloc[peak_frame] \
                            if 'T1_left_phase' in grp.columns else np.nan
        n_stance_at_peak  = grp['n_legs_stance'].iloc[peak_frame] \
                            if 'n_legs_stance' in grp.columns else np.nan

        lift_events.append({
            'bout_id':           bid,
            'fly_id':            fly_id,
            'bout_frame_start':  int(frames_arr[start]),
            'bout_frame_end':    int(frames_arr[end]),
            'duration_frames':   dur,
            'duration_ms':       dur / FPS * 1000,
            'peak_abd_z_mm':     float(peak_z),
            'mean_speed_during': float(mean_spd),
            'mean_speed_before': float(pre_spd),
            'mean_speed_after':  float(post_spd),
            'T1L_phase_at_peak': float(t1l_phase_at_peak),
            'n_legs_stance_at_peak': float(n_stance_at_peak),
        })

lift_df = pd.DataFrame(lift_events)
print(f"Detected {len(lift_df)} abdomen lift events across {lift_df['bout_id'].nunique()} bouts")
print(f"Duration distribution (ms):")
print(lift_df['duration_ms'].describe().to_string())
print(f"\nEvents per fly:")
print(lift_df.groupby('fly_id').size().to_string())
```

### Cell I4 — Mark lift events in df_valid

```python
# Add binary column to df_valid: 1 during a lift event, 0 otherwise
df_valid['is_abd_lift'] = 0

for _, ev in lift_df.iterrows():
    bid = ev['bout_id']
    t0  = ev['bout_frame_start']
    t1  = ev['bout_frame_end']
    mask = (df_valid['bout_id'] == bid) & \
           (df_valid['frame'] >= t0) & (df_valid['frame'] <= t1)
    df_valid.loc[mask, 'is_abd_lift'] = 1

print(f"Frames marked as abdomen lift: {df_valid['is_abd_lift'].sum()} "
      f"({df_valid['is_abd_lift'].mean()*100:.1f}% of frames)")
```

---

## Cell I5 — Characterize lift events

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel A: Duration distribution
axes[0, 0].hist(lift_df['duration_ms'], bins=20, edgecolor='k')
axes[0, 0].axvline(30, color='r', ls='--', label='30 ms target')
axes[0, 0].set_xlabel('Duration (ms)'); axes[0, 0].set_ylabel('N events')
axes[0, 0].set_title('A  Lift duration distribution')
axes[0, 0].legend()

# Panel B: Speed before/during/after
_spd_df = pd.DataFrame({
    'Before': lift_df['mean_speed_before'],
    'During': lift_df['mean_speed_during'],
    'After':  lift_df['mean_speed_after'],
})
import seaborn as sns
sns.boxplot(data=_spd_df.melt(var_name='Period', value_name='Speed'),
            x='Period', y='Speed', ax=axes[0, 1])
axes[0, 1].set_ylabel('Forward speed (mm/s)')
axes[0, 1].set_title('B  Speed around lift events')

# Panel C: Peak amplitude distribution
axes[0, 2].hist(lift_df['peak_abd_z_mm'], bins=20, edgecolor='k')
axes[0, 2].set_xlabel('Peak abdomen Z (mm)'); axes[0, 2].set_ylabel('N events')
axes[0, 2].set_title('C  Lift amplitude distribution')

# Panel D: Gait phase at lift peak (circular histogram)
_phases = lift_df['T1L_phase_at_peak'].dropna().values
_phase_bins = np.linspace(-np.pi, np.pi, 25)
axes[1, 0].hist(_phases, bins=_phase_bins, edgecolor='k')
axes[1, 0].axvline(-np.pi, color='b', ls='--', lw=1, label='T1L liftoff')
axes[1, 0].axvline(0,      color='b', ls=':',  lw=1, label='T1L mid-swing')
axes[1, 0].set_xlabel('T1_left phase at lift peak (rad)')
axes[1, 0].set_ylabel('N events')
axes[1, 0].set_title('D  Gait phase at lift peak\n(uniform = random; peak = preferred phase)')
axes[1, 0].legend(fontsize=8)

# Panel E: N legs in stance during lift
axes[1, 1].hist(lift_df['n_legs_stance_at_peak'].dropna().values,
                bins=np.arange(-0.5, 7.5, 1), edgecolor='k')
axes[1, 1].set_xlabel('N legs in stance during lift')
axes[1, 1].set_ylabel('N events')
axes[1, 1].set_title('E  Leg configuration during lift\n(6 = fully stopped)')
axes[1, 1].set_xticks(range(7))

# Panel F: Events per fly
lift_per_fly = lift_df.groupby('fly_id').agg(
    n_events=('bout_id', 'count'),
    rate_per_min=('bout_id', lambda x: len(x) / (
        df_valid[df_valid['fly_id'] == x.index[0] if hasattr(x.index, '__getitem__') else x.name].shape[0] / FPS / 60
    ) if False else len(x))   # placeholder rate
).reset_index()
axes[1, 2].bar(range(len(lift_per_fly)), lift_per_fly['n_events'])
axes[1, 2].set_xticks(range(len(lift_per_fly)))
axes[1, 2].set_xticklabels(lift_per_fly['fly_id'], rotation=30, fontsize=8)
axes[1, 2].set_ylabel('N events')
axes[1, 2].set_title('F  Lift events per fly')

plt.suptitle('Abdomen lift characterization', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'abdomen_lift_characterization.pdf', bbox_inches='tight')
plt.show()
```

---

## Cell I6 — Visual validation: triggered average around lift events

Before trusting the detector, validate it by plotting the average time series of multiple signals around all detected events.

```python
# Triggered average: align all events to peak abd_z, plot ±500 ms window
WINDOW_MS   = 500
WINDOW_HALF = int(WINDOW_MS / 1000 * FPS)   # frames

signals_to_avg = {
    'abd_tip_detrended_mm':  'Abd tip Z detrended (mm)',
    'forward_speed':         'Forward speed (mm/s)',
    'n_legs_stance':         'N legs stance',
    'thorax_z_detrended':    'Thorax Z detrended (mm)',
}

triggered = {k: [] for k in signals_to_avg}

for _, ev in lift_df.iterrows():
    bid  = ev['bout_id']
    peak = int(ev['bout_frame_start'] +
               (ev['bout_frame_end'] - ev['bout_frame_start']) // 2)
    grp  = df_valid[df_valid['bout_id'] == bid].sort_values('frame')
    grp_frames = grp['frame'].values
    peak_pos   = np.searchsorted(grp_frames, peak)
    t0 = peak_pos - WINDOW_HALF
    t1 = peak_pos + WINDOW_HALF + 1
    if t0 < 0 or t1 > len(grp):
        continue
    for sig in signals_to_avg:
        if sig in grp.columns:
            triggered[sig].append(grp[sig].values[t0:t1])

t_axis_ms = np.linspace(-WINDOW_MS, WINDOW_MS, 2 * WINDOW_HALF + 1)

fig, axes = plt.subplots(len(signals_to_avg), 1, figsize=(10, 12), sharex=True)
for ax, (sig, label) in zip(axes, signals_to_avg.items()):
    if not triggered[sig]:
        continue
    arr = np.array([a for a in triggered[sig] if len(a) == 2*WINDOW_HALF+1])
    mean_trace = np.nanmean(arr, axis=0)
    sem_trace  = np.nanstd(arr, axis=0) / np.sqrt(len(arr))
    ax.fill_between(t_axis_ms, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3)
    ax.plot(t_axis_ms, mean_trace, lw=2)
    ax.axvline(0, color='r', ls='--', lw=1, label='lift peak')
    ax.set_ylabel(label, fontsize=9)
    ax.legend(fontsize=8)

axes[-1].set_xlabel('Time relative to lift peak (ms)')
plt.suptitle(f'Triggered average around {len(lift_df)} lift events', fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'abdomen_lift_triggered_avg.pdf', bbox_inches='tight')
plt.show()
```

**What to look for:**
- **Abd tip Z** should show a clear peak at t=0 with ~30 ms duration
- **Forward speed** should be low before/during and may increase after (fly resumes walking)
- **N legs stance** should be close to 6 (all legs on ground) during the lift
- **Thorax Z** should be mostly flat (lift is isolated to abdomen, not whole body rising)

If the triggered average looks as expected, the detector is working. If thorax Z also shows a large peak, the detector may be picking up whole-body bobbing, not abdomen lifts specifically.

---

## Cell I7 — UMAP integration: add lift as color variable

```python
# Use is_abd_lift as a color variable in the existing UMAP embeddings
# Requires umap_result_fly_norm from Part B of UMAP plan

if 'umap_result_fly_norm' in dir():
    _is_lift = df_valid['is_abd_lift'].values.astype(float)
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot non-lift frames first (background)
    ax.scatter(umap_result_fly_norm[_is_lift == 0, 0],
               umap_result_fly_norm[_is_lift == 0, 1],
               s=1, alpha=0.2, c='lightgray', rasterized=True)
    # Overlay lift frames
    ax.scatter(umap_result_fly_norm[_is_lift == 1, 0],
               umap_result_fly_norm[_is_lift == 1, 1],
               s=8, alpha=0.8, c='red', label=f'Lift events (n={_is_lift.sum():.0f} frames)')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_title('Abdomen lift events in UMAP space')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'umap_abdomen_lift_overlay.pdf', bbox_inches='tight')
    plt.show()
```

---

## Execution Order

```
Cell 1–14  (main pipeline)
Optional: Cell H1 (thorax_z_detrended, needed for triggered average validation)
│
├── Cell I1  ← Load abdomen signals from HDF5 → df_valid
│
├── Cell I2  ← Compute per-bout detrended abdomen signal
│
├── Cell I3  ← Run lift detector → lift_df
│               Tune LIFT_THRESHOLD_SD here (try 1.5, then 2.0)
│
├── Cell I4  ← Mark lift events in df_valid['is_abd_lift']
│
├── Cell I5  ← Characterize events (duration, speed, phase, stance)
│
├── Cell I6  ← Triggered average validation (most important QC step)
│               If traces look wrong: adjust parameters in I3 and re-run
│
└── Cell I7  ← UMAP overlay (requires umap_result_fly_norm from UMAP plan Part B)
```

---

## Key Variables

| Variable | Cell | Description |
|---|---|---|
| `df_valid['abd_tip_ego_z_mm']` | I1 | Abdomen tip Z in egocentric frame, mm |
| `df_valid['abd_tip_detrended_mm']` | I2 | Detrended (baseline-subtracted) abd_tip Z |
| `df_valid['is_abd_lift']` | I4 | Binary: 1 during detected lift event |
| `lift_df` | I3 | One row per event: duration, speed, phase, amplitude |

---

## Parameter Tuning Guide

| Parameter | Start | Too low | Too high |
|---|---|---|---|
| `LIFT_THRESHOLD_SD` | 1.5 | Many false positives; speed filter alone must reject them | Events too short to trigger; few events detected |
| `LIFT_MIN_FRAMES` | 8 (10 ms) | Risk of noise spikes | Misses short lifts |
| `LIFT_MAX_FRAMES` | 120 (150 ms) | — | Includes postural shifts that aren't lifts |
| `LIFT_MERGE_FRAMES` | 12 (15 ms) | Double-counts events with brief interruptions | Over-merges distinct events |
| `LIFT_SPEED_PERCENTILE` | 50 | May miss lifts during slow walking | Includes too many walking frames as candidates |

Use Cell I6 (triggered average) as the primary diagnostic. A good detector shows: sharp peak in abd_z, flat thorax_z, n_legs_stance ≥ 5, speed dip centered near t=0.

---

## Connection to Scutellum/Running Analysis

- Use `df_valid['is_abd_lift'] == 0` as an exclusion mask in Cell H5 (walking/running phase test) if abdomen lifts are frequent enough to bias the signal.
- The stopping detection infrastructure (speed threshold, duration filter) from this plan can be reused for identifying all stopping events, not just those with lifts.
- Once lift rate per fly is established, cross-reference with the inter-fly UMAP differences from `UMAP_interfly_analysis_plan.md` — flies with more frequent lifts may occupy a distinct UMAP region.
