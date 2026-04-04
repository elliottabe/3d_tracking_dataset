# Scutellum Height Oscillation and Walking vs. Running Analysis Plan

*Generated 2026-03-09. Companion to `Joint_Kinematics_Analysis.ipynb`.*

---

## Background and Motivation

### The observation
The scutellum oscillates in Z (world-frame height) at **twice the stride frequency** — once per tripod swing — suggesting the body rises each time a tripod lifts. This is visible in the raw videos. Additionally, mean body height correlates with locomotion speed.

### The biomechanics question
An expert framing: *"the biomechanics definition of walking vs. running is whether kinetic energy and gravitational potential energy are out of phase (walking) or in phase (running). Kinematically this reduces to whether CoM velocity fluctuations and CoM height are in phase or out of phase."*

This gives a precise, testable criterion:
- **Walking (inverted pendulum):** height oscillation and speed oscillation are **~π out of phase** — CoM is highest when the stance leg is most vertical (low velocity), lowest during transition.
- **Running (spring-mass):** height oscillation and speed oscillation are **~0 in phase** — CoM is lowest at mid-stance (leg loaded/compressed), highest during aerial phase.

Your observation (scutellum rises *mid tripod swing*) is already consistent with walking: the stance tripod straightening → body rises → this is the inverted pendulum peak. The question is whether this phase relationship flips at high speeds.

### Three layers of analysis
1. **Oscillation characterization:** Confirm the 2× frequency relationship and the phase anchor to T1_left.
2. **DC speed-height relationship:** Does mean height change systematically with mean speed?
3. **Walking vs. running test:** Does the height-speed phase relationship change with speed?

---

## Signal

### What NOT to use
- `xpos_egocentric[:, 0, 2]` (Scutellum in body frame): **constant**, std ≈ 0.00001. Useless for oscillation — it is a rigid body marker fixed to the thorax.

### What to use
- **`df_valid['thorax_z']`** — world-frame thorax Z from `xpos[:, 1, 2]`, already in df_valid. In model units; multiply by 10 for mm. This is the CoM height proxy.
- `kp_data.reshape(T, 50, 3)[:, 0, 2]` (Scutellum from raw JARVIS, in each bout's HDF5 group) is equivalent and can cross-validate, but `thorax_z` is sufficient and already aligned to df_valid.

### Relationship to Sandbox notebook
The Sandbox uses `data3D.csv` → Scutellum Z (mm) as body height. `df_valid['thorax_z'] × 10` is the same signal from the IK pipeline. Both represent the dorsal thorax height above the floor.

### Signal decomposition
```
thorax_z  =  slow_baseline  +  oscillatory_component  +  noise
              (changes with speed,       (at 2× stride freq,
               stopping, arena position)  amplitude ~0.1–0.5 mm)
```
The DC component and oscillatory component need to be separated for different analyses.

---

## Implementation

All cells insert in `Joint_Kinematics_Analysis.ipynb` after Cell 14 (after df_valid is built with PC columns). Group them as a new section: **"Section 12 — Body Height and Walking/Running Analysis"**.

Prerequisites: Cells 1–14 run. `df_valid` must contain `thorax_z`, `forward_speed`, `T1_left_phase`, `step_cycle_mean_speed`, `T1_left_swing_stance`, and `n_legs_stance`.

---

### Cell H1 — Add detrended thorax height to df_valid

```python
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt

FPS = 800  # Hz

# ── Compute thorax height in mm ───────────────────────────────────────────────
df_valid['thorax_z_mm'] = df_valid['thorax_z'] * 10.0   # model-m → mm

# ── Per-bout slow baseline (rolling mean, 500 ms window) ─────────────────────
# Captures speed-correlated height changes; removing it isolates the oscillation
BASELINE_FRAMES = int(0.500 * FPS)   # 400 frames

df_valid['thorax_z_baseline'] = np.nan
df_valid['thorax_z_detrended'] = np.nan

for bid, grp in df_valid.groupby('bout_id', sort=False):
    idx = grp.index
    z_mm = grp['thorax_z_mm'].values
    # Rolling mean baseline (edge-padded)
    baseline = uniform_filter1d(z_mm, size=BASELINE_FRAMES, mode='nearest')
    df_valid.loc[idx, 'thorax_z_baseline'] = baseline
    df_valid.loc[idx, 'thorax_z_detrended'] = z_mm - baseline

# ── Also: oscillatory component via bandpass (5–50 Hz) ───────────────────────
# This is an alternative to detrending; isolates exactly the stride-frequency band
b_bp, a_bp = butter(3, [5 / (FPS / 2), 50 / (FPS / 2)], btype='band')

df_valid['thorax_z_osc'] = np.nan
for bid, grp in df_valid.groupby('bout_id', sort=False):
    idx = grp.index
    if len(idx) < 30:   # skip very short bouts
        continue
    z_mm = grp['thorax_z_mm'].values
    df_valid.loc[idx, 'thorax_z_osc'] = filtfilt(b_bp, a_bp, z_mm)

print(f"thorax_z_mm:       mean={df_valid['thorax_z_mm'].mean():.3f} mm  std={df_valid['thorax_z_mm'].std():.3f} mm")
print(f"thorax_z_detrended: mean≈0  std={df_valid['thorax_z_detrended'].std():.4f} mm")
print(f"thorax_z_osc:       mean≈0  std={df_valid['thorax_z_osc'].std():.4f} mm")
```

---

### Cell H2 — Layer 1: Phase-averaged height oscillation

**Question:** Does thorax Z oscillate at 2× stride frequency, and what is its phase anchor to T1_left?

The stride cycle is defined by `T1_left_phase ∈ [-π, π]`. If the body oscillates at 2×, the phase-averaged signal should show **two peaks** per stride cycle.

```python
N_BINS = 48   # bins per stride cycle

# Phase-bin the detrended thorax Z
phase_bins = np.linspace(-np.pi, np.pi, N_BINS + 1)
bin_centers = 0.5 * (phase_bins[:-1] + phase_bins[1:])
bin_idx = np.digitize(df_valid['T1_left_phase'], phase_bins) - 1
bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

# Compute mean and SEM per phase bin, overall and by speed quartile
speed_quartiles = pd.qcut(df_valid['step_cycle_mean_speed'], q=4,
                          labels=['Q1 slow', 'Q2', 'Q3', 'Q4 fast'])
df_valid['_sq'] = speed_quartiles

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: overall phase average
z_phase = np.array([
    df_valid['thorax_z_detrended'].values[bin_idx == b].mean()
    for b in range(N_BINS)
])
z_sem = np.array([
    df_valid['thorax_z_detrended'].values[bin_idx == b].std() /
    np.sqrt((bin_idx == b).sum())
    for b in range(N_BINS)
])
axes[0].fill_between(bin_centers, z_phase - z_sem, z_phase + z_sem, alpha=0.3)
axes[0].plot(bin_centers, z_phase, lw=2)
axes[0].axhline(0, color='k', lw=0.5, ls='--')
axes[0].set_xlabel('T1_left phase (rad)')
axes[0].set_ylabel('Detrended thorax Z (mm)')
axes[0].set_title('Phase-averaged body height (all speeds)')
axes[0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axes[0].set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

# Annotate T1_left swing onset (phase ≈ -π) and mid-swing (phase ≈ 0)
axes[0].axvline(-np.pi, color='blue', ls=':', lw=1, label='T1L liftoff')
axes[0].axvline(0, color='blue', ls='--', lw=1, label='T1L mid-swing')
axes[0].legend(fontsize=8)

# Right: by speed quartile
colors = plt.cm.viridis(np.linspace(0.2, 0.9, 4))
for qi, (qlabel, qgrp) in enumerate(df_valid.groupby('_sq', observed=True)):
    _bidx = bin_idx[qgrp.index]
    # Map to positional indices in qgrp
    _pos_bidx = np.array([
        np.where(df_valid.index == i)[0][0] for i in qgrp.index
    ])
    _z_vals = df_valid['thorax_z_detrended'].values
    qz = np.array([_z_vals[_pos_bidx[_bidx == b]].mean()
                   if (_bidx == b).any() else np.nan
                   for b in range(N_BINS)])
    axes[1].plot(bin_centers, qz, color=colors[qi], lw=2, label=qlabel)

axes[1].axhline(0, color='k', lw=0.5, ls='--')
axes[1].set_xlabel('T1_left phase (rad)')
axes[1].set_ylabel('Detrended thorax Z (mm)')
axes[1].set_title('Phase-averaged body height by speed quartile')
axes[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axes[1].set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
axes[1].legend(fontsize=8)

plt.suptitle('Body height oscillation vs. stride phase\n'
             '(2 peaks = 2× stride freq; peak phase = phase of body rise)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'height_phase_avg.pdf', bbox_inches='tight')
plt.show()
```

**What to look for:**
- **2 peaks per stride cycle** → confirms 2× frequency relationship
- **Peak positions** relative to T1_left phase → phase anchor. If peaks occur at T1L_phase ≈ 0 and ≈ π, the body rises at mid-swing of each tripod as expected.
- **Speed-dependent shift:** if the phase of the peaks shifts left (earlier) at high speed, that's the first hint of a gait transition.

---

### Cell H3 — Layer 1 (continued): Hilbert phase of height oscillation

For a cleaner phase characterization, compute the instantaneous phase of the thorax height oscillation and compare it directly to the T1_left phase.

```python
from scipy.signal import hilbert

# Compute Hilbert phase of thorax_z_osc per bout
df_valid['thorax_z_hilbert_phase'] = np.nan

for bid, grp in df_valid.groupby('bout_id', sort=False):
    idx = grp.index
    z_osc = grp['thorax_z_osc'].values
    if np.all(np.isnan(z_osc)) or len(z_osc) < 30:
        continue
    analytic = hilbert(z_osc)
    df_valid.loc[idx, 'thorax_z_hilbert_phase'] = np.angle(analytic)

# Phase difference: thorax_z_hilbert_phase runs at 2× T1L phase
# To compare, we look at thorax_z_phase mod π (folds 2× frequency onto same range)
# Or: directly plot thorax_z_phase vs T1_left_phase as a scatter

# Quick summary: circular mean of thorax_z_phase at each T1L phase bin
_z_phases = df_valid['thorax_z_hilbert_phase'].values
_t1_phases = df_valid['T1_left_phase'].values
_valid = ~(np.isnan(_z_phases) | np.isnan(_t1_phases))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(_t1_phases[_valid][::20], _z_phases[_valid][::20],
           s=1, alpha=0.1, c='steelblue')
ax.set_xlabel('T1_left phase (rad)')
ax.set_ylabel('Thorax Z Hilbert phase (rad)')
ax.set_title('Instantaneous phase coupling: body height vs. stride phase')
ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(['-π', '0', 'π'])
ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels(['-π', '0', 'π'])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'height_hilbert_phase_scatter.pdf', bbox_inches='tight')
plt.show()
```

---

### Cell H4 — Layer 2: DC speed-height relationship

**Question:** Does mean body height correlate with mean forward speed (separate from the oscillation)?

This extends the Sandbox analysis (`analyze_height_speed_correlation()`) using the IK pipeline's speed computation and per-frame resolution.

```python
from scipy.stats import spearmanr, pearsonr

# ── Per-bout: mean height vs. mean speed ─────────────────────────────────────
bout_stats = df_valid.groupby('bout_id').agg(
    mean_speed=('forward_speed', 'mean'),
    median_speed=('forward_speed', 'median'),
    mean_height=('thorax_z_mm', 'mean'),
    std_height=('thorax_z_mm', 'std'),
    fly_id=('fly_id', 'first'),
).reset_index()
# Also add sex if session_metadata.csv was loaded (Part D)
if 'sex' in df_valid.columns:
    bout_stats['sex'] = bout_stats['fly_id'].map(
        df_valid.drop_duplicates('fly_id').set_index('fly_id')['sex'])

r_bout, p_bout = pearsonr(bout_stats['mean_speed'], bout_stats['mean_height'])

# ── Within-bout: frame-level correlation ─────────────────────────────────────
r_within = []
for bid, grp in df_valid.groupby('bout_id'):
    if len(grp) < 50:
        continue
    r, p = spearmanr(grp['forward_speed'], grp['thorax_z_mm'], nan_policy='omit')
    r_within.append({'bout_id': bid, 'r': r, 'p': p,
                     'fly_id': grp['fly_id'].iloc[0],
                     'mean_speed': grp['forward_speed'].mean()})
r_within_df = pd.DataFrame(r_within)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: per-bout scatter
colors_fly = {fid: plt.cm.tab10(i) for i, fid in enumerate(bout_stats['fly_id'].unique())}
for fid, grp in bout_stats.groupby('fly_id'):
    axes[0].scatter(grp['mean_speed'], grp['mean_height'],
                    c=[colors_fly[fid]], s=40, alpha=0.7, label=fid)
axes[0].set_xlabel('Mean forward speed (mm/s)')
axes[0].set_ylabel('Mean thorax height (mm)')
axes[0].set_title(f'Mean height vs. mean speed per bout\nPearson r={r_bout:.2f}, p={p_bout:.3f}')

# Add regression line
_xs = np.linspace(bout_stats['mean_speed'].min(), bout_stats['mean_speed'].max(), 100)
_slope, _int = np.polyfit(bout_stats['mean_speed'], bout_stats['mean_height'], 1)
axes[0].plot(_xs, _slope * _xs + _int, 'k--', lw=1.5)
axes[0].legend(fontsize=6, ncol=2)

# Middle: distribution of within-bout Spearman r
axes[1].hist(r_within_df['r'], bins=30, edgecolor='k')
axes[1].axvline(0, color='k', ls='--')
axes[1].axvline(r_within_df['r'].median(), color='r', ls='-', label=f'median r={r_within_df["r"].median():.2f}')
axes[1].set_xlabel('Within-bout Spearman r (speed vs. height)')
axes[1].set_ylabel('N bouts')
axes[1].set_title('Within-bout height-speed correlation')
axes[1].legend()

# Right: binned height vs. speed (pooled across all frames)
speed_bins_dc = pd.qcut(df_valid['forward_speed'], q=10)
height_by_speed = df_valid.groupby(speed_bins_dc, observed=True)['thorax_z_mm'].agg(['mean', 'sem'])
speed_centers = df_valid.groupby(speed_bins_dc, observed=True)['forward_speed'].mean()
axes[2].errorbar(speed_centers, height_by_speed['mean'], yerr=height_by_speed['sem'],
                 fmt='o-', capsize=3)
axes[2].set_xlabel('Forward speed bin (mm/s)')
axes[2].set_ylabel('Mean thorax height (mm)')
axes[2].set_title('Height vs. speed (decile bins, all frames)')

plt.suptitle('DC speed-height relationship', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'height_speed_dc.pdf', bbox_inches='tight')
plt.show()

# Print summary
print(f"Per-bout Pearson r = {r_bout:.3f} (p = {p_bout:.4f})")
print(f"Within-bout Spearman r: median = {r_within_df['r'].median():.3f}, "
      f"mean = {r_within_df['r'].mean():.3f}")
sign = 'higher' if r_bout > 0 else 'lower'
print(f"→ Faster flies walk {sign} on average.")
```

**What to look for:**
- Positive r → flies walk higher at higher speed (like humans; faster = more extended legs)
- Negative r → flies squat at high speed (like some insects)
- Near-zero r → height and speed decouple; depends on gait rather than speed

---

### Cell H5 — Layer 3: Walking vs. running — phase offset between height and speed oscillations

**This is the core test.** Compute the instantaneous phase offset between `thorax_z_osc` and the oscillatory component of `forward_speed`, per frame. Then ask: does this phase offset change with speed?

```python
from scipy.signal import butter, filtfilt, hilbert

# ── Bandpass forward_speed to isolate stride-frequency oscillation ─────────────
b_bp, a_bp = butter(3, [5 / (FPS / 2), 50 / (FPS / 2)], btype='band')

df_valid['speed_osc'] = np.nan
df_valid['height_speed_phase_diff'] = np.nan

for bid, grp in df_valid.groupby('bout_id', sort=False):
    idx = grp.index
    if len(idx) < 60:
        continue
    spd = grp['forward_speed'].values
    z_osc = grp['thorax_z_osc'].values

    if np.all(np.isnan(z_osc)):
        continue

    # Bandpass speed to same range
    spd_osc = filtfilt(b_bp, a_bp, spd)
    df_valid.loc[idx, 'speed_osc'] = spd_osc

    # Hilbert analytic signals → instantaneous phase difference
    z_analytic   = hilbert(z_osc)
    spd_analytic = hilbert(spd_osc)

    # Phase difference: φ_height - φ_speed
    # Walking: ~π (out of phase); Running: ~0 (in phase)
    phase_diff = np.angle(z_analytic * np.conj(spd_analytic))
    df_valid.loc[idx, 'height_speed_phase_diff'] = phase_diff

# ── Filter to active walking frames ───────────────────────────────────────────
# Exclude stopping, very slow frames, and frames with high turning rate
_activity_mask = (
    (df_valid['forward_speed'] > df_valid['forward_speed'].quantile(0.25)) &
    (df_valid['mean_abs_vel']  > df_valid['mean_abs_vel'].quantile(0.25))
)
df_walk = df_valid[_activity_mask].copy()

# ── Global summary ─────────────────────────────────────────────────────────────
from scipy.stats import circmean, circstd
_pd = df_walk['height_speed_phase_diff'].dropna().values
global_mean_phase = circmean(_pd, high=np.pi, low=-np.pi)
global_std_phase  = circstd(_pd, high=np.pi, low=-np.pi)

print(f"Global mean phase offset (height - speed): {np.degrees(global_mean_phase):.1f}°")
print(f"  (0° = in phase = running; ±180° = out of phase = walking)")
print(f"Circular std: {np.degrees(global_std_phase):.1f}°")

# ── Plot 1: phase offset distribution ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(_pd, bins=72, range=(-np.pi, np.pi), edgecolor='k', lw=0.3)
axes[0].axvline(global_mean_phase, color='r', lw=2,
                label=f'mean = {np.degrees(global_mean_phase):.0f}°')
axes[0].axvline(np.pi,  color='b', ls='--', lw=1, label='π = walking')
axes[0].axvline(-np.pi, color='b', ls='--', lw=1)
axes[0].axvline(0,      color='g', ls='--', lw=1, label='0 = running')
axes[0].set_xlabel('Phase offset (height − speed, rad)')
axes[0].set_ylabel('N frames')
axes[0].set_title('Height-speed phase offset distribution')
axes[0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axes[0].set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
axes[0].legend()

# ── Plot 2: phase offset vs. instantaneous speed ───────────────────────────────
# Bin by speed, compute circular mean phase per bin
speed_bins_h = pd.qcut(df_walk['forward_speed'], q=10)
phase_by_speed = df_walk.groupby(speed_bins_h, observed=True)['height_speed_phase_diff'].apply(
    lambda x: circmean(x.dropna().values, high=np.pi, low=-np.pi)
)
speed_centers_h = df_walk.groupby(speed_bins_h, observed=True)['forward_speed'].mean()

axes[1].plot(speed_centers_h, np.degrees(phase_by_speed), 'o-', lw=2, markersize=8)
axes[1].axhline(180,  color='b', ls='--', lw=1, label='180° = walking')
axes[1].axhline(-180, color='b', ls='--', lw=1)
axes[1].axhline(0,    color='g', ls='--', lw=1, label='0° = running')
axes[1].set_xlabel('Forward speed (mm/s)')
axes[1].set_ylabel('Circular mean phase offset (°)')
axes[1].set_title('Phase offset vs. speed\n(does it flip? → gait transition)')
axes[1].legend()
axes[1].set_ylim(-190, 190)

# ── Plot 3: per-fly phase offset (box/violin) ──────────────────────────────────
fly_order_h = df_walk.groupby('fly_id')['forward_speed'].median().sort_values().index.tolist()
import seaborn as sns
sns.boxplot(data=df_walk, x='fly_id', y='height_speed_phase_diff',
            order=fly_order_h, ax=axes[2])
axes[2].axhline(np.pi,  color='b', ls='--', lw=1)
axes[2].axhline(-np.pi, color='b', ls='--', lw=1)
axes[2].axhline(0, color='g', ls='--', lw=1)
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30, fontsize=8)
axes[2].set_ylabel('Phase offset height − speed (rad)')
axes[2].set_title('Per-fly phase offset distribution')

plt.suptitle('Walking vs. Running: height-speed phase relationship',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'walking_running_phase.pdf', bbox_inches='tight')
plt.show()
```

**How to interpret:**
- If the mean phase offset is near **±π (±180°)** at all speeds → flies walk at all recorded speeds (inverted pendulum throughout)
- If the mean phase offset shifts toward **0°** at high speeds → potential gait transition to running
- If there's a **bimodal distribution** in the histogram → two gait modes coexisting
- If the per-fly panel shows consistent ~π offset → the result is robust across individuals

---

### Cell H6 — Amplitude of oscillation vs. speed

**Question:** Does the body oscillate *more* at higher speeds, and does it oscillate differently at the same speed across flies?

```python
# Compute per-step-cycle amplitude of thorax_z_osc
# Assign step_cycle_id from existing column, compute RMS amplitude per cycle

cycle_amp = df_valid.groupby(['bout_id', 'step_cycle_id']).agg(
    speed=('step_cycle_mean_speed', 'first'),
    z_amp=('thorax_z_osc', lambda x: np.sqrt(np.nanmean(x**2))),  # RMS amplitude
    fly_id=('fly_id', 'first'),
).reset_index().dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: RMS amplitude vs. speed (scatter + binned mean)
axes[0].scatter(cycle_amp['speed'], cycle_amp['z_amp'],
                s=4, alpha=0.2, c='steelblue', rasterized=True)
speed_bins_a = pd.qcut(cycle_amp['speed'], q=8)
amp_by_speed = cycle_amp.groupby(speed_bins_a, observed=True)['z_amp'].agg(['mean', 'sem'])
spd_c_a = cycle_amp.groupby(speed_bins_a, observed=True)['speed'].mean()
axes[0].errorbar(spd_c_a, amp_by_speed['mean'], yerr=amp_by_speed['sem'],
                 fmt='ro-', lw=2, markersize=8, capsize=3, label='Mean ± SEM')
axes[0].set_xlabel('Step cycle mean speed (mm/s)')
axes[0].set_ylabel('Thorax Z RMS amplitude (mm)')
axes[0].set_title('Body oscillation amplitude vs. speed')
axes[0].legend()

# Right: per-fly mean amplitude
fly_order_a = cycle_amp.groupby('fly_id')['speed'].median().sort_values().index.tolist()
sns.boxplot(data=cycle_amp, x='fly_id', y='z_amp', order=fly_order_a, ax=axes[1])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, fontsize=8)
axes[1].set_ylabel('RMS amplitude (mm)')
axes[1].set_title('Per-fly oscillation amplitude')

plt.suptitle('Thorax height oscillation amplitude', fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'height_oscillation_amplitude.pdf', bbox_inches='tight')
plt.show()
```

---

### Cell H7 — Combined summary figure (paper-quality)

A 4-panel summary combining all layers: oscillation shape, DC relationship, phase test, and amplitude.

```python
fig = plt.figure(figsize=(18, 10))
gs  = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.4)

# Panel A: phase-averaged oscillation (slow vs. fast)
ax_a = fig.add_subplot(gs[0, :2])
colors_q = plt.cm.viridis(np.linspace(0.2, 0.9, 4))
# [re-plot phase average by quartile from Cell H2 data — use stored z_phase arrays]
ax_a.set_title('A  Body height vs. stride phase', fontweight='bold', loc='left')
ax_a.set_xlabel('T1_left phase (rad)')
ax_a.set_ylabel('ΔHeight (mm)')

# Panel B: DC height-speed
ax_b = fig.add_subplot(gs[0, 2:])
# [re-plot binned height vs. speed from Cell H4 data]
ax_b.set_title('B  Mean height vs. speed (DC)', fontweight='bold', loc='left')

# Panel C: phase offset distribution
ax_c = fig.add_subplot(gs[1, :2])
ax_c.hist(_pd, bins=72, range=(-np.pi, np.pi))
ax_c.set_title('C  Height-speed phase offset', fontweight='bold', loc='left')
ax_c.set_xlabel('Phase (rad)')

# Panel D: phase offset vs. speed
ax_d = fig.add_subplot(gs[1, 2:])
ax_d.plot(speed_centers_h, np.degrees(phase_by_speed), 'o-')
ax_d.set_title('D  Phase offset vs. speed', fontweight='bold', loc='left')
ax_d.set_xlabel('Speed (mm/s)')
ax_d.set_ylabel('Phase (°)')

plt.savefig(OUTPUT_DIR / 'walking_running_summary.pdf', bbox_inches='tight')
plt.show()
```

---

## Execution Order

```
Cell 1–14   (main pipeline — must be run first)
│
├── Cell H1  ← Add thorax_z_mm, thorax_z_detrended, thorax_z_osc to df_valid
│               (run once; adds 3 columns)
│
├── Cell H2  ← Phase-averaged oscillation + speed quartile overlay
│               prerequisite: H1
│
├── Cell H3  ← Hilbert phase scatter (height phase vs. stride phase)
│               prerequisite: H1
│
├── Cell H4  ← DC speed-height correlation (per-bout + frame-level)
│               prerequisite: H1
│
├── Cell H5  ← Walking vs. running: phase offset histogram + speed binning
│               prerequisite: H1; uses step_cycle_id from Cell 11
│
├── Cell H6  ← Amplitude vs. speed per step cycle
│               prerequisite: H1; uses step_cycle_id from Cell 11
│
└── Cell H7  ← Combined summary figure
                prerequisite: H2, H4, H5 data
```

---

## Key Variables After Running

| Variable | Cell | Description |
|---|---|---|
| `df_valid['thorax_z_mm']` | H1 | World-frame body height in mm |
| `df_valid['thorax_z_baseline']` | H1 | Slow baseline (500 ms rolling mean), mm |
| `df_valid['thorax_z_detrended']` | H1 | Height minus slow baseline (oscillatory+noise), mm |
| `df_valid['thorax_z_osc']` | H1 | Bandpass 5–50 Hz height oscillation, mm |
| `df_valid['speed_osc']` | H5 | Bandpass 5–50 Hz speed oscillation |
| `df_valid['height_speed_phase_diff']` | H5 | Instantaneous phase(height) − phase(speed) per frame |
| `bout_stats` | H4 | Per-bout mean height, speed, fly_id, sex |
| `r_within_df` | H4 | Per-bout Spearman r (height vs. speed) |
| `cycle_amp` | H6 | Per-step-cycle RMS height amplitude and mean speed |

---

## Decision Points

**After Cell H2 (phase average):**
- If the curve shows **2 peaks**: 2× frequency confirmed. Note the phases of the peaks — they should correspond to mid-swing of each tripod (T1L phase ≈ 0 and ≈ ±π for the two tripods).
- If the curve shows **1 peak**: the second harmonic is absent; body oscillates at stride frequency, not twice. Revisit the frequency assumption.
- If the peaks **shift with speed** (right panel): phase of oscillation is speed-dependent — important for the walking/running test.

**After Cell H4 (DC relationship):**
- **Positive slope** (higher = faster): flies extend legs at high speed. Common in insects, consistent with aerial phases at high speed.
- **Negative slope** (lower = faster): flies crouch at high speed. Would argue against running.
- **Flat**: height and speed decouple; gait mode may not change.
- **Within-bout r strongly positive** with high between-bout variance: the relationship varies by individual (behavioral strategy differences).

**After Cell H5 (phase test):**
- **Mean phase ≈ ±180°, constant across speed**: pure walking throughout. Case closed.
- **Mean phase starts at ±180° and decreases toward 0° at high speed**: evidence of gait transition. Find the speed at which the phase crosses ±90° — that's your putative walk→run transition speed.
- **High variance / bimodal distribution**: coexistence of walking and running patterns within the dataset. Separate by speed and re-plot.
- **Different phase for different flies**: inter-individual locomotion strategy difference (cross-reference with the UMAP inter-fly analysis).

---

## Notes on Signal Quality

**Potential confounds to be aware of:**
1. **Stopping transitions:** Near the start/end of bouts, the fly decelerates/accelerates. The bandpass filter will ring at these edges. Apply an activity mask (`forward_speed > quantile(0.25)`) before interpreting phase results.
2. **Turning:** During turns, the body yaw changes; this can affect the forward_speed computation and the apparent height. Consider filtering to low `turning_rate` frames for the cleanest result.
3. **Abdomen lifts:** These are discrete events (~30 ms) during near-stopping that shift the CoM upward. They are NOT part of the stride-frequency oscillation but could contaminate the baseline if they're frequent. Since they occur at near-zero speed, the activity mask should exclude most of them.
4. **Model joint limits:** The thorax Z from `xpos[:, 1, 2]` is from IK forward kinematics. The STAC model has abdomen joint limits (±0.15 rad) but the THORAX position is the free joint (no limits); it faithfully tracks the data. Use `kp_data.reshape(T,50,3)[:, 0, 2]` as a cross-check if needed.

---

## Relationship to UMAP Analysis

Once `height_speed_phase_diff` is in df_valid, it can be added as a color variable to the UMAP embeddings (Cell 29 `COLOR_BY` toggle). Frames where the phase offset is ~0° vs. ~π will occupy different regions of the UMAP if the embedding is capturing gait dynamics — this is a strong validation test for both analyses.

Also: the oscillation amplitude (Cell H6 `z_amp` per cycle) can be added to the step-cycle UMAP metadata (`sc_meta`) as another variable to color by in Cell 53.

---

## Per-Cycle Peak Phase Analysis (Cells P0–P1)

### Motivation

Phase-averaged traces (H2, S0) blur the signal: the std band reflects both biological variability and the fact that different cycles have different amplitudes, not just different phases. A complementary view skips the averaging entirely and asks directly: *at what gait phase do the signal peaks actually occur?* This avoids averaging in phase-space and lets the data speak as a distribution.

### Step 1 — Hilbert phase as a gait clock

For each frame, the T1_left leg tip speed `s(t)` is mean-centered and passed through the Hilbert transform to produce the analytic signal:

```
z(t) = s(t) + i·H{s(t)}
φ(t) = atan2( Im(z(t)), Re(z(t)) )    ∈ [-π, π]
```

`H{s}` is the 90°-phase-shifted copy of `s`, so `z(t)` is always a rotating phasor. `φ(t)` is its instantaneous angle — a continuous clock reading that advances monotonically through the stride cycle. The convention is fixed by the signal peak: `s` is maximal at mid-swing, so `φ = 0` at mid-swing. Tracing forward: touchdown ≈ `+π/2`, mid-stance ≈ `±π`, liftoff ≈ `-π/2`.

Every frame in the dataset now carries a gait phase value that is independent of stride duration or speed.

### Step 2 — Stride cycle segmentation

Step cycles are defined as liftoff-to-liftoff on T1_left. Liftoff = upward crossing of `φ = -π/2` (the moment leg speed starts rising from near-zero). Within each cycle, `φ` sweeps from `-π/2` back around to the next `-π/2` crossing.

### Step 3 — Peak detection within each cycle

Within each stride segment we apply `scipy.signal.find_peaks` to `thorax_z_detrended` and `forward_speed_osc` independently. We take the **top-2 peaks by prominence** (ranked by `props['prominences']`), then sort those two by frame position to label them `peak_rank=0` (early) and `peak_rank=1` (late).

For each detected peak at frame `t*` we record a single number: `φ(t*)`. This is the gait-clock reading at the exact moment the signal hit its maximum. No binning, no averaging — just one phase sample per peak per cycle.

Key parameters (tunable in P0):
- `Z_PROM = 0.3 µm` — minimum prominence to count as a height peak
- `SPD_PROM = 2.0 mm/s` — minimum prominence for a speed peak
- `PEAK_MIN_DIST = 15 frames` — minimum frame separation between peaks within one cycle

### Step 4 — Circular mean across cycles and flies

A set of phase samples `{φ₁, φ₂, …, φₙ}` cannot be averaged arithmetically because `-π` and `+π` are the same angle. The correct estimator is the **circular mean**:

```
μ = atan2( Σ sin(φᵢ),  Σ cos(φᵢ) )
```

Geometrically: project each angle onto the unit circle as a unit vector, sum the vectors, and take the angle of the resultant. If all samples cluster tightly, the resultant is long and the mean is well-defined. If they are uniformly scattered the resultant cancels toward zero — the circular analog of high variance.

Per-fly means are computed first (`scipy.stats.circmean` per fly), then shown as individual dots on the polar. The overall arrow is the circular mean over all cycles pooled.

### Step 5 — Rose histogram and polar layout

The rose histogram bins `{φᵢ}` into `N_PBINS = 24` equal-width bins over `[-π, π]` and plots `counts / counts.sum()` as bar height — i.e., the empirical probability distribution of peak phases. Normalising by total count makes the height signal and speed signal directly comparable in shape regardless of their different event counts.

The figure has one polar per speed quantile (`N_Q_POLAR`, default 3). On each polar:
- **Blue rose** = height peak phases (early + late pooled)
- **Red rose** = speed peak phases, same normalisation, 50% alpha
- **Arrows** = circular mean per (signal × rank): solid = early peak, dashed = late peak
- **Dots** = per-fly circular means: ○ = early peak, △ = late peak

### Interpreting the overlap

When the blue and red roses land on the same phase bins — and the arrows point in the same direction — it means: at the moment body height is maximal, forward speed is also maximal, and they are both locked to the same gait-clock reading. The fact that this holds consistently across all three speed quantiles rules out a speed-dependent confound and establishes a fixed kinematic coupling between body height oscillation and speed oscillation, both anchored to the T1_left stride phase.
