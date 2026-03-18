# Amputation Longitudinal Adaptation — Notebook Plan

*Generated 2026-03-11. Companion to `Amputation_Coordination.ipynb`.
Implement as `Amputation_Longitudinal.ipynb`.*

---

## Background and Motivation

T1L-amputated *Drosophila* undergo spontaneous behavioral recovery over 7 days.
Cross-sectional observations (0 d vs 7 d, different flies) show fewer falls, lower
walking speed, more anterior T2L foot placement, and a shift in leg coordination.
This notebook characterizes these changes **longitudinally** — the same 8 flies
(3 died before day 7) tracked daily from day 0 to day 7.

**No inverse kinematics available.** All metrics derive directly from JARVIS 3D
keypoints (`data3D.csv`).

---

## Data Organization

### Metadata CSV (`amputation_metadata.csv`)

One row per (fly, day) recording session. Create this file manually before running:

```
fly_id,day,sex,path
fly01,0,m,/home/user/src/JARVIS-HybridNet/projects/.../Predictions_3D_.../
fly01,1,m,/home/user/src/JARVIS-HybridNet/projects/.../Predictions_3D_.../
fly02,0,f,...
```

- `fly_id`: unique string per animal (e.g. `fly01`–`fly08`)
- `day`: integer 0–7
- `sex`: `m` or `f`
- `path`: folder containing `data3D.csv`

Missing (fly, day) combinations simply have no row (3 flies died mid-experiment).

### Keypoints in data3D.csv

| Region | Available keypoints | Notes |
|---|---|---|
| Body | `Antenna_Base`, `EyeL`, `EyeR`, `Scutellum`, `Abd_A4`, `Abd_tip` | Full chain |
| T1R (intact front right) | `T1R_ThxCx`, `T1R_Tro`, `T1R_FeTi`, `T1R_TiTa`, `T1R_TaT1`, `T1R_TaT3`, `T1R_TaTip` | Full leg |
| T2L, T2R, T3L, T3R | `{leg}_Tro`, `{leg}_FeTi`, `{leg}_TiTa`, `{leg}_TaT1`, `{leg}_TaT3`, `{leg}_TaTip` | **No ThxCx/coxa** |
| T1L (stump) | `T1L_ThxCx` only | — |

Scale factor: divide raw CSV units by `jarvis_scale = 10.0` to get mm.

### Egocentric Body Frame (computed from raw keypoints)

Used for all body-relative metrics (leg placement, leg spread).

- **Origin**: `Scutellum` world position
- **X (anterior/forward)**: unit vector from `Scutellum` → `Antenna_Base`,
  projected to the horizontal (world-XY) plane
- **Y (right lateral)**: X × world-Z
- **Z (up)**: world vertical

---

## Pipeline Overview

```
amputation_metadata.csv
        │
        ▼ Cell 0c — load metadata DataFrame
for each (fly_id, day, sex, path):
        ▼ Cell 1a — load_3d_data()  →  df_raw
        ▼ Cell 1b — detect_walking_bouts()  →  valid_bouts + rejected_bouts
        │
        ▼ SECTION 2 — QC: every bout, every fly, every day  ← INSPECT BEFORE PROCEEDING
        │    (speed, head Z, 5× leg world Z, XY trajectory, boundary reasons)
        │
        ▼ Cell 1c — compute_body_frame()  →  R, origin  (per session, done once)
        ▼ Cell 1d — detect_swing_stance()  →  swing_dict, ego_dict, wz_dict (per bout)
        ▼ Cell 1e — compute_bout_metrics()  →  dict of scalars + arrays
        ▼ Cell 3  — append to df_long + phase_store
        │
        ▼ Section 4 — longitudinal plots (metric vs day, per-fly traces, ±SEM by sex)
        ▼ Section 5 — phase coordination detail (polar histograms, R-matrix)
```

---

## Metrics Computed per Walking Bout

| Metric | Description |
|---|---|
| `speed_mm_s` | Mean Scutellum XY speed (mm/s) |
| `n_falls` | Fall event count (head Z dips) |
| `fall_rate_per_s` | Falls per second of walking |
| `fall_depth_mean_mm` | Mean fall prominence (mm) |
| `body_pitch_deg` | Mean elevation angle of Antenna_Base–Abd_A4 axis (°); + = head up |
| `height_over_legs_mm` | Mean Scutellum Z minus mean stance-leg tip Z (mm) |
| `leg_spread_mm2` | Mean convex hull area of 5 leg-tip egocentric XY positions (mm²) |
| `T2L_forward_ego_mm` | Mean egocentric X of T2L_TaTip during stance (mm); + = anterior |
| `mean_n_legs_stance` | Mean number of legs in stance per frame |
| `R_{leg_A}→{leg_B}` | Phase coordination R (mean resultant length, 0–1) per pair |

---

## Section 0 — Imports & Configuration

### Cell 0a — Imports

```python
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from scipy.stats import sem as scipy_sem, circmean
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
print("Imports OK")
```

### Cell 0b — Configuration

```python
CFG = {
    # ── Paths ─────────────────────────────────────────────────────────────
    "metadata_csv":   Path("amputation_metadata.csv"),  # edit path as needed
    "output_dir":     Path("output/amputation_longitudinal"),

    # ── Recording ─────────────────────────────────────────────────────────
    "fps":            800,
    "jarvis_scale":   10.0,      # raw CSV units → mm
    "conf_threshold": 0.8,
    "conf_gap_bridge":15,        # bridge confident segments across ≤ this many frames

    # ── Keypoints ──────────────────────────────────────────────────────────
    "intact_leg_tips": ["T1R_TaTip", "T2L_TaTip", "T2R_TaTip", "T3L_TaTip", "T3R_TaTip"],
    "stump_kp":        "T1L_ThxCx",
    "scutellum_kp":    "Scutellum",
    "antenna_kp":      "Antenna_Base",
    "abd_kp":          "Abd_A4",          # posterior landmark for pitch
    "head_kps":        ["Antenna_Base", "EyeL", "EyeR"],  # averaged for head Z

    # ── Walking bout detection ─────────────────────────────────────────────
    "min_walking_cycles":        2,      # minimum swing peaks per intact leg
    "min_body_displacement_mm":  3.0,
    "min_bout_duration_frames":  400,    # 0.5 s at 800 Hz
    "max_gap_frames":            100,    # bridge gaps ≤ this between valid regions

    # ── Swing / stance ─────────────────────────────────────────────────────
    # Method: Z-velocity threshold on world-frame leg-tip Z (same as
    # Amputation_Coordination.ipynb; tip rising fast → swing)
    "swing_z_vel_sigma":          3,     # Gaussian smoothing σ before differentiation
    "swing_z_vel_threshold_mm_s": 1.5,  # dZ/dt above this (mm/s) = swing

    # ── Fall detection ──────────────────────────────────────────────────────
    "fall_prominence_mm":         1.5,   # minimum head-Z dip depth (mm)
    "fall_min_duration_frames":   10,    # minimum dip width (frames)

    # ── Phase pairs (leg_A relative to leg_B's cycle) ───────────────────────
    "phase_pairs": [
        ("T1R", "T2L"),   # right tripod: are T1R and T2L still coupled?
        ("T1R", "T2R"),   # contralateral front–mid
        ("T2L", "T2R"),   # bilateral mid
        ("T3L", "T3R"),   # bilateral hind
        ("T2L", "T3L"),   # ipsilateral left
        ("T2R", "T3R"),   # ipsilateral right
        ("T1R", "T3R"),   # right tripod: T1R vs T3R
        ("T2R", "T3L"),   # left-tripod remnant
    ],

    # ── Arena & QC ─────────────────────────────────────────────────────────
    "arena_x_mm":              None,   # fill in actual arena width  (None = skip rectangle)
    "arena_y_mm":              None,   # fill in actual arena height (None = skip rectangle)
    "stationary_speed_threshold_mm_s": 2.0,  # below this = immobile (boundary reason)
    "qc_context_frames":       200,    # pre/post padding frames around each bout
}

FPS   = CFG["fps"]
SCALE = CFG["jarvis_scale"]
CFG["output_dir"].mkdir(parents=True, exist_ok=True)
print(f"Output dir: {CFG['output_dir']}")
```

### Cell 0c — Load Metadata

```python
meta = pd.read_csv(CFG["metadata_csv"])
assert {"fly_id", "day", "sex", "path"}.issubset(meta.columns), \
    f"Missing columns in metadata"
meta["path"] = meta["path"].apply(Path)
meta["day"]  = meta["day"].astype(int)

print(f"Metadata: {len(meta)} sessions — {meta['fly_id'].nunique()} flies, "
      f"days {meta['day'].min()}–{meta['day'].max()}")
print("\nFly × sex:")
print(meta.drop_duplicates('fly_id')[['fly_id', 'sex']].to_string(index=False))
print("\nDays per fly:")
print(meta.groupby('fly_id')['day'].apply(lambda d: sorted(d.tolist())).to_string())
```

---

## Section 1 — Helper Functions

### Cell 1a — Data Loading

```python
def load_3d_data(folder):
    """Load data3D.csv from a JARVIS Predictions_3D folder.
    Returns df (DataFrame) and kp_names (list of unique keypoint base names).
    """
    csv_path = Path(folder) / "data3D.csv"
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    df = df.iloc[:-1].reset_index(drop=True)   # drop last incomplete row
    seen, kp_names = set(), []
    for col in df.columns:
        base = col.split('.')[0]
        if base not in seen:
            seen.add(base)
            kp_names.append(base)
    return df, kp_names


def extract_xyzc(df, kp_name, scale=SCALE):
    """Return x, y, z (mm) and confidence for one keypoint."""
    cols = df.columns.tolist()
    idx  = cols.index(kp_name)
    x    = df.iloc[:, idx    ].values.astype(float) / scale
    y    = df.iloc[:, idx + 1].values.astype(float) / scale
    z    = df.iloc[:, idx + 2].values.astype(float) / scale
    conf = df.iloc[:, idx + 3].values.astype(float)
    return x, y, z, conf


def get_xyz(df, kp_name, scale=SCALE):
    """Return (T, 3) world-frame position (mm)."""
    x, y, z, _ = extract_xyzc(df, kp_name, scale)
    return np.column_stack([x, y, z])


def mask_low_confidence(df, kp_names, threshold=None, gap_bridge=None):
    """Set x/y/z to NaN for frames below confidence threshold. Modifies df in-place."""
    thr = threshold  or CFG["conf_threshold"]
    gap = gap_bridge or CFG["conf_gap_bridge"]
    for kp in kp_names:
        cols = df.columns.tolist()
        if kp not in cols:
            continue
        idx  = cols.index(kp)
        conf = df.iloc[:, idx + 3].values.astype(float)
        good = conf >= thr
        # Bridge short gaps in confidence
        in_gap = False; gap_start = 0
        for i in range(len(good)):
            if not good[i]:
                if not in_gap: gap_start = i; in_gap = True
            else:
                if in_gap:
                    if i - gap_start <= gap: good[gap_start:i] = True
                    in_gap = False
        bad = ~good
        df.iloc[bad, idx] = np.nan; df.iloc[bad, idx+1] = np.nan
        df.iloc[bad, idx+2] = np.nan
    return df


print("Data loading helpers defined.")
```

### Cell 1b — Walking Bout Detection

```python
def compute_scutellum_speed(df, scale=SCALE):
    """Per-frame Scutellum XY speed (mm/s) and raw x/y/z arrays."""
    sx, sy, sz, _ = extract_xyzc(df, CFG["scutellum_kp"], scale)
    disp = np.sqrt(np.diff(sx)**2 + np.diff(sy)**2)
    speed = np.concatenate([[disp[0] if len(disp) else 0.0], disp]) * FPS
    return speed, sx, sy, sz


def _count_leg_cycles(df, tip, s, e):
    """Count Z-velocity peaks (swing cycles) for one leg tip within frames [s, e]."""
    _, _, z, _ = extract_xyzc(df, tip)
    z_b = z[s:e+1].copy()
    nans = np.isnan(z_b)
    if nans.all(): return 0
    if nans.any():
        vi = np.where(~nans)[0]
        z_b[nans] = np.interp(np.where(nans)[0], vi, z_b[vi])
    z_sm = gaussian_filter1d(z_b.astype(float), sigma=5)
    peaks, _ = find_peaks(z_sm, prominence=0.05, distance=8, width=(1, 35))
    return len(peaks)


def detect_walking_bouts(df, kp_names):
    """Detect walking bouts; return both valid and rejected candidates.

    Returns:
        valid_bouts   : list of dicts with keys:
                          'start', 'end', 'duration_s', 'mean_speed',
                          'per_leg_cycles'       {tip: int},
                          'total_distance_mm'    float,
                          'net_displacement_mm'  float
        rejected_bouts: same keys plus 'rejection_reason' string
        speed_arr     : per-frame Scutellum speed (mm/s)
        scut_xyz      : (T, 3) Scutellum world positions (mm)
    """
    speed_arr, sx, sy, sz = compute_scutellum_speed(df)
    scut_xyz = np.column_stack([sx, sy, sz])

    # Per-frame swing activity score: count of intact legs with a swing peak nearby
    swing_score = np.zeros(len(df), dtype=int)
    for tip in CFG["intact_leg_tips"]:
        if tip not in kp_names:
            continue
        _, _, z, _ = extract_xyzc(df, tip)
        z_fill = z.copy()
        nans   = np.isnan(z_fill)
        if nans.all(): continue
        if nans.any():
            vi = np.where(~nans)[0]
            z_fill[nans] = np.interp(np.where(nans)[0], vi, z_fill[vi])
        z_sm   = gaussian_filter1d(z_fill, sigma=5)
        peaks, _ = find_peaks(z_sm, prominence=0.05, distance=8, width=(1, 35))
        for p in peaks:
            swing_score[max(0, p-20):min(len(df), p+20)] += 1

    gate = swing_score >= CFG["min_walking_cycles"]

    # Bridge short gaps in gate
    bridged = gate.copy()
    in_gap = False; gap_start = 0
    for i in range(len(bridged)):
        if not bridged[i]:
            if not in_gap: gap_start = i; in_gap = True
        else:
            if in_gap:
                if i - gap_start <= CFG["max_gap_frames"]: bridged[gap_start:i] = True
                in_gap = False

    # Collect contiguous regions as candidates
    candidates = []
    in_bout = False; bout_start = 0
    for i in range(len(bridged)):
        if bridged[i] and not in_bout:
            bout_start = i; in_bout = True
        elif not bridged[i] and in_bout:
            candidates.append((bout_start, i - 1)); in_bout = False
    if in_bout:
        candidates.append((bout_start, len(bridged) - 1))

    valid_bouts = []; rejected_bouts = []

    for (bs, be) in candidates:
        dur_frames = be - bs + 1
        x_seg = sx[bs:be+1]; y_seg = sy[bs:be+1]
        ok = ~(np.isnan(x_seg) | np.isnan(y_seg))

        # Compute path metrics
        total_dist = float(np.nansum(np.sqrt(
            np.diff(x_seg[ok])**2 + np.diff(y_seg[ok])**2))) if ok.sum() > 1 else 0.0
        net_disp = float(np.sqrt(
            (x_seg[ok][-1]-x_seg[ok][0])**2 +
            (y_seg[ok][-1]-y_seg[ok][0])**2)) if ok.sum() > 1 else 0.0

        # Per-leg cycle counts within this window
        per_leg = {}
        for tip in CFG["intact_leg_tips"]:
            if tip in kp_names:
                per_leg[tip] = _count_leg_cycles(df, tip, bs, be)

        bout_info = {
            'start':              bs,
            'end':                be,
            'duration_s':         dur_frames / FPS,
            'mean_speed':         float(np.nanmean(speed_arr[bs:be+1])),
            'per_leg_cycles':     per_leg,
            'total_distance_mm':  total_dist,
            'net_displacement_mm': net_disp,
        }

        # ── Rejection tests (in order of priority) ────────────────────────
        if dur_frames < CFG["min_bout_duration_frames"]:
            bout_info['rejection_reason'] = (
                f"too short: {dur_frames} frames < "
                f"{CFG['min_bout_duration_frames']} required")
            rejected_bouts.append(bout_info)
            continue

        if net_disp < CFG["min_body_displacement_mm"]:
            bout_info['rejection_reason'] = (
                f"low displacement: {net_disp:.2f} mm < "
                f"{CFG['min_body_displacement_mm']} mm required")
            rejected_bouts.append(bout_info)
            continue

        failing_legs = [t for t, c in per_leg.items()
                        if c < CFG["min_walking_cycles"]]
        if failing_legs:
            bout_info['rejection_reason'] = (
                f"legs with <{CFG['min_walking_cycles']} cycles: "
                + ", ".join(t.replace('_TaTip', '') for t in failing_legs))
            rejected_bouts.append(bout_info)
            continue

        valid_bouts.append(bout_info)

    return valid_bouts, rejected_bouts, speed_arr, scut_xyz


print("Walking bout detection defined.")
```

### Cell 1c — Egocentric Body Frame

```python
def compute_body_frame(df, scale=SCALE):
    """Per-frame egocentric body frame from Scutellum + Antenna_Base.

    Returns:
        R      : (T, 3, 3) rotation matrices;  R[t] @ world_vec  →  ego_vec
                   row 0 = anterior (X),  row 1 = right (Y),  row 2 = up (Z)
        origin : (T, 3) Scutellum world position (mm)

    Coordinate convention:
        ego_x > 0  →  anterior (toward head)
        ego_y > 0  →  fly's right
        ego_z > 0  →  above Scutellum
    """
    scut = get_xyz(df, CFG["scutellum_kp"], scale)   # (T, 3)
    ant  = get_xyz(df, CFG["antenna_kp"],    scale)   # (T, 3)
    T    = len(scut)

    fwd  = ant - scut                       # Scutellum → Antenna_Base
    fwd[:, 2] = 0.0                         # project to horizontal plane
    norm = np.linalg.norm(fwd, axis=1, keepdims=True)
    bad  = norm[:, 0] < 1e-3               # undefined when antenna ≈ scutellum
    fwd[bad] = [1.0, 0.0, 0.0]; norm[bad] = 1.0
    fwd /= norm                             # unit anterior vectors (T, 3)

    up    = np.zeros((T, 3)); up[:, 2] = 1.0   # world Z
    right = np.cross(fwd, up)
    right /= (np.linalg.norm(right, axis=1, keepdims=True) + 1e-8)

    R = np.stack([fwd, right, up], axis=1)  # (T, 3, 3)  — rows are axes
    return R, scut


def world_to_ego(world_xyz, R, origin):
    """Transform (T, 3) world positions to body-egocentric frame."""
    rel = world_xyz - origin                      # (T, 3)
    return np.einsum('tij,tj->ti', R, rel)        # R[t] @ rel[t]  for each t


print("Egocentric frame functions defined.")
```

### Cell 1d — Swing / Stance Detection (Z-velocity threshold, per-bout)

```python
def detect_swing_stance(df, R, origin, bout_start, bout_end, scale=SCALE):
    """Swing (1) / stance (0) per intact leg, plus egocentric positions.

    Method: dZ/dt > threshold → swing (1); else stance (0).
    Applied within the bout window to avoid cross-bout edge artifacts.

    Returns:
        swing_dict   : {tip → (n_frames,) int8}  — 1=swing, 0=stance
        ego_dict     : {tip → (n_frames, 3) mm}  — egocentric positions
        world_z_dict : {tip → (n_frames,) mm}    — world-frame Z
    """
    s, e    = bout_start, bout_end + 1
    R_b     = R[s:e]; orig_b = origin[s:e]
    sigma   = CFG["swing_z_vel_sigma"]
    thresh  = CFG["swing_z_vel_threshold_mm_s"]

    swing_dict = {}; ego_dict = {}; world_z_dict = {}

    for tip in CFG["intact_leg_tips"]:
        wx, wy, wz, _ = extract_xyzc(df, tip, scale)
        wx_b = wx[s:e]; wy_b = wy[s:e]; wz_b = wz[s:e]

        # Fill NaN before differentiation
        wz_fill = wz_b.copy()
        nans = np.isnan(wz_fill)
        if not nans.all() and nans.any():
            vi = np.where(~nans)[0]
            wz_fill[nans] = np.interp(np.where(nans)[0], vi, wz_fill[vi])

        dz_dt = np.gradient(gaussian_filter1d(wz_fill.astype(float), sigma)) * FPS
        sw    = (dz_dt > thresh).astype(np.int8)
        sw[nans] = 0   # treat missing frames as stance

        world_pos = np.column_stack([wx_b, wy_b, wz_b])
        ego_pos   = world_to_ego(world_pos, R_b, orig_b)

        swing_dict[tip]    = sw
        ego_dict[tip]      = ego_pos
        world_z_dict[tip]  = wz_b

    return swing_dict, ego_dict, world_z_dict


print("Swing/stance detection defined.")
```

### Cell 1e — Metric Computation Functions

```python
def compute_head_z(df, scale=SCALE):
    """Mean Z (mm) of head keypoints per frame."""
    return np.nanmean(
        np.vstack([extract_xyzc(df, kp, scale)[2] for kp in CFG["head_kps"]]),
        axis=0
    )


def detect_falls(head_z_bout):
    """Detect downward dips in head Z. Returns (peak_indices, prominences, widths)."""
    inv = -head_z_bout.astype(float)
    nans = np.isnan(inv)
    if nans.all(): return np.array([]), np.array([]), np.array([])
    if nans.any():
        vi = np.where(~nans)[0]
        inv[nans] = np.interp(np.where(nans)[0], vi, inv[vi])
    peaks, props = find_peaks(inv, prominence=CFG["fall_prominence_mm"],
                              width=CFG["fall_min_duration_frames"])
    return peaks, props.get('prominences', np.array([])), props.get('widths', np.array([]))


def compute_body_pitch(df, scale=SCALE):
    """Per-frame body pitch (rad). + = head higher than abdomen (head-up posture)."""
    ant = get_xyz(df, CFG["antenna_kp"], scale)   # (T, 3)
    abd = get_xyz(df, CFG["abd_kp"],     scale)   # (T, 3)
    vec = ant - abd                               # posterior → anterior
    return np.arctan2(vec[:, 2], np.sqrt(vec[:, 0]**2 + vec[:, 1]**2))


def compute_height_over_legs(scut_z, world_z_dict, swing_dict):
    """Per-frame Scutellum Z minus mean stance-leg tip Z (mm).
    NaN when all legs in swing. Both inputs must be same length (one bout)."""
    tips    = list(world_z_dict.keys())
    leg_z   = np.column_stack([world_z_dict[t] for t in tips])
    stance  = np.column_stack([(1 - swing_dict[t]) for t in tips]).astype(bool)
    mean_gnd = np.nanmean(np.where(stance, leg_z, np.nan), axis=1)
    return scut_z - mean_gnd


def compute_leg_spread_ego(ego_dict, subsample=4):
    """Per-frame convex hull area (mm²) of 5 leg-tip egocentric XY positions.

    Args:
        ego_dict  : {tip → (n_frames, 3)}
        subsample : compute every Nth frame to keep runtime manageable at 800 Hz.
    Returns:
        area_full : (n_frames,) interpolated to full frame rate.
    """
    tips    = list(ego_dict.keys())
    T       = len(next(iter(ego_dict.values())))
    pts_all = np.stack([ego_dict[t][:, :2] for t in tips], axis=1)  # (T, n, 2)
    idx     = np.arange(0, T, subsample)
    area_sub = np.full(len(idx), np.nan)
    for ii, t in enumerate(idx):
        pts   = pts_all[t]
        valid = ~np.any(np.isnan(pts), axis=1)
        if valid.sum() < 3: continue
        try:
            area_sub[ii] = ConvexHull(pts[valid]).volume  # 2-D: volume = area
        except Exception:
            pass
    # Interpolate back to full frame rate
    area_full = np.interp(np.arange(T), idx, area_sub,
                          left=np.nan, right=np.nan)
    return area_full


def compute_T2L_forward_stance(ego_dict, swing_dict):
    """Mean egocentric X of T2L_TaTip during stance (mm). + = forward/anterior."""
    tip = "T2L_TaTip"
    if tip not in ego_dict: return np.nan
    x_ego  = ego_dict[tip][:, 0]
    stance = (swing_dict[tip] == 0) & ~np.isnan(x_ego)
    return float(np.nanmean(x_ego[stance])) if stance.any() else np.nan


def _get_swing_onsets(sw_binary):
    """Frame indices where swing starts (0 → 1 transitions)."""
    return np.where(np.diff(sw_binary.astype(int), prepend=0) == 1)[0]


def compute_phase_offsets(swing_dict):
    """Onset-based phase offsets for all configured pairs.
    Returns {label: (phases_rad_array, R_scalar)}."""
    results = {}
    for leg_a, leg_b in CFG["phase_pairs"]:
        tip_a = f"{leg_a}_TaTip"; tip_b = f"{leg_b}_TaTip"
        if tip_a not in swing_dict or tip_b not in swing_dict: continue
        ons_a = _get_swing_onsets(swing_dict[tip_a])
        ons_b = _get_swing_onsets(swing_dict[tip_b])
        if len(ons_a) < 1 or len(ons_b) < 2: continue
        phases = []
        for t_a in ons_a:
            prec = ons_b[ons_b < t_a]
            if not len(prec): continue
            t0 = prec[-1]
            fol = ons_b[ons_b > t0]
            if not len(fol): continue
            period = fol[0] - t0
            if period <= 0: continue
            phases.append((t_a - t0) / period * 2 * np.pi)
        if not phases: continue
        phases = np.array(phases)
        R = float(np.abs(np.mean(np.exp(1j * phases))))
        results[f"{leg_a}→{leg_b}"] = (phases, R)
    return results


def compute_bout_metrics(df, bout, R_mat, origin, head_z_full, speed_arr, scut_z,
                         scale=SCALE):
    """Compute all per-bout scalar metrics + store arrays for phase plots."""
    s, e = bout['start'], bout['end']
    dur_s = (e - s + 1) / FPS

    swing_dict, ego_dict, wz_dict = detect_swing_stance(df, R_mat, origin, s, e, scale)

    # ── Speed ──────────────────────────────────────────────────────────────
    mean_speed = float(np.nanmean(speed_arr[s:e+1]))

    # ── Falls ───────────────────────────────────────────────────────────────
    head_z_b  = head_z_full[s:e+1]
    fall_pk, fall_pr, _ = detect_falls(head_z_b)
    n_falls       = len(fall_pk)
    fall_rate     = n_falls / dur_s
    fall_depth    = float(np.nanmean(fall_pr)) if len(fall_pr) else np.nan

    # ── Body pitch ──────────────────────────────────────────────────────────
    pitch_full    = compute_body_pitch(df, scale)
    mean_pitch    = float(np.degrees(np.nanmean(pitch_full[s:e+1])))

    # ── Height over legs ────────────────────────────────────────────────────
    scut_z_b      = scut_z[s:e+1]
    height        = compute_height_over_legs(scut_z_b, wz_dict, swing_dict)
    mean_height   = float(np.nanmean(height))

    # ── Leg spread ───────────────────────────────────────────────────────────
    spread        = compute_leg_spread_ego(ego_dict)
    mean_spread   = float(np.nanmean(spread))

    # ── T2L forward placement ───────────────────────────────────────────────
    T2L_fwd       = compute_T2L_forward_stance(ego_dict, swing_dict)

    # ── n_legs_stance ────────────────────────────────────────────────────────
    n_stance = np.sum(
        np.column_stack([1 - swing_dict[t] for t in swing_dict]), axis=1
    )
    mean_n_stance = float(np.nanmean(n_stance))

    # ── Phase offsets ────────────────────────────────────────────────────────
    phase_res = compute_phase_offsets(swing_dict)
    phase_R   = {k: v[1] for k, v in phase_res.items()}

    return {
        'duration_s':            dur_s,
        'speed_mm_s':            mean_speed,
        'n_falls':               n_falls,
        'fall_rate_per_s':       fall_rate,
        'fall_depth_mean_mm':    fall_depth,
        'body_pitch_deg':        mean_pitch,
        'height_over_legs_mm':   mean_height,
        'leg_spread_mm2':        mean_spread,
        'T2L_forward_ego_mm':    T2L_fwd,
        'mean_n_legs_stance':    mean_n_stance,
        **{f'R_{k}': v for k, v in phase_R.items()},
        # Internal arrays (prefixed _) — used by Section 5, excluded from df_long
        '_phase_offsets':  phase_res,   # {label: (phases_arr, R)}
        '_swing_dict':     swing_dict,
        '_ego_dict':       ego_dict,
        '_head_z':         head_z_b,
        '_speed_arr':      speed_arr[s:e+1],
    }


print("Metric functions defined.")
```

---

## Section 2 — QC: Full Bout Visualization (All Flies, All Days)

**Run this section before Section 3.**
Every detected bout — valid AND rejected — for every fly on every day is plotted
with pre/post context padding, an XY chamber trajectory, and annotated boundary
reasons. Inspect these PDFs in `output/amputation_longitudinal/qc/` before
accepting the batch results.

### Cell 2a — Colour Palette and Boundary-Reason Helper

```python
# ── Colour palette for leg traces ────────────────────────────────────────────
LEG_COLOR = {
    'T1R_TaTip': '#457B9D', 'T2L_TaTip': '#2D6A4F',
    'T2R_TaTip': '#B5838D', 'T3L_TaTip': '#6D6875', 'T3R_TaTip': '#E07A5F',
}
STANCE_ALPHA = 0.22
QC_VALID_COLOR   = '#2ca02c'   # green title bar for valid bouts
QC_REJECT_COLOR  = '#d62728'   # red title bar for rejected bouts
QC_CTX = CFG["qc_context_frames"]


def _boundary_reason(speed_arr, scut_xyz, kp_names, df, s, e, n_check=25):
    """Infer why a bout started and ended at frames s and e.

    Checks n_check frames just outside [s, e]:
      - Immobility  : mean speed < stationary_speed_threshold
      - Conf loss   : ≥ 2 intact leg tips have NaN (masked by mask_low_confidence)
      - Arena wall  : Scutellum XY near arena boundary (if arena_x_mm set in CFG)
      - Recording   : boundary is at frame 0 or len(df)-1

    Returns (start_reason, end_reason) — both plain strings.
    """
    T = len(df)
    thr_spd = CFG["stationary_speed_threshold_mm_s"]

    def _reason_at(boundary_frame, direction):
        # direction: 'before' = look left of s, 'after' = look right of e
        if direction == 'before':
            frames = np.arange(max(0, boundary_frame - n_check), boundary_frame)
        else:
            frames = np.arange(boundary_frame + 1,
                               min(T, boundary_frame + 1 + n_check))

        if len(frames) == 0:
            return "recording edge"
        if frames[0] == 0 or frames[-1] == T - 1:
            return "recording edge"

        # Low speed → immobility
        sp_local = speed_arr[frames]
        if np.nanmean(sp_local) < thr_spd:
            return "immobility"

        # Confidence loss on ≥ 2 legs
        tips_present = [t for t in CFG["intact_leg_tips"] if t in kp_names]
        n_bad_legs = 0
        for tip in tips_present:
            _, _, z, _ = extract_xyzc(df, tip)
            if np.sum(np.isnan(z[frames])) > len(frames) * 0.5:
                n_bad_legs += 1
        if n_bad_legs >= 2:
            return "tracking failure"

        # Arena wall (optional)
        if CFG.get("arena_x_mm") and CFG.get("arena_y_mm"):
            xy = scut_xyz[frames, :2]
            ax_lim = CFG["arena_x_mm"] / 2
            ay_lim = CFG["arena_y_mm"] / 2
            margin = 3.0   # mm from edge counts as "wall"
            near = (np.abs(xy[:, 0]) > ax_lim - margin) | \
                   (np.abs(xy[:, 1]) > ay_lim - margin)
            if near.any():
                return "arena wall"

        return "unknown"

    start_reason = _reason_at(s, 'before')
    end_reason   = _reason_at(e, 'after')
    return start_reason, end_reason


print("QC helpers defined.")
```

### Cell 2b — `plot_bout_qc()`: Single-Bout Figure

```python
def plot_bout_qc(df, bout_info, valid_bouts, rejected_bouts,
                 speed_arr, scut_xyz, kp_names,
                 fly_id='', day=0, bout_global_idx=0,
                 scale=SCALE, save_path=None):
    """Plot one bout: time-series panels (left 75%) + XY trajectory (right 25%).

    Layout
    ------
    Left column (GridSpec width_ratios 3:1):
      Row 0  : Scutellum speed  + bout boundary annotations
      Row 1  : Head Z + fall markers
      Rows 2–6: World-frame Z of each intact leg tip + swing/stance shading
    Right column (spans all rows):
      XY chamber trajectory with viridis time gradient;
      pre/post context in gray dashed/dotted;
      arena rectangle if arena_x_mm/arena_y_mm in CFG.

    Bout status:
      Valid   → green (#2ca02c) background on suptitle bar
      Rejected→ red   (#d62728) background on suptitle bar; reason shown
    """
    from matplotlib.collections import LineCollection
    from matplotlib.gridspec import GridSpec

    s   = bout_info['start']
    e   = bout_info['end']
    ctx = QC_CTX
    T   = len(df)

    s_ctx = max(0, s - ctx)
    e_ctx = min(T - 1, e + ctx)

    is_valid = 'rejection_reason' not in bout_info
    status_color = QC_VALID_COLOR if is_valid else QC_REJECT_COLOR

    # ── Swing/stance within the bout only ─────────────────────────────────
    # Use a minimal body-frame computation for this bout
    R_mat, origin = compute_body_frame(df, scale)
    swing_dict, _, world_z_dict = detect_swing_stance(df, R_mat, origin, s, e, scale)
    legs = [t for t in CFG["intact_leg_tips"] if t in world_z_dict]

    n_ts_panels = 2 + len(legs)
    fig = plt.figure(figsize=(16, max(6, 1.55 * n_ts_panels)))
    gs  = GridSpec(n_ts_panels, 2, figure=fig,
                   width_ratios=[3, 1], hspace=0.06, wspace=0.08)

    axes_left = [fig.add_subplot(gs[i, 0]) for i in range(n_ts_panels)]
    ax_xy     = fig.add_subplot(gs[:, 1])

    # ── Time axis: absolute session time in seconds ───────────────────────
    t_full = np.arange(s_ctx, e_ctx + 1) / FPS    # seconds (absolute)
    t_bout = np.arange(s, e + 1) / FPS
    t_pre  = np.arange(s_ctx, s)   / FPS
    t_post = np.arange(e + 1, e_ctx + 1) / FPS

    bout_span_s  = s  / FPS
    bout_span_e  = (e + 1) / FPS

    def _shade_bout(ax):
        ax.axvspan(bout_span_s, bout_span_e, color=status_color, alpha=0.07,
                   zorder=0, linewidth=0)

    # ── Panel 0: Scutellum speed ──────────────────────────────────────────
    ax = axes_left[0]
    _shade_bout(ax)
    ax.plot(np.arange(s_ctx, s)    / FPS, speed_arr[s_ctx:s],
            color='#aaaaaa', lw=0.7, ls='--', zorder=1)
    ax.plot(t_bout, speed_arr[s:e+1], color='k', lw=0.9, zorder=2)
    ax.plot(np.arange(e+1, e_ctx+1) / FPS, speed_arr[e+1:e_ctx+1],
            color='#aaaaaa', lw=0.7, ls=':', zorder=1)
    ax.axhline(CFG["stationary_speed_threshold_mm_s"],
               color='#888888', lw=0.6, ls='--')
    ax.set_ylabel('Speed\n(mm/s)', fontsize=8)

    # Boundary reason annotations
    start_reason, end_reason = _boundary_reason(
        speed_arr, scut_xyz, kp_names, df, s, e)
    for (x_pos, reason, ha) in [(bout_span_s, f'◀ {start_reason}', 'left'),
                                  (bout_span_e, f'{end_reason} ▶', 'right')]:
        ax.annotate(
            reason,
            xy=(x_pos, 1), xycoords=('data', 'axes fraction'),
            xytext=(6 if ha == 'left' else -6, -8), textcoords='offset points',
            fontsize=7.5, color=QC_REJECT_COLOR, fontweight='bold',
            ha=ha, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=QC_REJECT_COLOR, alpha=0.85)
        )

    # ── Panel 1: Head Z + fall markers ────────────────────────────────────
    ax = axes_left[1]
    _shade_bout(ax)
    head_z_full = compute_head_z(df, scale)
    ax.plot(np.arange(s_ctx, s)    / FPS, head_z_full[s_ctx:s],
            color='#aaaaaa', lw=0.7, ls='--')
    ax.plot(t_bout, head_z_full[s:e+1], color='#222222', lw=0.9)
    ax.plot(np.arange(e+1, e_ctx+1) / FPS, head_z_full[e+1:e_ctx+1],
            color='#aaaaaa', lw=0.7, ls=':')
    fpk, fpr, _ = detect_falls(head_z_full[s:e+1])
    if len(fpk):
        ax.scatter((fpk + s) / FPS, head_z_full[s:e+1][fpk],
                   color='red', s=28, zorder=5, label=f'Falls n={len(fpk)}')
        ax.legend(fontsize=7, loc='upper right', framealpha=0.6)
    ax.set_ylabel('Head Z\n(mm)', fontsize=8)

    # ── Panels 2+: Leg tip world Z + swing/stance shading ─────────────────
    for pi, tip in enumerate(legs):
        ax  = axes_left[2 + pi]
        col = LEG_COLOR.get(tip, '#888888')
        _shade_bout(ax)

        _, _, wz_full, _ = extract_xyzc(df, tip, scale)
        ax.plot(np.arange(s_ctx, s)    / FPS, wz_full[s_ctx:s],
                color='#cccccc', lw=0.7, ls='--')
        ax.plot(t_bout, world_z_dict[tip], color=col, lw=0.9)
        ax.plot(np.arange(e+1, e_ctx+1) / FPS, wz_full[e+1:e_ctx+1],
                color='#cccccc', lw=0.7, ls=':')
        ax.set_ylabel(tip.replace('_TaTip', '') + '\nZ (mm)', fontsize=8)

        # Shade stance periods within the bout only
        sw = swing_dict[tip]
        in_st = False; st0 = 0
        for fi in range(len(sw)):
            if sw[fi] == 0 and not in_st:
                st0 = fi; in_st = True
            elif sw[fi] == 1 and in_st:
                ax.axvspan((s + st0) / FPS, (s + fi) / FPS,
                           alpha=STANCE_ALPHA, color=col, linewidth=0)
                in_st = False
        if in_st:
            ax.axvspan((s + st0) / FPS, (e + 1) / FPS,
                       alpha=STANCE_ALPHA, color=col, linewidth=0)

    axes_left[-1].set_xlabel('Session time (s)', fontsize=8)
    for ax in axes_left:
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        # Share x across all left panels
        ax.set_xlim(s_ctx / FPS, e_ctx / FPS)

    # ── XY trajectory panel ───────────────────────────────────────────────
    ax_xy.set_aspect('equal')

    # Pre context: gray dashed
    if s_ctx < s:
        ax_xy.plot(scut_xyz[s_ctx:s, 0], scut_xyz[s_ctx:s, 1],
                   color='#aaaaaa', lw=0.8, ls='--', zorder=1)
    # Post context: gray dotted
    if e + 1 < e_ctx:
        ax_xy.plot(scut_xyz[e+1:e_ctx+1, 0], scut_xyz[e+1:e_ctx+1, 1],
                   color='#aaaaaa', lw=0.8, ls=':', zorder=1)

    # Bout trajectory: viridis gradient
    x_b = scut_xyz[s:e+1, 0]; y_b = scut_xyz[s:e+1, 1]
    pts = np.array([x_b, y_b]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    t_norm = np.linspace(0, 1, max(1, len(segs)))
    lc = LineCollection(segs, cmap='viridis', norm=plt.Normalize(0, 1), lw=2.0, zorder=2)
    lc.set_array(t_norm)
    ax_xy.add_collection(lc)
    ax_xy.scatter(x_b[0],  y_b[0],  color=QC_VALID_COLOR,  s=40, zorder=5,
                  marker='o', label='start')
    ax_xy.scatter(x_b[-1], y_b[-1], color=QC_REJECT_COLOR,  s=40, zorder=5,
                  marker='^', label='end')

    # Arena rectangle
    if CFG.get("arena_x_mm") and CFG.get("arena_y_mm"):
        from matplotlib.patches import Rectangle
        ax_xy.add_patch(Rectangle(
            (-CFG["arena_x_mm"]/2, -CFG["arena_y_mm"]/2),
            CFG["arena_x_mm"], CFG["arena_y_mm"],
            linewidth=1.2, edgecolor='#555555', facecolor='none', ls='--', zorder=0))

    ax_xy.autoscale_view()
    ax_xy.legend(fontsize=6, loc='upper right', framealpha=0.7)
    ax_xy.set_xlabel('X (mm)', fontsize=8); ax_xy.set_ylabel('Y (mm)', fontsize=8)
    ax_xy.tick_params(labelsize=7)
    ax_xy.spines[['top', 'right']].set_visible(False)

    # ── Colour map bar (minimal) ───────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_xy, fraction=0.06, pad=0.04)
    cb.set_label('bout time (0→1)', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # ── Title ──────────────────────────────────────────────────────────────
    per_leg_str = '  '.join(f"{t.replace('_TaTip','')}: {c}"
                             for t, c in bout_info.get('per_leg_cycles', {}).items())
    status_str  = 'VALID' if is_valid else f"REJECTED — {bout_info.get('rejection_reason','')}"
    title = (f"fly={fly_id}  day={day}  bout#{bout_global_idx:03d}  [{status_str}]\n"
             f"dur={bout_info['duration_s']:.2f}s  "
             f"speed={bout_info['mean_speed']:.1f}mm/s  "
             f"net_disp={bout_info['net_displacement_mm']:.1f}mm  "
             f"cycles: {per_leg_str}")
    fig.suptitle(title, fontsize=8.5, fontweight='bold',
                 color='white',
                 bbox=dict(facecolor=status_color, alpha=0.90,
                           boxstyle='round,pad=0.4', edgecolor='none'))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)


print("plot_bout_qc() defined.")
```

### Cell 2c — `run_qc_all_sessions()`: Batch QC Runner

```python
def run_qc_all_sessions(meta, show_plots=False):
    """Generate QC figures for every bout (valid + rejected) in every session.

    For each (fly_id, day):
      1. Load + mask_low_confidence
      2. detect_walking_bouts → valid_bouts, rejected_bouts
      3. Sort all bouts by start frame
      4. Call plot_bout_qc() for each → save PDF to qc/ subfolder
      5. Build summary DataFrame row per bout

    Returns
    -------
    qc_df : DataFrame with columns:
              fly_id, day, sex, bout_global_idx, status (valid/rejected),
              rejection_reason, start, end, duration_s, mean_speed,
              net_displacement_mm, per_leg_cycles (dict as str)
    """
    from matplotlib import use as mpl_use
    if not show_plots:
        mpl_use('Agg')   # suppress interactive windows during batch

    qc_dir = CFG["output_dir"] / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    n_sess = len(meta)

    for ridx, row in meta.iterrows():
        fly_id = row['fly_id']
        day    = int(row['day'])
        sex    = row['sex']
        path   = row['path']
        print(f"[{ridx+1}/{n_sess}] fly={fly_id}  day={day}", end='  ')

        df_s, kp_s = load_3d_data(path)
        df_s       = mask_low_confidence(df_s, kp_s)
        v_bouts, r_bouts, speed_arr, scut_xyz = detect_walking_bouts(df_s, kp_s)
        print(f"{len(v_bouts)} valid  {len(r_bouts)} rejected")

        # Merge + sort by start frame; tag each with status
        all_bouts = (
            [dict(b, _status='valid')    for b in v_bouts] +
            [dict(b, _status='rejected') for b in r_bouts]
        )
        all_bouts.sort(key=lambda b: b['start'])

        for g_idx, bout_info in enumerate(all_bouts):
            is_valid = (bout_info['_status'] == 'valid')
            tag = 'valid' if is_valid else 'rejected'
            fname = (f"{fly_id}_day{day:02d}_bout{g_idx:03d}_{tag}.pdf")
            save_path = qc_dir / fname

            plot_bout_qc(
                df=df_s,
                bout_info=bout_info,
                valid_bouts=v_bouts,
                rejected_bouts=r_bouts,
                speed_arr=speed_arr,
                scut_xyz=scut_xyz,
                kp_names=kp_s,
                fly_id=fly_id,
                day=day,
                bout_global_idx=g_idx,
                save_path=save_path,
            )

            summary_rows.append({
                'fly_id':            fly_id,
                'day':               day,
                'sex':               sex,
                'bout_global_idx':   g_idx,
                'status':            tag,
                'rejection_reason':  bout_info.get('rejection_reason', ''),
                'start':             bout_info['start'],
                'end':               bout_info['end'],
                'duration_s':        bout_info['duration_s'],
                'mean_speed':        bout_info['mean_speed'],
                'net_displacement_mm': bout_info['net_displacement_mm'],
                'per_leg_cycles':    str(bout_info.get('per_leg_cycles', {})),
            })

    qc_df = pd.DataFrame(summary_rows)
    out_csv = CFG["output_dir"] / "qc_session_summary.csv"
    qc_df.to_csv(out_csv, index=False)
    print(f"\nQC complete. {len(qc_df)} bouts ({qc_df['status'].value_counts().to_dict()})")
    print(f"PDFs in:  {qc_dir}")
    print(f"Summary:  {out_csv}")
    return qc_df


print("run_qc_all_sessions() defined.")
```

### Cell 2d — Run QC

```python
# ── Run once before batch processing ─────────────────────────────────────────
# show_plots=False writes PDFs silently; set True to pop up each figure
qc_summary = run_qc_all_sessions(meta, show_plots=False)

# ── Quick audit printout ──────────────────────────────────────────────────────
print("\n=== Valid bouts per session ===")
v_counts = (qc_summary[qc_summary['status'] == 'valid']
            .groupby(['fly_id', 'day']).size().unstack(fill_value=0))
print(v_counts.to_string())

print("\n=== Rejection reasons ===")
rej = qc_summary[qc_summary['status'] == 'rejected']
if len(rej):
    print(rej['rejection_reason'].value_counts().to_string())
else:
    print("(none)")
```

**What to look for in the QC PDFs:**
- **Speed panel**: bout boundaries align with speed transitions; boundary
  annotations (`immobility`, `tracking failure`, `arena wall`, `recording edge`)
  explain why the bout ended where it did.
- **Head Z**: falls appear as brief, prominent downward dips — not flat-line
  artefacts or slow drifts.
- **Leg Z**: swing events (unshaded) are brief and rhythmic; stance periods
  (shaded) alternate across legs. No leg shows constant swing (tracking loss).
- **XY trajectory**: viridis gradient shows directionality; pre/post context
  (gray dashed/dotted) shows where the fly was coming from and going to.
- **Rejected bouts**: check whether rejections are reasonable (genuinely short
  or stationary segments) vs systematic (poor threshold choice).

*If swing/stance looks wrong: adjust `swing_z_vel_threshold_mm_s` in CFG and
re-run Cell 1b + 2d. Try 0.5–3.0 mm/s. If too many bouts are rejected for
displacement, lower `min_body_displacement_mm`.*

---

## Section 3 — Batch Processing → `df_long`

### Cell 3 — Loop, Compute Metrics, Build DataFrame

```python
records     = []   # one entry per (fly, day, bout)
phase_store = {}   # (fly_id, day) → {pair_label: [arrays per bout]}

n_sess = len(meta)
for ridx, row in meta.iterrows():
    fly_id = row['fly_id']
    day    = int(row['day'])
    sex    = row['sex']
    path   = row['path']
    print(f"\n[{ridx+1}/{n_sess}] fly={fly_id}  day={day}  sex={sex}")

    # ── Load ───────────────────────────────────────────────────────────────
    df_s, kp_s = load_3d_data(path)
    df_s       = mask_low_confidence(df_s, kp_s)
    print(f"   {len(df_s):,} frames")

    # ── Session-level arrays (computed once per session) ───────────────────
    R_mat, origin = compute_body_frame(df_s)
    head_z_full   = compute_head_z(df_s)
    speed_arr, sx, sy, scut_z_arr = compute_scutellum_speed(df_s)

    # ── Detect walking bouts ───────────────────────────────────────────────
    bouts, rejected, _, _ = detect_walking_bouts(df_s, kp_s)
    print(f"   {len(bouts)} valid bouts  ({len(rejected)} rejected)")
    if not bouts:
        continue

    # ── Per-bout metrics ───────────────────────────────────────────────────
    sess_key = (fly_id, day)
    phase_store[sess_key] = {}

    for bidx, bout in enumerate(bouts):
        try:
            m = compute_bout_metrics(df_s, bout, R_mat, origin,
                                     head_z_full, speed_arr, scut_z_arr)
        except Exception as ex:
            print(f"   ! Bout {bidx} failed: {ex}")
            continue

        # Accumulate phase arrays for Section 5
        for label, (phases_arr, _) in m['_phase_offsets'].items():
            phase_store[sess_key].setdefault(label, []).append(phases_arr)

        # Flat record (exclude _ arrays)
        rec = {'fly_id': fly_id, 'day': day, 'sex': sex,
               'bout_idx': bidx, 'path': str(path)}
        rec.update({k: v for k, v in m.items() if not k.startswith('_')})
        records.append(rec)

# ── Concatenate phase arrays ───────────────────────────────────────────────
for key in phase_store:
    for label in phase_store[key]:
        phase_store[key][label] = np.concatenate(phase_store[key][label])

# ── Build df_long ──────────────────────────────────────────────────────────
df_long = pd.DataFrame(records)
print(f"\n{'='*55}")
print(f"df_long: {len(df_long)} rows — "
      f"{df_long['fly_id'].nunique()} flies, {df_long['day'].nunique()} days")
print(df_long.groupby(['fly_id', 'day'])['speed_mm_s'].mean().unstack())

df_long.to_csv(CFG["output_dir"] / "df_long.csv", index=False)
print(f"\nSaved: {CFG['output_dir'] / 'df_long.csv'}")
```

---

## Section 3b — Dataset Consistency Report

**Run immediately after Section 3 before interpreting any longitudinal trend.**
The goal is to flag whether observed day-to-day differences could be explained by
a single outlier fly, a recording session with unusual properties, or systematic
over/under-sampling of certain sessions. Each plot offers a different diagnostic lens.

### Cell 3b-1 — Coverage Heatmap (fly × day)

```python
import seaborn as sns

# ── Valid bout count per session ──────────────────────────────────────────────
pivot_n = (df_long.groupby(['fly_id', 'day']).size()
           .unstack(fill_value=0))

# ── Total walking time per session (minutes) ──────────────────────────────────
pivot_t = (df_long.groupby(['fly_id', 'day'])['duration_s']
           .sum().unstack(fill_value=0) / 60.0)

# Fly order: females first (consistent with downstream palette)
_fly_order = (
    [f for f, s in fly_sex.items() if s == 'f'] +
    [f for f, s in fly_sex.items() if s == 'm']
)
_day_order = sorted(df_long['day'].unique())

fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.8 * len(_day_order)),
                                         max(4, 0.8 * len(_fly_order))))

for ax, (pivot, title, fmt) in zip(
    axes,
    [(pivot_n, 'N valid bouts', 'd'),
     (pivot_t, 'Walking time (min)', '.1f')]
):
    data = pivot.reindex(index=_fly_order, columns=_day_order, fill_value=0)
    im = ax.imshow(data.values, aspect='auto',
                   cmap='YlGn', vmin=0, vmax=data.values.max() or 1)
    ax.set_xticks(range(len(_day_order)));  ax.set_xticklabels(_day_order, fontsize=9)
    ax.set_yticks(range(len(_fly_order)));  ax.set_yticklabels(_fly_order, fontsize=9)
    ax.set_xlabel('Day', fontsize=9);  ax.set_ylabel('Fly', fontsize=9)
    ax.set_title(title, fontweight='bold')
    for i, fly in enumerate(_fly_order):
        for j, day in enumerate(_day_order):
            val = data.loc[fly, day] if fly in data.index and day in data.columns else 0
            ax.text(j, i, format(val, fmt), ha='center', va='center',
                    fontsize=7.5,
                    color='white' if val > data.values.max() * 0.6 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

plt.suptitle('Dataset coverage — valid walking bouts per session',
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(CFG["output_dir"] / 'coverage_heatmap.pdf', bbox_inches='tight')
plt.show()
print("Red flag: blank cells = missing session (expected for 3 deceased flies).")
print("Red flag: one cell with 1 bout while neighbours have 10+ = poor detection.")
```

### Cell 3b-2 — Per-Metric Distributions by Fly (Overlaid Histograms)

```python
# Six key metrics for consistency check
DIST_METRICS = [
    ('speed_mm_s',          'Speed (mm/s)',            0,   None),
    ('height_over_legs_mm', 'Height over legs (mm)',   0,   None),
    ('leg_spread_mm2',      'Leg spread (mm²)',         0,   None),
    ('body_pitch_deg',      'Body pitch (°)',           None, None),
    ('T2L_forward_ego_mm',  'T2L forward ego (mm)',    None, None),
    ('fall_rate_per_s',     'Fall rate (falls/s)',      0,   None),
]

n_metrics = len(DIST_METRICS)
ncols = 3; nrows = int(np.ceil(n_metrics / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows),
                          gridspec_kw={'hspace': 0.45, 'wspace': 0.35})
axes_flat = axes.flatten()

for (metric, label, xmin, xmax), ax in zip(DIST_METRICS, axes_flat):
    if metric not in df_long.columns:
        ax.set_visible(False); continue

    for fly in _fly_order:
        s_data = df_long.loc[df_long['fly_id'] == fly, metric].dropna()
        if len(s_data) < 3: continue
        # Clip at 99th percentile to suppress extreme artefacts
        clip = s_data.quantile(0.99)
        s_data = s_data[s_data <= clip]
        col = FLY_COLOR.get(fly, '#888888')
        ax.hist(s_data, bins=40, density=True, alpha=0.40, color=col,
                label=fly, histtype='stepfilled', edgecolor='none')

    ax.set_xlabel(label, fontsize=8.5)
    ax.set_ylabel('Density', fontsize=8)
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7)

for ax in axes_flat[n_metrics:]:
    ax.set_visible(False)

plt.suptitle('Per-fly metric distributions (frame-level, across all bouts + days)',
             fontsize=11, fontweight='bold')
plt.savefig(CFG["output_dir"] / 'consistency_histograms.pdf', bbox_inches='tight')
plt.show()
print("Red flag: one fly's distribution shifted far from the others "
      "→ possible calibration or arena difference.")
```

### Cell 3b-3 — Strip Plots: Metric vs Day, Fly-Coloured

```python
# Per-bout means (one dot per bout; consistent with df_long granularity)
STRIP_METRICS = [
    ('speed_mm_s',          'Speed (mm/s)'),
    ('height_over_legs_mm', 'Height over legs (mm)'),
    ('body_pitch_deg',      'Body pitch (°)'),
    ('T2L_forward_ego_mm',  'T2L forward ego (mm)'),
    ('fall_rate_per_s',     'Fall rate (falls/s)'),
    ('mean_n_legs_stance',  'Mean N legs in stance'),
]

n_strip = len(STRIP_METRICS)
ncols = 3; nrows = int(np.ceil(n_strip / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                          gridspec_kw={'hspace': 0.5, 'wspace': 0.35})
axes_flat = axes.flatten()

for (metric, label), ax in zip(STRIP_METRICS, axes_flat):
    if metric not in df_long.columns:
        ax.set_visible(False); continue

    # Overlay per-fly strips with consistent colour
    strip_df = df_long[['fly_id', 'day', metric]].dropna()
    for fly in _fly_order:
        sub = strip_df[strip_df['fly_id'] == fly]
        if sub.empty: continue
        ax.scatter(sub['day'] + np.random.uniform(-0.25, 0.25, len(sub)),
                   sub[metric],
                   color=FLY_COLOR.get(fly, '#888888'),
                   s=9, alpha=0.55, linewidths=0, label=fly, zorder=2)

    # Population mean ± SEM per day (black line on top)
    pop = strip_df.groupby('day')[metric].agg(['mean', 'sem']).reset_index()
    ax.errorbar(pop['day'], pop['mean'], yerr=pop['sem'],
                color='k', lw=1.8, marker='o', markersize=5, zorder=4,
                capsize=3, label='mean±SEM')

    ax.set_xlabel('Day', fontsize=8.5);  ax.set_ylabel(label, fontsize=8.5)
    ax.set_xticks(_day_order)
    ax.legend(fontsize=6, ncol=2, framealpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7)

for ax in axes_flat[n_strip:]:
    ax.set_visible(False)

plt.suptitle('Bout-level metric vs day — each dot = one bout, coloured by fly',
             fontsize=11, fontweight='bold')
plt.savefig(CFG["output_dir"] / 'consistency_stripplots.pdf', bbox_inches='tight')
plt.show()
print("Red flag: one fly's dots separated from all others on every day "
      "→ the fly may have a calibration offset or unusual baseline behaviour.")
print("Red flag: day effect present in strip plot but absent in longitudinal "
      "→ caused by fly dropout (dead flies were different from survivors).")
```

### Cell 3b-4 — Session Quality Summary

```python
# Bout duration distribution, N bouts per session, total walking time
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ── Panel A: Bout duration histogram, coloured by fly ─────────────────────────
ax = axes[0]
for fly in _fly_order:
    d = df_long.loc[df_long['fly_id'] == fly, 'duration_s'].dropna()
    if len(d) < 2: continue
    ax.hist(d, bins=30, alpha=0.45, color=FLY_COLOR.get(fly, '#888888'),
            label=fly, density=True, histtype='stepfilled', edgecolor='none')
ax.set_xlabel('Bout duration (s)', fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title('A  Bout duration distribution', fontweight='bold', loc='left')
ax.legend(fontsize=7, ncol=2, framealpha=0.5)
ax.spines[['top', 'right']].set_visible(False)

# ── Panel B: N bouts per session (strip plot ordered by day) ──────────────────
ax = axes[1]
n_bouts_per_sess = df_long.groupby(['fly_id', 'day']).size().reset_index(name='n_bouts')
for fly in _fly_order:
    sub = n_bouts_per_sess[n_bouts_per_sess['fly_id'] == fly]
    if sub.empty: continue
    ax.scatter(sub['day'] + np.random.uniform(-0.2, 0.2, len(sub)),
               sub['n_bouts'],
               color=FLY_COLOR.get(fly, '#888888'),
               s=30, alpha=0.7, label=fly, zorder=2)
ax.set_xlabel('Day', fontsize=9);  ax.set_ylabel('N valid bouts', fontsize=9)
ax.set_title('B  Bouts per session', fontweight='bold', loc='left')
ax.set_xticks(_day_order)
ax.spines[['top', 'right']].set_visible(False)

# ── Panel C: Total walking time per session ───────────────────────────────────
ax = axes[2]
walk_time = (df_long.groupby(['fly_id', 'day'])['duration_s']
             .sum().reset_index(name='total_s'))
walk_time['total_min'] = walk_time['total_s'] / 60.0
for fly in _fly_order:
    sub = walk_time[walk_time['fly_id'] == fly]
    if sub.empty: continue
    ax.plot(sub['day'], sub['total_min'],
            color=FLY_COLOR.get(fly, '#888888'), marker='o',
            lw=1.2, markersize=5, alpha=0.75, label=fly)
ax.set_xlabel('Day', fontsize=9);  ax.set_ylabel('Total walking time (min)', fontsize=9)
ax.set_title('C  Total walking time / session', fontweight='bold', loc='left')
ax.set_xticks(_day_order)
ax.legend(fontsize=7, ncol=2, framealpha=0.5)
ax.spines[['top', 'right']].set_visible(False)

plt.suptitle('Session quality metrics', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(CFG["output_dir"] / 'session_quality.pdf', bbox_inches='tight')
plt.show()

# ── Text summary ──────────────────────────────────────────────────────────────
print("\n=== Session quality summary ===")
print(f"Median bout duration:  {df_long['duration_s'].median():.2f} s")
print(f"Mean bouts / session:  {n_bouts_per_sess['n_bouts'].mean():.1f}")
print(f"Total walking time:    {df_long['duration_s'].sum()/60:.1f} min across all sessions")
```

---

## Section 4 — Longitudinal Analysis

### Cell 4a — Plotting Helper

```python
# ── Colour palette: warm = female, cool = male ────────────────────────────
_PAL_F = ['#C0392B', '#E67E22', '#F39C12', '#8E44AD']
_PAL_M = ['#1A5276', '#117A65', '#1F618D', '#145A32', '#1B4F72']

fly_sex = meta.drop_duplicates('fly_id').set_index('fly_id')['sex'].to_dict()
_f_flies = [f for f, s in fly_sex.items() if s == 'f']
_m_flies = [f for f, s in fly_sex.items() if s == 'm']
FLY_COLOR = {f: _PAL_F[i % len(_PAL_F)] for i, f in enumerate(_f_flies)}
FLY_COLOR.update({f: _PAL_M[i % len(_PAL_M)] for i, f in enumerate(_m_flies)})

SEX_COLOR  = {'f': '#C0392B', 'm': '#1A5276'}
SEX_LABEL  = {'f': 'Female',  'm': 'Male'}

DAYS_ALL   = sorted(df_long['day'].unique())


def _daily_means(metric):
    """Per-fly per-day mean of metric across bouts."""
    return (
        df_long.dropna(subset=[metric])
        .groupby(['fly_id', 'day', 'sex'])[metric]
        .mean().reset_index()
    )


def plot_longitudinal(metric, ylabel, ax=None, ymin=None, ymax=None, hline=None):
    """Per-fly thin lines + population mean±SEM by sex, metric vs day."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    fd = _daily_means(metric)

    for sex in ['f', 'm']:
        col = SEX_COLOR[sex]
        sub = fd[fd['sex'] == sex]
        if sub.empty: continue

        # Thin per-fly lines
        for fid, grp in sub.groupby('fly_id'):
            grp_s = grp.sort_values('day')
            ax.plot(grp_s['day'], grp_s[metric],
                    color=FLY_COLOR.get(fid, col), alpha=0.35, lw=1.2,
                    marker='o', markersize=3)

        # Population mean ± SEM
        pop = sub.groupby('day')[metric].agg(['mean', 'sem']).reset_index()
        ax.plot(pop['day'], pop['mean'], color=col, lw=2.5,
                marker='o', markersize=6, label=SEX_LABEL[sex])
        ax.fill_between(pop['day'], pop['mean'] - pop['sem'],
                        pop['mean'] + pop['sem'], color=col, alpha=0.18)

    if hline is not None:
        ax.axhline(hline, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Day post-amputation', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(DAYS_ALL)
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    return ax


print("Longitudinal plotting helper defined.")
```

### Cell 4b — Falls and Speed

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_longitudinal('fall_rate_per_s', 'Fall rate (falls/s)', ax=axes[0], ymin=0)
axes[0].set_title('A  Fall rate', fontweight='bold', loc='left')

plot_longitudinal('speed_mm_s', 'Walking speed (mm/s)', ax=axes[1], ymin=0)
axes[1].set_title('B  Walking speed', fontweight='bold', loc='left')

plt.suptitle('T1L amputation — falls and speed across days',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(CFG["output_dir"] / 'longitudinal_falls_speed.pdf', bbox_inches='tight')
plt.show()
```

### Cell 4c — Body Posture

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_longitudinal('body_pitch_deg',
                  'Body pitch (°)\n+ = head up', ax=axes[0], hline=0)
axes[0].set_title('C  Body pitch', fontweight='bold', loc='left')

plot_longitudinal('height_over_legs_mm',
                  'Body height over legs (mm)', ax=axes[1], ymin=0)
axes[1].set_title('D  Height above stance legs', fontweight='bold', loc='left')

plot_longitudinal('leg_spread_mm2',
                  'Leg spread — hull area (mm²)', ax=axes[2], ymin=0)
axes[2].set_title('E  Leg spread', fontweight='bold', loc='left')

plt.suptitle('Body posture adaptation across days',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(CFG["output_dir"] / 'longitudinal_posture.pdf', bbox_inches='tight')
plt.show()
```

### Cell 4d — T2L Foot Placement

```python
fig, ax = plt.subplots(figsize=(8, 5))
plot_longitudinal('T2L_forward_ego_mm',
                  'T2L tip egocentric X (mm)\n+ = anterior / forward', ax=ax, hline=0)
ax.set_title('F  T2L foot placement (forward shift)',
             fontweight='bold', loc='left')
plt.tight_layout()
plt.savefig(CFG["output_dir"] / 'longitudinal_T2L_placement.pdf', bbox_inches='tight')
plt.show()
```

### Cell 4e — Phase Coordination R Values

```python
# Show coordination strength for key pairs across days
HIGHLIGHT = ['T1R→T2L', 'T1R→T2R', 'T2L→T2R', 'T2L→T3L', 'T2R→T3R']
pair_cols = [f'R_{p}' for p in HIGHLIGHT if f'R_{p}' in df_long.columns]

if pair_cols:
    fig, axes = plt.subplots(1, len(pair_cols),
                             figsize=(5 * len(pair_cols), 5), sharey=True)
    if len(pair_cols) == 1: axes = [axes]

    for ax, col in zip(axes, pair_cols):
        label = col.replace('R_', '')
        plot_longitudinal(col, f'R ({label})', ax=ax, ymin=0, ymax=1, hline=0.5)
        ax.set_title(label, fontweight='bold', loc='left', fontsize=9)

    plt.suptitle('Phase coordination strength across days',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CFG["output_dir"] / 'longitudinal_coordination_R.pdf',
                bbox_inches='tight')
    plt.show()
```

### Cell 4f — 8-Panel Summary Figure

```python
PANELS = [
    ('fall_rate_per_s',       'Fall rate (falls/s)',          'A', 0,    None),
    ('speed_mm_s',            'Speed (mm/s)',                  'B', 0,    None),
    ('body_pitch_deg',        'Body pitch (°)',                'C', None, None),
    ('height_over_legs_mm',   'Height over legs (mm)',         'D', 0,    None),
    ('leg_spread_mm2',        'Leg spread (mm²)',              'E', 0,    None),
    ('T2L_forward_ego_mm',    'T2L forward (mm)',              'F', None, None),
    ('mean_n_legs_stance',    'Mean N legs stance',            'G', 0,    5),
]
if 'R_T1R→T2L' in df_long.columns:
    PANELS.append(('R_T1R→T2L', 'R T1R→T2L', 'H', 0, 1))

n_panels = len(PANELS)
ncols = 4; nrows = int(np.ceil(n_panels / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                          gridspec_kw={'hspace': 0.5, 'wspace': 0.35})
axes_flat = axes.flatten()

for (metric, ylabel, label, ymin, ymax), ax in zip(PANELS, axes_flat):
    plot_longitudinal(metric, ylabel, ax=ax, ymin=ymin, ymax=ymax)
    ax.set_title(f'{label}  {ylabel}', fontsize=9, fontweight='bold', loc='left')
    ax.tick_params(labelsize=7)

for ax in axes_flat[n_panels:]:
    ax.set_visible(False)

plt.suptitle('T1L amputation — longitudinal adaptation summary',
             fontsize=13, fontweight='bold')
plt.savefig(CFG["output_dir"] / 'longitudinal_summary.pdf', bbox_inches='tight')
plt.show()
```

---

## Section 5 — Phase Coordination Detail

### Cell 5a — Polar Histograms by Day (Key Pairs)

Requires `phase_store` from Cell 3.

```python
POLAR_PAIRS = ['T1R→T2L', 'T1R→T2R', 'T2L→T3L']  # adjust as needed
n_days = len(DAYS_ALL)
n_bins = 24
bin_edges   = np.linspace(0, 2 * np.pi, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

for pair in POLAR_PAIRS:
    fig, axes = plt.subplots(1, n_days, figsize=(3.5 * n_days, 3.5),
                              subplot_kw={'projection': 'polar'})
    if n_days == 1: axes = [axes]

    rmax = 0.0
    hists = {}
    for day in DAYS_ALL:
        all_phases = []
        for fly_id in df_long['fly_id'].unique():
            key = (fly_id, day)
            if key in phase_store and pair in phase_store[key]:
                all_phases.append(phase_store[key][pair])
        if all_phases:
            ph = np.concatenate(all_phases)
            counts, _ = np.histogram(ph, bins=bin_edges)
            counts_n  = counts / counts.sum()
            hists[day] = (counts_n, ph)
            rmax = max(rmax, counts_n.max())

    for ax, day in zip(axes, DAYS_ALL):
        if day not in hists:
            ax.set_title(f'Day {day}\n(no data)', fontsize=8)
            continue
        counts_n, ph = hists[day]
        R = float(np.abs(np.mean(np.exp(1j * ph))))
        bars = ax.bar(bin_centers, counts_n,
                      width=2 * np.pi / n_bins, align='center',
                      color='steelblue', alpha=0.75, edgecolor='white', lw=0.4)
        ax.set_ylim(0, rmax * 1.15)
        ax.set_title(f'Day {day}\nR={R:.2f}', fontsize=8)
        ax.set_xticklabels([]); ax.set_yticklabels([])

    fig.suptitle(f'Phase coordination: {pair}\n(each bar = fraction of onsets)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CFG["output_dir"] / f'polar_day_{pair.replace("→","_")}.pdf',
                bbox_inches='tight')
    plt.show()
```

### Cell 5b — R-Matrix Heatmap: Day 0 vs Day 7

```python
LEG_ORDER  = ['T1R', 'T2L', 'T2R', 'T3L', 'T3R']

def _build_R_matrix(target_day):
    """5×5 mean R matrix across all flies for one day."""
    mat = np.full((5, 5), np.nan)
    for i, la in enumerate(LEG_ORDER):
        for j, lb in enumerate(LEG_ORDER):
            if i == j: continue
            label = f"{la}→{lb}"
            col   = f"R_{label}"
            if col not in df_long.columns: continue
            sub = df_long[df_long['day'] == target_day][col].dropna()
            if len(sub): mat[i, j] = sub.mean()
    return mat

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, day in zip(axes, [0, 7]):
    mat = _build_R_matrix(day)
    if np.all(np.isnan(mat)):
        ax.set_title(f'Day {day} — no data'); continue
    im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(LEG_ORDER, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(LEG_ORDER, fontsize=9)
    ax.set_xlabel('Reference (B)', fontsize=9)
    ax.set_ylabel('Test (A)', fontsize=9)
    ax.set_title(f'Day {day} post-amputation\nR matrix (coordination strength)',
                 fontweight='bold')
    for ii in range(5):
        for jj in range(5):
            if not np.isnan(mat[ii, jj]):
                ax.text(jj, ii, f"{mat[ii, jj]:.2f}", ha='center', va='center',
                        fontsize=7, color='white' if mat[ii, jj] < 0.6 else 'black')

plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='R')
plt.suptitle('Phase coordination matrix: day 0 vs day 7',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(CFG["output_dir"] / 'R_matrix_day0_vs_day7.pdf', bbox_inches='tight')
plt.show()
```

---

## Execution Order

```
Cell 0a  imports
Cell 0b  configuration  ← edit metadata_csv path, arena dims, and parameters here
Cell 0c  load metadata  ← verify fly × sex × day count

Cell 1a  data loading helpers
Cell 1b  walking bout detection  ← returns valid + rejected bouts
Cell 1c  egocentric frame
Cell 1d  swing/stance detection
Cell 1e  metric functions
│
├── SECTION 2: QC — ALL bouts, ALL flies, ALL days  ← INSPECT BEFORE BATCH
│   Cell 2a  colour palette + _boundary_reason()
│   Cell 2b  plot_bout_qc()        ← multi-panel figure per bout
│   Cell 2c  run_qc_all_sessions() ← batch PDFs + summary CSV
│   Cell 2d  run QC + audit valid/rejected counts
│            ↳ PDFs in output/amputation_longitudinal/qc/
│            ↳ Adjust CFG thresholds if needed, then re-run Cell 1b + 2d
│
└── SECTION 3: Batch (run after QC confirms data quality)
    Cell 3   loop + build df_long + phase_store
    │
    ├── SECTION 3b: Dataset consistency report  ← spot outliers before fitting
    │   Cell 3b-1  coverage heatmap (fly × day)
    │   Cell 3b-2  per-metric distributions by fly (overlaid histograms)
    │   Cell 3b-3  strip plots: metric vs day, fly-coloured
    │   Cell 3b-4  session quality (bout N, duration, walking time)
    │
    ├── SECTION 4: Longitudinal plots
    │   Cell 4a  plotting helper
    │   Cell 4b  falls + speed
    │   Cell 4c  posture (pitch, height, spread)
    │   Cell 4d  T2L foot placement
    │   Cell 4e  coordination R values
    │   Cell 4f  8-panel summary figure
    │
    └── SECTION 5: Phase detail
        Cell 5a  polar histograms by day
        Cell 5b  R-matrix day 0 vs day 7
```

---

## Key Variables

| Variable | Created in | Description |
|---|---|---|
| `meta` | Cell 0c | Metadata DataFrame: fly_id, day, sex, path |
| `df_long` | Cell 3 | One row per (fly, day, bout); all scalar metrics |
| `phase_store` | Cell 3 | `{(fly_id, day): {pair_label: phases_array}}` |
| `FLY_COLOR` | Cell 4a | `{fly_id: hex_color}` — consistent across all plots |
| `DAYS_ALL` | Cell 4a | Sorted list of day integers in the dataset |

---

## Parameter Tuning Guide

| Parameter | Default | Too low | Too high |
|---|---|---|---|
| `swing_z_vel_threshold_mm_s` | 1.5 | Stance frames classified as swing; noisy gait diagram | Many real swings missed; all legs appear in stance |
| `swing_z_vel_sigma` | 3 | Noisy dZ/dt; spurious swing detections | Swing peaks blurred; onset timing offset |
| `fall_prominence_mm` | 1.5 | Noise spikes detected as falls; rate inflated | Real falls missed; rate underestimated |
| `min_bout_duration_frames` | 400 (0.5 s) | Short artefact bouts included | Long recovery strides not captured |
| `conf_threshold` | 0.8 | Poor tracking frames pass into analysis | Too many frames masked → short bouts lost |

**Primary diagnostic**: Cell 2c (QC visualization). A correct swing/stance detector shows:
- Clear alternating stance (shaded) and swing (clear) for each leg
- Swing durations ~30–80 ms (24–64 frames at 800 Hz)
- T1R and T2L swinging roughly alternately with T2R/T3L (post-recovery)
- Head Z baseline stable; falls are brief, prominent dips

---

## Notes on Signal Interpretation

**Body pitch** (`body_pitch_deg`): After T1L amputation, the fly loses anterior
support on the left. Expect slightly negative pitch (head-down tilt) on day 0 when
the fly overreaches with T1R. Recovery may show pitch returning to near zero.

**Height over legs** (`height_over_legs_mm`): Lower values = fly body closer to
ground (crouching). May increase as fly learns to distribute weight more evenly
across the remaining 5 legs.

**T2L forward placement** (`T2L_forward_ego_mm`): Positive = T2L tip anterior to
Scutellum origin. An increase from day 0 to day 7 means the fly is placing the
ipsilateral middle leg further forward — compensating for the missing T1L support.
Compute this during stance only (foot on ground = mechanical load-bearing).

**Phase coordination** (`R_T1R→T2L`): In intact tripod gait R ≈ 0.8–0.9 (T1R and
T2L are synchronized in the right tripod). After amputation, if the fly shifts to
wave-like coordination, this R should decrease (T1R and T2L become less coupled).
Watch for `R_T2L→T3L` increasing if a T1R-T2L-T3L wave on the left emerges.
