# Despike cell for fly1 bout_065 (and siblings) — design

## Context

In `notebooks/Courtship_Song_Analysis.ipynb`, the two-fly render of bout 65 shows
a handful of rapid tracking-error jumps that leak through the upstream
preprocessing pipeline — roughly ≈ 500 ms, ≈ 700 ms, and ≈ 1300 ms — and they
appear in *both* keypoints (`kp_data`) and joint angles (`qpos`) for fly1. We
want a light, notebook-local fix that identifies these jumps and interpolates
across them so the render and any downstream per-bout song analysis see clean
data for this session, without re-running the stac / preprocessing pipeline and
without touching real wing motion during song.

Scope: a single new cell in the notebook, operating on a single fly / single
bout at a time, with per-run idempotency. Not a reusable `utils/` function and
not a global preprocessing change — those can be lifted later if the approach
works out.

## Signals and detector

Capture rate is `FS = 800 Hz` (already defined earlier in the notebook).

Two independent per-feature velocity detectors; the flagged set is the union.

### Keypoint velocity

- Reshape `kp_data` from `(T, N*3)` to `(T, N, 3)`.
- `v_kp[t, n] = ‖kp[t+1, n] − kp[t, n]‖` — one scalar speed per keypoint per
  frame.
- Per-keypoint robust scale: `sigma_n = MAD_n / 0.6745`, where
  `MAD_n = median(|v_kp[:, n] − median(v_kp[:, n])|)`.
- Flag frame `t` as kp-bad if `v_kp[t, n] > KP_K · sigma_n` for **any** `n`.
- `KP_K = 6.0` by default.

### qpos velocity

- `v_q[t, j] = |q[t+1, j] − q[t, j]|` — one scalar speed per joint per frame.
- Per-joint robust scale (same MAD formula).
- Flag frame `t` as q-bad if `v_q[t, j] > QPOS_K · sigma_j` for **any** `j`.
- `QPOS_K = 6.0` by default.

Per-feature scaling matters: wing-yaw joints and wing-tip keypoints genuinely
oscillate fast during song. Using a *per-joint* / *per-keypoint* MAD means each
feature is measured against its own typical motion — fast-but-consistent wing
oscillation stays well below its own `6·sigma` bar, while a tracking glitch on
(say) the thorax-mounted Scutellum marker stands out immediately.

### Run post-processing

- `bad = bad_kp | bad_q`.
- Dilate `bad` by `± DILATE` frames (default `DILATE = 1`) — tracking spikes
  typically smear onto the immediate neighbor via the velocity computation.
- Split `bad` into maximal runs of consecutive flagged frames. Keep runs of
  length ≤ `MAX_GAP` (default `4` frames ≈ 5 ms at 800 Hz). Longer runs are
  **not** interpolated — they are recorded in the report as "skipped long
  runs" so they remain visible and you can decide manually what to do with
  them.

## Interpolation

For each coordinate of each keypoint (3·N scalar channels) and each joint
(`nq` scalar channels) independently:

- Replace the values at the kept-flagged frames with NaN.
- PCHIP (monotone cubic) interpolant fit on the remaining finite values,
  evaluated at the flagged frames. This matches the interpolant already used
  in `utils/keypoint_filter.py` (`_nan_interp_1d` at line 41) — C¹-smooth and
  non-overshooting between anchors, which matters for quaternion-like qpos
  components.
- Edges: if a flagged run touches `t = 0` or `t = T − 1`, it cannot be PCHIP-
  interpolated, so it is reported as "skipped edge run" and left alone.

Result is re-flattened back to `(T, N*3)` for `kp_data`; `qpos` is already
`(T, nq)`.

## Idempotency

On first run, the cell stashes untouched copies:

```python
bd = fly_data[DESPIKE_FLY][DESPIKE_BOUT_KEY]
bd.setdefault('kp_data__raw', np.asarray(bd['kp_data']).copy())
bd.setdefault('qpos__raw',    np.asarray(bd['qpos']).copy())
```

Every run — including re-runs with different thresholds — starts from the
`__raw` copies, so tuning `KP_K` / `QPOS_K` / `MAX_GAP` does not compound.

The cell writes the cleaned arrays back into `bd['kp_data']` and `bd['qpos']`,
so the two-fly render cell and any later per-bout analysis in the same kernel
transparently consume despiked data.

## Exposed knobs (top of cell)

```python
DESPIKE_FLY      = 'fly1'
DESPIKE_BOUT_KEY = f'bout_{bout:03d}'
KP_K             = 6.0      # MAD multiplier for kp velocity
QPOS_K           = 6.0      # MAD multiplier for qpos velocity
MAX_GAP          = 4        # max consecutive flagged frames to interpolate
DILATE           = 1        # ± frames to widen each flagged run
```

## Report

Text summary:

```
[despike fly1 bout_065]  kp-bad: 37 frames, q-bad: 22 frames, union: 48 frames
  fixed runs (≤4 frames): 6    total frames fixed: 11
  skipped long runs (>4 frames): 2
    frames 412-420 (9 frames)
    frames 1041-1049 (9 frames)
  skipped edge runs: 0
```

Diagnostic figure (2 rows, shared time axis in ms):

- Row 1: max-over-keypoints `v_kp[t]` trace, with the kept-flagged frames
  overplotted as red dots and skipped long runs overplotted as orange spans.
- Row 2: max-over-joints `v_q[t]` trace, same overlay style.

This gives an at-a-glance confirmation that the detector flagged the three
events near 500 ms / 700 ms / 1300 ms and did not drag a red line across real
wing-song segments.

## Files touched

- `notebooks/Courtship_Song_Analysis.ipynb` — one new code cell inserted
  immediately after the cell `id=42d16128` (the `fly0_qpos = ... / fly1_qpos
  = ...` data-load cell) and before the two-fly render cell `id=05512411`.
  No edits to existing cells; no changes under `utils/`.

## Verification

1. Run the new cell with defaults. The printed report should list a small
   number of fixed runs (on the order of the three events you noticed) and
   at most a couple of skipped long runs.
2. The diagnostic figure should show red dots sitting on top of the obvious
   velocity spikes and **no** red dots on the smooth high-frequency wing
   oscillations.
3. Re-render bout 65 via the two-fly render cell. The visible hops in fly1
   near 500 ms / 700 ms / 1300 ms should disappear; the wing song should look
   unchanged.
4. Re-run the despike cell a second time with different `KP_K` (e.g. `4.0`
   and `8.0`) — the output should be consistent because each run starts from
   the stashed `__raw` copies.
5. Set `DESPIKE_FLY = 'fly0'` and confirm that fly0, which you said looked
   clean, flags very few or zero frames — a negative-control sanity check.
6. If you have a per-fly per-bout song-analysis pipeline downstream, run it
   on bout 65 before and after despike and confirm no legitimate pulse /
   sine events near the fixed frames get lost.
