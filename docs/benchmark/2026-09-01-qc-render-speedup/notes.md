# QC + overlay render speedup (behaviour-preserving)

Bout: `Session0/2025_10_20_13_20_04` `bout_00028/fly0` — 2007 frames, 7 cameras,
1936x448 frames, `mesh_subset: fps_500+wing` (600 verts), 50 model keypoints.
Everything below is wall clock on one GPU node (CPU-only work; the node was
already loaded, ~130 load average, during the render measurements).

## Where the time went (measured, cProfile — not guessed)

Per-bout wall clock before, from artifact mtimes:

| stage | before |
|---|---|
| triangulation | 5 s |
| STAC IK (not a target) | 12 m 26 s |
| bridge + FK | 2 m 34 s |
| **qc.json** | **3 m 39 s** |
| **qc_perframe.npz** | **2 m 52 s** |
| **overlays (7 videos)** | **11 m 25 s** |
| sidebyside.mp4 | 1 m 55 s |

QC profile (60 frames, `cProfile`, cumulative):
`mesh_mask_iou_report` 70% of `qc_report`, of which `rt.reproject_point`
(335 996 scalar calls in 60 frames) 4.8 s and `soft_iou_of_verts` 3.1 s;
`loo_reproj` 13%; `ik_reproj_report` 11%; `per_camera_reproj_error` 6%.
`qc_perframe` recomputed the same two metrics from scratch.

Overlay profile (per 1936x448 frame, real camera):
matplotlib Agg draw 34 ms (26 ms of it `imshow`'s resample), h264 decode 18 ms,
libx264 encode 10 ms, projection 0.05 ms. **Nothing in this path uses a GPU.**

## What changed

* `geometry/reprojection_tool.py`: added `reproject_points` (batched, `np.einsum`
  — BLAS `@`/`tensordot` reassociate the 4-term dot and drift 2.3e-13 px) and
  `reconstruct_points` (stacked SVD). Both **bit-identical** to the scalar loops.
* `tracking/qc.py`: every metric vectorised over keypoints/vertices/cameras via
  those primitives, with a scalar fallback for `rt` objects that only implement
  the old API. New `frame_metrics()` holds the per-(frame, camera) IoU +
  reprojection arrays that `qc_report` and `per_frame_qc` both need.
* `tracking/mesh_iou.py`: hard IoU counts on the touched pixels instead of
  materialising an (H, W) `pred` mask; soft IoU splats/exps/sums over the
  splat's bounding box (~50 k px) instead of the full frame (867 k px).
* `tracking/qc_perframe.py`, `scripts/run_bout.py` Stage E: compute
  `frame_metrics` once, pass it to both artifacts.
* `tracking/reproj_video.py`: one reused Agg figure per camera
  (`OverlayRenderer`), a decode-prefetch thread, and `render_camera_overlays()`
  which renders cameras concurrently as isolated
  `python -m jarvis_jax.tracking.reproj_video <job.npz>` subprocesses.
  `cfg.outputs.overlay_workers` (default 4) caps concurrency.

## After

| stage | before | after | speedup |
|---|---|---|---|
| qc.json (standalone) | 269.5 s | 15.5 s | 17.4x |
| qc_perframe.npz (standalone) | 181.1 s | 11.3 s | 16.0x |
| both, sharing `frame_metrics` (what Stage E does) | 450.7 s | 15.4 s | **29.3x** |
| overlays, 1 camera | 97.9 s | 72 s | 1.36x |
| overlays, 7 cameras | 685 s | 195 s (7 workers) / 201 s (4, the default) | **3.4x** |
| **the three target stages together** | **1076 s** (mtimes: 219+172+685) | **215 s** | **5.0x** |

(The `before` figures for QC are a rerun of the pre-change code on identical
inputs in the same process, which is why they differ slightly from the mtime
column above.)

## Numbers unchanged

`qc.json`, new code vs the **artifact already on disk** (written by the old
code): **280 / 280 numeric leaves identical, max abs diff 0.0**. The only key
difference is `silhouette_iou` -> `mesh_mask_iou`, the pre-existing rename in
commit 0bc36fe.

`qc.json`, new vs a rerun of the old code in-process: **283 / 283 leaves
identical, max abs and max rel diff 0.0** (`qc_equivalence.json`).

`qc_perframe.npz` vs on disk: `hard_iou`, `reproj_px`, `n_cams` bit-identical;
`soft_iou` max abs 1.11e-16, max rel 5.8e-16 (1-2 ulp). That single deviation is
the only non-exact quantity in the change and it is understood: the bounding-box
soft IoU rearranges two reductions,
`sum(soft + mask - soft*mask)` -> `|mask| + sum(soft*(1-mask))`; the splat grid
itself is bit-identical.

Overlay videos vs the on-disk mp4s the old code wrote: **all 7 md5-identical
and byte-for-byte the same size**, same frame count (2007), decoded-pixel
max |old - new| = 0 on sampled frames, and the 4-worker output identical to the
7-worker output (`render_equivalence.json`). Visual check in
`figures/2026-09-01-qc-render-speedup/overlay_old_vs_new_frame1000.png`: fly0
(female) at frame 1000 in Cam2012630 and Cam2012855, old beside new — cyan mesh
+ magenta keypoints on the female, the male beside her un-overlaid, panels
indistinguishable.

### Caveat on the render speedup
3.4x, not the 5-10x the QC stages reached. The overlay is CPU-bound in
matplotlib and the measurement ran on a node whose cores were already consumed
by an unrelated 4756%-CPU MATLAB job (load average 130+ on 32 cores), so
7 cameras x 72 s = 504 s of work finished in 195 s, i.e. only ~2.6x effective
parallelism. On an idle node the floor is one camera's 72 s (9.5x), and 4 vs 7
workers made no difference here precisely because no cores were free. Beyond
that, the only remaining lever is leaving matplotlib or moving the h264 to
NVENC, and both change output pixels/bytes.

## Would JAX / the GPU help what is left?

Profile of the optimized QC (200 frames, `tottime`), i.e. what the remaining
~15 s is made of:

| | share |
|---|---|
| `np.ufunc.reduce` — mostly the full-frame `mask.sum()` per (frame, camera), memory-bandwidth-bound over the 12.2 GB mask array | 19% |
| `_soft_splat_bbox` + `np.add.at` — the Gaussian splat's scatter-add | 30% |
| `np.linalg.svd` — ~700 k stacked (12,4) LOO DLT solves in LAPACK | 16% |
| `_soft_iou_from_bbox`, `np.unique` | 11% |
| **`np.einsum` — all the reprojection math** | **2%** |

The reprojection, the one part a GPU is good at, is now 2% of the cost; the
bottleneck was never FLOPs, it was ~7 M Python-level calls. The rest is scatter-
add, a bandwidth-bound reduction, and small-matrix LAPACK — none of which a GPU
wins by much at this size. And porting them would break the acceptance
criterion: JAX is float32 unless `jax_enable_x64`, and even in float64 its
reductions/einsum lower to different kernels with a different summation order,
so the numbers WOULD move. The pipeline's remaining JAX-worthy stage is the STAC
IK (12 m 26 s), which is already JAX.

## Deliberately NOT changed

* **STAC IK** (12 m 26 s) — out of scope.
* **`interpolation='nearest'`** on the overlay's `imshow` would cut 17 of 34 ms,
  and **h264_nvenc** (available on this node) would move the encode to the GPU.
  Both change output pixels/bytes, so neither was taken.
* **`-threads` on libx264** would stop 7 concurrent encoders from spawning
  ~48 threads each, but a different thread count changes the h264 bitstream.
* **sidebyside.mp4** (1 m 55 s) — already frame-capped at
  `outputs.sidebyside_frames: 300` and dominated by MuJoCo/EGL rendering.
* **`tests/test_qc.py`'s two failures** (`silhouette_iou` key) and
  **two config tests** (missing bouts CSV) fail identically before and after.
* **12.2 GB resident mask array**: `load_bout_masks` materialises
  `(T, C, H, W)` bool masks and Stage E's peak RSS is ~27 GB. Not touched — it
  is a memory issue, not a speed one, and the QC path now reads each mask twice
  instead of four times.

## Reproduce

```
python -m pytest third_party/jarvis_jax/tests/test_qc_vectorized_equivalence.py \
                 third_party/jarvis_jax/tests/test_reprojection_tool_batched.py \
                 third_party/jarvis_jax/tests/test_reproj_video_fast.py
```
The equivalence tests carry the pre-change implementations verbatim and assert
`==` (not `allclose`) everywhere except soft IoU.
