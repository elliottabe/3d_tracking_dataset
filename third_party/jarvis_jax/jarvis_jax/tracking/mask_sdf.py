"""Cropped signed-distance fields from SAM masks, in ORIGINAL image pixels.

For each (frame, camera) mask: crop to the fly bbox + margin, resize to a fixed
grid, and compute a SIGNED distance field -- negative inside, positive outside --
so a differentiable containment residual can bilinearly sample it at projected
mesh vertices. Pure NumPy/SciPy/cv2; no JAX, no qpos.

Recovered from `silhouette_sdf.py` (deleted in 0bc36fe with the silhouette
polish) and re-verified rather than trusted: the sign convention checks out
(-25.3 inside / +27.8 outside on a test box), but its ORIGINAL-pixel scaling was
WRONG and is fixed here.

THE BUG, and why it mattered. The old code resized the crop ANISOTROPICALLY to a
square grid, ran an isotropic distance transform on that grid, then rescaled by a
single AVERAGED factor:

    px_scale = ((x1 - x0) / W + (y1 - y0) / H) / 2.0     # one factor, two axes
    sdf = sdf_resized * px_scale

No single factor can be correct in both axes. Measured on a 40x60 box (crop
108x72 -> grid 64x64, a 1.50x anisotropy) it read -25.3 px at the box centre
against a true inradius of 20.0 px: a 27% direction-dependent error, which for a
containment penalty mis-weights the pull depending on which way a vertex lies
outside the mask.

THE FIX. `scipy.ndimage.distance_transform_edt` takes per-axis pixel spacing via
`sampling=`, so the transform itself can work in original pixels and no rescale
is needed. With `sampling=(dy, dx)` the same test box reads exactly -20.0.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BIG = 1.0e6          # SDF fill for a (frame, camera) with no usable mask

# Cap on the auto worker count. The per-(frame, camera) job is a few hundred
# microseconds of cv2/scipy C code, so past ~16 threads the pool's dispatch
# overhead eats the gain and a shared node's other jobs start to suffer.
_MAX_AUTO_WORKERS = 16


def _n_workers(workers, n_jobs, env=None):
    """Resolve the requested worker count against the CPUs actually available.

    ``SLURM_CPUS_PER_TASK`` first, then ``os.sched_getaffinity`` (never bare
    ``cpu_count``): the pipeline runs up to four bout processes per node under
    ``--cpus-per-task=8``, and sizing a pool off the node's 32-128 cores would
    have each of them oversubscribe the whole machine.
    """
    if workers is None:
        env = os.environ if env is None else env
        avail = None
        try:
            avail = int(env["SLURM_CPUS_PER_TASK"])
        except (KeyError, ValueError, TypeError):
            pass
        if not avail or avail < 1:
            try:
                avail = len(os.sched_getaffinity(0))
            except AttributeError:                   # pragma: no cover - non-Linux
                avail = os.cpu_count() or 1
        workers = min(_MAX_AUTO_WORKERS, max(1, avail))
    return max(1, min(int(workers), max(1, n_jobs)))


def mask_bbox(mask, margin=0.4):
    """(x0, y0, x1, y1) float bbox of the mask, grown by `margin` of its size.
    None if the mask is empty."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (x0 - mx, y0 - my, x1 + mx, y1 + my)


def mask_to_sdf_crop(mask, bbox, out_hw):
    """Crop -> resize -> SIGNED distance field in ORIGINAL pixels.

    Returns (sdf (H,W) float32, grid_scale (2,) float32, grid_offset (2,)
    float32), or None if the crop is empty. The sampling convention is
    ``grid_xy = (orig_xy - grid_offset) * grid_scale``, unchanged from the
    original so existing containment code keeps working.

    Distances are in ORIGINAL pixels EXACTLY, not approximately: the per-axis
    spacing goes into the distance transform instead of being averaged into one
    post-hoc factor.
    """
    import cv2
    from scipy import ndimage

    m = np.asarray(mask).astype(np.uint8)
    H, W = int(out_hw[0]), int(out_hw[1])
    x0, y0, x1, y1 = bbox
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(m.shape[1], int(np.ceil(x1))), min(m.shape[0], int(np.ceil(y1)))
    if x1i <= x0i or y1i <= y0i:
        return None
    crop = m[y0i:y1i, x0i:x1i]
    if crop.sum() == 0:
        return None
    rc = cv2.resize(crop, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    # per-axis ORIGINAL px per resized px -- this is the fix
    dy = (y1i - y0i) / float(H)
    dx = (x1i - x0i) / float(W)
    din = ndimage.distance_transform_edt(rc, sampling=(dy, dx))    # >0 inside
    dout = ndimage.distance_transform_edt(~rc, sampling=(dy, dx))  # >0 outside
    sdf = (dout - din).astype(np.float32)                          # ORIGINAL px

    grid_scale = np.array([W / (x1i - x0i), H / (y1i - y0i)], np.float32)
    grid_offset = np.array([x0i, y0i], np.float32)
    return sdf, grid_scale, grid_offset


def sdf_stack_from_masks(masks, valid, *, out_hw=(128, 128), bbox_margin=0.4,
                         workers=None):
    """(T,C,H,W) masks + (T,C) valid -> per-(frame,camera) SDF stack.

    Replaces the recovered `build_sdf_stack`, which indexed a COCO annotation
    file. The pipeline already carries masks as an array (sam3_masks.npz), so
    taking them directly removes a dataset dependency the pipeline does not have.

    Returns (sdf (T,C,H,W) float32, grid_scale (T,C,2), grid_offset (T,C,2),
    present (T,C) bool). A (frame, camera) with no usable mask is left
    `present=False` and its SDF filled with BIG, so a containment residual gated
    on `present` contributes nothing there rather than pulling toward garbage.

    `workers` threads the (frame, camera) loop (None = auto, 1 = serial). This
    is a pure SCHEDULING change and the result is bit-identical to the serial
    order by construction: the output arrays are preallocated and each job
    writes only its own `(t, c)` slot, so no two threads touch the same memory
    and nothing depends on completion order. Pinned by
    `tests/test_mask_sdf.py::test_threaded_sdf_stack_is_bit_identical_to_serial`.
    Threading (rather than processes) is what fits: `cv2.resize` and
    `scipy.ndimage.distance_transform_edt` are C extensions that release the
    GIL, and the (T,C,H,W) mask array would have to be pickled to subprocesses.
    Measured on Session0/2025_10_20_13_20_04 bout 28 fly0 (2007x7 masks):
    34.4 s serial -> 4.8 s.
    """
    masks = np.asarray(masks)
    valid = np.asarray(valid, bool)
    T, C = masks.shape[:2]
    H, W = int(out_hw[0]), int(out_hw[1])
    sdf = np.full((T, C, H, W), BIG, np.float32)
    gs = np.ones((T, C, 2), np.float32)
    go = np.zeros((T, C, 2), np.float32)
    present = np.zeros((T, C), bool)

    def _one(tc):
        t, c = tc
        bb = mask_bbox(masks[t, c], bbox_margin)
        if bb is None:
            return
        out = mask_to_sdf_crop(masks[t, c], bb, (H, W))
        if out is None:
            return
        sdf[t, c], gs[t, c], go[t, c] = out
        present[t, c] = True

    jobs = [(t, c) for t in range(T) for c in range(C) if valid[t, c]]
    n = _n_workers(workers, len(jobs))
    if n <= 1:
        for tc in jobs:
            _one(tc)
    else:
        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(_one, jobs))
    return sdf, gs, go, present
