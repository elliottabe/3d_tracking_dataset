"""Tracking quality from the SAM masks, and the per-frame validity gate.

THE PROBLEM THIS SOLVES. Session0 bout 28's female (fly0) is well tracked for
frames 0-1499 and badly tracked for 1500-2006: her median SAM mask area
collapses from 15 862 px to 4 882 px and 40% of the masks still flagged `valid`
are slivers -- the female pressed against a wall, seen as a truncated fragment.
`min_present_cameras: 3` is blind to it, because the masks ARE present; they
are just tiny. Anything that fits against them (the post-STAC wing-pitch
refinement, Stage D2) then solves a third of a fly and produces a 130 deg pitch
swing, which every band-limited parameterisation renders as a *smooth* wrong
answer -- the more dangerous kind, because it looks plausible.

The user's ruling (2026-09-01) is that skipping such fly-frames is a FEATURE:
the correction is for flies that are well tracked, and a poorly tracked female
may simply be left at her marker-solve pose.

TWO TOOLS, AND THEY ARE NOT THE SAME TOOL.

  `bout_fly_quality` is MASK-ONLY -- area, validity, centroids -- so a whole
  recording's bouts can be ranked for tracking quality with no pose, no
  triangulation and no GPU. It is how a well-tracked bout gets chosen.

  `camera_validity` / `frame_keep` / `gate_envelope` are the PER-FRAME GATE
  that runs inside the stage, and that one is POSE-AWARE. An area test cannot
  catch a wrong-fly or a merged-fly mask: those have perfectly normal area and
  give themselves away only by disagreeing with the pose. `body_inside_fraction`
  is that signal -- what fraction of the projected body lands inside the mask --
  and it is the generalisation of the number that already made the failure
  visible (the control's own wing `inside%` drops 86.4 -> 40.2 on that stretch).

WHY THE GATE RETURNS AN ENVELOPE RATHER THAN A BOOLEAN. In `param_mode='free'`
zeroing a frame's `present` row freezes it, because the frame gate lives on the
parameter update. In `spline`/`lowpass` it does NOT: one knot spans many frames,
so the knots interpolate across a gated frame and the low-pass smears across it.
A gate that only zeroed `present` would therefore decay the correction toward
STAC over roughly one knot spacing while the frames at the edge of the gap
stayed fitted to slivers. `gate_envelope` multiplies the correction to EXACTLY
zero on the skipped frames, and ramps back to 1 over one knot spacing so the
gate itself cannot add content faster than the basis can express -- a hard 0/1
step is broadband, which is precisely what those modes exist to keep out of the
6.4-frame song band.

THE AREA REFERENCE IS PER CAMERA AND IS A HIGH QUANTILE, NOT A MEDIAN. Cameras
see the fly at very different scales (bout 28 fly0: 14 238 px on Cam2012861
against 20 156 px on Cam2012853), so an absolute floor would condemn a whole
camera or wave a sliver through depending on which camera it is. And a median
reference is dragged down by a long enough bad stretch -- at which point the
slivers define "normal" and the gate silently stops firing; p75 keeps the
reference on the healthy population for a bout up to a quarter bad, which the
worst measured case (25% of bout 28) exactly is.

Every array here is (T, C) -- frames by cameras -- the same convention as
`load_bout_masks`, `sdf_stack_from_masks` and `present`. `sam3_masks.npz`
stores its own arrays as (A, C, T, ...); `mask_areas_from_packed` takes ONE
fly's (C, T, H, Wp) slab and transposes, so the transpose happens once, in a
named place, rather than at each call site.
"""
from __future__ import annotations

import numpy as np

#: popcount of every uint8 value -- `np.unpackbits(...).sum()` per frame would
#: materialise the full (T, C, H, W) bool stack, which is 12 GB for one bout.
_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)

#: Default gate constants, MEASURED on Session0 bout 28 (both flies, all four
#: `param_mode` arms). See the module docstring and
#: docs/benchmark/2026-09-01-wing-mask-fit/notes.md.
DEFAULT_AREA_REF_PCT = 75.0
DEFAULT_MIN_AREA_FRAC = 0.25
DEFAULT_MIN_BODY_INSIDE = 0.5
DEFAULT_MIN_CAMERAS = 3


# ---------------------------------------------------------------------------
# areas
# ---------------------------------------------------------------------------
def mask_areas_from_packed(packed_fly, width=None):
    """(T, C) int32 mask area in pixels, from ONE fly's bit-packed slab.

    Args:
        packed_fly: (C, T, H, ceil(W/8)) uint8, i.e. `npz['packed'][fly]`.
        width: the true image width. `np.packbits` zero-pads the last byte, so
            for a real mask this changes nothing; it matters only for a caller
            that packed a mask whose trailing bits were set.

    The return is (T, C), NOT the npz's (C, T): every array the pipeline gates
    on -- `valid`, `present`, `sdf` -- is frames-first, and mixing the two
    conventions is the index-space error this repo has been bitten by twice.
    """
    packed_fly = np.asarray(packed_fly)
    if packed_fly.ndim != 4:
        raise ValueError(
            f"packed_fly must be (C, T, H, Wp) for ONE fly, got "
            f"{packed_fly.shape} -- pass npz['packed'][fly], not npz['packed']")
    C, T = packed_fly.shape[0], packed_fly.shape[1]
    if width is not None:
        n_bytes = packed_fly.shape[-1]
        full = int(width) // 8
        area = np.zeros((C, T), np.int32)
        for c in range(C):
            a = _POPCOUNT[packed_fly[c, :, :, :full]].sum(axis=(1, 2),
                                                          dtype=np.int32)
            if full < n_bytes:
                tail = np.unpackbits(packed_fly[c, :, :, full:], axis=-1)
                a = a + tail[:, :, :int(width) - 8 * full].sum(
                    axis=(1, 2), dtype=np.int32)
            area[c] = a
    else:
        area = np.zeros((C, T), np.int32)
        for c in range(C):
            area[c] = _POPCOUNT[packed_fly[c]].sum(axis=(1, 2), dtype=np.int32)
    return area.T.copy()


def mask_areas_from_masks(masks):
    """(T, C) int32 mask area from an already-unpacked (T, C, H, W) bool stack."""
    m = np.asarray(masks)
    if m.ndim != 4:
        raise ValueError(f"masks must be (T, C, H, W), got {m.shape}")
    return m.reshape(m.shape[0], m.shape[1], -1).sum(axis=2, dtype=np.int32)


# ---------------------------------------------------------------------------
# the reference area
# ---------------------------------------------------------------------------
def area_reference(area, valid, pct=DEFAULT_AREA_REF_PCT):
    """(C,) the fly's own HEALTHY mask area per camera, as a high quantile.

    NaN for a camera with no valid frame -- deliberately not 0, which would
    make `area / ref` infinite and every sliver on that camera look healthy.
    """
    area = np.asarray(area, np.float64)
    valid = np.asarray(valid, bool)
    if area.shape != valid.shape:
        raise ValueError(f"area {area.shape} != valid {valid.shape}")
    C = area.shape[1]
    ref = np.full(C, np.nan)
    for c in range(C):
        col = area[valid[:, c], c]
        if col.size:
            ref[c] = np.percentile(col, float(pct))
    return ref


# ---------------------------------------------------------------------------
# the per-(frame, camera) gate
# ---------------------------------------------------------------------------
def camera_validity(valid, area, ref, *, min_area_frac=DEFAULT_MIN_AREA_FRAC,
                    body_inside=None, min_body_inside=DEFAULT_MIN_BODY_INSIDE):
    """(T, C) bool: this camera's mask is USABLE for this frame.

    Three conditions, and the third is the one that matters:

      * `valid` -- SAM found the fly at all;
      * the area is at least `min_area_frac` of that camera's own `ref`, which
        catches the truncated sliver;
      * if `body_inside` is given, at least `min_body_inside` of the projected
        BODY vertices land inside the mask. THIS IS THE POSE-AWARE HALF: a
        wrong-fly or merged-fly mask has a perfectly normal area and is
        invisible to the area test.

    A NaN `ref` (camera never valid) or a NaN `body_inside` fails, since the
    honest reading of "unknown" here is "not usable".
    """
    valid = np.asarray(valid, bool)
    area = np.asarray(area, np.float64)
    ref = np.asarray(ref, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        big_enough = area >= float(min_area_frac) * ref[None, :]
    ok = valid & np.nan_to_num(big_enough, nan=False).astype(bool)
    if body_inside is not None:
        bi = np.asarray(body_inside, np.float64)
        if bi.shape != valid.shape:
            raise ValueError(f"body_inside {bi.shape} != valid {valid.shape}")
        with np.errstate(invalid="ignore"):
            ok = ok & (bi >= float(min_body_inside))
        ok = ok & np.isfinite(bi)
    return ok


def frame_keep(camera_ok, min_cameras=DEFAULT_MIN_CAMERAS):
    """(T,) bool: enough USABLE cameras to fit this frame.

    Note the input is `camera_ok`, not `valid`: counting valid cameras is what
    `min_present_cameras` already did, and it is what missed the sliver stretch
    entirely (90.7% of those masks were flagged valid).
    """
    return np.asarray(camera_ok, bool).sum(axis=1) >= int(min_cameras)


def gate_envelope(keep, ramp=0):
    """(T,) float32 multiplier for the pose CORRECTION: 0 on skipped frames.

    `ramp` is the number of frames over which the envelope returns to 1 beside a
    skipped frame; pass the mode's `knot_spacing` in `spline`/`lowpass` and 0 in
    `free`. The value at a frame `d` frames from the nearest skipped frame is
    `min(1, d / ramp)`, so the fastest slope the gate can introduce is `1/ramp`
    -- no faster than the basis itself. `ramp=0` is a hard 0/1 gate, which is
    the right and minimal behaviour in `free` mode, where the correction is
    already independent per frame and there is nothing to smear.
    """
    keep = np.asarray(keep, bool)
    if keep.ndim != 1:
        raise ValueError(f"keep must be (T,), got {keep.shape}")
    ramp = int(ramp)
    if ramp <= 0:
        return keep.astype(np.float32)
    if keep.all():
        return np.ones(keep.shape[0], np.float32)
    if not keep.any():
        return np.zeros(keep.shape[0], np.float32)
    # forward/backward distance to the nearest skipped frame, in frames
    T = keep.shape[0]
    big = np.float32(T + ramp + 1)
    d = np.where(keep, big, 0.0).astype(np.float32)
    for i in range(1, T):
        d[i] = min(d[i], d[i - 1] + 1.0)
    for i in range(T - 2, -1, -1):
        d[i] = min(d[i], d[i + 1] + 1.0)
    return np.minimum(1.0, d / np.float32(ramp)).astype(np.float32)


def body_inside_fraction(body_uv, masks, valid):
    """(T, C) fraction of projected body vertices landing inside the mask.

    Args:
        body_uv: (T, C, V, 2) float projected pixel coordinates, in the SAME
            camera order as `masks`. Non-finite vertices are dropped from both
            numerator and denominator (a vertex behind the camera is not
            evidence either way).
        masks: (T, C, H, W) bool, or anything indexable as `masks[t][c]`.
        valid: (T, C) bool.

    NaN where the camera is invalid or has no finite vertex, so a caller cannot
    read "0% inside" off a camera that simply was not looked at.
    """
    uv = np.asarray(body_uv, np.float64)
    valid = np.asarray(valid, bool)
    if uv.ndim != 4 or uv.shape[-1] != 2:
        raise ValueError(f"body_uv must be (T, C, V, 2), got {uv.shape}")
    T, C = valid.shape
    if uv.shape[:2] != (T, C):
        raise ValueError(f"body_uv {uv.shape[:2]} != valid {valid.shape}")
    out = np.full((T, C), np.nan)
    for t in range(T):
        for c in range(C):
            if not valid[t, c]:
                continue
            mk = np.asarray(masks[t][c], bool)
            H, W = mk.shape
            p = uv[t, c]
            fin = np.isfinite(p).all(axis=1)
            if not fin.any():
                continue
            xy = np.rint(p[fin]).astype(np.int64)
            inb = ((xy[:, 0] >= 0) & (xy[:, 0] < W)
                   & (xy[:, 1] >= 0) & (xy[:, 1] < H))
            hit = np.zeros(xy.shape[0], bool)
            hit[inb] = mk[xy[inb, 1], xy[inb, 0]]
            out[t, c] = hit.mean()
    return out


def wing_fit_validity_gate(masks, valid, *, body_uv=None,
                           min_area_frac=DEFAULT_MIN_AREA_FRAC,
                           min_body_inside=DEFAULT_MIN_BODY_INSIDE,
                           min_cameras=DEFAULT_MIN_CAMERAS,
                           area_ref_pct=DEFAULT_AREA_REF_PCT):
    """The whole per-frame gate, in one call. Returns a dict.

    `camera_ok` (T, C) is what to hand `sdf_stack_from_masks` as `valid`, so
    sliver and wrong-pose evidence never enters the objective at all;
    `frame_keep` (T,) is what to hand `refine_wing_pitch`, so the frames it
    skips are HELD at the marker-solve pose instead of being interpolated by
    the neighbouring knots. BOTH are needed -- see the module docstring.

    `body_uv` (T, C, V, 2) is optional only so a caller without a pose can get
    the area half; omitting it makes the gate area-only, which CANNOT catch a
    wrong-fly or merged-fly mask.
    """
    valid = np.asarray(valid, bool)
    area = mask_areas_from_masks(masks) if np.ndim(masks) == 4 else np.asarray(masks)
    ref = area_reference(area, valid, pct=area_ref_pct)
    with np.errstate(invalid="ignore"):
        area_ok = valid & np.nan_to_num(
            area >= float(min_area_frac) * ref[None, :], nan=False).astype(bool)
    inside = (body_inside_fraction(body_uv, masks, valid)
              if body_uv is not None else None)
    ok = camera_validity(valid, area, ref, min_area_frac=min_area_frac,
                         body_inside=inside, min_body_inside=min_body_inside)
    keep = frame_keep(ok, min_cameras=min_cameras)
    return dict(
        area=area, area_ref=ref, body_inside=inside, camera_ok=ok,
        frame_keep=keep,
        n_sliver_camera_frames=int((valid & ~area_ok).sum()),
        n_pose_reject_camera_frames=int((area_ok & ~ok).sum()),
        n_valid_camera_frames=int(valid.sum()),
    )


# ---------------------------------------------------------------------------
# the bout-level, mask-only ranking
# ---------------------------------------------------------------------------
def bout_fly_quality(area, valid, centroids, *,
                     min_area_frac=DEFAULT_MIN_AREA_FRAC,
                     min_cameras=DEFAULT_MIN_CAMERAS,
                     pct=DEFAULT_AREA_REF_PCT, window=200):
    """Mask-only tracking quality for ONE (bout, fly). All inputs (T, C[, 2]).

    Returns a dict with, among others:
      `frac_frames_ok`      fraction of frames with >= `min_cameras` usable ones
      `worst_window_frac_ok` the same over the WORST `window`-frame block --
          reported because a bout average hides a bad quarter, which is exactly
          how bout 28 fly0's frames 1500-2006 went unscored through two tasks
      `sliver_frac`         fraction of VALID (frame, camera) masks under
          `min_area_frac` of that camera's own reference
      `score`               the ranking scalar (see below)
      `frame_ok`            (T,) bool, the per-frame flags the summary is from

    `score` is `frac_frames_ok * worst_window_frac_ok * mean_cams_ok/C`.

    The first two factors say a bout is only as good as its worst stretch --
    that is where the fit misbehaves -- and multiplying keeps a bout that is 90%
    fine with one dead block below a bout that is uniformly 90% fine. The third
    is the TIE-BREAK, and it is not cosmetic: measured over Session0's 30 bouts,
    all 30 males and 22 of 30 females saturate the first two factors at 1.000,
    so without it the ranking cannot choose at all. A fly seen usably by 4.00 of
    7 cameras is not as well tracked as one seen by 7.00, and the difference is
    exactly the conditioning of the silhouette fit.

    Centroid jump is REPORTED, never scored -- measured on bout 28 it does not
    separate the known good/bad split at all (the male, fine throughout, jumps
    MORE than the collapsed female, because he moves more), so scoring on it
    would rank motion, not quality. There is likewise NO song proxy here: mask
    area p90/p50, the obvious candidate for wing extension, is 1.119 on bout
    28's SINGING male and 1.215 on its non-singing female -- it measures
    occlusion, not song, and using it would select for the wrong thing.
    """
    area = np.asarray(area, np.float64)
    valid = np.asarray(valid, bool)
    T, C = valid.shape
    ref = area_reference(area, valid, pct=pct)
    ok = camera_validity(valid, area, ref, min_area_frac=min_area_frac)
    keep = frame_keep(ok, min_cameras=min_cameras)

    n_valid = int(valid.sum())
    sliver = valid & ~ok
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = area / ref[None, :]

    w = min(int(window), T)
    if w > 0:
        cs = np.concatenate([[0.0], np.cumsum(keep.astype(np.float64))])
        block = (cs[w:] - cs[:-w]) / w
        worst = float(block.min()) if block.size else float(keep.mean())
    else:
        worst = float(keep.mean())

    cent = np.asarray(centroids, np.float64)
    both = valid[:-1] & valid[1:] if T > 1 else np.zeros((0, C), bool)
    jump = (np.linalg.norm(cent[1:] - cent[:-1], axis=-1)
            if T > 1 else np.zeros((0, C)))
    jump = np.where(both, jump, np.nan)

    return {
        "n_frames": int(T),
        "n_cameras": int(C),
        "valid_frac": float(valid.mean()),
        "mean_cams_valid": float(valid.sum(axis=1).mean()),
        "mean_cams_ok": float(ok.sum(axis=1).mean()),
        "frac_frames_ok": float(keep.mean()),
        "worst_window_frac_ok": worst,
        "worst_window_frames": int(w),
        "sliver_frac": float(sliver.sum() / n_valid) if n_valid else 1.0,
        "median_area_px": (float(np.median(area[valid])) if n_valid else 0.0),
        "median_area_ratio": (float(np.median(ratio[valid])) if n_valid else 0.0),
        "area_ref_px": [float(x) for x in ref],
        "centroid_jump_p95_px": (float(np.nanpercentile(jump, 95))
                                 if np.isfinite(jump).any() else float("nan")),
        "centroid_jump_max_px": (float(np.nanmax(jump))
                                 if np.isfinite(jump).any() else float("nan")),
        "score": float(keep.mean()) * worst * float(ok.sum(axis=1).mean()) / C,
        "frame_ok": keep,
        "camera_ok": ok,
    }
