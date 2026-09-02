"""The SDF's ORIGINAL-pixel scaling must be exact, not averaged.

The recovered silhouette_sdf averaged the two axis scale factors into one, which
cannot be right for a non-square crop: measured 27% error on a 40x60 box. These
tests pin the fix.
"""
import numpy as np
import pytest

from jarvis_jax.tracking.mask_sdf import (mask_bbox, mask_to_sdf_crop,
                                          sdf_stack_from_masks, BIG)


def _box(h=40, w=60, H=200, W=300):
    m = np.zeros((H, W), bool)
    y0, x0 = (H - h) // 2, (W - w) // 2
    m[y0:y0 + h, x0:x0 + w] = True
    return m, (y0 + h / 2, x0 + w / 2)


def test_sign_convention_negative_inside_positive_outside():
    m, (cy, cx) = _box()
    bb = mask_bbox(m, 0.4)
    sdf, gs, go = mask_to_sdf_crop(m, bb, (64, 64))
    at = lambda y, x: float(sdf[int(round((y - go[1]) * gs[1])),
                               int(round((x - go[0]) * gs[0]))])
    assert at(cy, cx) < 0, "inside must be negative"
    assert float(sdf[0, 0]) > 0, "the crop corner is outside the box"


def test_distance_is_exact_original_pixels_on_an_ANISOTROPIC_crop():
    """THE REGRESSION. A 40-tall x 60-wide box has inradius 20 px. The old code
    averaged the axis scales and read 25.3 px here -- a 27% error."""
    m, (cy, cx) = _box(h=40, w=60)
    bb = mask_bbox(m, 0.4)
    sdf, gs, go = mask_to_sdf_crop(m, bb, (64, 64))
    assert abs(gs[0] / gs[1] - 1.0) > 0.2, "this crop must actually be anisotropic"
    centre = float(sdf[int(round((cy - go[1]) * gs[1])),
                       int(round((cx - go[0]) * gs[0]))])
    assert abs(abs(centre) - 20.0) < 1.2, f"inradius should read ~20.0 px, got {centre}"


def test_a_square_crop_agrees_with_the_anisotropic_path():
    """Sanity: where the old averaging was harmless (square crop), the fixed code
    gives the same answer, so the fix is not a behaviour change in that case."""
    m, (cy, cx) = _box(h=50, w=50)
    bb = mask_bbox(m, 0.4)
    sdf, gs, go = mask_to_sdf_crop(m, bb, (64, 64))
    centre = float(sdf[int(round((cy - go[1]) * gs[1])),
                       int(round((cx - go[0]) * gs[0]))])
    assert abs(abs(centre) - 25.0) < 1.5, f"inradius should read ~25.0 px, got {centre}"


def test_distance_grows_linearly_away_from_the_edge_in_original_px():
    """Walk out along +x from the box edge; each original pixel of travel must add
    ~1 to the SDF, in BOTH axes. An averaged scale fails one of them."""
    m, (cy, cx) = _box(h=40, w=60)
    bb = mask_bbox(m, 0.4)
    sdf, gs, go = mask_to_sdf_crop(m, bb, (64, 64))
    x_edge = cx + 30.0
    y_edge = cy + 20.0
    samp = lambda y, x: float(sdf[np.clip(int(round((y - go[1]) * gs[1])), 0, 63),
                                  np.clip(int(round((x - go[0]) * gs[0])), 0, 63)])
    for axis, (a0, step) in (("x", (x_edge, (0.0, 1.0))), ("y", (y_edge, (1.0, 0.0)))):
        d5 = samp(cy + 5 * step[0] if axis == "y" else cy,
                  cx if axis == "y" else a0 + 5)
        d15 = samp(cy + 15 * step[0] if axis == "y" else cy,
                   cx if axis == "y" else a0 + 15)
        if axis == "y":
            d5 = samp(a0 + 5, cx); d15 = samp(a0 + 15, cx)
        slope = (d15 - d5) / 10.0
        assert 0.75 < slope < 1.25, f"{axis}: {slope:.2f} original px per px"


def test_empty_and_invalid_frames_are_marked_absent_not_filled_with_garbage():
    T, C, H, W = 3, 2, 40, 60
    masks = np.zeros((T, C, H, W), bool)
    masks[0, 0, 10:30, 20:40] = True
    valid = np.ones((T, C), bool)
    valid[1, :] = False                       # invalid views
    sdf, gs, go, present = sdf_stack_from_masks(masks, valid, out_hw=(32, 32))
    assert present[0, 0] and not present[0, 1], "empty mask must be absent"
    assert not present[1].any(), "invalid views must be absent"
    assert np.allclose(sdf[~present], BIG), "absent entries must be BIG, not 0"
    assert sdf[0, 0].min() < 0, "the real mask must produce an interior"


def test_bbox_margin_grows_the_box():
    m, _ = _box(h=40, w=60)
    a = mask_bbox(m, 0.0)
    b = mask_bbox(m, 0.5)
    assert (b[2] - b[0]) > (a[2] - a[0]) and (b[3] - b[1]) > (a[3] - a[1])
    assert mask_bbox(np.zeros((10, 10), bool)) is None


# ---------------------------------------------------------------------------
# Threading (Phase 2 / Task 9): the per-(frame, camera) loop is embarrassingly
# parallel and was measured at 34.4 s serial on one bout-fly, dominating the
# wing-mask-fit stage. Threading it must be a pure SCHEDULING change: every
# worker writes only its own (t, c) slot of preallocated arrays, so the output
# cannot depend on how the work was split.
# ---------------------------------------------------------------------------

def _rng_masks(T=6, C=4, H=48, W=72, seed=0):
    """Masks with a per-(t,c) blob of a DIFFERENT size and position, so a
    mis-indexed write (the race this pins) produces a different array."""
    rng = np.random.default_rng(seed)
    masks = np.zeros((T, C, H, W), bool)
    valid = np.ones((T, C), bool)
    for t in range(T):
        for c in range(C):
            h = int(rng.integers(6, 20)); w = int(rng.integers(6, 28))
            y0 = int(rng.integers(0, H - h)); x0 = int(rng.integers(0, W - w))
            masks[t, c, y0:y0 + h, x0:x0 + w] = True
    valid[2, 1] = False                       # an invalid view
    masks[4, 3] = False                       # an empty mask
    return masks, valid


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_threaded_sdf_stack_is_bit_identical_to_serial(workers):
    masks, valid = _rng_masks()
    ref = sdf_stack_from_masks(masks, valid, out_hw=(32, 32), workers=1)
    got = sdf_stack_from_masks(masks, valid, out_hw=(32, 32), workers=workers)
    names = ("sdf", "grid_scale", "grid_offset", "present")
    for name, a, b in zip(names, ref, got):
        assert a.dtype == b.dtype, f"{name} dtype changed"
        assert np.array_equal(a, b), (
            f"{name} differs between workers=1 and workers={workers}: "
            f"max|d| = {np.abs(a.astype(float) - b.astype(float)).max()}")


def test_threaded_sdf_stack_places_each_frame_camera_in_its_own_slot():
    """Ordering guard. Each (t, c) blob has a distinct area, so the SDF minimum
    (deepest interior point) is a per-slot fingerprint; a worker writing another
    worker's slot would permute these even while the multiset stayed the same."""
    masks, valid = _rng_masks(seed=3)
    ref = sdf_stack_from_masks(masks, valid, out_hw=(32, 32), workers=1)[0]
    got = sdf_stack_from_masks(masks, valid, out_hw=(32, 32), workers=8)[0]
    assert np.array_equal(ref.min(axis=(2, 3)), got.min(axis=(2, 3)))


def test_workers_default_does_not_change_the_numbers():
    masks, valid = _rng_masks(seed=7)
    ref = sdf_stack_from_masks(masks, valid, out_hw=(32, 32), workers=1)
    got = sdf_stack_from_masks(masks, valid, out_hw=(32, 32))   # auto
    for a, b in zip(ref, got):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("env,cpus,expected", [
    ({"SLURM_CPUS_PER_TASK": "8"}, 4096, 8),      # the batch case: 4 bouts x 8 CPUs
    ({"SLURM_CPUS_PER_TASK": "64"}, 4096, 16),    # never above the auto cap
    ({}, 4096, None),                             # interactive: affinity, capped at 16
    ({"SLURM_CPUS_PER_TASK": "not-a-number"}, 4096, None),
])
def test_auto_worker_count_respects_the_slurm_allocation(env, cpus, expected):
    """A bout process must not size its pool off the whole node: the pipeline
    runs up to four of them per node under --cpus-per-task=8."""
    from jarvis_jax.tracking.mask_sdf import _n_workers, _MAX_AUTO_WORKERS
    got = _n_workers(None, cpus, env=env)
    if expected is None:
        assert 1 <= got <= _MAX_AUTO_WORKERS
    else:
        assert got == expected


def test_worker_count_never_exceeds_the_job_count():
    from jarvis_jax.tracking.mask_sdf import _n_workers
    assert _n_workers(None, 3, env={"SLURM_CPUS_PER_TASK": "8"}) == 3
    assert _n_workers(8, 2) == 2
    assert _n_workers(0, 100) == 1
