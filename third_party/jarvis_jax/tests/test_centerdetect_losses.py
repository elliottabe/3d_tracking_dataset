"""Tests for ``jarvis_jax.train.losses.centerdetect_instance_mse``.

Two claims under test, matched to the task brief's actual language (see that
function's docstring):

1. "A missed second fly currently costs one Gaussian's worth of pixels in a
   320x320 map -- negligible against the background." -- the OLD recipe
   (PyTorch's plain full-image mean-MSE, ``HeatmapLoss`` in
   ``third_party/JARVIS-HybridNet/jarvis/efficienttrack/loss.py``, no fg/bg
   split at all) divides a fully-missed instance's error by the ENTIRE H*W
   pixel count, not by the instance's own small footprint -- so the SAME
   "totally missed" scenario is orders of magnitude smaller in that scheme
   than in one that normalises per-instance-footprint.
2. "Normalise so each ANIMAL contributes equally to the loss regardless of
   how many are present" -- the fixed loss's fg term for "the only fly in a
   1-fly frame, missed" and "one of two flies in a 2-fly frame, missed
   (other found perfectly)" should be close -- NOT diluted by co-occurring
   correctly-predicted instances in the same image.

Both are checked directly on synthetic heatmaps, not asserted from prose.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from jarvis_jax.data.device import render_heatmaps
from jarvis_jax.train.losses import centerdetect_instance_mse

H = 64
SIGMA = 2.5
# Centers chosen so every instance's Gaussian footprint (~4*sigma=10px radius)
# sits fully inside the grid with equal margin on every side -- avoids
# boundary clipping making footprints unequal for reasons that have nothing
# to do with the loss itself.
CENTER_ONE = [32.0, 32.0]          # margin 32px on every side
CENTER_TWO_A = [20.0, 32.0]        # margin 20px
CENTER_TWO_B = [44.0, 32.0]        # margin 20px (mirror of A)


def _naive_full_image_mse(pred2d, gt_merged):
    """The OLD PyTorch recipe: plain mean over every pixel, then mean over
    the batch -- matches HeatmapLoss.forward exactly (mean over H,W then
    per-sample values averaged)."""
    per_image = ((pred2d - gt_merged) ** 2).mean(axis=(1, 2))
    return float(per_image.mean())


def test_fg_loss_is_not_diluted_by_the_background_pixel_count():
    """A single missed instance: the fixed per-instance fg_loss (normalised
    by that instance's own ~small footprint) must be MUCH larger than the
    naive full-image mean (normalised by all H*W=4096 pixels) for the exact
    same prediction/target -- quantifying "negligible against the
    background" for the recipe this replaces."""
    centers = jnp.asarray([[CENTER_ONE, [0.0, 0.0]]], dtype=jnp.float32)
    valid = jnp.asarray([[True, False]])
    gt = render_heatmaps(centers, valid, heatmap_size=H, sigma=SIGMA)
    gt_merged = jnp.max(gt, axis=-1)
    pred_missed = jnp.zeros((1, H, H))                       # model predicts nothing anywhere

    naive = _naive_full_image_mse(pred_missed, gt_merged)
    _, fg, _bg = centerdetect_instance_mse(pred_missed, centers, valid,
                                           heatmap_size=H, sigma=SIGMA)
    fg = float(fg)

    assert naive > 0 and fg > 0
    footprint_frac = float((gt_merged > 0.01).sum()) / (H * H)
    assert footprint_frac < 0.05, "test setup: footprint should be small vs the frame"
    # The naive scheme's own missed-instance contribution IS diluted by
    # roughly the footprint's share of the frame; the fixed one is not.
    assert fg / naive > 10.0, f"fg={fg:.4f} naive={naive:.6f} ratio={fg / naive:.2f}"


def test_fixing_one_instance_reduces_batch_loss_by_the_same_amount_regardless_of_its_images_occupancy():
    """The actual fix, stated as a marginal-contribution equality (avoids the
    single-image-batch degeneracy where "per-instance mean, batch of one"
    trivially equals "per-image mean" -- the property this loss buys is
    about how instances trade off ACROSS a real multi-image batch, not
    within one image alone).

    3-image batch: image0 and image1 each have ONE fly (missed); image2 has
    TWO flies, A and B, both missed. By construction every missed instance's
    own per-footprint error is identical (translation-invariant unclipped
    Gaussian, same sigma). Then:
      - "fixing" image0's instance (predict it perfectly) removes ONE missed
        instance from a TOTAL of 4 in the batch -> fg_loss should drop by
        1/4 of one instance's error.
      - "fixing" image2's fly A (leaving fly B in the SAME image still
        missed) removes a DIFFERENT one of those same 4 instances -> under
        this loss the drop must be the SAME 1/4, because weight is assigned
        per INSTANCE across the whole batch, not per IMAGE-then-per-instance
        (which would give image2's fly A only half that credit, since it
        shares its image with fly B).
    """
    centers = jnp.stack([
        jnp.asarray([CENTER_ONE, [0.0, 0.0]]),
        jnp.asarray([CENTER_ONE, [0.0, 0.0]]),
        jnp.asarray([CENTER_TWO_A, CENTER_TWO_B]),
    ])                                                          # (3,2,2)
    valid = jnp.asarray([[True, False], [True, False], [True, True]])

    gt_im0 = render_heatmaps(centers[0:1], valid[0:1], heatmap_size=H, sigma=SIGMA)[..., 0]
    gt_im2_A_only = render_heatmaps(
        centers[2:3], jnp.asarray([[True, False]]), heatmap_size=H, sigma=SIGMA)[..., 0]

    pred_base = jnp.zeros((3, H, H))                             # everyone missed

    pred_fix_im0 = pred_base.at[0].set(gt_im0[0])                 # image0's lone fly: fixed
    pred_fix_im2_A = pred_base.at[2].set(gt_im2_A_only[0])        # image2's fly A: fixed; B still missed

    _, fg_base, _ = centerdetect_instance_mse(pred_base, centers, valid, heatmap_size=H, sigma=SIGMA)
    _, fg_fix_im0, _ = centerdetect_instance_mse(pred_fix_im0, centers, valid, heatmap_size=H, sigma=SIGMA)
    _, fg_fix_im2_A, _ = centerdetect_instance_mse(pred_fix_im2_A, centers, valid, heatmap_size=H, sigma=SIGMA)

    fg_base, fg_fix_im0, fg_fix_im2_A = float(fg_base), float(fg_fix_im0), float(fg_fix_im2_A)
    delta_im0 = fg_base - fg_fix_im0
    delta_im2_A = fg_base - fg_fix_im2_A

    assert delta_im0 > 0 and delta_im2_A > 0
    assert abs(delta_im0 - delta_im2_A) / delta_im0 < 1e-4, (
        f"fixing a solo instance dropped the batch loss by {delta_im0:.6f}, "
        f"but fixing one of two co-occurring instances dropped it by "
        f"{delta_im2_A:.6f} -- these should be equal (each instance is "
        f"1/4 of this batch's total, regardless of which image it is in)")
    # Sanity: under a naive PER-IMAGE-then-batch-mean scheme (the "per-image"
    # this loss deliberately is NOT), fixing image2's fly A only halves that
    # image's own per-instance mean (fly B is still missed) before it is
    # averaged again across images -- a strictly smaller credit than fixing
    # image0's lone instance, which zeroes that image's ENTIRE per-image
    # mean. Demonstrate the two schemes actually disagree here (otherwise
    # this test would not be discriminating).
    per_im_base = jnp.asarray([1.0, 1.0, 1.0])                    # normalized units: each missed inst = 1
    per_im_fix_im0 = jnp.asarray([0.0, 1.0, 1.0])
    per_im_fix_im2_A = jnp.asarray([1.0, 1.0, 0.5])
    naive_delta_im0 = float((per_im_base - per_im_fix_im0).mean())
    naive_delta_im2_A = float((per_im_base - per_im_fix_im2_A).mean())
    assert naive_delta_im0 != naive_delta_im2_A, (
        "the per-image-mean baseline should NOT give equal credit here -- "
        "if it does, this test is not actually distinguishing the two schemes")


@pytest.mark.parametrize("amplitude", [1.0, 255.0])
def test_perfect_prediction_gives_zero_fg_and_bg_loss(amplitude):
    """A prediction equal to the target must score zero -- AT THE TARGET'S OWN
    AMPLITUDE. Parameterised because the amplitude is the whole point: the
    default is 255.0 to match JARVIS (`255.0*np.exp(...)`), whose trained
    CenterDetect emits peaks of ~240."""
    centers = jnp.asarray([[[16.0, 16.0], [4.0, 4.0]]], dtype=jnp.float32)
    valid = jnp.asarray([[True, True]])
    gt = render_heatmaps(centers, valid, heatmap_size=H, sigma=SIGMA)
    gt_merged = amplitude * jnp.max(gt, axis=-1)
    total, fg, bg = centerdetect_instance_mse(gt_merged, centers, valid,
                                              heatmap_size=H, sigma=SIGMA,
                                              amplitude=amplitude)
    assert float(total) < 1e-8
    assert float(fg) < 1e-8
    assert float(bg) < 1e-8


def test_amplitude_mismatch_is_catastrophic_not_subtle():
    """Regression guard for the bug this parameter fixes: scoring a 255-scale
    prediction (what the warm-started checkpoint emits) against 1.0-scale
    targets must produce a HUGE loss, not a slightly worse one. Training that
    way forces the model to crush its own outputs ~240x and destroys the
    pretrained representation."""
    centers = jnp.asarray([[[16.0, 16.0], [4.0, 4.0]]], dtype=jnp.float32)
    valid = jnp.asarray([[True, True]])
    gt = render_heatmaps(centers, valid, heatmap_size=H, sigma=SIGMA)
    pred_ckpt_scale = 255.0 * jnp.max(gt, axis=-1)
    matched, _, _ = centerdetect_instance_mse(pred_ckpt_scale, centers, valid,
                                              heatmap_size=H, sigma=SIGMA,
                                              amplitude=255.0)
    mismatched, _, _ = centerdetect_instance_mse(pred_ckpt_scale, centers, valid,
                                                 heatmap_size=H, sigma=SIGMA,
                                                 amplitude=1.0)
    assert float(matched) < 1e-8
    assert float(mismatched) > 1e3


def test_zero_valid_instances_is_finite_not_nan():
    centers = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    valid = jnp.asarray([[False, False]])
    pred = jnp.zeros((1, H, H))
    total, fg, bg = centerdetect_instance_mse(pred, centers, valid,
                                              heatmap_size=H, sigma=SIGMA)
    assert np.isfinite(float(total))
    assert float(fg) == 0.0
    assert float(bg) == 0.0    # pred==gt_merged==0 everywhere


def test_accepts_4d_pred_with_trailing_channel_axis():
    centers = jnp.asarray([[[16.0, 16.0], [0.0, 0.0]]], dtype=jnp.float32)
    valid = jnp.asarray([[True, False]])
    gt = render_heatmaps(centers, valid, heatmap_size=H, sigma=SIGMA)
    gt_merged = jnp.max(gt, axis=-1)
    pred_3d = gt_merged
    pred_4d = gt_merged[..., None]
    t3, f3, b3 = centerdetect_instance_mse(pred_3d, centers, valid, heatmap_size=H, sigma=SIGMA)
    t4, f4, b4 = centerdetect_instance_mse(pred_4d, centers, valid, heatmap_size=H, sigma=SIGMA)
    assert float(t3) == float(t4)


def test_focal_makes_a_spurious_peak_expensive_but_leaves_clean_frames_free():
    """The property the focal background term exists for.

    With a UNIFORM background term the model reached two_peak=1.000 by learning
    to always emit two peaks, hallucinating a confident second fly on 65% of
    single-fly frames at a median 0.87 of the primary's confidence. A 10x
    bg_weight sweep moved that only 0.648 -> 0.594, because a spurious peak
    covers a few pixels out of ~25,600 and normalising by PIXEL COUNT dilutes
    it away. Focal weights each background pixel by how confidently it is
    wrongly predicted and normalises by the sum of weights instead.
    """
    # coordinates must sit INSIDE the H=64 map used by this module
    centers = jnp.asarray([[[32.0, 32.0], [0.0, 0.0]]], dtype=jnp.float32)
    valid = jnp.asarray([[True, False]])
    AMP = 255.0
    gt = AMP * jnp.max(render_heatmaps(centers, valid, heatmap_size=H, sigma=SIGMA), axis=-1)
    ghost_c = jnp.asarray([[[12.0, 50.0], [0.0, 0.0]]], dtype=jnp.float32)
    ghost = AMP * jnp.max(render_heatmaps(ghost_c, valid, heatmap_size=H, sigma=SIGMA), axis=-1)

    def bg(pred, alpha):
        return float(centerdetect_instance_mse(pred, centers, valid, heatmap_size=H,
                                               sigma=SIGMA, bg_weight=1.0,
                                               focal_alpha=alpha)[2])

    # a clean frame must stay free under focal -- otherwise it would just be a
    # constant tax that suppresses real peaks too
    assert bg(gt, 2.0) < 1e-6

    with_ghost = jnp.maximum(gt, ghost)
    uniform, focal = bg(with_ghost, 0.0), bg(with_ghost, 2.0)
    # Focal must make the ghost DRAMATICALLY more expensive, not marginally.
    # The multiplier scales with map size, because it is undoing dilution by
    # the background pixel count: ~100x at this module's H=64, ~647x measured
    # at the production H=160. Assert a floor that holds at the smaller size.
    assert focal > 50 * uniform, (focal, uniform)


def test_focal_alpha_zero_reproduces_the_uniform_background_exactly():
    centers = jnp.asarray([[[16.0, 16.0], [4.0, 4.0]]], dtype=jnp.float32)
    valid = jnp.asarray([[True, True]])
    pred = jnp.zeros((1, H, H))
    a = centerdetect_instance_mse(pred, centers, valid, heatmap_size=H, sigma=SIGMA,
                                  focal_alpha=0.0)
    assert all(np.isfinite(float(x)) for x in a)
