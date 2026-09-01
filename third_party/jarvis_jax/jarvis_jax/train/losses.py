"""Keypoint training losses (JAX)."""
import jax
import jax.numpy as jnp


def heatmap_mse(pred, gt, vis, bg_weight=0.1, fg_thresh=0.01, eps=1e-6,
                joint_weight=None):
    """Foreground-weighted heatmap MSE. pred,gt: (B,H,W,K); vis: (B,K) bool.

    Plain mean-MSE over all H*W pixels dilutes the sparse keypoint-foreground
    gradient ~50000x, so the model collapses to predicting all-zeros and the
    peak never forms. Instead average the squared error separately over
    foreground (gt > fg_thresh) and background pixels per keypoint, then combine
    as mse_fg + bg_weight * mse_bg. This gives the peak a full-strength gradient.
    Averaged over visible keypoints only.

    ``joint_weight`` (optional ``(K,)``) scales each channel's contribution — used
    to emphasise hard channels (e.g. wing vertices) during a focused fine-tune.
    """
    d = (pred - gt) ** 2
    fg = (gt > fg_thresh).astype(pred.dtype)
    nfg = fg.sum(axis=(1, 2)) + eps
    nbg = (1.0 - fg).sum(axis=(1, 2)) + eps
    per_kp = (d * fg).sum(axis=(1, 2)) / nfg + bg_weight * (
        (d * (1.0 - fg)).sum(axis=(1, 2)) / nbg)          # (B,K)
    w = vis.astype(pred.dtype)
    if joint_weight is not None:
        w = w * jnp.asarray(joint_weight, dtype=pred.dtype)[None, :]
    return (per_kp * w).sum() / jnp.maximum(w.sum(), 1.0)


def centerdetect_instance_mse(pred, centers_xy, valid, *, heatmap_size, sigma,
                              fg_thresh=0.01, bg_weight=0.1, eps=1e-6,
                              amplitude=255.0, focal_alpha=2.0):
    """Per-INSTANCE-normalised heatmap MSE for CenterDetect (multi-animal,
    single output channel).

    Why this exists (see the coordinating task brief / centerdetect-jax-port
    report): the PyTorch CenterDetect reference (``jarvis/efficienttrack/
    loss.py::HeatmapLoss``) is PLAIN mean-MSE over every H*W pixel of a
    single-channel heatmap that may contain 0, 1, or 2 Gaussian blobs. A
    missed second fly costs one small blob's worth of squared error averaged
    over the WHOLE map -- and, worse, averaged together with the first fly's
    near-zero error in an image-level mean, a missed 2nd-fly blob in a
    2-fly image is diluted to roughly HALF the loss magnitude a fully-missed
    single fly gets in a 1-fly image (their footprints are equal size --
    JARVIS's own sigma-by-output-resolution convention, reproduced in
    ``sigma`` below, is NOT size-dependent -- so it is literally an average
    of one "correct" blob and one "wrong" blob vs. one "wrong" blob alone).
    Only 6.6% of train images are two-fly, so this arithmetic is most of why
    the PyTorch retrain's two-peak rate rose then collapsed with more
    training (see report): the gradient overwhelmingly rewards confidently
    predicting ONE peak.

    Fix, applied at the INSTANCE level (this repo's precedent is
    ``heatmap_mse``'s fg/bg split, itself per-CHANNEL; this generalises the
    same idea to per-INSTANCE within a single channel, which per-channel
    normalisation cannot reach since both flies' blobs share channel 0):

      1. Render each instance's OWN (unmerged) Gaussian via
         ``jarvis_jax.data.device.render_heatmaps`` -- reused, not
         reimplemented: treating "K_MAX instances" as if they were
         "K_MAX keypoint channels" is exactly what that function already
         computes (peak=1.0 Gaussians, zeroed wherever `valid` is False).
      2. For each valid instance, take the mean squared error over ONLY
         that instance's own small foreground footprint (gt_instance >
         fg_thresh) against the merged (max-combined) target.
      3. Average those per-instance means over EVERY valid instance in the
         WHOLE BATCH (flat, not image-then-batch) -- so an instance in a
         2-fly frame gets the SAME weight as the sole instance in a 1-fly
         frame, rather than half of it. This is the actual fix: it is what
         makes "was fly 2 found" cost as much, per animal, as "was the only
         fly found" regardless of how many co-occur in the same image.
      4. A separate, image-weighted background term (mean squared error
         over every pixel outside any instance's footprint) keeps the
         old fg/bg-split behaviour of discouraging spurious activations
         elsewhere -- unaffected by how many instances are present.

    Args:
        pred: (B,H,W,1) or (B,H,W) predicted heatmap (one CenterDetect scale).
        centers_xy: (B,K_MAX,2) instance centers in heatmap-px, this scale.
        valid: (B,K_MAX) bool -- which of centers_xy are real instances.
        heatmap_size: H (== W) of `pred`, matching `centers_xy`'s scale.
        sigma: target Gaussian std (heatmap px) for this scale -- e.g.
            JARVIS's own ``1.0 * output_res / 64`` convention (sigma=-2 in
            ``dataset2D.py::HeatmapGenerator``): 1.25 @80, 2.5 @160.
        fg_thresh: per-instance foreground threshold on the UNMERGED
            per-instance Gaussian (default matches ``heatmap_mse``).
        bg_weight: background term weight (default matches ``heatmap_mse``).
        focal_alpha: CornerNet/CenterNet-style penalty-reduction exponent on the
            BACKGROUND term. 0.0 restores the old uniform background exactly.

            Why this is needed, measured: with a uniform background term the
            model reached a two-peak rate of 1.000 on real two-fly frames by
            learning to ALWAYS emit two peaks -- hallucinating a confident
            second fly on 65% of SINGLE-fly frames, at a median 0.87 of the
            primary peak's confidence, i.e. unfilterable by any threshold. A
            10x sweep of `bg_weight` (0.1 -> 0.5 -> 1.0) moved that only
            0.648 -> 0.594 and the confidence ratio only 0.868 -> 0.782.

            It cannot work, because a spurious peak covers a handful of pixels
            out of ~25,600 on a 160x160 map: normalising by PIXEL COUNT dilutes
            its penalty to nothing no matter how `bg_weight` is scaled, while
            the per-instance foreground term charges full price for a MISSED
            instance. The loss had no term saying "a confident peak far from any
            instance is bad".

            The focal fix is CornerNet's: weight each background pixel by how
            confident the (wrong) prediction there is, ``(pred/amplitude)**
            focal_alpha``, and normalise by the SUM OF WEIGHTS rather than the
            pixel count. Quiet background contributes ~0 and stops drowning the
            signal; one confident false peak now dominates the term.
        amplitude: PEAK VALUE of the regression target. MUST be 255.0 to match
            JARVIS, which renders targets as ``255.0*np.exp(...)``
            (``dataset2D.py:420,432``) -- so its trained CenterDetect emits
            peaks of ~240 (measured: the port's own parity fixture has
            ``res2.max() == 240.119``). ``render_heatmaps`` produces peak-1.0
            Gaussians, the jarvis_jax convention for the KEYPOINT path, and
            training a 255-scale warm-started checkpoint against 1.0-scale
            targets is not merely a slower start -- it is actively harmful,
            because the model must first crush its own outputs ~240x, which
            destroys the pretrained representation. Measured before this
            argument existed: epoch-1 train_loss 840.8 collapsing to 0.59 by
            epoch 2, then a from-scratch relearn that had still not produced a
            SINGLE two-peak hit by epoch 12 (median_dist 495 -> 300 px against
            a 40 px capture radius), while the PyTorch fine-tune from the SAME
            checkpoint reached an 80% two-peak rate by epoch 10.
            Note `fg_thresh` stays relative to the UNSCALED peak-1.0 Gaussian
            below, so it keeps its meaning as a fraction of peak and does not
            need rescaling alongside this.

    Returns:
        (total, fg_loss, bg_loss) -- three JAX scalars; `total` is what a
        caller sums across scales for the actual training loss, the other
        two are for logging/diagnosis.
    """
    from jarvis_jax.data.device import render_heatmaps
    pred2d = pred[..., 0] if pred.ndim == 4 else pred                # (B,H,W)
    valid_f = valid.astype(pred2d.dtype)                              # (B,K)

    gt_inst = render_heatmaps(centers_xy, valid, heatmap_size=heatmap_size,
                              sigma=sigma)                            # (B,H,W,K)
    # fg_mask (below) is computed on the UNSCALED peak-1.0 gt_inst so fg_thresh
    # stays a fraction-of-peak; only the regression TARGET carries amplitude.
    gt_merged = amplitude * jnp.max(gt_inst, axis=-1)                 # (B,H,W)

    se = (pred2d - gt_merged) ** 2                                    # (B,H,W)

    fg_mask = gt_inst > fg_thresh                                     # (B,H,W,K)
    per_inst_count = fg_mask.sum(axis=(1, 2)) + eps                    # (B,K)
    per_inst_sum = (se[..., None] * fg_mask).sum(axis=(1, 2))          # (B,K)
    per_inst_mean = per_inst_sum / per_inst_count                      # (B,K)
    fg_loss = (per_inst_mean * valid_f).sum() / (valid_f.sum() + eps)

    # NOTE the `amplitude *` here: `fg_thresh` is a FRACTION of peak (it is
    # applied to the unscaled peak-1.0 `gt_inst` above), but `gt_merged` is
    # amplitude-scaled. Comparing the scaled target against the raw fraction
    # left a ring of Gaussian-skirt pixels (0.0000392 < g <= 0.01) in NEITHER
    # the foreground nor the background mask, silently excluded from the loss.
    bg_mask = (gt_merged <= amplitude * fg_thresh).astype(pred2d.dtype)  # (B,H,W)

    if focal_alpha > 0.0:
        # CornerNet-style penalty reduction: a background pixel matters in
        # proportion to how confidently it is (wrongly) predicted.
        conf = jnp.clip(pred2d / amplitude, 0.0, 1.0)                   # (B,H,W)
        bg_w = bg_mask * conf ** focal_alpha
        # Normalise by the SUM OF WEIGHTS, not the pixel count -- this is the
        # part that actually fixes the dilution. A floor of 1.0 keeps a fully
        # quiet background at ~0 loss instead of dividing by ~0.
        denom = jnp.maximum(bg_w.sum(axis=(1, 2)), 1.0)                 # (B,)
        bg_per_image = (se * bg_w).sum(axis=(1, 2)) / denom             # (B,)
    else:
        nbg = bg_mask.sum(axis=(1, 2)) + eps                             # (B,)
        bg_per_image = (se * bg_mask).sum(axis=(1, 2)) / nbg             # (B,)
    bg_loss = bg_per_image.mean()

    total = fg_loss + bg_weight * bg_loss
    return total, fg_loss, bg_loss


def mask_containment(pred, mask224, *, dilate=0, eps=1e-6):
    """Fraction of positive predicted heatmap mass outside the (dilated) mask.

    pred: (B,H,W,K); mask224: (B,H,W) float {0,1}. Returns a scalar in [0,1].
    ``dilate`` (static int) grows the mask by a dilate x dilate max-pool before
    the penalty (edge slack for wings/legs that legitimately extend past the raw
    SAM mask). dilate=0 == the original behaviour. Monotone non-increasing in
    dilate. Mirrors the PyTorch mask_containment_loss dilation.
    """
    from jarvis_jax.models.dilation import dilate_mask_jax
    m = dilate_mask_jax(mask224, dilate) if dilate and dilate > 1 else mask224
    p = jax.nn.relu(pred)                                 # positive mass
    outside = p * (1.0 - m[..., None])
    return outside.sum() / (p.sum() + eps)
