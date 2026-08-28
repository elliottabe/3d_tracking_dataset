"""D3 numpy frameset builder: SAM3 masks + frames -> V3-format crops4 + centerHM."""
import cv2
import numpy as np

from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.geometry.center3d import triangulate_dlt_batched, project_center_to_cameras

CROP = 448


def _dilate(mask, radius):
    """Binary dilation by an elliptical structuring element of `radius` px."""
    if radius <= 0:
        return np.asarray(mask, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(np.asarray(mask, np.uint8), k).astype(bool)


def build_frameset(frame_imgs, masks, centroids, valid, cameraMatrices, *,
                   distractor_masks=None, crop=CROP, distractor_dilate=0,
                   target_protect=None):
    """Build a V3-format frameset for one fly at one frame.

    frame_imgs (nc,H,W,3) uint8 RGB; masks (nc,H,W) bool; centroids (nc,2) full-px;
    valid (nc,) bool; cameraMatrices (nc,4,3). Returns (crops4 (nc,crop,crop,4) uint8
    | None, centerHM (nc,2) f32, n_valid). center3D for placement = triangulated valid
    centroids -> reprojected to all cameras -> clamped crop centers (V3 crop_origin).

    distractor_masks (nc,H,W) bool | None: union of OTHER animals' masks. When
    given, distractor-fly pixels in the RGB crop are replaced with the crop's
    mean colour (target-overlap excluded) so the detector sees a single clean
    fly + the target-mask channel. This matches JARVIS's dataset2D training
    crop and build_4ch_crops inference crop; omitting it feeds two overlapping
    flies on courtship frames and collapses the 3D reconstruction.
    """
    nc, H, W = masks.shape
    valid = np.asarray(valid, bool)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return None, np.zeros((nc, 2), np.float32), n_valid

    center3D = np.asarray(triangulate_dlt_batched(
        centroids[None].astype(np.float32), cameraMatrices[None].astype(np.float32),
        valid[None]))[0]                                   # (3,)
    proj = project_center_to_cameras(center3D, cameraMatrices)  # (nc,2) full px
    half = crop / 2.0
    crops4 = np.zeros((nc, crop, crop, 4), np.uint8)
    centerHM = np.zeros((nc, 2), np.float32)
    for c in range(nc):
        cx, cy = float(proj[c, 0]), float(proj[c, 1])
        x0, y0 = crop_origin([cx, cy, 0.0, 0.0], W, H, crop)   # clamped top-left
        centerHM[c] = [x0 + half, y0 + half]
        rgb = frame_imgs[c, y0:y0 + crop, x0:x0 + crop, :3]
        m = masks[c, y0:y0 + crop, x0:x0 + crop]
        # zero-pad if the (clamped) crop is short at an edge
        hh, ww = rgb.shape[0], rgb.shape[1]
        crops4[c, :hh, :ww, :3] = rgb
        crops4[c, :m.shape[0], :m.shape[1], 3] = m.astype(np.uint8)
        # Gray-fill distractor-fly pixels with the crop mean (matches JARVIS
        # dataset2D:253-262 / build_4ch_crops). Exclude target-overlap pixels
        # so we never gray out part of the target fly. Mean is taken over the
        # crop *before* filling, as in JARVIS.
        if distractor_masks is not None:
            d = distractor_masks[c, y0:y0 + crop, x0:x0 + crop].astype(bool)
            d = d[:hh, :ww]
            t = m[:hh, :ww].astype(bool)
            # A SAM3 mask is the fly's BODY silhouette: the legs, wings and
            # antennae fall OUTSIDE it and survive the fill, so the crop still
            # contains a limbed, fly-shaped object. Measured on
            # Session0/2025_10_20_13_20_04 bout_00028 while the female is
            # edge-on against the wall (her mask drops to 2-5% of its area on
            # five cameras): the detector labels that leftover male as the
            # target on 87-100% of frames, on cameras where HER mask is 100%
            # valid, which drags her 3D track onto his (Scutellum separation
            # 25.7 -> 0.85 units, against a 23.8-unit body span).
            # Dilating closes the limbs into the filled blob. The TARGET is
            # dilated by the same radius and excluded, so the target's own
            # limbs are never greyed out -- without that the fix is symmetric
            # and would damage the fly we are trying to label.
            if distractor_dilate > 0:
                # The TARGET's wings are outside its mask too, and an extended
                # wing reaches far past the body -- often straight at the other
                # fly. Protecting only `distractor_dilate` px around the target
                # body therefore ERASED the target's own wing tip: measured on
                # bout_00028, fly0's wing tip fell inside the fill in 12.1% of
                # views with no dilation and 24.2% at a symmetric 15px, and the
                # re-detected bouts came back with wing L/R identity flipping
                # 3.33% -> 5.71% of adjacent frames. Protect the target with a
                # LARGER radius than the distractor is grown by: at 15/60 the
                # target wing is erased in 7.4% of views -- better than the
                # undilated original -- while the grey still covers 1.65x the
                # undilated area, keeping the limb closure.
                protect = (distractor_dilate if target_protect is None
                           else int(target_protect))
                d = _dilate(d, distractor_dilate) & ~_dilate(t, protect)
            else:
                d = d & ~t
            if d.any():
                region = crops4[c, :hh, :ww, :3]
                mean_color = region.reshape(-1, 3).mean(axis=0)
                region[d] = mean_color.astype(np.uint8)
    return crops4, centerHM, n_valid
