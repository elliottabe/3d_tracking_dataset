"""D3 numpy frameset builder: SAM3 masks + frames -> V3-format crops4 + centerHM."""
import numpy as np

from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.geometry.center3d import triangulate_dlt_batched, project_center_to_cameras

CROP = 448


def build_frameset(frame_imgs, masks, centroids, valid, cameraMatrices, *,
                   distractor_masks=None, crop=CROP):
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
            d = d[:hh, :ww] & ~m[:hh, :ww].astype(bool)
            if d.any():
                region = crops4[c, :hh, :ww, :3]
                mean_color = region.reshape(-1, 3).mean(axis=0)
                region[d] = mean_color.astype(np.uint8)
    return crops4, centerHM, n_valid
